//! ARM multi-row (prefill) K-quant tiles.
//!
//! Q8_K activation rows are repacked into four-row groups whose quants are
//! interleaved in 4-byte runs ([`Q8K4Block`]). The indexed `SDOT` form then
//! yields one weight column's integer dot against four activation rows in the
//! four lanes of a vector, so each unpacked weight chunk serves four rows and
//! each activation vector serves a tile of columns without horizontal sums.
//!
//! Numerics match [`super::arm::q4`] / [`super::arm::q6`] and the scalar
//! oracle bit for bit: every per-(row, column, super-block) integer total and
//! min correction is an exact `i32`, converted with the same round-to-nearest
//! conversion, and the float step is the same expression with separate
//! multiplies/adds (`sum += a.d * (d * T - dmin * C)` for Q4_K,
//! `sum += a.d * d * T` for Q6_K), accumulated over super-blocks in order.
use super::*;
use std::arch::aarch64::*;

/// Four Q8_K activation rows of one super-block.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(super) struct Q8K4Block {
    /// `qs[16 * v + 4 * r + k]` is quant `4 * v + k` of row `r`.
    qs: [i8; 4 * QK_K],
    /// Row scales (`Q8KBlock::d`); zero for padding rows.
    d: [f32; 4],
    /// `pair_sums[s][r]`: sum of row `r`'s quants `32 * s .. 32 * s + 32`
    /// (`bsums[2s] + bsums[2s + 1]`).
    pair_sums: [[i32; 4]; QK_K / 32],
}

impl Q8K4Block {
    const ZERO: Self = Self {
        qs: [0; 4 * QK_K],
        d: [0.0; 4],
        pair_sums: [[0; 4]; QK_K / 32],
    };
}

/// Repack `rows` rows of `blocks` Q8_K blocks into `rows.div_ceil(4)` groups.
pub(super) fn pack_rows(
    input: &[Q8KBlock],
    rows: usize,
    blocks: usize,
    packed: &mut Vec<Q8K4Block>,
) {
    packed.clear();
    packed.resize(rows.div_ceil(4) * blocks, Q8K4Block::ZERO);
    for (group, dst) in packed.chunks_mut(blocks).enumerate() {
        for r in 0..4 {
            let row = group * 4 + r;
            if row >= rows {
                break;
            }
            for (src, out) in input[row * blocks..(row + 1) * blocks]
                .iter()
                .zip(dst.iter_mut())
            {
                out.d[r] = src.d;
                for v in 0..QK_K / 4 {
                    for k in 0..4 {
                        out.qs[16 * v + 4 * r + k] = src.qs[4 * v + k];
                    }
                }
                for s in 0..QK_K / 32 {
                    out.pair_sums[s][r] =
                        i32::from(src.bsums[2 * s]) + i32::from(src.bsums[2 * s + 1]);
                }
            }
        }
    }
}

/// `acc + dot(a[4r..4r+4], b[4I..4I+4])` in lane `r` (indexed SDOT).
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot_lane<const I: i32>(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut result = acc;
    // SAFETY: the caller guarantees dotprod; indexed `sdot` works on the
    // three registers only and touches no memory.
    unsafe {
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[{i}]",
            acc = inout(vreg) result,
            a = in(vreg) a,
            b = in(vreg) b,
            i = const I,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    result
}

/// Dot of 16 weight quants `w` (elements `e..e + 16`) with four rows.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot16(acc: int32x4_t, x: *const i8, w: int8x16_t) -> int32x4_t {
    // SAFETY: the caller guarantees dotprod and 64 readable bytes at `x`
    // (16 quants of each of the four interleaved rows).
    unsafe {
        let mut acc = sdot_lane::<0>(acc, vld1q_s8(x), w);
        acc = sdot_lane::<1>(acc, vld1q_s8(x.add(16)), w);
        acc = sdot_lane::<2>(acc, vld1q_s8(x.add(32)), w);
        sdot_lane::<3>(acc, vld1q_s8(x.add(48)), w)
    }
}

#[inline]
fn f16_at(block: &[u8], offset: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([block[offset], block[offset + 1]])).to_f32()
}

/// Q4_K × four packed rows for `C` consecutive columns from `column`.
/// Returns per-column row-lane sums over every super-block.
///
/// # Safety
/// Requires NEON + dotprod; `group` holds `blocks` packed blocks and the
/// columns exist in `data`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
#[allow(clippy::needless_range_loop)]
pub(super) unsafe fn q4_tile<const C: usize>(
    data: &[u8],
    blocks: usize,
    column: usize,
    group: &[Q8K4Block],
) -> [float32x4_t; C] {
    let row_bytes = blocks * Q4_K_BLOCK_BYTES;
    let weights = &data[column * row_bytes..(column + C) * row_bytes];
    assert_eq!(group.len(), blocks);
    let mut sums = [vdupq_n_f32(0.0); C];
    // SAFETY: the caller guarantees dotprod. `weights` is a bounds-checked
    // slice of `C` whole rows and `b < blocks`, so each 16-byte weight load at
    // `16 + g * 32 + j` stays inside super-block `b` of column `c`. `x` points
    // at the 1024 interleaved quants of a `Q8K4Block`; `dot16` reads 64 bytes
    // from at most offset `4 * 240`. The `d` and `pair_sums` loads read whole
    // four-element arrays.
    unsafe {
        for (b, activation) in group.iter().enumerate() {
            let x = activation.qs.as_ptr();
            let mut total = [vdupq_n_s32(0); C];
            let mut correction = [vdupq_n_s32(0); C];
            // Per column: scales and mins widened to i32, four per vector.
            let mut header = [[vdupq_n_s32(0); 4]; C];
            for (c, widened) in header.iter_mut().enumerate() {
                let block = &weights[c * row_bytes + b * Q4_K_BLOCK_BYTES..][..16];
                let (scales, mins) = unpack_k4_scales(&block[4..16]);
                let scales16 = vmovl_u8(vld1_u8(scales.as_ptr()));
                let mins16 = vmovl_u8(vld1_u8(mins.as_ptr()));
                *widened = [
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales16))),
                    vreinterpretq_s32_u32(vmovl_high_u16(scales16)),
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(mins16))),
                    vreinterpretq_s32_u32(vmovl_high_u16(mins16)),
                ];
            }
            for g in 0..4 {
                let mut lo = [vdupq_n_s32(0); C];
                let mut hi = [vdupq_n_s32(0); C];
                for j in [0, 16] {
                    let x_lo = x.add(4 * (g * 64 + j));
                    let x_hi = x.add(4 * (g * 64 + 32 + j));
                    for c in 0..C {
                        let q = vld1q_u8(
                            weights
                                .as_ptr()
                                .add(c * row_bytes + b * Q4_K_BLOCK_BYTES + 16 + g * 32 + j),
                        );
                        let low = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(15)));
                        let high = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q));
                        lo[c] = dot16(lo[c], x_lo, low);
                        hi[c] = dot16(hi[c], x_hi, high);
                    }
                }
                for c in 0..C {
                    let scales = header[c][g / 2];
                    total[c] = super::arm::mla_lane(total[c], lo[c], scales, (2 * g) % 4);
                    total[c] = super::arm::mla_lane(total[c], hi[c], scales, (2 * g + 1) % 4);
                }
            }
            let row_d = vld1q_f32(activation.d.as_ptr());
            for c in 0..C {
                for s in 0..QK_K / 32 {
                    correction[c] = super::arm::mla_lane(
                        correction[c],
                        vld1q_s32(activation.pair_sums[s].as_ptr()),
                        header[c][2 + s / 4],
                        s % 4,
                    );
                }
                let block = &weights[c * row_bytes + b * Q4_K_BLOCK_BYTES..][..4];
                let d = f16_at(block, 0);
                let dmin = f16_at(block, 2);
                let inner = vsubq_f32(
                    vmulq_n_f32(vcvtq_f32_s32(total[c]), d),
                    vmulq_n_f32(vcvtq_f32_s32(correction[c]), dmin),
                );
                sums[c] = vaddq_f32(sums[c], vmulq_f32(row_d, inner));
            }
        }
    }
    sums
}

/// Q6_K × four packed rows for `C` consecutive columns from `column`.
///
/// # Safety
/// As [`q4_tile`].
#[inline]
#[target_feature(enable = "neon,dotprod")]
#[allow(clippy::needless_range_loop)]
pub(super) unsafe fn q6_tile<const C: usize>(
    data: &[u8],
    blocks: usize,
    column: usize,
    group: &[Q8K4Block],
) -> [float32x4_t; C] {
    let row_bytes = blocks * Q6_K_BLOCK_BYTES;
    let weights = &data[column * row_bytes..(column + C) * row_bytes];
    assert_eq!(group.len(), blocks);
    let mut sums = [vdupq_n_f32(0.0); C];
    // SAFETY: the caller guarantees dotprod. `weights` is a bounds-checked
    // slice of `C` whole rows and `b < blocks`, so `block` points at a whole
    // 210-byte Q6_K super-block: the low-bit loads end at or before byte 128,
    // the high-bit loads at or before byte 192, and the scale bytes read are
    // 192..208. `x` points at the 1024 interleaved quants of a `Q8K4Block`;
    // `dot16` reads 64 bytes from at most offset `4 * 240`.
    unsafe {
        let six = vdupq_n_u8(63);
        let bias = vdupq_n_s8(32);
        for (b, activation) in group.iter().enumerate() {
            let x = activation.qs.as_ptr();
            let mut total = [vdupq_n_s32(0); C];
            // Sixteen sign-extended scales per column, four per vector.
            let mut sc = [[vdupq_n_s32(0); 4]; C];
            for (c, scales) in sc.iter_mut().enumerate() {
                let block = weights.as_ptr().add(c * row_bytes + b * Q6_K_BLOCK_BYTES);
                let scales8 = vld1q_s8(block.add(192).cast::<i8>());
                let s16 = [vmovl_s8(vget_low_s8(scales8)), vmovl_high_s8(scales8)];
                *scales = [
                    vmovl_s16(vget_low_s16(s16[0])),
                    vmovl_high_s16(s16[0]),
                    vmovl_s16(vget_low_s16(s16[1])),
                    vmovl_high_s16(s16[1]),
                ];
            }
            for half in 0..2 {
                for j in [0, 16] {
                    for c in 0..C {
                        let block = weights.as_ptr().add(c * row_bytes + b * Q6_K_BLOCK_BYTES);
                        let l0 = vld1q_u8(block.add(half * 64 + j));
                        let l1 = vld1q_u8(block.add(half * 64 + 32 + j));
                        let h = vld1q_u8(block.add(128 + half * 32 + j));
                        let h2 = vshrq_n_u8::<2>(h);
                        // Same bytes as masking and or-ing the high bits in.
                        let quants = [
                            vandq_u8(vsliq_n_u8::<4>(l0, h), six),
                            vandq_u8(vsliq_n_u8::<4>(l1, h2), six),
                            vandq_u8(vsriq_n_u8::<4>(h, l0), six),
                            vandq_u8(vsriq_n_u8::<4>(h2, l1), six),
                        ];
                        for (segment, q) in quants.into_iter().enumerate() {
                            let signed = vsubq_s8(vreinterpretq_s8_u8(q), bias);
                            let s = half * 8 + segment * 2 + j / 16;
                            let element = half * 128 + segment * 32 + j;
                            let dot = dot16(vdupq_n_s32(0), x.add(4 * element), signed);
                            total[c] = super::arm::mla_lane(total[c], dot, sc[c][s / 4], s % 4);
                        }
                    }
                }
            }
            let row_d = vld1q_f32(activation.d.as_ptr());
            for c in 0..C {
                let block = &weights[c * row_bytes + b * Q6_K_BLOCK_BYTES..][..Q6_K_BLOCK_BYTES];
                let d = f16_at(block, 208);
                let scaled = vmulq_f32(vmulq_n_f32(row_d, d), vcvtq_f32_s32(total[c]));
                sums[c] = vaddq_f32(sums[c], scaled);
            }
        }
    }
    sums
}

/// Tile width in columns.
const TILE_COLUMNS: usize = 4;

/// Row groups per parallel task.
const TASK_GROUPS: usize = 8;

/// `dst[row, column] += x·w` for row groups `groups` and columns
/// `first..first + count`: each four-column weight tile is reused across the
/// task's row groups while it is cache-resident.
///
/// # Safety
/// Requires NEON + dotprod and a validated `CompressedArm` weight. `add`
/// must accept every `(row, column)` pair with `row < rows` of the range.
unsafe fn tiles(
    packed: &[Q8K4Block],
    rows: usize,
    w: &KQuantWeight,
    groups: std::ops::Range<usize>,
    first: usize,
    count: usize,
    mut add: impl FnMut(usize, usize, f32),
) {
    let blocks = w.blocks_per_row();
    let mut emit = |group: usize, column: usize, lanes: &[float32x4_t]| {
        for (c, lane_values) in lanes.iter().enumerate() {
            let mut values = [0.0f32; 4];
            // SAFETY: plain 16-byte store into a local array.
            unsafe { vst1q_f32(values.as_mut_ptr(), *lane_values) };
            for (r, value) in values.into_iter().enumerate() {
                let row = 4 * group + r;
                if row < rows {
                    add(row, column + c, value);
                }
            }
        }
    };
    let end = first + count;
    let mut column = first;
    // SAFETY: caller guarantees the CPU features and weight layout.
    unsafe {
        while column < end {
            let width = if column + TILE_COLUMNS <= end {
                TILE_COLUMNS
            } else {
                1
            };
            for group in groups.clone() {
                let packed_group = &packed[group * blocks..(group + 1) * blocks];
                if width == TILE_COLUMNS {
                    let lanes = match w.dtype() {
                        KQuantDtype::Q4K => {
                            q4_tile::<TILE_COLUMNS>(w.data(), blocks, column, packed_group)
                        }
                        KQuantDtype::Q6K => {
                            q6_tile::<TILE_COLUMNS>(w.data(), blocks, column, packed_group)
                        }
                    };
                    emit(group, column, &lanes);
                } else {
                    let lanes = match w.dtype() {
                        KQuantDtype::Q4K => q4_tile::<1>(w.data(), blocks, column, packed_group),
                        KQuantDtype::Q6K => q6_tile::<1>(w.data(), blocks, column, packed_group),
                    };
                    emit(group, column, &lanes);
                }
            }
            column += width;
        }
    }
}

/// Serial `dst += x·w` over every row and column.
///
/// # Safety
/// Requires NEON + dotprod and a validated `CompressedArm` weight; `dst`
/// is `rows × out_features`.
pub(super) unsafe fn serial(packed: &[Q8K4Block], rows: usize, w: &KQuantWeight, dst: &mut [f32]) {
    let out_features = w.out_features();
    assert_eq!(dst.len(), rows * out_features);
    let groups = rows.div_ceil(4);
    // SAFETY: forwarded caller contract.
    unsafe {
        tiles(
            packed,
            rows,
            w,
            0..groups,
            0,
            out_features,
            |row, column, value| {
                dst[row * out_features + column] += value;
            },
        );
    }
}

/// Raw destination shared by tasks that own pairwise-disjoint elements.
#[derive(Clone, Copy)]
struct DstPtr(*mut f32);
// SAFETY: every task writes a distinct (row group, column range) rectangle.
unsafe impl Send for DstPtr {}
// SAFETY: as for `Send`; shared access never aliases a written element.
unsafe impl Sync for DstPtr {}

impl DstPtr {
    /// Method access makes closures capture the whole (`Sync`) wrapper.
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Parallel `dst += x·w`: disjoint (row-group chunk × column chunk) tasks,
/// sized for ~4 tasks per worker, so short prompts still spread over the
/// pool and long prompts keep weight tiles cache-resident.
///
/// # Safety
/// As [`serial`].
pub(super) unsafe fn parallel(
    packed: &[Q8K4Block],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) {
    use rayon::prelude::*;
    let out_features = w.out_features();
    assert_eq!(dst.len(), rows * out_features);
    let groups = rows.div_ceil(4);
    let row_tasks = groups.div_ceil(TASK_GROUPS);
    let threads = rayon::current_num_threads().max(1);
    let want_col_tasks = (4 * threads).div_ceil(row_tasks).max(1);
    let col_chunk = out_features
        .div_ceil(want_col_tasks)
        .next_multiple_of(TILE_COLUMNS)
        .clamp(TILE_COLUMNS, 256);
    let col_tasks = out_features.div_ceil(col_chunk);
    let base = DstPtr(dst.as_mut_ptr());
    (0..row_tasks * col_tasks)
        .into_par_iter()
        .for_each(|index| {
            let g0 = (index / col_tasks) * TASK_GROUPS;
            let g1 = (g0 + TASK_GROUPS).min(groups);
            let c0 = (index % col_tasks) * col_chunk;
            let count = col_chunk.min(out_features - c0);
            let out = base.get();
            #[cfg(test)]
            super::route_probe::record_worker(w);
            // SAFETY: forwarded caller contract; this task alone writes rows of
            // groups `g0..g1` in columns `c0..c0 + count`, all within `dst`.
            unsafe {
                tiles(packed, rows, w, g0..g1, c0, count, |row, column, value| {
                    *out.add(row * out_features + column) += value;
                });
            }
        });
}
