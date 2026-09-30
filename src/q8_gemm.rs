//! Multi-row (prefill) Q8_0 × Q8_0 GEMM for aarch64 NEON + dotprod.
//! (Compiled on aarch64 only; other targets keep the existing batch kernels.)
//!
//! Activation rows are repacked once per call into four-row interleaved
//! blocks ([`Q8x4Block`]) so the indexed `SDOT` form produces the integer dot
//! of one weight block against four activation rows directly in the four
//! lanes of a vector, with no horizontal reduction. Each weight block is then
//! reused across a tile of activation rows and each activation block across a
//! tile of output columns.
//!
//! Numerics are bit-identical to every other Q8_0 matmul path
//! (`matmul_q8_0_decode_*`, `matmul_q8_0_batch_dotprod`, the scalar
//! fallback): for each output element the per-block integer dot is exact,
//! and the float accumulation is `sum += (f32(dot) * w_scale) * x_scale`
//! with the blocks visited in ascending order, starting from `0.0`, using
//! separate multiplies and adds (no fused multiply-add).

use crate::quant::{Q8_0_BLOCK_SIZE, Q8_0_TYPE_SIZE};
use half::f16;
use rayon::prelude::*;
use std::cell::RefCell;

/// Four activation rows of one Q8_0 block, interleaved in 4-byte groups:
/// `quants[16 * i + 4 * r + k]` is byte `4 * i + k` of row `r`'s block, and
/// `scales[r]` is row `r`'s f16 block scale widened (exactly) to f32.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Q8x4Block {
    quants: [i8; 4 * Q8_0_BLOCK_SIZE],
    scales: [f32; 4],
}

impl Q8x4Block {
    const ZERO: Self = Self {
        quants: [0; 4 * Q8_0_BLOCK_SIZE],
        scales: [0.0; 4],
    };
}

thread_local! {
    /// Packed activations for the calling thread. Moved out while Rayon runs
    /// so a nested call on the same OS thread gets independent storage.
    static PACKED_INPUT: RefCell<Vec<Q8x4Block>> = const { RefCell::new(Vec::new()) };
}

/// Output columns per register tile.
const TILE_COLS: usize = 4;
/// Row groups (of four rows) per register tile.
const TILE_GROUPS: usize = 2;
/// Row groups per parallel task.
const TASK_GROUPS: usize = 16;

/// Pack `rows` Q8_0-encoded activation rows (`blocks` blocks each) into
/// four-row interleaved groups. Padding rows of the last group are zero.
fn pack_rows(x: &[u8], rows: usize, blocks: usize, packed: &mut Vec<Q8x4Block>) {
    let groups = rows.div_ceil(4);
    let row_bytes = blocks * Q8_0_TYPE_SIZE;
    packed.clear();
    packed.resize(groups * blocks, Q8x4Block::ZERO);
    let pack_group = |(group, dst): (usize, &mut [Q8x4Block])| {
        for r in 0..4 {
            let row = group * 4 + r;
            if row >= rows {
                break;
            }
            let src = &x[row * row_bytes..(row + 1) * row_bytes];
            for (block, out) in src.chunks_exact(Q8_0_TYPE_SIZE).zip(dst.iter_mut()) {
                out.scales[r] = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                for i in 0..Q8_0_BLOCK_SIZE / 4 {
                    for k in 0..4 {
                        out.quants[16 * i + 4 * r + k] = block[2 + 4 * i + k] as i8;
                    }
                }
            }
        }
    };
    if groups * row_bytes >= 1 << 16 {
        packed
            .par_chunks_mut(blocks)
            .enumerate()
            .for_each(pack_group);
    } else {
        packed.chunks_mut(blocks).enumerate().for_each(pack_group);
    }
}

/// Raw output pointer shared by tasks that write disjoint elements.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: tasks write pairwise-disjoint (row, column) elements; see `run`.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    /// Method access makes closures capture the whole (`Sync`) wrapper.
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Whether the tiled prefill GEMM can run on this CPU.
#[inline]
pub(crate) fn supported() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
        && std::arch::is_aarch64_feature_detected!("fp16")
}

/// `out[rows, out_features] = x × wᵀ` for Q8_0-encoded activation rows `x`
/// and Q8_0 weight bytes `data` (`out_features` rows of `blocks` blocks).
///
/// # Panics
/// Panics if [`supported`] is false or the slices do not match the shape.
pub(crate) fn matmul(
    x: &[u8],
    rows: usize,
    data: &[u8],
    out_features: usize,
    blocks: usize,
    out: &mut [f32],
) {
    assert!(supported(), "q8 prefill GEMM requires dotprod and fp16");
    let row_bytes = blocks * Q8_0_TYPE_SIZE;
    assert_eq!(x.len(), rows * row_bytes);
    assert!(data.len() >= out_features * row_bytes);
    assert_eq!(out.len(), rows * out_features);
    if rows == 0 || out_features == 0 {
        return;
    }
    let mut packed = PACKED_INPUT.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    pack_rows(x, rows, blocks, &mut packed);
    run(&packed, rows, data, out_features, blocks, out);
    PACKED_INPUT.with(|cell| *cell.borrow_mut() = packed);
}

fn run(
    packed: &[Q8x4Block],
    rows: usize,
    data: &[u8],
    out_features: usize,
    blocks: usize,
    out: &mut [f32],
) {
    let groups = rows.div_ceil(4);
    let threads = rayon::current_num_threads().max(1);
    let row_tasks = groups.div_ceil(TASK_GROUPS);
    // Aim for ~4 tasks per worker so stealing evens out P/E-core speed.
    let want_col_tasks = (4 * threads).div_ceil(row_tasks).max(1);
    let col_chunk = out_features
        .div_ceil(want_col_tasks)
        .next_multiple_of(TILE_COLS)
        .clamp(TILE_COLS, 256);
    let col_tasks = out_features.div_ceil(col_chunk);
    let out_ptr = OutPtr(out.as_mut_ptr());
    let task = |index: usize| {
        let row_task = index / col_tasks;
        let col_task = index % col_tasks;
        let g0 = row_task * TASK_GROUPS;
        let g1 = (g0 + TASK_GROUPS).min(groups);
        let c0 = col_task * col_chunk;
        let c1 = (c0 + col_chunk).min(out_features);
        // SAFETY: the feature gate was checked in `matmul`; the tile covers
        // rows `4*g0..min(4*g1, rows)` × columns `c0..c1`, disjoint from every
        // other task, and all indices are within the validated shapes.
        unsafe {
            neon::task(
                packed,
                rows,
                data,
                out_features,
                blocks,
                (g0, g1),
                (c0, c1),
                out_ptr.get(),
            );
        }
    };
    let total = row_tasks * col_tasks;
    if total == 1 || threads == 1 {
        (0..total).for_each(task);
    } else {
        (0..total).into_par_iter().for_each(task);
    }
}

mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// `acc + dot(a[4r..4r+4], b[4I..4I+4])` in lane `r` (indexed SDOT).
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn sdot_lane<const I: i32>(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut result = acc;
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

    #[inline]
    #[target_feature(enable = "fp16")]
    unsafe fn load_scale(ptr: *const u8) -> f32 {
        unsafe {
            let bits = u16::from_le(std::ptr::read_unaligned(ptr.cast::<u16>()));
            let value: f32;
            std::arch::asm!("fcvt {out:s}, {bits:h}", out = out(vreg) value, bits = in(vreg) bits, options(pure, nomem, nostack, preserves_flags));
            value
        }
    }

    /// One register tile: `C` output columns × `G` four-row groups over every
    /// block, accumulated in ascending block order.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    #[target_feature(enable = "neon,dotprod,fp16")]
    unsafe fn tile<const C: usize, const G: usize>(
        packed: *const Q8x4Block,
        blocks: usize,
        w: *const u8,
        row_bytes: usize,
    ) -> [[float32x4_t; G]; C] {
        unsafe {
            let mut acc = [[vdupq_n_f32(0.0); G]; C];
            for b in 0..blocks {
                for g in 0..G {
                    let xb = packed.add(g * blocks + b);
                    let q = (*xb).quants.as_ptr();
                    let x = [
                        vld1q_s8(q),
                        vld1q_s8(q.add(16)),
                        vld1q_s8(q.add(32)),
                        vld1q_s8(q.add(48)),
                        vld1q_s8(q.add(64)),
                        vld1q_s8(q.add(80)),
                        vld1q_s8(q.add(96)),
                        vld1q_s8(q.add(112)),
                    ];
                    let xs = vld1q_f32((*xb).scales.as_ptr());
                    for (c, column) in acc.iter_mut().enumerate() {
                        let wp = w.add(c * row_bytes + b * Q8_0_TYPE_SIZE);
                        let ws = load_scale(wp);
                        let w0 = vld1q_s8(wp.add(2).cast());
                        let w1 = vld1q_s8(wp.add(18).cast());
                        let mut dot = vdupq_n_s32(0);
                        dot = sdot_lane::<0>(dot, x[0], w0);
                        dot = sdot_lane::<1>(dot, x[1], w0);
                        dot = sdot_lane::<2>(dot, x[2], w0);
                        dot = sdot_lane::<3>(dot, x[3], w0);
                        dot = sdot_lane::<0>(dot, x[4], w1);
                        dot = sdot_lane::<1>(dot, x[5], w1);
                        dot = sdot_lane::<2>(dot, x[6], w1);
                        dot = sdot_lane::<3>(dot, x[7], w1);
                        let scaled = vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(dot), ws), xs);
                        column[g] = vaddq_f32(column[g], scaled);
                    }
                }
            }
            acc
        }
    }

    /// Store a tile's lanes for rows `< rows`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store<const C: usize, const G: usize>(
        acc: &[[float32x4_t; G]; C],
        out: *mut f32,
        out_features: usize,
        rows: usize,
        g0: usize,
        c0: usize,
    ) {
        unsafe {
            for (c, column) in acc.iter().enumerate() {
                for (g, lanes) in column.iter().enumerate() {
                    let mut values = [0.0f32; 4];
                    vst1q_f32(values.as_mut_ptr(), *lanes);
                    for (r, value) in values.into_iter().enumerate() {
                        let row = 4 * (g0 + g) + r;
                        if row < rows {
                            *out.add(row * out_features + c0 + c) = value;
                        }
                    }
                }
            }
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon,dotprod,fp16")]
    unsafe fn columns<const G: usize>(
        packed: &[Q8x4Block],
        rows: usize,
        data: &[u8],
        out_features: usize,
        blocks: usize,
        g: usize,
        (c0, c1): (usize, usize),
        out: *mut f32,
    ) {
        let row_bytes = blocks * Q8_0_TYPE_SIZE;
        let x = packed[g * blocks..].as_ptr();
        unsafe {
            let mut c = c0;
            while c + TILE_COLS <= c1 {
                let w = data.as_ptr().add(c * row_bytes);
                let acc = tile::<TILE_COLS, G>(x, blocks, w, row_bytes);
                store(&acc, out, out_features, rows, g, c);
                c += TILE_COLS;
            }
            while c < c1 {
                let w = data.as_ptr().add(c * row_bytes);
                let acc = tile::<1, G>(x, blocks, w, row_bytes);
                store(&acc, out, out_features, rows, g, c);
                c += 1;
            }
        }
    }

    /// Row groups `g0..g1` × columns `c0..c1`.
    ///
    /// # Safety
    /// Requires NEON, dotprod and FP16. `out` must be valid for
    /// `rows * out_features` elements and no other thread may access the
    /// covered elements concurrently.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon,dotprod,fp16")]
    pub(super) unsafe fn task(
        packed: &[Q8x4Block],
        rows: usize,
        data: &[u8],
        out_features: usize,
        blocks: usize,
        (g0, g1): (usize, usize),
        cols: (usize, usize),
        out: *mut f32,
    ) {
        debug_assert!(packed.len() >= g1 * blocks);
        unsafe {
            let mut g = g0;
            while g + TILE_GROUPS <= g1 {
                columns::<TILE_GROUPS>(packed, rows, data, out_features, blocks, g, cols, out);
                g += TILE_GROUPS;
            }
            while g < g1 {
                columns::<1>(packed, rows, data, out_features, blocks, g, cols, out);
                g += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            })
            .collect()
    }

    /// Random Q8_0 rows with finite, varied f16 scales (incl. zero blocks).
    fn seeded_q8_rows(rows: usize, blocks: usize, seed: u64) -> Vec<u8> {
        let mut bytes = seeded_bytes(rows * blocks * Q8_0_TYPE_SIZE, seed);
        for (index, block) in bytes.chunks_exact_mut(Q8_0_TYPE_SIZE).enumerate() {
            let scale = match index % 7 {
                0 => 0.0,
                1 => 1.0e-3,
                2 => 3.5,
                _ => (index % 97) as f32 * 0.0137 + 1.0e-4,
            };
            let sign = if index % 3 == 0 { -1.0 } else { 1.0 };
            block[..2].copy_from_slice(&f16::from_f32(sign * scale).to_bits().to_le_bytes());
            for quant in &mut block[2..] {
                if *quant == 0x80 {
                    *quant = 0x81; // Q8_0 quants are clamped to [-127, 127]
                }
            }
        }
        bytes
    }

    fn reference(x: &[u8], rows: usize, data: &[u8], outputs: usize, blocks: usize) -> Vec<f32> {
        let row_bytes = blocks * Q8_0_TYPE_SIZE;
        let mut out = vec![0.0f32; rows * outputs];
        for row in 0..rows {
            for column in 0..outputs {
                let mut sum = 0.0f32;
                for b in 0..blocks {
                    let xb = &x[row * row_bytes + b * Q8_0_TYPE_SIZE..][..Q8_0_TYPE_SIZE];
                    let wb = &data[column * row_bytes + b * Q8_0_TYPE_SIZE..][..Q8_0_TYPE_SIZE];
                    let xs = f16::from_bits(u16::from_le_bytes([xb[0], xb[1]])).to_f32();
                    let ws = f16::from_bits(u16::from_le_bytes([wb[0], wb[1]])).to_f32();
                    let dot: i32 = (0..Q8_0_BLOCK_SIZE)
                        .map(|j| i32::from(xb[2 + j] as i8) * i32::from(wb[2 + j] as i8))
                        .sum();
                    sum += dot as f32 * ws * xs;
                }
                out[row * outputs + column] = sum;
            }
        }
        out
    }

    #[test]
    fn tiled_gemm_is_bit_identical_to_scalar_block_order() {
        if !supported() {
            return;
        }
        let shapes = [
            (1, 1, 1),
            (2, 3, 2),
            (3, 5, 1),
            (4, 4, 3),
            (5, 7, 4),
            (7, 9, 2),
            (8, 8, 8),
            (9, 13, 5),
            (13, 17, 3),
            (26, 64, 8),
            (33, 130, 6),
            (67, 36, 64),
            (130, 257, 2),
        ];
        for (seed, &(rows, outputs, blocks)) in shapes.iter().enumerate() {
            let x = seeded_q8_rows(rows, blocks, seed as u64 * 2 + 1);
            let w = seeded_q8_rows(outputs, blocks, seed as u64 * 2 + 2);
            let expected = reference(&x, rows, &w, outputs, blocks);
            let mut actual = vec![f32::NAN; rows * outputs];
            matmul(&x, rows, &w, outputs, blocks, &mut actual);
            let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            let actual_bits: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                actual_bits, expected_bits,
                "shape {rows}x{outputs}x{blocks}"
            );
        }
    }

    #[test]
    fn tiled_gemm_matches_existing_batch_and_decode_kernels() {
        if !supported() {
            return;
        }
        for (rows, outputs, blocks) in [(2, 12, 4), (6, 20, 9), (26, 44, 16), (41, 32, 64)] {
            let x = seeded_q8_rows(rows, blocks, 11 + rows as u64);
            let data = seeded_q8_rows(outputs, blocks, 17 + outputs as u64);
            let weight = crate::quant::QuantizedWeight::new(
                data.clone(),
                vec![outputs, blocks * Q8_0_BLOCK_SIZE],
            );
            let row_bytes = blocks * Q8_0_TYPE_SIZE;
            let mut decode = vec![0.0f32; rows * outputs];
            for (x_row, out_row) in x
                .chunks_exact(row_bytes)
                .zip(decode.chunks_exact_mut(outputs))
            {
                crate::simd::matmul_q8_0_decode(x_row, &weight, out_row);
            }
            let mut legacy = vec![0.0f32; rows * outputs];
            crate::simd::matmul_q8_0_batch_legacy(&x, rows, &weight, &mut legacy);
            let mut tiled = vec![0.0f32; rows * outputs];
            matmul(&x, rows, &data, outputs, blocks, &mut tiled);
            let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&tiled),
                bits(&decode),
                "decode {rows}x{outputs}x{blocks}"
            );
            assert_eq!(
                bits(&tiled),
                bits(&legacy),
                "legacy {rows}x{outputs}x{blocks}"
            );
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// `cargo test --release --lib q8_gemm::bench -- --ignored --nocapture`
    #[test]
    #[ignore = "timing probe"]
    fn q8_gemm_throughput() {
        for (rows, outputs, inputs) in [
            (26, 2048, 2048),
            (128, 2048, 2048),
            (512, 2048, 2048),
            (512, 512, 2048),
            (512, 8192, 2048),
            (512, 2048, 8192),
        ] {
            let blocks = inputs / Q8_0_BLOCK_SIZE;
            let x: Vec<u8> = (0..rows * blocks * Q8_0_TYPE_SIZE)
                .map(|i| (i * 7 % 13) as u8)
                .collect();
            let data: Vec<u8> = (0..outputs * blocks * Q8_0_TYPE_SIZE)
                .map(|i| (i * 11 % 17) as u8)
                .collect();
            let weight = crate::quant::QuantizedWeight::new(data.clone(), vec![outputs, inputs]);
            let mut out = vec![0.0f32; rows * outputs];
            let mut best = [f64::MAX; 2];
            for _ in 0..7 {
                let t = std::time::Instant::now();
                matmul(&x, rows, &data, outputs, blocks, &mut out);
                best[0] = best[0].min(t.elapsed().as_secs_f64());
                let t = std::time::Instant::now();
                crate::simd::matmul_q8_0_batch_legacy(&x, rows, &weight, &mut out);
                best[1] = best[1].min(t.elapsed().as_secs_f64());
            }
            let ops = 2.0 * (rows * outputs * inputs) as f64;
            println!(
                "{rows}x{outputs}x{inputs}: tiled {:.0} GOPS ({:.2} ms)  legacy {:.0} GOPS",
                ops / best[0] / 1e9,
                best[0] * 1e3,
                ops / best[1] / 1e9
            );
        }
    }
}
