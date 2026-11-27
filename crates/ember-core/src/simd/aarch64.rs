//! # Safety blanket
//! Every `unsafe fn` in this module requires the caller to have
//! runtime-checked the `#[target_feature]` set on the function, and to
//! pass slices with the length relationships the function's dispatcher
//! asserts (the `SAFETY` comment in each body names that dispatcher).
//! Tests that call a kernel directly build buffers of exactly those
//! sizes behind the same feature check.
use super::*;
use crate::quant::QuantizedWeightInterleaved;
use std::arch::aarch64::*;

// Rust 1.92 still gates the SDOT intrinsic; keep the instruction wrapper
// local and require the same CPU feature as its caller.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn signed_dot(acc: int32x4_t, x: int8x16_t, w: int8x16_t) -> int32x4_t {
    let mut result = acc;
    // SAFETY: the caller guarantees dotprod; `sdot` works on the three
    // registers only and touches no memory.
    unsafe {
        std::arch::asm!(
            "sdot {acc:v}.4s, {x:v}.16b, {w:v}.16b",
            acc = inout(vreg) result,
            x = in(vreg) x,
            w = in(vreg) w,
            options(pure, nomem, nostack),
        );
    }
    result
}

#[inline]
#[target_feature(enable = "neon,dotprod,fp16")]
unsafe fn q8_tile<const N: usize>(x: &[u8], data: &[u8], blocks: usize, dst: &mut [f32]) {
    // SAFETY: the caller guarantees the CPU features, `blocks` encoded
    // blocks in `x` and `N` rows of `blocks` blocks in `data` (asserted by
    // `matmul_q8_0_decode_dotprod`); each 34-byte block holds the two-byte
    // scale and the 32 quants read at offsets 2 and 18.
    unsafe {
        let mut sums = [0.0f32; N];
        for b in 0..blocks {
            let xp = x.as_ptr().add(b * Q8_0_TYPE_SIZE);
            let xs = load_scale(xp);
            let x0 = vld1q_s8(xp.add(2).cast());
            let x1 = vld1q_s8(xp.add(18).cast());
            for (r, sum) in sums.iter_mut().enumerate() {
                let wp = data.as_ptr().add((r * blocks + b) * Q8_0_TYPE_SIZE);
                let ws = load_scale(wp);
                let dot = signed_dot(
                    signed_dot(vdupq_n_s32(0), x0, vld1q_s8(wp.add(2).cast())),
                    x1,
                    vld1q_s8(wp.add(18).cast()),
                );
                *sum += vaddvq_s32(dot) as f32 * ws * xs;
            }
        }
        dst.copy_from_slice(&sums);
    }
}

#[inline]
#[target_feature(enable = "fp16")]
unsafe fn load_scale(ptr: *const u8) -> f32 {
    // SAFETY: the caller guarantees two readable bytes at `ptr` (read
    // unaligned) and fp16; `fcvt` only converts the loaded register.
    unsafe {
        let bits = u16::from_le(std::ptr::read_unaligned(ptr.cast::<u16>()));
        let value: f32;
        std::arch::asm!("fcvt {out:s}, {bits:h}", out = out(vreg) value, bits = in(vreg) bits, options(pure, nomem, nostack, preserves_flags));
        value
    }
}

/// Two independent output rows reuse each activation load. Integer dot
/// products are exact, and scaling preserves the scalar block order, so
/// every row matches the one-row tile bit for bit.
///
/// Two concurrent weight streams measured faster than four on M1: with
/// four rows in flight the long rows of the MLP down projection (8.5 KiB)
/// dropped a core to ~23 GB/s versus ~39 GB/s with two.
///
/// # Safety
/// Requires NEON, FP16 conversion and the signed byte dot-product extension.
#[target_feature(enable = "neon,dotprod,fp16")]
pub unsafe fn matmul_q8_0_decode_dotprod(x: &[u8], data: &[u8], blocks: usize, out: &mut [f32]) {
    assert!(x.len() >= blocks * Q8_0_TYPE_SIZE);
    assert!(data.len() >= out.len() * blocks * Q8_0_TYPE_SIZE);
    // SAFETY: the caller guarantees the CPU features; the asserts above
    // give `x` one encoded row and `data` a row for every output, and each
    // tile receives the data suffix starting at its first row.
    unsafe {
        let full_rows = out.len() / 2 * 2;
        for (tile, dst) in out[..full_rows].chunks_exact_mut(2).enumerate() {
            q8_tile::<2>(x, &data[tile * 2 * blocks * Q8_0_TYPE_SIZE..], blocks, dst);
        }
        for (row, dst) in out[full_rows..].chunks_mut(1).enumerate() {
            q8_tile::<1>(
                x,
                &data[(full_rows + row) * blocks * Q8_0_TYPE_SIZE..],
                blocks,
                dst,
            );
        }
    }
}

/// Where block `b` of weight row `r` lives in a decode weight layout.
pub(crate) trait DecodeRowBlocks {
    /// Pointer to the 32 signed quants.
    fn quants(&self, row: usize, block: usize) -> *const u8;
    /// Pointer to the little-endian f16 scale.
    fn scale(&self, row: usize, block: usize) -> *const u8;
}

/// Row-contiguous GGUF Q8_0 blocks (`[scale, 32 quants]` per block).
pub(crate) struct RowsBlocks {
    pub(crate) data: *const u8,
    pub(crate) blocks: usize,
}

impl DecodeRowBlocks for RowsBlocks {
    #[inline(always)]
    fn quants(&self, row: usize, block: usize) -> *const u8 {
        self.data
            .wrapping_add((row * self.blocks + block) * Q8_0_TYPE_SIZE + 2)
    }
    #[inline(always)]
    fn scale(&self, row: usize, block: usize) -> *const u8 {
        self.data
            .wrapping_add((row * self.blocks + block) * Q8_0_TYPE_SIZE)
    }
}

/// Four-row interleaved stripes (see `QuantizedWeightInterleaved`).
pub(crate) struct InterleavedBlocks {
    pub(crate) quants: *const u8,
    pub(crate) scales: *const u8,
    pub(crate) blocks: usize,
}

impl DecodeRowBlocks for InterleavedBlocks {
    #[inline(always)]
    fn quants(&self, row: usize, block: usize) -> *const u8 {
        self.quants
            .wrapping_add(((row / 4) * self.blocks + block) * 128 + (row % 4) * 32)
    }
    #[inline(always)]
    fn scale(&self, row: usize, block: usize) -> *const u8 {
        self.scales
            .wrapping_add(((row / 4) * self.blocks + block) * 8 + (row % 4) * 2)
    }
}

/// `R` weight rows × `C` activation columns. Each weight block is loaded
/// once for all columns. Every `(row, column)` accumulates
/// `sum += (dot as f32 * ws) * xs` in block order — the single-row decode
/// tile's exact arithmetic — so each column is bit-identical to decoding
/// that activation alone.
#[inline]
#[target_feature(enable = "neon,dotprod,fp16")]
unsafe fn q8_columns_tile<L: DecodeRowBlocks, const R: usize, const C: usize>(
    layout: &L,
    x: *const u8,
    row_bytes: usize,
    row: usize,
    out: *mut f32,
    out_stride: usize,
) {
    // SAFETY: the caller guarantees the CPU features, `C` activation
    // rows of `row_bytes` at `x` (each 34-byte block read at offsets 0,
    // 2 and 18), a `layout` addressing every block of weight rows
    // `row..row + R`, and `out` valid for `out[c * out_stride + row + r]`
    // for every `(r, c)` written below.
    unsafe {
        let mut sums = [[0.0f32; C]; R];
        for b in 0..row_bytes / Q8_0_TYPE_SIZE {
            let mut xs = [0.0f32; C];
            let mut x0 = [vdupq_n_s8(0); C];
            let mut x1 = [vdupq_n_s8(0); C];
            for c in 0..C {
                let xp = x.add(c * row_bytes + b * Q8_0_TYPE_SIZE);
                xs[c] = load_scale(xp);
                x0[c] = vld1q_s8(xp.add(2).cast());
                x1[c] = vld1q_s8(xp.add(18).cast());
            }
            for (r, row_sums) in sums.iter_mut().enumerate() {
                let wp = layout.quants(row + r, b);
                let ws = load_scale(layout.scale(row + r, b));
                let w0 = vld1q_s8(wp.cast());
                let w1 = vld1q_s8(wp.add(16).cast());
                for c in 0..C {
                    let dot = signed_dot(signed_dot(vdupq_n_s32(0), x0[c], w0), x1[c], w1);
                    row_sums[c] += vaddvq_s32(dot) as f32 * ws * xs[c];
                }
            }
        }
        for (r, row_sums) in sums.iter().enumerate() {
            for (c, &value) in row_sums.iter().enumerate() {
                *out.add(c * out_stride + row + r) = value;
            }
        }
    }
}

/// `R` weight rows × four activation columns with the column scales in
/// one vector. Lane `c` computes `(dot_c as f32 * ws) * xs_c` and adds it
/// to its running sum: the pairwise integer reduction is exact, the
/// conversion rounds like the scalar cast, and the two multiplies and the
/// add stay separate IEEE operations, so every lane equals
/// [`q8_columns_tile`] (and single-row decode) bit for bit.
#[inline]
#[target_feature(enable = "neon,dotprod,fp16")]
unsafe fn q8_columns4_tile<L: DecodeRowBlocks, const R: usize>(
    layout: &L,
    x: *const u8,
    row_bytes: usize,
    row: usize,
    out: *mut f32,
    out_stride: usize,
) {
    // SAFETY: as for `q8_columns_tile` with four activation rows at `x`;
    // `scales` and `lanes` are local four-element arrays.
    unsafe {
        let mut sums = [vdupq_n_f32(0.0); R];
        for b in 0..row_bytes / Q8_0_TYPE_SIZE {
            let mut scales = [0.0f32; 4];
            let mut x0 = [vdupq_n_s8(0); 4];
            let mut x1 = [vdupq_n_s8(0); 4];
            for c in 0..4 {
                let xp = x.add(c * row_bytes + b * Q8_0_TYPE_SIZE);
                scales[c] = load_scale(xp);
                x0[c] = vld1q_s8(xp.add(2).cast());
                x1[c] = vld1q_s8(xp.add(18).cast());
            }
            let xs = vld1q_f32(scales.as_ptr());
            for (r, sum) in sums.iter_mut().enumerate() {
                let wp = layout.quants(row + r, b);
                let ws = load_scale(layout.scale(row + r, b));
                let w0 = vld1q_s8(wp.cast());
                let w1 = vld1q_s8(wp.add(16).cast());
                let zero = vdupq_n_s32(0);
                let d0 = signed_dot(signed_dot(zero, x0[0], w0), x1[0], w1);
                let d1 = signed_dot(signed_dot(zero, x0[1], w0), x1[1], w1);
                let d2 = signed_dot(signed_dot(zero, x0[2], w0), x1[2], w1);
                let d3 = signed_dot(signed_dot(zero, x0[3], w0), x1[3], w1);
                let dots = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
                let products = vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(dots), ws), xs);
                *sum = vaddq_f32(*sum, products);
            }
        }
        for (r, sum) in sums.iter().enumerate() {
            let mut lanes = [0.0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), *sum);
            for (c, value) in lanes.into_iter().enumerate() {
                *out.add(c * out_stride + row + r) = value;
            }
        }
    }
}

/// Weight rows `row_start..row_end` against `columns` quantized
/// activation rows (`x` holds `columns * row_bytes` bytes). Output column
/// `c` row `r` is written to `out[c * out_stride + r]`.
///
/// # Safety
/// Requires NEON, FP16 and dotprod; `layout` must address every block of
/// the requested rows, `x` must hold `columns` activation rows, and `out`
/// must be valid for the addressed writes.
#[target_feature(enable = "neon,dotprod,fp16")]
#[allow(
    clippy::too_many_arguments,
    reason = "kernel signature: buffers, strides and shape stay separate arguments on the hot path"
)]
pub(crate) unsafe fn matmul_q8_0_decode_columns_dotprod<L: DecodeRowBlocks>(
    layout: &L,
    x: &[u8],
    row_bytes: usize,
    columns: usize,
    row_start: usize,
    row_end: usize,
    out: *mut f32,
    out_stride: usize,
) {
    assert!(x.len() >= columns * row_bytes);
    // SAFETY: the caller contract above is forwarded; the assert covers
    // the activation rows, and every tile stays within `row_end` rows and
    // `columns` columns.
    unsafe {
        // Wider batches are compute-bound, so taller row tiles amortize
        // the activation loads; single-column work keeps two streams.
        let tall = if columns >= 4 { 4 } else { 2 };
        let mut row = row_start;
        while row < row_end {
            let height = if row + tall <= row_end {
                tall
            } else if row + 2 <= row_end {
                2
            } else {
                1
            };
            let mut column = 0;
            while column < columns {
                let xp = x.as_ptr().add(column * row_bytes);
                let op = out.add(column * out_stride);
                let width = match columns - column {
                    4.. => 4,
                    2..=3 => 2,
                    _ => 1,
                };
                macro_rules! tile {
                    ($r:literal) => {
                        match width {
                            4 => q8_columns4_tile::<L, $r>(
                                layout, xp, row_bytes, row, op, out_stride,
                            ),
                            2 => q8_columns_tile::<L, $r, 2>(
                                layout, xp, row_bytes, row, op, out_stride,
                            ),
                            _ => q8_columns_tile::<L, $r, 1>(
                                layout, xp, row_bytes, row, op, out_stride,
                            ),
                        }
                    };
                }
                match height {
                    4 => tile!(4),
                    2 => tile!(2),
                    _ => tile!(1),
                }
                column += width;
            }
            row += height;
        }
    }
}

/// Decode four adjacent outputs from the packed four-row weight stripes.
///
/// # Safety
/// Caller validates the logical output range and runtime CPU features.
#[target_feature(enable = "neon,dotprod,fp16")]
pub unsafe fn matmul_q8_0_interleaved_dotprod(
    x: &[u8],
    w: &QuantizedWeightInterleaved,
    out: &mut [f32],
    start: usize,
) {
    // SAFETY: the caller guarantees the CPU features and that rows
    // `start..start + out.len()` lie inside the weight; a four-row tile is
    // only issued when its first row is stripe-aligned.
    unsafe {
        let mut done = 0;
        while done < out.len() && !(start + done).is_multiple_of(4) {
            q8_interleaved_tile::<1>(x, w, start + done, &mut out[done..done + 1]);
            done += 1;
        }
        while done + 4 <= out.len() {
            q8_interleaved_tile::<4>(x, w, start + done, &mut out[done..done + 4]);
            done += 4;
        }
        while done < out.len() {
            q8_interleaved_tile::<1>(x, w, start + done, &mut out[done..done + 1]);
            done += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "neon,dotprod,fp16")]
unsafe fn q8_interleaved_tile<const N: usize>(
    x: &[u8],
    w: &QuantizedWeightInterleaved,
    start: usize,
    out: &mut [f32],
) {
    // SAFETY: the caller guarantees the CPU features, one encoded row in
    // `x`, and rows `start..start + N` inside one stripe of the weight.
    // The validated `quants`/`scales` lengths give every stripe 128 quant
    // and 8 scale bytes per block, which covers lanes `lane..lane + N`.
    unsafe {
        let stripe = start / 4;
        let lane = start % 4;
        let quants = w
            .quants()
            .as_ptr()
            .add(stripe * w.blocks_per_row * 128 + lane * 32);
        let scales = w
            .scales()
            .as_ptr()
            .add(stripe * w.blocks_per_row * 8 + lane * 2);
        let mut sums = [0.0f32; N];
        for b in 0..w.blocks_per_row {
            let xp = x.as_ptr().add(b * Q8_0_TYPE_SIZE);
            let xs = load_scale(xp);
            let x0 = vld1q_s8(xp.add(2).cast());
            let x1 = vld1q_s8(xp.add(18).cast());
            for (r, sum) in sums.iter_mut().enumerate() {
                let wp = quants.add(b * 128 + r * 32);
                let ws = load_scale(scales.add(b * 8 + r * 2));
                let dot = signed_dot(
                    signed_dot(vdupq_n_s32(0), x0, vld1q_s8(wp.cast())),
                    x1,
                    vld1q_s8(wp.add(16).cast()),
                );
                *sum += vaddvq_s32(dot) as f32 * ws * xs;
            }
        }
        out.copy_from_slice(&sums);
    }
}

#[inline]
#[target_feature(enable = "neon,fp16")]
unsafe fn load_f16x4(ptr: *const f16) -> float32x4_t {
    // SAFETY: the caller guarantees `ptr` is readable for four `f16`
    // values and that fp16 is available; `fcvtl` only widens the loaded
    // register.
    unsafe {
        let bits = vld1_u16(ptr.cast());
        let result: float32x4_t;
        std::arch::asm!("fcvtl {result:v}.4s, {bits:v}.4h",
            result = out(vreg) result, bits = in(vreg) bits,
            options(pure, nomem, nostack, preserves_flags));
        result
    }
}

/// FP16 conversion and products are vectorized; reduction order is unchanged.
///
/// # Safety
/// Requires NEON/FP16 and equal input lengths.
#[target_feature(enable = "neon,fp16")]
pub unsafe fn dot_product_f16_ordered(a: &[f32], b: &[f16]) -> f32 {
    // SAFETY: the caller guarantees the CPU features and equal lengths
    // (asserted by `dot_product_f16`); every 4-lane load is guarded by `i
    // + 4 <= a.len()`.
    unsafe {
        // Rust's floating-point Sum identity is negative zero.
        let mut sum = -0.0f32;
        let mut i = 0;
        while i + 4 <= a.len() {
            let products = vmulq_f32(vld1q_f32(a.as_ptr().add(i)), load_f16x4(b.as_ptr().add(i)));
            sum += vgetq_lane_f32::<0>(products);
            sum += vgetq_lane_f32::<1>(products);
            sum += vgetq_lane_f32::<2>(products);
            sum += vgetq_lane_f32::<3>(products);
            i += 4;
        }
        while i < a.len() {
            sum += a[i] * b[i].to_f32();
            i += 1;
        }
        sum
    }
}

/// SIMD FP16 weighted accumulation without FMA contraction.
///
/// # Safety
/// Requires NEON/FP16 and equal source/destination lengths.
#[target_feature(enable = "neon,fp16")]
pub unsafe fn weighted_add_f16_neon(acc: &mut [f32], src: &[f16], weight: f32) {
    // SAFETY: the caller guarantees the CPU features and equal lengths
    // (asserted by `weighted_add_f16`); every 4-lane load/store is guarded
    // by `i + 4 <= acc.len()`.
    unsafe {
        let mut i = 0;
        while i + 4 <= acc.len() {
            let dst = acc.as_mut_ptr().add(i);
            let product = vmulq_n_f32(load_f16x4(src.as_ptr().add(i)), weight);
            vst1q_f32(dst, vaddq_f32(vld1q_f32(dst), product));
            i += 4;
        }
        while i < acc.len() {
            acc[i] += weight * src[i].to_f32();
            i += 1;
        }
    }
}

/// Prompt tile: reuse each weight block for N activation rows.
///
/// # Safety
/// Requires NEON, dotprod and FP16; dimensions must describe the slices.
#[target_feature(enable = "neon,dotprod,fp16")]
pub unsafe fn matmul_q8_0_batch_dotprod<const N: usize>(
    x: &[u8],
    data: &[u8],
    outputs: usize,
    blocks: usize,
    out: &mut [f32],
) {
    let row_bytes = blocks * Q8_0_TYPE_SIZE;
    assert_eq!(x.len(), N * row_bytes);
    assert!(data.len() >= outputs * row_bytes);
    assert_eq!(out.len(), N * outputs);
    // SAFETY: the caller guarantees the CPU features; the asserts above
    // give `x` exactly `N` encoded rows and `data` at least `outputs`
    // rows, so every 34-byte block read is in bounds.
    unsafe {
        for column in 0..outputs {
            let mut sums = [0.0f32; N];
            for b in 0..blocks {
                let wp = data.as_ptr().add(column * row_bytes + b * Q8_0_TYPE_SIZE);
                let ws = load_scale(wp);
                let w0 = vld1q_s8(wp.add(2).cast());
                let w1 = vld1q_s8(wp.add(18).cast());
                for (row, sum) in sums.iter_mut().enumerate() {
                    let xp = x.as_ptr().add(row * row_bytes + b * Q8_0_TYPE_SIZE);
                    let xs = load_scale(xp);
                    let dot = signed_dot(
                        signed_dot(vdupq_n_s32(0), w0, vld1q_s8(xp.add(2).cast())),
                        w1,
                        vld1q_s8(xp.add(18).cast()),
                    );
                    *sum += vaddvq_s32(dot) as f32 * ws * xs;
                }
            }
            for (row, value) in sums.into_iter().enumerate() {
                out[row * outputs + column] = value;
            }
        }
    }
}

/// NEON-accelerated Q8_0 row dequantization.
///
/// Processes 32 quants per block in 8 batches of 4 f32 values using
/// 128-bit SIMD registers (i8 → i16 → i32 → f32 → mul).
///
/// # Safety
///
/// Caller must ensure NEON is supported (checked by dispatch above).
#[target_feature(enable = "neon")]
pub unsafe fn dequantize_row_neon(
    data: &[u8],
    block_start: usize,
    blocks_per_row: usize,
    dst: &mut [f32],
) {
    // SAFETY: the caller guarantees the CPU features;
    // `dequantize_q8_0_row` asserts that blocks `block_start..block_start
    // + blocks_per_row` lie inside `data` and that `dst` holds 32 values
    // per block, which bounds the 34-byte block reads and the eight 4-lane
    // stores.
    unsafe {
        for b in 0..blocks_per_row {
            let byte_offset = (block_start + b) * Q8_0_TYPE_SIZE;
            let base_ptr = data.as_ptr().add(byte_offset);

            // -- scale: load 2-byte f16, convert to f32, broadcast ---------
            let d_bits = u16::from_le_bytes(*(base_ptr as *const [u8; 2]));
            let d = f16::from_bits(d_bits).to_f32();
            let d_vec = vdupq_n_f32(d);

            // -- quants: load two 128-bit vectors of 16 i8 values each ----
            let quants_ptr = base_ptr.add(2) as *const i8;
            let q0 = vld1q_s8(quants_ptr);
            let q1 = vld1q_s8(quants_ptr.add(16));

            let out_offset = b * Q8_0_BLOCK_SIZE;
            let out_ptr = dst.as_mut_ptr().add(out_offset);

            // helper: dequantize 16 i8 values → 4 × float32x4_t
            #[inline(always)]
            unsafe fn process16(src: int8x16_t, scale: float32x4_t, out: *mut f32) {
                // SAFETY: internal helper, only called from the enclosing
                // kernel, which guarantees NEON and 16 writable values at
                // `out`.
                unsafe {
                    // low 8 i8 → i16
                    let i16_lo = vmovl_s8(vget_low_s8(src));
                    // high 8 i8 → i16
                    let i16_hi = vmovl_s8(vget_high_s8(src));

                    // i16 → i32 → f32 → mul → store (4 lanes each)
                    let f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16_lo))), scale);
                    let f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16_lo))), scale);
                    let f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16_hi))), scale);
                    let f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16_hi))), scale);

                    vst1q_f32(out, f0);
                    vst1q_f32(out.add(4), f1);
                    vst1q_f32(out.add(8), f2);
                    vst1q_f32(out.add(12), f3);
                }
            }

            // 16 quants → bytes 0..15, 16 quants → bytes 16..31
            process16(q0, d_vec, out_ptr);
            process16(q1, d_vec, out_ptr.add(16));
        }
    }
}

/// SIMD sum of squares using NEON FMA.
///
/// # Safety
///
/// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn sum_squares_neon(x: &[f32]) -> f32 {
    // SAFETY: the caller guarantees the CPU features; every 4-lane load is
    // guarded by `i + 4 <= n`.
    unsafe {
        let n = x.len();
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 4 <= n {
            let v = vld1q_f32(x.as_ptr().add(i));
            acc = vfmaq_f32(acc, v, v);
            i += 4;
        }

        let mut sum = vgetq_lane_f32::<0>(acc)
            + vgetq_lane_f32::<1>(acc)
            + vgetq_lane_f32::<2>(acc)
            + vgetq_lane_f32::<3>(acc);

        while i < n {
            sum += x[i] * x[i];
            i += 1;
        }
        sum
    }
}

/// SIMD `out[i] = x[i] * scale * weight[i]` using NEON.
///
/// # Safety
///
/// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn scale_weight_mul_neon(x: &[f32], scale: f32, weight: &[f32], out: &mut [f32]) {
    // SAFETY: the caller guarantees the CPU features; `scale_weight_mul`
    // asserts all three lengths equal, and every 4-lane load/store is
    // guarded by `i + 4 <= n`.
    unsafe {
        let n = x.len();
        let s = vdupq_n_f32(scale);
        let mut i = 0;

        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let wv = vld1q_f32(weight.as_ptr().add(i));
            let r = vmulq_f32(vmulq_f32(xv, s), wv);
            vst1q_f32(out.as_mut_ptr().add(i), r);
            i += 4;
        }
        while i < n {
            out[i] = x[i] * scale * weight[i];
            i += 1;
        }
    }
}

/// NEON adjacent-pair RoPE (Llama layout: pairs `(2d, 2d+1)`).
///
/// `vld2q_f32` de-interleaves four pairs into `x0`/`x1` lanes and
/// `vst2q_f32` re-interleaves the results. `vfmaq_f32(a, b, c)` is the
/// fused `a + b*c`, so `x0' = fma(x0, c, -(x1*s))` and
/// `x1' = fma(x0, s, x1*c)` round exactly like the scalar `mul_add`
/// loop: bit-identical.
///
/// # Safety
///
/// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rope_adjacent_pair_neon(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    // SAFETY: the caller guarantees the CPU features; `rope_adjacent_pair`
    // asserts `x.len() == n_heads * head_dim` and `cos`/`sin` hold
    // `head_dim / 2` values; the `d + 8 <= half` and `d + 4 <= half`
    // guards keep each 8-value pair load inside head `h` and each 4-value
    // table load inside `cos`/`sin`.
    unsafe {
        let half = head_dim / 2;
        for h in 0..n_heads {
            let off = h * head_dim;
            let mut d = 0;
            while d + 8 <= half {
                // Two independent 4-pair groups per iteration for ILP.
                let ptr = x.as_mut_ptr().add(off + 2 * d);
                let a = vld2q_f32(ptr);
                let b = vld2q_f32(ptr.add(8));
                let ca = vld1q_f32(cos.as_ptr().add(d));
                let sa = vld1q_f32(sin.as_ptr().add(d));
                let cb = vld1q_f32(cos.as_ptr().add(d + 4));
                let sb = vld1q_f32(sin.as_ptr().add(d + 4));
                let a0 = vfmaq_f32(vnegq_f32(vmulq_f32(a.1, sa)), a.0, ca);
                let a1 = vfmaq_f32(vmulq_f32(a.1, ca), a.0, sa);
                let b0 = vfmaq_f32(vnegq_f32(vmulq_f32(b.1, sb)), b.0, cb);
                let b1 = vfmaq_f32(vmulq_f32(b.1, cb), b.0, sb);
                vst2q_f32(ptr, float32x4x2_t(a0, a1));
                vst2q_f32(ptr.add(8), float32x4x2_t(b0, b1));
                d += 8;
            }
            while d + 4 <= half {
                let ptr = x.as_mut_ptr().add(off + 2 * d);
                let pairs = vld2q_f32(ptr);
                let (x0, x1) = (pairs.0, pairs.1);
                let c = vld1q_f32(cos.as_ptr().add(d));
                let s = vld1q_f32(sin.as_ptr().add(d));
                let out0 = vfmaq_f32(vnegq_f32(vmulq_f32(x1, s)), x0, c);
                let out1 = vfmaq_f32(vmulq_f32(x1, c), x0, s);
                vst2q_f32(ptr, float32x4x2_t(out0, out1));
                d += 4;
            }
            rope_adjacent_pair_scalar_head(&mut x[off..off + head_dim], d, cos, sin);
        }
    }
}

/// SIMD element-wise multiply using NEON.
///
/// # Safety
///
/// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn elemul_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: the caller guarantees the CPU features; `elemul` asserts all
    // three lengths equal, and every 4-lane load/store is guarded by `i +
    // 4 <= n`.
    unsafe {
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(av, bv));
            i += 4;
        }
        while i < n {
            out[i] = a[i] * b[i];
            i += 1;
        }
    }
}

/// SIMD dot product using NEON FMA.
///
/// # Safety
///
/// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: the caller guarantees the CPU features; `dot_product`
    // asserts equal lengths, and every 4-lane load is guarded by `i + 16
    // <= n` or `i + 4 <= n`.
    unsafe {
        let n = a.len();
        // Pinned ggml F32 reduction: four independent NEON accumulators,
        // then a fixed pairwise tree. Its compiled tail multiplies groups of
        // four separately, adds their lanes in order, and fuses the final 1–3.
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i + 16 <= n {
            for (j, sum) in acc.iter_mut().enumerate() {
                let av = vld1q_f32(a.as_ptr().add(i + j * 4));
                let bv = vld1q_f32(b.as_ptr().add(i + j * 4));
                *sum = vfmaq_f32(*sum, av, bv);
            }
            i += 16;
        }
        let paired = vaddq_f32(vaddq_f32(acc[0], acc[2]), vaddq_f32(acc[1], acc[3]));
        let mut sum = vaddvq_f32(paired);
        while i + 4 <= n {
            let products = vmulq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            sum += vgetq_lane_f32::<0>(products);
            sum += vgetq_lane_f32::<1>(products);
            sum += vgetq_lane_f32::<2>(products);
            sum += vgetq_lane_f32::<3>(products);
            i += 4;
        }
        while i < n {
            sum = a[i].mul_add(b[i], sum);
            i += 1;
        }
        sum
    }
}

/// SIMD element-wise add using NEON.
///
/// # Safety
///
/// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: the caller guarantees the CPU features; `add` asserts all
    // three lengths equal, and every 4-lane load/store is guarded by `i +
    // 4 <= n`.
    unsafe {
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            vst1q_f32(out.as_mut_ptr().add(i), vaddq_f32(av, bv));
            i += 4;
        }
        while i < n {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }
}

/// SIMD weighted accumulate using NEON FMA.
///
/// # Safety
///
/// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn weighted_add_neon(acc: &mut [f32], src: &[f32], weight: f32) {
    // SAFETY: the caller guarantees the CPU features; `weighted_add`
    // asserts equal lengths, and every 4-lane load/store is guarded by `i
    // + 4 <= n`.
    unsafe {
        let n = acc.len();
        let w = vdupq_n_f32(weight);
        let mut i = 0;

        while i + 4 <= n {
            let sv = vld1q_f32(src.as_ptr().add(i));
            let av = vld1q_f32(acc.as_ptr().add(i));
            vst1q_f32(acc.as_mut_ptr().add(i), vfmaq_f32(av, sv, w));
            i += 4;
        }
        while i < n {
            acc[i] += weight * src[i];
            i += 1;
        }
    }
}
