//! SIMD-accelerated Q8_0 dequantization and matrix multiplication kernels.
//!
//! Platform-specific kernels with runtime dispatch via `std::arch` and a
//! portable scalar fallback.  The dispatch function selects the fastest
//! available kernel at runtime based on CPU feature detection.
//!
//! ## kernels
//!
//! | arch     | feature | width | notes                        |
//! |----------|---------|-------|------------------------------|
//! | x86-64   | avx2+f16c | 256 | 8 f32 per op, 4 ops / block |
//! | aarch64  | neon    | 128   | 4 f32 per op, 8 ops / block  |
//! | aarch64  | neon+dotprod+fp16 | 128 | Q8 integer dots, tiled decode/prefill |
//! | fallback | (none)  | —     | scalar, matches original     |
//!
//! One Q8_0 block = 34 bytes (2-byte f16 scale + 32 i8 quants) → 32 f32 values.

use crate::quant::{QuantizedWeight, Q8_0_BLOCK_SIZE, Q8_0_TYPE_SIZE};
use half::f16;
use rayon::prelude::*;
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

// Four-way row splitting becomes beneficial well below the old 8M-MAC gate
// now that decode uses cheaper Q8 × Q8 dots. This includes Gemma's Q/O
// projections while keeping its tiny 256-wide PLE gates serial.
const PARALLEL_Q8_DECODE_MIN_WORK: usize = 1_048_576;

/// Whether this CPU can execute the interleaved Q8_0 decode kernel.
#[inline]
#[doc(hidden)]
pub fn interleaved_q8_0_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
    }
    #[cfg(target_arch = "aarch64")]
    {
        is_aarch64_feature_detected!("dotprod") && is_aarch64_feature_detected!("fp16")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Whether this CPU can execute the 16-output packed Q8_0 decode kernel.
#[inline]
pub(crate) fn packed_q8_0_vnni_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
// ---------------------------------------------------------------------------
// public dispatch
// ---------------------------------------------------------------------------

/// Dequantize `blocks_per_row` consecutive Q8_0 blocks starting at
/// `block_start` from `data` into `dst`.
///
/// Automatically selects the fastest available kernel for the current CPU.
/// Called from `QuantizedWeight::dequantize_row`.
#[inline]
pub fn dequantize_q8_0_row(
    data: &[u8],
    block_start: usize,
    blocks_per_row: usize,
    dst: &mut [f32],
) {
    let end_block = block_start
        .checked_add(blocks_per_row)
        .expect("q8_0 block range overflow");
    let required_bytes = end_block
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("q8_0 byte range overflow");
    let required_values = blocks_per_row
        .checked_mul(Q8_0_BLOCK_SIZE)
        .expect("q8_0 destination length overflow");
    assert!(
        required_bytes <= data.len(),
        "q8_0 source too short: need {required_bytes} bytes for blocks {block_start}..{end_block}, got {}",
        data.len()
    );
    assert!(
        required_values <= dst.len(),
        "q8_0 destination too short: need {required_values} floats, got {}",
        dst.len()
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            // SAFETY: features checked; the asserts above bound both the
            // source block range and the destination.
            unsafe {
                return x86_64::dequantize_row_avx2(data, block_start, blocks_per_row, dst);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: feature checked; the asserts above bound both the
            // source block range and the destination.
            unsafe {
                return aarch64::dequantize_row_neon(data, block_start, blocks_per_row, dst);
            }
        }
    }
    dequantize_row_scalar(data, block_start, blocks_per_row, dst);
}

// ---------------------------------------------------------------------------
// in-place / fused dispatch
// ---------------------------------------------------------------------------

/// RMS normalization into a pre-allocated destination: `dst = x * weight / rms(x)`.
/// `dst` must have the same length as `x` and `weight`.
#[inline]
pub fn rms_norm_into(x: &[f32], weight: &[f32], eps: f32, dst: &mut [f32]) {
    let n = x.len();
    assert!(!x.is_empty(), "rms_norm_into requires a non-empty input");
    assert_eq!(dst.len(), n, "rms_norm_into destination length mismatch");
    assert_eq!(weight.len(), n, "rms_norm_into weight length mismatch");
    assert!(
        eps.is_finite() && eps >= 0.0,
        "rms_norm_into requires finite eps >= 0"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features checked; lengths asserted equal above.
            unsafe {
                return x86_64::rms_norm_into_avx2(x, weight, eps, dst);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") {
        // Match CpuTensor::rms_norm's NEON reduction order exactly. A scalar
        // reduction here perturbs later activation quantization in planned decode.
        let rstd = (sum_squares(x) / n as f32 + eps).sqrt().recip();
        scale_weight_mul(x, rstd, weight, dst);
        return;
    }
    rms_norm_into_scalar(x, weight, eps, dst);
}

/// Preserve the established Q8 fast-decode normalization order on ARM.
/// Its scalar reduction is part of existing generation/reproduction behavior;
/// planned K decode instead uses the tensor-reference NEON order above.
#[inline]
pub(crate) fn rms_norm_q8_decode_into(x: &[f32], weight: &[f32], eps: f32, dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        assert!(!x.is_empty());
        assert_eq!(x.len(), weight.len());
        assert_eq!(x.len(), dst.len());
        assert!(eps.is_finite() && eps >= 0.0);
        let sum: f32 = x.iter().map(|value| value * value).sum();
        let rstd = (sum / x.len() as f32 + eps).sqrt().recip();
        // Vectorize the apply step without changing the reduction or multiply order.
        scale_weight_mul(x, rstd, weight, dst);
    }
    #[cfg(not(target_arch = "aarch64"))]
    rms_norm_into(x, weight, eps, dst);
}

/// Fused SiLU multiply: `dst[i] = silu(gate[i]) * up[i]`.
/// All three slices must have the same length.
#[inline]
pub fn silu_mul_into(gate: &[f32], up: &[f32], dst: &mut [f32]) {
    let n = gate.len();
    assert_eq!(up.len(), n, "silu_mul_into input length mismatch");
    assert_eq!(dst.len(), n, "silu_mul_into destination length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features checked; lengths asserted equal above.
            unsafe {
                return x86_64::silu_mul_into_avx2(gate, up, dst);
            }
        }
    }
    silu_mul_into_scalar(gate, up, dst);
}

// ---------------------------------------------------------------------------
// scalar fallback (always compiled)
// ---------------------------------------------------------------------------

fn rms_norm_into_scalar(x: &[f32], weight: &[f32], eps: f32, dst: &mut [f32]) {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rstd = (sum_sq / n as f32 + eps).sqrt().recip();
    for i in 0..n {
        dst[i] = x[i] * rstd * weight[i];
    }
}

fn silu_mul_into_scalar(gate: &[f32], up: &[f32], dst: &mut [f32]) {
    for i in 0..gate.len() {
        let g = gate[i];
        dst[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

fn dequantize_row_scalar(data: &[u8], block_start: usize, blocks_per_row: usize, dst: &mut [f32]) {
    for b in 0..blocks_per_row {
        let byte_offset = (block_start + b) * Q8_0_TYPE_SIZE;

        let d_bits = u16::from_le_bytes(data[byte_offset..byte_offset + 2].try_into().unwrap());
        let d = f16::from_bits(d_bits).to_f32();

        let out_offset = b * Q8_0_BLOCK_SIZE;
        for j in 0..Q8_0_BLOCK_SIZE {
            let q = data[byte_offset + 2 + j] as i8;
            dst[out_offset + j] = q as f32 * d;
        }
    }
}

// ---------------------------------------------------------------------------
// Fast exp (softmax-class kernels)
// ---------------------------------------------------------------------------

/// Elementwise `exp(src) -> dst` for softmax-style workloads.
///
/// Vectorized path: AVX2+FMA polynomial approximation with two-part ln2
/// range reduction (Cephes expf coefficients, degree-6 minimax on
/// [-ln2/2, +ln2/2]). Measured relative error vs the libm reference is
/// ~1e-7 — at or below one f32 ulp of typical results, i.e. fp32
/// rounding noise. Inputs are clamped to >= -88 before evaluation:
/// exp(-88) ≈ 6e-39 is already indistinguishable from zero after
/// probability normalization, and the clamp keeps masked rows (additive
/// f32::MIN bias) well-defined instead of overflowing the exponent-bit
/// scaling.
///
/// The scalar fallback is the exact libm `expf`, so non-AVX2 hosts keep
/// bit-identical reference behavior. Callers must treat the AVX2 output
/// as numerically-equivalent-but-not-bit-identical and validate through
/// their own error ladders.
pub fn fast_exp_into(src: &[f32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "fast_exp_into: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: feature set checked immediately above via runtime detection.
            unsafe { x86_64::fast_exp_raw(src.as_ptr(), dst.as_mut_ptr(), src.len()) };
            return;
        }
    }
    fast_exp_scalar(src, dst);
}

/// In-place variant of [`fast_exp_into`]. The AVX2 body loads each
/// 8-lane block completely before storing it, so same-slice operation is
/// well-defined; the scalar tail processes elements strictly forward.
pub fn fast_exp_in_place(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let ptr = v.as_mut_ptr();
            let n = v.len();
            // Safety: distinct in-borrow pointers into one slice; the core
            // never reads a lane after writing it.
            unsafe { x86_64::fast_exp_raw(ptr, ptr, n) };
            return;
        }
    }
    for d in v.iter_mut() {
        *d = d.max(-88.0).exp();
    }
}

pub(crate) fn fast_exp_scalar(src: &[f32], dst: &mut [f32]) {
    const CLAMP_LO: f32 = -88.0;
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s.max(CLAMP_LO).exp();
    }
}

// ---------------------------------------------------------------------------
// x86-64 AVX2 kernel
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86_64;

// ---------------------------------------------------------------------------
// aarch64 NEON kernel
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod q8_decode;
pub(crate) use q8_decode::*;

// ---------------------------------------------------------------------------
// element-wise SIMD helpers (rms_norm, elemul, silu, rope)
// ---------------------------------------------------------------------------

/// SIMD sum of squares reduction: `Σ x[i]²`.
///
/// Used by rms_norm to compute mean_sq before the sqrt+recip step.
#[inline]
pub(crate) fn sum_squares(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features checked; the kernel only reads `x`.
            return unsafe { x86_64::sum_squares_avx2(x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: feature checked; the kernel only reads `x`.
            return unsafe { aarch64::sum_squares_neon(x) };
        }
    }
    x.iter().map(|v| v * v).sum()
}

/// SIMD element-wise multiply: `out[i] = a[i] * b[i]`.
#[inline]
pub(crate) fn elemul(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "elemul input length mismatch");
    assert_eq!(a.len(), out.len(), "elemul output length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: features checked; lengths asserted equal above.
            unsafe {
                return x86_64::elemul_avx2(a, b, out);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: feature checked; lengths asserted equal above.
            unsafe {
                return aarch64::elemul_neon(a, b, out);
            }
        }
    }
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// SIMD dot product: `Σ a[i] · b[i]`.
#[inline]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features checked; lengths asserted equal above.
            unsafe {
                return x86_64::dot_product_avx2(a, b);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: feature checked; lengths asserted equal above.
            unsafe {
                return aarch64::dot_product_neon(a, b);
            }
        }
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// SIMD dot product between an F32 row and an F16 cache row.
#[inline]
pub(crate) fn dot_product_f16(a: &[f32], b: &[f16]) -> f32 {
    assert_eq!(a.len(), b.len(), "mixed dot product length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            // SAFETY: features checked; lengths asserted equal above.
            unsafe {
                return x86_64::dot_product_f16_avx2(a, b);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("fp16") {
        // SAFETY: lengths checked above; NEON is baseline on AArch64.
        return unsafe { aarch64::dot_product_f16_ordered(a, b) };
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y.to_f32()).sum()
}

/// SIMD weighted accumulate: `acc[i] += weight * src[i]`.
#[inline]
pub(crate) fn weighted_add(acc: &mut [f32], src: &[f32], weight: f32) {
    assert_eq!(acc.len(), src.len(), "weighted add length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features checked; lengths asserted equal above.
            unsafe {
                return x86_64::weighted_add_avx2(acc, src, weight);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: feature checked; lengths asserted equal above.
            unsafe {
                return aarch64::weighted_add_neon(acc, src, weight);
            }
        }
    }
    for i in 0..acc.len() {
        acc[i] += weight * src[i];
    }
}

/// SIMD weighted accumulate from an F16 cache row.
#[inline]
pub(crate) fn weighted_add_f16(acc: &mut [f32], src: &[f16], weight: f32) {
    assert_eq!(acc.len(), src.len(), "mixed weighted add length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            // SAFETY: features checked; lengths asserted equal above.
            unsafe {
                return x86_64::weighted_add_f16_avx2(acc, src, weight);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("fp16") {
        // SAFETY: lengths checked above; NEON is baseline on AArch64.
        return unsafe { aarch64::weighted_add_f16_neon(acc, src, weight) };
    }
    for i in 0..acc.len() {
        acc[i] += weight * src[i].to_f32();
    }
}

/// SIMD in-place addition: `dst[i] += src[i]`.
#[inline]
pub(crate) fn add_assign(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len(), "add-assign length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature checked; lengths asserted equal above.
            unsafe {
                return x86_64::add_assign_avx2(dst, src);
            }
        }
    }
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

/// SIMD element-wise addition: `out[i] = a[i] + b[i]`.
#[inline]
pub(crate) fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "add input length mismatch");
    assert_eq!(a.len(), out.len(), "add output length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: features checked; lengths asserted equal above.
            unsafe {
                return x86_64::add_avx2(a, b, out);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: feature checked; lengths asserted equal above.
            unsafe {
                return aarch64::add_neon(a, b, out);
            }
        }
    }
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// SIMD scale-and-weight: `out[i] = x[i] * scale * weight[i]`.
///
/// Used by rms_norm for the element-wise apply step after computing rstd.
#[inline]
pub(crate) fn scale_weight_mul(x: &[f32], scale: f32, weight: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), weight.len(), "scale/weight input length mismatch");
    assert_eq!(x.len(), out.len(), "scale/weight output length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: features checked; lengths asserted equal above.
            unsafe {
                return x86_64::scale_weight_mul_avx2(x, scale, weight, out);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: feature checked; lengths asserted equal above.
            unsafe {
                return aarch64::scale_weight_mul_neon(x, scale, weight, out);
            }
        }
    }
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

// ---------------------------------------------------------------------------
// softmax
// ---------------------------------------------------------------------------

/// SIMD prefix softmax: `row[i] = exp(row[i] - max) / Σ exp(row[j] - max)`.
/// SIMD split-half RoPE: `x[i0] = x[i0]*c - x[i1]*s; x[i1] = x[i0]*s + x[i1]*c`
/// where `i0 = head_offset + d`, `i1 = head_offset + d + head_dim/2`.
#[inline]
pub(crate) fn rope_split_half(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "RoPE head dimension must be positive and even"
    );
    let expected_x = n_heads
        .checked_mul(head_dim)
        .expect("RoPE input length overflow");
    assert_eq!(x.len(), expected_x, "RoPE input length mismatch");
    assert_eq!(cos.len(), half, "RoPE cosine length mismatch");
    assert_eq!(sin.len(), half, "RoPE sine length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features checked; lengths asserted above.
            unsafe {
                return x86_64::rope_split_half_avx2(x, n_heads, head_dim, cos, sin);
            }
        }
    }
    for h in 0..n_heads {
        let off = h * head_dim;
        for d in 0..half {
            let i0 = off + d;
            let i1 = off + d + half;
            let x0 = x[i0];
            let x1 = x[i1];
            x[i0] = x0.mul_add(cos[d], -(x1 * sin[d]));
            x[i1] = x0.mul_add(sin[d], x1 * cos[d]);
        }
    }
}

/// SIMD adjacent-pair RoPE (Llama layout): for each head and pair `d`,
/// `x[2d] = x[2d]*c - x[2d+1]*s; x[2d+1] = x[2d]*s + x[2d+1]*c` with the
/// products and fused adds of [`rope_adjacent_pair_scalar`]. Every SIMD path
/// is bit-identical to that scalar loop (same `mul_add` rounding per lane),
/// which the tests below pin including odd tails.
#[inline]
pub(crate) fn rope_adjacent_pair(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "RoPE head dimension must be positive and even"
    );
    let expected_x = n_heads
        .checked_mul(head_dim)
        .expect("RoPE input length overflow");
    assert_eq!(x.len(), expected_x, "RoPE input length mismatch");
    assert_eq!(cos.len(), half, "RoPE cosine length mismatch");
    assert_eq!(sin.len(), half, "RoPE sine length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features checked; lengths asserted above.
            unsafe {
                return x86_64::rope_adjacent_pair_avx2(x, n_heads, head_dim, cos, sin);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: feature checked; lengths asserted above.
            unsafe {
                return aarch64::rope_adjacent_pair_neon(x, n_heads, head_dim, cos, sin);
            }
        }
    }
    rope_adjacent_pair_scalar(x, n_heads, head_dim, cos, sin);
}

/// Scalar adjacent-pair RoPE: the reference the SIMD kernels must match.
pub(crate) fn rope_adjacent_pair_scalar(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    for h in 0..n_heads {
        let off = h * head_dim;
        rope_adjacent_pair_scalar_head(&mut x[off..off + head_dim], 0, cos, sin);
    }
}

/// Scalar adjacent-pair RoPE over one head, pairs `start..head_dim/2`
/// (the SIMD kernels' tail).
#[inline]
fn rope_adjacent_pair_scalar_head(head: &mut [f32], start: usize, cos: &[f32], sin: &[f32]) {
    for d in start..head.len() / 2 {
        let x0 = head[2 * d];
        let x1 = head[2 * d + 1];
        head[2 * d] = x0.mul_add(cos[d], -(x1 * sin[d]));
        head[2 * d + 1] = x0.mul_add(sin[d], x1 * cos[d]);
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
