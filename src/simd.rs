//! SIMD-accelerated Q8_0 dequantization kernels.
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
//! | fallback | (none)  | —     | scalar, matches original     |
//!
//! One Q8_0 block = 34 bytes (2-byte f16 scale + 32 i8 quants) → 32 f32 values.

use crate::quant::{Q8_0_BLOCK_SIZE, Q8_0_TYPE_SIZE};
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
pub(crate) fn interleaved_q8_0_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
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
    // Safety: the arch-specific kernels are only called when the
    // corresponding CPU feature is detected at runtime.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            unsafe {
                return x86_64::dequantize_row_avx2(data, block_start, blocks_per_row, dst);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
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
            unsafe {
                return x86_64::rms_norm_into_avx2(x, weight, eps, dst);
            }
        }
    }
    rms_norm_into_scalar(x, weight, eps, dst);
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
            unsafe {
                return x86_64::silu_mul_into_avx2(gate, up, dst);
            }
        }
    }
    silu_mul_into_scalar(gate, up, dst);
}

/// SiLU in-place: `dst[i] = xxx[i] / (1.0 + exp(-xxx[i]))`.
/// Reads from `src`, writes to `dst` (may alias).
#[inline]
pub fn silu_into(src: &[f32], dst: &mut [f32]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "silu_into destination length mismatch"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return x86_64::silu_into_avx2(src, dst);
            }
        }
    }
    silu_into_scalar(src, dst);
}

/// Fused RMS norm + residual add: `dst = (x * weight / rms(x)) + residual`.
/// `x`, `weight`, `residual`, and `dst` must all have the same length.
#[inline]
pub fn rms_norm_residual_into(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    residual: &[f32],
    dst: &mut [f32],
) {
    let n = x.len();
    assert!(
        !x.is_empty(),
        "rms_norm_residual_into requires a non-empty input"
    );
    assert_eq!(
        weight.len(),
        n,
        "rms_norm_residual_into weight length mismatch"
    );
    assert_eq!(
        residual.len(),
        n,
        "rms_norm_residual_into residual length mismatch"
    );
    assert_eq!(
        dst.len(),
        n,
        "rms_norm_residual_into destination length mismatch"
    );
    assert!(
        eps.is_finite() && eps >= 0.0,
        "rms_norm_residual_into requires finite eps >= 0"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return x86_64::rms_norm_residual_into_avx2(x, weight, eps, residual, dst);
            }
        }
    }
    rms_norm_residual_into_scalar(x, weight, eps, residual, dst);
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

fn silu_into_scalar(src: &[f32], dst: &mut [f32]) {
    for i in 0..src.len() {
        dst[i] = src[i] / (1.0 + (-src[i]).exp());
    }
}

fn rms_norm_residual_into_scalar(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    residual: &[f32],
    dst: &mut [f32],
) {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rstd = (sum_sq / n as f32 + eps).sqrt().recip();
    for i in 0..n {
        dst[i] = x[i] * rstd * weight[i] + residual[i];
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
// x86-64 AVX2 kernel
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "f16c")]
    unsafe fn f16_bits_to_f32(bits: u16) -> f32 {
        // Safety: Caller must ensure the required x86 feature set (`f16c`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
    }

    // -- fused / in-place SIMD kernels -------------------------------

    /// SIMD RMS norm into pre-allocated dst using AVX2+FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,fma`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn rms_norm_into_avx2(x: &[f32], weight: &[f32], eps: f32, dst: &mut [f32]) {
        unsafe {
            let n = x.len();
            // 1. sum of squares
            let mut ss0 = _mm256_setzero_ps();
            let mut ss1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= n {
                let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
                let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
                ss0 = _mm256_fmadd_ps(v0, v0, ss0);
                ss1 = _mm256_fmadd_ps(v1, v1, ss1);
                i += 16;
            }
            while i + 8 <= n {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                ss0 = _mm256_fmadd_ps(v, v, ss0);
                i += 8;
            }
            let acc = _mm256_add_ps(ss0, ss1);
            let low = _mm256_castps256_ps128(acc);
            let high = _mm256_extractf128_ps::<1>(acc);
            let sum128 = _mm_add_ps(low, high);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let mut sum_sq = _mm_cvtss_f32(sum128);
            while i < n {
                sum_sq += x[i] * x[i];
                i += 1;
            }

            let rstd = _mm256_set1_ps((sum_sq / n as f32 + eps).sqrt().recip());

            // 2. dst[i] = x[i] * rstd * weight[i]
            i = 0;
            while i + 8 <= n {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                let wv = _mm256_loadu_ps(weight.as_ptr().add(i));
                let r = _mm256_mul_ps(_mm256_mul_ps(xv, rstd), wv);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), r);
                i += 8;
            }
            while i < n {
                dst[i] = x[i] * (sum_sq / n as f32 + eps).sqrt().recip() * weight[i];
                i += 1;
            }
        }
    }

    /// SIMD fused SiLU * up using AVX2+FMA: `dst[i] = silu(gate[i]) * up[i]`.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,fma`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn silu_mul_into_avx2(gate: &[f32], up: &[f32], dst: &mut [f32]) {
        unsafe {
            let n = gate.len();
            let one = _mm256_set1_ps(1.0);
            let mut i = 0;
            while i + 8 <= n {
                let g = _mm256_loadu_ps(gate.as_ptr().add(i));
                let u = _mm256_loadu_ps(up.as_ptr().add(i));
                // silu(g) = g / (1 + exp(-g))
                let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
                let exp_neg = exp_ps(neg_g);
                let denom = _mm256_add_ps(one, exp_neg);
                let silu = _mm256_div_ps(g, denom);
                let r = _mm256_mul_ps(silu, u);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), r);
                i += 8;
            }
            while i < n {
                let g = gate[i];
                dst[i] = (g / (1.0 + (-g).exp())) * up[i];
                i += 1;
            }
        }
    }

    /// SIMD SiLU into pre-allocated dst using AVX2+FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,fma`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn silu_into_avx2(src: &[f32], dst: &mut [f32]) {
        unsafe {
            let n = src.len();
            let one = _mm256_set1_ps(1.0);
            let mut i = 0;
            while i + 8 <= n {
                let x = _mm256_loadu_ps(src.as_ptr().add(i));
                let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
                let exp_neg = exp_ps(neg_x);
                let denom = _mm256_add_ps(one, exp_neg);
                let r = _mm256_div_ps(x, denom);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), r);
                i += 8;
            }
            while i < n {
                dst[i] = src[i] / (1.0 + (-src[i]).exp());
                i += 1;
            }
        }
    }

    /// SIMD fused RMS norm + residual add: `dst[i] = (x[i] * rstd * weight[i]) + residual[i]`.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,fma`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn rms_norm_residual_into_avx2(
        x: &[f32],
        weight: &[f32],
        eps: f32,
        residual: &[f32],
        dst: &mut [f32],
    ) {
        unsafe {
            let n = x.len();
            // sum of squares
            let mut ss0 = _mm256_setzero_ps();
            let mut ss1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= n {
                let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
                let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
                ss0 = _mm256_fmadd_ps(v0, v0, ss0);
                ss1 = _mm256_fmadd_ps(v1, v1, ss1);
                i += 16;
            }
            while i + 8 <= n {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                ss0 = _mm256_fmadd_ps(v, v, ss0);
                i += 8;
            }
            let acc = _mm256_add_ps(ss0, ss1);
            let low = _mm256_castps256_ps128(acc);
            let high = _mm256_extractf128_ps::<1>(acc);
            let sum128 = _mm_add_ps(low, high);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let mut sum_sq = _mm_cvtss_f32(sum128);
            while i < n {
                sum_sq += x[i] * x[i];
                i += 1;
            }

            let rstd = _mm256_set1_ps((sum_sq / n as f32 + eps).sqrt().recip());

            // dst[i] = x[i] * rstd * weight[i] + residual[i]
            i = 0;
            while i + 8 <= n {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                let wv = _mm256_loadu_ps(weight.as_ptr().add(i));
                let rv = _mm256_loadu_ps(residual.as_ptr().add(i));
                let normed = _mm256_mul_ps(_mm256_mul_ps(xv, rstd), wv);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_add_ps(normed, rv));
                i += 8;
            }
            while i < n {
                dst[i] = x[i] * (sum_sq / n as f32 + eps).sqrt().recip() * weight[i] + residual[i];
                i += 1;
            }
        }
    }

    /// AVX2-accelerated Q8_0 row dequantization.
    ///
    /// Processes 32 quants per block in 4 batches of 8 f32 values using
    /// 256-bit SIMD registers.  One block per iteration.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and F16C are supported (checked by dispatch above).
    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn dequantize_row_avx2(
        data: &[u8],
        block_start: usize,
        blocks_per_row: usize,
        dst: &mut [f32],
    ) {
        unsafe {
            for b in 0..blocks_per_row {
                let byte_offset = (block_start + b) * Q8_0_TYPE_SIZE;
                let base_ptr = data.as_ptr().add(byte_offset);

                // -- scale: load 2-byte f16, convert to f32, broadcast ---------
                let d_bits = u16::from_le_bytes(*(base_ptr as *const [u8; 2]));
                let d = f16_bits_to_f32(d_bits);
                let d_vec = _mm256_set1_ps(d);

                // -- quants: load 32 i8 values as 256-bit vector --------------
                let quants_ptr = base_ptr.add(2) as *const i8;
                let quants = _mm256_loadu_si256(quants_ptr as *const __m256i);

                // split into two 128-bit halves
                let low128 = _mm256_castsi256_si128(quants);
                let high128 = _mm256_extracti128_si256::<1>(quants);

                let out_offset = b * Q8_0_BLOCK_SIZE;
                let out_ptr = dst.as_mut_ptr().add(out_offset);

                // -- batch 0: bytes 0..7 of low128 ----------------------------
                let q0_i32 = _mm256_cvtepi8_epi32(low128);
                let q0_f32 = _mm256_cvtepi32_ps(q0_i32);
                _mm256_storeu_ps(out_ptr, _mm256_mul_ps(q0_f32, d_vec));

                // -- batch 1: bytes 8..15 of low128 ---------------------------
                let q1_i32 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(low128, 8));
                let q1_f32 = _mm256_cvtepi32_ps(q1_i32);
                _mm256_storeu_ps(out_ptr.add(8), _mm256_mul_ps(q1_f32, d_vec));

                // -- batch 2: bytes 0..7 of high128 ---------------------------
                let q2_i32 = _mm256_cvtepi8_epi32(high128);
                let q2_f32 = _mm256_cvtepi32_ps(q2_i32);
                _mm256_storeu_ps(out_ptr.add(16), _mm256_mul_ps(q2_f32, d_vec));

                // -- batch 3: bytes 8..15 of high128 --------------------------
                let q3_i32 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(high128, 8));
                let q3_f32 = _mm256_cvtepi32_ps(q3_i32);
                _mm256_storeu_ps(out_ptr.add(24), _mm256_mul_ps(q3_f32, d_vec));
            }
        }
    }

    /// Q8_0 × Q8_0 dot product using AVX2 integer pairwise multiply-add.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and F16C are supported.
    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn matmul_q8_0_decode_avx2(
        x: &[u8],
        data: &[u8],
        _out_features: usize,
        blocks_per_row: usize,
        out: &mut [f32],
    ) {
        unsafe {
            let ones = _mm256_set1_epi16(1);
            for (row, out_val) in out.iter_mut().enumerate() {
                let row_start = row * blocks_per_row;
                let mut acc = _mm256_setzero_ps();

                for b in 0..blocks_per_row {
                    let byte_offset = (row_start + b) * Q8_0_TYPE_SIZE;
                    let x_offset = b * Q8_0_TYPE_SIZE;
                    let weight_scale = f16_bits_to_f32(u16::from_le_bytes(
                        *(data.as_ptr().add(byte_offset) as *const [u8; 2]),
                    ));
                    let input_scale = f16_bits_to_f32(u16::from_le_bytes(
                        *(x.as_ptr().add(x_offset) as *const [u8; 2]),
                    ));
                    let weights =
                        _mm256_loadu_si256(data.as_ptr().add(byte_offset + 2) as *const __m256i);
                    let input = _mm256_loadu_si256(x.as_ptr().add(x_offset + 2) as *const __m256i);
                    let abs_weights = _mm256_abs_epi8(weights);
                    let signed_input = _mm256_sign_epi8(input, weights);
                    let pair16 = _mm256_maddubs_epi16(abs_weights, signed_input);
                    let pair32 = _mm256_madd_epi16(pair16, ones);
                    let products = _mm256_cvtepi32_ps(pair32);
                    let scale = _mm256_set1_ps(weight_scale * input_scale);
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(products, scale));
                }

                let low = _mm256_castps256_ps128(acc);
                let high = _mm256_extractf128_ps::<1>(acc);
                let sum128 = _mm_add_ps(low, high);
                let sum128 = _mm_hadd_ps(sum128, sum128);
                let sum128 = _mm_hadd_ps(sum128, sum128);
                *out_val = _mm_cvtss_f32(sum128);
            }
        }
    }

    /// Q8_0 × Q8_0 dot product using AVX-512 VNNI's packed integer dot.
    /// Processes 8 output rows simultaneously (up from 4) for 2× ILP.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,f16c,fma,avx512vl,avx512vnni`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,f16c,fma,avx512vl,avx512vnni")]
    pub unsafe fn matmul_q8_0_decode_avx512_vnni(
        x: &[u8],
        data: &[u8],
        _out_features: usize,
        blocks_per_row: usize,
        out: &mut [f32],
    ) {
        unsafe {
            let grouped_rows = out.len() / 8 * 8;
            let grouped_blocks = blocks_per_row / 4 * 4;

            for row_start in (0..grouped_rows).step_by(8) {
                let mut acc = [_mm256_setzero_ps(); 8];
                for block_start in (0..grouped_blocks).step_by(4) {
                    // Load 4 input blocks (shared across all 8 output rows)
                    let mut input_quants = [_mm256_setzero_si256(); 4];
                    let mut input_scales = [0.0f32; 4];
                    for bl in 0..4 {
                        let x_off = (block_start + bl) * Q8_0_TYPE_SIZE;
                        input_scales[bl] = f16_bits_to_f32(u16::from_le_bytes(
                            *(x.as_ptr().add(x_off) as *const [u8; 2]),
                        ));
                        input_quants[bl] =
                            _mm256_loadu_si256(x.as_ptr().add(x_off + 2) as *const __m256i);
                    }
                    // Prefetch weight data for next block group (4 blocks = 136 bytes ahead)
                    if block_start + 4 < grouped_blocks {
                        let pf_offset =
                            (row_start * blocks_per_row + block_start + 4) * Q8_0_TYPE_SIZE;
                        #[cfg(target_arch = "x86_64")]
                        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                            data.as_ptr().add(pf_offset) as *const i8,
                        );
                    }
                    for (lane, lane_acc) in acc.iter_mut().enumerate() {
                        for bl in 0..4 {
                            let block = block_start + bl;
                            let byte_off =
                                ((row_start + lane) * blocks_per_row + block) * Q8_0_TYPE_SIZE;
                            let weight_scale = f16_bits_to_f32(u16::from_le_bytes(
                                *(data.as_ptr().add(byte_off) as *const [u8; 2]),
                            ));
                            let weights = _mm256_loadu_si256(
                                data.as_ptr().add(byte_off + 2) as *const __m256i
                            );
                            let abs_w = _mm256_abs_epi8(weights);
                            let signed_in = _mm256_sign_epi8(input_quants[bl], weights);
                            let sums =
                                _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs_w, signed_in);
                            *lane_acc = _mm256_fmadd_ps(
                                _mm256_cvtepi32_ps(sums),
                                _mm256_set1_ps(weight_scale * input_scales[bl]),
                                *lane_acc,
                            );
                        }
                    }
                }
                // Tail blocks (not a multiple of 4)
                for b in grouped_blocks..blocks_per_row {
                    let x_off = b * Q8_0_TYPE_SIZE;
                    let in_scale = f16_bits_to_f32(u16::from_le_bytes(
                        *(x.as_ptr().add(x_off) as *const [u8; 2]),
                    ));
                    let in_q = _mm256_loadu_si256(x.as_ptr().add(x_off + 2) as *const __m256i);
                    for (lane, lane_acc) in acc.iter_mut().enumerate() {
                        let byte_off = ((row_start + lane) * blocks_per_row + b) * Q8_0_TYPE_SIZE;
                        let w_scale = f16_bits_to_f32(u16::from_le_bytes(
                            *(data.as_ptr().add(byte_off) as *const [u8; 2]),
                        ));
                        let w =
                            _mm256_loadu_si256(data.as_ptr().add(byte_off + 2) as *const __m256i);
                        let abs_w = _mm256_abs_epi8(w);
                        let signed_in = _mm256_sign_epi8(in_q, w);
                        let sums = _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs_w, signed_in);
                        *lane_acc = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(sums),
                            _mm256_set1_ps(w_scale * in_scale),
                            *lane_acc,
                        );
                    }
                }
                // Horizontal sum and store
                for lane in 0..8 {
                    let a = acc[lane];
                    let low = _mm256_castps256_ps128(a);
                    let high = _mm256_extractf128_ps::<1>(a);
                    let s = _mm_add_ps(low, high);
                    let s = _mm_hadd_ps(s, s);
                    let s = _mm_hadd_ps(s, s);
                    out[row_start + lane] = _mm_cvtss_f32(s);
                }
            }
            // Scalar tail for remaining rows
            for (row, out_val) in out.iter_mut().enumerate().skip(grouped_rows) {
                let row_start = row * blocks_per_row;
                let mut acc = _mm256_setzero_ps();
                for b in 0..blocks_per_row {
                    let byte_off = (row_start + b) * Q8_0_TYPE_SIZE;
                    let x_off = b * Q8_0_TYPE_SIZE;
                    let w_scale = f16_bits_to_f32(u16::from_le_bytes(
                        *(data.as_ptr().add(byte_off) as *const [u8; 2]),
                    ));
                    let in_scale = f16_bits_to_f32(u16::from_le_bytes(
                        *(x.as_ptr().add(x_off) as *const [u8; 2]),
                    ));
                    let w = _mm256_loadu_si256(data.as_ptr().add(byte_off + 2) as *const __m256i);
                    let in_q = _mm256_loadu_si256(x.as_ptr().add(x_off + 2) as *const __m256i);
                    let abs_w = _mm256_abs_epi8(w);
                    let signed_in = _mm256_sign_epi8(in_q, w);
                    let sums = _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs_w, signed_in);
                    acc = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(sums),
                        _mm256_set1_ps(w_scale * in_scale),
                        acc,
                    );
                }
                let low = _mm256_castps256_ps128(acc);
                let high = _mm256_extractf128_ps::<1>(acc);
                let s = _mm_add_ps(low, high);
                let s = _mm_hadd_ps(s, s);
                let s = _mm_hadd_ps(s, s);
                *out_val = _mm_cvtss_f32(s);
            }
        }
    }

    /// Q8_0 × Q8_0 decode over a 16-output packed weight layout.
    ///
    /// Each VNNI lane accumulates the same four input coordinates for 16
    /// output rows. Eight vector accumulators preserve the row-contiguous
    /// kernel's floating-point reduction order exactly.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,f16c,fma,avx512f,avx512bw,avx512vnni`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,f16c,fma,avx512f,avx512bw,avx512vnni")]
    pub unsafe fn matmul_q8_0_decode_packed16_avx512_vnni(
        x: &[u8],
        data: &[u8],
        blocks_per_row: usize,
        out: &mut [f32],
        global_row_offset: usize,
    ) {
        unsafe {
            use crate::quant::{VNNI_BLOCK_RECORD_SIZE, VNNI_OUT_TILE};

            debug_assert!(global_row_offset.is_multiple_of(VNNI_OUT_TILE));
            let first_tile = global_row_offset / VNNI_OUT_TILE;

            for (local_tile, out_tile) in out.chunks_mut(VNNI_OUT_TILE).enumerate() {
                let tile = first_tile + local_tile;
                let mut accumulators = [_mm512_setzero_ps(); Q8_0_BLOCK_SIZE / 4];

                for block in 0..blocks_per_row {
                    let record = (tile * blocks_per_row + block) * VNNI_BLOCK_RECORD_SIZE;
                    let scales_offset = record + VNNI_OUT_TILE * Q8_0_BLOCK_SIZE;
                    let weight_scales = _mm512_cvtph_ps(_mm256_loadu_si256(
                        data.as_ptr().add(scales_offset) as *const __m256i,
                    ));
                    let input_offset = block * Q8_0_TYPE_SIZE;
                    let input_scale = f16_bits_to_f32(u16::from_le_bytes(
                        *(x.as_ptr().add(input_offset) as *const [u8; 2]),
                    ));
                    let combined_scales = _mm512_mul_ps(weight_scales, _mm512_set1_ps(input_scale));

                    for (group, accumulator) in accumulators.iter_mut().enumerate() {
                        // The public dispatcher verifies one complete encoded
                        // activation row. `group < 8`, so this unaligned load is
                        // contained in the current 34-byte Q8_0 block.
                        let input_group = core::ptr::read_unaligned(
                            x.as_ptr().add(input_offset + 2 + group * 4) as *const i32,
                        );
                        let input = _mm512_set1_epi32(input_group);
                        let weights = _mm512_loadu_si512(
                            data.as_ptr().add(record + group * VNNI_OUT_TILE * 4) as *const __m512i,
                        );
                        let absolute_weights = _mm512_abs_epi8(weights);
                        let zero = _mm512_setzero_si512();
                        let negative_weights = _mm512_cmpgt_epi8_mask(zero, weights);
                        let negated_input = _mm512_sub_epi8(zero, input);
                        let signed_input =
                            _mm512_mask_mov_epi8(input, negative_weights, negated_input);
                        let products = _mm512_dpbusd_epi32(
                            _mm512_setzero_si512(),
                            absolute_weights,
                            signed_input,
                        );
                        *accumulator = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(products),
                            combined_scales,
                            *accumulator,
                        );
                    }
                }

                // Match the two 128-bit horizontal adds used by the existing
                // eight-lane row-contiguous kernel.
                let sum04 = _mm512_add_ps(accumulators[0], accumulators[4]);
                let sum15 = _mm512_add_ps(accumulators[1], accumulators[5]);
                let sum26 = _mm512_add_ps(accumulators[2], accumulators[6]);
                let sum37 = _mm512_add_ps(accumulators[3], accumulators[7]);
                let result =
                    _mm512_add_ps(_mm512_add_ps(sum04, sum15), _mm512_add_ps(sum26, sum37));

                if out_tile.len() == VNNI_OUT_TILE {
                    _mm512_storeu_ps(out_tile.as_mut_ptr(), result);
                } else {
                    let mut tail = [0.0_f32; VNNI_OUT_TILE];
                    _mm512_storeu_ps(tail.as_mut_ptr(), result);
                    out_tile.copy_from_slice(&tail[..out_tile.len()]);
                }
            }
        }
    }

    /// Q8_0 × Q8_0 dot product using interleaved weight layout.
    /// Processes 4 output rows per stripe, loading all 4 rows' quants
    /// for each block in one contiguous 128-byte read.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,f16c,fma,avx512vl,avx512vnni`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,f16c,fma,avx512vl,avx512vnni")]
    pub unsafe fn matmul_q8_0_decode_interleaved_avx512_vnni(
        x: &[u8],
        quants: &[u8],
        scales: &[u8],
        out_features: usize,
        blocks_per_row: usize,
        out: &mut [f32],
        global_row_offset: usize,
    ) {
        unsafe {
            use crate::quant::INTERLEAVE;
            let grouped_rows = out_features / INTERLEAVE * INTERLEAVE;
            let quants_per_block = INTERLEAVE * Q8_0_BLOCK_SIZE; // 128 bytes
            let scales_per_block = INTERLEAVE * 2; // 8 bytes

            for local_row_start in (0..grouped_rows).step_by(INTERLEAVE) {
                let global_row = global_row_offset + local_row_start;
                let stripe = global_row / INTERLEAVE;
                let q_stripe = &quants[stripe * blocks_per_row * quants_per_block..];
                let s_stripe = &scales[stripe * blocks_per_row * scales_per_block..];

                let mut acc = [_mm256_setzero_ps(); INTERLEAVE];

                for b in 0..blocks_per_row {
                    let q_off = b * quants_per_block;
                    let s_off = b * scales_per_block;

                    // Load 4 rows' quants (128 bytes = 4 × 32)
                    let w0 = _mm256_loadu_si256(q_stripe[q_off..].as_ptr() as *const __m256i);
                    let w1 = _mm256_loadu_si256(q_stripe[q_off + 32..].as_ptr() as *const __m256i);
                    let w2 = _mm256_loadu_si256(q_stripe[q_off + 64..].as_ptr() as *const __m256i);
                    let w3 = _mm256_loadu_si256(q_stripe[q_off + 96..].as_ptr() as *const __m256i);

                    // Load 4 scales (8 bytes), convert each f16 → f32
                    let s_bits = u64::from_le_bytes(
                        *(s_stripe[s_off..s_off + 8].as_ptr() as *const [u8; 8]),
                    );
                    let ws = [
                        f16_bits_to_f32(s_bits as u16),
                        f16_bits_to_f32((s_bits >> 16) as u16),
                        f16_bits_to_f32((s_bits >> 32) as u16),
                        f16_bits_to_f32((s_bits >> 48) as u16),
                    ];

                    // Input activation for this block
                    let x_off = b * Q8_0_TYPE_SIZE;
                    let x_scale = f16_bits_to_f32(u16::from_le_bytes(
                        *(x.as_ptr().add(x_off) as *const [u8; 2]),
                    ));
                    let xq = _mm256_loadu_si256(x.as_ptr().add(x_off + 2) as *const __m256i);

                    let weights = [w0, w1, w2, w3];
                    for lane in 0..INTERLEAVE {
                        let abs_w = _mm256_abs_epi8(weights[lane]);
                        let signed_in = _mm256_sign_epi8(xq, weights[lane]);
                        let sums = _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs_w, signed_in);
                        acc[lane] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(sums),
                            _mm256_set1_ps(ws[lane] * x_scale),
                            acc[lane],
                        );
                    }
                }

                // Horizontal sum and store
                for lane in 0..INTERLEAVE {
                    let a = acc[lane];
                    let low = _mm256_castps256_ps128(a);
                    let high = _mm256_extractf128_ps::<1>(a);
                    let s = _mm_add_ps(low, high);
                    let s = _mm_hadd_ps(s, s);
                    let s = _mm_hadd_ps(s, s);
                    out[local_row_start + lane] = _mm_cvtss_f32(s);
                }
            }

            // Scalar tail for remaining rows (not a multiple of INTERLEAVE)
            for (local_row, out_val) in out.iter_mut().enumerate().skip(grouped_rows) {
                let global_row = global_row_offset + local_row;
                let stripe = global_row / INTERLEAVE;
                let lane = global_row % INTERLEAVE;
                let q_stripe = &quants[stripe * blocks_per_row * quants_per_block..];
                let s_stripe = &scales[stripe * blocks_per_row * scales_per_block..];
                let mut acc = _mm256_setzero_ps();
                for b in 0..blocks_per_row {
                    let q_off = b * quants_per_block + lane * Q8_0_BLOCK_SIZE;
                    let s_off = b * scales_per_block + lane * 2;
                    let w = _mm256_loadu_si256(q_stripe[q_off..].as_ptr() as *const __m256i);
                    let ws = f16_bits_to_f32(u16::from_le_bytes(
                        *(s_stripe[s_off..s_off + 2].as_ptr() as *const [u8; 2]),
                    ));
                    let x_off = b * Q8_0_TYPE_SIZE;
                    let xs = f16_bits_to_f32(u16::from_le_bytes(
                        *(x.as_ptr().add(x_off) as *const [u8; 2]),
                    ));
                    let xq = _mm256_loadu_si256(x.as_ptr().add(x_off + 2) as *const __m256i);
                    let abs_w = _mm256_abs_epi8(w);
                    let signed_in = _mm256_sign_epi8(xq, w);
                    let sums = _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs_w, signed_in);
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums), _mm256_set1_ps(ws * xs), acc);
                }
                let low = _mm256_castps256_ps128(acc);
                let high = _mm256_extractf128_ps::<1>(acc);
                let s = _mm_add_ps(low, high);
                let s = _mm_hadd_ps(s, s);
                let s = _mm_hadd_ps(s, s);
                *out_val = _mm_cvtss_f32(s);
            }
        }
    }

    /// Q8_0 × Q8_0 matrix multiply for a small tile of activation rows.
    ///
    /// Each weight block is loaded once and reused across all rows in the
    /// tile, reducing mmap/cache bandwidth during prompt prefill.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,f16c`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn matmul_q8_0_batch_avx2(
        x: &[u8],
        rows: usize,
        data: &[u8],
        out_features: usize,
        blocks_per_row: usize,
        out: &mut [f32],
    ) {
        unsafe {
            debug_assert!((1..=4).contains(&rows));
            let encoded_row_len = blocks_per_row * Q8_0_TYPE_SIZE;
            let ones = _mm256_set1_epi16(1);

            for output_index in 0..out_features {
                let weight_row_start = output_index * blocks_per_row;
                let mut accumulators = [_mm256_setzero_ps(); 4];

                for b in 0..blocks_per_row {
                    let weight_offset = (weight_row_start + b) * Q8_0_TYPE_SIZE;
                    let weight_scale = f16_bits_to_f32(u16::from_le_bytes(
                        *(data.as_ptr().add(weight_offset) as *const [u8; 2]),
                    ));
                    let weights =
                        _mm256_loadu_si256(data.as_ptr().add(weight_offset + 2) as *const __m256i);
                    let abs_weights = _mm256_abs_epi8(weights);

                    for (row, accumulator) in accumulators.iter_mut().enumerate().take(rows) {
                        let input_offset = row * encoded_row_len + b * Q8_0_TYPE_SIZE;
                        let input_scale = f16_bits_to_f32(u16::from_le_bytes(
                            *(x.as_ptr().add(input_offset) as *const [u8; 2]),
                        ));
                        let input =
                            _mm256_loadu_si256(x.as_ptr().add(input_offset + 2) as *const __m256i);
                        let signed_input = _mm256_sign_epi8(input, weights);
                        let pair16 = _mm256_maddubs_epi16(abs_weights, signed_input);
                        let pair32 = _mm256_madd_epi16(pair16, ones);
                        let products = _mm256_cvtepi32_ps(pair32);
                        let scale = _mm256_set1_ps(weight_scale * input_scale);
                        *accumulator = _mm256_add_ps(*accumulator, _mm256_mul_ps(products, scale));
                    }
                }

                for (row, accumulator) in accumulators.iter().enumerate().take(rows) {
                    let low = _mm256_castps256_ps128(*accumulator);
                    let high = _mm256_extractf128_ps::<1>(*accumulator);
                    let sum128 = _mm_add_ps(low, high);
                    let sum128 = _mm_hadd_ps(sum128, sum128);
                    let sum128 = _mm_hadd_ps(sum128, sum128);
                    out[row * out_features + output_index] = _mm_cvtss_f32(sum128);
                }
            }
        }
    }

    /// AVX-512 VNNI variant of the tiled Q8_0 matrix multiply.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,f16c,fma,avx512vl,avx512vnni`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,f16c,fma,avx512vl,avx512vnni")]
    pub unsafe fn matmul_q8_0_batch_avx512_vnni(
        x: &[u8],
        rows: usize,
        data: &[u8],
        out_features: usize,
        blocks_per_row: usize,
        out: &mut [f32],
    ) {
        unsafe {
            debug_assert!((1..=4).contains(&rows));
            let encoded_row_len = blocks_per_row * Q8_0_TYPE_SIZE;

            if rows == 4 {
                let grouped_outputs = out_features / 4 * 4;
                for output_start in (0..grouped_outputs).step_by(4) {
                    let mut accumulators = [_mm256_setzero_ps(); 16];

                    for b in 0..blocks_per_row {
                        let mut inputs = [_mm256_setzero_si256(); 4];
                        let mut input_scales = [0.0f32; 4];
                        for row in 0..4 {
                            let input_offset = row * encoded_row_len + b * Q8_0_TYPE_SIZE;
                            input_scales[row] = f16_bits_to_f32(u16::from_le_bytes(
                                *(x.as_ptr().add(input_offset) as *const [u8; 2]),
                            ));
                            inputs[row] = _mm256_loadu_si256(
                                x.as_ptr().add(input_offset + 2) as *const __m256i
                            );
                        }

                        for output_lane in 0..4 {
                            let weight_offset = ((output_start + output_lane) * blocks_per_row + b)
                                * Q8_0_TYPE_SIZE;
                            let weight_scale = f16_bits_to_f32(u16::from_le_bytes(
                                *(data.as_ptr().add(weight_offset) as *const [u8; 2]),
                            ));
                            let weights = _mm256_loadu_si256(
                                data.as_ptr().add(weight_offset + 2) as *const __m256i
                            );
                            let abs_weights = _mm256_abs_epi8(weights);

                            for row in 0..4 {
                                let signed_input = _mm256_sign_epi8(inputs[row], weights);
                                let sums = _mm256_dpbusd_epi32(
                                    _mm256_setzero_si256(),
                                    abs_weights,
                                    signed_input,
                                );
                                let products = _mm256_cvtepi32_ps(sums);
                                let scale = _mm256_set1_ps(weight_scale * input_scales[row]);
                                let accumulator = &mut accumulators[output_lane * 4 + row];
                                *accumulator = _mm256_fmadd_ps(products, scale, *accumulator);
                            }
                        }
                    }

                    for output_lane in 0..4 {
                        for row in 0..4 {
                            let accumulator = accumulators[output_lane * 4 + row];
                            let low = _mm256_castps256_ps128(accumulator);
                            let high = _mm256_extractf128_ps::<1>(accumulator);
                            let sum128 = _mm_add_ps(low, high);
                            let sum128 = _mm_hadd_ps(sum128, sum128);
                            let sum128 = _mm_hadd_ps(sum128, sum128);
                            out[row * out_features + output_start + output_lane] =
                                _mm_cvtss_f32(sum128);
                        }
                    }
                }

                if grouped_outputs == out_features {
                    return;
                }
            }

            let output_start = if rows == 4 { out_features / 4 * 4 } else { 0 };
            for output_index in output_start..out_features {
                let weight_row_start = output_index * blocks_per_row;
                let mut accumulators = [_mm256_setzero_ps(); 4];

                for b in 0..blocks_per_row {
                    let weight_offset = (weight_row_start + b) * Q8_0_TYPE_SIZE;
                    let weight_scale = f16_bits_to_f32(u16::from_le_bytes(
                        *(data.as_ptr().add(weight_offset) as *const [u8; 2]),
                    ));
                    let weights =
                        _mm256_loadu_si256(data.as_ptr().add(weight_offset + 2) as *const __m256i);
                    let abs_weights = _mm256_abs_epi8(weights);

                    for (row, accumulator) in accumulators.iter_mut().enumerate().take(rows) {
                        let input_offset = row * encoded_row_len + b * Q8_0_TYPE_SIZE;
                        let input_scale = f16_bits_to_f32(u16::from_le_bytes(
                            *(x.as_ptr().add(input_offset) as *const [u8; 2]),
                        ));
                        let input =
                            _mm256_loadu_si256(x.as_ptr().add(input_offset + 2) as *const __m256i);
                        let signed_input = _mm256_sign_epi8(input, weights);
                        let sums =
                            _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs_weights, signed_input);
                        let products = _mm256_cvtepi32_ps(sums);
                        let scale = _mm256_set1_ps(weight_scale * input_scale);
                        *accumulator = _mm256_fmadd_ps(products, scale, *accumulator);
                    }
                }

                for (row, accumulator) in accumulators.iter().enumerate().take(rows) {
                    let low = _mm256_castps256_ps128(*accumulator);
                    let high = _mm256_extractf128_ps::<1>(*accumulator);
                    let sum128 = _mm_add_ps(low, high);
                    let sum128 = _mm_hadd_ps(sum128, sum128);
                    let sum128 = _mm_hadd_ps(sum128, sum128);
                    out[row * out_features + output_index] = _mm_cvtss_f32(sum128);
                }
            }
        }
    }

    /// SIMD sum of squares: `Σ x[i]²` using AVX2 FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,fma`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn sum_squares_avx2(x: &[f32]) -> f32 {
        unsafe {
            let n = x.len();
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;

            while i + 16 <= n {
                let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
                let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
                acc0 = _mm256_fmadd_ps(v0, v0, acc0);
                acc1 = _mm256_fmadd_ps(v1, v1, acc1);
                i += 16;
            }
            while i + 8 <= n {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                acc0 = _mm256_fmadd_ps(v, v, acc0);
                i += 8;
            }

            let acc = _mm256_add_ps(acc0, acc1);
            let low = _mm256_castps256_ps128(acc);
            let high = _mm256_extractf128_ps::<1>(acc);
            let sum128 = _mm_add_ps(low, high);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let mut sum = _mm_cvtss_f32(sum128);

            while i < n {
                sum += x[i] * x[i];
                i += 1;
            }
            sum
        }
    }

    /// SIMD `out[i] = x[i] * scale * weight[i]` using AVX2.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn scale_weight_mul_avx2(
        x: &[f32],
        scale: f32,
        weight: &[f32],
        out: &mut [f32],
    ) {
        unsafe {
            let n = x.len();
            let s = _mm256_set1_ps(scale);
            let mut i = 0;

            while i + 8 <= n {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                let wv = _mm256_loadu_ps(weight.as_ptr().add(i));
                let r = _mm256_mul_ps(_mm256_mul_ps(xv, s), wv);
                _mm256_storeu_ps(out.as_mut_ptr().add(i), r);
                i += 8;
            }
            while i < n {
                out[i] = x[i] * scale * weight[i];
                i += 1;
            }
        }
    }

    /// SIMD element-wise multiply using AVX2.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn elemul_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        unsafe {
            let n = a.len();
            let mut i = 0;
            while i + 8 <= n {
                let av = _mm256_loadu_ps(a.as_ptr().add(i));
                let bv = _mm256_loadu_ps(b.as_ptr().add(i));
                _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(av, bv));
                i += 8;
            }
            while i < n {
                out[i] = a[i] * b[i];
                i += 1;
            }
        }
    }

    /// Accurate SIMD exp using range reduction and a fifth-degree polynomial.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,fma`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp_ps(x: __m256) -> __m256 {
        let log2e = _mm256_set1_ps(core::f32::consts::LOG2_E);
        let ln2 = _mm256_set1_ps(core::f32::consts::LN_2);
        let magic = _mm256_set1_ps(12_582_912.0_f32);
        let p0 = _mm256_set1_ps(1.0_f32);
        let p1 = _mm256_set1_ps(1.0_f32);
        let p2 = _mm256_set1_ps(0.5_f32);
        let p3 = _mm256_set1_ps(0.166_666_67_f32);
        let p4 = _mm256_set1_ps(0.041_666_668_f32);
        let p5 = _mm256_set1_ps(0.008_333_334_f32);

        let a = _mm256_mul_ps(x, log2e);
        let k = _mm256_sub_ps(_mm256_add_ps(a, magic), magic);
        let r = _mm256_fnmadd_ps(k, ln2, x);
        let poly = _mm256_fmadd_ps(p5, r, p4);
        let poly = _mm256_fmadd_ps(poly, r, p3);
        let poly = _mm256_fmadd_ps(poly, r, p2);
        let poly = _mm256_fmadd_ps(poly, r, p1);
        let poly = _mm256_fmadd_ps(poly, r, p0);
        let k_i32 = _mm256_cvtps_epi32(k);
        let pow2 = _mm256_slli_epi32::<23>(_mm256_add_epi32(k_i32, _mm256_set1_epi32(127)));
        _mm256_mul_ps(poly, _mm256_castsi256_ps(pow2))
    }

    /// SIMD dot product using AVX2 FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,fma`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;

            while i + 16 <= n {
                let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
                let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
                let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                i += 16;
            }
            while i + 8 <= n {
                let av = _mm256_loadu_ps(a.as_ptr().add(i));
                let bv = _mm256_loadu_ps(b.as_ptr().add(i));
                acc0 = _mm256_fmadd_ps(av, bv, acc0);
                i += 8;
            }

            let acc = _mm256_add_ps(acc0, acc1);
            let low = _mm256_castps256_ps128(acc);
            let high = _mm256_extractf128_ps::<1>(acc);
            let sum128 = _mm_add_ps(low, high);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let mut sum = _mm_cvtss_f32(sum128);

            while i < n {
                sum += a[i] * b[i];
                i += 1;
            }
            sum
        }
    }

    /// SIMD dot product between F32 queries and compact F16 cache rows.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,fma,f16c`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(crate) unsafe fn dot_product_f16_avx2(a: &[f32], b: &[f16]) -> f32 {
        unsafe {
            let n = a.len();
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;

            while i + 16 <= n {
                let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
                let b0 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
                let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                let b1 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i + 8) as *const __m128i));
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                i += 16;
            }
            while i + 8 <= n {
                let av = _mm256_loadu_ps(a.as_ptr().add(i));
                let bv = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
                acc0 = _mm256_fmadd_ps(av, bv, acc0);
                i += 8;
            }

            let acc = _mm256_add_ps(acc0, acc1);
            let low = _mm256_castps256_ps128(acc);
            let high = _mm256_extractf128_ps::<1>(acc);
            let sum128 = _mm_add_ps(low, high);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let mut sum = _mm_cvtss_f32(sum128);

            while i < n {
                sum += a[i] * b[i].to_f32();
                i += 1;
            }
            sum
        }
    }

    /// SIMD in-place add: `dst[i] += src[i]`.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn add_assign_avx2(dst: &mut [f32], src: &[f32]) {
        unsafe {
            let n = dst.len();
            let mut i = 0;
            while i + 8 <= n {
                let dv = _mm256_loadu_ps(dst.as_ptr().add(i));
                let sv = _mm256_loadu_ps(src.as_ptr().add(i));
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_add_ps(dv, sv));
                i += 8;
            }
            while i < n {
                dst[i] += src[i];
                i += 1;
            }
        }
    }

    /// SIMD element-wise add using AVX2.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        unsafe {
            let n = a.len();
            let mut i = 0;
            while i + 8 <= n {
                let av = _mm256_loadu_ps(a.as_ptr().add(i));
                let bv = _mm256_loadu_ps(b.as_ptr().add(i));
                _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_add_ps(av, bv));
                i += 8;
            }
            while i < n {
                out[i] = a[i] + b[i];
                i += 1;
            }
        }
    }

    /// SIMD weighted accumulate using AVX2 FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,fma`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn weighted_add_avx2(acc: &mut [f32], src: &[f32], weight: f32) {
        unsafe {
            let n = acc.len();
            let w = _mm256_set1_ps(weight);
            let mut i = 0;

            while i + 8 <= n {
                let sv = _mm256_loadu_ps(src.as_ptr().add(i));
                let av = _mm256_loadu_ps(acc.as_ptr().add(i));
                _mm256_storeu_ps(acc.as_mut_ptr().add(i), _mm256_fmadd_ps(sv, w, av));
                i += 8;
            }
            while i < n {
                acc[i] += weight * src[i];
                i += 1;
            }
        }
    }

    /// SIMD weighted accumulate from a compact F16 cache row.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2,fma,f16c`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(crate) unsafe fn weighted_add_f16_avx2(acc: &mut [f32], src: &[f16], weight: f32) {
        unsafe {
            let n = acc.len();
            let w = _mm256_set1_ps(weight);
            let mut i = 0;

            while i + 8 <= n {
                let sv = _mm256_cvtph_ps(_mm_loadu_si128(src.as_ptr().add(i) as *const __m128i));
                let av = _mm256_loadu_ps(acc.as_ptr().add(i));
                _mm256_storeu_ps(acc.as_mut_ptr().add(i), _mm256_fmadd_ps(sv, w, av));
                i += 8;
            }
            while i < n {
                acc[i] += weight * src[i].to_f32();
                i += 1;
            }
        }
    }

    /// SIMD split-half RoPE using AVX2: 8 pairs per iteration.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required x86 feature set (`avx2`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn rope_split_half_avx2(
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        cos: &[f32],
        sin: &[f32],
    ) {
        unsafe {
            let half = head_dim / 2;
            for h in 0..n_heads {
                let off = h * head_dim;
                let mut d = 0;
                while d + 8 <= half {
                    // Load 8 x0 values (first half of each pair)
                    let x0 = _mm256_loadu_ps(x.as_ptr().add(off + d));
                    // Load 8 x1 values (second half of each pair)
                    let x1 = _mm256_loadu_ps(x.as_ptr().add(off + d + half));
                    let c = _mm256_loadu_ps(cos.as_ptr().add(d));
                    let s = _mm256_loadu_ps(sin.as_ptr().add(d));
                    // x0' = x0*c - x1*s,  x1' = x0*s + x1*c
                    let x0c = _mm256_mul_ps(x0, c);
                    let x1s = _mm256_mul_ps(x1, s);
                    let x0s = _mm256_mul_ps(x0, s);
                    let x1c = _mm256_mul_ps(x1, c);
                    _mm256_storeu_ps(x.as_mut_ptr().add(off + d), _mm256_sub_ps(x0c, x1s));
                    _mm256_storeu_ps(x.as_mut_ptr().add(off + d + half), _mm256_add_ps(x0s, x1c));
                    d += 8;
                }
                // Tail
                for d in d..half {
                    let i0 = off + d;
                    let i1 = off + d + half;
                    let x0 = x[i0];
                    let x1 = x[i1];
                    x[i0] = x0 * cos[d] - x1 * sin[d];
                    x[i1] = x0 * sin[d] + x1 * cos[d];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON kernel
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;
    use std::arch::aarch64::*;

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
        for b in 0..blocks_per_row {
            let byte_offset = (block_start + b) * Q8_0_TYPE_SIZE;
            let base_ptr = unsafe { data.as_ptr().add(byte_offset) };

            // -- scale: load 2-byte f16, convert to f32, broadcast ---------
            let d_bits = u16::from_le_bytes(unsafe { *(base_ptr as *const [u8; 2]) });
            let d = f16::from_bits(d_bits).to_f32();
            let d_vec = vdupq_n_f32(d);

            // -- quants: load two 128-bit vectors of 16 i8 values each ----
            let quants_ptr = unsafe { base_ptr.add(2) as *const i8 };
            let q0 = unsafe { vld1q_s8(quants_ptr) };
            let q1 = unsafe { vld1q_s8(quants_ptr.add(16)) };

            let out_offset = b * Q8_0_BLOCK_SIZE;
            let out_ptr = unsafe { dst.as_mut_ptr().add(out_offset) };

            // helper: dequantize 16 i8 values → 4 × float32x4_t
            #[inline(always)]
            unsafe fn process16(src: int8x16_t, scale: float32x4_t, out: *mut f32) {
                // Safety: Internal helper; only callable from other `unsafe fn`s in this module that already guarantee the required CPU features.
                // low 8 i8 → i16
                let i16_lo = vmovl_s8(vget_low_s8(src));
                // high 8 i8 → i16
                let i16_hi = vmovl_s8(vget_high_s8(src));

                // i16 → i32 → f32 → mul → store (4 lanes each)
                let f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16_lo))), scale);
                let f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16_lo))), scale);
                let f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16_hi))), scale);
                let f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16_hi))), scale);

                unsafe { vst1q_f32(out, f0) };
                unsafe { vst1q_f32(out.add(4), f1) };
                unsafe { vst1q_f32(out.add(8), f2) };
                unsafe { vst1q_f32(out.add(12), f3) };
            }

            // 16 quants → bytes 0..15, 16 quants → bytes 16..31
            unsafe { process16(q0, d_vec, out_ptr) };
            unsafe { process16(q1, d_vec, out_ptr.add(16)) };
        }
    }

    /// Fused Q8_0 dot product using NEON.  32 quants per block in 8 batches
    /// of 4 f32 values each (i8 → i16 → i32 → f32 → mul scale → fma with x).
    ///
    /// # Safety
    ///
    /// SIMD sum of squares using NEON FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn sum_squares_neon(x: &[f32]) -> f32 {
        let n = x.len();
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 4 <= n {
            let v = unsafe { vld1q_f32(x.as_ptr().add(i)) };
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

    /// SIMD `out[i] = x[i] * scale * weight[i]` using NEON.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn scale_weight_mul_neon(
        x: &[f32],
        scale: f32,
        weight: &[f32],
        out: &mut [f32],
    ) {
        let n = x.len();
        let s = vdupq_n_f32(scale);
        let mut i = 0;

        while i + 4 <= n {
            let xv = unsafe { vld1q_f32(x.as_ptr().add(i)) };
            let wv = unsafe { vld1q_f32(weight.as_ptr().add(i)) };
            let r = vmulq_f32(vmulq_f32(xv, s), wv);
            unsafe { vst1q_f32(out.as_mut_ptr().add(i), r) };
            i += 4;
        }
        while i < n {
            out[i] = x[i] * scale * weight[i];
            i += 1;
        }
    }

    /// SIMD element-wise multiply using NEON.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn elemul_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let av = unsafe { vld1q_f32(a.as_ptr().add(i)) };
            let bv = unsafe { vld1q_f32(b.as_ptr().add(i)) };
            unsafe { vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(av, bv)) };
            i += 4;
        }
        while i < n {
            out[i] = a[i] * b[i];
            i += 1;
        }
    }

    /// SIMD dot product using NEON FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 4 <= n {
            let av = unsafe { vld1q_f32(a.as_ptr().add(i)) };
            let bv = unsafe { vld1q_f32(b.as_ptr().add(i)) };
            acc = vfmaq_f32(acc, av, bv);
            i += 4;
        }

        let mut sum = vgetq_lane_f32::<0>(acc)
            + vgetq_lane_f32::<1>(acc)
            + vgetq_lane_f32::<2>(acc)
            + vgetq_lane_f32::<3>(acc);

        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }

    /// SIMD element-wise add using NEON.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn add_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let av = unsafe { vld1q_f32(a.as_ptr().add(i)) };
            let bv = unsafe { vld1q_f32(b.as_ptr().add(i)) };
            unsafe { vst1q_f32(out.as_mut_ptr().add(i), vaddq_f32(av, bv)) };
            i += 4;
        }
        while i < n {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }

    /// SIMD weighted accumulate using NEON FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure the required aarch64 feature set (`neon`) is supported at runtime (dispatched via `is_aarch64_feature_detected!`) before calling this function.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn weighted_add_neon(acc: &mut [f32], src: &[f32], weight: f32) {
        let n = acc.len();
        let w = vdupq_n_f32(weight);
        let mut i = 0;

        while i + 4 <= n {
            let sv = unsafe { vld1q_f32(src.as_ptr().add(i)) };
            let av = unsafe { vld1q_f32(acc.as_ptr().add(i)) };
            unsafe { vst1q_f32(acc.as_mut_ptr().add(i), vfmaq_f32(av, sv, w)) };
            i += 4;
        }
        while i < n {
            acc[i] += weight * src[i];
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// fused Q8_0 decode (seq_len = 1)
// ---------------------------------------------------------------------------
//
// When the input is a single row (decode), quantize it once to Q8_0 and use
// packed integer dot products against each compressed weight row.
//
//   out[j] = Σblocks (d_x · d_w · Σₖ q_x[k] · q_w[j][k])

use crate::quant::QuantizedWeight;

/// Fused Q8_0 dot product for single-row input (decode, seq_len = 1).
///
/// Computes a Q8_0 × Q8_0 matrix-vector product for every output row.
///
/// # Panics
///
/// Panics if `x` is not a valid quantized input row, `out.len() !=
/// out_features`, or the
/// Exact branch-and-bound argmax over a Q8_0 decode matmul.
///
/// Mirrors [`matmul_q8_0_decode_scalar`] accumulation exactly (row-major,
/// per-block f32 sum), so the returned argmax is bit-for-bit the scalar
/// path's argmax. Rows whose running sum plus a Cauchy-Schwarz bound on the
/// remaining in-blocks cannot beat the running maximum are pruned; the
/// bound uses the same quantized values the matmul consumes, so pruning can
/// never remove the true argmax.
#[allow(clippy::needless_range_loop)]
pub(crate) fn matmul_q8_0_decode_argmax(
    x: &[u8],
    w: &QuantizedWeight,
    norms: &crate::quant::Q8TopkNorms,
    margin: f32,
) -> (u32, f32) {
    let out_features = norms.out_features();
    let in_blocks = norms.in_blocks();
    let data = w.data();
    assert_eq!(
        out_features,
        w.out_features(),
        "argmax norms/weight row mismatch"
    );
    assert_eq!(
        in_blocks,
        w.in_features() / Q8_0_BLOCK_SIZE,
        "argmax norms/weight column mismatch"
    );
    assert!(out_features > 0, "argmax requires at least one output row");
    assert!(
        out_features <= u32::MAX as usize,
        "argmax output row count exceeds u32 token range"
    );
    assert!(
        margin.is_finite() && margin >= 1.0,
        "argmax pruning margin must be finite and at least 1"
    );
    let encoded_row_len = in_blocks
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("argmax input size overflow");
    assert_eq!(x.len(), encoded_row_len, "argmax input length mismatch");

    // activation suffix bound: b_g = |scale_x_g| * ||qx_g||
    let mut act_suffix = vec![0.0f32; in_blocks + 1];
    {
        let mut acc = 0.0f64;
        for b in (0..in_blocks).rev() {
            let x_offset = b * Q8_0_TYPE_SIZE;
            let scale = half::f16::from_bits(u16::from_le_bytes(
                x[x_offset..x_offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            let block = &x[x_offset + 2..x_offset + 2 + Q8_0_BLOCK_SIZE];
            let q_norm: f64 = block
                .iter()
                .map(|&v| f64::from(f32::from(i8::from_le_bytes([v]))).powi(2))
                .sum();
            acc += f64::from(scale).powi(2) * q_norm;
            act_suffix[b] = acc.sqrt() as f32;
        }
    }

    // A few fully-computed rows provide a *complete* solution: the optimum
    // is at least the best of them, so pruning against that lower bound can
    // never remove the true argmax. (Pruning against a running *partial*
    // maximum would be unsound: a partial can peak above the final maximum.)
    const SAMPLE_ROWS: usize = 16;
    let sample = out_features.min(SAMPLE_ROWS);
    let mut sums = vec![0.0f32; out_features];
    let mut best_solution = f32::NEG_INFINITY;
    let mut best_row = 0usize;
    for row in 0..sample {
        let mut sum = 0.0f32;
        for b in 0..in_blocks {
            let offset = (row * in_blocks + b) * Q8_0_TYPE_SIZE;
            let weight_scale = half::f16::from_bits(u16::from_le_bytes(
                data[offset..offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            let input_scale = half::f16::from_bits(u16::from_le_bytes(
                x[b * Q8_0_TYPE_SIZE..b * Q8_0_TYPE_SIZE + 2]
                    .try_into()
                    .unwrap(),
            ))
            .to_f32();
            let mut block_sum = 0i32;
            for j in 0..Q8_0_BLOCK_SIZE {
                let weight = data[offset + 2 + j] as i8 as i32;
                let input = x[b * Q8_0_TYPE_SIZE + 2 + j] as i8 as i32;
                block_sum += weight * input;
            }
            sum += block_sum as f32 * weight_scale * input_scale;
        }
        sums[row] = sum;
        if sum > best_solution {
            best_solution = sum;
            best_row = row;
        }
    }

    let mut active = vec![true; out_features];
    for b in 0..in_blocks {
        let x_offset = b * Q8_0_TYPE_SIZE;
        let input_scale = half::f16::from_bits(u16::from_le_bytes(
            x[x_offset..x_offset + 2].try_into().unwrap(),
        ))
        .to_f32();
        for row in sample..out_features {
            if !active[row] {
                continue;
            }
            let offset = (row * in_blocks + b) * Q8_0_TYPE_SIZE;
            let weight_scale = half::f16::from_bits(u16::from_le_bytes(
                data[offset..offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            let mut block_sum = 0i32;
            for j in 0..Q8_0_BLOCK_SIZE {
                let weight = data[offset + 2 + j] as i8 as i32;
                let input = x[x_offset + 2 + j] as i8 as i32;
                block_sum += weight * input;
            }
            sums[row] += block_sum as f32 * weight_scale * input_scale;
        }
        let act_bound = act_suffix[b + 1] * margin;
        for row in sample..out_features {
            if !active[row] {
                continue;
            }
            if sums[row] + norms.suffix(row, b + 1) * act_bound < best_solution {
                active[row] = false;
            }
        }
    }

    let mut max_val = best_solution;
    let mut argmax = best_row as u32;
    for row in 0..out_features {
        if !active[row] {
            continue;
        }
        if sums[row] > max_val {
            max_val = sums[row];
            argmax = row as u32;
        }
    }
    (argmax, max_val)
}

/// weight data is not a valid Q8_0 encoding.
#[inline]
pub(crate) fn matmul_q8_0_decode(x: &[u8], w: &QuantizedWeight, out: &mut [f32]) {
    let encoded_row_len = (w.in_features() / Q8_0_BLOCK_SIZE)
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("q8_0 decode input size overflow");
    assert_eq!(x.len(), encoded_row_len);
    assert_eq!(out.len(), w.out_features());

    let blocks_per_row = w.in_features() / Q8_0_BLOCK_SIZE;
    if should_parallel_q8_decode(w.out_features(), w.in_features()) {
        matmul_q8_0_decode_parallel(x, w, blocks_per_row, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                return x86_64::matmul_q8_0_decode_avx512_vnni(
                    x,
                    w.data(),
                    w.out_features(),
                    blocks_per_row,
                    out,
                );
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            unsafe {
                return x86_64::matmul_q8_0_decode_avx2(
                    x,
                    w.data(),
                    w.out_features(),
                    blocks_per_row,
                    out,
                );
            }
        }
    }
    matmul_q8_0_decode_scalar(x, w.data(), w.out_features(), blocks_per_row, out);
}

/// Packed Q8_0 matrix multiply for multiple activation rows.
///
/// Rows are scheduled together so prompt prefill enters Rayon once instead
/// of launching and joining a separate parallel projection for every token.
pub(crate) fn matmul_q8_0_batch(x: &[u8], rows: usize, w: &QuantizedWeight, out: &mut [f32]) {
    let encoded_row_len = (w.in_features() / Q8_0_BLOCK_SIZE)
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("q8_0 batch row size overflow");
    let expected_input_len = rows
        .checked_mul(encoded_row_len)
        .expect("q8_0 batch input size overflow");
    let expected_output_len = rows
        .checked_mul(w.out_features())
        .expect("q8_0 batch output size overflow");
    assert_eq!(x.len(), expected_input_len);
    assert_eq!(out.len(), expected_output_len);

    let tile_rows = q8_batch_tile_rows(rows);

    if tile_rows == 1 {
        x.par_chunks(encoded_row_len)
            .zip(out.par_chunks_mut(w.out_features()))
            .for_each(|(x_row, out_row)| matmul_q8_0_decode(x_row, w, out_row));
        return;
    }

    let input_tile_len = encoded_row_len
        .checked_mul(tile_rows)
        .expect("q8_0 batch input tile size overflow");
    let output_tile_len = w
        .out_features()
        .checked_mul(tile_rows)
        .expect("q8_0 batch output tile size overflow");
    let blocks_per_row = w.in_features() / Q8_0_BLOCK_SIZE;
    x.par_chunks(input_tile_len)
        .zip(out.par_chunks_mut(output_tile_len))
        .for_each(|(x_tile, out_tile)| {
            let tile_len = x_tile.len() / encoded_row_len;
            matmul_q8_0_batch_dispatch_tile(
                x_tile,
                tile_len,
                w.data(),
                w.out_features(),
                blocks_per_row,
                out_tile,
            );
        });
}

fn q8_batch_tile_rows(rows: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    if rows >= threads.saturating_mul(4) {
        4
    } else if rows >= threads.saturating_mul(2) {
        2
    } else {
        1
    }
}

#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
fn matmul_q8_0_batch_dispatch_tile(
    x: &[u8],
    rows: usize,
    data: &[u8],
    out_features: usize,
    blocks_per_row: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                return x86_64::matmul_q8_0_batch_avx512_vnni(
                    x,
                    rows,
                    data,
                    out_features,
                    blocks_per_row,
                    out,
                );
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            unsafe {
                return x86_64::matmul_q8_0_batch_avx2(
                    x,
                    rows,
                    data,
                    out_features,
                    blocks_per_row,
                    out,
                );
            }
        }
    }

    let encoded_row_len = blocks_per_row * Q8_0_TYPE_SIZE;
    for (x_row, out_row) in x
        .chunks_exact(encoded_row_len)
        .zip(out.chunks_exact_mut(out_features))
    {
        matmul_q8_0_decode_dispatch_chunk(x_row, data, blocks_per_row, out_row);
    }
}

fn should_parallel_q8_decode(out_features: usize, in_features: usize) -> bool {
    rayon::current_num_threads() > 1
        && out_features.saturating_mul(in_features) >= PARALLEL_Q8_DECODE_MIN_WORK
}

#[inline]
pub(crate) fn q8_decode_uses_row_parallel(out_features: usize, in_features: usize) -> bool {
    should_parallel_q8_decode(out_features, in_features)
}

fn matmul_q8_0_decode_parallel(
    x: &[u8],
    w: &QuantizedWeight,
    blocks_per_row: usize,
    out: &mut [f32],
) {
    let threads = rayon::current_num_threads().max(1);
    let chunk_rows = w.out_features().div_ceil(threads).max(64);
    out.par_chunks_mut(chunk_rows)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * chunk_rows;
            let data_start = row_start * blocks_per_row * Q8_0_TYPE_SIZE;
            let data = &w.data()[data_start..];
            matmul_q8_0_decode_dispatch_chunk(x, data, blocks_per_row, out_chunk);
        });
}

fn matmul_q8_0_decode_dispatch_chunk(
    x: &[u8],
    data: &[u8],
    blocks_per_row: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                return x86_64::matmul_q8_0_decode_avx512_vnni(
                    x,
                    data,
                    out.len(),
                    blocks_per_row,
                    out,
                );
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            unsafe {
                return x86_64::matmul_q8_0_decode_avx2(x, data, out.len(), blocks_per_row, out);
            }
        }
    }
    matmul_q8_0_decode_scalar(x, data, out.len(), blocks_per_row, out);
}

// -- interleaved Q8_0 dispatch -----------------------------------------------

use crate::quant::QuantizedWeightInterleaved;
use crate::quant::{QuantizedWeightVnni, VNNI_OUT_TILE};

/// Matrix-vector multiply using the 16-output packed Q8_0 layout.
pub(crate) fn matmul_q8_0_decode_packed16_parallel(
    x: &[u8],
    weight: &QuantizedWeightVnni,
    out: &mut [f32],
) {
    assert!(packed_q8_0_vnni_supported());
    assert_eq!(x.len(), weight.blocks_per_row * Q8_0_TYPE_SIZE);
    assert_eq!(out.len(), weight.out_features());
    let threads = rayon::current_num_threads().max(1);
    let chunk_rows = weight
        .out_features()
        .div_ceil(threads)
        .next_multiple_of(VNNI_OUT_TILE)
        .max(64);
    out.par_chunks_mut(chunk_rows).enumerate().for_each(
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
        |(chunk_index, out_chunk)| {
            let global_row_offset = chunk_index * chunk_rows;
            #[cfg(target_arch = "x86_64")]
            unsafe {
                // Safety: runtime feature checks above cover every enabled ISA
                // extension. Repacking pads the last tile and guarantees one
                // complete record for every output chunk addressed here.
                x86_64::matmul_q8_0_decode_packed16_avx512_vnni(
                    x,
                    &weight.data,
                    weight.blocks_per_row,
                    out_chunk,
                    global_row_offset,
                );
            }
        },
    );
}

/// Matrix multiply using interleaved Q8_0 weight layout.
/// Dispatches to the fastest available kernel.
/// `global_row_start` is the absolute row index within the full weight matrix
/// that `out[0]` corresponds to. Pass 0 when processing all rows.
#[inline]
pub(crate) fn matmul_q8_0_decode_interleaved(
    x: &[u8],
    w: &QuantizedWeightInterleaved,
    out: &mut [f32],
    global_row_start: usize,
) {
    let out_rows = out.len();
    if out_rows == 0 {
        return;
    }
    let input_len = w
        .blocks_per_row
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("interleaved q8 input length overflow");
    assert_eq!(x.len(), input_len, "interleaved q8 input length mismatch");
    let row_end = global_row_start
        .checked_add(out_rows)
        .expect("interleaved q8 output range overflow");
    assert!(
        row_end <= w.out_features(),
        "interleaved q8 output range {global_row_start}..{row_end} exceeds {} rows",
        w.out_features()
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                return x86_64::matmul_q8_0_decode_interleaved_avx512_vnni(
                    x,
                    &w.quants,
                    &w.scales,
                    out_rows,
                    w.blocks_per_row,
                    out,
                    global_row_start,
                );
            }
        }
    }
    // Fallback: dequantize to f32 and use standard matmul (unlikely path)
    for (row, out_val) in out.iter_mut().enumerate() {
        let global_row = global_row_start + row;
        let stripe = global_row / crate::quant::INTERLEAVE;
        let lane = global_row % crate::quant::INTERLEAVE;
        let q_stripe =
            &w.quants[stripe * w.blocks_per_row * crate::quant::INTERLEAVE * Q8_0_BLOCK_SIZE..];
        let s_stripe = &w.scales[stripe * w.blocks_per_row * crate::quant::INTERLEAVE * 2..];
        let mut sum = 0.0f32;
        for b in 0..w.blocks_per_row {
            let q_off = b * crate::quant::INTERLEAVE * Q8_0_BLOCK_SIZE + lane * Q8_0_BLOCK_SIZE;
            let s_off = b * crate::quant::INTERLEAVE * 2 + lane * 2;
            let ws =
                f16::from_bits(u16::from_le_bytes([s_stripe[s_off], s_stripe[s_off + 1]])).to_f32();
            let xs = f16::from_bits(u16::from_le_bytes([
                x[b * Q8_0_TYPE_SIZE],
                x[b * Q8_0_TYPE_SIZE + 1],
            ]))
            .to_f32();
            for j in 0..Q8_0_BLOCK_SIZE {
                sum += (x[b * Q8_0_TYPE_SIZE + 2 + j] as i8) as f32
                    * (q_stripe[q_off + j] as i8) as f32
                    * ws
                    * xs;
            }
        }
        *out_val = sum;
    }
}

/// Parallel version: splits output rows across Rayon threads.
/// Each chunk receives its global row offset so stripe indexing is correct.
pub(crate) fn matmul_q8_0_decode_interleaved_parallel(
    x: &[u8],
    w: &QuantizedWeightInterleaved,
    out: &mut [f32],
) {
    assert_eq!(
        out.len(),
        w.out_features(),
        "interleaved q8 output length mismatch"
    );
    let input_len = w
        .blocks_per_row
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("interleaved q8 input length overflow");
    assert_eq!(x.len(), input_len, "interleaved q8 input length mismatch");
    let threads = rayon::current_num_threads().max(1);
    let chunk_rows = w
        .out_features()
        .div_ceil(threads)
        .next_multiple_of(crate::quant::INTERLEAVE)
        .max(64);
    out.par_chunks_mut(chunk_rows)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let global_start = chunk_idx * chunk_rows;
            matmul_q8_0_decode_interleaved(x, w, out_chunk, global_start);
        });
}

// -- scalar fallback --------------------------------------------------------

fn matmul_q8_0_decode_scalar(
    x: &[u8],
    data: &[u8],
    _out_features: usize,
    blocks_per_row: usize,
    out: &mut [f32],
) {
    for (row, out_val) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let row_start = row * blocks_per_row;
        for b in 0..blocks_per_row {
            let byte_offset = (row_start + b) * Q8_0_TYPE_SIZE;
            let x_offset = b * Q8_0_TYPE_SIZE;
            let weight_scale = f16::from_bits(u16::from_le_bytes(
                data[byte_offset..byte_offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            let input_scale = f16::from_bits(u16::from_le_bytes(
                x[x_offset..x_offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            let mut block_sum = 0i32;
            for j in 0..Q8_0_BLOCK_SIZE {
                let weight = data[byte_offset + 2 + j] as i8 as i32;
                let input = x[x_offset + 2 + j] as i8 as i32;
                block_sum += weight * input;
            }
            sum += block_sum as f32 * weight_scale * input_scale;
        }
        *out_val = sum;
    }
}

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
            return unsafe { x86_64::sum_squares_avx2(x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
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
            unsafe {
                return x86_64::elemul_avx2(a, b, out);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
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
            unsafe {
                return x86_64::dot_product_avx2(a, b);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
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
            unsafe {
                return x86_64::dot_product_f16_avx2(a, b);
            }
        }
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
            unsafe {
                return x86_64::weighted_add_avx2(acc, src, weight);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
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
            unsafe {
                return x86_64::weighted_add_f16_avx2(acc, src, weight);
            }
        }
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
            unsafe {
                return x86_64::add_avx2(a, b, out);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
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
            unsafe {
                return x86_64::scale_weight_mul_avx2(x, scale, weight, out);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
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
        if is_x86_feature_detected!("avx2") {
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
            x[i0] = x0 * cos[d] - x1 * sin[d];
            x[i1] = x0 * sin[d] + x1 * cos[d];
        }
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::quant::{quantize_q8_0_into, Q8TopkNorms, QuantizedWeight};

    /// The branch-and-bound argmax must match the scalar decode matmul
    /// bit-for-bit (same accumulation order) across randomized inputs.
    #[test]
    fn argmax_kernel_matches_scalar_exactly() {
        let mut seed = 0x9e37_79b9u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        };
        for _trial in 0..32 {
            let out = 512;
            let in_ = 256;
            let blocks_per_row = in_ / Q8_0_BLOCK_SIZE;
            let mut bytes = Vec::with_capacity(out * blocks_per_row * Q8_0_TYPE_SIZE);
            for _ in 0..out * blocks_per_row {
                // random f16 scale (finite) + 32 random int8 quants
                let mut scale_bits: u16 = ((next() as u16) << 8) | next() as u16;
                scale_bits &= 0x7FFF;
                if scale_bits >= 0x7C00 {
                    scale_bits = 0x3C00;
                }
                bytes.extend_from_slice(&scale_bits.to_le_bytes());
                for _ in 0..Q8_0_BLOCK_SIZE {
                    bytes.push(next());
                }
            }
            let weight = QuantizedWeight::try_new(bytes, vec![out, in_]).unwrap();
            let norms = Q8TopkNorms::compute(&weight);
            // random activation, quantized exactly like the decode path
            let mut act = vec![0.0f32; in_];
            let mut aseed = seed;
            for value in &mut act {
                aseed = aseed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *value = (((aseed >> 40) as f32) / 16_777_216.0 - 0.5) * 3.0;
            }
            let mut encoded = Vec::new();
            quantize_q8_0_into(&act, &mut encoded);
            let mut full = vec![0.0f32; out];
            matmul_q8_0_decode_scalar(&encoded, weight.data(), out, blocks_per_row, &mut full);
            let (token, logit) = matmul_q8_0_decode_argmax(&encoded, &weight, &norms, 1.001);
            // first-index max semantics (matches the kernel's strict-greater
            // update; ties resolve to the lowest index)
            let expected = full.iter().enumerate().fold(
                (0usize, f32::NEG_INFINITY),
                |(best_idx, best_val), (idx, &value)| {
                    if value > best_val {
                        (idx, value)
                    } else {
                        (best_idx, best_val)
                    }
                },
            );
            if token as usize != expected.0 {
                eprintln!(
                    "trial {_trial}: argmax {token} (val {logit}) vs expected {} (val {}); \
                     best_solution-relevant: sample rows 0..{}",
                    expected.0,
                    expected.1,
                    16.min(out)
                );
            }
            assert_eq!(
                token as usize, expected.0,
                "trial {_trial}: argmax mismatch"
            );
            assert!(
                (logit - expected.1).abs() < 1e-3,
                "trial {_trial}: logit mismatch {logit} vs {}",
                expected.1
            );
            seed = aseed;
        }
    }

    use super::*;

    const BOUNDARY_LENGTHS: &[usize] = &[0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33];

    fn patterned_vec(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + phase) * 0.37).sin() * 3.0 + (i % 5) as f32 * 0.125)
            .collect()
    }

    fn quantized_input(values: &[f32]) -> Vec<u8> {
        let mut data = Vec::new();
        crate::quant::quantize_q8_0_into(values, &mut data);
        data
    }

    fn max_diff(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut max_idx = 0usize;
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            let abs = (x - y).abs();
            let rel = abs / x.abs().max(y.abs()).max(1.0);
            if abs > max_abs {
                max_abs = abs;
                max_rel = rel;
                max_idx = i;
            }
        }
        (max_abs, max_rel, max_idx)
    }

    fn assert_close(label: &str, got: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32) {
        let (max_abs, max_rel, max_idx) = max_diff(got, expected);
        assert!(
            max_abs <= abs_tol || max_rel <= rel_tol,
            "{label}: max_abs={max_abs} max_rel={max_rel} idx={max_idx} got={} expected={}",
            got.get(max_idx).copied().unwrap_or(0.0),
            expected.get(max_idx).copied().unwrap_or(0.0)
        );
    }

    /// Build a single Q8_0 block (34 bytes) with known scale and quants.
    fn make_block(scale: f32, quants: &[i8; 32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Q8_0_TYPE_SIZE);
        let s = f16::from_f32(scale);
        buf.extend_from_slice(&s.to_bits().to_le_bytes());
        for &q in quants {
            buf.push(q as u8);
        }
        assert_eq!(buf.len(), Q8_0_TYPE_SIZE);
        buf
    }

    #[test]
    fn simd_helper_boundary_parity() {
        for &len in BOUNDARY_LENGTHS {
            let a = patterned_vec(len, 0.25);
            let b = patterned_vec(len, 1.75);

            let expected_sum_squares: f32 = a.iter().map(|v| v * v).sum();
            let got_sum_squares = sum_squares(&a);
            assert!(
                (got_sum_squares - expected_sum_squares).abs() <= 1e-5 * len.max(1) as f32,
                "sum_squares len={len}: got={got_sum_squares} expected={expected_sum_squares}"
            );

            let expected_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let got_dot = dot_product(&a, &b);
            assert!(
                (got_dot - expected_dot).abs() <= 1e-5 * len.max(1) as f32,
                "dot_product len={len}: got={got_dot} expected={expected_dot}"
            );

            let mut got = vec![0.0; len];
            let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
            elemul(&a, &b, &mut got);
            assert_close(&format!("elemul len={len}"), &got, &expected, 1e-6, 1e-6);

            let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
            add(&a, &b, &mut got);
            assert_close(&format!("add len={len}"), &got, &expected, 1e-6, 1e-6);

            let mut acc = a.clone();
            let mut expected = a.clone();
            for (dst, src) in expected.iter_mut().zip(&b) {
                *dst += -0.75 * src;
            }
            weighted_add(&mut acc, &b, -0.75);
            assert_close(
                &format!("weighted_add len={len}"),
                &acc,
                &expected,
                1e-5,
                1e-6,
            );

            let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * 0.5 * y).collect();
            scale_weight_mul(&a, 0.5, &b, &mut got);
            assert_close(
                &format!("scale_weight_mul len={len}"),
                &got,
                &expected,
                1e-6,
                1e-6,
            );
        }
    }

    #[test]
    fn f16_cache_simd_helpers_match_rounded_scalar() {
        for &len in BOUNDARY_LENGTHS {
            let query = patterned_vec(len, 0.25);
            let source = patterned_vec(len, 1.75);
            let source_f16 = source
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>();

            let expected_dot = query
                .iter()
                .zip(&source_f16)
                .map(|(a, b)| a * b.to_f32())
                .sum::<f32>();
            let got_dot = dot_product_f16(&query, &source_f16);
            assert!(
                (got_dot - expected_dot).abs() <= 1e-5 * len.max(1) as f32,
                "f16 dot len={len}: got={got_dot} expected={expected_dot}"
            );

            let mut expected = patterned_vec(len, 3.25);
            for (dst, src) in expected.iter_mut().zip(&source_f16) {
                *dst += 0.375 * src.to_f32();
            }
            let mut got = patterned_vec(len, 3.25);
            weighted_add_f16(&mut got, &source_f16, 0.375);
            assert_close("f16 weighted add", &got, &expected, 1e-5, 1e-5);
        }
    }

    /// Build `n` consecutive Q8_0 blocks with alternating scale patterns.
    fn make_row(blocks_per_row: usize) -> (Vec<u8>, Vec<f32>) {
        let mut data = Vec::with_capacity(blocks_per_row * Q8_0_TYPE_SIZE);
        let mut expected = vec![0.0f32; blocks_per_row * Q8_0_BLOCK_SIZE];

        for b in 0..blocks_per_row {
            let scale_f32 = 0.5 + (b as f32) * 0.1;
            // expected values must account for f16 round-trip precision loss:
            // the scale is stored as f16, so the effective scale is
            // f16::from_f32(scale_f32).to_f32(), not the original f32 value.
            let scale_effective = f16::from_f32(scale_f32).to_f32();
            let mut quants = [0i8; 32];
            for j in 0..32 {
                quants[j] = (j as i8) - 16; // range [-16, 15]
                expected[b * 32 + j] = (quants[j] as f32) * scale_effective;
            }
            data.extend_from_slice(&make_block(scale_f32, &quants));
        }
        (data, expected)
    }

    #[test]
    fn scalar_dequant_row_matches_expected() {
        let (data, expected) = make_row(4);
        let mut dst = vec![0.0f32; 4 * 32];
        dequantize_row_scalar(&data, 0, 4, &mut dst);
        for (i, (a, b)) in dst.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "mismatch at {i}: scalar={a} expected={b}"
            );
        }
    }

    #[test]
    fn dispatch_produces_same_output_as_scalar() {
        let blocks = 16;
        let (data, _expected) = make_row(blocks);
        let mut scalar_out = vec![0.0f32; blocks * 32];
        let mut simd_out = vec![0.0f32; blocks * 32];

        dequantize_row_scalar(&data, 0, blocks, &mut scalar_out);
        dequantize_q8_0_row(&data, 0, blocks, &mut simd_out);

        for (i, (s, d)) in scalar_out.iter().zip(simd_out.iter()).enumerate() {
            assert!(
                (s - d).abs() < 1e-6,
                "dispatch mismatch at {i}: scalar={s} dispatch={d}"
            );
        }
    }

    #[test]
    fn dispatch_with_offset_produces_same_output() {
        // simulate a weight matrix with multiple rows
        let blocks_per_row = 8;
        let (row0_data, _row0_expected) = make_row(blocks_per_row);
        let (row1_data, _row1_expected) = make_row(blocks_per_row);

        let mut data = row0_data.clone();
        data.extend_from_slice(&row1_data);

        let mut scalar_out = vec![0.0f32; blocks_per_row * 32];
        let mut dispatch_out = vec![0.0f32; blocks_per_row * 32];

        // row 0 (block_start = 0)
        dequantize_row_scalar(&data, 0, blocks_per_row, &mut scalar_out);
        dequantize_q8_0_row(&data, 0, blocks_per_row, &mut dispatch_out);
        for (i, (s, d)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
            assert!((s - d).abs() < 1e-6, "row0 mismatch at {i}");
        }

        // row 1 (block_start = blocks_per_row)
        dequantize_row_scalar(&data, blocks_per_row, blocks_per_row, &mut scalar_out);
        dequantize_q8_0_row(&data, blocks_per_row, blocks_per_row, &mut dispatch_out);
        for (i, (s, d)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
            assert!((s - d).abs() < 1e-6, "row1 mismatch at {i}");
        }
    }

    #[test]
    fn edge_case_min_max_values() {
        // test extreme quant values: min i8 (-128), max i8 (127), zero scale
        let mut data = Vec::new();

        // block 0: scale=0.0, quants=[-128, 127, 0, ...]
        let s0 = f16::from_f32(0.0);
        data.extend_from_slice(&s0.to_bits().to_le_bytes());
        data.push((-128i8) as u8);
        data.push(127i8 as u8);
        data.push(0u8);
        data.resize(data.len() + 29, 0);

        // block 1: scale=1.0, quants=[-128, 127, 0, ...]
        let s1 = f16::from_f32(1.0);
        data.extend_from_slice(&s1.to_bits().to_le_bytes());
        data.push((-128i8) as u8);
        data.push(127i8 as u8);
        data.push(0u8);
        data.resize(data.len() + 29, 0);

        let blocks = 2;
        let mut scalar_out = vec![0.0f32; blocks * 32];
        let mut dispatch_out = vec![0.0f32; blocks * 32];

        dequantize_row_scalar(&data, 0, blocks, &mut scalar_out);
        dequantize_q8_0_row(&data, 0, blocks, &mut dispatch_out);

        for (i, (s, d)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
            assert!(
                (s - d).abs() < 1e-6,
                "edge mismatch at {i}: scalar={s} dispatch={d}"
            );
        }
    }

    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut, unused_variables))]
    #[test]
    fn explicit_avx2_call_matches_scalar() {
        let blocks = 4;
        let (data, _expected) = make_row(blocks);
        let mut scalar_out = vec![0.0f32; blocks * 32];
        let mut avx2_out = vec![0.0f32; blocks * 32];

        dequantize_row_scalar(&data, 0, blocks, &mut scalar_out);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
                unsafe {
                    x86_64::dequantize_row_avx2(&data, 0, blocks, &mut avx2_out);
                }
                for (i, (s, a)) in scalar_out.iter().zip(avx2_out.iter()).enumerate() {
                    assert!(
                        (s - a).abs() < 1e-6,
                        "avx2 mismatch at {i}: scalar={s} avx2={a}"
                    );
                }
            }
        }
    }

    #[test]
    fn explicit_neon_call_matches_scalar() {
        let blocks = 4;
        let (data, _expected) = make_row(blocks);
        let mut scalar_out = vec![0.0f32; blocks * 32];
        let mut _neon_out = vec![0.0f32; blocks * 32];

        dequantize_row_scalar(&data, 0, blocks, &mut scalar_out);

        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                unsafe {
                    aarch64::dequantize_row_neon(&data, 0, blocks, &mut _neon_out);
                }
                for (i, (s, n)) in scalar_out.iter().zip(_neon_out.iter()).enumerate() {
                    assert!(
                        (s - n).abs() < 1e-6,
                        "neon mismatch at {i}: scalar={s} neon={n}"
                    );
                }
            }
        }
    }

    // -- benchmark ------------------------------------------------------

    /// Build random Q8_0 weight data for shape `[out_features, in_features]`.
    /// Each block gets a random f16 scale and random i8 quants.
    fn random_q8_0_data(out_features: usize, in_features: usize) -> Vec<u8> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
        let total_blocks = out_features * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q8_0_TYPE_SIZE];

        for b in 0..total_blocks {
            let offset = b * Q8_0_TYPE_SIZE;
            let scale = f16::from_f32(rng.r#gen::<f32>() * 2.0);
            data[offset..offset + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
            for j in 0..Q8_0_BLOCK_SIZE {
                data[offset + 2 + j] = rng.r#gen::<i8>() as u8;
            }
        }
        data
    }

    // -- decode path correctness --------------------------------------------

    #[test]
    fn decode_path_matches_scalar() {
        use rand::Rng;
        let out_features = 66;
        // Five input blocks and 66 outputs exercise both 4x4 tile tails.
        let in_features = 160;
        let data = random_q8_0_data(out_features, in_features);
        let w = QuantizedWeight::try_new(data.clone(), vec![out_features, in_features]).unwrap();

        let mut rng = rand::thread_rng();
        let x: Vec<f32> = (0..in_features)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect();
        let qx = quantized_input(&x);

        let mut decode_out = vec![0.0f32; out_features];
        matmul_q8_0_decode(&qx, &w, &mut decode_out);

        let mut scalar_out = vec![0.0f32; out_features];
        matmul_q8_0_decode_scalar(
            &qx,
            &data,
            out_features,
            in_features / Q8_0_BLOCK_SIZE,
            &mut scalar_out,
        );

        for (i, (d, s)) in decode_out.iter().zip(scalar_out.iter()).enumerate() {
            let diff = (d - s).abs();
            let max_val = d.abs().max(s.abs()).max(1.0);
            // SIMD FMA accumulates in a different order than scalar;
            // allow 1e-3 relative tolerance for the dot product.
            assert!(
                diff / max_val < 1e-3,
                "decode mismatch at {i}: decode={d} scalar={s} diff={diff} rel={}",
                diff / max_val
            );
        }
    }

    #[test]
    fn interleaved_decode_matches_row_contiguous_decode() {
        use rand::Rng;

        // A non-power-of-two row count exercises the aligned parallel chunks
        // and the final partial interleave stripe.
        let out_features = 70;
        let in_features = 256;
        let data = random_q8_0_data(out_features, in_features);
        let weight = QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap();
        let interleaved = QuantizedWeightInterleaved::from_quantized(&weight);
        let mut rng = rand::thread_rng();
        let input = (0..in_features)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect::<Vec<_>>();
        let quantized = quantized_input(&input);

        let mut expected = vec![0.0; out_features];
        matmul_q8_0_decode(&quantized, &weight, &mut expected);
        let mut actual = vec![0.0; out_features];
        matmul_q8_0_decode_interleaved_parallel(&quantized, &interleaved, &mut actual);

        assert_close("interleaved q8 decode", &actual, &expected, 0.1, 0.02);
    }

    #[test]
    fn packed16_decode_matches_row_contiguous_decode() {
        if !packed_q8_0_vnni_supported() {
            return;
        }

        for (out_features, in_features) in [(70, 256), (32, 2_048), (32, 8_192)] {
            let data = random_q8_0_data(out_features, in_features);
            let weight = QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap();
            let packed = QuantizedWeightVnni::from_quantized(&weight);
            assert_eq!(
                packed.byte_len(),
                out_features.div_ceil(VNNI_OUT_TILE) * VNNI_OUT_TILE * in_features
                    / Q8_0_BLOCK_SIZE
                    * Q8_0_TYPE_SIZE
            );
            let input = patterned_vec(in_features, 0.625);
            let quantized = quantized_input(&input);

            let mut expected = vec![0.0; out_features];
            matmul_q8_0_decode(&quantized, &weight, &mut expected);
            let mut actual = vec![0.0; out_features];
            matmul_q8_0_decode_packed16_parallel(&quantized, &packed, &mut actual);

            for (index, (packed_value, row_value)) in actual.iter().zip(expected.iter()).enumerate()
            {
                assert_eq!(
                    packed_value.to_bits(),
                    row_value.to_bits(),
                    "packed Q8 decode mismatch at row {index} for {out_features}x{in_features}: \
                     packed={packed_value} row={row_value}"
                );
            }
        }
    }

    /// Compare the existing row-contiguous and packed-16 projection kernels.

    #[test]
    fn decode_path_matches_packed_batch_path() {
        use crate::backend::{Backend, CpuBackend};
        use rand::Rng;

        let out_features = 66;
        let in_features = 256;
        let data = random_q8_0_data(out_features, in_features);
        let w = QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap();

        let mut rng = rand::thread_rng();
        let x: Vec<f32> = (0..in_features)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect();
        let qx = quantized_input(&x);

        let mut decode_out = vec![0.0f32; out_features];
        matmul_q8_0_decode(&qx, &w, &mut decode_out);

        // Two identical rows exercise the packed multi-row scheduling path.
        let mut x2 = Vec::with_capacity(in_features * 2);
        x2.extend_from_slice(&x);
        x2.extend_from_slice(&x);
        let backend = CpuBackend;
        let prefill = backend
            .matmul_q8_0(
                &crate::tensor::CpuTensor::from_data(vec![2, in_features], x2),
                &w,
            )
            .unwrap();
        let prefill_first_row = &prefill.data()[..out_features];
        assert_close(
            "q8 decode vs packed batch",
            &decode_out,
            prefill_first_row,
            0.1,
            0.02,
        );
    }

    #[test]
    fn tiled_q8_batch_matches_independent_rows() {
        use rand::Rng;

        let rows = 8;
        let out_features = 64;
        let in_features = 256;
        let data = random_q8_0_data(out_features, in_features);
        let w = QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap();
        let mut rng = rand::thread_rng();
        let input = (0..rows * in_features)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect::<Vec<_>>();
        let packed = quantized_input(&input);
        let encoded_row_len = in_features / Q8_0_BLOCK_SIZE * Q8_0_TYPE_SIZE;

        let mut expected = vec![0.0; rows * out_features];
        for row in 0..rows {
            matmul_q8_0_decode(
                &packed[row * encoded_row_len..(row + 1) * encoded_row_len],
                &w,
                &mut expected[row * out_features..(row + 1) * out_features],
            );
        }

        let mut got = vec![0.0; rows * out_features];
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| matmul_q8_0_batch(&packed, rows, &w, &mut got));

        assert_close("tiled q8 batch", &got, &expected, 1e-5, 1e-5);
    }

    #[test]
    fn scale_weight_mul_simd_matches_scalar() {
        let n = 1536; // embed_dim
        let mut x = vec![0.0f32; n];
        let mut weight = vec![0.0f32; n];
        let scale = 0.5_f32.sqrt().recip(); // typical rstd value

        // Realistic values: x ~ N(0, 1), weight from output_norm (0..118)
        for i in 0..n {
            x[i] = (i as f32).sin() * 2.0;
            weight[i] = (i as f32 / n as f32) * 118.0; // max weight seen in GGUF
        }

        let mut simd_out = vec![0.0f32; n];
        let mut scalar_out = vec![0.0f32; n];

        // SIMD path
        crate::simd::scale_weight_mul(&x, scale, &weight, &mut simd_out);

        // Scalar path
        for i in 0..n {
            scalar_out[i] = x[i] * scale * weight[i];
        }

        for i in 0..n {
            assert!(
                (simd_out[i] - scalar_out[i]).abs() < 1e-6,
                "mismatch at {i}: simd={} scalar={}",
                simd_out[i],
                scalar_out[i]
            );
        }
    }

    #[test]
    fn sum_squares_simd_matches_scalar() {
        let n = 1536;
        let mut x = vec![0.0f32; n];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32).sin() * 100.0;
        }
        let simd = super::sum_squares(&x);
        let scalar: f32 = x.iter().map(|v| v * v).sum();
        let diff = (simd - scalar).abs();
        let rel = if scalar > 0.0 { diff / scalar } else { diff };
        assert!(
            rel < 1e-6,
            "sum_squares mismatch: simd={}, scalar={}, diff={}",
            simd,
            scalar,
            diff
        );
    }
}
