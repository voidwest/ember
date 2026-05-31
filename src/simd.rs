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
//! | x86-64   | avx2    | 256   | 8 f32 per op, 4 ops / block  |
//! | aarch64  | neon    | 128   | 4 f32 per op, 8 ops / block  |
//! | fallback | (none)  | —     | scalar, matches original     |
//!
//! One Q8_0 block = 34 bytes (2-byte f16 scale + 32 i8 quants) → 32 f32 values.

use crate::quant::{Q8_0_BLOCK_SIZE, Q8_0_TYPE_SIZE};
use half::f16;
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
    // Safety: the arch-specific kernels are only called when the
    // corresponding CPU feature is detected at runtime.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
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
// scalar fallback (always compiled)
// ---------------------------------------------------------------------------

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

    /// AVX2-accelerated Q8_0 row dequantization.
    ///
    /// Processes 32 quants per block in 4 batches of 8 f32 values using
    /// 256-bit SIMD registers.  One block per iteration.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported (checked by dispatch above).
    #[target_feature(enable = "avx2")]
    pub unsafe fn dequantize_row_avx2(
        data: &[u8],
        block_start: usize,
        blocks_per_row: usize,
        dst: &mut [f32],
    ) {
        for b in 0..blocks_per_row {
            let byte_offset = (block_start + b) * Q8_0_TYPE_SIZE;
            let base_ptr = data.as_ptr().add(byte_offset);

            // -- scale: load 2-byte f16, convert to f32, broadcast ---------
            let d_bits = u16::from_le_bytes(*(base_ptr as *const [u8; 2]));
            let d = f16::from_bits(d_bits).to_f32();
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

    /// Fused Q8_0 dot product using AVX2 for integer widening and FMA for
    /// multiply-accumulate.  Processes 32 quants per block in 4 batches of
    /// 8 f32 values each, with two independent accumulators for ILP.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and FMA are supported.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn matmul_q8_0_decode_avx2_fma(
        x: &[f32],
        data: &[u8],
        _out_features: usize,
        blocks_per_row: usize,
        out: &mut [f32],
    ) {
        for (row, out_val) in out.iter_mut().enumerate() {
            let row_start = row * blocks_per_row;
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();

            for b in 0..blocks_per_row {
                let byte_offset = (row_start + b) * Q8_0_TYPE_SIZE;
                let base_ptr = data.as_ptr().add(byte_offset);

                // -- scale -----------------------------------------------
                let d_bits = u16::from_le_bytes(*(base_ptr as *const [u8; 2]));
                let d = f16::from_bits(d_bits).to_f32();
                let d_vec = _mm256_set1_ps(d);

                // -- quants: 32 i8 values as 256-bit vector ---------------
                let quants_ptr = base_ptr.add(2) as *const i8;
                let quants = _mm256_loadu_si256(quants_ptr as *const __m256i);

                let low128 = _mm256_castsi256_si128(quants);
                let high128 = _mm256_extracti128_si256::<1>(quants);

                let x_offset = b * Q8_0_BLOCK_SIZE;
                let x_ptr = x.as_ptr().add(x_offset);

                // -- batch 0: bytes 0..7 of low128 → acc0 ----------------
                let q0_i32 = _mm256_cvtepi8_epi32(low128);
                let q0_f32 = _mm256_cvtepi32_ps(q0_i32);
                let q0_scaled = _mm256_mul_ps(q0_f32, d_vec);
                let x0 = _mm256_loadu_ps(x_ptr);
                acc0 = _mm256_fmadd_ps(q0_scaled, x0, acc0);

                // -- batch 1: bytes 8..15 of low128 → acc1 ---------------
                let q1_i32 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(low128, 8));
                let q1_f32 = _mm256_cvtepi32_ps(q1_i32);
                let q1_scaled = _mm256_mul_ps(q1_f32, d_vec);
                let x1 = _mm256_loadu_ps(x_ptr.add(8));
                acc1 = _mm256_fmadd_ps(q1_scaled, x1, acc1);

                // -- batch 2: bytes 0..7 of high128 → acc0 ---------------
                let q2_i32 = _mm256_cvtepi8_epi32(high128);
                let q2_f32 = _mm256_cvtepi32_ps(q2_i32);
                let q2_scaled = _mm256_mul_ps(q2_f32, d_vec);
                let x2 = _mm256_loadu_ps(x_ptr.add(16));
                acc0 = _mm256_fmadd_ps(q2_scaled, x2, acc0);

                // -- batch 3: bytes 8..15 of high128 → acc1 --------------
                let q3_i32 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(high128, 8));
                let q3_f32 = _mm256_cvtepi32_ps(q3_i32);
                let q3_scaled = _mm256_mul_ps(q3_f32, d_vec);
                let x3 = _mm256_loadu_ps(x_ptr.add(24));
                acc1 = _mm256_fmadd_ps(q3_scaled, x3, acc1);
            }

            // -- horizontal sum: acc0 + acc1 → scalar --------------------
            let acc = _mm256_add_ps(acc0, acc1);
            let low = _mm256_castps256_ps128(acc);
            let high = _mm256_extractf128_ps::<1>(acc);
            let sum128 = _mm_add_ps(low, high);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            let sum128 = _mm_hadd_ps(sum128, sum128);
            *out_val = _mm_cvtss_f32(sum128);
        }
    }

    /// SIMD sum of squares: `Σ x[i]²` using AVX2 FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn sum_squares_avx2(x: &[f32]) -> f32 {
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

    /// SIMD `out[i] = x[i] * scale * weight[i]` using AVX2.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn scale_weight_mul_avx2(
        x: &[f32],
        scale: f32,
        weight: &[f32],
        out: &mut [f32],
    ) {
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

    /// SIMD element-wise multiply using AVX2.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn elemul_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
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

    /// Accurate SIMD exp using range reduction + 5th-degree polynomial.
    ///
    /// Algorithm: reduce to [-ln2/2, ln2/2], evaluate minimax polynomial,
    /// scale by 2^n.  Relative error < 2e-6 — indistinguishable from
    /// `libm::expf` in practice.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp_ps(x: __m256) -> __m256 {
        let log2e = _mm256_set1_ps(core::f32::consts::LOG2_E);
        let ln2 = _mm256_set1_ps(core::f32::consts::LN_2);
        let magic = _mm256_set1_ps(12582912.0_f32); // 1.5 * 2^23

        // polynomial coefficients for exp(y), y ∈ [-ln2/2, ln2/2]
        let p0 = _mm256_set1_ps(1.0_f32);
        let p1 = _mm256_set1_ps(1.0_f32);
        let p2 = _mm256_set1_ps(0.5_f32);
        let p3 = _mm256_set1_ps(0.166_666_67_f32); // 1/6
        let p4 = _mm256_set1_ps(0.041_666_668_f32); // 1/24
        let p5 = _mm256_set1_ps(0.008_333_334_f32); // 1/120

        // 1. k = round(x * log2(e))
        let a = _mm256_mul_ps(x, log2e);
        let k = _mm256_sub_ps(_mm256_add_ps(a, magic), magic);

        // 2. r = x - k * ln(2)  (reduced argument)
        let r = _mm256_fnmadd_ps(k, ln2, x); // x - k*ln2 = -(k*ln2 - x)

        // 3. polynomial: p5*r + p4 → Horner
        let poly = _mm256_fmadd_ps(p5, r, p4);
        let poly = _mm256_fmadd_ps(poly, r, p3);
        let poly = _mm256_fmadd_ps(poly, r, p2);
        let poly = _mm256_fmadd_ps(poly, r, p1);
        let poly = _mm256_fmadd_ps(poly, r, p0);

        // 4. scale by 2^k: (k+127) << 23
        let k_i32 = _mm256_cvtps_epi32(k);
        let pow2 = _mm256_slli_epi32::<23>(_mm256_add_epi32(k_i32, _mm256_set1_epi32(127)));
        let pow2_f = _mm256_castsi256_ps(pow2);

        _mm256_mul_ps(poly, pow2_f)
    }

    /// SIMD SiLU using accurate polynomial exp.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn silu_avx2(x: &[f32], out: &mut [f32]) {
        let n = x.len();
        let one = _mm256_set1_ps(1.0_f32);
        let zero = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= n {
            let xv = _mm256_loadu_ps(x.as_ptr().add(i));
            let neg_x = _mm256_sub_ps(zero, xv);
            let exp_neg = exp_ps(neg_x);
            let denom = _mm256_add_ps(one, exp_neg);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_div_ps(xv, denom));
            i += 8;
        }
        while i < n {
            out[i] = x[i] / (1.0 + (-x[i]).exp());
            i += 1;
        }
    }

    /// SIMD dot product using AVX2 FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
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

    /// SIMD element-wise add using AVX2.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
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

    /// SIMD weighted accumulate using AVX2 FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn weighted_add_avx2(acc: &mut [f32], src: &[f32], weight: f32) {
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

    /// SIMD prefix softmax using AVX2 + accurate polynomial exp.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    pub(crate) unsafe fn softmax_prefix_avx2(row: &mut [f32], len: usize) {
        let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);

        // 1. find max
        let mut max_vec = neg_inf;
        let mut i = 0;
        while i + 8 <= len {
            let v = _mm256_loadu_ps(row.as_ptr().add(i));
            max_vec = _mm256_max_ps(max_vec, v);
            i += 8;
        }
        let low = _mm256_castps256_ps128(max_vec);
        let high = _mm256_extractf128_ps::<1>(max_vec);
        let mut max_val = _mm_cvtss_f32(_mm_max_ps(low, high));
        // handle tail
        for j in i..len {
            max_val = max_val.max(row[j]);
        }

        if max_val == f32::NEG_INFINITY {
            let uniform = 1.0 / (len as f32);
            let u = _mm256_set1_ps(uniform);
            let mut i = 0;
            while i + 8 <= len {
                _mm256_storeu_ps(row.as_mut_ptr().add(i), u);
                i += 8;
            }
            for j in i..len {
                row[j] = uniform;
            }
            return;
        }

        // 2. exp(x - max) + sum
        let max_splat = _mm256_set1_ps(max_val);
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        i = 0;
        while i + 16 <= len {
            let v0 = _mm256_loadu_ps(row.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(row.as_ptr().add(i + 8));
            let e0 = exp_ps(_mm256_sub_ps(v0, max_splat));
            let e1 = exp_ps(_mm256_sub_ps(v1, max_splat));
            _mm256_storeu_ps(row.as_mut_ptr().add(i), e0);
            _mm256_storeu_ps(row.as_mut_ptr().add(i + 8), e1);
            sum0 = _mm256_add_ps(sum0, e0);
            sum1 = _mm256_add_ps(sum1, e1);
            i += 16;
        }
        while i + 8 <= len {
            let v = _mm256_loadu_ps(row.as_ptr().add(i));
            let e = exp_ps(_mm256_sub_ps(v, max_splat));
            _mm256_storeu_ps(row.as_mut_ptr().add(i), e);
            sum0 = _mm256_add_ps(sum0, e);
            i += 8;
        }
        let sum_vec = _mm256_add_ps(sum0, sum1);
        let low = _mm256_castps256_ps128(sum_vec);
        let high = _mm256_extractf128_ps::<1>(sum_vec);
        let sum128 = _mm_add_ps(low, high);
        let sum128 = _mm_hadd_ps(sum128, sum128);
        let sum128 = _mm_hadd_ps(sum128, sum128);
        let mut total = _mm_cvtss_f32(sum128);
        // tail
        for j in i..len {
            let e = (row[j] - max_val).exp();
            row[j] = e;
            total += e;
        }

        // 3. scale by 1/sum
        let inv = _mm256_set1_ps(total.recip());
        let mut i = 0;
        while i + 8 <= len {
            let v = _mm256_loadu_ps(row.as_ptr().add(i));
            _mm256_storeu_ps(row.as_mut_ptr().add(i), _mm256_mul_ps(v, inv));
            i += 8;
        }
        for j in i..len {
            row[j] *= total.recip();
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
            let out_ptr = dst.as_mut_ptr().add(out_offset) as *mut f32;

            // helper: dequantize 16 i8 values → 4 × float32x4_t
            #[inline(always)]
            unsafe fn process16(src: int8x16_t, scale: float32x4_t, out: *mut f32) {
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

            // 16 quants → bytes 0..15, 16 quants → bytes 16..31
            process16(q0, d_vec, out_ptr);
            process16(q1, d_vec, out_ptr.add(16));
        }
    }

    /// Fused Q8_0 dot product using NEON.  32 quants per block in 8 batches
    /// of 4 f32 values each (i8 → i16 → i32 → f32 → mul scale → fma with x).
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is supported.
    #[target_feature(enable = "neon")]
    pub unsafe fn matmul_q8_0_decode_neon(
        x: &[f32],
        data: &[u8],
        _out_features: usize,
        blocks_per_row: usize,
        out: &mut [f32],
    ) {
        for (row, out_val) in out.iter_mut().enumerate() {
            let row_start = row * blocks_per_row;
            let mut acc = vdupq_n_f32(0.0);

            for b in 0..blocks_per_row {
                let byte_offset = (row_start + b) * Q8_0_TYPE_SIZE;
                let base_ptr = data.as_ptr().add(byte_offset);

                let d_bits = u16::from_le_bytes(*(base_ptr as *const [u8; 2]));
                let d = f16::from_bits(d_bits).to_f32();
                let d_vec = vdupq_n_f32(d);

                let quants_ptr = base_ptr.add(2) as *const i8;
                let q0 = vld1q_s8(quants_ptr);
                let q1 = vld1q_s8(quants_ptr.add(16));

                let x_offset = b * Q8_0_BLOCK_SIZE;
                let x_ptr = x.as_ptr().add(x_offset) as *const f32;

                // helper: process 16 i8 → 4 × f32, fma with x, accumulate
                unsafe fn fma16(
                    src: int8x16_t,
                    scale: float32x4_t,
                    xp: *const f32,
                    acc: &mut float32x4_t,
                ) {
                    let i16_lo = vmovl_s8(vget_low_s8(src));
                    let i16_hi = vmovl_s8(vget_high_s8(src));

                    let f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16_lo))), scale);
                    *acc = vfmaq_f32(*acc, f0, vld1q_f32(xp));

                    let f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16_lo))), scale);
                    *acc = vfmaq_f32(*acc, f1, vld1q_f32(xp.add(4)));

                    let f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16_hi))), scale);
                    *acc = vfmaq_f32(*acc, f2, vld1q_f32(xp.add(8)));

                    let f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16_hi))), scale);
                    *acc = vfmaq_f32(*acc, f3, vld1q_f32(xp.add(12)));
                }

                fma16(q0, d_vec, x_ptr, &mut acc);
                fma16(q1, d_vec, x_ptr.add(16), &mut acc);
            }

            *out_val = vgetq_lane_f32::<0>(acc)
                + vgetq_lane_f32::<1>(acc)
                + vgetq_lane_f32::<2>(acc)
                + vgetq_lane_f32::<3>(acc);
        }
    }

    /// SIMD sum of squares using NEON FMA.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn sum_squares_neon(x: &[f32]) -> f32 {
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

    /// SIMD `out[i] = x[i] * scale * weight[i]` using NEON.
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

    /// SIMD element-wise multiply using NEON.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn elemul_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
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

    /// Accurate SIMD exp using NEON (range reduction + polynomial).
    #[target_feature(enable = "neon")]
    unsafe fn exp_ps(x: float32x4_t) -> float32x4_t {
        let log2e = vdupq_n_f32(core::f32::consts::LOG2_E);
        let ln2 = vdupq_n_f32(core::f32::consts::LN_2);
        let magic = vdupq_n_f32(12582912.0_f32);
        let p0 = vdupq_n_f32(1.0_f32);
        let p1 = vdupq_n_f32(1.0_f32);
        let p2 = vdupq_n_f32(0.5_f32);
        let p3 = vdupq_n_f32(0.166_666_67_f32);
        let p4 = vdupq_n_f32(0.041_666_668_f32);
        let p5 = vdupq_n_f32(0.008_333_334_f32);

        let a = vmulq_f32(x, log2e);
        let k = vsubq_f32(vaddq_f32(a, magic), magic);
        let r = vfmsq_f32(x, k, ln2); // x - k * ln2

        let poly = vfmaq_f32(p4, p5, r);
        let poly = vfmaq_f32(p3, poly, r);
        let poly = vfmaq_f32(p2, poly, r);
        let poly = vfmaq_f32(p1, poly, r);
        let poly = vfmaq_f32(p0, poly, r);

        let k_i32 = vcvtq_s32_f32(k);
        let pow2 = vshlq_s32(vaddq_s32(k_i32, vdupq_n_s32(127)), vdupq_n_s32(23));
        let pow2_f = vreinterpretq_f32_s32(pow2);
        vmulq_f32(poly, pow2_f)
    }

    /// SIMD SiLU using NEON with accurate polynomial exp.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn silu_neon(x: &[f32], out: &mut [f32]) {
        let n = x.len();
        let one = vdupq_n_f32(1.0_f32);
        let zero = vdupq_n_f32(0.0_f32);
        let mut i = 0;

        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let neg_x = vsubq_f32(zero, xv);
            let exp_neg = exp_ps(neg_x);
            let denom = vaddq_f32(one, exp_neg);
            vst1q_f32(out.as_mut_ptr().add(i), vdivq_f32(xv, denom));
            i += 4;
        }
        while i < n {
            out[i] = x[i] / (1.0 + (-x[i]).exp());
            i += 1;
        }
    }

    /// SIMD dot product using NEON FMA.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
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
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn add_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
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

    /// SIMD weighted accumulate using NEON FMA.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn weighted_add_neon(acc: &mut [f32], src: &[f32], weight: f32) {
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

    /// SIMD prefix softmax using NEON + accurate polynomial exp.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn softmax_prefix_neon(row: &mut [f32], len: usize) {
        let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
        let zero = vdupq_n_f32(0.0);

        // 1. find max
        let mut max_vec = neg_inf;
        let mut i = 0;
        while i + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            max_vec = vmaxq_f32(max_vec, v);
            i += 4;
        }
        let mut max_val = vgetq_lane_f32::<0>(max_vec)
            .max(vgetq_lane_f32::<1>(max_vec))
            .max(vgetq_lane_f32::<2>(max_vec))
            .max(vgetq_lane_f32::<3>(max_vec));
        for j in i..len {
            max_val = max_val.max(row[j]);
        }

        if max_val == f32::NEG_INFINITY {
            let uniform = 1.0 / (len as f32);
            let u = vdupq_n_f32(uniform);
            let mut i = 0;
            while i + 4 <= len {
                vst1q_f32(row.as_mut_ptr().add(i), u);
                i += 4;
            }
            for j in i..len {
                row[j] = uniform;
            }
            return;
        }

        // 2. exp(x - max) + sum
        let max_splat = vdupq_n_f32(max_val);
        let mut sum_vec = vdupq_n_f32(0.0);
        i = 0;
        while i + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            let shifted = vsubq_f32(v, max_splat);
            let e = exp_ps(shifted);
            vst1q_f32(row.as_mut_ptr().add(i), e);
            sum_vec = vaddq_f32(sum_vec, e);
            i += 4;
        }
        let mut total = vgetq_lane_f32::<0>(sum_vec)
            + vgetq_lane_f32::<1>(sum_vec)
            + vgetq_lane_f32::<2>(sum_vec)
            + vgetq_lane_f32::<3>(sum_vec);
        for j in i..len {
            let e = (row[j] - max_val).exp();
            row[j] = e;
            total += e;
        }

        // 3. scale
        let inv = vdupq_n_f32(total.recip());
        let mut i = 0;
        while i + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            vst1q_f32(row.as_mut_ptr().add(i), vmulq_f32(v, inv));
            i += 4;
        }
        for j in i..len {
            row[j] *= total.recip();
        }
    }
}

// ---------------------------------------------------------------------------
// fused Q8_0 decode (seq_len = 1)
// ---------------------------------------------------------------------------
//
// When the input is a single row (decode), each output value is a dot product
// between the input row and one dequantized weight row.  Skipping the dense
// w_block buffer and computing the dot product directly from the compressed
// Q8_0 data eliminates the temporary allocation and sgemm dispatch overhead.
//
//   out[j] = Σᵢ x[i] · dequant(w[j][i])
//
// Each Q8_0 block contributes 32 terms:  Σₖ x[off+k] · (qₖ · d).

use crate::quant::QuantizedWeight;

/// Fused Q8_0 dot product for single-row input (decode, seq_len = 1).
///
/// Computes `out[j] = Σᵢ x[i] · dequant(w[j][i])` for every output row j
/// directly from the compressed weight data.  No dense temporary is allocated.
///
/// # Panics
///
/// Panics if `x.len() != in_features`, `out.len() != out_features`, or the
/// weight data is not a valid Q8_0 encoding.
#[inline]
pub(crate) fn matmul_q8_0_decode(x: &[f32], w: &QuantizedWeight, out: &mut [f32]) {
    debug_assert_eq!(x.len(), w.in_features());
    debug_assert_eq!(out.len(), w.out_features());

    let blocks_per_row = w.in_features() / Q8_0_BLOCK_SIZE;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return x86_64::matmul_q8_0_decode_avx2_fma(
                    x,
                    &w.data,
                    w.out_features(),
                    blocks_per_row,
                    out,
                );
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe {
                return aarch64::matmul_q8_0_decode_neon(
                    x,
                    &w.data,
                    w.out_features(),
                    blocks_per_row,
                    out,
                );
            }
        }
    }
    matmul_q8_0_decode_scalar(x, &w.data, w.out_features(), blocks_per_row, out);
}

// -- scalar fallback --------------------------------------------------------

fn matmul_q8_0_decode_scalar(
    x: &[f32],
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
            let d_bits = u16::from_le_bytes(data[byte_offset..byte_offset + 2].try_into().unwrap());
            let d = f16::from_bits(d_bits).to_f32();
            let x_offset = b * Q8_0_BLOCK_SIZE;
            for j in 0..Q8_0_BLOCK_SIZE {
                let q = data[byte_offset + 2 + j] as i8;
                sum += x[x_offset + j] * (q as f32) * d;
            }
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
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

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

/// SIMD SiLU: `out[i] = x[i] / (1 + exp(-x[i]))`.
///
/// Uses a fast polynomial exp approximation suitable for inference.
/// Maximum relative error < 2% in the critical region [-8, 8].
///
/// NOTE: uses accurate polynomial exp (not Schraudolph).
/// Currently not wired — libm::expf is faster on scalar.
#[inline]
#[allow(dead_code)]
pub(crate) fn silu(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return x86_64::silu_avx2(x, out);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe {
                return aarch64::silu_neon(x, out);
            }
        }
    }
    for i in 0..x.len() {
        out[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

/// SIMD dot product: `Σ a[i] · b[i]`.
#[inline]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

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

/// SIMD weighted accumulate: `acc[i] += weight * src[i]`.
#[inline]
pub(crate) fn weighted_add(acc: &mut [f32], src: &[f32], weight: f32) {
    debug_assert_eq!(acc.len(), src.len());

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

/// SIMD element-wise addition: `out[i] = a[i] + b[i]`.
#[inline]
pub(crate) fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
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
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());

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
///
/// Uses accurate polynomial exp.  Currently not wired — libm::expf is faster.
#[inline]
#[allow(dead_code)]
pub(crate) fn softmax_prefix(row: &mut [f32], len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return x86_64::softmax_prefix_avx2(row, len);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe {
                return aarch64::softmax_prefix_neon(row, len);
            }
        }
    }
    // scalar fallback
    let max_val = row[..len].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max_val == f32::NEG_INFINITY {
        let uniform = 1.0 / (len as f32);
        for slot in row.iter_mut().take(len) {
            *slot = uniform;
        }
        return;
    }
    let mut sum = 0.0;
    for slot in row.iter_mut().take(len) {
        *slot = (*slot - max_val).exp();
        sum += *slot;
    }
    let inv_sum = sum.recip();
    for slot in row.iter_mut().take(len) {
        *slot *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const BOUNDARY_LENGTHS: &[usize] = &[0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33];

    fn patterned_vec(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + phase) * 0.37).sin() * 3.0 + (i % 5) as f32 * 0.125)
            .collect()
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

            let silu_input: Vec<f32> = a.iter().map(|v| v.clamp(-8.0, 8.0)).collect();
            let expected: Vec<f32> = silu_input.iter().map(|x| x / (1.0 + (-x).exp())).collect();
            silu(&silu_input, &mut got);
            assert_close(&format!("silu len={len}"), &got, &expected, 3e-3, 3e-3);

            let mut row = patterned_vec(len + 3, 2.5);
            let mut expected = row.clone();
            scalar_softmax_prefix(&mut expected, len);
            softmax_prefix(&mut row, len);
            assert_close(
                &format!("softmax_prefix len={len}"),
                &row,
                &expected,
                3e-5,
                3e-5,
            );
        }
    }

    fn scalar_softmax_prefix(row: &mut [f32], len: usize) {
        let max_val = row[..len].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        if max_val == f32::NEG_INFINITY {
            let uniform = 1.0 / len as f32;
            for slot in row.iter_mut().take(len) {
                *slot = uniform;
            }
            return;
        }
        let mut sum = 0.0;
        for slot in row.iter_mut().take(len) {
            *slot = (*slot - max_val).exp();
            sum += *slot;
        }
        let inv_sum = sum.recip();
        for slot in row.iter_mut().take(len) {
            *slot *= inv_sum;
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
        data.extend(std::iter::repeat(0u8).take(29));

        // block 1: scale=1.0, quants=[-128, 127, 0, ...]
        let s1 = f16::from_f32(1.0);
        data.extend_from_slice(&s1.to_bits().to_le_bytes());
        data.push((-128i8) as u8);
        data.push(127i8 as u8);
        data.push(0u8);
        data.extend(std::iter::repeat(0u8).take(29));

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

    #[test]
    fn explicit_avx2_call_matches_scalar() {
        let blocks = 4;
        let (data, _expected) = make_row(blocks);
        let mut scalar_out = vec![0.0f32; blocks * 32];
        let mut avx2_out = vec![0.0f32; blocks * 32];

        dequantize_row_scalar(&data, 0, blocks, &mut scalar_out);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
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

    /// Timing benchmark: compare scalar vs dispatch dequantization on a
    /// realistic-sized weight matrix (e.g. 4096 × 4096).
    ///
    /// Run with:
    ///   cargo test --release -- bench_dequant --nocapture --ignored
    #[test]
    #[ignore]
    fn bench_dequant() {
        const OUT_FEATURES: usize = 4096;
        const IN_FEATURES: usize = 4096;
        const BLOCKS_PER_ROW: usize = IN_FEATURES / Q8_0_BLOCK_SIZE; // 128
        const WARMUP_ITERS: usize = 20;
        const MEASURE_ITERS: usize = 200;

        println!("\n--- Q8_0 dequantization benchmark ---");
        println!(
            "weight shape: [{}, {}]  blocks/row: {}  row size: {} f32\n",
            OUT_FEATURES, IN_FEATURES, BLOCKS_PER_ROW, IN_FEATURES
        );

        let data = random_q8_0_data(OUT_FEATURES, IN_FEATURES);
        let mut dst = vec![0.0f32; IN_FEATURES];

        // detect which kernel the dispatch path will use
        let kernel_name = {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    "avx2"
                } else {
                    "scalar (no avx2)"
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if is_aarch64_feature_detected!("neon") {
                    "neon"
                } else {
                    "scalar (no neon)"
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                "scalar (fallback)"
            }
        };
        println!("dispatch kernel: {kernel_name}\n");

        // warmup — scalar
        for _ in 0..WARMUP_ITERS {
            for row in 0..OUT_FEATURES {
                dequantize_row_scalar(&data, row * BLOCKS_PER_ROW, BLOCKS_PER_ROW, &mut dst);
            }
        }

        // measure — scalar
        let t0 = std::time::Instant::now();
        for _ in 0..MEASURE_ITERS {
            for row in 0..OUT_FEATURES {
                dequantize_row_scalar(&data, row * BLOCKS_PER_ROW, BLOCKS_PER_ROW, &mut dst);
            }
        }
        let scalar_elapsed = t0.elapsed();
        let scalar_total_rows = MEASURE_ITERS * OUT_FEATURES;
        let scalar_us_per_row = scalar_elapsed.as_micros() as f64 / scalar_total_rows as f64;

        println!("scalar:    {scalar_elapsed:.2?}  ({MEASURE_ITERS} × {OUT_FEATURES} rows)");
        println!("           {scalar_us_per_row:.2} µs/row");

        // warmup — dispatch
        for _ in 0..WARMUP_ITERS {
            for row in 0..OUT_FEATURES {
                dequantize_q8_0_row(&data, row * BLOCKS_PER_ROW, BLOCKS_PER_ROW, &mut dst);
            }
        }

        // measure — dispatch
        let t0 = std::time::Instant::now();
        for _ in 0..MEASURE_ITERS {
            for row in 0..OUT_FEATURES {
                dequantize_q8_0_row(&data, row * BLOCKS_PER_ROW, BLOCKS_PER_ROW, &mut dst);
            }
        }
        let dispatch_elapsed = t0.elapsed();
        let dispatch_us_per_row = dispatch_elapsed.as_micros() as f64 / scalar_total_rows as f64;

        println!("dispatch:  {dispatch_elapsed:.2?}  ({MEASURE_ITERS} × {OUT_FEATURES} rows)");
        println!("           {dispatch_us_per_row:.2} µs/row");

        let speedup = scalar_us_per_row / dispatch_us_per_row;
        println!("\n  speedup: {speedup:.2}×");
        assert!(speedup >= 1.0, "SIMD path should not be slower than scalar");
    }

    // -- decode path correctness --------------------------------------------

    #[test]
    fn decode_path_matches_scalar() {
        use rand::Rng;
        let out_features = 64;
        let in_features = 256; // 8 blocks per row
        let data = random_q8_0_data(out_features, in_features);
        let w = QuantizedWeight::try_new(data.clone(), vec![out_features, in_features]).unwrap();

        let mut rng = rand::thread_rng();
        let x: Vec<f32> = (0..in_features)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect();

        let mut decode_out = vec![0.0f32; out_features];
        matmul_q8_0_decode(&x, &w, &mut decode_out);

        let mut scalar_out = vec![0.0f32; out_features];
        matmul_q8_0_decode_scalar(
            &x,
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
    fn decode_path_matches_blockwise_prefill_path() {
        use crate::backend::{Backend, CpuBackend};
        use rand::Rng;

        let out_features = 64;
        let in_features = 256;
        let data = random_q8_0_data(out_features, in_features);
        let w = QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap();

        let mut rng = rand::thread_rng();
        let x: Vec<f32> = (0..in_features)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect();

        let mut decode_out = vec![0.0f32; out_features];
        matmul_q8_0_decode(&x, &w, &mut decode_out);

        // Force the existing block-wise prefill implementation by using two
        // identical rows; CpuBackend only takes the fused decode branch for
        // seq_len == 1.
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
            "q8 decode vs blockwise prefill",
            &decode_out,
            prefill_first_row,
            1e-3,
            1e-3,
        );
    }

    /// Full decode path benchmark: compare fused decode vs block-wise sgemm
    /// on a realistic weight matrix.
    ///
    /// Run with:
    ///   cargo test --release -- bench_decode --nocapture --ignored
    #[test]
    #[ignore]
    fn bench_decode() {
        const OUT_FEATURES: usize = 4096;
        const IN_FEATURES: usize = 4096;
        const BLOCKS_PER_ROW: usize = IN_FEATURES / Q8_0_BLOCK_SIZE;
        const WARMUP_ITERS: usize = 40;
        const MEASURE_ITERS: usize = 400;

        println!("\n--- Fused Q8_0 decode benchmark ---");
        println!(
            "weight shape: [{}, {}]  blocks/row: {}\n",
            OUT_FEATURES, IN_FEATURES, BLOCKS_PER_ROW
        );

        let data = random_q8_0_data(OUT_FEATURES, IN_FEATURES);
        let w = QuantizedWeight::try_new(data.clone(), vec![OUT_FEATURES, IN_FEATURES]).unwrap();

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let x: Vec<f32> = (0..IN_FEATURES)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect();
        let mut out = vec![0.0f32; OUT_FEATURES];

        let kernel_name = {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    "avx2+fma"
                } else {
                    "scalar"
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if is_aarch64_feature_detected!("neon") {
                    "neon"
                } else {
                    "scalar"
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                "scalar"
            }
        };
        println!("dispatch kernel: {kernel_name}\n");

        // warmup — scalar fallback
        for _ in 0..WARMUP_ITERS {
            matmul_q8_0_decode_scalar(&x, &data, OUT_FEATURES, BLOCKS_PER_ROW, &mut out);
        }

        // measure — scalar
        let t0 = std::time::Instant::now();
        for _ in 0..MEASURE_ITERS {
            matmul_q8_0_decode_scalar(&x, &data, OUT_FEATURES, BLOCKS_PER_ROW, &mut out);
        }
        let scalar_elapsed = t0.elapsed();
        let scalar_us = scalar_elapsed.as_micros() as f64 / MEASURE_ITERS as f64;
        println!("scalar:     {scalar_elapsed:.2?}  ({MEASURE_ITERS} iterations)");
        println!("            {scalar_us:.2} µs/call");

        // warmup — dispatch
        for _ in 0..WARMUP_ITERS {
            matmul_q8_0_decode(&x, &w, &mut out);
        }

        // measure — dispatch
        let t0 = std::time::Instant::now();
        for _ in 0..MEASURE_ITERS {
            matmul_q8_0_decode(&x, &w, &mut out);
        }
        let dispatch_elapsed = t0.elapsed();
        let dispatch_us = dispatch_elapsed.as_micros() as f64 / MEASURE_ITERS as f64;
        println!("dispatch:   {dispatch_elapsed:.2?}  ({MEASURE_ITERS} iterations)");
        println!("            {dispatch_us:.2} µs/call");

        let speedup = scalar_us / dispatch_us;
        println!("\n  speedup: {speedup:.2}×");
        assert!(speedup >= 1.0, "SIMD path should not be slower than scalar");
    }
}
