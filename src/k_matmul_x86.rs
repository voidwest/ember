//! Optimized x86 (AVX2+FMA) compressed-resident K-quant matmul kernels.
//!
//! One optimized path for the current development target, per the v0.3
//! contracts: explicit feature detection and dispatch, scalar fallback
//! recorded (never silent), no hook changes, no persistent f32 expansion.
//! The scalar kernels in `k_matmul` remain the correctness oracle.
//!
//! Q6_K dequantization math mirrors `quant_k::dequant_q6_k` exactly:
//! per 32-lane group, four 6-bit sub-values per lane are reconstructed
//! from `ql`/`qh` nibble pairs and scaled by the per-16-lane int8
//! scales. The dot product accumulates in f32 with FMA.

use crate::quant_k::{KQuantDtype, KQuantWeight, Q6_K_BLOCK_BYTES, QK_K};

/// Whether the AVX2+FMA feature set required by these kernels is present.
pub fn avx2_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX2 dispatch entry: `dst = src × w` over the packed K-quant weight.
///
/// # Safety
///
/// The caller must have validated the layout (`src.len() == rows *
/// in_features`, `dst.len() == rows * out_features`, `dst`
/// zero-initialized) and `avx2_supported()`; the loader records the
/// execution decision at load time, so a mismatch here is a bug, not a
/// fallback.
#[cfg(target_arch = "x86_64")]
pub unsafe fn matmul_k_avx2_into(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    match w.dtype() {
        KQuantDtype::Q6K => {
            x86_64::matmul_q6_k_avx2_into(src, rows, w, dst);
            Ok(())
        }
        KQuantDtype::Q4K => {
            Err("matmul_k_avx2: q4_k AVX2 kernel not yet wired (v0.3 commit 8)".to_string())
        }
    }
}

/// Non-x86 stub: the dispatch entry never reaches this on non-x86 builds
/// because `avx2_supported()` is false there, but the error is explicit.
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn matmul_k_avx2_into(
    _src: &[f32],
    _rows: usize,
    _w: &KQuantWeight,
    _dst: &mut [f32],
) -> Result<(), String> {
    Err("matmul_k_avx2: AVX2 kernels require x86_64".to_string())
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;
    use core::arch::x86_64::*;

    /// Reconstruct the four signed 6-bit sub-values for one 8-lane chunk
    /// of a 32-lane group (bytes are `u8`; values are 6-bit minus 32).
    ///
    /// The per-byte nibble math uses 16-bit unit shifts with masks that
    /// recover each byte's contribution independently:
    /// `(x >> s) & m` on 16-bit units equals per-byte `(b >> s) & m`
    /// because cross-byte carry lands in masked-out bits.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn q6_values(
        ql_lo: __m128i,
        ql_hi: __m128i,
        qh: __m128i,
    ) -> (__m128i, __m128i, __m128i, __m128i) {
        let mask_0f = _mm_set1_epi8(0x0F);
        let mask_03 = _mm_set1_epi8(0x03);
        let mask_f0 = _mm_set1_epi8(-16); // 0xF0
        let thirty_two = _mm_set1_epi8(32);

        let qh_lo = _mm_and_si128(qh, mask_03);
        let qh_sh2 = _mm_and_si128(_mm_srli_epi16(qh, 2), mask_03);
        let qh_sh4 = _mm_and_si128(_mm_srli_epi16(qh, 4), mask_03);
        let qh_sh6 = _mm_and_si128(_mm_srli_epi16(qh, 6), mask_03);
        let shl4 = |v: __m128i| _mm_and_si128(_mm_slli_epi16(v, 4), mask_f0);

        let q1 = _mm_sub_epi8(
            _mm_or_si128(_mm_and_si128(ql_lo, mask_0f), shl4(qh_lo)),
            thirty_two,
        );
        let q2 = _mm_sub_epi8(
            _mm_or_si128(_mm_and_si128(ql_hi, mask_0f), shl4(qh_sh2)),
            thirty_two,
        );
        let q3 = _mm_sub_epi8(
            _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16(ql_lo, 4), mask_0f),
                shl4(qh_sh4),
            ),
            thirty_two,
        );
        let q4 = _mm_sub_epi8(
            _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16(ql_hi, 4), mask_0f),
                shl4(qh_sh6),
            ),
            thirty_two,
        );
        (q1, q2, q3, q4)
    }

    /// Accumulate one 8-lane dot: `acc += x[0..8] * q * scale`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot8_acc(q: __m128i, scale: f32, x: *const f32, acc: &mut __m256) {
        let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
        let qf = _mm256_mul_ps(qf, _mm256_set1_ps(scale));
        let xv = _mm256_loadu_ps(x);
        *acc = _mm256_fmadd_ps(xv, qf, *acc);
    }

    /// AVX2 Q6_K matmul over packed super-blocks.
    ///
    /// Layout invariants are the `KQuantWeight` construction guarantees:
    /// `data.len() == out_features * blocks_per_row * 210` and
    /// `in_features` is a multiple of 256, so every slice below is in
    /// bounds.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn matmul_q6_k_avx2_into(
        src: &[f32],
        rows: usize,
        w: &KQuantWeight,
        dst: &mut [f32],
    ) {
        let in_features = w.in_features();
        let out_features = w.out_features();
        let blocks_per_row = w.blocks_per_row();
        let data = w.data();

        for j in 0..out_features {
            let row_bytes = j * blocks_per_row * Q6_K_BLOCK_BYTES;
            for b in 0..blocks_per_row {
                let block =
                    &data[row_bytes + b * Q6_K_BLOCK_BYTES..row_bytes + (b + 1) * Q6_K_BLOCK_BYTES];
                let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
                let scales = &block[192..208];
                let ql = &block[0..128];
                let qh = &block[128..192];
                let x_base = b * QK_K;

                for half in 0..2 {
                    let q = half * 64;
                    let h = half * 32;
                    let s = half * 8;
                    let y = half * 128;
                    for c in 0..4 {
                        let l8 = c * 8;
                        let ql_lo = _mm_loadl_epi64(ql[q + l8..].as_ptr() as *const __m128i);
                        let ql_hi = _mm_loadl_epi64(ql[q + 32 + l8..].as_ptr() as *const __m128i);
                        let qh8 = _mm_loadl_epi64(qh[h + l8..].as_ptr() as *const __m128i);
                        let (q1, q2, q3, q4) = q6_values(ql_lo, ql_hi, qh8);
                        let sc_idx = s + c / 2;
                        // scales are int8 in the reference; >= 0x80 are negative
                        let sc = |i: usize| i32::from(i8::from_le_bytes([scales[i]])) as f32;
                        let d1 = d * sc(sc_idx);
                        let d2 = d * sc(sc_idx + 2);
                        let d3 = d * sc(sc_idx + 4);
                        let d4 = d * sc(sc_idx + 6);

                        for r in 0..rows {
                            // one accumulator per output row: the whole
                            // block's 256 values dot into the same output
                            // element `dst[r * out_features + j]`
                            let mut acc = _mm256_setzero_ps();
                            let xr = r * in_features + x_base;
                            let x = src.as_ptr().add(xr + y);
                            dot8_acc(q1, d1, x.add(c * 8), &mut acc);
                            dot8_acc(q2, d2, x.add(32 + c * 8), &mut acc);
                            dot8_acc(q3, d3, x.add(64 + c * 8), &mut acc);
                            dot8_acc(q4, d4, x.add(96 + c * 8), &mut acc);
                            let mut lanes = [0.0f32; 8];
                            _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
                            let block_sum = lanes.iter().sum::<f32>();
                            dst[r * out_features + j] += block_sum;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k_matmul::matmul_k_scalar_into;
    use crate::tensor::CpuTensor;

    fn seeded_q6_blocks(blocks: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut bytes = vec![0u8; blocks * Q6_K_BLOCK_BYTES];
        for byte in &mut bytes {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (state >> 33) as u8;
        }
        for block in bytes.chunks_exact_mut(Q6_K_BLOCK_BYTES) {
            let bits = u16::from_le_bytes([block[208], block[209]]);
            let bits = bits & 0x7FFF;
            let bits = if bits >= 0x7C00 { 0x3C00 } else { bits };
            block[208..210].copy_from_slice(&bits.to_le_bytes());
        }
        bytes
    }

    fn seeded_activations(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (state >> 33) as u32 as i32;
            values.push(v as f32 * (4.0 / 2147483648.0));
        }
        values
    }

    /// The eager-f32 oracle (dequant-then-gemm), identical to the scalar
    /// kernel tests' reference.
    fn eager_reference(w: &KQuantWeight, src: &[f32], rows: usize) -> Vec<f32> {
        let w_full = w.dequantize_all().transpose();
        let x = CpuTensor::from_data(vec![rows, w.in_features()], src.to_vec());
        x.matmul(&w_full).data().to_vec()
    }

    fn assert_gate_d(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
        let mut max_abs = 0.0f32;
        for (a, e) in actual.iter().zip(expected) {
            max_abs = max_abs.max((a - e).abs());
        }
        let gate = 1e-4 * scale;
        assert!(
            max_abs <= gate,
            "Gate D exceeded: max_abs {max_abs} > {gate} (scale {scale})"
        );
    }

    fn run_avx2(src: &[f32], rows: usize, w: &KQuantWeight) -> Vec<f32> {
        let mut dst = vec![0.0f32; rows * w.out_features()];
        unsafe {
            matmul_k_avx2_into(src, rows, w, &mut dst).expect("avx2 dispatch on a q6 weight");
        }
        dst
    }

    #[test]
    fn avx2_q6_k_matches_eager_oracle_across_shapes() {
        if !avx2_supported() {
            eprintln!("skipped: AVX2 unavailable");
            return;
        }
        for (rows, in_features, out_features) in [
            (1, 256, 128),
            (2, 512, 512),
            (8, 2048, 1536),
            (32, 256, 512),
            (1, 3584, 256),
            (4, 2048, 2048),
        ] {
            let blocks = in_features / 256 * out_features;
            let weight = KQuantWeight::new(
                seeded_q6_blocks(blocks, 0x71_00 + rows as u64 * 101 + in_features as u64),
                [out_features, in_features],
                KQuantDtype::Q6K,
            );
            let src = seeded_activations(rows * in_features, 0x7C_00 + out_features as u64);
            let expected = eager_reference(&weight, &src, rows);
            let actual = run_avx2(&src, rows, &weight);
            assert_gate_d(&actual, &expected);
        }
    }

    #[test]
    fn avx2_q6_k_matches_scalar_within_gate_d() {
        // both kernels read the same packed bytes and accumulate per
        // (block, chunk); the AVX2 path uses FMA (single rounding) while
        // the scalar path rounds the product first, so agreement is
        // within Gate D, not bit-for-bit
        if !avx2_supported() {
            eprintln!("skipped: AVX2 unavailable");
            return;
        }
        let weight = KQuantWeight::new(seeded_q6_blocks(6, 0x7D), [2, 768], KQuantDtype::Q6K);
        let src = seeded_activations(3 * 768, 0x7E);
        let mut scalar = vec![0.0f32; 3 * 2];
        matmul_k_scalar_into(&src, 3, &weight, &mut scalar).unwrap();
        let avx2 = run_avx2(&src, 3, &weight);
        assert_gate_d(&avx2, &scalar);
    }

    #[test]
    fn avx2_q6_k_is_deterministic() {
        if !avx2_supported() {
            eprintln!("skipped: AVX2 unavailable");
            return;
        }
        let weight = KQuantWeight::new(seeded_q6_blocks(4, 0x7F), [2, 512], KQuantDtype::Q6K);
        let src = seeded_activations(2 * 512, 0x80);
        assert_eq!(run_avx2(&src, 2, &weight), run_avx2(&src, 2, &weight));
    }

    #[test]
    fn q4_k_avx2_arm_is_hard_error_until_wired() {
        if !avx2_supported() {
            eprintln!("skipped: AVX2 unavailable");
            return;
        }
        let weight = KQuantWeight::new(vec![0u8; 2 * 144], [2, 256], KQuantDtype::Q4K);
        let mut dst = vec![0.0f32; 4];
        let err = unsafe { matmul_k_avx2_into(&[0.0f32; 512], 2, &weight, &mut dst) }.unwrap_err();
        assert!(err.contains("q4_k"), "unexpected error: {err}");
    }
}
