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

use crate::quant_k::{
    get_scale_min_k4, KQuantDtype, KQuantWeight, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, QK_K,
};

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
            x86_64::matmul_q4_k_avx2_into(src, rows, w, dst);
            Ok(())
        }
    }
}

/// Column-parallel AVX2 decode matvec (rows = 1): the output dimension is
/// split across the rayon pool; each task writes a disjoint `dst` range with
/// the identical per-column accumulation order, so results are bit-identical
/// to the serial kernel. Only ever called for rows = 1.
///
/// # Safety
///
/// The caller must have validated the layout (`src.len() == in_features`,
/// `dst.len() == out_features`, `dst` zero-initialized), `rows == 1`, and
/// `avx2_supported()`.
#[cfg(target_arch = "x86_64")]
pub unsafe fn matmul_k_avx2_into_parallel(src: &[f32], w: &KQuantWeight, dst: &mut [f32]) {
    match w.dtype() {
        KQuantDtype::Q6K => x86_64::matmul_q6_k_avx2_into_parallel(src, w, dst),
        KQuantDtype::Q4K => x86_64::matmul_q4_k_avx2_into_parallel(src, w, dst),
    }
}

/// Benchmark-only accessor for the pre-GEMV AVX2 row-1 kernel (Q4_K).
/// Not used by any production path; kept for old-vs-new measurements.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn bench_legacy_q4k_row1_avx2(
    src: &[f32],
    w: &KQuantWeight,
    j0: usize,
    dst: &mut [f32],
) {
    if crate::k_matmul_x86::avx2_supported() {
        x86_64::matmul_q4_k_avx2_row1_into(src, w, j0, dst);
    }
}

/// Benchmark-only accessor for the pre-GEMV AVX2 row-1 kernel (Q6_K).
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn bench_legacy_q6k_row1_avx2(
    src: &[f32],
    w: &KQuantWeight,
    j0: usize,
    dst: &mut [f32],
) {
    if crate::k_matmul_x86::avx2_supported() {
        x86_64::matmul_q6_k_avx2_row1_into(src, w, j0, dst);
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

/// Non-x86 stub for the column-parallel entry (unreachable; explicit).
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn matmul_k_avx2_into_parallel(_src: &[f32], _w: &KQuantWeight, _dst: &mut [f32]) {
    unreachable!("matmul_k_avx2_into_parallel: AVX2 kernels require x86_64")
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
        if rows == 1 {
            // SAFETY: same invariants; the single-row body is shared with
            // the column-parallel entry so serial and parallel stay
            // bit-identical.
            unsafe { matmul_q6_k_avx2_row1_into(src, w, 0, dst) }
            return;
        }
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

    /// Single-row (decode) Q6_K body over `dst_chunk` columns starting at
    /// `j0`, with one scalar accumulator per output column (bit-identical
    /// to the serial accumulation into a zeroed `dst`).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn matmul_q6_k_avx2_row1_into(
        src: &[f32],
        w: &KQuantWeight,
        j0: usize,
        dst_chunk: &mut [f32],
    ) {
        let blocks_per_row = w.blocks_per_row();
        let data = w.data();
        for (i, j) in (j0..j0 + dst_chunk.len()).enumerate() {
            let mut acc_j = 0.0f32;
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
                        let sc = |i: usize| i32::from(i8::from_le_bytes([scales[i]])) as f32;
                        let d1 = d * sc(sc_idx);
                        let d2 = d * sc(sc_idx + 2);
                        let d3 = d * sc(sc_idx + 4);
                        let d4 = d * sc(sc_idx + 6);

                        let mut acc = _mm256_setzero_ps();
                        let x = src.as_ptr().add(x_base + y);
                        dot8_acc(q1, d1, x.add(c * 8), &mut acc);
                        dot8_acc(q2, d2, x.add(32 + c * 8), &mut acc);
                        dot8_acc(q3, d3, x.add(64 + c * 8), &mut acc);
                        dot8_acc(q4, d4, x.add(96 + c * 8), &mut acc);
                        let mut lanes = [0.0f32; 8];
                        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
                        acc_j += lanes.iter().sum::<f32>();
                    }
                }
            }
            dst_chunk[i] = acc_j;
        }
    }

    /// Column-parallel Q6_K decode matvec (rows = 1): the output dimension
    /// is split across the rayon pool; each task writes a disjoint `dst`
    /// range with the identical per-column accumulation order, so results
    /// are bit-identical to the serial kernel.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn matmul_q6_k_avx2_into_parallel(
        src: &[f32],
        w: &KQuantWeight,
        dst: &mut [f32],
    ) {
        use rayon::prelude::*;
        let chunk = 256usize;
        dst.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(c, dst_chunk)| {
                let j0 = c * chunk;
                // SAFETY: same invariants as the serial body; each task
                // writes only its own disjoint dst_chunk.
                unsafe { matmul_q6_k_avx2_row1_into(src, w, j0, dst_chunk) }
            });
    }

    /// AVX2 Q4_K matmul over packed super-blocks.
    ///
    /// Q4_K values are `d * sc * q - min * m` with 6-bit (scale, min)
    /// pairs unpacked by `get_scale_min_k4` (12-byte bit-reshuffle); the
    /// 32-lane groups split into 32 low-nibble values and 32 high-nibble
    /// values with independent pairs.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn matmul_q4_k_avx2_into(
        src: &[f32],
        rows: usize,
        w: &KQuantWeight,
        dst: &mut [f32],
    ) {
        if rows == 1 {
            // SAFETY: same invariants; the single-row body is shared with
            // the column-parallel entry so serial and parallel stay
            // bit-identical.
            unsafe { matmul_q4_k_avx2_row1_into(src, w, 0, dst) }
            return;
        }
        let in_features = w.in_features();
        let out_features = w.out_features();
        let blocks_per_row = w.blocks_per_row();
        let data = w.data();

        for j in 0..out_features {
            let row_bytes = j * blocks_per_row * Q4_K_BLOCK_BYTES;
            for b in 0..blocks_per_row {
                let block =
                    &data[row_bytes + b * Q4_K_BLOCK_BYTES..row_bytes + (b + 1) * Q4_K_BLOCK_BYTES];
                let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                let min = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
                let scales = &block[4..16];
                let qs = &block[16..144];
                let x_base = b * QK_K;

                for g in 0..4 {
                    // 64 values per group: 32 low-nibble + 32 high-nibble
                    let (sc_low, m_low) = get_scale_min_k4(2 * g, scales);
                    let (sc_high, m_high) = get_scale_min_k4(2 * g + 1, scales);
                    let d1 = d * f32::from(sc_low);
                    let m1 = min * f32::from(m_low);
                    let d2 = d * f32::from(sc_high);
                    let m2 = min * f32::from(m_high);

                    let qs32 = &qs[g * 32..g * 32 + 32];
                    let mask_0f = _mm_set1_epi8(0x0F);

                    for c in 0..4 {
                        let l8 = c * 8;
                        let qs8 = _mm_loadl_epi64(qs32.as_ptr().byte_add(l8) as *const __m128i);
                        let q_low = _mm256_cvtepu8_epi32(_mm_and_si128(qs8, mask_0f));
                        let q_high =
                            _mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(qs8, 4), mask_0f));
                        // value = d*sc*q - min*m, computed as fma(q, d1, -m1)
                        let v_low = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(q_low),
                            _mm256_set1_ps(d1),
                            _mm256_set1_ps(-m1),
                        );
                        let v_high = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(q_high),
                            _mm256_set1_ps(d2),
                            _mm256_set1_ps(-m2),
                        );

                        for r in 0..rows {
                            let acc = _mm256_setzero_ps();
                            let xr = r * in_features + x_base + g * 64;
                            let x = src.as_ptr().add(xr);
                            let xv_low = _mm256_loadu_ps(x.add(c * 8));
                            let xv_high = _mm256_loadu_ps(x.add(32 + c * 8));
                            let acc = _mm256_fmadd_ps(xv_low, v_low, acc);
                            let acc = _mm256_fmadd_ps(xv_high, v_high, acc);
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

    /// Single-row (decode) Q4_K body over `dst_chunk` columns starting at
    /// `j0`, with one scalar accumulator per output column (bit-identical
    /// to the serial accumulation into a zeroed `dst`).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn matmul_q4_k_avx2_row1_into(
        src: &[f32],
        w: &KQuantWeight,
        j0: usize,
        dst_chunk: &mut [f32],
    ) {
        let blocks_per_row = w.blocks_per_row();
        let data = w.data();
        for (i, j) in (j0..j0 + dst_chunk.len()).enumerate() {
            let mut acc_j = 0.0f32;
            let row_bytes = j * blocks_per_row * Q4_K_BLOCK_BYTES;
            for b in 0..blocks_per_row {
                let block =
                    &data[row_bytes + b * Q4_K_BLOCK_BYTES..row_bytes + (b + 1) * Q4_K_BLOCK_BYTES];
                let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                let min = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
                let scales = &block[4..16];
                let qs = &block[16..144];
                let x_base = b * QK_K;

                for g in 0..4 {
                    // 64 values per group: 32 low-nibble + 32 high-nibble
                    let (sc_low, m_low) = get_scale_min_k4(2 * g, scales);
                    let (sc_high, m_high) = get_scale_min_k4(2 * g + 1, scales);
                    let d1 = d * f32::from(sc_low);
                    let m1 = min * f32::from(m_low);
                    let d2 = d * f32::from(sc_high);
                    let m2 = min * f32::from(m_high);

                    let qs32 = &qs[g * 32..g * 32 + 32];
                    let mask_0f = _mm_set1_epi8(0x0F);

                    for c in 0..4 {
                        let l8 = c * 8;
                        let qs8 = _mm_loadl_epi64(qs32.as_ptr().byte_add(l8) as *const __m128i);
                        let q_low = _mm256_cvtepu8_epi32(_mm_and_si128(qs8, mask_0f));
                        let q_high =
                            _mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(qs8, 4), mask_0f));
                        // value = d*sc*q - min*m, computed as fma(q, d1, -m1)
                        let v_low = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(q_low),
                            _mm256_set1_ps(d1),
                            _mm256_set1_ps(-m1),
                        );
                        let v_high = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(q_high),
                            _mm256_set1_ps(d2),
                            _mm256_set1_ps(-m2),
                        );

                        let acc = _mm256_setzero_ps();
                        let x = src.as_ptr().add(x_base + g * 64);
                        let xv_low = _mm256_loadu_ps(x.add(c * 8));
                        let xv_high = _mm256_loadu_ps(x.add(32 + c * 8));
                        let acc = _mm256_fmadd_ps(xv_low, v_low, acc);
                        let acc = _mm256_fmadd_ps(xv_high, v_high, acc);
                        let mut lanes = [0.0f32; 8];
                        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
                        acc_j += lanes.iter().sum::<f32>();
                    }
                }
            }
            dst_chunk[i] = acc_j;
        }
    }

    /// Column-parallel Q4_K decode matvec (rows = 1): the output dimension
    /// is split across the rayon pool; each task writes a disjoint `dst`
    /// range with the identical per-column accumulation order, so results
    /// are bit-identical to the serial kernel.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn matmul_q4_k_avx2_into_parallel(
        src: &[f32],
        w: &KQuantWeight,
        dst: &mut [f32],
    ) {
        use rayon::prelude::*;
        let chunk = 256usize;
        dst.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(c, dst_chunk)| {
                let j0 = c * chunk;
                // SAFETY: same invariants as the serial body; each task
                // writes only its own disjoint dst_chunk.
                unsafe { matmul_q4_k_avx2_row1_into(src, w, j0, dst_chunk) }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k_matmul::matmul_k_scalar_into;
    use crate::k_matmul::tests::{
        eager_reference, seeded_activations, seeded_q4_blocks, seeded_q6_blocks,
    };

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
    fn q4_k_avx2_matches_eager_oracle_across_shapes() {
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
                seeded_q4_blocks(blocks, 0x81_00 + rows as u64 * 101 + in_features as u64),
                [out_features, in_features],
                KQuantDtype::Q4K,
            );
            let src = seeded_activations(rows * in_features, 0x8C_00 + out_features as u64);
            let expected = eager_reference(&weight, &src, rows);
            let mut dst = vec![0.0f32; rows * out_features];
            unsafe {
                matmul_k_avx2_into(&src, rows, &weight, &mut dst)
                    .expect("avx2 dispatch on a q4 weight");
            }
            assert_gate_d(&dst, &expected);
        }
    }

    #[test]
    fn q4_k_avx2_zero_scale_and_min_contribute_exactly_zero() {
        if !avx2_supported() {
            eprintln!("skipped: AVX2 unavailable");
            return;
        }
        // d = 0 and min = 0: every value is exactly 0.0
        let mut bytes = seeded_q4_blocks(2, 0x8D);
        for block in bytes.chunks_exact_mut(Q4_K_BLOCK_BYTES) {
            block[0..4].fill(0);
        }
        let weight = KQuantWeight::new(bytes, [1, 512], KQuantDtype::Q4K);
        let src = seeded_activations(2 * 512, 0x8E);
        let mut dst = vec![1.0f32; 2]; // nonzero sentinel
        unsafe {
            matmul_k_avx2_into(&src, 2, &weight, &mut dst).unwrap();
        }
        assert_eq!(dst, vec![1.0; 2]);
    }

    #[test]
    fn q4_k_avx2_is_deterministic() {
        if !avx2_supported() {
            eprintln!("skipped: AVX2 unavailable");
            return;
        }
        let weight = KQuantWeight::new(seeded_q4_blocks(4, 0x8F), [2, 512], KQuantDtype::Q4K);
        let src = seeded_activations(2 * 512, 0x90);
        let mut a = vec![0.0f32; 4];
        let mut b = vec![0.0f32; 4];
        unsafe {
            matmul_k_avx2_into(&src, 2, &weight, &mut a).unwrap();
            matmul_k_avx2_into(&src, 2, &weight, &mut b).unwrap();
        }
        assert_eq!(a, b);
    }
}
