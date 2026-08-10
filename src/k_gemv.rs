//! Bandwidth-competitive batch-1 (`rows == 1`) K-quant GEMV.
//!
//! The v0.3 K-quant kernels (scalar and AVX2) dequantize every 256-value
//! super-block into a scratch/register working set and reduce the vector
//! accumulator after every 8-lane chunk, which keeps them ~8× below the
//! machine's memory bandwidth (measured 4.3 GB/s effective vs ~35 GB/s for
//! the Q8_0 path on the dossier host). This module replaces that structure
//! for batch-1 decode with a per-output-column traversal that:
//!
//! - keeps **exact f32 activations** (no activation quantization — the
//!   frozen eager-vs-compressed logit gates must keep their meaning);
//! - unpacks each block's scale/min metadata **once per block** into
//!   registers (the K4 12-byte reshuffle, same math as
//!   `quant_k::get_scale_min_k4`);
//! - carries vector accumulators across the **whole block** (4× 512-bit or
//!   8× 256-bit independent chains) and performs **one horizontal
//!   reduction per output element**;
//! - applies scale/min with broadcast FMAs directly into the accumulator
//!   (no per-value scratch round-trip, no per-8-value horizontal sums);
//! - parallelizes over coarse static output-column chunks with a
//!   shape-dependent threshold (bit-identical per-column body, so serial
//!   and parallel are bit-identical by construction);
//! - dispatches AVX-512 (512-bit) > AVX2 (256-bit) > portable scalar by
//!   runtime feature detection, mirroring the Q8_0 path's dispatch style.
//!
//! Numerics: same f32 math as the scalar reference, different summation
//! order (per sub-block pairs instead of k-linear); deltas are bounded by
//! the existing Gate-A/B envelopes (validated in tests and `k_parity`).
//! The weight bytes are still read exactly once per matmul; the goal is to
//! make per-value ALU/L1 work cheap enough that DRAM bandwidth is the
//! limit.

use crate::quant_k::{
    KExecution, KQuantDtype, KQuantWeight, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, QK_K,
};

/// Minimum output width for the coarse column-parallel split (measured;
/// small projections pay more in rayon join latency than they save).
const PARALLEL_MIN_OUT: usize = 512;
/// Minimum matvec size (MACs) for the column-parallel split.
///
/// Measured on the dossier host (`benches/k_gemv`): the 2048x2048 q/o
/// projections (4.2M MACs) scale ~4x from 1->4 threads, while the 2048x512
/// k/v projections (1.05M MACs) gain only ~1.5x at 2-4 threads and
/// *regress* at 8 (join + memory contention > work); gate/up/down (16.8M)
/// scale ~4x. 2M MACs is the measured crossover.
const PARALLEL_MIN_MACS: usize = 2_000_000;
/// Coarse static chunk of output columns per rayon task.
const PARALLEL_CHUNK: usize = 256;

// ---------------------------------------------------------------------------
// K4 scale/min unpacking (shared by every ISA path)
// ---------------------------------------------------------------------------

/// Unpack the 12-byte K4 (scale, min) array into the 8 sub-block `d`
/// scales and 8 sub-block `m` mins (6-bit each).
///
/// Same bit reshuffle as llama.cpp's `get_scale_min_k4` (and Ember's
/// `quant_k::get_scale_min_k4`), vectorized into byte arrays so the hot
/// loop can broadcast from registers.
#[inline]
fn unpack_k4_scales(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    const K1: u32 = 0x0303_0303;
    const K2: u32 = 0x0f0f_0f0f;
    const K3: u32 = 0x3f3f_3f3f;
    let s0 = u32::from_le_bytes(scales[0..4].try_into().expect("k4 scales slice"));
    let s1 = u32::from_le_bytes(scales[4..8].try_into().expect("k4 scales slice"));
    let s2 = u32::from_le_bytes(scales[8..12].try_into().expect("k4 scales slice"));
    let u0 = s0 & K3;
    let u1 = (s2 & K2) | (((s0 >> 6) & K1) << 4);
    let u2 = s1 & K3;
    let u3 = ((s2 >> 4) & K2) | (((s1 >> 6) & K1) << 4);
    let b0 = u0.to_le_bytes();
    let b1 = u1.to_le_bytes();
    let b2 = u2.to_le_bytes();
    let b3 = u3.to_le_bytes();
    let mut ds = [0u8; 8];
    let mut ms = [0u8; 8];
    ds[0] = b0[0];
    ds[1] = b0[1];
    ds[2] = b0[2];
    ds[3] = b0[3];
    ds[4] = b1[0];
    ds[5] = b1[1];
    ds[6] = b1[2];
    ds[7] = b1[3];
    ms[0] = b2[0];
    ms[1] = b2[1];
    ms[2] = b2[2];
    ms[3] = b2[3];
    ms[4] = b3[0];
    ms[5] = b3[1];
    ms[6] = b3[2];
    ms[7] = b3[3];
    (ds, ms)
}

// ---------------------------------------------------------------------------
// portable scalar body (also the correctness reference for this module)
// ---------------------------------------------------------------------------

/// Scalar Q4_K per-column dot: `dst[j] += x · dequant(row_j)`.
#[inline]
fn q4k_col_scalar(src: &[f32], data: &[u8], blocks_per_row: usize, j: usize, dst: &mut f32) {
    let row_bytes = j * blocks_per_row * Q4_K_BLOCK_BYTES;
    let mut acc = 0.0f32;
    for b in 0..blocks_per_row {
        let block = &data[row_bytes + b * Q4_K_BLOCK_BYTES..];
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let min = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let (ds, ms) = unpack_k4_scales(&block[4..16]);
        let qs = &block[16..144];
        let xb = b * QK_K;
        for g in 0..4 {
            let d1 = d * f32::from(ds[2 * g]);
            let m1 = min * f32::from(ms[2 * g]);
            let d2 = d * f32::from(ds[2 * g + 1]);
            let m2 = min * f32::from(ms[2 * g + 1]);
            for l in 0..32 {
                let ql = f32::from(qs[g * 32 + l] & 0x0F);
                let qh = f32::from(qs[g * 32 + l] >> 4);
                acc += src[xb + g * 64 + l] * (d1 * ql - m1);
                acc += src[xb + g * 64 + 32 + l] * (d2 * qh - m2);
            }
        }
    }
    *dst += acc;
}

/// Scalar Q6_K per-column dot: `dst[j] += x · dequant(row_j)`.
#[inline]
fn q6k_col_scalar(src: &[f32], data: &[u8], blocks_per_row: usize, j: usize, dst: &mut f32) {
    let row_bytes = j * blocks_per_row * Q6_K_BLOCK_BYTES;
    let mut acc = 0.0f32;
    for b in 0..blocks_per_row {
        let block = &data[row_bytes + b * Q6_K_BLOCK_BYTES..];
        let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
        let scales = &block[192..208];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let xb = b * QK_K;
        for half in 0..2 {
            let q = half * 64;
            let h = half * 32;
            let s = half * 8;
            let y = half * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = f32::from((ql[q + l] & 0x0F) | ((qh[h + l] & 3) << 4)) - 32.0;
                let q2 = f32::from((ql[q + l + 32] & 0x0F) | (((qh[h + l] >> 2) & 3) << 4)) - 32.0;
                let q3 = f32::from((ql[q + l] >> 4) | (((qh[h + l] >> 4) & 3) << 4)) - 32.0;
                let q4 = f32::from((ql[q + l + 32] >> 4) | (((qh[h + l] >> 6) & 3) << 4)) - 32.0;
                let sc = |i: usize| f32::from(i8::from_le_bytes([scales[i]]));
                acc += src[xb + y + l] * (d * sc(s + is) * q1);
                acc += src[xb + y + 32 + l] * (d * sc(s + is + 2) * q2);
                acc += src[xb + y + 64 + l] * (d * sc(s + is + 4) * q3);
                acc += src[xb + y + 96 + l] * (d * sc(s + is + 6) * q4);
            }
        }
    }
    *dst += acc;
}

// ---------------------------------------------------------------------------
// x86 bodies (AVX2 256-bit, AVX-512 512-bit)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use core::arch::x86_64::*;

    /// AVX2 Q4_K per-column dot (8 f32 lanes, 4 independent accumulators).
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)] // SIMD accumulator banking: index is structural
    pub(super) unsafe fn q4k_col_avx2(
        src: &[f32],
        data: &[u8],
        blocks_per_row: usize,
        j: usize,
        dst: &mut f32,
    ) {
        let row_bytes = j * blocks_per_row * Q4_K_BLOCK_BYTES;
        let mask0f = _mm_set1_epi8(0x0F);
        let mut acc = [_mm256_setzero_ps(); 4];
        for b in 0..blocks_per_row {
            let block = &data[row_bytes + b * Q4_K_BLOCK_BYTES..];
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let min = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
            let (ds, ms) = unpack_k4_scales(&block[4..16]);
            let qs = &block[16..144];
            let xb = b * QK_K;
            for g in 0..4 {
                let d1 = d * f32::from(ds[2 * g]);
                let m1 = -min * f32::from(ms[2 * g]);
                let d2 = d * f32::from(ds[2 * g + 1]);
                let m2 = -min * f32::from(ms[2 * g + 1]);
                let bd1 = _mm256_set1_ps(d1);
                let bm1 = _mm256_set1_ps(m1);
                let bd2 = _mm256_set1_ps(d2);
                let bm2 = _mm256_set1_ps(m2);
                let qs32 = qs.as_ptr().add(g * 32);
                let xg = src.as_ptr().add(xb + g * 64);
                for c in 0..4 {
                    // 8 bytes = 16 nibbles (8 low + 8 high)
                    let q8 = _mm_loadl_epi64(qs32.add(c * 8) as *const __m128i);
                    let ql = _mm_and_si128(q8, mask0f);
                    let qh = _mm_and_si128(_mm_srli_epi16(q8, 4), mask0f);
                    let v_low =
                        _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(ql)), bd1, bm1);
                    let v_high =
                        _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qh)), bd2, bm2);
                    acc[c] = _mm256_fmadd_ps(_mm256_loadu_ps(xg.add(c * 8)), v_low, acc[c]);
                    acc[c] = _mm256_fmadd_ps(_mm256_loadu_ps(xg.add(32 + c * 8)), v_high, acc[c]);
                }
            }
        }
        let s = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        // reduce 8 lanes
        let hi = _mm256_extractf128_ps(s, 1);
        let lo = _mm256_castps256_ps128(s);
        let sum = _mm_add_ps(lo, hi);
        let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        let sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
        *dst += _mm_cvtss_f32(sum);
    }

    /// AVX2 Q6_K per-column dot (8 f32 lanes, 4 independent accumulators).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q6k_col_avx2(
        src: &[f32],
        data: &[u8],
        blocks_per_row: usize,
        j: usize,
        dst: &mut f32,
    ) {
        let row_bytes = j * blocks_per_row * Q6_K_BLOCK_BYTES;
        let mask0f = _mm_set1_epi8(0x0F);
        let mask03 = _mm_set1_epi8(0x03);
        let thirty_two = _mm_set1_epi8(32);
        let mut acc = [_mm256_setzero_ps(); 4];
        for b in 0..blocks_per_row {
            let block = &data[row_bytes + b * Q6_K_BLOCK_BYTES..];
            let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
            let scales = &block[192..208];
            let ql = &block[0..128];
            let qh = &block[128..192];
            let xb = b * QK_K;
            for half in 0..2 {
                let q = half * 64;
                let h = half * 32;
                let s = half * 8;
                let y = half * 128;
                let sc = |i: usize| f32::from(i8::from_le_bytes([scales[i]]));
                for c in 0..4 {
                    // 8 values per lane group: q1..q4 from 8 ql bytes + 8 qh bytes
                    let ql_lo = _mm_loadl_epi64(ql.as_ptr().add(q + c * 8) as *const __m128i);
                    let ql_hi = _mm_loadl_epi64(ql.as_ptr().add(q + 32 + c * 8) as *const __m128i);
                    let qh8 = _mm_loadl_epi64(qh.as_ptr().add(h + c * 8) as *const __m128i);

                    let qh_lo = _mm_and_si128(qh8, mask03);
                    let qh_sh2 = _mm_and_si128(_mm_srli_epi16(qh8, 2), mask03);
                    let qh_sh4 = _mm_and_si128(_mm_srli_epi16(qh8, 4), mask03);
                    let qh_sh6 = _mm_and_si128(_mm_srli_epi16(qh8, 6), mask03);
                    let shl4 = |v: __m128i| _mm_and_si128(_mm_slli_epi16(v, 4), _mm_set1_epi8(-16));

                    let q1 = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(ql_lo, mask0f), shl4(qh_lo)),
                        thirty_two,
                    );
                    let q2 = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(ql_hi, mask0f), shl4(qh_sh2)),
                        thirty_two,
                    );
                    let q3 = _mm_sub_epi8(
                        _mm_or_si128(
                            _mm_and_si128(_mm_srli_epi16(ql_lo, 4), mask0f),
                            shl4(qh_sh4),
                        ),
                        thirty_two,
                    );
                    let q4 = _mm_sub_epi8(
                        _mm_or_si128(
                            _mm_and_si128(_mm_srli_epi16(ql_hi, 4), mask0f),
                            shl4(qh_sh6),
                        ),
                        thirty_two,
                    );
                    let c2 = c / 2;
                    let d1 = d * sc(s + c2);
                    let d2 = d * sc(s + c2 + 2);
                    let d3 = d * sc(s + c2 + 4);
                    let d4 = d * sc(s + c2 + 6);
                    let x = src.as_ptr().add(xb + y + c * 8);
                    let v1 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)),
                        _mm256_set1_ps(d1),
                    );
                    let v2 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2)),
                        _mm256_set1_ps(d2),
                    );
                    let v3 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3)),
                        _mm256_set1_ps(d3),
                    );
                    let v4 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q4)),
                        _mm256_set1_ps(d4),
                    );
                    acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(x), v1, acc[0]);
                    acc[1] = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(32)), v2, acc[1]);
                    acc[2] = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(64)), v3, acc[2]);
                    acc[3] = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(96)), v4, acc[3]);
                }
            }
        }
        let s = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let hi = _mm256_extractf128_ps(s, 1);
        let lo = _mm256_castps256_ps128(s);
        let sum = _mm_add_ps(lo, hi);
        let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        let sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
        *dst += _mm_cvtss_f32(sum);
    }

    /// AVX-512 Q4_K per-column dot (16 f32 lanes, 4 independent accumulators).
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn q4k_col_avx512(
        src: &[f32],
        data: &[u8],
        blocks_per_row: usize,
        j: usize,
        dst: &mut f32,
    ) {
        let row_bytes = j * blocks_per_row * Q4_K_BLOCK_BYTES;
        let mask0f = _mm_set1_epi8(0x0F);
        let mut acc = [_mm512_setzero_ps(); 4];
        for b in 0..blocks_per_row {
            let block = &data[row_bytes + b * Q4_K_BLOCK_BYTES..];
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let min = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
            let (ds, ms) = unpack_k4_scales(&block[4..16]);
            let qs = &block[16..144];
            let xb = b * QK_K;
            for g in 0..4 {
                let d1 = d * f32::from(ds[2 * g]);
                let m1 = -min * f32::from(ms[2 * g]);
                let d2 = d * f32::from(ds[2 * g + 1]);
                let m2 = -min * f32::from(ms[2 * g + 1]);
                let bd1 = _mm512_set1_ps(d1);
                let bm1 = _mm512_set1_ps(m1);
                let bd2 = _mm512_set1_ps(d2);
                let bm2 = _mm512_set1_ps(m2);
                let qs32 = qs.as_ptr().add(g * 32);
                let xg = src.as_ptr().add(xb + g * 64);
                for c in 0..2 {
                    // 16 bytes = 32 nibbles (16 low + 16 high)
                    let q16 = _mm_loadu_si128(qs32.add(c * 16) as *const __m128i);
                    let ql = _mm_and_si128(q16, mask0f);
                    let qh = _mm_and_si128(_mm_srli_epi16(q16, 4), mask0f);
                    let v_low =
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ql)), bd1, bm1);
                    let v_high =
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(qh)), bd2, bm2);
                    acc[c] = _mm512_fmadd_ps(_mm512_loadu_ps(xg.add(c * 16)), v_low, acc[c]);
                    acc[2 + c] =
                        _mm512_fmadd_ps(_mm512_loadu_ps(xg.add(32 + c * 16)), v_high, acc[2 + c]);
                }
            }
        }
        let s = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
        *dst += _mm512_reduce_add_ps(s);
    }

    /// AVX-512 Q4_K two-column dot: processes output columns `j` and
    /// `j+1` together so each 16-lane activation chunk is loaded once and
    /// consumed by both columns (halves the activation L1 re-read traffic,
    /// which dominates for large out_features).
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn q4k_col2_avx512(
        src: &[f32],
        data: &[u8],
        blocks_per_row: usize,
        j: usize,
        dst: &mut [f32],
    ) {
        let row_a = j * blocks_per_row * Q4_K_BLOCK_BYTES;
        let row_b = (j + 1) * blocks_per_row * Q4_K_BLOCK_BYTES;
        let mask0f = _mm_set1_epi8(0x0F);
        let mut acc = [[_mm512_setzero_ps(); 4]; 2];
        for b in 0..blocks_per_row {
            let block_a = &data[row_a + b * Q4_K_BLOCK_BYTES..];
            let block_b = &data[row_b + b * Q4_K_BLOCK_BYTES..];
            let d_a = half::f16::from_bits(u16::from_le_bytes([block_a[0], block_a[1]])).to_f32();
            let min_a = half::f16::from_bits(u16::from_le_bytes([block_a[2], block_a[3]])).to_f32();
            let d_b = half::f16::from_bits(u16::from_le_bytes([block_b[0], block_b[1]])).to_f32();
            let min_b = half::f16::from_bits(u16::from_le_bytes([block_b[2], block_b[3]])).to_f32();
            let (ds_a, ms_a) = unpack_k4_scales(&block_a[4..16]);
            let (ds_b, ms_b) = unpack_k4_scales(&block_b[4..16]);
            let qs_a = block_a.as_ptr().add(16);
            let qs_b = block_b.as_ptr().add(16);
            let xb = b * QK_K;
            for g in 0..4 {
                let (d1a, m1a, d2a, m2a) = (
                    d_a * f32::from(ds_a[2 * g]),
                    -min_a * f32::from(ms_a[2 * g]),
                    d_a * f32::from(ds_a[2 * g + 1]),
                    -min_a * f32::from(ms_a[2 * g + 1]),
                );
                let (d1b, m1b, d2b, m2b) = (
                    d_b * f32::from(ds_b[2 * g]),
                    -min_b * f32::from(ms_b[2 * g]),
                    d_b * f32::from(ds_b[2 * g + 1]),
                    -min_b * f32::from(ms_b[2 * g + 1]),
                );
                let xg = src.as_ptr().add(xb + g * 64);
                for c in 0..2 {
                    let q16a = _mm_loadu_si128(qs_a.add(g * 32 + c * 16) as *const __m128i);
                    let q16b = _mm_loadu_si128(qs_b.add(g * 32 + c * 16) as *const __m128i);
                    let xv = _mm512_loadu_ps(xg.add(c * 16));
                    let xv2 = _mm512_loadu_ps(xg.add(32 + c * 16));
                    let (qla, qha) = (
                        _mm_and_si128(q16a, mask0f),
                        _mm_and_si128(_mm_srli_epi16(q16a, 4), mask0f),
                    );
                    let (qlb, qhb) = (
                        _mm_and_si128(q16b, mask0f),
                        _mm_and_si128(_mm_srli_epi16(q16b, 4), mask0f),
                    );
                    let vla = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(qla)),
                        _mm512_set1_ps(d1a),
                        _mm512_set1_ps(m1a),
                    );
                    let vha = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(qha)),
                        _mm512_set1_ps(d2a),
                        _mm512_set1_ps(m2a),
                    );
                    let vlb = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(qlb)),
                        _mm512_set1_ps(d1b),
                        _mm512_set1_ps(m1b),
                    );
                    let vhb = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(qhb)),
                        _mm512_set1_ps(d2b),
                        _mm512_set1_ps(m2b),
                    );
                    acc[0][c] = _mm512_fmadd_ps(xv, vla, acc[0][c]);
                    acc[0][2 + c] = _mm512_fmadd_ps(xv2, vha, acc[0][2 + c]);
                    acc[1][c] = _mm512_fmadd_ps(xv, vlb, acc[1][c]);
                    acc[1][2 + c] = _mm512_fmadd_ps(xv2, vhb, acc[1][2 + c]);
                }
            }
        }
        let s0 = _mm512_add_ps(
            _mm512_add_ps(acc[0][0], acc[0][1]),
            _mm512_add_ps(acc[0][2], acc[0][3]),
        );
        let s1 = _mm512_add_ps(
            _mm512_add_ps(acc[1][0], acc[1][1]),
            _mm512_add_ps(acc[1][2], acc[1][3]),
        );
        dst[0] += _mm512_reduce_add_ps(s0);
        dst[1] += _mm512_reduce_add_ps(s1);
    }

    /// AVX-512 Q4_K four-column dot: loads each 16-lane activation chunk
    /// once and consumes it for four output columns (quarters the activation
    /// L1 re-read traffic vs the single-column body; bit-identical per
    /// column because the per-column FMA sequence is unchanged).
    #[target_feature(enable = "avx512f")]
    #[allow(clippy::needless_range_loop)] // SIMD accumulator banking: index is structural
    pub(super) unsafe fn q4k_col4_avx512(
        src: &[f32],
        data: &[u8],
        blocks_per_row: usize,
        j: usize,
        dst: &mut [f32],
    ) {
        let mask0f = _mm_set1_epi8(0x0F);
        let mut acc = [[_mm512_setzero_ps(); 4]; 4];
        for b in 0..blocks_per_row {
            let xg = src.as_ptr().add(b * QK_K);
            for g in 0..4 {
                let xv = [
                    _mm512_loadu_ps(xg.add(g * 64)),
                    _mm512_loadu_ps(xg.add(g * 64 + 16)),
                    _mm512_loadu_ps(xg.add(g * 64 + 32)),
                    _mm512_loadu_ps(xg.add(g * 64 + 48)),
                ];
                for col in 0..4 {
                    let row = (j + col) * blocks_per_row * Q4_K_BLOCK_BYTES;
                    let block = &data[row + b * Q4_K_BLOCK_BYTES..];
                    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                    let min =
                        half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
                    let (ds, ms) = unpack_k4_scales(&block[4..16]);
                    let d1 = d * f32::from(ds[2 * g]);
                    let m1 = -min * f32::from(ms[2 * g]);
                    let d2 = d * f32::from(ds[2 * g + 1]);
                    let m2 = -min * f32::from(ms[2 * g + 1]);
                    let bd1 = _mm512_set1_ps(d1);
                    let bm1 = _mm512_set1_ps(m1);
                    let bd2 = _mm512_set1_ps(d2);
                    let bm2 = _mm512_set1_ps(m2);
                    let qs32 = block.as_ptr().add(16 + g * 32);
                    for c in 0..2 {
                        let q16 = _mm_loadu_si128(qs32.add(c * 16) as *const __m128i);
                        let ql = _mm_and_si128(q16, mask0f);
                        let qh = _mm_and_si128(_mm_srli_epi16(q16, 4), mask0f);
                        let v_low =
                            _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ql)), bd1, bm1);
                        let v_high =
                            _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(qh)), bd2, bm2);
                        acc[col][c] = _mm512_fmadd_ps(xv[c], v_low, acc[col][c]);
                        acc[col][2 + c] = _mm512_fmadd_ps(xv[2 + c], v_high, acc[col][2 + c]);
                    }
                }
            }
        }
        for col in 0..4 {
            let s = _mm512_add_ps(
                _mm512_add_ps(acc[col][0], acc[col][1]),
                _mm512_add_ps(acc[col][2], acc[col][3]),
            );
            dst[col] += _mm512_reduce_add_ps(s);
        }
    }

    /// AVX-512 Q6_K per-column dot (16 f32 lanes, 4 independent accumulators).
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn q6k_col_avx512(
        src: &[f32],
        data: &[u8],
        blocks_per_row: usize,
        j: usize,
        dst: &mut f32,
    ) {
        let row_bytes = j * blocks_per_row * Q6_K_BLOCK_BYTES;
        let mask0f = _mm_set1_epi8(0x0F);
        let mask03 = _mm_set1_epi8(0x03);
        let thirty_two = _mm_set1_epi8(32);
        let mut acc = [_mm512_setzero_ps(); 4];
        for b in 0..blocks_per_row {
            let block = &data[row_bytes + b * Q6_K_BLOCK_BYTES..];
            let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
            let scales = &block[192..208];
            let ql = &block[0..128];
            let qh = &block[128..192];
            let xb = b * QK_K;
            for half in 0..2 {
                let q = half * 64;
                let h = half * 32;
                let s = half * 8;
                let y = half * 128;
                let sc = |i: usize| f32::from(i8::from_le_bytes([scales[i]]));
                for c16 in 0..2 {
                    // 16 values per lane group: q1..q4 from 16 ql bytes + 16 qh bytes
                    let ql_lo = _mm_loadu_si128(ql.as_ptr().add(q + c16 * 16) as *const __m128i);
                    let ql_hi =
                        _mm_loadu_si128(ql.as_ptr().add(q + 32 + c16 * 16) as *const __m128i);
                    let qh16 = _mm_loadu_si128(qh.as_ptr().add(h + c16 * 16) as *const __m128i);
                    let qh_lo = _mm_and_si128(qh16, mask03);
                    let qh_sh2 = _mm_and_si128(_mm_srli_epi16(qh16, 2), mask03);
                    let qh_sh4 = _mm_and_si128(_mm_srli_epi16(qh16, 4), mask03);
                    let qh_sh6 = _mm_and_si128(_mm_srli_epi16(qh16, 6), mask03);
                    let shl4 = |v: __m128i| _mm_and_si128(_mm_slli_epi16(v, 4), _mm_set1_epi8(-16));
                    let q1 = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(ql_lo, mask0f), shl4(qh_lo)),
                        thirty_two,
                    );
                    let q2 = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(ql_hi, mask0f), shl4(qh_sh2)),
                        thirty_two,
                    );
                    let q3 = _mm_sub_epi8(
                        _mm_or_si128(
                            _mm_and_si128(_mm_srli_epi16(ql_lo, 4), mask0f),
                            shl4(qh_sh4),
                        ),
                        thirty_two,
                    );
                    let q4 = _mm_sub_epi8(
                        _mm_or_si128(
                            _mm_and_si128(_mm_srli_epi16(ql_hi, 4), mask0f),
                            shl4(qh_sh6),
                        ),
                        thirty_two,
                    );
                    let d1 = d * sc(s + c16);
                    let d2 = d * sc(s + c16 + 2);
                    let d3 = d * sc(s + c16 + 4);
                    let d4 = d * sc(s + c16 + 6);
                    let x = src.as_ptr().add(xb + y + c16 * 16);
                    let v1 = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1)),
                        _mm512_set1_ps(d1),
                    );
                    let v2 = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q2)),
                        _mm512_set1_ps(d2),
                    );
                    let v3 = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q3)),
                        _mm512_set1_ps(d3),
                    );
                    let v4 = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q4)),
                        _mm512_set1_ps(d4),
                    );
                    acc[0] = _mm512_fmadd_ps(_mm512_loadu_ps(x), v1, acc[0]);
                    acc[1] = _mm512_fmadd_ps(_mm512_loadu_ps(x.add(32)), v2, acc[1]);
                    acc[2] = _mm512_fmadd_ps(_mm512_loadu_ps(x.add(64)), v3, acc[2]);
                    acc[3] = _mm512_fmadd_ps(_mm512_loadu_ps(x.add(96)), v4, acc[3]);
                }
            }
        }
        let s = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
        *dst += _mm512_reduce_add_ps(s);
    }
    /// AVX-512 Q6_K two-column dot: loads each 16-lane activation chunk
    /// once and consumes it for both output columns (halves the activation
    /// L1 re-read traffic vs the single-column body).
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn q6k_col2_avx512(
        src: &[f32],
        data: &[u8],
        blocks_per_row: usize,
        j: usize,
        dst: &mut [f32],
    ) {
        let row_a = j * blocks_per_row * Q6_K_BLOCK_BYTES;
        let row_b = (j + 1) * blocks_per_row * Q6_K_BLOCK_BYTES;
        let mask0f = _mm_set1_epi8(0x0F);
        let mask03 = _mm_set1_epi8(0x03);
        let thirty_two = _mm_set1_epi8(32);
        let mut acc_a = [_mm512_setzero_ps(); 4];
        let mut acc_b = [_mm512_setzero_ps(); 4];
        for b in 0..blocks_per_row {
            let block_a = &data[row_a + b * Q6_K_BLOCK_BYTES..];
            let block_b = &data[row_b + b * Q6_K_BLOCK_BYTES..];
            let d_a =
                half::f16::from_bits(u16::from_le_bytes([block_a[208], block_a[209]])).to_f32();
            let d_b =
                half::f16::from_bits(u16::from_le_bytes([block_b[208], block_b[209]])).to_f32();
            let scales_a = &block_a[192..208];
            let scales_b = &block_b[192..208];
            let ql_a = &block_a[0..128];
            let qh_a = &block_a[128..192];
            let ql_b = &block_b[0..128];
            let qh_b = &block_b[128..192];
            let xb = b * QK_K;
            for half in 0..2 {
                let q = half * 64;
                let h = half * 32;
                let s = half * 8;
                let y = half * 128;
                let sc_a = |i: usize| f32::from(i8::from_le_bytes([scales_a[i]]));
                let sc_b = |i: usize| f32::from(i8::from_le_bytes([scales_b[i]]));
                for c16 in 0..2 {
                    let ql_a_lo =
                        _mm_loadu_si128(ql_a.as_ptr().add(q + c16 * 16) as *const __m128i);
                    let ql_a_hi =
                        _mm_loadu_si128(ql_a.as_ptr().add(q + 32 + c16 * 16) as *const __m128i);
                    let qh_a16 = _mm_loadu_si128(qh_a.as_ptr().add(h + c16 * 16) as *const __m128i);
                    let ql_b_lo =
                        _mm_loadu_si128(ql_b.as_ptr().add(q + c16 * 16) as *const __m128i);
                    let ql_b_hi =
                        _mm_loadu_si128(ql_b.as_ptr().add(q + 32 + c16 * 16) as *const __m128i);
                    let qh_b16 = _mm_loadu_si128(qh_b.as_ptr().add(h + c16 * 16) as *const __m128i);
                    let qh_a_lo = _mm_and_si128(qh_a16, mask03);
                    let qh_a_sh2 = _mm_and_si128(_mm_srli_epi16(qh_a16, 2), mask03);
                    let qh_a_sh4 = _mm_and_si128(_mm_srli_epi16(qh_a16, 4), mask03);
                    let qh_a_sh6 = _mm_and_si128(_mm_srli_epi16(qh_a16, 6), mask03);
                    let qh_b_lo = _mm_and_si128(qh_b16, mask03);
                    let qh_b_sh2 = _mm_and_si128(_mm_srli_epi16(qh_b16, 2), mask03);
                    let qh_b_sh4 = _mm_and_si128(_mm_srli_epi16(qh_b16, 4), mask03);
                    let qh_b_sh6 = _mm_and_si128(_mm_srli_epi16(qh_b16, 6), mask03);
                    let shl4 = |v: __m128i| _mm_and_si128(_mm_slli_epi16(v, 4), _mm_set1_epi8(-16));
                    let q1a = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(ql_a_lo, mask0f), shl4(qh_a_lo)),
                        thirty_two,
                    );
                    let q2a = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(ql_a_hi, mask0f), shl4(qh_a_sh2)),
                        thirty_two,
                    );
                    let q3a = _mm_sub_epi8(
                        _mm_or_si128(
                            _mm_and_si128(_mm_srli_epi16(ql_a_lo, 4), mask0f),
                            shl4(qh_a_sh4),
                        ),
                        thirty_two,
                    );
                    let q4a = _mm_sub_epi8(
                        _mm_or_si128(
                            _mm_and_si128(_mm_srli_epi16(ql_a_hi, 4), mask0f),
                            shl4(qh_a_sh6),
                        ),
                        thirty_two,
                    );
                    let q1b = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(ql_b_lo, mask0f), shl4(qh_b_lo)),
                        thirty_two,
                    );
                    let q2b = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(ql_b_hi, mask0f), shl4(qh_b_sh2)),
                        thirty_two,
                    );
                    let q3b = _mm_sub_epi8(
                        _mm_or_si128(
                            _mm_and_si128(_mm_srli_epi16(ql_b_lo, 4), mask0f),
                            shl4(qh_b_sh4),
                        ),
                        thirty_two,
                    );
                    let q4b = _mm_sub_epi8(
                        _mm_or_si128(
                            _mm_and_si128(_mm_srli_epi16(ql_b_hi, 4), mask0f),
                            shl4(qh_b_sh6),
                        ),
                        thirty_two,
                    );
                    let d1a = d_a * sc_a(s + c16);
                    let d2a = d_a * sc_a(s + c16 + 2);
                    let d3a = d_a * sc_a(s + c16 + 4);
                    let d4a = d_a * sc_a(s + c16 + 6);
                    let d1b = d_b * sc_b(s + c16);
                    let d2b = d_b * sc_b(s + c16 + 2);
                    let d3b = d_b * sc_b(s + c16 + 4);
                    let d4b = d_b * sc_b(s + c16 + 6);
                    let x = src.as_ptr().add(xb + y + c16 * 16);
                    let xv0 = _mm512_loadu_ps(x);
                    let xv1 = _mm512_loadu_ps(x.add(32));
                    let xv2 = _mm512_loadu_ps(x.add(64));
                    let xv3 = _mm512_loadu_ps(x.add(96));
                    let v1a = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1a)),
                        _mm512_set1_ps(d1a),
                    );
                    let v2a = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q2a)),
                        _mm512_set1_ps(d2a),
                    );
                    let v3a = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q3a)),
                        _mm512_set1_ps(d3a),
                    );
                    let v4a = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q4a)),
                        _mm512_set1_ps(d4a),
                    );
                    let v1b = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1b)),
                        _mm512_set1_ps(d1b),
                    );
                    let v2b = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q2b)),
                        _mm512_set1_ps(d2b),
                    );
                    let v3b = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q3b)),
                        _mm512_set1_ps(d3b),
                    );
                    let v4b = _mm512_mul_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q4b)),
                        _mm512_set1_ps(d4b),
                    );
                    acc_a[0] = _mm512_fmadd_ps(xv0, v1a, acc_a[0]);
                    acc_a[1] = _mm512_fmadd_ps(xv1, v2a, acc_a[1]);
                    acc_a[2] = _mm512_fmadd_ps(xv2, v3a, acc_a[2]);
                    acc_a[3] = _mm512_fmadd_ps(xv3, v4a, acc_a[3]);
                    acc_b[0] = _mm512_fmadd_ps(xv0, v1b, acc_b[0]);
                    acc_b[1] = _mm512_fmadd_ps(xv1, v2b, acc_b[1]);
                    acc_b[2] = _mm512_fmadd_ps(xv2, v3b, acc_b[2]);
                    acc_b[3] = _mm512_fmadd_ps(xv3, v4b, acc_b[3]);
                }
            }
        }
        let s_a = _mm512_add_ps(
            _mm512_add_ps(acc_a[0], acc_a[1]),
            _mm512_add_ps(acc_a[2], acc_a[3]),
        );
        let s_b = _mm512_add_ps(
            _mm512_add_ps(acc_b[0], acc_b[1]),
            _mm512_add_ps(acc_b[2], acc_b[3]),
        );
        dst[0] += _mm512_reduce_add_ps(s_a);
        dst[1] += _mm512_reduce_add_ps(s_b);
    }
}

// ---------------------------------------------------------------------------
// dispatch over a column range
// ---------------------------------------------------------------------------

/// Tuned column-blocking width for the AVX-512 per-column dot bodies.
///
/// CB = number of output columns processed per activation L1 load. The
/// kernels are bit-identical per column regardless of CB (same per-column
/// FMA sequence), so the choice is pure performance tuning. Measured on the
/// reference host (Tiger Lake, 4c/8t): CB=1 is best end-to-end — multi-
/// column bodies interleave 2/4 weight-row streams and defeat the sequential
/// DRAM prefetcher (CB=2 ≈ -12%, CB=4 ≈ -35% on bench-decode Q4_K_M).
/// Overridable with `EMBER_KGEMV_CB=2|4` for other hosts; default 1.
fn gemv_cb() -> u8 {
    static CB: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *CB.get_or_init(|| match std::env::var("EMBER_KGEMV_CB") {
        Ok(v) => v.parse::<u8>().unwrap_or(1).clamp(1, 4),
        Err(_) => 1,
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn gemv_chunk_into_x86(src: &[f32], w: &KQuantWeight, j0: usize, dst_chunk: &mut [f32]) {
    let data = w.data();
    let blocks_per_row = w.blocks_per_row();
    let q4 = matches!(w.dtype(), KQuantDtype::Q4K);
    let use_avx512 = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512dq");
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    let mut i = 0usize;
    let cb = gemv_cb();
    // AVX-512: process CB columns per activation load (Q4_K: CB=4 then
    // CB=2 remainder; Q6_K: CB=2 then single remainder). Q4_K without
    // AVX-512 and all scalar/AVX2 bodies stay single-column.
    if use_avx512 && cb > 1 {
        if q4 && cb >= 4 {
            while i + 3 < dst_chunk.len() {
                x86::q4k_col4_avx512(src, data, blocks_per_row, j0 + i, &mut dst_chunk[i..i + 4]);
                i += 4;
            }
            while i + 1 < dst_chunk.len() {
                x86::q4k_col2_avx512(src, data, blocks_per_row, j0 + i, &mut dst_chunk[i..i + 2]);
                i += 2;
            }
        } else if q4 {
            while i + 1 < dst_chunk.len() {
                x86::q4k_col2_avx512(src, data, blocks_per_row, j0 + i, &mut dst_chunk[i..i + 2]);
                i += 2;
            }
        } else {
            while i + 1 < dst_chunk.len() {
                x86::q6k_col2_avx512(src, data, blocks_per_row, j0 + i, &mut dst_chunk[i..i + 2]);
                i += 2;
            }
        }
    }
    while i < dst_chunk.len() {
        let j = j0 + i;
        if use_avx512 {
            if q4 {
                x86::q4k_col_avx512(src, data, blocks_per_row, j, &mut dst_chunk[i]);
            } else {
                x86::q6k_col_avx512(src, data, blocks_per_row, j, &mut dst_chunk[i]);
            }
        } else if use_avx2 {
            if q4 {
                x86::q4k_col_avx2(src, data, blocks_per_row, j, &mut dst_chunk[i]);
            } else {
                x86::q6k_col_avx2(src, data, blocks_per_row, j, &mut dst_chunk[i]);
            }
        } else if q4 {
            q4k_col_scalar(src, data, blocks_per_row, j, &mut dst_chunk[i]);
        } else {
            q6k_col_scalar(src, data, blocks_per_row, j, &mut dst_chunk[i]);
        }
        i += 1;
    }
}

/// Column-range body (any ISA). Dispatches per call; the per-column dot is
/// the same function in serial and parallel so results are bit-identical.
#[inline]
fn gemv_chunk_into(src: &[f32], w: &KQuantWeight, j0: usize, dst_chunk: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: layout validated by the caller; the x86 bodies only touch
        // their own disjoint dst range and read src/w immutably.
        unsafe { gemv_chunk_into_x86(src, w, j0, dst_chunk) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let data = w.data();
        let blocks_per_row = w.blocks_per_row();
        for (i, j) in (j0..j0 + dst_chunk.len()).enumerate() {
            match w.dtype() {
                KQuantDtype::Q4K => q4k_col_scalar(src, data, blocks_per_row, j, &mut dst_chunk[i]),
                KQuantDtype::Q6K => q6k_col_scalar(src, data, blocks_per_row, j, &mut dst_chunk[i]),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// public entries
// ---------------------------------------------------------------------------

/// Serial batch-1 K-quant GEMV: `dst[j] += x · dequant(row_j)`.
///
/// `src.len() == in_features`, `dst.len() == out_features` (zero-filled or
/// previously accumulated; the kernel accumulates, matching the
/// `matmul_k_into` contract). Only `rows == 1` is supported.
pub fn matmul_k_gemv_serial(src: &[f32], w: &KQuantWeight, dst: &mut [f32]) -> Result<(), String> {
    if src.len() != w.in_features() {
        return Err(format!(
            "k_gemv: src len {} != in_features {}",
            src.len(),
            w.in_features()
        ));
    }
    if dst.len() != w.out_features() {
        return Err(format!(
            "k_gemv: dst len {} != out_features {}",
            dst.len(),
            w.out_features()
        ));
    }
    if !matches!(
        w.execution(),
        KExecution::CompressedScalar | KExecution::CompressedX86
    ) {
        return Err("k_gemv: eager-f32 tensors have no packed payload".to_string());
    }
    gemv_chunk_into(src, w, 0, dst);
    Ok(())
}

/// Batch-1 K-quant GEMV with a coarse column-parallel split when the shape
/// is large enough (measured threshold) and the rayon pool has > 1 thread.
/// The per-column body is identical to the serial entry, so results are
/// bit-identical to [`matmul_k_gemv_serial`].
pub fn matmul_k_gemv_parallel(
    src: &[f32],
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    let macs = w.in_features().saturating_mul(w.out_features());
    if rayon::current_num_threads() <= 1
        || w.out_features() < PARALLEL_MIN_OUT
        || macs < PARALLEL_MIN_MACS
    {
        return matmul_k_gemv_serial(src, w, dst);
    }
    if src.len() != w.in_features() {
        return Err(format!(
            "k_gemv: src len {} != in_features {}",
            src.len(),
            w.in_features()
        ));
    }
    if dst.len() != w.out_features() {
        return Err(format!(
            "k_gemv: dst len {} != out_features {}",
            dst.len(),
            w.out_features()
        ));
    }
    // Recursive `rayon::join` split: static, deterministic, and
    // allocation-free on the caller (unlike `par_chunks_mut`, whose split
    // machinery can allocate a job per parallel matvec — the Gate-E
    // steady-state allocation budget counts caller-thread allocations).
    fn parallel_rec(src: &[f32], w: &KQuantWeight, dst: &mut [f32], start: usize, len: usize) {
        // invariant: `dst` covers columns [start, start + len), indexed
        // relatively (dst[i] is column start + i)
        if len <= PARALLEL_CHUNK {
            gemv_chunk_into(src, w, start, &mut dst[..len]);
        } else {
            let half = len / 2;
            let (lo, hi) = dst.split_at_mut(half);
            rayon::join(
                || parallel_rec(src, w, lo, start, half),
                || parallel_rec(src, w, hi, start + half, len - half),
            );
        }
    }
    parallel_rec(src, w, dst, 0, dst.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k_matmul::tests::{seeded_activations, seeded_q4_blocks, seeded_q6_blocks};
    use crate::quant_k::KQuantWeight;

    #[test]
    fn unpack_k4_scales_matches_get_scale_min_k4() {
        use crate::quant_k::get_scale_min_k4;
        let mut state = 0x1234_5678u64;
        let mut scales = [0u8; 12];
        for b in &mut scales {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (state >> 33) as u8;
        }
        let (ds, ms) = unpack_k4_scales(&scales);
        for j in 0..8 {
            let (d, m) = get_scale_min_k4(j, &scales);
            assert_eq!(d, ds[j], "d_s[{j}]");
            assert_eq!(m, ms[j], "m_s[{j}]");
        }
    }

    #[test]
    fn gemv_matches_scalar_reference_on_seeded_data() {
        let shapes = [
            (2048usize, 2048usize),
            (2048, 512),
            (2048, 8192),
            (8192, 2048),
        ];
        for &(in_features, out_features) in &shapes {
            for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
                let blocks = in_features / 256 * out_features;
                let payload = match dtype {
                    KQuantDtype::Q4K => seeded_q4_blocks(blocks, 0x11_00 + in_features as u64),
                    KQuantDtype::Q6K => seeded_q6_blocks(blocks, 0x21_00 + in_features as u64),
                };
                let src = seeded_activations(in_features, 0x31_00 + out_features as u64);
                let weight = KQuantWeight::new(payload, [out_features, in_features], dtype);
                let mut reference = vec![0.0f32; out_features];
                let mut candidate = vec![0.0f32; out_features];
                // reference: the old scalar row-1 kernel via the existing
                // serial entry (still the v0.3 body for rows > 1; the
                // rows == 1 branch is the new GEMV, so compare against the
                // eager dequant path instead for a true oracle).
                let f32_tensor = weight.dequantize_all();
                for (j, row) in f32_tensor.data().chunks_exact(in_features).enumerate() {
                    let mut acc = 0.0f32;
                    for k in 0..in_features {
                        acc += src[k] * row[k];
                    }
                    reference[j] = acc;
                }
                matmul_k_gemv_serial(&src, &weight, &mut candidate).unwrap();
                // same math, different summation order: bound per element
                let scale = reference
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max)
                    .max(1.0f32);
                let mut max_abs = 0.0f32;
                for j in 0..out_features {
                    max_abs = max_abs.max((reference[j] - candidate[j]).abs());
                }
                assert!(
                    max_abs <= 1e-3 * scale,
                    "{in_features}x{out_features} {dtype:?}: max_abs {max_abs} > 1e-3*scale {scale}"
                );
                // serial and parallel must be bit-identical
                let mut parallel = vec![0.0f32; out_features];
                matmul_k_gemv_parallel(&src, &weight, &mut parallel).unwrap();
                assert_eq!(candidate, parallel, "serial/parallel divergence");
            }
        }
    }

    /// All AVX-512 dispatch bodies (CB=1/2/4, Q4_K and Q6_K) must agree
    /// with the eager-f32 oracle within the same tolerance as the default
    /// path, and serial must stay bit-identical to parallel at every CB.
    #[test]
    fn all_column_blocking_widths_match_oracle() {
        for cb in [1u8, 2, 4] {
            std::env::set_var("EMBER_KGEMV_CB", cb.to_string());
            for &(in_features, out_features) in &[(2048usize, 2048usize), (2048, 512), (8192, 2048)]
            {
                for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
                    let blocks = in_features / 256 * out_features;
                    let payload = match dtype {
                        KQuantDtype::Q4K => {
                            seeded_q4_blocks(blocks, 0x51_00 + cb as u64 * 100 + in_features as u64)
                        }
                        KQuantDtype::Q6K => {
                            seeded_q6_blocks(blocks, 0x61_00 + cb as u64 * 100 + in_features as u64)
                        }
                    };
                    let src = seeded_activations(
                        in_features,
                        0x71_00 + cb as u64 * 100 + out_features as u64,
                    );
                    let weight = KQuantWeight::new(payload, [out_features, in_features], dtype);
                    let f32_tensor = weight.dequantize_all();
                    let mut reference = vec![0.0f32; out_features];
                    for (j, row) in f32_tensor.data().chunks_exact(in_features).enumerate() {
                        let mut acc = 0.0f32;
                        for k in 0..in_features {
                            acc += src[k] * row[k];
                        }
                        reference[j] = acc;
                    }
                    let mut candidate = vec![0.0f32; out_features];
                    matmul_k_gemv_serial(&src, &weight, &mut candidate).unwrap();
                    let scale = reference
                        .iter()
                        .fold(0.0f32, |m, v| m.max(v.abs()))
                        .max(1.0f32);
                    let mut max_abs = 0.0f32;
                    for j in 0..out_features {
                        max_abs = max_abs.max((reference[j] - candidate[j]).abs());
                    }
                    assert!(
                        max_abs <= 1e-3 * scale,
                        "cb={cb} {in_features}x{out_features} {dtype:?}: max_abs {max_abs} > 1e-3*scale {scale}"
                    );
                    let mut parallel = vec![0.0f32; out_features];
                    matmul_k_gemv_parallel(&src, &weight, &mut parallel).unwrap();
                    assert_eq!(candidate, parallel, "serial/parallel divergence at cb={cb}");
                }
            }
        }
        // restore the default so later tests measure the shipped path
        std::env::remove_var("EMBER_KGEMV_CB");
    }
}
