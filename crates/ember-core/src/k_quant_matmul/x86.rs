//! x86_64 AVX2 kernel internals.
//!
//! # Safety blanket
//! Every `unsafe fn` in this module requires the `#[target_feature]` set
//! on the function to be runtime-checked by the caller: dispatch happens
//! via `is_x86_feature_detected!` in `x86_k_supported`/the public entry
//! points, never from safe code. Internal helpers (`f16_to_f32`,
//! `hsum_*`, `quantize_eight`, `scales_for_32`) are only callable from
//! the gated unsafe kernels in this module.
use super::*;
use core::arch::x86_64::*;

/// vpshufb masks for the Q4_K scale broadcast. `k4_shuffle(i)` broadcasts
/// scale word `i` (i = 2g for the low-nibble scale of group g, 2g + 1 for
/// the high-nibble scale) across all 16 i16 lanes of the scale register;
/// both 128-bit lanes of the source hold the same eight scale words.
const K4_SCALE_SHUFFLES: [u8; 256] = [
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
    4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
    6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
    8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
    10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
    12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15,
    14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
];

/// vpshufb masks for the Q6_K scale broadcast. `q6_shuffle(s)` broadcasts
/// scale pair (2s, 2s + 1) across the 16 bytes: lanes 0..7 carry scale
/// 2s, lanes 8..15 carry scale 2s + 1, matching the 32-weight segment
/// layout of the dot product.
const Q6_SCALE_SHUFFLES: [u8; 128] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
    11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14,
    14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
];

#[inline]
unsafe fn k4_shuffle(index: usize) -> __m256i {
    debug_assert!(index < 8);
    // SAFETY: callers pass `index < 8`, so the unaligned 32-byte load
    // stays inside the 256-byte table; the gated kernels that call this
    // guarantee AVX.
    unsafe { _mm256_loadu_si256(K4_SCALE_SHUFFLES.as_ptr().add(32 * index).cast()) }
}

#[inline]
unsafe fn q6_shuffle(index: usize) -> __m128i {
    debug_assert!(index < 8);
    // SAFETY: callers pass `index < 8`, so the unaligned 16-byte load
    // stays inside the 128-byte table; SSE2 is part of the x86_64
    // baseline.
    unsafe { _mm_loadu_si128(Q6_SCALE_SHUFFLES.as_ptr().add(16 * index).cast()) }
}

/// Experimental AVX-512 gate (see `q4_k_dot_q8_k_avx512`): opt-in via
/// `EMBER_K_AVX512=1`, runtime-checked, read once per process.
#[inline]
pub(super) fn k_avx512_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("EMBER_K_AVX512").is_some_and(|v| v == "1")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
    })
}

#[inline]
#[target_feature(enable = "f16c")]
unsafe fn f16_to_f32(bits: u16) -> f32 {
    let packed = _mm_cvtsi32_si128(i32::from(bits));
    _mm_cvtss_f32(_mm_cvtph_ps(packed))
}

#[inline]
#[target_feature(enable = "avx2,ssse3")]
unsafe fn hsum_i32x8(value: __m256i) -> i32 {
    let low = _mm256_castsi256_si128(value);
    let high = _mm256_extracti128_si256::<1>(value);
    let sum = _mm_add_epi32(low, high);
    let sum = _mm_hadd_epi32(sum, sum);
    let sum = _mm_hadd_epi32(sum, sum);
    _mm_cvtsi128_si32(sum)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn quantize_eight(values: *const f32, inverse_scale: __m256, dst: *mut i8) -> __m256i {
    // SAFETY: the caller guarantees eight readable `f32` at `values` and
    // eight writable bytes at `dst` (the 64-bit store), both unaligned.
    unsafe {
        let scaled = _mm256_mul_ps(_mm256_loadu_ps(values), inverse_scale);
        let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
        let clipped = _mm256_min_ps(
            _mm256_set1_ps(127.0),
            _mm256_max_ps(_mm256_set1_ps(-127.0), rounded),
        );
        // Rust's saturating float-to-int cast maps NaN to zero. `cvttps2dq`
        // maps it to i32::MIN, so mask unordered lanes before narrowing.
        let ordered = _mm256_cmp_ps::<_CMP_ORD_Q>(scaled, scaled);
        let quantized =
            _mm256_and_si256(_mm256_cvttps_epi32(clipped), _mm256_castps_si256(ordered));
        let low = _mm256_castsi256_si128(quantized);
        let high = _mm256_extracti128_si256::<1>(quantized);
        let packed_i16 = _mm_packs_epi32(low, high);
        let packed_i8 = _mm_packs_epi16(packed_i16, _mm_setzero_si128());
        _mm_storel_epi64(dst.cast(), packed_i8);
        quantized
    }
}

/// Bit-identical finite-input Q8_K packing for the recorded x86 tier.
#[target_feature(enable = "avx2,ssse3")]
/// # Safety
///
/// Caller must ensure the AVX2+SSSE3 feature set is runtime-checked and
/// `dst` has capacity for `len` Q8_K blocks.
pub(super) unsafe fn quantize_q8_k_into(
    src: &[f32],
    dst: &mut Vec<Q8KBlock>,
) -> Result<(), &'static str> {
    // SAFETY: the caller guarantees the feature set. `values` is a whole
    // `QK_K`-element chunk and `block.qs` holds `QK_K` quants, so each
    // `quantize_eight` call at `offset` or `offset + 8` (`offset <=
    // QK_K - 16`) reads and writes eight in-bounds elements.
    unsafe {
        debug_assert!(src.len().is_multiple_of(QK_K));
        let blocks = src.len() / QK_K;
        dst.resize_with(blocks, Q8KBlock::default);

        for (values, block) in src.chunks_exact(QK_K).zip(dst.iter_mut()) {
            let mut max = 0.0f32;
            let mut amax = 0.0f32;
            for &value in values {
                if !value.is_finite() {
                    return Err("K-quant matmul requires finite activations");
                }
                let abs = value.abs();
                if abs > amax {
                    amax = abs;
                    max = value;
                }
            }
            if amax == 0.0 {
                *block = Q8KBlock::default();
                continue;
            }

            let inverse_scale_scalar = -127.0 / max;
            let inverse_scale = _mm256_set1_ps(inverse_scale_scalar);
            for group in 0..QK_K / 16 {
                let offset = group * 16;
                let first = quantize_eight(
                    values.as_ptr().add(offset),
                    inverse_scale,
                    block.qs.as_mut_ptr().add(offset),
                );
                let second = quantize_eight(
                    values.as_ptr().add(offset + 8),
                    inverse_scale,
                    block.qs.as_mut_ptr().add(offset + 8),
                );
                block.bsums[group] = (hsum_i32x8(first) + hsum_i32x8(second)) as i16;
            }
            block.d = inverse_scale_scalar.recip();
        }
        Ok(())
    }
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn hsum_f32x8(value: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(value, 1);
    let lo = _mm256_castps256_ps128(value);
    let sum = _mm_add_ps(lo, hi);
    let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum = _mm_add_ss(sum, _mm_shuffle_ps::<0x55>(sum, sum));
    _mm_cvtss_f32(sum)
}

#[inline]
unsafe fn hsum_f32x4(value: __m128) -> f32 {
    // SAFETY: register-only SSE operations; SSE is part of the x86_64
    // baseline and no memory is accessed.
    unsafe {
        let sum = _mm_add_ps(value, _mm_movehl_ps(value, value));
        let sum = _mm_add_ss(sum, _mm_shuffle_ps::<0x55>(sum, sum));
        _mm_cvtss_f32(sum)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn scales_for_32(first: i8, second: i8) -> __m256i {
    _mm256_setr_epi16(
        i16::from(first),
        i16::from(first),
        i16::from(first),
        i16::from(first),
        i16::from(first),
        i16::from(first),
        i16::from(first),
        i16::from(first),
        i16::from(second),
        i16::from(second),
        i16::from(second),
        i16::from(second),
        i16::from(second),
        i16::from(second),
        i16::from(second),
        i16::from(second),
    )
}

#[target_feature(enable = "avx2,fma,f16c,ssse3")]
/// # Safety
///
/// Caller must ensure the AVX2+FMA+F16C+SSSE3 feature set is
/// runtime-checked, `data` is a valid Q4_K weight block row for
/// `blocks_per_row` blocks, `column` is in range, and `input` holds
/// exactly `blocks_per_row` Q8_K blocks.
pub(super) unsafe fn q4_k_dot_q8_k(
    data: &[u8],
    blocks_per_row: usize,
    column: usize,
    input: &[Q8KBlock],
) -> f32 {
    // SAFETY: the caller guarantees the feature set and that `data` holds
    // row `column` of `blocks_per_row` Q4_K super-blocks, one per `input`
    // element. Within a 144-byte block the unaligned reads cover the
    // 16-byte header and four 32-byte quant groups ending at byte 144;
    // each activation load reads 32 of a `Q8KBlock`'s 256 quants or its
    // sixteen `bsums`.
    unsafe {
        let row = data
            .as_ptr()
            .add(column * blocks_per_row * Q4_K_BLOCK_BYTES);
        let nibble_mask = _mm256_set1_epi8(0x0f);
        let mut acc = _mm256_setzero_ps();
        let mut min_acc = _mm_setzero_ps();
        for (block_index, activation) in input.iter().enumerate() {
            let block = row.add(block_index * Q4_K_BLOCK_BYTES);
            let d = f16_to_f32(block.cast::<u16>().read_unaligned());
            let dmin = f16_to_f32(block.add(2).cast::<u16>().read_unaligned());
            // Unpack the 12 scale/min bytes into one 16-byte vector
            // (8 scales + 8 mins) with the four-scalar shuffle (same
            // dataflow as ggml's q4_K dot): a handful of scalar ops
            // instead of a per-byte unpack with stack spills, and the
            // scale register then feeds shuffle-based broadcasts.
            let s0 = block.add(4).cast::<u32>().read_unaligned();
            let s1 = block.add(8).cast::<u32>().read_unaligned();
            let s2 = block.add(12).cast::<u32>().read_unaligned();
            let utmp0 = s0 & 0x3f3f_3f3f;
            let utmp1 = (s2 & 0x0f0f_0f0f) | (((s0 >> 6) & 0x0303_0303) << 4);
            let utmp2 = s1 & 0x3f3f_3f3f;
            let utmp3 = ((s2 >> 4) & 0x0f0f_0f0f) | (((s1 >> 6) & 0x0303_0303) << 4);
            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp3 as i32,
                utmp2 as i32,
                utmp1 as i32,
                utmp0 as i32,
            ));
            let sc128 = _mm256_extracti128_si256::<0>(mins_and_scales);
            let scales = _mm256_set_m128i(sc128, sc128);
            let mut integer_sum = _mm256_setzero_si256();
            for group in 0..4 {
                let packed = _mm256_loadu_si256(block.add(16 + group * 32).cast());
                let low = _mm256_and_si256(packed, nibble_mask);
                let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
                let q8_low = _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64).cast());
                let q8_high =
                    _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64 + 32).cast());
                let low_pairs = _mm256_maddubs_epi16(low, q8_low);
                let high_pairs = _mm256_maddubs_epi16(high, q8_high);
                let low_scale = _mm256_shuffle_epi8(scales, k4_shuffle(2 * group));
                let high_scale = _mm256_shuffle_epi8(scales, k4_shuffle(2 * group + 1));
                integer_sum =
                    _mm256_add_epi32(integer_sum, _mm256_madd_epi16(low_scale, low_pairs));
                integer_sum =
                    _mm256_add_epi32(integer_sum, _mm256_madd_epi16(high_scale, high_pairs));
            }
            let scale = _mm256_set1_ps(activation.d * d);
            acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(integer_sum), acc);
            let q8sums = _mm256_loadu_si256(activation.bsums.as_ptr().cast());
            let paired_sums = _mm_hadd_epi16(
                _mm256_extracti128_si256::<0>(q8sums),
                _mm256_extracti128_si256::<1>(q8sums),
            );
            let mins16 = _mm256_extracti128_si256::<1>(mins_and_scales);
            let min_product = _mm_madd_epi16(mins16, paired_sums);
            min_acc = _mm_fmadd_ps(
                _mm_set1_ps(-activation.d * dmin),
                _mm_cvtepi32_ps(min_product),
                min_acc,
            );
        }
        hsum_f32x8(acc) + hsum_f32x4(min_acc)
    }
}

#[target_feature(enable = "avx2,fma,f16c,ssse3")]
/// # Safety
///
/// Caller must ensure the AVX2+FMA+F16C+SSSE3 feature set is
/// runtime-checked, `data` is a valid Q6_K weight block row for
/// `blocks_per_row` blocks, `column` is in range, and `input` holds
/// exactly `blocks_per_row` Q8_K blocks.
pub(super) unsafe fn q6_k_dot_q8_k(
    data: &[u8],
    blocks_per_row: usize,
    column: usize,
    input: &[Q8KBlock],
) -> f32 {
    // SAFETY: the caller guarantees the feature set and that `data` holds
    // row `column` of `blocks_per_row` Q6_K super-blocks, one per `input`
    // element. Within a 210-byte block the unaligned reads cover the low
    // bits (bytes 0..128), high bits (128..192), scales (192..208) and the
    // scale word (208..210); each activation load reads 32 of a
    // `Q8KBlock`'s 256 quants or its sixteen `bsums`.
    unsafe {
        let row = data
            .as_ptr()
            .add(column * blocks_per_row * Q6_K_BLOCK_BYTES);
        let mask3 = _mm256_set1_epi8(3);
        let mask12 = _mm256_set1_epi8(12);
        let mask15 = _mm256_set1_epi8(15);
        let mask48 = _mm256_set1_epi8(48);
        let mask_signed = _mm256_set1_epi8(-64);
        let mut acc = _mm256_setzero_ps();
        for (block_index, activation) in input.iter().enumerate() {
            let block = row.add(block_index * Q6_K_BLOCK_BYTES);
            let d = f16_to_f32(block.add(208).cast::<u16>().read_unaligned());
            // Unpack the 16 scale bytes once per block: they feed the
            // eight segment broadcasts and the bsums offset correction.
            let scale_bytes = _mm_loadu_si128(block.add(192).cast());
            let scale_words = _mm256_cvtepi8_epi16(scale_bytes);
            let sums = _mm256_loadu_si256(activation.bsums.as_ptr().cast());
            let offset = _mm256_slli_epi32(_mm256_madd_epi16(sums, scale_words), 5);
            let mut integer_sum = _mm256_setzero_si256();
            for half in 0..2 {
                let ql = block.add(half * 64);
                let qh = block.add(128 + half * 32);
                let q8 = activation.qs.as_ptr().add(half * 128);
                let low_a = _mm256_loadu_si256(ql.cast());
                let low_b = _mm256_loadu_si256(ql.add(32).cast());
                let high_bits = _mm256_loadu_si256(qh.cast());
                let raw = [
                    _mm256_or_si256(
                        _mm256_and_si256(low_a, mask15),
                        _mm256_slli_epi16(_mm256_and_si256(high_bits, mask3), 4),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(low_b, mask15),
                        _mm256_slli_epi16(_mm256_and_si256(high_bits, mask12), 2),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(low_a, 4), mask15),
                        _mm256_and_si256(high_bits, mask48),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(low_b, 4), mask15),
                        _mm256_srli_epi16(_mm256_and_si256(high_bits, mask_signed), 2),
                    ),
                ];
                for (segment, &raw_values) in raw.iter().enumerate() {
                    let q8_values = _mm256_loadu_si256(q8.add(segment * 32).cast());
                    let pairs = _mm256_maddubs_epi16(raw_values, q8_values);
                    // Segment (half, s) uses scale pair (half * 8 + 2s,
                    // half * 8 + 2s + 1): one pshufb + one sign-extend
                    // replaces a two-scalar broadcast build per segment.
                    let scale_shuffled =
                        _mm_shuffle_epi8(scale_bytes, q6_shuffle(half * 4 + segment));
                    let scale_vector = _mm256_cvtepi8_epi16(scale_shuffled);
                    integer_sum =
                        _mm256_add_epi32(integer_sum, _mm256_madd_epi16(scale_vector, pairs));
                }
            }
            integer_sum = _mm256_sub_epi32(integer_sum, offset);
            let combined_scale = activation.d * d;
            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(combined_scale),
                _mm256_cvtepi32_ps(integer_sum),
                acc,
            );
        }
        hsum_f32x8(acc)
    }
}

/// Experimental AVX-512 Q4_K dot (EVEX-encoded, 32 vector registers).
/// Same integer math as [`q4_k_dot_q8_k`]: bit-identical results, but the
/// extra register file lets the eight scale-shuffle constants stay live.
/// Gated behind `EMBER_K_AVX512=1` until measured on healthy hosts.
#[target_feature(enable = "avx2,fma,f16c,ssse3,avx512f,avx512bw,avx512dq,avx512vl")]
/// # Safety
///
/// Caller must ensure the full feature set is runtime-checked and the
/// data/input layout contract of [`q4_k_dot_q8_k`] holds.
pub(super) unsafe fn q4_k_dot_q8_k_avx512(
    data: &[u8],
    blocks_per_row: usize,
    column: usize,
    input: &[Q8KBlock],
) -> f32 {
    // SAFETY: the caller guarantees the feature set and that `data` holds
    // row `column` of `blocks_per_row` Q4_K super-blocks, one per `input`
    // element. Within a 144-byte block the unaligned reads cover the
    // 16-byte header and four 32-byte quant groups ending at byte 144;
    // each activation load reads 32 of a `Q8KBlock`'s 256 quants or its
    // sixteen `bsums`.
    unsafe {
        let row = data
            .as_ptr()
            .add(column * blocks_per_row * Q4_K_BLOCK_BYTES);
        let nibble_mask = _mm256_set1_epi8(0x0f);
        let mut acc = _mm256_setzero_ps();
        let mut min_acc = _mm_setzero_ps();
        let shuffles = [
            k4_shuffle(0),
            k4_shuffle(1),
            k4_shuffle(2),
            k4_shuffle(3),
            k4_shuffle(4),
            k4_shuffle(5),
            k4_shuffle(6),
            k4_shuffle(7),
        ];
        for (block_index, activation) in input.iter().enumerate() {
            let block = row.add(block_index * Q4_K_BLOCK_BYTES);
            let d = f16_to_f32(block.cast::<u16>().read_unaligned());
            let dmin = f16_to_f32(block.add(2).cast::<u16>().read_unaligned());
            let s0 = block.add(4).cast::<u32>().read_unaligned();
            let s1 = block.add(8).cast::<u32>().read_unaligned();
            let s2 = block.add(12).cast::<u32>().read_unaligned();
            let utmp0 = s0 & 0x3f3f_3f3f;
            let utmp1 = (s2 & 0x0f0f_0f0f) | (((s0 >> 6) & 0x0303_0303) << 4);
            let utmp2 = s1 & 0x3f3f_3f3f;
            let utmp3 = ((s2 >> 4) & 0x0f0f_0f0f) | (((s1 >> 6) & 0x0303_0303) << 4);
            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp3 as i32,
                utmp2 as i32,
                utmp1 as i32,
                utmp0 as i32,
            ));
            let sc128 = _mm256_extracti128_si256::<0>(mins_and_scales);
            let scales = _mm256_set_m128i(sc128, sc128);
            let mut integer_sum = _mm256_setzero_si256();
            for group in 0..4 {
                let packed = _mm256_loadu_si256(block.add(16 + group * 32).cast());
                let low = _mm256_and_si256(packed, nibble_mask);
                let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
                let q8_low = _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64).cast());
                let q8_high =
                    _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64 + 32).cast());
                let low_pairs = _mm256_maddubs_epi16(low, q8_low);
                let high_pairs = _mm256_maddubs_epi16(high, q8_high);
                let low_scale = _mm256_shuffle_epi8(scales, shuffles[2 * group]);
                let high_scale = _mm256_shuffle_epi8(scales, shuffles[2 * group + 1]);
                integer_sum =
                    _mm256_add_epi32(integer_sum, _mm256_madd_epi16(low_scale, low_pairs));
                integer_sum =
                    _mm256_add_epi32(integer_sum, _mm256_madd_epi16(high_scale, high_pairs));
            }
            let scale = _mm256_set1_ps(activation.d * d);
            acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(integer_sum), acc);
            let q8sums = _mm256_loadu_si256(activation.bsums.as_ptr().cast());
            let paired_sums = _mm_hadd_epi16(
                _mm256_extracti128_si256::<0>(q8sums),
                _mm256_extracti128_si256::<1>(q8sums),
            );
            let mins16 = _mm256_extracti128_si256::<1>(mins_and_scales);
            let min_product = _mm_madd_epi16(mins16, paired_sums);
            min_acc = _mm_fmadd_ps(
                _mm_set1_ps(-activation.d * dmin),
                _mm_cvtepi32_ps(min_product),
                min_acc,
            );
        }
        hsum_f32x8(acc) + hsum_f32x4(min_acc)
    }
}

/// Experimental AVX-512 Q6_K dot: EVEX encoding plus `vpternlogq` to fuse
/// the nibble/high-bit assembly (four two-operand and/or ops become one
/// three-input boolean op per segment). Bit-identical to
/// [`q6_k_dot_q8_k`]. Gated behind `EMBER_K_AVX512=1` (see above).
#[target_feature(enable = "avx2,fma,f16c,ssse3,avx512f,avx512bw,avx512dq,avx512vl")]
/// # Safety
///
/// Caller must ensure the full feature set is runtime-checked and the
/// data/input layout contract of [`q6_k_dot_q8_k`] holds.
pub(super) unsafe fn q6_k_dot_q8_k_avx512(
    data: &[u8],
    blocks_per_row: usize,
    column: usize,
    input: &[Q8KBlock],
) -> f32 {
    // SAFETY: the caller guarantees the feature set and that `data` holds
    // row `column` of `blocks_per_row` Q6_K super-blocks, one per `input`
    // element. Within a 210-byte block the unaligned reads cover the low
    // bits (bytes 0..128), high bits (128..192), scales (192..208) and the
    // scale word (208..210); each activation load reads 32 of a
    // `Q8KBlock`'s 256 quants or its sixteen `bsums`.
    unsafe {
        let row = data
            .as_ptr()
            .add(column * blocks_per_row * Q6_K_BLOCK_BYTES);
        let mask3 = _mm256_set1_epi8(3);
        let mask12 = _mm256_set1_epi8(12);
        let mask48 = _mm256_set1_epi8(48);
        let mask_signed = _mm256_set1_epi8(-64);
        // Byte-uniform selector for `vpternlogq`: imm 0xE4 = (a & c) |
        // (b & ~c), i.e. bits 0-3 from `a` and bits 4-7 from `b`.
        let select_low = _mm256_set1_epi8(0x0f);
        let mut acc = _mm256_setzero_ps();
        for (block_index, activation) in input.iter().enumerate() {
            let block = row.add(block_index * Q6_K_BLOCK_BYTES);
            let d = f16_to_f32(block.add(208).cast::<u16>().read_unaligned());
            let scale_bytes = _mm_loadu_si128(block.add(192).cast());
            let scale_words = _mm256_cvtepi8_epi16(scale_bytes);
            let sums = _mm256_loadu_si256(activation.bsums.as_ptr().cast());
            let offset = _mm256_slli_epi32(_mm256_madd_epi16(sums, scale_words), 5);
            let mut integer_sum = _mm256_setzero_si256();
            for half in 0..2 {
                let ql = block.add(half * 64);
                let qh = block.add(128 + half * 32);
                let q8 = activation.qs.as_ptr().add(half * 128);
                let low_a = _mm256_loadu_si256(ql.cast());
                let low_b = _mm256_loadu_si256(ql.add(32).cast());
                let high_bits = _mm256_loadu_si256(qh.cast());
                let high_shifted = [
                    _mm256_slli_epi16(_mm256_and_si256(high_bits, mask3), 4),
                    _mm256_slli_epi16(_mm256_and_si256(high_bits, mask12), 2),
                    high_bits,
                    high_bits,
                ];
                // Every segment is the same select: the 4-bit low quant
                // from `a` in bits 0-3, the 2 high bits (already moved to
                // bits 4-5, zero elsewhere) from `b` in bits 4-7. Selecting
                // bits 0-3 of `a` also discards its upper nibble, which
                // belongs to the next segment.
                let raw = [
                    _mm256_ternarylogic_epi32(low_a, high_shifted[0], select_low, 0xE4),
                    _mm256_ternarylogic_epi32(low_b, high_shifted[1], select_low, 0xE4),
                    _mm256_ternarylogic_epi32(
                        _mm256_srli_epi16(low_a, 4),
                        _mm256_and_si256(high_shifted[2], mask48),
                        select_low,
                        0xE4,
                    ),
                    _mm256_ternarylogic_epi32(
                        _mm256_srli_epi16(low_b, 4),
                        _mm256_srli_epi16(_mm256_and_si256(high_shifted[3], mask_signed), 2),
                        select_low,
                        0xE4,
                    ),
                ];
                for (segment, &raw_values) in raw.iter().enumerate() {
                    let q8_values = _mm256_loadu_si256(q8.add(segment * 32).cast());
                    let pairs = _mm256_maddubs_epi16(raw_values, q8_values);
                    let scale_shuffled =
                        _mm_shuffle_epi8(scale_bytes, q6_shuffle(half * 4 + segment));
                    let scale_vector = _mm256_cvtepi8_epi16(scale_shuffled);
                    integer_sum =
                        _mm256_add_epi32(integer_sum, _mm256_madd_epi16(scale_vector, pairs));
                }
            }
            integer_sum = _mm256_sub_epi32(integer_sum, offset);
            let combined_scale = activation.d * d;
            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(combined_scale),
                _mm256_cvtepi32_ps(integer_sum),
                acc,
            );
        }
        hsum_f32x8(acc)
    }
}

/// Four-row Q4_K × Q8_K tile. Weight headers/quants are loaded and
/// unpacked once, then consumed by four activation rows.
#[target_feature(enable = "avx2,fma,f16c,ssse3")]
#[allow(
    clippy::needless_range_loop,
    reason = "the index is also used in arithmetic or to index several arrays"
)]
/// # Safety
///
/// Caller must ensure the AVX2+FMA+F16C+SSSE3 feature set is
/// runtime-checked and `input` holds exactly four packed Q8_K rows for
/// `blocks_per_row` blocks.
pub(super) unsafe fn q4_k_dot_q8_k_x4(
    data: &[u8],
    blocks_per_row: usize,
    column: usize,
    input: &[Q8KBlock],
) -> [f32; 4] {
    // SAFETY: the caller guarantees the feature set. `block` is a
    // bounds-checked 144-byte slice, so the four 32-byte quant loads from
    // offset 16 end at byte 144; `input` is indexed with bounds checks and
    // each activation load reads 32 of a `Q8KBlock`'s 256 quants, eight
    // of its sixteen `bsums`, or the eight-byte `mins` array.
    unsafe {
        debug_assert_eq!(input.len(), 4 * blocks_per_row);
        let row_start = column * blocks_per_row * Q4_K_BLOCK_BYTES;
        let nibble_mask = _mm256_set1_epi8(0x0f);
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut min_acc = [_mm_setzero_ps(); 4];
        for block_index in 0..blocks_per_row {
            let start = row_start + block_index * Q4_K_BLOCK_BYTES;
            let block = &data[start..start + Q4_K_BLOCK_BYTES];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (scales, mins) = unpack_k4_scales(&block[4..16]);
            let mut integer_sum = [_mm256_setzero_si256(); 4];
            for group in 0..4 {
                let packed = _mm256_loadu_si256(block.as_ptr().add(16 + group * 32).cast());
                let low = _mm256_and_si256(packed, nibble_mask);
                let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
                let low_scale = _mm256_set1_epi16(i16::from(scales[2 * group]));
                let high_scale = _mm256_set1_epi16(i16::from(scales[2 * group + 1]));
                for row in 0..4 {
                    let activation = &input[row * blocks_per_row + block_index];
                    let q8_low = _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64).cast());
                    let q8_high =
                        _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64 + 32).cast());
                    integer_sum[row] = _mm256_add_epi32(
                        integer_sum[row],
                        _mm256_madd_epi16(low_scale, _mm256_maddubs_epi16(low, q8_low)),
                    );
                    integer_sum[row] = _mm256_add_epi32(
                        integer_sum[row],
                        _mm256_madd_epi16(high_scale, _mm256_maddubs_epi16(high, q8_high)),
                    );
                }
            }
            for row in 0..4 {
                let activation = &input[row * blocks_per_row + block_index];
                acc[row] = _mm256_fmadd_ps(
                    _mm256_set1_ps(activation.d * d),
                    _mm256_cvtepi32_ps(integer_sum[row]),
                    acc[row],
                );
                let mins8 = _mm_loadl_epi64(mins.as_ptr().cast());
                let mins16 = _mm_unpacklo_epi8(mins8, _mm_setzero_si128());
                let sums_low = _mm_loadu_si128(activation.bsums.as_ptr().cast());
                let sums_high = _mm_loadu_si128(activation.bsums.as_ptr().add(8).cast());
                let paired_sums = _mm_hadd_epi16(sums_low, sums_high);
                let min_product = _mm_madd_epi16(mins16, paired_sums);
                min_acc[row] = _mm_fmadd_ps(
                    _mm_set1_ps(-activation.d * dmin),
                    _mm_cvtepi32_ps(min_product),
                    min_acc[row],
                );
            }
        }
        [
            hsum_f32x8(acc[0]) + hsum_f32x4(min_acc[0]),
            hsum_f32x8(acc[1]) + hsum_f32x4(min_acc[1]),
            hsum_f32x8(acc[2]) + hsum_f32x4(min_acc[2]),
            hsum_f32x8(acc[3]) + hsum_f32x4(min_acc[3]),
        ]
    }
}

/// Four-row Q4_K × Q8_K tile reading pre-split nibbles (opt-in, prefill-only).
#[target_feature(enable = "avx2,fma,f16c,ssse3")]
#[allow(
    clippy::needless_range_loop,
    reason = "the index is also used in arithmetic or to index several arrays"
)]
/// # Safety
///
/// Caller must ensure the AVX2+FMA+F16C+SSSE3 feature set is
/// runtime-checked, `presplit` was built for this weight, and `input`
/// holds exactly four packed Q8_K rows.
pub(super) unsafe fn q4_k_dot_q8_k_x4_presplit(
    data: &[u8],
    presplit: &[u8],
    blocks_per_row: usize,
    column: usize,
    input: &[Q8KBlock],
) -> [f32; 4] {
    // SAFETY: the caller guarantees the feature set and that `presplit`
    // holds 256 bytes per super-block of this weight, so the 32-byte
    // loads at `qstart + group * 64 (+ 32)` stay inside block
    // `block_index` of row `column`. `block` and `input` are
    // bounds-checked; each activation load reads 32 of a `Q8KBlock`'s 256
    // quants, eight of its sixteen `bsums`, or the eight-byte `mins`
    // array.
    unsafe {
        debug_assert_eq!(input.len(), 4 * blocks_per_row);
        let row_start = column * blocks_per_row * Q4_K_BLOCK_BYTES;
        let presplit_start = column * blocks_per_row * 256;
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut min_acc = [_mm_setzero_ps(); 4];
        for block_index in 0..blocks_per_row {
            let start = row_start + block_index * Q4_K_BLOCK_BYTES;
            let block = &data[start..start + Q4_K_BLOCK_BYTES];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (scales, mins) = unpack_k4_scales(&block[4..16]);
            let qstart = presplit_start + block_index * 256;
            let mut integer_sum = [_mm256_setzero_si256(); 4];
            for group in 0..4 {
                let low = _mm256_loadu_si256(presplit.as_ptr().add(qstart + group * 64).cast());
                let high =
                    _mm256_loadu_si256(presplit.as_ptr().add(qstart + group * 64 + 32).cast());
                let low_scale = _mm256_set1_epi16(i16::from(scales[2 * group]));
                let high_scale = _mm256_set1_epi16(i16::from(scales[2 * group + 1]));
                for row in 0..4 {
                    let activation = &input[row * blocks_per_row + block_index];
                    let q8_low = _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64).cast());
                    let q8_high =
                        _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64 + 32).cast());
                    integer_sum[row] = _mm256_add_epi32(
                        integer_sum[row],
                        _mm256_madd_epi16(low_scale, _mm256_maddubs_epi16(low, q8_low)),
                    );
                    integer_sum[row] = _mm256_add_epi32(
                        integer_sum[row],
                        _mm256_madd_epi16(high_scale, _mm256_maddubs_epi16(high, q8_high)),
                    );
                }
            }
            for row in 0..4 {
                let activation = &input[row * blocks_per_row + block_index];
                acc[row] = _mm256_fmadd_ps(
                    _mm256_set1_ps(activation.d * d),
                    _mm256_cvtepi32_ps(integer_sum[row]),
                    acc[row],
                );
                let mins8 = _mm_loadl_epi64(mins.as_ptr().cast());
                let mins16 = _mm_unpacklo_epi8(mins8, _mm_setzero_si128());
                let sums_low = _mm_loadu_si128(activation.bsums.as_ptr().cast());
                let sums_high = _mm_loadu_si128(activation.bsums.as_ptr().add(8).cast());
                let paired_sums = _mm_hadd_epi16(sums_low, sums_high);
                let min_product = _mm_madd_epi16(mins16, paired_sums);
                min_acc[row] = _mm_fmadd_ps(
                    _mm_set1_ps(-activation.d * dmin),
                    _mm_cvtepi32_ps(min_product),
                    min_acc[row],
                );
            }
        }
        [
            hsum_f32x8(acc[0]) + hsum_f32x4(min_acc[0]),
            hsum_f32x8(acc[1]) + hsum_f32x4(min_acc[1]),
            hsum_f32x8(acc[2]) + hsum_f32x4(min_acc[2]),
            hsum_f32x8(acc[3]) + hsum_f32x4(min_acc[3]),
        ]
    }
}

/// Four-row Q6_K × Q8_K tile; see [`q4_k_dot_q8_k_x4`].
#[target_feature(enable = "avx2,fma,f16c,ssse3")]
#[allow(
    clippy::needless_range_loop,
    reason = "the index is also used in arithmetic or to index several arrays"
)]
/// # Safety
///
/// Caller must ensure the AVX2+FMA+F16C+SSSE3 feature set is
/// runtime-checked and `input` holds exactly four packed Q8_K rows for
/// `blocks_per_row` blocks.
pub(super) unsafe fn q6_k_dot_q8_k_x4(
    data: &[u8],
    blocks_per_row: usize,
    column: usize,
    input: &[Q8KBlock],
) -> [f32; 4] {
    // SAFETY: the caller guarantees the feature set. `block` is a
    // bounds-checked 210-byte slice: the low-bit loads end at byte 128,
    // the high-bit loads at byte 192, and `scales` is the 16-byte slice
    // 192..208. `input` is indexed with bounds checks and each activation
    // load reads 32 of a `Q8KBlock`'s 256 quants or its sixteen `bsums`.
    unsafe {
        debug_assert_eq!(input.len(), 4 * blocks_per_row);
        let row_start = column * blocks_per_row * Q6_K_BLOCK_BYTES;
        let mask3 = _mm256_set1_epi8(3);
        let mask15 = _mm256_set1_epi8(15);
        let mut acc = [_mm256_setzero_ps(); 4];
        for block_index in 0..blocks_per_row {
            let start = row_start + block_index * Q6_K_BLOCK_BYTES;
            let block = &data[start..start + Q6_K_BLOCK_BYTES];
            let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let scales = &block[192..208];
            let mut integer_sum = [_mm256_setzero_si256(); 4];
            for half in 0..2 {
                let ql = block.as_ptr().add(half * 64);
                let qh = block.as_ptr().add(128 + half * 32);
                let low_a = _mm256_loadu_si256(ql.cast());
                let low_b = _mm256_loadu_si256(ql.add(32).cast());
                let high_bits = _mm256_loadu_si256(qh.cast());
                let raw = [
                    _mm256_or_si256(
                        _mm256_and_si256(low_a, mask15),
                        _mm256_slli_epi16(_mm256_and_si256(high_bits, mask3), 4),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(low_b, mask15),
                        _mm256_slli_epi16(_mm256_and_si256(high_bits, _mm256_set1_epi8(12)), 2),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(low_a, 4), mask15),
                        _mm256_and_si256(high_bits, _mm256_set1_epi8(48)),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(low_b, 4), mask15),
                        _mm256_srli_epi16(_mm256_and_si256(high_bits, _mm256_set1_epi8(-64)), 2),
                    ),
                ];
                for segment in 0..4 {
                    let scale_base = half * 8 + segment * 2;
                    let scale_vector = scales_for_32(
                        i8::from_le_bytes([scales[scale_base]]),
                        i8::from_le_bytes([scales[scale_base + 1]]),
                    );
                    for row in 0..4 {
                        let activation = &input[row * blocks_per_row + block_index];
                        let q8 = activation.qs.as_ptr().add(half * 128 + segment * 32);
                        let pairs =
                            _mm256_maddubs_epi16(raw[segment], _mm256_loadu_si256(q8.cast()));
                        integer_sum[row] = _mm256_add_epi32(
                            integer_sum[row],
                            _mm256_madd_epi16(scale_vector, pairs),
                        );
                    }
                }
            }
            for row in 0..4 {
                let activation = &input[row * blocks_per_row + block_index];
                let sums = _mm256_loadu_si256(activation.bsums.as_ptr().cast());
                let scale_bytes = _mm_loadu_si128(scales.as_ptr().cast());
                let scale_words = _mm256_cvtepi8_epi16(scale_bytes);
                let offset = _mm256_slli_epi32(_mm256_madd_epi16(sums, scale_words), 5);
                integer_sum[row] = _mm256_sub_epi32(integer_sum[row], offset);
                let combined_scale = activation.d * d;
                acc[row] = _mm256_fmadd_ps(
                    _mm256_set1_ps(combined_scale),
                    _mm256_cvtepi32_ps(integer_sum[row]),
                    acc[row],
                );
            }
        }
        [
            hsum_f32x8(acc[0]),
            hsum_f32x8(acc[1]),
            hsum_f32x8(acc[2]),
            hsum_f32x8(acc[3]),
        ]
    }
}

/// Four-row Q6_K × Q8_K tile reading pre-expanded quants (opt-in, prefill-only).
#[target_feature(enable = "avx2,fma,f16c,ssse3")]
#[allow(
    clippy::needless_range_loop,
    reason = "the index is also used in arithmetic or to index several arrays"
)]
/// # Safety
///
/// Caller must ensure the AVX2+FMA+F16C+SSSE3 feature set is
/// runtime-checked, `presplit` was built for this weight, and `input`
/// holds exactly four packed Q8_K rows.
pub(super) unsafe fn q6_k_dot_q8_k_x4_presplit(
    data: &[u8],
    presplit: &[u8],
    blocks_per_row: usize,
    column: usize,
    input: &[Q8KBlock],
) -> [f32; 4] {
    // SAFETY: the caller guarantees the feature set and that `presplit`
    // holds 256 bytes per super-block of this weight, so the 32-byte
    // loads at `qstart + half * 128 + segment * 32` stay inside block
    // `block_index` of row `column`. `block` and `input` are
    // bounds-checked, `scales` is the 16-byte slice 192..208, and each
    // activation load reads 32 of a `Q8KBlock`'s 256 quants or its
    // sixteen `bsums`.
    unsafe {
        debug_assert_eq!(input.len(), 4 * blocks_per_row);
        let row_start = column * blocks_per_row * Q6_K_BLOCK_BYTES;
        let presplit_start = column * blocks_per_row * 256;
        let mut acc = [_mm256_setzero_ps(); 4];
        for block_index in 0..blocks_per_row {
            let start = row_start + block_index * Q6_K_BLOCK_BYTES;
            let block = &data[start..start + Q6_K_BLOCK_BYTES];
            let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let scales = &block[192..208];
            let qstart = presplit_start + block_index * 256;
            let mut integer_sum = [_mm256_setzero_si256(); 4];
            for half in 0..2 {
                for segment in 0..4 {
                    let scale_base = half * 8 + segment * 2;
                    let scale_vector = scales_for_32(
                        i8::from_le_bytes([scales[scale_base]]),
                        i8::from_le_bytes([scales[scale_base + 1]]),
                    );
                    let raw = _mm256_loadu_si256(
                        presplit
                            .as_ptr()
                            .add(qstart + half * 128 + segment * 32)
                            .cast(),
                    );
                    for row in 0..4 {
                        let activation = &input[row * blocks_per_row + block_index];
                        let q8 = activation.qs.as_ptr().add(half * 128 + segment * 32);
                        let pairs = _mm256_maddubs_epi16(raw, _mm256_loadu_si256(q8.cast()));
                        integer_sum[row] = _mm256_add_epi32(
                            integer_sum[row],
                            _mm256_madd_epi16(scale_vector, pairs),
                        );
                    }
                }
            }
            for row in 0..4 {
                let activation = &input[row * blocks_per_row + block_index];
                let sums = _mm256_loadu_si256(activation.bsums.as_ptr().cast());
                let scale_bytes = _mm_loadu_si128(scales.as_ptr().cast());
                let scale_words = _mm256_cvtepi8_epi16(scale_bytes);
                let offset = _mm256_slli_epi32(_mm256_madd_epi16(sums, scale_words), 5);
                integer_sum[row] = _mm256_sub_epi32(integer_sum[row], offset);
                let combined_scale = activation.d * d;
                acc[row] = _mm256_fmadd_ps(
                    _mm256_set1_ps(combined_scale),
                    _mm256_cvtepi32_ps(integer_sum[row]),
                    acc[row],
                );
            }
        }
        [
            hsum_f32x8(acc[0]),
            hsum_f32x8(acc[1]),
            hsum_f32x8(acc[2]),
            hsum_f32x8(acc[3]),
        ]
    }
}
