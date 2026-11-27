//! # Safety blanket
//! Every `unsafe fn` in this module requires the caller to have
//! runtime-checked the `#[target_feature]` set on the function, and to
//! pass slices with the length relationships the function's dispatcher
//! asserts (the `SAFETY` comment in each body names that dispatcher).
//! Tests that call a kernel directly build buffers of exactly those
//! sizes behind the same feature check.
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
    // SAFETY: the caller guarantees the CPU features; `rms_norm_into`
    // asserts `weight` and `dst` match `x.len()`, and every 8-lane
    // load/store is guarded by `i + 8 <= n`.
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
    // SAFETY: the caller guarantees the CPU features; `silu_mul_into`
    // asserts all three lengths equal, and every 8-lane load/store is
    // guarded by `i + 8 <= n`.
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
    // SAFETY: the caller guarantees the CPU features;
    // `dequantize_q8_0_row` asserts that blocks `block_start..block_start
    // + blocks_per_row` lie inside `data` and that `dst` holds 32 values
    // per block, which bounds the 34-byte block reads and the four 8-lane
    // stores.
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
    // SAFETY: the caller guarantees the CPU features;
    // `matmul_q8_0_decode_dispatch_chunk` asserts that `x` holds
    // `blocks_per_row` blocks and `data` holds `blocks_per_row` blocks for
    // each of the `out.len()` rows, so every 34-byte block read is in
    // bounds.
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
    // SAFETY: the caller guarantees the CPU features;
    // `matmul_q8_0_decode_dispatch_chunk` asserts that `x` holds
    // `blocks_per_row` blocks and `data` holds `blocks_per_row` blocks for
    // each of the `out.len()` rows, so every 34-byte block read (and the
    // prefetch address) is in bounds.
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
                    let pf_offset = (row_start * blocks_per_row + block_start + 4) * Q8_0_TYPE_SIZE;
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
                        let weights =
                            _mm256_loadu_si256(data.as_ptr().add(byte_off + 2) as *const __m256i);
                        let abs_w = _mm256_abs_epi8(weights);
                        let signed_in = _mm256_sign_epi8(input_quants[bl], weights);
                        let sums = _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs_w, signed_in);
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
                    let w = _mm256_loadu_si256(data.as_ptr().add(byte_off + 2) as *const __m256i);
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
    // SAFETY: the caller guarantees the CPU features; the dispatcher
    // asserts `x` is one encoded row, and `QuantizedWeightVnni` validates
    // that `data` holds one complete record per (tile, block) addressed
    // here, including the padded last tile.
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
                    let signed_input = _mm512_mask_mov_epi8(input, negative_weights, negated_input);
                    let products =
                        _mm512_dpbusd_epi32(_mm512_setzero_si512(), absolute_weights, signed_input);
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
            let result = _mm512_add_ps(_mm512_add_ps(sum04, sum15), _mm512_add_ps(sum26, sum37));

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
    // SAFETY: the caller guarantees the CPU features;
    // `matmul_q8_0_decode_interleaved` asserts `x` is one encoded row and
    // that the row range lies inside the weight, whose validated
    // `quants`/`scales` lengths cover every stripe (128 quant and 8 scale
    // bytes per block) read below.
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
                let s_bits =
                    u64::from_le_bytes(*(s_stripe[s_off..s_off + 8].as_ptr() as *const [u8; 8]));
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
    // SAFETY: the caller guarantees the CPU features;
    // `matmul_q8_0_batch_dispatch_tile` asserts that `x` holds `rows`
    // encoded rows and `data` holds `out_features` rows of
    // `blocks_per_row` blocks, so every 34-byte block read is in bounds.
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
    // SAFETY: the caller guarantees the CPU features;
    // `matmul_q8_0_batch_dispatch_tile` asserts that `x` holds `rows`
    // encoded rows and `data` holds `out_features` rows of
    // `blocks_per_row` blocks, so every 34-byte block read is in bounds.
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
                        inputs[row] =
                            _mm256_loadu_si256(x.as_ptr().add(input_offset + 2) as *const __m256i);
                    }

                    for output_lane in 0..4 {
                        let weight_offset =
                            ((output_start + output_lane) * blocks_per_row + b) * Q8_0_TYPE_SIZE;
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
    // SAFETY: the caller guarantees the CPU features; every 8-lane load is
    // guarded by `i + 8 <= n`.
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
pub(crate) unsafe fn scale_weight_mul_avx2(x: &[f32], scale: f32, weight: &[f32], out: &mut [f32]) {
    // SAFETY: the caller guarantees the CPU features; `scale_weight_mul`
    // asserts all three lengths equal, and every 8-lane load/store is
    // guarded by `i + 8 <= n`.
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
    // SAFETY: the caller guarantees the CPU features; `elemul` asserts all
    // three lengths equal, and every 8-lane load/store is guarded by `i +
    // 8 <= n`.
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

    // 2^k is built by writing k + 127 into the exponent field, which
    // only holds a normal float for k in [-126, 127]. Clamp to the range
    // where that holds (ln(FLT_MIN) up to Cephes' exp_hi): beyond it the
    // field wraps into the sign bit and exp(-88.5) came out as -inf.
    let x = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(-87.336_54_f32)),
        _mm256_set1_ps(88.376_26_f32),
    );
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
    // SAFETY: the caller guarantees the CPU features; `dot_product`
    // asserts equal lengths, and every 8-lane load is guarded by `i + 8 <=
    // n`.
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
    // SAFETY: the caller guarantees the CPU features; `dot_product_f16`
    // asserts equal lengths, and every load of eight `f32` or eight `f16`
    // values is guarded by `i + 8 <= n`.
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
    // SAFETY: the caller guarantees the CPU features; `add_assign` asserts
    // equal lengths, and every 8-lane load/store is guarded by `i + 8 <=
    // n`.
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
    // SAFETY: the caller guarantees the CPU features; `add` asserts all
    // three lengths equal, and every 8-lane load/store is guarded by `i +
    // 8 <= n`.
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
    // SAFETY: the caller guarantees the CPU features; `weighted_add`
    // asserts equal lengths, and every 8-lane load/store is guarded by `i
    // + 8 <= n`.
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
    // SAFETY: the caller guarantees the CPU features; `weighted_add_f16`
    // asserts equal lengths, and every load of eight `f32` or eight `f16`
    // values is guarded by `i + 8 <= n`.
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
/// Caller must ensure the required x86 feature set (`avx2` and `fma`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn rope_split_half_avx2(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    // SAFETY: the caller guarantees the CPU features; `rope_split_half`
    // asserts `x.len() == n_heads * head_dim` and `cos`/`sin` hold
    // `head_dim / 2` values; `d + 8 <= half` keeps both half-row loads
    // inside head `h` and the table loads inside `cos`/`sin`.
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
                let x1s = _mm256_mul_ps(x1, s);
                let x1c = _mm256_mul_ps(x1, c);
                _mm256_storeu_ps(x.as_mut_ptr().add(off + d), _mm256_fmsub_ps(x0, c, x1s));
                _mm256_storeu_ps(
                    x.as_mut_ptr().add(off + d + half),
                    _mm256_fmadd_ps(x0, s, x1c),
                );
                d += 8;
            }
            // Tail
            for d in d..half {
                let i0 = off + d;
                let i1 = off + d + half;
                let x0 = x[i0];
                let x1 = x[i1];
                x[i0] = x0.mul_add(cos[d], -(x1 * sin[d]));
                x[i1] = x0.mul_add(sin[d], x1 * cos[d]);
            }
        }
    }
}

/// AVX2+FMA adjacent-pair RoPE (Llama layout: pairs `(2d, 2d+1)`).
///
/// Works on the interleaved layout directly: each 8-lane vector holds four
/// pairs, `vs` is the same vector with every pair swapped and `cc`/`ss`
/// repeat each cos/sin value twice. Even lanes take
/// `fma(x0, c, -(x1*s))`, odd lanes `fma(x0, s, x1*c)` — exactly the
/// scalar `mul_add` expressions, one rounding for each product and one
/// for each fused op, so the result is bit-identical to the scalar loop.
///
/// # Safety
///
/// Caller must ensure the required x86 feature set (`avx2` and `fma`) is supported at runtime (dispatched via `is_x86_feature_detected!`) before calling this function.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn rope_adjacent_pair_avx2(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    // SAFETY: the caller guarantees the CPU features; `rope_adjacent_pair`
    // asserts `x.len() == n_heads * head_dim` and `cos`/`sin` hold
    // `head_dim / 2` values; `d + 4 <= half` keeps the 8-value load inside
    // head `h` and the 4-value table loads inside `cos`/`sin`.
    unsafe {
        let half = head_dim / 2;
        for h in 0..n_heads {
            let off = h * head_dim;
            let mut d = 0;
            while d + 4 <= half {
                let ptr = x.as_mut_ptr().add(off + 2 * d);
                // [x0_0, x1_0, x0_1, x1_1, ...] and its pair-swapped copy
                let v = _mm256_loadu_ps(ptr);
                let vs = _mm256_permute_ps::<0b1011_0001>(v);
                let c4 = _mm_loadu_ps(cos.as_ptr().add(d));
                let s4 = _mm_loadu_ps(sin.as_ptr().add(d));
                let cc = _mm256_set_m128(_mm_unpackhi_ps(c4, c4), _mm_unpacklo_ps(c4, c4));
                let ss = _mm256_set_m128(_mm_unpackhi_ps(s4, s4), _mm_unpacklo_ps(s4, s4));
                // even lanes: x0*c - x1*s (x1*s rounded, then fused)
                let even = _mm256_fmsub_ps(v, cc, _mm256_mul_ps(vs, ss));
                // odd lanes: x0*s + x1*c (x1*c rounded, then fused)
                let odd = _mm256_fmadd_ps(vs, ss, _mm256_mul_ps(v, cc));
                _mm256_storeu_ps(ptr, _mm256_blend_ps::<0b1010_1010>(even, odd));
                d += 4;
            }
            rope_adjacent_pair_scalar_head(&mut x[off..off + head_dim], d, cos, sin);
        }
    }
}

/// AVX2+FMA fast exp: n = round(x·log2e), r = x − n·ln2 (split hi/lo),
/// AVX2+FMA fast exp core over raw pointers (shared by the into and
/// in-place wrappers): n = round(x·log2e), r = x − n·ln2 (split
/// hi/lo), degree-7 Taylor Horner, 2^n via exponent-bit scaling.
///
/// # Safety
///
/// `src`/`dst` must be valid for `n` f32 reads/writes. Aliasing is
/// permitted only when `src == dst`: each 8-lane block is fully loaded
/// before any lane of that block is stored, and lanes are never read
/// after being written.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn fast_exp_raw(src: *const f32, dst: *mut f32, n: usize) {
    // SAFETY: the caller guarantees the CPU features and `n`
    // readable/writable values at `src`/`dst` (see the contract above);
    // every access is at an index below `n`.
    unsafe {
        // f32-rounded constants (clippy's "approximate" lints are the
        // point here: these must be the f32 values the reference expf
        // range reduction uses, not higher-precision literals)
        #[allow(
            clippy::excessive_precision,
            clippy::approx_constant,
            reason = "f32-rounded constants of the reference expf range reduction"
        )]
        const LOG2E: f32 = 1.442_695;
        #[allow(
            clippy::excessive_precision,
            clippy::approx_constant,
            reason = "f32-rounded constants of the reference expf range reduction"
        )]
        const LN2_HI: f32 = 6.931_471_8e-1; // ln2, high 12 bits
        #[allow(
            clippy::excessive_precision,
            clippy::approx_constant,
            reason = "f32-rounded constants of the reference expf range reduction"
        )]
        const LN2_LO: f32 = 1.908_214_9e-10; // ln2 remainder
                                             // Exact Taylor coefficients 1/k!, k = 7..0. Truncation error on
                                             // |r| <= ln2/2 is r^8/8! < 5e-9 — far below one f32 ulp.
        const C7: f32 = 1.0 / 5040.0;
        const C6: f32 = 1.0 / 720.0;
        const C5: f32 = 1.0 / 120.0;
        const C4: f32 = 1.0 / 24.0;
        const C3: f32 = 1.0 / 6.0;
        const C2: f32 = 0.5;

        let v_log2e = _mm256_set1_ps(LOG2E);
        let v_half = _mm256_set1_ps(0.5);
        let v_ln2_hi = _mm256_set1_ps(LN2_HI);
        let v_ln2_lo = _mm256_set1_ps(LN2_LO);
        let v_c2 = _mm256_set1_ps(C2);
        let v_c3 = _mm256_set1_ps(C3);
        let v_c4 = _mm256_set1_ps(C4);
        let v_c5 = _mm256_set1_ps(C5);
        let v_c6 = _mm256_set1_ps(C6);
        let v_c7 = _mm256_set1_ps(C7);
        let v_one = _mm256_set1_ps(1.0);
        let v_clamp_lo = _mm256_set1_ps(-88.0);

        let n_len = n;
        let mut i = 0;
        while i + 8 <= n_len {
            {
                let xd = _mm256_max_ps(_mm256_loadu_ps(src.add(i)), v_clamp_lo);
                // n = floor(x*log2e + 0.5); r = x - n*ln2 (two-part)
                let nf = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(xd, v_log2e), v_half));
                let r = _mm256_fnmadd_ps(nf, v_ln2_hi, xd);
                let r = _mm256_fnmadd_ps(nf, v_ln2_lo, r);
                // degree-7 Horner: 1 + r(1 + r(1/2 + r(1/6 + ...)))
                let p = _mm256_fmadd_ps(v_c7, r, v_c6);
                let p = _mm256_fmadd_ps(p, r, v_c5);
                let p = _mm256_fmadd_ps(p, r, v_c4);
                let p = _mm256_fmadd_ps(p, r, v_c3);
                let p = _mm256_fmadd_ps(p, r, v_c2);
                let p = _mm256_fmadd_ps(p, r, v_one);
                let p = _mm256_fmadd_ps(p, r, v_one);
                // scale by 2^n through the biased-exponent bits
                let ni = _mm256_cvtps_epi32(nf);
                let bits = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
                let scale = _mm256_castsi256_ps(bits);
                _mm256_storeu_ps(dst.add(i), _mm256_mul_ps(p, scale));
            }
            i += 8;
        }
        while i < n_len {
            dst.add(i)
                .write_unaligned(src.add(i).read().max(-88.0).exp());
            i += 1;
        }
    }
}
