//! Prefill (multi-row) cached-attention inner kernels.
//!
//! [`cached_row_head`] computes exactly what
//! [`crate::backend::cached_attention_row_head`] computes for one (row, head),
//! with the same arithmetic in the same order for every output element:
//!
//! * each score is `dot(q, k_j) * scale`, where the dot starts at `-0.0` and
//!   adds the rounded products `q[i] * k_j[i]` in ascending `i`
//!   ([`crate::simd::dot_product_f16`]'s ordered NEON form). Independent keys
//!   are evaluated in separate vector lanes instead of one after another, so
//!   the add chain is no longer latency-bound;
//! * the softmax is the shared [`crate::backend::softmax_range`];
//! * each output element adds `weight_j * v_j[i]` for ascending `j`, skipping
//!   zero weights ([`crate::simd::weighted_add_f16`]), but keeps the running
//!   sums in registers across keys instead of reloading them per key.

use half::f16;

/// One (row, head) of cached attention; returns `false` (having touched
/// nothing) when this CPU or shape has no fast kernel, in which case the
/// caller runs the generic path.
///
/// `k_base` / `v_base` index key/value `min_j`'s row at `stride` elements
/// per position. `scores` must hold at least `max_j + 1` values.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cached_row_head(
    q: &[f32],
    cached_k: &[f16],
    cached_v: &[f16],
    kv_offset: usize,
    stride: usize,
    scale: f32,
    min_j: usize,
    max_j: usize,
    scores: &mut [f32],
    out: &mut [f32],
) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        let head_dim = q.len();
        if !std::arch::is_aarch64_feature_detected!("fp16")
            || !head_dim.is_multiple_of(4)
            || head_dim == 0
            || out.len() != head_dim
        {
            return false;
        }
        assert!(min_j <= max_j && max_j < scores.len());
        let last = kv_offset + max_j * stride + head_dim;
        assert!(last <= cached_k.len() && last <= cached_v.len());
        // SAFETY: fp16 is runtime-checked; the bounds above cover every key
        // and value row read, and `scores`/`out` are sized for the writes.
        unsafe {
            neon::scores(q, cached_k, kv_offset, stride, scale, min_j, max_j, scores);
        }
        crate::backend::softmax_range(scores, min_j, max_j + 1);
        // SAFETY: as above.
        unsafe {
            neon::weighted_values(cached_v, kv_offset, stride, min_j, max_j, scores, out);
        }
        true
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (
            q, cached_k, cached_v, kv_offset, stride, scale, min_j, max_j, scores, out,
        );
        false
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon,fp16")]
    unsafe fn load_f16x4(ptr: *const f16) -> float32x4_t {
        // SAFETY: the caller guarantees `ptr` is readable for four `f16`
        // values and that fp16 is available; `fcvtl` only widens the loaded
        // register and touches no memory.
        unsafe {
            let bits = vld1_u16(ptr.cast());
            let result: float32x4_t;
            std::arch::asm!("fcvtl {result:v}.4s, {bits:v}.4h",
                result = out(vreg) result, bits = in(vreg) bits,
                options(pure, nomem, nostack, preserves_flags));
            result
        }
    }

    /// Dims `i..i+4` of four keys, transposed so vector `d` holds dim `i+d`
    /// of keys 0..4.
    #[inline]
    #[target_feature(enable = "neon,fp16")]
    unsafe fn transposed(k: *const f16, stride: usize, i: usize) -> [float32x4_t; 4] {
        // SAFETY: the caller guarantees four key rows `stride` apart starting
        // at `k`, each readable through element `i + 3`.
        unsafe {
            let r0 = load_f16x4(k.add(i));
            let r1 = load_f16x4(k.add(stride + i));
            let r2 = load_f16x4(k.add(2 * stride + i));
            let r3 = load_f16x4(k.add(3 * stride + i));
            let t0 = vtrn1q_f32(r0, r1);
            let t1 = vtrn2q_f32(r0, r1);
            let t2 = vtrn1q_f32(r2, r3);
            let t3 = vtrn2q_f32(r2, r3);
            [
                vreinterpretq_f32_f64(vtrn1q_f64(
                    vreinterpretq_f64_f32(t0),
                    vreinterpretq_f64_f32(t2),
                )),
                vreinterpretq_f32_f64(vtrn1q_f64(
                    vreinterpretq_f64_f32(t1),
                    vreinterpretq_f64_f32(t3),
                )),
                vreinterpretq_f32_f64(vtrn2q_f64(
                    vreinterpretq_f64_f32(t0),
                    vreinterpretq_f64_f32(t2),
                )),
                vreinterpretq_f32_f64(vtrn2q_f64(
                    vreinterpretq_f64_f32(t1),
                    vreinterpretq_f64_f32(t3),
                )),
            ]
        }
    }

    /// Scores for `4 * B` consecutive keys starting at `k`, lane per key.
    #[inline]
    #[target_feature(enable = "neon,fp16")]
    unsafe fn score_block<const B: usize>(
        q: &[f32],
        k: *const f16,
        stride: usize,
        scale: f32,
        dst: *mut f32,
    ) {
        // SAFETY: the caller guarantees `4 * B` key rows of `q.len()` values
        // (a multiple of four) starting at `k`, and `4 * B` writable scores at
        // `dst`; `i` stays below `q.len()` so every `q` load is in bounds.
        unsafe {
            let mut sums = [vdupq_n_f32(-0.0); B];
            let mut i = 0;
            while i < q.len() {
                let qv = vld1q_f32(q.as_ptr().add(i));
                for (block, sum) in sums.iter_mut().enumerate() {
                    let c = transposed(k.add(4 * block * stride), stride, i);
                    *sum = vaddq_f32(*sum, vmulq_laneq_f32::<0>(c[0], qv));
                    *sum = vaddq_f32(*sum, vmulq_laneq_f32::<1>(c[1], qv));
                    *sum = vaddq_f32(*sum, vmulq_laneq_f32::<2>(c[2], qv));
                    *sum = vaddq_f32(*sum, vmulq_laneq_f32::<3>(c[3], qv));
                }
                i += 4;
            }
            for (block, sum) in sums.into_iter().enumerate() {
                vst1q_f32(dst.add(4 * block), vmulq_n_f32(sum, scale));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon,fp16")]
    pub(super) unsafe fn scores(
        q: &[f32],
        cached_k: &[f16],
        kv_offset: usize,
        stride: usize,
        scale: f32,
        min_j: usize,
        max_j: usize,
        scores: &mut [f32],
    ) {
        // SAFETY: `cached_row_head` asserts `max_j < scores.len()` and that
        // key row `max_j` ends inside `cached_k`; each block below reads keys
        // `j..j + 4 * B` and writes the same score indices, all `<= max_j`.
        unsafe {
            let k = cached_k.as_ptr().add(kv_offset);
            let dst = scores.as_mut_ptr();
            let mut j = min_j;
            while j + 16 <= max_j + 1 {
                score_block::<4>(q, k.add(j * stride), stride, scale, dst.add(j));
                j += 16;
            }
            while j + 4 <= max_j + 1 {
                score_block::<1>(q, k.add(j * stride), stride, scale, dst.add(j));
                j += 4;
            }
            while j <= max_j {
                let key = &cached_k[kv_offset + j * stride..][..q.len()];
                *dst.add(j) = crate::simd::dot_product_f16(q, key) * scale;
                j += 1;
            }
        }
    }

    /// `out[c..c + 4 * N] += Σ_j w_j * v_j[c..]` in ascending `j`.
    #[inline]
    #[target_feature(enable = "neon,fp16")]
    unsafe fn weighted_chunk<const N: usize>(
        v: *const f16,
        stride: usize,
        min_j: usize,
        max_j: usize,
        weights: &[f32],
        out: *mut f32,
    ) {
        // SAFETY: the caller guarantees `4 * N` readable and writable values
        // at `out`, value rows `min_j..=max_j` each readable for `4 * N`
        // values from `v`, and `max_j < weights.len()`.
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); N];
            for (lane, value) in acc.iter_mut().enumerate() {
                *value = vld1q_f32(out.add(4 * lane));
            }
            for j in min_j..=max_j {
                let weight = *weights.get_unchecked(j);
                if weight == 0.0 {
                    continue;
                }
                let row = v.add(j * stride);
                for (lane, value) in acc.iter_mut().enumerate() {
                    let product = vmulq_n_f32(load_f16x4(row.add(4 * lane)), weight);
                    *value = vaddq_f32(*value, product);
                }
            }
            for (lane, value) in acc.into_iter().enumerate() {
                vst1q_f32(out.add(4 * lane), value);
            }
        }
    }

    #[target_feature(enable = "neon,fp16")]
    pub(super) unsafe fn weighted_values(
        cached_v: &[f16],
        kv_offset: usize,
        stride: usize,
        min_j: usize,
        max_j: usize,
        weights: &[f32],
        out: &mut [f32],
    ) {
        // SAFETY: `cached_row_head` checks that `out.len()` is a multiple of
        // four equal to the head dimension, that value row `max_j` ends inside
        // `cached_v`, and that `max_j < weights.len()`; `c` advances in whole
        // chunks that stay within `out.len()`.
        unsafe {
            let v = cached_v.as_ptr().add(kv_offset);
            let dst = out.as_mut_ptr();
            let mut c = 0;
            while c + 32 <= out.len() {
                weighted_chunk::<8>(v.add(c), stride, min_j, max_j, weights, dst.add(c));
                c += 32;
            }
            while c < out.len() {
                weighted_chunk::<1>(v.add(c), stride, min_j, max_j, weights, dst.add(c));
                c += 4;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state % 20_001) as f32 / 10_000.0 - 1.0
            })
            .collect()
    }

    /// The fast kernel reproduces `cached_attention_row_head` bit for bit,
    /// including sliding windows, key counts off the 4/16 tiling, head dims
    /// off the 32-value chunking, zero softmax weights and a padded stride.
    #[test]
    fn fast_row_head_matches_generic_bits() {
        for &(head_dim, stride) in &[(64usize, 64usize), (4, 4), (36, 40), (128, 128), (12, 16)] {
            let positions = 45;
            let n_repeat = 2;
            let kv_heads = 2;
            let n_heads = n_repeat * kv_heads;
            let head_stride = positions * stride;
            let to_f16 = |v: Vec<f32>| v.into_iter().map(f16::from_f32).collect::<Vec<_>>();
            let mut k = to_f16(seeded(kv_heads * head_stride, 3));
            let v = to_f16(seeded(kv_heads * head_stride, 5));
            // A very negative key drives its softmax weight to exactly zero.
            for value in &mut k[7 * stride..7 * stride + head_dim] {
                *value = f16::from_f32(-60_000.0);
            }
            let mut q = seeded(n_heads * head_dim, 7);
            q[..head_dim]
                .iter_mut()
                .for_each(|value| *value = value.abs() * 4.0);
            let embed_dim = n_heads * head_dim;
            let scale = (head_dim as f32).sqrt().recip();
            for head in 0..n_heads {
                for &(min_j, max_j) in &[(0, 0), (0, 3), (0, 16), (0, 44), (5, 38), (9, 30)] {
                    let mut expected_scores = vec![0.0f32; positions];
                    let mut expected = seeded(head_dim, head as u32 + 11);
                    crate::backend::cached_attention_row_head(
                        &q,
                        &k,
                        &v,
                        0,
                        head,
                        embed_dim,
                        head_dim,
                        stride,
                        n_repeat,
                        scale,
                        head_stride,
                        max_j,
                        min_j,
                        &mut expected_scores,
                        &mut expected,
                    );
                    let mut scores = vec![0.0f32; positions];
                    let mut actual = seeded(head_dim, head as u32 + 11);
                    let handled = cached_row_head(
                        &q[head * head_dim..(head + 1) * head_dim],
                        &k,
                        &v,
                        (head / n_repeat) * head_stride,
                        stride,
                        scale,
                        min_j,
                        max_j,
                        &mut scores,
                        &mut actual,
                    );
                    if !handled {
                        continue;
                    }
                    let bits =
                        |values: &[f32]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "{head_dim} {head} {min_j}..={max_j}"
                    );
                    assert_eq!(
                        bits(&scores[min_j..=max_j]),
                        bits(&expected_scores[min_j..=max_j])
                    );
                }
            }
        }
    }
}
