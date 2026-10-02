//! ARM byte-dot kernels. Scale arithmetic and block accumulation match scalar.
//!
//! Integer lanes accumulate a whole super-block before horizontal reduction.
//! Even with -128 activations, Q6_K's absolute block bound is
//! 256 * 32 * 128 * 128 = 134,217,728, so neither lanes nor their sum overflow i32.
use super::*;
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_acc(mut acc: int32x4_t, x: int8x16_t, y: int8x16_t) -> int32x4_t {
    // SAFETY: the caller guarantees dotprod; `sdot` works on the three
    // registers only and touches no memory.
    unsafe {
        std::arch::asm!("sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
            acc = inout(vreg) acc, x = in(vreg) x, y = in(vreg) y,
            options(pure, nomem, nostack, preserves_flags));
    }
    acc
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot(x: int8x16_t, y: int8x16_t) -> int32x4_t {
    // SAFETY: same CPU-feature contract as `dot_acc`; no memory is accessed.
    unsafe { dot_acc(vdupq_n_s32(0), x, y) }
}

#[cfg(test)]
/// Caller must check dotprod and supply a validated weight row and Q8_K input.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn q4_reference<const N: usize>(
    data: &[u8],
    blocks: usize,
    column: usize,
    input: &[Q8KBlock],
) -> [f32; N] {
    let row = &data[column * blocks * Q4_K_BLOCK_BYTES..][..blocks * Q4_K_BLOCK_BYTES];
    assert_eq!(N * blocks, input.len());
    let mut sum = [0.0f32; N];
    // SAFETY: the caller guarantees dotprod. Each `block` is a whole Q4_K
    // super-block, so the 16-byte weight loads at `16 + g * 32 + j` end at or
    // before byte 144; each activation load at `g * 64 + 32 + j` ends at or
    // before quant 256 of a bounds-checked `Q8KBlock`.
    unsafe {
        for (b, block) in row.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let dm = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
            let (scales, mins) = unpack_k4_scales(&block[4..16]);
            let mut total = [vdupq_n_s32(0); N];
            let mut correction = [0i32; N];
            for g in 0..4 {
                let mut lo = [vdupq_n_s32(0); N];
                let mut hi = [vdupq_n_s32(0); N];
                for j in [0, 16] {
                    let q = vld1q_u8(block.as_ptr().add(16 + g * 32 + j));
                    let low = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(15)));
                    let high = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q));
                    for lane in 0..N {
                        let a = &input[lane * blocks + b];
                        lo[lane] = dot_acc(lo[lane], low, vld1q_s8(a.qs.as_ptr().add(g * 64 + j)));
                        hi[lane] =
                            dot_acc(hi[lane], high, vld1q_s8(a.qs.as_ptr().add(g * 64 + 32 + j)));
                    }
                }
                for lane in 0..N {
                    let a = &input[lane * blocks + b];
                    total[lane] = vmlaq_n_s32(total[lane], lo[lane], i32::from(scales[2 * g]));
                    total[lane] = vmlaq_n_s32(total[lane], hi[lane], i32::from(scales[2 * g + 1]));
                    correction[lane] += i32::from(mins[2 * g])
                        * (i32::from(a.bsums[4 * g]) + i32::from(a.bsums[4 * g + 1]));
                    correction[lane] += i32::from(mins[2 * g + 1])
                        * (i32::from(a.bsums[4 * g + 2]) + i32::from(a.bsums[4 * g + 3]));
                }
            }
            for lane in 0..N {
                sum[lane] += input[lane * blocks + b].d
                    * (d * vaddvq_s32(total[lane]) as f32 - dm * correction[lane] as f32);
            }
        }
    }
    sum
}

#[cfg(test)]
/// Caller must check dotprod and supply a validated weight row and Q8_K input.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn q6_reference<const N: usize>(
    data: &[u8],
    blocks: usize,
    column: usize,
    input: &[Q8KBlock],
) -> [f32; N] {
    let row = &data[column * blocks * Q6_K_BLOCK_BYTES..][..blocks * Q6_K_BLOCK_BYTES];
    assert_eq!(N * blocks, input.len());
    let mut sum = [0.0f32; N];
    // SAFETY: the caller guarantees dotprod. Each `block` is a whole Q6_K
    // super-block, so the low-bit loads end at or before byte 128 and the
    // high-bit loads at or before byte 192; each activation load at
    // `half * 128 + segment * 32 + j` ends at or before quant 256 of a
    // bounds-checked `Q8KBlock`.
    unsafe {
        for (b, block) in row.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
            let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
            let mut total = [vdupq_n_s32(0); N];
            for half in 0..2 {
                for j in [0, 16] {
                    let l0 = vld1q_u8(block.as_ptr().add(half * 64 + j));
                    let l1 = vld1q_u8(block.as_ptr().add(half * 64 + 32 + j));
                    let h = vld1q_u8(block.as_ptr().add(128 + half * 32 + j));
                    let mask = vdupq_n_u8(15);
                    let high_mask = vdupq_n_u8(48);
                    let quants = [
                        vorrq_u8(vandq_u8(l0, mask), vandq_u8(vshlq_n_u8::<4>(h), high_mask)),
                        vorrq_u8(vandq_u8(l1, mask), vandq_u8(vshlq_n_u8::<2>(h), high_mask)),
                        vorrq_u8(vshrq_n_u8::<4>(l0), vandq_u8(h, high_mask)),
                        vorrq_u8(vshrq_n_u8::<4>(l1), vandq_u8(vshrq_n_u8::<2>(h), high_mask)),
                    ];
                    for (segment, q) in quants.into_iter().enumerate() {
                        let signed = vsubq_s8(vreinterpretq_s8_u8(q), vdupq_n_s8(32));
                        let scale = block[192 + half * 8 + segment * 2 + j / 16] as i8 as i32;
                        for lane in 0..N {
                            let a = &input[lane * blocks + b];
                            total[lane] = vmlaq_n_s32(
                                total[lane],
                                dot(
                                    signed,
                                    vld1q_s8(a.qs.as_ptr().add(half * 128 + segment * 32 + j)),
                                ),
                                scale,
                            );
                        }
                    }
                }
            }
            for lane in 0..N {
                sum[lane] += input[lane * blocks + b].d * d * vaddvq_s32(total[lane]) as f32;
            }
        }
    }
    sum
}

/// `acc + x * v[lane]`. Callers pass lanes that loop unrolling makes constant.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn mla_lane(acc: int32x4_t, x: int32x4_t, v: int32x4_t, lane: usize) -> int32x4_t {
    match lane & 3 {
        0 => vmlaq_laneq_s32::<0>(acc, x, v),
        1 => vmlaq_laneq_s32::<1>(acc, x, v),
        2 => vmlaq_laneq_s32::<2>(acc, x, v),
        _ => vmlaq_laneq_s32::<3>(acc, x, v),
    }
}

/// Q4_K dots of weight column `column` against `N` Q8_K activation rows.
///
/// Bit-identical to [`q4_reference`]: the per-block integer total and min
/// correction are the same exact `i32` values (only the order of integer
/// additions differs: scales are applied from vector lanes and the min
/// correction is a widening vector multiply of the 32-quant pair sums), and
/// the float step is unchanged.
///
/// # Safety
/// Requires NEON + dotprod; `data` must hold a validated Q4_K weight with at
/// least `column + 1` rows of `blocks` super-blocks.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn q4<const N: usize>(
    data: &[u8],
    blocks: usize,
    column: usize,
    input: &[Q8KBlock],
) -> [f32; N] {
    let row = &data[column * blocks * Q4_K_BLOCK_BYTES..][..blocks * Q4_K_BLOCK_BYTES];
    assert_eq!(N * blocks, input.len());
    let mut sum = [0.0f32; N];
    let nibble = vdupq_n_u8(15);
    for (b, block) in row.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dm = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let (scales, mins) = unpack_k4_scales(&block[4..16]);
        // SAFETY: the caller guarantees dotprod. `scales`/`mins` are 8-byte
        // locals; each `block` is a whole Q4_K super-block, so the 16-byte
        // weight loads at `16 + g * 32 + j` end at or before byte 144; each
        // activation load at `g * 64 + 32 + j` ends at or before quant 256
        // and the two `bsums` loads cover its 16 sums exactly.
        unsafe {
            let scales16 = vmovl_u8(vld1_u8(scales.as_ptr()));
            let sc = [
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales16))),
                vreinterpretq_s32_u32(vmovl_high_u16(scales16)),
            ];
            let mins16 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mins.as_ptr())));
            let mut total = [vdupq_n_s32(0); N];
            for g in 0..4 {
                let mut lo = [vdupq_n_s32(0); N];
                let mut hi = [vdupq_n_s32(0); N];
                for j in [0, 16] {
                    let q = vld1q_u8(block.as_ptr().add(16 + g * 32 + j));
                    let low = vreinterpretq_s8_u8(vandq_u8(q, nibble));
                    let high = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q));
                    for lane in 0..N {
                        let a = &input[lane * blocks + b];
                        lo[lane] = dot_acc(lo[lane], low, vld1q_s8(a.qs.as_ptr().add(g * 64 + j)));
                        hi[lane] =
                            dot_acc(hi[lane], high, vld1q_s8(a.qs.as_ptr().add(g * 64 + 32 + j)));
                    }
                }
                for lane in 0..N {
                    total[lane] = mla_lane(total[lane], lo[lane], sc[g / 2], (2 * g) % 4);
                    total[lane] = mla_lane(total[lane], hi[lane], sc[g / 2], (2 * g + 1) % 4);
                }
            }
            for lane in 0..N {
                let a = &input[lane * blocks + b];
                // `pairs[s] = bsums[2s] + bsums[2s + 1]` (|value| <= 4096).
                let pairs = vpaddq_s16(
                    vld1q_s16(a.bsums.as_ptr()),
                    vld1q_s16(a.bsums.as_ptr().add(8)),
                );
                let correction = vaddvq_s32(vmlal_high_s16(
                    vmull_s16(vget_low_s16(mins16), vget_low_s16(pairs)),
                    mins16,
                    pairs,
                ));
                sum[lane] += a.d * (d * vaddvq_s32(total[lane]) as f32 - dm * correction as f32);
            }
        }
    }
    sum
}

/// Q6_K dots of weight column `column` against `N` Q8_K activation rows.
///
/// Bit-identical to [`q6_reference`]. Two exact integer rewrites:
/// the six-bit quants are assembled with shift-insert instructions (same
/// bytes), and instead of subtracting 32 from every quant the kernel dots
/// the unsigned quants and subtracts `32 * sum(scale[s] * bsums[s])` (the
/// sixteen-quant groups of `bsums` are exactly the scale groups). Both give
/// the same `i32` block total (two's-complement lane arithmetic is exact
/// modulo 2^32 and the true total fits, see the module bound), and the
/// float step is unchanged.
///
/// # Safety
/// Requires NEON + dotprod; `data` must hold a validated Q6_K weight with at
/// least `column + 1` rows of `blocks` super-blocks.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn q6<const N: usize>(
    data: &[u8],
    blocks: usize,
    column: usize,
    input: &[Q8KBlock],
) -> [f32; N] {
    let row = &data[column * blocks * Q6_K_BLOCK_BYTES..][..blocks * Q6_K_BLOCK_BYTES];
    assert_eq!(N * blocks, input.len());
    let mut sum = [0.0f32; N];
    let six = vdupq_n_u8(63);
    for (b, block) in row.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
        let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
        // SAFETY: the caller guarantees dotprod. Each `block` is a whole Q6_K
        // super-block: low-bit loads end at or before byte 128, high-bit
        // loads at or before byte 192 and the scale load at byte 208; each
        // activation load at `half * 128 + segment * 32 + j` ends at or
        // before quant 256 and the two `bsums` loads cover its 16 sums.
        unsafe {
            let scales8 = vld1q_s8(block.as_ptr().add(192).cast::<i8>());
            let scales16 = [vmovl_s8(vget_low_s8(scales8)), vmovl_high_s8(scales8)];
            let sc = [
                vmovl_s16(vget_low_s16(scales16[0])),
                vmovl_high_s16(scales16[0]),
                vmovl_s16(vget_low_s16(scales16[1])),
                vmovl_high_s16(scales16[1]),
            ];
            let mut total = [vdupq_n_s32(0); N];
            for half in 0..2 {
                for j in [0, 16] {
                    let l0 = vld1q_u8(block.as_ptr().add(half * 64 + j));
                    let l1 = vld1q_u8(block.as_ptr().add(half * 64 + 32 + j));
                    let h = vld1q_u8(block.as_ptr().add(128 + half * 32 + j));
                    let h2 = vshrq_n_u8::<2>(h);
                    // Bits 0-1, 2-3, 4-5 and 6-7 of `h` become bits 4-5 of
                    // the four segments' quants.
                    let quants = [
                        vandq_u8(vsliq_n_u8::<4>(l0, h), six),
                        vandq_u8(vsliq_n_u8::<4>(l1, h2), six),
                        vandq_u8(vsriq_n_u8::<4>(h, l0), six),
                        vandq_u8(vsriq_n_u8::<4>(h2, l1), six),
                    ];
                    for (segment, q) in quants.into_iter().enumerate() {
                        let q = vreinterpretq_s8_u8(q);
                        let s = half * 8 + segment * 2 + j / 16;
                        for lane in 0..N {
                            let a = &input[lane * blocks + b];
                            total[lane] = mla_lane(
                                total[lane],
                                dot(
                                    q,
                                    vld1q_s8(a.qs.as_ptr().add(half * 128 + segment * 32 + j)),
                                ),
                                sc[s / 4],
                                s % 4,
                            );
                        }
                    }
                }
            }
            for lane in 0..N {
                let a = &input[lane * blocks + b];
                let bsums = [
                    vld1q_s16(a.bsums.as_ptr()),
                    vld1q_s16(a.bsums.as_ptr().add(8)),
                ];
                let mut offset = vmull_s16(vget_low_s16(scales16[0]), vget_low_s16(bsums[0]));
                offset = vmlal_high_s16(offset, scales16[0], bsums[0]);
                offset = vmlal_s16(offset, vget_low_s16(scales16[1]), vget_low_s16(bsums[1]));
                offset = vmlal_high_s16(offset, scales16[1], bsums[1]);
                let block_total = vsubq_s32(total[lane], vshlq_n_s32::<5>(offset));
                sum[lane] += a.d * d * vaddvq_s32(block_total) as f32;
            }
        }
    }
    sum
}
