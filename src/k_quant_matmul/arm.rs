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
    unsafe { dot_acc(vdupq_n_s32(0), x, y) }
}

/// Caller must check dotprod and supply a validated weight row and Q8_K input.
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

/// Caller must check dotprod and supply a validated weight row and Q8_K input.
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
