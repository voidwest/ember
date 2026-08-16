//! Canonical K-quant matrix multiplication: Q4_K/Q6_K weights × Q8_K activations.
//!
//! K-family weights are designed to be dotted with a transient Q8_K row.  We
//! quantize each activation row once, then reuse that packed row for every
//! output column.  This is the same dataflow used by ggml: integer dot products
//! in the hot loop, with the exact-f32 dequantize-and-dot code retained only as
//! a reference in `k_matmul`.

use crate::quant_k::{
    KExecution, KQuantDtype, KQuantWeight, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, QK_K,
};
use rayon::join;
use std::cell::RefCell;

const PARALLEL_LEAF_OUTPUTS: usize = 256;
const PARALLEL_MIN_MACS: usize = 2_000_000;

#[cfg(test)]
mod route_probe {
    use super::KQuantWeight;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, Mutex};

    pub(super) static TARGET_WEIGHT: AtomicUsize = AtomicUsize::new(0);
    pub(super) static X86_PACKS: AtomicUsize = AtomicUsize::new(0);
    pub(super) static X86_DOTS: AtomicUsize = AtomicUsize::new(0);
    pub(super) static PARALLEL_BODIES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static PARALLEL_LEAVES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static PARALLEL_WORKERS: AtomicU64 = AtomicU64::new(0);
    static FIRST_TWO_LEAVES: Mutex<Option<Arc<Barrier>>> = Mutex::new(None);
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    pub(super) fn lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().expect("route test lock")
    }

    pub(super) fn reset(weight: &KQuantWeight) {
        TARGET_WEIGHT.store(weight as *const KQuantWeight as usize, Ordering::Relaxed);
        X86_PACKS.store(0, Ordering::Relaxed);
        X86_DOTS.store(0, Ordering::Relaxed);
        PARALLEL_BODIES.store(0, Ordering::Relaxed);
        PARALLEL_LEAVES.store(0, Ordering::Relaxed);
        PARALLEL_WORKERS.store(0, Ordering::Relaxed);
        *FIRST_TWO_LEAVES.lock().expect("route barrier lock") = Some(Arc::new(Barrier::new(2)));
    }

    pub(super) fn finish() {
        *FIRST_TWO_LEAVES.lock().expect("route barrier lock") = None;
        TARGET_WEIGHT.store(0, Ordering::Relaxed);
    }

    pub(super) fn is_target(weight: &KQuantWeight) -> bool {
        TARGET_WEIGHT.load(Ordering::Relaxed) == weight as *const KQuantWeight as usize
    }

    pub(super) fn record_worker(weight: &KQuantWeight) {
        if !is_target(weight) {
            return;
        }
        let leaf = PARALLEL_LEAVES.fetch_add(1, Ordering::Relaxed);
        let barrier = FIRST_TWO_LEAVES.lock().expect("route barrier lock").clone();
        if leaf < 2 {
            barrier.expect("route probe barrier installed").wait();
        }
        if let Some(index) = rayon::current_thread_index() {
            if index < u64::BITS as usize {
                PARALLEL_WORKERS.fetch_or(1u64 << index, Ordering::Relaxed);
            }
        }
    }
}

/// Bytes in one transient ggml `block_q8_K` activation block.
pub const Q8_K_BLOCK_BYTES: usize = core::mem::size_of::<Q8KBlock>();

/// ggml-compatible transient activation block (`block_q8_K`).
///
/// This is deliberately typed rather than encoded as bytes: Q8_K is never a
/// persisted model tensor in Ember.  It exists only for one matmul call.
#[repr(C)]
#[derive(Clone)]
struct Q8KBlock {
    d: f32,
    qs: [i8; QK_K],
    bsums: [i16; QK_K / 16],
}

impl Default for Q8KBlock {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        }
    }
}

thread_local! {
    /// One transient Q8_K buffer per invoking OS thread. Capacity grows to the
    /// largest packed activation seen and is retained after the call; the Vec
    /// is moved out while Rayon runs so a nested invocation gets independent
    /// storage rather than overlapping a `RefCell` borrow. This cache is not
    /// part of the execution-plan arena.
    static Q8_K_INPUT: RefCell<Vec<Q8KBlock>> = const { RefCell::new(Vec::new()) };
}

/// Linear destination plus an owned interval of its strided columns.
///
/// The only way to create siblings is [`DstColumns::split`], which consumes the
/// parent interval and returns two disjoint intervals. This makes the invariant
/// supporting `Send` part of the type instead of relying on parallel callers to
/// keep separate `first_column` arguments in sync with freely copied pointers.
struct DstColumns {
    base: *mut f32,
    rows: usize,
    stride: usize,
    first: usize,
    count: usize,
}

// SAFETY: a value exclusively owns `first..first + count` for every row. It is
// not Clone/Copy; `split` is the only duplication operation and makes disjoint
// child intervals. The Rayon join completes before the original `&mut [f32]`
// may be observed again.
unsafe impl Send for DstColumns {}

impl DstColumns {
    fn new(dst: &mut [f32], rows: usize, stride: usize) -> Self {
        debug_assert_eq!(dst.len(), rows * stride);
        Self {
            base: dst.as_mut_ptr(),
            rows,
            stride,
            first: 0,
            count: stride,
        }
    }

    fn split(self) -> (Self, Self) {
        debug_assert!(self.count > 1);
        let low_count = self.count / 2;
        let high = Self {
            base: self.base,
            rows: self.rows,
            stride: self.stride,
            first: self.first + low_count,
            count: self.count - low_count,
        };
        let low = Self {
            count: low_count,
            ..self
        };
        (low, high)
    }

    /// Add to one element inside this task's owned column interval.
    unsafe fn add_assign(&mut self, row: usize, column: usize, value: f32) {
        debug_assert!(row < self.rows);
        debug_assert!((self.first..self.first + self.count).contains(&column));
        // SAFETY: construction validates `rows * stride`; the assertions above
        // express the bounds and each live task owns a disjoint column interval.
        unsafe {
            *self.base.add(row * self.stride + column) += value;
        }
    }
}

/// Runtime capability used both by dispatch and the explicit `--k-strategy
/// x86` loader policy.
#[inline]
pub fn x86_k_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("ssse3")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// llama.cpp-compatible Q8_K row quantization.
fn quantize_q8_k_into_scalar(src: &[f32], dst: &mut Vec<Q8KBlock>) -> Result<(), &'static str> {
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

        // q8_K uses the signed value with the largest magnitude to choose the
        // scale. `round_ties_even` matches ggml's `nearest_int` helper.
        let inverse_scale = -127.0 / max;
        for (quant, &value) in block.qs.iter_mut().zip(values) {
            let rounded = (inverse_scale * value).round_ties_even();
            *quant = rounded.clamp(-127.0, 127.0) as i8;
        }
        for (sum, group) in block.bsums.iter_mut().zip(block.qs.chunks_exact(16)) {
            *sum = group.iter().map(|&value| i16::from(value)).sum();
        }
        block.d = inverse_scale.recip();
    }
    Ok(())
}

/// Unpack the K4 12-byte scale/min header into eight 6-bit scales and mins.
#[inline]
fn unpack_k4_scales(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    debug_assert!(scales.len() >= 12);
    const K1: u32 = 0x0303_0303;
    const K2: u32 = 0x0f0f_0f0f;
    const K3: u32 = 0x3f3f_3f3f;
    let s0 = u32::from_le_bytes(scales[0..4].try_into().expect("K4 scale header"));
    let s1 = u32::from_le_bytes(scales[4..8].try_into().expect("K4 scale header"));
    let s2 = u32::from_le_bytes(scales[8..12].try_into().expect("K4 scale header"));
    let d0 = (s0 & K3).to_le_bytes();
    let d1 = ((s2 & K2) | (((s0 >> 6) & K1) << 4)).to_le_bytes();
    let m0 = (s1 & K3).to_le_bytes();
    let m1 = (((s2 >> 4) & K2) | (((s1 >> 6) & K1) << 4)).to_le_bytes();
    (
        [d0[0], d0[1], d0[2], d0[3], d1[0], d1[1], d1[2], d1[3]],
        [m0[0], m0[1], m0[2], m0[3], m1[0], m1[1], m1[2], m1[3]],
    )
}

#[inline]
fn q4_k_dot_q8_k_scalar(
    data: &[u8],
    blocks_per_row: usize,
    column: usize,
    input: &[Q8KBlock],
) -> f32 {
    let row_start = column * blocks_per_row * Q4_K_BLOCK_BYTES;
    let mut sum = 0.0f32;
    for (block_index, activation) in input.iter().enumerate() {
        let start = row_start + block_index * Q4_K_BLOCK_BYTES;
        let block = &data[start..start + Q4_K_BLOCK_BYTES];
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let (scales, mins) = unpack_k4_scales(&block[4..16]);
        let quants = &block[16..144];
        let mut dot = 0i32;
        let mut min_dot = 0i32;
        for group in 0..4 {
            let scale_low = i32::from(scales[2 * group]);
            let scale_high = i32::from(scales[2 * group + 1]);
            for index in 0..32 {
                let packed = quants[group * 32 + index];
                dot += scale_low
                    * i32::from(packed & 0x0f)
                    * i32::from(activation.qs[group * 64 + index]);
                dot += scale_high
                    * i32::from(packed >> 4)
                    * i32::from(activation.qs[group * 64 + 32 + index]);
            }
            min_dot += i32::from(mins[2 * group])
                * (i32::from(activation.bsums[4 * group])
                    + i32::from(activation.bsums[4 * group + 1]));
            min_dot += i32::from(mins[2 * group + 1])
                * (i32::from(activation.bsums[4 * group + 2])
                    + i32::from(activation.bsums[4 * group + 3]));
        }
        sum += activation.d * (d * dot as f32 - dmin * min_dot as f32);
    }
    sum
}

#[inline]
fn q6_k_dot_q8_k_scalar(
    data: &[u8],
    blocks_per_row: usize,
    column: usize,
    input: &[Q8KBlock],
) -> f32 {
    let row_start = column * blocks_per_row * Q6_K_BLOCK_BYTES;
    let mut sum = 0.0f32;
    for (block_index, activation) in input.iter().enumerate() {
        let start = row_start + block_index * Q6_K_BLOCK_BYTES;
        let block = &data[start..start + Q6_K_BLOCK_BYTES];
        let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
        let ql = &block[..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let mut dot = 0i32;
        for half in 0..2 {
            let q = half * 64;
            let h = half * 32;
            let scale = half * 8;
            let y = half * 128;
            for index in 0..32 {
                let values = [
                    (ql[q + index] & 0x0f) | ((qh[h + index] & 3) << 4),
                    (ql[q + index + 32] & 0x0f) | (((qh[h + index] >> 2) & 3) << 4),
                    (ql[q + index] >> 4) | (((qh[h + index] >> 4) & 3) << 4),
                    (ql[q + index + 32] >> 4) | (((qh[h + index] >> 6) & 3) << 4),
                ];
                for (segment, &quant) in values.iter().enumerate() {
                    let scale_index = scale + 2 * segment + index / 16;
                    let scale_value = i32::from(i8::from_le_bytes([scales[scale_index]]));
                    dot += scale_value
                        * (i32::from(quant) - 32)
                        * i32::from(activation.qs[y + segment * 32 + index]);
                }
            }
        }
        sum += activation.d * d * dot as f32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
mod x86 {
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
        unsafe {
            let scaled = _mm256_mul_ps(_mm256_loadu_ps(values), inverse_scale);
            let rounded =
                _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
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
        unsafe {
            let row_start = column * blocks_per_row * Q4_K_BLOCK_BYTES;
            let nibble_mask = _mm256_set1_epi8(0x0f);
            let mut acc = _mm256_setzero_ps();
            let mut min_acc = _mm_setzero_ps();
            for (block_index, activation) in input.iter().enumerate() {
                let start = row_start + block_index * Q4_K_BLOCK_BYTES;
                let block = &data[start..start + Q4_K_BLOCK_BYTES];
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let (scales, mins) = unpack_k4_scales(&block[4..16]);
                let mut integer_sum = _mm256_setzero_si256();
                for group in 0..4 {
                    let packed = _mm256_loadu_si256(block.as_ptr().add(16 + group * 32).cast());
                    let low = _mm256_and_si256(packed, nibble_mask);
                    let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
                    let q8_low = _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64).cast());
                    let q8_high =
                        _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64 + 32).cast());
                    let low_pairs = _mm256_maddubs_epi16(low, q8_low);
                    let high_pairs = _mm256_maddubs_epi16(high, q8_high);
                    let low_scale = _mm256_set1_epi16(i16::from(scales[2 * group]));
                    let high_scale = _mm256_set1_epi16(i16::from(scales[2 * group + 1]));
                    integer_sum =
                        _mm256_add_epi32(integer_sum, _mm256_madd_epi16(low_scale, low_pairs));
                    integer_sum =
                        _mm256_add_epi32(integer_sum, _mm256_madd_epi16(high_scale, high_pairs));
                }
                let scale = _mm256_set1_ps(activation.d * d);
                acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(integer_sum), acc);
                let mins8 = _mm_loadl_epi64(mins.as_ptr().cast());
                let mins16 = _mm_unpacklo_epi8(mins8, _mm_setzero_si128());
                let sums_low = _mm_loadu_si128(activation.bsums.as_ptr().cast());
                let sums_high = _mm_loadu_si128(activation.bsums.as_ptr().add(8).cast());
                let paired_sums = _mm_hadd_epi16(sums_low, sums_high);
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
        unsafe {
            let row_start = column * blocks_per_row * Q6_K_BLOCK_BYTES;
            let mask3 = _mm256_set1_epi8(3);
            let mask15 = _mm256_set1_epi8(15);
            let mut acc = _mm256_setzero_ps();
            for (block_index, activation) in input.iter().enumerate() {
                let start = row_start + block_index * Q6_K_BLOCK_BYTES;
                let block = &data[start..start + Q6_K_BLOCK_BYTES];
                let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
                let scales = &block[192..208];
                let mut integer_sum = _mm256_setzero_si256();
                for half in 0..2 {
                    let ql = block.as_ptr().add(half * 64);
                    let qh = block.as_ptr().add(128 + half * 32);
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
                            _mm256_slli_epi16(_mm256_and_si256(high_bits, _mm256_set1_epi8(12)), 2),
                        ),
                        _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16(low_a, 4), mask15),
                            _mm256_and_si256(high_bits, _mm256_set1_epi8(48)),
                        ),
                        _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16(low_b, 4), mask15),
                            _mm256_srli_epi16(
                                _mm256_and_si256(high_bits, _mm256_set1_epi8(-64)),
                                2,
                            ),
                        ),
                    ];
                    for (segment, &raw_values) in raw.iter().enumerate() {
                        let q8_values = _mm256_loadu_si256(q8.add(segment * 32).cast());
                        let pairs = _mm256_maddubs_epi16(raw_values, q8_values);
                        let scale_base = half * 8 + segment * 2;
                        let scale_vector = scales_for_32(
                            i8::from_le_bytes([scales[scale_base]]),
                            i8::from_le_bytes([scales[scale_base + 1]]),
                        );
                        integer_sum =
                            _mm256_add_epi32(integer_sum, _mm256_madd_epi16(scale_vector, pairs));
                    }
                }
                let sums = _mm256_loadu_si256(activation.bsums.as_ptr().cast());
                let scale_bytes = _mm_loadu_si128(scales.as_ptr().cast());
                let scale_words = _mm256_cvtepi8_epi16(scale_bytes);
                let offset = _mm256_slli_epi32(_mm256_madd_epi16(sums, scale_words), 5);
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
    #[allow(clippy::needless_range_loop)]
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
                        let q8_low =
                            _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64).cast());
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
    #[allow(clippy::needless_range_loop)]
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
                        let q8_low =
                            _mm256_loadu_si256(activation.qs.as_ptr().add(group * 64).cast());
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
    #[allow(clippy::needless_range_loop)]
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
                            _mm256_srli_epi16(
                                _mm256_and_si256(high_bits, _mm256_set1_epi8(-64)),
                                2,
                            ),
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
    #[allow(clippy::needless_range_loop)]
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
}

#[inline]
fn dot_column(w: &KQuantWeight, column: usize, input: &[Q8KBlock]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if matches!(w.execution(), KExecution::CompressedX86) {
        #[cfg(test)]
        if route_probe::is_target(w) {
            route_probe::X86_DOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: `validate` rejects a recorded x86 strategy when AVX2/FMA/F16C/SSSE3 is
        // unavailable. Weight layout is validated at construction.
        return unsafe {
            match w.dtype() {
                KQuantDtype::Q4K => x86::q4_k_dot_q8_k(w.data(), w.blocks_per_row(), column, input),
                KQuantDtype::Q6K => x86::q6_k_dot_q8_k(w.data(), w.blocks_per_row(), column, input),
            }
        };
    }
    match w.dtype() {
        KQuantDtype::Q4K => q4_k_dot_q8_k_scalar(w.data(), w.blocks_per_row(), column, input),
        KQuantDtype::Q6K => q6_k_dot_q8_k_scalar(w.data(), w.blocks_per_row(), column, input),
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
fn dot_four_rows(w: &KQuantWeight, column: usize, input: &[Q8KBlock]) -> Option<[f32; 4]> {
    if matches!(w.execution(), KExecution::CompressedX86) {
        #[cfg(test)]
        if route_probe::is_target(w) {
            route_probe::X86_DOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: the recorded strategy is validated before execution, and
        // `input` contains exactly four packed rows.
        return Some(unsafe {
            match w.dtype() {
                KQuantDtype::Q4K => match w.presplit() {
                    Some(presplit) => x86::q4_k_dot_q8_k_x4_presplit(
                        w.data(),
                        presplit,
                        w.blocks_per_row(),
                        column,
                        input,
                    ),
                    None => x86::q4_k_dot_q8_k_x4(w.data(), w.blocks_per_row(), column, input),
                },
                KQuantDtype::Q6K => match w.presplit() {
                    Some(presplit) => x86::q6_k_dot_q8_k_x4_presplit(
                        w.data(),
                        presplit,
                        w.blocks_per_row(),
                        column,
                        input,
                    ),
                    None => x86::q6_k_dot_q8_k_x4(w.data(), w.blocks_per_row(), column, input),
                },
            }
        });
    }
    None
}

#[cfg(not(target_arch = "x86_64"))]
fn dot_four_rows(_w: &KQuantWeight, _column: usize, _input: &[Q8KBlock]) -> Option<[f32; 4]> {
    None
}

fn validate(src: &[f32], rows: usize, w: &KQuantWeight, dst: &[f32]) -> Result<(), String> {
    if rows == 0 {
        return Err("matmul_k_q8: rows must be nonzero".to_string());
    }
    let expected_src = rows
        .checked_mul(w.in_features())
        .ok_or_else(|| "matmul_k_q8: input shape product overflow".to_string())?;
    if src.len() != expected_src {
        return Err(format!(
            "matmul_k_q8: src len {} != rows {rows} * in_features {}",
            src.len(),
            w.in_features()
        ));
    }
    let expected_dst = rows
        .checked_mul(w.out_features())
        .ok_or_else(|| "matmul_k_q8: output shape product overflow".to_string())?;
    if dst.len() != expected_dst {
        return Err(format!(
            "matmul_k_q8: dst len {} != rows {rows} * out_features {}",
            dst.len(),
            w.out_features()
        ));
    }
    match w.execution() {
        KExecution::EagerF32 => {
            return Err("matmul_k_q8: eager-f32 tensors have no packed payload".to_string());
        }
        KExecution::CompressedX86 if !x86_k_supported() => {
            return Err(
                "matmul_k_q8: compressed-x86 was recorded but AVX2/FMA/F16C/SSSE3 is unavailable"
                    .to_string(),
            );
        }
        KExecution::CompressedScalar | KExecution::CompressedX86 => {}
    }
    Ok(())
}

fn serial_body(input: &[Q8KBlock], rows: usize, w: &KQuantWeight, dst: &mut [f32]) {
    let blocks_per_row = w.blocks_per_row();
    let out_features = w.out_features();
    // Row-tile-major traversal: a four-row activation tile is loaded once and
    // reused across every output column, keeping it L1-resident. For multi-token
    // prefill the activation is the operand that would otherwise be re-read once
    // per column (the dominant traffic), so this order streams the weight once
    // per row tile instead of re-streaming the activation per column.
    let mut row = 0;
    while row + 4 <= rows {
        let packed_rows = &input[row * blocks_per_row..(row + 4) * blocks_per_row];
        for column in 0..out_features {
            if let Some(values) = dot_four_rows(w, column, packed_rows) {
                for (lane, value) in values.into_iter().enumerate() {
                    dst[(row + lane) * out_features + column] += value;
                }
            } else {
                for lane in 0..4 {
                    let packed_row =
                        &input[(row + lane) * blocks_per_row..(row + lane + 1) * blocks_per_row];
                    dst[(row + lane) * out_features + column] += dot_column(w, column, packed_row);
                }
            }
        }
        row += 4;
    }
    while row < rows {
        let packed_row = &input[row * blocks_per_row..(row + 1) * blocks_per_row];
        for column in 0..out_features {
            dst[row * out_features + column] += dot_column(w, column, packed_row);
        }
        row += 1;
    }
}

fn parallel_body(input: &[Q8KBlock], rows: usize, w: &KQuantWeight, dst: &mut [f32]) {
    #[cfg(test)]
    if route_probe::is_target(w) {
        route_probe::PARALLEL_BODIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn recurse(input: &[Q8KBlock], rows: usize, w: &KQuantWeight, mut dst: DstColumns) {
        if dst.count <= PARALLEL_LEAF_OUTPUTS {
            #[cfg(test)]
            route_probe::record_worker(w);
            let blocks_per_row = w.blocks_per_row();
            // Row-tile-major: keep the four-row activation tile L1-resident
            // across this leaf's column interval (see `serial_body`).
            let mut row = 0;
            while row + 4 <= rows {
                let packed_rows = &input[row * blocks_per_row..(row + 4) * blocks_per_row];
                for column in dst.first..dst.first + dst.count {
                    if let Some(values) = dot_four_rows(w, column, packed_rows) {
                        for (lane, value) in values.into_iter().enumerate() {
                            // SAFETY: this leaf owns `column`; sibling tasks own
                            // disjoint column intervals created by `split`.
                            unsafe {
                                dst.add_assign(row + lane, column, value);
                            }
                        }
                    } else {
                        for lane in 0..4 {
                            let packed_row = &input
                                [(row + lane) * blocks_per_row..(row + lane + 1) * blocks_per_row];
                            // SAFETY: as above; construction validated the full matrix.
                            unsafe {
                                dst.add_assign(
                                    row + lane,
                                    column,
                                    dot_column(w, column, packed_row),
                                );
                            }
                        }
                    }
                }
                row += 4;
            }
            while row < rows {
                let packed_row = &input[row * blocks_per_row..(row + 1) * blocks_per_row];
                for column in dst.first..dst.first + dst.count {
                    // SAFETY: as above; construction validated the full matrix.
                    unsafe {
                        dst.add_assign(row, column, dot_column(w, column, packed_row));
                    }
                }
                row += 1;
            }
            return;
        }
        let (low_dst, high_dst) = dst.split();
        join(
            move || recurse(input, rows, w, low_dst),
            move || recurse(input, rows, w, high_dst),
        );
    }
    recurse(input, rows, w, DstColumns::new(dst, rows, w.out_features()));
}

#[inline]
fn should_use_parallel(rows: usize, w: &KQuantWeight, requested: bool) -> bool {
    let macs = rows
        .saturating_mul(w.in_features())
        .saturating_mul(w.out_features());
    requested
        && rayon::current_num_threads() > 1
        && w.out_features() > PARALLEL_LEAF_OUTPUTS
        && macs >= PARALLEL_MIN_MACS
}

/// Report the scheduler that this call would actually use in the current
/// Rayon pool. Diagnostics and benchmarks use the same predicate as execution
/// rather than inferring a path from a requested flag.
pub fn scheduler_name(rows: usize, w: &KQuantWeight, requested: bool) -> &'static str {
    if should_use_parallel(rows, w, requested) {
        "column-parallel-rayon"
    } else {
        "serial"
    }
}

/// Compute `dst += src × dequant(w)` using transient Q8_K activation rows.
///
/// Activations must be finite. A non-finite value returns an error before any
/// destination element is modified. The transient Q8_K buffer is cached per
/// OS thread but moved out of TLS while Rayon runs, so nested Rayon matmuls do
/// not hold overlapping `RefCell` borrows. Serial/parallel bit identity
/// additionally assumes the caller and Rayon workers use the same floating-
/// point control state (rounding mode and MXCSR FTZ/DAZ settings on x86).
pub fn matmul_k_q8_into_with_dispatch(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
    parallel: bool,
) -> Result<&'static str, String> {
    validate(src, rows, w, dst)?;
    let mut input = Q8_K_INPUT.with(|cache| std::mem::take(&mut *cache.borrow_mut()));

    #[cfg(target_arch = "x86_64")]
    let pack_result = if matches!(w.execution(), KExecution::CompressedX86) {
        #[cfg(test)]
        if route_probe::is_target(w) {
            route_probe::X86_PACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: x86 execution is validated against AVX2/FMA/F16C/SSSE3 at load.
        unsafe { x86::quantize_q8_k_into(src, &mut input) }
    } else {
        quantize_q8_k_into_scalar(src, &mut input)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let pack_result = quantize_q8_k_into_scalar(src, &mut input);

    if let Err(message) = pack_result {
        Q8_K_INPUT.with(|cache| *cache.borrow_mut() = input);
        return Err(message.to_string());
    }

    let dispatch = if should_use_parallel(rows, w, parallel) {
        parallel_body(&input, rows, w, dst);
        "column-parallel-rayon"
    } else {
        serial_body(&input, rows, w, dst);
        "serial"
    };
    Q8_K_INPUT.with(|cache| *cache.borrow_mut() = input);
    Ok(dispatch)
}

/// Compute the canonical product and discard the diagnostic dispatch result.
pub fn matmul_k_q8_into(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
    parallel: bool,
) -> Result<(), String> {
    matmul_k_q8_into_with_dispatch(src, rows, w, dst, parallel).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k_matmul::tests::{seeded_activations, seeded_q4_blocks, seeded_q6_blocks};

    fn weight(dtype: KQuantDtype, out: usize, input: usize, seed: u64) -> KQuantWeight {
        let blocks = out * (input / QK_K);
        let bytes = match dtype {
            KQuantDtype::Q4K => seeded_q4_blocks(blocks, seed),
            KQuantDtype::Q6K => seeded_q6_blocks(blocks, seed),
        };
        KQuantWeight::try_new(bytes, [out, input], dtype).unwrap()
    }

    #[test]
    fn q8_k_zero_and_signed_scale() {
        let mut packed = Vec::new();
        quantize_q8_k_into_scalar(&[0.0; QK_K], &mut packed).unwrap();
        assert_eq!(Q8_K_BLOCK_BYTES, 292, "ggml block_q8_K ABI");
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].d, 0.0);
        assert!(packed[0].qs.iter().all(|&q| q == 0));
        assert!(packed[0].bsums.iter().all(|&sum| sum == 0));

        let mut values = [0.0; QK_K];
        values[0] = -4.0;
        values[1] = 2.0;
        quantize_q8_k_into_scalar(&values, &mut packed).unwrap();
        assert_eq!(packed[0].qs[0], -127);
        assert_eq!(packed[0].qs[1], 64);
        assert!((packed[0].d * -127.0 + 4.0).abs() < 1e-6);
    }

    #[test]
    fn required_x86_ci_tier_is_available() {
        if std::env::var("EMBER_REQUIRE_X86_TESTS").as_deref() == Ok("1") {
            assert!(
                x86_k_supported(),
                "dedicated x86 gate requires AVX2/FMA/F16C/SSSE3"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_q8_k_packing_is_bit_identical_to_scalar() {
        if !x86_k_supported() {
            return;
        }
        let mut values = seeded_activations(32 * QK_K, 0x8a);
        for (index, block) in values.chunks_exact_mut(QK_K).enumerate() {
            let scale = match index % 4 {
                0 => 1.0,
                1 => 1e-20,
                2 => 1e20,
                _ => -3.25,
            };
            for value in block {
                *value *= scale;
            }
        }

        // Zero, signed-maximum ties, and every half-integer rounding lane.
        values[..QK_K].fill(0.0);
        let boundary = &mut values[QK_K..2 * QK_K];
        boundary[0] = -127.0;
        for (index, value) in boundary[1..].iter_mut().enumerate() {
            *value = (index % 253) as f32 - 126.0 + 0.5;
        }
        boundary[1..9].copy_from_slice(&[-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]);
        let ties = &mut values[2 * QK_K..3 * QK_K];
        ties.fill(0.0);
        ties[0] = 4.0;
        ties[1] = -4.0;
        let reverse_ties = &mut values[6 * QK_K..7 * QK_K];
        reverse_ties.fill(0.0);
        reverse_ties[0] = -4.0;
        reverse_ties[1] = 4.0;

        // Finite subnormals can overflow the inverse scale; zero lanes then
        // produce NaN products and must still narrow exactly like Rust casts.
        let tiny = &mut values[3 * QK_K..4 * QK_K];
        tiny.fill(0.0);
        tiny[0] = f32::from_bits(1);
        tiny[1] = -f32::from_bits(1);
        let extremes = &mut values[4 * QK_K..5 * QK_K];
        extremes.fill(0.25);
        extremes[0] = f32::MAX;
        extremes[1] = -f32::MAX;
        values[5 * QK_K..6 * QK_K].fill(-0.0);

        let mut scalar = Vec::new();
        let mut x86 = Vec::new();
        quantize_q8_k_into_scalar(&values, &mut scalar).unwrap();
        assert_eq!(&scalar[1].qs[1..9], &[-4, -2, -2, 0, 0, 2, 2, 4]);
        assert!(scalar[2].d.is_sign_negative());
        assert!(!scalar[6].d.is_sign_negative());
        assert_eq!(&scalar[2].qs[..2], &[-127, 127]);
        assert_eq!(&scalar[6].qs[..2], &[-127, 127]);
        // SAFETY: guarded by the complete x86 feature predicate.
        unsafe { super::x86::quantize_q8_k_into(&values, &mut x86) }.unwrap();
        assert_eq!(x86.len(), scalar.len());
        for (index, (actual, expected)) in x86.iter().zip(&scalar).enumerate() {
            assert_eq!(
                actual.d.to_bits(),
                expected.d.to_bits(),
                "block {index} scale"
            );
            assert_eq!(actual.qs, expected.qs, "block {index} quants");
            assert_eq!(actual.bsums, expected.bsums, "block {index} sums");
            for (group, (&sum, quants)) in actual
                .bsums
                .iter()
                .zip(actual.qs.chunks_exact(16))
                .enumerate()
            {
                let stored_sum: i16 = quants.iter().map(|&value| i16::from(value)).sum();
                assert_eq!(sum, stored_sum, "block {index} group {group}");
            }
        }
    }

    #[test]
    fn q8_k_pack_and_matmul_reject_non_finite_activations() {
        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = [0.0; QK_K];
            values[17] = non_finite;
            let mut packed = Vec::new();
            assert!(quantize_q8_k_into_scalar(&values, &mut packed).is_err());

            #[cfg(target_arch = "x86_64")]
            if x86_k_supported() {
                // SAFETY: guarded by the complete x86 feature predicate.
                assert!(unsafe { super::x86::quantize_q8_k_into(&values, &mut packed) }.is_err());
            }

            let scalar = weight(KQuantDtype::Q4K, 3, 2 * QK_K, 0x51);
            let mut tiers = vec![("scalar", scalar.clone())];
            if x86_k_supported() {
                tiers.push(("x86", scalar.with_execution(KExecution::CompressedX86)));
            }
            let mut full_src = vec![0.25; 4 * QK_K];
            full_src[3 * QK_K + 17] = non_finite;
            for (tier, weight) in tiers {
                for parallel in [false, true] {
                    let mut dst = [1.0, -0.0, 7.5, -2.0, 3.0, 9.0];
                    let before = dst.map(f32::to_bits);
                    let error =
                        matmul_k_q8_into(&full_src, 2, &weight, &mut dst, parallel).unwrap_err();
                    assert!(error.contains("finite activations"));
                    assert_eq!(
                        dst.map(f32::to_bits),
                        before,
                        "destination changed on {tier}/parallel={parallel} error"
                    );
                }
            }
        }
    }

    #[test]
    fn integer_dots_match_dequantized_q8_reference() {
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let w = weight(dtype, 7, 512, 11 + dtype.gguf_code() as u64);
            let src = seeded_activations(512, 91);
            let mut packed = Vec::new();
            quantize_q8_k_into_scalar(&src, &mut packed).unwrap();
            let quantized_src: Vec<f32> = packed
                .iter()
                .flat_map(|block| block.qs.iter().map(|&q| block.d * f32::from(q)))
                .collect();
            let full = w.dequantize_all();
            for column in 0..w.out_features() {
                let row = &full.data()[column * 512..(column + 1) * 512];
                let expected: f32 = row.iter().zip(&quantized_src).map(|(a, b)| a * b).sum();
                let scalar = match dtype {
                    KQuantDtype::Q4K => {
                        q4_k_dot_q8_k_scalar(w.data(), w.blocks_per_row(), column, &packed)
                    }
                    KQuantDtype::Q6K => {
                        q6_k_dot_q8_k_scalar(w.data(), w.blocks_per_row(), column, &packed)
                    }
                };
                let scale = expected.abs().max(1.0);
                assert!(
                    (scalar - expected).abs() <= 2e-5 * scale,
                    "{dtype:?} column {column}: scalar {scalar} reference {expected}"
                );
                let dispatched_weight = if x86_k_supported() {
                    w.clone().with_execution(KExecution::CompressedX86)
                } else {
                    w.clone()
                };
                let dispatched = dot_column(&dispatched_weight, column, &packed);
                assert!(
                    (dispatched - scalar).abs() <= 2e-5 * scale,
                    "{dtype:?} column {column}: dispatched {dispatched} scalar {scalar}"
                );
            }
        }
    }

    #[test]
    fn pinned_llama_cpp_known_answer_vector() {
        // Generated by tools/verify_k_quant_llamacpp.c at llama.cpp
        // 47c786924ad1ab7e91da2cdc72fcdb563780c2bd (generic/reference path).
        let src: Vec<f32> = (0..512)
            .map(|index| ((index * 37) % 257 - 128) as f32 / 16.0)
            .collect();
        let mut packed = Vec::new();
        quantize_q8_k_into_scalar(&src, &mut packed).unwrap();
        let mut packed_bytes = Vec::with_capacity(2 * Q8_K_BLOCK_BYTES);
        for block in &packed {
            packed_bytes.extend_from_slice(&block.d.to_le_bytes());
            packed_bytes.extend(block.qs.iter().map(|&value| value as u8));
            for &sum in &block.bsums {
                packed_bytes.extend_from_slice(&sum.to_le_bytes());
            }
        }
        use sha2::{Digest, Sha256};
        assert_eq!(
            format!("{:x}", Sha256::digest(&packed_bytes)),
            "d1fbb94a39f658c146b9a6e797cb8849fa3d8dc39fd412b113b5b0415c1d9bde"
        );

        let mut q4 = Vec::with_capacity(2 * KQuantDtype::Q4K.block_bytes());
        let mut q6 = Vec::with_capacity(2 * KQuantDtype::Q6K.block_bytes());
        for block_index in 0..2 {
            let mut block = vec![0; KQuantDtype::Q4K.block_bytes()];
            let d = if block_index == 0 { 0x3400u16 } else { 0xb800 };
            let dmin = if block_index == 0 { 0x3000u16 } else { 0x3800 };
            block[0..2].copy_from_slice(&d.to_le_bytes());
            block[2..4].copy_from_slice(&dmin.to_le_bytes());
            for (index, value) in block[4..16].iter_mut().enumerate() {
                *value = (block_index * 53 + index * 37 + 11) as u8;
            }
            for (index, value) in block[16..].iter_mut().enumerate() {
                *value = (block_index * 29 + index * 73 + 19) as u8;
            }
            q4.extend_from_slice(&block);

            let mut block = vec![0; KQuantDtype::Q6K.block_bytes()];
            for (index, value) in block[..128].iter_mut().enumerate() {
                *value = (block_index * 31 + index * 67 + 7) as u8;
            }
            for (index, value) in block[128..192].iter_mut().enumerate() {
                *value = (block_index * 47 + index * 43 + 23) as u8;
            }
            for (index, value) in block[192..208].iter_mut().enumerate() {
                *value = ((block_index * 59 + index * 41) as i32 - 121) as i8 as u8;
            }
            let d = if block_index == 0 { 0x3800u16 } else { 0xb400 };
            block[208..210].copy_from_slice(&d.to_le_bytes());
            q6.extend_from_slice(&block);
        }

        let q4_weight = KQuantWeight::try_new(q4, [1, 512], KQuantDtype::Q4K).unwrap();
        let q6_weight = KQuantWeight::try_new(q6, [1, 512], KQuantDtype::Q6K).unwrap();
        let q4_scalar = q4_k_dot_q8_k_scalar(q4_weight.data(), 2, 0, &packed);
        let q6_scalar = q6_k_dot_q8_k_scalar(q6_weight.data(), 2, 0, &packed);
        let q4_reference = f32::from_bits(0xc5eb_1fdf);
        let q6_reference = f32::from_bits(0xc701_2543);
        assert!(
            (q4_scalar - q4_reference).abs() <= 0.001,
            "q4 scalar {q4_scalar} reference {q4_reference}"
        );
        assert!(
            (q6_scalar - q6_reference).abs() <= 0.004,
            "q6 scalar {q6_scalar} reference {q6_reference}"
        );

        #[cfg(target_arch = "x86_64")]
        if x86_k_supported() {
            // SAFETY: guarded by the complete x86 feature predicate.
            let q4_x86 = unsafe { super::x86::q4_k_dot_q8_k(q4_weight.data(), 2, 0, &packed) };
            // SAFETY: guarded by the complete x86 feature predicate.
            let q6_x86 = unsafe { super::x86::q6_k_dot_q8_k(q6_weight.data(), 2, 0, &packed) };
            assert_eq!(q4_x86.to_bits(), 0xc5eb_1fde);
            assert_eq!(q6_x86.to_bits(), 0xc701_2543);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_extrema_dots_do_not_saturate_or_overflow() {
        if !x86_k_supported() {
            return;
        }
        let input = [Q8KBlock {
            d: 1.0,
            qs: [127; QK_K],
            bsums: [16 * 127; QK_K / 16],
        }];

        let mut q4 = vec![0xff; KQuantDtype::Q4K.block_bytes()];
        q4[0..2].copy_from_slice(&[0x00, 0x3c]); // f16(1.0) d
        q4[2..4].copy_from_slice(&[0x00, 0x3c]); // f16(1.0) dmin
        let q4_scalar = q4_k_dot_q8_k_scalar(&q4, 1, 0, &input);
        // SAFETY: guarded by the complete x86 feature predicate.
        let q4_x86 = unsafe { super::x86::q4_k_dot_q8_k(&q4, 1, 0, &input) };
        assert_eq!(q4_x86.to_bits(), q4_scalar.to_bits());
        assert!(q4_scalar.is_finite() && q4_scalar.abs() > 1.0e6);

        let mut q6 = vec![0; KQuantDtype::Q6K.block_bytes()];
        q6[192..208].fill(0x80); // scale -128; reconstructed quants are -32
        q6[208..210].copy_from_slice(&[0x00, 0x3c]); // f16(1.0) d
        let q6_scalar = q6_k_dot_q8_k_scalar(&q6, 1, 0, &input);
        // SAFETY: guarded by the complete x86 feature predicate.
        let q6_x86 = unsafe { super::x86::q6_k_dot_q8_k(&q6, 1, 0, &input) };
        assert_eq!(q6_x86.to_bits(), q6_scalar.to_bits());
        assert!(q6_scalar.is_finite() && q6_scalar.abs() > 1.0e8);
    }

    #[test]
    fn canonical_matmul_accumulates_for_fused_output_projection() {
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let scalar = weight(dtype, 5, 512, 0xf5 + dtype.gguf_code() as u64);
            let mut tiers = vec![("scalar", scalar.clone())];
            if x86_k_supported() {
                tiers.push(("x86", scalar.with_execution(KExecution::CompressedX86)));
            }
            let src = seeded_activations(512, 0x55);
            for (tier, weight) in tiers {
                let mut projection = vec![0.0; 5];
                matmul_k_q8_into(&src, 1, &weight, &mut projection, false).unwrap();
                let initial = [2.0, -1.0, 0.25, -0.0, 7.0];
                let mut fused_destination = initial;
                matmul_k_q8_into(&src, 1, &weight, &mut fused_destination, false).unwrap();
                for index in 0..5 {
                    assert_eq!(
                        fused_destination[index].to_bits(),
                        (initial[index] + projection[index]).to_bits(),
                        "{dtype:?}/{tier} F5 accumulate at {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn full_api_covers_dtype_tier_shape_and_row_remainders() {
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            for input_features in [QK_K, 2 * QK_K] {
                for rows in [1usize, 2, 3, 4, 5, 7] {
                    for output_features in [1usize, 3, 17] {
                        let scalar_weight = weight(
                            dtype,
                            output_features,
                            input_features,
                            0x400 + rows as u64 + output_features as u64,
                        );
                        let src = seeded_activations(rows * input_features, 0x501 + rows as u64);
                        let initial: Vec<f32> = (0..rows * output_features)
                            .map(|index| (index % 11) as f32 * 0.125 - 0.5)
                            .collect();
                        let mut packed = Vec::new();
                        quantize_q8_k_into_scalar(&src, &mut packed).unwrap();
                        let blocks_per_row = input_features / QK_K;
                        let mut expected = initial.clone();
                        for row in 0..rows {
                            let packed_row =
                                &packed[row * blocks_per_row..(row + 1) * blocks_per_row];
                            for column in 0..output_features {
                                expected[row * output_features + column] += match dtype {
                                    KQuantDtype::Q4K => q4_k_dot_q8_k_scalar(
                                        scalar_weight.data(),
                                        blocks_per_row,
                                        column,
                                        packed_row,
                                    ),
                                    KQuantDtype::Q6K => q6_k_dot_q8_k_scalar(
                                        scalar_weight.data(),
                                        blocks_per_row,
                                        column,
                                        packed_row,
                                    ),
                                };
                            }
                        }

                        let mut tiers = vec![("scalar", scalar_weight.clone())];
                        if x86_k_supported() {
                            tiers.push((
                                "x86",
                                scalar_weight
                                    .clone()
                                    .with_execution(KExecution::CompressedX86),
                            ));
                        }
                        for (tier, weight) in tiers {
                            let mut actual = initial.clone();
                            matmul_k_q8_into(&src, rows, &weight, &mut actual, false).unwrap();
                            for (index, (&actual, &expected)) in
                                actual.iter().zip(&expected).enumerate()
                            {
                                let tolerance = if tier == "scalar" {
                                    0.0
                                } else {
                                    5e-4 * expected.abs().max(1.0)
                                };
                                assert!(
                                    actual.is_finite() && (actual - expected).abs() <= tolerance,
                                    "{dtype:?}/{tier} in={input_features} out={output_features} rows={rows} index={index}: {actual} vs {expected}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_weight_scales_leave_destination_unchanged() {
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let rows = 5;
            let input_features = 512;
            let output_features = 9;
            let blocks = output_features * input_features / QK_K;
            let mut bytes = match dtype {
                KQuantDtype::Q4K => seeded_q4_blocks(blocks, 0x41),
                KQuantDtype::Q6K => seeded_q6_blocks(blocks, 0x61),
            };
            for block in bytes.chunks_exact_mut(dtype.block_bytes()) {
                match dtype {
                    KQuantDtype::Q4K => block[..4].fill(0),
                    KQuantDtype::Q6K => block[208..210].fill(0),
                }
            }
            let scalar_weight =
                KQuantWeight::try_new(bytes, [output_features, input_features], dtype).unwrap();
            let mut tiers = vec![scalar_weight.clone()];
            if x86_k_supported() {
                tiers.push(scalar_weight.with_execution(KExecution::CompressedX86));
            }
            let src = seeded_activations(rows * input_features, 0x71);
            for weight in tiers {
                let expected: Vec<f32> = (0..rows * output_features)
                    .map(|index| index as f32 * 0.25 - 3.0)
                    .collect();
                let mut actual = expected.clone();
                matmul_k_q8_into(&src, rows, &weight, &mut actual, false).unwrap();
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{dtype:?} {:?}",
                    weight.execution()
                );
            }
        }
    }

    #[test]
    fn warmed_serial_matmul_allocates_zero_times_on_calling_thread() {
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let rows = 3;
            let scalar = weight(dtype, 7, 512, 0x31 + dtype.gguf_code() as u64);
            let mut tiers = vec![("scalar", scalar.clone())];
            if x86_k_supported() {
                tiers.push(("x86", scalar.with_execution(KExecution::CompressedX86)));
            }
            for (tier, weight) in tiers {
                let src = seeded_activations(rows * weight.in_features(), 0x91);
                let mut dst = vec![0.0; rows * weight.out_features()];
                matmul_k_q8_into(&src, rows, &weight, &mut dst, false).unwrap();
                dst.fill(0.0);
                let (result, allocations) = crate::alloc_counter::count_allocations(|| {
                    matmul_k_q8_into(&src, rows, &weight, &mut dst, false)
                });
                result.unwrap();
                assert_eq!(allocations, 0, "{dtype:?}/{tier} warmed workspace");
            }
        }
    }

    #[test]
    fn serial_and_actual_parallel_are_bit_identical() {
        let _route_guard = route_probe::lock();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let rows = 7;
            let scalar_weight = weight(dtype, 769, 512, 42 + dtype.gguf_code() as u64);
            let mut tiers = vec![("scalar", scalar_weight.clone())];
            if x86_k_supported() {
                tiers.push((
                    "x86",
                    scalar_weight.with_execution(KExecution::CompressedX86),
                ));
            }
            let src = seeded_activations(rows * 512, 7);
            for (tier, weight) in tiers {
                let mut serial: Vec<f32> = (0..rows * weight.out_features())
                    .map(|index| (index % 17) as f32 * 0.125 - 1.0)
                    .collect();
                let mut parallel = serial.clone();
                matmul_k_q8_into(&src, rows, &weight, &mut serial, false).unwrap();
                route_probe::reset(&weight);
                pool.install(|| {
                    assert_eq!(rayon::current_num_threads(), 2);
                    assert!(should_use_parallel(rows, &weight, true));
                    matmul_k_q8_into(&src, rows, &weight, &mut parallel, true).unwrap();
                });
                use std::sync::atomic::Ordering;
                assert_eq!(route_probe::PARALLEL_BODIES.load(Ordering::Relaxed), 1);
                assert!(route_probe::PARALLEL_LEAVES.load(Ordering::Relaxed) >= 2);
                assert_eq!(
                    route_probe::PARALLEL_WORKERS
                        .load(Ordering::Relaxed)
                        .count_ones(),
                    2,
                    "{dtype:?}/{tier} did not execute leaves on both dedicated-pool workers"
                );
                if tier == "x86" {
                    assert_eq!(route_probe::X86_PACKS.load(Ordering::Relaxed), 1);
                    assert!(route_probe::X86_DOTS.load(Ordering::Relaxed) > 0);
                } else {
                    assert_eq!(route_probe::X86_PACKS.load(Ordering::Relaxed), 0);
                    assert_eq!(route_probe::X86_DOTS.load(Ordering::Relaxed), 0);
                }
                route_probe::finish();
                for (index, (&actual, &expected)) in parallel.iter().zip(&serial).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{dtype:?} {tier} destination {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn nested_rayon_matmuls_use_independent_tls_storage() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let rows = 7;
        let weight = weight(KQuantDtype::Q4K, 769, 512, 0xa5);
        let src_a = seeded_activations(rows * 512, 0xb1);
        let src_b = seeded_activations(rows * 512, 0xb2);
        let mut expected_a = vec![0.0; rows * weight.out_features()];
        let mut expected_b = expected_a.clone();
        matmul_k_q8_into(&src_a, rows, &weight, &mut expected_a, false).unwrap();
        matmul_k_q8_into(&src_b, rows, &weight, &mut expected_b, false).unwrap();
        let mut dst_a = vec![0.0; rows * weight.out_features()];
        let mut dst_b = dst_a.clone();
        pool.install(|| {
            rayon::join(
                || matmul_k_q8_into(&src_a, rows, &weight, &mut dst_a, true).unwrap(),
                || matmul_k_q8_into(&src_b, rows, &weight, &mut dst_b, true).unwrap(),
            );
        });
        assert_eq!(
            dst_a
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_a
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            dst_b
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_b
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn four_row_tile_matches_independent_dots() {
        if !x86_k_supported() {
            return;
        }
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let w = weight(dtype, 5, 512, 73 + dtype.gguf_code() as u64)
                .with_execution(KExecution::CompressedX86);
            let src = seeded_activations(4 * 512, 19);
            let mut packed = Vec::new();
            quantize_q8_k_into_scalar(&src, &mut packed).unwrap();
            for column in 0..w.out_features() {
                let tiled = dot_four_rows(&w, column, &packed).unwrap();
                for row in 0..4 {
                    let single = dot_column(
                        &w,
                        column,
                        &packed[row * w.blocks_per_row()..(row + 1) * w.blocks_per_row()],
                    );
                    assert_eq!(
                        tiled[row].to_bits(),
                        single.to_bits(),
                        "{dtype:?} column {column} row {row}"
                    );
                }
            }
        }
    }

    #[test]
    fn q4_presplit_dot_four_rows_matches_packed() {
        if !x86_k_supported() {
            return;
        }
        let w_packed =
            weight(KQuantDtype::Q4K, 5, 512, 0x5a).with_execution(KExecution::CompressedX86);
        let w_presplit = w_packed.clone().with_presplit();
        let src = seeded_activations(4 * 512, 0x5b);
        let mut packed = Vec::new();
        quantize_q8_k_into_scalar(&src, &mut packed).unwrap();
        for column in 0..w_packed.out_features() {
            let expected = dot_four_rows(&w_packed, column, &packed).unwrap();
            let actual = dot_four_rows(&w_presplit, column, &packed).unwrap();
            for row in 0..4 {
                assert_eq!(
                    expected[row].to_bits(),
                    actual[row].to_bits(),
                    "column {column} row {row}"
                );
            }
        }
    }

    #[test]
    fn q6_presplit_dot_four_rows_matches_packed() {
        if !x86_k_supported() {
            return;
        }
        let w_packed =
            weight(KQuantDtype::Q6K, 5, 512, 0x6a).with_execution(KExecution::CompressedX86);
        let w_presplit = w_packed.clone().with_presplit();
        let src = seeded_activations(4 * 512, 0x6b);
        let mut packed = Vec::new();
        quantize_q8_k_into_scalar(&src, &mut packed).unwrap();
        for column in 0..w_packed.out_features() {
            let expected = dot_four_rows(&w_packed, column, &packed).unwrap();
            let actual = dot_four_rows(&w_presplit, column, &packed).unwrap();
            for row in 0..4 {
                assert_eq!(
                    expected[row].to_bits(),
                    actual[row].to_bits(),
                    "column {column} row {row}"
                );
            }
        }
    }

    #[test]
    fn unsplittable_output_range_reports_serial_even_above_mac_threshold() {
        let _route_guard = route_probe::lock();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let rows = 1;
        let weight = weight(KQuantDtype::Q4K, PARALLEL_LEAF_OUTPUTS, 8192, 0xc1);
        let src = seeded_activations(8192, 0xc2);
        let mut dst = vec![0.0; PARALLEL_LEAF_OUTPUTS];
        route_probe::reset(&weight);
        let dispatch = pool.install(|| {
            assert_eq!(scheduler_name(rows, &weight, true), "serial");
            matmul_k_q8_into_with_dispatch(&src, rows, &weight, &mut dst, true).unwrap()
        });
        assert_eq!(dispatch, "serial");
        assert_eq!(
            route_probe::PARALLEL_BODIES.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        route_probe::finish();
    }

    #[test]
    fn invalid_shapes_and_execution_states_are_errors() {
        let w = weight(KQuantDtype::Q4K, 3, 256, 5);
        let mut dst = [0.0; 6];
        assert!(matmul_k_q8_into(&[], 0, &w, &mut [], false)
            .unwrap_err()
            .contains("rows must be nonzero"));
        assert!(matmul_k_q8_into(&[0.0; 255], 1, &w, &mut dst[..3], false).is_err());
        assert!(matmul_k_q8_into(&[0.0; 512], 2, &w, &mut dst[..5], false).is_err());
        assert!(matmul_k_q8_into(&[], usize::MAX, &w, &mut [], false)
            .unwrap_err()
            .contains("shape product overflow"));

        if !x86_k_supported() {
            let unsupported = w.with_execution(KExecution::CompressedX86);
            assert!(
                matmul_k_q8_into(&[0.0; 256], 1, &unsupported, &mut dst[..3], false,)
                    .unwrap_err()
                    .contains("unavailable")
            );
        }
    }
}
