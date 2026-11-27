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

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "aarch64")]
mod arm_tiles;

/// Four-row interleaved activation groups for the ARM prefill tiles; other
/// targets never build them.
#[cfg(target_arch = "aarch64")]
type Packed4 = arm_tiles::Q8K4Block;
#[cfg(not(target_arch = "aarch64"))]
type Packed4 = ();

/// Runtime feature gate for the recorded ARM K-quant tier.
pub fn arm_k_supported() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
            && std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

const PARALLEL_LEAF_OUTPUTS: usize = 256;
/// Columns per decode-team chunk for single-token matvecs: small enough that
/// a slow or late member holds back little work, large enough that claiming a
/// chunk (one CAS) is noise next to the dots.
const DECODE_TEAM_CHUNK_COLUMNS: usize = 64;
const PARALLEL_MIN_MACS: usize = 512_000;

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
        if let Some(index) = rayon::current_thread_index()
            && index < u64::BITS as usize
        {
            PARALLEL_WORKERS.fetch_or(1u64 << index, Ordering::Relaxed);
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
    /// Four-row repack of `Q8_K_INPUT` for multi-row ARM calls (same
    /// ownership discipline).
    static Q8_K_INPUT4: RefCell<Vec<Packed4>> = const { RefCell::new(Vec::new()) };
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

/// Whether the opt-in AVX-512 K-quant dot tier is active for this process
/// (`EMBER_K_AVX512=1` plus runtime feature checks). Diagnostics only.
#[inline]
pub fn k_avx512_opt_in() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        x86::k_avx512_enabled()
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

    // Blocks are independent: a prefill-sized input is encoded in parallel
    // by the same per-block code. Validation fails if any block is
    // non-finite, exactly as the serial loop does (the matmul destination
    // has not been touched yet either way).
    if blocks >= PARALLEL_Q8_K_QUANTIZE_MIN_BLOCKS && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        return src
            .par_chunks(PARALLEL_Q8_K_QUANTIZE_CHUNK_BLOCKS * QK_K)
            .zip(dst.par_chunks_mut(PARALLEL_Q8_K_QUANTIZE_CHUNK_BLOCKS))
            .try_for_each(|(values, encoded)| quantize_q8_k_blocks(values, encoded));
    }
    quantize_q8_k_blocks(src, dst)
}

/// Q8_K inputs with at least this many blocks quantize in parallel.
const PARALLEL_Q8_K_QUANTIZE_MIN_BLOCKS: usize = 256;
/// Blocks per parallel Q8_K quantization task.
const PARALLEL_Q8_K_QUANTIZE_CHUNK_BLOCKS: usize = 32;

fn quantize_q8_k_blocks(src: &[f32], dst: &mut [Q8KBlock]) -> Result<(), &'static str> {
    #[cfg(target_arch = "aarch64")]
    {
        arm::quantize_q8_k_blocks(src, dst)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        quantize_q8_k_blocks_scalar(src, dst)
    }
}

/// The reference Q8_K encoder; the NEON encoder reproduces it bit for bit.
#[cfg(any(not(target_arch = "aarch64"), test))]
fn quantize_q8_k_blocks_scalar(src: &[f32], dst: &mut [Q8KBlock]) -> Result<(), &'static str> {
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
mod x86;

#[inline]
fn dot_column(w: &KQuantWeight, column: usize, input: &[Q8KBlock]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if matches!(w.execution(), KExecution::CompressedArm) {
        // SAFETY: validate checks the feature gate; weight construction checks layout.
        return unsafe {
            match w.dtype() {
                KQuantDtype::Q4K => arm::q4::<1>(w.data(), w.blocks_per_row(), column, input)[0],
                KQuantDtype::Q6K => arm::q6::<1>(w.data(), w.blocks_per_row(), column, input)[0],
            }
        };
    }
    #[cfg(target_arch = "x86_64")]
    if matches!(w.execution(), KExecution::CompressedX86) {
        #[cfg(test)]
        if route_probe::is_target(w) {
            route_probe::X86_DOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if x86::k_avx512_enabled() {
            // SAFETY: the flag gate runtime-checks the AVX-512 feature set;
            // weight layout is validated at construction.
            return unsafe {
                match w.dtype() {
                    KQuantDtype::Q4K => {
                        x86::q4_k_dot_q8_k_avx512(w.data(), w.blocks_per_row(), column, input)
                    }
                    KQuantDtype::Q6K => {
                        x86::q6_k_dot_q8_k_avx512(w.data(), w.blocks_per_row(), column, input)
                    }
                }
            };
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
fn dot_four_rows(w: &KQuantWeight, column: usize, input: &[Q8KBlock]) -> Option<[f32; 4]> {
    #[cfg(target_arch = "aarch64")]
    if matches!(w.execution(), KExecution::CompressedArm) {
        // SAFETY: validate checks the recorded tier's CPU features and shapes.
        return Some(unsafe {
            match w.dtype() {
                KQuantDtype::Q4K => arm::q4::<4>(w.data(), w.blocks_per_row(), column, input),
                KQuantDtype::Q6K => arm::q6::<4>(w.data(), w.blocks_per_row(), column, input),
            }
        });
    }
    let _ = (w, column, input);
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
        KExecution::CompressedArm if !arm_k_supported() => {
            return Err(
                "matmul_k_q8: compressed-arm was recorded but NEON+dotprod is unavailable"
                    .to_string(),
            );
        }
        KExecution::CompressedScalar | KExecution::CompressedX86 | KExecution::CompressedArm => {}
    }
    Ok(())
}

fn serial_body(
    input: &[Q8KBlock],
    packed4: &[Packed4],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) {
    let blocks_per_row = w.blocks_per_row();
    let out_features = w.out_features();
    #[cfg(target_arch = "aarch64")]
    if !packed4.is_empty() {
        // SAFETY: packed groups exist only for a validated CompressedArm weight.
        unsafe { arm_tiles::serial(packed4, rows, w, dst) };
        return;
    }
    let _ = packed4;
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

fn parallel_body(
    input: &[Q8KBlock],
    packed4: &[Packed4],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) {
    #[cfg(test)]
    if route_probe::is_target(w) {
        route_probe::PARALLEL_BODIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(target_arch = "aarch64")]
    if !packed4.is_empty() {
        // Multi-row ARM prefill: 2-D (row group × column) tiles.
        // SAFETY: packed groups exist only for a validated CompressedArm weight.
        unsafe { arm_tiles::parallel(packed4, rows, w, dst) };
        return;
    }
    let _ = packed4;
    // Single-token decode: hand fixed column chunks to the spin-waiting decode
    // team (performance cores only, no fork/join per projection). Each column
    // is computed by the same serial kernel, so the result is independent of
    // which member claims it. Falls through to the Rayon tree when the team
    // is unavailable (a nested call from a Rayon worker, or a busy team).
    if rows == 1 {
        let out_features = w.out_features();
        let chunks = out_features.div_ceil(DECODE_TEAM_CHUNK_COLUMNS);
        let out = crate::decode_pool::SharedMut::new(dst);
        let ran = crate::decode_pool::run(chunks, &|chunk| {
            let first = chunk * DECODE_TEAM_CHUNK_COLUMNS;
            let count = DECODE_TEAM_CHUNK_COLUMNS.min(out_features - first);
            // SAFETY: chunks cover disjoint column ranges of `dst`, which
            // outlives the region.
            let columns = unsafe { out.range(first, count) };
            for (offset, value) in columns.iter_mut().enumerate() {
                *value += dot_column(w, first + offset, input);
            }
        });
        if ran {
            return;
        }
    }
    // Split into ~two leaves per worker so work-stealing can smooth the tail
    // (low-column projections such as K/V otherwise leave most workers idle).
    // Floor at 128 columns per leaf: below that, per-leaf dispatch and
    // crossbeam steal overhead dominate the dots themselves.
    let leaf_outputs = PARALLEL_LEAF_OUTPUTS
        .min(
            w.out_features()
                .div_ceil(2 * rayon::current_num_threads().max(1)),
        )
        .max(128);
    fn recurse(
        input: &[Q8KBlock],
        rows: usize,
        w: &KQuantWeight,
        mut dst: DstColumns,
        leaf_outputs: usize,
    ) {
        if dst.count <= leaf_outputs {
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
            move || recurse(input, rows, w, low_dst, leaf_outputs),
            move || recurse(input, rows, w, high_dst, leaf_outputs),
        );
    }
    recurse(
        input,
        rows,
        w,
        DstColumns::new(dst, rows, w.out_features()),
        leaf_outputs,
    );
}

/// Whether a `[out_features, in_features]` decode matvec takes the
/// column-parallel route under a pool of `threads` workers.
///
/// This is the exact shape predicate `should_use_parallel` applies to a
/// resident weight; the runtime schedule reports it, so the thresholds live
/// in one place.
#[inline]
pub fn parallel_for_shape(
    rows: usize,
    in_features: usize,
    out_features: usize,
    threads: usize,
    requested: bool,
) -> bool {
    let macs = rows
        .saturating_mul(in_features)
        .saturating_mul(out_features);
    requested && threads > 1 && out_features > PARALLEL_LEAF_OUTPUTS && macs >= PARALLEL_MIN_MACS
}

#[inline]
fn should_use_parallel(rows: usize, w: &KQuantWeight, requested: bool) -> bool {
    parallel_for_shape(
        rows,
        w.in_features(),
        w.out_features(),
        rayon::current_num_threads(),
        requested,
    )
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

    // Multi-row ARM calls repack the Q8_K rows into four-row groups for the
    // prefill tiles (bit-identical to the per-column kernels).
    let mut packed4 = Q8_K_INPUT4.with(|cache| std::mem::take(&mut *cache.borrow_mut()));
    packed4.clear();
    #[cfg(target_arch = "aarch64")]
    if rows > 1 && matches!(w.execution(), KExecution::CompressedArm) {
        arm_tiles::pack_rows(&input, rows, w.blocks_per_row(), &mut packed4);
    }

    let dispatch = if should_use_parallel(rows, w, parallel) {
        parallel_body(&input, &packed4, rows, w, dst);
        "column-parallel-rayon"
    } else {
        serial_body(&input, &packed4, rows, w, dst);
        "serial"
    };
    Q8_K_INPUT.with(|cache| *cache.borrow_mut() = input);
    Q8_K_INPUT4.with(|cache| *cache.borrow_mut() = packed4);
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
mod tests;
