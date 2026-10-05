//! F16/BF16 weights kept in their GGUF encoding and widened on use.
//!
//! The loader used to widen every F16/BF16 tensor into an owned f32 buffer,
//! and the Llama builder then transposed each linear weight into a second
//! f32 buffer: about twice the file size in anonymous memory, built before
//! the first token. A [`HalfWeight`] keeps the 16-bit encoding instead
//! (shared with the file mapping when loaded from disk) and widens values
//! where they are consumed, with the conversion the loader used, so every
//! result is bit-identical to the eager path:
//!
//! * [`widen_into`] is the loader's element conversion, vectorized; it is
//!   checked against the scalar conversion on all 65,536 encodings.
//! * [`half_matmul_into`] widens a block of output rows into an f32 scratch
//!   and calls the same `matrixmultiply::sgemm` the eager path calls, with
//!   the weight described by strides instead of a transposed copy. sgemm's
//!   arithmetic for one output element depends only on `k` (its fixed KC
//!   blocking and the per-element FMA chain), not on `n`, on the operand
//!   strides or on where the element falls in a micro-tile, so splitting the
//!   output columns into blocks reproduces the full product bit for bit.
//! * [`half_matvec_sequential_into`] reproduces the planned decode matvec:
//!   one left-to-right `acc += x * w` sum per output, with no fused
//!   multiply-add.

use std::cell::RefCell;
use std::ops::Range;
use std::sync::Arc;

use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use rayon::prelude::*;

use crate::tensor::CpuTensor;

/// Encoding of a [`HalfWeight`]'s elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfDtype {
    /// IEEE 754 binary16 (GGML dtype 1).
    F16,
    /// bfloat16, the top half of an f32 (GGML dtype 30).
    Bf16,
}

impl HalfDtype {
    /// Lower-case dtype name as GGUF tooling prints it.
    pub fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Clone)]
enum HalfData {
    Owned(Arc<[u16]>),
    /// A byte range of the read-only model mapping. Construction checks that
    /// it is 2-byte aligned with an even length on a little-endian target, so
    /// it can be viewed as `[u16]` holding the GGUF little-endian values.
    Mapped {
        mmap: Arc<memmap2::Mmap>,
        range: Range<usize>,
    },
}

/// A tensor of F16 or BF16 values in GGUF order.
///
/// `dims` are the GGUF-native dimensions (first dimension contiguous), as
/// the eager loader recorded them on the f32 tensor it produced. For a 2-D
/// linear weight that is `[in_features, out_features]`, so each output
/// feature is one contiguous row of `in_features` values; for an embedding
/// table it is `[embed, vocab]` with one contiguous row per token. Cloning
/// shares the storage.
#[derive(Clone)]
pub struct HalfWeight {
    dtype: HalfDtype,
    dims: Vec<usize>,
    data: HalfData,
}

impl std::fmt::Debug for HalfWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HalfWeight")
            .field("dtype", &self.dtype)
            .field("dims", &self.dims)
            .field("mapped", &self.is_mapped())
            .finish()
    }
}

fn element_count(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
}

impl HalfWeight {
    /// Wrap owned 16-bit values (already decoded from little-endian bytes).
    ///
    /// # Panics
    ///
    /// If `bits.len()` is not the product of `dims`.
    pub fn from_bits(dtype: HalfDtype, dims: Vec<usize>, bits: Vec<u16>) -> Self {
        assert_eq!(
            Some(bits.len()),
            element_count(&dims),
            "HalfWeight: {} values do not fill dims {dims:?}",
            bits.len()
        );
        Self {
            dtype,
            dims,
            data: HalfData::Owned(bits.into()),
        }
    }

    /// Share a byte range of the model mapping. Returns `None` when the range
    /// cannot be viewed in place (misaligned, a length that does not match
    /// `dims`, or a big-endian target); the caller then copies the values.
    pub(crate) fn from_mmap(
        dtype: HalfDtype,
        dims: Vec<usize>,
        mmap: Arc<memmap2::Mmap>,
        range: Range<usize>,
    ) -> Option<Self> {
        if cfg!(target_endian = "big") || range.start > range.end || range.end > mmap.len() {
            return None;
        }
        let bytes = range.end - range.start;
        if Some(bytes) != element_count(&dims).and_then(|count| count.checked_mul(2)) {
            return None;
        }
        let start = (mmap.as_ptr() as usize).checked_add(range.start)?;
        if !start.is_multiple_of(std::mem::align_of::<u16>()) {
            return None;
        }
        Some(Self {
            dtype,
            dims,
            data: HalfData::Mapped { mmap, range },
        })
    }

    pub fn dtype(&self) -> HalfDtype {
        self.dtype
    }

    /// GGUF-native dimensions (first dimension contiguous).
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Whether the values live in the shared model mapping.
    pub fn is_mapped(&self) -> bool {
        matches!(self.data, HalfData::Mapped { .. })
    }

    /// The raw 16-bit values, GGUF order.
    pub fn bits(&self) -> &[u16] {
        match &self.data {
            HalfData::Owned(bits) => bits,
            HalfData::Mapped { mmap, range } => {
                let bytes = &mmap[range.clone()];
                // SAFETY: `from_mmap` admitted this range only when its start
                // address is 2-byte aligned, its byte length is even (twice
                // the element count) and the target is little-endian, so the
                // bytes are `bytes.len() / 2` properly aligned `u16`s equal
                // to the GGUF little-endian values. Every bit pattern is a
                // valid `u16`. The mapping is read-only and owned by the
                // `Arc` in `self`, so the borrow cannot outlive or alias a
                // mutable view of it.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), bytes.len() / 2) }
            }
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.bits().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Encoded size in bytes.
    pub fn byte_len(&self) -> usize {
        self.len() * 2
    }

    /// Length of one contiguous row: the first GGUF dimension (input
    /// features of a linear weight, embedding width of a token table).
    pub fn row_len(&self) -> usize {
        self.dims.first().copied().unwrap_or(1)
    }

    /// Number of contiguous rows (output features, or vocabulary size).
    pub fn rows(&self) -> usize {
        self.dims.iter().skip(1).product()
    }

    /// Input features of a 2-D linear weight.
    pub fn in_features(&self) -> usize {
        self.row_len()
    }

    /// Output features of a 2-D linear weight.
    pub fn out_features(&self) -> usize {
        self.rows()
    }

    /// Widen row `row` into `dst` (`dst.len() == row_len()`).
    ///
    /// # Panics
    ///
    /// If `row` is out of range or `dst` has the wrong length.
    pub fn dequantize_row(&self, row: usize, dst: &mut [f32]) {
        let row_len = self.row_len();
        assert!(row < self.rows(), "HalfWeight: row {row} out of range");
        widen_into(
            self.dtype,
            &self.bits()[row * row_len..(row + 1) * row_len],
            dst,
        );
    }

    /// The f32 tensor the eager loader produced for this GGUF tensor: every
    /// value widened, shape = GGUF-native dims.
    pub fn to_f32_tensor(&self) -> CpuTensor {
        let bits = self.bits();
        let mut data = vec![0.0f32; bits.len()];
        widen_into(self.dtype, bits, &mut data);
        CpuTensor::from_data(self.dims.clone(), data)
    }
}

/// Scalar reference: the conversion the eager loader applied per element.
#[inline]
pub fn widen_scalar(dtype: HalfDtype, bits: u16) -> f32 {
    match dtype {
        HalfDtype::F16 => half::f16::from_bits(bits).to_f32(),
        HalfDtype::Bf16 => f32::from_bits(u32::from(bits) << 16),
    }
}

/// Widen `src` into `dst`, bit-identical to [`widen_scalar`] per element.
///
/// # Panics
///
/// If the slices differ in length.
#[inline]
pub fn widen_into(dtype: HalfDtype, src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "widen_into: length mismatch");
    match dtype {
        HalfDtype::F16 => src
            .reinterpret_cast::<half::f16>()
            .convert_to_f32_slice(dst),
        HalfDtype::Bf16 => {
            for (value, &bits) in dst.iter_mut().zip(src) {
                *value = f32::from_bits(u32::from(bits) << 16);
            }
        }
    }
}

thread_local! {
    /// Per-thread widened-row scratch, reused across calls so steady-state
    /// decode does not allocate.
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if scratch.len() < len {
            scratch.resize(len, 0.0);
        }
        f(&mut scratch[..len])
    })
}

/// Output features widened per `sgemm` call in [`half_matmul_into`]. The
/// block size never changes results (see the module docs); it trades the
/// per-call packing and dispatch cost against keeping the widened block
/// cache-resident.
const MATMUL_BLOCK_COLS: usize = 64;

/// Below this many multiply-adds a matmul runs on the calling thread.
const MATMUL_PARALLEL_MIN_MACS: usize = 1 << 20;

/// A raw output pointer shared by the column-block tasks of one matmul.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);

impl OutPtr {
    /// Taking `self` makes closures capture the whole (`Sync`) wrapper
    /// rather than the raw-pointer field.
    fn get(self) -> *mut f32 {
        self.0
    }
}
// SAFETY: `OutPtr` is only dereferenced (by sgemm) inside
// `half_matmul_into`, where each task writes a disjoint set of output
// columns and the owning `&mut [f32]` outlives every task.
unsafe impl Send for OutPtr {}
// SAFETY: as above; tasks never write the same element.
unsafe impl Sync for OutPtr {}

/// `out[m, n] = x[m, k] · Wᵀ` for a linear weight `w` with `in_features ==
/// k` and `out_features == n`.
///
/// Bit-identical to `CpuTensor::par_matmul` (and `matmul`) of `x` with the
/// eager path's row-major `[k, n]` f32 weight, i.e.
/// `try_gguf_to_row_major_f32(w.to_f32_tensor())`; see the module docs.
///
/// # Panics
///
/// If `w` is not 2-D or the slice lengths do not match `m`, `k` and `n`.
pub fn half_matmul_into(x: &[f32], m: usize, w: &HalfWeight, out: &mut [f32]) {
    assert_eq!(w.dims().len(), 2, "half_matmul_into: weight must be 2-D");
    let (k, n) = (w.in_features(), w.out_features());
    assert_eq!(Some(x.len()), m.checked_mul(k), "half_matmul_into: x shape");
    assert_eq!(
        Some(out.len()),
        m.checked_mul(n),
        "half_matmul_into: out shape"
    );
    if m == 0 || n == 0 {
        return;
    }
    let bits = w.bits();
    let dtype = w.dtype();
    let out_ptr = OutPtr(out.as_mut_ptr());
    let run_block = |block: usize| {
        let first = block * MATMUL_BLOCK_COLS;
        let cols = MATMUL_BLOCK_COLS.min(n - first);
        with_scratch(cols * k, |widened| {
            widen_into(dtype, &bits[first * k..(first + cols) * k], widened);
            // `widened` holds `cols` rows of `k` values: as the `k x cols`
            // operand B, element (p, j) is at `j * k + p`, so B's row stride
            // is 1 and its column stride `k`.
            // SAFETY: A is `x`, `m * k` contiguous f32 (row stride k, column
            // stride 1); B is `widened`, `cols * k` f32 with the strides
            // above; C starts at column `first` of the `m x n` row-major
            // `out` (row stride n, column stride 1), so sgemm writes exactly
            // out[r * n + first + j] for r < m, j < cols, which is in bounds
            // because first + cols <= n. Blocks partition the columns, so
            // concurrent tasks write disjoint elements, and none of them
            // overlaps `x` or the thread-local `widened`. beta = 0 means C
            // is not read.
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    cols,
                    1.0,
                    x.as_ptr(),
                    k as isize,
                    1,
                    widened.as_ptr(),
                    1,
                    k as isize,
                    0.0,
                    out_ptr.get().add(first),
                    n as isize,
                    1,
                );
            }
        });
    };
    let blocks = n.div_ceil(MATMUL_BLOCK_COLS);
    let macs = m.saturating_mul(k).saturating_mul(n);
    if blocks > 1 && macs >= MATMUL_PARALLEL_MIN_MACS && rayon::current_num_threads() > 1 {
        (0..blocks).into_par_iter().for_each(run_block);
    } else {
        (0..blocks).for_each(run_block);
    }
}

/// Output rows per group in [`half_matvec_sequential_into`].
const MATVEC_GROUP_ROWS: usize = 8;

/// Below this many weights the sequential matvec runs on the calling thread.
const MATVEC_PARALLEL_MIN_WEIGHTS: usize = 1 << 18;

/// Whether [`half_matvec_sequential_into`] splits its output groups across
/// the rayon pool for this weight (only when the caller allows it).
pub fn half_matvec_runs_parallel(w: &HalfWeight, parallel: bool) -> bool {
    parallel
        && w.len() >= MATVEC_PARALLEL_MIN_WEIGHTS
        && w.rows() > MATVEC_GROUP_ROWS
        && rayon::current_num_threads() > 1
}

/// The planned decode's dense matvec over a half weight: for each output
/// `j`, `acc = 0; for i in 0..k { acc += x[i] * w[j][i] }`, then
/// `dst[j] = acc` (or `dst[j] += acc` with `accumulate`).
///
/// Bit-identical to that loop over the eager f32 weight: each output keeps
/// its own left-to-right sum with a separate multiply and add (Rust never
/// contracts them into an FMA). Eight outputs are computed side by side
/// from a transposed widened block so the compiler can vectorize across
/// outputs without changing any output's order of operations.
///
/// # Panics
///
/// If `w` is not 2-D or the slice lengths do not match it.
/// With `parallel`, large weights split their output groups across the
/// rayon pool; each output is still one thread's sequential sum.
pub fn half_matvec_sequential_into(
    x: &[f32],
    w: &HalfWeight,
    dst: &mut [f32],
    accumulate: bool,
    parallel: bool,
) {
    assert_eq!(w.dims().len(), 2, "half_matvec: weight must be 2-D");
    let (k, n) = (w.in_features(), w.out_features());
    assert_eq!(x.len(), k, "half_matvec: x length");
    assert_eq!(dst.len(), n, "half_matvec: dst length");
    let bits = w.bits();
    let dtype = w.dtype();
    let run_group = |group: usize, dst: &mut [f32]| {
        let first = group * MATVEC_GROUP_ROWS;
        let rows = dst.len();
        with_scratch(2 * MATVEC_GROUP_ROWS * k, |scratch| {
            let (widened, transposed) = scratch.split_at_mut(MATVEC_GROUP_ROWS * k);
            widen_into(
                dtype,
                &bits[first * k..(first + rows) * k],
                &mut widened[..rows * k],
            );
            // Lanes past `rows` are padding: their sums are never stored.
            if rows < MATVEC_GROUP_ROWS {
                transposed.fill(0.0);
            }
            for (row, values) in widened[..rows * k].chunks_exact(k.max(1)).enumerate() {
                for (i, &value) in values.iter().enumerate() {
                    transposed[i * MATVEC_GROUP_ROWS + row] = value;
                }
            }
            let mut acc = [0.0f32; MATVEC_GROUP_ROWS];
            for (&xi, lanes) in x.iter().zip(transposed.chunks_exact(MATVEC_GROUP_ROWS)) {
                for lane in 0..MATVEC_GROUP_ROWS {
                    acc[lane] += xi * lanes[lane];
                }
            }
            for (value, sum) in dst.iter_mut().zip(acc) {
                if accumulate {
                    *value += sum;
                } else {
                    *value = sum;
                }
            }
        });
    };
    if half_matvec_runs_parallel(w, parallel) {
        dst.par_chunks_mut(MATVEC_GROUP_ROWS)
            .enumerate()
            .for_each(|(group, dst)| run_group(group, dst));
    } else {
        dst.chunks_mut(MATVEC_GROUP_ROWS)
            .enumerate()
            .for_each(|(group, dst)| run_group(group, dst));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift stream.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f32(&mut self) -> f32 {
            ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// Random finite weights with a sprinkling of subnormals and zeros of
    /// both signs, plus (when `specials`) NaN and infinities.
    fn random_bits(dtype: HalfDtype, len: usize, rng: &mut Rng, specials: bool) -> Vec<u16> {
        (0..len)
            .map(|_| {
                let roll = rng.next() % 64;
                match (dtype, roll) {
                    (_, 0) => 0x0000,
                    (_, 1) => 0x8000,
                    (HalfDtype::F16, 2) => (rng.next() % 0x3FF) as u16 + 1, // subnormal
                    (HalfDtype::F16, 3) => 0x8000 | ((rng.next() % 0x3FF) as u16 + 1),
                    (HalfDtype::Bf16, 2) => (rng.next() % 0x7F) as u16 + 1, // subnormal
                    (HalfDtype::F16, 4) if specials => 0x7C00,              // +inf
                    (HalfDtype::F16, 5) if specials => 0xFC00,              // -inf
                    (HalfDtype::F16, 6) if specials => 0x7E01,              // qNaN payload
                    (HalfDtype::F16, 7) if specials => 0x7C01,              // sNaN
                    (HalfDtype::Bf16, 4) if specials => 0x7F80,
                    (HalfDtype::Bf16, 5) if specials => 0xFF80,
                    (HalfDtype::Bf16, 6) if specials => 0x7FC1,
                    (HalfDtype::Bf16, 7) if specials => 0x7F81,
                    (HalfDtype::F16, _) => half::f16::from_f32(rng.f32() * 4.0).to_bits(),
                    (HalfDtype::Bf16, _) => half::bf16::from_f32(rng.f32() * 4.0).to_bits(),
                }
            })
            .collect()
    }

    /// The eager loader's tensor: scalar-widened values in GGUF dims.
    fn eager_tensor(dtype: HalfDtype, dims: &[usize], bits: &[u16]) -> CpuTensor {
        let data = bits.iter().map(|&b| widen_scalar(dtype, b)).collect();
        CpuTensor::from_data(dims.to_vec(), data)
    }

    /// The eager loader's row-major `[k, n]` matrix (GGUF dims `[k, n]`).
    fn eager_row_major(dtype: HalfDtype, k: usize, n: usize, bits: &[u16]) -> CpuTensor {
        crate::loader::try_gguf_to_row_major_f32(eager_tensor(dtype, &[k, n], bits)).unwrap()
    }

    fn assert_bits_eq(expected: &[f32], actual: &[f32], what: &str) {
        assert_eq!(expected.len(), actual.len(), "{what}: length");
        for (index, (e, a)) in expected.iter().zip(actual).enumerate() {
            assert_eq!(
                e.to_bits(),
                a.to_bits(),
                "{what}: element {index} differs ({e} vs {a})"
            );
        }
    }

    #[test]
    fn widen_matches_scalar_conversion_for_every_encoding() {
        let all: Vec<u16> = (0..=u16::MAX).collect();
        for dtype in [HalfDtype::F16, HalfDtype::Bf16] {
            let mut widened = vec![0.0f32; all.len()];
            widen_into(dtype, &all, &mut widened);
            for (&bits, value) in all.iter().zip(&widened) {
                assert_eq!(
                    value.to_bits(),
                    widen_scalar(dtype, bits).to_bits(),
                    "{dtype:?} {bits:#06x}"
                );
            }
            // Odd lengths and offsets exercise the vector tails.
            for start in 0..9 {
                for len in [0, 1, 3, 7, 9, 31, 33] {
                    let src = &all[start * 977..start * 977 + len];
                    let mut dst = vec![0.0f32; len];
                    widen_into(dtype, src, &mut dst);
                    let expected: Vec<f32> = src.iter().map(|&b| widen_scalar(dtype, b)).collect();
                    assert_bits_eq(&expected, &dst, "widen tail");
                }
            }
        }
    }

    #[test]
    fn f16_scalar_conversion_is_the_eager_loaders() {
        // The eager loader decoded F16 with half::f16::from_bits(..).to_f32()
        // and BF16 with a 16-bit shift; pin a few representative encodings.
        assert_eq!(widen_scalar(HalfDtype::F16, 0x3C00), 1.0);
        assert_eq!(widen_scalar(HalfDtype::F16, 0x0001), 2f32.powi(-24));
        assert_eq!(widen_scalar(HalfDtype::F16, 0x8000).to_bits(), 0x8000_0000);
        assert!(widen_scalar(HalfDtype::F16, 0x7C01).is_nan());
        assert_eq!(widen_scalar(HalfDtype::Bf16, 0x3F80), 1.0);
        assert_eq!(widen_scalar(HalfDtype::Bf16, 0x7F81).to_bits(), 0x7F81_0000);
    }

    /// Shapes covering: single row/column, sizes below/above the 64-column
    /// block and the 8-row group, k below/above sgemm's 256 KC block (and a
    /// non-multiple of it), and m across the decode, small-batch and
    /// parallel-prefill regimes.
    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (1, 7, 3),
        (1, 33, 65),
        (1, 256, 64),
        (1, 300, 130),
        (1, 513, 200),
        (2, 17, 9),
        (3, 64, 129),
        (5, 257, 63),
        (8, 129, 191),
        (17, 70, 1),
        (64, 96, 160),
        (65, 300, 77),
        (130, 64, 70),
    ];

    #[test]
    fn matmul_is_bit_identical_to_eager_par_matmul() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for dtype in [HalfDtype::F16, HalfDtype::Bf16] {
            for &(m, k, n) in SHAPES {
                for specials in [false, true] {
                    let bits = random_bits(dtype, k * n, &mut rng, specials);
                    let x: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
                    let weight = HalfWeight::from_bits(dtype, vec![k, n], bits.clone());
                    let eager = eager_row_major(dtype, k, n, &bits);
                    let x_tensor = CpuTensor::from_data(vec![m, k], x.clone());
                    let expected = x_tensor.par_matmul(&eager);
                    assert_bits_eq(expected.data(), x_tensor.matmul(&eager).data(), "serial");

                    let mut out = vec![f32::NAN; m * n];
                    half_matmul_into(&x, m, &weight, &mut out);
                    assert_bits_eq(
                        expected.data(),
                        &out,
                        &format!("{dtype:?} m={m} k={k} n={n} specials={specials}"),
                    );
                }
            }
        }
    }

    #[test]
    fn matmul_is_bit_identical_on_the_parallel_path() {
        // Large enough to cross MATMUL_PARALLEL_MIN_MACS in a 4-thread pool.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut rng = Rng(42);
        for (m, k, n) in [(1, 1100, 1000), (64, 300, 333)] {
            let bits = random_bits(HalfDtype::F16, k * n, &mut rng, false);
            let x: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
            let weight = HalfWeight::from_bits(HalfDtype::F16, vec![k, n], bits.clone());
            let eager = eager_row_major(HalfDtype::F16, k, n, &bits);
            let expected =
                pool.install(|| CpuTensor::from_data(vec![m, k], x.clone()).par_matmul(&eager));
            let mut out = vec![0.0f32; m * n];
            pool.install(|| half_matmul_into(&x, m, &weight, &mut out));
            assert_bits_eq(expected.data(), &out, "parallel");
        }
    }

    /// The planned decode's f32 matvec, verbatim.
    fn eager_sequential(x: &[f32], weight: &[f32], out_dim: usize) -> Vec<f32> {
        let mut dst = vec![0.0f32; out_dim];
        for (j, value) in dst.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (i, &x) in x.iter().enumerate() {
                acc += x * weight[i * out_dim + j];
            }
            *value = acc;
        }
        dst
    }

    #[test]
    fn sequential_matvec_is_bit_identical_to_planned_f32_loop() {
        let mut rng = Rng(7);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut shapes: Vec<(usize, usize)> = SHAPES.iter().map(|&(_, k, n)| (k, n)).collect();
        shapes.push((600, 500)); // crosses MATVEC_PARALLEL_MIN_WEIGHTS
        for dtype in [HalfDtype::F16, HalfDtype::Bf16] {
            for &(k, n) in &shapes {
                for specials in [false, true] {
                    let bits = random_bits(dtype, k * n, &mut rng, specials);
                    let x: Vec<f32> = (0..k).map(|_| rng.f32()).collect();
                    let weight = HalfWeight::from_bits(dtype, vec![k, n], bits.clone());
                    let eager = eager_row_major(dtype, k, n, &bits);
                    let expected = eager_sequential(&x, eager.data(), n);
                    let mut out = vec![f32::NAN; n];
                    pool.install(|| {
                        half_matvec_sequential_into(&x, &weight, &mut out, false, true)
                    });
                    assert_bits_eq(&expected, &out, &format!("{dtype:?} k={k} n={n}"));

                    // accumulate: dst[j] += acc
                    let base: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
                    let mut accumulated = base.clone();
                    half_matvec_sequential_into(&x, &weight, &mut accumulated, true, false);
                    let expected: Vec<f32> =
                        base.iter().zip(&expected).map(|(b, e)| b + e).collect();
                    assert_bits_eq(&expected, &accumulated, "accumulate");
                }
            }
        }
    }

    #[test]
    fn rows_and_tensor_match_the_eager_loader() {
        let mut rng = Rng(99);
        for dtype in [HalfDtype::F16, HalfDtype::Bf16] {
            let (embed, vocab) = (37, 11);
            let bits = random_bits(dtype, embed * vocab, &mut rng, true);
            let weight = HalfWeight::from_bits(dtype, vec![embed, vocab], bits.clone());
            let eager = eager_tensor(dtype, &[embed, vocab], &bits);
            assert_eq!(weight.to_f32_tensor().shape(), eager.shape());
            assert_bits_eq(eager.data(), weight.to_f32_tensor().data(), "tensor");
            let mut row = vec![0.0f32; embed];
            for token in 0..vocab {
                weight.dequantize_row(token, &mut row);
                assert_bits_eq(
                    &eager.data()[token * embed..(token + 1) * embed],
                    &row,
                    "row",
                );
            }
            // 1-D tensors (norms) keep their single dimension.
            let norm = HalfWeight::from_bits(dtype, vec![5], bits[..5].to_vec());
            assert_eq!(norm.to_f32_tensor().shape(), &[5]);
        }
    }
}
