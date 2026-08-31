use crate::tensor::CpuTensor;
use anyhow::{bail, Result};
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Arc;

/// number of float elements per q8_0 quantization block
pub const Q8_0_BLOCK_SIZE: usize = 32;
/// total byte size of one q8_0 block (2 byte fp16 scale + 32 int8 values)
pub const Q8_0_TYPE_SIZE: usize = 34;

/// Precomputed per-row suffix norms for exact branch-and-bound argmax
/// decode on a Q8_0 weight.
///
/// For row `i` and in-block boundary `b`, `suffix[i][b]` is the L2 norm of
/// the row's *actual* weight values (quantized ints times block scale) from
/// in-block `b` onward. Cauchy-Schwarz then bounds the remaining contribution
/// of a partially-accumulated row, so rows that provably cannot beat the
/// running maximum are pruned without changing the argmax.
pub struct Q8TopkNorms {
    out_features: usize,
    in_blocks: usize,
    suffix: Vec<f32>,
}

impl Q8TopkNorms {
    /// Compute the suffix-norm table for a Q8_0 weight.
    pub fn compute(weight: &QuantizedWeight) -> Self {
        let out_features = weight.shape[0];
        let in_features = weight.shape[1];
        let in_blocks = in_features / Q8_0_BLOCK_SIZE;
        let bytes = weight.data();
        let mut suffix = vec![0.0f32; out_features * (in_blocks + 1)];
        for row in 0..out_features {
            let mut acc = 0.0f64;
            for b in (0..in_blocks).rev() {
                let offset = (row * in_blocks + b) * Q8_0_TYPE_SIZE;
                let scale = half::f16::from_bits(u16::from_le_bytes(
                    bytes[offset..offset + 2].try_into().unwrap(),
                ))
                .to_f32();
                let block = &bytes[offset + 2..offset + 2 + Q8_0_BLOCK_SIZE];
                let q_norm: f64 = block
                    .iter()
                    .map(|&v| f64::from(f32::from(i8::from_le_bytes([v]))).powi(2))
                    .sum();
                acc += f64::from(scale).powi(2) * q_norm;
                suffix[row * (in_blocks + 1) + b] = acc.sqrt() as f32;
            }
        }
        Self {
            out_features,
            in_blocks,
            suffix,
        }
    }

    #[inline]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    #[inline]
    pub fn in_blocks(&self) -> usize {
        self.in_blocks
    }

    /// Suffix norm of row `row` from in-block `b` onward.
    #[inline]
    pub fn suffix(&self, row: usize, b: usize) -> f32 {
        self.suffix[row * (self.in_blocks + 1) + b]
    }
}
/// Compute the encoded byte length for `n_floats` values.
///
/// Each 32-float block → 34 bytes (2B f16 scale + 32B i8 quants).
/// Panics if `n_floats` is not a multiple of 32.
#[inline]
pub fn q8_0_encoded_len(n_floats: usize) -> usize {
    assert!(
        n_floats.is_multiple_of(Q8_0_BLOCK_SIZE),
        "q8_0_encoded_len: n_floats ({}) must be a multiple of {}",
        n_floats,
        Q8_0_BLOCK_SIZE
    );
    (n_floats / Q8_0_BLOCK_SIZE)
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("q8_0 encoded length overflow")
}

/// Quantize one or more contiguous rows to llama.cpp-compatible Q8_0 blocks.
///
/// Each 32-value block stores an FP16 `amax / 127` scale followed by 32
/// rounded signed bytes. `dst` is resized and reused by decode callers.
pub fn quantize_q8_0_into(src: &[f32], dst: &mut Vec<u8>) {
    assert!(
        src.len().is_multiple_of(Q8_0_BLOCK_SIZE),
        "q8_0 input length must be a multiple of 32"
    );
    let n_blocks = src.len() / Q8_0_BLOCK_SIZE;
    let encoded_len = n_blocks
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("q8_0 encoded length overflow");
    dst.resize(encoded_len, 0);

    for block in 0..n_blocks {
        let src_start = block * Q8_0_BLOCK_SIZE;
        let values = &src[src_start..src_start + Q8_0_BLOCK_SIZE];
        let amax = values
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale = amax / 127.0;
        let inv_scale = if scale != 0.0 { scale.recip() } else { 0.0 };
        let dst_start = block * Q8_0_TYPE_SIZE;
        dst[dst_start..dst_start + 2]
            .copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        for (index, value) in values.iter().enumerate() {
            let quantized = (*value * inv_scale).round().clamp(-127.0, 127.0) as i8;
            dst[dst_start + 2 + index] = quantized as u8;
        }
    }
}

/// a q8_0 weight matrix kept in raw block-compressed form.
///
/// weights are never stored as f32 - `dequantize_row(j)` dequantizes
/// one output-feature column on demand during matmul.  this keeps the
/// in-memory footprint at the quantized size (~4x smaller than f32).
///
/// ## layout
///
/// the loader reverses gguf dims from `[in, out]` to `[out, in]` so
/// q8_0 blocks (which run along the in_features dimension) are
/// contiguous per output feature.  `shape[0]` is `out_features`,
/// `shape[1]` is `in_features`.
/// Whether load-time quantized-weight integrity verification is enabled
/// (`EMBER_VERIFY_QUANT=1|true|yes`). When set, every constructed quantized
/// weight is scanned once for layout and finite-scale violations before it
/// is used (EmberSEC Phase V). Off by default: the scan costs one pass over
/// the packed bytes, and compressed-resident Q8 loads deliberately avoid
/// touching file pages.
pub(crate) fn quant_verify_enabled() -> bool {
    matches!(
        std::env::var("EMBER_VERIFY_QUANT").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

#[derive(Clone)]
pub(crate) enum QuantizedData {
    Owned(Arc<[u8]>),
    Mapped {
        mmap: Arc<Mmap>,
        range: Range<usize>,
    },
}

impl QuantizedData {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(data) => data,
            Self::Mapped { mmap, range } => &mmap[range.clone()],
        }
    }
}

impl core::fmt::Debug for QuantizedData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Owned(data) => f.debug_struct("Owned").field("len", &data.len()).finish(),
            Self::Mapped { range, .. } => f
                .debug_struct("Mapped")
                .field("range", range)
                .finish_non_exhaustive(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct QuantizedWeight {
    /// raw q8_0 bytes: [block0_scale(2B) | block0_q(32B) | block1_scale(2B) | ...]
    ///
    /// File-loaded weights retain a shared mmap range instead of copying model
    /// bytes into anonymous memory. Cloning a weight is therefore constant-time
    /// and does not increase the model's resident allocation.
    data: QuantizedData,
    /// logical shape [out_features, in_features] (reversed from gguf dims)
    pub(crate) shape: Vec<usize>,
}

impl QuantizedWeight {
    /// create a quantized weight from raw q8_0 bytes and logical shape
    /// `[out_features, in_features]`.
    pub fn new(data: Vec<u8>, shape: Vec<usize>) -> Self {
        Self::try_new(data, shape).expect("invalid q8_0 weight")
    }

    /// fallible constructor for q8_0 weights loaded from external model files.
    pub fn try_new(data: Vec<u8>, shape: Vec<usize>) -> Result<Self> {
        Self::try_new_storage(QuantizedData::Owned(data.into()), shape)
    }

    pub(crate) fn try_from_mmap(
        mmap: Arc<Mmap>,
        range: Range<usize>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        if range.start > range.end || range.end > mmap.len() {
            bail!(
                "QuantizedWeight: mmap range {:?} exceeds mapping length {}",
                range,
                mmap.len()
            );
        }
        Self::try_new_storage(QuantizedData::Mapped { mmap, range }, shape)
    }

    fn try_new_storage(data: QuantizedData, shape: Vec<usize>) -> Result<Self> {
        if shape.len() != 2 {
            bail!("QuantizedWeight: expected 2D shape, got {:?}", shape);
        }
        if shape.contains(&0) {
            bail!(
                "QuantizedWeight: dimensions must be non-zero, got {:?}",
                shape
            );
        }
        if !shape[1].is_multiple_of(Q8_0_BLOCK_SIZE) {
            bail!(
                "QuantizedWeight: in_features ({}) must be a multiple of {}",
                shape[1],
                Q8_0_BLOCK_SIZE
            );
        }
        let expected_elements = shape[0]
            .checked_mul(shape[1])
            .ok_or_else(|| anyhow::anyhow!("QuantizedWeight: shape product overflow"))?;
        let expected_blocks = expected_elements / Q8_0_BLOCK_SIZE;
        let expected_len = expected_blocks
            .checked_mul(Q8_0_TYPE_SIZE)
            .ok_or_else(|| anyhow::anyhow!("QuantizedWeight: byte length overflow"))?;
        if data.as_slice().len() != expected_len {
            bail!(
                "QuantizedWeight: data len ({}) != expected ({})",
                data.as_slice().len(),
                expected_len
            );
        }
        let weight = Self { data, shape };
        if quant_verify_enabled() {
            weight
                .validate_integrity()
                .map_err(|e| anyhow::anyhow!("quant integrity check failed: {e}"))?;
        }
        Ok(weight)
    }

    /// Raw Q8_0 storage.
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Mutable access to the packed bytes for owned weights (fault-injection
    /// harnesses, integrity tooling). Returns `None` for file-mapped weights,
    /// whose bytes must stay read-only.
    pub fn data_mut(&mut self) -> Option<&mut [u8]> {
        match &mut self.data {
            QuantizedData::Owned(data) => Some(Arc::make_mut(data)),
            QuantizedData::Mapped { .. } => None,
        }
    }

    /// Content-level integrity check: block-layout math and every block's
    /// f16 scale must be finite. A NaN/Inf scale (e.g. from a single
    /// corrupted scale byte) would otherwise propagate non-finite logits
    /// into sampling, where the sampler's argmax asserts on NaN.
    pub fn validate_integrity(&self) -> Result<(), String> {
        let expected_elements = self
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| "q8_0 shape product overflow".to_string())?;
        let blocks = expected_elements / Q8_0_BLOCK_SIZE;
        let expected_len = blocks
            .checked_mul(Q8_0_TYPE_SIZE)
            .ok_or_else(|| "q8_0 byte length overflow".to_string())?;
        if self.byte_len() != expected_len {
            return Err(format!(
                "q8_0 data len {} != expected {expected_len}",
                self.byte_len()
            ));
        }
        let data = self.data();
        for block in 0..blocks {
            let off = block * Q8_0_TYPE_SIZE;
            let scale = f16::from_bits(u16::from_le_bytes([data[off], data[off + 1]])).to_f32();
            if !scale.is_finite() {
                return Err(format!("q8_0 block {block} has non-finite scale {scale}"));
            }
        }
        Ok(())
    }

    /// Compressed byte size of this weight.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.data.as_slice().len()
    }

    /// Whether this weight directly references a read-only model-file mapping.
    #[inline]
    pub fn is_mapped(&self) -> bool {
        matches!(&self.data, QuantizedData::Mapped { .. })
    }

    /// Drop resident file-backed pages after constructing an alternate layout.
    ///
    /// # Safety
    ///
    /// The caller must ensure no slices into this mapping are live while
    /// `MADV_DONTNEED` executes. The weight remains valid and will fault its
    /// original pages back from the GGUF file if a generic path uses it later.
    #[cfg(unix)]
    pub(crate) unsafe fn evict_mapped_pages(&self) -> std::io::Result<bool> {
        if let QuantizedData::Mapped { mmap, range } = &self.data {
            // Safety: delegated to the caller above. Model construction invokes
            // this only after repacking has returned and before inference can
            // share or borrow the model.
            unsafe {
                mmap.unchecked_advise_range(
                    memmap2::UncheckedAdvice::DontNeed,
                    range.start,
                    range.len(),
                )
            }?;
            return Ok(true);
        }
        Ok(false)
    }

    /// dequantize one output-feature column into `dst`.
    ///
    /// `dst` must have length `in_features` (= `shape[1]`).
    /// output feature `j` occupies `in_features / 32` consecutive blocks
    /// starting at byte offset `j * blocks_per_col * 34`.
    #[inline]
    pub fn dequantize_row(&self, row: usize, dst: &mut [f32]) {
        self.validate_row_bounds(row)
            .expect("q8_0 row out of bounds");
        assert_eq!(
            dst.len(),
            self.shape[1],
            "dequantize_row destination len ({}) != in_features ({})",
            dst.len(),
            self.shape[1]
        );
        let in_features = self.shape[1];
        let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
        let row_start = row * blocks_per_row;

        crate::simd::dequantize_q8_0_row(self.data(), row_start, blocks_per_row, dst);
    }

    /// fully dequantize to a f32 `CpuTensor` with shape `[out_features, in_features]`.
    ///
    /// data is column-major (contiguous per output feature).
    /// transpose the result if you need row-major `[in_features, out_features]`.
    pub fn dequantize_all(&self) -> CpuTensor {
        let n_elements: usize = self.shape.iter().product();
        let mut data = vec![0.0f32; n_elements];
        let in_features = self.shape[1];
        let out_features = self.shape[0];

        // Parallel across rows: each row writes a disjoint slice, so results
        // are bit-identical to the serial loop. Small tensors stay serial.
        if in_features > 0 && out_features >= 16 {
            data.par_chunks_exact_mut(in_features)
                .enumerate()
                .for_each(|(i, row_dst)| self.dequantize_row(i, row_dst));
        } else {
            for i in 0..out_features {
                let row_start = i * in_features;
                self.dequantize_row(i, &mut data[row_start..row_start + in_features]);
            }
        }
        CpuTensor::from_data(self.shape.clone(), data)
    }

    /// number of output features (first dimension, `shape[0]`).
    pub fn out_features(&self) -> usize {
        self.shape[0]
    }

    /// number of input features (second dimension, `shape[1]`).
    pub fn in_features(&self) -> usize {
        self.shape[1]
    }

    fn validate_row_bounds(&self, row: usize) -> Result<()> {
        if row >= self.shape[0] {
            bail!(
                "QuantizedWeight: row {} out of bounds for {} rows",
                row,
                self.shape[0]
            );
        }
        Ok(())
    }
}

/// Interleaved Q8_0 layout: 4 consecutive output rows' blocks are stored
/// together so that a single cache-line load serves multiple rows. Quants
/// and scales are split into separate contiguous arrays for SIMD-friendly access.
///
/// For each stripe of `INTERLEAVE` rows, block b is stored as:
///   quants: [row0_b(32B) | row1_b(32B) | row2_b(32B) | row3_b(32B)]
///   scales: [row0_b(2B)  | row1_b(2B)  | row2_b(2B)  | row3_b(2B)]
///
/// Total stripe size = blocks_per_row × (INTERLEAVE × 32 + INTERLEAVE × 2) bytes.
/// Total size matches the original row-contiguous layout exactly.
pub const INTERLEAVE: usize = 4;

#[derive(Clone, Debug)]
pub struct QuantizedWeightInterleaved {
    /// Interleaved quants, grouped by stripe.
    pub(crate) quants: alloc::vec::Vec<u8>,
    /// Interleaved scales, grouped by stripe.
    pub(crate) scales: alloc::vec::Vec<u8>,
    /// Logical shape [out_features, in_features].
    pub(crate) shape: Vec<usize>,
    /// Blocks per row = in_features / 32.
    pub(crate) blocks_per_row: usize,
}

impl QuantizedWeightInterleaved {
    /// Repack a row-contiguous `QuantizedWeight` into the interleaved layout.
    ///
    /// This is a one-time cost at model load. The interleaved layout enables
    /// the VNNI kernel to process 4 output rows simultaneously with contiguous
    /// weight reads, reducing DRAM transactions by ~2-3× for large matrices.
    pub fn from_quantized(w: &QuantizedWeight) -> Self {
        let out_features = w.out_features();
        let in_features = w.in_features();
        let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
        let data = w.data();

        let stripes = out_features.div_ceil(INTERLEAVE);
        let quants_per_stripe = blocks_per_row * INTERLEAVE * Q8_0_BLOCK_SIZE;
        let scales_per_stripe = blocks_per_row * INTERLEAVE * 2; // 2 bytes per f16

        let mut quants = vec![0u8; stripes * quants_per_stripe];
        let mut scales = vec![0u8; stripes * scales_per_stripe];

        let row_bytes = blocks_per_row * Q8_0_TYPE_SIZE;

        for stripe in 0..stripes {
            let row_base = stripe * INTERLEAVE;
            let q_base = stripe * quants_per_stripe;
            let s_base = stripe * scales_per_stripe;

            for b in 0..blocks_per_row {
                let q_block_off = q_base + b * INTERLEAVE * Q8_0_BLOCK_SIZE;
                let s_block_off = s_base + b * INTERLEAVE * 2;

                for lane in 0..INTERLEAVE {
                    let row = row_base + lane;
                    if row >= out_features {
                        // Pad with zeros for incomplete final stripe
                        continue;
                    }
                    let src = row * row_bytes + b * Q8_0_TYPE_SIZE;
                    // Copy quants (skip 2-byte scale prefix in source)
                    let q_dst = q_block_off + lane * Q8_0_BLOCK_SIZE;
                    quants[q_dst..q_dst + Q8_0_BLOCK_SIZE]
                        .copy_from_slice(&data[src + 2..src + 2 + Q8_0_BLOCK_SIZE]);
                    // Copy scale
                    let s_dst = s_block_off + lane * 2;
                    scales[s_dst..s_dst + 2].copy_from_slice(&data[src..src + 2]);
                }
            }
        }

        Self {
            quants,
            scales,
            shape: vec![out_features, in_features],
            blocks_per_row,
        }
    }

    #[inline]
    pub fn out_features(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    pub fn in_features(&self) -> usize {
        self.shape[1]
    }
}

/// Q8_0 weights packed for a batch-1 AVX-512 VNNI matrix-vector kernel.
///
/// Output rows are grouped in tiles of 16. Within each input block, weights
/// are transposed in groups of four input bytes:
///
/// ```text
/// [k0..k3 for row 0, k0..k3 for row 1, ... k0..k3 for row 15]
/// ```
///
/// Eight such 64-byte groups cover one Q8_0 block. The 16 FP16 scales follow
/// the quants. A complete tile/block record is therefore 512 + 32 = 544
/// bytes, exactly the same size as 16 row-contiguous Q8_0 blocks.
pub const VNNI_OUT_TILE: usize = 16;
pub const VNNI_BLOCK_RECORD_SIZE: usize =
    VNNI_OUT_TILE * (Q8_0_BLOCK_SIZE + core::mem::size_of::<u16>());

#[derive(Clone, Debug)]
pub struct QuantizedWeightVnni {
    /// Tile-major packed records described above.
    pub(crate) data: alloc::vec::Vec<u8>,
    /// Logical shape `[out_features, in_features]`.
    pub(crate) shape: Vec<usize>,
    /// Blocks per row = `in_features / 32`.
    pub(crate) blocks_per_row: usize,
}

impl QuantizedWeightVnni {
    /// Repack a row-contiguous Q8_0 matrix without changing its encoded size.
    ///
    /// Set `EMBER_PARALLEL_REPACK=1` to parallelize warm-cache startup; the
    /// default stays sequential to avoid random cold mmap faults.
    pub fn from_quantized(weight: &QuantizedWeight) -> Self {
        let parallel_repack =
            std::env::var_os("EMBER_PARALLEL_REPACK").is_some_and(|value| value == "1");
        Self::from_quantized_with_mode(weight, parallel_repack)
    }

    fn from_quantized_with_mode(weight: &QuantizedWeight, parallel_repack: bool) -> Self {
        let out_features = weight.out_features();
        let in_features = weight.in_features();
        let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
        let output_tiles = out_features.div_ceil(VNNI_OUT_TILE);
        let tile_bytes = blocks_per_row * VNNI_BLOCK_RECORD_SIZE;
        let mut data = vec![0_u8; output_tiles * tile_bytes];
        let source = weight.data();
        let source_row_size = blocks_per_row * Q8_0_TYPE_SIZE;
        const TILES_PER_TASK: usize = 16;
        let task_bytes = tile_bytes * TILES_PER_TASK;
        let pack_task = |task: usize, task_data: &mut [u8]| {
            let first_tile = task * TILES_PER_TASK;
            let tiles = task_data.len() / tile_bytes;
            for tile_offset in 0..tiles {
                let tile = first_tile + tile_offset;
                let tile_data =
                    &mut task_data[tile_offset * tile_bytes..(tile_offset + 1) * tile_bytes];
                for block in 0..blocks_per_row {
                    let record = block * VNNI_BLOCK_RECORD_SIZE;
                    let scales = record + VNNI_OUT_TILE * Q8_0_BLOCK_SIZE;

                    for lane in 0..VNNI_OUT_TILE {
                        let row = tile * VNNI_OUT_TILE + lane;
                        if row >= out_features {
                            continue;
                        }
                        let source_block = row * source_row_size + block * Q8_0_TYPE_SIZE;
                        tile_data[scales + lane * 2..scales + lane * 2 + 2]
                            .copy_from_slice(&source[source_block..source_block + 2]);

                        for group in 0..Q8_0_BLOCK_SIZE / 4 {
                            let destination = record + group * VNNI_OUT_TILE * 4 + lane * 4;
                            let source_quants = source_block + 2 + group * 4;
                            tile_data[destination..destination + 4]
                                .copy_from_slice(&source[source_quants..source_quants + 4]);
                        }
                    }
                }
            }
        };

        // Parallel tile repacking improves warm startup, but parallel page
        // faults can hurt cold file-backed startup. Keep the safe default
        // sequential; deployments with warm page cache may opt in.
        if parallel_repack {
            data.par_chunks_mut(task_bytes)
                .enumerate()
                .for_each(|(task, task_data)| pack_task(task, task_data));
        } else {
            for (task, task_data) in data.chunks_mut(task_bytes).enumerate() {
                pack_task(task, task_data);
            }
        }

        Self {
            data,
            shape: vec![out_features, in_features],
            blocks_per_row,
        }
    }

    #[inline]
    pub fn out_features(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    pub fn in_features(&self) -> usize {
        self.shape[1]
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_q8_0_zero_block_matches_reference_layout() {
        let mut encoded = Vec::new();
        quantize_q8_0_into(&[0.0; Q8_0_BLOCK_SIZE], &mut encoded);
        assert_eq!(encoded, vec![0; Q8_0_TYPE_SIZE]);
    }

    #[test]
    fn quantize_q8_0_matches_llama_reference_vector() {
        let values = (0..Q8_0_BLOCK_SIZE)
            .map(|index| index as f32 - 16.0)
            .collect::<Vec<_>>();
        let mut encoded = Vec::new();
        quantize_q8_0_into(&values, &mut encoded);

        let expected_scale = f16::from_f32(16.0 / 127.0);
        assert_eq!(
            &encoded[..2],
            &expected_scale.to_bits().to_le_bytes(),
            "Q8_0 scale must be stored as little-endian FP16"
        );
        let scale = (16.0f32 / 127.0).recip();
        for (index, value) in values.iter().enumerate() {
            let expected = (*value * scale).round() as i8;
            assert_eq!(encoded[2 + index] as i8, expected, "index {index}");
        }
    }

    #[test]
    fn parallel_vnni_repack_is_byte_identical_to_sequential() {
        let weight = QuantizedWeight::new(
            vec![0x5a; 4 * 64 / Q8_0_BLOCK_SIZE * Q8_0_TYPE_SIZE],
            vec![4, 64],
        );
        let sequential = QuantizedWeightVnni::from_quantized_with_mode(&weight, false);
        let parallel = QuantizedWeightVnni::from_quantized_with_mode(&weight, true);
        assert_eq!(parallel.data, sequential.data);
        assert_eq!(parallel.shape, sequential.shape);
        assert_eq!(parallel.blocks_per_row, sequential.blocks_per_row);
    }

    #[test]
    fn quantized_weight_rejects_non_block_aligned_rows() {
        assert!(QuantizedWeight::try_new(vec![], vec![1, 31]).is_err());
    }
}
