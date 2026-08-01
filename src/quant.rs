use crate::tensor::CpuTensor;
use anyhow::{bail, Result};
use half::f16;
use memmap2::Mmap;
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
    (n_floats / Q8_0_BLOCK_SIZE) * Q8_0_TYPE_SIZE
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
    dst.resize(n_blocks * Q8_0_TYPE_SIZE, 0);

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

/// dequantize a q8_0 block-compressed buffer into f32 values.
///
/// each block: 2-byte fp16 scale `d`, followed by 32 int8 quantized values `q`.
/// output: `dst[j] = (q[j] as f32) * d`.
#[inline]
pub fn dequantize_q8_0(src: &[u8], dst: &mut [f32]) -> Result<()> {
    if !src.len().is_multiple_of(Q8_0_TYPE_SIZE) {
        bail!(
            "q8_0 source length {} is not a multiple of block size {}",
            src.len(),
            Q8_0_TYPE_SIZE
        );
    }
    let n_blocks = src.len() / Q8_0_TYPE_SIZE;
    let expected_dst_len = n_blocks
        .checked_mul(Q8_0_BLOCK_SIZE)
        .ok_or_else(|| anyhow::anyhow!("q8_0 output length overflow"))?;
    if dst.len() != expected_dst_len {
        bail!(
            "q8_0 destination length {} does not match expected {}",
            dst.len(),
            expected_dst_len
        );
    }

    for i in 0..n_blocks {
        let block_start = i * Q8_0_TYPE_SIZE;
        let out_start = i * Q8_0_BLOCK_SIZE;

        let d_bits = u16::from_le_bytes(src[block_start..block_start + 2].try_into()?);
        let d = f16::from_bits(d_bits).to_f32();

        for j in 0..Q8_0_BLOCK_SIZE {
            let q = src[block_start + 2 + j] as i8;
            dst[out_start + j] = q as f32 * d;
        }
    }
    Ok(())
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
#[derive(Clone)]
enum QuantizedData {
    Owned(Arc<[u8]>),
    Mapped {
        mmap: Arc<Mmap>,
        range: Range<usize>,
    },
}

impl QuantizedData {
    #[inline]
    fn as_slice(&self) -> &[u8] {
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
    pub shape: Vec<usize>,
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
        Ok(Self { data, shape })
    }

    /// Raw Q8_0 storage.
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data.as_slice()
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

        for i in 0..out_features {
            let row_start = i * in_features;
            self.dequantize_row(i, &mut data[row_start..row_start + in_features]);
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
    pub quants: alloc::vec::Vec<u8>,
    /// Interleaved scales, grouped by stripe.
    pub scales: alloc::vec::Vec<u8>,
    /// Logical shape [out_features, in_features].
    pub shape: Vec<usize>,
    /// Blocks per row = in_features / 32.
    pub blocks_per_row: usize,
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
    pub data: alloc::vec::Vec<u8>,
    /// Logical shape `[out_features, in_features]`.
    pub shape: Vec<usize>,
    /// Blocks per row = `in_features / 32`.
    pub blocks_per_row: usize,
}

impl QuantizedWeightVnni {
    /// Repack a row-contiguous Q8_0 matrix without changing its encoded size.
    pub fn from_quantized(weight: &QuantizedWeight) -> Self {
        let out_features = weight.out_features();
        let in_features = weight.in_features();
        let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
        let output_tiles = out_features.div_ceil(VNNI_OUT_TILE);
        let mut data = vec![0_u8; output_tiles * blocks_per_row * VNNI_BLOCK_RECORD_SIZE];
        let source = weight.data();
        let source_row_size = blocks_per_row * Q8_0_TYPE_SIZE;

        for tile in 0..output_tiles {
            for block in 0..blocks_per_row {
                let record = (tile * blocks_per_row + block) * VNNI_BLOCK_RECORD_SIZE;
                let scales = record + VNNI_OUT_TILE * Q8_0_BLOCK_SIZE;

                for lane in 0..VNNI_OUT_TILE {
                    let row = tile * VNNI_OUT_TILE + lane;
                    if row >= out_features {
                        continue;
                    }
                    let source_block = row * source_row_size + block * Q8_0_TYPE_SIZE;
                    data[scales + lane * 2..scales + lane * 2 + 2]
                        .copy_from_slice(&source[source_block..source_block + 2]);

                    for group in 0..Q8_0_BLOCK_SIZE / 4 {
                        let destination = record + group * VNNI_OUT_TILE * 4 + lane * 4;
                        let source_quants = source_block + 2 + group * 4;
                        data[destination..destination + 4]
                            .copy_from_slice(&source[source_quants..source_quants + 4]);
                    }
                }
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
    fn quantized_weight_rejects_non_block_aligned_rows() {
        assert!(QuantizedWeight::try_new(vec![], vec![1, 31]).is_err());
    }

    #[test]
    fn dequantize_rejects_malformed_buffer_lengths() {
        assert!(dequantize_q8_0(&[0; Q8_0_TYPE_SIZE - 1], &mut [0.0; Q8_0_BLOCK_SIZE]).is_err());
        assert!(dequantize_q8_0(&[0; Q8_0_TYPE_SIZE], &mut [0.0; Q8_0_BLOCK_SIZE - 1]).is_err());
    }
}
