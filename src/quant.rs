use crate::tensor::CpuTensor;
use anyhow::Result;
use half::f16;

/// number of float elements per q8_0 quantization block
pub const Q8_0_BLOCK_SIZE: usize = 32;
/// total byte size of one q8_0 block (2 byte fp16 scale + 32 int8 values)
pub const Q8_0_TYPE_SIZE: usize = 34;

/// dequantize a q8_0 block-compressed buffer into f32 values.
///
/// each block: 2-byte fp16 scale `d`, followed by 32 int8 quantized values `q`.
/// output: `dst[j] = (q[j] as f32) * d`.
#[inline]
pub fn dequantize_q8_0(src: &[u8], dst: &mut [f32]) -> Result<()> {
    let n_blocks = src.len() / Q8_0_TYPE_SIZE;

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
#[derive(Clone, Debug)]
pub struct QuantizedWeight {
    /// raw q8_0 bytes: [block0_scale(2B) | block0_q(32B) | block1_scale(2B) | ...]
    pub data: Vec<u8>,
    /// logical shape [out_features, in_features] (reversed from gguf dims)
    pub shape: Vec<usize>,
}

impl QuantizedWeight {
    /// create a quantized weight from raw q8_0 bytes and logical shape
    /// `[out_features, in_features]`.
    ///
    /// # panics
    /// panics if `shape[1]` (in_features) is not a multiple of 32.
    pub fn new(data: Vec<u8>, shape: Vec<usize>) -> Self {
        assert_eq!(
            shape[1] % Q8_0_BLOCK_SIZE,
            0,
            "QuantizedWeight: in_features ({}) must be a multiple of {}",
            shape[1],
            Q8_0_BLOCK_SIZE
        );
        let expected_blocks = shape[0] * shape[1] / Q8_0_BLOCK_SIZE;
        assert_eq!(
            data.len(),
            expected_blocks * Q8_0_TYPE_SIZE,
            "QuantizedWeight: data len ({}) != expected ({})",
            data.len(),
            expected_blocks * Q8_0_TYPE_SIZE
        );
        Self { data, shape }
    }

    /// dequantize one output-feature column into `dst`.
    ///
    /// `dst` must have length `in_features` (= `shape[1]`).
    /// output feature `j` occupies `in_features / 32` consecutive blocks
    /// starting at byte offset `j * blocks_per_col * 34`.
    #[inline]
    pub fn dequantize_row(&self, row: usize, dst: &mut [f32]) {
        let in_features = self.shape[1];
        let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
        let row_start = row * blocks_per_row;

        for b in 0..blocks_per_row {
            let byte_offset = (row_start + b) * Q8_0_TYPE_SIZE;

            let d_bits =
                u16::from_le_bytes(self.data[byte_offset..byte_offset + 2].try_into().unwrap());
            let d = f16::from_bits(d_bits).to_f32();

            let out_offset = b * Q8_0_BLOCK_SIZE;
            for j in 0..Q8_0_BLOCK_SIZE {
                let q = self.data[byte_offset + 2 + j] as i8;
                dst[out_offset + j] = q as f32 * d;
            }
        }
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
}
