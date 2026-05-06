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
