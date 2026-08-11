//! K-family matmul routing and exact-f32 reference implementation.
//!
//! Production execution uses the format's canonical Q8_K activation packing
//! and integer dot products in [`crate::k_quant_matmul`].  The scalar
//! dequantize-and-dot path remains here as an explicit correctness oracle; it
//! is not used to dictate production numerical semantics.

use crate::quant_k::{
    dequant_q4_k, dequant_q6_k, KQuantDtype, KQuantWeight, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, QK_K,
};

type DequantizeBlock = fn(&[u8], &mut [f32]);

thread_local! {
    static BLOCK_SCRATCH: std::cell::RefCell<[f32; QK_K]> =
        const { std::cell::RefCell::new([0.0; QK_K]) };
}

fn validate(
    label: &str,
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &[f32],
) -> Result<(), String> {
    let expected_src = rows
        .checked_mul(w.in_features())
        .ok_or_else(|| format!("{label}: input shape product overflow"))?;
    if src.len() != expected_src {
        return Err(format!(
            "{label}: src len {} != rows {rows} * in_features {}",
            src.len(),
            w.in_features()
        ));
    }
    let expected_dst = rows
        .checked_mul(w.out_features())
        .ok_or_else(|| format!("{label}: output shape product overflow"))?;
    if dst.len() != expected_dst {
        return Err(format!(
            "{label}: dst len {} != rows {rows} * out_features {}",
            dst.len(),
            w.out_features()
        ));
    }
    Ok(())
}

/// Slow exact-f32 oracle: dequantize one weight super-block at a time and
/// accumulate against the original f32 activation rows.
pub fn matmul_k_scalar_into(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    validate("matmul_k_scalar", src, rows, w, dst)?;
    let (dequant, block_bytes): (DequantizeBlock, usize) = match w.dtype() {
        KQuantDtype::Q4K => (dequant_q4_k, Q4_K_BLOCK_BYTES),
        KQuantDtype::Q6K => (dequant_q6_k, Q6_K_BLOCK_BYTES),
    };
    let input_features = w.in_features();
    let output_features = w.out_features();
    let blocks_per_row = w.blocks_per_row();
    let data = w.data();

    for column in 0..output_features {
        let weight_row = column * blocks_per_row * block_bytes;
        for block_index in 0..blocks_per_row {
            let start = weight_row + block_index * block_bytes;
            let block = &data[start..start + block_bytes];
            BLOCK_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                dequant(block, &mut scratch[..]);
                let input_block = block_index * QK_K;
                for row in 0..rows {
                    let input_start = row * input_features + input_block;
                    let mut sum = 0.0f32;
                    for index in 0..QK_K {
                        sum += src[input_start + index] * scratch[index];
                    }
                    dst[row * output_features + column] += sum;
                }
            });
        }
    }
    Ok(())
}

/// Production K-quant matmul using Q8_K-packed activations and integer dots.
pub fn matmul_k_into(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    crate::k_quant_matmul::matmul_k_q8_into(src, rows, w, dst, false)
}

/// Parallel production K-quant matmul.  Decode and prefill share the same
/// primitive; only the scheduling decision changes.
pub fn matmul_k_into_parallel(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    crate::k_quant_matmul::matmul_k_q8_into(src, rows, w, dst, true)
}

/// Benchmark hook for the slow exact-f32 oracle.
#[doc(hidden)]
pub fn bench_exact_f32(src: &[f32], w: &KQuantWeight, dst: &mut [f32]) {
    let rows = src.len() / w.in_features();
    matmul_k_scalar_into(src, rows, w, dst).expect("valid benchmark shape");
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tensor::CpuTensor;

    /// Deterministic pseudo-random Q6_K payload with finite f16 scales.
    pub(crate) fn seeded_q6_blocks(blocks: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut bytes = vec![0u8; blocks * Q6_K_BLOCK_BYTES];
        for byte in &mut bytes {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (state >> 33) as u8;
        }
        for block in bytes.chunks_exact_mut(Q6_K_BLOCK_BYTES) {
            let bits = u16::from_le_bytes([block[208], block[209]]) & 0x7fff;
            let finite = if bits >= 0x7c00 { 0x3c00 } else { bits };
            block[208..210].copy_from_slice(&finite.to_le_bytes());
        }
        bytes
    }

    /// Deterministic pseudo-random Q4_K payload with finite f16 scales.
    pub(crate) fn seeded_q4_blocks(blocks: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut bytes = vec![0u8; blocks * Q4_K_BLOCK_BYTES];
        for byte in &mut bytes {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (state >> 33) as u8;
        }
        for block in bytes.chunks_exact_mut(Q4_K_BLOCK_BYTES) {
            for offset in [0, 2] {
                let bits = u16::from_le_bytes([block[offset], block[offset + 1]]) & 0x7fff;
                let finite = if bits >= 0x7c00 { 0x3c00 } else { bits };
                block[offset..offset + 2].copy_from_slice(&finite.to_le_bytes());
            }
        }
        bytes
    }

    /// Deterministic activations in [-4, 4).
    pub(crate) fn seeded_activations(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let value = (state >> 33) as u32 as i32;
            values.push(value as f32 / i32::MAX as f32 * 4.0);
        }
        values
    }

    fn eager_matmul(src: &[f32], rows: usize, w: &KQuantWeight) -> Vec<f32> {
        let full = w.dequantize_all().transpose();
        CpuTensor::from_data(vec![rows, w.in_features()], src.to_vec())
            .matmul(&full)
            .data()
            .to_vec()
    }

    #[test]
    fn exact_oracle_matches_eager_dequant_for_q4_and_q6() {
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let rows = 3;
            let input = 512;
            let output = 5;
            let blocks = output * input / QK_K;
            let bytes = match dtype {
                KQuantDtype::Q4K => seeded_q4_blocks(blocks, 1),
                KQuantDtype::Q6K => seeded_q6_blocks(blocks, 2),
            };
            let w = KQuantWeight::try_new(bytes, [output, input], dtype).unwrap();
            let src = seeded_activations(rows * input, 3);
            let expected = eager_matmul(&src, rows, &w);
            let mut actual = vec![0.0; rows * output];
            matmul_k_scalar_into(&src, rows, &w, &mut actual).unwrap();
            for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                let scale = b.abs().max(1.0);
                assert!(
                    (a - b).abs() <= 1e-4 * scale,
                    "{dtype:?} index {index}: {a} != {b}"
                );
            }
        }
    }

    #[test]
    fn production_serial_and_parallel_match() {
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let input = 512;
            let output = 513;
            let blocks = output * input / QK_K;
            let bytes = match dtype {
                KQuantDtype::Q4K => seeded_q4_blocks(blocks, 4),
                KQuantDtype::Q6K => seeded_q6_blocks(blocks, 5),
            };
            let w = KQuantWeight::try_new(bytes, [output, input], dtype).unwrap();
            let src = seeded_activations(input, 6);
            let mut serial = vec![0.0; output];
            let mut parallel = vec![0.0; output];
            matmul_k_into(&src, 1, &w, &mut serial).unwrap();
            matmul_k_into_parallel(&src, 1, &w, &mut parallel).unwrap();
            assert_eq!(serial, parallel);
        }
    }

    #[test]
    fn length_mismatches_are_rejected() {
        let w = KQuantWeight::try_new(seeded_q4_blocks(2, 9), [2, 256], KQuantDtype::Q4K).unwrap();
        let mut dst = [0.0; 2];
        assert!(matmul_k_scalar_into(&[0.0; 255], 1, &w, &mut dst).is_err());
        assert!(matmul_k_into(&[0.0; 256], 1, &w, &mut dst[..1]).is_err());
    }
}
