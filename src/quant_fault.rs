//! EmberSEC Phase V: quantized-inference fault sensitivity and integrity.
//!
//! A single corrupted byte inside a packed quantized weight can be:
//! - **bounded**: a payload-bit flip changes a few dequantized values by one
//!   quantization step (logit drift proportional to the block scale);
//! - **severe**: a scale/header-bit flip can turn an f16 scale into NaN or
//!   Inf, propagating non-finite logits into sampling — where the sampler's
//!   argmax asserts on NaN (crash) and the CLI's logit validation bails.
//!
//! This module provides the hermetic, deterministic harness for measuring
//! that fault surface (`inject_bit_flip`, `measure_impact`, `k_decode`,
//! `q8_decode`) plus the integrity checks on [`crate::quant`] /
//! [`crate::quant_k`] weights. It runs on synthetic seeded blocks — no model
//! files, no GPU, no 8B-host requirements.

use crate::k_quant_matmul::matmul_k_q8_into;
use crate::quant::{quantize_q8_0_into, QuantizedWeight};
use crate::quant_k::KQuantWeight;
use crate::simd::matmul_q8_0_decode;

/// One single-bit fault at a byte offset inside one block of a weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitFault {
    /// Block index (Q4_K: 144 B/block, Q6_K: 210 B/block, Q8_0: 34 B/block).
    pub block: usize,
    /// Byte offset within the block.
    pub byte: usize,
    /// Bit index 0..7 within the byte.
    pub bit: u8,
}

/// Impact of a faulted weight on one decode pass, relative to pristine.
#[derive(Debug, Clone, Copy)]
pub struct FaultImpact {
    /// Max absolute logit difference over the output row.
    pub max_abs_logit_diff: f32,
    /// Relative L2 of the logit difference (norm of pristine).
    pub rel_l2: f32,
    /// Whether the argmax token changed (both sides finite).
    pub top1_flipped: bool,
    /// Whether every faulted logit stayed finite.
    pub logits_finite: bool,
}

/// Flip one bit in a byte slice (the only mutation primitive the harness
/// uses). Bounds-checked: out-of-range offsets are errors, not panics.
pub fn inject_bit_flip(data: &mut [u8], byte: usize, bit: u8) -> Result<(), String> {
    if byte >= data.len() {
        return Err(format!(
            "inject_bit_flip: byte {byte} out of range {}",
            data.len()
        ));
    }
    if bit >= 8 {
        return Err(format!("inject_bit_flip: bit {bit} out of range 0..8"));
    }
    data[byte] ^= 1u8 << bit;
    Ok(())
}

/// Compare pristine vs faulted logits.
pub fn measure_impact(pristine: &[f32], faulted: &[f32]) -> FaultImpact {
    debug_assert_eq!(pristine.len(), faulted.len());
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut norm_sq = 0.0f32;
    let mut logits_finite = true;
    for (p, f) in pristine.iter().zip(faulted.iter()) {
        if !f.is_finite() {
            logits_finite = false;
        }
        let d = (f - p).abs();
        max_abs = max_abs.max(d);
        sum_sq += d * d;
        norm_sq += p * p;
    }
    let rel_l2 = (sum_sq / norm_sq.max(f32::MIN_POSITIVE)).sqrt();
    let top1_flipped = logits_finite && argmax(pristine) != argmax(faulted);
    FaultImpact {
        max_abs_logit_diff: max_abs,
        rel_l2,
        top1_flipped,
        logits_finite,
    }
}

/// One decode pass over a K-quant weight (dst += src x dequant(w), as the
/// inference path uses it). `rows` activation rows, single output row set.
pub fn k_decode(w: &KQuantWeight, src: &[f32], parallel: bool) -> Result<Vec<f32>, String> {
    let mut dst = vec![0.0f32; w.out_features()];
    matmul_k_q8_into(src, 1, w, &mut dst, parallel)?;
    Ok(dst)
}

/// One decode pass over a Q8_0 weight (single activation row).
pub fn q8_decode(w: &QuantizedWeight, src: &[f32]) -> Result<Vec<f32>, String> {
    if src.len() != w.in_features() {
        return Err(format!(
            "q8_decode: src len {} != in_features {}",
            src.len(),
            w.in_features()
        ));
    }
    let mut x = Vec::new();
    quantize_q8_0_into(src, &mut x);
    let mut out = vec![0.0f32; w.out_features()];
    matmul_q8_0_decode(&x, w, &mut out);
    Ok(out)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
            if v > best_v {
                (i, v)
            } else {
                (best_i, best_v)
            }
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k_matmul::tests::{seeded_activations, seeded_q4_blocks, seeded_q6_blocks};
    use crate::quant_k::{KQuantDtype, QK_K};

    /// Deterministic K-quant weight with realistic per-block scales.
    /// `seeded_*_blocks` produces valid layouts but wild f16 headers (up to
    /// ~65504) that exaggerate absolute logit magnitudes; the fault tests
    /// want realistic `d`/`min` so the bounds are meaningful.
    fn kweight(dtype: KQuantDtype, out: usize, input: usize, seed: u64, d: f32) -> KQuantWeight {
        let blocks = out * (input / QK_K);
        let mut bytes = match dtype {
            KQuantDtype::Q4K => seeded_q4_blocks(blocks, seed),
            KQuantDtype::Q6K => seeded_q6_blocks(blocks, seed),
        };
        let (block_bytes, d_off, min_off) = match dtype {
            KQuantDtype::Q4K => (144usize, 0usize, 2usize),
            KQuantDtype::Q6K => (210usize, 208usize, 208usize),
        };
        let d_bits = half::f16::from_f32(d).to_bits();
        let min_bits = half::f16::from_f32(-0.02).to_bits();
        for block in 0..blocks {
            let base = block * block_bytes;
            bytes[base + d_off..base + d_off + 2].copy_from_slice(&d_bits.to_le_bytes());
            if dtype == KQuantDtype::Q4K {
                bytes[base + min_off..base + min_off + 2].copy_from_slice(&min_bits.to_le_bytes());
            }
        }
        KQuantWeight::try_new(bytes, [out, input], dtype).unwrap()
    }

    fn q8weight(out: usize, input: usize, seed: u64) -> QuantizedWeight {
        let mut data = Vec::new();
        for row in 0..out {
            let values: Vec<f32> = (0..input)
                .map(|i| ((i * 31 + row * 17 + seed as usize * 13) as f32 / 97.0).sin() * 0.7)
                .collect();
            let mut row_bytes = Vec::new();
            quantize_q8_0_into(&values, &mut row_bytes);
            data.extend_from_slice(&row_bytes);
        }
        QuantizedWeight::try_new(data, vec![out, input]).unwrap()
    }

    const OUT: usize = 8;

    #[test]
    fn payload_bit_flips_are_bounded_and_finite() {
        let src = seeded_activations(QK_K, 7);
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let w = kweight(dtype, OUT, QK_K, 42, 0.05);
            let pristine = k_decode(&w, &src, false).unwrap();
            let (payload_start, payload_end) = match dtype {
                KQuantDtype::Q4K => (16usize, 144usize),
                KQuantDtype::Q6K => (0usize, 192usize),
            };
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for block in 0..OUT {
                for byte in payload_start..payload_end {
                    let mut faulted = w.clone();
                    let data = faulted.data_mut().expect("owned weight");
                    let off = block
                        * match dtype {
                            KQuantDtype::Q4K => 144usize,
                            KQuantDtype::Q6K => 210usize,
                        };
                    inject_bit_flip(
                        &mut data[off..off + (payload_end - payload_start)],
                        byte - payload_start,
                        3,
                    )
                    .expect("in bounds");
                    let out = k_decode(&faulted, &src, false).unwrap();
                    let impact = measure_impact(&pristine, &out);
                    assert!(
                        impact.logits_finite,
                        "{dtype:?} block {block} byte {byte} went non-finite"
                    );
                    max_abs = max_abs.max(impact.max_abs_logit_diff);
                    max_rel = max_rel.max(impact.rel_l2);
                }
            }
            // every payload-bit fault is bounded: the largest possible logit
            // delta from one block is 256 * (d * max_scale * nibble_step),
            // i.e. 256 * 0.05 * 63 * 8 = 6451 for Q4_K and
            // 256 * 0.05 * 128 * 8 = 13107 for Q6_K — never Inf/NaN
            let bound = match dtype {
                KQuantDtype::Q4K => 7_000.0f32,
                KQuantDtype::Q6K => 15_000.0f32,
            };
            assert!(max_abs < bound, "{dtype:?} max_abs {max_abs}");
            assert!(max_rel < 0.5, "{dtype:?} max_rel {max_rel}");
        }

        // Q8_0 payload: bounded as well
        let w = q8weight(OUT, QK_K, 11);
        let pristine = q8_decode(&w, &src).unwrap();
        let mut max_abs = 0.0f32;
        for block in 0..OUT {
            let mut faulted = w.clone();
            let data = faulted.data_mut().expect("owned weight");
            inject_bit_flip(&mut data[block * 34 + 2..block * 34 + 34], 16, 5).unwrap();
            let out = q8_decode(&faulted, &src).unwrap();
            let impact = measure_impact(&pristine, &out);
            assert!(impact.logits_finite);
            max_abs = max_abs.max(impact.max_abs_logit_diff);
        }
        // per-block worst case: 256 * (scale ~0.0055) * 32 = ~45
        assert!(max_abs < 100.0, "q8_0 max_abs {max_abs}");
    }

    #[test]
    fn scale_bit_flips_can_produce_non_finite_logits() {
        let src = seeded_activations(QK_K, 7);
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let w = kweight(dtype, OUT, QK_K, 42, 1.0);
            let pristine = k_decode(&w, &src, false).unwrap();
            let d_off = match dtype {
                KQuantDtype::Q4K => 0usize,
                KQuantDtype::Q6K => 208usize,
            };
            let block_bytes = match dtype {
                KQuantDtype::Q4K => 144usize,
                KQuantDtype::Q6K => 210usize,
            };
            let mut nonfinite = 0usize;
            for block in 0..OUT {
                let mut hit = false;
                for bit in 10..=14 {
                    let mut faulted = w.clone();
                    let data = faulted.data_mut().expect("owned weight");
                    let base = block * block_bytes + d_off;
                    // little-endian 16-bit f16: bit >= 8 lives in byte 1
                    inject_bit_flip(&mut data[base..base + 2], usize::from(bit >= 8), bit % 8)
                        .unwrap();
                    let out = k_decode(&faulted, &src, false).unwrap();
                    if !measure_impact(&pristine, &out).logits_finite {
                        hit = true;
                        break;
                    }
                }
                if hit {
                    nonfinite += 1;
                }
            }
            assert!(
                nonfinite > 0,
                "{dtype:?}: expected at least one scale-bit flip to produce non-finite logits"
            );
        }
    }

    #[test]
    fn validate_integrity_detects_corrupted_headers() {
        // K-quant: NaN d scale -> error
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            let mut w = kweight(dtype, OUT, QK_K, 42, 0.05);
            assert!(w.validate_integrity().is_ok());
            let data = w.data_mut().expect("owned");
            let d_off = match dtype {
                KQuantDtype::Q4K => 0usize,
                KQuantDtype::Q6K => 208usize,
            };
            // f16 NaN: exponent all-ones, nonzero mantissa (0x7E00)
            data[d_off..d_off + 2].copy_from_slice(&0x7E00u16.to_le_bytes());
            let err = w.validate_integrity().expect_err("NaN header must fail");
            assert!(err.contains("non-finite"), "{err}");
        }
        // Q8_0: NaN scale -> error
        let mut w = q8weight(OUT, QK_K, 11);
        assert!(w.validate_integrity().is_ok());
        let data = w.data_mut().expect("owned");
        data[0..2].copy_from_slice(&0x7E00u16.to_le_bytes());
        let err = w.validate_integrity().expect_err("NaN scale must fail");
        assert!(err.contains("non-finite"), "{err}");
    }

    #[test]
    fn inject_bit_flip_is_bounds_checked() {
        let mut data = vec![0u8; 4];
        assert!(inject_bit_flip(&mut data, 4, 0).is_err());
        assert!(inject_bit_flip(&mut data, 0, 8).is_err());
        inject_bit_flip(&mut data, 0, 0).unwrap();
        assert_eq!(data[0], 1);
    }
}
