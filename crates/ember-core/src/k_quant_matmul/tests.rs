use super::*;
use crate::k_matmul::tests::{seeded_activations, seeded_k_blocks};

fn weight(dtype: KQuantDtype, out: usize, input: usize, seed: u64) -> KQuantWeight {
    let bytes = seeded_k_blocks(dtype, out * (input / QK_K), seed);
    KQuantWeight::try_new(bytes, [out, input], dtype).unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The scalar tier plus, where the host supports it, the x86 tier.
fn tiers(scalar: KQuantWeight) -> Vec<(&'static str, KQuantWeight)> {
    let mut tiers = vec![("scalar", scalar.clone())];
    if x86_k_supported() {
        tiers.push(("x86", scalar.with_execution(KExecution::CompressedX86)));
    }
    tiers
}

/// A single-token matvec issued outside a Rayon pool takes the decode
/// team; whichever member claims a column chunk, the result must equal
/// the serial product bit for bit, including the accumulate-into-`dst`
/// contract and a last chunk shorter than the chunk width.
#[test]
fn decode_team_matvec_is_bit_identical_to_serial() {
    for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
        for outputs in [
            PARALLEL_LEAF_OUTPUTS + 1,
            4 * DECODE_TEAM_CHUNK_COLUMNS + 17,
        ] {
            let input_features = 8 * QK_K;
            let weight = weight(dtype, outputs, input_features, 0x5eed + outputs as u64);
            let src = seeded_activations(input_features, 0xfeed);
            let initial: Vec<f32> = (0..outputs).map(|i| i as f32 * 0.125 - 3.0).collect();
            let mut serial = initial.clone();
            matmul_k_q8_into(&src, 1, &weight, &mut serial, false).unwrap();
            let mut team = initial;
            matmul_k_q8_into(&src, 1, &weight, &mut team, true).unwrap();
            assert_eq!(bits(&team), bits(&serial), "{dtype:?} outputs={outputs}");
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn arm_k_kernels_match_scalar_and_preserve_accumulation() {
    if !arm_k_supported() {
        return;
    }
    for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
        for outputs in [17, 513] {
            for seed in [0, 17, 913] {
                let scalar = weight(dtype, outputs, 3 * QK_K, seed);
                let arm = scalar.clone().with_execution(KExecution::CompressedArm);
                let mut packed = vec![Q8KBlock::default(); 3];
                for (b, a) in packed.iter_mut().enumerate() {
                    a.d = (b as f32 - 1.0) * 0.125;
                    for (j, q) in a.qs.iter_mut().enumerate() {
                        *q = [i8::MIN, i8::MAX, -1, 0, 1, (j * 37 + seed as usize) as i8][j % 6];
                    }
                    for (dst, qs) in a.bsums.iter_mut().zip(a.qs.chunks_exact(16)) {
                        *dst = qs.iter().map(|&q| i16::from(q)).sum();
                    }
                }
                for column in 0..outputs {
                    assert_eq!(
                        dot_column(&scalar, column, &packed).to_bits(),
                        dot_column(&arm, column, &packed).to_bits(),
                        "{dtype:?} seed={seed} column={column}"
                    );
                }
                for rows in [1, 4, 7] {
                    let src = seeded_activations(rows * 3 * QK_K, seed + 7);
                    let mut expected = vec![0.25; rows * outputs];
                    matmul_k_q8_into(&src, rows, &scalar, &mut expected, false).unwrap();
                    for parallel in [false, true] {
                        let mut actual = vec![0.25; rows * outputs];
                        matmul_k_q8_into(&src, rows, &arm, &mut actual, parallel).unwrap();
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}

/// The decode kernels (`arm::q4`/`arm::q6`, one and four activation
/// rows) reproduce the original NEON kernels bit for bit on random
/// weights (Q6_K scales include -128), random and extreme activations
/// (all-zero blocks, +-127 runs) and odd shapes.
#[cfg(target_arch = "aarch64")]
#[test]
fn arm_decode_kernels_match_reference_kernels_bitwise() {
    if !arm_k_supported() {
        return;
    }
    for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
        for (outputs, blocks) in [(1usize, 1usize), (3, 2), (17, 3), (33, 8), (5, 32)] {
            for seed in [1u64, 99, 4242] {
                let input = blocks * QK_K;
                let w = weight(dtype, outputs, input, seed);
                let mut src = seeded_activations(4 * input, seed + 3);
                src[..QK_K].fill(0.0);
                for (index, value) in src[input..input + QK_K].iter_mut().enumerate() {
                    *value = if index % 3 == 0 { -1.0e6 } else { 1.0e6 };
                }
                for value in &mut src[2 * input + 7..2 * input + 40] {
                    *value *= 1.0e-30;
                }
                let mut packed = Vec::new();
                quantize_q8_k_into_scalar(&src, &mut packed).unwrap();
                let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                for column in 0..outputs {
                    // SAFETY: the feature gate was checked above and
                    // `column < outputs` of a validated weight.
                    let (new4, old4) = unsafe {
                        match dtype {
                            KQuantDtype::Q4K => (
                                arm::q4::<4>(w.data(), blocks, column, &packed),
                                arm::q4_reference::<4>(w.data(), blocks, column, &packed),
                            ),
                            KQuantDtype::Q6K => (
                                arm::q6::<4>(w.data(), blocks, column, &packed),
                                arm::q6_reference::<4>(w.data(), blocks, column, &packed),
                            ),
                        }
                    };
                    assert_eq!(
                        bits(&new4),
                        bits(&old4),
                        "{dtype:?} {outputs}x{blocks} seed={seed} column={column}"
                    );
                    for row in 0..4 {
                        let one = &packed[row * blocks..(row + 1) * blocks];
                        // SAFETY: as above.
                        let (new1, old1) = unsafe {
                            match dtype {
                                KQuantDtype::Q4K => (
                                    arm::q4::<1>(w.data(), blocks, column, one),
                                    arm::q4_reference::<1>(w.data(), blocks, column, one),
                                ),
                                KQuantDtype::Q6K => (
                                    arm::q6::<1>(w.data(), blocks, column, one),
                                    arm::q6_reference::<1>(w.data(), blocks, column, one),
                                ),
                            }
                        };
                        assert_eq!(
                            new1[0].to_bits(),
                            old1[0].to_bits(),
                            "{dtype:?} row={row} column={column}"
                        );
                        assert_eq!(new1[0].to_bits(), new4[row].to_bits());
                    }
                }
            }
        }
    }
}

/// The four-row interleaved prefill tiles reproduce the per-column ARM
/// dots (and hence the scalar oracle) bit for bit: row counts off the
/// four-row groups, columns off the tile width, zero/extreme activations
/// and accumulation into a non-zero destination.
#[cfg(target_arch = "aarch64")]
#[test]
fn arm_prefill_tiles_match_per_column_dots_bitwise() {
    if !arm_k_supported() {
        return;
    }
    for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
        for (outputs, input) in [(1, QK_K), (5, 2 * QK_K), (19, 3 * QK_K), (300, QK_K)] {
            let arm = weight(dtype, outputs, input, 0x77 + outputs as u64)
                .with_execution(KExecution::CompressedArm);
            let blocks = input / QK_K;
            for rows in [2usize, 3, 4, 5, 8, 9, 13] {
                let mut src = seeded_activations(rows * input, 0x99 + rows as u64);
                src[..QK_K].fill(0.0); // an all-zero Q8_K block
                src[input] = 1.0e30;
                let mut packed = Vec::new();
                quantize_q8_k_into_scalar(&src, &mut packed).unwrap();
                let initial: Vec<f32> = (0..rows * outputs)
                    .map(|index| (index % 7) as f32 * 0.25 - 0.75)
                    .collect();
                let mut expected = initial.clone();
                for row in 0..rows {
                    let packed_row = &packed[row * blocks..(row + 1) * blocks];
                    for column in 0..outputs {
                        expected[row * outputs + column] += dot_column(&arm, column, packed_row);
                    }
                }
                for parallel in [false, true] {
                    let mut actual = initial.clone();
                    matmul_k_q8_into(&src, rows, &arm, &mut actual, parallel).unwrap();
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "{dtype:?} rows={rows} out={outputs} in={input} parallel={parallel}"
                    );
                }
            }
        }
    }
}

/// `cargo test --release --lib k_quant_matmul::tests::k_prefill_throughput -- --ignored --nocapture`
#[test]
#[ignore = "timing probe"]
fn k_prefill_throughput() {
    for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
        for (rows, outputs, input) in [
            (26, 2048, 2048),
            (128, 2048, 2048),
            (512, 2048, 2048),
            (512, 8192, 2048),
            (512, 2048, 8192),
        ] {
            let mut w = weight(dtype, outputs, input, 5);
            if arm_k_supported() {
                w = w.with_execution(KExecution::CompressedArm);
            }
            let src = seeded_activations(rows * input, 9);
            let mut out = vec![0.0f32; rows * outputs];
            let mut best = f64::MAX;
            for _ in 0..7 {
                let t = std::time::Instant::now();
                matmul_k_q8_into(&src, rows, &w, &mut out, true).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            let ops = 2.0 * (rows * outputs * input) as f64;
            println!(
                "{dtype:?} {rows}x{outputs}x{input}: {:.0} GOPS ({:.2} ms)",
                ops / best / 1e9,
                best * 1e3
            );
        }
    }
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

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn vector_q8_k_packing_is_bit_identical_to_scalar() {
    #[cfg(target_arch = "x86_64")]
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

    let mut scalar = vec![Q8KBlock::default(); values.len() / QK_K];
    quantize_q8_k_blocks_scalar(&values, &mut scalar).unwrap();
    assert_eq!(&scalar[1].qs[1..9], &[-4, -2, -2, 0, 0, 2, 2, 4]);
    assert!(scalar[2].d.is_sign_negative());
    assert!(!scalar[6].d.is_sign_negative());
    assert_eq!(&scalar[2].qs[..2], &[-127, 127]);
    assert_eq!(&scalar[6].qs[..2], &[-127, 127]);
    #[cfg(target_arch = "x86_64")]
    let vector = {
        let mut vector = Vec::new();
        // SAFETY: guarded by the complete x86 feature predicate.
        unsafe { super::x86::quantize_q8_k_into(&values, &mut vector) }.unwrap();
        vector
    };
    #[cfg(target_arch = "aarch64")]
    let vector = {
        let mut vector = vec![Q8KBlock::default(); scalar.len()];
        super::arm::quantize_q8_k_blocks(&values, &mut vector).unwrap();
        vector
    };
    assert_eq!(vector.len(), scalar.len());
    for (index, (actual, expected)) in vector.iter().zip(&scalar).enumerate() {
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

        let mut full_src = vec![0.25; 4 * QK_K];
        full_src[3 * QK_K + 17] = non_finite;
        for (tier, weight) in tiers(weight(KQuantDtype::Q4K, 3, 2 * QK_K, 0x51)) {
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
        // The AVX-512 kernels are documented bit-identical to AVX2. They
        // are opt-in at runtime, so test them whenever the host could run
        // them rather than only when EMBER_K_AVX512 happens to be set.
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: guarded by the x86 and AVX-512 feature predicates.
            let q4_avx512 =
                unsafe { super::x86::q4_k_dot_q8_k_avx512(q4_weight.data(), 2, 0, &packed) };
            // SAFETY: as above.
            let q6_avx512 =
                unsafe { super::x86::q6_k_dot_q8_k_avx512(q6_weight.data(), 2, 0, &packed) };
            assert_eq!(q4_avx512.to_bits(), q4_x86.to_bits());
            assert_eq!(q6_avx512.to_bits(), q6_x86.to_bits());
        }
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
        let src = seeded_activations(512, 0x55);
        for (tier, weight) in tiers(scalar) {
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
                        let packed_row = &packed[row * blocks_per_row..(row + 1) * blocks_per_row];
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

                    for (tier, weight) in tiers(scalar_weight) {
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
        let seed = match dtype {
            KQuantDtype::Q4K => 0x41,
            KQuantDtype::Q6K => 0x61,
        };
        let mut bytes = seeded_k_blocks(dtype, blocks, seed);
        for block in bytes.chunks_exact_mut(dtype.block_bytes()) {
            match dtype {
                KQuantDtype::Q4K => block[..4].fill(0),
                KQuantDtype::Q6K => block[208..210].fill(0),
            }
        }
        let scalar_weight =
            KQuantWeight::try_new(bytes, [output_features, input_features], dtype).unwrap();
        let src = seeded_activations(rows * input_features, 0x71);
        for (_, weight) in tiers(scalar_weight) {
            let expected: Vec<f32> = (0..rows * output_features)
                .map(|index| index as f32 * 0.25 - 3.0)
                .collect();
            let mut actual = expected.clone();
            matmul_k_q8_into(&src, rows, &weight, &mut actual, false).unwrap();
            assert_eq!(
                bits(&actual),
                bits(&expected),
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
        for (tier, weight) in tiers(scalar) {
            let src = seeded_activations(rows * weight.in_features(), 0x91);
            let mut dst = vec![0.0; rows * weight.out_features()];
            matmul_k_q8_into(&src, rows, &weight, &mut dst, false).unwrap();
            dst.fill(0.0);
            assert!(crate::alloc_counter::counting_active());
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
        let src = seeded_activations(rows * 512, 7);
        for (tier, weight) in tiers(scalar_weight) {
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
    assert_eq!(bits(&dst_a), bits(&expected_a));
    assert_eq!(bits(&dst_b), bits(&expected_b));
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
fn presplit_dot_four_rows_matches_packed() {
    if !x86_k_supported() {
        return;
    }
    for (dtype, weight_seed, src_seed) in [
        (KQuantDtype::Q4K, 0x5a, 0x5b),
        (KQuantDtype::Q6K, 0x6a, 0x6b),
    ] {
        let w_packed = weight(dtype, 5, 512, weight_seed).with_execution(KExecution::CompressedX86);
        let w_presplit = w_packed.clone().with_presplit();
        let src = seeded_activations(4 * 512, src_seed);
        let mut packed = Vec::new();
        quantize_q8_k_into_scalar(&src, &mut packed).unwrap();
        for column in 0..w_packed.out_features() {
            let expected = dot_four_rows(&w_packed, column, &packed).unwrap();
            let actual = dot_four_rows(&w_presplit, column, &packed).unwrap();
            for row in 0..4 {
                assert_eq!(
                    expected[row].to_bits(),
                    actual[row].to_bits(),
                    "{dtype:?} column {column} row {row}"
                );
            }
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

/// `parallel_for_shape` (the runtime schedule's predicate) must agree with
/// the kernel's own `scheduler_name` at every threshold boundary.
#[test]
fn parallel_for_shape_matches_scheduler_name_at_thresholds() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        let threads = rayon::current_num_threads();
        assert_eq!(threads, 2);
        for (out_features, in_features) in [
            (PARALLEL_LEAF_OUTPUTS, 8192usize),
            (PARALLEL_LEAF_OUTPUTS * 2, 8192),
            (512, 1024),
            (512, 512),
            (512, 768),
            (2048, 2048),
        ] {
            let weight = weight(KQuantDtype::Q4K, out_features, in_features, 0x77);
            let decision = parallel_for_shape(1, in_features, out_features, threads, true);
            let scheduled = scheduler_name(1, &weight, true);
            assert_eq!(
                scheduled,
                if decision {
                    "column-parallel-rayon"
                } else {
                    "serial"
                },
                "shape {out_features}x{in_features}"
            );
        }
        // The request flag and worker count short-circuit before shape.
        assert!(!parallel_for_shape(1, 2048, 2048, threads, false));
        assert!(!parallel_for_shape(1, 2048, 2048, 1, true));
    });
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
