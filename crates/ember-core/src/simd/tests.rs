#[cfg(target_arch = "aarch64")]
use super::q8_decode::matmul_q8_0_batch_dispatch_tile;
use super::q8_decode::{matmul_q8_0_decode_dispatch_chunk, matmul_q8_0_decode_scalar};
use crate::quant::{QuantizedWeightInterleaved, QuantizedWeightVnni, VNNI_OUT_TILE};

/// SiLU must stay correct where exp over- or underflows f32. The AVX2
/// exp used to wrap its exponent field there, so silu(88.5) came out as
/// -0.0 and silu(-90) as -90. Runs on every architecture's dispatch.
#[test]
fn silu_is_correct_at_large_magnitudes() {
    let src: Vec<f32> = vec![
        -120.0, -95.0, -90.0, -88.5, -87.5, -50.0, -1.0, 0.0, 1.0, 50.0, 87.5, 88.5, 90.0, 95.0,
        120.0, 3.0,
    ];
    let ones = vec![1.0; src.len()];
    let mut fused = vec![0.0; src.len()];
    super::silu_mul_into(&src, &ones, &mut fused);
    for (&x, &z) in src.iter().zip(&fused) {
        let expected = (x as f64 / (1.0 + (-(x as f64)).exp())) as f32;
        let tolerance = 1e-5 * x.abs().max(1.0);
        assert!(
            (z - expected).abs() <= tolerance,
            "silu_mul({x}) = {z}, expected {expected}"
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn arm_inplace_norm_is_exact_with_tensor_reference() {
    use crate::tensor::CpuTensor;
    for n in [1, 3, 4, 5, 31, 2048] {
        let x: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.17).sin() * (i % 7 + 1) as f32)
            .collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.11).cos()).collect();
        let expected = CpuTensor::from_data(vec![1, n], x.clone())
            .rms_norm(&CpuTensor::from_data(vec![n], w.clone()), 1e-5);
        let mut out = vec![0.0; n];
        rms_norm_into(&x, &w, 1e-5, &mut out);
        assert_eq!(out, expected.data());
        let mut scalar = vec![0.0; n];
        rms_norm_into_scalar(&x, &w, 1e-5, &mut scalar);
        rms_norm_q8_decode_into(&x, &w, 1e-5, &mut out);
        assert_eq!(
            out, scalar,
            "Q8 fast decode must retain scalar reduction order"
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn arm_q8_batch_is_exact_across_tile_and_output_tails() {
    if !is_aarch64_feature_detected!("dotprod") || !is_aarch64_feature_detected!("fp16") {
        return;
    }
    for rows in [1, 2, 3, 4] {
        for outputs in [1, 3, 4, 17] {
            for blocks in [1, 7, 64] {
                let mut x = vec![0u8; 1 + rows * blocks * 34];
                let mut data = vec![0u8; 1 + outputs * blocks * 34];
                for (i, block) in x[1..]
                    .chunks_mut(34)
                    .chain(data[1..].chunks_mut(34))
                    .enumerate()
                {
                    block[..2].copy_from_slice(
                        &f16::from_f32((i as f32 % 17.0 - 8.0) / 8.0)
                            .to_bits()
                            .to_le_bytes(),
                    );
                    for (j, q) in block[2..].iter_mut().enumerate() {
                        *q = (i * 37 + j * 13) as u8;
                    }
                }
                let mut expected = vec![0.0; rows * outputs];
                for row in 0..rows {
                    matmul_q8_0_decode_scalar(
                        &x[1 + row * blocks * 34..][..blocks * 34],
                        &data[1..],
                        outputs,
                        blocks,
                        &mut expected[row * outputs..][..outputs],
                    );
                }
                let mut actual = vec![0.0; rows * outputs];
                matmul_q8_0_batch_dispatch_tile(
                    &x[1..],
                    rows,
                    &data[1..],
                    outputs,
                    blocks,
                    &mut actual,
                );
                assert_eq!(
                    actual, expected,
                    "rows={rows} outputs={outputs} blocks={blocks}"
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn arm_q8_dotprod_matches_scalar_extremes_and_tails() {
    if !is_aarch64_feature_detected!("dotprod") || !is_aarch64_feature_detected!("fp16") {
        return;
    }
    for rows in [1, 3, 4, 5, 17] {
        for blocks in [1, 3, 64] {
            let mut x = vec![0u8; 1 + blocks * 34];
            let mut w = vec![0u8; 1 + rows * blocks * 34];
            for (i, block) in x[1..]
                .chunks_mut(34)
                .chain(w[1..].chunks_mut(34))
                .enumerate()
            {
                block[..2].copy_from_slice(
                    &[0u16, 1, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x8001, 0xbc00][i % 8].to_le_bytes(),
                );
                for (j, q) in block[2..].iter_mut().enumerate() {
                    *q = [128, 127, 0, 255, 1, (i * 37 + j * 13) as u8][(i + j) % 6];
                }
            }
            let mut expected = vec![0.0; rows];
            let mut actual = vec![0.0; rows];
            matmul_q8_0_decode_scalar(&x[1..], &w[1..], rows, blocks, &mut expected);
            // SAFETY: dotprod and fp16 are checked at the top of the
            // test; `x` holds one row and `w` holds `rows` rows.
            unsafe {
                aarch64::matmul_q8_0_decode_dotprod(&x[1..], &w[1..], blocks, &mut actual);
            }
            assert_eq!(actual, expected, "rows={rows} blocks={blocks}");
        }
    }
}

use crate::quant::QuantizedWeight;

/// Fast-exp error ladder, level 1 (kernel): the AVX2 approximation vs
/// libm `expf` over a dense grid plus adversarial points. Gate: max
/// relative error <= 4e-7 (~3 f32 ulp of typical results; measured is
/// usually < 1e-7).
#[test]
fn fast_exp_accuracy_vs_libm() {
    let mut cases: Vec<f32> = Vec::new();
    // dense grid over the softmax-relevant domain
    let mut x = -88.0f32;
    while x <= 0.0 {
        cases.push(x);
        x += 0.005;
    }
    // positive range too (general kernel contract)
    let mut x = 0.0f32;
    while x <= 88.0 {
        cases.push(x);
        x += 0.011;
    }
    // adversarial: exact powers of ln2 (r == 0), boundaries, denormal
    for k in -126..=87 {
        cases.push(k as f32 * std::f64::consts::LN_2 as f32);
    }
    cases.extend([0.0f32, -0.0, -87.9, -88.0, -1e6]);

    let reference: Vec<f32> = cases.iter().map(|&v| v.max(-88.0).exp()).collect();
    let mut got = vec![0.0f32; cases.len()];
    super::fast_exp_into(&cases, &mut got);
    let mut max_rel = 0.0f32;
    for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
        assert!(
            g.is_finite() && g >= 0.0,
            "non-finite exp at {} -> {g}",
            cases[i]
        );
        if r > 1e-30 {
            let rel = ((g - r) / r).abs();
            max_rel = max_rel.max(rel);
        }
    }
    assert!(
        max_rel <= 4e-7,
        "fast_exp rel err {max_rel:.3e} exceeds gate"
    );

    // in-place must match the into variant exactly
    let mut inplace = cases.clone();
    super::fast_exp_in_place(&mut inplace);
    assert_eq!(got, inplace);
}

use super::*;

const BOUNDARY_LENGTHS: &[usize] = &[0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33];

fn patterned_vec(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 + phase) * 0.37).sin() * 3.0 + (i % 5) as f32 * 0.125)
        .collect()
}

fn quantized_input(values: &[f32]) -> Vec<u8> {
    let mut data = Vec::new();
    crate::quant::quantize_q8_0_into(values, &mut data);
    data
}

fn max_diff(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let abs = (x - y).abs();
        let rel = abs / x.abs().max(y.abs()).max(1.0);
        if abs > max_abs {
            max_abs = abs;
            max_rel = rel;
            max_idx = i;
        }
    }
    (max_abs, max_rel, max_idx)
}

fn assert_close(label: &str, got: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32) {
    let (max_abs, max_rel, max_idx) = max_diff(got, expected);
    assert!(
        max_abs <= abs_tol || max_rel <= rel_tol,
        "{label}: max_abs={max_abs} max_rel={max_rel} idx={max_idx} got={} expected={}",
        got.get(max_idx).copied().unwrap_or(0.0),
        expected.get(max_idx).copied().unwrap_or(0.0)
    );
}

/// Build a single Q8_0 block (34 bytes) with known scale and quants.
fn make_block(scale: f32, quants: &[i8; 32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(Q8_0_TYPE_SIZE);
    let s = f16::from_f32(scale);
    buf.extend_from_slice(&s.to_bits().to_le_bytes());
    for &q in quants {
        buf.push(q as u8);
    }
    assert_eq!(buf.len(), Q8_0_TYPE_SIZE);
    buf
}

#[test]
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn arm_dot_matches_retained_reference_lengths_and_tails() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/dot-reference/arm-f32.json"
    ))
    .unwrap();
    for record in fixture["records"].as_array().unwrap() {
        let decode = |key: &str| -> Vec<f32> {
            record[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
                .collect()
        };
        let a = decode("a_bits");
        let b = decode("b_bits");
        let actual = dot_product(&a, &b);
        assert_eq!(
            actual.to_bits(),
            record["expected_bits"].as_u64().unwrap() as u32,
            "reference dot case {} length {}",
            record["name"],
            a.len()
        );
    }
}

#[test]
fn simd_helper_boundary_parity() {
    for &len in BOUNDARY_LENGTHS {
        let a = patterned_vec(len, 0.25);
        let b = patterned_vec(len, 1.75);

        let expected_sum_squares: f32 = a.iter().map(|v| v * v).sum();
        let got_sum_squares = sum_squares(&a);
        assert!(
            (got_sum_squares - expected_sum_squares).abs() <= 1e-5 * len.max(1) as f32,
            "sum_squares len={len}: got={got_sum_squares} expected={expected_sum_squares}"
        );

        let expected_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let got_dot = dot_product(&a, &b);
        assert!(
            (got_dot - expected_dot).abs() <= 1e-5 * len.max(1) as f32,
            "dot_product len={len}: got={got_dot} expected={expected_dot}"
        );

        let mut got = vec![0.0; len];
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        elemul(&a, &b, &mut got);
        assert_close(&format!("elemul len={len}"), &got, &expected, 1e-6, 1e-6);

        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        add(&a, &b, &mut got);
        assert_close(&format!("add len={len}"), &got, &expected, 1e-6, 1e-6);

        let mut acc = a.clone();
        let mut expected = a.clone();
        for (dst, src) in expected.iter_mut().zip(&b) {
            *dst += -0.75 * src;
        }
        weighted_add(&mut acc, &b, -0.75);
        assert_close(
            &format!("weighted_add len={len}"),
            &acc,
            &expected,
            1e-5,
            1e-6,
        );

        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * 0.5 * y).collect();
        scale_weight_mul(&a, 0.5, &b, &mut got);
        assert_close(
            &format!("scale_weight_mul len={len}"),
            &got,
            &expected,
            1e-6,
            1e-6,
        );
    }
}

#[test]
fn f16_cache_simd_helpers_match_rounded_scalar() {
    for &len in BOUNDARY_LENGTHS {
        let query = patterned_vec(len, 0.25);
        let source = patterned_vec(len, 1.75);
        let source_f16 = source
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();

        let expected_dot = query
            .iter()
            .zip(&source_f16)
            .map(|(a, b)| a * b.to_f32())
            .sum::<f32>();
        let got_dot = dot_product_f16(&query, &source_f16);
        assert!(
            (got_dot - expected_dot).abs() <= 1e-5 * len.max(1) as f32,
            "f16 dot len={len}: got={got_dot} expected={expected_dot}"
        );

        let mut expected = patterned_vec(len, 3.25);
        for (dst, src) in expected.iter_mut().zip(&source_f16) {
            *dst += 0.375 * src.to_f32();
        }
        let mut got = patterned_vec(len, 3.25);
        weighted_add_f16(&mut got, &source_f16, 0.375);
        assert_close("f16 weighted add", &got, &expected, 1e-5, 1e-5);
    }
}

/// Build `n` consecutive Q8_0 blocks with alternating scale patterns.
fn make_row(blocks_per_row: usize) -> (Vec<u8>, Vec<f32>) {
    let mut data = Vec::with_capacity(blocks_per_row * Q8_0_TYPE_SIZE);
    let mut expected = vec![0.0f32; blocks_per_row * Q8_0_BLOCK_SIZE];

    for b in 0..blocks_per_row {
        let scale_f32 = 0.5 + (b as f32) * 0.1;
        // expected values must account for f16 round-trip precision loss:
        // the scale is stored as f16, so the effective scale is
        // f16::from_f32(scale_f32).to_f32(), not the original f32 value.
        let scale_effective = f16::from_f32(scale_f32).to_f32();
        let mut quants = [0i8; 32];
        for j in 0..32 {
            quants[j] = (j as i8) - 16; // range [-16, 15]
            expected[b * 32 + j] = (quants[j] as f32) * scale_effective;
        }
        data.extend_from_slice(&make_block(scale_f32, &quants));
    }
    (data, expected)
}

#[test]
fn scalar_dequant_row_matches_expected() {
    let (data, expected) = make_row(4);
    let mut dst = vec![0.0f32; 4 * 32];
    dequantize_row_scalar(&data, 0, 4, &mut dst);
    for (i, (a, b)) in dst.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "mismatch at {i}: scalar={a} expected={b}"
        );
    }
}

#[test]
fn dispatch_with_offset_produces_same_output() {
    // simulate a weight matrix with multiple rows
    let blocks_per_row = 8;
    let (row0_data, _row0_expected) = make_row(blocks_per_row);
    let (row1_data, _row1_expected) = make_row(blocks_per_row);

    let mut data = row0_data;
    data.extend_from_slice(&row1_data);

    let mut scalar_out = vec![0.0f32; blocks_per_row * 32];
    let mut dispatch_out = vec![0.0f32; blocks_per_row * 32];

    // row 0 (block_start = 0)
    dequantize_row_scalar(&data, 0, blocks_per_row, &mut scalar_out);
    dequantize_q8_0_row(&data, 0, blocks_per_row, &mut dispatch_out);
    for (i, (s, d)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
        assert!((s - d).abs() < 1e-6, "row0 mismatch at {i}");
    }

    // row 1 (block_start = blocks_per_row)
    dequantize_row_scalar(&data, blocks_per_row, blocks_per_row, &mut scalar_out);
    dequantize_q8_0_row(&data, blocks_per_row, blocks_per_row, &mut dispatch_out);
    for (i, (s, d)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
        assert!((s - d).abs() < 1e-6, "row1 mismatch at {i}");
    }
}

#[test]
fn edge_case_min_max_values() {
    // test extreme quant values: min i8 (-128), max i8 (127), zero scale
    let mut data = Vec::new();

    // block 0: scale=0.0, quants=[-128, 127, 0, ...]
    let s0 = f16::from_f32(0.0);
    data.extend_from_slice(&s0.to_bits().to_le_bytes());
    data.push((-128i8) as u8);
    data.push(127i8 as u8);
    data.push(0u8);
    data.resize(data.len() + 29, 0);

    // block 1: scale=1.0, quants=[-128, 127, 0, ...]
    let s1 = f16::from_f32(1.0);
    data.extend_from_slice(&s1.to_bits().to_le_bytes());
    data.push((-128i8) as u8);
    data.push(127i8 as u8);
    data.push(0u8);
    data.resize(data.len() + 29, 0);

    let blocks = 2;
    let mut scalar_out = vec![0.0f32; blocks * 32];
    let mut dispatch_out = vec![0.0f32; blocks * 32];

    dequantize_row_scalar(&data, 0, blocks, &mut scalar_out);
    dequantize_q8_0_row(&data, 0, blocks, &mut dispatch_out);

    for (i, (s, d)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
        assert!(
            (s - d).abs() < 1e-6,
            "edge mismatch at {i}: scalar={s} dispatch={d}"
        );
    }
}

/// Build random Q8_0 weight data for shape `[out_features, in_features]`.
/// Each block gets a random f16 scale and random i8 quants.
fn random_q8_0_data(out_features: usize, in_features: usize) -> Vec<u8> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
    let total_blocks = out_features * blocks_per_row;
    let mut data = vec![0u8; total_blocks * Q8_0_TYPE_SIZE];

    for b in 0..total_blocks {
        let offset = b * Q8_0_TYPE_SIZE;
        let scale = f16::from_f32(rng.r#gen::<f32>() * 2.0);
        data[offset..offset + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
        for j in 0..Q8_0_BLOCK_SIZE {
            data[offset + 2 + j] = rng.r#gen::<i8>() as u8;
        }
    }
    data
}

// -- decode region bit identity -------------------------------------------

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Scheduled decode regions (spin team, fused projections, fine chunks)
/// must reproduce the serial whole-matrix kernel bit for bit, including
/// odd row counts that leave partial tiles and partial chunks.
#[test]
fn decode_regions_are_bit_identical_to_serial_kernels() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    // (out_features, in_features): shapes below and above the parallel
    // threshold, odd row counts, and a wide down-projection-like input.
    let shapes = [
        (1, 64),
        (3, 2048),
        (7, 96),
        (17, 4096),
        (513, 2048),
        (1023, 2048),
        (2051, 1024),
        (263, 8192),
    ];
    for &(out_features, in_features) in &shapes {
        let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
        let x: Vec<f32> = (0..in_features)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect();
        let qx = quantized_input(&x);
        let weights: Vec<QuantizedWeight> = (0..3)
            .map(|extra| {
                let rows = out_features + extra * 5;
                QuantizedWeight::try_new(
                    random_q8_0_data(rows, in_features),
                    vec![rows, in_features],
                )
                .unwrap()
            })
            .collect();
        let serial: Vec<Vec<f32>> = weights
            .iter()
            .map(|w| {
                let mut out = vec![0.0f32; w.out_features()];
                matmul_q8_0_decode_dispatch_chunk(&qx, w.data(), blocks_per_row, &mut out);
                out
            })
            .collect();

        // On ARM the dot-product tiles implement exactly the scalar
        // per-row formula `sum += (dot as f32 * ws) * xs` in block order,
        // so the tile height (and any schedule) must not change a bit.
        #[cfg(target_arch = "aarch64")]
        for (w, expected) in weights.iter().zip(&serial) {
            let mut scalar = vec![0.0f32; w.out_features()];
            matmul_q8_0_decode_scalar(&qx, w.data(), w.out_features(), blocks_per_row, &mut scalar);
            assert_eq!(bits(expected), bits(&scalar), "tile vs scalar formula");
        }

        for (w, expected) in weights.iter().zip(&serial) {
            let mut out = vec![f32::NAN; w.out_features()];
            matmul_q8_0_decode(&qx, w, &mut out);
            assert_eq!(
                bits(&out),
                bits(expected),
                "single {out_features}x{in_features}"
            );
        }

        let mut outs: Vec<Vec<f32>> = weights
            .iter()
            .map(|w| vec![f32::NAN; w.out_features()])
            .collect();
        {
            let [a, b, c] = &mut outs[..] else {
                unreachable!()
            };
            matmul_q8_0_decode_rows(
                &qx,
                &mut [(&weights[0], a), (&weights[1], b), (&weights[2], c)],
            );
        }
        for (out, expected) in outs.iter().zip(&serial) {
            assert_eq!(bits(out), bits(expected));
        }

        if interleaved_q8_0_supported() {
            let interleaved = QuantizedWeightInterleaved::from_quantized(&weights[0]);
            let mut expected = vec![0.0f32; out_features];
            matmul_q8_0_decode_interleaved(&qx, &interleaved, &mut expected, 0);
            let mut out = vec![f32::NAN; out_features];
            matmul_q8_0_decode_interleaved_parallel(&qx, &interleaved, &mut out);
            assert_eq!(bits(&out), bits(&expected), "interleaved {out_features}");
        }
    }
}

/// The same regions requested from inside a Rayon worker (where the
/// decode team declines) and from many threads at once still match.
#[test]
fn decode_regions_match_under_nested_and_concurrent_callers() {
    use rand::Rng;
    let (out_features, in_features) = (1031, 2048);
    let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
    let w = QuantizedWeight::try_new(
        random_q8_0_data(out_features, in_features),
        vec![out_features, in_features],
    )
    .unwrap();
    let mut rng = rand::thread_rng();
    let x: Vec<f32> = (0..in_features)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect();
    let qx = quantized_input(&x);
    let mut expected = vec![0.0f32; out_features];
    matmul_q8_0_decode_dispatch_chunk(&qx, w.data(), blocks_per_row, &mut expected);
    let expected = bits(&expected);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    pool.install(|| {
        (0..6).into_par_iter().for_each(|_| {
            let mut out = vec![f32::NAN; out_features];
            matmul_q8_0_decode(&qx, &w, &mut out);
            assert_eq!(bits(&out), expected);
        })
    });
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..50 {
                    let mut out = vec![f32::NAN; out_features];
                    matmul_q8_0_decode(&qx, &w, &mut out);
                    assert_eq!(bits(&out), expected);
                }
            });
        }
    });
}

/// Multi-column decode tasks (batched sequences) must give every column
/// exactly the single-activation result, for both decode layouts, odd
/// row counts and every column count used by the batched decode tests.
#[test]
fn decode_column_tasks_match_single_column_bit_for_bit() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for &(out_features, in_features) in &[(5, 64), (37, 256), (1027, 2048), (131, 8192)] {
        let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
        let row_bytes = blocks_per_row * Q8_0_TYPE_SIZE;
        let weight = QuantizedWeight::try_new(
            random_q8_0_data(out_features, in_features),
            vec![out_features, in_features],
        )
        .unwrap();
        let interleaved = QuantizedWeightInterleaved::from_quantized(&weight);
        for columns in [1usize, 2, 3, 4, 5, 7, 8, 17] {
            let mut x = Vec::with_capacity(columns * row_bytes);
            for _ in 0..columns {
                let values: Vec<f32> = (0..in_features)
                    .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
                    .collect();
                x.extend_from_slice(&quantized_input(&values));
            }
            let mut layouts = vec![Q8DecodeWeight::Rows(&weight)];
            if interleaved_q8_0_supported() {
                layouts.push(Q8DecodeWeight::Interleaved(&interleaved));
            }
            for layout in layouts {
                let mut batched = vec![f32::NAN; columns * out_features];
                matmul_q8_0_decode_tasks(&mut [Q8DecodeTask::columns(
                    &x,
                    layout,
                    columns,
                    &mut batched,
                )]);
                for column in 0..columns {
                    let mut single = vec![f32::NAN; out_features];
                    matmul_q8_0_decode_tasks(&mut [Q8DecodeTask::columns(
                        &x[column * row_bytes..(column + 1) * row_bytes],
                        layout,
                        1,
                        &mut single,
                    )]);
                    assert_eq!(
                        bits(&batched[column * out_features..(column + 1) * out_features]),
                        bits(&single),
                        "{out_features}x{in_features} columns={columns} column={column}"
                    );
                }
            }
        }
    }
}

/// Team-split activation preparation is byte-identical to the serial
/// encoder and SiLU.
#[test]
fn decode_activation_preparation_matches_serial() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for len in [32usize, 2048, 8192, 32 * 33] {
        let a: Vec<f32> = (0..len).map(|_| rng.r#gen::<f32>() * 8.0 - 4.0).collect();
        let b: Vec<f32> = (0..len).map(|_| rng.r#gen::<f32>() * 8.0 - 4.0).collect();
        let mut serial = Vec::new();
        crate::quant::quantize_q8_0_into(&a, &mut serial);
        let mut team = Vec::new();
        quantize_q8_0_decode_into(&a, &mut team);
        assert_eq!(serial, team);

        let mut gated = vec![0.0f32; len];
        silu_mul_into(&a, &b, &mut gated);
        crate::quant::quantize_q8_0_into(&gated, &mut serial);
        let mut fused_gated = vec![f32::NAN; len];
        silu_mul_quantize_q8_0_decode_into(&a, &b, &mut fused_gated, &mut team);
        assert_eq!(bits(&gated), bits(&fused_gated));
        assert_eq!(serial, team);
    }
}

// -- decode path correctness --------------------------------------------

#[test]
fn decode_path_matches_scalar() {
    use rand::Rng;
    let out_features = 66;
    // Five input blocks and 66 outputs exercise both 4x4 tile tails.
    let in_features = 160;
    let data = random_q8_0_data(out_features, in_features);
    let w = QuantizedWeight::try_new(data.clone(), vec![out_features, in_features]).unwrap();

    let mut rng = rand::thread_rng();
    let x: Vec<f32> = (0..in_features)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect();
    let qx = quantized_input(&x);

    let mut decode_out = vec![0.0f32; out_features];
    matmul_q8_0_decode(&qx, &w, &mut decode_out);

    let mut scalar_out = vec![0.0f32; out_features];
    matmul_q8_0_decode_scalar(
        &qx,
        &data,
        out_features,
        in_features / Q8_0_BLOCK_SIZE,
        &mut scalar_out,
    );

    for (i, (d, s)) in decode_out.iter().zip(scalar_out.iter()).enumerate() {
        let diff = (d - s).abs();
        let max_val = d.abs().max(s.abs()).max(1.0);
        // SIMD FMA accumulates in a different order than scalar;
        // allow 1e-3 relative tolerance for the dot product.
        assert!(
            diff / max_val < 1e-3,
            "decode mismatch at {i}: decode={d} scalar={s} diff={diff} rel={}",
            diff / max_val
        );
    }
}

#[test]
fn interleaved_decode_matches_row_contiguous_decode() {
    use rand::Rng;

    // A non-power-of-two row count exercises the aligned parallel chunks
    // and the final partial interleave stripe.
    let out_features = 70;
    let in_features = 256;
    let data = random_q8_0_data(out_features, in_features);
    let weight = QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap();
    let interleaved = QuantizedWeightInterleaved::from_quantized(&weight);
    let mut rng = rand::thread_rng();
    let input = (0..in_features)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect::<Vec<_>>();
    let quantized = quantized_input(&input);

    let mut expected = vec![0.0; out_features];
    matmul_q8_0_decode(&quantized, &weight, &mut expected);
    let mut actual = vec![0.0; out_features];
    matmul_q8_0_decode_interleaved_parallel(&quantized, &interleaved, &mut actual);

    assert_close("interleaved q8 decode", &actual, &expected, 0.1, 0.02);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn arm_interleaved_exact_for_unaligned_output_ranges() {
    if !interleaved_q8_0_supported() {
        return;
    }
    for outputs in [1, 3, 4, 5, 17, 70] {
        for blocks in [1, 7, 64] {
            let mut bytes = vec![0u8; outputs * blocks * 34];
            for (i, b) in bytes.chunks_exact_mut(34).enumerate() {
                b[..2]
                    .copy_from_slice(&[1u16, 0x03ff, 0x0400, 0x3c00, 0xbc00][i % 5].to_le_bytes());
                for (j, q) in b[2..].iter_mut().enumerate() {
                    *q = (i * 37 + j * 13) as u8;
                }
            }
            let input = bytes[..blocks * 34].to_vec();
            let weight = QuantizedWeight::try_new(bytes, vec![outputs, blocks * 32]).unwrap();
            let packed = QuantizedWeightInterleaved::from_quantized(&weight);
            let mut expected = vec![0.0; outputs];
            matmul_q8_0_decode(&input, &weight, &mut expected);
            for start in 0..outputs {
                for len in [1, (outputs - start).min(5), outputs - start] {
                    let mut actual = vec![0.0; len];
                    matmul_q8_0_decode_interleaved(&input, &packed, &mut actual, start);
                    assert_eq!(
                        actual,
                        expected[start..start + len],
                        "outputs={outputs} blocks={blocks} start={start} len={len}"
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn arm_f16_attention_is_bit_exact_with_ordered_scalar() {
    if !is_aarch64_feature_detected!("fp16") {
        return;
    }
    // Every finite half value, plus all SIMD tail lengths and zero signs.
    let values: Vec<f16> = (0..=u16::MAX)
        .filter(|b| b & 0x7c00 != 0x7c00)
        .map(f16::from_bits)
        .collect();
    for offset in 0..4 {
        for len in [0, 1, 3, 4, 5, 7, 64, 129, values.len() - offset] {
            let src = &values[offset..offset + len];
            let query: Vec<f32> = (0..len)
                .map(|i| [0.0, -0.0, 0.375, -1.25, 1e-30][i % 5])
                .collect();
            let expected: f32 = query.iter().zip(src).map(|(a, b)| a * b.to_f32()).sum();
            assert_eq!(
                dot_product_f16(&query, src).to_bits(),
                expected.to_bits(),
                "dot offset={offset} len={len}"
            );
            for weight in [0.0, -0.0, 0.375, -1.25, 1e-30] {
                let mut expected = query.clone();
                for (a, b) in expected.iter_mut().zip(src) {
                    *a += weight * b.to_f32();
                }
                let mut actual = query.clone();
                weighted_add_f16(&mut actual, src, weight);
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "add offset={offset} len={len} weight={weight}"
                );
            }
        }
    }
}

#[test]
fn packed16_decode_matches_row_contiguous_decode() {
    if !packed_q8_0_vnni_supported() {
        return;
    }

    for (out_features, in_features) in [(70, 256), (32, 2_048), (32, 8_192)] {
        let data = random_q8_0_data(out_features, in_features);
        let weight = QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap();
        let packed = QuantizedWeightVnni::from_quantized(&weight);
        assert_eq!(
            packed.byte_len(),
            out_features.div_ceil(VNNI_OUT_TILE) * VNNI_OUT_TILE * in_features / Q8_0_BLOCK_SIZE
                * Q8_0_TYPE_SIZE
        );
        let input = patterned_vec(in_features, 0.625);
        let quantized = quantized_input(&input);

        let mut expected = vec![0.0; out_features];
        matmul_q8_0_decode(&quantized, &weight, &mut expected);
        let mut actual = vec![0.0; out_features];
        matmul_q8_0_decode_packed16_parallel(&quantized, &packed, &mut actual);

        for (index, (packed_value, row_value)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                packed_value.to_bits(),
                row_value.to_bits(),
                "packed Q8 decode mismatch at row {index} for {out_features}x{in_features}: \
                 packed={packed_value} row={row_value}"
            );
        }
    }
}

#[test]
fn decode_path_matches_packed_batch_path() {
    use crate::backend::{Backend, CpuBackend};
    use rand::Rng;

    let out_features = 66;
    let in_features = 256;
    let data = random_q8_0_data(out_features, in_features);
    let w = QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap();

    let mut rng = rand::thread_rng();
    let x: Vec<f32> = (0..in_features)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect();
    let qx = quantized_input(&x);

    let mut decode_out = vec![0.0f32; out_features];
    matmul_q8_0_decode(&qx, &w, &mut decode_out);

    // Two identical rows exercise the packed multi-row scheduling path.
    let mut x2 = Vec::with_capacity(in_features * 2);
    x2.extend_from_slice(&x);
    x2.extend_from_slice(&x);
    let backend = CpuBackend;
    let prefill = backend
        .matmul_q8_0(
            &crate::tensor::CpuTensor::from_data(vec![2, in_features], x2),
            &w,
        )
        .unwrap();
    let prefill_first_row = &prefill.data()[..out_features];
    assert_close(
        "q8 decode vs packed batch",
        &decode_out,
        prefill_first_row,
        0.1,
        0.02,
    );
}

#[test]
fn tiled_q8_batch_matches_independent_rows() {
    use rand::Rng;

    let rows = 8;
    let out_features = 64;
    let in_features = 256;
    let data = random_q8_0_data(out_features, in_features);
    let w = QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap();
    let mut rng = rand::thread_rng();
    let input = (0..rows * in_features)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
        .collect::<Vec<_>>();
    let packed = quantized_input(&input);
    let encoded_row_len = in_features / Q8_0_BLOCK_SIZE * Q8_0_TYPE_SIZE;

    let mut expected = vec![0.0; rows * out_features];
    for row in 0..rows {
        matmul_q8_0_decode(
            &packed[row * encoded_row_len..(row + 1) * encoded_row_len],
            &w,
            &mut expected[row * out_features..(row + 1) * out_features],
        );
    }

    let mut got = vec![0.0; rows * out_features];
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| matmul_q8_0_batch(&packed, rows, &w, &mut got));

    assert_close("tiled q8 batch", &got, &expected, 1e-5, 1e-5);
}

/// Deterministic values spanning signs, magnitudes and subnormal-ish
/// ranges, so FMA-vs-separate rounding differences would show.
fn rope_test_values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            let scale = [1.0e-3, 0.37, 1.0, 17.5, 3.0e4][i % 5];
            unit * scale
        })
        .collect()
}

fn rope_tables(half: usize, position: usize) -> (Vec<f32>, Vec<f32>) {
    (0..half)
        .map(|d| {
            let freq = 10_000f32.powf(-(2.0 * d as f32) / (2 * half) as f32);
            let angle = position as f32 * freq;
            (angle.cos(), angle.sin())
        })
        .unzip()
}

/// The dispatched adjacent-pair RoPE (NEON / AVX2+FMA / scalar) must be
/// bit-identical to the scalar reference loop, including heads whose pair
/// count leaves a tail shorter than one SIMD vector.
#[test]
fn rope_adjacent_pair_simd_is_bit_identical_to_scalar() {
    for head_dim in [2usize, 4, 6, 8, 10, 14, 30, 64, 66, 128, 130] {
        for n_heads in [1usize, 3, 8] {
            for position in [0usize, 1, 7, 513] {
                let (cos, sin) = rope_tables(head_dim / 2, position);
                let input = rope_test_values(n_heads * head_dim, (head_dim * 31 + n_heads) as u32);
                let mut expected = input.clone();
                for h in 0..n_heads {
                    for d in 0..head_dim / 2 {
                        let i0 = h * head_dim + 2 * d;
                        let (x0, x1) = (expected[i0], expected[i0 + 1]);
                        expected[i0] = x0.mul_add(cos[d], -(x1 * sin[d]));
                        expected[i0 + 1] = x0.mul_add(sin[d], x1 * cos[d]);
                    }
                }
                let mut simd = input.clone();
                super::rope_adjacent_pair(&mut simd, n_heads, head_dim, &cos, &sin);
                let mut scalar = input;
                super::rope_adjacent_pair_scalar(&mut scalar, n_heads, head_dim, &cos, &sin);
                for (i, ((got, reference), want)) in
                    simd.iter().zip(&scalar).zip(&expected).enumerate()
                {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "dispatched head_dim={head_dim} n_heads={n_heads} pos={position} i={i}"
                    );
                    assert_eq!(
                        reference.to_bits(),
                        want.to_bits(),
                        "scalar head_dim={head_dim} n_heads={n_heads} pos={position} i={i}"
                    );
                }
            }
        }
    }
}

/// Split-half RoPE dispatch stays bit-identical to its scalar loop too
/// (the eager prefill now routes both layouts through `simd`).
#[test]
fn rope_split_half_simd_is_bit_identical_to_scalar() {
    for head_dim in [2usize, 6, 16, 18, 64, 128, 130] {
        for n_heads in [1usize, 3] {
            let (cos, sin) = rope_tables(head_dim / 2, 11);
            let input = rope_test_values(n_heads * head_dim, head_dim as u32);
            let half = head_dim / 2;
            let mut expected = input.clone();
            for h in 0..n_heads {
                for d in 0..half {
                    let (i0, i1) = (h * head_dim + d, h * head_dim + d + half);
                    let (x0, x1) = (expected[i0], expected[i1]);
                    expected[i0] = x0.mul_add(cos[d], -(x1 * sin[d]));
                    expected[i1] = x0.mul_add(sin[d], x1 * cos[d]);
                }
            }
            let mut got = input;
            super::rope_split_half(&mut got, n_heads, head_dim, &cos, &sin);
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got_bits, want_bits, "head_dim={head_dim} n_heads={n_heads}");
        }
    }
}

#[test]
fn scale_weight_mul_simd_matches_scalar() {
    let n = 1536; // embed_dim
    let scale = 0.5_f32.sqrt().recip(); // typical rstd value
                                        // Realistic values: x ~ N(0, 1), weight from output_norm (0..118,
                                        // the max weight seen in GGUF).
    let x: Vec<f32> = (0..n).map(|i| (i as f32).sin() * 2.0).collect();
    let weight: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 118.0).collect();
    let mut simd_out = vec![0.0f32; n];
    crate::simd::scale_weight_mul(&x, scale, &weight, &mut simd_out);
    for i in 0..n {
        let scalar = x[i] * scale * weight[i];
        assert!(
            (simd_out[i] - scalar).abs() < 1e-6,
            "mismatch at {i}: simd={} scalar={scalar}",
            simd_out[i]
        );
    }
}

#[test]
fn sum_squares_simd_matches_scalar() {
    let n = 1536;
    let mut x = vec![0.0f32; n];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (i as f32).sin() * 100.0;
    }
    let simd = super::sum_squares(&x);
    let scalar: f32 = x.iter().map(|v| v * v).sum();
    let diff = (simd - scalar).abs();
    let rel = if scalar > 0.0 { diff / scalar } else { diff };
    assert!(
        rel < 1e-6,
        "sum_squares mismatch: simd={}, scalar={}, diff={}",
        simd,
        scalar,
        diff
    );
}
