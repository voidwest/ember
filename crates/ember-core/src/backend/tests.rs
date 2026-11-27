use super::*;

fn test_q8_weight(out_features: usize, in_features: usize, phase: f32) -> QuantizedWeight {
    let values = (0..out_features * in_features)
        .map(|index| {
            let value = index as f32 * 0.03125 + phase;
            value.sin() * 0.75 + value.cos() * 0.125
        })
        .collect::<Vec<_>>();
    let mut data = Vec::new();
    crate::quant::quantize_q8_0_into(&values, &mut data);
    QuantizedWeight::new(data, vec![out_features, in_features])
}

#[test]
fn packed_pair_matches_generic_for_measured_short_prompt_regime() {
    if !crate::simd::packed_q8_0_vnni_supported() {
        return;
    }

    let backend = CpuBackend;
    let rows = 6;
    let in_features = 64;
    let first = test_q8_weight(64, in_features, 0.25);
    let second = test_q8_weight(64, in_features, 1.5);
    let packed_first = QuantizedWeightVnni::from_quantized(&first);
    let packed_second = QuantizedWeightVnni::from_quantized(&second);
    let input = CpuTensor::from_data(
        vec![rows, in_features],
        (0..rows * in_features)
            .map(|index| (index as f32 * 0.017).sin())
            .collect(),
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let expected = pool
        .install(|| backend.matmul_q8_0_pair(&input, &first, &second))
        .unwrap();
    let actual = pool
        .install(|| backend.matmul_q8_0_packed_pair(&input, &packed_first, &packed_second))
        .unwrap()
        .expect("four-thread, six-row input should use packed pair");

    assert_eq!(actual, expected);
}

#[test]
fn packed_pair_defers_to_generic_batch_kernel_after_six_rows() {
    if !crate::simd::packed_q8_0_vnni_supported() {
        return;
    }

    let backend = CpuBackend;
    let first = test_q8_weight(16, 32, 0.25);
    let second = test_q8_weight(16, 32, 1.5);
    let input = CpuTensor::zeroes(&[8, 32]);
    let actual = backend
        .matmul_q8_0_packed_pair(
            &input,
            &QuantizedWeightVnni::from_quantized(&first),
            &QuantizedWeightVnni::from_quantized(&second),
        )
        .unwrap();

    assert!(actual.is_none());
}

#[test]
fn cached_attention_into_matches_tensor_api() {
    let backend = CpuBackend;
    let query = CpuTensor::from_data(vec![1, 4], vec![0.25, -0.5, 0.75, 1.0]);
    let cached_k = [0.5, 0.25, -0.75, 1.0, 1.0, 0.0, 0.5, -0.5].map(f16::from_f32);
    let cached_v = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0].map(f16::from_f32);
    let spec = CachedAttentionSpec {
        n_heads: 2,
        n_kv_heads: 2,
        head_dim: 2,
        max_seq_len: 2,
        total_seq_len: 2,
    };

    let expected = backend
        .cached_causal_attention(&query, &cached_k, &cached_v, spec)
        .unwrap();
    let mut scratch = Vec::new();
    let mut actual = vec![0.0; query.len()];
    backend
        .cached_causal_attention_into(
            query.data(),
            &cached_k,
            &cached_v,
            spec,
            &mut scratch,
            &mut actual,
        )
        .unwrap();

    assert_eq!(actual, expected.data());
}

/// Batched decode attention over sequences of different lengths equals
/// attending each sequence on its own, bit for bit, on every route the
/// single call can take (serial below the parallel threshold, decode
/// team above it).
#[test]
fn batched_decode_attention_matches_per_sequence_attention() {
    let backend = CpuBackend;
    let (n_heads, n_kv_heads, head_dim, max_seq_len) = (8usize, 2usize, 64usize, 96usize);
    let width = n_heads * head_dim;
    let cache_len = n_kv_heads * max_seq_len * head_dim;
    let value =
        |seed: usize, index: usize| ((seed * 7919 + index * 104_729) % 2003) as f32 / 1001.0 - 1.0;
    let lengths = [1usize, 5, 40, 96, 63, 17];
    let queries: Vec<Vec<f32>> = (0..lengths.len())
        .map(|seed| (0..width).map(|index| value(seed + 1, index)).collect())
        .collect();
    let caches: Vec<(Vec<f16>, Vec<f16>)> = (0..lengths.len())
        .map(|seed| {
            let fill = |salt: usize| {
                (0..cache_len)
                    .map(|index| f16::from_f32(value(seed * 31 + salt, index)))
                    .collect::<Vec<_>>()
            };
            (fill(3), fill(5))
        })
        .collect();
    let spec = |total_seq_len: usize| CachedAttentionSpec {
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq_len,
        total_seq_len,
    };
    let expected: Vec<Vec<f32>> = lengths
        .iter()
        .enumerate()
        .map(|(index, &length)| {
            let mut out = vec![0.0; width];
            backend
                .cached_causal_attention_into(
                    &queries[index],
                    &caches[index].0,
                    &caches[index].1,
                    spec(length),
                    &mut Vec::new(),
                    &mut out,
                )
                .unwrap();
            out
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        crate::decode_pool::session(|| {
            let mut outs = vec![vec![f32::NAN; width]; lengths.len()];
            let mut items: Vec<DecodeAttention<'_>> = outs
                .iter_mut()
                .enumerate()
                .map(|(index, out)| DecodeAttention {
                    q: &queries[index],
                    cached_k: &caches[index].0,
                    cached_v: &caches[index].1,
                    spec: spec(lengths[index]),
                    out,
                })
                .collect();
            // The team may be held by a concurrently running test.
            let ran = backend
                .cached_decode_attention_batch_into(&mut items)
                .unwrap();
            drop(items);
            if ran {
                for (index, (got, want)) in outs.iter().zip(&expected).enumerate() {
                    let got: Vec<u32> = got.iter().map(|value| value.to_bits()).collect();
                    let want: Vec<u32> = want.iter().map(|value| value.to_bits()).collect();
                    assert_eq!(got, want, "sequence {index}");
                }
            }
        });
    });
}

#[cfg(test)]
mod reference_softmax_diagnostic {
    #[test]
    #[ignore = "requires pinned reference attention score captures"]
    fn diagnose_reference_input_softmax() {
        let source = std::env::var("EMBER_PARITY_REFERENCE_TENSORS").unwrap();
        let capture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&source).unwrap()).unwrap();
        let rows = capture["tensors"].as_array().unwrap();
        let values = |name: &str| -> Vec<f32> {
            let row = rows
                .iter()
                .find(|r| r["name"] == name)
                .expect("required score row");
            assert_eq!(row["step"], 10);
            assert_eq!(row["row"], 0);
            assert_eq!(row["width"], 16);
            let data: Vec<f32> = serde_json::from_value(row["values"].clone()).unwrap();
            assert_eq!(data.len(), 16);
            assert!(data.iter().all(|x| x.is_finite()));
            data
        };
        let raw = values("kq-0");
        let expected = values("kq_soft_max-0");
        // Pinned Llama 3.2 1B: 64 dimensions/head, scale=1/sqrt(64), no ALiBi.
        let scale = 1.0 / 64.0_f32.sqrt();
        let mut actual: Vec<f32> = raw.iter().map(|v| v * scale).collect();
        super::softmax_prefix(&mut actual, raw.len());
        let vector = |name: &str| -> Vec<f32> {
            let row = rows
                .iter()
                .find(|r| r["name"] == name)
                .expect("rotation capture required");
            let data: Vec<f32> = serde_json::from_value(row["values"].clone()).unwrap();
            assert!(data.len() >= 64 && data.iter().all(|x| x.is_finite()));
            data
        };
        let q = vector("q_rope-0");
        let k = vector("k_rope-0");
        let dot = crate::simd::dot_product(&q[..64], &k[..64]);
        let cached_k: Vec<half::f16> = k[..64].iter().copied().map(half::f16::from_f32).collect();
        let mixed_dot = crate::simd::dot_product_f16(&q[..64], &cached_k);
        let rounded_q: Vec<f32> = q[..64]
            .iter()
            .map(|x| half::f16::from_f32(*x).to_f32())
            .collect();
        let both_rounded_dot = crate::simd::dot_product_f16(&rounded_q, &cached_k);
        let output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(std::env::var("EMBER_PARITY_DIAGNOSTIC").unwrap())
            .unwrap();
        serde_json::to_writer_pretty(output, &serde_json::json!({
            "scope": "actual Ember softmax on identical reference first-head scores at Arabic evaluation 10",
            "reference_source": source,
            "dot_product_check": {"scope": "current query/key first head at position 15",
                "actual": dot, "reference": raw[15]},
            "mixed_precision_check": {"scope": "identical captured Q/K before optional f16 conversion",
                "f32_query_f16_key": mixed_dot, "f16_query_f16_key": both_rounded_dot},
            "records": [{"operation": "softmax", "scale": scale, "input": raw,
                "actual": actual, "reference": expected}]
        })).unwrap();
    }
}
