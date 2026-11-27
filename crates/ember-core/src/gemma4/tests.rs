use super::*;
use crate::experiments::test_support::RecordingExperiment;
use crate::experiments::{
    ExecutionPhase, ExperimentHook, ModelContext, ModelFamily, TracingState, ZeroLayerOutput,
    ZeroLayerOutputSpec, ZeroLayerOutputStage,
};
use std::collections::HashMap;

fn loader_with(metadata: HashMap<String, GgufValue>) -> GgufLoader {
    GgufLoader::for_test(metadata, HashMap::new())
}

/// Metadata for the 2-wide, single-head, 4-token tiny Gemma 4 fixture.
fn tiny_metadata(block_count: u32) -> HashMap<String, GgufValue> {
    [
        ("block_count", block_count),
        ("embedding_length", 2),
        ("attention.head_count", 1),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 2),
        ("feed_forward_length", 2),
        ("vocab_size", 4),
        ("context_length", 8),
    ]
    .into_iter()
    .map(|(name, value)| (format!("gemma4.{name}"), GgufValue::U32(value)))
    .collect()
}

fn tiny_tensor(shape: &[usize], value: f32) -> LoadedTensor {
    LoadedTensor::F32(CpuTensor::from_data(
        shape.to_vec(),
        vec![value; shape.iter().product()],
    ))
}

fn tiny_weight(shape: &[usize]) -> LoadedTensor {
    let rows = shape[0];
    let cols = shape[1];
    let mut data = vec![0.0; rows * cols];
    for i in 0..rows.min(cols) {
        data[i * cols + i] = 0.1;
    }
    LoadedTensor::F32(CpuTensor::from_data(shape.to_vec(), data))
}

fn insert_tiny_gemma4_block_tensors(
    tensors: &mut HashMap<String, LoadedTensor>,
    layer: usize,
    include_kv: bool,
) {
    for name in ["attn_q", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
        tensors.insert(
            format!("blk.{}.{}.weight", layer, name),
            tiny_weight(&[2, 2]),
        );
    }
    if include_kv {
        for name in ["attn_k", "attn_v"] {
            tensors.insert(
                format!("blk.{}.{}.weight", layer, name),
                tiny_weight(&[2, 2]),
            );
        }
    }
    for name in [
        "attn_q_norm",
        "attn_k_norm",
        "attn_norm",
        "attn_post_norm",
        "ffn_norm",
        "ffn_post_norm",
    ] {
        tensors.insert(
            format!("blk.{}.{}.weight", layer, name),
            tiny_tensor(&[2], 1.0),
        );
    }
}

fn insert_tiny_gemma4_tensors(tensors: &mut HashMap<String, LoadedTensor>) {
    tensors.insert(
        "token_embd.weight".to_string(),
        LoadedTensor::F32(CpuTensor::from_data(
            // GGUF logical dims are [embed, vocab] with the vocab dim
            // contiguous (data stays vocab-major as written below).
            vec![2, 4],
            vec![0.1, 0.2, 0.2, 0.1, 0.0, 0.3, 0.3, 0.0],
        )),
    );
    tensors.insert("output_norm.weight".to_string(), tiny_tensor(&[2], 1.0));
    tensors.insert("output.weight".to_string(), tiny_weight(&[2, 4]));
    insert_tiny_gemma4_block_tensors(tensors, 0, true);
}

fn tiny_gemma4_model() -> Gemma4<CpuBackend> {
    let mut metadata = tiny_metadata(1);
    metadata.insert(
        "gemma4.attention.sliding_window".to_string(),
        GgufValue::U32(2),
    );
    metadata.insert(
        "gemma4.final_logit_softcap".to_string(),
        GgufValue::F32(10.0),
    );

    let mut tensors = HashMap::new();
    insert_tiny_gemma4_tensors(&mut tensors);
    Gemma4::from_loader(GgufLoader::for_test(metadata, tensors)).unwrap()
}

fn tiny_heterogeneous_gemma4_loader() -> GgufLoader {
    let mut metadata = HashMap::new();
    for (name, value) in [
        ("block_count", 4),
        ("embedding_length", 4),
        ("attention.head_count", 4),
        ("attention.local_head_count_kv", 1),
        ("attention.global_head_count_kv", 2),
        ("attention.key_length_swa", 2),
        ("attention.global_key_length", 4),
        ("attention.sliding_window", 2),
        ("attention.shared_kv_layers", 2),
        ("feed_forward_length", 4),
        ("vocab_size", 4),
        ("context_length", 16),
    ] {
        metadata.insert(format!("gemma4.{name}"), GgufValue::U32(value));
    }
    metadata.insert(
        "gemma4.attention.sliding_window_pattern".into(),
        GgufValue::Array(
            [true, false, true, false]
                .into_iter()
                .map(GgufValue::Bool)
                .collect(),
        ),
    );
    let weight = |input: usize, output: usize, phase: f32| {
        LoadedTensor::F32(CpuTensor::from_data(
            vec![input, output],
            (0..input * output)
                .map(|index| (index as f32 * 0.37 + phase).sin() * 0.125)
                .collect(),
        ))
    };
    let mut tensors = HashMap::from([
        ("token_embd.weight".into(), weight(4, 4, 0.0)),
        ("output.weight".into(), weight(4, 4, 0.25)),
        ("output_norm.weight".into(), tiny_tensor(&[4], 1.0)),
    ]);
    for layer in 0..4 {
        let (head_dim, kv_heads) = if layer % 2 == 0 { (2, 1) } else { (4, 2) };
        for (name, input, output) in [
            ("attn_q", 4, 4 * head_dim),
            ("attn_output", 4 * head_dim, 4),
            ("ffn_gate", 4, 4),
            ("ffn_up", 4, 4),
            ("ffn_down", 4, 4),
        ] {
            tensors.insert(
                format!("blk.{layer}.{name}.weight"),
                weight(input, output, layer as f32),
            );
        }
        if layer < 2 {
            for name in ["attn_k", "attn_v"] {
                tensors.insert(
                    format!("blk.{layer}.{name}.weight"),
                    weight(4, kv_heads * head_dim, 0.5 + layer as f32),
                );
            }
        }
        for (name, width) in [
            ("attn_q_norm", head_dim),
            ("attn_k_norm", head_dim),
            ("attn_norm", 4),
            ("attn_post_norm", 4),
            ("ffn_norm", 4),
            ("ffn_post_norm", 4),
        ] {
            tensors.insert(
                format!("blk.{layer}.{name}.weight"),
                tiny_tensor(&[width], 1.0),
            );
        }
    }
    let mut loader = loader_with(metadata);
    loader.tensors = tensors;
    loader
}

fn tiny_heterogeneous_gemma4_model() -> Gemma4<CpuBackend> {
    Gemma4::from_loader(tiny_heterogeneous_gemma4_loader()).unwrap()
}

#[test]
fn heterogeneous_shared_cache_matches_padded_prefill_decode_and_rollback() {
    let backend = CpuBackend;
    let model = tiny_heterogeneous_gemma4_model();
    let mut compact = model.create_cache(&backend, 16);
    let mut padded = KVCache::new(4, 2, 4, 16);
    assert_eq!(compact.storage_bytes(), (2 + 2 * 4) * 16 * 4);
    assert_eq!(padded.storage_bytes(), 4 * 2 * 4 * 16 * 4);
    assert_eq!(compact.get(0).0.as_ptr(), compact.get(2).0.as_ptr());
    assert_eq!(compact.get(1).1.as_ptr(), compact.get(3).1.as_ptr());

    let assert_active_equal = |compact: &KVCache, padded: &KVCache| {
        assert_eq!(compact.cursor(), padded.cursor());
        for (layer, block) in model.blocks.iter().enumerate() {
            let source = block.attn.shared_source_layer.unwrap_or(layer);
            let (actual_k, actual_v) = compact.get(layer);
            let (expected_k, expected_v) = padded.get(source);
            for head in 0..block.attn.n_kv_heads {
                for pos in 0..compact.cursor() {
                    let actual = (head * 16 + pos) * compact.layer_head_dim(layer);
                    let expected = (head * 16 + pos) * padded.layer_head_dim(source);
                    let width = block.attn.head_dim;
                    assert_eq!(
                        &actual_k[actual..actual + width],
                        &expected_k[expected..expected + width]
                    );
                    assert_eq!(
                        &actual_v[actual..actual + width],
                        &expected_v[expected..expected + width]
                    );
                }
            }
        }
    };
    for tokens in [&[0, 1, 2, 3, 0][..], &[1][..], &[2][..]] {
        let start = compact.cursor();
        let actual = model
            .forward_with_cache(&backend, tokens, &mut compact, start)
            .unwrap();
        let expected = model
            .forward_with_cache(&backend, tokens, &mut padded, start)
            .unwrap();
        assert_eq!(actual, expected);
        assert!(actual.data.iter().any(|value| *value != 0.0));
        assert_active_equal(&compact, &padded);
    }
    compact.truncate_to(3);
    padded.truncate_to(3);
    let actual = model
        .forward_with_cache(&backend, &[2, 1], &mut compact, 3)
        .unwrap();
    let expected = model
        .forward_with_cache(&backend, &[2, 1], &mut padded, 3)
        .unwrap();
    assert_eq!(actual, expected);
    assert_active_equal(&compact, &padded);
    let mut fresh = model.create_cache(&backend, 16);
    let replay = model
        .forward_with_cache(&backend, &[0, 1, 2, 2, 1], &mut fresh, 0)
        .unwrap();
    assert_eq!(actual.data, replay.data[3 * 4..]);
    assert_active_equal(&fresh, &padded);
}

#[test]
fn gemma_cached_attention_parallel_matches_serial_with_window_and_padding() {
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let max_seq_len = 1540;
    let total_seq_len = 1536;
    let n_heads = 4;
    let n_kv_heads = 2;
    let head_dim = 8;
    let cache_head_dim = 12;
    let keys: Vec<_> = (0..n_kv_heads * max_seq_len * cache_head_dim)
        .map(|index| f16::from_f32((index as f32 * 0.07).sin()))
        .collect();
    let values: Vec<_> = (0..keys.len())
        .map(|index| f16::from_f32((index as f32 * 0.03).cos()))
        .collect();
    for seq_len in [1, 5] {
        let query = CpuTensor::from_data(
            vec![seq_len, n_heads * head_dim],
            (0..seq_len * n_heads * head_dim)
                .map(|index| (index as f32 * 0.13).sin())
                .collect(),
        );
        let mut full_output = None;
        for sliding_window in [None, Some(1024)] {
            let attended = sliding_window.unwrap_or(total_seq_len);
            assert!(
                !serial.install(|| crate::backend::should_parallel_attention(
                    n_heads, seq_len, attended, head_dim
                ))
            );
            assert!(
                parallel.install(|| crate::backend::should_parallel_attention(
                    n_heads, seq_len, attended, head_dim
                ))
            );
            let run = || {
                cached_attention_with_scratch(
                    &CpuBackend,
                    &query,
                    &keys,
                    &values,
                    Gemma4CachedAttentionSpec {
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cache_head_dim,
                        max_seq_len,
                        total_seq_len,
                        sliding_window,
                        scale: 0.5,
                    },
                    &mut Vec::new(),
                )
                .unwrap()
            };
            let expected = serial.install(run);
            assert_eq!(parallel.install(run), expected);
            if let Some(full) = &full_output {
                assert_ne!(
                    &expected, full,
                    "window must actually exclude visible values"
                );
            } else {
                full_output = Some(expected);
            }
        }
    }
}

#[test]
fn cached_local_attention_masks_prefix_and_future_with_padded_heads() {
    let query = CpuTensor::zeroes(&[2, 8]);
    let keys = vec![f16::ZERO; 2 * 5 * 4];
    let mut values = vec![f16::from_f32(-999.0); keys.len()];
    for head in 0..2 {
        for (pos, value) in [1.0, 2.0, 4.0, 8.0].into_iter().enumerate() {
            let offset = (head * 5 + pos) * 4;
            values[offset..offset + 2].fill(f16::from_f32(value + head as f32 * 16.0));
        }
    }
    let actual = cached_attention_with_scratch(
        &CpuBackend,
        &query,
        &keys,
        &values,
        Gemma4CachedAttentionSpec {
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 2,
            cache_head_dim: 4,
            max_seq_len: 5,
            total_seq_len: 4,
            sliding_window: Some(2),
            scale: 1.0,
        },
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(
        actual.data,
        [3.0, 3.0, 3.0, 3.0, 19.0, 19.0, 19.0, 19.0, 6.0, 6.0, 6.0, 6.0, 22.0, 22.0, 22.0, 22.0]
    );
}

#[test]
fn owned_qkv_normalization_reuses_cpu_payloads() {
    let query = CpuTensor::from_data(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    let pointer = query.data.as_ptr();
    let norm = CpuTensor::from_data(vec![2], vec![1.0, 0.5]);
    let cos = CpuTensor::from_data(vec![1, 1], vec![0.0]);
    let sin = CpuTensor::from_data(vec![1, 1], vec![1.0]);
    let output =
        apply_rope_and_qk_norm(&CpuBackend, query, &norm, &cos, &sin, 0, 2, 2, 1e-6).unwrap();
    assert_eq!(pointer, output.data.as_ptr());
    let mut expected = Vec::new();
    for (a, b) in [(1.0_f32, 2.0_f32), (3.0, 4.0)] {
        let rstd = ((a * a + b * b) / 2.0 + 1e-6).sqrt().recip();
        expected.extend([-b * rstd * 0.5, a * rstd]);
    }
    assert_eq!(output.data, expected);
    let value = CpuTensor::from_data(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    let pointer = value.data.as_ptr();
    let output = apply_v_rms_norm(&CpuBackend, value, 2, 2, 1e-6).unwrap();
    assert_eq!(pointer, output.data.as_ptr());
    assert_eq!(
        output.data,
        vec![
            expected[1],
            -expected[0] * 2.0,
            expected[3],
            -expected[2] * 2.0
        ]
    );
}

#[test]
fn gemma4_config_rejects_moe() {
    let mut metadata = HashMap::new();
    metadata.insert("gemma4.enable_moe_block".to_string(), GgufValue::Bool(true));
    let err = Gemma4Config::from_gguf_metadata(&loader_with(metadata)).unwrap_err();
    assert!(err.to_string().contains("MoE Gemma 4"));
}

#[test]
fn gemma4_config_accepts_dense_text_defaults() {
    let mut metadata = HashMap::new();
    metadata.insert("gemma4.block_count".to_string(), GgufValue::U32(2));
    metadata.insert("gemma4.embedding_length".to_string(), GgufValue::U32(8));
    metadata.insert("gemma4.attention.head_count".to_string(), GgufValue::U32(2));
    metadata.insert("gemma4.vocab_size".to_string(), GgufValue::U32(16));
    let cfg = Gemma4Config::from_gguf_metadata(&loader_with(metadata)).unwrap();
    assert_eq!(cfg.n_layers, 2);
    assert_eq!(cfg.embed_dim, 8);
    assert_eq!(cfg.local_head_dim, 4);
}

#[test]
fn gemma4_config_rejects_odd_dimensions_and_large_rope_tables() {
    let mut metadata = HashMap::new();
    metadata.insert(
        "gemma4.attention.local_key_length".to_string(),
        GgufValue::U32(3),
    );
    let odd = Gemma4Config::from_gguf_metadata(&loader_with(metadata))
        .expect_err("odd local RoPE dimensions must be rejected");
    assert!(odd.to_string().contains("positive and even"), "{odd}");

    let mut metadata = HashMap::new();
    metadata.insert(
        "gemma4.context_length".to_string(),
        GgufValue::U32(1_000_000),
    );
    metadata.insert(
        "gemma4.attention.local_key_length".to_string(),
        GgufValue::U32(512),
    );
    metadata.insert(
        "gemma4.attention.global_key_length".to_string(),
        GgufValue::U32(512),
    );
    let oversized = Gemma4Config::from_gguf_metadata(&loader_with(metadata))
        .expect_err("RoPE table products must be bounded");
    assert!(
        oversized.to_string().contains("RoPE-table element limit"),
        "{oversized}"
    );
}

#[test]
fn gemma4_rejects_hostile_rope_factors_and_linear_shapes() {
    fn loader_with_rope(shape: Vec<usize>, values: Vec<f32>) -> GgufLoader {
        let mut metadata = HashMap::new();
        metadata.insert("gemma4.block_count".to_string(), GgufValue::U32(1));
        metadata.insert("gemma4.embedding_length".to_string(), GgufValue::U32(4));
        metadata.insert("gemma4.attention.head_count".to_string(), GgufValue::U32(1));
        metadata.insert("gemma4.attention.key_length".to_string(), GgufValue::U32(4));
        metadata.insert("gemma4.vocab_size".to_string(), GgufValue::U32(4));
        metadata.insert("gemma4.context_length".to_string(), GgufValue::U32(8));
        let tensors = HashMap::from([
            ("token_embd.weight".to_string(), tiny_tensor(&[4, 4], 0.1)),
            (
                "rope_freqs.weight".to_string(),
                LoadedTensor::F32(CpuTensor::from_data(shape, values)),
            ),
        ]);
        GgufLoader::for_test(metadata, tensors)
    }

    let shape_err = Gemma4::from_loader(loader_with_rope(vec![1], vec![1.0]))
        .err()
        .expect("wrong RoPE factor shape must be rejected");
    assert!(
        shape_err.to_string().contains("rope_freqs.weight"),
        "{shape_err}"
    );

    let value_err = Gemma4::from_loader(loader_with_rope(vec![2], vec![f32::INFINITY, 1.0]))
        .err()
        .expect("non-finite RoPE factors must be rejected");
    assert!(
        value_err.to_string().contains("finite positive factors"),
        "{value_err}"
    );

    let mut loader = GgufLoader::for_test(
        HashMap::new(),
        HashMap::from([("bad.weight".to_string(), tiny_tensor(&[4], 0.0))]),
    );
    let linear_err = take_gemma4_linear(&mut loader, "bad.weight")
        .err()
        .expect("non-2D linear weights must be rejected");
    assert!(
        linear_err.to_string().contains("2D linear weight"),
        "{linear_err}"
    );
}

#[test]
fn softcap_transforms_logits_and_reuses_allocation_when_enabled_or_disabled() {
    let backend = CpuBackend;
    let logits = CpuTensor::from_data(vec![1, 3], vec![-100.0, 0.0, 100.0]);
    let allocation = logits.data().as_ptr();
    let capped = softcap_logits(&backend, logits, Some(30.0)).unwrap();
    assert_eq!(capped.data().as_ptr(), allocation, "enabled softcap");
    assert!(capped.data()[0] > -30.0);
    assert_eq!(capped.data()[1], 0.0);
    assert!(capped.data()[2] < 30.0);

    let logits = CpuTensor::from_data(vec![1, 3], vec![1.0, 2.0, 3.0]);
    let allocation = logits.data().as_ptr();
    let uncapped = softcap_logits(&backend, logits, None).unwrap();
    assert_eq!(uncapped.data().as_ptr(), allocation, "disabled softcap");
}

#[test]
fn attention_scale_uses_query_pre_attn_scalar() {
    let mut metadata = HashMap::new();
    metadata.insert("gemma4.attention.head_count".into(), GgufValue::U32(1));
    metadata.insert("gemma4.embedding_length".into(), GgufValue::U32(4));
    metadata.insert("gemma4.attention.key_length".into(), GgufValue::U32(4));
    metadata.insert(
        "gemma4.attention.query_pre_attn_scalar".into(),
        GgufValue::F32(16.0),
    );
    let cfg = Gemma4Config::from_gguf_metadata(&loader_with(metadata)).unwrap();
    assert!((cfg.attention_scale - 0.25).abs() < 1e-6);
}

#[test]
fn qk_norm_happens_before_rope_and_uses_config_eps() {
    let backend = CpuBackend;
    let x = CpuTensor::from_data(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    let norm = CpuTensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let rope_cos = CpuTensor::from_data(vec![1, 2], vec![0.0, 1.0]);
    let rope_sin = CpuTensor::from_data(vec![1, 2], vec![1.0, 0.0]);
    let out =
        apply_rope_and_qk_norm(&backend, x, &norm, &rope_cos, &rope_sin, 0, 1, 4, 0.0).unwrap();
    // RMSNorm first: rstd = 1 / sqrt(mean([1, 4, 9, 16])).
    // RoPE then rotates the first/second half pairs with cos=[0,1],
    // sin=[1,0].
    let rstd = (7.5_f32).sqrt().recip();
    let expected = [-9.0 * rstd, 4.0 * rstd, 1.0 * rstd, 16.0 * rstd];
    for (got, expected) in out.data().iter().zip(expected) {
        assert!((got - expected).abs() < 1e-6);
    }
}

#[test]
fn kv_cache_strided_append_keeps_source_layer_values() {
    let mut cache = KVCache::new(2, 1, 4, 3);
    cache.append_with_head_dim(0, 0, &[1.0, 2.0], &[3.0, 4.0], 2);
    let (k, v) = cache.get(0);
    assert_eq!(
        k[..4]
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>(),
        [1.0, 2.0, 0.0, 0.0]
    );
    assert_eq!(
        v[..4]
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>(),
        [3.0, 4.0, 0.0, 0.0]
    );
}

#[test]
fn tiny_gemma4_loader_runs_forward_pass() {
    let model = tiny_gemma4_model();
    let backend = CpuBackend;
    let mut cache = model.create_cache(&backend, 4);
    let logits = model
        .forward_with_cache(&backend, &[0, 1], &mut cache, 0)
        .unwrap();
    assert_eq!(logits.shape(), &[2, 4]);
    assert!(logits.data().iter().all(|v| v.is_finite()));

    let mut trace_cache = model.create_cache(&backend, 4);
    assert!(crate::trace::enable_tracing("prefill", 0));
    let traced = model.forward_last_logits_with_cache(&backend, &[0, 1], &mut trace_cache, 0);
    let report = crate::trace::disable_tracing().expect("Gemma trace report");
    traced.unwrap();
    let names = report
        .events
        .iter()
        .map(|event| event.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "embedding",
            "attn_rms_norm",
            "qkv_proj",
            "rope_q_qk_norm",
            "rope_k_qk_norm",
            "kv_cache_store",
            "attention_score",
            "o_proj",
            "post_attn_rms_norm",
            "attn_residual_add",
            "ffn_rms_norm",
            "gate_up_proj",
            "gelu",
            "elemul",
            "down_proj",
            "post_ffn_rms_norm",
            "ffn_residual_add",
            "final_norm",
            "lm_head",
            "logit_softcap",
        ]
    );

    let groups = vec![vec![0, 1]];
    let hidden_only = model
        .forward_pooled_hidden_states(&backend, &[0, 1], &groups)
        .unwrap();
    let (with_logits, _) = model
        .forward_pooled_activations(&backend, &[0, 1], &groups)
        .unwrap();
    assert_eq!(hidden_only, with_logits);
}

#[test]
fn gemma_observation_hooks_preserve_logits_and_stage_order() {
    let model = tiny_gemma4_model();
    let backend = CpuBackend;
    let mut normal_cache = model.create_cache(&backend, 4);
    let normal = model
        .forward_last_logits_with_cache(&backend, &[0, 1], &mut normal_cache, 0)
        .unwrap();

    let model_context = ModelContext::new(ModelFamily::Gemma4, None, "gemma4", 1, 2);
    let execution = ExecutionContext::new(
        model_context,
        ExecutionPhase::Prefill,
        0,
        2,
        TracingState::Disabled,
    );
    let (experiment, records) = RecordingExperiment::new();
    let mut runner = ExperimentRunner::new(experiment);
    let mut observed_cache = model.create_cache(&backend, 4);
    let observed = ExperimentalForwardModel::forward_last_logits_with_experiment(
        &model,
        &backend,
        &[0, 1],
        &mut observed_cache,
        0,
        execution,
        &mut runner,
    )
    .unwrap();

    assert_eq!(observed, normal);
    let records = records.lock().unwrap();
    assert_eq!(
        records.iter().map(|record| record.hook).collect::<Vec<_>>(),
        [
            ExperimentHook::BeforeLayer,
            ExperimentHook::AfterAttention,
            ExperimentHook::AfterMlp,
            ExperimentHook::AfterLayer,
            ExperimentHook::BeforeLogits,
            ExperimentHook::AfterLogits,
        ]
    );
    assert_eq!(records[0].shape, Some([2, 2]));
    assert_eq!(records[4].shape, Some([1, 2]));
    assert_eq!(records[5].shape, Some([1, 4]));
    assert!(records
        .iter()
        .all(|record| record.phase == Some(ExecutionPhase::Prefill)));
}

/// Gemma's attention path does not fire the per-head stage, so a run
/// that asks for it fails instead of recording nothing.
#[test]
fn gemma_refuses_the_per_head_stage() {
    struct WantsHeads;
    impl crate::experiments::Experiment for WantsHeads {
        fn name(&self) -> &'static str {
            "wants-heads"
        }
        fn uses_activation_stage(&self, stage: crate::hook_types::ActivationStage) -> bool {
            stage == crate::hook_types::ActivationStage::BeforeOProj
        }
    }
    let model = tiny_gemma4_model();
    let backend = CpuBackend;
    let model_context = ModelContext::new(ModelFamily::Gemma4, None, "gemma4", 1, 2);
    let execution = ExecutionContext::new(
        model_context,
        ExecutionPhase::Prefill,
        0,
        2,
        TracingState::Disabled,
    );
    let mut runner = ExperimentRunner::new(WantsHeads);
    let mut cache = model.create_cache(&backend, 4);
    let error = ExperimentalForwardModel::forward_last_logits_with_experiment(
        &model,
        &backend,
        &[0, 1],
        &mut cache,
        0,
        execution,
        &mut runner,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("before-o-proj is not supported for gemma4"),
        "{error}"
    );
}

#[test]
fn zero_attention_intervention_changes_gemma_logits() {
    let model = tiny_gemma4_model();
    let backend = CpuBackend;
    let mut normal_cache = model.create_cache(&backend, 4);
    let normal = model
        .forward_last_logits_with_cache(&backend, &[0, 1], &mut normal_cache, 0)
        .unwrap();

    let model_context = ModelContext::new(ModelFamily::Gemma4, None, "gemma4", 1, 2);
    let mut runner = ExperimentRunner::new(ZeroLayerOutput::new(ZeroLayerOutputSpec::new(
        0,
        ZeroLayerOutputStage::Attention,
    )));
    runner.on_model_loaded(&model_context).unwrap();
    let execution = ExecutionContext::new(
        model_context,
        ExecutionPhase::Prefill,
        0,
        2,
        TracingState::Disabled,
    );
    let mut experiment_cache = model.create_cache(&backend, 4);
    let intervened = ExperimentalForwardModel::forward_last_logits_with_experiment(
        &model,
        &backend,
        &[0, 1],
        &mut experiment_cache,
        0,
        execution,
        &mut runner,
    )
    .unwrap();

    assert_ne!(intervened, normal);
}

#[test]
fn loader_rejects_malformed_optional_output_weight_geometry() {
    let metadata = tiny_metadata(1);

    let mut tensors = HashMap::new();
    insert_tiny_gemma4_tensors(&mut tensors);
    // Corrupt the optional LM-head weight: expected [embed_dim=2, vocab=4].
    tensors.insert("output.weight".to_string(), tiny_weight(&[2, 3]));
    let loader = GgufLoader::for_test(metadata, tensors);
    let error = Gemma4::from_loader(loader)
        .err()
        .expect("malformed optional output.weight must be rejected before allocation");
    assert!(error.to_string().contains("output.weight"), "{error}");
    assert!(error.to_string().contains("dimensions"), "{error}");
}

#[test]
fn loader_rejects_malformed_ple_block_geometry() {
    let mut metadata = tiny_metadata(1);
    metadata.insert(
        "gemma4.hidden_size_per_layer_input".to_string(),
        GgufValue::U32(2),
    );

    let mut tensors = HashMap::new();
    insert_tiny_gemma4_tensors(&mut tensors);
    // A PLE layer requires inp_gate [embed, per_layer_dim]; give it the
    // wrong width so the preflight must reject before block allocation.
    tensors.insert("blk.0.proj.weight".to_string(), tiny_weight(&[2, 2]));
    tensors.insert("blk.0.inp_gate.weight".to_string(), tiny_weight(&[3, 2]));
    let loader = GgufLoader::for_test(metadata, tensors);
    let error = Gemma4::from_loader(loader)
        .err()
        .expect("malformed PLE block tensors must be rejected before allocation");
    assert!(error.to_string().contains("inp_gate.weight"), "{error}");
    assert!(error.to_string().contains("dimensions"), "{error}");
}

#[test]
fn loader_accepts_double_wide_mlp_on_shared_layers() {
    let mut metadata = tiny_metadata(2);
    metadata.insert(
        "gemma4.attention.shared_kv_layers".to_string(),
        GgufValue::U32(1),
    );

    let mut tensors = HashMap::new();
    insert_tiny_gemma4_tensors(&mut tensors);
    insert_tiny_gemma4_block_tensors(&mut tensors, 1, false);
    // The KV-shared layer carries the doubled MLP (intermediate 2 -> 4).
    tensors.insert("blk.1.ffn_gate.weight".into(), tiny_weight(&[2, 4]));
    tensors.insert("blk.1.ffn_up.weight".into(), tiny_weight(&[2, 4]));
    tensors.insert("blk.1.ffn_down.weight".into(), tiny_weight(&[4, 2]));

    let loader = GgufLoader::for_test(metadata, tensors);
    let model = Gemma4::from_loader(loader).unwrap();

    assert_eq!(model.blocks.len(), 2);
    assert_eq!(model.blocks[1].attn.shared_source_layer, Some(0));
}

#[test]
fn ple_vectors_slice_by_layer_and_token() {
    let backend = CpuBackend;
    let model = Gemma4 {
        embed_tokens: Gemma4Embedding::F32(Arc::new(CpuTensor::zeroes(&[3, 2]))),
        blocks: Vec::new(),
        norm: CpuTensor::zeroes(&[2]),
        head: Gemma4Head::Linear(Linear::new(CpuTensor::zeroes(&[2, 3]), None)),
        ple: Some(Gemma4Ple::Hidden(CpuTensor::from_data(
            vec![2, 3, 2],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            ],
        ))),
        config: Gemma4Config {
            n_layers: 2,
            n_heads: 1,
            n_local_kv_heads: 1,
            n_global_kv_heads: 1,
            embed_dim: 2,
            intermediate_dim: 2,
            vocab_size: 3,
            local_head_dim: 2,
            global_head_dim: 2,
            max_seq_len: 4,
            rope_theta: 10_000.0,
            local_rope_theta: 10_000.0,
            global_rope_theta: 10_000.0,
            norm_eps: 1e-6,
            attention_scale: 1.0,
            sliding_window: 2,
            layer_types: vec![Gemma4AttentionType::Local, Gemma4AttentionType::Local],
            final_logit_softcap: None,
            vocab_size_per_layer_input: Some(3),
            hidden_size_per_layer_input: Some(2),
            num_kv_shared_layers: 0,
        },
        per_layer_model_proj: None,
        per_layer_proj_norm: None,
    };
    let dummy_hidden = CpuTensor::zeroes(&[2, 2]);
    let ple = model
        .ple_vectors(&backend, &[2, 1], &dummy_hidden)
        .unwrap()
        .unwrap();
    // With per_layer_dim=2 and ple_scale=sqrt(2), raw PLE values are scaled.
    let s = 2.0f32.sqrt();
    assert_eq!(ple[0].data(), &[5.0 * s, 6.0 * s, 3.0 * s, 4.0 * s]);
    assert_eq!(ple[1].data(), &[50.0 * s, 60.0 * s, 30.0 * s, 40.0 * s]);
}
