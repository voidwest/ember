use super::*;
use crate::experiments::test_support::HookRecord;
use crate::experiments::test_support::RecordingExperiment;
use crate::experiments::{
    ExecutionPhase, ExperimentHook, ModelContext, ModelFamily, TracingState, ZeroLayerOutput,
    ZeroLayerOutputSpec, ZeroLayerOutputStage,
};
use crate::loader::{GgufLoader, GgufValue, LoadedTensor};
use crate::plan::{FusionState, KernelId, PlannedOp};
use crate::planned_decode::planned_causal_attention;
use crate::quant::{QuantizedWeight, Q8_0_BLOCK_SIZE, Q8_0_TYPE_SIZE};
use std::collections::HashMap;

#[test]
fn llama_config_honors_full_context_length_metadata() {
    let mut metadata = HashMap::new();
    metadata.insert("llama.context_length".to_string(), GgufValue::U32(131_072));
    let loader = GgufLoader::for_test(metadata, HashMap::new());

    let config = LlamaConfig::from_gguf_metadata(&loader).unwrap();

    assert_eq!(config.max_seq_len, 131_072);
}

#[test]
fn llama_config_rejects_odd_rope_dimensions_and_large_tables() {
    fn config_with(values: &[(&str, u32)]) -> LlamaConfig {
        let metadata = values
            .iter()
            .map(|(key, value)| ((*key).to_string(), GgufValue::U32(*value)))
            .collect();
        LlamaConfig::from_gguf_metadata(&GgufLoader::for_test(metadata, HashMap::new())).unwrap()
    }

    let odd = config_with(&[("llama.attention.key_length", 1)])
        .validate()
        .expect_err("odd RoPE dimensions must be rejected before table construction");
    assert!(odd.to_string().contains("positive and even"), "{odd}");

    // Both individual values are within their sanity caps, but their
    // product would allocate an unreasonably large pair of tables.
    let oversized = config_with(&[
        ("llama.context_length", 1_000_000),
        ("llama.attention.key_length", 512),
    ])
    .validate()
    .expect_err("RoPE table product must be bounded");
    assert!(
        oversized.to_string().contains("RoPE-table element limit"),
        "{oversized}"
    );
}

#[test]
fn llama_rejects_hostile_rope_factors_and_linear_shapes() {
    fn base_metadata() -> HashMap<String, GgufValue> {
        HashMap::from([
            ("llama.embedding_length".to_string(), GgufValue::U32(4)),
            ("llama.attention.head_count".to_string(), GgufValue::U32(1)),
            ("llama.attention.key_length".to_string(), GgufValue::U32(4)),
            ("llama.context_length".to_string(), GgufValue::U32(8)),
        ])
    }
    fn loader_with_rope(
        metadata: HashMap<String, GgufValue>,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> GgufLoader {
        GgufLoader::for_test(
            metadata,
            HashMap::from([(
                "rope_freqs.weight".to_string(),
                LoadedTensor::F32(CpuTensor::from_data(shape, values)),
            )]),
        )
    }

    let shape_err = Llama::from_loader(loader_with_rope(base_metadata(), vec![1], vec![1.0]))
        .err()
        .expect("wrong RoPE factor shape must be rejected");
    assert!(
        shape_err.to_string().contains("rope_freqs.weight"),
        "{shape_err}"
    );

    let value_err = Llama::from_loader(loader_with_rope(
        base_metadata(),
        vec![2],
        vec![f32::NAN, 1.0],
    ))
    .err()
    .expect("non-finite RoPE factors must be rejected");
    assert!(
        value_err.to_string().contains("finite positive factors"),
        "{value_err}"
    );

    let mut loader = GgufLoader::for_test(
        HashMap::new(),
        HashMap::from([(
            "bad.weight".to_string(),
            LoadedTensor::F32(CpuTensor::from_data(vec![4], vec![0.0; 4])),
        )]),
    );
    let mut packing_ns = 0u64;
    let linear_err = take_llama_linear(
        &mut loader,
        "bad.weight",
        None,
        false,
        &mut packing_ns,
        None,
    )
    .err()
    .expect("non-2D linear weights must be rejected");
    assert!(
        linear_err.to_string().contains("2D linear weight"),
        "{linear_err}"
    );
}

fn metadata_loader(metadata: HashMap<String, GgufValue>) -> GgufLoader {
    GgufLoader::for_test(metadata, HashMap::new())
}

#[test]
fn llama_config_accepts_lossless_numeric_types_and_rejects_others() {
    let config = LlamaConfig::from_gguf_metadata(&metadata_loader(HashMap::from([
        (
            "general.architecture".to_string(),
            GgufValue::Str("qwen3".into()),
        ),
        ("qwen3.block_count".to_string(), GgufValue::U64(3)),
        ("qwen3.attention.head_count".to_string(), GgufValue::I32(4)),
        (
            "qwen3.attention.head_count_kv".to_string(),
            GgufValue::U16(2),
        ),
        ("llama.embedding_length".to_string(), GgufValue::I64(64)),
        ("qwen3.context_length".to_string(), GgufValue::F32(512.0)),
        (
            "qwen3.rope.freq_base".to_string(),
            GgufValue::U32(1_000_000),
        ),
        (
            "qwen3.attention.layer_norm_rms_epsilon".to_string(),
            GgufValue::F64(1e-6),
        ),
    ])))
    .unwrap();
    assert_eq!(config.n_layers, 3);
    assert_eq!(config.n_heads, 4);
    assert_eq!(config.n_kv_heads, 2);
    assert_eq!(config.embed_dim, 64, "llama.* fallback key is still read");
    assert_eq!(config.head_dim, 16);
    assert_eq!(config.max_seq_len, 512);
    assert_eq!(config.rope_theta, 1_000_000.0);
    assert_eq!(config.norm_eps, 1e-6f64 as f32);
    assert_eq!(config.vocab_size, 32000, "missing keys keep their defaults");

    for (key, value, needle) in [
        (
            "llama.context_length",
            GgufValue::Str("4096".into()),
            "must be numeric",
        ),
        (
            "llama.block_count",
            GgufValue::Bool(true),
            "must be numeric",
        ),
        ("llama.block_count", GgufValue::I32(-1), "does not fit u32"),
        (
            "llama.context_length",
            GgufValue::U64(1 << 40),
            "does not fit u32",
        ),
        (
            "llama.context_length",
            GgufValue::F32(1.5),
            "does not fit u32",
        ),
        (
            "llama.rope.freq_base",
            GgufValue::U64((1 << 24) + 1),
            "does not fit f32",
        ),
        (
            "llama.attention.layer_norm_rms_epsilon",
            GgufValue::Array(vec![]),
            "must be numeric",
        ),
    ] {
        let error = LlamaConfig::from_gguf_metadata(&metadata_loader(HashMap::from([(
            key.to_string(),
            value,
        )])))
        .expect_err(key)
        .to_string();
        assert!(error.contains(key) && error.contains(needle), "{error}");
    }
}

#[test]
fn qk_norm_uses_the_model_norm_epsilon() {
    let f32_tensor = |shape: Vec<usize>, seed: usize| {
        let len = shape.iter().product::<usize>();
        let values = (0..len)
            .map(|index| ((index * 7 + seed * 3) % 11) as f32 * 0.1 - 0.5)
            .collect();
        LoadedTensor::F32(CpuTensor::from_data(shape, values))
    };
    // GGUF dimension order: [ne0 = in/embed, ne1 = out/vocab].
    let tensors = HashMap::from([
        ("token_embd.weight".to_string(), f32_tensor(vec![8, 4], 1)),
        ("output_norm.weight".to_string(), f32_tensor(vec![8], 2)),
        ("blk.0.attn_norm.weight".to_string(), f32_tensor(vec![8], 3)),
        ("blk.0.ffn_norm.weight".to_string(), f32_tensor(vec![8], 4)),
        ("blk.0.attn_q.weight".to_string(), f32_tensor(vec![8, 8], 5)),
        ("blk.0.attn_k.weight".to_string(), f32_tensor(vec![8, 4], 6)),
        ("blk.0.attn_v.weight".to_string(), f32_tensor(vec![8, 4], 7)),
        (
            "blk.0.attn_output.weight".to_string(),
            f32_tensor(vec![8, 8], 8),
        ),
        (
            "blk.0.ffn_gate.weight".to_string(),
            f32_tensor(vec![8, 8], 9),
        ),
        (
            "blk.0.ffn_up.weight".to_string(),
            f32_tensor(vec![8, 8], 10),
        ),
        (
            "blk.0.ffn_down.weight".to_string(),
            f32_tensor(vec![8, 8], 11),
        ),
        (
            "blk.0.attn_q_norm.weight".to_string(),
            f32_tensor(vec![4], 12),
        ),
        (
            "blk.0.attn_k_norm.weight".to_string(),
            f32_tensor(vec![4], 13),
        ),
    ]);
    let metadata = HashMap::from([
        (
            "general.architecture".to_string(),
            GgufValue::Str("qwen3".into()),
        ),
        ("qwen3.block_count".to_string(), GgufValue::U32(1)),
        ("qwen3.embedding_length".to_string(), GgufValue::U32(8)),
        ("qwen3.attention.head_count".to_string(), GgufValue::U32(2)),
        (
            "qwen3.attention.head_count_kv".to_string(),
            GgufValue::U32(1),
        ),
        ("qwen3.attention.key_length".to_string(), GgufValue::U32(4)),
        ("qwen3.context_length".to_string(), GgufValue::U32(8)),
        ("qwen3.vocab_size".to_string(), GgufValue::U32(4)),
        (
            "qwen3.attention.layer_norm_rms_epsilon".to_string(),
            GgufValue::F32(0.25),
        ),
    ]);
    let model = Llama::from_loader(GgufLoader::for_test(metadata, tensors)).unwrap();
    let attention = &model.blocks[0].self_attn;
    assert_eq!(attention.qk_norm_eps, 0.25);

    // The decode Q/K norm really uses it: compare against a manual
    // RMSNorm with eps = 0.25 (before RoPE at position 0, where RoPE is
    // the identity).
    let norm = attention.q_norm.as_ref().unwrap();
    let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.05 - 0.2).collect();
    let mut actual = input.clone();
    attention.apply_decode_rope_and_qk_norm(&mut actual, 2, 0, Some(norm));
    let mut expected = input;
    for head in expected.chunks_exact_mut(4) {
        let mean_square = head.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let rstd = (mean_square + 0.25).sqrt().recip();
        for (value, weight) in head.iter_mut().zip(norm.data()) {
            *value = *value * rstd * weight;
        }
    }
    for (a, e) in actual.iter().zip(&expected) {
        assert!(
            (a - e).abs() <= 1e-6,
            "actual={actual:?} expected={expected:?}"
        );
    }
}

fn test_f32_linear(out_features: usize, in_features: usize, seed: usize) -> Linear<CpuBackend> {
    let weights = (0..in_features * out_features)
        .map(|index| ((index * 17 + seed * 11) % 61) as f32 * 0.002 - 0.06)
        .collect();
    Linear::new(
        CpuTensor::from_data(vec![in_features, out_features], weights),
        None,
    )
}

fn test_llama_model() -> Llama<CpuBackend> {
    test_llama_model_with_layers(1)
}

fn test_llama_model_with_layers(n_layers: usize) -> Llama<CpuBackend> {
    test_llama_model_with_dims(n_layers, 32, 32)
}

fn test_llama_model_with_dims(
    n_layers: usize,
    embed_dim: usize,
    vocab_size: usize,
) -> Llama<CpuBackend> {
    testkit::q8_llama(n_layers, embed_dim, vocab_size)
}

fn configure_as_test_qwen(model: &mut Llama<CpuBackend>) {
    model.config.rope_layout = RopeLayout::SplitHalf;
    model.config.qk_norm_order = QkNormOrder::BeforeRope;
    for block in &mut model.blocks {
        block.self_attn.rope_layout = RopeLayout::SplitHalf;
        block.self_attn.qk_norm_order = QkNormOrder::BeforeRope;
        block.self_attn.q_norm = Some(CpuTensor::from_data(
            vec![model.config.head_dim],
            vec![1.0; model.config.head_dim],
        ));
        block.self_attn.k_norm = Some(CpuTensor::from_data(
            vec![model.config.head_dim],
            vec![1.0; model.config.head_dim],
        ));
    }
    model.fast_decode_inter_dim = model.eligible_fast_decode_inter_dim();
}

#[test]
fn fast_q8_decode_matches_generic_path() {
    let model = test_llama_model();
    let backend = CpuBackend;
    let mut generic_cache = model.create_cache(&backend, model.config.max_seq_len);
    let generic =
        Llama::forward_last_logits_with_cache(&model, &backend, &[3], &mut generic_cache, 0)
            .unwrap();
    let mut fast_cache = model.create_cache(&backend, model.config.max_seq_len);
    let fast =
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &[3], &mut fast_cache, 0)
            .unwrap();

    for (index, (expected, actual)) in generic.data().iter().zip(fast.data()).enumerate() {
        let tolerance = 1e-4 * expected.abs().max(actual.abs()).max(1.0);
        assert!(
            (expected - actual).abs() <= tolerance,
            "logit {index}: generic={expected} fast={actual}"
        );
    }
}

/// Synthetic model whose lm_head input is known exactly: the attention
/// and MLP outputs are zero (all-zero Q8_0 scales), so the final hidden
/// state is the token embedding, which is block-periodic. Every Q8_0
/// block of the quantized head activation is therefore identical, and
/// head rows that are block permutations of one another have
/// mathematically equal logits that differ only by float summation
/// order — a constructed near-tie that a kernel with a different
/// accumulation order would resolve differently.
fn near_tie_head_model() -> Llama<CpuBackend> {
    const EMBED: usize = 256;
    const VOCAB: usize = 64;
    let blocks = EMBED / Q8_0_BLOCK_SIZE;
    let mut model = test_llama_model_with_dims(1, EMBED, VOCAB);
    let zero_q8 = |out_features: usize, in_features: usize| {
        let n_blocks = out_features * in_features / Q8_0_BLOCK_SIZE;
        Linear::new_q8_0(
            QuantizedWeight::try_new(
                vec![0u8; n_blocks * Q8_0_TYPE_SIZE],
                vec![out_features, in_features],
            )
            .unwrap(),
            None,
        )
    };
    for block in &mut model.blocks {
        block.self_attn.o_proj = zero_q8(EMBED, EMBED);
        block.mlp.down_proj = zero_q8(EMBED, 2 * EMBED);
    }
    let embedding: Vec<f32> = (0..VOCAB * EMBED)
        .map(|index| {
            let k = index % Q8_0_BLOCK_SIZE;
            ((k * 7 + 3) % 13) as f32 * 0.1 - 0.55
        })
        .collect();
    model.embed_tokens = LlamaEmbedding::F32(CpuTensor::from_data(vec![VOCAB, EMBED], embedding));

    // One base row with a wide dynamic range of block scales and mixed
    // signs, so the cross-block sum is order-sensitive.
    let base: Vec<(half::f16, [i8; Q8_0_BLOCK_SIZE])> = (0..blocks)
        .map(|b| {
            let scale =
                half::f16::from_f32((2.0f32).powi(3 * b as i32 - 9) * (1.0 + 0.37 * b as f32));
            let mut quants = [0i8; Q8_0_BLOCK_SIZE];
            for (k, quant) in quants.iter_mut().enumerate() {
                let sign = if (b + k) % 2 == 0 { 1 } else { -1 };
                *quant = (sign * (((k * 11 + b * 5) % 97) as i32 + 17)) as i8;
            }
            (scale, quants)
        })
        .collect();
    let mut data = Vec::with_capacity(VOCAB * blocks * Q8_0_TYPE_SIZE);
    for row in 0..VOCAB {
        // Row 63 duplicates row 5 exactly: an exact tie, which must go
        // to the lowest index.
        let source = if row == VOCAB - 1 { 5 } else { row };
        for b in 0..blocks {
            let (scale, quants) = &base[(b * (2 * source + 1) + source / 3) % blocks];
            data.extend_from_slice(&scale.to_bits().to_le_bytes());
            data.extend(quants.iter().map(|&quant| quant as u8));
        }
    }
    model.head = Linear::new_q8_0(
        QuantizedWeight::try_new(data, vec![VOCAB, EMBED]).unwrap(),
        None,
    );
    model
}

/// Greedy must pick exactly `argmax` of the logits the sampling route
/// computes. With this head, an AVX2-order reduction (8 lane partial
/// sums, then a horizontal add) ranks row 16 first while the sequential
/// scalar block order the old fused greedy kernel used ranks row 0 first
/// (checked by simulating both orders), so on x86 the old path failed
/// this; on aarch64 the dotprod kernel keeps scalar block order.
#[test]
fn greedy_token_is_argmax_of_full_logits_including_near_ties() {
    let model = near_tie_head_model();
    let backend = CpuBackend;
    assert!(model.fast_decode_inter_dim.is_some());
    let mut logits_cache = model.create_cache(&backend, model.config.max_seq_len);
    let mut greedy_cache = model.create_cache(&backend, model.config.max_seq_len);
    for (position, token) in [3u32, 3, 7].into_iter().enumerate() {
        let logits = ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &[token],
            &mut logits_cache,
            position,
        )
        .unwrap();
        let logits = logits.data();
        let expected = crate::sampler::argmax_token(logits);
        // The construction must really be a near-tie: many rows share the
        // same mathematical logit.
        let near = logits
            .iter()
            .filter(|value| {
                (*value - logits[expected]).abs() <= 1e-4 * logits[expected].abs().max(1.0)
            })
            .count();
        assert!(near >= 8, "expected a dense near-tie, found {near} rows");
        let (greedy, greedy_logit) = ForwardModel::greedy_next_token_with_cache(
            &model,
            &backend,
            &[token],
            &mut greedy_cache,
            position,
        )
        .unwrap();
        assert_eq!(greedy as usize, expected, "position {position}");
        assert_eq!(greedy_logit.to_bits(), logits[expected].to_bits());
    }
}

#[test]
fn greedy_decode_is_allocation_free_after_warmup() {
    let model = test_llama_model();
    let backend = CpuBackend;
    let mut cache = model.create_cache(&backend, model.config.max_seq_len);
    ForwardModel::greedy_next_token_with_cache(&model, &backend, &[3], &mut cache, 0).unwrap();
    ForwardModel::greedy_next_token_with_cache(&model, &backend, &[4], &mut cache, 1).unwrap();
    assert!(crate::alloc_counter::counting_active());
    let (result, allocations) = crate::alloc_counter::count_allocations(|| {
        ForwardModel::greedy_next_token_with_cache(&model, &backend, &[5], &mut cache, 2)
    });
    result.unwrap();
    assert_eq!(allocations, 0, "steady-state greedy decode allocated");
}

fn warmed_disabled_decode_allocation_count(n_layers: usize) -> usize {
    let model = test_llama_model_with_layers(n_layers);
    let backend = CpuBackend;
    let mut cache = model.create_cache(&backend, model.config.max_seq_len);
    ForwardModel::forward_last_logits_with_cache(&model, &backend, &[3], &mut cache, 0).unwrap();
    assert!(crate::alloc_counter::counting_active());
    let (result, allocations) = crate::alloc_counter::count_allocations(|| {
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &[5], &mut cache, 1)
    });
    result.unwrap();
    allocations
}

#[test]
fn disabled_hooks_add_no_per_layer_allocations() {
    let one_layer = warmed_disabled_decode_allocation_count(1);
    let four_layers = warmed_disabled_decode_allocation_count(4);

    assert_eq!(
        four_layers, one_layer,
        "disabled hook dispatch must not allocate once per layer"
    );
}

/// Allocation events of a multi-token (generic/eager) prefill into a
/// fresh cache, after one warm-up prefill has sized the thread-local
/// activation and attention-score buffers. Runs on a one-thread Rayon
/// pool so every kernel executes (and is counted) on the measuring
/// thread and no count depends on the global job injector.
fn warmed_prefill_allocation_count(n_layers: usize, qwen: bool) -> usize {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let mut model = test_llama_model_with_layers(n_layers);
        if qwen {
            configure_as_test_qwen(&mut model);
        }
        let backend = CpuBackend;
        let prompt = [3, 4, 5, 6, 7];
        let mut warm = model.create_cache(&backend, model.config.max_seq_len);
        let mut cache = model.create_cache(&backend, model.config.max_seq_len);
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &prompt, &mut warm, 0)
            .unwrap();
        assert!(crate::alloc_counter::counting_active());
        let (result, allocations) = crate::alloc_counter::count_allocations(|| {
            ForwardModel::forward_last_logits_with_cache(&model, &backend, &prompt, &mut cache, 0)
        });
        result.unwrap();
        allocations
    })
}

/// The eager prefill reuses one set of scratch buffers across layers:
/// its allocation count must not grow with depth, and stays within a
/// small per-forward budget (embedding, scratch set-up, final logits).
/// Before scratch reuse every layer allocated ~50 times (three
/// allocations per intermediate tensor).
#[test]
fn prefill_allocations_do_not_scale_with_layers() {
    for qwen in [false, true] {
        let one_layer = warmed_prefill_allocation_count(1, qwen);
        let four_layers = warmed_prefill_allocation_count(4, qwen);
        assert_eq!(
            four_layers, one_layer,
            "eager prefill allocated per layer (qwen={qwen})"
        );
        assert!(
            one_layer <= PREFILL_ALLOCATION_BUDGET,
            "eager prefill made {one_layer} allocations, budget {PREFILL_ALLOCATION_BUDGET} (qwen={qwen})"
        );
    }
}

const PREFILL_ALLOCATION_BUDGET: usize = 48;

#[test]
fn split_half_rope_remains_on_generic_decode_path() {
    let mut model = test_llama_model();
    configure_as_test_qwen(&mut model);
    assert!(model.fast_decode_inter_dim.is_none());
}

#[test]
fn experiment_hook_order_covers_prefill_and_fast_decode() {
    let model = test_llama_model();
    let backend = CpuBackend;
    let model_context =
        ModelContext::new(ModelFamily::Llama, None, "llama", 1, model.config.embed_dim);
    let (experiment, records) = RecordingExperiment::new();
    let mut runner = ExperimentRunner::new(experiment);
    let mut cache = model.create_cache(&backend, model.config.max_seq_len);

    let prefill = ExecutionContext::new(
        model_context,
        ExecutionPhase::Prefill,
        0,
        2,
        TracingState::Disabled,
    );
    ExperimentalForwardModel::forward_last_logits_with_experiment(
        &model,
        &backend,
        &[3, 5],
        &mut cache,
        0,
        prefill,
        &mut runner,
    )
    .unwrap();

    let decode = ExecutionContext::new(
        model_context,
        ExecutionPhase::Decode,
        2,
        1,
        TracingState::Disabled,
    );
    ExperimentalForwardModel::forward_last_logits_with_experiment(
        &model,
        &backend,
        &[7],
        &mut cache,
        2,
        decode,
        &mut runner,
    )
    .unwrap();

    let records = records.lock().unwrap();
    let expected_per_evaluation = [
        ExperimentHook::BeforeLayer,
        ExperimentHook::BeforeOProj,
        ExperimentHook::AfterAttention,
        ExperimentHook::AfterMlp,
        ExperimentHook::AfterLayer,
        ExperimentHook::BeforeLogits,
        ExperimentHook::AfterLogits,
    ];
    assert_eq!(records.len(), expected_per_evaluation.len() * 2);
    assert_eq!(
        records.iter().map(|record| record.hook).collect::<Vec<_>>(),
        expected_per_evaluation
            .into_iter()
            .chain(expected_per_evaluation)
            .collect::<Vec<_>>()
    );
    for record in &records[..expected_per_evaluation.len()] {
        assert_eq!(record.phase, Some(ExecutionPhase::Prefill));
        assert_eq!(record.sequence_length, Some(2));
    }
    for record in &records[expected_per_evaluation.len()..] {
        assert_eq!(record.phase, Some(ExecutionPhase::Decode));
        assert_eq!(record.sequence_length, Some(3));
    }
    let q_dim = model.config.n_heads * model.config.head_dim;
    assert_eq!(records[0].shape, Some([2, model.config.embed_dim]));
    // The per-head attention output: every head of every prefill row,
    // then the single decode row.
    assert_eq!(records[1].shape, Some([2, q_dim]));
    assert_eq!(records[5].shape, Some([1, model.config.embed_dim]));
    assert_eq!(records[6].shape, Some([1, model.config.vocab_size]));
    assert_eq!(records[7].shape, Some([1, model.config.embed_dim]));
    assert_eq!(records[8].shape, Some([1, q_dim]));
    for record in records.iter().filter(|record| record.layer_index.is_some()) {
        assert_eq!(record.layer_index, Some(0));
    }
}

#[test]
fn qwen_observation_hooks_preserve_generic_logits_exactly() {
    let mut model = test_llama_model();
    configure_as_test_qwen(&mut model);
    let backend = CpuBackend;
    let mut normal_cache = model.create_cache(&backend, model.config.max_seq_len);
    let normal =
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &[3], &mut normal_cache, 0)
            .unwrap();

    let model_context =
        ModelContext::new(ModelFamily::Qwen3, None, "qwen3", 1, model.config.embed_dim);
    let execution = ExecutionContext::new(
        model_context,
        ExecutionPhase::Decode,
        0,
        1,
        TracingState::Disabled,
    );
    let (experiment, records) = RecordingExperiment::new();
    let mut runner = ExperimentRunner::new(experiment);
    let mut observed_cache = model.create_cache(&backend, model.config.max_seq_len);
    let observed = ExperimentalForwardModel::forward_last_logits_with_experiment(
        &model,
        &backend,
        &[3],
        &mut observed_cache,
        0,
        execution,
        &mut runner,
    )
    .unwrap();

    assert_eq!(observed, normal);
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 7);
    assert!(records
        .iter()
        .all(|record| record.phase == Some(ExecutionPhase::Decode)));
}

#[test]
fn zero_attention_intervention_changes_fast_decode_logits() {
    let model = test_llama_model();
    let backend = CpuBackend;

    let mut normal_cache = model.create_cache(&backend, model.config.max_seq_len);
    ForwardModel::forward_last_logits_with_cache(&model, &backend, &[3, 5], &mut normal_cache, 0)
        .unwrap();
    let normal =
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &[7], &mut normal_cache, 2)
            .unwrap();

    let mut experiment_cache = model.create_cache(&backend, model.config.max_seq_len);
    ForwardModel::forward_last_logits_with_cache(
        &model,
        &backend,
        &[3, 5],
        &mut experiment_cache,
        0,
    )
    .unwrap();
    let model_context =
        ModelContext::new(ModelFamily::Llama, None, "llama", 1, model.config.embed_dim);
    let mut runner = ExperimentRunner::new(ZeroLayerOutput::new(ZeroLayerOutputSpec::new(
        0,
        ZeroLayerOutputStage::Attention,
    )));
    runner.on_model_loaded(&model_context).unwrap();
    let execution = ExecutionContext::new(
        model_context,
        ExecutionPhase::Decode,
        2,
        1,
        TracingState::Disabled,
    );
    let intervened = ExperimentalForwardModel::forward_last_logits_with_experiment(
        &model,
        &backend,
        &[7],
        &mut experiment_cache,
        2,
        execution,
        &mut runner,
    )
    .unwrap();

    assert_ne!(intervened, normal);
}

#[test]
fn trace_guard_preserves_semantic_events() {
    let model = test_llama_model();
    let backend = CpuBackend;
    let mut cache = model.create_cache(&backend, model.config.max_seq_len);

    assert!(crate::trace::enable_tracing("prefill", 0));
    let result =
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &[3], &mut cache, 0);
    let report = crate::trace::disable_tracing().expect("trace report");
    result.unwrap();

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
            "q_proj",
            "k_proj",
            "v_proj",
            "rope_q",
            "rope_k",
            "kv_cache_store",
            "attention_score",
            "o_proj",
            "attn_residual_add",
            "mlp_rms_norm",
            "gate_proj",
            "silu",
            "up_proj",
            "elemul",
            "down_proj",
            "mlp_residual_add",
            "final_norm",
            "lm_head",
        ]
    );
}

#[test]
fn hidden_only_pooling_matches_logits_path() {
    let model = test_llama_model();
    let backend = CpuBackend;
    let groups = vec![vec![0, 1]];
    let hidden_only = model
        .forward_pooled_hidden_states(&backend, &[3, 5], &groups)
        .unwrap();
    let (with_logits, _) = model
        .forward_pooled_activations(&backend, &[3, 5], &groups)
        .unwrap();

    assert_eq!(hidden_only, with_logits);
}

#[test]
fn block_pooled_forward_matches_independent_sequences() {
    let model = test_llama_model();
    let backend = CpuBackend;
    let first_tokens = [3, 5];
    let second_tokens = [7, 11, 13];
    let token_ids = [first_tokens.as_slice(), second_tokens.as_slice()].concat();
    let boundaries = [0, first_tokens.len()];
    let groups = vec![vec![0, 1], vec![2, 3, 4]];

    let (batched_pooled, batched_logits) = ForwardModel::forward_pooled_with_blocks(
        &model,
        &backend,
        &token_ids,
        &boundaries,
        &groups,
    )
    .unwrap();

    for (block_index, tokens) in [first_tokens.as_slice(), second_tokens.as_slice()]
        .into_iter()
        .enumerate()
    {
        let local_group = vec![(0..tokens.len()).collect()];
        let (independent_pooled, independent_logits) = model
            .forward_pooled_activations(&backend, tokens, &local_group)
            .unwrap();
        assert_eq!(
            batched_pooled[block_index], independent_pooled[0],
            "pooled activations differ for block {block_index}"
        );

        let vocab_size = model.config.vocab_size;
        let batched_row =
            &batched_logits.data()[block_index * vocab_size..(block_index + 1) * vocab_size];
        let independent_data = independent_logits.data();
        let independent_row =
            &independent_data[independent_data.len() - vocab_size..independent_data.len()];
        assert_eq!(batched_row, independent_row);
    }
}

// -----------------------------------------------------------------------
// v0.4 execution plan construction
// -----------------------------------------------------------------------

/// Planned, hooks-disabled plan at capacity 8 without provenance hashes.
fn default_plan(model: &Llama<CpuBackend>) -> Arc<ExecutionPlan> {
    model
        .execution_plan(
            ExecutionMode::Planned,
            HookMode::Disabled,
            &[],
            8,
            None,
            None,
        )
        .unwrap()
}

#[test]
fn execution_plan_is_deterministic_and_complete() {
    let model = test_llama_model_with_layers(2);
    let plan_a = default_plan(&model);
    let plan_b = default_plan(&model);
    assert_eq!(
        plan_a.plan_hash, plan_b.plan_hash,
        "plans must hash identically"
    );
    assert!(
        Arc::ptr_eq(&plan_a, &plan_b),
        "plan cache must return the same Arc for an identical key"
    );
    assert_eq!(plan_a.preamble.len(), 1, "one embedding op");
    assert_eq!(plan_a.layers.len(), 2, "one layer plan per block");
    assert_eq!(plan_a.layers[0].ops.len(), 17, "unfused layer op count");
    assert_eq!(plan_a.final_ops.len(), 2, "output norm + logits");
    assert_eq!(plan_a.operation_count(), 1 + 2 * 17 + 2);
    assert!(
        plan_a
            .layers
            .iter()
            .all(|l| l.fusion == FusionState::Unfused),
        "phase-3 plans are unfused"
    );
    assert_eq!(
        plan_a
            .scratch
            .regions
            .iter()
            .filter(|r| r.name == "scores")
            .count(),
        1,
        "one shared score region"
    );
    assert!(plan_a.scratch.total_bytes > 0);
}

#[test]
fn execution_plan_cache_keys_capacity_and_provenance() {
    let model = test_llama_model();
    let small = model
        .execution_plan(
            ExecutionMode::Planned,
            HookMode::Disabled,
            &[],
            4,
            None,
            None,
        )
        .unwrap();
    let large = model
        .execution_plan(
            ExecutionMode::Planned,
            HookMode::Disabled,
            &[],
            8,
            Some(&"aa".repeat(32)),
            Some(&"bb".repeat(32)),
        )
        .unwrap();
    assert!(!Arc::ptr_eq(&small, &large));
    assert_eq!(small.kv.max_seq, 4);
    assert_eq!(large.kv.max_seq, 8);
    assert!(small.model_sha256.is_empty());
    assert_eq!(large.model_sha256, "aa".repeat(32));
}

#[test]
fn execution_plan_cache_keys_rayon_thread_count() {
    let model = test_llama_model();
    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let two_threads = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let (model, serial) = one_thread.install(move || {
        let plan = default_plan(&model);
        (model, plan)
    });
    let (_model, parallel) = two_threads.install(move || {
        let plan = default_plan(&model);
        (model, plan)
    });
    assert!(!Arc::ptr_eq(&serial, &parallel));
    assert_eq!(serial.cpu.threads, 1);
    assert_eq!(serial.dispatch.thread_strategy, "serial");
    assert_eq!(parallel.cpu.threads, 2);
    assert_eq!(parallel.dispatch.thread_strategy, "column-parallel-rayon");
}

#[test]
fn execution_plan_resolves_kernels_and_shapes() {
    let model = test_llama_model();
    let plan = default_plan(&model);
    // layer 0 op 1 is the q_proj matvec
    let q_w = match &plan.layers[0].ops[1] {
        PlannedOp::Matvec { weight, .. } => *weight,
        other => panic!("expected q_proj matvec, got {other:?}"),
    };
    let q_record = plan
        .tensor_table
        .iter()
        .find(|t| t.id == q_w.id)
        .expect("q_proj weight in tensor table");
    assert_eq!(q_record.shape, vec![32, 32], "test model embed = 32");
    assert_eq!(q_record.gguf_dtype, "q8_0");
    assert_eq!(q_record.kernel, KernelId::Q8Packed);
    let embed = plan
        .tensor_table
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("embedding weight recorded");
    assert_eq!(embed.shape, vec![32, 32]);
    assert_eq!(embed.kernel, KernelId::EmbeddingF32Row);
    // final logits op is untied in the test model
    let tied = match &plan.final_ops[1] {
        PlannedOp::Logits { tied, .. } => *tied,
        other => panic!("expected logits op, got {other:?}"),
    };
    assert!(!tied);
}

#[test]
fn execution_plan_preserves_loader_fallback_provenance() {
    let mut model = test_llama_model();
    model.blocks[0].self_attn.q_proj = test_f32_linear(32, 32, 77);
    model.k_decisions.insert(
        "blk.0.attn_q.weight".into(),
        crate::quant_k::KTensorDecision {
            gguf_dtype: 10,
            execution: crate::quant_k::KExecution::EagerF32,
            fallback_reason: Some("Q2_K has no canonical native kernel".into()),
        },
    );
    let plan = default_plan(&model);
    let tensor = plan
        .tensor_table
        .iter()
        .find(|record| record.name == "blk.0.attn_q.weight")
        .unwrap();
    assert_eq!(tensor.gguf_dtype, "q2_k");
    assert_eq!(tensor.execution, "eager_f32");
    assert_eq!(tensor.kernel, KernelId::EagerF32);
    let dispatch = plan
        .dispatch
        .kernel_per_tensor
        .iter()
        .find(|entry| entry.tensor == tensor.id)
        .unwrap();
    assert_eq!(
        dispatch.fallback.as_deref(),
        Some("Q2_K has no canonical native kernel")
    );
}

#[test]
fn execution_plan_validation_rejects_kernel_identity_tampering() {
    let model = test_llama_model();
    let plan = default_plan(&model);

    let mut wrong_revision = (*plan).clone();
    wrong_revision.kernel_revision -= 1;
    assert!(wrong_revision
        .validate()
        .unwrap_err()
        .to_string()
        .contains("kernel revision"));

    let mut wrong_kernel = (*plan).clone();
    let entry = wrong_kernel
        .dispatch
        .kernel_per_tensor
        .first_mut()
        .expect("test plan has resident tensors");
    entry.kernel = if entry.kernel == KernelId::EagerF32 {
        KernelId::KQuantScalarQ6K
    } else {
        KernelId::EagerF32
    };
    entry.cpu_feature = entry.kernel.cpu_feature().map(str::to_string);
    assert!(wrong_kernel
        .validate()
        .unwrap_err()
        .to_string()
        .contains("disagrees with tensor-table kernel"));
}

#[test]
fn execution_plan_hook_sites_follow_active_stages() {
    let model = test_llama_model();
    let plan = model
        .execution_plan(
            ExecutionMode::Planned,
            HookMode::Observe,
            &["after-attention"],
            8,
            None,
            None,
        )
        .unwrap();
    assert_eq!(plan.hook_sites.active, vec!["after-attention"]);
    let after_attn: Vec<_> = plan
        .hook_sites
        .sites
        .iter()
        .filter(|s| s.stage == "after-attention")
        .collect();
    assert!(!after_attn.is_empty());
    assert!(after_attn
        .iter()
        .all(|s| s.materialized && s.tensor.is_some()));
    assert!(
        plan.hook_sites
            .sites
            .iter()
            .filter(|s| s.stage != "after-attention")
            .all(|s| !s.materialized),
        "inactive stages must be recorded as not materialized"
    );
}

#[test]
fn execution_plan_rejects_unknown_stages() {
    let model = test_llama_model();
    assert!(
        model
            .execution_plan(
                ExecutionMode::Planned,
                HookMode::Observe,
                &["nonsense"],
                8,
                None,
                None
            )
            .is_err(),
        "unknown hook stage must fail at plan build"
    );
}

#[test]
fn execution_plan_serializes_to_json() {
    let model = test_llama_model_with_layers(2);
    let plan = default_plan(&model);
    let json = serde_json::to_string_pretty(&*plan).unwrap();
    assert!(json.contains("\"architecture\": \"llama\""));
    assert!(json.contains("\"op\": \"matvec\""));
    assert!(json.contains("\"plan_hash\": \""));
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Which entry point a greedy run uses for its multi-token prefill.
#[derive(Clone, Copy)]
enum Prefill {
    /// The inherent `Llama::forward_with_cache`.
    Inherent,
    /// `ForwardModel::forward_last_logits_with_cache`.
    Trait,
}

/// Prefill `[3, 1, 7]` in `mode`, then greedy-decode four tokens through
/// the trait path (ForwardModel) so the v0.4 cpu dispatch runs. Returns
/// the token stream and the last decode logits.
fn run_greedy(
    model: &Llama<CpuBackend>,
    mode: ExecutionMode,
    prefill: Prefill,
) -> (Vec<u32>, CpuTensor) {
    let backend = CpuBackend;
    model.set_execution_mode(mode);
    let mut cache = model.create_cache(&backend, 64);
    let prompt = [3u32, 1, 7];
    match prefill {
        Prefill::Inherent => {
            model
                .forward_with_cache(&backend, &prompt, &mut cache, 0)
                .unwrap();
        }
        Prefill::Trait => {
            ForwardModel::forward_last_logits_with_cache(model, &backend, &prompt, &mut cache, 0)
                .unwrap();
        }
    }
    let mut tokens = prompt.to_vec();
    let mut last_logits = None;
    for start_pos in prompt.len()..prompt.len() + 4 {
        let last = *tokens.last().unwrap();
        let logits = ForwardModel::forward_last_logits_with_cache(
            model,
            &backend,
            &[last],
            &mut cache,
            start_pos,
        )
        .unwrap();
        let best = crate::sampler::argmax_token(logits.data());
        tokens.push(best as u32);
        last_logits = Some(logits);
    }
    (tokens, last_logits.expect("decode produced logits"))
}

/// Prefill `[3, 1, 7]` in `mode`, then decode `token` at position 3.
fn plain_decode(model: &Llama<CpuBackend>, mode: ExecutionMode, token: u32) -> CpuTensor {
    let backend = CpuBackend;
    model.set_execution_mode(mode);
    let mut cache = model.create_cache(&backend, 64);
    ForwardModel::forward_last_logits_with_cache(model, &backend, &[3, 1, 7], &mut cache, 0)
        .unwrap();
    ForwardModel::forward_last_logits_with_cache(model, &backend, &[token], &mut cache, 3).unwrap()
}

/// Prefill `[3, 1, 7]` in `mode`, then decode token 5 at position 3
/// through the experiment path with `runner`.
fn decode_with_runner(
    model: &Llama<CpuBackend>,
    mode: ExecutionMode,
    runner: &mut ExperimentRunner,
) -> CpuTensor {
    let backend = CpuBackend;
    model.set_execution_mode(mode);
    let mut cache = model.create_cache(&backend, 64);
    ForwardModel::forward_last_logits_with_cache(model, &backend, &[3, 1, 7], &mut cache, 0)
        .unwrap();
    let model_context =
        ModelContext::new(ModelFamily::Llama, None, "llama", 2, model.config.embed_dim);
    let execution = ExecutionContext::new(
        model_context,
        ExecutionPhase::Decode,
        3,
        1,
        TracingState::Disabled,
    );
    ExperimentalForwardModel::forward_last_logits_with_experiment(
        model,
        &backend,
        &[5],
        &mut cache,
        3,
        execution,
        runner,
    )
    .unwrap()
}

/// `decode_with_runner` with a no-op recording experiment: the logits and
/// every hook record it fired.
fn decode_recording_hooks(
    model: &Llama<CpuBackend>,
    mode: ExecutionMode,
) -> (CpuTensor, Vec<HookRecord>) {
    let (experiment, records) = RecordingExperiment::new();
    let logits = decode_with_runner(model, mode, &mut ExperimentRunner::new(experiment));
    let hook_records = records.lock().unwrap().clone();
    (logits, hook_records)
}

#[test]
fn planned_decode_matches_reference_greedy() {
    let mut model = test_llama_model_with_layers(2);
    // Disable the Q8 workspace fast path so the comparison exercises the
    // generic (reference) path and the planned interpreter directly.
    model.fast_decode_inter_dim = None;

    let (reference_tokens, reference_logits) =
        run_greedy(&model, ExecutionMode::Reference, Prefill::Inherent);
    let (planned_tokens, planned_logits) =
        run_greedy(&model, ExecutionMode::Planned, Prefill::Inherent);

    assert_eq!(
        reference_tokens, planned_tokens,
        "greedy token sequences must be identical"
    );
    let max_abs = max_abs_diff(reference_logits.data(), planned_logits.data());
    assert!(
        max_abs <= 1e-4,
        "planned logits diverge from reference: max_abs {max_abs}"
    );
    // prove the planned interpreter actually ran (not a silent fallback)
    assert!(
        model.decode_state.borrow().is_some(),
        "planned decode must populate the decode session"
    );
}

#[test]
fn planned_decode_multi_token_uses_generic_path() {
    // Multi-token (prefill) calls are outside the single-token planned
    // scope; they must produce the generic reference output.
    let mut model = test_llama_model();
    model.fast_decode_inter_dim = None;
    let backend = CpuBackend;
    let mut cache = model.create_cache(&backend, 64);
    let prompt = [3u32, 1, 7];
    model
        .forward_with_cache(&backend, &prompt, &mut cache, 0)
        .unwrap();
    model.set_execution_mode(ExecutionMode::Planned);
    let planned =
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &[2, 5], &mut cache, 3)
            .unwrap();
    model.set_execution_mode(ExecutionMode::Reference);
    let mut reference_cache = model.create_cache(&backend, 64);
    model
        .forward_with_cache(&backend, &prompt, &mut reference_cache, 0)
        .unwrap();
    let reference = ForwardModel::forward_last_logits_with_cache(
        &model,
        &backend,
        &[2, 5],
        &mut reference_cache,
        3,
    )
    .unwrap();
    assert_eq!(
        planned.data(),
        reference.data(),
        "multi-token planned must take the generic path"
    );
    assert!(
        model.decode_state.borrow().is_none(),
        "multi-token calls must not build a decode session"
    );
}

/// Gate E (contract section 13): after warmup, the planned decode loop
/// with hooks disabled performs zero heap allocations per token other
/// than the documented logits tensor materialization — `CpuTensor` for
/// the `[1, vocab]` output allocates shape, strides, and data (3).
#[test]
fn planned_decode_is_zero_steady_state_allocation() {
    let mut model = test_llama_model_with_layers(2);
    model.fast_decode_inter_dim = None;
    model.set_execution_mode(ExecutionMode::Planned);
    let backend = CpuBackend;
    let mut cache = model.create_cache(&backend, 64);
    let prompt = [3u32, 1, 7];
    model
        .forward_with_cache(&backend, &prompt, &mut cache, 0)
        .unwrap();
    // warmup: build the plan + decode session + run one token
    ForwardModel::forward_last_logits_with_cache(&model, &backend, &[3], &mut cache, 3).unwrap();
    assert!(crate::alloc_counter::counting_active());
    let (_, allocations) = crate::alloc_counter::count_allocations(|| {
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &[5], &mut cache, 4)
            .unwrap();
    });
    assert!(
        allocations <= 3,
        "planned decode allocated {allocations} times per token; expected at most 3 (the logits CpuTensor shape + strides + data)"
    );
}

#[test]
fn reusable_planned_logits_preserve_results_without_allocating() {
    let backend = CpuBackend;
    for mode in [ExecutionMode::Planned, ExecutionMode::PlannedFused] {
        let mut model = test_llama_model_with_layers(2);
        model.fast_decode_inter_dim = None;
        model.set_execution_mode(mode);
        let mut cache = model.create_cache(&backend, 64);
        let mut output = ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &[3, 1, 7],
            &mut cache,
            0,
        )
        .unwrap();
        let mut reference_cache = cache.clone();
        let pointer = output.data().as_ptr();
        for (position, token) in [(3, 3), (4, 5), (5, 2)] {
            let expected = ForwardModel::forward_last_logits_with_cache(
                &model,
                &backend,
                &[token],
                &mut reference_cache,
                position,
            )
            .unwrap();
            assert!(crate::alloc_counter::counting_active());
            let (result, allocations) = crate::alloc_counter::count_allocations(|| {
                ForwardModel::forward_last_logits_with_cache_reusing(
                    &model,
                    &backend,
                    &[token],
                    &mut cache,
                    position,
                    &mut output,
                )
            });
            result.unwrap();
            assert_eq!(allocations, 0, "{mode:?}: reusable logits allocated");
            assert_eq!(output.data().as_ptr(), pointer);
            assert_eq!(output.data(), expected.data());
            assert_eq!(
                cache.export_compact_prefix(position + 1).unwrap(),
                reference_cache.export_compact_prefix(position + 1).unwrap()
            );
        }
    }
}

#[test]
fn planned_logits_reject_wrong_output_size_before_cache_mutation() {
    let mut model = test_llama_model();
    model.fast_decode_inter_dim = None;
    model.set_execution_mode(ExecutionMode::Planned);
    let mut cache = model.create_cache(&CpuBackend, 64);
    let before = cache.export_compact_prefix(0).unwrap();
    let mut output = [0.0];
    let error = forward_last_logits_planned_into(
        &model,
        &[3],
        &mut cache,
        0,
        None,
        HookMode::Disabled,
        &[],
        &mut output,
    )
    .unwrap_err();
    assert!(error.to_string().contains("wrong size"));
    assert_eq!(cache.cursor(), 0);
    assert_eq!(cache.export_compact_prefix(0).unwrap(), before);
}

/// Gate C (contract section 12): the planned path with an initialized
/// hook system but a no-op experiment must be bit-identical to the
/// plain planned path (hooks disabled).
#[test]
fn planned_inactive_hooks_match_plain_planned() {
    let mut model = test_llama_model_with_layers(2);
    model.fast_decode_inter_dim = None;

    // plain planned decode (hooks fully disabled)
    let plain = plain_decode(&model, ExecutionMode::Planned, 5);

    // planned decode through the experiment path with a no-op (recording)
    // experiment: same kernels, same sites, nothing mutates values.
    let (hooked, records) = decode_recording_hooks(&model, ExecutionMode::Planned);

    assert_eq!(
        plain.data(),
        hooked.data(),
        "planned decode with inactive hooks must be bit-identical to hooks disabled"
    );
    assert!(
        !records.is_empty(),
        "the hook system must actually have fired on the planned path"
    );
}

/// `(layer, shape, values)` of every per-head row a probe saw.
type HeadRows = Vec<(usize, [usize; 2], Vec<f32>)>;

/// Uses only the per-head stage: records every per-head row it sees and,
/// with `zero = Some((layer, head))`, zeroes that head (`None`: every
/// head) of that layer before the output projection reads it.
struct HeadsProbe {
    head_dim: usize,
    zero: Option<(usize, Option<usize>)>,
    seen: std::sync::Arc<std::sync::Mutex<HeadRows>>,
}

impl crate::experiments::Experiment for HeadsProbe {
    fn name(&self) -> &'static str {
        "heads-probe"
    }

    fn intervenes(&self) -> bool {
        self.zero.is_some()
    }

    fn uses_activation_stage(&self, stage: crate::hook_types::ActivationStage) -> bool {
        stage == crate::hook_types::ActivationStage::BeforeOProj
    }

    fn before_o_proj(
        &mut self,
        ctx: &crate::experiments::LayerContext<'_>,
        heads: &mut crate::experiments::TensorAccess<'_>,
    ) -> Result<(), crate::experiments::ExperimentError> {
        let shape = *heads.shape();
        self.seen
            .lock()
            .unwrap()
            .push((ctx.layer_index, shape, heads.values().to_vec()));
        if let Some((layer, head)) = self.zero
            && layer == ctx.layer_index
        {
            let width = self.head_dim;
            for row in heads.values_mut().chunks_exact_mut(shape[1]) {
                match head {
                    None => row.fill(0.0),
                    Some(head) => row[head * width..(head + 1) * width].fill(0.0),
                }
            }
        }
        Ok(())
    }
}

/// Decode through `mode` with a [`HeadsProbe`]: the logits and every
/// `(layer, shape, values)` the probe saw.
fn decode_with_heads_probe(
    model: &Llama<CpuBackend>,
    mode: ExecutionMode,
    zero: Option<(usize, Option<usize>)>,
) -> (CpuTensor, HeadRows) {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let probe = HeadsProbe {
        head_dim: model.config.head_dim,
        zero,
        seen: std::sync::Arc::clone(&seen),
    };
    let logits = decode_with_runner(model, mode, &mut ExperimentRunner::new(probe));
    let seen = seen.lock().unwrap().clone();
    (logits, seen)
}

/// The per-head site fires once per layer with one `[1, n_heads *
/// head_dim]` row on the Q8 fast, generic, planned and planned-fused
/// decode paths, and observing it changes no logit. The generic, planned
/// and planned-fused values are bit-identical; the fast path quantizes
/// its activations on its own route, so it agrees to 1% of the row's
/// largest value.
#[test]
fn per_head_site_fires_on_every_decode_path() {
    let fast = test_llama_model_with_layers(2);
    assert!(fast.fast_decode_inter_dim.is_some());
    let mut generic = test_llama_model_with_layers(2);
    generic.fast_decode_inter_dim = None;
    let q_dim = generic.config.n_heads * generic.config.head_dim;

    let (_, reference) = decode_with_heads_probe(&generic, ExecutionMode::Reference, None);
    assert_eq!(
        reference
            .iter()
            .map(|(layer, shape, _)| (*layer, *shape))
            .collect::<Vec<_>>(),
        [(0, [1, q_dim]), (1, [1, q_dim])]
    );
    let paths = [
        ("fast", &fast, ExecutionMode::Reference),
        ("generic", &generic, ExecutionMode::Reference),
        ("planned", &generic, ExecutionMode::Planned),
        ("planned-fused", &generic, ExecutionMode::PlannedFused),
    ];
    for (label, model, mode) in paths {
        let (logits, seen) = decode_with_heads_probe(model, mode, None);
        assert_eq!(seen.len(), 2, "{label}");
        for ((layer, shape, values), (_, _, expected)) in seen.iter().zip(&reference) {
            assert_eq!(*shape, [1, q_dim], "{label} layer {layer}");
            let max_abs = max_abs_diff(values, expected);
            if label == "fast" {
                let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
                assert!(max_abs <= 0.01 * scale, "{label} layer {layer}: {max_abs}");
            } else {
                assert_eq!(values, expected, "{label} layer {layer}");
            }
        }
        let plain = plain_decode(model, mode, 5);
        assert_eq!(
            plain.data(),
            logits.data(),
            "{label}: observing the per-head site changed the logits"
        );
    }
}

/// On a model whose output projection planned-fused fuses (F5), the
/// per-head site leaves F5 fused: its tensor is the attention op's own
/// output, which F5 reads.
#[test]
fn per_head_site_keeps_f5_fused() {
    let mut model = test_llama_model_with_layers(2);
    model.fast_decode_inter_dim = None;
    for (layer, block) in model.blocks.iter_mut().enumerate() {
        block.self_attn.o_proj = test_f32_linear(32, 32, 100 + layer);
    }
    let (_, reference) = decode_with_heads_probe(&model, ExecutionMode::Reference, None);
    crate::planned_decode::reset_fused_execution_counts();
    let (logits, seen) = decode_with_heads_probe(&model, ExecutionMode::PlannedFused, None);
    let counts = crate::planned_decode::fused_execution_counts();
    assert!(counts.f5_output_proj_residual > 0, "{counts:?}");
    assert_eq!(seen, reference);
    assert_eq!(
        logits.data(),
        plain_decode(&model, ExecutionMode::PlannedFused, 5).data()
    );
    // An intervention there lands before F5 reads the tensor.
    let (zeroed_reference, _) =
        decode_with_heads_probe(&model, ExecutionMode::Reference, Some((1, Some(1))));
    crate::planned_decode::reset_fused_execution_counts();
    let (zeroed_fused, _) =
        decode_with_heads_probe(&model, ExecutionMode::PlannedFused, Some((1, Some(1))));
    let counts = crate::planned_decode::fused_execution_counts();
    assert!(counts.f5_output_proj_residual > 0, "{counts:?}");
    let max_abs = max_abs_diff(zeroed_reference.data(), zeroed_fused.data());
    assert!(max_abs <= 1e-4, "{max_abs}");
    assert_ne!(zeroed_fused.data(), logits.data());
}

/// Zeroing every head before the output projection is zeroing the
/// attention output (the projection has no bias), on every decode path.
#[test]
fn zeroing_every_head_matches_zeroing_the_attention_output() {
    let fast = test_llama_model_with_layers(2);
    let mut generic = test_llama_model_with_layers(2);
    generic.fast_decode_inter_dim = None;
    let paths = [
        ("fast", &fast, ExecutionMode::Reference),
        ("generic", &generic, ExecutionMode::Reference),
        ("planned", &generic, ExecutionMode::Planned),
        ("planned-fused", &generic, ExecutionMode::PlannedFused),
    ];
    for (label, model, mode) in paths {
        let (heads_zeroed, _) = decode_with_heads_probe(model, mode, Some((0, None)));
        let mut runner = ExperimentRunner::new(ZeroLayerOutput::new(ZeroLayerOutputSpec::new(
            0,
            ZeroLayerOutputStage::Attention,
        )));
        let output_zeroed = decode_with_runner(model, mode, &mut runner);
        assert_eq!(heads_zeroed.data(), output_zeroed.data(), "{label}");
        let (one_head, _) = decode_with_heads_probe(model, mode, Some((0, Some(0))));
        assert_ne!(
            one_head.data(),
            plain_decode(model, mode, 5).data(),
            "{label}"
        );
        assert_ne!(one_head.data(), heads_zeroed.data(), "{label}");
    }
}

/// A plan records the per-head site only where it is active; every other
/// part of the plan is the plan without it.
#[test]
fn per_head_site_is_recorded_only_when_active() {
    let model = test_llama_model_with_layers(2);
    for mode in [ExecutionMode::Planned, ExecutionMode::PlannedFused] {
        let plan = |sites: &[&str]| {
            model
                .execution_plan(mode, HookMode::Observe, sites, 8, None, None)
                .unwrap()
        };
        let base = plan(&["after-mlp"]);
        let with = plan(&["after-mlp", "before-o-proj@1"]);
        assert!(base
            .hook_sites
            .sites
            .iter()
            .all(|site| site.stage != "before-o-proj"));
        let records: Vec<_> = with
            .hook_sites
            .sites
            .iter()
            .filter(|site| site.stage == "before-o-proj")
            .collect();
        assert_eq!(records.len(), 1);
        let attention_out = with.layers[1]
            .ops
            .iter()
            .find_map(|op| match op {
                PlannedOp::Attention { out, .. } => Some(out.id),
                _ => None,
            })
            .unwrap();
        assert_eq!(records[0].layer, Some(1));
        assert_eq!(records[0].tensor, Some(attention_out));
        assert!(records[0].materialized);
        let others: Vec<_> = with
            .hook_sites
            .sites
            .iter()
            .filter(|site| site.stage != "before-o-proj")
            .map(|site| serde_json::to_value(site).unwrap())
            .collect();
        let base_sites: Vec<_> = base
            .hook_sites
            .sites
            .iter()
            .map(|site| serde_json::to_value(site).unwrap())
            .collect();
        assert_eq!(others, base_sites, "{mode:?}");
        assert_eq!(
            serde_json::to_value(&with.layers).unwrap(),
            serde_json::to_value(&base.layers).unwrap(),
            "{mode:?}: the per-head site changed the op sequence"
        );
        assert_eq!(with.validate().ok(), Some(()));
    }
}

/// Gate C: the planned path fires the six semantic hook sites with the
/// same stages, layers, and shapes as the reference path.
#[test]
fn planned_hook_sites_match_reference() {
    let mut model = test_llama_model_with_layers(2);
    model.fast_decode_inter_dim = None;

    let (reference, reference_records) = decode_recording_hooks(&model, ExecutionMode::Reference);
    let (planned, planned_records) = decode_recording_hooks(&model, ExecutionMode::Planned);

    assert_eq!(
        reference_records.len(),
        planned_records.len(),
        "reference and planned must fire the same number of hooks"
    );
    for (expected, actual) in reference_records.iter().zip(&planned_records) {
        assert_eq!(expected.hook, actual.hook, "hook stage mismatch");
        assert_eq!(
            expected.layer_index, actual.layer_index,
            "hook layer mismatch"
        );
        assert_eq!(expected.shape, actual.shape, "hook shape mismatch");
    }
    let max_abs = max_abs_diff(reference.data(), planned.data());
    assert!(
        max_abs <= 1e-4,
        "hooked planned logits diverge from hooked reference: {max_abs}"
    );
}

/// Gate C: an intervention (zero-layer-output patch) on the planned path
/// lands at the same tensor and produces the same patched output as the
/// reference path.
#[test]
fn planned_patch_intervention_matches_reference() {
    let mut model = test_llama_model_with_layers(2);
    model.fast_decode_inter_dim = None;

    let run_patched = |mode: ExecutionMode| -> CpuTensor {
        let mut runner = ExperimentRunner::new(ZeroLayerOutput::new(ZeroLayerOutputSpec::new(
            0,
            ZeroLayerOutputStage::Attention,
        )));
        decode_with_runner(&model, mode, &mut runner)
    };

    let reference = run_patched(ExecutionMode::Reference);
    let planned = run_patched(ExecutionMode::Planned);
    let max_abs = max_abs_diff(reference.data(), planned.data());
    assert!(
        max_abs <= 1e-4,
        "patched planned logits diverge from patched reference: {max_abs}"
    );
    // the patch must actually have changed the planned output
    let plain = plain_decode(&model, ExecutionMode::Planned, 5);
    assert_ne!(plain.data(), planned.data(), "the intervention must fire");
}

/// Phase 7: the planned single-token attention must be bit-identical to
/// the reference `cached_causal_attention_with_scratch` single-token
/// branch across sequence lengths (same math, same simd primitives).
#[test]
fn planned_attention_matches_reference_single_token() {
    let backend = CpuBackend;
    let (n_heads, n_kv_heads, head_dim, max_seq_len) = (8, 4, 32, 2048);
    let mut cache = crate::kv_cache::KVCache::new(1, n_kv_heads, head_dim, max_seq_len);
    // deterministic fill (xorshift-style LCG)
    let mut state = 0x1234_5678u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f32 / (1u64 << 31) as f32
    };
    for pos in 0..1025 {
        let k: Vec<f32> = (0..n_kv_heads * head_dim).map(|_| next()).collect();
        let v: Vec<f32> = (0..n_kv_heads * head_dim).map(|_| next()).collect();
        cache.append_with_layout(0, pos, &k, &v, n_kv_heads, head_dim);
    }
    let q_data: Vec<f32> = (0..n_heads * head_dim).map(|_| next()).collect();
    let q = CpuTensor::from_data(vec![1, n_heads * head_dim], q_data.clone());
    let (cached_k, cached_v) = cache.get(0);
    let mut qk_scratch = Vec::new();
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for total_seq_len in [1usize, 2, 3, 5, 8, 129, 1025] {
        let reference = serial.install(|| {
            backend
                .cached_causal_attention_with_scratch(
                    &q,
                    cached_k,
                    cached_v,
                    CachedAttentionSpec {
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_seq_len,
                        total_seq_len,
                    },
                    &mut qk_scratch,
                )
                .unwrap()
        });
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut scores = vec![0.0f32; n_heads * max_seq_len];
            let mut out = vec![0.0f32; n_heads * head_dim];
            pool.install(|| {
                planned_causal_attention(
                    &q_data,
                    cached_k,
                    cached_v,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_seq_len,
                    total_seq_len,
                    &mut scores,
                    &mut out,
                )
                .unwrap()
            });
            assert_eq!(
            reference.data(),
            &out[..],
            "planned attention diverged from reference at total_seq_len {total_seq_len}, threads {threads}"
        );
        }
    }
}

/// Phase 8: the fused planned decode (F1-F5) must match the reference
/// greedy tokens and stay within the frozen Gate A/B logit envelope.
#[test]
fn planned_fused_matches_reference_greedy() {
    let mut model = test_llama_model_with_layers(2);
    model.fast_decode_inter_dim = None;
    for (layer, block) in model.blocks.iter_mut().enumerate() {
        block.self_attn.o_proj = test_f32_linear(32, 32, 100 + layer);
    }

    let (reference_tokens, reference_logits) =
        run_greedy(&model, ExecutionMode::Reference, Prefill::Trait);
    crate::planned_decode::reset_fused_execution_counts();
    let (fused_tokens, fused_logits) =
        run_greedy(&model, ExecutionMode::PlannedFused, Prefill::Trait);
    let counts = crate::planned_decode::fused_execution_counts();
    assert!(
        counts.f1_rmsnorm_linear > 0,
        "F1 never executed: {counts:?}"
    );
    assert!(
        counts.f3_qkv_orchestration > 0,
        "F3 never executed: {counts:?}"
    );
    assert!(
        counts.f4_rope_in_attention > 0,
        "F4 never executed: {counts:?}"
    );
    assert!(
        counts.f5_output_proj_residual > 0,
        "F5 never executed: {counts:?}"
    );

    assert_eq!(
        reference_tokens, fused_tokens,
        "fused greedy tokens must match the reference"
    );
    let max_abs = max_abs_diff(reference_logits.data(), fused_logits.data());
    assert!(
        max_abs <= 1e-4,
        "fused logits diverge from reference: max_abs {max_abs}"
    );
}

#[test]
fn planned_fused_defuses_f5_for_q8_assignment_kernel() {
    let mut model = test_llama_model_with_layers(2);
    model.fast_decode_inter_dim = None;
    let reference = plain_decode(&model, ExecutionMode::Reference, 7);
    crate::planned_decode::reset_fused_execution_counts();
    let fused = plain_decode(&model, ExecutionMode::PlannedFused, 7);
    let counts = crate::planned_decode::fused_execution_counts();
    assert_eq!(counts.f5_output_proj_residual, 0, "{counts:?}");
    assert!(counts.f2_residual_rmsnorm > 0, "{counts:?}");
    let max_abs = max_abs_diff(reference.data(), fused.data());
    assert!(max_abs <= 1e-4, "Q8 defused logits diverged: {max_abs}");

    let plan = model
        .execution_plan(
            ExecutionMode::PlannedFused,
            HookMode::Disabled,
            &[],
            64,
            None,
            None,
        )
        .unwrap();
    assert!(plan.layers.iter().all(|layer| {
        layer.fusion == FusionState::PartiallyFused
            && layer
                .fusion_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("assignment semantics"))
    }));
}

/// Phase 8 defusion: planned-fused with an after_attention hook must
/// defuse F5, fire the hook on the materialized o tensor, and stay
/// within the Gate C envelope of the hooked reference path.
#[test]
fn planned_fused_defuses_after_attention_hook() {
    let mut model = test_llama_model_with_layers(2);
    model.fast_decode_inter_dim = None;

    // the reference path fires every stage (unfused)
    let (reference, reference_records) = decode_recording_hooks(&model, ExecutionMode::Reference);
    crate::planned_decode::reset_fused_execution_counts();
    let (fused, fused_records) = decode_recording_hooks(&model, ExecutionMode::PlannedFused);
    let counts = crate::planned_decode::fused_execution_counts();
    assert!(
        counts.f2_residual_rmsnorm > 0,
        "F2 never executed: {counts:?}"
    );
    assert_eq!(
        counts.f5_output_proj_residual, 0,
        "F5 executed despite after_attention defusion: {counts:?}"
    );

    assert!(
        fused_records
            .iter()
            .any(|r| r.hook == ExperimentHook::AfterAttention),
        "after_attention must fire on the fused path (F5 defused)"
    );
    assert_eq!(
        reference_records.len(),
        fused_records.len(),
        "defused fused path must fire the same hook sequence as reference"
    );
    let max_abs = max_abs_diff(reference.data(), fused.data());
    assert!(
        max_abs <= 1e-4,
        "defused fused logits diverge from hooked reference: {max_abs}"
    );
}

#[test]
fn request_cache_is_sized_to_the_request() {
    // The request-sized cache must never exceed the actual work
    // (prompt + generated tokens), and never the model's context.
    let model = test_llama_model_with_layers(2);
    let backend = CpuBackend;
    // the test model's metadata context is 8; requests above it clamp
    let cache = model.create_request_cache(&backend, 102, 16);
    assert_eq!(cache.max_seq_len(), 8);
    let cache = model.create_request_cache(&backend, 5, 1);
    assert_eq!(cache.max_seq_len(), 6);
}
#[test]
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn retained_reference_rope_matches_reference_and_planned_layouts() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/rope-reference/llama-q6-position15.json"
    ))
    .unwrap();
    let position = fixture["position"].as_u64().unwrap() as usize;
    let dim = fixture["head_dim"].as_u64().unwrap() as usize;
    let factors: Vec<f32> = serde_json::from_value(fixture["freq_factors"].clone()).unwrap();
    let (cos, sin) = crate::tensor::compute_rope_freqs(
        position + 1,
        dim,
        fixture["theta_base"].as_f64().unwrap() as f32,
        Some(&factors),
    );
    for layout in [RopeLayout::AdjacentPair, RopeLayout::SplitHalf] {
        let attention = LlamaAttention::new(
            testkit::q8_linear(dim, dim, 1),
            testkit::q8_linear(dim, dim, 2),
            testkit::q8_linear(dim, dim, 3),
            testkit::q8_linear(dim, dim, 4),
            cos.clone(),
            sin.clone(),
            1,
            1,
            dim,
            layout,
            QkNormOrder::AfterRope,
            None,
            None,
        );
        for record in fixture["records"].as_array().unwrap() {
            let mut input: Vec<f32> = serde_json::from_value(record["input"].clone()).unwrap();
            let mut expected: Vec<f32> =
                serde_json::from_value(record["expected"].clone()).unwrap();
            if layout == RopeLayout::SplitHalf {
                let reorder = |v: Vec<f32>| {
                    v.iter()
                        .step_by(2)
                        .chain(v.iter().skip(1).step_by(2))
                        .copied()
                        .collect()
                };
                input = reorder(input);
                expected = reorder(expected);
            }
            let actual = apply_rope_and_qk_norm(
                &CpuBackend,
                CpuTensor::from_data(vec![1, dim], input.clone()),
                &cos,
                &sin,
                RopeQkNormSpec {
                    qk_norm_eps: DEFAULT_QK_NORM_EPS,
                    start_pos: position,
                    n_heads: 1,
                    head_dim: dim,
                    rope_layout: layout,
                    qk_norm_order: QkNormOrder::AfterRope,
                },
                None,
                None,
            )
            .unwrap();
            let mut planned = input;
            attention.apply_decode_rope_and_qk_norm(&mut planned, 1, position, None);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(actual.data()),
                bits(&expected),
                "reference {:?} {}",
                layout,
                record["name"]
            );
            assert_eq!(
                bits(&planned),
                bits(&expected),
                "planned {:?} {}",
                layout,
                record["name"]
            );
        }
    }
}

// -- batched multi-sequence decode ------------------------------------------

/// Records every hook site's activation bits and applies a
/// sequence-specific intervention after layer 0 attention and on the
/// layer 1 per-head attention output.
struct BatchProbeHooks {
    nudge: f32,
    records: Vec<(u8, usize, Vec<u32>)>,
    dispatches: Vec<DispatchPath>,
}

impl BatchProbeHooks {
    fn new(sequence: usize) -> Self {
        Self {
            nudge: sequence as f32 * 0.03125,
            records: Vec::new(),
            dispatches: Vec::new(),
        }
    }

    fn record(&mut self, site: u8, layer: usize, tensor: &mut SliceActivation<'_>) {
        let values = tensor.values_mut_for_test();
        if (site == 1 && layer == 0) || (site == 6 && layer == 1) {
            values[0] += self.nudge;
        }
        self.records
            .push((site, layer, values.iter().map(|v| v.to_bits()).collect()));
    }
}

impl<'a> LayerHooks<SliceActivation<'a>, CpuError> for BatchProbeHooks {
    fn note_dispatch(&mut self, path: DispatchPath) {
        self.dispatches.push(path);
    }
    fn before_layer(&mut self, layer: usize, t: &mut SliceActivation<'a>) -> Result<(), CpuError> {
        self.record(0, layer, t);
        Ok(())
    }
    fn before_o_proj(&mut self, layer: usize, t: &mut SliceActivation<'a>) -> Result<(), CpuError> {
        self.record(6, layer, t);
        Ok(())
    }
    fn after_attention(
        &mut self,
        layer: usize,
        t: &mut SliceActivation<'a>,
    ) -> Result<(), CpuError> {
        self.record(1, layer, t);
        Ok(())
    }
    fn after_mlp(&mut self, layer: usize, t: &mut SliceActivation<'a>) -> Result<(), CpuError> {
        self.record(2, layer, t);
        Ok(())
    }
    fn after_layer(&mut self, layer: usize, t: &mut SliceActivation<'a>) -> Result<(), CpuError> {
        self.record(3, layer, t);
        Ok(())
    }
    fn before_logits(&mut self, t: &mut SliceActivation<'a>) -> Result<(), CpuError> {
        self.record(4, usize::MAX, t);
        Ok(())
    }
    fn after_logits(&mut self, t: &mut SliceActivation<'a>) -> Result<(), CpuError> {
        self.record(5, usize::MAX, t);
        Ok(())
    }
}

/// Wider synthetic Llama whose projections cross the parallel decode
/// threshold once several sequences are batched (and the LM head alone).
fn batch_test_model() -> Llama<CpuBackend> {
    test_llama_model_with_dims(2, 256, 4099)
}

fn bits_of(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn batched_decode_matches_independent_single_decodes_bit_for_bit() {
    let model = batch_test_model();
    let backend = CpuBackend;
    let vocab = model.config.vocab_size;
    assert!(model.supports_batched_decode());
    for n in 1..=17usize {
        // Sequence i starts after a prefill of 1 + i % 3 tokens, so the
        // batch mixes positions and cache lengths.
        let prompt = |i: usize| -> Vec<u32> {
            (0..1 + i % 3)
                .map(|t| ((i * 7 + t * 3) % vocab) as u32)
                .collect()
        };
        let mut batch_caches = Vec::new();
        let mut single_caches = Vec::new();
        let mut positions = Vec::new();
        for i in 0..n {
            let tokens = prompt(i);
            for caches in [&mut batch_caches, &mut single_caches] {
                let mut cache = model.create_cache(&backend, model.config.max_seq_len);
                ForwardModel::forward_last_logits_with_cache(
                    &model, &backend, &tokens, &mut cache, 0,
                )
                .unwrap();
                caches.push(cache);
            }
            positions.push(tokens.len());
        }
        let steps = if n <= 4 { 4 } else { 2 };
        for step in 0..steps {
            let tokens: Vec<u32> = (0..n)
                .map(|i| ((i * 11 + step * 5 + 1) % vocab) as u32)
                .collect();
            let mut batch_logits = vec![vec![f32::NAN; vocab]; n];
            let mut batch_hooks: Vec<BatchProbeHooks> = (0..n).map(BatchProbeHooks::new).collect();
            {
                let mut sequences: Vec<DecodeBatchSequence<'_>> = batch_caches
                    .iter_mut()
                    .zip(batch_logits.iter_mut())
                    .enumerate()
                    .map(|(i, (cache, logits))| DecodeBatchSequence {
                        token_id: tokens[i],
                        cache,
                        start_pos: positions[i],
                        logits,
                    })
                    .collect();
                model
                    .forward_decode_batch_hooked(&backend, &mut sequences, &mut batch_hooks)
                    .unwrap();
            }
            for i in 0..n {
                let mut hooks = BatchProbeHooks::new(i);
                let mut logits = vec![f32::NAN; vocab];
                model
                    .forward_decode_fast_hooked(
                        &backend,
                        &[tokens[i]],
                        &mut single_caches[i],
                        positions[i],
                        &mut hooks,
                        Some(&mut logits),
                    )
                    .expect("fast path eligible")
                    .unwrap();
                assert_eq!(
                    bits_of(&batch_logits[i]),
                    bits_of(&logits),
                    "logits n={n} step={step} seq={i}"
                );
                assert_eq!(batch_hooks[i].records, hooks.records, "hooks n={n} seq={i}");
                assert_eq!(batch_hooks[i].dispatches, hooks.dispatches);
                assert_eq!(batch_caches[i].cursor(), single_caches[i].cursor());
                for layer in 0..model.blocks.len() {
                    let (bk, bv) = batch_caches[i].get(layer);
                    let (sk, sv) = single_caches[i].get(layer);
                    assert!(bk.iter().zip(sk).all(|(a, b)| a.to_bits() == b.to_bits()));
                    assert!(bv.iter().zip(sv).all(|(a, b)| a.to_bits() == b.to_bits()));
                }
                positions[i] += 1;
            }
        }
    }
}

#[test]
fn batched_decode_with_experiments_matches_single_experiment_decode() {
    let model = batch_test_model();
    let backend = CpuBackend;
    let vocab = model.config.vocab_size;
    let model_context =
        ModelContext::new(ModelFamily::Llama, None, "llama", 2, model.config.embed_dim);
    // Even sequences zero layer-0 attention, odd sequences zero layer-1 MLP.
    let make_runner = |i: usize| {
        let spec = if i.is_multiple_of(2) {
            ZeroLayerOutputSpec::new(0, ZeroLayerOutputStage::Attention)
        } else {
            ZeroLayerOutputSpec::new(1, ZeroLayerOutputStage::Mlp)
        };
        let mut runner = ExperimentRunner::new(ZeroLayerOutput::new(spec));
        runner.on_model_loaded(&model_context).unwrap();
        runner
    };
    let n = 5;
    let mut batch_caches = Vec::new();
    let mut single_caches = Vec::new();
    for i in 0..n {
        for caches in [&mut batch_caches, &mut single_caches] {
            let mut cache = model.create_cache(&backend, model.config.max_seq_len);
            ForwardModel::forward_last_logits_with_cache(
                &model,
                &backend,
                &[(i + 2) as u32],
                &mut cache,
                0,
            )
            .unwrap();
            caches.push(cache);
        }
    }
    let mut batch_runners: Vec<ExperimentRunner> = (0..n).map(make_runner).collect();
    let mut batch_logits = vec![vec![0.0f32; vocab]; n];
    let execution = ExecutionContext::new(
        model_context,
        ExecutionPhase::Decode,
        1,
        1,
        TracingState::Disabled,
    );
    {
        let mut sequences: Vec<DecodeBatchSequence<'_>> = batch_caches
            .iter_mut()
            .zip(batch_logits.iter_mut())
            .enumerate()
            .map(|(i, (cache, logits))| DecodeBatchSequence {
                token_id: (i * 13 % vocab) as u32,
                cache,
                start_pos: 1,
                logits,
            })
            .collect();
        let mut experiments: Vec<DecodeBatchExperiment<'_, '_>> = batch_runners
            .iter_mut()
            .map(|runner| DecodeBatchExperiment { runner, execution })
            .collect();
        model
            .forward_decode_batch_with_experiments(&backend, &mut sequences, &mut experiments)
            .unwrap();
    }
    for i in 0..n {
        let mut runner = make_runner(i);
        let single = ExperimentalForwardModel::forward_last_logits_with_experiment(
            &model,
            &backend,
            &[(i * 13 % vocab) as u32],
            &mut single_caches[i],
            1,
            execution,
            &mut runner,
        )
        .unwrap();
        assert_eq!(bits_of(&batch_logits[i]), bits_of(single.data()), "seq {i}");
    }
    // Different interventions really produced different sequences.
    assert_ne!(bits_of(&batch_logits[0]), bits_of(&batch_logits[1]));
}

#[test]
fn batched_decode_rejects_bad_shapes_before_touching_caches() {
    let model = batch_test_model();
    let backend = CpuBackend;
    let mut cache = model.create_cache(&backend, model.config.max_seq_len);
    let mut short = vec![0.0f32; 3];
    let mut sequences = [DecodeBatchSequence {
        token_id: 1,
        cache: &mut cache,
        start_pos: 0,
        logits: &mut short,
    }];
    assert!(model
        .forward_decode_batch(&backend, &mut sequences)
        .is_err());
    assert_eq!(cache.cursor(), 0);
    assert!(model.forward_decode_batch(&backend, &mut []).is_ok());

    // A position that disagrees with the cursor, a full cache and a
    // cache of the wrong geometry are errors, not panics.
    let vocab = model.config.vocab_size;
    let mut logits = vec![0.0f32; vocab];
    let mut sequences = [DecodeBatchSequence {
        token_id: 1,
        cache: &mut cache,
        start_pos: 3,
        logits: &mut logits,
    }];
    let error = model
        .forward_decode_batch(&backend, &mut sequences)
        .unwrap_err();
    assert!(error.to_string().contains("cursor"), "{error}");
    let mut full = model.create_cache(&backend, 1);
    let mut sequences = [DecodeBatchSequence {
        token_id: 1,
        cache: &mut full,
        start_pos: 0,
        logits: &mut logits,
    }];
    model
        .forward_decode_batch(&backend, &mut sequences)
        .unwrap();
    assert_eq!(full.cursor(), 1);
    let mut sequences = [DecodeBatchSequence {
        token_id: 1,
        cache: &mut full,
        start_pos: 1,
        logits: &mut logits,
    }];
    let error = model
        .forward_decode_batch(&backend, &mut sequences)
        .unwrap_err();
    assert!(error.to_string().contains("full"), "{error}");
    let mut wide = crate::kv_cache::KVCache::new(
        model.blocks.len(),
        model.config.n_kv_heads,
        model.config.head_dim * 2,
        8,
    );
    let mut sequences = [DecodeBatchSequence {
        token_id: 1,
        cache: &mut wide,
        start_pos: 0,
        logits: &mut logits,
    }];
    let error = model
        .forward_decode_batch(&backend, &mut sequences)
        .unwrap_err();
    assert!(error.to_string().contains("KV heads"), "{error}");
}

#[cfg(test)]
mod reference_rope_diagnostic {
    use super::*;

    #[test]
    #[ignore = "requires pinned Q6 model and reference projection/rotation captures"]
    fn diagnose_reference_input_rope() {
        let model = std::env::var("EMBER_PARITY_MODEL").expect("model required");
        assert_eq!(
            crate::plan::hex(&<sha2::Sha256 as sha2::Digest>::digest(
                std::fs::read(&model).unwrap()
            )),
            "4bf385159856b7c50a938b1228112318d9f99238a76880ea0f6381ab879982b3"
        );
        let source = std::env::var("EMBER_PARITY_REFERENCE_TENSORS").unwrap();
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&source).unwrap()).unwrap();
        let control: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::env::var("EMBER_PARITY_REFERENCE_OUTPUT").unwrap()).unwrap(),
        )
        .unwrap();
        let prompt_len = control["input_token_ids"].as_array().unwrap().len();
        // Evaluation 1 consumes the prompt; evaluation 10 consumes generated token 9.
        let position = prompt_len.checked_add(8).unwrap();
        let rows: std::collections::BTreeMap<String, Vec<f32>> = reference["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                assert_eq!(t["step"], 10);
                assert_eq!(t["row"], 0);
                let values: Vec<f32> = serde_json::from_value(t["values"].clone()).unwrap();
                assert_eq!(values.len(), t["width"].as_u64().unwrap() as usize);
                assert!(values.iter().all(|x| x.is_finite()));
                (t["name"].as_str().unwrap().to_string(), values)
            })
            .collect();
        let loader =
            crate::loader::load_gguf_with_k_strategy(&model, crate::quant_k::KStrategy::Arm, false)
                .unwrap();
        let config = LlamaConfig::from_gguf_metadata(&loader).unwrap();
        let factors = loader.tensors.get("rope_freqs.weight").map(|weight| {
            let crate::loader::LoadedTensor::F32(tensor) = weight else {
                panic!("expected f32 factors")
            };
            tensor.data()
        });
        let (cos, sin) = crate::tensor::compute_rope_freqs(
            position + 1,
            config.head_dim,
            config.rope_theta,
            factors,
        );
        // Diagnostic alternative matching the pinned reference's frequency recurrence.
        // The retained context log confirms freq_scale=1 for this pinned model/run.
        let half = config.head_dim / 2;
        let theta_scale = config.rope_theta.powf(-2.0 / config.head_dim as f32);
        let mut recurrent_cos = vec![0.0; (position + 1) * half];
        let mut recurrent_sin = vec![0.0; (position + 1) * half];
        for p in 0..=position {
            let mut theta = p as f32;
            for i in 0..half {
                let angle = theta / factors.map_or(1.0, |values| values[i]);
                recurrent_cos[p * half + i] = angle.cos();
                recurrent_sin[p * half + i] = angle.sin();
                theta *= theta_scale;
            }
        }
        let recurrent_cos = CpuTensor::from_data(vec![position + 1, half], recurrent_cos);
        let recurrent_sin = CpuTensor::from_data(vec![position + 1, half], recurrent_sin);
        let mut records = Vec::new();
        for (table_kind, cos, sin) in [
            ("production", &cos, &sin),
            ("reference_recurrence", &recurrent_cos, &recurrent_sin),
        ] {
            for (input_name, output_name, heads) in [
                ("q_projection-0", "q_rope-0", config.n_heads),
                ("k_projection-0", "k_rope-0", config.n_kv_heads),
            ] {
                let input = &rows[input_name];
                assert_eq!(input.len(), heads * config.head_dim);
                let actual = apply_rope_and_qk_norm(
                    &CpuBackend,
                    CpuTensor::from_data(vec![1, input.len()], input.clone()),
                    cos,
                    sin,
                    RopeQkNormSpec {
                        qk_norm_eps: DEFAULT_QK_NORM_EPS,
                        start_pos: position,
                        n_heads: heads,
                        head_dim: config.head_dim,
                        rope_layout: config.rope_layout,
                        qk_norm_order: config.qk_norm_order,
                    },
                    None,
                    None,
                )
                .unwrap();
                assert_eq!(actual.data().len(), rows[output_name].len());
                assert!(actual.data().iter().all(|x| x.is_finite()));
                records.push(
                serde_json::json!({"operation": output_name, "input": input_name,
                "position": position, "table_kind": table_kind, "actual": actual.data(), "reference": rows[output_name]}),
            );
            }
        }
        assert_eq!(config.rope_layout, RopeLayout::AdjacentPair);
        for (input_name, output_name) in [
            ("q_projection-0", "q_rope-0"),
            ("k_projection-0", "k_rope-0"),
        ] {
            for fuse_first in [false, true] {
                for fuse_second in [false, true] {
                    let mut actual = rows[input_name].clone();
                    for head in actual.chunks_exact_mut(config.head_dim) {
                        for i in 0..half {
                            let x0 = head[2 * i];
                            let x1 = head[2 * i + 1];
                            let c = recurrent_cos.data()[position * half + i];
                            let si = recurrent_sin.data()[position * half + i];
                            head[2 * i] = if fuse_first {
                                x0.mul_add(c, -(x1 * si))
                            } else {
                                (-x1).mul_add(si, x0 * c)
                            };
                            head[2 * i + 1] = if fuse_second {
                                x0.mul_add(si, x1 * c)
                            } else {
                                x1.mul_add(c, x0 * si)
                            };
                        }
                    }
                    assert!(actual.iter().all(|x| x.is_finite()));
                    records.push(serde_json::json!({"operation": output_name, "input": input_name,
                        "position": position, "table_kind": "reference_recurrence_fused_diagnostic",
                        "fuse_first_product_for_even": fuse_first, "fuse_first_product_for_odd": fuse_second,
                        "actual": actual, "reference": rows[output_name]}));
                }
            }
        }
        let output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(std::env::var("EMBER_PARITY_DIAGNOSTIC").unwrap())
            .unwrap();
        serde_json::to_writer_pretty(output, &serde_json::json!({
            "scope": "actual Ember RoPE plus explicitly labeled diagnostic fused variants on identical reference projections; no acceptance threshold",
            "reference_source": source, "records": records})).unwrap();
    }
}
