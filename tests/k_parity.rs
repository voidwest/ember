//! Real-model validation for the exact-f32 oracle, production Q8_K path,
//! planned routes, hooks, and allocation contract.
//!
//! Env-gated (skipped without the variables) because it loads real GGUFs:
//!
//! - `EMBER_PARITY_MODEL` — path to a Q4_K_M or Q6_K GGUF
//! - `EMBER_PARITY_TOKENIZER` — tokenizer.json path
//! - `EMBER_PARITY_ARCH` — optional arch (default `auto`)
//! - `EMBER_PARITY_TOKENS` — optional greedy decode length (default 12)
//!
//! Run with the release profile (`cargo test --release --test k_parity`)
//! or via `scripts/validate_k_parity.sh`. Gates are frozen in
//! docs/v03-execution-contracts.md section 9:
//!
//! The exact-f32 path is a slow oracle, not the production numerical
//! contract. Production Q8_K activation packing is checked here for behavioral
//! parity and a broad numerical sanity envelope; trusted numerical gates are
//! the llama.cpp golden-logit artifacts under `artifacts/golden-v03`.
//!
//! - per-layer representations remain finite and cosine >= 0.99
//! - logits cosine >= 0.99 on shared-prefix steps
//! - greedy tokens identical within a compressed tier. The cross-tier
//!   eager-f32 comparison keeps the numeric envelope and records token flips
//!   (the 2026-08-11 amendment scopes the original Gate B; Gate C/llama.cpp
//!   is the authoritative model-level numerical gate for production K-quant).

// Gate E reads allocation counts; the library does not install the counting
// allocator, so this test binary registers it.
#[global_allocator]
static GLOBAL_ALLOCATOR: ember::alloc_counter::CountingAllocator =
    ember::alloc_counter::CountingAllocator;

use ember::backend::CpuBackend;
use ember::experiments::ExperimentalForwardModel;
use ember::loader::{load_gguf_with_k_strategy, GgufLoader};
use ember::model::ForwardModel;
use ember::quant_k::{KExecution, KQuantDtype, KStrategy};
use ember::tensor::CpuTensor;
use ember::tokenizer::EmberTokenizer;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::sync::OnceLock;

/// Frozen prompt set (contract section 9): canonical English prompts,
/// the smoke set, and Arabic morphology prompts.
const FROZEN_PROMPTS: &[&str] = &[
    "The capital of France is",
    "The quick brown fox jumps over the",
    "Once upon a time in a small",
    "ما هي عاصمة فرنسا؟",
    "الطقس جميل اليوم في",
    "أحب اللغة العربية لأنها",
];

fn parity_env() -> Option<(String, String, String, usize)> {
    let required = std::env::var("EMBER_PARITY_REQUIRED").as_deref() == Ok("1");
    let model = match std::env::var("EMBER_PARITY_MODEL") {
        Ok(value) => value,
        Err(error) if required => panic!("EMBER_PARITY_MODEL is required: {error}"),
        Err(_) => return None,
    };
    let tokenizer = match std::env::var("EMBER_PARITY_TOKENIZER") {
        Ok(value) => value,
        Err(error) if required => panic!("EMBER_PARITY_TOKENIZER is required: {error}"),
        Err(_) => return None,
    };
    let arch = std::env::var("EMBER_PARITY_ARCH").unwrap_or_else(|_| "auto".to_string());
    let tokens = std::env::var("EMBER_PARITY_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(12);
    validate_model_file_contract(&model);
    Some((model, tokenizer, arch, tokens))
}

static MODEL_SHA256: OnceLock<String> = OnceLock::new();

fn validate_model_file_contract(model_path: &str) {
    let metadata = std::fs::metadata(model_path).expect("parity model metadata");
    assert!(metadata.is_file(), "parity model is not a regular file");
    // The dedicated ladder is 1B/1.5B. This cap fails before a larger model is
    // mapped or materialized and comfortably covers their Q4/Q6 artifacts.
    assert!(
        metadata.len() <= 2_500_000_000,
        "parity model is {} bytes; the validation ladder is capped at 2.5 GB",
        metadata.len()
    );
    if let Ok(expected) = std::env::var("EMBER_PARITY_EXPECT_SHA256") {
        let actual = MODEL_SHA256.get_or_init(|| {
            let mut file = std::fs::File::open(model_path).expect("open parity model for hashing");
            let mut digest = Sha256::new();
            let mut buffer = [0u8; 1024 * 1024];
            loop {
                let count = file.read(&mut buffer).expect("hash parity model");
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
            }
            format!("{:x}", digest.finalize())
        });
        assert!(
            actual.eq_ignore_ascii_case(&expected),
            "parity model SHA-256 {actual} != expected {expected}"
        );
    }
}

fn configured_compressed_strategy() -> KStrategy {
    let value = std::env::var("EMBER_PARITY_TIER").unwrap_or_else(|_| "auto".into());
    let strategy = KStrategy::from_cli(&value).expect("EMBER_PARITY_TIER");
    assert!(
        !matches!(strategy, KStrategy::EagerF32),
        "EMBER_PARITY_TIER must select a compressed tier"
    );
    strategy
}

/// One full run over a frozen prompt: prefill per-layer hidden states and
/// final logits, plus the greedy decode logits and token sequence.
struct Run {
    prefill_layers: Vec<Vec<f32>>,
    prefill_logits: Vec<f32>,
    decode_logits: Vec<Vec<f32>>,
    tokens: Vec<u32>,
}

fn load_llama(
    model_path: &str,
    tokenizer_path: &str,
    strategy: KStrategy,
) -> (ember::llama::Llama<CpuBackend>, EmberTokenizer, bool) {
    let loader: GgufLoader = load_gguf_with_k_strategy(model_path, strategy, false)
        .unwrap_or_else(|e| panic!("failed to load '{model_path}' with {strategy:?}: {e}"));
    assert!(
        !loader.k_decisions.is_empty(),
        "parity model has no K-family tensor inventory"
    );
    assert!(
        loader
            .k_decisions
            .values()
            .all(|decision| decision.fallback_reason.is_none()),
        "fail-closed parity load recorded a fallback: {:?}",
        loader.k_decisions
    );
    if let Ok(expected_dtype) = std::env::var("EMBER_PARITY_EXPECT_DTYPE") {
        assert!(
            loader.k_decisions.values().any(|decision| {
                ember::loader::ggml_dtype_name(decision.gguf_dtype) == Some(expected_dtype.as_str())
            }),
            "model inventory has no expected dtype {expected_dtype}: {:?}",
            loader.k_decisions
        );
    }
    if let Ok(expected_count) = std::env::var("EMBER_PARITY_EXPECT_K_TENSORS") {
        assert_eq!(
            loader.k_decisions.len(),
            expected_count
                .parse::<usize>()
                .expect("EMBER_PARITY_EXPECT_K_TENSORS"),
            "K-family tensor inventory count"
        );
    }
    for (name, decision) in &loader.k_decisions {
        if KQuantDtype::from_gguf(decision.gguf_dtype).is_none() {
            continue;
        }
        let expected = match strategy {
            KStrategy::EagerF32 => KExecution::EagerF32,
            KStrategy::Scalar => KExecution::CompressedScalar,
            KStrategy::X86 => KExecution::CompressedX86,
            KStrategy::Arm => KExecution::CompressedArm,
            KStrategy::Auto if ember::k_quant_matmul::arm_k_supported() => {
                KExecution::CompressedArm
            }
            KStrategy::Auto if ember::k_quant_matmul::x86_k_supported() => {
                KExecution::CompressedX86
            }
            KStrategy::Auto => KExecution::CompressedScalar,
        };
        assert_eq!(decision.execution, expected, "{name}: dispatch tier");
    }
    if std::env::var("EMBER_PARITY_REQUIRE_PARALLEL").as_deref() == Ok("1")
        && !matches!(strategy, KStrategy::EagerF32)
    {
        assert!(
            rayon::current_num_threads() > 1,
            "dedicated gate needs Rayon >1"
        );
        let routes_parallel = loader.tensors.values().any(|tensor| match tensor {
            ember::loader::LoadedTensor::KQuant(weight) => {
                ember::k_quant_matmul::scheduler_name(1, weight, true) == "column-parallel-rayon"
            }
            _ => false,
        });
        assert!(
            routes_parallel,
            "no real-model projection selected the parallel scheduler"
        );
    }
    // Whether the model has any compressed K-quant tensors. The v0.4 planned
    // decode path only runs for K-quant models: Q8_0 keeps the v0.3 native
    // fast path (contract D1: "Q8_0 is never rerouted through the plan").
    // Tests that assert on the *planned* path must skip pure-Q8_0/F32 models.
    let has_k_quant = loader
        .tensors
        .values()
        .any(|t| matches!(t, ember::loader::LoadedTensor::KQuant(_)));
    match loader.metadata.get("general.architecture") {
        Some(ember::loader::GgufValue::Str(arch))
            if matches!(arch.as_str(), "llama" | "qwen2" | "qwen3") => {}
        other => panic!("k-parity requires a llama-family model, got {other:?}"),
    }
    let model = ember::llama::Llama::from_loader_with_max_seq_len(loader, Some(2048))
        .expect("model construction");
    if let Ok(expected_layers) = std::env::var("EMBER_PARITY_EXPECT_LAYERS") {
        assert_eq!(
            model.n_layers(),
            expected_layers
                .parse::<usize>()
                .expect("EMBER_PARITY_EXPECT_LAYERS"),
            "model rung/layer count"
        );
    }
    if has_k_quant {
        let plan = model
            .execution_plan(
                ember::plan::ExecutionMode::Planned,
                ember::plan::HookMode::Disabled,
                &[],
                2048,
                None,
                None,
            )
            .expect("execution plan provenance");
        assert_eq!(plan.kernel_revision, ember::plan::PLAN_KERNEL_REVISION);
        assert!(plan.dispatch.kernel_per_tensor.iter().any(|entry| {
            matches!(
                entry.kernel,
                ember::plan::KernelId::KQuantScalarQ4K
                    | ember::plan::KernelId::KQuantScalarQ6K
                    | ember::plan::KernelId::KQuantAvx2Q4K
                    | ember::plan::KernelId::KQuantArmQ4K
                    | ember::plan::KernelId::KQuantArmQ6K
                    | ember::plan::KernelId::KQuantAvx2Q6K
            )
        }));
    }
    let tokenizer = EmberTokenizer::from_file(tokenizer_path).expect("tokenizer load");
    let backend = CpuBackend;
    tokenizer
        .validate_model_vocab(model.vocab_size(&backend))
        .expect("tokenizer/model vocab contract");
    (model, tokenizer, has_k_quant)
}

fn run_frozen_prompt(
    model: &ember::llama::Llama<CpuBackend>,
    tokenizer: &EmberTokenizer,
    prompt: &str,
    decode_tokens: usize,
) -> Run {
    let backend = CpuBackend;
    let ids = tokenizer
        .encode(prompt)
        .unwrap_or_else(|e| panic!("encode '{prompt}': {e}"));
    let vocab = model.vocab_size(&backend);
    assert!(
        ids.iter().all(|&id| (id as usize) < vocab),
        "prompt token out of vocabulary"
    );

    // prefill: per-layer hidden states + final logits (the probing entry)
    let (prefill_layers, logits_tensor) = model
        .forward_with_activations(&backend, &ids)
        .expect("prefill forward");
    let prefill_logits = logits_tensor.data().to_vec();

    // greedy decode through the cache
    let mut cache = model.create_cache(&backend, 2048);
    let mut tokens = Vec::new();
    let mut decode_logits = Vec::new();
    let mut position = 0usize;
    let mut current = ids.clone();
    for step in 0..decode_tokens {
        // Trait path (ForwardModel) so v0.4 execution-mode dispatch runs;
        // the inherent Llama method would shadow it.
        let logits = ForwardModel::forward_last_logits_with_cache(
            model, &backend, &current, &mut cache, position,
        )
        .expect("decode forward");
        let data = logits.data();
        let token = ember::sampler::argmax_token(data);
        decode_logits.push(data.to_vec());
        tokens.push(token as u32);
        position += current.len();
        current = vec![token as u32];
        if step + 1 >= decode_tokens {
            break;
        }
    }
    Run {
        prefill_layers,
        prefill_logits,
        decode_logits,
        tokens,
    }
}

/// Cross-tier numerical sanity for the production Q8_K path against the
/// exact-f32 oracle. This is deliberately not a golden gate: the trusted
/// reference for native Q8_K execution is llama.cpp, not Ember's exact-f32
/// oracle. Per the 2026-08-11 amendment in `docs/v03-execution-contracts.md`
/// the original Gate B greedy-token equality no longer defines the production
/// K-quant algorithm; the llama.cpp golden ladder (Gate C) is the authoritative
/// model-level numerical gate, and exact greedy-token equality is asserted only
/// within a tier. Here the compressed path must track the oracle within the
/// frozen numeric envelope on every step whose generated prefix is still
/// shared; a token flip at a near-tie margin is recorded, not required away.
fn assert_production_sanity(reference: &Run, candidate: &Run, label: &str) {
    let first_divergence = reference
        .tokens
        .iter()
        .zip(&candidate.tokens)
        .position(|(a, b)| a != b);
    if let Some(step) = first_divergence {
        let top = |row: &[f32]| {
            let mut values: Vec<_> = row.iter().copied().enumerate().collect();
            values.sort_by(|a, b| b.1.total_cmp(&a.1));
            values.truncate(8);
            values
        };
        let expected = &reference.decode_logits[step];
        let actual = &candidate.decode_logits[step];
        let diagnostic = serde_json::json!({
            "label": label,
            "first_divergence_step": step + 1,
            "common_generated_prefix": &reference.tokens[..step],
            "eager_top_logits": top(expected),
            "compressed_top_logits": top(actual),
            "eager_tokens": reference.tokens,
            "compressed_tokens": candidate.tokens,
            "eager_logit_at_compressed_choice": expected[candidate.tokens[step] as usize],
            "compressed_logit_at_eager_choice": actual[reference.tokens[step] as usize],
        });
        eprintln!("first-divergence diagnostic: {diagnostic}");
        if let Ok(path) = std::env::var("EMBER_PARITY_DIAGNOSTIC") {
            std::fs::write(path, serde_json::to_vec_pretty(&diagnostic).unwrap()).unwrap();
        }
    }
    // The first generated token comes from the prefill logits, which see the
    // same prompt in both tiers; it must agree under the envelope.
    assert_eq!(
        reference.tokens.first(),
        candidate.tokens.first(),
        "{label}: first generated token (prefill argmax) diverged"
    );
    assert_eq!(
        reference.prefill_layers.len(),
        candidate.prefill_layers.len(),
        "{label}: prefill layer count"
    );
    assert_eq!(
        reference.prefill_logits.len(),
        candidate.prefill_logits.len(),
        "{label}: prefill vocab width"
    );
    assert_eq!(
        reference.decode_logits.len(),
        candidate.decode_logits.len(),
        "{label}: decode step count"
    );

    for (li, (expected, actual)) in reference
        .prefill_layers
        .iter()
        .zip(&candidate.prefill_layers)
        .enumerate()
    {
        assert_eq!(expected.len(), actual.len(), "{label} layer {li}: width");
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&x, &y) in expected.iter().zip(actual) {
            assert!(
                x.is_finite() && y.is_finite(),
                "{label} layer {li}: non-finite value"
            );
            dot += f64::from(x) * f64::from(y);
            norm_a += f64::from(x) * f64::from(x);
            norm_b += f64::from(y) * f64::from(y);
        }
        let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
        assert!(cosine >= 0.99, "{label} layer {li}: cosine {cosine} < 0.99");
    }

    let cosine = |expected: &[f32], actual: &[f32]| {
        assert_eq!(expected.len(), actual.len(), "{label}: vector width");
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&x, &y) in expected.iter().zip(actual) {
            assert!(x.is_finite() && y.is_finite(), "{label}: non-finite logit");
            dot += f64::from(x) * f64::from(y);
            norm_a += f64::from(x) * f64::from(x);
            norm_b += f64::from(y) * f64::from(y);
        }
        dot / (norm_a.sqrt() * norm_b.sqrt())
    };
    let prefill_cosine = cosine(&reference.prefill_logits, &candidate.prefill_logits);
    assert!(
        prefill_cosine >= 0.99,
        "{label}: prefill logits cosine {prefill_cosine} < 0.99"
    );

    // Decode logits are comparable only while both tiers consumed the same
    // generated prefix; compare through the first divergence step inclusive and
    // skip later rows, which are conditioned on different prefixes.
    let comparable_steps = first_divergence
        .map_or(reference.decode_logits.len(), |step| step + 1)
        .min(reference.decode_logits.len());
    for (step, (expected, actual)) in reference
        .decode_logits
        .iter()
        .zip(&candidate.decode_logits)
        .take(comparable_steps)
        .enumerate()
    {
        let decode_cosine = cosine(expected, actual);
        assert!(
            decode_cosine >= 0.99,
            "{label}: decode step {step} logits cosine {decode_cosine} < 0.99"
        );
    }
}

fn max_abs_finite(expected: &[f32], actual: &[f32], label: &str) -> f32 {
    assert_eq!(expected.len(), actual.len(), "{label}: vector width");
    let mut max_abs = 0.0f32;
    for (&left, &right) in expected.iter().zip(actual) {
        assert!(
            left.is_finite() && right.is_finite(),
            "{label}: non-finite value"
        );
        max_abs = max_abs.max((left - right).abs());
    }
    max_abs
}

#[test]
fn arm_q8_k_is_bit_exact_with_scalar_on_frozen_prompts() {
    let Some((model_path, tokenizer_path, _, decode_tokens)) = parity_env() else {
        eprintln!("skipped: EMBER_PARITY_MODEL/EMBER_PARITY_TOKENIZER not set");
        return;
    };
    if !ember::k_quant_matmul::arm_k_supported() {
        assert_ne!(
            std::env::var("EMBER_PARITY_REQUIRE_ARM").as_deref(),
            Ok("1"),
            "required ARM tier unavailable"
        );
        return;
    }
    let (scalar_model, tokenizer, _) = load_llama(&model_path, &tokenizer_path, KStrategy::Scalar);
    let (arm_model, _, _) = load_llama(&model_path, &tokenizer_path, KStrategy::Arm);
    for &prompt in FROZEN_PROMPTS {
        let reference = run_frozen_prompt(&scalar_model, &tokenizer, prompt, decode_tokens);
        let candidate = run_frozen_prompt(&arm_model, &tokenizer, prompt, decode_tokens);
        assert_eq!(reference.tokens, candidate.tokens, "{prompt}: tokens");
        assert_eq!(
            reference.prefill_layers, candidate.prefill_layers,
            "{prompt}: layers"
        );
        assert_eq!(
            reference.prefill_logits, candidate.prefill_logits,
            "{prompt}: prefill logits"
        );
        assert_eq!(
            reference.decode_logits, candidate.decode_logits,
            "{prompt}: decode logits"
        );
    }
}

#[test]
fn production_q8_k_keeps_oracle_behavior_across_frozen_prompts() {
    let Some((model_path, tokenizer_path, arch, decode_tokens)) = parity_env() else {
        eprintln!("skipped: EMBER_PARITY_MODEL/EMBER_PARITY_TOKENIZER not set");
        return;
    };
    let _ = arch; // arch is inferred from the GGUF metadata by the loader
    let x86_supported = ember::k_quant_matmul::x86_k_supported();

    for &prompt in FROZEN_PROMPTS {
        let label = format!("{model_path} | {prompt}");

        let (eager_model, eager_tok, _) =
            load_llama(&model_path, &tokenizer_path, KStrategy::EagerF32);
        let eager = run_frozen_prompt(&eager_model, &eager_tok, prompt, decode_tokens);
        drop(eager_model);

        let (scalar_model, scalar_tok, _) =
            load_llama(&model_path, &tokenizer_path, KStrategy::Scalar);
        let scalar = run_frozen_prompt(&scalar_model, &scalar_tok, prompt, decode_tokens);
        drop(scalar_model);
        assert_production_sanity(&eager, &scalar, &format!("{label} [scalar]"));

        if x86_supported {
            let (x86_model, x86_tok, _) = load_llama(&model_path, &tokenizer_path, KStrategy::X86);
            let x86 = run_frozen_prompt(&x86_model, &x86_tok, prompt, decode_tokens);
            drop(x86_model);
            assert_production_sanity(&eager, &x86, &format!("{label} [x86]"));
        } else if std::env::var("EMBER_PARITY_REQUIRE_X86").as_deref() == Ok("1") {
            panic!("dedicated x86 gate requested but AVX2/FMA/F16C/SSSE3 is unavailable");
        } else {
            eprintln!("skipped x86 comparison for {label}: full x86 tier unavailable");
        }
    }
}

/// Collect comparative evidence without changing the release assertions.
#[test]
#[ignore = "diagnostic artifact generation requires a pinned real model and output path"]
fn diagnose_compressed_and_eager_generation() {
    let (model_path, tokenizer_path, _, decode_tokens) = parity_env().expect("model required");
    let output = std::env::var("EMBER_PARITY_DIAGNOSTIC").expect("diagnostic output path required");
    let (compressed, tokenizer, _) = load_llama(
        &model_path,
        &tokenizer_path,
        configured_compressed_strategy(),
    );
    let (eager, _, _) = load_llama(&model_path, &tokenizer_path, KStrategy::EagerF32);
    let mut records = Vec::new();
    for &prompt in FROZEN_PROMPTS {
        let a = run_frozen_prompt(&eager, &tokenizer, prompt, decode_tokens);
        let b = run_frozen_prompt(&compressed, &tokenizer, prompt, decode_tokens);
        for row in a.decode_logits.iter().chain(&b.decode_logits) {
            assert!(row.iter().all(|value| value.is_finite()));
        }
        let divergence = a.tokens.iter().zip(&b.tokens).position(|(a, b)| a != b);
        let steps: Vec<_> = a
            .decode_logits
            .iter()
            .zip(&b.decode_logits)
            .enumerate()
            .map(|(step, (eager_logits, compressed_logits))| {
                let top = |row: &[f32]| {
                    let mut ranked: Vec<_> = row.iter().copied().enumerate().collect();
                    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    ranked.truncate(8);
                    ranked
                };
                serde_json::json!({
                    "step": step + 1,
                    "common_input_prefix": a.tokens[..step] == b.tokens[..step],
                    "eager_top_logits": top(eager_logits),
                    "compressed_top_logits": top(compressed_logits),
                    "eager_logit_at_compressed_choice": eager_logits[b.tokens[step] as usize],
                    "compressed_logit_at_eager_choice": compressed_logits[a.tokens[step] as usize],
                })
            })
            .collect();
        records.push(serde_json::json!({
            "prompt": prompt,
            "input_token_ids": tokenizer.encode(prompt).unwrap(),
            "eager_tokens": a.tokens,
            "compressed_tokens": b.tokens,
            "first_divergence_step": divergence.map(|step| step + 1),
            "steps": steps,
        }));
    }
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&serde_json::json!({
            "scope": "diagnostic, not a replacement for frozen gates",
            "model": model_path,
            "tokenizer": tokenizer_path,
            "decode_tokens": decode_tokens,
            "records": records,
        }))
        .unwrap(),
    )
    .unwrap();
}

/// A no-op experiment: every hook is observational, so active-hook
/// plumbing must leave outputs bit-identical to the uninstrumented path.
struct IntermediateObserver {
    position: usize,
    rows: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, Vec<f32>>>>,
}

impl IntermediateObserver {
    fn record(
        &self,
        name: String,
        ctx: &ember::experiments::ExecutionContext<'_>,
        tensor: &ember::experiments::TensorAccess<'_>,
    ) {
        if ctx.phase != ember::experiments::ExecutionPhase::Decode
            || ctx.start_position != self.position
        {
            return;
        }
        assert_eq!(tensor.shape()[0], 1);
        assert!(tensor.values().iter().all(|v| v.is_finite()));
        assert!(
            self.rows
                .lock()
                .unwrap()
                .insert(name, tensor.values().to_vec())
                .is_none(),
            "duplicate intermediate capture"
        );
    }
}

impl ember::experiments::Experiment for IntermediateObserver {
    fn name(&self) -> &'static str {
        "q6-intermediate-diagnostic"
    }

    fn uses_activation_site(
        &self,
        _: ember::artifact::ActivationStage,
        _: Option<usize>,
        phase: ember::experiments::ExecutionPhase,
    ) -> bool {
        phase == ember::experiments::ExecutionPhase::Decode
    }

    fn after_attention(
        &mut self,
        ctx: &ember::experiments::LayerContext<'_>,
        tensor: &mut ember::experiments::TensorAccess<'_>,
    ) -> Result<(), ember::experiments::ExperimentError> {
        self.record(
            format!("attn_out-{}", ctx.layer_index),
            &ctx.execution,
            tensor,
        );
        Ok(())
    }

    fn after_mlp(
        &mut self,
        ctx: &ember::experiments::LayerContext<'_>,
        tensor: &mut ember::experiments::TensorAccess<'_>,
    ) -> Result<(), ember::experiments::ExperimentError> {
        self.record(
            format!("ffn_out-{}", ctx.layer_index),
            &ctx.execution,
            tensor,
        );
        Ok(())
    }

    fn after_layer(
        &mut self,
        ctx: &ember::experiments::LayerContext<'_>,
        tensor: &mut ember::experiments::TensorAccess<'_>,
    ) -> Result<(), ember::experiments::ExperimentError> {
        self.record(format!("l_out-{}", ctx.layer_index), &ctx.execution, tensor);
        Ok(())
    }

    fn before_logits(
        &mut self,
        ctx: &ember::experiments::ExecutionContext<'_>,
        tensor: &mut ember::experiments::TensorAccess<'_>,
    ) -> Result<(), ember::experiments::ExperimentError> {
        self.record("result_norm".into(), ctx, tensor);
        Ok(())
    }

    fn after_logits(
        &mut self,
        ctx: &ember::experiments::ExecutionContext<'_>,
        tensor: &mut ember::experiments::TensorAccess<'_>,
    ) -> Result<(), ember::experiments::ExperimentError> {
        self.record("result_output".into(), ctx, tensor);
        Ok(())
    }
}

#[test]
#[ignore = "diagnostic intermediate capture requires pinned Q6 model and new output file"]
fn diagnose_arabic_intermediates() {
    diagnose_intermediates(FROZEN_PROMPTS[5], 10);
}

#[test]
#[ignore = "diagnostic first-divergence capture requires pinned Q6 model and new output file"]
fn diagnose_english_first_divergence() {
    diagnose_intermediates(FROZEN_PROMPTS[1], 3);
}

fn diagnose_intermediates(prompt: &str, capture_step: usize) {
    assert!(capture_step >= 2);
    let (model_path, tokenizer_path, _, decode_tokens) = parity_env().expect("model required");
    assert!(decode_tokens >= capture_step);
    let output = std::env::var("EMBER_PARITY_DIAGNOSTIC").expect("output required");
    let backend = CpuBackend;
    let mut records = Vec::new();
    for (name, strategy) in [
        ("eager", KStrategy::EagerF32),
        ("compressed", configured_compressed_strategy()),
    ] {
        let (model, tokenizer, _) = load_llama(&model_path, &tokenizer_path, strategy);
        let plain = run_frozen_prompt(&model, &tokenizer, prompt, decode_tokens);
        let ids = tokenizer.encode(prompt).unwrap();
        let rows = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new()));
        let mut runner = ember::experiments::ExperimentRunner::new(IntermediateObserver {
            position: ids.len() + capture_step - 2,
            rows: rows.clone(),
        });
        let context = ember::experiments::ModelContext::new(
            ember::experiments::ModelFamily::Llama,
            None,
            "llama",
            model.n_layers(),
            model.embed_dim(),
        );
        let mut cache = model.create_cache(&backend, 2048);
        let mut current = ids.clone();
        let mut position = 0;
        let mut tokens = Vec::new();
        for step in 0..decode_tokens {
            let phase = if position == 0 {
                ember::experiments::ExecutionPhase::Prefill
            } else {
                ember::experiments::ExecutionPhase::Decode
            };
            let execution = ember::experiments::ExecutionContext::new(
                context,
                phase,
                position,
                current.len(),
                ember::experiments::TracingState::Disabled,
            );
            let logits = model
                .forward_last_logits_with_experiment(
                    &backend,
                    &current,
                    &mut cache,
                    position,
                    execution,
                    &mut runner,
                )
                .unwrap();
            assert!(
                logits
                    .data()
                    .iter()
                    .zip(&plain.decode_logits[step])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: observer changed step {}",
                step + 1
            );
            assert_eq!(logits.data().len(), plain.decode_logits[step].len());
            let token = ember::sampler::argmax_token(logits.data()) as u32;
            tokens.push(token);
            position += current.len();
            current = vec![token];
        }
        assert_eq!(tokens, plain.tokens);
        let captured = rows.lock().unwrap();
        assert_eq!(captured.len(), 3 * model.n_layers() + 2);
        records.push(serde_json::json!({
            "strategy": name, "step": capture_step, "input_token_ids": ids,
            "generated_token_ids": tokens, "observer_all_logits_bit_exact": true,
            "tensors": *captured,
        }));
    }
    // Cross-strategy comparisons are meaningful only before histories diverge.
    let eager_tokens = records[0]["generated_token_ids"].as_array().unwrap();
    let compressed_tokens = records[1]["generated_token_ids"].as_array().unwrap();
    assert_eq!(
        &eager_tokens[..capture_step - 1],
        &compressed_tokens[..capture_step - 1],
        "intermediate comparison requires identical input history"
    );
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    serde_json::to_writer_pretty(
        &mut file,
        &serde_json::json!({
            "scope": "diagnostic intermediates; not a numerical parity gate",
            "prompt": prompt, "records": records,
        }),
    )
    .unwrap();
}

#[test]
#[ignore = "identical-input projection diagnostic requires pinned Q6 model and new output file"]
fn diagnose_english_layer_zero_projections() {
    use ember::loader::LoadedTensor;
    let (model_path, _, _, _) = parity_env().expect("model required");
    let output = std::env::var("EMBER_PARITY_DIAGNOSTIC").expect("output required");
    let source = std::env::var("EMBER_PARITY_REFERENCE_TENSORS").expect("capture required");
    let captured: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source).unwrap()).unwrap();
    let records = captured["records"].as_array().unwrap();
    assert_eq!(records.len(), 2);
    for record in records {
        assert_eq!(record["step"], 3);
        assert_eq!(record["generated_token_ids"][1], 5679);
        assert_eq!(record["observer_all_logits_bit_exact"], true);
    }
    let token = 5679usize;
    let mut common_input: Option<Vec<f32>> = None;
    let mut results = Vec::new();
    for (name, strategy) in [
        ("eager", KStrategy::EagerF32),
        ("compressed", configured_compressed_strategy()),
    ] {
        let mut loader = load_gguf_with_k_strategy(&model_path, strategy, false).unwrap();
        let embedding = loader.tensors.remove("token_embd.weight").unwrap();
        let row = match embedding {
            LoadedTensor::F32(t) => {
                let width = t.shape()[0];
                t.data()[token * width..(token + 1) * width].to_vec()
            }
            LoadedTensor::KQuant(w) => {
                let mut row = vec![0.0; w.in_features()];
                w.dequantize_row(token, &mut row);
                row
            }
            LoadedTensor::Half(w) => {
                let mut row = vec![0.0; w.row_len()];
                w.dequantize_row(token, &mut row);
                row
            }
            LoadedTensor::Q8_0(w) => {
                let mut row = vec![0.0; w.in_features()];
                w.dequantize_row(token, &mut row);
                row
            }
        };
        let LoadedTensor::F32(norm) = loader.tensors.remove("blk.0.attn_norm.weight").unwrap()
        else {
            panic!("expected f32 normalization")
        };
        let ember::loader::GgufValue::F32(eps) =
            loader.metadata["llama.attention.layer_norm_rms_epsilon"]
        else {
            panic!("expected epsilon")
        };
        let mut normalized = vec![0.0; row.len()];
        ember::simd::rms_norm_into(&row, norm.data(), eps, &mut normalized);
        if let Some(common) = &common_input {
            assert_eq!(common.len(), normalized.len());
            assert!(
                common
                    .iter()
                    .zip(&normalized)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "layer-zero normalized inputs differ"
            );
        } else {
            common_input = Some(normalized.clone());
        }
        let input = CpuTensor::from_data(vec![1, normalized.len()], normalized);
        for projection in ["q", "k", "v"] {
            let weight_name = format!("blk.0.attn_{projection}.weight");
            let linear = match loader.tensors.remove(&weight_name).unwrap() {
                LoadedTensor::F32(w) => {
                    ember::model::Linear::new(ember::loader::gguf_to_row_major_f32(w), None)
                }
                LoadedTensor::KQuant(w) => ember::model::Linear::new_k(w, None),
                LoadedTensor::Q8_0(_) | LoadedTensor::Half(_) => panic!("expected Q6 projection"),
            };
            let actual = linear.forward(&CpuBackend, &input).unwrap();
            assert!(actual.data().iter().all(|v| v.is_finite()));
            results.push(
                serde_json::json!({"strategy": name, "projection": projection,
                "values": actual.data()}),
            );
        }
    }
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    serde_json::to_writer_pretty(file, &serde_json::json!({
        "scope": "actual Linear paths on bit-identical layer-zero normalized input; no cache or RoPE",
        "token": token, "normalized_input": common_input, "records": results,
    })).unwrap();
}

#[test]
#[ignore = "isolated numerical diagnostic requires reference input tensors and pinned Q6 weights"]
fn diagnose_reference_input_mlp_kernels() {
    let (model_path, _, _, _) = parity_env().expect("model required");
    let source =
        std::env::var("EMBER_PARITY_REFERENCE_TENSORS").expect("reference tensors required");
    let output = std::env::var("EMBER_PARITY_DIAGNOSTIC").expect("output required");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&source).unwrap()).unwrap();
    let rows: std::collections::BTreeMap<String, Vec<f32>> = reference["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tensor| {
            assert_eq!(tensor["step"], 10);
            assert_eq!(tensor["row"], 0);
            let values: Vec<f32> = serde_json::from_value(tensor["values"].clone()).unwrap();
            assert!(values.iter().all(|x| x.is_finite()));
            assert_eq!(values.len(), tensor["width"].as_u64().unwrap() as usize);
            (tensor["name"].as_str().unwrap().to_string(), values)
        })
        .collect();
    let mut records = Vec::new();
    for (name, strategy) in [("arm", KStrategy::Arm), ("scalar", KStrategy::Scalar)] {
        let loader = load_gguf_with_k_strategy(&model_path, strategy, false).unwrap();
        let mut projections = vec![
            ("ffn_norm-0", "blk.0.ffn_up.weight", "ffn_up-0"),
            ("ffn_norm-0", "blk.0.ffn_gate.weight", "ffn_gate-0"),
            ("ffn_swiglu-0", "blk.0.ffn_down.weight", "ffn_out-0"),
        ];
        if std::env::var_os("EMBER_PARITY_ATTENTION_DETAIL").is_some() {
            assert!(rows.contains_key("kqv_out-0"), "attention input required");
            projections.push(("kqv_out-0", "blk.0.attn_output.weight", "attn_out-0"));
        }
        if std::env::var_os("EMBER_PARITY_QKV_DETAIL").is_some() {
            projections.extend([
                ("attn_norm-0", "blk.0.attn_q.weight", "q_projection-0"),
                ("attn_norm-0", "blk.0.attn_k.weight", "k_projection-0"),
                ("attn_norm-0", "blk.0.attn_v.weight", "v_projection-0"),
            ]);
        }
        for (input_name, weight_name, output_name) in projections {
            let ember::loader::LoadedTensor::KQuant(weight) = &loader.tensors[weight_name] else {
                panic!("{weight_name}: expected compressed K weight");
            };
            assert_eq!(weight.dtype(), KQuantDtype::Q6K);
            let input = &rows[input_name];
            assert_eq!(input.len(), weight.in_features());
            let mut actual = vec![0.0; weight.out_features()];
            ember::k_quant_matmul::matmul_k_q8_into(input, 1, weight, &mut actual, true).unwrap();
            assert!(actual.iter().all(|x| x.is_finite()));
            assert_eq!(actual.len(), rows[output_name].len());
            records.push(serde_json::json!({
                "strategy": name, "operation": output_name,
                "reference_input": input_name, "weight": weight_name,
                "actual": actual, "reference": rows[output_name],
            }));
        }
        if name == "arm" {
            let mut actual = vec![0.0; rows["ffn_gate-0"].len()];
            ember::simd::silu_mul_into(&rows["ffn_gate-0"], &rows["ffn_up-0"], &mut actual);
            records.push(serde_json::json!({
                "strategy": "native", "operation": "ffn_swiglu-0",
                "reference_input": ["ffn_gate-0", "ffn_up-0"],
                "actual": actual, "reference": rows["ffn_swiglu-0"],
            }));
            let ember::loader::LoadedTensor::F32(weight) = &loader.tensors["blk.0.ffn_norm.weight"]
            else {
                panic!("normalization weight must be f32");
            };
            let ember::loader::GgufValue::F32(eps) =
                loader.metadata["llama.attention.layer_norm_rms_epsilon"]
            else {
                panic!("normalization epsilon must be f32");
            };
            let mut actual = vec![0.0; rows["ffn_inp-0"].len()];
            ember::simd::rms_norm_into(&rows["ffn_inp-0"], weight.data(), eps, &mut actual);
            records.push(serde_json::json!({
                "strategy": "native", "operation": "ffn_norm-0", "epsilon": eps,
                "reference_input": "ffn_inp-0", "actual": actual, "reference": rows["ffn_norm-0"],
            }));
        }
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    serde_json::to_writer_pretty(&mut file, &serde_json::json!({
        "scope": "isolated operations on identical reference inputs; no new acceptance thresholds",
        "reference_source": source, "records": records,
    })).unwrap();
}

/// A no-op observer used by the frozen hook-neutrality gate.
struct NoopExperiment;

impl ember::experiments::Experiment for NoopExperiment {
    fn name(&self) -> &'static str {
        "noop"
    }
}

/// Inactive-hook equivalence on the compressed path (contract section 8):
/// firing the full ActiveHooks machinery with a no-op experiment must not
/// alter logits or tokens.
#[test]
fn inactive_hooks_do_not_alter_compressed_outputs() {
    let Some((model_path, tokenizer_path, _, decode_tokens)) = parity_env() else {
        eprintln!("skipped: EMBER_PARITY_MODEL/EMBER_PARITY_TOKENIZER not set");
        return;
    };
    let (model, tokenizer, _) = load_llama(
        &model_path,
        &tokenizer_path,
        configured_compressed_strategy(),
    );
    let backend = CpuBackend;
    let prompt = FROZEN_PROMPTS[0];
    let ids = tokenizer.encode(prompt).expect("encode");

    // plain run (DisabledHooks)
    let plain = run_frozen_prompt(&model, &tokenizer, prompt, decode_tokens);

    // hooked run (ActiveHooks -> noop experiment)
    let model_context = ember::experiments::ModelContext::new(
        ember::experiments::ModelFamily::Llama,
        None,
        "llama",
        model.n_layers(),
        model.embed_dim(),
    );
    let mut cache = model.create_cache(&backend, 2048);
    let mut tokens = Vec::new();
    let mut hooked_logits = Vec::new();
    let mut position = 0usize;
    let mut current = ids.clone();
    for _ in 0..decode_tokens {
        let token_count = current.len();
        let execution = ember::experiments::ExecutionContext::new(
            model_context,
            if position == 0 {
                ember::experiments::ExecutionPhase::Prefill
            } else {
                ember::experiments::ExecutionPhase::Decode
            },
            position,
            token_count,
            ember::experiments::TracingState::Disabled,
        );
        let mut runner = ember::experiments::ExperimentRunner::new(NoopExperiment);
        let logits = model
            .forward_last_logits_with_experiment(
                &backend,
                &current,
                &mut cache,
                position,
                execution,
                &mut runner,
            )
            .expect("hooked forward");
        let data = logits.data();
        hooked_logits.push(data.to_vec());
        tokens.push(ember::sampler::argmax_token(data) as u32);
        position += current.len();
        current = vec![*tokens.last().expect("token pushed")];
    }

    assert_eq!(
        plain.tokens, tokens,
        "hooked (noop) run diverged tokens from the plain run"
    );
    assert_eq!(plain.decode_logits.len(), hooked_logits.len());
    for (step, (expected, actual)) in plain.decode_logits.iter().zip(&hooked_logits).enumerate() {
        assert_eq!(
            expected, actual,
            "hooked (noop) run diverged logits at decode step {step}"
        );
    }
}

/// Gate B for v0.4 planned execution: the plan-driven interpreter must
/// reproduce the reference greedy tokens and stay within the frozen logit
/// envelope on the real model (docs/v04-execution-contract.md section 13).
#[test]
fn v04_planned_matches_reference_real_model() {
    let Some((model_path, tokenizer_path, _, decode_tokens)) = parity_env() else {
        eprintln!("skipped: EMBER_PARITY_MODEL/EMBER_PARITY_TOKENIZER not set");
        return;
    };
    let (model, tokenizer, has_k_quant) = load_llama(
        &model_path,
        &tokenizer_path,
        configured_compressed_strategy(),
    );
    if !has_k_quant {
        // Q8_0/F32 models keep the v0.3 native fast path (contract D1: Q8_0
        // is never rerouted through the plan), so the plain run uses the fast
        // path while the hooked run uses the generic hooked path — different
        // dispatch, legitimately different float accumulation (tokens still
        // match). This test asserts bit-exact logits and is only meaningful
        // when both runs execute the *planned* interpreter, i.e. K-quant.
        eprintln!(
            "skipped: {model_path} has no K-quant tensors (planned path not exercised; \
             Q8_0 keeps the v0.3 fast path per contract D1)"
        );
        return;
    }
    use ember::plan::ExecutionMode;

    for &prompt in FROZEN_PROMPTS {
        model.set_execution_mode(ExecutionMode::Reference);
        let reference = run_frozen_prompt(&model, &tokenizer, prompt, decode_tokens);
        model.set_execution_mode(ExecutionMode::Planned);
        let planned = run_frozen_prompt(&model, &tokenizer, prompt, decode_tokens);
        model.set_execution_mode(ExecutionMode::PlannedFused);
        let fused = run_frozen_prompt(&model, &tokenizer, prompt, decode_tokens);
        assert_eq!(
            reference.tokens, planned.tokens,
            "{model_path} | {prompt}: greedy tokens diverged under planned execution"
        );
        assert_eq!(
            reference.tokens, fused.tokens,
            "{model_path} | {prompt}: greedy tokens diverged under fused planned execution"
        );
        assert_eq!(reference.decode_logits.len(), planned.decode_logits.len());
        for (step, (expected, actual)) in reference
            .decode_logits
            .iter()
            .zip(&planned.decode_logits)
            .enumerate()
        {
            let label = format!("{model_path} | {prompt}: planned decode step {step}");
            let max_abs = max_abs_finite(expected, actual, &label);
            assert!(max_abs <= 1e-3, "{label}: logits max_abs {max_abs} > 1e-3");
        }
        assert_eq!(reference.decode_logits.len(), fused.decode_logits.len());
        for (step, (expected, actual)) in reference
            .decode_logits
            .iter()
            .zip(&fused.decode_logits)
            .enumerate()
        {
            let label = format!("{model_path} | {prompt}: fused decode step {step}");
            let max_abs = max_abs_finite(expected, actual, &label);
            assert!(max_abs <= 1e-3, "{label}: logits max_abs {max_abs} > 1e-3");
        }
    }
}

/// Gate C on the real model for v0.4 planned execution (contract section
/// 12): the planned path with the hook system initialized but a no-op
/// experiment must stay bit-identical to the plain planned path.
#[test]
fn v04_planned_inactive_hooks_real_model() {
    let Some((model_path, tokenizer_path, _, decode_tokens)) = parity_env() else {
        eprintln!("skipped: EMBER_PARITY_MODEL/EMBER_PARITY_TOKENIZER not set");
        return;
    };
    let (model, tokenizer, has_k_quant) = load_llama(
        &model_path,
        &tokenizer_path,
        configured_compressed_strategy(),
    );
    if !has_k_quant {
        // Q8_0/F32 models keep the v0.3 native fast path (contract D1: Q8_0
        // is never rerouted through the plan), so the plain run uses the fast
        // path while the hooked run uses the generic hooked path — different
        // dispatch, legitimately different float accumulation (tokens still
        // match). This test asserts bit-exact logits and is only meaningful
        // when both runs execute the *planned* interpreter, i.e. K-quant.
        eprintln!(
            "skipped: {model_path} has no K-quant tensors (planned path not exercised; \
             Q8_0 keeps the v0.3 fast path per contract D1)"
        );
        return;
    }
    use ember::plan::ExecutionMode;
    let backend = CpuBackend;
    let prompt = FROZEN_PROMPTS[0];
    let ids = tokenizer.encode(prompt).expect("encode");
    model.set_execution_mode(ExecutionMode::Planned);

    // plain planned run
    let plain = run_frozen_prompt(&model, &tokenizer, prompt, decode_tokens);

    // planned run through the experiment machinery with a noop experiment
    let model_context = ember::experiments::ModelContext::new(
        ember::experiments::ModelFamily::Llama,
        None,
        "llama",
        model.n_layers(),
        model.embed_dim(),
    );
    let mut cache = model.create_cache(&backend, 2048);
    let mut tokens = Vec::new();
    let mut hooked_logits = Vec::new();
    let mut position = 0usize;
    let mut current = ids.clone();
    for _ in 0..decode_tokens {
        let token_count = current.len();
        let execution = ember::experiments::ExecutionContext::new(
            model_context,
            if position == 0 {
                ember::experiments::ExecutionPhase::Prefill
            } else {
                ember::experiments::ExecutionPhase::Decode
            },
            position,
            token_count,
            ember::experiments::TracingState::Disabled,
        );
        let mut runner = ember::experiments::ExperimentRunner::new(NoopExperiment);
        let logits = model
            .forward_last_logits_with_experiment(
                &backend,
                &current,
                &mut cache,
                position,
                execution,
                &mut runner,
            )
            .expect("hooked planned forward");
        let data = logits.data();
        hooked_logits.push(data.to_vec());
        tokens.push(ember::sampler::argmax_token(data) as u32);
        position += current.len();
        current = vec![*tokens.last().expect("token pushed")];
    }

    assert_eq!(
        plain.tokens, tokens,
        "planned hooked (noop) run diverged tokens from the plain planned run"
    );
    assert_eq!(plain.decode_logits.len(), hooked_logits.len());
    for (step, (expected, actual)) in plain.decode_logits.iter().zip(&hooked_logits).enumerate() {
        assert_eq!(
            expected, actual,
            "planned hooked (noop) run diverged logits at decode step {step}"
        );
    }
}

/// Gate E on the real model (contract section 13): after warmup, the
/// planned decode loop with hooks disabled performs zero heap allocations
/// per token other than the logits tensor materialization (3 documented).
#[test]
fn v04_planned_allocations_stay_within_existing_scheduler_bound() {
    let Some((model_path, tokenizer_path, _, _)) = parity_env() else {
        eprintln!("skipped: EMBER_PARITY_MODEL/EMBER_PARITY_TOKENIZER not set");
        return;
    };
    let (model, tokenizer, _) = load_llama(
        &model_path,
        &tokenizer_path,
        configured_compressed_strategy(),
    );
    use ember::plan::ExecutionMode;
    let backend = CpuBackend;
    let ids = tokenizer.encode(FROZEN_PROMPTS[0]).expect("encode");
    model.set_execution_mode(ExecutionMode::Planned);
    let mut cache = model.create_cache(&backend, 2048);
    ForwardModel::forward_last_logits_with_cache(&model, &backend, &ids, &mut cache, 0)
        .expect("prefill");
    // warmup decode: plan build + decode session + rayon pool
    ForwardModel::forward_last_logits_with_cache(
        &model,
        &backend,
        &[ids[0]],
        &mut cache,
        ids.len(),
    )
    .expect("warmup decode");
    // measure two consecutive decodes: a one-shot lazy-init allocation on
    // the first (e.g. rayon pool internals) is distinguishable from a
    // per-token allocation
    assert!(ember::alloc_counter::counting_active());
    let mut counts = Vec::new();
    for step in 0..2 {
        let (_, allocations) = ember::alloc_counter::count_allocations(|| {
            ForwardModel::forward_last_logits_with_cache(
                &model,
                &backend,
                &[ids[1]],
                &mut cache,
                ids.len() + 1 + step,
            )
            .expect("measured decode");
        });
        counts.push(allocations);
    }
    eprintln!("gate-e allocation counts: {counts:?}");
    // Preserve the existing bound: 3 logits allocations plus up to 2
    // scheduler allocations. External-thread Rayon injection can allocate
    // Crossbeam queue blocks even without competing workloads (traced on M1).
    // This short sample does not prove zero allocation or bound every token
    // of a longer workload; the release benchmark checks that separately.
    for (step, allocations) in counts.into_iter().enumerate() {
        assert!(
            allocations <= 5,
            "planned decode step {step} allocated {allocations} times; existing bound is 5 (3 logits + up to 2 scheduler allocations), not zero"
        );
    }
}

/// The into-buffer decode route must produce bit-identical logits while
/// avoiding the per-token logits allocation entirely. The existing allowance
/// for scheduler allocations is a bounded-allocation check, not proof of the
/// release contract's zero steady-state allocation requirement.
#[test]
fn v04_planned_into_route_is_bit_identical_with_bounded_scheduler_allocations() {
    let Some((model_path, tokenizer_path, _, _)) = parity_env() else {
        eprintln!("skipped: EMBER_PARITY_MODEL/EMBER_PARITY_TOKENIZER not set");
        return;
    };
    let (model, tokenizer, _) = load_llama(
        &model_path,
        &tokenizer_path,
        configured_compressed_strategy(),
    );
    use ember::plan::ExecutionMode;
    let backend = CpuBackend;
    let ids = tokenizer.encode(FROZEN_PROMPTS[0]).expect("encode");
    model.set_execution_mode(ExecutionMode::Planned);
    let mut cache_ref = model.create_cache(&backend, 2048);
    let mut cache_into = model.create_cache(&backend, 2048);
    let prefill =
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &ids, &mut cache_ref, 0)
            .expect("prefill");
    ForwardModel::forward_last_logits_with_cache(&model, &backend, &ids, &mut cache_into, 0)
        .expect("prefill");
    let vocab = prefill.data().len();

    // Warm the plan session through the allocating route, then compare the
    // two routes on identical sequences (independent caches, same prefix).
    let token = ids[0];
    let reference = ForwardModel::forward_last_logits_with_cache(
        &model,
        &backend,
        &[token],
        &mut cache_ref,
        ids.len(),
    )
    .expect("reference decode");
    let mut buffer = CpuTensor::zeroes(&[1, vocab]);
    assert!(ember::alloc_counter::counting_active());
    let (result, allocations) = ember::alloc_counter::count_allocations(|| {
        ForwardModel::forward_last_logits_with_cache_reusing(
            &model,
            &backend,
            &[token],
            &mut cache_into,
            ids.len(),
            &mut buffer,
        )
    });
    result.expect("reusing decode");
    assert_eq!(
        reference.data(),
        buffer.data(),
        "reused-buffer logits must be bit-identical to the materialized route"
    );
    eprintln!("gate-e into-route allocation count: {allocations}");
    assert!(
        allocations <= 2,
        "into-buffer decode allocated {allocations} times; rayon job structures (<=2) are the only documented allocation"
    );
}

/// The production session boundary must remove external injector traffic over
/// a long stream, while retaining the parallel K-quant scheduler.
#[test]
#[ignore = "requires explicit real K-quant model and tokenizer"]
fn cpu_session_has_zero_steady_state_allocations() {
    let (model_path, tokenizer_path, _, _) = parity_env().expect("set parity model/tokenizer");
    assert!(rayon::current_num_threads() > 1, "requires a parallel pool");
    let (model, tokenizer, has_k_quant) = load_llama(
        &model_path,
        &tokenizer_path,
        configured_compressed_strategy(),
    );
    assert!(has_k_quant);
    let backend = CpuBackend;
    model.set_execution_mode(ember::plan::ExecutionMode::Planned);
    let ids = tokenizer.encode(FROZEN_PROMPTS[0]).unwrap();
    // Establish exact output expectations on an external calling thread.
    let mut external_cache = model.create_cache(&backend, 2048);
    let mut external_logits = ForwardModel::forward_last_logits_with_cache(
        &model,
        &backend,
        &ids,
        &mut external_cache,
        0,
    )
    .unwrap();
    let mut expected = Vec::with_capacity(72);
    for offset in 0..72 {
        ForwardModel::forward_last_logits_with_cache_reusing(
            &model,
            &backend,
            &[ids[0]],
            &mut external_cache,
            ids.len() + offset,
            &mut external_logits,
        )
        .unwrap();
        expected.push(external_logits.data().to_vec());
    }
    drop(external_cache);
    ember::model::with_cpu_session(move || {
        let mut cache = model.create_cache(&backend, 2048);
        let mut logits =
            ForwardModel::forward_last_logits_with_cache(&model, &backend, &ids, &mut cache, 0)
                .unwrap();
        for offset in 0..8 {
            ForwardModel::forward_last_logits_with_cache_reusing(
                &model,
                &backend,
                &[ids[0]],
                &mut cache,
                ids.len() + offset,
                &mut logits,
            )
            .unwrap();
        }
        assert!(ember::alloc_counter::counting_active());
        let global_tracking = ember::alloc_counter::track_global();
        let before = ember::alloc_counter::total_allocations();
        let (_, caller) = ember::alloc_counter::count_allocations(|| {
            for offset in 8..72 {
                ForwardModel::forward_last_logits_with_cache_reusing(
                    &model,
                    &backend,
                    &[ids[0]],
                    &mut cache,
                    ids.len() + offset,
                    &mut logits,
                )
                .unwrap();
                assert_eq!(logits.data().len(), expected[offset].len());
                for (actual, expected) in logits.data().iter().zip(&expected[offset]) {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "session changed a logit"
                    );
                }
            }
        });
        let global = ember::alloc_counter::total_allocations() - before;
        drop(global_tracking);
        assert_eq!(
            caller, 0,
            "caller allocations across 64 steady-state tokens"
        );
        assert_eq!(
            global, 0,
            "process allocations across 64 steady-state tokens; run this test in isolation"
        );
    });
}
