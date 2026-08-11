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
//! - logits cosine >= 0.99
//! - greedy tokens identical.

use ember::backend::CpuBackend;
use ember::experiments::ExperimentalForwardModel;
use ember::loader::{load_gguf_with_k_strategy, GgufLoader};
use ember::model::ForwardModel;
use ember::quant_k::{KExecution, KQuantDtype, KStrategy};
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

/// Internal production-vs-oracle sanity check. This is deliberately not
/// called a golden gate: the trusted reference for native Q8_K execution is
/// llama.cpp, not Ember's exact-f32 oracle.
fn assert_production_sanity(reference: &Run, candidate: &Run, label: &str) {
    assert_eq!(
        reference.tokens, candidate.tokens,
        "{label}: greedy token sequences diverged"
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

    for (step, (expected, actual)) in reference
        .decode_logits
        .iter()
        .zip(&candidate.decode_logits)
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

/// A no-op experiment: every hook is observational, so active-hook
/// plumbing must leave outputs bit-identical to the uninstrumented path.
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
fn v04_planned_zero_steady_state_allocation_real_model() {
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
    let allocations = counts[0];
    // Accounted steady-state allocations: 3 for the logits CpuTensor
    // (shape + strides + data) plus 1 for rayon's per-iterator job
    // structure of the column-parallel matvec when the shared pool is busy
    // (measured 0 on a quiet pool, 0 with serial matvecs). Anything beyond
    // this documented constant is a leak.
    assert!(
        allocations <= 4,
        "planned decode allocated {allocations} times per token on the real model; expected at most 4 (3 logits + 1 rayon job under pool contention)"
    );
}
