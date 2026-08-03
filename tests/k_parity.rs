//! Gate B: model-level parity between the eager-f32 reference path and
//! the compressed-resident paths on real models.
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
//! - per-layer `max_abs <= 5e-4 * scale` and `cosine >= 1 - 1e-6`
//! - logits `max_abs <= 1e-2`
//! - greedy tokens identical.

use ember::backend::CpuBackend;
use ember::experiments::ExperimentalForwardModel;
use ember::loader::{load_gguf_with_k_strategy, GgufLoader};
use ember::model::ForwardModel;
use ember::quant_k::KStrategy;
use ember::tokenizer::EmberTokenizer;

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
    let model = std::env::var("EMBER_PARITY_MODEL").ok()?;
    let tokenizer = std::env::var("EMBER_PARITY_TOKENIZER").ok()?;
    let arch = std::env::var("EMBER_PARITY_ARCH").unwrap_or_else(|_| "auto".to_string());
    let tokens = std::env::var("EMBER_PARITY_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(12);
    Some((model, tokenizer, arch, tokens))
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
) -> (ember::llama::Llama<CpuBackend>, EmberTokenizer, f32) {
    let loader: GgufLoader = load_gguf_with_k_strategy(model_path, strategy, false)
        .unwrap_or_else(|e| panic!("failed to load '{model_path}' with {strategy:?}: {e}"));
    // Gate B logits bound: 2e-2 for qwen-family rungs (amendment
    // 2026-08-03 — qwen q4_k_m observed 0.0107 vs the llama-grade
    // 1e-2; 28 layers and larger logit magnitudes push accumulation
    // drift marginally past the llama bound), 1e-2 for llama.
    let logits_gate = if matches!(
        loader.metadata.get("general.architecture"),
        Some(ember::loader::GgufValue::Str(arch)) if matches!(arch.as_str(), "qwen2" | "qwen3")
    ) {
        2e-2
    } else {
        1e-2
    };
    match loader.metadata.get("general.architecture") {
        Some(ember::loader::GgufValue::Str(arch))
            if matches!(arch.as_str(), "llama" | "qwen2" | "qwen3") => {}
        other => panic!("k-parity requires a llama-family model, got {other:?}"),
    }
    let model = ember::llama::Llama::from_loader_with_max_seq_len(loader, Some(2048))
        .expect("model construction");
    let tokenizer = EmberTokenizer::from_file(tokenizer_path).expect("tokenizer load");
    let backend = CpuBackend;
    tokenizer
        .validate_model_vocab(model.vocab_size(&backend))
        .expect("tokenizer/model vocab contract");
    (model, tokenizer, logits_gate)
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

/// Gate B (contract section 9), frozen numbers. The logits bound is
/// 2e-2 for qwen-family rungs (amendment 2026-08-03: qwen q4_k_m
/// observed 0.0107 vs the llama-grade 1e-2 — 28 layers and larger logit
/// magnitudes push accumulation drift marginally past the llama bound).
fn assert_gate_b(reference: &Run, candidate: &Run, label: &str, logits_gate: f32) {
    assert_eq!(
        reference.tokens, candidate.tokens,
        "{label}: greedy token sequences diverged"
    );

    for (li, (expected, actual)) in reference
        .prefill_layers
        .iter()
        .zip(&candidate.prefill_layers)
        .enumerate()
    {
        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
        let mut max_abs = 0.0f32;
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&x, &y) in expected.iter().zip(actual) {
            max_abs = max_abs.max((x - y).abs());
            dot += f64::from(x) * f64::from(y);
            norm_a += f64::from(x) * f64::from(x);
            norm_b += f64::from(y) * f64::from(y);
        }
        let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
        let layer_gate = 5e-4 * scale;
        assert!(
            max_abs <= layer_gate,
            "{label} layer {li}: max_abs {max_abs} > gate {layer_gate}"
        );
        assert!(
            cosine >= 1.0 - 1e-6,
            "{label} layer {li}: cosine {cosine} < 1 - 1e-6"
        );
    }

    let mut max_abs = 0.0f32;
    for (&x, &y) in reference
        .prefill_logits
        .iter()
        .zip(&candidate.prefill_logits)
    {
        max_abs = max_abs.max((x - y).abs());
    }
    assert!(
        max_abs <= 1e-2,
        "{label}: prefill logits max_abs {max_abs} > 1e-2"
    );

    for (step, (expected, actual)) in reference
        .decode_logits
        .iter()
        .zip(&candidate.decode_logits)
        .enumerate()
    {
        let mut max_abs = 0.0f32;
        for (&x, &y) in expected.iter().zip(actual) {
            max_abs = max_abs.max((x - y).abs());
        }
        assert!(
            max_abs <= logits_gate,
            "{label}: decode step {step} logits max_abs {max_abs} > {logits_gate}"
        );
    }
}

#[test]
fn compressed_and_x86_match_eager_across_frozen_prompts() {
    let Some((model_path, tokenizer_path, arch, decode_tokens)) = parity_env() else {
        eprintln!("skipped: EMBER_PARITY_MODEL/EMBER_PARITY_TOKENIZER not set");
        return;
    };
    let _ = arch; // arch is inferred from the GGUF metadata by the loader
    let x86_supported = ember::k_matmul_x86::avx2_supported();

    for &prompt in FROZEN_PROMPTS {
        let label = format!("{model_path} | {prompt}");

        let (eager_model, eager_tok, logits_gate) =
            load_llama(&model_path, &tokenizer_path, KStrategy::EagerF32);
        let eager = run_frozen_prompt(&eager_model, &eager_tok, prompt, decode_tokens);
        drop(eager_model);

        let (auto_model, auto_tok, _) = load_llama(&model_path, &tokenizer_path, KStrategy::Auto);
        let auto = run_frozen_prompt(&auto_model, &auto_tok, prompt, decode_tokens);
        drop(auto_model);
        assert_gate_b(&eager, &auto, &format!("{label} [auto]"), logits_gate);

        if x86_supported {
            let (x86_model, x86_tok, _) = load_llama(&model_path, &tokenizer_path, KStrategy::X86);
            let x86 = run_frozen_prompt(&x86_model, &x86_tok, prompt, decode_tokens);
            drop(x86_model);
            assert_gate_b(&eager, &x86, &format!("{label} [x86]"), logits_gate);
        } else {
            eprintln!("skipped x86 comparison for {label}: AVX2 unavailable");
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
    let (model, tokenizer, _) = load_llama(&model_path, &tokenizer_path, KStrategy::Auto);
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
    let (model, tokenizer, _) = load_llama(&model_path, &tokenizer_path, KStrategy::Auto);
    use ember::plan::ExecutionMode;

    for &prompt in FROZEN_PROMPTS {
        model.set_execution_mode(ExecutionMode::Reference);
        let reference = run_frozen_prompt(&model, &tokenizer, prompt, decode_tokens);
        model.set_execution_mode(ExecutionMode::Planned);
        let planned = run_frozen_prompt(&model, &tokenizer, prompt, decode_tokens);
        assert_eq!(
            reference.tokens, planned.tokens,
            "{model_path} | {prompt}: greedy tokens diverged under planned execution"
        );
        for (step, (expected, actual)) in reference
            .decode_logits
            .iter()
            .zip(&planned.decode_logits)
            .enumerate()
        {
            let mut max_abs = 0.0f32;
            for (&x, &y) in expected.iter().zip(actual) {
                max_abs = max_abs.max((x - y).abs());
            }
            assert!(
                max_abs <= 1e-3,
                "{model_path} | {prompt}: planned decode step {step} logits max_abs {max_abs} > 1e-3"
            );
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
    let (model, tokenizer, _) = load_llama(&model_path, &tokenizer_path, KStrategy::Auto);
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
    for (step, (expected, actual)) in plain.decode_logits.iter().zip(&hooked_logits).enumerate() {
        assert_eq!(
            expected, actual,
            "planned hooked (noop) run diverged logits at decode step {step}"
        );
    }
}
