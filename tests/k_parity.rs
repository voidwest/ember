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
) -> (ember::llama::Llama<CpuBackend>, EmberTokenizer) {
    let loader: GgufLoader = load_gguf_with_k_strategy(model_path, strategy, false)
        .unwrap_or_else(|e| panic!("failed to load '{model_path}' with {strategy:?}: {e}"));
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
    (model, tokenizer)
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
        let logits = model
            .forward_last_logits_with_cache(&backend, &current, &mut cache, position)
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

/// Gate B (contract section 9), frozen numbers.
fn assert_gate_b(reference: &Run, candidate: &Run, label: &str) {
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
            max_abs <= 1e-2,
            "{label}: decode step {step} logits max_abs {max_abs} > 1e-2"
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

        let (eager_model, eager_tok) =
            load_llama(&model_path, &tokenizer_path, KStrategy::EagerF32);
        let eager = run_frozen_prompt(&eager_model, &eager_tok, prompt, decode_tokens);
        drop(eager_model);

        let (auto_model, auto_tok) = load_llama(&model_path, &tokenizer_path, KStrategy::Auto);
        let auto = run_frozen_prompt(&auto_model, &auto_tok, prompt, decode_tokens);
        drop(auto_model);
        assert_gate_b(&eager, &auto, &format!("{label} [auto]"));

        if x86_supported {
            let (x86_model, x86_tok) = load_llama(&model_path, &tokenizer_path, KStrategy::X86);
            let x86 = run_frozen_prompt(&x86_model, &x86_tok, prompt, decode_tokens);
            drop(x86_model);
            assert_gate_b(&eager, &x86, &format!("{label} [x86]"));
        } else {
            eprintln!("skipped x86 comparison for {label}: AVX2 unavailable");
        }
    }
}
