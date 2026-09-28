//! One-shot latent handoff injection (pilot, v1).
//!
//! Loads a pre-assembled `[seq_len, embed_dim]` f32 embedding matrix
//! (`[K virtual rows; T prompt rows]`, built out-of-band by
//! `/tmp/opencode/handoff_pilot/stage/assemble_live.py`), prefills the
//! resident model through the existing precomputed-embedding path
//! (`forward_last_logits_embeddings_with_cache`, shared with the audio/vision
//! conditioning paths), then runs the standard greedy single-token decode
//! loop. Writes a run directory with an evidence envelope recording every
//! input hash, so a verifier can recompute provenance without trusting us.
//!
//! Scope: Llama/Qwen-family models exposing the embeddings-with-cache seam.
//! Deterministic: greedy argmax only (temperature is fixed at 0).

use anyhow::Context;
use clap::Args;
use ember::backend::{Backend, CpuBackend};
use ember::extraction::{git_commit, sha256_bytes, sha256_file_result};
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::npy::read_npy_2d;
use ember::quant_k::KStrategy;
use ember::tensor::CpuTensor;
use ember::tokenizer::EmberTokenizer;

#[derive(Args)]
pub(crate) struct HandoffInjectCommand {
    /// Generalist GGUF model path.
    #[arg(long)]
    model: String,
    /// Tokenizer JSON path (used for the prompt record + decode).
    #[arg(long)]
    tokenizer: String,
    /// Architecture hint; auto reads general.architecture from the GGUF.
    #[arg(long, default_value = "auto", value_parser = ["auto", "llama", "qwen3"])]
    arch: String,
    /// Text prompt (the T prompt rows must correspond to this prompt).
    #[arg(long)]
    prompt: String,
    /// Assembled [K+T, embed_dim] f32 npy: K virtual rows then T prompt rows.
    #[arg(long)]
    prefix_embeddings: String,
    /// Number of leading virtual rows K in the npy.
    #[arg(long)]
    prefix_rows: usize,
    /// Map sidecar JSON from assemble_live.py (recorded by hash + inline).
    #[arg(long)]
    map_manifest: String,
    /// Max new tokens to generate (greedy).
    #[arg(long, default_value_t = 40)]
    max_tokens: usize,
    /// Seed recorded for provenance (greedy decode is deterministic; the seed
    /// pins the documented RNG contract for future sampling modes).
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Output run directory (created; fails if it already exists).
    #[arg(long)]
    out: String,
    /// Allow overwriting an existing output directory.
    #[arg(long)]
    overwrite: bool,
}

pub(crate) fn run_handoff_inject_command(
    command: &HandoffInjectCommand,
    k_strategy: KStrategy,
    allow_fallback: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        command.max_tokens > 0,
        "handoff-inject needs max_tokens > 0"
    );
    anyhow::ensure!(
        command.prefix_rows > 0,
        "handoff-inject needs prefix_rows >= 1 (K virtual rows)"
    );
    let out_dir = std::path::Path::new(&command.out);
    if out_dir.exists() {
        anyhow::ensure!(
            command.overwrite,
            "output dir '{}' exists; pass --overwrite",
            command.out
        );
    } else {
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("failed to create out dir '{}'", command.out))?;
    }

    // -- assembled embeddings -------------------------------------------
    let (shape, values) = read_npy_2d(&command.prefix_embeddings)?;
    anyhow::ensure!(shape.len() == 2, "prefix npy must be 2D");
    let (seq_len, embed_dim) = (shape[0], shape[1]);
    anyhow::ensure!(
        seq_len > command.prefix_rows,
        "npy has {seq_len} rows but prefix_rows is {}; need K+T with T >= 1",
        command.prefix_rows
    );
    let prompt_rows = seq_len - command.prefix_rows;
    let prefix_sha = sha256_file_result(&command.prefix_embeddings)?;
    let map_bytes = std::fs::read(&command.map_manifest)
        .with_context(|| format!("failed to read map manifest '{}'", command.map_manifest))?;
    let map_sha = sha256_bytes(&map_bytes);
    let map_value: serde_json::Value =
        serde_json::from_slice(&map_bytes).context("map manifest is not valid JSON")?;

    // -- model + tokenizer ----------------------------------------------
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, allow_fallback)?;
    let architecture = ember::loader::resolve_generation_architecture(&command.arch, &loader)?;
    anyhow::ensure!(
        architecture == "llama" || architecture == "qwen3",
        "handoff-inject supports llama/qwen3 models (got '{architecture}')"
    );
    let model = ember::llama::Llama::from_loader_with_max_seq_len(loader, None)?;
    anyhow::ensure!(
        model.config.embed_dim == embed_dim,
        "npy embed_dim {embed_dim} != model embed_dim {}",
        model.config.embed_dim
    );
    let tokenizer = EmberTokenizer::from_file(&command.tokenizer)?;
    tokenizer.validate_model_vocab(model.config.vocab_size)?;
    let prompt_ids = tokenizer.encode(&command.prompt)?;
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt produced no token IDs");
    let capacity_needed = seq_len
        .checked_add(command.max_tokens)
        .context("context length overflow")?;
    anyhow::ensure!(
        capacity_needed <= model.config.max_seq_len,
        "need {capacity_needed} positions but model context is {}",
        model.config.max_seq_len
    );
    let model_sha = sha256_file_result(&command.model)?;
    let tokenizer_sha = sha256_file_result(&command.tokenizer)?;

    // -- prefill over assembled rows, then greedy decode ------------------
    let backend = CpuBackend;
    let embeddings = CpuTensor::from_data(vec![seq_len, embed_dim], values);
    let mut cache = model.create_request_cache(&backend, seq_len, command.max_tokens);
    let t0 = std::time::Instant::now();
    let mut logits =
        model.forward_last_logits_embeddings_with_cache(&backend, &embeddings, &mut cache, 0)?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
    let eos_ids = tokenizer.eos_token_ids();
    let mut generated: Vec<u32> = Vec::new();
    let t1 = std::time::Instant::now();
    for step in 0..command.max_tokens {
        let data = backend.data(&logits);
        let best = ember::sampler::argmax_token(data);
        let best = u32::try_from(best).map_err(|_| anyhow::anyhow!("vocab exceeds u32"))?;
        generated.push(best);
        if eos_ids.contains(&best) {
            break;
        }
        if step + 1 < command.max_tokens {
            logits = ForwardModel::forward_last_logits_with_cache(
                &model,
                &backend,
                &[best],
                &mut cache,
                seq_len + step,
            )?;
        }
    }
    let decode_ms = t1.elapsed().as_secs_f64() * 1e3;
    let text = tokenizer.decode(&generated)?;

    // -- evidence envelope -------------------------------------------------
    let envelope = serde_json::json!({
        "schema": "ember.handoff-inject.v1",
        "prompt": command.prompt,
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": prompt_rows,
        "prefix_rows": command.prefix_rows,
        "assembled_rows": seq_len,
        "assembled_npy_sha256": prefix_sha,
        "assembled_embed_dim": embed_dim,
        "map_manifest_sha256": map_sha,
        "map_manifest": map_value,
        "model_sha256": model_sha,
        "tokenizer_sha256": tokenizer_sha,
        "architecture": architecture,
        "sampler": "greedy-argmax",
        "seed": command.seed,
        "max_tokens": command.max_tokens,
        "generated_token_ids": generated,
        "generated_text": text,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "ember_git_commit": git_commit().unwrap_or_else(|| "unknown".to_string()),
        "trust_note": "assembled tail-row == prompt-embedding correspondence is NOT rechecked here; see verify_envelope.py (re-tokenize + GGUF matrix re-read)",
    });
    std::fs::write(
        out_dir.join("output.json"),
        serde_json::to_string_pretty(&envelope)? + "\n",
    )?;
    std::fs::write(out_dir.join("generated.txt"), format!("{text}\n"))?;
    println!("{text}");
    Ok(())
}
