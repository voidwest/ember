//! Forced-choice log-prob scoring with an optional centered-residual patch (Phase C).
//!
//! Resident loop over a jobs file: the model loads once, then each job
//! prefills its prompt through the embedding path (patch armed for that one
//! forward only) and teacher-forces two candidate continuations from cloned
//! caches. Patch `None` (ZERO) traverses the hooked path with structural
//! no-ops; the parity test compares it against the stock token path.

use anyhow::Context;
use clap::Args;
use ember::backend::{Backend, CpuBackend};
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::quant_k::KStrategy;
use ember::residual_patch::{prefill_embed_with_patch, PatchSite, SpanPatch};
use ember::tokenizer::EmberTokenizer;

#[derive(Args)]
pub(crate) struct InterveneScoreCommand {
    /// Receiver GGUF model path.
    #[arg(long)]
    model: String,
    /// Tokenizer JSON path.
    #[arg(long)]
    tokenizer: String,
    /// Architecture hint.
    #[arg(long, default_value = "auto", value_parser = ["auto", "llama", "qwen3"])]
    arch: String,
    /// Jobs JSONL: {key,prompt,rows,site,layer,delta,alpha,candidates,meta}.
    #[arg(long)]
    jobs: String,
    /// Output results JSONL path.
    #[arg(long)]
    out: String,
    #[arg(long)]
    overwrite: bool,
}

pub(crate) fn run_intervene_score_command(
    command: &InterveneScoreCommand,
    k_strategy: KStrategy,
    allow_fallback: bool,
) -> anyhow::Result<()> {
    if std::path::Path::new(&command.out).exists() && !command.overwrite {
        anyhow::bail!("out '{}' exists; pass --overwrite", command.out);
    }
    let jobs_text = std::fs::read_to_string(&command.jobs).with_context(|| "read jobs file")?;
    let jobs: Vec<serde_json::Value> = jobs_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .context("parse jobs jsonl")?;
    anyhow::ensure!(!jobs.is_empty(), "no jobs");

    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, allow_fallback)?;
    let architecture = ember::loader::resolve_generation_architecture(&command.arch, &loader)?;
    anyhow::ensure!(
        architecture == "llama" || architecture == "qwen3",
        "intervene-score supports llama/qwen3 (got '{architecture}')"
    );
    let model = ember::llama::Llama::from_loader_with_max_seq_len(loader, None)?;
    let tokenizer = EmberTokenizer::from_file(&command.tokenizer)?;
    let vocab = model.config.vocab_size;
    tokenizer.validate_model_vocab(vocab)?;
    let backend = CpuBackend;

    // stream results as they complete so killed runs keep their prefix;
    // drivers resume by filtering keys already present in the output.
    let out_file = std::fs::File::create(&command.out)
        .with_context(|| format!("create out '{}'", command.out))?;
    let mut out_writer = std::io::BufWriter::new(out_file);
    use std::io::Write as _;
    let mut done = 0usize;
    for job in jobs.iter() {
        let key = job["key"].as_str().unwrap_or("?").to_string();
        let prompt = job["prompt"]
            .as_str()
            .context("job missing prompt")?
            .to_string();
        let prompt_ids = tokenizer.encode(&prompt)?;
        anyhow::ensure!(!prompt_ids.is_empty(), "job {key}: empty prompt ids");
        let rows: Vec<usize> = serde_json::from_value(job["rows"].clone()).context("rows")?;
        let site = match job["site"].as_str().unwrap_or("embed") {
            "embed" => PatchSite::EmbedSpan,
            "hidden" => PatchSite::HiddenLayer(
                job["layer"].as_u64().context("hidden site needs layer")? as usize,
            ),
            s => anyhow::bail!("job {key}: unknown site '{s}'"),
        };
        let alpha = job["alpha"].as_f64().unwrap_or(0.0) as f32;
        let patch = match &job["delta"] {
            serde_json::Value::Null => None,
            v => {
                let delta: Vec<f32> = serde_json::from_value(v.clone()).context("delta")?;
                Some(SpanPatch {
                    site,
                    rows: rows.clone(),
                    delta,
                    alpha,
                })
            }
        };
        let cands: Vec<Vec<u32>> =
            serde_json::from_value(job["candidates"].clone()).context("candidates")?;
        anyhow::ensure!(cands.len() == 2, "job {key}: need exactly 2 candidates");
        for c in cands.iter() {
            anyhow::ensure!(!c.is_empty(), "job {key}: empty candidate");
        }
        let max_new = cands.iter().map(|c| c.len()).max().unwrap_or(0);
        let need = prompt_ids
            .len()
            .checked_add(max_new)
            .context("context overflow")?;
        anyhow::ensure!(
            need <= model.config.max_seq_len,
            "job {key}: need {need} > context {}",
            model.config.max_seq_len
        );
        let mut cache = model.create_request_cache(&backend, prompt_ids.len(), max_new);
        let (logits_t, diag) =
            prefill_embed_with_patch(&model, &backend, &prompt_ids, &mut cache, 0, patch)?;
        let shape = backend.shape(&logits_t);
        anyhow::ensure!(
            shape == [1, vocab],
            "job {key}: prefill logits shape {shape:?} != [1, {vocab}]"
        );
        // teacher-force each candidate from a cloned prefill cache
        let mut lps = Vec::with_capacity(2);
        for cand in cands.iter() {
            let mut ccache = cache.clone();
            let mut total = 0.0f64;
            let mut step_ids: Vec<u32> = Vec::new();
            for (t, &tok) in cand.iter().enumerate() {
                let step_logits = if t == 0 {
                    logits_t.clone()
                } else {
                    ForwardModel::forward_last_logits_with_cache(
                        &model,
                        &backend,
                        &step_ids,
                        &mut ccache,
                        prompt_ids.len() + t - 1,
                    )?
                };
                let data = backend.data(&step_logits);
                anyhow::ensure!(data.len() == vocab, "job {key}: step logits len");
                let m = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let mut s = 0.0f64;
                for &v in data.iter() {
                    s += ((v as f64) - m).exp();
                }
                let lse = m + s.ln();
                anyhow::ensure!(
                    (tok as usize) < vocab,
                    "job {key}: candidate id out of range"
                );
                total += (data[tok as usize] as f64) - lse;
                step_ids = vec![tok];
            }
            lps.push(total);
        }
        let row = serde_json::json!({
            "key": key,
            "prompt_ids": prompt_ids,
            "logits_shape": shape,
            "lp": lps,
            "margin": lps[0] - lps[1],
            "diag": {"fired": diag.fired, "h_norm": diag.h_norm,
                     "hp_norm": diag.hp_norm, "cos": diag.cos,
                     "delta_norm": diag.delta_norm},
            "meta": job.get("meta").cloned().unwrap_or(serde_json::Value::Null),
        });
        writeln!(out_writer, "{}", serde_json::to_string(&row)?)?;
        done += 1;
        if done.is_multiple_of(25) {
            out_writer.flush()?;
        }
    }
    out_writer.flush()?;
    Ok(())
}

/// Collect per-layer span-mean states (Phase E direction estimation).
///
/// Input JSONL rows: {"id","prompt","rows":[absolute prefill positions]}.
/// Runs one uncached forward per prompt. Writes:
/// - `<out>.npy`: f32 C-order matrix [(N*L), d], row (i*L+l) = example i layer l.
/// - `<out>.json`: {ids, n_layers, embed_dim, model/tokenizer/prompts SHAs}.
#[derive(Args)]
pub(crate) struct CollectSpansCommand {
    /// Receiver GGUF model path.
    #[arg(long)]
    model: String,
    /// Tokenizer JSON path.
    #[arg(long)]
    tokenizer: String,
    /// Architecture hint.
    #[arg(long, default_value = "auto", value_parser = ["auto", "llama", "qwen3"])]
    arch: String,
    /// Prompts JSONL path.
    #[arg(long)]
    prompts: String,
    /// Output JSON path.
    #[arg(long)]
    out: String,
    #[arg(long)]
    overwrite: bool,
    /// Use the cached embedding-prefill path shared with intervene-score.
    #[arg(long)]
    cached: bool,
}

pub(crate) fn run_collect_spans_command(
    command: &CollectSpansCommand,
    k_strategy: KStrategy,
    allow_fallback: bool,
) -> anyhow::Result<()> {
    use ember::extraction::sha256_file_result;
    if std::path::Path::new(&command.out).exists() && !command.overwrite {
        anyhow::bail!("out '{}' exists; pass --overwrite", command.out);
    }
    let prompts_text =
        std::fs::read_to_string(&command.prompts).with_context(|| "read prompts file")?;
    let prompts_sha = ember::extraction::sha256_bytes(prompts_text.as_bytes());
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, allow_fallback)?;
    let architecture = ember::loader::resolve_generation_architecture(&command.arch, &loader)?;
    anyhow::ensure!(
        architecture == "llama" || architecture == "qwen3",
        "collect-spans supports llama/qwen3 (got '{architecture}')"
    );
    let model = ember::llama::Llama::from_loader_with_max_seq_len(loader, None)?;
    let tokenizer = EmberTokenizer::from_file(&command.tokenizer)?;
    let vocab = model.config.vocab_size;
    tokenizer.validate_model_vocab(vocab)?;
    let backend = CpuBackend;
    let n_layers = model.blocks.len();
    let embed_dim = model.config.embed_dim;
    let npy_path = format!("{}.npy", command.out);
    if std::path::Path::new(&npy_path).exists() && !command.overwrite {
        anyhow::bail!("out '{npy_path}' exists; pass --overwrite");
    }
    // collect rows first so the npy stream shape is exact up front
    let mut inputs: Vec<(String, Vec<u32>, Vec<usize>)> = Vec::new();
    for line in prompts_text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(line)?;
        let id = row["id"].as_str().context("row missing id")?.to_string();
        let prompt = row["prompt"].as_str().context("row missing prompt")?;
        let rows: Vec<usize> = serde_json::from_value(row["rows"].clone()).context("row rows")?;
        anyhow::ensure!(!rows.is_empty(), "row {id}: empty span rows");
        let token_ids = tokenizer.encode(prompt)?;
        anyhow::ensure!(!token_ids.is_empty(), "row {id}: empty prompt ids");
        for &r in rows.iter() {
            anyhow::ensure!(
                r < token_ids.len(),
                "row {id}: span row {r} >= seq {}",
                token_ids.len()
            );
        }
        inputs.push((id, token_ids, rows));
    }
    anyhow::ensure!(!inputs.is_empty(), "no prompt rows");
    let mut writer =
        ember::npy::NpyStreamWriter::create(&npy_path, &[inputs.len() * n_layers, embed_dim])
            .context("create npy stream")?;
    let mut ids = Vec::with_capacity(inputs.len());
    for (id, token_ids, rows) in inputs.iter() {
        if command.cached {
            let means = ember::residual_patch::collect_cached_span_means(
                &model, &backend, token_ids, rows,
            )?;
            anyhow::ensure!(means.len() == n_layers, "row {id}: layer count");
            for mean in means {
                writer.write_f32s(&mean)?;
            }
            ids.push(id.clone());
            continue;
        }
        let (states, _) =
            ember::llama::Llama::forward_with_activations(&model, &backend, token_ids)
                .map_err(|e| anyhow::anyhow!("forward failed for {id}: {e:?}"))?;
        anyhow::ensure!(states.len() == n_layers, "row {id}: layer count");
        let seq_len = token_ids.len();
        for st in states.iter() {
            anyhow::ensure!(st.len() == seq_len * embed_dim, "row {id}: state shape");
            let mut mean = vec![0.0f32; embed_dim];
            for &r in rows.iter() {
                let base = r * embed_dim;
                for (j, acc) in mean.iter_mut().enumerate() {
                    *acc += st[base + j];
                }
            }
            let n = rows.len() as f32;
            for acc in mean.iter_mut() {
                *acc /= n;
            }
            writer.write_f32s(&mean)?;
        }
        ids.push(id.clone());
    }
    writer.finish()?;
    let payload = serde_json::json!({
        "ids": ids,
        "n_examples": ids.len(),
        "n_layers": n_layers,
        "embed_dim": embed_dim,
        "npy": npy_path,
        "model_sha256": sha256_file_result(&command.model)?,
        "tokenizer_sha256": sha256_file_result(&command.tokenizer)?,
        "prompts_sha256": prompts_sha,
        "capture_path": if command.cached { "cached_embedding_prefill" } else { "uncached" },
    });
    std::fs::write(&command.out, serde_json::to_string_pretty(&payload)?)?;
    Ok(())
}
