//! `ember score-batch` — batch greedy generation and next-token log-probability
//! scoring with a single resident model (loads the model once per process).
//!
//! Input: JSONL lines `{"id": "...", "kind": "gen"|"ll", "prompt": "...",
//!                      "max_tokens": N (gen, optional), "next_id": N (ll)}`
//! Output: JSONL lines `{"id": ..., "text": ...}` (gen) or
//!         `{"id": ..., "lp": f, "top1_id": i, "top1_lp": f}` (ll).
//!
//! `ll` log-probs are computed from the full-vocab logits at the last prompt
//! position with float64 log-softmax — numerically identical to the
//! `--dump-logits` + Python pipeline used for calibration (parity asserted by
//! the pilot runner before the main run).

use anyhow::Context;
use clap::Args as ClapArgs;
use ember::backend::{Backend, CpuBackend};
use ember::loader::load_gguf_with_k_strategy;
use ember::model::{ForwardModel, Gpt2};
use ember::quant_k::KStrategy;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use crate::cli_commands::ensure_sequence_fits;
use crate::cli_support::{default_tokenizer_for_arch, validate_token_ids_for_model};
use crate::rayon_current_num_threads;
use ember::loader::resolve_generation_architecture;

#[derive(ClapArgs)]
pub(crate) struct ScoreBatchCommand {
    /// path to the GGUF model
    #[arg(short, long)]
    model: String,

    /// path to tokenizer.json (defaults per architecture)
    #[arg(long)]
    tokenizer: Option<String>,

    /// model architecture override; auto reads general.architecture from GGUF
    #[arg(long, default_value = "auto", value_parser = ["auto", "gpt2", "llama", "qwen3", "gemma4"])]
    arch: String,

    /// optional context-size cap
    #[arg(long)]
    max_seq_len: Option<usize>,

    /// JSONL input: {"id","kind":"gen"|"ll","prompt","max_tokens"?,"next_id"?}
    #[arg(long)]
    input: String,

    /// JSONL output path
    #[arg(long)]
    output: String,
}

/// float64 log-softmax over the last-position logits; returns
/// (lp of next_id, top1 id, top1 lp). Mirrors the Python dump-logits pipeline.
fn ll_logprob<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    prompt: &str,
    next_id: usize,
    context_limit: usize,
) -> anyhow::Result<(f64, u32, f64)>
where
    B::Error: Send + Sync + 'static,
{
    let token_ids = tokenizer
        .encode(prompt)
        .with_context(|| format!("failed to tokenize prompt {prompt:?}"))?;
    if token_ids.is_empty() {
        anyhow::bail!("cannot score an empty prompt");
    }
    let vocab = model.vocab_size(backend);
    tokenizer.validate_model_vocab(vocab)?;
    validate_token_ids_for_model(&token_ids, vocab, "score-batch prompt")?;
    ensure_sequence_fits(token_ids.len(), 0, context_limit)?;
    if next_id >= vocab {
        anyhow::bail!("next_id {next_id} out of range (vocab {vocab})");
    }
    let mut cache = model.create_cache(backend, token_ids.len());
    let logits = model.forward_last_logits_with_cache(backend, &token_ids, &mut cache, 0)?;
    let data = backend.data(&logits);
    anyhow::ensure!(
        data.len() == vocab,
        "logits length {} != vocab {vocab}",
        data.len()
    );
    let m = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum = 0.0f64;
    for &v in data.iter() {
        sum += ((v as f64) - m).exp();
    }
    let logsum = m + sum.ln();
    let top1 = data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("f32 cmp"))
        .map(|(i, &v)| (i as u32, (v as f64) - logsum))
        .context("empty logits")?;
    let lp = (data[next_id] as f64) - logsum;
    Ok((lp, top1.0, top1.1))
}

pub(crate) fn run_score_batch_command(
    command: &ScoreBatchCommand,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, k_allow_fallback)?;
    let arch = resolve_generation_architecture(&command.arch, &loader)?;
    let tokenizer_path = command
        .tokenizer
        .as_deref()
        .unwrap_or_else(|| default_tokenizer_for_arch(&arch));
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(tokenizer_path)?;
    let backend = CpuBackend;

    let reader = BufReader::new(File::open(&command.input)?);
    let out = BufWriter::new(File::create(&command.output)?);

    match arch.as_str() {
        "gpt2" => {
            let model = Gpt2::from_loader(loader)?;
            run_lines(&backend, &model, &tokenizer, command, reader, out)
        }
        "llama" | "qwen3" => {
            use ember::llama::Llama;
            let model = Llama::from_loader_with_max_seq_len(loader, command.max_seq_len)?;
            crate::validate_tokenizer_model_contract(&backend, &model, &tokenizer)?;
            run_lines(&backend, &model, &tokenizer, command, reader, out)
        }
        "gemma4" => {
            use ember::gemma4::Gemma4;
            let model = Gemma4::from_loader(loader)?;
            run_lines(&backend, &model, &tokenizer, command, reader, out)
        }
        other => anyhow::bail!("unsupported score-batch architecture '{other}'"),
    }
}

fn run_lines<B: Backend, M: ForwardModel<B>>(
    backend: &B,
    model: &M,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    command: &ScoreBatchCommand,
    reader: BufReader<File>,
    mut out: BufWriter<File>,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let context_limit = match command.max_seq_len {
        Some(cap) => cap.min(model.max_seq_len(backend)),
        None => model.max_seq_len(backend),
    };
    let mut n_ok = 0usize;
    let mut n_err = 0usize;
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let v: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("line {}: bad JSON", lineno + 1))?;
        let id = v.get("id").and_then(|x| x.as_str()).unwrap_or_default();
        let kind = v.get("kind").and_then(|x| x.as_str()).unwrap_or("gen");
        let prompt = v.get("prompt").and_then(|x| x.as_str()).unwrap_or_default();
        let line_result: anyhow::Result<serde_json::Value> = match kind {
            "gen" => {
                let max_tokens =
                    v.get("max_tokens").and_then(|x| x.as_u64()).unwrap_or(12) as usize;
                let text = crate::cli_generation::generate(
                    backend,
                    model,
                    tokenizer,
                    prompt,
                    max_tokens,
                    0.0,
                    None,
                    None,
                    false,
                    false,
                    None,
                    false,
                    false,
                    rayon_current_num_threads(),
                    context_limit,
                    None,
                )?;
                Ok(serde_json::json!({ "id": id, "text": text }))
            }
            "ll" => {
                let next_id = v
                    .get("next_id")
                    .and_then(|x| x.as_u64())
                    .context("ll line requires next_id")? as usize;
                let (lp, top1_id, top1_lp) =
                    ll_logprob(backend, model, tokenizer, prompt, next_id, context_limit)?;
                Ok(serde_json::json!({
                    "id": id, "lp": lp, "top1_id": top1_id, "top1_lp": top1_lp
                }))
            }
            "top1" => {
                // full-vocab distribution stats at the last position (calibration)
                let token_ids = tokenizer.encode(prompt).context("tokenize failed")?;
                if token_ids.is_empty() {
                    anyhow::bail!("cannot score an empty prompt");
                }
                let vocab = model.vocab_size(backend);
                tokenizer.validate_model_vocab(vocab)?;
                validate_token_ids_for_model(&token_ids, vocab, "score-batch prompt")?;
                ensure_sequence_fits(token_ids.len(), 0, context_limit)?;
                let mut cache = model.create_cache(backend, token_ids.len());
                let logits =
                    model.forward_last_logits_with_cache(backend, &token_ids, &mut cache, 0)?;
                let data = backend.data(&logits);
                let m = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let mut sum = 0.0f64;
                for &v in data.iter() {
                    sum += ((v as f64) - m).exp();
                }
                let logsum = m + sum.ln();
                let mut top1_id = 0u32;
                let mut top1_lp = f64::NEG_INFINITY;
                let mut entropy = 0.0f64;
                for (i, &v) in data.iter().enumerate() {
                    let lp = (v as f64) - logsum;
                    if lp > top1_lp {
                        top1_lp = lp;
                        top1_id = i as u32;
                    }
                    entropy -= lp.exp() * lp;
                }
                Ok(serde_json::json!({
                    "id": id, "top1_id": top1_id, "top1_lp": top1_lp, "entropy": entropy
                }))
            }
            other => Err(anyhow::anyhow!("unknown kind {other:?}")),
        };
        match line_result {
            Ok(v) => {
                writeln!(out, "{}", serde_json::to_string(&v)?)?;
                n_ok += 1;
            }
            Err(e) => {
                writeln!(
                    out,
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "id": id, "error": format!("{e:#}")
                    }))?
                )?;
                n_err += 1;
            }
        }
    }
    out.flush()?;
    eprintln!(
        "score-batch: {n_ok} ok, {n_err} errors -> {}",
        command.output
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ember::backend::CpuError;
    use ember::kv_cache::KVCache;
    use ember::tensor::CpuTensor;
    use std::cell::RefCell;

    #[derive(Default)]
    struct ScoringModel {
        capacities: RefCell<Vec<usize>>,
    }

    impl ForwardModel<CpuBackend> for ScoringModel {
        fn create_cache(&self, _: &CpuBackend, capacity: usize) -> KVCache {
            self.capacities.borrow_mut().push(capacity);
            KVCache::new(1, 1, 2, capacity)
        }
        fn max_seq_len(&self, _: &CpuBackend) -> usize {
            131_072
        }
        fn n_layers(&self) -> usize {
            1
        }
        fn embed_dim(&self) -> usize {
            2
        }
        fn vocab_size(&self, _: &CpuBackend) -> usize {
            3
        }
        fn forward_with_cache(
            &self,
            _: &CpuBackend,
            _: &[u32],
            _: &mut KVCache,
            _: usize,
        ) -> Result<CpuTensor, CpuError> {
            panic!("scoring must request only last-position logits")
        }
        fn forward_last_logits_with_cache(
            &self,
            _: &CpuBackend,
            ids: &[u32],
            cache: &mut KVCache,
            start: usize,
        ) -> Result<CpuTensor, CpuError> {
            assert_eq!(cache.max_seq_len(), ids.len());
            assert_eq!(start, 0);
            Ok(CpuTensor::from_data(vec![1, 3], vec![1.0, 2.0, 0.0]))
        }
        fn forward_with_activations(
            &self,
            _: &CpuBackend,
            _: &[u32],
        ) -> Result<(Vec<Vec<f32>>, CpuTensor), CpuError> {
            panic!("scoring must not capture activations")
        }
    }

    fn tokenizer() -> ember::tokenizer::EmberTokenizer {
        ember::tokenizer::EmberTokenizer::from_bytes(
            r#"{
            "version":"1.0", "truncation":null, "padding":null,
            "added_tokens":[], "normalizer":null, "pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel","vocab":{"x":0,"y":1,"[UNK]":2},"unk_token":"[UNK]"}
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn logits_only_batch_sizes_cache_to_prompt_and_preserves_scores() {
        let root = std::env::temp_dir().join(format!(
            "ember-score-capacity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let input = root.join("in.jsonl");
        let output = root.join("out.jsonl");
        std::fs::write(
            &input,
            concat!(
                "{\"id\":\"a\",\"kind\":\"ll\",\"prompt\":\"x y\",\"next_id\":1}\n",
                "{\"id\":\"b\",\"kind\":\"top1\",\"prompt\":\"x\"}\n"
            ),
        )
        .unwrap();
        let command = ScoreBatchCommand {
            model: "unused".into(),
            tokenizer: None,
            arch: "auto".into(),
            max_seq_len: None,
            input: input.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
        };
        let model = ScoringModel::default();
        run_lines(
            &CpuBackend,
            &model,
            &tokenizer(),
            &command,
            BufReader::new(File::open(&input).unwrap()),
            BufWriter::new(File::create(&output).unwrap()),
        )
        .unwrap();
        assert_eq!(*model.capacities.borrow(), vec![2, 1]);
        let records: Vec<serde_json::Value> = std::fs::read_to_string(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let expected = -(1.0 + (-1.0f64).exp() + (-2.0f64).exp()).ln();
        assert_eq!(records[0]["top1_id"], 1);
        assert_eq!(records[1]["top1_id"], 1);
        assert!((records[0]["lp"].as_f64().unwrap() - expected).abs() < 1e-15);
        assert_eq!(records[0]["top1_lp"], records[1]["top1_lp"]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scoring_rejects_overlong_prompts_before_allocating_cache() {
        let model = ScoringModel::default();
        assert!(ll_logprob(&CpuBackend, &model, &tokenizer(), "x y", 1, 1).is_err());
        assert!(model.capacities.borrow().is_empty());
    }

    #[test]
    fn batch_tokenizer_default_matches_generation_for_qwen() {
        assert_eq!(default_tokenizer_for_arch("qwen3"), "tokenizer-qwen3.json");
        assert_eq!(default_tokenizer_for_arch("llama"), "tokenizer.json");
        assert_eq!(
            default_tokenizer_for_arch("gemma4"),
            "tokenizer-gemma4.json"
        );
    }
}
