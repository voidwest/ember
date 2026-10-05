//! `ember experiment lens`: logit lens over a verified bundle's
//! residual-stream captures.
//!
//! The bundle is verified (and optionally anchored) first; the supplied
//! model and the tokenizer must hash to the values the bundle recorded
//! (the same fail-closed rules as `reproduce`). Nothing is written into the
//! bundle: the report goes to stdout and, with `--out`, to a path outside it.

use crate::cli_experiment::{load_anchored_bundle, AnchorArgs};
use crate::cli_support::{default_tokenizer_for_arch, resolve_tokenizer, ResolvedTokenizer};
use anyhow::Context;
use clap::Args as ClapArgs;
use ember::extraction::sha256_file_result;
use ember::llama::Llama;
use ember::loader::load_gguf_with_k_strategy;
use ember::quant_k::KStrategy;
use ember::tokenizer::EmberTokenizer;
use ember::v05::lens::{compute_lens, LensOptions, LensReport, ModelLens, MAX_TOP_K};
use ember::v05::manifest::sha256_hex;
use std::path::{Path, PathBuf};

#[derive(ClapArgs)]
pub(crate) struct LensArgs {
    /// Bundle directory (verified before use; never modified).
    pub bundle: PathBuf,
    /// Model file (must hash to the bundle's recorded model SHA-256).
    #[arg(long, value_name = "model.gguf")]
    pub model: PathBuf,
    /// Tokenizer file (default: the path the bundle's spec names). Either
    /// way it must match the bundle's recorded SHA-256.
    #[arg(long, value_name = "tokenizer.json")]
    pub tokenizer: Option<PathBuf>,
    /// Anchors for the bundle, checked before anything is loaded.
    #[command(flatten)]
    pub anchor: AnchorArgs,
    /// Tokens to list per layer.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..=MAX_TOP_K as i64))]
    pub top_k: u32,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
    /// Also write the JSON report to this path (outside the bundle).
    #[arg(long, value_name = "lens.json")]
    pub out: Option<PathBuf>,
}

/// Hash and load a tokenizer from one read of its bytes, so the tokenizer
/// used is the one whose hash was checked.
fn load_pinned_tokenizer(
    resolved: &ResolvedTokenizer,
    expected_sha256: &str,
) -> anyhow::Result<EmberTokenizer> {
    let (sha, tokenizer) = match resolved {
        ResolvedTokenizer::File(path) => {
            let bytes =
                std::fs::read(path).with_context(|| format!("cannot read tokenizer '{path}'"))?;
            (sha256_hex(&bytes), Some(bytes))
        }
        ResolvedTokenizer::EmbeddedLlama => (resolved.sha256()?, None),
    };
    anyhow::ensure!(
        sha == expected_sha256,
        "tokenizer '{}' hashes to {sha} but the bundle records {expected_sha256}; the lens \
         requires the identical tokenizer",
        resolved.identity()
    );
    match tokenizer {
        Some(bytes) => EmberTokenizer::from_bytes(bytes),
        None => resolved.load(),
    }
}

/// Refuse an output path inside the bundle: bundles are immutable.
fn ensure_outside(bundle: &Path, path: &Path) -> anyhow::Result<()> {
    let bundle_root = bundle
        .canonicalize()
        .with_context(|| format!("cannot resolve '{}'", bundle.display()))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = parent
        .canonicalize()
        .with_context(|| format!("cannot resolve '{}'", parent.display()))?;
    anyhow::ensure!(
        !parent.starts_with(&bundle_root),
        "--out must point outside the bundle; the lens never modifies the bundle"
    );
    Ok(())
}

pub(crate) fn run_lens_command(
    command: &LensArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    // Verify (and anchor) once; everything below uses the verified bytes.
    let bundle = load_anchored_bundle(&command.bundle, &command.anchor)?;
    let manifest = &bundle.semantic_manifest;

    // Fail before any work if --out would land in the bundle.
    if let Some(out) = &command.out {
        ensure_outside(&command.bundle, out)?;
    }

    // -- model: must be the file the bundle recorded --
    let model_file = crate::cli_experiment::ModelFileIdentity::of(&command.model)?;
    let model_sha = sha256_file_result(&command.model)
        .with_context(|| format!("failed to hash '{}'", command.model.display()))?;
    if model_sha != manifest.model.sha256 {
        anyhow::bail!(
            "model '{}' hashes to {} but the bundle records {}; the lens requires the identical \
             model file",
            command.model.display(),
            model_sha,
            manifest.model.sha256
        );
    }
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, k_allow_fallback)?;
    let architecture = ember::loader::resolve_generation_architecture("auto", &loader)?;
    anyhow::ensure!(
        matches!(architecture.as_str(), "llama" | "qwen3"),
        "the lens supports llama-family models (llama/qwen3); got architecture '{architecture}'"
    );
    let model = Llama::from_loader_with_max_seq_len(loader, None)?;
    model_file.ensure_unchanged(&command.model)?;

    // -- tokenizer: --tokenizer, else the path the bound spec names --
    let tokenizer_path = match &command.tokenizer {
        Some(path) => path.display().to_string(),
        None => ember::v05::verify::bound_resolved_experiment(&bundle)
            .map_err(anyhow::Error::msg)?
            .model
            .tokenizer
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| default_tokenizer_for_arch(&architecture).to_string()),
    };
    let tokenizer = load_pinned_tokenizer(
        &resolve_tokenizer(&tokenizer_path),
        &manifest.tokenizer.sha256,
    )?;
    tokenizer.validate_model_vocab(model.config.vocab_size)?;

    let decode = |token: u32| {
        tokenizer
            .decode(&[token])
            .ok()
            .filter(|text| !text.is_empty())
            .or_else(|| tokenizer.token_piece(token))
            .unwrap_or_else(|| format!("<{token}>"))
    };
    let head = ModelLens::new(&model);
    let report = compute_lens(
        &bundle,
        &head,
        &decode,
        LensOptions {
            top_k: command.top_k as usize,
        },
    )
    .map_err(anyhow::Error::msg)?;

    if let Some(out) = &command.out {
        ensure_outside(&command.bundle, out)?;
        let mut bytes = serde_json::to_vec_pretty(&report)?;
        bytes.push(b'\n');
        ember::atomic_file::atomic_write(out, &bytes)
            .with_context(|| format!("cannot write '{}'", out.display()))?;
    }
    if command.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_table(&report, command.anchor.is_anchored());
        if let Some(out) = &command.out {
            println!("report written to {}", out.display());
        }
    }
    Ok(())
}

fn print_table(report: &LensReport, anchored: bool) {
    println!("logit lens: {}", report.bundle);
    println!(
        "  semantic hash: {} ({})",
        report.semantic_hash,
        if anchored {
            "anchored"
        } else {
            "not anchored: self-consistent only"
        }
    );
    println!(
        "  model {} layers, vocab {}; KL is {}",
        report.n_layers, report.vocab_size, report.kl_direction
    );
    let check = &report.final_layer_check;
    if check.checked > 0 {
        println!(
            "  final-layer check: lens top-1 equals the generated next token for {}/{} rows{}",
            check.top1_matches,
            check.checked,
            match check.greedy {
                Some(true) => " (greedy run: all expected)",
                Some(false) => " (sampled run: informational)",
                None => "",
            }
        );
    }
    for skipped in &report.skipped {
        println!(
            "  skipped capture '{}' at {}: {}",
            skipped.capture_id, skipped.site, skipped.reason
        );
    }
    for row in &report.rows {
        println!();
        println!(
            "{} | input {} | position {} ({}) | {} | route {} | {}",
            row.capture_id,
            row.input_id,
            row.position,
            row.phase,
            row.site,
            row.hook_route,
            row.dtype
        );
        let token = row
            .token
            .as_ref()
            .map(|t| format!("{:?}", t.text))
            .unwrap_or_else(|| "-".into());
        let next = row
            .next_token
            .as_ref()
            .map(|t| format!("{:?} (id {}, {})", t.text, t.token_id, t.source))
            .unwrap_or_else(|| "- (end of sequence)".into());
        println!("  token {token} -> next {next}");
        println!(
            "  {:>5} {:>5} {:>8} {:>9} {:>8} {:>9}  top-{}",
            "layer", "depth", "entropy", "KL>final", "nextrank", "next p", report.top_k
        );
        for layer in &row.layers {
            let top: Vec<String> = layer
                .top
                .iter()
                .map(|t| format!("{:?} {:.3}", t.text, t.probability))
                .collect();
            println!(
                "  {:>5} {:>5} {:>8.3} {:>9} {:>8} {:>9}  {}",
                layer.layer,
                layer.depth,
                layer.entropy_nats,
                layer
                    .kl_to_final_nats
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "-".into()),
                layer
                    .next_token_rank
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into()),
                layer
                    .next_token_probability
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "-".into()),
                top.join(", ")
            );
        }
    }
    for note in &report.notes {
        println!("note: {note}");
    }
}
