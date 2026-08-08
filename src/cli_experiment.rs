//! Ember v0.5 experiment CLI driver: validate, run, inspect, verify,
//! compare, reproduce, tokenize.
//!
//! The run path loads the model and tokenizer, drives every input through
//! the existing generation machinery with a v0.5 experiment attached, and
//! assembles + self-verifies the deterministic bundle.

use crate::cli_support::{default_tokenizer_for_arch, gguf_metadata_json, resolve_tokenizer};
use anyhow::Context;
use clap::{Args as ClapArgs, Subcommand};
use ember::experiments::{
    ExecutionContext, Experiment, ExperimentError, ExperimentRunner, GenerationContext,
    LayerContext, ModelContext, ModelFamily, TensorAccess,
};
use ember::extraction::sha256_file_result;
use ember::llama::Llama;
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::plan::{ExecutionMode, HookMode};
use ember::quant_k::KStrategy;
use ember::tokenizer::EmberTokenizer;
use ember::v05::compare::compare_bundles;
use ember::v05::manifest::BundleIdentity;
use ember::v05::run::{
    write_bundle, BundleMaterials, ModelBundleMeta, RuntimeMetrics, TokenizerBundleMeta,
};
use ember::v05::runner::{
    load_bundle_source, BundleSource, InputResult, ModelFacts, V05Experiment,
};
use ember::v05::spec::{RawExperimentSpec, EXPERIMENT_SCHEMA_V1};
use ember::v05::token_select::{tokenize_for_selection, TextNormalization};
use ember::v05::verify::{verify_bundle, VerifyOptions};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// `ember experiment ...` subcommands.
#[derive(ClapArgs)]
pub(crate) struct ExperimentCommand {
    #[command(subcommand)]
    pub command: ExperimentSubcommand,
}

#[derive(Subcommand)]
pub(crate) enum ExperimentSubcommand {
    /// Validate an experiment specification without inference.
    Validate(ValidateArgs),
    /// Resolve, execute, and bundle an experiment.
    Run(RunArgs),
    /// Summarize a bundle's contents.
    Inspect(InspectArgs),
    /// Verify a bundle offline (optionally deep against a model file).
    Verify(VerifyArgs),
    /// Compare two bundles semantically and numerically.
    Compare(CompareArgs),
    /// Re-run a bundle's experiment and classify reproduction.
    Reproduce(ReproduceArgs),
    /// Inspect tokenization and span matching.
    Tokenize(TokenizeArgs),
}

#[derive(ClapArgs)]
pub(crate) struct ValidateArgs {
    /// Path to the experiment specification (TOML).
    pub spec: PathBuf,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct RunArgs {
    /// Path to the experiment specification (TOML).
    pub spec: PathBuf,
    /// Override the specification's execution mode.
    #[arg(long, value_name = "reference|planned|planned-fused")]
    pub execution: Option<String>,
    /// Override the specification's thread count.
    #[arg(long)]
    pub threads: Option<usize>,
    /// Override the specification's output directory.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Keep the staging directory on failure (clearly marked incomplete).
    #[arg(long)]
    pub retain_incomplete: bool,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct InspectArgs {
    /// Bundle directory.
    pub bundle: PathBuf,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct VerifyArgs {
    /// Bundle directory.
    pub bundle: PathBuf,
    /// Deep verification against a model file.
    #[arg(long, value_name = "model.gguf")]
    pub model: Option<PathBuf>,
    /// Deep tokenizer verification.
    #[arg(long, value_name = "tokenizer.json")]
    pub tokenizer: Option<PathBuf>,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct CompareArgs {
    /// First bundle directory.
    pub a: PathBuf,
    /// Second bundle directory.
    pub b: PathBuf,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct ReproduceArgs {
    /// Bundle directory to reproduce.
    pub bundle: PathBuf,
    /// Model file to re-run with (validated against the bundle hash).
    #[arg(long, value_name = "model.gguf")]
    pub model: PathBuf,
    /// Output directory for the new bundle (default:
    /// `<bundle>-reproduced`).
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Keep the staging directory on failure.
    #[arg(long)]
    pub retain_incomplete: bool,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct TokenizeArgs {
    /// GGUF model file.
    #[arg(long)]
    pub model: PathBuf,
    /// Architecture override (`auto`, `gpt2`, `llama`, `qwen3`, `gemma4`).
    #[arg(long, default_value = "auto")]
    pub arch: String,
    /// tokenizer.json path.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Text to tokenize.
    #[arg(long)]
    pub text: String,
    /// Optional span to match (matched-span selection, occurrence 0,
    /// all subtokens).
    #[arg(long)]
    pub match_span: Option<String>,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

/// Adapter that lets the v0.5 experiment ride the v0.4 hook machinery
/// while the driver keeps access through the shared handle.
struct V05Adapter(Arc<Mutex<V05Experiment>>);

impl Experiment for V05Adapter {
    fn name(&self) -> &'static str {
        "v05-experiment"
    }

    fn intervenes(&self) -> bool {
        self.0.lock().expect("v05 experiment lock").intervenes()
    }

    fn arguments(&self) -> serde_json::Value {
        serde_json::json!({"kind": "v05-experiment"})
    }

    fn on_model_loaded(&mut self, ctx: &ModelContext<'_>) -> Result<(), ExperimentError> {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .on_model_loaded(ctx)
    }

    fn before_prefill(&mut self, ctx: &ExecutionContext<'_>) -> Result<(), ExperimentError> {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .before_prefill(ctx)
    }

    fn before_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .before_layer(ctx, hidden)
    }

    fn after_attention(
        &mut self,
        ctx: &LayerContext<'_>,
        attention_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .after_attention(ctx, attention_output)
    }

    fn after_mlp(
        &mut self,
        ctx: &LayerContext<'_>,
        mlp_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .after_mlp(ctx, mlp_output)
    }

    fn after_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .after_layer(ctx, hidden)
    }

    fn before_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .before_logits(ctx, hidden)
    }

    fn after_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        logits: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .after_logits(ctx, logits)
    }

    fn on_generation_complete(
        &mut self,
        ctx: &GenerationContext<'_>,
    ) -> Result<(), ExperimentError> {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .on_generation_complete(ctx)
    }
}

fn family_for_arch(arch: &str) -> ModelFamily {
    match arch {
        "llama" => ModelFamily::Llama,
        "qwen3" => ModelFamily::Qwen3,
        "gemma4" => ModelFamily::Gemma4,
        _ => ModelFamily::Llama,
    }
}

/// Run one fully-resolved experiment and write + self-verify its bundle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_resolved(
    resolved: &ember::v05::spec::ExperimentSpecV1,
    spec_text: &str,
    output_directory: &std::path::Path,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
    retain_incomplete: bool,
) -> anyhow::Result<(
    PathBuf,
    BundleIdentity,
    ember::v05::verify::VerificationReport,
    Vec<InputResult>,
)> {
    let threads = if resolved.execution.threads > 0 {
        resolved.execution.threads
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("failed to build the experiment thread pool")?
        .install(|| {
            let prepared = prepare_run(resolved, k_strategy, k_allow_fallback)?;
            execute_prepared(
                &prepared,
                resolved,
                spec_text,
                output_directory,
                retain_incomplete,
            )
        })
}

/// A fully loaded, reusable experiment session.
///
/// Loading is separated from execution so the GUI can keep one model
/// resident across many runs (baseline, intervention, restore). The CLI
/// path is unchanged: `execute_resolved` prepares and executes in one call.
pub(crate) struct PreparedRun {
    pub model: Llama<ember::backend::CpuBackend>,
    pub tokenizer: EmberTokenizer,
    pub architecture: String,
    pub n_layers: usize,
    pub embed_dim: usize,
    pub model_sha: String,
    pub tokenizer_sha: String,
    pub gguf_metadata: serde_json::Value,
    pub model_path: PathBuf,
}

/// Load the model + tokenizer for a resolved experiment and validate
/// provenance hashes (model/tokenizer SHA when the spec pins them).
/// No inference happens here; the loaded model is reusable across runs.
pub(crate) fn prepare_run(
    resolved: &ember::v05::spec::ExperimentSpecV1,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<PreparedRun> {
    // -- model --
    let loader = load_gguf_with_k_strategy(&resolved.model.path, k_strategy, k_allow_fallback)?;
    let architecture =
        crate::cli_support::resolve_generation_architecture(&resolved.model.arch, &loader)?;
    if !matches!(architecture.as_str(), "llama" | "qwen3") {
        anyhow::bail!(
            "experiments support llama-family models (llama/qwen3); got architecture \
             '{architecture}'"
        );
    }
    let gguf_metadata = gguf_metadata_json(&loader);
    let model = Llama::from_loader_with_max_seq_len(loader, None)?;
    let n_layers = model.config.n_layers;
    let embed_dim = model.config.embed_dim;

    let model_sha = sha256_file_result(&resolved.model.path)
        .with_context(|| format!("failed to hash model '{}'", resolved.model.path.display()))?;
    if !resolved.model.expected_sha256.is_empty() && resolved.model.expected_sha256 != model_sha {
        anyhow::bail!(
            "model SHA-256 mismatch: spec expects {} but '{}' hashes to {}",
            resolved.model.expected_sha256,
            resolved.model.path.display(),
            model_sha
        );
    }

    // -- tokenizer --
    let tokenizer_path = resolved
        .model
        .tokenizer
        .clone()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| default_tokenizer_for_arch(&architecture).to_string());
    let resolved_tokenizer = resolve_tokenizer(&tokenizer_path);
    let tokenizer_sha = resolved_tokenizer.sha256()?;
    if !resolved.model.tokenizer_expected_sha256.is_empty()
        && resolved.model.tokenizer_expected_sha256 != tokenizer_sha
    {
        anyhow::bail!(
            "tokenizer SHA-256 mismatch: spec expects {} but '{}' hashes to {}",
            resolved.model.tokenizer_expected_sha256,
            resolved_tokenizer.identity(),
            tokenizer_sha
        );
    }
    let tokenizer = resolved_tokenizer.load()?;
    tokenizer.validate_model_vocab(model.config.vocab_size)?;

    Ok(PreparedRun {
        model,
        tokenizer,
        architecture,
        n_layers,
        embed_dim,
        model_sha,
        tokenizer_sha,
        gguf_metadata,
        model_path: resolved.model.path.clone(),
    })
}

/// Execute a resolved experiment against an already-loaded session:
/// build the plan, run every input through generation with the v0.5
/// experiment attached, assemble + write the bundle, and self-verify it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_prepared(
    prepared: &PreparedRun,
    resolved: &ember::v05::spec::ExperimentSpecV1,
    spec_text: &str,
    output_directory: &std::path::Path,
    retain_incomplete: bool,
) -> anyhow::Result<(
    PathBuf,
    BundleIdentity,
    ember::v05::verify::VerificationReport,
    Vec<InputResult>,
)> {
    let backend = ember::backend::CpuBackend;
    let mode = resolved.execution.mode;
    let threads = if resolved.execution.threads > 0 {
        resolved.execution.threads
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    };
    let model = &prepared.model;
    let tokenizer = &prepared.tokenizer;
    let architecture = &prepared.architecture;
    let model_sha = &prepared.model_sha;
    let tokenizer_sha = &prepared.tokenizer_sha;
    let n_layers = prepared.n_layers;
    let embed_dim = prepared.embed_dim;

    // -- execution plan --
    let has_captures = !resolved.captures.is_empty();
    let has_interventions = !resolved.interventions.is_empty();
    let hook_mode = if has_interventions {
        HookMode::Intervene
    } else if has_captures {
        HookMode::Observe
    } else {
        HookMode::Disabled
    };
    let all_stages: [&str; 6] = [
        "before-layer",
        "after-attention",
        "after-mlp",
        "after-layer",
        "before-logits",
        "after-logits",
    ];
    let stages: &[&str] = if hook_mode == HookMode::Disabled {
        &[]
    } else {
        &all_stages
    };
    let plan = model.execution_plan(
        mode,
        hook_mode,
        stages,
        model.config.max_seq_len,
        Some(model_sha),
        Some(tokenizer_sha),
    )?;
    model.set_execution_mode(mode);

    // -- cross-bundle sources --
    let mut bundle_sources: Vec<BundleSource> = Vec::new();
    for intervention in &resolved.interventions {
        if let Some(source) = &intervention.source {
            if let ember::v05::intervention::InterventionSource::CaptureFromBundle { .. } = source {
                let loaded =
                    load_bundle_source(intervention, source, model_sha, tokenizer_sha, n_layers)
                        .map_err(anyhow::Error::msg)?;
                bundle_sources.push(loaded);
            }
        }
    }

    // -- run every input --
    let facts = ModelFacts {
        n_layers,
        embed_dim,
        vocab_size: model.config.vocab_size,
    };
    let context_limit = model.max_seq_len(&backend);
    let mut results = Vec::new();
    let start = std::time::Instant::now();
    let mut total_generated = 0usize;
    let warnings = Vec::new();

    for (index, input) in resolved.inputs.iter().enumerate() {
        eprintln!(
            "experiment: input {} ({}) tokens={} mode={}",
            index + 1,
            input.id,
            input.text.len(),
            mode.name()
        );
        let inner = Arc::new(Mutex::new(V05Experiment::new(
            (*resolved).clone(),
            index,
            facts,
            Some(model_sha.clone()),
            Some(tokenizer_sha.clone()),
        )));
        {
            let mut experiment = inner.lock().expect("v05 experiment lock");
            let info = tokenize_for_selection(tokenizer, &input.text, TextNormalization::None)
                .map_err(anyhow::Error::msg)?;
            experiment.inject_tokenization(info);
            for source in &bundle_sources {
                experiment.inject_bundle_source(source.clone());
            }
        }
        let adapter = V05Adapter(Arc::clone(&inner));
        let mut runner = ExperimentRunner::new(adapter);
        let model_context = ModelContext::new(
            family_for_arch(architecture),
            Some(prepared.model_path.to_str().unwrap_or("model.gguf")),
            architecture,
            n_layers,
            embed_dim,
        )
        .with_provenance(Some(model_sha), Some(tokenizer_sha));
        let generated_text = crate::cli_generation::generate_with_experiment(
            &backend,
            model,
            &mut runner,
            model_context,
            tokenizer,
            &input.text,
            resolved.generation.max_new_tokens,
            resolved.generation.temperature,
            None,
            None,
            false,
            false,
            None,
            false,
            false,
            threads,
            context_limit,
            if resolved.generation.temperature > 0.0 && resolved.experiment.seed != 0 {
                Some(resolved.experiment.seed)
            } else {
                None
            },
        )?;
        {
            let mut experiment = inner.lock().expect("v05 experiment lock");
            experiment.set_generated_text(generated_text);
            let result = experiment.into_result().map_err(anyhow::Error::msg)?;
            total_generated += result.generated_token_ids.len();
            results.push(result);
        }
    }
    let wall_clock_ms = start.elapsed().as_secs_f64() * 1000.0;

    // -- assemble + write + self-verify --
    let runtime = RuntimeMetrics {
        wall_clock_ms,
        decode_throughput_tps: if wall_clock_ms > 0.0 {
            Some(total_generated as f64 / (wall_clock_ms / 1000.0))
        } else {
            None
        },
        prefill_throughput_tps: None,
        first_token_latency_ms: None,
        peak_rss_kb: peak_rss_kb(),
        threads,
    };
    let mut resolved_with_output = (*resolved).clone();
    resolved_with_output.output.directory = output_directory.to_path_buf();
    let materials = BundleMaterials {
        spec_text: spec_text.to_string(),
        resolved: resolved_with_output,
        ember_version: env!("CARGO_PKG_VERSION").to_string(),
        ember_commit: ember::extraction::git_commit().unwrap_or_else(|| "unknown".to_string()),
        model_meta: ModelBundleMeta {
            sha256: model_sha.clone(),
            architecture: architecture.clone(),
            layer_count: n_layers,
            embed_dim,
            vocab_size: model.config.vocab_size,
            gguf_metadata: prepared.gguf_metadata.clone(),
        },
        tokenizer_meta: TokenizerBundleMeta {
            sha256: tokenizer_sha.clone(),
            vocab_size: tokenizer.vocab_size(),
        },
        plan: (*plan).clone(),
        results: results.clone(),
        warnings,
        runtime,
    };
    let (path, identity) =
        write_bundle(&materials, retain_incomplete).map_err(anyhow::Error::msg)?;
    let report = verify_bundle(&path, &VerifyOptions::default()).map_err(anyhow::Error::msg)?;
    Ok((path, identity, report, results))
}

fn peak_rss_kb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                return rest.trim().trim_end_matches(" kB").parse().ok();
            }
        }
    }
    None
}

/// Load the spec file and resolve it, applying CLI overrides.
fn resolve_spec_file(
    spec: &std::path::Path,
    execution: Option<&str>,
    threads: Option<usize>,
) -> anyhow::Result<(String, ember::v05::spec::ExperimentSpecV1)> {
    let spec_text = std::fs::read_to_string(spec)
        .with_context(|| format!("cannot read experiment spec '{}'", spec.display()))?;
    let raw = RawExperimentSpec::from_toml_str(&spec_text)
        .map_err(|error| anyhow::anyhow!("{}", error))?;
    let mut resolved = raw
        .resolve()
        .map_err(|error| anyhow::anyhow!("{}", error))?;
    if let Some(mode) = execution {
        resolved.execution.mode = ExecutionMode::from_cli(mode).map_err(anyhow::Error::msg)?;
    }
    if let Some(threads) = threads {
        resolved.execution.threads = threads;
    }
    Ok((spec_text, resolved))
}

pub(crate) fn run_validate_command(command: &ValidateArgs) -> anyhow::Result<()> {
    let (_, resolved) = resolve_spec_file(&command.spec, None, None)?;
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "schema": EXPERIMENT_SCHEMA_V1,
                "experiment": resolved.experiment.name,
                "execution_mode": resolved.execution.mode.name(),
                "captures": resolved.captures.len(),
                "interventions": resolved.interventions.len(),
                "inputs": resolved.inputs.len(),
                "defaults": resolved.defaults,
            }))?
        );
    } else {
        println!("specification OK");
        println!("  schema: {}", EXPERIMENT_SCHEMA_V1);
        println!("  experiment: {}", resolved.experiment.name);
        println!("  execution mode: {}", resolved.execution.mode.name());
        println!("  inputs: {}", resolved.inputs.len());
        println!("  captures: {}", resolved.captures.len());
        println!("  interventions: {}", resolved.interventions.len());
        println!("  defaults applied: {}", resolved.defaults.len());
    }
    Ok(())
}

pub(crate) fn run_experiment_command(
    command: &RunArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let (spec_text, resolved) =
        resolve_spec_file(&command.spec, command.execution.as_deref(), command.threads)?;
    let output_directory = command
        .output
        .clone()
        .unwrap_or_else(|| resolved.output.directory.clone());
    let (path, identity, report, _results) = execute_resolved(
        &resolved,
        &spec_text,
        &output_directory,
        k_strategy,
        k_allow_fallback,
        command.retain_incomplete,
    )?;
    if !report.ok {
        anyhow::bail!(
            "bundle self-verification failed: {} check(s) failed",
            report.checks.iter().filter(|check| !check.ok).count()
        );
    }
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "bundle": path.display().to_string(),
                "semantic_hash": identity.semantic_hash,
                "payload_hash": identity.payload_hash,
                "verification": report,
            }))?
        );
    } else {
        println!("bundle written to {}", path.display());
        println!("  semantic hash: {}", identity.semantic_hash);
        println!("  payload hash:  {}", identity.payload_hash);
        println!("  verification: {} check(s) passed", report.checks.len());
    }
    Ok(())
}

pub(crate) fn run_inspect_command(command: &InspectArgs) -> anyhow::Result<()> {
    let bundle =
        ember::v05::verify::load_bundle_for_source(&command.bundle).map_err(anyhow::Error::msg)?;
    let manifest = bundle.semantic_manifest;
    let index = &bundle.capture_index;
    let summary = serde_json::json!({
        "bundle": command.bundle.display().to_string(),
        "bundle_schema": manifest.bundle_schema,
        "experiment": manifest.experiment.name,
        "model_sha256": manifest.model.sha256,
        "architecture": manifest.model.architecture,
        "execution_mode": manifest.execution.mode,
        "plan_hash": manifest.execution.plan_hash,
        "inputs": manifest.inputs.iter().map(|input| input.id.clone()).collect::<Vec<_>>(),
        "captures": index.len(),
        "interventions": manifest.interventions.len(),
        "generated_token_ids": manifest.generated.token_ids,
        "payloads": manifest.payloads.keys().collect::<Vec<_>>(),
        "warnings": manifest.warnings,
    });
    if command.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("experiment bundle: {}", command.bundle.display());
        println!("  schema: {}", manifest.bundle_schema);
        println!("  experiment: {}", manifest.experiment.name);
        println!(
            "  model: {} ({})",
            &manifest.model.sha256[..12],
            manifest.model.architecture
        );
        println!(
            "  execution: {} plan {}",
            manifest.execution.mode,
            &manifest.execution.plan_hash[..12]
        );
        println!(
            "  inputs: {:?}",
            manifest
                .inputs
                .iter()
                .map(|i| i.id.clone())
                .collect::<Vec<_>>()
        );
        println!("  captures: {}", index.len());
        for entry in index.iter().take(10) {
            let tensor = if entry.summary.is_some() {
                "summary".to_string()
            } else {
                entry
                    .shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            };
            println!(
                "    {} @ {} layer {}: {} [{}]",
                entry.capture_id, entry.site, entry.layer, tensor, entry.dtype
            );
        }
        if index.len() > 10 {
            println!("    ... {} more", index.len() - 10);
        }
        println!("  interventions: {}", manifest.interventions.len());
        println!("  warnings: {}", manifest.warnings.len());
    }
    Ok(())
}

pub(crate) fn run_verify_command(command: &VerifyArgs) -> anyhow::Result<()> {
    let options = VerifyOptions {
        model_path: command.model.clone(),
        tokenizer_path: command.tokenizer.clone(),
    };
    let report = verify_bundle(&command.bundle, &options).map_err(anyhow::Error::msg)?;
    if command.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("verification of {}", command.bundle.display());
        for check in &report.checks {
            println!(
                "  [{}] {}: {}",
                if check.ok { "ok" } else { "FAIL" },
                check.name,
                check.detail
            );
        }
        println!("verdict: {}", if report.ok { "verified" } else { "FAILED" });
    }
    if !report.ok {
        std::process::exit(1);
    }
    Ok(())
}

pub(crate) fn run_compare_command(command: &CompareArgs) -> anyhow::Result<()> {
    let result = compare_bundles(&command.a, &command.b).map_err(anyhow::Error::msg)?;
    if command.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    let identity = &result.identity;
    println!(
        "comparing {} vs {}",
        command.a.display(),
        command.b.display()
    );
    println!("identity:");
    println!("  schema compatible: {}", yesno(identity.schema_compatible));
    println!(
        "  semantic hash equal: {}",
        yesno(identity.semantic_hash_equal)
    );
    println!("  model hash equal: {}", yesno(identity.model_hash_equal));
    println!(
        "  tokenizer hash equal: {}",
        yesno(identity.tokenizer_hash_equal)
    );
    println!(
        "  execution mode equal: {}",
        yesno(identity.execution_mode_equal)
    );
    println!("  plan hash equal: {}", yesno(identity.plan_hash_equal));
    println!("  input ids equal: {}", yesno(identity.input_ids_equal));
    println!("  prompts equal: {}", yesno(identity.prompts_equal));
    println!(
        "  tokenization equal: {}",
        yesno(identity.tokenization_equal)
    );
    println!("outputs:");
    for output in &result.outputs {
        println!(
            "  {}: tokens {} text {} top1 {} divergence {}",
            output.input_id,
            yesno(output.generated_tokens_equal),
            yesno(output.generated_text_equal),
            yesno(output.final_top1_equal),
            output
                .first_divergence_step
                .map(|step| format!("step {step}"))
                .unwrap_or_else(|| "none".into())
        );
    }
    println!("captures:");
    for capture in &result.captures {
        if let Some(metrics) = &capture.metrics {
            println!(
                "  {} @ {} layer {}: exact {} max-abs {:.2e} mean-abs {:.2e} rel-l2 {:.2e} cosine {:.4}",
                capture.capture_id,
                capture.site,
                capture.layer,
                yesno(metrics.exact),
                metrics.maximum_absolute_difference.unwrap_or(f64::NAN),
                metrics.mean_absolute_difference.unwrap_or(f64::NAN),
                metrics.relative_l2_difference.unwrap_or(f64::NAN),
                metrics.cosine_similarity.unwrap_or(f64::NAN),
            );
        } else {
            println!(
                "  {} @ {} layer {}: present only in {}",
                capture.capture_id,
                capture.site,
                capture.layer,
                if capture.present_in_a { "a" } else { "b" }
            );
        }
    }
    println!("interventions:");
    for intervention in &result.interventions {
        println!(
            "  {}: operation {} source {} tokens {} defusion-route {} ({} vs {} events)",
            intervention.intervention_id,
            yesno(intervention.operation_equal),
            yesno(intervention.source_equal),
            yesno(intervention.selected_tokens_equal),
            yesno(intervention.defusion_route_equal),
            intervention.events_in_a,
            intervention.events_in_b,
        );
    }
    println!("runtime (not semantic):");
    println!(
        "  decode tps: {} vs {}",
        fmt_opt(result.runtime.decode_throughput_tps_a),
        fmt_opt(result.runtime.decode_throughput_tps_b)
    );
    println!(
        "  peak rss kb: {} vs {}",
        fmt_opt_u64(result.runtime.peak_rss_kb_a),
        fmt_opt_u64(result.runtime.peak_rss_kb_b)
    );
    Ok(())
}

fn yesno(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn fmt_opt_u64(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "-".into())
}

pub(crate) fn run_reproduce_command(
    command: &ReproduceArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    // Load the bundle's resolved experiment + verbatim spec.
    let bundle =
        ember::v05::verify::load_bundle_for_source(&command.bundle).map_err(anyhow::Error::msg)?;
    let manifest = bundle.semantic_manifest;
    let spec_text = std::fs::read_to_string(command.bundle.join("experiment.toml"))
        .context("bundle lacks experiment.toml")?;
    let raw = RawExperimentSpec::from_toml_str(&spec_text)
        .map_err(|error| anyhow::anyhow!("{}", error))?;
    let mut resolved = raw
        .resolve()
        .map_err(|error| anyhow::anyhow!("{}", error))?;

    // Validate the supplied model against the bundle's recorded hash.
    let model_sha = sha256_file_result(&command.model)
        .with_context(|| format!("failed to hash '{}'", command.model.display()))?;
    if model_sha != manifest.model.sha256 {
        anyhow::bail!(
            "model '{}' hashes to {} but the bundle records {}; reproduction requires the \
             identical model file",
            command.model.display(),
            model_sha,
            manifest.model.sha256
        );
    }
    resolved.model.path = command.model.clone();
    let output = command
        .output
        .clone()
        .unwrap_or_else(|| command.bundle.with_extension("reproduced"));

    let (path, identity, report, _results) = execute_resolved(
        &resolved,
        &spec_text,
        &output,
        k_strategy,
        k_allow_fallback,
        command.retain_incomplete,
    )?;
    if !report.ok {
        anyhow::bail!("reproduction bundle failed self-verification");
    }

    // Classify against the original.
    let comparison = compare_bundles(&command.bundle, &path).map_err(anyhow::Error::msg)?;
    let tokens_equal = comparison
        .outputs
        .iter()
        .all(|output| output.generated_tokens_equal);
    let captures_exact = comparison
        .captures
        .iter()
        .all(|capture| capture.metrics.as_ref().map(|m| m.exact).unwrap_or(false));
    let captures_within_envelope = comparison.captures.iter().all(|capture| {
        capture
            .metrics
            .as_ref()
            .map(|m| {
                m.maximum_absolute_difference
                    .map(|diff| diff <= 1e-4)
                    .unwrap_or(false)
            })
            .unwrap_or(true)
    });
    let top1_equal = comparison
        .outputs
        .iter()
        .all(|output| output.final_top1_equal);
    let verdict = if comparison.identity.semantic_hash_equal {
        "exact-semantic"
    } else if tokens_equal && captures_exact {
        "exact"
    } else if tokens_equal && captures_within_envelope {
        "output-equivalent"
    } else if top1_equal {
        "top1-equivalent"
    } else {
        "failed"
    };
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "verdict": verdict,
                "original": command.bundle.display().to_string(),
                "reproduction": path.display().to_string(),
                "semantic_hash": identity.semantic_hash,
                "tokens_equal": tokens_equal,
                "captures_exact": captures_exact,
                "captures_within_envelope": captures_within_envelope,
                "top1_equal": top1_equal,
            }))?
        );
    } else {
        println!("reproduction written to {}", path.display());
        println!("  verdict: {verdict}");
        println!(
            "  tokens equal: {}; captures exact: {}; top1 equal: {}",
            yesno(tokens_equal),
            yesno(captures_exact),
            yesno(top1_equal)
        );
        println!("  semantic hash: {}", identity.semantic_hash);
    }
    if verdict == "failed" {
        std::process::exit(2);
    }
    Ok(())
}

pub(crate) fn run_tokenize_command(
    command: &TokenizeArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, k_allow_fallback)?;
    let architecture = crate::cli_support::resolve_generation_architecture(&command.arch, &loader)?;
    let tokenizer_path = command
        .tokenizer
        .clone()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| default_tokenizer_for_arch(&architecture).to_string());
    let resolved_tokenizer = resolve_tokenizer(&tokenizer_path);
    let tokenizer: EmberTokenizer = resolved_tokenizer.load()?;
    let info = tokenize_for_selection(&tokenizer, &command.text, TextNormalization::None)
        .map_err(anyhow::Error::msg)?;
    let selection = match &command.match_span {
        Some(span) => {
            let selector = ember::v05::token_select::TokenSelector::MatchedTextSpan {
                text: span.clone(),
                occurrence: 0,
                subtoken_selection: ember::v05::token_select::SubtokenSelection::All,
                normalization: TextNormalization::None,
            };
            Some(
                ember::v05::token_select::resolve_static_selector(&selector, &info)
                    .map_err(anyhow::Error::msg)?,
            )
        }
        None => None,
    };
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "token_ids": info.token_ids,
                "pieces": info.pieces,
                "byte_offsets": info.byte_offsets,
                "selection": selection,
            }))?
        );
    } else {
        println!("tokenization of {:?}", command.text);
        for (index, ((id, piece), offset)) in info
            .token_ids
            .iter()
            .zip(info.pieces.iter())
            .zip(info.byte_offsets.iter())
            .enumerate()
        {
            println!("  [{index}] id={id:<8} bytes={offset:?} {piece:?}");
        }
        if let Some(selection) = selection {
            println!(
                "match {:?}: span {:?}, selected {:?}, coverage {:?}",
                command.match_span.as_deref().unwrap_or(""),
                selection.matched_byte_span,
                selection.selected_indices,
                selection.coverage
            );
        }
    }
    Ok(())
}
