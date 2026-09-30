//! Ember v0.5 experiment CLI driver: validate, run, inspect, verify,
//! compare, reproduce, tokenize (and `lens`, in `cli_experiment_lens`).
//!
//! The run path loads the model and tokenizer, drives every input through
//! the existing generation machinery with a v0.5 experiment attached, and
//! assembles + self-verifies the deterministic bundle.

use crate::cli_support::{default_tokenizer_for_arch, gguf_metadata_json, resolve_tokenizer};
use anyhow::Context;
use clap::{Args as ClapArgs, Subcommand};
use ember::artifact::ActivationStage;
use ember::experiments::{
    ExecutionContext, ExecutionPhase, Experiment, ExperimentError, ExperimentRunner,
    GenerationContext, LayerContext, ModelContext, ModelFamily, TensorAccess,
};
use ember::extraction::sha256_file_result;
use ember::llama::Llama;
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::plan::{ExecutionMode, HookMode};
use ember::quant_k::KStrategy;
use ember::tokenizer::EmberTokenizer;
use ember::v05::compare::compare_loaded;
use ember::v05::manifest::BundleIdentity;
use ember::v05::run::{
    write_bundle, BundleMaterials, ModelBundleMeta, RuntimeMetrics, TokenizerBundleMeta,
};
use ember::v05::runner::{
    load_bundle_source, BundleSource, InputResult, ModelFacts, V05Experiment,
};
use ember::v05::spec::{RawExperimentSpec, EXPERIMENT_SCHEMA_V1};
use ember::v05::token_select::{tokenize_for_selection, TextNormalization};
use ember::v05::verify::{load_verified_bundle, verify_bundle, LoadedBundle, VerifyOptions};
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
    /// Logit lens: project each captured residual-stream row through the
    /// model's final norm and LM head.
    Lens(crate::cli_experiment_lens::LensArgs),
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
    /// Also run this intervention spec (repeatable), starting its prefill
    /// from the prompt prefix this run computes instead of recomputing it.
    /// Each variant writes its own bundle (its spec's output directory),
    /// bit-identical to running it alone; runtime.json records the path.
    #[arg(long = "variant", value_name = "spec.toml")]
    pub variants: Vec<PathBuf>,
    /// Sign the bundle's manifest.json with this private key (from
    /// `ember evidence init`) and write the signed-evidence-v2 envelope next
    /// to the bundle as `<bundle>.evidence.json`. Defaults to the
    /// `EMBER_SIGN_KEY` environment variable when set.
    #[arg(long, value_name = "key", conflicts_with = "no_sign")]
    pub sign_key: Option<PathBuf>,
    /// Do not sign, even when `EMBER_SIGN_KEY` is set.
    #[arg(long)]
    pub no_sign: bool,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

/// Environment variable naming the default signing key for
/// `experiment run`.
pub(crate) const SIGN_KEY_ENV: &str = "EMBER_SIGN_KEY";

/// Where `experiment run --sign-key` writes a bundle's evidence envelope,
/// and where `--trusted-key` looks for one without `--expect-evidence`:
/// `<bundle>.evidence.json`, a sibling of the bundle directory. It lives
/// outside the bundle because a file inside would change the bundle's
/// inventory (and so fail verification of the very bundle it signs).
pub(crate) fn bundle_evidence_path(bundle: &std::path::Path) -> PathBuf {
    let named = bundle
        .file_name()
        .map(|_| bundle.to_path_buf())
        .or_else(|| bundle.canonicalize().ok())
        .unwrap_or_else(|| bundle.to_path_buf());
    let mut name = named
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "bundle".into());
    name.push(".evidence.json");
    named.with_file_name(name)
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
    #[command(flatten)]
    pub anchor: AnchorArgs,
    /// Also write the JSON report to this path (never inside the bundle;
    /// verification does not modify the bundle it checks).
    #[arg(long, value_name = "report.json")]
    pub write_report: Option<PathBuf>,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

/// External anchors for a bundle's identity. Every hash a bundle carries is
/// written by the bundle's producer, so verification alone shows only that
/// the bundle is self-consistent: an edited bundle can be fully resealed.
/// An anchor compares the recomputed identity with a value from outside.
#[derive(ClapArgs, Clone, Default)]
pub(crate) struct AnchorArgs {
    /// Require this semantic hash (64 hex chars, obtained from a trusted
    /// record such as a paper or a lab notebook).
    #[arg(long, value_name = "hex", value_parser = parse_sha256_hex)]
    pub expect_semantic_hash: Option<String>,
    /// Require a signed evidence envelope over the bundle's manifest.json
    /// (`ember evidence sign --manifest <bundle>/manifest.json`) whose signed
    /// semantic and payload hashes equal the bundle's. `experiment run
    /// --sign-key` writes one as `<bundle>.evidence.json`, the default when
    /// only `--trusted-key` is given.
    #[arg(long, value_name = "envelope.json", requires = "trusted_key")]
    pub expect_evidence: Option<PathBuf>,
    /// The envelope signer's public key (`.pub` file or hex fingerprint).
    /// Without `--expect-evidence` the envelope is looked up next to the
    /// bundle as `<bundle>.evidence.json`.
    #[arg(long, value_name = "key.pub")]
    pub trusted_key: Option<String>,
}

impl AnchorArgs {
    pub(crate) fn is_anchored(&self) -> bool {
        self.expect_semantic_hash.is_some() || self.trusted_key.is_some()
    }

    /// The envelope to check for `bundle` and its trusted key: the explicit
    /// `--expect-evidence`, else the sibling `<bundle>.evidence.json`.
    fn evidence_for(&self, bundle: &std::path::Path) -> Option<(PathBuf, &str)> {
        let trusted_key = self.trusted_key.as_deref()?;
        let envelope = self
            .expect_evidence
            .clone()
            .unwrap_or_else(|| bundle_evidence_path(bundle));
        Some((envelope, trusted_key))
    }
}

#[derive(ClapArgs)]
pub(crate) struct CompareArgs {
    /// First bundle directory.
    pub a: PathBuf,
    /// Second bundle directory.
    pub b: PathBuf,
    /// Require the first bundle to have this semantic hash.
    #[arg(long, value_name = "hex", value_parser = parse_sha256_hex)]
    pub expect_a_semantic_hash: Option<String>,
    /// Require the second bundle to have this semantic hash.
    #[arg(long, value_name = "hex", value_parser = parse_sha256_hex)]
    pub expect_b_semantic_hash: Option<String>,
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
    /// Tokenizer file to re-run with (default: the path the bundle's spec
    /// names). Either way it must match the bundle's recorded SHA-256.
    #[arg(long, value_name = "tokenizer.json")]
    pub tokenizer: Option<PathBuf>,
    /// Anchors for the original bundle, checked before anything runs.
    #[command(flatten)]
    pub anchor: AnchorArgs,
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

    fn uses_activation_site(
        &self,
        stage: ActivationStage,
        layer: Option<usize>,
        phase: ExecutionPhase,
    ) -> bool {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .uses_activation_site(stage, layer, phase)
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
    let threads = pool_threads(resolved)?;

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("failed to build the experiment thread pool")?
        .install(|| {
            let mut prepared = prepare_run(resolved, k_strategy, k_allow_fallback)?;
            execute_prepared(
                &mut prepared,
                resolved,
                spec_text,
                output_directory,
                retain_incomplete,
                None,
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
    /// The model section the session was prepared from; shared execution
    /// admits only specs naming the same model, tokenizer and architecture.
    pub model_spec: ember::v05::spec::ModelSpec,
    /// Contrastive directions already computed in this session, keyed by
    /// everything they depend on (a sweep over alpha computes each once).
    pub direction_cache: Mutex<std::collections::BTreeMap<String, DirectionLayers>>,
}

/// Per-layer direction vectors.
pub(crate) type DirectionLayers = std::collections::BTreeMap<usize, Vec<f32>>;

/// Refuse a spec that names a different model, tokenizer or architecture
/// than the loaded session, or pins hashes the session does not have.
pub(crate) fn ensure_same_session(
    prepared: &PreparedRun,
    resolved: &ember::v05::spec::ExperimentSpecV1,
) -> anyhow::Result<()> {
    let session = &prepared.model_spec;
    let model = &resolved.model;
    if model.path != session.path
        || model.tokenizer != session.tokenizer
        || model.arch != session.arch
    {
        anyhow::bail!(
            "experiment '{}' names a different model, tokenizer or architecture than the \
             loaded session ('{}'); specs run together must share them",
            resolved.experiment.name,
            session.path.display()
        );
    }
    if !model.expected_sha256.is_empty() && model.expected_sha256 != prepared.model_sha {
        anyhow::bail!(
            "experiment '{}' expects model SHA-256 {} but the session's model hashes to {}",
            resolved.experiment.name,
            model.expected_sha256,
            prepared.model_sha
        );
    }
    if !model.tokenizer_expected_sha256.is_empty()
        && model.tokenizer_expected_sha256 != prepared.tokenizer_sha
    {
        anyhow::bail!(
            "experiment '{}' expects tokenizer SHA-256 {} but the session's tokenizer hashes \
             to {}",
            resolved.experiment.name,
            model.tokenizer_expected_sha256,
            prepared.tokenizer_sha
        );
    }
    Ok(())
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
        ember::loader::resolve_generation_architecture(&resolved.model.arch, &loader)?;
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
        model_spec: resolved.model.clone(),
        direction_cache: Mutex::new(std::collections::BTreeMap::new()),
    })
}

/// Execute a resolved experiment against an already-loaded session:
/// build the plan, run every input through generation with the v0.5
/// experiment attached, assemble + write the bundle, and self-verify it.
///
/// `cancel` follows the generation contract in `docs/cancellation.md`: it is
/// checked before prefill and at every decode step of every input, and once
/// more before the bundle is written. A cancelled run returns
/// [`ember::cancel::Cancelled`] and writes nothing -- no bundle and no staging
/// directory -- so there is nothing partial to clean up.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_prepared(
    prepared: &mut PreparedRun,
    resolved: &ember::v05::spec::ExperimentSpecV1,
    spec_text: &str,
    output_directory: &std::path::Path,
    retain_incomplete: bool,
    cancel: Option<&ember::cancel::CancelToken>,
) -> anyhow::Result<(
    PathBuf,
    BundleIdentity,
    ember::v05::verify::VerificationReport,
    Vec<InputResult>,
)> {
    let threads = pool_threads(resolved)?;
    if rayon::current_thread_index().is_some() && rayon::current_num_threads() == threads {
        return execute_prepared_inner(
            prepared,
            resolved,
            spec_text,
            output_directory,
            retain_incomplete,
            cancel,
        );
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("failed to build the prepared experiment thread pool")?
        .install(move || {
            execute_prepared_inner(
                prepared,
                resolved,
                spec_text,
                output_directory,
                retain_incomplete,
                cancel,
            )
        })
}

fn execute_prepared_inner(
    prepared: &PreparedRun,
    resolved: &ember::v05::spec::ExperimentSpecV1,
    spec_text: &str,
    output_directory: &std::path::Path,
    retain_incomplete: bool,
    cancel: Option<&ember::cancel::CancelToken>,
) -> anyhow::Result<(
    PathBuf,
    BundleIdentity,
    ember::v05::verify::VerificationReport,
    Vec<InputResult>,
)> {
    let mut active = activate_spec(prepared, resolved)?;
    let mut results = Vec::new();
    let mut timing = RunTiming::default();
    for index in 0..resolved.inputs.len() {
        let started = std::time::Instant::now();
        let experiment = new_input_experiment(prepared, resolved, Some(&active), index)?;
        let result = run_input(prepared, resolved, &active, &experiment, None, None, cancel)?;
        timing.add(started.elapsed(), &result);
        results.push(result);
    }
    // Analysis workflows run after the inputs, with their own capture and
    // patch passes; their reports become bundle artifacts.
    if let Some(attribution) = &resolved.attribution {
        let started = std::time::Instant::now();
        let files =
            crate::cli_experiment_attribution::run_attribution(prepared, resolved, attribution)?;
        timing.elapsed += started.elapsed();
        active.artifacts.extend(files);
    }
    if let Some(probe) = &resolved.probe {
        let started = std::time::Instant::now();
        let files = crate::cli_experiment_probe::run_probe_bridge(prepared, resolved, probe)?;
        timing.elapsed += started.elapsed();
        active.artifacts.extend(files);
    }
    let outcome = finish_bundle(
        prepared,
        &RunTarget {
            resolved,
            spec_text,
            output_directory,
            retain_incomplete,
        },
        &active,
        results,
        timing,
        None,
        cancel,
    )?;
    Ok((
        outcome.path,
        outcome.identity,
        outcome.report,
        outcome.results,
    ))
}

/// One experiment to execute against a prepared session.
#[derive(Clone, Copy)]
pub(crate) struct RunTarget<'a> {
    pub resolved: &'a ember::v05::spec::ExperimentSpecV1,
    pub spec_text: &'a str,
    pub output_directory: &'a std::path::Path,
    pub retain_incomplete: bool,
}

/// A written, self-verified bundle and the results it holds.
pub(crate) struct RunOutcome {
    pub path: PathBuf,
    pub identity: BundleIdentity,
    pub report: ember::v05::verify::VerificationReport,
    pub results: Vec<InputResult>,
    /// Which shared-prefix path produced it (also in runtime.json).
    pub prefix: Option<ember::v05::prefix::PrefixReuseRecord>,
}

/// Per-spec execution state: the authoritative plan the model now runs and
/// the cross-bundle sources the spec's interventions consume.
pub(crate) struct ActiveSpec {
    pub plan: std::sync::Arc<ember::plan::ExecutionPlan>,
    pub bundle_sources: Vec<BundleSource>,
    /// Directions of `vector-file`/`contrastive` sources, resolved before
    /// execution and written into the bundle as artifacts.
    pub directions: Vec<ember::v05::steering::ResolvedDirection>,
    /// Analysis artifacts (attribution, probe bridge) produced after the
    /// inputs ran, written under `artifacts/`.
    pub artifacts: std::collections::BTreeMap<String, Vec<u8>>,
    pub threads: usize,
}

/// Wall time and generated-token count accumulated over a run's inputs.
#[derive(Default, Clone, Copy)]
pub(crate) struct RunTiming {
    pub elapsed: std::time::Duration,
    pub generated: usize,
}

impl RunTiming {
    pub fn add(&mut self, elapsed: std::time::Duration, result: &InputResult) {
        self.elapsed += elapsed;
        self.generated += result.generated_token_ids.len();
    }
}

/// Build the spec's execution plan (making it the model's active plan) and
/// load its cross-bundle sources and directions.
///
/// Directions are resolved first: a contrastive direction runs the model
/// over its prompts (under a capture-only plan of its own), so the spec's
/// own plan is built afterwards and is the one the model is left running.
pub(crate) fn activate_spec(
    prepared: &PreparedRun,
    resolved: &ember::v05::spec::ExperimentSpecV1,
) -> anyhow::Result<ActiveSpec> {
    let directions = crate::cli_experiment_steering::resolve_directions(prepared, resolved)?;
    let mode = resolved.execution.mode;
    let threads = pool_threads(resolved)?;
    let model = &prepared.model;
    let model_sha = &prepared.model_sha;
    let tokenizer_sha = &prepared.tokenizer_sha;
    let n_layers = prepared.n_layers;

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
    // Planned execution is single-token decode. Record the exact union of
    // generated-step sites used by any input so every runtime decode builds
    // the same authoritative plan; prompt-only hooks stay on generic prefill.
    let stage_keys: Vec<String> = ember::v05::prefix::generated_step_stage_keys(resolved, n_layers)
        .map_err(anyhow::Error::msg)?
        .into_iter()
        .collect();
    let stages: Vec<&str> = stage_keys.iter().map(String::as_str).collect();
    model.set_plan_provenance(
        model_sha.clone(),
        tokenizer_sha.clone(),
        model.config.max_seq_len,
    );
    let plan = model.execution_plan(
        mode,
        hook_mode,
        &stages,
        model.config.max_seq_len,
        Some(model_sha),
        Some(tokenizer_sha),
    )?;
    model.set_execution_mode(mode);

    // -- cross-bundle sources --
    let mut bundle_sources: Vec<BundleSource> = Vec::new();
    for intervention in &resolved.interventions {
        if let Some(source) = &intervention.source
            && let ember::v05::intervention::InterventionSource::CaptureFromBundle { .. } = source
        {
            let loaded =
                load_bundle_source(intervention, source, model_sha, tokenizer_sha, n_layers)
                    .map_err(anyhow::Error::msg)?;
            bundle_sources.push(loaded);
        }
    }
    Ok(ActiveSpec {
        plan,
        bundle_sources,
        directions,
        artifacts: std::collections::BTreeMap::new(),
        threads,
    })
}

/// A fresh experiment for input `index`, with its tokenization and the
/// spec's cross-bundle sources and directions injected (`active` is `None`
/// for an observer, which never intervenes).
pub(crate) fn new_input_experiment(
    prepared: &PreparedRun,
    resolved: &ember::v05::spec::ExperimentSpecV1,
    active: Option<&ActiveSpec>,
    index: usize,
) -> anyhow::Result<Arc<Mutex<V05Experiment>>> {
    let facts = ModelFacts {
        n_layers: prepared.n_layers,
        embed_dim: prepared.embed_dim,
        vocab_size: prepared.model.config.vocab_size,
    };
    let input = resolved
        .inputs
        .get(index)
        .ok_or_else(|| anyhow::anyhow!("input index {index} out of range"))?;
    let mut experiment = V05Experiment::new(
        (*resolved).clone(),
        index,
        facts,
        Some(prepared.model_sha.clone()),
        Some(prepared.tokenizer_sha.clone()),
    );
    let info = tokenize_for_selection(&prepared.tokenizer, &input.text, TextNormalization::None)
        .map_err(anyhow::Error::msg)?;
    experiment.inject_tokenization(info);
    if let Some(active) = active {
        for source in &active.bundle_sources {
            experiment.inject_bundle_source(source.clone());
        }
        for direction in &active.directions {
            for (layer, values) in &direction.layers {
                experiment.inject_direction(&direction.intervention_id, *layer, values.clone());
            }
        }
    }
    Ok(Arc::new(Mutex::new(experiment)))
}

/// Drive one input through generation with `runner_experiment` attached
/// (the input's own experiment, or a shared pass wrapping it) and collect
/// the input's result from `experiment`.
pub(crate) fn run_input(
    prepared: &PreparedRun,
    resolved: &ember::v05::spec::ExperimentSpecV1,
    active: &ActiveSpec,
    experiment: &Arc<Mutex<V05Experiment>>,
    pass: Option<crate::cli_experiment_shared::SharedPass>,
    role: Option<crate::cli_generation::PrefixRole<'_>>,
    cancel: Option<&ember::cancel::CancelToken>,
) -> anyhow::Result<InputResult> {
    let backend = ember::backend::CpuBackend;
    let model = &prepared.model;
    let architecture = &prepared.architecture;
    let index = experiment.lock().expect("v05 experiment lock").input_index;
    let input = &resolved.inputs[index];
    eprintln!(
        "experiment: input {} ({}) tokens={} mode={}",
        index + 1,
        input.id,
        input.text.len(),
        resolved.execution.mode.name()
    );
    let model_context = ModelContext::new(
        family_for_arch(architecture),
        Some(prepared.model_path.to_str().unwrap_or("model.gguf")),
        architecture,
        prepared.n_layers,
        prepared.embed_dim,
    )
    .with_provenance(Some(&prepared.model_sha), Some(&prepared.tokenizer_sha));
    let seed = if resolved.generation.temperature > 0.0 && resolved.experiment.seed != 0 {
        Some(resolved.experiment.seed)
    } else {
        None
    };
    let context_limit = model.max_seq_len(&backend);
    let mut runner = match pass {
        None => ExperimentRunner::new(V05Adapter(Arc::clone(experiment))),
        Some(pass) => ExperimentRunner::new(pass),
    };
    let generated_text = match role {
        None => {
            crate::cli_generation::generate_with_experiment(
                &backend,
                model,
                &mut runner,
                model_context,
                &prepared.tokenizer,
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
                active.threads,
                context_limit,
                seed,
                // The CLI passes no token (experiment runs are not signal-
                // cancellable yet); the native console passes its Cancel token.
                cancel,
            )?
        }
        Some(role) => crate::cli_generation::generate_with_experiment_prefix(
            &backend,
            model,
            &mut runner,
            model_context,
            &prepared.tokenizer,
            &input.text,
            resolved.generation.max_new_tokens,
            resolved.generation.temperature,
            active.threads,
            context_limit,
            seed,
            role,
            cancel,
        )?,
    };
    let mut experiment = experiment.lock().expect("v05 experiment lock");
    experiment.set_generated_text(generated_text);
    experiment.into_result().map_err(anyhow::Error::msg)
}

/// Assemble, write, and self-verify a bundle from finished input results.
pub(crate) fn finish_bundle(
    prepared: &PreparedRun,
    target: &RunTarget<'_>,
    active: &ActiveSpec,
    results: Vec<InputResult>,
    timing: RunTiming,
    prefix: Option<ember::v05::prefix::PrefixReuseRecord>,
    cancel: Option<&ember::cancel::CancelToken>,
) -> anyhow::Result<RunOutcome> {
    // Last check point: a cancel that lands after the final decode step must
    // still leave no bundle behind.
    if cancel.is_some_and(ember::cancel::CancelToken::is_cancelled) {
        return Err(anyhow::Error::new(ember::cancel::Cancelled));
    }
    let resolved = target.resolved;
    let wall_clock_ms = timing.elapsed.as_secs_f64() * 1000.0;
    let runtime = RuntimeMetrics {
        wall_clock_ms,
        decode_throughput_tps: if wall_clock_ms > 0.0 {
            Some(timing.generated as f64 / (wall_clock_ms / 1000.0))
        } else {
            None
        },
        prefill_throughput_tps: None,
        first_token_latency_ms: None,
        peak_rss_kb: peak_rss_kb(),
        threads: active.threads,
        prefix_reuse: prefix.as_ref().map(|record| record.to_json()),
    };
    let mut resolved_with_output = (*resolved).clone();
    resolved_with_output.output.directory = target.output_directory.to_path_buf();
    let mut artifacts =
        ember::v05::steering::direction_artifact_files(&active.directions, &resolved.interventions)
            .map_err(anyhow::Error::msg)?;
    artifacts.extend(active.artifacts.clone());
    let materials = BundleMaterials {
        spec_text: target.spec_text.to_string(),
        resolved: resolved_with_output,
        ember_version: env!("CARGO_PKG_VERSION").to_string(),
        ember_commit: ember::extraction::git_commit().unwrap_or_else(|| "unknown".to_string()),
        model_meta: ModelBundleMeta {
            sha256: prepared.model_sha.clone(),
            architecture: prepared.architecture.clone(),
            layer_count: prepared.n_layers,
            embed_dim: prepared.embed_dim,
            vocab_size: prepared.model.config.vocab_size,
            gguf_metadata: prepared.gguf_metadata.clone(),
        },
        tokenizer_meta: TokenizerBundleMeta {
            sha256: prepared.tokenizer_sha.clone(),
            vocab_size: prepared.tokenizer.vocab_size(),
        },
        plan: (*active.plan).clone(),
        results: results.clone(),
        warnings: Vec::new(),
        runtime,
        artifacts,
    };
    let (path, identity) =
        write_bundle(&materials, target.retain_incomplete).map_err(anyhow::Error::msg)?;
    let report = verify_bundle(&path, &VerifyOptions::default()).map_err(anyhow::Error::msg)?;
    Ok(RunOutcome {
        path,
        identity,
        report,
        results,
        prefix,
    })
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
/// The spec text when the file declares `[sweep]`, otherwise `None`.
fn read_sweep_spec(spec: &std::path::Path) -> anyhow::Result<Option<String>> {
    let text = std::fs::read_to_string(spec)
        .with_context(|| format!("cannot read experiment spec '{}'", spec.display()))?;
    Ok(crate::cli_experiment_sweep::is_sweep_spec(&text).then_some(text))
}

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
    if let Some(text) = read_sweep_spec(&command.spec)? {
        return crate::cli_experiment_sweep::run_validate_sweep(command, &text);
    }
    let (_, resolved) = resolve_spec_file(&command.spec, None, None)?;
    let direction_files = crate::cli_experiment_steering::check_direction_files(&resolved)?;
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
                "direction_files_checked": direction_files,
                "attribution": resolved.attribution.is_some(),
                "probe": resolved.probe.is_some(),
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
        if direction_files > 0 {
            println!("  direction files: {direction_files} (hash and shape checked)");
        }
        if let Some(attribution) = &resolved.attribution {
            println!(
                "  attribution: {} -> {}, {} site(s), verify top {}",
                attribution.clean,
                attribution.corrupted,
                attribution.sites.len(),
                attribution.verify_top_k
            );
        }
        if let Some(probe) = &resolved.probe {
            println!(
                "  probe bridge: {} layer {}, {} train / {} test examples, {} variant(s)",
                probe.site,
                probe.layer,
                probe.train.len(),
                probe.test.len(),
                probe.variants().len()
            );
            if let Some(file) = &probe.file {
                ember::v05::steering::read_direction_file(
                    &file.path,
                    &file.sha256,
                    file.tensor.as_deref(),
                )
                .map_err(|error| anyhow::anyhow!("probe.file: {error}"))?;
                println!("  probe file: hash and shape checked");
            }
        }
        println!("  defaults applied: {}", resolved.defaults.len());
    }
    Ok(())
}

pub(crate) fn run_experiment_command(
    command: &RunArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    if let Some(text) = read_sweep_spec(&command.spec)? {
        return crate::cli_experiment_sweep::run_sweep(
            command,
            &text,
            k_strategy,
            k_allow_fallback,
        );
    }
    let (spec_text, resolved) =
        resolve_spec_file(&command.spec, command.execution.as_deref(), command.threads)?;
    let output_directory = command
        .output
        .clone()
        .unwrap_or_else(|| resolved.output.directory.clone());
    if !command.variants.is_empty() {
        return run_with_variants(
            command,
            &spec_text,
            &resolved,
            &output_directory,
            k_strategy,
            k_allow_fallback,
        );
    }
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
    if !command.json {
        crate::cli_experiment_attribution::print_bundle_reports(&path)?;
    }
    let sign_key = resolve_sign_key(command);
    let evidence = match &sign_key {
        Some(key) => Some(sign_bundle(&path, key)?),
        None => None,
    };
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "bundle": path.display().to_string(),
                "semantic_hash": identity.semantic_hash,
                "payload_hash": identity.payload_hash,
                "evidence": evidence.as_ref().map(|(envelope, signer)| serde_json::json!({
                    "path": envelope.display().to_string(),
                    "signer_fingerprint": signer,
                })),
                "verification": report,
            }))?
        );
    } else {
        println!("bundle written to {}", path.display());
        println!("  semantic hash: {}", identity.semantic_hash);
        println!("  payload hash:  {}", identity.payload_hash);
        println!("  verification: {} check(s) passed", report.checks.len());
        if let Some((envelope, signer)) = &evidence {
            println!(
                "  signed evidence: {} (signer {signer})",
                envelope.display()
            );
            println!(
                "    check with: ember experiment verify {} --trusted-key <key.pub>",
                path.display()
            );
        }
    }
    Ok(())
}

/// `experiment run base.toml --variant a.toml --variant b.toml`: one model
/// load, the base computes the shared prefix, each variant resumes from it.
fn run_with_variants(
    command: &RunArgs,
    spec_text: &str,
    resolved: &ember::v05::spec::ExperimentSpecV1,
    output_directory: &std::path::Path,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let mut variants = Vec::with_capacity(command.variants.len());
    for path in &command.variants {
        variants.push(resolve_spec_file(
            path,
            command.execution.as_deref(),
            command.threads,
        )?);
    }
    let mut prepared = prepare_run(resolved, k_strategy, k_allow_fallback)?;
    let targets: Vec<RunTarget<'_>> = variants
        .iter()
        .map(|(text, spec)| RunTarget {
            resolved: spec,
            spec_text: text,
            output_directory: &spec.output.directory,
            retain_incomplete: command.retain_incomplete,
        })
        .collect();
    let (base, _, outcomes) = crate::cli_experiment_shared::execute_shared(
        &mut prepared,
        RunTarget {
            resolved,
            spec_text,
            output_directory,
            retain_incomplete: command.retain_incomplete,
        },
        &[],
        &targets,
        None,
    )?;
    let all: Vec<&RunOutcome> = std::iter::once(&base).chain(outcomes.iter()).collect();
    for outcome in &all {
        if !outcome.report.ok {
            anyhow::bail!(
                "bundle {} failed self-verification: {} check(s) failed",
                outcome.path.display(),
                outcome
                    .report
                    .checks
                    .iter()
                    .filter(|check| !check.ok)
                    .count()
            );
        }
    }
    let mut evidence = Vec::new();
    if let Some(key) = resolve_sign_key(command) {
        for outcome in &all {
            evidence.push(Some(sign_bundle(&outcome.path, &key)?));
        }
    } else {
        evidence.resize(all.len(), None);
    }
    if command.json {
        let bundles: Vec<serde_json::Value> = all
            .iter()
            .zip(&evidence)
            .map(|(outcome, evidence)| {
                serde_json::json!({
                    "bundle": outcome.path.display().to_string(),
                    "semantic_hash": outcome.identity.semantic_hash,
                    "payload_hash": outcome.identity.payload_hash,
                    "prefix_reuse": outcome.prefix.as_ref().map(|record| record.to_json()),
                    "evidence": evidence.as_ref().map(|(envelope, signer)| serde_json::json!({
                        "path": envelope.display().to_string(),
                        "signer_fingerprint": signer,
                    })),
                    "verification": outcome.report,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"ok": true, "bundles": bundles}))?
        );
    } else {
        for (outcome, evidence) in all.into_iter().zip(&evidence) {
            println!("bundle written to {}", outcome.path.display());
            if let Some((envelope, signer)) = evidence {
                println!(
                    "  signed evidence: {} (signer {signer})",
                    envelope.display()
                );
            }
            println!("  semantic hash: {}", outcome.identity.semantic_hash);
            println!("  payload hash:  {}", outcome.identity.payload_hash);
            println!(
                "  verification: {} check(s) passed",
                outcome.report.checks.len()
            );
            if let Some(record) = &outcome.prefix {
                println!("  prefix: {}", describe_prefix(record));
            }
        }
    }
    Ok(())
}

/// One line on how a bundle was computed.
pub(crate) fn describe_prefix(record: &ember::v05::prefix::PrefixReuseRecord) -> String {
    use ember::v05::prefix::PrefixPath;
    if record.inputs.is_empty() {
        return format!(
            "{}{}",
            record.role,
            record
                .note
                .as_ref()
                .map(|note| format!(" ({note})"))
                .unwrap_or_default()
        );
    }
    let parts: Vec<String> = record
        .inputs
        .iter()
        .map(|input| match &input.path {
            PrefixPath::Resumed { resume_layer } => {
                format!("{} resumed at block {resume_layer}", input.input_id)
            }
            PrefixPath::FullRecompute { reason } => {
                format!("{} recomputed in full ({reason})", input.input_id)
            }
        })
        .collect();
    format!("{}: {}", record.role, parts.join("; "))
}

/// The signing key a run uses: `--sign-key`, else `EMBER_SIGN_KEY`, unless
/// `--no-sign`.
pub(crate) fn resolve_sign_key(command: &RunArgs) -> Option<PathBuf> {
    if command.no_sign {
        None
    } else {
        command.sign_key.clone().or_else(|| {
            std::env::var_os(SIGN_KEY_ENV)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
    }
}

/// Sign `<bundle>/manifest.json` into `<bundle>.evidence.json`. Returns the
/// envelope path and the signer fingerprint.
pub(crate) fn sign_bundle(
    bundle: &std::path::Path,
    key: &std::path::Path,
) -> anyhow::Result<(PathBuf, String)> {
    let envelope_path = bundle_evidence_path(bundle);
    let envelope = crate::cli_evidence::sign_record_file(
        &bundle.join("manifest.json"),
        &key.to_string_lossy(),
        &envelope_path,
    )
    .with_context(|| format!("failed to sign bundle '{}'", bundle.display()))?;
    let signer = envelope["signer_fingerprint"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    Ok((envelope_path, signer))
}

pub(crate) fn run_inspect_command(command: &InspectArgs) -> anyhow::Result<()> {
    if ember::v05::sweep::is_sweep_dir(&command.bundle) {
        return crate::cli_experiment_sweep::run_inspect_sweep(&command.bundle, command.json);
    }
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
            manifest
                .model
                .sha256
                .get(..12)
                .unwrap_or(&manifest.model.sha256),
            manifest.model.architecture
        );
        println!(
            "  execution: {} plan {}",
            manifest.execution.mode,
            manifest
                .execution
                .plan_hash
                .get(..12)
                .unwrap_or(&manifest.execution.plan_hash)
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

/// Clap parser for an expected SHA-256: 64 hex characters, lowercased.
fn parse_sha256_hex(value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(value.to_ascii_lowercase())
    } else {
        Err("expected a SHA-256 as 64 hex characters".into())
    }
}

/// Check a signed evidence envelope over a bundle's `manifest.json` against
/// the bundle's recomputed identity. Returns (ok, detail).
fn evidence_anchor(
    envelope: &std::path::Path,
    trusted_key: &str,
    semantic_hash: &str,
    payload_hash: &str,
) -> (bool, String) {
    let verified =
        match crate::cli_evidence::verify_envelope_file_with_trusted_key(envelope, trusted_key) {
            Ok(verified) => verified,
            Err(error) => return (false, format!("{error:#}")),
        };
    let input = &verified.input;
    let field = |name: &str| input.get(name).and_then(|value| value.as_str());
    if field("kind") != Some(ember::v05::manifest::BUNDLE_KIND) {
        return (
            false,
            "the signed record is not an experiment bundle manifest.json".into(),
        );
    }
    let semantic_ok = field("semantic_hash") == Some(semantic_hash);
    let payload_ok = field("payload_hash") == Some(payload_hash);
    let signer = verified
        .signer_fingerprint
        .get(..12)
        .unwrap_or(&verified.signer_fingerprint);
    let timestamp = if verified.timestamp_signed {
        ""
    } else {
        "; its signing time is not covered by the v1 signature"
    };
    if semantic_ok && payload_ok {
        (
            true,
            format!(
                "{} from trusted signer {signer} binds this semantic and payload hash{timestamp}",
                verified.schema
            ),
        )
    } else {
        (
            false,
            format!(
                "signed record (trusted signer {signer}) names semantic {} and payload {}, not \
                 this bundle's (semantic {}, payload {})",
                field("semantic_hash").unwrap_or("-"),
                field("payload_hash").unwrap_or("-"),
                semantic_ok,
                payload_ok
            ),
        )
    }
}

/// Load and verify a bundle with its anchors; fails on any failed check.
pub(crate) fn load_anchored_bundle(
    bundle: &std::path::Path,
    anchor: &AnchorArgs,
) -> anyhow::Result<LoadedBundle> {
    let options = VerifyOptions {
        expected_semantic_hash: anchor.expect_semantic_hash.clone(),
        ..VerifyOptions::default()
    };
    let loaded = load_verified_bundle(bundle, &options).map_err(anyhow::Error::msg)?;
    if let Some((envelope, trusted_key)) = anchor.evidence_for(bundle) {
        let (ok, detail) = evidence_anchor(
            &envelope,
            trusted_key,
            &loaded.semantic_hash,
            &loaded.payload_hash,
        );
        anyhow::ensure!(
            ok,
            "bundle '{}' failed its evidence anchor: {detail}",
            bundle.display()
        );
    }
    Ok(loaded)
}

/// Refuse a report path inside the bundle: verification must not add files
/// to the bundle it checks.
fn write_report_outside(
    bundle: &std::path::Path,
    path: &std::path::Path,
    report: &ember::v05::verify::VerificationReport,
) -> anyhow::Result<()> {
    let bundle_root = bundle
        .canonicalize()
        .with_context(|| format!("cannot resolve '{}'", bundle.display()))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let parent = parent
        .canonicalize()
        .with_context(|| format!("cannot resolve '{}'", parent.display()))?;
    anyhow::ensure!(
        !parent.starts_with(&bundle_root),
        "--write-report must point outside the bundle; verification never modifies the bundle"
    );
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    ember::atomic_file::atomic_write(path, &bytes)
        .with_context(|| format!("cannot write report '{}'", path.display()))
}

pub(crate) fn run_verify_command(command: &VerifyArgs) -> anyhow::Result<()> {
    let options = VerifyOptions {
        model_path: command.model.clone(),
        tokenizer_path: command.tokenizer.clone(),
        expected_semantic_hash: command.anchor.expect_semantic_hash.clone(),
    };
    if ember::v05::sweep::is_sweep_dir(&command.bundle) {
        return crate::cli_experiment_sweep::run_verify_sweep(
            &command.bundle,
            &options,
            command.anchor.expect_evidence.is_some(),
            command.write_report.as_deref(),
            command.json,
        );
    }
    let mut report = verify_bundle(&command.bundle, &options).map_err(anyhow::Error::msg)?;
    if let Some((envelope, trusted_key)) = command.anchor.evidence_for(&command.bundle) {
        let (ok, detail) = if report.semantic_hash.is_empty() {
            (
                false,
                "the bundle's identity could not be recomputed".to_string(),
            )
        } else {
            evidence_anchor(
                &envelope,
                trusted_key,
                &report.semantic_hash,
                &report.payload_hash,
            )
        };
        report.add_check("evidence anchor", ok, detail);
    }
    let anchored = command.anchor.is_anchored();
    if !anchored {
        report.warnings.push(
            "not anchored: 'verified' means the bundle is self-consistent. Every hash it checks \
             is written by the bundle's producer, so an edited bundle can be resealed; pass \
             --expect-semantic-hash or --expect-evidence/--trusted-key to bind it to an identity \
             obtained elsewhere"
                .into(),
        );
    }
    if let Some(path) = &command.write_report {
        write_report_outside(&command.bundle, path, &report)?;
    }
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
        if report.ok {
            println!();
            println!("  semantic hash: {}", report.semantic_hash);
            println!("  payload hash:  {}", report.payload_hash);
            if anchored {
                println!("  anchored: the identity matches the externally supplied value");
            } else {
                println!(
                    "  note: 'verified' means self-consistent only; every hash checked is \
                     bundle-authored."
                );
                println!(
                    "        Record the semantic hash somewhere you trust and pass \
                     --expect-semantic-hash <hex>"
                );
                println!(
                    "        (or --expect-evidence <envelope> --trusted-key <key.pub>) to \
                     detect a resealed bundle."
                );
            }
        }
        for warning in report
            .warnings
            .iter()
            .filter(|warning| !warning.starts_with("not anchored"))
        {
            println!("  warning: {warning}");
        }
    }
    if !report.ok {
        return Err(crate::cli_support::VerificationFailed.into());
    }
    Ok(())
}

pub(crate) fn run_compare_command(command: &CompareArgs) -> anyhow::Result<()> {
    if ember::v05::sweep::is_sweep_dir(&command.a) || ember::v05::sweep::is_sweep_dir(&command.b) {
        return crate::cli_experiment_sweep::run_compare_sweeps(
            &command.a,
            &command.b,
            command.expect_a_semantic_hash.as_deref(),
            command.expect_b_semantic_hash.as_deref(),
            command.json,
        );
    }
    let anchor = |expected: &Option<String>| AnchorArgs {
        expect_semantic_hash: expected.clone(),
        ..AnchorArgs::default()
    };
    let bundle_a = load_anchored_bundle(&command.a, &anchor(&command.expect_a_semantic_hash))?;
    let bundle_b = load_anchored_bundle(&command.b, &anchor(&command.expect_b_semantic_hash))?;
    let result = compare_loaded(&bundle_a, &bundle_b).map_err(anyhow::Error::msg)?;
    let host = host_differences(&bundle_a, &bundle_b);
    if command.json {
        let mut value = serde_json::to_value(&result)?;
        value["host_differences"] = serde_json::to_value(&host)?;
        println!("{}", serde_json::to_string_pretty(&value)?);
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
    if results_differ(&result) {
        println!("host (why the numbers may differ):");
        for line in ember::v05::host_profile::report_lines(&host) {
            println!("{line}");
        }
    } else if !host.differences.is_empty() {
        println!(
            "host: {} difference(s), none of which changed the results",
            host.differences.len()
        );
    }
    Ok(())
}

/// Host differences between two bundles, from their runtime.json files.
fn host_differences(
    a: &LoadedBundle,
    b: &LoadedBundle,
) -> ember::v05::host_profile::HostDifferenceReport {
    let runtime = |bundle: &LoadedBundle| {
        bundle
            .file("runtime.json")
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(bytes).ok())
    };
    ember::v05::host_profile::explain_host_differences(runtime(a).as_ref(), runtime(b).as_ref())
}

/// Whether any output or capture differs between the compared bundles.
fn results_differ(result: &ember::v05::compare::CompareResult) -> bool {
    result
        .outputs
        .iter()
        .any(|output| !output.generated_tokens_equal || !output.generated_text_equal)
        || result.captures.iter().any(|capture| {
            capture
                .metrics
                .as_ref()
                .map(|metrics| !metrics.exact)
                .unwrap_or(true)
        })
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

/// Worker threads for a resolved spec: its explicit count, or every core.
///
/// Specs also arrive from bundles (`reproduce`), so the count is untrusted
/// and bounded before a pool is built from it.
pub(crate) fn pool_threads(resolved: &ember::v05::spec::ExperimentSpecV1) -> anyhow::Result<usize> {
    const MAX_THREADS: usize = 1024;
    let requested = resolved.execution.threads;
    anyhow::ensure!(
        requested <= MAX_THREADS,
        "execution.threads = {requested} exceeds the limit of {MAX_THREADS}"
    );
    Ok(if requested > 0 {
        requested
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

pub(crate) fn run_reproduce_command(
    command: &ReproduceArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    // Verify (and anchor) the original once; everything below uses the
    // verified bytes, never a second read of the bundle.
    let original = load_anchored_bundle(&command.bundle, &command.anchor)?;
    let manifest = &original.semantic_manifest;
    let spec_text = std::str::from_utf8(
        original
            .required_file("experiment.toml")
            .map_err(anyhow::Error::msg)?,
    )
    .context("bundle experiment.toml is not UTF-8")?
    .to_string();
    // resolved-experiment.json is outside the semantic hash; it is used
    // only after it has been bound to the hashed spec and manifest.
    let mut resolved =
        ember::v05::verify::bound_resolved_experiment(&original).map_err(anyhow::Error::msg)?;

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
    resolved.model.expected_sha256 = manifest.model.sha256.clone();
    // The tokenizer path comes from the bundle; whatever file it (or
    // --tokenizer) names must be the tokenizer the bundle recorded.
    if let Some(tokenizer) = &command.tokenizer {
        resolved.model.tokenizer = Some(tokenizer.clone());
    }
    resolved.model.tokenizer_expected_sha256 = manifest.tokenizer.sha256.clone();
    // Nothing from the bundle may widen what this command does to the
    // filesystem: replacing an existing directory needs the user's say-so,
    // not the bundle's.
    resolved.output.overwrite = false;
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

    // Classify against the original, as verified above.
    let reproduction =
        ember::v05::verify::load_bundle_for_source(&path).map_err(anyhow::Error::msg)?;
    let comparison = compare_loaded(&original, &reproduction).map_err(anyhow::Error::msg)?;
    let tokens_equal = comparison
        .outputs
        .iter()
        .all(|output| output.generated_tokens_equal);
    let captures_declared = !comparison.captures.is_empty();
    let captures_aligned = captures_declared
        && comparison
            .captures
            .iter()
            .all(|capture| capture.present_in_a && capture.present_in_b);
    let captures_exact = captures_aligned
        && comparison
            .captures
            .iter()
            .all(|capture| capture.metrics.as_ref().map(|m| m.exact).unwrap_or(false));
    let captures_within_envelope = captures_aligned
        && comparison.captures.iter().all(|capture| {
            capture
                .metrics
                .as_ref()
                .and_then(|m| m.maximum_absolute_difference)
                .map(|diff| diff <= 1e-4)
                .unwrap_or(false)
        });
    let captures_misaligned = captures_declared && !captures_aligned;
    let top1_equal = comparison
        .outputs
        .iter()
        .all(|output| output.final_top1_equal);
    // Output agreement means nothing unless both ran the same inputs: outputs
    // are paired by position, so a reproduction over a subset or different
    // prompts could otherwise still grade as exact.
    let inputs_equal = comparison.identity.input_ids_equal && comparison.identity.prompts_equal;
    let verdict = if comparison.identity.semantic_hash_equal {
        "exact-semantic"
    } else if !inputs_equal {
        "inputs-differ"
    } else if tokens_equal && (captures_exact || !captures_declared) {
        "exact"
    } else if tokens_equal && (!captures_declared || captures_within_envelope) {
        "output-equivalent"
    } else if captures_misaligned {
        "captures-misaligned"
    } else if top1_equal {
        "top1-equivalent"
    } else {
        "failed"
    };
    let host = host_differences(&original, &reproduction);
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "host_differences": host,
                "verdict": verdict,
                "original": command.bundle.display().to_string(),
                "reproduction": path.display().to_string(),
                "original_semantic_hash": original.semantic_hash,
                "original_anchored": command.anchor.is_anchored(),
                "semantic_hash": identity.semantic_hash,
                "inputs_equal": inputs_equal,
                "tokens_equal": tokens_equal,
                "captures_declared": captures_declared,
                "captures_aligned": captures_aligned,
                "captures_exact": captures_exact,
                "captures_within_envelope": captures_within_envelope,
                "top1_equal": top1_equal,
            }))?
        );
    } else {
        println!("reproduction written to {}", path.display());
        println!("  verdict: {verdict}");
        println!(
            "  tokens equal: {}; captures exact: {}; captures aligned: {}; top1 equal: {}",
            yesno(tokens_equal),
            yesno(captures_exact),
            yesno(captures_aligned),
            yesno(top1_equal)
        );
        println!("  semantic hash: {}", identity.semantic_hash);
        println!(
            "  original semantic hash: {} ({})",
            original.semantic_hash,
            if command.anchor.is_anchored() {
                "anchored"
            } else {
                "not anchored: self-consistent only"
            }
        );
        if !matches!(verdict, "exact-semantic" | "exact") {
            println!("  host (why the numbers may differ):");
            for line in ember::v05::host_profile::report_lines(&host) {
                println!("  {line}");
            }
        }
    }
    if verdict == "failed" || verdict == "captures-misaligned" {
        return Err(crate::cli_support::VerificationFailed.into());
    }
    Ok(())
}

pub(crate) fn run_tokenize_command(
    command: &TokenizeArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, k_allow_fallback)?;
    let architecture = ember::loader::resolve_generation_architecture(&command.arch, &loader)?;
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

#[cfg(test)]
mod tests {
    use super::bundle_evidence_path;
    use std::path::{Path, PathBuf};

    #[test]
    fn evidence_envelope_sits_next_to_the_bundle_not_inside_it() {
        assert_eq!(
            bundle_evidence_path(Path::new("runs/probe")),
            PathBuf::from("runs/probe.evidence.json")
        );
        assert_eq!(
            bundle_evidence_path(Path::new("runs/probe/")),
            PathBuf::from("runs/probe.evidence.json")
        );
        assert_eq!(
            bundle_evidence_path(Path::new("probe.v1")),
            PathBuf::from("probe.v1.evidence.json")
        );
    }
}
