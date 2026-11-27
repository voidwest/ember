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
use ember::extraction::{sha256_file_result, sha256_file_result_unless};
use ember::llama::Llama;
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::plan::{ExecutionMode, HookMode};
use ember::quant_k::KStrategy;
use ember::tokenizer::EmberTokenizer;
use ember::v05::compare::compare_loaded;
use ember::v05::manifest::BundleIdentity;
use ember::v05::run::{BundleMaterials, ModelBundleMeta, RuntimeMetrics, TokenizerBundleMeta};
use ember::v05::runner::{
    load_bundle_source, BundleSource, InputResult, ModelFacts, V05Experiment,
};
use ember::v05::spec::{RawExperimentSpec, EXPERIMENT_SCHEMA_V1};
use ember::v05::token_select::{tokenize_for_selection, TextNormalization};
use ember::v05::verify::{load_verified_bundle, verify_bundle, LoadedBundle, VerifyOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

mod args;
mod commands;
pub(crate) use args::*;
pub(crate) use commands::*;

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

    fn before_o_proj(
        &mut self,
        ctx: &LayerContext<'_>,
        heads: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.0
            .lock()
            .expect("v05 experiment lock")
            .before_o_proj(ctx, heads)
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
#[allow(
    clippy::too_many_arguments,
    reason = "application call path: each argument is a separate input"
)]
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
    in_session_pool(pool_threads(resolved)?, || {
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
    /// `tokenizer.vocab_size()`, counted once: the tokenizers crate builds
    /// the whole vocabulary map to answer it, and every bundle records it.
    pub tokenizer_vocab_size: usize,
    pub gguf_metadata: serde_json::Value,
    pub model_path: PathBuf,
    /// The model section the session was prepared from; shared execution
    /// admits only specs naming the same model, tokenizer and architecture.
    pub model_spec: ember::v05::spec::ModelSpec,
    /// Contrastive directions already computed in this session, keyed by
    /// everything they depend on (a sweep over alpha computes each once).
    pub direction_cache: Mutex<std::collections::BTreeMap<String, DirectionLayers>>,
    /// Dictionaries already loaded in this session, keyed by folder and
    /// both pinned hashes (a sweep loads each once).
    pub dictionary_cache: Mutex<DictionaryCache>,
}

/// Loaded dictionaries by (folder, weights SHA-256, cfg.json SHA-256): an
/// entry that pins another config hash loads, and is checked, on its own.
pub(crate) type DictionaryCache =
    std::collections::BTreeMap<(PathBuf, String, String), Arc<ember::v05::dictionary::Dictionary>>;

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
    // Hashing the whole GGUF (~0.7 s for 1.3 GB) and reading, hashing and
    // parsing the tokenizer do not depend on building the model, so they run
    // on helper threads while it loads. Results are checked in the order
    // they were when everything ran in sequence, with the same errors, and
    // the tokenizer is parsed only after its bytes match the spec's pin.
    let model_file = ModelFileIdentity::of(&resolved.model.path)?;
    let abandon = AtomicBool::new(false);
    let prepared = std::thread::scope(|scope| {
        // After an early error (or a panic) the helpers' results go unused;
        // stop them instead of waiting out a whole-model hash when the scope
        // joins. A drop guard covers the unwinding path too.
        let _abandon_on_exit = AbandonOnDrop(&abandon);
        let model_hash = scope.spawn(|| sha256_file_result_unless(&resolved.model.path, &abandon));
        prepare_run_overlapped(
            resolved,
            k_strategy,
            k_allow_fallback,
            scope,
            model_hash,
            &abandon,
        )
    })?;
    model_file.ensure_unchanged(&resolved.model.path)?;
    Ok(prepared)
}

/// What identifies a model file on disk. The SHA-256 and the weights come
/// from separate opens (the hash runs beside the load); a file replaced or
/// rewritten in between would otherwise be used under the recorded hash.
///
/// On Unix this includes the inode change time, which the kernel updates on
/// every write, rename or chmod and an unprivileged process cannot set back
/// (unlike the modification time), so an in-place rewrite that restores
/// size and mtime is still caught. It does not cover a writer that keeps
/// modifying the file after loading: the weights are memory-mapped.
pub(crate) struct ModelFileIdentity {
    len: u64,
    modified: Option<std::time::SystemTime>,
    #[cfg(unix)]
    inode: (u64, u64),
    #[cfg(unix)]
    changed: (i64, i64),
}

impl ModelFileIdentity {
    pub(crate) fn of(path: &std::path::Path) -> anyhow::Result<Self> {
        let metadata = std::fs::metadata(path)
            .with_context(|| format!("cannot stat model '{}'", path.display()))?;
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            inode: {
                use std::os::unix::fs::MetadataExt as _;
                (metadata.dev(), metadata.ino())
            },
            #[cfg(unix)]
            changed: {
                use std::os::unix::fs::MetadataExt as _;
                (metadata.ctime(), metadata.ctime_nsec())
            },
        })
    }

    pub(crate) fn ensure_unchanged(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let now = Self::of(path)?;
        let same = now.len == self.len && now.modified == self.modified;
        #[cfg(unix)]
        let same = same && now.inode == self.inode && now.changed == self.changed;
        anyhow::ensure!(
            same,
            "model '{}' changed while it was being hashed and loaded; its recorded SHA-256 \
             may not describe the weights in use",
            path.display()
        );
        Ok(())
    }
}

/// Sets the flag when dropped: tells `prepare_run`'s helper threads that
/// their results will not be used.
struct AbandonOnDrop<'a>(&'a AtomicBool);

impl Drop for AbandonOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Join a helper thread, re-raising its panic as one would have been raised
/// had its work run on the calling thread.
fn join_helper<T>(handle: std::thread::ScopedJoinHandle<'_, T>) -> T {
    handle
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
}

fn prepare_run_overlapped<'scope, 'env>(
    resolved: &'env ember::v05::spec::ExperimentSpecV1,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
    scope: &'scope std::thread::Scope<'scope, 'env>,
    model_hash: std::thread::ScopedJoinHandle<'scope, anyhow::Result<String>>,
    abandon: &'env AtomicBool,
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
    let tokenizer_path = resolved
        .model
        .tokenizer
        .clone()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| default_tokenizer_for_arch(&architecture).to_string());
    let resolved_tokenizer = resolve_tokenizer(&tokenizer_path);
    let tokenizer_pin = resolved.model.tokenizer_expected_sha256.as_str();
    let tokenizer_job = scope.spawn(move || resolved_tokenizer.load_pinned(tokenizer_pin, abandon));
    let gguf_metadata = gguf_metadata_json(&loader);
    let model = Llama::from_loader_with_max_seq_len(loader, None)?;
    let n_layers = model.config.n_layers;
    let embed_dim = model.config.embed_dim;

    let model_sha = join_helper(model_hash)
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
    let (tokenizer_sha, tokenizer) = join_helper(tokenizer_job)?;
    tokenizer.validate_model_vocab(model.config.vocab_size)?;
    let tokenizer_vocab_size = tokenizer.vocab_size();

    Ok(PreparedRun {
        model,
        tokenizer,
        architecture,
        n_layers,
        embed_dim,
        model_sha,
        tokenizer_sha,
        tokenizer_vocab_size,
        gguf_metadata,
        model_path: resolved.model.path.clone(),
        model_spec: resolved.model.clone(),
        direction_cache: Mutex::new(std::collections::BTreeMap::new()),
        dictionary_cache: Mutex::new(DictionaryCache::new()),
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
#[allow(
    clippy::too_many_arguments,
    reason = "application call path: each argument is a separate input"
)]
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
    in_session_pool(pool_threads(resolved)?, move || {
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

/// Run `f` as a CPU inference session on a pool of `threads` workers
/// (directly when the caller already runs on one of that size).
///
/// The session mark (`ember::model::with_cpu_session`) lets the sequential
/// decode loops inside `f` use the spin-waiting decode team although they run
/// on a Rayon worker; without it every decode region fell back to Rayon.
pub(crate) fn in_session_pool<T: Send>(
    threads: usize,
    f: impl FnOnce() -> anyhow::Result<T> + Send,
) -> anyhow::Result<T> {
    if rayon::current_thread_index().is_some() && rayon::current_num_threads() == threads {
        return ember::model::with_cpu_session(f);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("failed to build the experiment thread pool")?
        .install(move || ember::model::with_cpu_session(f))
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
    /// The spec's feature dictionaries (ADR-0003), in spec order.
    pub dictionaries: Vec<Arc<ember::v05::dictionary::Dictionary>>,
    pub threads: usize,
}

/// Wall time and generated-token count accumulated over a run's inputs,
/// and how its decode steps ran.
#[derive(Default, Clone, Copy)]
pub(crate) struct RunTiming {
    pub elapsed: std::time::Duration,
    pub generated: usize,
    pub decode: DecodeRoute,
}

/// How a run's decode steps executed; `runtime.json` records it as
/// `decode_batch` (never part of the semantic identity).
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeRoute {
    /// A standalone run: its own sequential decode; nothing is recorded.
    #[default]
    Standalone,
    /// Every decode step shared one batched forward with the other runs of
    /// a shared pass still generating (`batch_size`: the largest batch).
    Batched {
        batch_size: usize,
        generations: usize,
    },
    /// A shared pass whose runs decoded one after another, and why.
    Sequential { reason: &'static str },
}

impl DecodeRoute {
    fn to_json(self) -> Option<serde_json::Value> {
        match self {
            DecodeRoute::Standalone => None,
            DecodeRoute::Batched {
                batch_size,
                generations,
            } => Some(serde_json::json!({
                "path": "batched",
                "batch_size": batch_size,
                "generations": generations,
            })),
            DecodeRoute::Sequential { reason } => Some(serde_json::json!({
                "path": "sequential",
                "reason": reason,
            })),
        }
    }
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
    let dictionaries = crate::cli_experiment_dictionary::load_dictionaries(prepared, resolved)?;
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
            for warning in &loaded.warnings {
                log::warn!("{warning}");
            }
            bundle_sources.push(loaded);
        }
    }
    Ok(ActiveSpec {
        plan,
        bundle_sources,
        directions,
        artifacts: std::collections::BTreeMap::new(),
        dictionaries,
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
        n_heads: prepared.model.config.n_heads,
        head_dim: prepared.model.config.head_dim,
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
        for dictionary in &active.dictionaries {
            experiment.inject_dictionary(Arc::clone(dictionary));
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
    let InputSetup {
        input,
        model_context,
        seed,
        context_limit,
        mut runner,
    } = input_setup(prepared, resolved, experiment, pass);
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

/// What driving one input needs, shared by [`run_input`] and [`start_input`].
struct InputSetup<'a> {
    input: &'a ember::v05::spec::InputSpec,
    model_context: ModelContext<'a>,
    seed: Option<u64>,
    context_limit: usize,
    runner: ExperimentRunner,
}

fn input_setup<'a>(
    prepared: &'a PreparedRun,
    resolved: &'a ember::v05::spec::ExperimentSpecV1,
    experiment: &Arc<Mutex<V05Experiment>>,
    pass: Option<crate::cli_experiment_shared::SharedPass>,
) -> InputSetup<'a> {
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
    let context_limit = prepared.model.max_seq_len(&ember::backend::CpuBackend);
    let runner = match pass {
        None => ExperimentRunner::new(V05Adapter(Arc::clone(experiment))),
        Some(pass) => ExperimentRunner::new(pass),
    };
    InputSetup {
        input,
        model_context,
        seed,
        context_limit,
        runner,
    }
}

/// [`run_input`] up to its first decode forward: the input's prefill is
/// done and its first token chosen. The caller advances it (together with
/// other inputs' generations) through `cli_generation::decode_together` and
/// collects the result with [`finish_input`]. Only for models
/// `cli_generation::batched_decode_ineligibility` accepts.
pub(crate) fn start_input<'a>(
    prepared: &'a PreparedRun,
    resolved: &'a ember::v05::spec::ExperimentSpecV1,
    experiment: &Arc<Mutex<V05Experiment>>,
    pass: Option<crate::cli_experiment_shared::SharedPass>,
    role: Option<crate::cli_generation::PrefixRole<'_>>,
    cancel: Option<&ember::cancel::CancelToken>,
) -> anyhow::Result<crate::cli_generation::SteppedGeneration<'a>> {
    let InputSetup {
        input,
        model_context,
        seed,
        context_limit,
        runner,
    } = input_setup(prepared, resolved, experiment, pass);
    crate::cli_generation::SteppedGeneration::start(
        &ember::backend::CpuBackend,
        &prepared.model,
        runner,
        model_context,
        &prepared.tokenizer,
        &input.text,
        resolved.generation.max_new_tokens,
        resolved.generation.temperature,
        context_limit,
        seed,
        role,
        cancel,
    )
}

/// Finish a generation from [`start_input`] and collect the input's result.
pub(crate) fn finish_input(
    prepared: &PreparedRun,
    experiment: &Arc<Mutex<V05Experiment>>,
    generation: crate::cli_generation::SteppedGeneration<'_>,
) -> anyhow::Result<InputResult> {
    let generated_text = generation.finish(&prepared.tokenizer)?;
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
        decode_batch: timing.decode.to_json(),
    };
    let mut resolved_with_output = (*resolved).clone();
    resolved_with_output.output.directory = target.output_directory.to_path_buf();
    let mut artifacts =
        ember::v05::steering::direction_artifact_files(&active.directions, &resolved.interventions)
            .map_err(anyhow::Error::msg)?;
    artifacts.extend(active.artifacts.clone());
    let captured: Vec<&ember::v05::runner::CapturedTensor> =
        results.iter().flat_map(|result| &result.captures).collect();
    artifacts.extend(
        ember::v05::dictionary::artifact_files(resolved, &active.dictionaries, &captured)
            .map_err(anyhow::Error::msg)?,
    );
    let materials = BundleMaterials {
        spec_text: target.spec_text.to_string(),
        resolved: resolved_with_output,
        ember_version: env!("CARGO_PKG_VERSION").to_string(),
        ember_commit: ember_commit().to_string(),
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
            vocab_size: prepared.tokenizer_vocab_size,
        },
        plan: (*active.plan).clone(),
        results: results.clone(),
        // Overrides a cross-bundle source needed are part of the result's
        // provenance (docs/interventions.md).
        warnings: active
            .bundle_sources
            .iter()
            .flat_map(|source| source.warnings.iter().cloned())
            .collect(),
        runtime,
        artifacts,
    };
    // The report of the verification that gated publication: the published
    // bundle is a rename of exactly the verified staged bytes.
    let (path, identity, report) =
        ember::v05::run::write_verified_bundle(&materials, target.retain_incomplete)
            .map_err(anyhow::Error::msg)?;
    Ok(RunOutcome {
        path,
        identity,
        report,
        results,
        prefix,
    })
}

/// The commit recorded as `ember_commit` in every bundle this process
/// writes: the commit the binary was built from. It is part of the semantic
/// manifest, so it must not depend on the directory the run starts in.
fn ember_commit() -> &'static str {
    ember::extraction::build_commit().unwrap_or("unknown")
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

    #[test]
    fn a_model_file_replaced_during_load_is_detected() {
        let dir = std::env::temp_dir().join(format!("ember-model-identity-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        std::fs::write(&model, b"weights").unwrap();
        let identity = super::ModelFileIdentity::of(&model).unwrap();
        identity.ensure_unchanged(&model).unwrap();
        // Replaced by rename (same length): a different file.
        let replacement = dir.join("other.gguf");
        std::fs::write(&replacement, b"WEIGHTS").unwrap();
        std::fs::rename(&replacement, &model).unwrap();
        assert!(identity.ensure_unchanged(&model).is_err());

        // Rewritten in place with its size and modification time restored.
        let identity = super::ModelFileIdentity::of(&model).unwrap();
        let modified = std::fs::metadata(&model).unwrap().modified().unwrap();
        std::fs::write(&model, b"wEIGHTS").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&model)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        assert_eq!(
            std::fs::metadata(&model).unwrap().modified().unwrap(),
            modified
        );
        #[cfg(unix)]
        assert!(identity.ensure_unchanged(&model).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_signature_anchor_on_a_sweep_is_refused_not_ignored() {
        let dir = std::env::temp_dir().join(format!("ember-sweep-anchor-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(ember::v05::sweep::SWEEP_MANIFEST_FILE), b"{}").unwrap();
        let command = super::VerifyArgs {
            bundle: dir.clone(),
            model: None,
            tokenizer: None,
            anchor: super::AnchorArgs {
                trusted_key: Some("lab.pub".into()),
                ..super::AnchorArgs::default()
            },
            write_report: None,
            json: true,
        };
        let error = super::run_verify_command(&command).unwrap_err().to_string();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(error.contains("--trusted-key"), "{error}");
    }
}
