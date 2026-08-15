//! Ember v0.6 browser experiment console (`ember web-gui`).
//!
//! A thin, offline presentation layer over the existing v0.5 experiment
//! pipeline. `ember web-gui` starts a tiny HTTP server on localhost serving
//! one self-contained page (no web framework, no external assets). Every
//! action is translated into an `ember.experiment.v1` specification,
//! resolved and validated through the exact same path as
//! `ember experiment run`, and executed with the same `prepare_run` /
//! `execute_prepared` code. The GUI adds no parallel experiment semantics,
//! no inference logic, and no weaker validation.
//!
//! The model is loaded once per selected GGUF file and kept resident, so the
//! demo loop (change layer/intervention -> Run -> compare -> verify) reuses
//! one loaded model instead of reloading per run.

use crate::cli_experiment::{execute_prepared, prepare_run, PreparedRun};
use anyhow::Context;
use clap::Args as ClapArgs;
use ember::plan::ExecutionMode;
use ember::quant_k::KStrategy;
use ember::v05::capture::{
    CaptureDType, CaptureSpec, CaptureStorage, InputSelector, LayerSelector,
};
use ember::v05::hook::SemanticHookSite;
use ember::v05::intervention::{
    CompatibilityPolicy, InterventionOperation, InterventionSource, InterventionSpec, ShapePolicy,
};
use ember::v05::runner::InputResult;
use ember::v05::spec::{
    RawExecutionSpec, RawExperimentMetadata, RawExperimentSpec, RawGenerationSpec, RawInputSpec,
    RawModelSpec, RawOutputSpec, EXPERIMENT_SCHEMA_V1,
};
use ember::v05::token_select::{SubtokenSelection, TextNormalization, TokenSelector};
use ember::v05::verify::VerificationReport;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

/// `ember web-gui` CLI arguments.
#[derive(ClapArgs)]
pub(crate) struct GuiArgs {
    /// interface to bind (use 0.0.0.0 only for a trusted network)
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// port to bind; 0 picks a free port
    #[arg(long, default_value_t = 8337)]
    pub port: u16,
    /// do not attempt to open a browser
    #[arg(long)]
    pub no_open: bool,
}

const PAGE: &str = include_str!("gui_page.html");

// ---------------------------------------------------------------------------
// session state
// ---------------------------------------------------------------------------

/// One resident model session. Runs are serialized through this mutex; the
/// model is reused across baseline/intervention/restore executions.
pub(crate) struct GuiSession {
    k_strategy: KStrategy,
    k_allow_fallback: bool,
    prepared: Option<PreparedRun>,
    load_ms: f64,
    load_error: Option<String>,
    last_baseline: Option<BaselineRecord>,
    run_counter: u64,
}

/// The baseline text a later restore run can be compared against. The
/// comparison is only meaningful when the configuration is unchanged, so
/// the configuration key is recorded alongside.
#[derive(Clone)]
struct BaselineRecord {
    text: String,
    config_key: String,
}

/// Typed summary of the resident model session (shared by the web and the
/// native console). Only constructed when the native console feature is on.
#[cfg(feature = "gui")]
#[derive(Debug, Clone, Serialize)]
pub(crate) struct SessionInfo {
    pub model_path: String,
    pub model_name: String,
    pub architecture: String,
    pub n_layers: usize,
    pub embed_dim: usize,
    pub vocab_size: usize,
    pub model_sha: String,
    pub tokenizer_sha: String,
    pub load_ms: f64,
}

/// Typed outcome of one baseline + intervention pair (shared by both GUIs).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct RunBundle {
    pub baseline: RunOutput,
    pub intervention: RunOutput,
    pub verification: VerificationReport,
    pub elapsed_ms_total: f64,
    pub elapsed_ms_baseline: f64,
    pub baseline_key: String,
}

/// Typed outcome of the restore-original leg (shared by both GUIs).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct RestoreBundle {
    pub output: RunOutput,
    pub verification: VerificationReport,
    pub matches_baseline: bool,
    pub baseline_comparable: bool,
    pub baseline_text: Option<String>,
}

impl GuiSession {
    pub(crate) fn new(k_strategy: KStrategy, k_allow_fallback: bool) -> Self {
        GuiSession {
            k_strategy,
            k_allow_fallback,
            prepared: None,
            load_ms: 0.0,
            load_error: None,
            last_baseline: None,
            run_counter: 0,
        }
    }

    /// Load (or reuse) the model for `model_path`.
    pub(crate) fn ensure_prepared(&mut self, model_path: &str) -> Result<(), String> {
        if let Some(prepared) = &self.prepared {
            if prepared.model_path.to_string_lossy() == model_path {
                return Ok(());
            }
        }
        let started = std::time::Instant::now();
        // Reuse the CLI's own model-loading path (same loader, strategy,
        // architecture resolution, tokenizer resolution, and SHA checks).
        let spec = dummy_model_spec(model_path)?;
        let prepared = prepare_run(&spec, self.k_strategy, self.k_allow_fallback)
            .map_err(|error| format!("{error:#}"))?;
        self.load_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.load_error = None;
        self.prepared = Some(prepared);
        Ok(())
    }

    /// Typed summary of the prepared session, if any.
    #[cfg(feature = "gui")]
    pub(crate) fn info(&self) -> Option<SessionInfo> {
        let prepared = self.prepared.as_ref()?;
        Some(SessionInfo {
            model_path: prepared.model_path.display().to_string(),
            model_name: model_basename(&prepared.model_path),
            architecture: prepared.architecture.clone(),
            n_layers: prepared.n_layers,
            embed_dim: prepared.embed_dim,
            vocab_size: prepared.model.config.vocab_size,
            model_sha: prepared.model_sha.clone(),
            tokenizer_sha: prepared.tokenizer_sha.clone(),
            load_ms: self.load_ms,
        })
    }

    /// Run the capture-only baseline and the capture+intervention pair for
    /// one configuration. Both runs write and self-verify real v0.5 bundles
    /// through `execute_prepared`; the baseline text is recorded for later
    /// restore comparison.
    pub(crate) fn run_baseline_intervention(
        &mut self,
        cfg: &RunConfig,
    ) -> Result<RunBundle, String> {
        let started = std::time::Instant::now();
        let (baseline, _baseline_report) = run_one(self, cfg, RunKind::Baseline)
            .map_err(|error| format!("baseline run failed: {error}"))?;
        let elapsed_ms_baseline = started.elapsed().as_secs_f64() * 1000.0;
        let (intervention, intervention_report) = run_one(self, cfg, RunKind::Intervention)
            .map_err(|error| format!("intervention run failed: {error}"))?;
        let elapsed_ms_total = started.elapsed().as_secs_f64() * 1000.0;
        let baseline_key = cfg.comparison_key();
        self.last_baseline = Some(BaselineRecord {
            text: baseline.text.clone(),
            config_key: baseline_key.clone(),
        });
        Ok(RunBundle {
            baseline,
            intervention,
            verification: intervention_report,
            elapsed_ms_total,
            elapsed_ms_baseline,
            baseline_key,
        })
    }

    /// Run the restore-original leg for one configuration and compare its
    /// generated text against the stored baseline (only when the shared
    /// configuration is unchanged since the last baseline run).
    pub(crate) fn run_restore_leg(&mut self, cfg: &RunConfig) -> Result<RestoreBundle, String> {
        let (output, report) = run_one(self, cfg, RunKind::Restore)
            .map_err(|error| format!("restore run failed: {error}"))?;
        let baseline = self.last_baseline.clone();
        let (matches_baseline, comparable) = match &baseline {
            Some(record) if record.config_key == cfg.comparison_key() => {
                (record.text == output.text, true)
            }
            _ => (false, false),
        };
        Ok(RestoreBundle {
            output,
            verification: report,
            matches_baseline,
            baseline_comparable: comparable,
            baseline_text: baseline.map(|record| record.text),
        })
    }
}

/// A minimal resolved spec carrying just the model path; `prepare_run`
/// reads everything it needs from it (tokenizer resolution and provenance
/// hashes come from defaults, exactly like a user-authored spec).
/// Shared raw-spec skeleton for the console: same model/execution/generation
/// boilerplate with per-run name, prompt, payloads, and output directory.
#[allow(clippy::too_many_arguments)]
fn raw_spec(
    name: &str,
    model_path: &str,
    execution_mode: &str,
    max_new_tokens: usize,
    prompt: &str,
    captures: Vec<CaptureSpec>,
    interventions: Vec<InterventionSpec>,
    output_dir: PathBuf,
    overwrite: bool,
) -> RawExperimentSpec {
    RawExperimentSpec {
        schema: EXPERIMENT_SCHEMA_V1.to_string(),
        experiment: RawExperimentMetadata {
            name: name.to_string(),
            description: Some("run from the ember experiment console".to_string()),
            seed: Some(0),
        },
        model: RawModelSpec {
            path: PathBuf::from(model_path),
            expected_sha256: None,
            tokenizer: None,
            tokenizer_expected_sha256: None,
            arch: Some("auto".to_string()),
        },
        execution: Some(RawExecutionSpec {
            mode: Some(execution_mode.to_string()),
            threads: Some(0),
            deterministic: Some(true),
        }),
        generation: Some(RawGenerationSpec {
            max_new_tokens: Some(max_new_tokens),
            temperature: Some(0.0),
        }),
        inputs: vec![RawInputSpec {
            id: "prompt-1".to_string(),
            text: prompt.to_string(),
        }],
        captures,
        interventions,
        output: RawOutputSpec {
            directory: output_dir,
            tensor_format: Some("safetensors".to_string()),
            overwrite: Some(overwrite),
        },
    }
}

fn dummy_model_spec(model_path: &str) -> Result<ember::v05::spec::ExperimentSpecV1, String> {
    raw_spec(
        "gui-model-load",
        model_path,
        "reference",
        1,
        "model load",
        Vec::new(),
        Vec::new(),
        PathBuf::from("runs/gui/_preload"),
        true,
    )
    .resolve()
    .map_err(|error| error.to_string())
}

// ---------------------------------------------------------------------------
// request / response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PrepareRequest {
    model_path: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RunRequest {
    pub model_path: String,
    pub prompt: String,
    pub max_new_tokens: usize,
    /// "reference" | "planned" | "planned-fused"
    pub execution: String,
    /// v0.4 hook stage id: before-layer, after-attention, after-mlp,
    /// after-layer, before-logits, after-logits
    pub site: String,
    /// Transformer layer (unused for before-logits / after-logits).
    pub layer: Option<usize>,
    /// "replace" | "zero" | "scale" | "interpolate" | "add-delta"
    pub operation: String,
    pub factor: Option<f32>,
    pub alpha: Option<f32>,
    /// "capture" | "zero" (zero is valid only for replace / interpolate)
    pub source: String,
    /// Layer the source capture is taken from (capture sources only).
    pub source_layer: Option<usize>,
    /// "prompt-final" | "matched-span"
    pub token_kind: String,
    pub span_text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RestoreRequest {
    model_path: String,
    prompt: String,
    max_new_tokens: usize,
    execution: String,
    site: String,
    layer: Option<usize>,
    token_kind: String,
    span_text: Option<String>,
}

#[derive(Debug, Serialize)]
struct ApiEnvelope {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

impl ApiEnvelope {
    fn ok(data: serde_json::Value) -> Self {
        ApiEnvelope {
            ok: true,
            error: None,
            data: Some(data),
        }
    }
    fn err(error: impl Into<String>) -> Self {
        ApiEnvelope {
            ok: false,
            error: Some(error.into()),
            data: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RunOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub bundle_dir: String,
    pub semantic_hash: String,
    pub payload_hash: String,
    pub wall_ms: f64,
    pub decode_tps: Option<f64>,
    pub events: Vec<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// run configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuiOperation {
    Replace,
    Zero,
    Scale,
    Interpolate,
    AddDelta,
    /// Internal: the restore-original verification leg (`/api/restore`).
    RestoreOriginal,
}

impl GuiOperation {
    fn parse(value: &str) -> Result<GuiOperation, String> {
        match value {
            "replace" => Ok(GuiOperation::Replace),
            "zero" => Ok(GuiOperation::Zero),
            "scale" => Ok(GuiOperation::Scale),
            "interpolate" => Ok(GuiOperation::Interpolate),
            "add-delta" => Ok(GuiOperation::AddDelta),
            "restore-original" => Ok(GuiOperation::RestoreOriginal),
            other => Err(format!(
                "unknown intervention '{other}'; expected replace, zero, scale, interpolate, or add-delta"
            )),
        }
    }
    fn requires_source(self) -> bool {
        matches!(
            self,
            GuiOperation::Replace | GuiOperation::Interpolate | GuiOperation::AddDelta
        )
    }
    fn is_restore(self) -> bool {
        matches!(self, GuiOperation::RestoreOriginal)
    }
    fn label(self, factor: f32, alpha: f32) -> String {
        match self {
            GuiOperation::Replace => "replace".to_string(),
            GuiOperation::Zero => "zero".to_string(),
            GuiOperation::Scale => format!("scale \u{00d7}{factor:.2}"),
            GuiOperation::Interpolate => format!("interpolate \u{03b1}={alpha:.2}"),
            GuiOperation::AddDelta => "add-delta".to_string(),
            GuiOperation::RestoreOriginal => "restore-original".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceKind {
    Capture,
    Zero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenKind {
    PromptFinal,
    MatchedSpan,
}

#[derive(Debug, Clone)]
pub(crate) struct RunConfig {
    pub model_path: String,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub execution: ExecutionMode,
    pub site: SemanticHookSite,
    pub layer: Option<usize>,
    pub operation: GuiOperation,
    pub factor: f32,
    pub alpha: f32,
    pub source: SourceKind,
    pub source_layer: Option<usize>,
    pub token: TokenKind,
    pub span_text: String,
}

impl RunConfig {
    /// Canonical key over the *shared* configuration (model, prompt, site,
    /// layer, token selection, generation limits). Used to decide whether a
    /// restore run is comparable to the stored baseline: the restore leg
    /// must revisit the same site/layer/selection, while the intervention
    /// operation itself is irrelevant to the comparison.
    fn comparison_key(&self) -> String {
        format!(
            "{}|{}|{:?}|{:?}|{:?}|{}|{}|{}",
            self.model_path,
            self.prompt,
            self.site,
            self.layer,
            self.token,
            self.span_text,
            self.max_new_tokens,
            self.execution.name(),
        )
    }
}

pub(crate) fn parse_run_request(req: &RunRequest) -> Result<RunConfig, String> {
    let site = SemanticHookSite::ALL
        .iter()
        .find(|site| site.stage_id() == req.site)
        .copied()
        .ok_or_else(|| {
            format!(
                "unknown hook stage '{}'; expected one of: before-layer, after-attention, \
                 after-mlp, after-layer, before-logits, after-logits",
                req.site
            )
        })?;
    if site.is_per_layer() && req.layer.is_none() {
        return Err(format!("layer is required for hook stage '{}'", req.site));
    }
    if !site.is_per_layer() && req.layer.is_some() {
        return Err(format!(
            "hook stage '{}' does not carry a layer; the layer control is disabled",
            req.site
        ));
    }
    let operation = GuiOperation::parse(&req.operation)?;
    let factor = req.factor.unwrap_or(1.0);
    let alpha = req.alpha.unwrap_or(0.5);
    if !factor.is_finite() {
        return Err("scale factor must be a finite number".to_string());
    }
    if !alpha.is_finite() {
        return Err("interpolate alpha must be a finite number".to_string());
    }
    let source = match req.source.as_str() {
        "capture" => SourceKind::Capture,
        "zero" => SourceKind::Zero,
        other => {
            return Err(format!(
                "unknown source '{other}'; expected capture or zero"
            ))
        }
    };
    if source == SourceKind::Zero
        && !matches!(operation, GuiOperation::Replace | GuiOperation::Interpolate)
    {
        return Err("the zero source is only meaningful for replace and interpolate".to_string());
    }
    if operation.requires_source() && source == SourceKind::Capture && site.is_per_layer() {
        let target = req.layer.expect("layer checked above");
        let source_layer = req.source_layer.unwrap_or(target.saturating_sub(1));
        if source_layer > target {
            return Err(format!(
                "the source capture layer {source_layer} must not be below the intervention \
                 layer {target} (the capture fires before the intervention in the same pass)"
            ));
        }
        // Keep the resolved source layer on the config for the spec builder.
        let mut cfg = run_config_from(req, site, operation, factor, alpha, source)?;
        cfg.source_layer = Some(source_layer);
        return Ok(cfg);
    }
    run_config_from(req, site, operation, factor, alpha, source)
}

fn run_config_from(
    req: &RunRequest,
    site: SemanticHookSite,
    operation: GuiOperation,
    factor: f32,
    alpha: f32,
    source: SourceKind,
) -> Result<RunConfig, String> {
    let token = match req.token_kind.as_str() {
        "prompt-final" => TokenKind::PromptFinal,
        "matched-span" => {
            let text = req.span_text.clone().unwrap_or_default();
            if text.trim().is_empty() {
                return Err("matched-span requires a span text".to_string());
            }
            TokenKind::MatchedSpan
        }
        other => return Err(format!("unknown token selector '{other}'")),
    };
    if req.prompt.trim().is_empty() {
        return Err("prompt must not be empty".to_string());
    }
    if req.max_new_tokens == 0 {
        return Err("max new tokens must be >= 1".to_string());
    }
    let execution = ExecutionMode::from_cli(&req.execution).map_err(|error| error.to_string())?;
    Ok(RunConfig {
        model_path: req.model_path.clone(),
        prompt: req.prompt.clone(),
        max_new_tokens: req.max_new_tokens,
        execution,
        site,
        layer: req.layer,
        operation,
        factor,
        alpha,
        source,
        source_layer: req.source_layer,
        token,
        span_text: req.span_text.clone().unwrap_or_default(),
    })
}

/// The v0.4 hook stage ids, taken from Ember's own hook definitions
/// (`SemanticHookSite::stage_id`), used by the page's hook selector.
pub(crate) fn hook_stages() -> Vec<&'static str> {
    SemanticHookSite::ALL
        .iter()
        .map(|site| site.stage_id())
        .collect()
}

// ---------------------------------------------------------------------------
// spec building (same raw TOML form a user would write, resolved + validated)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum RunKind {
    Baseline,
    Intervention,
    Restore,
}

/// Build the raw (user-form) spec for one run kind and resolve it through
/// the standard `RawExperimentSpec::resolve()` gate.
fn build_and_resolve_spec(
    cfg: &RunConfig,
    kind: RunKind,
    output_dir: &str,
) -> Result<(String, ember::v05::spec::ExperimentSpecV1), String> {
    let tokens = match cfg.token {
        TokenKind::PromptFinal => TokenSelector::PromptFinal,
        TokenKind::MatchedSpan => TokenSelector::MatchedTextSpan {
            text: cfg.span_text.clone(),
            occurrence: 0,
            subtoken_selection: SubtokenSelection::All,
            normalization: TextNormalization::None,
        },
    };
    let layers_for = |layer: Option<usize>| match layer {
        Some(layer) => LayerSelector::List(vec![layer]),
        None => LayerSelector::All("all".to_string()),
    };

    // The source capture layer: the source layer for capture sources, the
    // intervention layer otherwise (so baseline and intervention bundles
    // carry the same capture and compare cleanly).
    let capture_layer = if cfg.source == SourceKind::Capture && cfg.operation.requires_source() {
        cfg.source_layer.or(cfg.layer)
    } else {
        cfg.layer
    };

    let capture = CaptureSpec {
        id: "cap-src".to_string(),
        site: cfg.site,
        layers: layers_for(capture_layer),
        tokens: tokens.clone(),
        inputs: InputSelector::All("all".to_string()),
        storage: CaptureStorage::SelectedRows,
        dtype: CaptureDType::F32,
    };

    let operation = match cfg.operation {
        GuiOperation::Replace => InterventionOperation::Replace,
        GuiOperation::Zero => InterventionOperation::Zero,
        GuiOperation::Scale => InterventionOperation::Scale { factor: cfg.factor },
        GuiOperation::Interpolate => InterventionOperation::Interpolate { alpha: cfg.alpha },
        GuiOperation::AddDelta => InterventionOperation::AddDelta,
        GuiOperation::RestoreOriginal => InterventionOperation::RestoreOriginal,
    };
    let source = match (&cfg.operation, &cfg.source) {
        (operation, _) if operation.is_restore() || !operation.requires_source() => None,
        (_, SourceKind::Zero) => Some(InterventionSource::Zero),
        (_, SourceKind::Capture) => Some(InterventionSource::CaptureFromCurrentRun {
            capture_id: "cap-src".to_string(),
        }),
    };
    let intervention = InterventionSpec {
        id: "iv-1".to_string(),
        site: cfg.site,
        layers: layers_for(cfg.layer),
        tokens: tokens.clone(),
        inputs: InputSelector::All("all".to_string()),
        operation,
        source,
        shape_policy: ShapePolicy::Strict,
        compatibility: CompatibilityPolicy::default(),
    };
    let restore = InterventionSpec {
        id: "restore-1".to_string(),
        site: cfg.site,
        layers: layers_for(cfg.layer),
        tokens,
        inputs: InputSelector::All("all".to_string()),
        operation: InterventionOperation::RestoreOriginal,
        source: None,
        shape_policy: ShapePolicy::Strict,
        compatibility: CompatibilityPolicy::default(),
    };

    let (name, captures, interventions) = match kind {
        RunKind::Baseline => ("gui-baseline", vec![capture], Vec::new()),
        RunKind::Intervention => ("gui-intervention", vec![capture], vec![intervention]),
        RunKind::Restore => ("gui-restore", Vec::new(), vec![restore]),
    };

    let raw = raw_spec(
        name,
        &cfg.model_path,
        cfg.execution.name(),
        cfg.max_new_tokens,
        &cfg.prompt,
        captures,
        interventions,
        PathBuf::from(output_dir),
        false,
    );
    let spec_text = toml::to_string_pretty(&raw)
        .map_err(|error| format!("cannot serialize the experiment specification: {error}"))?;
    let resolved = raw.resolve().map_err(|error| error.to_string())?;
    Ok((spec_text, resolved))
}

// ---------------------------------------------------------------------------
// execution
// ---------------------------------------------------------------------------

fn next_output_dir(session: &mut GuiSession, kind: &str) -> String {
    session.run_counter += 1;
    format!(
        "runs/gui/{kind}-{}-{:02}",
        unix_timestamp_seconds(),
        session.run_counter % 100
    )
}

fn unix_timestamp_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn run_one(
    session: &mut GuiSession,
    cfg: &RunConfig,
    kind: RunKind,
) -> Result<(RunOutput, VerificationReport), String> {
    session.ensure_prepared(&cfg.model_path)?;
    let output_dir = next_output_dir(
        session,
        match kind {
            RunKind::Baseline => "baseline",
            RunKind::Intervention => "intervention",
            RunKind::Restore => "restore",
        },
    );
    let prepared = session
        .prepared
        .as_ref()
        .ok_or_else(|| "model session is not prepared".to_string())?;
    let (spec_text, resolved) = build_and_resolve_spec(cfg, kind, &output_dir)?;
    let (path, identity, report, results) = execute_prepared(
        prepared,
        &resolved,
        &spec_text,
        Path::new(&output_dir),
        false,
    )
    .map_err(|error| format!("{error:#}"))?;
    let result: &InputResult = results
        .first()
        .ok_or_else(|| "the run produced no input results".to_string())?;
    let (wall_ms, decode_tps) = read_runtime_metrics(&path);
    let events: Vec<serde_json::Value> = result
        .events
        .iter()
        .map(|event| {
            serde_json::json!({
                "intervention_id": event.intervention_id,
                "site": event.site.stage_id(),
                "layer": event.layer,
                "positions": event.positions,
                "operation": event.operation.kind_name(),
                "source": event.source_kind,
                "snapshot_checksum": event.snapshot_checksum,
                "applied": event.applied,
            })
        })
        .collect();
    Ok((
        RunOutput {
            text: result.generated_text.clone(),
            prompt_tokens: result.tokenization.token_ids.len(),
            generated_tokens: result.generated_token_ids.len(),
            bundle_dir: path.display().to_string(),
            semantic_hash: identity.semantic_hash.clone(),
            payload_hash: identity.payload_hash.clone(),
            wall_ms,
            decode_tps,
            events,
        },
        report,
    ))
}

/// Read the honest wall-clock and decode throughput from the bundle's
/// `runtime.json` (written by `write_bundle` from `RuntimeMetrics`).
fn read_runtime_metrics(bundle_dir: &Path) -> (f64, Option<f64>) {
    let runtime_path = bundle_dir.join("runtime.json");
    let Ok(text) = std::fs::read_to_string(&runtime_path) else {
        return (0.0, None);
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return (0.0, None);
    };
    let wall_ms = value
        .get("wall_clock_ms")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let tps = value
        .get("decode_throughput_tps")
        .and_then(serde_json::Value::as_f64);
    (wall_ms, tps)
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

/// Entry point for `ember web-gui`.
pub(crate) fn run_gui_command(
    gui: &GuiArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let session = Arc::new(Mutex::new(GuiSession::new(k_strategy, k_allow_fallback)));
    let server = tiny_http::Server::http((gui.host.as_str(), gui.port)).map_err(|error| {
        anyhow::anyhow!(
            "cannot bind the GUI server on {}:{}: {error}",
            gui.host,
            gui.port
        )
    })?;
    let url = format!("http://{}/", server.server_addr());
    eprintln!(
        "EMBER experiment console v{} — {url}",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  press Ctrl-C to stop; the GUI is fully offline.");
    if !gui.no_open {
        open_browser(&url);
    }
    let limiter = Arc::new(SlotLimiter::new());
    for request in server.incoming_requests() {
        let session = Arc::clone(&session);
        let limiter = Arc::clone(&limiter);
        std::thread::spawn(move || {
            let _slot = limiter.acquire();
            if let Err(error) = handle_request(request, &session) {
                log::error!("gui request failed: {error:#}");
            }
        });
    }
    Ok(())
}

fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd")
        .args(["/c", "start", "", url])
        .spawn();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

fn handle_request(
    mut request: tiny_http::Request,
    session: &Arc<Mutex<GuiSession>>,
) -> anyhow::Result<()> {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or(&url);
    let method = request.method().clone();
    let is_root = method == tiny_http::Method::Get && path == "/";
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        build_response(is_root, &method, path, &mut request, session)
    }));
    let response = match built {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            log::error!("gui request failed: {error:#}");
            json_response(&ApiEnvelope::err(format!("internal error: {error}")))?
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            log::error!("gui request panicked: {message}");
            json_response(&ApiEnvelope::err(
                "internal error: request handler panicked",
            ))?
        }
    };
    request.respond(response)?;
    Ok(())
}

fn build_response(
    is_root: bool,
    method: &tiny_http::Method,
    path: &str,
    request: &mut tiny_http::Request,
    session: &Arc<Mutex<GuiSession>>,
) -> anyhow::Result<tiny_http::Response<std::io::Cursor<Vec<u8>>>> {
    if is_root {
        return Ok(tiny_http::Response::from_string(PAGE.to_string())
            .with_header(header("Content-Type", "text/html; charset=utf-8"))
            .with_header(header("Cache-Control", "no-store")));
    }
    let envelope = match (method, path) {
        (tiny_http::Method::Get, "/api/state") => state_payload(session),
        (tiny_http::Method::Post, "/api/prepare") => match read_json::<PrepareRequest>(request) {
            Ok(req) => {
                let mut session = lock_session(session);
                match session.ensure_prepared(&req.model_path) {
                    Ok(()) => {
                        let info = session_info_payload(&session);
                        ApiEnvelope::ok(info.unwrap_or_else(|| serde_json::json!({})))
                    }
                    Err(error) => {
                        session.load_error = Some(error.clone());
                        ApiEnvelope::err(error)
                    }
                }
            }
            Err(error) => ApiEnvelope::err(format!("malformed request: {error}")),
        },
        (tiny_http::Method::Post, "/api/run") => match read_json::<RunRequest>(request) {
            Ok(req) => run_experiment(session, req),
            Err(error) => ApiEnvelope::err(format!("malformed request: {error}")),
        },
        (tiny_http::Method::Post, "/api/restore") => match read_json::<RestoreRequest>(request) {
            Ok(req) => run_restore(session, req),
            Err(error) => ApiEnvelope::err(format!("malformed request: {error}")),
        },
        _ => ApiEnvelope::err("not found"),
    };
    json_response(&envelope)
}

fn header(name: &str, value: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes())
        .expect("static header bytes are valid")
}

fn read_json<T: for<'de> Deserialize<'de>>(request: &mut tiny_http::Request) -> anyhow::Result<T> {
    let mut body = Vec::new();
    request
        .as_reader()
        .read_to_end(&mut body)
        .context("cannot read request body")?;
    serde_json::from_slice(&body).context("malformed JSON request body")
}

fn json_response(
    value: &ApiEnvelope,
) -> anyhow::Result<tiny_http::Response<std::io::Cursor<Vec<u8>>>> {
    let body = serde_json::to_vec(value).context("cannot serialize JSON response")?;
    Ok(tiny_http::Response::from_data(body)
        .with_header(header("Content-Type", "application/json; charset=utf-8"))
        .with_header(header("Cache-Control", "no-store")))
}

/// Lock the GUI session, recovering from poisoning so a single panicking
/// request cannot permanently brick the console.
fn lock_session(session: &Arc<Mutex<GuiSession>>) -> std::sync::MutexGuard<'_, GuiSession> {
    session.lock().unwrap_or_else(|poisoned| {
        log::warn!("gui session mutex was poisoned; recovering guarded state");
        poisoned.into_inner()
    })
}

/// Bounds concurrent request-handler threads so a slow or malicious client
/// cannot exhaust the process with unbounded thread-per-request spawning.
struct SlotLimiter {
    active: Mutex<usize>,
    released: Condvar,
}

impl SlotLimiter {
    fn new() -> Self {
        Self {
            active: Mutex::new(0),
            released: Condvar::new(),
        }
    }

    fn acquire(self: &Arc<Self>) -> SlotGuard {
        const MAX_HANDLER_THREADS: usize = 8;
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while *active >= MAX_HANDLER_THREADS {
            active = self
                .released
                .wait(active)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        *active += 1;
        SlotGuard {
            limiter: Arc::clone(self),
        }
    }
}

struct SlotGuard {
    limiter: Arc<SlotLimiter>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        let mut active = self
            .limiter
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *active = active.saturating_sub(1);
        self.limiter.released.notify_one();
    }
}

// ---------------------------------------------------------------------------
// API handlers
// ---------------------------------------------------------------------------

fn state_payload(session: &Arc<Mutex<GuiSession>>) -> ApiEnvelope {
    let session = lock_session(session);
    ApiEnvelope::ok(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "commit": ember::extraction::git_commit().unwrap_or_else(|| "unknown".to_string()),
        "models": discover_models(),
        "hook_stages": hook_stages(),
        "session": session_info_payload(&session),
    }))
}

fn session_info_payload(session: &GuiSession) -> Option<serde_json::Value> {
    let prepared = session.prepared.as_ref()?;
    Some(serde_json::json!({
        "model_path": prepared.model_path.display().to_string(),
        "model_name": model_basename(&prepared.model_path),
        "architecture": prepared.architecture,
        "n_layers": prepared.n_layers,
        "embed_dim": prepared.embed_dim,
        "vocab_size": prepared.model.config.vocab_size,
        "model_sha": prepared.model_sha,
        "tokenizer_sha": prepared.tokenizer_sha,
        "load_ms": session.load_ms,
        "load_error": session.load_error,
    }))
}

fn model_basename(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn verification_json(report: &VerificationReport) -> serde_json::Value {
    serde_json::json!({
        "ok": report.ok,
        "checks": report.checks.iter().map(|check| serde_json::json!({
            "name": check.name,
            "ok": check.ok,
            "detail": check.detail,
        })).collect::<Vec<_>>(),
        "warnings": report.warnings,
    })
}

fn run_experiment(session: &Arc<Mutex<GuiSession>>, req: RunRequest) -> ApiEnvelope {
    let cfg = match parse_run_request(&req) {
        Ok(cfg) => cfg,
        Err(error) => return ApiEnvelope::err(error),
    };
    let mut session = lock_session(session);
    match session.run_baseline_intervention(&cfg) {
        Ok(bundle) => ApiEnvelope::ok(serde_json::json!({
            "baseline": bundle.baseline,
            "intervention": bundle.intervention,
            "verification": verification_json(&bundle.verification),
            "elapsed_ms_total": bundle.elapsed_ms_total,
            "elapsed_ms_baseline": bundle.elapsed_ms_baseline,
            "config": serde_json::json!({
                "site": cfg.site.stage_id(),
                "layer": cfg.layer,
                "operation": cfg.operation.label(cfg.factor, cfg.alpha),
                "max_new_tokens": cfg.max_new_tokens,
                "execution": cfg.execution.name(),
            }),
        })),
        Err(error) => ApiEnvelope::err(error),
    }
}

fn run_restore(session: &Arc<Mutex<GuiSession>>, req: RestoreRequest) -> ApiEnvelope {
    // The restore run reuses the full run configuration; only the
    // operation differs (restore-original instead of the user intervention).
    let run_req = RunRequest {
        model_path: req.model_path,
        prompt: req.prompt.clone(),
        max_new_tokens: req.max_new_tokens,
        execution: req.execution,
        site: req.site,
        layer: req.layer,
        operation: "restore-original".to_string(),
        factor: None,
        alpha: None,
        source: "capture".to_string(),
        source_layer: None,
        token_kind: req.token_kind,
        span_text: req.span_text,
    };
    let cfg = match parse_run_request(&run_req) {
        Ok(cfg) => cfg,
        Err(error) => return ApiEnvelope::err(error),
    };
    let mut session = lock_session(session);
    match session.run_restore_leg(&cfg) {
        Ok(bundle) => ApiEnvelope::ok(serde_json::json!({
            "restore": bundle.output,
            "verification": verification_json(&bundle.verification),
            "restore_matches_baseline": bundle.matches_baseline,
            "baseline_comparable": bundle.baseline_comparable,
            "baseline_text": bundle.baseline_text,
            "restore_text": bundle.output.text,
        })),
        Err(error) => ApiEnvelope::err(error),
    }
}

// ---------------------------------------------------------------------------
// model discovery
// ---------------------------------------------------------------------------

/// Discover GGUF files near the working directory (depth-limited, skipping
/// build/vendor trees) so the page can offer a model picker.
pub(crate) fn discover_models() -> Vec<String> {
    let mut found = BTreeSet::new();
    let mut stack = vec![(
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        0,
    )];
    let skipped = [
        "target",
        ".git",
        ".venv",
        "node_modules",
        "runs",
        "logs",
        "paper",
        ".cache",
        "data",
    ];
    while let Some((dir, depth)) = stack.pop() {
        if depth > 4 {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if path.is_dir() {
                if skipped.contains(&name.as_str()) {
                    continue;
                }
                stack.push((path, depth + 1));
            } else if path.extension().is_some_and(|ext| ext == "gguf") {
                if let Ok(absolute) = std::fs::canonicalize(&path) {
                    found.insert(absolute.display().to_string());
                } else {
                    found.insert(path.display().to_string());
                }
            }
        }
    }
    found.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_request() -> RunRequest {
        RunRequest {
            model_path: "model.gguf".to_string(),
            prompt: "\u{0627}\u{0643}\u{062a}\u{0628} \u{062c}\u{0645}\u{0644}\u{0629}".to_string(),
            max_new_tokens: 16,
            execution: "reference".to_string(),
            site: "after-mlp".to_string(),
            layer: Some(7),
            operation: "scale".to_string(),
            factor: Some(0.5),
            alpha: None,
            source: "capture".to_string(),
            source_layer: None,
            token_kind: "prompt-final".to_string(),
            span_text: None,
        }
    }

    fn cfg_of(req: &RunRequest) -> RunConfig {
        parse_run_request(req).expect("request parses")
    }

    #[test]
    fn every_operation_builds_valid_specs() {
        for (operation, extra) in [
            ("replace", "capture"),
            ("zero", "capture"),
            ("scale", "capture"),
            ("interpolate", "capture"),
            ("add-delta", "capture"),
        ] {
            let mut req = base_request();
            req.operation = operation.to_string();
            req.source = extra.to_string();
            if operation == "scale" {
                req.factor = Some(0.25);
            }
            if operation == "interpolate" {
                req.alpha = Some(0.4);
            }
            let cfg = cfg_of(&req);
            for kind in [RunKind::Baseline, RunKind::Intervention, RunKind::Restore] {
                let (spec_text, resolved) =
                    build_and_resolve_spec(&cfg, kind, "runs/gui/_test").expect("spec validates");
                // The raw form must be parseable by the strict raw parser.
                let reparsed =
                    RawExperimentSpec::from_toml_str(&spec_text).expect("raw round trip");
                reparsed.resolve().expect("raw re-resolves");
                assert_eq!(
                    resolved.interventions.len(),
                    usize::from(kind == RunKind::Intervention || kind == RunKind::Restore)
                );
                if kind == RunKind::Intervention {
                    assert_eq!(resolved.captures.len(), 1);
                }
            }
        }
    }

    #[test]
    fn non_per_layer_sites_use_all_layers() {
        for site in ["before-logits", "after-logits"] {
            let mut req = base_request();
            req.site = site.to_string();
            req.layer = None;
            let cfg = cfg_of(&req);
            let (_, resolved) =
                build_and_resolve_spec(&cfg, RunKind::Intervention, "runs/gui/_test").unwrap();
            let layers = resolved.interventions[0].layers.clone();
            assert!(matches!(layers, LayerSelector::All(_)));
        }
    }

    #[test]
    fn layer_required_for_per_layer_sites() {
        let mut req = base_request();
        req.layer = None;
        assert!(parse_run_request(&req).is_err());
    }

    #[test]
    fn source_layer_ordering_is_enforced() {
        let mut req = base_request();
        req.operation = "replace".to_string();
        req.source_layer = Some(9);
        assert!(parse_run_request(&req).is_err()); // 9 > 7
        req.source_layer = Some(6);
        assert!(parse_run_request(&req).is_ok()); // 6 <= 7
    }

    #[test]
    fn zero_source_only_for_replace_and_interpolate() {
        for op in ["zero", "scale", "add-delta"] {
            let mut req = base_request();
            req.operation = op.to_string();
            req.source = "zero".to_string();
            assert!(parse_run_request(&req).is_err(), "{op} + zero source");
        }
        for op in ["replace", "interpolate"] {
            let mut req = base_request();
            req.operation = op.to_string();
            req.source = "zero".to_string();
            assert!(parse_run_request(&req).is_ok(), "{op} + zero source");
        }
    }

    #[test]
    fn matched_span_requires_text() {
        let mut req = base_request();
        req.token_kind = "matched-span".to_string();
        req.span_text = Some("   ".to_string());
        assert!(parse_run_request(&req).is_err());
        req.span_text = Some("\u{0643}\u{062a}\u{0627}\u{0628}".to_string());
        let cfg = parse_run_request(&req).unwrap();
        let (_, resolved) =
            build_and_resolve_spec(&cfg, RunKind::Intervention, "runs/gui/_test").unwrap();
        assert!(matches!(
            resolved.captures[0].tokens,
            TokenSelector::MatchedTextSpan { .. }
        ));
    }

    #[test]
    fn restore_spec_uses_restore_original() {
        let cfg = cfg_of(&base_request());
        let (_, resolved) =
            build_and_resolve_spec(&cfg, RunKind::Restore, "runs/gui/_test").unwrap();
        assert_eq!(resolved.interventions.len(), 1);
        assert!(matches!(
            resolved.interventions[0].operation,
            InterventionOperation::RestoreOriginal
        ));
        assert!(resolved.captures.is_empty());
    }

    #[test]
    fn comparison_keys_cover_shared_configuration() {
        // The comparison key deliberately ignores the intervention
        // operation (restore is compared against the same site/selection).
        let a = base_request();
        let mut same_op = base_request();
        same_op.operation = "zero".to_string();
        assert_eq!(
            cfg_of(&a).comparison_key(),
            cfg_of(&same_op).comparison_key()
        );
        // But it does cover site, layer, and token selection.
        let mut other_site = base_request();
        other_site.site = "after-attention".to_string();
        assert_ne!(
            cfg_of(&a).comparison_key(),
            cfg_of(&other_site).comparison_key()
        );
        let mut other_layer = base_request();
        other_layer.layer = Some(11);
        assert_ne!(
            cfg_of(&a).comparison_key(),
            cfg_of(&other_layer).comparison_key()
        );
        let mut other_tokens = base_request();
        other_tokens.token_kind = "matched-span".to_string();
        other_tokens.span_text = Some("\u{0643}".to_string());
        assert_ne!(
            cfg_of(&a).comparison_key(),
            cfg_of(&other_tokens).comparison_key()
        );
    }
}
