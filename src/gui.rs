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

use crate::cli_experiment::DecodeRoute;
use crate::cli_experiment::{execute_prepared, prepare_run, PreparedRun, RunOutcome, RunTarget};
use crate::cli_experiment_shared::{
    batch_ineligibility, run_base_pass, run_batched_pass, run_variant, BasePass, SharedPrefix,
};
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
    /// Where run bundles go (`runs/gui`; tests point it elsewhere).
    output_root: String,
    #[cfg_attr(
        not(feature = "gui"),
        allow(dead_code, reason = "only the `gui` feature uses it")
    )]
    /// The shared pass of the layer sweep in progress (see
    /// `run_sweep_point`).
    sweep_cache: Option<SweepCache>,
    /// Cooperative cancellation for the run in flight (native console only;
    /// the web console never sets it). Checked at every decode step of each
    /// leg, between the legs, and before a bundle is written.
    cancel: Option<ember::cancel::CancelToken>,
}

#[cfg_attr(
    not(feature = "gui"),
    allow(dead_code, reason = "only the `gui` feature uses it")
)]
/// One layer sweep's shared pass: every layer's baseline bundle (the
/// baselines differ only in which layer their source capture records, so
/// one generation serves them all) and the recorded prompt prefix each
/// layer's intervention run resumes from. Points are taken out as they run.
struct SweepCache {
    key: String,
    model_path: String,
    baselines: std::collections::BTreeMap<usize, (RunOutput, String)>,
    /// The recorded prefix pending intervention runs resume from (`None`
    /// when the pass batched every intervention run).
    prefix: Option<SharedPrefix>,
    /// layer -> its intervention run, pending or already computed.
    variants: std::collections::BTreeMap<usize, SweepVariant>,
    /// Time the pass took, charged to the first point's baseline.
    pass_ms: f64,
}

/// One layer's intervention run in a sweep pass.
enum SweepVariant {
    /// Runs on request from the recorded prefix: (variant index in the
    /// pass, spec text, spec, output dir).
    Pending(
        usize,
        String,
        Box<ember::v05::spec::ExperimentSpecV1>,
        String,
    ),
    /// Decoded in the pass's batch; its bundle is written.
    Finished(Box<RunOutcome>),
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
    /// Compact chart-ready metrics derived from the two verified bundles.
    /// Raw activation tensors remain bundle-owned and are never cloned into
    /// the presentation layer.
    pub comparison: ExperimentComparison,
    pub verification: VerificationReport,
    pub elapsed_ms_total: f64,
    pub elapsed_ms_baseline: f64,
    pub baseline_key: String,
}

/// UI-facing comparison of one deterministic baseline/intervention pair.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExperimentComparison {
    pub layers: Vec<LayerMetric>,
    pub tokens: Vec<TokenMetric>,
    pub first_token_divergence: Option<usize>,
    pub generated_tokens_equal: bool,
    pub generated_text_equal: bool,
    /// High-value landmarks derived once on the worker so the overview can
    /// answer where divergence begins and whether token IDs later re-align.
    pub landmarks: DivergenceLandmarks,
    /// Reserved row-major layer × token metrics. Phase 1 deliberately leaves
    /// this absent until a capture configuration records a defensible matrix.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_token_grid: Option<LayerTokenGrid>,
}

/// Compact experiment landmarks for the overview. A stable token tail means
/// the remaining emitted token IDs are exactly equal; it deliberately does
/// not claim that two autoregressive trajectories semantically reconverged.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct DivergenceLandmarks {
    pub first_layer_divergence: Option<usize>,
    pub peak_layer: Option<usize>,
    pub peak_relative_l2: Option<f64>,
    /// One-based decode step at which a non-empty equal suffix begins.
    pub stable_token_tail_step: Option<usize>,
}

/// Compact, renderer-independent groundwork for a future native heatmap.
/// `values` is row-major over `layers × token_positions`; missing cells stay
/// `None` so sparse captures are represented without fabricated zeros.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct LayerTokenGrid {
    pub metric_name: String,
    pub layers: Vec<usize>,
    pub token_positions: Vec<usize>,
    pub values: Vec<Option<f64>>,
}

/// One sparse point in the residual-stream divergence trace.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct LayerMetric {
    pub layer: usize,
    /// Relative L2 distance, using the baseline activation as the reference.
    pub relative_l2_difference: Option<f64>,
    pub cosine_distance: Option<f64>,
    pub maximum_absolute_difference: Option<f64>,
    pub exact: bool,
}

/// Token-id pairing for the generated continuation. Text stays in
/// `RunOutput`; this compact sequence supports exact divergence markers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct TokenMetric {
    /// One-based decode step.
    pub position: usize,
    pub baseline_token_id: Option<u32>,
    pub intervention_token_id: Option<u32>,
    pub baseline_text: Option<String>,
    pub intervention_text: Option<String>,
    pub differs: bool,
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
            output_root: "runs/gui".to_string(),
            sweep_cache: None,
            cancel: None,
        }
    }

    /// Attach (or clear) the cancel token the next runs honour.
    #[cfg(feature = "gui")]
    pub(crate) fn set_cancel(&mut self, token: Option<ember::cancel::CancelToken>) {
        self.cancel = token;
    }

    fn cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(ember::cancel::CancelToken::is_cancelled)
    }

    /// Load (or reuse) the model for `model_path`.
    pub(crate) fn ensure_prepared(&mut self, model_path: &str) -> Result<(), String> {
        if let Some(prepared) = &self.prepared
            && prepared.model_path.to_string_lossy() == model_path
        {
            return Ok(());
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
        // A recorded prefix belongs to the model that computed it.
        self.end_sweep();
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
    /// through the shared pass (the intervention resumes from the
    /// baseline's prefix, and on the Q8_0 fast path both decode in one
    /// batch); the bundles equal `execute_prepared`'s. The baseline text is
    /// recorded for later restore comparison.
    pub(crate) fn run_baseline_intervention(
        &mut self,
        cfg: &RunConfig,
    ) -> Result<RunBundle, String> {
        let started = std::time::Instant::now();
        self.ensure_prepared(&cfg.model_path)?;
        let baseline_dir = next_output_dir(self, "baseline");
        let intervention_dir = next_output_dir(self, "intervention");
        let (baseline_text, baseline_spec) =
            build_and_resolve_spec(cfg, RunKind::Baseline, &baseline_dir)?;
        let (intervention_text, intervention_spec) =
            build_and_resolve_spec(cfg, RunKind::Intervention, &intervention_dir)?;
        let cancel = self.cancel.clone();
        let prepared = self
            .prepared
            .as_mut()
            .ok_or_else(|| "model session is not prepared".to_string())?;
        let baseline_target = RunTarget {
            resolved: &baseline_spec,
            spec_text: &baseline_text,
            output_directory: Path::new(&baseline_dir),
            retain_incomplete: false,
        };
        let intervention_target = RunTarget {
            resolved: &intervention_spec,
            spec_text: &intervention_text,
            output_directory: Path::new(&intervention_dir),
            retain_incomplete: false,
        };
        let batch = batch_ineligibility(prepared, &baseline_target, &[], &[intervention_target])
            .map_err(|error| format!("baseline run failed: {error:#}"))?;
        let (baseline, intervention, elapsed_ms_baseline) = match batch {
            // Both runs decode together: the intervention resumes from the
            // baseline's prefix and each decode step is one batched forward.
            // A cancelled pair writes neither bundle.
            None => {
                let (base, _, mut variants) = match run_batched_pass(
                    prepared,
                    baseline_target,
                    &[],
                    &[intervention_target],
                    cancel.as_ref(),
                ) {
                    Ok(outcomes) => outcomes,
                    Err(_)
                        if cancel
                            .as_ref()
                            .is_some_and(ember::cancel::CancelToken::is_cancelled) =>
                    {
                        return Err("cancelled".to_string())
                    }
                    Err(error) => {
                        return Err(format!("baseline/intervention run failed: {error:#}"))
                    }
                };
                let (baseline, _baseline_report) = run_output(prepared, base)?;
                let elapsed_ms_baseline = started.elapsed().as_secs_f64() * 1000.0;
                (baseline, variants.remove(0), elapsed_ms_baseline)
            }
            Some(reason) => {
                // The baseline computes the prompt prefix once; the
                // intervention run starts its prefill at the intervened
                // block from that state (bit-identical to a full recompute;
                // see `ember::v05::prefix`).
                let BasePass {
                    base, mut prefix, ..
                } = run_base_pass(
                    prepared,
                    baseline_target,
                    &[],
                    &[&intervention_spec],
                    DecodeRoute::Sequential { reason },
                    cancel.as_ref(),
                )
                .map_err(|error| format!("baseline run failed: {error:#}"))?;
                let (baseline, _baseline_report) = run_output(prepared, base)?;
                let elapsed_ms_baseline = started.elapsed().as_secs_f64() * 1000.0;
                // A cancelled pair leaves nothing behind, like a cancelled CLI
                // run: the finished baseline bundle is half an experiment, so
                // it goes too.
                let discard_baseline = || {
                    let _ = std::fs::remove_dir_all(&baseline.bundle_dir);
                };
                if self.cancelled() {
                    discard_baseline();
                    return Err("cancelled".to_string());
                }
                let cancel = self.cancel.clone();
                let prepared = self
                    .prepared
                    .as_mut()
                    .ok_or_else(|| "model session is not prepared".to_string())?;
                match run_variant(
                    prepared,
                    &mut prefix,
                    0,
                    intervention_target,
                    cancel.as_ref(),
                ) {
                    Ok(outcome) => (baseline, outcome, elapsed_ms_baseline),
                    Err(error) => {
                        if self.cancelled() {
                            discard_baseline();
                            return Err("cancelled".to_string());
                        }
                        return Err(format!("intervention run failed: {error:#}"));
                    }
                }
            }
        };
        let prepared = self
            .prepared
            .as_mut()
            .ok_or_else(|| "model session is not prepared".to_string())?;
        let (intervention, intervention_report) = run_output(prepared, intervention)?;
        let raw_comparison = ember::v05::compare::compare_bundles(
            Path::new(&baseline.bundle_dir),
            Path::new(&intervention.bundle_dir),
        )
        .map_err(|error| format!("baseline/intervention comparison failed: {error}"))?;
        let comparison = derive_experiment_comparison(
            &raw_comparison,
            &baseline.generated_token_ids,
            &baseline.generated_token_texts,
            &intervention.generated_token_ids,
            &intervention.generated_token_texts,
        );
        let elapsed_ms_total = started.elapsed().as_secs_f64() * 1000.0;
        let baseline_key = cfg.comparison_key();
        self.last_baseline = Some(BaselineRecord {
            text: baseline.text.clone(),
            config_key: baseline_key.clone(),
        });
        Ok(RunBundle {
            baseline,
            intervention,
            comparison,
            verification: intervention_report,
            elapsed_ms_total,
            elapsed_ms_baseline,
            baseline_key,
        })
    }

    #[cfg_attr(
        not(feature = "gui"),
        allow(dead_code, reason = "only the `gui` feature uses it")
    )]
    /// Run one point of a layer sweep: the same baseline/intervention pair
    /// `run_baseline_intervention` produces for `cfg` (bit-identical
    /// bundles), but the first point of a sweep runs one shared pass for
    /// every layer in `planned`: one generation writes every layer's
    /// baseline bundle and records the prompt prefix, and each point's
    /// intervention run then resumes from it at its own layer. On the Q8_0
    /// fast path the pass also runs every intervention, all decoding in one
    /// batch, and later points only read their finished bundles.
    pub(crate) fn run_sweep_point(
        &mut self,
        cfg: &RunConfig,
        planned: &[usize],
    ) -> Result<RunBundle, String> {
        let started = std::time::Instant::now();
        self.ensure_prepared(&cfg.model_path)?;
        let layer = cfg
            .layer
            .ok_or_else(|| "a sweep point needs a layer".to_string())?;
        let key = cfg.sweep_key();
        let reusable = self.sweep_cache.as_ref().is_some_and(|cache| {
            cache.key == key
                && cache.model_path == cfg.model_path
                && cache.variants.contains_key(&layer)
        });
        if !reusable {
            self.end_sweep();
            let mut layers: Vec<usize> = planned.to_vec();
            if !layers.contains(&layer) {
                layers.insert(0, layer);
            }
            self.sweep_cache = Some(self.build_sweep_cache(cfg, layer, &layers)?);
        }
        let mut cache = self
            .sweep_cache
            .take()
            .ok_or_else(|| "the sweep pass is missing".to_string())?;
        let pass_ms = std::mem::take(&mut cache.pass_ms);
        let result = self.finish_sweep_point(cfg, layer, &mut cache, started, pass_ms);
        if self.cancelled() {
            // Nothing of a cancelled sweep is kept: this point's baseline
            // (the worker discards a finished pair) and every layer's
            // baseline that will now never be shown.
            if let Some((baseline, _)) = cache.baselines.get(&layer)
                && result.is_err()
            {
                let _ = std::fs::remove_dir_all(&baseline.bundle_dir);
            }
            discard_unused_baselines(&cache);
            return result.and(Err("cancelled".to_string()));
        }
        if !cache.variants.is_empty() && result.is_ok() {
            self.sweep_cache = Some(cache);
        } else {
            discard_unused_baselines(&cache);
        }
        result
    }

    /// The sweep is over (finished, stopped or abandoned): remove the
    /// baseline bundles of layers that never ran.
    #[cfg_attr(
        not(feature = "gui"),
        allow(dead_code, reason = "only the `gui` feature uses it")
    )]
    pub(crate) fn end_sweep(&mut self) {
        if let Some(cache) = self.sweep_cache.take() {
            discard_unused_baselines(&cache);
        }
    }

    #[cfg_attr(
        not(feature = "gui"),
        allow(dead_code, reason = "only the `gui` feature uses it")
    )]
    fn build_sweep_cache(
        &mut self,
        cfg: &RunConfig,
        first: usize,
        layers: &[usize],
    ) -> Result<SweepCache, String> {
        let started = std::time::Instant::now();
        // Specs for every layer the form accepts; a layer whose spec does
        // not build is left out (the console skips it too), except the one
        // being run now, whose error is the point's error.
        let mut built = Vec::new();
        for &layer in layers {
            let mut point = cfg.clone();
            point.layer = Some(layer);
            let baseline_dir = next_output_dir(self, "baseline");
            let intervention_dir = next_output_dir(self, "intervention");
            let specs = build_and_resolve_spec(&point, RunKind::Baseline, &baseline_dir).and_then(
                |baseline| {
                    build_and_resolve_spec(&point, RunKind::Intervention, &intervention_dir)
                        .map(|intervention| (baseline, intervention))
                },
            );
            match specs {
                Ok((baseline, intervention)) => built.push((
                    layer,
                    point.comparison_key(),
                    baseline,
                    baseline_dir,
                    intervention,
                    intervention_dir,
                )),
                Err(error) if layer == first => return Err(error),
                Err(_) => {}
            }
        }
        // The point being run first is the pass's base.
        let first_index = built
            .iter()
            .position(|entry| entry.0 == first)
            .ok_or_else(|| "the sweep has no runnable layer".to_string())?;
        built.swap(0, first_index);
        let cancel = self.cancel.clone();
        let prepared = self
            .prepared
            .as_mut()
            .ok_or_else(|| "model session is not prepared".to_string())?;
        let target = |spec: &(String, ember::v05::spec::ExperimentSpecV1), dir: &str| {
            (spec.1.clone(), spec.0.clone(), dir.to_string())
        };
        let baselines: Vec<_> = built
            .iter()
            .map(|entry| target(&entry.2, &entry.3))
            .collect();
        let interventions: Vec<_> = built
            .iter()
            .map(|entry| target(&entry.4, &entry.5))
            .collect();
        let co: Vec<RunTarget<'_>> = baselines.iter().skip(1).map(run_target).collect();
        let variant_targets: Vec<RunTarget<'_>> = interventions.iter().map(run_target).collect();
        let batch =
            batch_ineligibility(prepared, &run_target(&baselines[0]), &co, &variant_targets)
                .map_err(|error| format!("baseline run failed: {error:#}"))?;
        let mut outputs = std::collections::BTreeMap::new();
        let (prefix, variants) = match batch {
            // Every layer's baseline and intervention in one pass whose
            // decode steps are batched; each point then only reads its
            // finished bundles.
            None => {
                let (base, co, finished) = run_batched_pass(
                    prepared,
                    run_target(&baselines[0]),
                    &co,
                    &variant_targets,
                    cancel.as_ref(),
                )
                .map_err(|error| format!("baseline run failed: {error:#}"))?;
                for (entry, outcome) in built.iter().zip(std::iter::once(base).chain(co)) {
                    let (output, _) = run_output(prepared, outcome)?;
                    outputs.insert(entry.0, (output, entry.1.clone()));
                }
                let variants = built
                    .iter()
                    .zip(finished)
                    .map(|(entry, outcome)| (entry.0, SweepVariant::Finished(Box::new(outcome))))
                    .collect();
                (None, variants)
            }
            Some(reason) => {
                let variant_specs: Vec<&ember::v05::spec::ExperimentSpecV1> =
                    interventions.iter().map(|(spec, _, _)| spec).collect();
                let BasePass { base, co, prefix } = run_base_pass(
                    prepared,
                    run_target(&baselines[0]),
                    &co,
                    &variant_specs,
                    DecodeRoute::Sequential { reason },
                    cancel.as_ref(),
                )
                .map_err(|error| format!("baseline run failed: {error:#}"))?;
                for (entry, outcome) in built.iter().zip(std::iter::once(base).chain(co)) {
                    let (output, _) = run_output(prepared, outcome)?;
                    outputs.insert(entry.0, (output, entry.1.clone()));
                }
                let variants = built
                    .iter()
                    .zip(interventions.iter().cloned())
                    .enumerate()
                    .map(|(index, (entry, (spec, text, dir)))| {
                        (
                            entry.0,
                            SweepVariant::Pending(index, text, Box::new(spec), dir),
                        )
                    })
                    .collect();
                (Some(prefix), variants)
            }
        };
        Ok(SweepCache {
            key: cfg.sweep_key(),
            model_path: cfg.model_path.clone(),
            baselines: outputs,
            prefix,
            variants,
            pass_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[cfg_attr(
        not(feature = "gui"),
        allow(dead_code, reason = "only the `gui` feature uses it")
    )]
    fn finish_sweep_point(
        &mut self,
        cfg: &RunConfig,
        layer: usize,
        cache: &mut SweepCache,
        started: std::time::Instant,
        pass_ms: f64,
    ) -> Result<RunBundle, String> {
        let (baseline, baseline_key) = cache
            .baselines
            .get(&layer)
            .cloned()
            .ok_or_else(|| format!("the sweep pass has no baseline for layer {layer}"))?;
        let variant = cache
            .variants
            .remove(&layer)
            .ok_or_else(|| format!("layer {layer} has already run in this sweep"))?;
        let elapsed_ms_baseline = if pass_ms > 0.0 { pass_ms } else { 0.0 };
        if self.cancelled() {
            return Err("cancelled".to_string());
        }
        let cancel = self.cancel.clone();
        let prepared = self
            .prepared
            .as_mut()
            .ok_or_else(|| "model session is not prepared".to_string())?;
        let outcome = match variant {
            SweepVariant::Finished(outcome) => *outcome,
            SweepVariant::Pending(index, text, spec, dir) => run_variant(
                prepared,
                cache
                    .prefix
                    .as_mut()
                    .ok_or_else(|| "the sweep pass recorded no prefix".to_string())?,
                index,
                RunTarget {
                    resolved: &spec,
                    spec_text: &text,
                    output_directory: Path::new(&dir),
                    retain_incomplete: false,
                },
                cancel.as_ref(),
            )
            .map_err(|error| format!("intervention run failed: {error:#}"))?,
        };
        let (intervention, intervention_report) = run_output(prepared, outcome)?;
        let raw_comparison = ember::v05::compare::compare_bundles(
            Path::new(&baseline.bundle_dir),
            Path::new(&intervention.bundle_dir),
        )
        .map_err(|error| format!("baseline/intervention comparison failed: {error}"))?;
        let comparison = derive_experiment_comparison(
            &raw_comparison,
            &baseline.generated_token_ids,
            &baseline.generated_token_texts,
            &intervention.generated_token_ids,
            &intervention.generated_token_texts,
        );
        debug_assert_eq!(baseline_key, cfg.comparison_key());
        self.last_baseline = Some(BaselineRecord {
            text: baseline.text.clone(),
            config_key: baseline_key.clone(),
        });
        Ok(RunBundle {
            baseline,
            intervention,
            comparison,
            verification: intervention_report,
            elapsed_ms_total: started.elapsed().as_secs_f64() * 1000.0,
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

/// Remove the bundles of sweep layers that were never shown: their
/// baselines, and their intervention bundles when the pass computed them.
#[cfg_attr(
    not(feature = "gui"),
    allow(dead_code, reason = "only the `gui` feature uses it")
)]
fn discard_unused_baselines(cache: &SweepCache) {
    for (layer, variant) in &cache.variants {
        if let Some((baseline, _)) = cache.baselines.get(layer) {
            let _ = std::fs::remove_dir_all(&baseline.bundle_dir);
        }
        if let SweepVariant::Finished(outcome) = variant {
            let _ = std::fs::remove_dir_all(&outcome.path);
        }
    }
}

/// A GUI run's (spec, text, output directory) as a shared-pass target.
fn run_target(
    (spec, text, dir): &(ember::v05::spec::ExperimentSpecV1, String, String),
) -> RunTarget<'_> {
    RunTarget {
        resolved: spec,
        spec_text: text,
        output_directory: Path::new(dir.as_str()),
        retain_incomplete: false,
    }
}

/// A minimal resolved spec carrying just the model path; `prepare_run`
/// reads everything it needs from it (tokenizer resolution and provenance
/// hashes come from defaults, exactly like a user-authored spec).
/// Shared raw-spec skeleton for the console: same model/execution/generation
/// boilerplate with per-run name, prompt, payloads, and output directory.
#[allow(
    clippy::too_many_arguments,
    reason = "application call path: each argument is a separate input"
)]
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
        captures: Some(
            captures
                .into_iter()
                .map(|capture| {
                    ember::v05::spec::RawDefinition::explicit(capture).expect("capture serializes")
                })
                .collect(),
        ),
        interventions: Some(
            interventions
                .into_iter()
                .map(|intervention| {
                    ember::v05::spec::RawDefinition::explicit(intervention)
                        .expect("intervention serializes")
                })
                .collect(),
        ),
        output: RawOutputSpec {
            directory: output_dir,
            tensor_format: Some("safetensors".to_string()),
            overwrite: Some(overwrite),
        },
        sweep: None,
        attribution: None,
        probe: None,
        dictionaries: None,
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
    pub generated_token_ids: Vec<u32>,
    pub generated_token_texts: Vec<String>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub bundle_dir: String,
    pub semantic_hash: String,
    pub payload_hash: String,
    pub wall_ms: f64,
    pub decode_tps: Option<f64>,
    pub events: Vec<serde_json::Value>,
}

const GUI_LAYER_TRACE_CAPTURE_ID: &str = "gui-layer-trace";

fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn derive_divergence_landmarks(
    layers: &[LayerMetric],
    baseline_tokens: &[u32],
    intervention_tokens: &[u32],
) -> DivergenceLandmarks {
    let divergent_layers = layers.iter().filter_map(|metric| {
        metric
            .relative_l2_difference
            .filter(|value| !metric.exact && *value > 0.0)
            .map(|value| (metric.layer, value))
    });
    let first_layer_divergence = divergent_layers.clone().map(|(layer, _)| layer).next();
    let peak = divergent_layers.max_by(|(_, left), (_, right)| left.total_cmp(right));

    let stable_token_tail_step = if baseline_tokens.len() == intervention_tokens.len() {
        baseline_tokens
            .iter()
            .zip(intervention_tokens)
            .position(|(baseline, intervention)| baseline != intervention)
            .and_then(|first_difference| {
                ((first_difference + 1)..baseline_tokens.len())
                    .find(|start| baseline_tokens[*start..] == intervention_tokens[*start..])
                    .map(|start| start + 1)
            })
    } else {
        None
    };

    DivergenceLandmarks {
        first_layer_divergence,
        peak_layer: peak.map(|(layer, _)| layer),
        peak_relative_l2: peak.map(|(_, value)| value),
        stable_token_tail_step,
    }
}

fn derive_experiment_comparison(
    comparison: &ember::v05::compare::CompareResult,
    baseline_tokens: &[u32],
    baseline_token_texts: &[String],
    intervention_tokens: &[u32],
    intervention_token_texts: &[String],
) -> ExperimentComparison {
    let mut layers: Vec<LayerMetric> = comparison
        .captures
        .iter()
        .filter(|capture| capture.capture_id == GUI_LAYER_TRACE_CAPTURE_ID)
        .map(|capture| {
            let metrics = capture.metrics.as_ref();
            let exact = metrics.is_some_and(|metrics| metrics.exact);
            let relative_l2_difference = metrics
                .and_then(|metrics| finite(metrics.relative_l2_difference))
                .or(exact.then_some(0.0));
            let cosine_distance = metrics
                .and_then(|metrics| finite(metrics.cosine_similarity))
                .map(|similarity| 1.0 - similarity.clamp(-1.0, 1.0))
                .or(exact.then_some(0.0));
            LayerMetric {
                layer: capture.layer,
                relative_l2_difference,
                cosine_distance,
                maximum_absolute_difference: metrics
                    .and_then(|metrics| finite(metrics.maximum_absolute_difference)),
                exact,
            }
        })
        .collect();
    layers.sort_by_key(|metric| metric.layer);

    let token_count = baseline_tokens.len().max(intervention_tokens.len());
    let tokens = (0..token_count)
        .map(|index| {
            let baseline_token_id = baseline_tokens.get(index).copied();
            let intervention_token_id = intervention_tokens.get(index).copied();
            TokenMetric {
                position: index + 1,
                baseline_token_id,
                intervention_token_id,
                baseline_text: baseline_token_texts.get(index).cloned(),
                intervention_text: intervention_token_texts.get(index).cloned(),
                differs: baseline_token_id != intervention_token_id,
            }
        })
        .collect();
    let output = comparison.outputs.first();
    let landmarks = derive_divergence_landmarks(&layers, baseline_tokens, intervention_tokens);
    ExperimentComparison {
        layers,
        tokens,
        first_token_divergence: output.and_then(|output| output.first_divergence_step),
        generated_tokens_equal: output.is_some_and(|output| output.generated_tokens_equal),
        generated_text_equal: output.is_some_and(|output| output.generated_text_equal),
        landmarks,
        layer_token_grid: None,
    }
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
    #[cfg_attr(
        not(feature = "gui"),
        allow(dead_code, reason = "only the `gui` feature uses it")
    )]
    /// Key over everything but the layer: the points of one layer sweep.
    fn sweep_key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{:?}|{:?}|{}|{}|{:?}|{:?}|{:?}|{}",
            self.model_path,
            self.prompt,
            self.max_new_tokens,
            self.execution.name(),
            self.site,
            self.operation,
            self.factor.to_bits(),
            self.alpha.to_bits(),
            self.source,
            self.source_layer,
            self.token,
            self.span_text,
        )
    }

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
                "unknown hook stage '{}'; expected one of: {}",
                req.site,
                hook_stages().join(", ")
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
    let execution = ExecutionMode::from_cli(&req.execution)?;
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
/// (`SemanticHookSite::stage_id`), used by both consoles' hook selectors.
pub(crate) fn hook_stages() -> Vec<&'static str> {
    SemanticHookSite::ALL
        .iter()
        .map(|site| site.stage_id())
        .collect()
}

/// The hook stage ids that carry a layer, for the browser console's layer
/// control (from `SemanticHookSite::is_per_layer`, not a copied table).
pub(crate) fn per_layer_hook_stages() -> Vec<&'static str> {
    SemanticHookSite::ALL
        .iter()
        .filter(|site| site.is_per_layer())
        .map(|site| site.stage_id())
        .collect()
}

mod server;
mod spec;
#[cfg(feature = "gui")]
pub(crate) use server::discover_models;
#[cfg(feature = "gui")]
use server::model_basename;
pub(crate) use server::run_gui_command;
pub(crate) use spec::*;

// ---------------------------------------------------------------------------
// execution
// ---------------------------------------------------------------------------

fn next_output_dir(session: &mut GuiSession, kind: &str) -> String {
    session.run_counter += 1;
    format!(
        "{}/{kind}-{}-{:02}",
        session.output_root,
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
    let cancel = session.cancel.clone();
    let prepared = session
        .prepared
        .as_mut()
        .ok_or_else(|| "model session is not prepared".to_string())?;
    let (spec_text, resolved) = build_and_resolve_spec(cfg, kind, &output_dir)?;
    let (path, identity, report, results) = execute_prepared(
        prepared,
        &resolved,
        &spec_text,
        Path::new(&output_dir),
        false,
        cancel.as_ref(),
    )
    .map_err(|error| format!("{error:#}"))?;
    run_output(
        prepared,
        RunOutcome {
            path,
            identity,
            report,
            results,
            prefix: None,
        },
    )
}

/// The GUI's view of a written bundle.
fn run_output(
    prepared: &PreparedRun,
    outcome: RunOutcome,
) -> Result<(RunOutput, VerificationReport), String> {
    let RunOutcome {
        path,
        identity,
        report,
        results,
        ..
    } = outcome;
    let result: &InputResult = results
        .first()
        .ok_or_else(|| "the run produced no input results".to_string())?;
    let (wall_ms, decode_tps) = read_runtime_metrics(&path);
    let generated_token_texts: Vec<String> = result
        .generated_token_ids
        .iter()
        .map(|token_id| {
            prepared
                .tokenizer
                .decode(&[*token_id])
                .unwrap_or_else(|_| format!("<{token_id}>"))
        })
        .collect();
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
            generated_token_ids: result.generated_token_ids.clone(),
            generated_token_texts,
            prompt_tokens: result.tokenization.token_ids.len(),
            generated_tokens: result.generated_token_ids.len(),
            bundle_dir: path.display().to_string(),
            semantic_hash: identity.semantic_hash.clone(),
            payload_hash: identity.payload_hash,
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

#[cfg(test)]
mod tests;
