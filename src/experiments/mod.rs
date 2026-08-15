//! Experimental execution hooks for built-in research interventions.
//!
//! This API is intentionally narrow and **unstable in Ember v0.1**. It
//! supports one statically compiled experiment per generation run. It is not
//! a dynamic plugin ABI, registry, event bus, or compatibility commitment.

mod activation_patch;
mod activation_stats;
mod capture;
mod context;
mod zero_layer_output;

pub use activation_patch::{ActivationPatch, PatchTarget};
pub use activation_stats::ActivationStats;
pub use capture::CaptureSink;
pub use context::{
    ExecutionContext, ExecutionPhase, GenerationContext, LayerContext, ModelContext, ModelFamily,
    TensorAccess, TensorDType, TracingState,
};
pub use zero_layer_output::{ZeroLayerOutput, ZeroLayerOutputSpec, ZeroLayerOutputStage};

use crate::artifact::{ActivationStage, DispatchObservation, DispatchPath, ManifestExperiment};
use crate::backend::{CpuBackend, CpuError};
use crate::kv_cache::KVCache;
use crate::model::ForwardModel;
use crate::tensor::CpuTensor;
use alloc::boxed::Box;
use alloc::string::String;

/// Error reported by an experiment implementation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ExperimentError {
    message: String,
}

impl ExperimentError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<String> for ExperimentError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for ExperimentError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

/// Hook associated with a structured experiment failure.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentHook {
    ModelLoaded,
    BeforePrefill,
    BeforeLayer,
    AfterAttention,
    AfterMlp,
    AfterLayer,
    BeforeLogits,
    AfterLogits,
    GenerationComplete,
}

impl core::fmt::Display for ExperimentHook {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::ModelLoaded => "on_model_loaded",
            Self::BeforePrefill => "before_prefill",
            Self::BeforeLayer => "before_layer",
            Self::AfterAttention => "after_attention",
            Self::AfterMlp => "after_mlp",
            Self::AfterLayer => "after_layer",
            Self::BeforeLogits => "before_logits",
            Self::AfterLogits => "after_logits",
            Self::GenerationComplete => "on_generation_complete",
        };
        formatter.write_str(name)
    }
}

/// Structured context attached when an experiment hook fails.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ExperimentFailure {
    experiment_name: &'static str,
    hook: ExperimentHook,
    phase: Option<ExecutionPhase>,
    layer_index: Option<usize>,
    source: ExperimentError,
}

impl ExperimentFailure {
    fn new(
        experiment_name: &'static str,
        hook: ExperimentHook,
        phase: Option<ExecutionPhase>,
        layer_index: Option<usize>,
        source: ExperimentError,
    ) -> Self {
        Self {
            experiment_name,
            hook,
            phase,
            layer_index,
            source,
        }
    }

    #[must_use]
    pub const fn experiment_name(&self) -> &'static str {
        self.experiment_name
    }

    #[must_use]
    pub const fn hook(&self) -> ExperimentHook {
        self.hook
    }

    #[must_use]
    pub const fn phase(&self) -> Option<ExecutionPhase> {
        self.phase
    }

    #[must_use]
    pub const fn layer_index(&self) -> Option<usize> {
        self.layer_index
    }
}

impl core::fmt::Display for ExperimentFailure {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "experiment '{}' failed in {}",
            self.experiment_name, self.hook
        )?;
        if self.phase.is_some() || self.layer_index.is_some() {
            formatter.write_str(" (")?;
            if let Some(phase) = self.phase {
                write!(formatter, "phase={phase}")?;
                if self.layer_index.is_some() {
                    formatter.write_str(", ")?;
                }
            }
            if let Some(layer) = self.layer_index {
                write!(formatter, "layer={layer}")?;
            }
            formatter.write_str(")")?;
        }
        write!(formatter, ": {}", self.source)
    }
}

impl core::error::Error for ExperimentFailure {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// One statically compiled research experiment.
///
/// Every hook is observational by default. Mutation is possible only through
/// the explicitly supplied [`TensorAccess`] view.
pub trait Experiment: Send {
    fn name(&self) -> &'static str;

    /// Whether this experiment mutates hidden states through its hooks
    /// (patch/intervention experiments) rather than only observing. Drives
    /// the v0.4 HookMode recorded in execution-plan provenance.
    fn intervenes(&self) -> bool {
        false
    }

    /// Whether this experiment can observe or mutate a semantic activation
    /// stage. The conservative default is `true` so third-party/internal
    /// experiments written before this method was added never lose a hook;
    /// implementations should override it for precise fusion planning.
    fn uses_activation_stage(&self, _stage: ActivationStage) -> bool {
        true
    }

    /// Layer-specific refinement for plan de-fusion. `None` denotes a global
    /// logits hook. The default preserves the stage-wide conservative behavior.
    fn uses_activation_site(
        &self,
        stage: ActivationStage,
        _layer: Option<usize>,
        _phase: ExecutionPhase,
    ) -> bool {
        self.uses_activation_stage(stage)
    }

    /// Structured arguments describing this experiment instance, recorded in
    /// capture artifacts for provenance.
    fn arguments(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    fn on_model_loaded(&mut self, _ctx: &ModelContext<'_>) -> Result<(), ExperimentError> {
        Ok(())
    }

    fn before_prefill(&mut self, _ctx: &ExecutionContext<'_>) -> Result<(), ExperimentError> {
        Ok(())
    }

    fn before_layer(
        &mut self,
        _ctx: &LayerContext<'_>,
        _hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        Ok(())
    }

    fn after_attention(
        &mut self,
        _ctx: &LayerContext<'_>,
        _attention_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        Ok(())
    }

    fn after_mlp(
        &mut self,
        _ctx: &LayerContext<'_>,
        _mlp_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        Ok(())
    }

    fn after_layer(
        &mut self,
        _ctx: &LayerContext<'_>,
        _hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        Ok(())
    }

    fn before_logits(
        &mut self,
        _ctx: &ExecutionContext<'_>,
        _hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        Ok(())
    }

    fn after_logits(
        &mut self,
        _ctx: &ExecutionContext<'_>,
        _logits: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        Ok(())
    }

    fn on_generation_complete(
        &mut self,
        _ctx: &GenerationContext<'_>,
    ) -> Result<(), ExperimentError> {
        Ok(())
    }
}

/// Owns the single experiment active for a generation run.
/// Generate the layer-scoped forwarding methods on `ExperimentRunner`.
/// Each generated method forwards to the active experiment (via a
/// `LayerContext`) and the capture sink (with the current dispatch path),
/// wrapping failures in `ExperimentFailure` for both.
macro_rules! forward_layer_hook {
    ($method:ident, $hook:path) => {
        pub(crate) fn $method(
            &mut self,
            execution: ExecutionContext<'_>,
            layer_index: usize,
            tensor: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentFailure> {
            let name = self.name();
            let ctx = LayerContext::new(execution, layer_index);
            if let Some(experiment) = self.experiment.as_mut() {
                experiment.$method(&ctx, tensor).map_err(|source| {
                    ExperimentFailure::new(
                        name,
                        $hook,
                        Some(execution.phase),
                        Some(layer_index),
                        source,
                    )
                })?;
            }
            if let Some(capture) = self.capture.as_mut() {
                let dispatch = self.current_dispatch;
                capture
                    .$method(&execution, layer_index, tensor, dispatch)
                    .map_err(|source| {
                        ExperimentFailure::new(
                            "capture-activations",
                            $hook,
                            Some(execution.phase),
                            Some(layer_index),
                            source,
                        )
                    })?;
            }
            Ok(())
        }
    };
}

/// Generate the logits-scoped forwarding methods on `ExperimentRunner`
/// (same shape as `forward_layer_hook!` minus the layer index).
macro_rules! forward_logits_hook {
    ($method:ident, $hook:path) => {
        pub(crate) fn $method(
            &mut self,
            execution: &ExecutionContext<'_>,
            tensor: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentFailure> {
            let name = self.name();
            if let Some(experiment) = self.experiment.as_mut() {
                experiment.$method(execution, tensor).map_err(|source| {
                    ExperimentFailure::new(name, $hook, Some(execution.phase), None, source)
                })?;
            }
            if let Some(capture) = self.capture.as_mut() {
                let dispatch = self.current_dispatch;
                capture
                    .$method(execution, tensor, dispatch)
                    .map_err(|source| {
                        ExperimentFailure::new(
                            "capture-activations",
                            $hook,
                            Some(execution.phase),
                            None,
                            source,
                        )
                    })?;
            }
            Ok(())
        }
    };
}

/// Generate the no-op `LayerHooks` impl for `DisabledHooks`: four
/// layer-scoped methods and two logits-scoped methods, all `Ok(())`.
macro_rules! impl_disabled_hooks {
    ($(fn $method:ident(&mut self, $($args:tt)*) -> Result<(), E> { Ok(()) })*) => {
        impl<T, E> LayerHooks<T, E> for DisabledHooks {
            $(
                #[inline(always)]
                fn $method(&mut self, $($args)*) -> Result<(), E> {
                    Ok(())
                }
            )*
        }
    };
}

pub struct ExperimentRunner {
    experiment: Option<Box<dyn Experiment>>,
    capture: Option<CaptureSink>,
    current_dispatch: DispatchPath,
    dispatch_observations: Vec<DispatchObservation>,
}

impl ExperimentRunner {
    #[must_use]
    pub fn new(experiment: impl Experiment + 'static) -> Self {
        Self {
            experiment: Some(Box::new(experiment)),
            capture: None,
            current_dispatch: DispatchPath::Unknown,
            dispatch_observations: Vec::new(),
        }
    }

    /// A runner with no experiment, used for capture-only runs.
    #[must_use]
    pub fn capture_only(capture: CaptureSink) -> Self {
        Self {
            experiment: None,
            capture: Some(capture),
            current_dispatch: DispatchPath::Unknown,
            dispatch_observations: Vec::new(),
        }
    }

    /// Attach the v0.2 capture facility to this runner. Capture rides
    /// alongside the single experiment; it is not itself an experiment.
    #[must_use]
    pub fn with_capture(mut self, capture: CaptureSink) -> Self {
        self.capture = Some(capture);
        self
    }

    #[must_use]
    pub fn has_experiment(&self) -> bool {
        self.experiment.is_some()
    }

    #[must_use]
    pub fn has_capture(&self) -> bool {
        self.capture.is_some()
    }

    /// Whether the active experiment mutates hidden states.
    #[must_use]
    pub fn intervenes(&self) -> bool {
        self.experiment
            .as_ref()
            .is_some_and(|experiment| experiment.intervenes())
    }

    /// The v0.4 hook mode for execution-plan provenance: `Intervene` when
    /// the experiment patches hidden states, `Observe` when an experiment
    /// or capture is present, `Disabled` otherwise.
    #[must_use]
    pub(crate) fn hook_mode(&self) -> crate::plan::HookMode {
        if self.intervenes() {
            crate::plan::HookMode::Intervene
        } else if self.experiment.is_some() || self.capture.is_some() {
            crate::plan::HookMode::Observe
        } else {
            crate::plan::HookMode::Disabled
        }
    }

    /// Whether a semantic site is active for exact per-layer fusion planning.
    pub(crate) fn uses_activation_site(
        &self,
        stage: ActivationStage,
        layer: Option<usize>,
        phase: ExecutionPhase,
    ) -> bool {
        self.experiment
            .as_ref()
            .is_some_and(|experiment| experiment.uses_activation_site(stage, layer, phase))
            || self.capture.as_ref().is_some_and(|capture| {
                let phase_name = match phase {
                    ExecutionPhase::Prefill => "prefill",
                    ExecutionPhase::Decode => "decode",
                };
                capture.selection().phase.includes(phase_name)
                    && capture.selection().stages.contains(&stage)
                    && layer.is_none_or(|layer| capture.selection().layers.contains(&layer))
            })
    }

    /// Canonical plan/cache keys: a bare stage means every layer; `stage@N`
    /// names one layer. Global logits stages never carry a suffix.
    pub(crate) fn active_plan_sites(
        &self,
        layer_count: usize,
        phase: ExecutionPhase,
    ) -> Vec<String> {
        let mut sites = Vec::new();
        for stage in [
            ActivationStage::BeforeLayer,
            ActivationStage::AfterAttention,
            ActivationStage::AfterMlp,
            ActivationStage::AfterLayer,
        ] {
            let active: Vec<usize> = (0..layer_count)
                .filter(|&layer| self.uses_activation_site(stage, Some(layer), phase))
                .collect();
            if active.len() == layer_count && layer_count != 0 {
                sites.push(stage.to_string());
            } else {
                sites.extend(active.into_iter().map(|layer| format!("{stage}@{layer}")));
            }
        }
        for stage in [ActivationStage::BeforeLogits, ActivationStage::AfterLogits] {
            if self.uses_activation_site(stage, None, phase) {
                sites.push(stage.to_string());
            }
        }
        sites
    }

    /// Record the kernel/dispatch path used by the current evaluation.
    /// A run can mix paths (generic prefill, fast/workspace decode), so this
    /// is recorded per evaluation and per captured record.
    pub(crate) fn note_dispatch(&mut self, phase: ExecutionPhase, path: DispatchPath) {
        self.current_dispatch = path;
        self.dispatch_observations.push(DispatchObservation {
            phase: match phase {
                ExecutionPhase::Prefill => "prefill",
                ExecutionPhase::Decode => "decode",
            }
            .to_string(),
            dispatch: path,
        });
    }

    #[must_use]
    pub fn name(&self) -> &'static str {
        match &self.experiment {
            Some(experiment) => experiment.name(),
            None => "none",
        }
    }

    pub fn on_model_loaded(&mut self, ctx: &ModelContext<'_>) -> Result<(), ExperimentFailure> {
        let name = self.name();
        if let Some(experiment) = self.experiment.as_mut() {
            experiment.on_model_loaded(ctx).map_err(|source| {
                ExperimentFailure::new(name, ExperimentHook::ModelLoaded, None, None, source)
            })?;
        }
        if let Some(capture) = self.capture.as_mut() {
            capture.on_model_loaded(ctx).map_err(|source| {
                ExperimentFailure::new(
                    "capture-activations",
                    ExperimentHook::ModelLoaded,
                    None,
                    None,
                    source,
                )
            })?;
        }
        Ok(())
    }

    pub fn before_prefill(&mut self, ctx: &ExecutionContext<'_>) -> Result<(), ExperimentFailure> {
        let name = self.name();
        if let Some(experiment) = self.experiment.as_mut() {
            experiment.before_prefill(ctx).map_err(|source| {
                ExperimentFailure::new(
                    name,
                    ExperimentHook::BeforePrefill,
                    Some(ctx.phase),
                    None,
                    source,
                )
            })?;
        }
        Ok(())
    }

    pub fn on_generation_complete(
        &mut self,
        ctx: &GenerationContext<'_>,
    ) -> Result<(), ExperimentFailure> {
        let name = self.name();
        if let Some(experiment) = self.experiment.as_mut() {
            experiment.on_generation_complete(ctx).map_err(|source| {
                ExperimentFailure::new(name, ExperimentHook::GenerationComplete, None, None, source)
            })?;
        }
        if let Some(capture) = self.capture.as_mut() {
            let experiment_meta = ManifestExperiment {
                name: name.to_string(),
                arguments: self
                    .experiment
                    .as_ref()
                    .map(|experiment| experiment.arguments())
                    .unwrap_or_else(|| serde_json::Value::Null),
            };
            let observations = core::mem::take(&mut self.dispatch_observations);
            capture
                .finalize(ctx, experiment_meta, observations)
                .map_err(|source| {
                    ExperimentFailure::new(
                        "capture-activations",
                        ExperimentHook::GenerationComplete,
                        None,
                        None,
                        source,
                    )
                })?;
        }
        Ok(())
    }

    forward_layer_hook!(before_layer, ExperimentHook::BeforeLayer);
    forward_layer_hook!(after_attention, ExperimentHook::AfterAttention);
    forward_layer_hook!(after_mlp, ExperimentHook::AfterMlp);
    forward_layer_hook!(after_layer, ExperimentHook::AfterLayer);
    forward_logits_hook!(before_logits, ExperimentHook::BeforeLogits);
    forward_logits_hook!(after_logits, ExperimentHook::AfterLogits);
}

impl core::fmt::Debug for ExperimentRunner {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ExperimentRunner")
            .field("experiment", &self.name())
            .field("capture", &self.capture.is_some())
            .finish()
    }
}

/// Experiment-aware cached inference implemented by the supported CPU models.
///
/// This trait exists only to connect Ember's CLI to the unstable v0.1
/// experiment API. It is not a general backend or third-party model extension
/// interface.
#[doc(hidden)]
pub trait ExperimentalForwardModel: ForwardModel<CpuBackend> {
    fn forward_last_logits_with_experiment(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut KVCache,
        start_pos: usize,
        execution: ExecutionContext<'_>,
        runner: &mut ExperimentRunner,
    ) -> Result<CpuTensor, CpuError>;
}

pub(crate) trait LayerHooks<T, E> {
    fn note_dispatch(&mut self, _path: DispatchPath) {}
    fn before_layer(&mut self, layer_index: usize, tensor: &mut T) -> Result<(), E>;
    fn after_attention(&mut self, layer_index: usize, tensor: &mut T) -> Result<(), E>;
    fn after_mlp(&mut self, layer_index: usize, tensor: &mut T) -> Result<(), E>;
    fn after_layer(&mut self, layer_index: usize, tensor: &mut T) -> Result<(), E>;
    fn before_logits(&mut self, tensor: &mut T) -> Result<(), E>;
    fn after_logits(&mut self, tensor: &mut T) -> Result<(), E>;
}

pub(crate) struct DisabledHooks;

impl_disabled_hooks! {
    fn before_layer(&mut self, _layer_index: usize, _tensor: &mut T) -> Result<(), E> { Ok(()) }
    fn after_attention(&mut self, _layer_index: usize, _tensor: &mut T) -> Result<(), E> { Ok(()) }
    fn after_mlp(&mut self, _layer_index: usize, _tensor: &mut T) -> Result<(), E> { Ok(()) }
    fn after_layer(&mut self, _layer_index: usize, _tensor: &mut T) -> Result<(), E> { Ok(()) }
    fn before_logits(&mut self, _tensor: &mut T) -> Result<(), E> { Ok(()) }
    fn after_logits(&mut self, _tensor: &mut T) -> Result<(), E> { Ok(()) }
}

pub(crate) struct ActiveHooks<'runner, 'model> {
    runner: &'runner mut ExperimentRunner,
    execution: ExecutionContext<'model>,
}

impl<'runner, 'model> ActiveHooks<'runner, 'model> {
    pub(crate) fn new(
        runner: &'runner mut ExperimentRunner,
        execution: ExecutionContext<'model>,
    ) -> Self {
        Self { runner, execution }
    }

    pub(crate) fn note_dispatch_path(&mut self, path: DispatchPath) {
        self.runner.note_dispatch(self.execution.phase, path);
    }
}

pub(crate) struct SliceActivation<'a> {
    shape: [usize; 2],
    values: &'a mut [f32],
}

impl<'a> SliceActivation<'a> {
    pub(crate) fn new(rows: usize, columns: usize, values: &'a mut [f32]) -> Self {
        let expected = rows
            .checked_mul(columns)
            .expect("activation shape product overflow");
        assert_eq!(expected, values.len(), "activation shape/data mismatch");
        Self {
            shape: [rows, columns],
            values,
        }
    }
}

trait ActivationStorage {
    fn shape_2d(&self) -> [usize; 2];
    fn values_mut(&mut self) -> &mut [f32];
}

impl ActivationStorage for CpuTensor {
    fn shape_2d(&self) -> [usize; 2] {
        assert_eq!(self.shape().len(), 2, "experiment hooks require 2D tensors");
        [self.shape()[0], self.shape()[1]]
    }

    fn values_mut(&mut self) -> &mut [f32] {
        self.data_mut()
    }
}

impl ActivationStorage for SliceActivation<'_> {
    fn shape_2d(&self) -> [usize; 2] {
        self.shape
    }

    fn values_mut(&mut self) -> &mut [f32] {
        self.values
    }
}

impl<T: ActivationStorage> LayerHooks<T, CpuError> for ActiveHooks<'_, '_> {
    fn before_layer(&mut self, layer_index: usize, tensor: &mut T) -> Result<(), CpuError> {
        let [rows, columns] = tensor.shape_2d();
        let mut access = TensorAccess::new(rows, columns, tensor.values_mut());
        self.runner
            .before_layer(self.execution, layer_index, &mut access)?;
        Ok(())
    }

    fn after_attention(&mut self, layer_index: usize, tensor: &mut T) -> Result<(), CpuError> {
        let [rows, columns] = tensor.shape_2d();
        let mut access = TensorAccess::new(rows, columns, tensor.values_mut());
        self.runner
            .after_attention(self.execution, layer_index, &mut access)?;
        Ok(())
    }

    fn after_mlp(&mut self, layer_index: usize, tensor: &mut T) -> Result<(), CpuError> {
        let [rows, columns] = tensor.shape_2d();
        let mut access = TensorAccess::new(rows, columns, tensor.values_mut());
        self.runner
            .after_mlp(self.execution, layer_index, &mut access)?;
        Ok(())
    }

    fn after_layer(&mut self, layer_index: usize, tensor: &mut T) -> Result<(), CpuError> {
        let [rows, columns] = tensor.shape_2d();
        let mut access = TensorAccess::new(rows, columns, tensor.values_mut());
        self.runner
            .after_layer(self.execution, layer_index, &mut access)?;
        Ok(())
    }

    fn before_logits(&mut self, tensor: &mut T) -> Result<(), CpuError> {
        let [rows, columns] = tensor.shape_2d();
        let mut access = TensorAccess::new(rows, columns, tensor.values_mut());
        self.runner.before_logits(&self.execution, &mut access)?;
        Ok(())
    }

    fn after_logits(&mut self, tensor: &mut T) -> Result<(), CpuError> {
        let [rows, columns] = tensor.shape_2d();
        let mut access = TensorAccess::new(rows, columns, tensor.values_mut());
        self.runner.after_logits(&self.execution, &mut access)?;
        Ok(())
    }

    fn note_dispatch(&mut self, path: DispatchPath) {
        self.runner.note_dispatch(self.execution.phase, path);
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct HookRecord {
        pub hook: ExperimentHook,
        pub phase: Option<ExecutionPhase>,
        pub layer_index: Option<usize>,
        pub sequence_length: Option<usize>,
        pub shape: Option<[usize; 2]>,
    }

    pub(crate) struct RecordingExperiment {
        records: Arc<Mutex<Vec<HookRecord>>>,
    }

    impl RecordingExperiment {
        pub(crate) fn new() -> (Self, Arc<Mutex<Vec<HookRecord>>>) {
            let records = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    records: Arc::clone(&records),
                },
                records,
            )
        }

        fn record_execution(
            &self,
            hook: ExperimentHook,
            execution: &ExecutionContext<'_>,
            layer_index: Option<usize>,
            shape: Option<[usize; 2]>,
        ) {
            self.records.lock().unwrap().push(HookRecord {
                hook,
                phase: Some(execution.phase),
                layer_index,
                sequence_length: Some(execution.sequence_length),
                shape,
            });
        }

        fn record_layer(
            &self,
            hook: ExperimentHook,
            ctx: &LayerContext<'_>,
            tensor: &TensorAccess<'_>,
        ) {
            self.record_execution(
                hook,
                &ctx.execution,
                Some(ctx.layer_index),
                Some(*tensor.shape()),
            );
        }
    }

    impl Experiment for RecordingExperiment {
        fn name(&self) -> &'static str {
            "recording"
        }

        fn on_model_loaded(&mut self, _ctx: &ModelContext<'_>) -> Result<(), ExperimentError> {
            self.records.lock().unwrap().push(HookRecord {
                hook: ExperimentHook::ModelLoaded,
                phase: None,
                layer_index: None,
                sequence_length: None,
                shape: None,
            });
            Ok(())
        }

        fn before_prefill(&mut self, ctx: &ExecutionContext<'_>) -> Result<(), ExperimentError> {
            self.record_execution(ExperimentHook::BeforePrefill, ctx, None, None);
            Ok(())
        }

        fn before_layer(
            &mut self,
            ctx: &LayerContext<'_>,
            hidden: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError> {
            self.record_layer(ExperimentHook::BeforeLayer, ctx, hidden);
            Ok(())
        }

        fn after_attention(
            &mut self,
            ctx: &LayerContext<'_>,
            attention_output: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError> {
            self.record_layer(ExperimentHook::AfterAttention, ctx, attention_output);
            Ok(())
        }

        fn after_mlp(
            &mut self,
            ctx: &LayerContext<'_>,
            mlp_output: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError> {
            self.record_layer(ExperimentHook::AfterMlp, ctx, mlp_output);
            Ok(())
        }

        fn after_layer(
            &mut self,
            ctx: &LayerContext<'_>,
            hidden: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError> {
            self.record_layer(ExperimentHook::AfterLayer, ctx, hidden);
            Ok(())
        }

        fn before_logits(
            &mut self,
            ctx: &ExecutionContext<'_>,
            hidden: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError> {
            self.record_execution(
                ExperimentHook::BeforeLogits,
                ctx,
                None,
                Some(*hidden.shape()),
            );
            Ok(())
        }

        fn after_logits(
            &mut self,
            ctx: &ExecutionContext<'_>,
            logits: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError> {
            self.record_execution(
                ExperimentHook::AfterLogits,
                ctx,
                None,
                Some(*logits.shape()),
            );
            Ok(())
        }

        fn on_generation_complete(
            &mut self,
            _ctx: &GenerationContext<'_>,
        ) -> Result<(), ExperimentError> {
            self.records.lock().unwrap().push(HookRecord {
                hook: ExperimentHook::GenerationComplete,
                phase: None,
                layer_index: None,
                sequence_length: None,
                shape: None,
            });
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FailingExperiment;

    impl Experiment for FailingExperiment {
        fn name(&self) -> &'static str {
            "failing"
        }

        fn after_attention(
            &mut self,
            _ctx: &LayerContext<'_>,
            _attention_output: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError> {
            Err(ExperimentError::new("intentional failure"))
        }
    }

    #[test]
    fn runner_attaches_hook_phase_layer_and_source() {
        let model = ModelContext::new(ModelFamily::Llama, None, "llama", 2, 4);
        let execution =
            ExecutionContext::new(model, ExecutionPhase::Decode, 7, 1, TracingState::Enabled);
        let mut runner = ExperimentRunner::new(FailingExperiment);
        let mut values = [1.0; 4];
        let mut tensor = TensorAccess::new(1, 4, &mut values);
        let failure = runner
            .after_attention(execution, 1, &mut tensor)
            .unwrap_err();

        assert_eq!(failure.experiment_name(), "failing");
        assert_eq!(failure.hook(), ExperimentHook::AfterAttention);
        assert_eq!(failure.phase(), Some(ExecutionPhase::Decode));
        assert_eq!(failure.layer_index(), Some(1));
        assert_eq!(
            failure.to_string(),
            "experiment 'failing' failed in after_attention (phase=decode, layer=1): intentional failure"
        );
    }

    #[test]
    fn execution_context_distinguishes_prefill_and_decode_positions() {
        let model = ModelContext::new(ModelFamily::Qwen3, None, "qwen3", 28, 1024);
        let prefill =
            ExecutionContext::new(model, ExecutionPhase::Prefill, 0, 6, TracingState::Disabled);
        let decode =
            ExecutionContext::new(model, ExecutionPhase::Decode, 6, 1, TracingState::Enabled);

        assert_eq!(prefill.sequence_length, 6);
        assert_eq!(prefill.token_position(), None);
        assert_eq!(decode.sequence_length, 7);
        assert_eq!(decode.token_position(), Some(6));
        assert!(decode.tracing.is_enabled());
    }

    #[test]
    fn runner_preserves_complete_lifecycle_order() {
        let model = ModelContext::new(ModelFamily::Gemma4, None, "gemma4", 1, 4);
        let execution =
            ExecutionContext::new(model, ExecutionPhase::Prefill, 0, 2, TracingState::Disabled);
        let (experiment, records) = test_support::RecordingExperiment::new();
        let mut runner = ExperimentRunner::new(experiment);
        runner.on_model_loaded(&model).unwrap();
        runner.before_prefill(&execution).unwrap();

        let mut values = [1.0; 8];
        let mut tensor = TensorAccess::new(2, 4, &mut values);
        runner.before_layer(execution, 0, &mut tensor).unwrap();
        runner.after_attention(execution, 0, &mut tensor).unwrap();
        runner.after_mlp(execution, 0, &mut tensor).unwrap();
        runner.after_layer(execution, 0, &mut tensor).unwrap();
        runner.before_logits(&execution, &mut tensor).unwrap();
        runner.after_logits(&execution, &mut tensor).unwrap();
        runner
            .on_generation_complete(&GenerationContext::new(
                model,
                2,
                1,
                0,
                TracingState::Disabled,
                &[],
                &[],
            ))
            .unwrap();

        assert_eq!(
            records
                .lock()
                .unwrap()
                .iter()
                .map(|record| record.hook)
                .collect::<Vec<_>>(),
            [
                ExperimentHook::ModelLoaded,
                ExperimentHook::BeforePrefill,
                ExperimentHook::BeforeLayer,
                ExperimentHook::AfterAttention,
                ExperimentHook::AfterMlp,
                ExperimentHook::AfterLayer,
                ExperimentHook::BeforeLogits,
                ExperimentHook::AfterLogits,
                ExperimentHook::GenerationComplete,
            ]
        );
    }
}
