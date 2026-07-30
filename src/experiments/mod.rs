//! Experimental execution hooks for built-in research interventions.
//!
//! This API is intentionally narrow and **unstable in Ember v0.1**. It
//! supports one statically compiled experiment per generation run. It is not
//! a dynamic plugin ABI, registry, event bus, or compatibility commitment.

mod context;
mod zero_layer_output;

pub use context::{
    ExecutionContext, ExecutionPhase, GenerationContext, LayerContext, ModelContext, ModelFamily,
    TensorAccess, TensorDType, TracingState,
};
pub use zero_layer_output::{ZeroLayerOutput, ZeroLayerOutputSpec, ZeroLayerOutputStage};

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
pub struct ExperimentRunner {
    experiment: Box<dyn Experiment>,
}

impl ExperimentRunner {
    #[must_use]
    pub fn new(experiment: impl Experiment + 'static) -> Self {
        Self {
            experiment: Box::new(experiment),
        }
    }

    #[must_use]
    pub fn name(&self) -> &'static str {
        self.experiment.name()
    }

    pub fn on_model_loaded(&mut self, ctx: &ModelContext<'_>) -> Result<(), ExperimentFailure> {
        let name = self.name();
        self.experiment.on_model_loaded(ctx).map_err(|source| {
            ExperimentFailure::new(name, ExperimentHook::ModelLoaded, None, None, source)
        })
    }

    pub fn before_prefill(&mut self, ctx: &ExecutionContext<'_>) -> Result<(), ExperimentFailure> {
        let name = self.name();
        self.experiment.before_prefill(ctx).map_err(|source| {
            ExperimentFailure::new(
                name,
                ExperimentHook::BeforePrefill,
                Some(ctx.phase),
                None,
                source,
            )
        })
    }

    pub fn on_generation_complete(
        &mut self,
        ctx: &GenerationContext<'_>,
    ) -> Result<(), ExperimentFailure> {
        let name = self.name();
        self.experiment
            .on_generation_complete(ctx)
            .map_err(|source| {
                ExperimentFailure::new(name, ExperimentHook::GenerationComplete, None, None, source)
            })
    }

    #[allow(dead_code)] // Wired into model execution in the next implementation commit.
    pub(crate) fn before_layer(
        &mut self,
        execution: ExecutionContext<'_>,
        layer_index: usize,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentFailure> {
        let name = self.name();
        let ctx = LayerContext::new(execution, layer_index);
        self.experiment
            .before_layer(&ctx, tensor)
            .map_err(|source| {
                ExperimentFailure::new(
                    name,
                    ExperimentHook::BeforeLayer,
                    Some(execution.phase),
                    Some(layer_index),
                    source,
                )
            })
    }

    #[allow(dead_code)] // Wired into model execution in the next implementation commit.
    pub(crate) fn after_attention(
        &mut self,
        execution: ExecutionContext<'_>,
        layer_index: usize,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentFailure> {
        let name = self.name();
        let ctx = LayerContext::new(execution, layer_index);
        self.experiment
            .after_attention(&ctx, tensor)
            .map_err(|source| {
                ExperimentFailure::new(
                    name,
                    ExperimentHook::AfterAttention,
                    Some(execution.phase),
                    Some(layer_index),
                    source,
                )
            })
    }

    #[allow(dead_code)] // Wired into model execution in the next implementation commit.
    pub(crate) fn after_mlp(
        &mut self,
        execution: ExecutionContext<'_>,
        layer_index: usize,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentFailure> {
        let name = self.name();
        let ctx = LayerContext::new(execution, layer_index);
        self.experiment.after_mlp(&ctx, tensor).map_err(|source| {
            ExperimentFailure::new(
                name,
                ExperimentHook::AfterMlp,
                Some(execution.phase),
                Some(layer_index),
                source,
            )
        })
    }

    #[allow(dead_code)] // Wired into model execution in the next implementation commit.
    pub(crate) fn after_layer(
        &mut self,
        execution: ExecutionContext<'_>,
        layer_index: usize,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentFailure> {
        let name = self.name();
        let ctx = LayerContext::new(execution, layer_index);
        self.experiment.after_layer(&ctx, tensor).map_err(|source| {
            ExperimentFailure::new(
                name,
                ExperimentHook::AfterLayer,
                Some(execution.phase),
                Some(layer_index),
                source,
            )
        })
    }

    #[allow(dead_code)] // Wired into model execution in the next implementation commit.
    pub(crate) fn before_logits(
        &mut self,
        execution: &ExecutionContext<'_>,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentFailure> {
        let name = self.name();
        self.experiment
            .before_logits(execution, tensor)
            .map_err(|source| {
                ExperimentFailure::new(
                    name,
                    ExperimentHook::BeforeLogits,
                    Some(execution.phase),
                    None,
                    source,
                )
            })
    }

    #[allow(dead_code)] // Wired into model execution in the next implementation commit.
    pub(crate) fn after_logits(
        &mut self,
        execution: &ExecutionContext<'_>,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentFailure> {
        let name = self.name();
        self.experiment
            .after_logits(execution, tensor)
            .map_err(|source| {
                ExperimentFailure::new(
                    name,
                    ExperimentHook::AfterLogits,
                    Some(execution.phase),
                    None,
                    source,
                )
            })
    }
}

impl core::fmt::Debug for ExperimentRunner {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ExperimentRunner")
            .field("experiment", &self.name())
            .finish()
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
}
