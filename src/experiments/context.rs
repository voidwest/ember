//! Lightweight metadata passed to experimental execution hooks.

/// Model family executing a hook.
///
/// This enum is experimental in Ember v0.1 and may gain variants or change
/// before the experiment API is stabilized.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Llama,
    Qwen3,
    Gemma4,
}

impl core::fmt::Display for ModelFamily {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::Llama => "llama",
            Self::Qwen3 => "qwen3",
            Self::Gemma4 => "gemma4",
        };
        formatter.write_str(name)
    }
}

/// Transformer execution phase.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPhase {
    Prefill,
    Decode,
}

impl core::fmt::Display for ExecutionPhase {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        };
        formatter.write_str(name)
    }
}

/// Whether Ember's structured operation tracing is active for an evaluation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracingState {
    Disabled,
    Enabled,
}

impl TracingState {
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

impl From<bool> for TracingState {
    fn from(enabled: bool) -> Self {
        if enabled {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }
}

/// Dtype exposed through [`TensorAccess`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F32,
}

/// Stable model metadata available to experiments.
///
/// This context deliberately omits weights, token buffers, the complete GGUF
/// metadata map, and model-family implementation details.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct ModelContext<'a> {
    pub family: ModelFamily,
    pub model_identifier: Option<&'a str>,
    pub architecture: &'a str,
    pub layer_count: usize,
    pub hidden_size: usize,
}

impl<'a> ModelContext<'a> {
    #[must_use]
    pub const fn new(
        family: ModelFamily,
        model_identifier: Option<&'a str>,
        architecture: &'a str,
        layer_count: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            family,
            model_identifier,
            architecture,
            layer_count,
            hidden_size,
        }
    }
}

/// Metadata for one prefill or decode model evaluation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct ExecutionContext<'a> {
    pub model: ModelContext<'a>,
    pub phase: ExecutionPhase,
    pub start_position: usize,
    pub input_token_count: usize,
    pub sequence_length: usize,
    pub tracing: TracingState,
}

impl<'a> ExecutionContext<'a> {
    #[must_use]
    pub const fn new(
        model: ModelContext<'a>,
        phase: ExecutionPhase,
        start_position: usize,
        input_token_count: usize,
        tracing: TracingState,
    ) -> Self {
        Self {
            model,
            phase,
            start_position,
            input_token_count,
            sequence_length: start_position.saturating_add(input_token_count),
            tracing,
        }
    }

    /// Return the absolute token position for a single-token evaluation.
    #[must_use]
    pub const fn token_position(self) -> Option<usize> {
        if self.input_token_count == 1 {
            Some(self.start_position)
        } else {
            None
        }
    }
}

/// Metadata for a transformer-layer hook.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct LayerContext<'a> {
    pub execution: ExecutionContext<'a>,
    pub layer_index: usize,
}

impl<'a> LayerContext<'a> {
    #[must_use]
    pub const fn new(execution: ExecutionContext<'a>, layer_index: usize) -> Self {
        Self {
            execution,
            layer_index,
        }
    }
}

/// Metadata supplied after successful generation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct GenerationContext<'a> {
    pub model: ModelContext<'a>,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub decode_evaluations: usize,
    pub tracing: TracingState,
}

impl<'a> GenerationContext<'a> {
    #[must_use]
    pub const fn new(
        model: ModelContext<'a>,
        prompt_token_count: usize,
        generated_token_count: usize,
        decode_evaluations: usize,
        tracing: TracingState,
    ) -> Self {
        Self {
            model,
            prompt_token_count,
            generated_token_count,
            decode_evaluations,
            tracing,
        }
    }
}

/// Narrow access to an existing contiguous 2D activation.
///
/// Experiments may inspect or replace values in place. They cannot resize the
/// allocation, change its shape, transfer ownership, or replace its storage.
#[non_exhaustive]
pub struct TensorAccess<'a> {
    shape: [usize; 2],
    values: &'a mut [f32],
}

impl<'a> TensorAccess<'a> {
    #[allow(dead_code)] // Wired into model execution in the next implementation commit.
    pub(crate) fn new(rows: usize, columns: usize, values: &'a mut [f32]) -> Self {
        debug_assert_eq!(rows.saturating_mul(columns), values.len());
        Self {
            shape: [rows, columns],
            values,
        }
    }

    #[must_use]
    pub const fn shape(&self) -> &[usize; 2] {
        &self.shape
    }

    #[must_use]
    pub const fn dtype(&self) -> TensorDType {
        TensorDType::F32
    }

    #[must_use]
    pub fn values(&self) -> &[f32] {
        self.values
    }

    /// Explicitly request mutable access to the existing tensor values.
    ///
    /// The returned slice cannot resize or replace the backing allocation.
    pub fn values_mut(&mut self) -> &mut [f32] {
        self.values
    }

    /// Replace every existing tensor value with zero.
    pub fn zero(&mut self) {
        self.values.fill(0.0);
    }
}

impl core::fmt::Debug for TensorAccess<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("TensorAccess")
            .field("shape", &self.shape)
            .field("dtype", &TensorDType::F32)
            .finish_non_exhaustive()
    }
}
