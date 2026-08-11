use super::{
    Experiment, ExperimentError, GenerationContext, LayerContext, ModelContext, TensorAccess,
};
use crate::artifact::ActivationStage;
use core::str::FromStr;

/// Supported intervention points for the example zero-layer-output experiment.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroLayerOutputStage {
    Attention,
    Mlp,
    Layer,
}

impl core::fmt::Display for ZeroLayerOutputStage {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::Attention => "attention",
            Self::Mlp => "mlp",
            Self::Layer => "layer",
        };
        formatter.write_str(name)
    }
}

impl FromStr for ZeroLayerOutputStage {
    type Err = ExperimentError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "attention" => Ok(Self::Attention),
            "mlp" => Ok(Self::Mlp),
            "layer" => Ok(Self::Layer),
            _ => Err(ExperimentError::new(format!(
                "unknown zero-layer-output stage '{value}'; expected attention, mlp, or layer"
            ))),
        }
    }
}

/// Parsed `LAYER:STAGE` configuration for zero-layer-output.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroLayerOutputSpec {
    layer: usize,
    stage: ZeroLayerOutputStage,
}

impl ZeroLayerOutputSpec {
    #[must_use]
    pub const fn new(layer: usize, stage: ZeroLayerOutputStage) -> Self {
        Self { layer, stage }
    }

    #[must_use]
    pub const fn layer(self) -> usize {
        self.layer
    }

    #[must_use]
    pub const fn stage(self) -> ZeroLayerOutputStage {
        self.stage
    }
}

impl core::fmt::Display for ZeroLayerOutputSpec {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}:{}", self.layer, self.stage)
    }
}

impl FromStr for ZeroLayerOutputSpec {
    type Err = ExperimentError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.matches(':').count() != 1 {
            return Err(ExperimentError::new(
                "zero-layer-output must use LAYER:STAGE, for example 4:attention",
            ));
        }
        let (layer, stage) = value
            .split_once(':')
            .expect("exactly one separator checked above");
        let layer = layer.parse::<usize>().map_err(|_| {
            ExperimentError::new(format!(
                "invalid zero-layer-output layer '{layer}'; expected a non-negative integer"
            ))
        })?;
        Ok(Self::new(layer, stage.parse()?))
    }
}

/// Example research intervention that zeros one layer contribution.
///
/// This implementation exists to demonstrate Ember's intervention hooks. It
/// is not an inference optimization or a supported model transformation.
pub struct ZeroLayerOutput {
    spec: ZeroLayerOutputSpec,
    interventions: usize,
}

impl ZeroLayerOutput {
    #[must_use]
    pub const fn new(spec: ZeroLayerOutputSpec) -> Self {
        Self {
            spec,
            interventions: 0,
        }
    }

    #[must_use]
    pub const fn spec(&self) -> ZeroLayerOutputSpec {
        self.spec
    }

    #[must_use]
    pub const fn intervention_count(&self) -> usize {
        self.interventions
    }

    fn intervene(
        &mut self,
        stage: ZeroLayerOutputStage,
        ctx: &LayerContext<'_>,
        tensor: &mut TensorAccess<'_>,
    ) {
        if ctx.layer_index == self.spec.layer && stage == self.spec.stage {
            tensor.zero();
            self.interventions += 1;
        }
    }
}

impl Experiment for ZeroLayerOutput {
    fn name(&self) -> &'static str {
        "zero-layer-output"
    }

    fn intervenes(&self) -> bool {
        true
    }

    fn uses_activation_stage(&self, stage: ActivationStage) -> bool {
        matches!(
            (self.spec.stage, stage),
            (
                ZeroLayerOutputStage::Attention,
                ActivationStage::AfterAttention
            ) | (ZeroLayerOutputStage::Mlp, ActivationStage::AfterMlp)
                | (ZeroLayerOutputStage::Layer, ActivationStage::AfterLayer)
        )
    }

    fn uses_activation_site(
        &self,
        stage: ActivationStage,
        layer: Option<usize>,
        _phase: super::ExecutionPhase,
    ) -> bool {
        layer == Some(self.spec.layer) && self.uses_activation_stage(stage)
    }

    fn arguments(&self) -> serde_json::Value {
        serde_json::json!({
            "layer": self.spec.layer(),
            "stage": self.spec.stage().to_string(),
            "modifies_execution": true,
        })
    }

    fn on_model_loaded(&mut self, ctx: &ModelContext<'_>) -> Result<(), ExperimentError> {
        if self.spec.layer >= ctx.layer_count {
            return Err(ExperimentError::new(format!(
                "layer {} does not exist for {} model '{}' (valid layers: 0..{})",
                self.spec.layer,
                ctx.family,
                ctx.model_identifier.unwrap_or(ctx.architecture),
                ctx.layer_count
            )));
        }
        Ok(())
    }

    fn after_attention(
        &mut self,
        ctx: &LayerContext<'_>,
        attention_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.intervene(ZeroLayerOutputStage::Attention, ctx, attention_output);
        Ok(())
    }

    fn after_mlp(
        &mut self,
        ctx: &LayerContext<'_>,
        mlp_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.intervene(ZeroLayerOutputStage::Mlp, ctx, mlp_output);
        Ok(())
    }

    fn after_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.intervene(ZeroLayerOutputStage::Layer, ctx, hidden);
        Ok(())
    }

    fn on_generation_complete(
        &mut self,
        _ctx: &GenerationContext<'_>,
    ) -> Result<(), ExperimentError> {
        eprintln!(
            "experiment zero-layer-output: {} intervention(s) at layer {} stage {}",
            self.interventions, self.spec.layer, self.spec.stage
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::{ExecutionContext, ExecutionPhase, ModelFamily, TracingState};

    fn model_context(layer_count: usize) -> ModelContext<'static> {
        ModelContext::new(
            ModelFamily::Gemma4,
            Some("tiny-gemma"),
            "gemma4",
            layer_count,
            4,
        )
    }

    #[test]
    fn spec_parses_all_supported_stages() {
        for (value, layer, stage) in [
            ("4:attention", 4, ZeroLayerOutputStage::Attention),
            ("0:mlp", 0, ZeroLayerOutputStage::Mlp),
            ("12:layer", 12, ZeroLayerOutputStage::Layer),
        ] {
            let spec: ZeroLayerOutputSpec = value.parse().unwrap();
            assert_eq!(spec.layer(), layer);
            assert_eq!(spec.stage(), stage);
            assert_eq!(spec.to_string(), value);
        }
    }

    #[test]
    fn spec_rejects_malformed_values() {
        for value in ["4", "4:", ":attention", "4:attention:extra", "-1:mlp"] {
            assert!(value.parse::<ZeroLayerOutputSpec>().is_err(), "{value}");
        }
        let error = "4:residual"
            .parse::<ZeroLayerOutputSpec>()
            .unwrap_err()
            .to_string();
        assert!(error.contains("attention, mlp, or layer"));
    }

    #[test]
    fn nonexistent_layer_fails_during_model_load_hook() {
        let mut experiment =
            ZeroLayerOutput::new(ZeroLayerOutputSpec::new(3, ZeroLayerOutputStage::Layer));
        let error = experiment.on_model_loaded(&model_context(3)).unwrap_err();
        assert!(error.to_string().contains("layer 3 does not exist"));
        assert!(error.to_string().contains("valid layers: 0..3"));
    }

    #[test]
    fn intervention_changes_only_selected_stage_and_layer() {
        let execution = ExecutionContext::new(
            model_context(3),
            ExecutionPhase::Prefill,
            0,
            2,
            TracingState::Disabled,
        );
        let selected = LayerContext::new(execution, 1);
        let other_layer = LayerContext::new(execution, 0);
        let mut experiment =
            ZeroLayerOutput::new(ZeroLayerOutputSpec::new(1, ZeroLayerOutputStage::Attention));

        let mut values = [1.0, 2.0, 3.0, 4.0];
        let mut tensor = TensorAccess::new(2, 2, &mut values);
        experiment.after_mlp(&selected, &mut tensor).unwrap();
        assert_eq!(tensor.values(), &[1.0, 2.0, 3.0, 4.0]);
        experiment
            .after_attention(&other_layer, &mut tensor)
            .unwrap();
        assert_eq!(tensor.values(), &[1.0, 2.0, 3.0, 4.0]);
        experiment.after_attention(&selected, &mut tensor).unwrap();
        assert_eq!(tensor.values(), &[0.0; 4]);
        assert_eq!(experiment.intervention_count(), 1);
    }

    #[test]
    fn intervention_count_includes_prefill_and_decode() {
        let model = model_context(2);
        let prefill = LayerContext::new(
            ExecutionContext::new(model, ExecutionPhase::Prefill, 0, 2, TracingState::Disabled),
            1,
        );
        let decode = LayerContext::new(
            ExecutionContext::new(model, ExecutionPhase::Decode, 2, 1, TracingState::Disabled),
            1,
        );
        let mut experiment =
            ZeroLayerOutput::new(ZeroLayerOutputSpec::new(1, ZeroLayerOutputStage::Mlp));
        let mut prefill_values = [1.0; 8];
        let mut prefill_tensor = TensorAccess::new(2, 4, &mut prefill_values);
        experiment.after_mlp(&prefill, &mut prefill_tensor).unwrap();
        let mut decode_values = [1.0; 4];
        let mut decode_tensor = TensorAccess::new(1, 4, &mut decode_values);
        experiment.after_mlp(&decode, &mut decode_tensor).unwrap();

        assert_eq!(experiment.intervention_count(), 2);
        assert!(prefill_tensor.values().iter().all(|value| *value == 0.0));
        assert!(decode_tensor.values().iter().all(|value| *value == 0.0));
    }
}
