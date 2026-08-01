use super::{
    Experiment, ExperimentError, GenerationContext, LayerContext, ModelContext, TensorAccess,
};
use crate::trace::compute_tensor_values;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const EXPERIMENT_NAME: &str = "activation-stats";

#[derive(Debug, Clone, Serialize)]
struct ActivationStatsModel {
    family: String,
    model_identifier: Option<String>,
    architecture: String,
    layer_count: usize,
    hidden_size: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ActivationStatsRecord {
    phase: &'static str,
    stage: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    layer_index: Option<usize>,
    start_position: usize,
    input_token_count: usize,
    sequence_length: usize,
    shape: [usize; 2],
    dtype: &'static str,
    l2_norm: f64,
    abs_max: f32,
    fingerprint: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ActivationStatsGeneration {
    prompt_token_count: usize,
    generated_token_count: usize,
    decode_evaluations: usize,
}

#[derive(Serialize)]
struct ActivationStatsArtifact<'a> {
    schema_version: u32,
    experiment: &'static str,
    observation_only: bool,
    model: &'a ActivationStatsModel,
    generation: ActivationStatsGeneration,
    records: &'a [ActivationStatsRecord],
}

/// Observation-only experiment that records activation norms and fingerprints.
///
/// The experiment reads existing activation values at Ember's approved hook
/// boundaries and writes one JSON artifact after successful generation. It
/// never requests mutable tensor access or changes inference values.
pub struct ActivationStats {
    output_path: PathBuf,
    model: Option<ActivationStatsModel>,
    records: Vec<ActivationStatsRecord>,
}

impl ActivationStats {
    #[must_use]
    pub fn new(output_path: impl Into<PathBuf>) -> Self {
        Self {
            output_path: output_path.into(),
            model: None,
            records: Vec::new(),
        }
    }

    #[must_use]
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    #[must_use]
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    fn record(&mut self, stage: &'static str, ctx: &LayerContext<'_>, tensor: &TensorAccess<'_>) {
        self.record_execution(stage, Some(ctx.layer_index), &ctx.execution, tensor);
    }

    fn record_execution(
        &mut self,
        stage: &'static str,
        layer_index: Option<usize>,
        ctx: &super::ExecutionContext<'_>,
        tensor: &TensorAccess<'_>,
    ) {
        let values = compute_tensor_values(tensor.values());
        self.records.push(ActivationStatsRecord {
            phase: match ctx.phase {
                super::ExecutionPhase::Prefill => "prefill",
                super::ExecutionPhase::Decode => "decode",
            },
            stage,
            layer_index,
            start_position: ctx.start_position,
            input_token_count: ctx.input_token_count,
            sequence_length: ctx.sequence_length,
            shape: *tensor.shape(),
            dtype: "f32",
            l2_norm: values.output_l2_norm,
            abs_max: values.output_abs_max,
            fingerprint: values.output_fingerprint,
        });
    }

    fn write_artifact(&self, ctx: &GenerationContext<'_>) -> Result<(), ExperimentError> {
        let model = self.model.as_ref().ok_or_else(|| {
            ExperimentError::new("activation statistics completed before model metadata was set")
        })?;
        let artifact = ActivationStatsArtifact {
            schema_version: 1,
            experiment: EXPERIMENT_NAME,
            observation_only: true,
            model,
            generation: ActivationStatsGeneration {
                prompt_token_count: ctx.prompt_token_count,
                generated_token_count: ctx.generated_token_count,
                decode_evaluations: ctx.decode_evaluations,
            },
            records: &self.records,
        };
        let file = File::create(&self.output_path).map_err(|error| {
            ExperimentError::new(format!(
                "could not create activation statistics artifact '{}': {error}",
                self.output_path.display()
            ))
        })?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &artifact).map_err(|error| {
            ExperimentError::new(format!(
                "could not serialize activation statistics artifact '{}': {error}",
                self.output_path.display()
            ))
        })?;
        writer.write_all(b"\n").map_err(|error| {
            ExperimentError::new(format!(
                "could not finish activation statistics artifact '{}': {error}",
                self.output_path.display()
            ))
        })?;
        writer.flush().map_err(|error| {
            ExperimentError::new(format!(
                "could not flush activation statistics artifact '{}': {error}",
                self.output_path.display()
            ))
        })
    }
}

impl Experiment for ActivationStats {
    fn name(&self) -> &'static str {
        EXPERIMENT_NAME
    }

    fn arguments(&self) -> serde_json::Value {
        serde_json::json!({
            "output_path": self.output_path.to_string_lossy(),
            "observation_only": true,
        })
    }

    fn on_model_loaded(&mut self, ctx: &ModelContext<'_>) -> Result<(), ExperimentError> {
        self.model = Some(ActivationStatsModel {
            family: ctx.family.to_string(),
            model_identifier: ctx.model_identifier.map(str::to_owned),
            architecture: ctx.architecture.to_owned(),
            layer_count: ctx.layer_count,
            hidden_size: ctx.hidden_size,
        });
        Ok(())
    }

    fn before_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.record("before_layer", ctx, hidden);
        Ok(())
    }

    fn after_attention(
        &mut self,
        ctx: &LayerContext<'_>,
        attention_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.record("after_attention", ctx, attention_output);
        Ok(())
    }

    fn after_mlp(
        &mut self,
        ctx: &LayerContext<'_>,
        mlp_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.record("after_mlp", ctx, mlp_output);
        Ok(())
    }

    fn after_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.record("after_layer", ctx, hidden);
        Ok(())
    }

    fn before_logits(
        &mut self,
        ctx: &super::ExecutionContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.record_execution("before_logits", None, ctx, hidden);
        Ok(())
    }

    fn after_logits(
        &mut self,
        ctx: &super::ExecutionContext<'_>,
        logits: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.record_execution("after_logits", None, ctx, logits);
        Ok(())
    }

    fn on_generation_complete(
        &mut self,
        ctx: &GenerationContext<'_>,
    ) -> Result<(), ExperimentError> {
        self.write_artifact(ctx)?;
        eprintln!(
            "experiment activation-stats: wrote {} observation record(s) to {}",
            self.records.len(),
            self.output_path.display()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::{ExecutionContext, ExecutionPhase, ModelFamily, TracingState};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

    fn temp_artifact_path() -> PathBuf {
        let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "ember-activation-stats-{}-{sequence}.json",
            std::process::id()
        ))
    }

    #[test]
    fn records_without_mutating_and_writes_self_describing_artifact() {
        let output_path = temp_artifact_path();
        let model = ModelContext::new(ModelFamily::Qwen3, Some("tiny-qwen.gguf"), "qwen3", 2, 4);
        let execution =
            ExecutionContext::new(model, ExecutionPhase::Prefill, 0, 2, TracingState::Enabled);
        let layer = LayerContext::new(execution, 1);
        let mut experiment = ActivationStats::new(&output_path);
        experiment.on_model_loaded(&model).unwrap();

        let original = [3.0, 4.0, -2.0, 1.0];
        let mut values = original;
        let mut tensor = TensorAccess::new(1, 4, &mut values);
        experiment.before_layer(&layer, &mut tensor).unwrap();
        experiment.after_attention(&layer, &mut tensor).unwrap();
        experiment.after_mlp(&layer, &mut tensor).unwrap();
        experiment.after_layer(&layer, &mut tensor).unwrap();
        experiment.before_logits(&execution, &mut tensor).unwrap();
        experiment.after_logits(&execution, &mut tensor).unwrap();
        assert_eq!(tensor.values(), &original);
        assert_eq!(experiment.record_count(), 6);

        experiment
            .on_generation_complete(&GenerationContext::new(
                model,
                2,
                1,
                0,
                TracingState::Enabled,
                &[1, 2],
                &[3],
            ))
            .unwrap();

        let artifact: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&output_path).unwrap()).unwrap();
        assert_eq!(artifact["schema_version"], 1);
        assert_eq!(artifact["experiment"], "activation-stats");
        assert_eq!(artifact["observation_only"], true);
        assert_eq!(artifact["model"]["family"], "qwen3");
        assert_eq!(artifact["generation"]["prompt_token_count"], 2);
        assert_eq!(artifact["records"].as_array().unwrap().len(), 6);
        assert_eq!(artifact["records"][0]["stage"], "before_layer");
        assert_eq!(artifact["records"][0]["phase"], "prefill");
        assert_eq!(artifact["records"][0]["layer_index"], 1);
        assert_eq!(artifact["records"][0]["shape"], serde_json::json!([1, 4]));
        assert_eq!(artifact["records"][0]["dtype"], "f32");
        assert_eq!(artifact["records"][0]["l2_norm"], 30.0_f64.sqrt());
        assert_eq!(artifact["records"][0]["abs_max"], 4.0);
        assert_eq!(
            artifact["records"][0]["fingerprint"],
            compute_tensor_values(&original).output_fingerprint
        );

        std::fs::remove_file(output_path).unwrap();
    }

    #[test]
    fn artifact_write_failure_names_the_output_path() {
        let output_path = temp_artifact_path().join("missing").join("stats.json");
        let model = ModelContext::new(ModelFamily::Llama, None, "llama", 1, 4);
        let mut experiment = ActivationStats::new(&output_path);
        experiment.on_model_loaded(&model).unwrap();
        let error = experiment
            .on_generation_complete(&GenerationContext::new(
                model,
                1,
                1,
                0,
                TracingState::Disabled,
                &[1],
                &[2],
            ))
            .unwrap_err();
        assert!(error
            .to_string()
            .contains(&output_path.display().to_string()));
    }
}
