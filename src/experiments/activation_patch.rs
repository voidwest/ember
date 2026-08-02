//! v0.2 `activation-patch` experiment: replace one live activation in place.
//!
//! The patch source is a v0.2 capture artifact. Each `--patch-target` must
//! resolve to **exactly one** source record — a position-qualified target or
//! a unique (layer, stage, phase) match; the first match is never chosen
//! implicitly. Source validation covers model family, layer range, stage,
//! phase, dtype, byte order, and tensor shape.
//!
//! The patch values are loaded once at construction; each hook application
//! is a bounds-checked `copy_from_slice` with **no allocation inside the
//! hook**. Patching is active only while this experiment is active: runs
//! without `--activation-patch` are never mutated.
//!
//! Patched runs are **not comparable to ordinary benchmark runs**. Treat
//! output from patched runs as research evidence with the patch provenance
//! attached, never as throughput or quality claims.

use super::{
    ExecutionContext, ExecutionPhase, Experiment, ExperimentError, GenerationContext, LayerContext,
    ModelContext, TensorAccess,
};
use crate::artifact::{
    load_manifest, resolve_record_path, resolve_unique_record, ActivationStage, CaptureRecord,
};
use core::str::FromStr;

/// One `LAYER:STAGE:PHASE[:POSITION]` patch target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchTarget {
    pub layer: usize,
    pub stage: ActivationStage,
    pub phase: ExecutionPhase,
    /// Absolute decode position; `None` requires the source manifest query
    /// to resolve to exactly one record.
    pub position: Option<usize>,
}

impl core::fmt::Display for PatchTarget {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}:{}:{}",
            self.layer,
            self.stage,
            match self.phase {
                ExecutionPhase::Prefill => "prefill",
                ExecutionPhase::Decode => "decode",
            }
        )?;
        if let Some(position) = self.position {
            write!(f, ":{position}")?;
        }
        Ok(())
    }
}

impl FromStr for PatchTarget {
    type Err = ExperimentError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = value.split(':').collect();
        if !(3..=4).contains(&parts.len()) {
            return Err(ExperimentError::new(format!(
                "patch target '{value}' must be LAYER:STAGE:PHASE[:POSITION], \
                 for example 4:after-mlp:decode or 4:after-mlp:decode:42"
            )));
        }
        let layer = parts[0].parse::<usize>().map_err(|_| {
            ExperimentError::new(format!(
                "invalid patch target layer '{}'; expected a non-negative integer",
                parts[0]
            ))
        })?;
        let stage = parts[1]
            .parse::<ActivationStage>()
            .map_err(ExperimentError::new)?;
        let phase = match parts[2] {
            "prefill" => ExecutionPhase::Prefill,
            "decode" => ExecutionPhase::Decode,
            other => {
                return Err(ExperimentError::new(format!(
                    "invalid patch target phase '{other}'; expected prefill or decode"
                )))
            }
        };
        let position = if parts.len() == 4 {
            let position = parts[3].parse::<usize>().map_err(|_| {
                ExperimentError::new(format!(
                    "invalid patch target position '{}'; expected a non-negative integer",
                    parts[3]
                ))
            })?;
            Some(position)
        } else {
            None
        };
        Ok(Self {
            layer,
            stage,
            phase,
            position,
        })
    }
}

/// One resolved patch: target + source record + loaded values.
#[derive(Debug)]
struct ResolvedPatch {
    target: PatchTarget,
    record: CaptureRecord,
    shape: [usize; 2],
    values: Vec<f32>,
    applied: usize,
    hook_reached: bool,
}

/// The built-in activation-patch experiment (v0.2, one target layer/stage).
#[derive(Debug)]
pub struct ActivationPatch {
    source_manifest: String,
    source_family: String,
    source_architecture: String,
    source_layer_count: usize,
    source_hidden_size: usize,
    source_model_sha256: Option<String>,
    source_tokenizer_sha256: Option<String>,
    source_input_token_ids: Vec<u32>,
    patches: Vec<ResolvedPatch>,
}

impl ActivationPatch {
    /// Load a source artifact and resolve every target to exactly one record.
    pub fn new(source_manifest: &str, targets: Vec<PatchTarget>) -> Result<Self, ExperimentError> {
        let manifest = load_manifest(source_manifest).map_err(ExperimentError::new)?;
        let mut patches = Vec::with_capacity(targets.len());
        for target in targets {
            let phase_name = match target.phase {
                ExecutionPhase::Prefill => "prefill",
                ExecutionPhase::Decode => "decode",
            };
            let record = resolve_unique_record(
                &manifest.records,
                target.layer,
                target.stage,
                phase_name,
                target.position,
            )
            .map_err(ExperimentError::new)?
            .clone();
            if record.dtype != "f32" {
                return Err(ExperimentError::new(format!(
                    "patch source record {} has dtype '{}'; only f32 is supported",
                    record.index, record.dtype
                )));
            }
            if record.byte_order != "little-endian" {
                return Err(ExperimentError::new(format!(
                    "patch source record {} has byte order '{}'; expected little-endian",
                    record.index, record.byte_order
                )));
            }
            let tensor_path =
                resolve_record_path(source_manifest, &record.path).map_err(ExperimentError::new)?;
            let tensor_path = tensor_path.to_str().ok_or_else(|| {
                ExperimentError::new("patch source tensor path is not valid UTF-8")
            })?;
            let (shape, values) = crate::npy::read_npy_2d(tensor_path).map_err(|e| {
                ExperimentError::new(format!(
                    "failed to read patch source tensor '{}': {e}",
                    tensor_path
                ))
            })?;
            if shape.len() != 2 {
                return Err(ExperimentError::new(format!(
                    "patch source record {} has shape {shape:?}; expected a 2D tensor",
                    record.index
                )));
            }
            let shape: [usize; 2] = [shape[0], shape[1]];
            if shape != record.shape {
                return Err(ExperimentError::new(format!(
                    "patch source record {} shape {shape:?} disagrees with manifest shape {:?}",
                    record.index, record.shape
                )));
            }
            patches.push(ResolvedPatch {
                target,
                record,
                shape,
                values,
                applied: 0,
                hook_reached: false,
            });
        }
        Ok(Self {
            source_manifest: source_manifest.to_string(),
            source_family: manifest.model.family,
            source_architecture: manifest.model.architecture,
            source_layer_count: manifest.model.layer_count,
            source_hidden_size: manifest.model.hidden_size,
            source_model_sha256: manifest.model.sha256,
            source_tokenizer_sha256: manifest.model.tokenizer_sha256,
            source_input_token_ids: manifest.run.input_token_ids,
            patches,
        })
    }

    fn apply(
        &mut self,
        stage: ActivationStage,
        ctx: &LayerContext<'_>,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        for patch in &mut self.patches {
            if patch.target.stage != stage {
                continue;
            }
            if patch.target.layer != ctx.layer_index || patch.target.phase != ctx.execution.phase {
                continue;
            }
            patch.hook_reached = true;
            if let Some(position) = patch.target.position {
                let matches_position = match ctx.execution.phase {
                    ExecutionPhase::Prefill => ctx.execution.start_position == position,
                    ExecutionPhase::Decode => ctx.execution.token_position() == Some(position),
                };
                if !matches_position {
                    continue;
                }
            }
            if tensor.shape() != &patch.shape {
                return Err(ExperimentError::new(format!(
                    "patch target {}: live tensor shape {:?} does not match source shape {:?} \
                     (same model and prompt required)",
                    patch.target,
                    tensor.shape(),
                    patch.shape
                )));
            }
            let live = tensor.values_mut();
            live.copy_from_slice(&patch.values);
            patch.applied += 1;
        }
        Ok(())
    }

    fn apply_logits(
        &mut self,
        stage: ActivationStage,
        ctx: &ExecutionContext<'_>,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        let layer_context = LayerContext::new(*ctx, 0);
        self.apply(stage, &layer_context, tensor)
    }

    /// Whether every target was applied at least once.
    pub fn all_applied(&self) -> bool {
        self.patches.iter().all(|patch| patch.applied > 0)
    }

    /// Total number of patch applications across all targets.
    pub fn total_applied(&self) -> usize {
        self.patches.iter().map(|patch| patch.applied).sum()
    }
}

impl Experiment for ActivationPatch {
    fn name(&self) -> &'static str {
        "activation-patch"
    }

    fn arguments(&self) -> serde_json::Value {
        serde_json::json!({
            "source_manifest": self.source_manifest,
            "source_family": self.source_family,
            "source_architecture": self.source_architecture,
            "source_model_sha256": self.source_model_sha256,
            "source_tokenizer_sha256": self.source_tokenizer_sha256,
            "targets": self.patches.iter().map(|patch| {
                serde_json::json!({
                    "target": patch.target.to_string(),
                    "source_record_index": patch.record.index,
                    "source_sha256": patch.record.sha256,
                })
            }).collect::<Vec<_>>(),
            "modifies_execution": true,
        })
    }

    fn on_model_loaded(&mut self, ctx: &ModelContext<'_>) -> Result<(), ExperimentError> {
        if self.source_family != ctx.family.to_string() {
            return Err(ExperimentError::new(format!(
                "patch source family '{}' does not match current model family '{}'",
                self.source_family, ctx.family
            )));
        }
        if self.source_layer_count != ctx.layer_count || self.source_hidden_size != ctx.hidden_size
        {
            return Err(ExperimentError::new(format!(
                "patch source model dimensions ({} layers, width {}) do not match current model ({} layers, width {})",
                self.source_layer_count,
                self.source_hidden_size,
                ctx.layer_count,
                ctx.hidden_size
            )));
        }
        match (self.source_model_sha256.as_deref(), ctx.model_sha256) {
            (Some(source), Some(current)) if source.eq_ignore_ascii_case(current) => {}
            (Some(source), Some(current)) => {
                return Err(ExperimentError::new(format!(
                    "patch source model SHA-256 {source} does not match current model {current}"
                )))
            }
            _ => {
                return Err(ExperimentError::new(
                    "activation patching requires model SHA-256 provenance on both source and current run",
                ))
            }
        }
        match (self.source_tokenizer_sha256.as_deref(), ctx.tokenizer_sha256) {
            (Some(source), Some(current)) if source.eq_ignore_ascii_case(current) => {}
            (Some(source), Some(current)) => {
                return Err(ExperimentError::new(format!(
                    "patch source tokenizer SHA-256 {source} does not match current tokenizer {current}"
                )))
            }
            _ => {
                return Err(ExperimentError::new(
                    "activation patching requires tokenizer SHA-256 provenance on both source and current run",
                ))
            }
        }
        for patch in &self.patches {
            if patch.target.layer >= ctx.layer_count {
                return Err(ExperimentError::new(format!(
                    "patch target {}: layer {} does not exist for {} model '{}' (valid layers: 0..{})",
                    patch.target,
                    patch.target.layer,
                    ctx.family,
                    ctx.model_identifier.unwrap_or(ctx.architecture),
                    ctx.layer_count
                )));
            }
            if patch.target.stage != ActivationStage::AfterLogits
                && patch.shape[1] != ctx.hidden_size
            {
                return Err(ExperimentError::new(format!(
                    "patch target {}: source width {} does not match {} hidden size {}",
                    patch.target, patch.shape[1], ctx.family, ctx.hidden_size
                )));
            }
        }
        Ok(())
    }

    fn before_prefill(&mut self, ctx: &ExecutionContext<'_>) -> Result<(), ExperimentError> {
        let current = ctx.input_token_ids.ok_or_else(|| {
            ExperimentError::new("activation patch prefill is missing current input token IDs")
        })?;
        if current != self.source_input_token_ids {
            return Err(ExperimentError::new(format!(
                "patch source input token IDs do not match current prompt (source {}, current {})",
                self.source_input_token_ids.len(),
                current.len()
            )));
        }
        Ok(())
    }

    fn before_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.apply(ActivationStage::BeforeLayer, ctx, hidden)
    }

    fn after_attention(
        &mut self,
        ctx: &LayerContext<'_>,
        attention_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.apply(ActivationStage::AfterAttention, ctx, attention_output)
    }

    fn after_mlp(
        &mut self,
        ctx: &LayerContext<'_>,
        mlp_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.apply(ActivationStage::AfterMlp, ctx, mlp_output)
    }

    fn after_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.apply(ActivationStage::AfterLayer, ctx, hidden)
    }

    fn before_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.apply_logits(ActivationStage::BeforeLogits, ctx, tensor)
    }

    fn after_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.apply_logits(ActivationStage::AfterLogits, ctx, tensor)
    }

    fn on_generation_complete(
        &mut self,
        _ctx: &GenerationContext<'_>,
    ) -> Result<(), ExperimentError> {
        for patch in &self.patches {
            if patch.applied == 0 {
                let reason = if patch.hook_reached {
                    "the target position never occurred in this run"
                } else {
                    "the target hook was never reached"
                };
                return Err(ExperimentError::new(format!(
                    "patch target {} was never applied: {reason}",
                    patch.target
                )));
            }
        }
        eprintln!(
            "experiment activation-patch: {} patch application(s) across {} target(s) from {}",
            self.total_applied(),
            self.patches.len(),
            self.source_manifest
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{DispatchObservation, DispatchPath, ManifestExperiment};
    use crate::experiments::{CaptureSink, ModelFamily, TracingState};

    const MODEL_SHA256: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TOKENIZER_SHA256: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn patch_model(layer_count: usize, hidden_size: usize) -> ModelContext<'static> {
        ModelContext::new(
            ModelFamily::Qwen3,
            Some("tiny.gguf"),
            "qwen3",
            layer_count,
            hidden_size,
        )
        .with_provenance(Some(MODEL_SHA256), Some(TOKENIZER_SHA256))
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ember_patch_test_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Build a tiny capture artifact with one prefill + two decode records at
    /// layer 1, stage after-mlp, with recognizable values.
    fn make_source_artifact(name: &str) -> (std::path::PathBuf, Vec<u32>) {
        let dir = temp_dir(name);
        let config_path = dir.join("capture.toml");
        std::fs::write(
            &config_path,
            format!(
                "schema_version = 1\noutput_dir = {:?}\nlayers = [1]\nstages = [\"after-mlp\"]\nphase = \"both\"\n",
                dir.to_str().unwrap()
            ),
        )
        .unwrap();
        let mut sink = CaptureSink::from_toml_path(
            config_path.to_str().unwrap(),
            "patch test prompt",
            1,
            serde_json::json!({}),
            Some(MODEL_SHA256.to_string()),
            Some(TOKENIZER_SHA256.to_string()),
            serde_json::json!({}),
        )
        .unwrap();
        let model = patch_model(4, 8);
        sink.on_model_loaded(&model).unwrap();

        let prefill =
            ExecutionContext::new(model, ExecutionPhase::Prefill, 0, 2, TracingState::Disabled);
        let mut prefill_values = vec![1.0f32; 16];
        let prefill_tensor = TensorAccess::new(2, 8, &mut prefill_values);
        sink.after_mlp(&prefill, 1, &prefill_tensor, DispatchPath::Generic)
            .unwrap();

        for position in [2usize, 3] {
            let decode = ExecutionContext::new(
                model,
                ExecutionPhase::Decode,
                position,
                1,
                TracingState::Disabled,
            );
            let mut decode_values = vec![(position as f32) * 10.0; 8];
            let decode_tensor = TensorAccess::new(1, 8, &mut decode_values);
            sink.after_mlp(&decode, 1, &decode_tensor, DispatchPath::Fast)
                .unwrap();
        }

        let generation =
            GenerationContext::new(model, 2, 2, 2, TracingState::Disabled, &[1, 2], &[3, 4]);
        let manifest_path = sink
            .finalize(
                &generation,
                ManifestExperiment {
                    name: "none".to_string(),
                    arguments: serde_json::Value::Null,
                },
                vec![DispatchObservation {
                    phase: "prefill".to_string(),
                    dispatch: DispatchPath::Generic,
                }],
            )
            .unwrap();
        (manifest_path, vec![1, 2])
    }

    #[test]
    fn target_parses_all_forms() {
        let target: PatchTarget = "4:after-mlp:decode".parse().unwrap();
        assert_eq!(target.layer, 4);
        assert_eq!(target.stage, ActivationStage::AfterMlp);
        assert_eq!(target.phase, ExecutionPhase::Decode);
        assert_eq!(target.position, None);
        let target: PatchTarget = "0:after-logits:prefill:17".parse().unwrap();
        assert_eq!(target.position, Some(17));
        assert_eq!(target.to_string(), "0:after-logits:prefill:17");
    }

    #[test]
    fn target_rejects_malformed_values() {
        for bad in [
            "4",
            "4:after-mlp",
            "4:after-mlp:decode:1:2",
            "x:after-mlp:decode",
            "4:bogus:decode",
            "4:after-mlp:sideways",
            "4:after-mlp:decode:x",
        ] {
            assert!(bad.parse::<PatchTarget>().is_err(), "{bad}");
        }
    }

    #[test]
    fn new_resolves_position_qualified_and_unique_targets() {
        let (manifest, _) = make_source_artifact("resolve_position");
        let manifest_str = manifest.to_str().unwrap();

        // position-qualified decode target
        let patch =
            ActivationPatch::new(manifest_str, vec!["1:after-mlp:decode:3".parse().unwrap()])
                .unwrap();
        assert_eq!(patch.patches.len(), 1);
        assert_eq!(patch.patches[0].record.start_position, 3);

        // unique triple (prefill has exactly one record)
        let patch =
            ActivationPatch::new(manifest_str, vec!["1:after-mlp:prefill".parse().unwrap()])
                .unwrap();
        assert_eq!(patch.patches[0].record.start_position, 0);

        // ambiguous: two decode records without a position
        let error = ActivationPatch::new(manifest_str, vec!["1:after-mlp:decode".parse().unwrap()])
            .unwrap_err();
        assert!(error.to_string().contains("ambiguous"), "{}", error);

        // none: no such layer/stage
        let error =
            ActivationPatch::new(manifest_str, vec!["3:after-mlp:decode:3".parse().unwrap()])
                .unwrap_err();
        assert!(
            error.to_string().contains("no captured record"),
            "{}",
            error
        );

        std::fs::remove_dir_all(manifest.parent().unwrap()).ok();
    }

    #[test]
    fn on_model_loaded_rejects_dimension_and_provenance_mismatch() {
        let (manifest, _) = make_source_artifact("load_width");
        let manifest_str = manifest.to_str().unwrap();
        let mut patch =
            ActivationPatch::new(manifest_str, vec!["1:after-mlp:prefill".parse().unwrap()])
                .unwrap();

        let model = patch_model(4, 4);
        let error = patch.on_model_loaded(&model).unwrap_err();
        assert!(error.to_string().contains("model dimensions"), "{}", error);

        let mut patch2 =
            ActivationPatch::new(manifest_str, vec!["1:after-mlp:prefill".parse().unwrap()])
                .unwrap();
        let model2 = ModelContext::new(ModelFamily::Qwen3, Some("different.gguf"), "qwen3", 4, 8)
            .with_provenance(
                Some("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"),
                Some(TOKENIZER_SHA256),
            );
        let error = patch2.on_model_loaded(&model2).unwrap_err();
        assert!(error.to_string().contains("model SHA-256"), "{}", error);

        let mut patch3 =
            ActivationPatch::new(manifest_str, vec!["1:after-mlp:prefill".parse().unwrap()])
                .unwrap();
        let model3 = ModelContext::new(ModelFamily::Qwen3, None, "qwen3", 4, 8)
            .with_provenance(Some(MODEL_SHA256), None);
        let error = patch3.on_model_loaded(&model3).unwrap_err();
        assert!(
            error.to_string().contains("tokenizer SHA-256 provenance"),
            "{}",
            error
        );

        std::fs::remove_dir_all(manifest.parent().unwrap()).ok();
    }

    #[test]
    fn apply_replaces_only_selected_stage_layer_phase_position() {
        let (manifest, _) = make_source_artifact("apply");
        let manifest_str = manifest.to_str().unwrap();
        let mut patch = ActivationPatch::new(
            manifest_str,
            vec![
                "1:after-mlp:decode:2".parse().unwrap(),
                "1:after-mlp:prefill".parse().unwrap(),
            ],
        )
        .unwrap();
        let model = patch_model(4, 8);
        patch.on_model_loaded(&model).unwrap();

        // decode at position 2: applied (values 20.0)
        let decode2 = LayerContext::new(
            ExecutionContext::new(model, ExecutionPhase::Decode, 2, 1, TracingState::Disabled),
            1,
        );
        let mut values = vec![-1.0f32; 8];
        let mut tensor = TensorAccess::new(1, 8, &mut values);
        patch.after_mlp(&decode2, &mut tensor).unwrap();
        assert!(tensor.values().iter().all(|v| *v == 20.0));

        // decode at position 3: not patched (different position)
        let decode3 = LayerContext::new(
            ExecutionContext::new(model, ExecutionPhase::Decode, 3, 1, TracingState::Disabled),
            1,
        );
        let mut values = vec![-1.0f32; 8];
        let mut tensor = TensorAccess::new(1, 8, &mut values);
        patch.after_mlp(&decode3, &mut tensor).unwrap();
        assert!(tensor.values().iter().all(|v| *v == -1.0));

        // different stage: not patched
        let mut values = vec![-1.0f32; 8];
        let mut tensor = TensorAccess::new(1, 8, &mut values);
        patch.after_attention(&decode2, &mut tensor).unwrap();
        assert!(tensor.values().iter().all(|v| *v == -1.0));

        // different layer: not patched
        let other_layer = LayerContext::new(
            ExecutionContext::new(model, ExecutionPhase::Decode, 2, 1, TracingState::Disabled),
            0,
        );
        let mut values = vec![-1.0f32; 8];
        let mut tensor = TensorAccess::new(1, 8, &mut values);
        patch.after_mlp(&other_layer, &mut tensor).unwrap();
        assert!(tensor.values().iter().all(|v| *v == -1.0));

        // prefill: applied with the [2, 8] source
        let prefill = LayerContext::new(
            ExecutionContext::new(model, ExecutionPhase::Prefill, 0, 2, TracingState::Disabled),
            1,
        );
        let mut values = vec![-2.0f32; 16];
        let mut tensor = TensorAccess::new(2, 8, &mut values);
        patch.after_mlp(&prefill, &mut tensor).unwrap();
        assert!(tensor.values().iter().all(|v| *v == 1.0));

        assert_eq!(patch.total_applied(), 2);
        assert!(patch.all_applied());

        std::fs::remove_dir_all(manifest.parent().unwrap()).ok();
    }

    #[test]
    fn shape_mismatch_at_hook_fails_clearly() {
        let (manifest, _) = make_source_artifact("shape_mismatch");
        let manifest_str = manifest.to_str().unwrap();
        let mut patch =
            ActivationPatch::new(manifest_str, vec!["1:after-mlp:prefill".parse().unwrap()])
                .unwrap();
        let model = patch_model(4, 8);
        patch.on_model_loaded(&model).unwrap();

        // wrong row count (different prompt length)
        let prefill = LayerContext::new(
            ExecutionContext::new(model, ExecutionPhase::Prefill, 0, 3, TracingState::Disabled),
            1,
        );
        let mut values = vec![0.0f32; 24];
        let mut tensor = TensorAccess::new(3, 8, &mut values);
        let error = patch.after_mlp(&prefill, &mut tensor).unwrap_err();
        assert!(
            error.to_string().contains("does not match source shape"),
            "{}",
            error
        );

        std::fs::remove_dir_all(manifest.parent().unwrap()).ok();
    }

    #[test]
    fn never_reached_target_fails_generation_complete() {
        let (manifest, _) = make_source_artifact("never_reached");
        let manifest_str = manifest.to_str().unwrap();
        // prefill target resolves (one record) but only decode hooks fire,
        // so the target hook is never reached in this run.
        let mut patch =
            ActivationPatch::new(manifest_str, vec!["1:after-mlp:prefill".parse().unwrap()])
                .unwrap();
        let model = patch_model(4, 8);
        patch.on_model_loaded(&model).unwrap();

        let decode = LayerContext::new(
            ExecutionContext::new(model, ExecutionPhase::Decode, 2, 1, TracingState::Disabled),
            1,
        );
        let mut values = vec![0.0f32; 8];
        let mut tensor = TensorAccess::new(1, 8, &mut values);
        patch.after_mlp(&decode, &mut tensor).unwrap();

        let generation =
            GenerationContext::new(model, 2, 1, 1, TracingState::Disabled, &[1, 2], &[3]);
        let error = patch.on_generation_complete(&generation).unwrap_err();
        assert!(error.to_string().contains("never applied"), "{}", error);

        std::fs::remove_dir_all(manifest.parent().unwrap()).ok();
    }

    #[test]
    fn before_prefill_requires_identical_input_tokens() {
        let (manifest, source_ids) = make_source_artifact("prompt_identity");
        let mut patch = ActivationPatch::new(
            manifest.to_str().unwrap(),
            vec!["1:after-mlp:prefill".parse().unwrap()],
        )
        .unwrap();
        let model = patch_model(4, 8);
        patch.on_model_loaded(&model).unwrap();

        let matching = ExecutionContext::new_with_token_ids(
            model,
            ExecutionPhase::Prefill,
            0,
            &source_ids,
            TracingState::Disabled,
        );
        patch.before_prefill(&matching).unwrap();

        let different_ids = [1, 99];
        let different = ExecutionContext::new_with_token_ids(
            model,
            ExecutionPhase::Prefill,
            0,
            &different_ids,
            TracingState::Disabled,
        );
        let error = patch.before_prefill(&different).unwrap_err();
        assert!(error.to_string().contains("do not match current prompt"));

        std::fs::remove_dir_all(manifest.parent().unwrap()).ok();
    }
}
