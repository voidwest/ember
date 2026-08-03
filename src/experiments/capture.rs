//! v0.2 capture facility: selectively record live activations at hook stages.
//!
//! Capture is a run-level facility, not an experiment: it rides alongside the
//! single active experiment (or runs alone) so that, for example, a
//! zero-layer-output run can capture the post-intervention values of the very
//! tensor it mutated.
//!
//! The capture path copies tensor values **only for explicitly selected
//! records** (layer/stage/phase/position filters). Everything else is a
//! branch plus no-op on memory Ember already owns. Records are buffered in
//! memory and hashed + written at generation completion — no file I/O in the
//! inference hot path.

use super::{ExperimentError, GenerationContext, ModelContext, TensorAccess, TracingState};
use crate::artifact::{
    record_file_name, ActivationManifest, ActivationStage, CaptureRecord, CaptureSelection,
    DispatchObservation, DispatchPath, ManifestExperiment, ManifestModel, ManifestRun,
};
use crate::extraction::{git_commit, stable_prompt_hash, unix_timestamp};
use std::sync::atomic::{AtomicU64, Ordering};

static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// One buffered record awaiting finalize.
struct PendingRecord {
    phase: &'static str,
    stage: ActivationStage,
    layer: usize,
    start_position: usize,
    token_count: usize,
    shape: [usize; 2],
    values: Vec<f32>,
    dispatch: DispatchPath,
}

struct StagingCleanup(std::path::PathBuf);

impl Drop for StagingCleanup {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// Selective activation capture for one run.
pub struct CaptureSink {
    selection: CaptureSelection,
    prompt: Option<String>,
    prompt_hash: String,
    thread_count: usize,
    cpu_metadata: serde_json::Value,
    model_sha256: Option<String>,
    tokenizer_sha256: Option<String>,
    gguf_metadata: serde_json::Value,
    model: Option<ManifestModel>,
    records: Vec<PendingRecord>,
    captured_bytes: usize,
    truncated: bool,
    finalized: bool,
    ember_version: &'static str,
    /// v0.3 execution provenance (per-tensor K-family decisions), attached
    /// to the artifact when the run loaded with an explicit strategy.
    execution: Option<crate::artifact::ExecutionInventory>,
}

impl CaptureSink {
    /// Build a capture sink from a TOML config path.
    ///
    /// `prompt` is stored only when the selection does not omit it.
    #[allow(clippy::too_many_arguments)]
    pub fn from_toml_path(
        path: &str,
        prompt: &str,
        thread_count: usize,
        cpu_metadata: serde_json::Value,
        model_sha256: Option<String>,
        tokenizer_sha256: Option<String>,
        gguf_metadata: serde_json::Value,
    ) -> Result<Self, String> {
        let mut selection = CaptureSelection::from_toml_path(path)?;
        let prompt_hash = stable_prompt_hash(prompt);
        let stored_prompt = if selection.omit_prompt_text {
            None
        } else {
            Some(prompt.to_string())
        };
        // make the stored selection reflect the effective prompt handling
        selection.omit_prompt_text = stored_prompt.is_none();
        Ok(Self {
            selection,
            prompt: stored_prompt,
            prompt_hash,
            thread_count,
            cpu_metadata,
            model_sha256,
            tokenizer_sha256,
            gguf_metadata,
            model: None,
            records: Vec::new(),
            captured_bytes: 0,
            truncated: false,
            finalized: false,
            ember_version: env!("CARGO_PKG_VERSION"),
            execution: None,
        })
    }

    /// Attach v0.3 execution provenance (per-tensor K-family decisions
    /// recorded at load time) to the artifact written by [`Self::finalize`].
    pub fn with_execution(mut self, execution: crate::artifact::ExecutionInventory) -> Self {
        self.execution = Some(execution);
        self
    }

    /// Capture config accessor (for CLI banners and tests).
    pub fn selection(&self) -> &CaptureSelection {
        &self.selection
    }

    pub(crate) fn on_model_loaded(
        &mut self,
        ctx: &ModelContext<'_>,
    ) -> Result<(), ExperimentError> {
        for layer in &self.selection.layers {
            if *layer >= ctx.layer_count {
                return Err(ExperimentError::new(format!(
                    "capture layer {} does not exist for {} model '{}' (valid layers: 0..{})",
                    layer,
                    ctx.family,
                    ctx.model_identifier.unwrap_or(ctx.architecture),
                    ctx.layer_count
                )));
            }
        }
        let gguf = self.gguf_metadata.clone();
        let architecture = gguf
            .get("general.architecture")
            .and_then(|value| value.as_str())
            .unwrap_or(ctx.architecture)
            .to_string();
        let quantization = serde_json::json!({
            "file_type": gguf.get("general.file_type"),
            "quantization_version": gguf.get("general.quantization_version"),
            "size_label": gguf.get("general.size_label"),
        });
        self.model = Some(ManifestModel {
            family: ctx.family.to_string(),
            identifier: ctx.model_identifier.map(str::to_string),
            architecture,
            layer_count: ctx.layer_count,
            hidden_size: ctx.hidden_size,
            sha256: self.model_sha256.clone(),
            file_size_bytes: ctx
                .model_identifier
                .and_then(|path| std::fs::metadata(path).ok())
                .map(|metadata| metadata.len()),
            tokenizer_sha256: self.tokenizer_sha256.clone(),
            gguf: serde_json::json!({
                "general.architecture": gguf.get("general.architecture"),
                "quantization": quantization,
            }),
        });
        Ok(())
    }

    fn selected(
        &self,
        stage: ActivationStage,
        layer: usize,
        phase: &str,
        token_position: Option<usize>,
    ) -> bool {
        if self.truncated {
            return false;
        }
        self.selection.selects(stage, layer, phase, token_position)
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        phase: &'static str,
        stage: ActivationStage,
        layer: usize,
        start_position: usize,
        token_count: usize,
        tensor: &TensorAccess<'_>,
        dispatch: DispatchPath,
    ) -> Result<(), ExperimentError> {
        if self.truncated {
            return Ok(());
        }
        let position = (phase == "decode").then_some(start_position);
        if !self.selected(stage, layer, phase, position) {
            return Ok(());
        }
        if self.selection.max_records > 0 && self.records.len() >= self.selection.max_records {
            self.truncated = true;
            return Ok(());
        }
        let [rows, columns] = *tensor.shape();
        let record_bytes = tensor
            .values()
            .len()
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| ExperimentError::new("capture tensor byte size overflow"))?;
        let next_bytes = self
            .captured_bytes
            .checked_add(record_bytes)
            .ok_or_else(|| ExperimentError::new("capture buffered byte count overflow"))?;
        if self.selection.max_bytes > 0 && next_bytes > self.selection.max_bytes {
            self.truncated = true;
            return Ok(());
        }
        self.records.push(PendingRecord {
            phase,
            stage,
            layer,
            start_position,
            token_count,
            shape: [rows, columns],
            values: tensor.values().to_vec(),
            dispatch,
        });
        self.captured_bytes = next_bytes;
        Ok(())
    }

    pub(crate) fn before_layer(
        &mut self,
        execution: &super::ExecutionContext<'_>,
        layer: usize,
        tensor: &TensorAccess<'_>,
        dispatch: DispatchPath,
    ) -> Result<(), ExperimentError> {
        self.record(
            phase_name(execution.phase),
            ActivationStage::BeforeLayer,
            layer,
            execution.start_position,
            execution.input_token_count,
            tensor,
            dispatch,
        )
    }

    pub(crate) fn after_attention(
        &mut self,
        execution: &super::ExecutionContext<'_>,
        layer: usize,
        tensor: &TensorAccess<'_>,
        dispatch: DispatchPath,
    ) -> Result<(), ExperimentError> {
        self.record(
            phase_name(execution.phase),
            ActivationStage::AfterAttention,
            layer,
            execution.start_position,
            execution.input_token_count,
            tensor,
            dispatch,
        )
    }

    pub(crate) fn after_mlp(
        &mut self,
        execution: &super::ExecutionContext<'_>,
        layer: usize,
        tensor: &TensorAccess<'_>,
        dispatch: DispatchPath,
    ) -> Result<(), ExperimentError> {
        self.record(
            phase_name(execution.phase),
            ActivationStage::AfterMlp,
            layer,
            execution.start_position,
            execution.input_token_count,
            tensor,
            dispatch,
        )
    }

    pub(crate) fn after_layer(
        &mut self,
        execution: &super::ExecutionContext<'_>,
        layer: usize,
        tensor: &TensorAccess<'_>,
        dispatch: DispatchPath,
    ) -> Result<(), ExperimentError> {
        self.record(
            phase_name(execution.phase),
            ActivationStage::AfterLayer,
            layer,
            execution.start_position,
            execution.input_token_count,
            tensor,
            dispatch,
        )
    }

    pub(crate) fn before_logits(
        &mut self,
        execution: &super::ExecutionContext<'_>,
        tensor: &TensorAccess<'_>,
        dispatch: DispatchPath,
    ) -> Result<(), ExperimentError> {
        self.record(
            phase_name(execution.phase),
            ActivationStage::BeforeLogits,
            0,
            execution.start_position,
            execution.input_token_count,
            tensor,
            dispatch,
        )
    }

    pub(crate) fn after_logits(
        &mut self,
        execution: &super::ExecutionContext<'_>,
        tensor: &TensorAccess<'_>,
        dispatch: DispatchPath,
    ) -> Result<(), ExperimentError> {
        self.record(
            phase_name(execution.phase),
            ActivationStage::AfterLogits,
            0,
            execution.start_position,
            execution.input_token_count,
            tensor,
            dispatch,
        )
    }

    /// Write the artifact (manifest.json + tensors/*.npy) and return the
    /// manifest path. Called once at generation completion.
    pub(crate) fn finalize(
        &mut self,
        generation: &GenerationContext<'_>,
        experiment: ManifestExperiment,
        dispatch_observations: Vec<DispatchObservation>,
    ) -> Result<std::path::PathBuf, ExperimentError> {
        if self.finalized {
            return Err(ExperimentError::new(
                "capture finalize called more than once".to_string(),
            ));
        }
        let model = self
            .model
            .clone()
            .ok_or_else(|| ExperimentError::new("capture finalized before model load"))?;
        let output_dir = self.selection.output_dir.clone();
        std::fs::create_dir_all(&output_dir).map_err(|e| {
            ExperimentError::new(format!(
                "failed to create capture output dir '{}': {e}",
                output_dir.display()
            ))
        })?;
        let published_tensors = output_dir.join("tensors");
        let published_manifest = output_dir.join("manifest.json");
        if published_tensors.exists() || published_manifest.exists() {
            return Err(ExperimentError::new(format!(
                "capture output '{}' already contains an artifact; refusing to overwrite",
                output_dir.display()
            )));
        }
        let staging_dir = output_dir.join(format!(
            ".ember-capture-staging-{}-{}-{}",
            std::process::id(),
            unix_timestamp(),
            STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&staging_dir).map_err(|e| {
            ExperimentError::new(format!(
                "failed to create capture staging dir '{}': {e}",
                staging_dir.display()
            ))
        })?;
        let _cleanup = StagingCleanup(staging_dir.clone());
        let tensors_dir = staging_dir.join("tensors");
        std::fs::create_dir(&tensors_dir).map_err(|e| {
            ExperimentError::new(format!(
                "failed to create capture tensor staging dir '{}': {e}",
                tensors_dir.display()
            ))
        })?;

        let mut records = Vec::with_capacity(self.records.len());
        for (index, pending) in self.records.iter().enumerate() {
            let file_name = record_file_name(
                pending.phase,
                pending.layer,
                pending.stage,
                pending.start_position,
            );
            let path = tensors_dir.join(&file_name);
            crate::npy::write_npy_2d(
                path.to_str().ok_or_else(|| {
                    ExperimentError::new("capture output path is not valid UTF-8")
                })?,
                &pending.values,
                &pending.shape,
            )
            .map_err(|e| {
                ExperimentError::new(format!("failed to write '{}': {e}", path.display()))
            })?;
            let sha256 = crate::extraction::sha256_file_result(&path).map_err(|e| {
                ExperimentError::new(format!("failed to hash '{}': {e}", path.display()))
            })?;
            let (l2_norm, abs_max) = tensor_stats(&pending.values);
            records.push(CaptureRecord {
                index,
                phase: pending.phase.to_string(),
                layer: pending.layer,
                stage: pending.stage,
                start_position: pending.start_position,
                token_count: pending.token_count,
                shape: pending.shape,
                dtype: "f32".to_string(),
                byte_order: "little-endian".to_string(),
                path: format!("tensors/{file_name}"),
                sha256,
                l2_norm,
                abs_max,
                dispatch: pending.dispatch,
            });
        }
        // deterministic record order for comparison (prefill before decode)
        records.sort_by_key(CaptureRecord::sort_key);
        let record_count = records.len();

        let manifest = ActivationManifest {
            schema_version: crate::artifact::ACTIVATION_ARTIFACT_SCHEMA.to_string(),
            artifact_kind: crate::artifact::ACTIVATION_ARTIFACT_KIND.to_string(),
            ember_version: self.ember_version.to_string(),
            git_commit: git_commit(),
            model,
            run: ManifestRun {
                prompt: self.prompt.clone(),
                prompt_hash: self.prompt_hash.clone(),
                input_token_ids: generation.input_token_ids.to_vec(),
                generated_token_ids: generation.generated_token_ids.to_vec(),
                thread_count: self.thread_count,
                tracing: match generation.tracing {
                    TracingState::Disabled => "disabled".to_string(),
                    TracingState::Enabled => "enabled".to_string(),
                },
                cpu: self.cpu_metadata.clone(),
                dispatch_observations,
                k_strategy: self
                    .execution
                    .as_ref()
                    .map(|inventory| inventory.requested_strategy.clone()),
            },
            experiment,
            capture_selection: self.selection.clone(),
            records,
            execution: self.execution.clone(),
            truncated: self.truncated,
            created_at_unix: unix_timestamp(),
        };
        let manifest_path = staging_dir.join("manifest.json");
        let text = serde_json::to_string_pretty(&manifest).map_err(|e| {
            ExperimentError::new(format!("failed to serialize capture manifest: {e}"))
        })?;
        std::fs::write(&manifest_path, text).map_err(|e| {
            ExperimentError::new(format!(
                "failed to write capture manifest '{}': {e}",
                manifest_path.display()
            ))
        })?;
        crate::artifact::load_manifest(
            manifest_path
                .to_str()
                .ok_or_else(|| ExperimentError::new("capture manifest path is not valid UTF-8"))?,
        )
        .map_err(|e| {
            ExperimentError::new(format!("capture artifact self-validation failed: {e}"))
        })?;
        std::fs::rename(&tensors_dir, &published_tensors).map_err(|e| {
            ExperimentError::new(format!(
                "failed to publish capture tensors to '{}': {e}",
                published_tensors.display()
            ))
        })?;
        if let Err(error) = std::fs::rename(&manifest_path, &published_manifest) {
            std::fs::remove_dir_all(&published_tensors).ok();
            return Err(ExperimentError::new(format!(
                "failed to publish capture manifest to '{}': {error}",
                published_manifest.display()
            )));
        }
        self.finalized = true;
        eprintln!(
            "capture: wrote {} record(s) to {} (truncated={})",
            record_count,
            output_dir.display(),
            self.truncated
        );
        Ok(published_manifest)
    }
}

fn phase_name(phase: super::ExecutionPhase) -> &'static str {
    match phase {
        super::ExecutionPhase::Prefill => "prefill",
        super::ExecutionPhase::Decode => "decode",
    }
}

fn tensor_stats(values: &[f32]) -> (f64, f32) {
    let mut sum_sq = 0.0f64;
    let mut abs_max = 0.0f32;
    for value in values {
        sum_sq += (*value as f64) * (*value as f64);
        abs_max = abs_max.max(value.abs());
    }
    (sum_sq.sqrt(), abs_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::CapturePhase;
    use crate::experiments::{ExecutionContext, ExecutionPhase, ModelContext, ModelFamily};

    fn model_context(layer_count: usize, hidden: usize) -> ModelContext<'static> {
        ModelContext::new(
            ModelFamily::Qwen3,
            Some("tiny-qwen.gguf"),
            "qwen3",
            layer_count,
            hidden,
        )
    }

    fn execution(phase: ExecutionPhase, start: usize, tokens: usize) -> ExecutionContext<'static> {
        ExecutionContext::new(
            model_context(4, 8),
            phase,
            start,
            tokens,
            TracingState::Disabled,
        )
    }

    fn make_sink(selection: CaptureSelection) -> CaptureSink {
        CaptureSink {
            selection,
            prompt: Some("test prompt".to_string()),
            prompt_hash: stable_prompt_hash("test prompt"),
            thread_count: 1,
            cpu_metadata: serde_json::json!({}),
            model_sha256: None,
            tokenizer_sha256: None,
            gguf_metadata: serde_json::json!({}),
            model: None,
            records: Vec::new(),
            captured_bytes: 0,
            truncated: false,
            finalized: false,
            ember_version: "test",
            execution: None,
        }
    }

    fn make_selection(
        layers: Vec<usize>,
        stages: Vec<&str>,
        phase: CapturePhase,
    ) -> CaptureSelection {
        CaptureSelection {
            output_dir: std::env::temp_dir().join(format!(
                "ember_capture_test_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )),
            layers,
            stages: stages
                .iter()
                .map(|s| s.parse::<ActivationStage>().unwrap())
                .collect(),
            phase,
            token_positions: Vec::new(),
            max_records: 0,
            max_bytes: 0,
            omit_prompt_text: false,
            config_hash: None,
        }
    }

    #[test]
    fn config_parses_and_hashes() {
        let text = r#"
schema_version = 1
output_dir = "runs/capture-demo"
layers = [4]
stages = ["after-mlp", "after-layer"]
phase = "decode"
token_positions = [6, 7]
max_records = 8
"#;
        let selection = CaptureSelection::from_toml_str(text).unwrap();
        assert_eq!(selection.layers, vec![4]);
        assert_eq!(selection.stages.len(), 2);
        assert_eq!(selection.phase, CapturePhase::Decode);
        assert_eq!(selection.token_positions, vec![6, 7]);
        assert_eq!(selection.max_records, 8);
        assert!(selection.config_hash.is_some());
        // identical bytes -> identical hash
        let again = CaptureSelection::from_toml_str(text).unwrap();
        assert_eq!(selection.config_hash, again.config_hash);
    }

    #[test]
    fn config_rejects_bad_input() {
        assert!(CaptureSelection::from_toml_str(
            "schema_version = 2\noutput_dir = \"x\"\nlayers = [0]\nstages = [\"after-mlp\"]\n"
        )
        .is_err());
        assert!(CaptureSelection::from_toml_str(
            "schema_version = 1\noutput_dir = \"x\"\nlayers = []\nstages = [\"after-mlp\"]\n"
        )
        .is_err());
        assert!(CaptureSelection::from_toml_str(
            "schema_version = 1\noutput_dir = \"x\"\nlayers = [0]\nstages = [\"after-mlp\"]\n"
        )
        .is_ok());
        assert!(CaptureSelection::from_toml_str(
            "schema_version = 1\noutput_dir = \"x\"\nlayers = [0]\nstages = [\"bogus-stage\"]\n"
        )
        .is_err());
        assert!(CaptureSelection::from_toml_str("schema_version = 1\noutput_dir = \"x\"\nlayers = [0]\nstages = [\"after-mlp\"]\nphase = \"sideways\"\n").is_err());
    }

    #[test]
    fn selection_filters_layer_stage_phase_position() {
        let selection = make_selection(
            vec![1, 3],
            vec!["after-mlp", "after-layer"],
            CapturePhase::Both,
        );
        assert!(selection.selects(ActivationStage::AfterMlp, 1, "prefill", None));
        assert!(selection.selects(ActivationStage::AfterLayer, 3, "decode", Some(7)));
        assert!(!selection.selects(ActivationStage::AfterAttention, 1, "prefill", None));
        assert!(!selection.selects(ActivationStage::AfterMlp, 2, "prefill", None));
        assert!(selection.selects(ActivationStage::AfterMlp, 1, "decode", None));
        let prefill_only = make_selection(vec![1], vec!["after-mlp"], CapturePhase::Prefill);
        assert!(!prefill_only.selects(ActivationStage::AfterMlp, 1, "decode", None));
    }

    #[test]
    fn selection_filters_by_decode_position() {
        let selection = make_selection(vec![0], vec!["after-mlp"], CapturePhase::Decode);
        let mut with_positions = selection.clone();
        with_positions.token_positions = vec![5];
        assert!(with_positions.selects(ActivationStage::AfterMlp, 0, "decode", Some(5)));
        assert!(!with_positions.selects(ActivationStage::AfterMlp, 0, "decode", Some(6)));
        // decode-only selection excludes prefill entirely
        assert!(!with_positions.selects(ActivationStage::AfterMlp, 0, "prefill", None));
        // with phase = both, prefill records are whole-sequence and not
        // position-filtered
        let mut both = make_selection(vec![0], vec!["after-mlp"], CapturePhase::Both);
        both.token_positions = vec![5];
        assert!(both.selects(ActivationStage::AfterMlp, 0, "prefill", None));
    }

    #[test]
    fn sink_records_only_selected_and_caps_records() {
        let mut selection = make_selection(vec![0], vec!["after-mlp"], CapturePhase::Decode);
        selection.max_records = 2;
        let mut sink = make_sink(selection);
        let execution = execution(ExecutionPhase::Decode, 6, 1);
        let mut values = [1.0f32; 8];
        let tensor = TensorAccess::new(1, 8, &mut values);
        sink.after_mlp(&execution, 0, &tensor, DispatchPath::Unknown)
            .unwrap();
        sink.after_mlp(&execution, 0, &tensor, DispatchPath::Unknown)
            .unwrap();
        sink.after_mlp(&execution, 0, &tensor, DispatchPath::Unknown)
            .unwrap();
        assert_eq!(sink.records.len(), 2);
        assert!(sink.truncated);
        // unselected stage not recorded
        let mut sink2 = make_sink(make_selection(
            vec![0],
            vec!["after-mlp"],
            CapturePhase::Decode,
        ));
        let mut other = [2.0f32; 8];
        let tensor2 = TensorAccess::new(1, 8, &mut other);
        sink2
            .after_layer(&execution, 0, &tensor2, DispatchPath::Unknown)
            .unwrap();
        assert!(sink2.records.is_empty());
    }

    #[test]
    fn sink_rejects_out_of_range_layer_at_load() {
        let mut sink = make_sink(make_selection(
            vec![7],
            vec!["after-mlp"],
            CapturePhase::Both,
        ));
        let error = sink.on_model_loaded(&model_context(4, 8)).unwrap_err();
        assert!(error.to_string().contains("layer 7 does not exist"));
    }

    #[test]
    fn finalize_writes_deterministic_artifact() {
        let mut sink = make_sink(make_selection(
            vec![0],
            vec!["after-mlp"],
            CapturePhase::Both,
        ));
        sink.on_model_loaded(&model_context(4, 8)).unwrap();
        let prefill = execution(ExecutionPhase::Prefill, 0, 3);
        let mut prefill_values: Vec<f32> = (1..=24).map(|value| value as f32).collect();
        let prefill_tensor = TensorAccess::new(3, 8, &mut prefill_values);
        sink.after_mlp(&prefill, 0, &prefill_tensor, DispatchPath::Generic)
            .unwrap();
        let decode = execution(ExecutionPhase::Decode, 3, 1);
        let mut decode_values = [0.5f32; 8];
        let decode_tensor = TensorAccess::new(1, 8, &mut decode_values);
        sink.after_mlp(&decode, 0, &decode_tensor, DispatchPath::Fast)
            .unwrap();

        let generation = GenerationContext::new(
            model_context(4, 8),
            3,
            1,
            1,
            TracingState::Disabled,
            &[1, 2, 3],
            &[9],
        );
        let manifest_path = sink
            .finalize(
                &generation,
                ManifestExperiment {
                    name: "test".to_string(),
                    arguments: serde_json::json!({}),
                },
                vec![DispatchObservation {
                    phase: "prefill".to_string(),
                    dispatch: DispatchPath::Generic,
                }],
            )
            .unwrap();
        let manifest: ActivationManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.schema_version, "0.2.0-experimental");
        assert_eq!(manifest.records.len(), 2);
        assert_eq!(manifest.run.input_token_ids, vec![1, 2, 3]);
        assert_eq!(manifest.run.generated_token_ids, vec![9]);
        assert_eq!(manifest.records[0].phase, "prefill");
        assert_eq!(manifest.records[1].phase, "decode");
        assert_eq!(manifest.records[0].dispatch, DispatchPath::Generic);
        assert_eq!(manifest.records[1].dispatch, DispatchPath::Fast);
        assert!(manifest.records[0].sha256.len() == 64);
        // tensor files exist and hash matches
        let tensor_path = manifest.base_dir().join(&manifest.records[0].path);
        assert!(tensor_path.exists());
        let (shape, values) = crate::npy::read_npy_2d(tensor_path.to_str().unwrap()).unwrap();
        assert_eq!(shape, vec![3, 8]);
        assert_eq!(values.len(), 24);

        // determinism: same records -> same hashes, names, and l2 norms
        let mut sink2 = make_sink(make_selection(
            vec![0],
            vec!["after-mlp"],
            CapturePhase::Both,
        ));
        sink2.on_model_loaded(&model_context(4, 8)).unwrap();
        let mut pv: Vec<f32> = (1..=24).map(|value| value as f32).collect();
        let pt = TensorAccess::new(3, 8, &mut pv);
        sink2
            .after_mlp(&prefill, 0, &pt, DispatchPath::Generic)
            .unwrap();
        let mut dv = [0.5f32; 8];
        let dt = TensorAccess::new(1, 8, &mut dv);
        sink2
            .after_mlp(&decode, 0, &dt, DispatchPath::Fast)
            .unwrap();
        let manifest2_path = sink2
            .finalize(
                &generation,
                ManifestExperiment {
                    name: "test".to_string(),
                    arguments: serde_json::json!({}),
                },
                Vec::new(),
            )
            .unwrap();
        let manifest2: ActivationManifest =
            serde_json::from_slice(&std::fs::read(&manifest2_path).unwrap()).unwrap();
        assert_eq!(manifest.records[0].sha256, manifest2.records[0].sha256);
        assert_eq!(manifest.records[0].path, manifest2.records[0].path);
        assert_eq!(manifest.records[0].l2_norm, manifest2.records[0].l2_norm);

        let dir = manifest.base_dir();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn finalize_without_inventory_keeps_v03_fields_null() {
        // the additive v0.3 fields serialize as null when no inventory is
        // attached, and a manifest without them still parses (backward
        // compatibility in both directions)
        let mut sink = make_sink(make_selection(
            vec![0],
            vec!["after-mlp"],
            CapturePhase::Prefill,
        ));
        sink.on_model_loaded(&model_context(4, 8)).unwrap();
        let prefill = execution(ExecutionPhase::Prefill, 0, 3);
        let mut prefill_values: Vec<f32> = (1..=24).map(|value| value as f32).collect();
        let prefill_tensor = TensorAccess::new(3, 8, &mut prefill_values);
        sink.after_mlp(&prefill, 0, &prefill_tensor, DispatchPath::Generic)
            .unwrap();
        let generation = GenerationContext::new(
            model_context(4, 8),
            3,
            1,
            1,
            TracingState::Disabled,
            &[1, 2, 3],
            &[9],
        );
        let manifest_path = sink
            .finalize(
                &generation,
                ManifestExperiment {
                    name: "test".to_string(),
                    arguments: serde_json::json!({}),
                },
                Vec::new(),
            )
            .unwrap();
        let manifest: ActivationManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        assert!(manifest.execution.is_none());
        assert!(manifest.run.k_strategy.is_none());

        // strip the null fields and re-parse: old-style manifests must load
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        value.as_object_mut().unwrap().remove("execution");
        value["run"].as_object_mut().unwrap().remove("k_strategy");
        let old_style: ActivationManifest = serde_json::from_value(value).unwrap();
        assert!(old_style.execution.is_none());
        assert!(old_style.run.k_strategy.is_none());

        std::fs::remove_dir_all(manifest.base_dir()).ok();
    }

    #[test]
    fn finalize_attaches_execution_inventory() {
        let inventory = crate::artifact::ExecutionInventory {
            requested_strategy: "auto".to_string(),
            tensors: vec![],
            summary: crate::artifact::ExecutionSummary {
                tensor_count: 0,
                fallback_count: 0,
                compressed_bytes: 0,
                expanded_bytes: 0,
                per_dtype: vec![],
            },
        };
        let mut sink = make_sink(make_selection(
            vec![0],
            vec!["after-mlp"],
            CapturePhase::Prefill,
        ))
        .with_execution(inventory);
        sink.on_model_loaded(&model_context(4, 8)).unwrap();
        let prefill = execution(ExecutionPhase::Prefill, 0, 3);
        let mut prefill_values: Vec<f32> = (1..=24).map(|value| value as f32).collect();
        let prefill_tensor = TensorAccess::new(3, 8, &mut prefill_values);
        sink.after_mlp(&prefill, 0, &prefill_tensor, DispatchPath::Generic)
            .unwrap();
        let generation = GenerationContext::new(
            model_context(4, 8),
            3,
            1,
            1,
            TracingState::Disabled,
            &[1, 2, 3],
            &[9],
        );
        let manifest_path = sink
            .finalize(
                &generation,
                ManifestExperiment {
                    name: "test".to_string(),
                    arguments: serde_json::json!({}),
                },
                Vec::new(),
            )
            .unwrap();
        let manifest: ActivationManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        let attached = manifest.execution.as_ref().expect("inventory attached");
        assert_eq!(attached.requested_strategy, "auto");
        assert_eq!(manifest.run.k_strategy.as_deref(), Some("auto"));

        std::fs::remove_dir_all(manifest.base_dir()).ok();
    }
}
