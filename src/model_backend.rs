use crate::backend::{Backend, CpuBackend};
use crate::extraction::{
    canonical_config_toml, layer_relative_path, load_input_samples, pooling_for_mode, run_dir,
    sample_order_hash, select_token_positions, sha256_file, source_field_for_position,
    source_span_for_position, source_value_for_position, stable_bytes_hash, stable_prompt_hash,
    unix_timestamp, ArtifactManifest, BackendHiddenStateOutput, BackendMetadata,
    ExecutionBackendName, ExtractionConfig, ExtractionRunOutput, LayerArtifact, LogitsArtifact,
    ModelMetadata, PositionArtifactRecord, SampleArtifactRecord, TensorContract,
    TokenizationArtifactRecord, TokenizedPrompt, ARTIFACT_CONTRACT_VERSION, ARTIFACT_LAYOUT,
    CHECKSUMS_FILENAME, CONFIG_FILENAME, LAYERS_DIRNAME, LOGITS_FILENAME, MANIFEST_FILENAME,
    POSITIONS_FILENAME, REPORT_FILENAME, SAMPLES_FILENAME, TOKENIZATION_FILENAME,
};
use crate::model::ForwardModel;
use crate::npy::NpyStreamWriter;
use crate::tokenizer::EmberTokenizer;
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

pub trait ModelBackend {
    fn backend_metadata(&self) -> BackendMetadata;
    fn model_metadata(&self) -> ModelMetadata;
    fn tokenize(&self, prompt: &str) -> Result<TokenizedPrompt>;
    fn extract_hidden_states(
        &mut self,
        request: HiddenStateRequest<'_>,
    ) -> Result<BackendHiddenStateOutput>;
}

#[derive(Debug, Clone)]
pub struct HiddenStateRequest<'a> {
    pub token_ids: &'a [u32],
    pub selected_token_positions: &'a [usize],
    pub layers: &'a [usize],
    pub max_seq_len: Option<usize>,
}

pub struct NativeModelBackend<M> {
    compute: CpuBackend,
    model: M,
    tokenizer: EmberTokenizer,
    model_metadata: ModelMetadata,
}

impl<M> NativeModelBackend<M>
where
    M: ForwardModel<CpuBackend>,
{
    pub fn new(
        model: M,
        tokenizer: EmberTokenizer,
        model_path: &str,
        architecture: Option<String>,
        gguf_metadata: Value,
        record_model_sha256: bool,
    ) -> Self {
        let compute = CpuBackend;
        let model_metadata = ModelMetadata {
            path: model_path.to_string(),
            architecture,
            n_layers: model.n_layers(),
            embed_dim: model.embed_dim(),
            max_seq_len: model.max_seq_len(&compute),
            file_size_bytes: fs::metadata(model_path).ok().map(|m| m.len()),
            sha256: if record_model_sha256 {
                sha256_file(model_path)
            } else {
                None
            },
            gguf_metadata,
        };
        Self {
            compute,
            model,
            tokenizer,
            model_metadata,
        }
    }
}

impl<M> ModelBackend for NativeModelBackend<M>
where
    M: ForwardModel<CpuBackend>,
    <CpuBackend as Backend>::Error: Send + Sync + 'static,
{
    fn backend_metadata(&self) -> BackendMetadata {
        BackendMetadata {
            name: ExecutionBackendName::Native.as_str().to_string(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
            executable: None,
            commit: git_commit(),
            details: serde_json::json!({
                "compute_backend": "CpuBackend",
                "crate": env!("CARGO_PKG_NAME"),
            }),
        }
    }

    fn model_metadata(&self) -> ModelMetadata {
        self.model_metadata.clone()
    }

    fn tokenize(&self, prompt: &str) -> Result<TokenizedPrompt> {
        let (token_ids, offsets) = self
            .tokenizer
            .encode_with_offsets(prompt)
            .context("failed to tokenize prompt with offsets")?;
        Ok(TokenizedPrompt { token_ids, offsets })
    }

    fn extract_hidden_states(
        &mut self,
        request: HiddenStateRequest<'_>,
    ) -> Result<BackendHiddenStateOutput> {
        if request.token_ids.is_empty() {
            anyhow::bail!("cannot extract hidden states from an empty token sequence");
        }
        if request.selected_token_positions.is_empty() {
            anyhow::bail!("selected_token_positions must not be empty");
        }
        let model_context_limit = self.model.max_seq_len(&self.compute);
        let context_limit = request
            .max_seq_len
            .unwrap_or(model_context_limit)
            .min(model_context_limit);
        if request.token_ids.len() > context_limit {
            anyhow::bail!(
                "prompt has {} tokens, exceeding context limit {}",
                request.token_ids.len(),
                context_limit
            );
        }
        for &position in request.selected_token_positions {
            if position >= request.token_ids.len() {
                anyhow::bail!(
                    "selected token position {} is outside token sequence length {}",
                    position,
                    request.token_ids.len()
                );
            }
        }

        let groups = vec![request.selected_token_positions.to_vec()];
        let (pooled_states, logits) =
            self.model
                .forward_pooled_activations(&self.compute, request.token_ids, &groups)?;
        let all_layers = &pooled_states[0];
        let embed_dim = self.model.embed_dim();
        let mut hidden_states = Vec::with_capacity(request.layers.len() * embed_dim);
        for &layer in request.layers {
            let start = layer * embed_dim;
            let end = start + embed_dim;
            hidden_states.extend_from_slice(&all_layers[start..end]);
        }
        let raw_logits_shape = self.compute.shape(&logits).to_vec();
        let (logits, logits_shape) = if raw_logits_shape.len() == 2 && raw_logits_shape[0] > 0 {
            let vocab_size = raw_logits_shape[1];
            let row_start = (raw_logits_shape[0] - 1) * vocab_size;
            let row_end = row_start + vocab_size;
            (
                Some(self.compute.data(&logits)[row_start..row_end].to_vec()),
                Some(vec![1, vocab_size]),
            )
        } else {
            (
                Some(self.compute.data(&logits).to_vec()),
                Some(raw_logits_shape),
            )
        };
        Ok(BackendHiddenStateOutput {
            hidden_states,
            hidden_states_shape: vec![request.layers.len(), embed_dim],
            logits_available: true,
            logits,
            logits_shape,
        })
    }
}

#[derive(Debug, Clone)]
pub struct LlamaCppBackend {
    executable: Option<String>,
}

impl LlamaCppBackend {
    pub fn from_config(config: &ExtractionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            executable: config.llama_cpp_binary.clone(),
        })
    }
}

impl ModelBackend for LlamaCppBackend {
    fn backend_metadata(&self) -> BackendMetadata {
        BackendMetadata {
            name: ExecutionBackendName::LlamaCpp.as_str().to_string(),
            version: llama_cpp_version(self.executable.as_deref()),
            executable: self.executable.clone(),
            commit: None,
            details: serde_json::json!({
                "integration": "external-process",
                "status": "not implemented",
                "requires": "patched/custom llama.cpp hidden-state extraction binary",
            }),
        }
    }

    fn model_metadata(&self) -> ModelMetadata {
        ModelMetadata {
            path: String::new(),
            architecture: None,
            n_layers: 0,
            embed_dim: 0,
            max_seq_len: 0,
            file_size_bytes: None,
            sha256: None,
            gguf_metadata: Value::Null,
        }
    }

    fn tokenize(&self, _prompt: &str) -> Result<TokenizedPrompt> {
        anyhow::bail!(
            "llama-cpp backend not implemented for hidden-state extraction yet; \
             expected a patched/custom external extraction binary"
        )
    }

    fn extract_hidden_states(
        &mut self,
        _request: HiddenStateRequest<'_>,
    ) -> Result<BackendHiddenStateOutput> {
        anyhow::bail!(
            "llama-cpp backend not implemented for hidden-state extraction yet; \
             expected a patched/custom external extraction binary"
        )
    }
}

pub fn run_extraction_with_backend<B: ModelBackend>(
    backend: &mut B,
    config: &ExtractionConfig,
) -> Result<ExtractionRunOutput> {
    config.validate()?;

    let model_metadata = backend.model_metadata();
    let backend_metadata = backend.backend_metadata();
    let layers = config.effective_layers(model_metadata.n_layers)?;
    let samples = load_input_samples(config)?;

    let run_dir = run_dir(config);
    let layers_dir = run_dir.join(LAYERS_DIRNAME);
    fs::create_dir_all(&layers_dir).with_context(|| {
        format!(
            "failed to create layers directory: {}",
            layers_dir.display()
        )
    })?;

    let config_path = run_dir.join(CONFIG_FILENAME);
    let manifest_path = run_dir.join(MANIFEST_FILENAME);
    let samples_path = run_dir.join(SAMPLES_FILENAME);
    let tokenization_path = run_dir.join(TOKENIZATION_FILENAME);
    let positions_path = run_dir.join(POSITIONS_FILENAME);
    let checksums_path = run_dir.join(CHECKSUMS_FILENAME);
    let report_path = run_dir.join(REPORT_FILENAME);
    let logits_path = run_dir.join(LOGITS_FILENAME);

    let config_path_str = path_to_string(&config_path)?;
    let manifest_path_str = path_to_string(&manifest_path)?;
    let samples_path_str = path_to_string(&samples_path)?;
    let tokenization_path_str = path_to_string(&tokenization_path)?;
    let positions_path_str = path_to_string(&positions_path)?;
    let checksums_path_str = path_to_string(&checksums_path)?;
    let report_path_str = path_to_string(&report_path)?;
    let logits_path_str = path_to_string(&logits_path)?;

    let canonical_config = canonical_config_toml(config)?;
    fs::write(&config_path, &canonical_config)
        .with_context(|| format!("failed to write canonical config: {}", config_path_str))?;
    let config_hash = stable_bytes_hash(canonical_config.as_bytes());

    let mut layer_writers = layers
        .iter()
        .map(|&layer| {
            let path = run_dir.join(layer_relative_path(layer));
            let path = path_to_string(&path)?;
            NpyStreamWriter::create(&path, &[samples.len(), model_metadata.embed_dim])
        })
        .collect::<Result<Vec<_>>>()?;
    let layer_artifacts = layers
        .iter()
        .map(|&layer| LayerArtifact {
            layer_index: layer,
            layer_name: crate::extraction::layer_name(layer),
            path: layer_relative_path(layer),
            shape: vec![samples.len(), model_metadata.embed_dim],
        })
        .collect::<Vec<_>>();

    let mut sample_writer = fs::File::create(&samples_path)
        .with_context(|| format!("failed to create samples artifact: {}", samples_path_str))?;
    let mut tokenization_writer = fs::File::create(&tokenization_path).with_context(|| {
        format!(
            "failed to create tokenization artifact: {}",
            tokenization_path_str
        )
    })?;
    let mut positions_writer = fs::File::create(&positions_path).with_context(|| {
        format!(
            "failed to create positions artifact: {}",
            positions_path_str
        )
    })?;

    let mut logits_writer: Option<NpyStreamWriter> = None;
    let mut logits_shape: Option<Vec<usize>> = None;
    let mut logits_written = false;
    let mut order_hash_inputs = Vec::with_capacity(samples.len());

    for (sample_index, sample) in samples.iter().enumerate() {
        let tokenized = backend
            .tokenize(&sample.prompt)
            .with_context(|| format!("failed to tokenize sample '{}'", sample.sample_id))?;
        let prompt_hash = stable_prompt_hash(&sample.prompt);
        let selected_token_positions = select_token_positions(
            &sample.prompt,
            &tokenized.token_ids,
            &tokenized.offsets,
            config,
            sample.word_value.as_deref(),
        )
        .with_context(|| {
            format!(
                "failed to select token positions for '{}'",
                sample.sample_id
            )
        })?;
        let output = backend.extract_hidden_states(HiddenStateRequest {
            token_ids: &tokenized.token_ids,
            selected_token_positions: &selected_token_positions,
            layers: &layers,
            max_seq_len: config.max_seq_len,
        })?;
        if output.hidden_states_shape != vec![layers.len(), model_metadata.embed_dim] {
            anyhow::bail!(
                "backend returned hidden-state shape {:?}, expected {:?}",
                output.hidden_states_shape,
                vec![layers.len(), model_metadata.embed_dim]
            );
        }
        for (layer_offset, writer) in layer_writers.iter_mut().enumerate() {
            let row_start = layer_offset * model_metadata.embed_dim;
            let row_end = row_start + model_metadata.embed_dim;
            writer.write_f32s(&output.hidden_states[row_start..row_end])?;
        }

        if config.write_logits {
            let logits = output
                .logits
                .as_ref()
                .context("config requested write_logits but backend did not return logits")?;
            let shape = output
                .logits_shape
                .as_ref()
                .context("backend returned logits without logits_shape")?;
            if shape.len() != 2 || shape[0] != 1 {
                anyhow::bail!(
                    "expected per-sample logits shape [1, vocab], got {:?}",
                    shape
                );
            }
            let vocab_size = shape[1];
            if logits.len() != vocab_size {
                anyhow::bail!(
                    "logits payload has {} values but logits_shape expects {}",
                    logits.len(),
                    vocab_size
                );
            }
            if logits_writer.is_none() {
                logits_writer = Some(NpyStreamWriter::create(
                    &logits_path_str,
                    &[samples.len(), vocab_size],
                )?);
                logits_shape = Some(vec![samples.len(), vocab_size]);
            }
            logits_writer
                .as_mut()
                .expect("logits writer initialized above")
                .write_f32s(logits)?;
            logits_written = true;
        }

        let token_count = tokenized.token_ids.len();
        order_hash_inputs.push((sample.sample_id.clone(), prompt_hash.clone()));

        let sample_record = SampleArtifactRecord {
            schema_version: ARTIFACT_CONTRACT_VERSION,
            sample_index,
            sample_id: sample.sample_id.clone(),
            input_index: sample.input_index,
            prompt: if config.prompt_hashes_only {
                None
            } else {
                Some(sample.prompt.clone())
            },
            prompt_hash: prompt_hash.clone(),
        };
        serde_json::to_writer(&mut sample_writer, &sample_record)?;
        sample_writer.write_all(b"\n")?;

        let tokenization_record = TokenizationArtifactRecord {
            schema_version: ARTIFACT_CONTRACT_VERSION,
            sample_index,
            sample_id: sample.sample_id.clone(),
            token_ids: tokenized.token_ids,
            token_count,
            prompt_hash,
            offsets: tokenized.offsets,
        };
        serde_json::to_writer(&mut tokenization_writer, &tokenization_record)?;
        tokenization_writer.write_all(b"\n")?;

        let position_record = PositionArtifactRecord {
            schema_version: ARTIFACT_CONTRACT_VERSION,
            sample_index,
            sample_id: sample.sample_id.clone(),
            position_mode: config.token_position.as_str().to_string(),
            pooling: pooling_for_mode(config.token_position).to_string(),
            selected_token_positions,
            source_field: source_field_for_position(config),
            source_value: source_value_for_position(config, sample.word_value.as_deref()),
            source_byte_span: source_span_for_position(
                &sample.prompt,
                config,
                sample.word_value.as_deref(),
            ),
        };
        serde_json::to_writer(&mut positions_writer, &position_record)?;
        positions_writer.write_all(b"\n")?;
    }

    for writer in &mut layer_writers {
        writer.finish()?;
    }
    if let Some(writer) = &mut logits_writer {
        writer.finish()?;
    }
    sample_writer.flush()?;
    tokenization_writer.flush()?;
    positions_writer.flush()?;

    let sample_order_hash = sample_order_hash(&order_hash_inputs);
    let logits_artifact = if logits_written {
        Some(LogitsArtifact {
            path: LOGITS_FILENAME.to_string(),
            shape: logits_shape.expect("logits shape recorded when logits are written"),
        })
    } else {
        None
    };

    let manifest = ArtifactManifest {
        schema_version: ARTIFACT_CONTRACT_VERSION,
        layout: ARTIFACT_LAYOUT.to_string(),
        artifact_kind: "ember_hidden_states".to_string(),
        created_at_unix: unix_timestamp(),
        run_id: config.run_id.clone(),
        run_dir: path_to_string(&run_dir)?,
        config_path: CONFIG_FILENAME.to_string(),
        samples_path: SAMPLES_FILENAME.to_string(),
        tokenization_path: TOKENIZATION_FILENAME.to_string(),
        positions_path: POSITIONS_FILENAME.to_string(),
        checksums_path: CHECKSUMS_FILENAME.to_string(),
        report_path: REPORT_FILENAME.to_string(),
        logits_path: logits_written.then(|| LOGITS_FILENAME.to_string()),
        tensor_contract: TensorContract {
            storage: "layer-sharded-npy".to_string(),
            dtype: config.dtype.as_str().to_string(),
            byte_order: "little-endian".to_string(),
            sample_axis: 0,
            hidden_axis: 1,
            layers: layer_artifacts,
            logits: logits_artifact,
        },
        sample_count: samples.len(),
        sample_order_hash,
        config_hash,
        dtype: config.dtype.as_str().to_string(),
        output_format: config.output_format.as_str().to_string(),
        model: model_metadata,
        backend: backend_metadata,
        extraction_config: config.clone(),
    };
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("failed to write manifest artifact: {}", manifest_path_str))?;

    let report = serde_json::json!({
        "schema_version": ARTIFACT_CONTRACT_VERSION,
        "layout": ARTIFACT_LAYOUT,
        "status": "complete",
        "sample_count": samples.len(),
        "layer_count": layers.len(),
        "layers": layers,
        "logits_written": logits_written,
        "resume": {
            "supported_by_contract": true,
            "native_runner_policy": "fresh-run",
            "rule": "resume only when existing JSONL line counts, layer row counts, config_hash, and sample_order_hash agree"
        },
        "stale_or_corrupt_detection": {
            "checksums": CHECKSUMS_FILENAME,
            "manifest": MANIFEST_FILENAME,
            "sample_order_hash": manifest.sample_order_hash,
            "config_hash": manifest.config_hash
        },
    });
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("failed to write report artifact: {}", report_path_str))?;

    let mut checksums = BTreeMap::new();
    checksum_insert(&mut checksums, &config_path, CONFIG_FILENAME);
    checksum_insert(&mut checksums, &manifest_path, MANIFEST_FILENAME);
    checksum_insert(&mut checksums, &samples_path, SAMPLES_FILENAME);
    checksum_insert(&mut checksums, &tokenization_path, TOKENIZATION_FILENAME);
    checksum_insert(&mut checksums, &positions_path, POSITIONS_FILENAME);
    checksum_insert(&mut checksums, &report_path, REPORT_FILENAME);
    for &layer in &layers {
        let rel = layer_relative_path(layer);
        checksum_insert(&mut checksums, &run_dir.join(&rel), &rel);
    }
    if logits_written {
        checksum_insert(&mut checksums, &logits_path, LOGITS_FILENAME);
    }
    fs::write(&checksums_path, serde_json::to_string_pretty(&checksums)?)
        .with_context(|| format!("failed to write checksums artifact: {}", checksums_path_str))?;

    Ok(ExtractionRunOutput {
        run_dir: path_to_string(&run_dir)?,
        manifest_path: manifest_path_str,
        samples_path: samples_path_str,
        tokenization_path: tokenization_path_str,
        positions_path: positions_path_str,
        checksums_path: checksums_path_str,
        report_path: report_path_str,
        sample_count: samples.len(),
        layer_paths: layers
            .iter()
            .map(|&layer| path_to_string(&run_dir.join(layer_relative_path(layer))))
            .collect::<Result<Vec<_>>>()?,
    })
}

fn checksum_insert(
    checksums: &mut BTreeMap<String, String>,
    absolute_path: &Path,
    relative_path: &str,
) {
    if let Some(sum) = sha256_file(absolute_path) {
        checksums.insert(relative_path.to_string(), sum);
    }
}

fn path_to_string(path: &Path) -> Result<String> {
    path.to_str()
        .map(str::to_string)
        .with_context(|| format!("path is not valid UTF-8: {}", path.display()))
}

fn git_commit() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let commit = String::from_utf8(output.stdout).ok()?;
    Some(commit.trim().to_string())
}

fn llama_cpp_version(executable: Option<&str>) -> Option<String> {
    let executable = executable?;
    let output = std::process::Command::new(executable)
        .arg("--version")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|s| s.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_cpp_placeholder_reports_not_implemented() {
        let config = ExtractionConfig {
            run_id: None,
            model_path: "model.gguf".to_string(),
            architecture: Some("llama".to_string()),
            tokenizer_path: None,
            backend: ExecutionBackendName::LlamaCpp,
            prompt_template: "{word}".to_string(),
            input_jsonl_path: "input.jsonl".to_string(),
            output_dir: "out".to_string(),
            layers: vec![0],
            token_position: crate::extraction::TokenPositionMode::PromptFinal,
            word_field: "word".to_string(),
            sample_id_field: "id".to_string(),
            batch_size: 1,
            dtype: crate::extraction::ArtifactDType::F32,
            output_format: crate::extraction::ArtifactOutputFormat::Npy,
            prompt_hashes_only: false,
            write_logits: false,
            resume: false,
            max_seq_len: None,
            record_model_sha256: false,
            llama_cpp_binary: None,
            run_metadata: Value::Null,
        };
        let backend = LlamaCppBackend::from_config(&config).expect("valid placeholder backend");
        let err = backend
            .tokenize("hello")
            .expect_err("placeholder should fail");
        assert!(err.to_string().contains("not implemented"));
    }
}
