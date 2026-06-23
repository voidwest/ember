use crate::backend::{Backend, CpuBackend};
use crate::extraction::{
    load_input_samples, select_token_positions, sha256_file, stable_prompt_hash, unix_timestamp,
    ArtifactMetadata, BackendHiddenStateOutput, BackendMetadata, ExecutionBackendName,
    ExtractionConfig, ExtractionRunOutput, ModelMetadata, SampleArtifactRecord, TokenizedPrompt,
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

    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("failed to create output directory: {}", config.output_dir))?;
    let hidden_states_path = Path::new(&config.output_dir).join("hidden_states.npy");
    let samples_path = Path::new(&config.output_dir).join("samples.jsonl");
    let metadata_path = Path::new(&config.output_dir).join("metadata.json");

    let hidden_states_shape = vec![samples.len(), layers.len(), model_metadata.embed_dim];
    let hidden_states_path_str = path_to_string(&hidden_states_path)?;
    let samples_path_str = path_to_string(&samples_path)?;
    let metadata_path_str = path_to_string(&metadata_path)?;

    let mut hidden_writer = NpyStreamWriter::create(&hidden_states_path_str, &hidden_states_shape)?;
    let mut sample_writer = fs::File::create(&samples_path)
        .with_context(|| format!("failed to create samples artifact: {}", samples_path_str))?;

    for sample in &samples {
        let tokenized = backend
            .tokenize(&sample.prompt)
            .with_context(|| format!("failed to tokenize sample '{}'", sample.sample_id))?;
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
        hidden_writer.write_f32s(&output.hidden_states)?;
        let token_count = tokenized.token_ids.len();

        let record = SampleArtifactRecord {
            schema_version: 1,
            sample_id: sample.sample_id.clone(),
            input_index: sample.input_index,
            token_ids: tokenized.token_ids,
            selected_token_positions,
            token_count,
            prompt: if config.prompt_hashes_only {
                None
            } else {
                Some(sample.prompt.clone())
            },
            prompt_hash: stable_prompt_hash(&sample.prompt),
            logits_available: output.logits_available,
            logits_shape: output.logits_shape,
        };
        serde_json::to_writer(&mut sample_writer, &record)?;
        sample_writer.write_all(b"\n")?;
    }

    hidden_writer.finish()?;
    sample_writer.flush()?;

    let mut checksums = BTreeMap::new();
    if let Some(sum) = sha256_file(&hidden_states_path) {
        checksums.insert("hidden_states.npy".to_string(), sum);
    }
    if let Some(sum) = sha256_file(&samples_path) {
        checksums.insert("samples.jsonl".to_string(), sum);
    }

    let metadata = ArtifactMetadata {
        schema_version: 1,
        artifact_kind: "ember_hidden_states".to_string(),
        created_at_unix: unix_timestamp(),
        output_dir: config.output_dir.clone(),
        hidden_states_path: hidden_states_path_str.clone(),
        samples_path: samples_path_str.clone(),
        hidden_states_shape: hidden_states_shape.clone(),
        dtype: config.dtype.as_str().to_string(),
        output_format: config.output_format.as_str().to_string(),
        layers,
        token_position: config.token_position.as_str().to_string(),
        model: model_metadata,
        backend: backend_metadata,
        extraction_config: config.clone(),
        checksums,
    };
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)
        .with_context(|| format!("failed to write metadata artifact: {}", metadata_path_str))?;

    Ok(ExtractionRunOutput {
        output_dir: config.output_dir.clone(),
        hidden_states_path: hidden_states_path_str,
        samples_path: samples_path_str,
        metadata_path: metadata_path_str,
        sample_count: samples.len(),
        hidden_states_shape,
    })
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
