use crate::backend::{Backend, CpuBackend};
use crate::extraction::{
    canonical_config_toml, git_commit, layer_relative_path, load_input_samples, pooling_for_mode,
    read_jsonl_records, run_dir, sample_order_hash, select_token_positions, sha256_file_result,
    source_field_for_position, source_span_for_position, source_value_for_position,
    stable_bytes_hash, stable_prompt_hash, unix_timestamp, validate_artifact_contract,
    ArtifactManifest, BackendHiddenStateOutput, BackendMetadata, ExecutionBackendName,
    ExtractionConfig, ExtractionRunOutput, LayerArtifact, LlamaCppExternalRequest, LogitsArtifact,
    ModelMetadata, PositionArtifactRecord, RunDirectoryTransaction, SampleArtifactRecord,
    TensorContract, TokenizationArtifactRecord, TokenizedPrompt, TokenizerMetadata,
    ARTIFACT_CONTRACT_VERSION, ARTIFACT_LAYOUT, CHECKSUMS_FILENAME, CONFIG_FILENAME,
    LAYERS_DIRNAME, LLAMA_CPP_REQUEST_FILENAME, LOGITS_FILENAME, MANIFEST_FILENAME,
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
use std::process::Command;

pub trait ModelBackend {
    fn backend_metadata(&self) -> BackendMetadata;
    fn model_metadata(&self) -> ModelMetadata;
    fn tokenizer_metadata(&self) -> Option<TokenizerMetadata> {
        None
    }
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
    pub include_logits: bool,
}

pub struct NativeModelBackend<M> {
    compute: CpuBackend,
    model: M,
    tokenizer: EmberTokenizer,
    model_metadata: ModelMetadata,
    tokenizer_metadata: TokenizerMetadata,
}

impl<M> NativeModelBackend<M>
where
    M: ForwardModel<CpuBackend>,
{
    pub fn new(
        model: M,
        tokenizer: EmberTokenizer,
        tokenizer_path: &str,
        model_path: &str,
        architecture: Option<String>,
        gguf_metadata: Value,
        record_model_sha256: bool,
    ) -> Result<Self> {
        let compute = CpuBackend;
        let file_metadata = fs::metadata(model_path)
            .with_context(|| format!("failed to stat model file: {model_path}"))?;
        let tokenizer_file_metadata = fs::metadata(tokenizer_path)
            .with_context(|| format!("failed to stat tokenizer file: {tokenizer_path}"))?;
        let model_metadata = ModelMetadata {
            path: model_path.to_string(),
            architecture,
            n_layers: model.n_layers(),
            embed_dim: model.embed_dim(),
            max_seq_len: model.max_seq_len(&compute),
            file_size_bytes: Some(file_metadata.len()),
            sha256: if record_model_sha256 {
                Some(sha256_file_result(model_path)?)
            } else {
                None
            },
            gguf_metadata,
        };
        Ok(Self {
            compute,
            model,
            tokenizer,
            model_metadata,
            tokenizer_metadata: TokenizerMetadata {
                path: tokenizer_path.to_string(),
                file_size_bytes: tokenizer_file_metadata.len(),
                sha256: sha256_file_result(tokenizer_path)?,
            },
        })
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

    fn tokenizer_metadata(&self) -> Option<TokenizerMetadata> {
        Some(self.tokenizer_metadata.clone())
    }

    fn tokenize(&self, prompt: &str) -> Result<TokenizedPrompt> {
        let (token_ids, offsets) = self
            .tokenizer
            .encode_with_offsets(prompt)
            .context("failed to tokenize prompt with offsets")?;
        if let Some((index, token_id)) = token_ids
            .iter()
            .enumerate()
            .find(|(_, token_id)| **token_id as usize >= self.model.vocab_size(&self.compute))
        {
            anyhow::bail!(
                "token ID {token_id} at prompt position {index} exceeds model vocabulary size {}",
                self.model.vocab_size(&self.compute)
            );
        }
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
        if !request
            .selected_token_positions
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        {
            anyhow::bail!("selected_token_positions must be strictly increasing");
        }
        if !request.layers.windows(2).all(|pair| pair[0] < pair[1]) {
            anyhow::bail!("requested layers must be strictly increasing");
        }
        if request
            .layers
            .iter()
            .any(|&layer| layer >= self.model.n_layers())
        {
            anyhow::bail!(
                "requested layer exceeds model layer count {}",
                self.model.n_layers()
            );
        }
        if let Some((index, token_id)) = request
            .token_ids
            .iter()
            .enumerate()
            .find(|(_, token_id)| **token_id as usize >= self.model.vocab_size(&self.compute))
        {
            anyhow::bail!(
                "token ID {token_id} at position {index} exceeds model vocabulary size {}",
                self.model.vocab_size(&self.compute)
            );
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
        let (pooled_states, logits) = if request.include_logits {
            let (pooled, logits) =
                self.model
                    .forward_pooled_activations(&self.compute, request.token_ids, &groups)?;
            (pooled, Some(logits))
        } else {
            (
                self.model.forward_pooled_hidden_states(
                    &self.compute,
                    request.token_ids,
                    &groups,
                )?,
                None,
            )
        };
        if pooled_states.len() != 1 {
            anyhow::bail!(
                "native model returned {} pooled groups, expected 1",
                pooled_states.len()
            );
        }
        let all_layers = &pooled_states[0];
        let embed_dim = self.model.embed_dim();
        let expected_all_layers = self
            .model
            .n_layers()
            .checked_mul(embed_dim)
            .context("pooled hidden-state shape overflow")?;
        if all_layers.len() != expected_all_layers {
            anyhow::bail!(
                "native model returned {} pooled hidden values, expected {expected_all_layers}",
                all_layers.len()
            );
        }
        if let Some((index, value)) = all_layers
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            anyhow::bail!(
                "native pooled hidden states contain non-finite value {value} at flat index {index}"
            );
        }
        let hidden_capacity = request
            .layers
            .len()
            .checked_mul(embed_dim)
            .context("selected hidden-state shape overflow")?;
        let mut hidden_states = Vec::with_capacity(hidden_capacity);
        for &layer in request.layers {
            let start = layer * embed_dim;
            let end = start + embed_dim;
            hidden_states.extend_from_slice(&all_layers[start..end]);
        }
        let (logits, logits_shape) = logits.map_or(Ok((None, None)), |logits| {
            let raw_shape = self.compute.shape(&logits).to_vec();
            let vocab_size = self.model.vocab_size(&self.compute);
            if raw_shape.len() == 2 && raw_shape[0] > 0 && raw_shape[1] == vocab_size {
                let data = self.compute.data(&logits);
                let expected = raw_shape[0]
                    .checked_mul(vocab_size)
                    .context("native logits shape overflow")?;
                if data.len() != expected {
                    anyhow::bail!(
                        "native logits payload has {} values, expected {expected}",
                        data.len()
                    );
                }
                let row_start = (raw_shape[0] - 1) * vocab_size;
                let row_end = row_start + vocab_size;
                let last = &data[row_start..row_end];
                if let Some((index, value)) = last
                    .iter()
                    .enumerate()
                    .find(|(_, value)| !value.is_finite())
                {
                    anyhow::bail!(
                        "native logits contain non-finite value {value} at vocabulary index {index}"
                    );
                }
                Ok((
                    Some(last.to_vec()),
                    Some(vec![1, vocab_size]),
                ))
            } else {
                anyhow::bail!(
                    "native logits shape must be [rows, {vocab_size}] with rows > 0, got {raw_shape:?}"
                )
            }
        })?;
        Ok(BackendHiddenStateOutput {
            hidden_states,
            hidden_states_shape: vec![request.layers.len(), embed_dim],
            logits_available: request.include_logits,
            logits,
            logits_shape,
        })
    }
}

#[derive(Debug, Clone)]
pub struct LlamaCppExternalBackend {
    executable: String,
}

impl LlamaCppExternalBackend {
    pub fn from_config(config: &ExtractionConfig) -> Result<Self> {
        config.validate()?;
        let executable = config
            .llama_cpp_binary
            .as_deref()
            .context("llama-cpp-external requires llama_cpp_binary or --llama-bin")?;
        validate_executable_path(executable)?;
        validate_model_path(&config.model_path)?;
        validate_input_path(&config.input_jsonl_path)?;
        if !config.layers.is_empty() {
            anyhow::bail!(
                "unsupported llama-cpp-external config: hidden-state layer extraction is not wired yet; leave layers empty for tokenization/logits plumbing"
            );
        }
        Ok(Self {
            executable: executable.to_string(),
        })
    }

    pub fn backend_metadata(&self) -> BackendMetadata {
        BackendMetadata {
            name: ExecutionBackendName::LlamaCppExternal.as_str().to_string(),
            version: llama_cpp_version(Some(&self.executable)),
            executable: Some(self.executable.clone()),
            commit: None,
            details: serde_json::json!({
                "integration": "external-process",
                "interface": "--request <json>",
                "supports_hidden_states": false,
            }),
        }
    }
}

pub fn run_llama_cpp_external_backend(config: &ExtractionConfig) -> Result<ExtractionRunOutput> {
    let backend = LlamaCppExternalBackend::from_config(config)?;
    let final_run_dir = run_dir(config);
    let transaction = RunDirectoryTransaction::begin(&final_run_dir)?;
    let run_dir = transaction.staging_path().to_path_buf();
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create run directory: {}", run_dir.display()))?;

    let config_path = run_dir.join(CONFIG_FILENAME);
    let request_path = run_dir.join(LLAMA_CPP_REQUEST_FILENAME);
    let manifest_path = run_dir.join(MANIFEST_FILENAME);
    let samples_path = run_dir.join(SAMPLES_FILENAME);
    let tokenization_path = run_dir.join(TOKENIZATION_FILENAME);
    let positions_path = run_dir.join(POSITIONS_FILENAME);
    let checksums_path = run_dir.join(CHECKSUMS_FILENAME);
    let report_path = run_dir.join(REPORT_FILENAME);
    let logits_path = run_dir.join(LOGITS_FILENAME);

    let canonical_config = canonical_config_toml(config)?;
    fs::write(&config_path, canonical_config).with_context(|| {
        format!(
            "failed to write external backend config: {}",
            config_path.display()
        )
    })?;

    let request = LlamaCppExternalRequest {
        schema_version: 1,
        contract_version: ARTIFACT_CONTRACT_VERSION,
        layout: ARTIFACT_LAYOUT.to_string(),
        backend: ExecutionBackendName::LlamaCppExternal.as_str().to_string(),
        model_path: config.model_path.clone(),
        input_jsonl_path: config.input_jsonl_path.clone(),
        output_dir: path_to_string(&final_run_dir)?,
        config_path: path_to_string(&config_path)?,
        manifest_path: path_to_string(&manifest_path)?,
        samples_path: path_to_string(&samples_path)?,
        tokenization_path: path_to_string(&tokenization_path)?,
        positions_path: path_to_string(&positions_path)?,
        checksums_path: path_to_string(&checksums_path)?,
        report_path: path_to_string(&report_path)?,
        logits_path: config
            .write_logits
            .then(|| path_to_string(&logits_path))
            .transpose()?,
        prompt_template: config.prompt_template.clone(),
        sample_id_field: config.sample_id_field.clone(),
        word_field: config.word_field.clone(),
        token_position: config.token_position.as_str().to_string(),
        layers: config.layers.clone(),
        write_logits: config.write_logits,
        prompt_hashes_only: config.prompt_hashes_only,
        max_seq_len: config.max_seq_len,
        run_metadata: config.run_metadata.clone(),
        extraction_config: config.clone(),
    };
    fs::write(&request_path, serde_json::to_string_pretty(&request)?).with_context(|| {
        format!(
            "failed to write llama-cpp external request: {}",
            request_path.display()
        )
    })?;

    let output = Command::new(&backend.executable)
        .arg("--request")
        .arg(&request_path)
        .output()
        .with_context(|| {
            format!(
                "failed to spawn llama-cpp external backend: {}",
                backend.executable
            )
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::bail!(
            "llama-cpp external backend failed with status {}:\nstderr:\n{}\nstdout:\n{}",
            output.status,
            stderr.trim(),
            stdout.trim()
        );
    }

    let summary = validate_artifact_contract(&run_dir, true)?;
    let sample_count = summary.sample_count;
    let published_run_dir = transaction.commit()?;
    Ok(ExtractionRunOutput {
        run_dir: path_to_string(&published_run_dir)?,
        manifest_path: path_to_string(&published_run_dir.join(MANIFEST_FILENAME))?,
        samples_path: path_to_string(&published_run_dir.join(SAMPLES_FILENAME))?,
        tokenization_path: path_to_string(&published_run_dir.join(TOKENIZATION_FILENAME))?,
        positions_path: path_to_string(&published_run_dir.join(POSITIONS_FILENAME))?,
        checksums_path: path_to_string(&published_run_dir.join(CHECKSUMS_FILENAME))?,
        report_path: path_to_string(&published_run_dir.join(REPORT_FILENAME))?,
        sample_count,
        layer_paths: Vec::new(),
    })
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BackendParityReport {
    pub native_run_dir: String,
    pub external_run_dir: String,
    pub sample_count: usize,
    pub sample_order_hash_matches: bool,
    pub prompt_hash_mismatches: Vec<usize>,
    pub token_id_mismatches: Vec<usize>,
    pub token_offset_mismatches: Vec<usize>,
    pub position_mismatches: Vec<usize>,
    pub logits_status: String,
    pub logits_comparison: Option<BackendLogitsComparison>,
    pub provenance_warnings: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BackendLogitsComparison {
    pub shape: Vec<usize>,
    pub exact_bits_equal: bool,
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
    pub rmse: f64,
    pub cosine_similarity: Option<f64>,
    pub top1_match_count: usize,
    pub top1_match_rate: f64,
}

pub fn compare_backend_artifacts(
    native_run_dir: impl AsRef<Path>,
    external_run_dir: impl AsRef<Path>,
) -> Result<BackendParityReport> {
    let native_summary = validate_artifact_contract(&native_run_dir, true)?;
    let external_summary = validate_artifact_contract(&external_run_dir, true)?;
    if native_summary.sample_count != external_summary.sample_count {
        anyhow::bail!(
            "sample_count mismatch: native={}, external={}",
            native_summary.sample_count,
            external_summary.sample_count
        );
    }
    let native_manifest: ArtifactManifest = serde_json::from_str(&fs::read_to_string(
        native_run_dir.as_ref().join(MANIFEST_FILENAME),
    )?)?;
    let external_manifest: ArtifactManifest = serde_json::from_str(&fs::read_to_string(
        external_run_dir.as_ref().join(MANIFEST_FILENAME),
    )?)?;
    if let (Some(native_sha), Some(external_sha)) = (
        native_manifest.model.sha256.as_deref(),
        external_manifest.model.sha256.as_deref(),
    ) {
        if !native_sha.eq_ignore_ascii_case(external_sha) {
            anyhow::bail!("backend comparison model SHA-256 values differ");
        }
    }
    let mut provenance_warnings = Vec::new();
    if native_manifest.model.sha256.is_none() || external_manifest.model.sha256.is_none() {
        provenance_warnings.push(
            "one or both runs omit model SHA-256; model identity is not cryptographically pinned"
                .to_string(),
        );
    }
    match (
        native_manifest.tokenizer.as_ref(),
        external_manifest.tokenizer.as_ref(),
    ) {
        (Some(native), Some(external)) if !native.sha256.eq_ignore_ascii_case(&external.sha256) => {
            anyhow::bail!("backend comparison tokenizer SHA-256 values differ")
        }
        (None, _) | (_, None) => provenance_warnings.push(
            "one or both runs omit tokenizer SHA-256; tokenizer identity is not fully pinned"
                .to_string(),
        ),
        _ => {}
    }

    let native_tokens: Vec<TokenizationArtifactRecord> =
        read_jsonl_records(native_run_dir.as_ref().join(TOKENIZATION_FILENAME))?;
    let external_tokens: Vec<TokenizationArtifactRecord> =
        read_jsonl_records(external_run_dir.as_ref().join(TOKENIZATION_FILENAME))?;
    let native_positions: Vec<PositionArtifactRecord> =
        read_jsonl_records(native_run_dir.as_ref().join(POSITIONS_FILENAME))?;
    let external_positions: Vec<PositionArtifactRecord> =
        read_jsonl_records(external_run_dir.as_ref().join(POSITIONS_FILENAME))?;

    let mut token_id_mismatches = Vec::new();
    let mut token_offset_mismatches = Vec::new();
    let mut prompt_hash_mismatches = Vec::new();
    let mut position_mismatches = Vec::new();
    for i in 0..native_tokens.len() {
        if native_tokens[i].sample_id != external_tokens[i].sample_id {
            anyhow::bail!(
                "sample_id mismatch at row {}: native={}, external={}",
                i,
                native_tokens[i].sample_id,
                external_tokens[i].sample_id
            );
        }
        if native_tokens[i].token_ids != external_tokens[i].token_ids {
            token_id_mismatches.push(i);
        }
        if native_tokens[i].prompt_hash != external_tokens[i].prompt_hash {
            prompt_hash_mismatches.push(i);
        }
        if native_tokens[i].offsets != external_tokens[i].offsets
            || native_tokens[i].offset_unit != external_tokens[i].offset_unit
        {
            token_offset_mismatches.push(i);
        }
        if native_positions[i].position_mode != external_positions[i].position_mode
            || native_positions[i].pooling != external_positions[i].pooling
            || native_positions[i].selected_token_positions
                != external_positions[i].selected_token_positions
            || native_positions[i].source_field != external_positions[i].source_field
            || native_positions[i].source_value != external_positions[i].source_value
            || native_positions[i].source_byte_span != external_positions[i].source_byte_span
        {
            position_mismatches.push(i);
        }
    }

    let (logits_status, logits_comparison) = match (
        native_summary.logits_present,
        external_summary.logits_present,
    ) {
        (false, false) => ("not_exposed".to_string(), None),
        (true, false) => ("native_only".to_string(), None),
        (false, true) => ("external_only".to_string(), None),
        (true, true) => {
            let comparison = compare_logits_artifacts(
                &native_run_dir.as_ref().join(LOGITS_FILENAME),
                &external_run_dir.as_ref().join(LOGITS_FILENAME),
            )?;
            let status = if comparison.exact_bits_equal {
                "identical"
            } else {
                "different"
            };
            (status.to_string(), Some(comparison))
        }
    };

    Ok(BackendParityReport {
        native_run_dir: native_summary.run_dir,
        external_run_dir: external_summary.run_dir,
        sample_count: native_summary.sample_count,
        sample_order_hash_matches: native_summary.sample_order_hash
            == external_summary.sample_order_hash,
        prompt_hash_mismatches,
        token_id_mismatches,
        token_offset_mismatches,
        position_mismatches,
        logits_status,
        logits_comparison,
        provenance_warnings,
    })
}

fn compare_logits_artifacts(
    native_path: &Path,
    external_path: &Path,
) -> Result<BackendLogitsComparison> {
    let native_path_text = path_to_string(native_path)?;
    let external_path_text = path_to_string(external_path)?;
    let (native_shape, native) = crate::npy::read_npy_2d(&native_path_text)?;
    let (external_shape, external) = crate::npy::read_npy_2d(&external_path_text)?;
    if native_shape != external_shape {
        anyhow::bail!(
            "backend logits shape mismatch: native {:?}, external {:?}",
            native_shape,
            external_shape
        );
    }
    if native_shape.len() != 2 || native_shape[0] == 0 || native_shape[1] == 0 {
        anyhow::bail!("backend logits must be a non-empty rank-2 tensor");
    }
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut dot = 0.0f64;
    let mut native_norm = 0.0f64;
    let mut external_norm = 0.0f64;
    let mut exact_bits_equal = true;
    for (&left, &right) in native.iter().zip(&external) {
        exact_bits_equal &= left.to_bits() == right.to_bits();
        let left = f64::from(left);
        let right = f64::from(right);
        let difference = left - right;
        let absolute = difference.abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
        sum_sq += difference * difference;
        dot += left * right;
        native_norm += left * left;
        external_norm += right * right;
    }
    let value_count = native.len() as f64;
    let cosine_similarity = match (native_norm, external_norm) {
        (0.0, 0.0) => Some(1.0),
        (0.0, _) | (_, 0.0) => None,
        _ => Some(dot / (native_norm.sqrt() * external_norm.sqrt())),
    };
    let rows = native_shape[0];
    let vocab_size = native_shape[1];
    let mut top1_match_count = 0usize;
    for row in 0..rows {
        let start = row * vocab_size;
        let end = start + vocab_size;
        if crate::sampler::argmax_token(&native[start..end])
            == crate::sampler::argmax_token(&external[start..end])
        {
            top1_match_count += 1;
        }
    }
    Ok(BackendLogitsComparison {
        shape: native_shape,
        exact_bits_equal,
        max_abs_diff: max_abs,
        mean_abs_diff: sum_abs / value_count,
        rmse: (sum_sq / value_count).sqrt(),
        cosine_similarity,
        top1_match_count,
        top1_match_rate: top1_match_count as f64 / rows as f64,
    })
}

pub fn run_extraction_with_backend<B: ModelBackend>(
    backend: &mut B,
    config: &ExtractionConfig,
) -> Result<ExtractionRunOutput> {
    config.validate()?;

    let model_metadata = backend.model_metadata();
    let tokenizer_metadata = backend.tokenizer_metadata();
    let backend_metadata = backend.backend_metadata();
    if model_metadata.n_layers == 0
        || model_metadata.embed_dim == 0
        || model_metadata.max_seq_len == 0
    {
        anyhow::bail!(
            "backend model metadata must have non-zero layers, hidden width, and context length"
        );
    }
    if backend_metadata.name.trim().is_empty() {
        anyhow::bail!("backend metadata name must not be empty");
    }
    let layers = config.effective_layers(model_metadata.n_layers)?;
    let samples = load_input_samples(config)?;

    let final_run_dir = run_dir(config);
    let transaction = RunDirectoryTransaction::begin(&final_run_dir)?;
    let run_dir = transaction.staging_path().to_path_buf();
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
        if tokenized.token_ids.is_empty() {
            anyhow::bail!("sample '{}' produced no token IDs", sample.sample_id);
        }
        crate::extraction::validate_token_offsets(
            Some(&sample.prompt),
            &tokenized.token_ids,
            &tokenized.offsets,
            sample_index,
        )?;
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
            include_logits: config.write_logits,
        })?;
        if output.hidden_states_shape != vec![layers.len(), model_metadata.embed_dim] {
            anyhow::bail!(
                "backend returned hidden-state shape {:?}, expected {:?}",
                output.hidden_states_shape,
                vec![layers.len(), model_metadata.embed_dim]
            );
        }
        let expected_hidden_values = layers
            .len()
            .checked_mul(model_metadata.embed_dim)
            .context("hidden-state output shape overflow")?;
        if output.hidden_states.len() != expected_hidden_values {
            anyhow::bail!(
                "backend returned {} hidden values, expected {expected_hidden_values}",
                output.hidden_states.len()
            );
        }
        if let Some((index, value)) = output
            .hidden_states
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            anyhow::bail!(
                "backend hidden states contain non-finite value {value} at flat index {index}"
            );
        }
        if output.logits_available != config.write_logits {
            anyhow::bail!(
                "backend logits_available={} does not match write_logits={}",
                output.logits_available,
                config.write_logits
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
            if vocab_size == 0 {
                anyhow::bail!("backend returned an empty logits vocabulary");
            }
            if logits.len() != vocab_size {
                anyhow::bail!(
                    "logits payload has {} values but logits_shape expects {}",
                    logits.len(),
                    vocab_size
                );
            }
            if let Some((index, value)) = logits
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                anyhow::bail!(
                    "backend logits contain non-finite value {value} at vocabulary index {index}"
                );
            }
            if let Some(existing_shape) = &logits_shape {
                if existing_shape[1] != vocab_size {
                    anyhow::bail!(
                        "backend logits vocabulary changed from {} to {vocab_size} at sample '{}'",
                        existing_shape[1],
                        sample.sample_id
                    );
                }
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
        } else if output.logits.is_some() || output.logits_shape.is_some() {
            anyhow::bail!("backend returned logits even though write_logits=false");
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
            offset_unit: "unicode_character_index".to_string(),
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
            )?,
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
        run_dir: path_to_string(&final_run_dir)?,
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
        tokenizer: tokenizer_metadata,
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
            "supported_by_contract": false,
            "native_runner_policy": "fresh-run",
            "rule": "existing run directories are rejected; resume is not implemented"
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
    checksum_insert(&mut checksums, &config_path, CONFIG_FILENAME)?;
    checksum_insert(&mut checksums, &manifest_path, MANIFEST_FILENAME)?;
    checksum_insert(&mut checksums, &samples_path, SAMPLES_FILENAME)?;
    checksum_insert(&mut checksums, &tokenization_path, TOKENIZATION_FILENAME)?;
    checksum_insert(&mut checksums, &positions_path, POSITIONS_FILENAME)?;
    checksum_insert(&mut checksums, &report_path, REPORT_FILENAME)?;
    for &layer in &layers {
        let rel = layer_relative_path(layer);
        checksum_insert(&mut checksums, &run_dir.join(&rel), &rel)?;
    }
    if logits_written {
        checksum_insert(&mut checksums, &logits_path, LOGITS_FILENAME)?;
    }
    fs::write(&checksums_path, serde_json::to_string_pretty(&checksums)?)
        .with_context(|| format!("failed to write checksums artifact: {}", checksums_path_str))?;

    validate_artifact_contract(&run_dir, false)?;
    let published_run_dir = transaction.commit()?;
    let published_path = |relative: &str| path_to_string(&published_run_dir.join(relative));

    Ok(ExtractionRunOutput {
        run_dir: path_to_string(&published_run_dir)?,
        manifest_path: published_path(MANIFEST_FILENAME)?,
        samples_path: published_path(SAMPLES_FILENAME)?,
        tokenization_path: published_path(TOKENIZATION_FILENAME)?,
        positions_path: published_path(POSITIONS_FILENAME)?,
        checksums_path: published_path(CHECKSUMS_FILENAME)?,
        report_path: published_path(REPORT_FILENAME)?,
        sample_count: samples.len(),
        layer_paths: layers
            .iter()
            .map(|&layer| path_to_string(&published_run_dir.join(layer_relative_path(layer))))
            .collect::<Result<Vec<_>>>()?,
    })
}

fn checksum_insert(
    checksums: &mut BTreeMap<String, String>,
    absolute_path: &Path,
    relative_path: &str,
) -> Result<()> {
    checksums.insert(
        relative_path.to_string(),
        sha256_file_result(absolute_path)?,
    );
    Ok(())
}

fn validate_executable_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    if !path.is_file() {
        anyhow::bail!("invalid llama.cpp external binary path: {}", path.display());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(path)
            .with_context(|| format!("failed to stat binary: {}", path.display()))?
            .permissions()
            .mode();
        if mode & 0o111 == 0 {
            anyhow::bail!(
                "invalid llama.cpp external binary path: {} is not executable",
                path.display()
            );
        }
    }
    Ok(())
}

fn validate_model_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    if !path.is_file() {
        anyhow::bail!("invalid GGUF model path: {}", path.display());
    }
    Ok(())
}

fn validate_input_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    if !path.is_file() {
        anyhow::bail!("invalid samples JSONL path: {}", path.display());
    }
    Ok(())
}

fn path_to_string(path: &Path) -> Result<String> {
    path.to_str()
        .map(str::to_string)
        .with_context(|| format!("path is not valid UTF-8: {}", path.display()))
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
    use crate::backend::CpuError;
    use crate::tensor::CpuTensor;
    use std::cell::Cell;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct CountingForwardModel {
        hidden_calls: Cell<usize>,
        logits_calls: Cell<usize>,
    }

    impl ForwardModel<CpuBackend> for CountingForwardModel {
        fn create_cache(
            &self,
            _backend: &CpuBackend,
            max_seq_len: usize,
        ) -> crate::kv_cache::KVCache {
            crate::kv_cache::KVCache::new(1, 1, 2, max_seq_len)
        }

        fn max_seq_len(&self, _backend: &CpuBackend) -> usize {
            16
        }

        fn forward_with_cache(
            &self,
            _backend: &CpuBackend,
            _token_ids: &[u32],
            _cache: &mut crate::kv_cache::KVCache,
            _start_pos: usize,
        ) -> Result<CpuTensor, CpuError> {
            unreachable!("not used by extraction")
        }

        fn forward_last_logits_with_cache(
            &self,
            _backend: &CpuBackend,
            _token_ids: &[u32],
            _cache: &mut crate::kv_cache::KVCache,
            _start_pos: usize,
        ) -> Result<CpuTensor, CpuError> {
            unreachable!("not used by extraction")
        }

        fn n_layers(&self) -> usize {
            1
        }

        fn embed_dim(&self) -> usize {
            2
        }

        fn vocab_size(&self, _backend: &CpuBackend) -> usize {
            3
        }

        fn forward_with_activations(
            &self,
            _backend: &CpuBackend,
            _token_ids: &[u32],
        ) -> Result<(Vec<Vec<f32>>, CpuTensor), CpuError> {
            unreachable!("not used by extraction")
        }

        fn forward_pooled_activations(
            &self,
            _backend: &CpuBackend,
            _token_ids: &[u32],
            _token_index_groups: &[Vec<usize>],
        ) -> Result<(Vec<Vec<f32>>, CpuTensor), CpuError> {
            self.logits_calls.set(self.logits_calls.get() + 1);
            Ok((
                vec![vec![1.0, 2.0]],
                CpuTensor::from_data(vec![1, 3], vec![3.0, 4.0, 5.0]),
            ))
        }

        fn forward_pooled_hidden_states(
            &self,
            _backend: &CpuBackend,
            _token_ids: &[u32],
            _token_index_groups: &[Vec<usize>],
        ) -> Result<Vec<Vec<f32>>, CpuError> {
            self.hidden_calls.set(self.hidden_calls.get() + 1);
            Ok(vec![vec![1.0, 2.0]])
        }
    }

    #[test]
    fn native_extraction_skips_logits_when_not_requested() {
        let model = CountingForwardModel {
            hidden_calls: Cell::new(0),
            logits_calls: Cell::new(0),
        };
        let tokenizer =
            EmberTokenizer::from_file("tokenizer.json").expect("repository test tokenizer");
        let mut backend = NativeModelBackend::new(
            model,
            tokenizer,
            "tokenizer.json",
            "Cargo.toml",
            Some("test".to_string()),
            Value::Null,
            false,
        )
        .unwrap();
        let request = |include_logits| HiddenStateRequest {
            token_ids: &[1],
            selected_token_positions: &[0],
            layers: &[0],
            max_seq_len: None,
            include_logits,
        };

        let hidden_only = backend.extract_hidden_states(request(false)).unwrap();
        assert!(!hidden_only.logits_available);
        assert!(hidden_only.logits.is_none());
        assert_eq!(backend.model.hidden_calls.get(), 1);
        assert_eq!(backend.model.logits_calls.get(), 0);

        let with_logits = backend.extract_hidden_states(request(true)).unwrap();
        assert!(with_logits.logits_available);
        assert_eq!(
            with_logits.logits.as_deref(),
            Some([3.0, 4.0, 5.0].as_slice())
        );
        assert_eq!(backend.model.hidden_calls.get(), 1);
        assert_eq!(backend.model.logits_calls.get(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn llama_cpp_external_rejects_invalid_binary_path() {
        let dir = temp_test_dir("invalid_bin");
        let model = write_file(&dir, "model.gguf", "dummy");
        let samples = write_file(
            &dir,
            "samples.jsonl",
            "{\"id\":\"s0\",\"prompt\":\"hello\"}\n",
        );
        let mut config = external_config(&dir, &model, &samples, &dir.join("missing-bin"));
        config.layers.clear();
        let err = LlamaCppExternalBackend::from_config(&config).expect_err("invalid binary");
        assert!(err
            .to_string()
            .contains("invalid llama.cpp external binary path"));
    }

    #[cfg(unix)]
    #[test]
    fn llama_cpp_external_rejects_invalid_model_path() {
        let dir = temp_test_dir("invalid_model");
        let script = write_executable(&dir, "extract.sh", "#!/bin/sh\nexit 0\n");
        let samples = write_file(
            &dir,
            "samples.jsonl",
            "{\"id\":\"s0\",\"prompt\":\"hello\"}\n",
        );
        let config = external_config(&dir, &dir.join("missing.gguf"), &samples, &script);
        let err = LlamaCppExternalBackend::from_config(&config).expect_err("invalid model");
        assert!(err.to_string().contains("invalid GGUF model path"));
    }

    #[cfg(unix)]
    #[test]
    fn llama_cpp_external_rejects_unsupported_layers() {
        let dir = temp_test_dir("unsupported_layers");
        let script = write_executable(&dir, "extract.sh", "#!/bin/sh\nexit 0\n");
        let model = write_file(&dir, "model.gguf", "dummy");
        let samples = write_file(
            &dir,
            "samples.jsonl",
            "{\"id\":\"s0\",\"prompt\":\"hello\"}\n",
        );
        let mut config = external_config(&dir, &model, &samples, &script);
        config.layers = vec![0];
        let err = LlamaCppExternalBackend::from_config(&config).expect_err("unsupported layers");
        assert!(err
            .to_string()
            .contains("hidden-state layer extraction is not wired yet"));
    }

    #[cfg(unix)]
    #[test]
    fn llama_cpp_external_captures_process_stderr() {
        let dir = temp_test_dir("process_failure");
        let script = write_executable(
            &dir,
            "extract.sh",
            "#!/bin/sh\necho external extractor failed >&2\nexit 23\n",
        );
        let model = write_file(&dir, "model.gguf", "dummy");
        let samples = write_file(
            &dir,
            "samples.jsonl",
            "{\"id\":\"s0\",\"prompt\":\"hello\"}\n",
        );
        let config = external_config(&dir.join("run"), &model, &samples, &script);
        let err = run_llama_cpp_external_backend(&config).expect_err("external failure");
        let text = err.to_string();
        assert!(text.contains("external extractor failed"));
        assert!(text.contains("status"));
    }

    #[cfg(unix)]
    #[test]
    fn llama_cpp_external_validates_produced_manifest_skeleton() {
        let dir = temp_test_dir("manifest_skeleton");
        let run_dir = dir.join("run");
        let model = write_file(&dir, "model.gguf", "dummy");
        let samples = write_file(
            &dir,
            "samples.jsonl",
            "{\"id\":\"s0\",\"prompt\":\"hello\"}\n",
        );
        let prompt_hash = stable_prompt_hash("hello");
        let order_hash = sample_order_hash(&[("s0".to_string(), prompt_hash.clone())]);
        let config = external_config(&run_dir, &model, &samples, &dir.join("extract.sh"));
        let canonical_config = canonical_config_toml(&config).unwrap();
        let config_hash = stable_bytes_hash(canonical_config.as_bytes());

        let manifest = serde_json::json!({
            "schema_version": ARTIFACT_CONTRACT_VERSION,
            "layout": ARTIFACT_LAYOUT,
            "artifact_kind": "ember_hidden_states",
            "created_at_unix": 0,
            "run_id": null,
            "run_dir": run_dir.to_string_lossy(),
            "config_path": CONFIG_FILENAME,
            "samples_path": SAMPLES_FILENAME,
            "tokenization_path": TOKENIZATION_FILENAME,
            "positions_path": POSITIONS_FILENAME,
            "checksums_path": CHECKSUMS_FILENAME,
            "report_path": REPORT_FILENAME,
            "logits_path": null,
            "tensor_contract": {
                "storage": "layer-sharded-npy",
                "dtype": "f32",
                "byte_order": "little-endian",
                "sample_axis": 0,
                "hidden_axis": 1,
                "layers": [],
                "logits": null
            },
            "sample_count": 1,
            "sample_order_hash": order_hash,
            "config_hash": config_hash,
            "dtype": "f32",
            "output_format": "npy",
            "model": {
                "path": model.to_string_lossy(),
                "architecture": null,
                "n_layers": 0,
                "embed_dim": 0,
                "max_seq_len": 0,
                "file_size_bytes": null,
                "sha256": null,
                "gguf_metadata": null
            },
            "backend": {
                "name": "llama-cpp-external",
                "version": null,
                "executable": null,
                "commit": null,
                "details": {}
            },
            "extraction_config": config
        });
        let samples_content = format!(
            "{{\"schema_version\":2,\"sample_index\":0,\"sample_id\":\"s0\",\"input_index\":0,\"prompt\":\"hello\",\"prompt_hash\":\"{prompt_hash}\"}}\n"
        );
        let tokenization_content = format!(
            "{{\"schema_version\":2,\"sample_index\":0,\"sample_id\":\"s0\",\"token_ids\":[1,2,3],\"token_count\":3,\"prompt_hash\":\"{prompt_hash}\",\"offsets\":[[0,0],[0,2],[2,5]],\"offset_unit\":\"unicode_character_index\"}}\n"
        );
        let positions_content = "{\"schema_version\":2,\"sample_index\":0,\"sample_id\":\"s0\",\"position_mode\":\"prompt_final\",\"pooling\":\"single\",\"selected_token_positions\":[2],\"source_field\":null,\"source_value\":null,\"source_byte_span\":null}\n";
        let manifest_content = format!("{}\n", serde_json::to_string_pretty(&manifest).unwrap());
        let report_content = "{\"schema_version\":2,\"layout\":\"ember.layer_sharded_npy.v1\",\"status\":\"complete\",\"sample_count\":1,\"layer_count\":0,\"logits_written\":false}\n";
        let checksums = serde_json::json!({
            CONFIG_FILENAME: crate::extraction::sha256_bytes(canonical_config.as_bytes()),
            MANIFEST_FILENAME: crate::extraction::sha256_bytes(manifest_content.as_bytes()),
            SAMPLES_FILENAME: crate::extraction::sha256_bytes(samples_content.as_bytes()),
            TOKENIZATION_FILENAME: crate::extraction::sha256_bytes(tokenization_content.as_bytes()),
            POSITIONS_FILENAME: crate::extraction::sha256_bytes(positions_content.as_bytes()),
            REPORT_FILENAME: crate::extraction::sha256_bytes(report_content.as_bytes()),
        });
        let checksums_content = format!("{}\n", serde_json::to_string_pretty(&checksums).unwrap());
        let script_body = format!(
            r#"#!/bin/sh
run_dir=$(dirname "$2")
cat > "$run_dir/{samples_filename}" <<'JSON'
{samples_content}JSON
cat > "$run_dir/{tokenization_filename}" <<'JSON'
{tokenization_content}JSON
cat > "$run_dir/{positions_filename}" <<'JSON'
{positions_content}JSON
cat > "$run_dir/{manifest_filename}" <<'JSON'
{manifest_content}JSON
cat > "$run_dir/{report_filename}" <<'JSON'
{report_content}JSON
cat > "$run_dir/{checksums_filename}" <<'JSON'
{checksums_content}JSON
"#,
            samples_filename = SAMPLES_FILENAME,
            tokenization_filename = TOKENIZATION_FILENAME,
            positions_filename = POSITIONS_FILENAME,
            manifest_filename = MANIFEST_FILENAME,
            report_filename = REPORT_FILENAME,
            checksums_filename = CHECKSUMS_FILENAME,
        );
        let script = write_executable(&dir, "extract.sh", &script_body);
        let mut config = config;
        config.llama_cpp_binary = Some(script.to_string_lossy().to_string());

        let output = run_llama_cpp_external_backend(&config).expect("external skeleton validates");
        assert_eq!(output.sample_count, 1);
        assert!(output.layer_paths.is_empty());
        assert!(run_dir.join(LLAMA_CPP_REQUEST_FILENAME).is_file());
    }

    #[cfg(unix)]
    fn external_config(
        out_dir: &std::path::Path,
        model: &std::path::Path,
        samples: &std::path::Path,
        binary: &std::path::Path,
    ) -> ExtractionConfig {
        ExtractionConfig {
            run_id: None,
            model_path: model.to_string_lossy().to_string(),
            architecture: None,
            tokenizer_path: None,
            backend: ExecutionBackendName::LlamaCppExternal,
            prompt_template: "{prompt}".to_string(),
            input_jsonl_path: samples.to_string_lossy().to_string(),
            output_dir: out_dir.to_string_lossy().to_string(),
            layers: Vec::new(),
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
            llama_cpp_binary: Some(binary.to_string_lossy().to_string()),
            run_metadata: Value::Null,
        }
    }

    #[cfg(unix)]
    fn temp_test_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ember_{}_{}_{}", name, std::process::id(), unique));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[cfg(unix)]
    fn write_file(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, content).expect("write temp file");
        path
    }

    #[cfg(unix)]
    fn write_executable(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = write_file(dir, name, content);
        let mut perms = fs::metadata(&path).expect("stat script").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms).expect("chmod script");
        path
    }
}
