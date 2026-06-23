use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionBackendName {
    Native,
    LlamaCpp,
}

impl ExecutionBackendName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::LlamaCpp => "llama-cpp",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenPositionMode {
    PromptFinal,
    WordFinalSubtoken,
    WordMean,
    FullPromptMean,
}

impl TokenPositionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PromptFinal => "prompt_final",
            Self::WordFinalSubtoken => "word_final_subtoken",
            Self::WordMean => "word_mean",
            Self::FullPromptMean => "full_prompt_mean",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactDType {
    F32,
}

impl ArtifactDType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactOutputFormat {
    Npy,
}

impl ArtifactOutputFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Npy => "npy",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    pub model_path: String,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub tokenizer_path: Option<String>,
    #[serde(default = "default_backend")]
    pub backend: ExecutionBackendName,
    pub prompt_template: String,
    pub input_jsonl_path: String,
    pub output_dir: String,
    #[serde(default)]
    pub layers: Vec<usize>,
    #[serde(default = "default_token_position")]
    pub token_position: TokenPositionMode,
    #[serde(default = "default_word_field")]
    pub word_field: String,
    #[serde(default = "default_sample_id_field")]
    pub sample_id_field: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_dtype")]
    pub dtype: ArtifactDType,
    #[serde(default = "default_output_format")]
    pub output_format: ArtifactOutputFormat,
    #[serde(default)]
    pub prompt_hashes_only: bool,
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    #[serde(default)]
    pub record_model_sha256: bool,
    #[serde(default)]
    pub llama_cpp_binary: Option<String>,
    #[serde(default)]
    pub run_metadata: Value,
}

impl ExtractionConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read extraction config: {}", path.display()))?;
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => serde_json::from_str(&text)
                .with_context(|| format!("failed to parse JSON config: {}", path.display())),
            _ => toml::from_str(&text)
                .with_context(|| format!("failed to parse TOML config: {}", path.display())),
        }
    }

    pub fn validate(&self) -> Result<()> {
        require_non_empty(&self.model_path, "model_path")?;
        require_non_empty(&self.prompt_template, "prompt_template")?;
        require_non_empty(&self.input_jsonl_path, "input_jsonl_path")?;
        require_non_empty(&self.output_dir, "output_dir")?;
        require_non_empty(&self.sample_id_field, "sample_id_field")?;
        if self.batch_size == 0 {
            anyhow::bail!("batch_size must be greater than 0");
        }
        if let Some(max_seq_len) = self.max_seq_len {
            if max_seq_len == 0 {
                anyhow::bail!("max_seq_len must be greater than 0 when set");
            }
        }
        if matches!(
            self.token_position,
            TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean
        ) {
            require_non_empty(&self.word_field, "word_field")?;
        }
        let mut seen_layers = BTreeMap::new();
        for layer in &self.layers {
            if seen_layers.insert(*layer, ()).is_some() {
                anyhow::bail!("layers must not contain duplicates; repeated layer {layer}");
            }
        }
        Ok(())
    }

    pub fn with_backend_override(mut self, backend: Option<ExecutionBackendName>) -> Self {
        if let Some(backend) = backend {
            self.backend = backend;
        }
        self
    }

    pub fn effective_layers(&self, n_layers: usize) -> Result<Vec<usize>> {
        if self.layers.is_empty() {
            return Ok((0..n_layers).collect());
        }
        for &layer in &self.layers {
            if layer >= n_layers {
                anyhow::bail!(
                    "requested layer {} but model only has {} layer(s)",
                    layer,
                    n_layers
                );
            }
        }
        Ok(self.layers.clone())
    }
}

#[derive(Debug, Clone)]
pub struct ExtractionInputSample {
    pub input_index: usize,
    pub sample_id: String,
    pub prompt: String,
    pub word_value: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendMetadata {
    pub name: String,
    pub version: Option<String>,
    pub executable: Option<String>,
    pub commit: Option<String>,
    pub details: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelMetadata {
    pub path: String,
    pub architecture: Option<String>,
    pub n_layers: usize,
    pub embed_dim: usize,
    pub max_seq_len: usize,
    pub file_size_bytes: Option<u64>,
    pub sha256: Option<String>,
    pub gguf_metadata: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactMetadata {
    pub schema_version: u32,
    pub artifact_kind: String,
    pub created_at_unix: u64,
    pub output_dir: String,
    pub hidden_states_path: String,
    pub samples_path: String,
    pub hidden_states_shape: Vec<usize>,
    pub dtype: String,
    pub output_format: String,
    pub layers: Vec<usize>,
    pub token_position: String,
    pub model: ModelMetadata,
    pub backend: BackendMetadata,
    pub extraction_config: ExtractionConfig,
    pub checksums: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleArtifactRecord {
    pub schema_version: u32,
    pub sample_id: String,
    pub input_index: usize,
    pub token_ids: Vec<u32>,
    pub selected_token_positions: Vec<usize>,
    pub token_count: usize,
    pub prompt: Option<String>,
    pub prompt_hash: String,
    pub logits_available: bool,
    pub logits_shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct TokenizedPrompt {
    pub token_ids: Vec<u32>,
    pub offsets: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct BackendHiddenStateOutput {
    pub hidden_states: Vec<f32>,
    pub hidden_states_shape: Vec<usize>,
    pub logits_available: bool,
    pub logits: Option<Vec<f32>>,
    pub logits_shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct ExtractionRunOutput {
    pub output_dir: String,
    pub hidden_states_path: String,
    pub samples_path: String,
    pub metadata_path: String,
    pub sample_count: usize,
    pub hidden_states_shape: Vec<usize>,
}

pub fn load_input_samples(config: &ExtractionConfig) -> Result<Vec<ExtractionInputSample>> {
    let text = fs::read_to_string(&config.input_jsonl_path)
        .with_context(|| format!("failed to read input JSONL: {}", config.input_jsonl_path))?;
    let mut samples = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).with_context(|| {
            format!(
                "failed to parse JSONL record {} from {}",
                line_index + 1,
                config.input_jsonl_path
            )
        })?;
        let object = value.as_object().with_context(|| {
            format!(
                "JSONL record {} must be an object in {}",
                line_index + 1,
                config.input_jsonl_path
            )
        })?;
        let sample_id = object
            .get(&config.sample_id_field)
            .and_then(value_to_string)
            .unwrap_or_else(|| line_index.to_string());
        let prompt = render_prompt(&config.prompt_template, object)
            .with_context(|| format!("failed to render prompt for record {}", line_index + 1))?;
        let word_value = object.get(&config.word_field).and_then(value_to_string);
        samples.push(ExtractionInputSample {
            input_index: line_index,
            sample_id,
            prompt,
            word_value,
        });
    }
    Ok(samples)
}

pub fn render_prompt(template: &str, object: &Map<String, Value>) -> Result<String> {
    let mut rendered = template.to_string();
    for (key, value) in object {
        if let Some(text) = value_to_string(value) {
            rendered = rendered.replace(&format!("{{{{{key}}}}}"), &text);
            rendered = rendered.replace(&format!("{{{key}}}"), &text);
        }
    }
    Ok(rendered)
}

pub fn select_token_positions(
    prompt: &str,
    token_ids: &[u32],
    offsets: &[(usize, usize)],
    config: &ExtractionConfig,
    word_value: Option<&str>,
) -> Result<Vec<usize>> {
    if token_ids.is_empty() {
        anyhow::bail!("cannot select token positions from an empty prompt");
    }
    match config.token_position {
        TokenPositionMode::PromptFinal => {
            let indices = non_special_token_indices(offsets, token_ids.len());
            indices
                .last()
                .copied()
                .map(|i| vec![i])
                .context("cannot select prompt_final from an empty prompt")
        }
        TokenPositionMode::FullPromptMean => {
            Ok(non_special_token_indices(offsets, token_ids.len()))
        }
        TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean => {
            let needle = word_value.with_context(|| {
                format!(
                    "token_position '{}' requires input JSONL field '{}'",
                    config.token_position.as_str(),
                    config.word_field
                )
            })?;
            let start = prompt.find(needle).with_context(|| {
                format!(
                    "could not locate word_field '{}' value '{}' in rendered prompt",
                    config.word_field, needle
                )
            })?;
            let mut indices = token_indices_for_offsets(offsets, start, start + needle.len());
            if indices.is_empty() {
                anyhow::bail!(
                    "could not map word_field '{}' value '{}' to tokenizer offsets",
                    config.word_field,
                    needle
                );
            }
            if config.token_position == TokenPositionMode::WordFinalSubtoken {
                let last = *indices.last().expect("indices is non-empty");
                indices.clear();
                indices.push(last);
            }
            Ok(indices)
        }
    }
}

pub fn stable_prompt_hash(prompt: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in prompt.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn sha256_file(path: impl AsRef<Path>) -> Option<String> {
    let output = std::process::Command::new("sha256sum")
        .arg(path.as_ref())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    stdout.split_whitespace().next().map(str::to_string)
}

fn token_indices_for_offsets(offsets: &[(usize, usize)], start: usize, end: usize) -> Vec<usize> {
    offsets
        .iter()
        .enumerate()
        .filter_map(|(i, &(tok_start, tok_end))| {
            if tok_start != tok_end && tok_start < end && tok_end > start {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

pub fn non_special_token_indices(offsets: &[(usize, usize)], token_count: usize) -> Vec<usize> {
    let indices = offsets
        .iter()
        .enumerate()
        .filter_map(|(i, &(start, end))| if start != end { Some(i) } else { None })
        .collect::<Vec<_>>();
    if indices.is_empty() {
        (0..token_count).collect()
    } else {
        indices
    }
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn require_non_empty(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        anyhow::bail!("{field} must not be empty");
    }
    Ok(())
}

fn default_backend() -> ExecutionBackendName {
    ExecutionBackendName::Native
}

fn default_token_position() -> TokenPositionMode {
    TokenPositionMode::PromptFinal
}

fn default_word_field() -> String {
    "word".to_string()
}

fn default_sample_id_field() -> String {
    "id".to_string()
}

fn default_batch_size() -> usize {
    1
}

fn default_dtype() -> ArtifactDType {
    ArtifactDType::F32
}

fn default_output_format() -> ArtifactOutputFormat {
    ArtifactOutputFormat::Npy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_validate() {
        let config = ExtractionConfig {
            model_path: "model.gguf".to_string(),
            architecture: Some("llama".to_string()),
            tokenizer_path: None,
            backend: ExecutionBackendName::Native,
            prompt_template: "word: {word}".to_string(),
            input_jsonl_path: "input.jsonl".to_string(),
            output_dir: "out".to_string(),
            layers: vec![0, 2],
            token_position: TokenPositionMode::PromptFinal,
            word_field: "word".to_string(),
            sample_id_field: "id".to_string(),
            batch_size: 1,
            dtype: ArtifactDType::F32,
            output_format: ArtifactOutputFormat::Npy,
            prompt_hashes_only: false,
            max_seq_len: None,
            record_model_sha256: false,
            llama_cpp_binary: None,
            run_metadata: Value::Null,
        };
        config.validate().expect("valid extraction config");
        assert_eq!(config.effective_layers(4).unwrap(), vec![0, 2]);
    }

    #[test]
    fn render_prompt_replaces_single_and_double_braces() {
        let mut object = Map::new();
        object.insert("word".to_string(), Value::String("kataba".to_string()));
        let rendered = render_prompt("{word} / {{word}}", &object).unwrap();
        assert_eq!(rendered, "kataba / kataba");
    }

    #[test]
    fn prompt_final_skips_zero_width_offsets() {
        let config = ExtractionConfig {
            model_path: "model.gguf".to_string(),
            architecture: None,
            tokenizer_path: None,
            backend: ExecutionBackendName::Native,
            prompt_template: "x".to_string(),
            input_jsonl_path: "input.jsonl".to_string(),
            output_dir: "out".to_string(),
            layers: Vec::new(),
            token_position: TokenPositionMode::PromptFinal,
            word_field: "word".to_string(),
            sample_id_field: "id".to_string(),
            batch_size: 1,
            dtype: ArtifactDType::F32,
            output_format: ArtifactOutputFormat::Npy,
            prompt_hashes_only: false,
            max_seq_len: None,
            record_model_sha256: false,
            llama_cpp_binary: None,
            run_metadata: Value::Null,
        };
        let selected =
            select_token_positions("abc", &[1, 2, 3], &[(0, 0), (0, 1), (1, 3)], &config, None)
                .unwrap();
        assert_eq!(selected, vec![2]);
    }
}
