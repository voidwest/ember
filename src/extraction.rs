use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const ARTIFACT_CONTRACT_VERSION: u32 = 2;
pub const ARTIFACT_LAYOUT: &str = "ember.layer_sharded_npy.v1";
pub const MANIFEST_FILENAME: &str = "manifest.json";
pub const CONFIG_FILENAME: &str = "config.toml";
pub const SAMPLES_FILENAME: &str = "samples.jsonl";
pub const TOKENIZATION_FILENAME: &str = "tokenization.jsonl";
pub const POSITIONS_FILENAME: &str = "positions.jsonl";
pub const CHECKSUMS_FILENAME: &str = "checksums.json";
pub const REPORT_FILENAME: &str = "report.json";
pub const LAYERS_DIRNAME: &str = "layers";
pub const LOGITS_FILENAME: &str = "logits.npy";
pub const LLAMA_CPP_REQUEST_FILENAME: &str = "llama_cpp_request.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionBackendName {
    Native,
    LlamaCpp,
    LlamaCppExternal,
}

impl ExecutionBackendName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::LlamaCpp => "llama-cpp",
            Self::LlamaCppExternal => "llama-cpp-external",
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
    #[serde(default)]
    pub run_id: Option<String>,
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
    pub write_logits: bool,
    #[serde(default)]
    pub resume: bool,
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
        if let Some(run_id) = &self.run_id {
            require_non_empty(run_id, "run_id")?;
            if run_id.contains('/') || run_id.contains('\\') {
                anyhow::bail!("run_id must be a single path component");
            }
        }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMetadata {
    pub name: String,
    pub version: Option<String>,
    pub executable: Option<String>,
    pub commit: Option<String>,
    pub details: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub layout: String,
    pub artifact_kind: String,
    pub created_at_unix: u64,
    pub run_id: Option<String>,
    pub run_dir: String,
    pub config_path: String,
    pub samples_path: String,
    pub tokenization_path: String,
    pub positions_path: String,
    pub checksums_path: String,
    pub report_path: String,
    pub logits_path: Option<String>,
    pub tensor_contract: TensorContract,
    pub sample_count: usize,
    pub sample_order_hash: String,
    pub config_hash: String,
    pub dtype: String,
    pub output_format: String,
    pub model: ModelMetadata,
    pub backend: BackendMetadata,
    pub extraction_config: ExtractionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorContract {
    pub storage: String,
    pub dtype: String,
    pub byte_order: String,
    pub sample_axis: usize,
    pub hidden_axis: usize,
    pub layers: Vec<LayerArtifact>,
    pub logits: Option<LogitsArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerArtifact {
    pub layer_index: usize,
    pub layer_name: String,
    pub path: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitsArtifact {
    pub path: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleArtifactRecord {
    pub schema_version: u32,
    pub sample_index: usize,
    pub sample_id: String,
    pub input_index: usize,
    pub prompt: Option<String>,
    pub prompt_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationArtifactRecord {
    pub schema_version: u32,
    pub sample_index: usize,
    pub sample_id: String,
    pub token_ids: Vec<u32>,
    pub token_count: usize,
    pub prompt_hash: String,
    pub offsets: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionArtifactRecord {
    pub schema_version: u32,
    pub sample_index: usize,
    pub sample_id: String,
    pub position_mode: String,
    pub pooling: String,
    pub selected_token_positions: Vec<usize>,
    pub source_field: Option<String>,
    pub source_value: Option<String>,
    pub source_byte_span: Option<[usize; 2]>,
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
    pub run_dir: String,
    pub manifest_path: String,
    pub samples_path: String,
    pub tokenization_path: String,
    pub positions_path: String,
    pub checksums_path: String,
    pub report_path: String,
    pub sample_count: usize,
    pub layer_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppExternalRequest {
    pub schema_version: u32,
    pub contract_version: u32,
    pub layout: String,
    pub backend: String,
    pub model_path: String,
    pub input_jsonl_path: String,
    pub output_dir: String,
    pub config_path: String,
    pub manifest_path: String,
    pub samples_path: String,
    pub tokenization_path: String,
    pub positions_path: String,
    pub checksums_path: String,
    pub report_path: String,
    pub logits_path: Option<String>,
    pub prompt_template: String,
    pub sample_id_field: String,
    pub word_field: String,
    pub token_position: String,
    pub layers: Vec<usize>,
    pub write_logits: bool,
    pub prompt_hashes_only: bool,
    pub max_seq_len: Option<usize>,
    pub run_metadata: Value,
}

#[derive(Debug, Clone)]
pub struct ArtifactValidationSummary {
    pub run_dir: String,
    pub sample_count: usize,
    pub layer_count: usize,
    pub logits_present: bool,
    pub sample_order_hash: String,
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

pub fn run_dir(config: &ExtractionConfig) -> std::path::PathBuf {
    match &config.run_id {
        Some(run_id) => Path::new(&config.output_dir).join(run_id),
        None => Path::new(&config.output_dir).to_path_buf(),
    }
}

pub fn layer_name(layer: usize) -> String {
    format!("layer_{layer:04}")
}

pub fn layer_filename(layer: usize) -> String {
    format!("{}.npy", layer_name(layer))
}

pub fn layer_relative_path(layer: usize) -> String {
    format!("{LAYERS_DIRNAME}/{}", layer_filename(layer))
}

pub fn pooling_for_mode(mode: TokenPositionMode) -> &'static str {
    match mode {
        TokenPositionMode::PromptFinal | TokenPositionMode::WordFinalSubtoken => "single",
        TokenPositionMode::WordMean | TokenPositionMode::FullPromptMean => "mean",
    }
}

pub fn source_span_for_position(
    prompt: &str,
    config: &ExtractionConfig,
    word_value: Option<&str>,
) -> Option<[usize; 2]> {
    match config.token_position {
        TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean => {
            let value = word_value?;
            let start = prompt.find(value)?;
            Some([start, start + value.len()])
        }
        TokenPositionMode::PromptFinal | TokenPositionMode::FullPromptMean => None,
    }
}

pub fn source_field_for_position(config: &ExtractionConfig) -> Option<String> {
    match config.token_position {
        TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean => {
            Some(config.word_field.clone())
        }
        TokenPositionMode::PromptFinal | TokenPositionMode::FullPromptMean => None,
    }
}

pub fn source_value_for_position(
    config: &ExtractionConfig,
    word_value: Option<&str>,
) -> Option<String> {
    match config.token_position {
        TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean => {
            word_value.map(str::to_string)
        }
        TokenPositionMode::PromptFinal | TokenPositionMode::FullPromptMean => None,
    }
}

pub fn sample_order_hash(records: &[(String, String)]) -> String {
    let mut payload = String::new();
    for (sample_id, prompt_hash) in records {
        payload.push_str(sample_id);
        payload.push('\t');
        payload.push_str(prompt_hash);
        payload.push('\n');
    }
    stable_prompt_hash(&payload)
}

pub fn validate_artifact_contract(
    run_dir: impl AsRef<Path>,
    allow_missing_layers: bool,
) -> Result<ArtifactValidationSummary> {
    let run_dir = run_dir.as_ref();
    let manifest_path = run_dir.join(MANIFEST_FILENAME);
    let manifest_text = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read manifest: {}", manifest_path.display()))?;
    let manifest: ArtifactManifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("failed to parse manifest: {}", manifest_path.display()))?;

    if manifest.schema_version != ARTIFACT_CONTRACT_VERSION {
        anyhow::bail!(
            "manifest schema_version {} does not match expected {}",
            manifest.schema_version,
            ARTIFACT_CONTRACT_VERSION
        );
    }
    if manifest.layout != ARTIFACT_LAYOUT {
        anyhow::bail!(
            "manifest layout '{}' does not match expected '{}'",
            manifest.layout,
            ARTIFACT_LAYOUT
        );
    }
    if manifest.tensor_contract.layers.is_empty() && !allow_missing_layers {
        anyhow::bail!("manifest has no layer shards");
    }

    let samples: Vec<SampleArtifactRecord> =
        read_jsonl_records(run_dir.join(&manifest.samples_path))?;
    let tokenization: Vec<TokenizationArtifactRecord> =
        read_jsonl_records(run_dir.join(&manifest.tokenization_path))?;
    let positions: Vec<PositionArtifactRecord> =
        read_jsonl_records(run_dir.join(&manifest.positions_path))?;

    if samples.len() != manifest.sample_count {
        anyhow::bail!(
            "samples.jsonl has {} rows but manifest sample_count is {}",
            samples.len(),
            manifest.sample_count
        );
    }
    if tokenization.len() != samples.len() || positions.len() != samples.len() {
        anyhow::bail!(
            "artifact row count mismatch: samples={}, tokenization={}, positions={}",
            samples.len(),
            tokenization.len(),
            positions.len()
        );
    }

    let mut order = Vec::with_capacity(samples.len());
    for (index, sample) in samples.iter().enumerate() {
        if sample.schema_version != ARTIFACT_CONTRACT_VERSION {
            anyhow::bail!(
                "sample row {index} has schema_version {}",
                sample.schema_version
            );
        }
        if sample.sample_index != index {
            anyhow::bail!(
                "samples.jsonl row {index} has sample_index {}",
                sample.sample_index
            );
        }
        let token_row = &tokenization[index];
        let position_row = &positions[index];
        if token_row.schema_version != ARTIFACT_CONTRACT_VERSION {
            anyhow::bail!(
                "tokenization row {index} has schema_version {}",
                token_row.schema_version
            );
        }
        if position_row.schema_version != ARTIFACT_CONTRACT_VERSION {
            anyhow::bail!(
                "position row {index} has schema_version {}",
                position_row.schema_version
            );
        }
        if token_row.sample_index != index || position_row.sample_index != index {
            anyhow::bail!("sample_index mismatch at row {index}");
        }
        if token_row.sample_id != sample.sample_id || position_row.sample_id != sample.sample_id {
            anyhow::bail!("sample_id mismatch at sample_index {index}");
        }
        if token_row.prompt_hash != sample.prompt_hash {
            anyhow::bail!("prompt_hash mismatch at sample_index {index}");
        }
        if token_row.token_count != token_row.token_ids.len() {
            anyhow::bail!(
                "token_count mismatch at sample_index {index}: {} vs {} token_ids",
                token_row.token_count,
                token_row.token_ids.len()
            );
        }
        if position_row.selected_token_positions.is_empty() {
            anyhow::bail!("empty selected_token_positions at sample_index {index}");
        }
        match position_row.pooling.as_str() {
            "single" => {
                if position_row.selected_token_positions.len() != 1 {
                    anyhow::bail!(
                        "single pooling at sample_index {index} selected {} positions",
                        position_row.selected_token_positions.len()
                    );
                }
            }
            "mean" => {}
            other => anyhow::bail!("unsupported pooling '{other}' at sample_index {index}"),
        }
        for position in &position_row.selected_token_positions {
            if *position >= token_row.token_count {
                anyhow::bail!(
                    "selected token position {} out of bounds for token_count {} at sample_index {index}",
                    position,
                    token_row.token_count
                );
            }
        }
        order.push((sample.sample_id.clone(), sample.prompt_hash.clone()));
    }

    let computed_order_hash = sample_order_hash(&order);
    if computed_order_hash != manifest.sample_order_hash {
        anyhow::bail!(
            "sample_order_hash mismatch: manifest {}, computed {}",
            manifest.sample_order_hash,
            computed_order_hash
        );
    }

    for layer in &manifest.tensor_contract.layers {
        if layer.shape.len() != 2 {
            anyhow::bail!("layer {} shape must be rank 2", layer.layer_name);
        }
        if layer.shape[0] != manifest.sample_count {
            anyhow::bail!(
                "layer {} first dimension {} does not match sample_count {}",
                layer.layer_name,
                layer.shape[0],
                manifest.sample_count
            );
        }
        let path = run_dir.join(&layer.path);
        if !path.is_file() {
            anyhow::bail!("missing layer shard: {}", path.display());
        }
    }
    if let Some(logits_path) = &manifest.logits_path {
        let path = run_dir.join(logits_path);
        if !path.is_file() {
            anyhow::bail!(
                "manifest declares logits but file is missing: {}",
                path.display()
            );
        }
    }

    let report_path = run_dir.join(&manifest.report_path);
    let report_text = fs::read_to_string(&report_path)
        .with_context(|| format!("failed to read report: {}", report_path.display()))?;
    let report: Value = serde_json::from_str(&report_text)
        .with_context(|| format!("failed to parse report: {}", report_path.display()))?;
    if report.get("status").and_then(Value::as_str) != Some("complete") {
        anyhow::bail!("report status is not complete");
    }

    let checksums_path = run_dir.join(&manifest.checksums_path);
    let checksums_text = fs::read_to_string(&checksums_path)
        .with_context(|| format!("failed to read checksums: {}", checksums_path.display()))?;
    let checksums: BTreeMap<String, String> = serde_json::from_str(&checksums_text)
        .with_context(|| format!("failed to parse checksums: {}", checksums_path.display()))?;
    for (relative_path, expected) in checksums {
        let path = run_dir.join(&relative_path);
        if !path.is_file() {
            anyhow::bail!("checksums.json references missing file: {relative_path}");
        }
        if let Some(actual) = sha256_file(&path) {
            if actual != expected {
                anyhow::bail!(
                    "checksum mismatch for {relative_path}: expected {expected}, got {actual}"
                );
            }
        }
    }

    Ok(ArtifactValidationSummary {
        run_dir: run_dir
            .to_str()
            .map(str::to_string)
            .unwrap_or_else(|| run_dir.display().to_string()),
        sample_count: manifest.sample_count,
        layer_count: manifest.tensor_contract.layers.len(),
        logits_present: manifest.logits_path.is_some(),
        sample_order_hash: manifest.sample_order_hash,
    })
}

pub fn read_jsonl_records<T>(path: impl AsRef<Path>) -> Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read JSONL artifact: {}", path.display()))?;
    let mut records = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record = serde_json::from_str(line).with_context(|| {
            format!(
                "failed to parse JSONL line {} from {}",
                line_index + 1,
                path.display()
            )
        })?;
        records.push(record);
    }
    Ok(records)
}

pub fn canonical_config_toml(config: &ExtractionConfig) -> Result<String> {
    let mut lines = Vec::new();
    if let Some(run_id) = &config.run_id {
        lines.push(toml_string_line("run_id", run_id));
    }
    lines.push(toml_string_line("model_path", &config.model_path));
    if let Some(architecture) = &config.architecture {
        lines.push(toml_string_line("architecture", architecture));
    }
    if let Some(tokenizer_path) = &config.tokenizer_path {
        lines.push(toml_string_line("tokenizer_path", tokenizer_path));
    }
    lines.push(toml_string_line("backend", config.backend.as_str()));
    lines.push(toml_string_line("prompt_template", &config.prompt_template));
    lines.push(toml_string_line(
        "input_jsonl_path",
        &config.input_jsonl_path,
    ));
    lines.push(toml_string_line("output_dir", &config.output_dir));
    lines.push(format!(
        "layers = [{}]",
        config
            .layers
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    ));
    lines.push(toml_string_line(
        "token_position",
        config.token_position.as_str(),
    ));
    lines.push(toml_string_line("word_field", &config.word_field));
    lines.push(toml_string_line("sample_id_field", &config.sample_id_field));
    lines.push(format!("batch_size = {}", config.batch_size));
    lines.push(toml_string_line("dtype", config.dtype.as_str()));
    lines.push(toml_string_line(
        "output_format",
        config.output_format.as_str(),
    ));
    lines.push(format!(
        "prompt_hashes_only = {}",
        config.prompt_hashes_only
    ));
    lines.push(format!("write_logits = {}", config.write_logits));
    lines.push(format!("resume = {}", config.resume));
    if let Some(max_seq_len) = config.max_seq_len {
        lines.push(format!("max_seq_len = {max_seq_len}"));
    }
    lines.push(format!(
        "record_model_sha256 = {}",
        config.record_model_sha256
    ));
    if let Some(binary) = &config.llama_cpp_binary {
        lines.push(toml_string_line("llama_cpp_binary", binary));
    }
    if !config.run_metadata.is_null() {
        let run_metadata_json = serde_json::to_string(&config.run_metadata)?;
        lines.push(toml_string_line("run_metadata_json", &run_metadata_json));
    }
    lines.push(String::new());
    Ok(lines.join("\n"))
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

pub fn stable_bytes_hash(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
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

fn toml_string_line(key: &str, value: &str) -> String {
    format!("{key} = \"{}\"", escape_toml_string(value))
}

fn escape_toml_string(value: &str) -> String {
    value
        .chars()
        .flat_map(|c| match c {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            c => vec![c],
        })
        .collect()
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
            run_id: None,
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
            write_logits: false,
            resume: false,
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
    fn contract_names_layers_and_pooling_stably() {
        assert_eq!(ARTIFACT_CONTRACT_VERSION, 2);
        assert_eq!(ARTIFACT_LAYOUT, "ember.layer_sharded_npy.v1");
        assert_eq!(layer_name(4), "layer_0004");
        assert_eq!(layer_relative_path(4), "layers/layer_0004.npy");
        assert_eq!(pooling_for_mode(TokenPositionMode::PromptFinal), "single");
        assert_eq!(pooling_for_mode(TokenPositionMode::WordMean), "mean");

        let order_a = sample_order_hash(&[
            ("a".to_string(), "fnv1a64:1111".to_string()),
            ("b".to_string(), "fnv1a64:2222".to_string()),
        ]);
        let order_b = sample_order_hash(&[
            ("b".to_string(), "fnv1a64:2222".to_string()),
            ("a".to_string(), "fnv1a64:1111".to_string()),
        ]);
        assert_ne!(order_a, order_b);
    }

    #[test]
    fn prompt_final_skips_zero_width_offsets() {
        let config = ExtractionConfig {
            run_id: None,
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
            write_logits: false,
            resume: false,
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
