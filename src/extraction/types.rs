use super::*;

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
        let _ = prompt_template_fields(&self.prompt_template)?;
        if let Some(run_id) = &self.run_id {
            require_non_empty(run_id, "run_id")?;
            let mut components = Path::new(run_id).components();
            if !matches!(components.next(), Some(std::path::Component::Normal(_)))
                || components.next().is_some()
            {
                anyhow::bail!("run_id must be a single path component");
            }
        }
        // Validated against the same alias table the architecture resolvers
        // use, so the accepted set cannot drift from what can actually run.
        if let Some(architecture) = &self.architecture
            && crate::support::EngineFamily::from_alias(architecture).is_none()
        {
            anyhow::bail!("unsupported architecture '{architecture}'");
        }
        if self.batch_size != 1 {
            anyhow::bail!(
                "batch_size={} is unsupported; extraction currently requires batch_size=1",
                self.batch_size
            );
        }
        if self.resume {
            anyhow::bail!("resume=true is unsupported; extraction currently creates a new run");
        }
        if let Some(max_seq_len) = self.max_seq_len
            && max_seq_len == 0
        {
            anyhow::bail!("max_seq_len must be greater than 0 when set");
        }
        if matches!(
            self.token_position,
            TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean
        ) {
            require_non_empty(&self.word_field, "word_field")?;
        }
        if !self.run_metadata.is_null() && !self.run_metadata.is_object() {
            anyhow::bail!("run_metadata must be a JSON object or null");
        }
        match self.backend {
            ExecutionBackendName::Native if self.llama_cpp_binary.is_some() => {
                anyhow::bail!("llama_cpp_binary is ignored by the native backend; remove it")
            }
            ExecutionBackendName::LlamaCppExternal => {
                let binary = self
                    .llama_cpp_binary
                    .as_deref()
                    .context("llama-cpp-external backend requires llama_cpp_binary")?;
                require_non_empty(binary, "llama_cpp_binary")?;
            }
            _ => {}
        }
        let mut seen_layers = BTreeMap::new();
        for layer in &self.layers {
            if seen_layers.insert(*layer, ()).is_some() {
                anyhow::bail!("layers must not contain duplicates; repeated layer {layer}");
            }
        }
        if !self.layers.windows(2).all(|pair| pair[0] < pair[1]) {
            anyhow::bail!("layers must be in strictly increasing order");
        }
        Ok(())
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
    /// Byte span of `word_value` in `prompt`, tracked at render time so it is
    /// unambiguous even when the value appears in more than one field.
    pub word_byte_span: Option<[usize; 2]>,
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
pub struct TokenizerMetadata {
    pub path: String,
    pub file_size_bytes: u64,
    pub sha256: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerMetadata>,
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
    #[serde(default = "default_offset_unit")]
    pub offset_unit: String,
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
    /// Complete, validated configuration used to create this request. Keeping
    /// this in the request prevents external adapters from having to recreate
    /// Rust defaults or infer the base output directory from staging paths.
    pub extraction_config: ExtractionConfig,
}

#[derive(Debug, Clone)]
pub struct ArtifactValidationSummary {
    pub run_dir: String,
    pub sample_count: usize,
    pub layer_count: usize,
    pub logits_present: bool,
    pub sample_order_hash: String,
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

fn default_offset_unit() -> String {
    "unicode_character_index".to_string()
}
