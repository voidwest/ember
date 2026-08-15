use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
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

static RUN_STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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
        if let Some(architecture) = &self.architecture {
            if !matches!(
                architecture.as_str(),
                "gpt2" | "llama" | "qwen2" | "qwen3" | "gemma3" | "gemma4"
            ) {
                anyhow::bail!("unsupported architecture '{architecture}'");
            }
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

pub fn load_input_samples(config: &ExtractionConfig) -> Result<Vec<ExtractionInputSample>> {
    let text = fs::read_to_string(&config.input_jsonl_path)
        .with_context(|| format!("failed to read input JSONL: {}", config.input_jsonl_path))?;
    let mut samples = Vec::new();
    let mut sample_ids = HashSet::new();
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
            .with_context(|| {
                format!(
                    "JSONL record {} is missing scalar sample_id_field '{}'",
                    line_index + 1,
                    config.sample_id_field
                )
            })?;
        if sample_id.trim().is_empty() {
            anyhow::bail!(
                "JSONL record {} has an empty sample ID in field '{}'",
                line_index + 1,
                config.sample_id_field
            );
        }
        if !sample_ids.insert(sample_id.clone()) {
            anyhow::bail!(
                "duplicate sample ID '{sample_id}' at JSONL record {}",
                line_index + 1
            );
        }
        let (prompt, word_byte_span) = if matches!(
            config.token_position,
            TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean
        ) {
            let (prompt, span) =
                render_prompt_with_span(&config.prompt_template, object, &config.word_field)
                    .with_context(|| {
                        format!("failed to render prompt for record {}", line_index + 1)
                    })?;
            (prompt, Some(span))
        } else {
            let prompt = render_prompt(&config.prompt_template, object).with_context(|| {
                format!("failed to render prompt for record {}", line_index + 1)
            })?;
            (prompt, None)
        };
        if prompt.trim().is_empty() {
            anyhow::bail!(
                "rendered prompt for JSONL record {} is empty",
                line_index + 1
            );
        }
        let word_value = object.get(&config.word_field).and_then(value_to_string);
        if matches!(
            config.token_position,
            TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean
        ) && word_value.as_deref().is_none_or(str::is_empty)
        {
            anyhow::bail!(
                "JSONL record {} is missing non-empty scalar word_field '{}'",
                line_index + 1,
                config.word_field
            );
        }
        samples.push(ExtractionInputSample {
            input_index: line_index,
            sample_id,
            prompt,
            word_value,
            word_byte_span,
        });
    }
    if samples.is_empty() {
        anyhow::bail!(
            "input JSONL contains no samples: {}",
            config.input_jsonl_path
        );
    }
    Ok(samples)
}

pub fn render_prompt(template: &str, object: &Map<String, Value>) -> Result<String> {
    let fields = prompt_template_fields(template)?;
    let mut rendered = template.to_string();
    for key in fields {
        let text = object
            .get(&key)
            .and_then(value_to_string)
            .with_context(|| format!("prompt template field '{key}' is missing or not scalar"))?;
        rendered = rendered.replace(&format!("{{{{{key}}}}}"), &text);
        rendered = rendered.replace(&format!("{{{key}}}"), &text);
    }
    Ok(rendered)
}

/// Render a prompt template and record the byte span of the first occurrence
/// of `tracked_field`'s rendered value. Fields are substituted in template
/// order, so the recorded span is the placeholder's position in the final
/// string — unlike substring search, this stays correct when the same value
/// appears in more than one field (e.g. surface repeated in Surface:/Token:).
pub fn render_prompt_with_span(
    template: &str,
    object: &Map<String, Value>,
    tracked_field: &str,
) -> Result<(String, [usize; 2])> {
    let fields = prompt_template_fields(template)?;
    let mut rendered = template.to_string();
    let mut tracked: Option<[usize; 2]> = None;
    for key in fields {
        let text = object
            .get(&key)
            .and_then(value_to_string)
            .with_context(|| format!("prompt template field '{key}' is missing or not scalar"))?;
        for needle in [format!("{{{{{key}}}}}"), format!("{{{key}}}")] {
            if let Some(pos) = rendered.find(&needle) {
                rendered.replace_range(pos..pos + needle.len(), &text);
                if key == tracked_field && tracked.is_none() {
                    tracked = Some([pos, pos + text.len()]);
                }
            }
        }
    }
    let span = tracked
        .with_context(|| format!("prompt template has no '{{{tracked_field}}}' placeholder"))?;
    Ok((rendered, span))
}

fn prompt_template_fields(template: &str) -> Result<Vec<String>> {
    let mut fields = Vec::new();
    let mut cursor = 0usize;
    while let Some(relative_start) = template[cursor..].find('{') {
        let start = cursor + relative_start;
        if template[cursor..start].contains('}') {
            anyhow::bail!("prompt template contains an unmatched closing brace");
        }
        let double = template[start..].starts_with("{{");
        let content_start = start + if double { 2 } else { 1 };
        let closing = if double { "}}" } else { "}" };
        let relative_end = template[content_start..]
            .find(closing)
            .context("prompt template contains an unmatched opening brace")?;
        let end = content_start + relative_end;
        let field = &template[content_start..end];
        if field.is_empty()
            || !field.chars().all(|character| {
                character.is_alphanumeric() || matches!(character, '_' | '-' | '.')
            })
        {
            anyhow::bail!("invalid prompt template field '{{{field}}}'");
        }
        if !fields.iter().any(|existing| existing == field) {
            fields.push(field.to_string());
        }
        cursor = end + closing.len();
    }
    if template[cursor..].contains('}') {
        anyhow::bail!("prompt template contains an unmatched closing brace");
    }
    Ok(fields)
}

pub fn run_dir(config: &ExtractionConfig) -> std::path::PathBuf {
    match &config.run_id {
        Some(run_id) => Path::new(&config.output_dir).join(run_id),
        None => Path::new(&config.output_dir).to_path_buf(),
    }
}

/// Fresh-run transaction for a complete extraction directory. Artifacts are
/// built and validated in a sibling staging directory, then one rename
/// publishes the run. Existing runs are never overwritten because resume is
/// not implemented by the contract.
pub struct RunDirectoryTransaction {
    final_path: std::path::PathBuf,
    staging_path: std::path::PathBuf,
    committed: bool,
}

impl RunDirectoryTransaction {
    pub fn begin(final_path: impl AsRef<Path>) -> Result<Self> {
        let final_path = final_path.as_ref().to_path_buf();
        if final_path.exists() {
            anyhow::bail!(
                "extraction run directory already exists and resume is unsupported: {}",
                final_path.display()
            );
        }
        let parent = final_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create extraction parent directory: {}",
                parent.display()
            )
        })?;
        let filename = final_path
            .file_name()
            .context("extraction output directory must have a final path component")?
            .to_string_lossy();
        for _ in 0..128 {
            let sequence = RUN_STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let staging_path = parent.join(format!(
                ".{filename}.ember-staging-{}-{sequence}",
                std::process::id()
            ));
            match fs::create_dir(&staging_path) {
                Ok(()) => {
                    return Ok(Self {
                        final_path,
                        staging_path,
                        committed: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "failed to create extraction staging directory next to {}",
                            final_path.display()
                        )
                    });
                }
            }
        }
        anyhow::bail!(
            "could not allocate a unique staging directory next to {}",
            final_path.display()
        )
    }

    pub fn final_path(&self) -> &Path {
        &self.final_path
    }

    pub fn staging_path(&self) -> &Path {
        &self.staging_path
    }

    pub fn commit(mut self) -> Result<std::path::PathBuf> {
        fs::rename(&self.staging_path, &self.final_path).with_context(|| {
            format!(
                "failed to publish extraction run '{}' from staging '{}'",
                self.final_path.display(),
                self.staging_path.display()
            )
        })?;
        self.committed = true;
        Ok(self.final_path.clone())
    }
}

impl Drop for RunDirectoryTransaction {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir_all(&self.staging_path);
        }
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
    config: &ExtractionConfig,
    word_byte_span: Option<[usize; 2]>,
) -> Result<Option<[usize; 2]>> {
    match config.token_position {
        TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean => Ok(word_byte_span),
        TokenPositionMode::PromptFinal | TokenPositionMode::FullPromptMean => Ok(None),
    }
}

/// Locate a source value exactly once in a rendered prompt. Byte spans are
/// retained in artifacts because they can slice UTF-8 text losslessly.
pub fn unique_substring_byte_span(prompt: &str, needle: &str) -> Result<[usize; 2]> {
    if needle.is_empty() {
        anyhow::bail!("cannot locate an empty source value in a prompt");
    }
    let mut matches = prompt.match_indices(needle);
    let (start, _) = matches
        .next()
        .with_context(|| format!("could not locate source value '{needle}' in rendered prompt"))?;
    if matches.next().is_some() {
        anyhow::bail!(
            "source value '{needle}' occurs more than once in rendered prompt; token position is ambiguous"
        );
    }
    Ok([start, start + needle.len()])
}

/// Convert a UTF-8 byte span into the Unicode-character offset unit emitted by
/// Hugging Face tokenizers.
pub fn byte_span_to_character_span(text: &str, byte_span: [usize; 2]) -> Result<[usize; 2]> {
    let [start, end] = byte_span;
    if start > end
        || end > text.len()
        || !text.is_char_boundary(start)
        || !text.is_char_boundary(end)
    {
        anyhow::bail!(
            "invalid UTF-8 byte span [{start}, {end}] for text with {} bytes",
            text.len()
        );
    }
    Ok([text[..start].chars().count(), text[..end].chars().count()])
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
    if !run_dir.is_dir() {
        anyhow::bail!(
            "artifact run directory does not exist: {}",
            run_dir.display()
        );
    }
    let canonical_run_dir = fs::canonicalize(run_dir).with_context(|| {
        format!(
            "failed to canonicalize run directory: {}",
            run_dir.display()
        )
    })?;
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
    if manifest.artifact_kind != "ember_hidden_states" {
        anyhow::bail!(
            "manifest artifact_kind '{}' does not match 'ember_hidden_states'",
            manifest.artifact_kind
        );
    }
    manifest.extraction_config.validate()?;
    if manifest.model.path != manifest.extraction_config.model_path {
        anyhow::bail!("manifest model path does not match extraction_config.model_path");
    }
    if manifest.backend.name != manifest.extraction_config.backend.as_str() {
        anyhow::bail!(
            "manifest backend '{}' does not match extraction config '{}'",
            manifest.backend.name,
            manifest.extraction_config.backend.as_str()
        );
    }
    if manifest.sample_count == 0 {
        anyhow::bail!("manifest sample_count must be greater than zero");
    }
    validate_stable_hash(&manifest.sample_order_hash, "manifest sample_order_hash")?;
    validate_stable_hash(&manifest.config_hash, "manifest config_hash")?;
    if manifest.dtype != "f32" || manifest.output_format != "npy" {
        anyhow::bail!(
            "manifest requires dtype=f32 and output_format=npy, got dtype='{}' output_format='{}'",
            manifest.dtype,
            manifest.output_format
        );
    }
    if manifest.backend.name.trim().is_empty() {
        anyhow::bail!("manifest backend.name is empty");
    }
    if let Some(sha256) = &manifest.model.sha256 {
        validate_sha256(sha256, "manifest model.sha256")?;
    }
    let model_path = Path::new(&manifest.model.path);
    if model_path.is_file() {
        if let Some(expected_size) = manifest.model.file_size_bytes {
            let actual_size = fs::metadata(model_path)
                .with_context(|| format!("failed to stat model: {}", model_path.display()))?
                .len();
            if actual_size != expected_size {
                anyhow::bail!(
                    "model file size mismatch: manifest {expected_size}, actual {actual_size}"
                );
            }
        }
        if let Some(expected_sha256) = &manifest.model.sha256 {
            let actual_sha256 = sha256_file_result(model_path)?;
            if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
                anyhow::bail!("model SHA-256 does not match the manifest");
            }
        }
    }
    if let Some(tokenizer) = &manifest.tokenizer {
        require_non_empty(&tokenizer.path, "manifest tokenizer.path")?;
        validate_sha256(&tokenizer.sha256, "manifest tokenizer.sha256")?;
        let tokenizer_path = Path::new(&tokenizer.path);
        if tokenizer_path.is_file() {
            let actual_size = fs::metadata(tokenizer_path)
                .with_context(|| format!("failed to stat tokenizer: {}", tokenizer_path.display()))?
                .len();
            if actual_size != tokenizer.file_size_bytes {
                anyhow::bail!(
                    "tokenizer file size mismatch: manifest {}, actual {actual_size}",
                    tokenizer.file_size_bytes
                );
            }
            let actual_sha256 = sha256_file_result(tokenizer_path)?;
            if !actual_sha256.eq_ignore_ascii_case(&tokenizer.sha256) {
                anyhow::bail!("tokenizer SHA-256 does not match the manifest");
            }
        }
    }
    if manifest.tensor_contract.storage != "layer-sharded-npy"
        || manifest.tensor_contract.dtype != "f32"
        || manifest.tensor_contract.byte_order != "little-endian"
        || manifest.tensor_contract.sample_axis != 0
        || manifest.tensor_contract.hidden_axis != 1
    {
        anyhow::bail!("manifest tensor_contract does not match the layer-sharded f32 contract");
    }
    if manifest.config_path != CONFIG_FILENAME
        || manifest.samples_path != SAMPLES_FILENAME
        || manifest.tokenization_path != TOKENIZATION_FILENAME
        || manifest.positions_path != POSITIONS_FILENAME
        || manifest.checksums_path != CHECKSUMS_FILENAME
        || manifest.report_path != REPORT_FILENAME
    {
        anyhow::bail!("manifest core paths do not match the canonical artifact layout");
    }
    if manifest.tensor_contract.layers.is_empty() && !allow_missing_layers {
        anyhow::bail!("manifest has no layer shards");
    }

    let config_path = resolve_artifact_path(
        &canonical_run_dir,
        &manifest.config_path,
        "manifest config_path",
    )?;
    let samples_path = resolve_artifact_path(
        &canonical_run_dir,
        &manifest.samples_path,
        "manifest samples_path",
    )?;
    let tokenization_path = resolve_artifact_path(
        &canonical_run_dir,
        &manifest.tokenization_path,
        "manifest tokenization_path",
    )?;
    let positions_path = resolve_artifact_path(
        &canonical_run_dir,
        &manifest.positions_path,
        "manifest positions_path",
    )?;
    let report_path = resolve_artifact_path(
        &canonical_run_dir,
        &manifest.report_path,
        "manifest report_path",
    )?;
    let checksums_path = resolve_artifact_path(
        &canonical_run_dir,
        &manifest.checksums_path,
        "manifest checksums_path",
    )?;
    let declared_core_paths = [
        manifest.config_path.as_str(),
        MANIFEST_FILENAME,
        manifest.samples_path.as_str(),
        manifest.tokenization_path.as_str(),
        manifest.positions_path.as_str(),
        manifest.report_path.as_str(),
        manifest.checksums_path.as_str(),
    ];
    if declared_core_paths
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len()
        != declared_core_paths.len()
    {
        anyhow::bail!("manifest core artifact paths must be distinct");
    }

    let config_bytes = fs::read(&config_path)
        .with_context(|| format!("failed to read config artifact: {}", config_path.display()))?;
    let computed_config_hash = stable_bytes_hash(&config_bytes);
    if computed_config_hash != manifest.config_hash {
        anyhow::bail!(
            "config_hash mismatch: manifest {}, computed {}",
            manifest.config_hash,
            computed_config_hash
        );
    }

    let mut expected_checksum_paths = HashSet::from([
        manifest.config_path.clone(),
        MANIFEST_FILENAME.to_string(),
        manifest.samples_path.clone(),
        manifest.tokenization_path.clone(),
        manifest.positions_path.clone(),
        manifest.report_path.clone(),
    ]);
    for layer in &manifest.tensor_contract.layers {
        if !expected_checksum_paths.insert(layer.path.clone()) {
            anyhow::bail!("layer path '{}' collides with another artifact", layer.path);
        }
    }
    if let Some(path) = &manifest.logits_path {
        if !expected_checksum_paths.insert(path.clone()) {
            anyhow::bail!("logits path '{path}' collides with another artifact");
        }
    }

    let checksums_text = fs::read_to_string(&checksums_path)
        .with_context(|| format!("failed to read checksums: {}", checksums_path.display()))?;
    let checksums: BTreeMap<String, String> = serde_json::from_str(&checksums_text)
        .with_context(|| format!("failed to parse checksums: {}", checksums_path.display()))?;
    for expected_path in &expected_checksum_paths {
        if !checksums.contains_key(expected_path) {
            anyhow::bail!("checksums.json is missing required artifact: {expected_path}");
        }
    }
    for (relative_path, expected) in &checksums {
        if !expected_checksum_paths.contains(relative_path) {
            anyhow::bail!("checksums.json contains undeclared artifact: {relative_path}");
        }
        validate_sha256(expected, &format!("checksum for {relative_path}"))?;
        let path = resolve_artifact_path(
            &canonical_run_dir,
            relative_path,
            "checksums.json artifact path",
        )?;
        let actual = sha256_file_result(&path)?;
        if !actual.eq_ignore_ascii_case(expected) {
            anyhow::bail!(
                "checksum mismatch for {relative_path}: expected {expected}, got {actual}"
            );
        }
    }

    let samples: Vec<SampleArtifactRecord> = read_jsonl_records(&samples_path)?;
    let tokenization: Vec<TokenizationArtifactRecord> = read_jsonl_records(&tokenization_path)?;
    let positions: Vec<PositionArtifactRecord> = read_jsonl_records(&positions_path)?;

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
    let mut sample_ids = HashSet::new();
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
        if sample.sample_id.trim().is_empty() {
            anyhow::bail!("samples.jsonl row {index} has an empty sample_id");
        }
        if !sample_ids.insert(sample.sample_id.clone()) {
            anyhow::bail!(
                "samples.jsonl contains duplicate sample_id '{}'",
                sample.sample_id
            );
        }
        validate_stable_hash(&sample.prompt_hash, &format!("sample {index} prompt_hash"))?;
        if manifest.extraction_config.prompt_hashes_only == sample.prompt.is_some() {
            anyhow::bail!(
                "sample prompt presence at sample_index {index} does not match prompt_hashes_only={}",
                manifest.extraction_config.prompt_hashes_only
            );
        }
        if let Some(prompt) = &sample.prompt {
            let computed = stable_prompt_hash(prompt);
            if computed != sample.prompt_hash {
                anyhow::bail!(
                    "prompt_hash mismatch at sample_index {index}: stored {}, computed {computed}",
                    sample.prompt_hash
                );
            }
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
        if token_row.token_ids.is_empty() {
            anyhow::bail!("empty token_ids at sample_index {index}");
        }
        if token_row.offset_unit != "unicode_character_index" {
            anyhow::bail!(
                "unsupported offset_unit '{}' at sample_index {index}",
                token_row.offset_unit
            );
        }
        validate_token_offsets(
            sample.prompt.as_deref(),
            &token_row.token_ids,
            &token_row.offsets,
            index,
        )?;
        if position_row.position_mode != manifest.extraction_config.token_position.as_str() {
            anyhow::bail!(
                "position_mode '{}' at sample_index {index} does not match extraction config '{}'",
                position_row.position_mode,
                manifest.extraction_config.token_position.as_str()
            );
        }
        let expected_pooling = pooling_for_mode(manifest.extraction_config.token_position);
        if position_row.pooling != expected_pooling {
            anyhow::bail!(
                "pooling '{}' at sample_index {index} does not match expected '{expected_pooling}'",
                position_row.pooling
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
        if !position_row
            .selected_token_positions
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        {
            anyhow::bail!(
                "selected_token_positions must be strictly increasing at sample_index {index}"
            );
        }
        let expected_source_field = source_field_for_position(&manifest.extraction_config);
        if position_row.source_field != expected_source_field {
            anyhow::bail!("source_field mismatch at sample_index {index}");
        }
        if matches!(
            manifest.extraction_config.token_position,
            TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean
        ) && position_row
            .source_value
            .as_deref()
            .is_none_or(str::is_empty)
        {
            anyhow::bail!("word-based position has no source_value at sample_index {index}");
        }
        if let (Some(prompt), Some(source_value)) = (
            sample.prompt.as_deref(),
            position_row.source_value.as_deref(),
        ) {
            if let Some([start, end]) = position_row.source_byte_span {
                if prompt.get(start..end) != Some(source_value) {
                    anyhow::bail!(
                        "source_byte_span [{start}, {end}] does not slice to source_value at sample_index {index}"
                    );
                }
            }
        }
        if !matches!(
            manifest.extraction_config.token_position,
            TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean
        ) && (position_row.source_value.is_some() || position_row.source_byte_span.is_some())
        {
            anyhow::bail!(
                "prompt-wide position unexpectedly declares source value/span at sample_index {index}"
            );
        }
        if sample.prompt.is_some() {
            let recomputed = select_token_positions(
                &token_row.token_ids,
                &token_row.offsets,
                &manifest.extraction_config,
                position_row.source_byte_span,
            )?;
            if recomputed != position_row.selected_token_positions {
                anyhow::bail!(
                    "selected_token_positions do not reproduce at sample_index {index}: stored {:?}, computed {:?}",
                    position_row.selected_token_positions,
                    recomputed
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

    let mut previous_layer = None;
    let mut layer_paths = HashSet::new();
    for layer in &manifest.tensor_contract.layers {
        if previous_layer.is_some_and(|previous| layer.layer_index <= previous) {
            anyhow::bail!("manifest layer indices must be strictly increasing");
        }
        previous_layer = Some(layer.layer_index);
        if layer.layer_index >= manifest.model.n_layers {
            anyhow::bail!(
                "layer index {} exceeds model n_layers {}",
                layer.layer_index,
                manifest.model.n_layers
            );
        }
        if layer.layer_name != layer_name(layer.layer_index)
            || layer.path != layer_relative_path(layer.layer_index)
        {
            anyhow::bail!(
                "layer {} name/path does not match canonical layer-sharded layout",
                layer.layer_index
            );
        }
        if !layer_paths.insert(layer.path.clone()) {
            anyhow::bail!("duplicate layer artifact path '{}'", layer.path);
        }
        let expected_shape = vec![manifest.sample_count, manifest.model.embed_dim];
        if layer.shape != expected_shape {
            anyhow::bail!(
                "layer {} shape {:?} does not match expected {:?}",
                layer.layer_name,
                layer.shape,
                expected_shape
            );
        }
        let path = resolve_artifact_path(&canonical_run_dir, &layer.path, "layer artifact path")?;
        let (actual_shape, values) = crate::npy::read_npy_2d(
            path.to_str()
                .with_context(|| format!("layer path is not UTF-8: {}", path.display()))?,
        )?;
        if actual_shape != layer.shape {
            anyhow::bail!(
                "layer {} npy shape {:?} does not match manifest {:?}",
                layer.layer_name,
                actual_shape,
                layer.shape
            );
        }
        validate_finite_tensor(&values, &format!("layer {}", layer.layer_name))?;
    }

    match (&manifest.logits_path, &manifest.tensor_contract.logits) {
        (None, None) => {}
        (Some(path), Some(logits)) if path == &logits.path => {
            if logits.shape.len() != 2
                || logits.shape[0] != manifest.sample_count
                || logits.shape[1] == 0
            {
                anyhow::bail!("invalid logits shape in tensor contract: {:?}", logits.shape);
            }
            let resolved = resolve_artifact_path(
                &canonical_run_dir,
                path,
                "manifest logits_path",
            )?;
            let (actual_shape, values) = crate::npy::read_npy_2d(
                resolved.to_str().with_context(|| {
                    format!("logits path is not UTF-8: {}", resolved.display())
                })?,
            )?;
            if actual_shape != logits.shape {
                anyhow::bail!(
                    "logits npy shape {:?} does not match manifest {:?}",
                    actual_shape,
                    logits.shape
                );
            }
            validate_finite_tensor(&values, "logits")?;
        }
        _ => anyhow::bail!(
            "manifest logits_path and tensor_contract.logits must either both be absent or declare the same path"
        ),
    }

    let report_text = fs::read_to_string(&report_path)
        .with_context(|| format!("failed to read report: {}", report_path.display()))?;
    let report: Value = serde_json::from_str(&report_text)
        .with_context(|| format!("failed to parse report: {}", report_path.display()))?;
    if report.get("status").and_then(Value::as_str) != Some("complete") {
        anyhow::bail!("report status is not complete");
    }
    if report.get("schema_version").and_then(Value::as_u64)
        != Some(u64::from(ARTIFACT_CONTRACT_VERSION))
        || report.get("layout").and_then(Value::as_str) != Some(ARTIFACT_LAYOUT)
        || report.get("sample_count").and_then(Value::as_u64)
            != u64::try_from(manifest.sample_count).ok()
        || report.get("layer_count").and_then(Value::as_u64)
            != u64::try_from(manifest.tensor_contract.layers.len()).ok()
        || report.get("logits_written").and_then(Value::as_bool)
            != Some(manifest.logits_path.is_some())
    {
        anyhow::bail!("report schema/layout/count/logits fields do not match the manifest");
    }

    Ok(ArtifactValidationSummary {
        run_dir: canonical_run_dir
            .to_str()
            .map(str::to_string)
            .unwrap_or_else(|| run_dir.display().to_string()),
        sample_count: manifest.sample_count,
        layer_count: manifest.tensor_contract.layers.len(),
        logits_present: manifest.logits_path.is_some(),
        sample_order_hash: manifest.sample_order_hash,
    })
}

fn resolve_artifact_path(
    canonical_run_dir: &Path,
    relative_path: &str,
    field: &str,
) -> Result<std::path::PathBuf> {
    require_non_empty(relative_path, field)?;
    let relative = Path::new(relative_path);
    if relative.is_absolute()
        || !relative
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        anyhow::bail!("{field} must be a normalized relative artifact path: {relative_path}");
    }
    let joined = canonical_run_dir.join(relative);
    if !joined.is_file() {
        anyhow::bail!("{field} is missing: {}", joined.display());
    }
    let canonical = fs::canonicalize(&joined)
        .with_context(|| format!("failed to canonicalize artifact: {}", joined.display()))?;
    if !canonical.starts_with(canonical_run_dir) {
        anyhow::bail!(
            "{field} escapes the artifact run directory through a symlink: {}",
            joined.display()
        );
    }
    Ok(canonical)
}

fn validate_stable_hash(value: &str, field: &str) -> Result<()> {
    let Some(hex) = value.strip_prefix("fnv1a64:") else {
        anyhow::bail!("{field} must use the fnv1a64:<16 hex digits> format");
    };
    if hex.len() != 16 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("{field} must use the fnv1a64:<16 hex digits> format");
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("{field} must contain exactly 64 hexadecimal digits");
    }
    Ok(())
}

pub(crate) fn validate_token_offsets(
    prompt: Option<&str>,
    token_ids: &[u32],
    offsets: &[(usize, usize)],
    sample_index: usize,
) -> Result<()> {
    if offsets.len() != token_ids.len() {
        anyhow::bail!(
            "offset count {} does not match token count {} at sample_index {sample_index}",
            offsets.len(),
            token_ids.len()
        );
    }
    let byte_length = prompt.map(|text| text.len());
    let mut previous = None;
    for (token_index, &(start, end)) in offsets.iter().enumerate() {
        if start > end || byte_length.is_some_and(|len| end > len) {
            anyhow::bail!(
                "invalid token offset ({start}, {end}) for token {token_index} at sample_index {sample_index}"
            );
        }
        if start == end {
            continue;
        }
        if previous.is_some_and(|(previous_start, previous_end)| {
            start < previous_start || end < previous_end
        }) {
            anyhow::bail!(
                "non-monotonic token offset ({start}, {end}) for token {token_index} at sample_index {sample_index}"
            );
        }
        previous = Some((start, end));
    }
    Ok(())
}

fn validate_finite_tensor(values: &[f32], artifact: &str) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("{artifact} contains non-finite value {value} at flat index {index}");
    }
    Ok(())
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
    token_ids: &[u32],
    offsets: &[(usize, usize)],
    config: &ExtractionConfig,
    word_byte_span: Option<[usize; 2]>,
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
            let [start, end] = word_byte_span.with_context(|| {
                format!(
                    "token_position '{}' requires input JSONL field '{}'",
                    config.token_position.as_str(),
                    config.word_field
                )
            })?;
            let mut indices = token_indices_for_offsets(offsets, start, end);
            if indices.is_empty() {
                anyhow::bail!(
                    "could not map word_field '{}' byte span [{start}, {end}] to tokenizer offsets",
                    config.word_field
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
    sha256_file_result(path).ok()
}

pub fn sha256_file_result(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open file for SHA-256: {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .with_context(|| format!("failed to hash file: {}", path.display()))?;
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn git_commit() -> Option<String> {
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

fn default_offset_unit() -> String {
    "unicode_character_index".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_config() -> ExtractionConfig {
        ExtractionConfig {
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
        }
    }

    #[test]
    fn config_defaults_validate() {
        let config = test_config();
        config.validate().expect("valid extraction config");
        assert_eq!(config.effective_layers(4).unwrap(), vec![0, 2]);
    }

    #[test]
    fn config_rejects_ignored_execution_flags() {
        let mut config = test_config();
        config.batch_size = 2;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("requires batch_size=1"));

        config.batch_size = 1;
        config.resume = true;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("resume=true is unsupported"));
    }

    #[test]
    fn render_prompt_replaces_single_and_double_braces() {
        let mut object = Map::new();
        object.insert("word".to_string(), Value::String("kataba".to_string()));
        let rendered = render_prompt("{word} / {{word}}", &object).unwrap();
        assert_eq!(rendered, "kataba / kataba");
        assert!(render_prompt("{missing}", &object).is_err());
        assert!(render_prompt("{word", &object).is_err());
    }

    #[test]
    fn arabic_source_spans_convert_from_utf8_bytes_to_tokenizer_characters() {
        let prompt = "قل كتب الآن";
        let byte_span = unique_substring_byte_span(prompt, "كتب").unwrap();
        assert_eq!(&prompt[byte_span[0]..byte_span[1]], "كتب");
        assert_eq!(
            byte_span_to_character_span(prompt, byte_span).unwrap(),
            [3, 6]
        );
        assert!(unique_substring_byte_span("كتب ثم كتب", "كتب").is_err());
    }

    #[test]
    fn run_directory_transaction_publishes_only_on_commit() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = std::env::temp_dir().join(format!(
            "ember_run_transaction_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&parent).unwrap();

        let abandoned = parent.join("abandoned");
        {
            let transaction = RunDirectoryTransaction::begin(&abandoned).unwrap();
            fs::write(transaction.staging_path().join("partial"), b"partial").unwrap();
            assert!(!abandoned.exists());
        }
        assert!(!abandoned.exists());

        let published = parent.join("published");
        let transaction = RunDirectoryTransaction::begin(&published).unwrap();
        fs::write(transaction.staging_path().join("complete"), b"complete").unwrap();
        transaction.commit().unwrap();
        assert_eq!(fs::read(published.join("complete")).unwrap(), b"complete");
        assert!(RunDirectoryTransaction::begin(&published).is_err());
        fs::remove_dir_all(parent).ok();
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
        let mut config = test_config();
        config.architecture = None;
        config.prompt_template = "x".to_string();
        config.layers.clear();
        let selected =
            select_token_positions(&[1, 2, 3], &[(0, 0), (0, 1), (1, 3)], &config, None).unwrap();
        assert_eq!(selected, vec![2]);
    }
}
