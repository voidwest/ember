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

mod types;
mod validate;
pub use types::*;
pub use validate::validate_artifact_contract;
pub(crate) use validate::validate_token_offsets;

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

/// Bytes read per call while hashing a file. The 8 KiB reads of
/// `std::io::copy` added ~10% to hashing a 1.3 GB model (0.80 s -> 0.72 s on
/// an M1 Pro); past 256 KiB larger reads gain nothing.
const SHA256_FILE_CHUNK: usize = 256 * 1024;

pub fn sha256_file_result(path: impl AsRef<Path>) -> Result<String> {
    sha256_file_result_unless(path, &std::sync::atomic::AtomicBool::new(false))
}

/// As [`sha256_file_result`], but gives up between reads once `abandon` is
/// set, returning an error. For a hash computed alongside other work that
/// can fail first: the caller sets `abandon` and the result goes unused.
pub fn sha256_file_result_unless(
    path: impl AsRef<Path>,
    abandon: &std::sync::atomic::AtomicBool,
) -> Result<String> {
    use std::io::Read;

    let path = path.as_ref();
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open file for SHA-256: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut chunk = vec![0u8; SHA256_FILE_CHUNK];
    loop {
        if abandon.load(std::sync::atomic::Ordering::Relaxed) {
            anyhow::bail!("abandoned hashing {}", path.display());
        }
        let read = match file.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to hash file: {}", path.display()));
            }
        };
        hasher.update(&chunk[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The commit this binary was built from, as captured by `ember-core`'s
/// build script (`EMBER_GIT_COMMIT`), or `None` for a build outside a git
/// checkout.
///
/// Use this for provenance that names what produced an artifact. Unlike
/// [`git_commit`], it does not depend on the directory the binary runs in, so
/// the same binary records the same commit wherever it is invoked.
pub fn build_commit() -> Option<&'static str> {
    crate::build_info::GIT_COMMIT
}

/// `git rev-parse HEAD` in the current working directory.
///
/// This describes the checkout the process runs in, not the binary: an
/// installed `ember` invoked inside another repository reports that
/// repository's HEAD. Only benchmark and support records, which deliberately
/// capture the source tree next to the binary's own [`build_commit`], use it.
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

pub fn token_indices_for_offsets(
    offsets: &[(usize, usize)],
    start: usize,
    end: usize,
) -> Vec<usize> {
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

#[cfg(test)]
mod tests;
