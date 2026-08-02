//! Ember v0.2 activation-artifact schema.
//!
//! **Experimental schema. No compatibility guarantee.** The manifest shape,
//! record naming, and file layout below are versioned for the v0.2 series but
//! may change in any future release. Treat every field as unstable unless a
//! later release explicitly stabilizes it.
//!
//! Artifacts may contain sensitive prompt or activation data. Capture configs
//! can omit prompt text (`omit_prompt_text = true`) while retaining token IDs
//! and hashes; see `docs/activation-artifacts.md`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;

/// Experimental schema version for v0.2 activation artifacts.
pub const ACTIVATION_ARTIFACT_SCHEMA: &str = "0.2.0-experimental";

/// Artifact kind written by the capture facility.
pub const ACTIVATION_ARTIFACT_KIND: &str = "ember-activation-capture";

/// The six semantic hook stages a capture record can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActivationStage {
    BeforeLayer,
    AfterAttention,
    AfterMlp,
    AfterLayer,
    BeforeLogits,
    AfterLogits,
}

impl ActivationStage {
    /// All supported stages, in hook order.
    pub const ALL: [ActivationStage; 6] = [
        ActivationStage::BeforeLayer,
        ActivationStage::AfterAttention,
        ActivationStage::AfterMlp,
        ActivationStage::AfterLayer,
        ActivationStage::BeforeLogits,
        ActivationStage::AfterLogits,
    ];
}

impl fmt::Display for ActivationStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::BeforeLayer => "before-layer",
            Self::AfterAttention => "after-attention",
            Self::AfterMlp => "after-mlp",
            Self::AfterLayer => "after-layer",
            Self::BeforeLogits => "before-logits",
            Self::AfterLogits => "after-logits",
        };
        f.write_str(name)
    }
}

impl FromStr for ActivationStage {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "before-layer" => Ok(Self::BeforeLayer),
            "after-attention" => Ok(Self::AfterAttention),
            "after-mlp" => Ok(Self::AfterMlp),
            "after-layer" => Ok(Self::AfterLayer),
            "before-logits" => Ok(Self::BeforeLogits),
            "after-logits" => Ok(Self::AfterLogits),
            _ => Err(format!(
                "unknown stage '{value}'; expected one of: before-layer, after-attention, \
                 after-mlp, after-layer, before-logits, after-logits"
            )),
        }
    }
}

/// Which execution phase a capture selection or record targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CapturePhase {
    Prefill,
    Decode,
    Both,
}

impl CapturePhase {
    /// Whether records for the given phase are included.
    pub fn includes(self, phase: &str) -> bool {
        match self {
            Self::Prefill => phase == "prefill",
            Self::Decode => phase == "decode",
            Self::Both => phase == "prefill" || phase == "decode",
        }
    }
}

impl fmt::Display for CapturePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prefill => f.write_str("prefill"),
            Self::Decode => f.write_str("decode"),
            Self::Both => f.write_str("both"),
        }
    }
}

impl FromStr for CapturePhase {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            "both" => Ok(Self::Both),
            _ => Err(format!(
                "unknown phase '{value}'; expected prefill, decode, or both"
            )),
        }
    }
}

/// Kernel/dispatch path used for an evaluation. A single run can mix paths
/// (generic prefill, fast/workspace decode), so dispatch is recorded per
/// evaluation and per captured record, plus as run-level observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DispatchPath {
    /// Allocation-free/workspace-backed single-token decode.
    Fast,
    /// Generic tensor path (prefill and ineligible decode).
    Generic,
    /// Unknown or not recorded.
    Unknown,
}

impl fmt::Display for DispatchPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fast => f.write_str("fast"),
            Self::Generic => f.write_str("generic"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

/// Parsed capture selection (TOML, typed; not a generic config language).
///
/// The exact config file hash is preserved in the artifact manifest so a
/// capture can be replayed from identical selection semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureSelection {
    /// Output directory for the artifact (manifest.json + tensors/).
    pub output_dir: PathBuf,
    /// Layer indices to capture (validated against the model at load).
    pub layers: Vec<usize>,
    /// Hook stages to capture.
    pub stages: Vec<ActivationStage>,
    /// Execution phases to capture.
    pub phase: CapturePhase,
    /// Absolute decode positions to capture; empty = all decode steps.
    /// Prefill records are whole-sequence tensors in the v0.2 MVP.
    pub token_positions: Vec<usize>,
    /// Maximum number of records; 0 = unlimited. Truncation is flagged.
    pub max_records: usize,
    /// Maximum buffered tensor payload bytes; 0 = unlimited. This protects
    /// long decode captures from exhausting process memory.
    #[serde(default)]
    pub max_bytes: usize,
    /// Omit prompt text from the manifest (token IDs and hash retained).
    pub omit_prompt_text: bool,
    /// Stable hash (fnv1a64) of the exact config file bytes, when loaded
    /// from a file, so a capture can be replayed from identical selection
    /// semantics.
    #[serde(default)]
    pub config_hash: Option<String>,
}

/// Typed TOML config mirror, with `config_hash` filled at load time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfigFile {
    pub schema_version: u32,
    pub output_dir: String,
    pub layers: Vec<usize>,
    pub stages: Vec<String>,
    #[serde(default = "default_phase")]
    pub phase: String,
    #[serde(default)]
    pub token_positions: Vec<usize>,
    #[serde(default)]
    pub max_records: usize,
    #[serde(default)]
    pub max_bytes: usize,
    #[serde(default)]
    pub omit_prompt_text: bool,
}

fn default_phase() -> String {
    "both".to_string()
}

impl CaptureSelection {
    /// Parse and validate a capture config from raw TOML text.
    pub fn from_toml_str(text: &str) -> Result<Self, String> {
        let config: CaptureConfigFile =
            toml::from_str(text).map_err(|e| format!("invalid capture config: {e}"))?;
        if config.schema_version != 1 {
            return Err(format!(
                "unsupported capture config schema_version {} (expected 1)",
                config.schema_version
            ));
        }
        if config.layers.is_empty() {
            return Err("capture config requires at least one layer".to_string());
        }
        if config.output_dir.trim().is_empty() {
            return Err("capture config requires output_dir".to_string());
        }
        let stages = config
            .stages
            .iter()
            .map(|s| s.parse::<ActivationStage>())
            .collect::<Result<Vec<_>, _>>()?;
        if stages.is_empty() {
            return Err("capture config requires at least one stage".to_string());
        }
        reject_duplicates(&config.layers, "capture layers")?;
        reject_duplicates(&stages, "capture stages")?;
        reject_duplicates(&config.token_positions, "capture token positions")?;
        let phase = config.phase.parse::<CapturePhase>()?;
        let config_hash = Some(crate::extraction::stable_bytes_hash(text.as_bytes()));
        Ok(Self {
            output_dir: PathBuf::from(config.output_dir),
            layers: config.layers,
            stages,
            phase,
            token_positions: config.token_positions,
            max_records: config.max_records,
            max_bytes: config.max_bytes,
            omit_prompt_text: config.omit_prompt_text,
            config_hash,
        })
    }

    /// Load a capture config from a TOML file path.
    pub fn from_toml_path(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read capture config '{path}': {e}"))?;
        Self::from_toml_str(&text).map_err(|e| format!("{e} (in '{path}')"))
    }

    /// Whether a hook at the given stage/layer/phase/position is selected.
    pub fn selects(
        &self,
        stage: ActivationStage,
        layer: usize,
        phase: &str,
        token_position: Option<usize>,
    ) -> bool {
        if !self.stages.contains(&stage) {
            return false;
        }
        // before-logits / after-logits are not per-layer: the layers filter
        // does not apply, and records for these stages always carry layer 0.
        let is_logits_stage = matches!(
            stage,
            ActivationStage::BeforeLogits | ActivationStage::AfterLogits
        );
        if !is_logits_stage && !self.layers.contains(&layer) {
            return false;
        }
        if !self.phase.includes(phase) {
            return false;
        }
        if !self.token_positions.is_empty() {
            match token_position {
                // Prefill records are whole-sequence in the MVP and are not
                // position-filtered; decode steps filter by absolute position.
                Some(position) if phase == "decode" => {
                    return self.token_positions.contains(&position);
                }
                _ => {}
            }
        }
        true
    }
}

/// One captured tensor record (metadata; values live in the tensor file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureRecord {
    pub index: usize,
    pub phase: String,
    pub layer: usize,
    pub stage: ActivationStage,
    pub start_position: usize,
    pub token_count: usize,
    pub shape: [usize; 2],
    pub dtype: String,
    pub byte_order: String,
    pub path: String,
    pub sha256: String,
    pub l2_norm: f64,
    pub abs_max: f32,
    pub dispatch: DispatchPath,
}

impl CaptureRecord {
    /// Semantic identity key shared with the compare and patch tooling.
    /// Phase is compared as a string so decode/prefill distinction is exact.
    pub fn alignment_key(&self) -> (String, usize, ActivationStage, usize) {
        (
            self.phase.clone(),
            self.layer,
            self.stage,
            self.start_position,
        )
    }

    /// Deterministic ordering key: prefill before decode, then layer, stage,
    /// and start position. Used to stabilize manifest and compare output
    /// ordering regardless of record insertion order.
    pub fn sort_key(&self) -> (usize, usize, ActivationStage, usize) {
        let phase_rank = if self.phase == "prefill" { 0 } else { 1 };
        (phase_rank, self.layer, self.stage, self.start_position)
    }
}

/// Model provenance section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestModel {
    pub family: String,
    pub identifier: Option<String>,
    pub architecture: String,
    pub layer_count: usize,
    pub hidden_size: usize,
    pub sha256: Option<String>,
    pub file_size_bytes: Option<u64>,
    pub tokenizer_sha256: Option<String>,
    /// Subset of GGUF metadata: architecture + quantization fields.
    pub gguf: serde_json::Value,
}

/// Run provenance section (prompt/token/dispatch).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestRun {
    /// Prompt text, or `null` when `omit_prompt_text` was set.
    pub prompt: Option<String>,
    pub prompt_hash: String,
    pub input_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub thread_count: usize,
    pub tracing: String,
    pub cpu: serde_json::Value,
    pub dispatch_observations: Vec<DispatchObservation>,
}

/// One (phase, dispatch path) observation; a run can mix paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchObservation {
    pub phase: String,
    pub dispatch: DispatchPath,
}

/// Active experiment provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestExperiment {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// The v0.2 activation artifact manifest.
///
/// Deterministic except for `created_at_unix`, which the compare tooling
/// explicitly ignores. Everything else compares exactly or is reported.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationManifest {
    pub schema_version: String,
    pub artifact_kind: String,
    pub ember_version: String,
    pub git_commit: Option<String>,
    pub model: ManifestModel,
    pub run: ManifestRun,
    pub experiment: ManifestExperiment,
    pub capture_selection: CaptureSelection,
    pub records: Vec<CaptureRecord>,
    /// True when `max_records` stopped capture early.
    pub truncated: bool,
    /// Provenance only; explicitly ignored by deterministic comparison.
    pub created_at_unix: u64,
}

impl ActivationManifest {
    /// Directory containing `manifest.json`.
    pub fn base_dir(&self) -> &std::path::Path {
        &self.capture_selection.output_dir
    }
}

/// Deterministic tensor file name for one record.
pub fn record_file_name(
    phase: &str,
    layer: usize,
    stage: ActivationStage,
    start_position: usize,
) -> String {
    format!("{phase}_layer{layer:03}_{stage}_pos{start_position:06}.npy")
}

/// Resolve a source record by (layer, stage, phase, optional position).
///
/// Selection must be unambiguous: with a position, exactly one record must
/// match; without one, exactly one record must match the (layer, stage,
/// phase) triple. Any other outcome is an error — the first match is never
/// chosen implicitly.
pub fn resolve_unique_record<'a>(
    records: &'a [CaptureRecord],
    layer: usize,
    stage: ActivationStage,
    phase: &str,
    position: Option<usize>,
) -> Result<&'a CaptureRecord, String> {
    let matches: Vec<&CaptureRecord> = records
        .iter()
        .filter(|record| {
            record.layer == layer
                && record.stage == stage
                && record.phase == phase
                && position.is_none_or(|position| record.start_position == position)
        })
        .collect();
    match matches.len() {
        0 => {
            let qualifier = position
                .map(|p| format!(" at position {p}"))
                .unwrap_or_default();
            Err(format!(
                "no captured record matches layer {layer} stage {stage} phase {phase}{qualifier} \
                 (artifact has {} record(s))",
                records.len()
            ))
        }
        1 => Ok(matches[0]),
        count => {
            let candidates: Vec<String> = matches
                .iter()
                .map(|record| {
                    format!(
                        "  index {}: {} layer {} {} pos {}",
                        record.index,
                        record.phase,
                        record.layer,
                        record.stage,
                        record.start_position
                    )
                })
                .collect();
            Err(format!(
                "ambiguous patch source: {count} records match layer {layer} stage {stage} \
                 phase {phase}; specify a position or narrow the selection:\n{}",
                candidates.join("\n")
            ))
        }
    }
}

/// Load and structurally validate a v0.2 manifest from `manifest.json`.
pub fn load_manifest(path: &str) -> Result<ActivationManifest, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read manifest '{path}': {e}"))?;
    let manifest: ActivationManifest = serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse manifest '{path}': {e}"))?;
    if manifest.schema_version != ACTIVATION_ARTIFACT_SCHEMA {
        return Err(format!(
            "unsupported artifact schema '{}' (expected '{}')",
            manifest.schema_version, ACTIVATION_ARTIFACT_SCHEMA
        ));
    }
    validate_manifest(path, &manifest)?;
    Ok(manifest)
}

fn reject_duplicates<T>(values: &[T], label: &str) -> Result<(), String>
where
    T: Ord + Clone + fmt::Debug,
{
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value.clone()) {
            return Err(format!("{label} contains duplicate value {value:?}"));
        }
    }
    Ok(())
}

/// Resolve a record path relative to its manifest and reject absolute paths,
/// parent traversal, and symlinks escaping the artifact directory.
pub fn resolve_record_path(manifest_path: &str, record_path: &str) -> Result<PathBuf, String> {
    let relative = Path::new(record_path);
    if relative.as_os_str().is_empty() || relative.is_absolute() {
        return Err(format!(
            "artifact record path must be non-empty and relative: '{record_path}'"
        ));
    }
    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "artifact record path is not normalized: '{record_path}'"
        ));
    }
    let manifest = Path::new(manifest_path);
    let base = manifest.parent().unwrap_or_else(|| Path::new("."));
    let canonical_base = base.canonicalize().map_err(|e| {
        format!(
            "failed to resolve artifact directory '{}': {e}",
            base.display()
        )
    })?;
    let joined = base.join(relative);
    let canonical = joined.canonicalize().map_err(|e| {
        format!(
            "failed to resolve artifact record '{}': {e}",
            joined.display()
        )
    })?;
    if !canonical.starts_with(&canonical_base) {
        return Err(format!(
            "artifact record '{}' resolves outside artifact directory '{}'",
            record_path,
            canonical_base.display()
        ));
    }
    Ok(canonical)
}

/// Validate manifest structure and every declared tensor payload.
pub fn validate_manifest(path: &str, manifest: &ActivationManifest) -> Result<(), String> {
    if manifest.schema_version != ACTIVATION_ARTIFACT_SCHEMA {
        return Err(format!(
            "unsupported artifact schema '{}' (expected '{}')",
            manifest.schema_version, ACTIVATION_ARTIFACT_SCHEMA
        ));
    }
    if manifest.artifact_kind != ACTIVATION_ARTIFACT_KIND {
        return Err(format!(
            "unsupported artifact kind '{}' (expected '{}')",
            manifest.artifact_kind, ACTIVATION_ARTIFACT_KIND
        ));
    }
    if manifest.ember_version.trim().is_empty() {
        return Err("artifact ember_version must not be empty".to_string());
    }
    if manifest.model.family.trim().is_empty() || manifest.model.architecture.trim().is_empty() {
        return Err("artifact model family and architecture must not be empty".to_string());
    }
    if manifest.model.layer_count == 0 || manifest.model.hidden_size == 0 {
        return Err("artifact model dimensions must be non-zero".to_string());
    }
    for (name, hash) in [
        ("model", manifest.model.sha256.as_deref()),
        ("tokenizer", manifest.model.tokenizer_sha256.as_deref()),
    ] {
        if let Some(hash) = hash {
            if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(format!("artifact {name} SHA-256 is invalid"));
            }
        }
    }
    if manifest.run.prompt_hash.trim().is_empty() {
        return Err("artifact prompt_hash must not be empty".to_string());
    }
    if manifest.run.input_token_ids.is_empty() {
        return Err("artifact input_token_ids must not be empty".to_string());
    }
    if manifest.run.thread_count == 0 {
        return Err("artifact thread_count must be non-zero".to_string());
    }
    if let Some(prompt) = &manifest.run.prompt {
        let expected = crate::extraction::stable_prompt_hash(prompt);
        if manifest.run.prompt_hash != expected {
            return Err(format!(
                "artifact prompt_hash does not match stored prompt: expected {expected}"
            ));
        }
    }
    if !matches!(manifest.run.tracing.as_str(), "enabled" | "disabled") {
        return Err(format!("invalid tracing state '{}'", manifest.run.tracing));
    }
    for observation in &manifest.run.dispatch_observations {
        if !matches!(observation.phase.as_str(), "prefill" | "decode") {
            return Err(format!(
                "dispatch observation has invalid phase '{}'",
                observation.phase
            ));
        }
    }
    if manifest.experiment.name.trim().is_empty() {
        return Err("artifact experiment name must not be empty".to_string());
    }
    if manifest.capture_selection.output_dir.as_os_str().is_empty()
        || manifest.capture_selection.layers.is_empty()
        || manifest.capture_selection.stages.is_empty()
    {
        return Err(
            "artifact capture selection requires output_dir, layers, and stages".to_string(),
        );
    }
    reject_duplicates(&manifest.capture_selection.layers, "capture layers")?;
    reject_duplicates(&manifest.capture_selection.stages, "capture stages")?;
    reject_duplicates(
        &manifest.capture_selection.token_positions,
        "capture token positions",
    )?;
    if manifest.records.is_empty() {
        return Err("activation artifact contains no tensor records".to_string());
    }

    let mut indices = BTreeSet::new();
    let mut keys = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut previous_sort_key = None;
    for record in &manifest.records {
        if !indices.insert(record.index) {
            return Err(format!("duplicate capture record index {}", record.index));
        }
        if !keys.insert(record.alignment_key()) {
            return Err(format!(
                "duplicate capture record key: {} layer {} {} position {}",
                record.phase, record.layer, record.stage, record.start_position
            ));
        }
        if !paths.insert(record.path.clone()) {
            return Err(format!("duplicate capture record path '{}'", record.path));
        }
        let sort_key = record.sort_key();
        if previous_sort_key.is_some_and(|previous| previous > sort_key) {
            return Err("capture records are not in deterministic sort order".to_string());
        }
        previous_sort_key = Some(sort_key);
        if !matches!(record.phase.as_str(), "prefill" | "decode") {
            return Err(format!(
                "record {} has invalid phase '{}'",
                record.index, record.phase
            ));
        }
        let is_logits = matches!(
            record.stage,
            ActivationStage::BeforeLogits | ActivationStage::AfterLogits
        );
        if (!is_logits && record.layer >= manifest.model.layer_count)
            || (is_logits && record.layer != 0)
        {
            return Err(format!(
                "record {} layer {} is invalid for stage {} and {} model layers",
                record.index, record.layer, record.stage, manifest.model.layer_count
            ));
        }
        if record.dtype != "f32" || record.byte_order != "little-endian" {
            return Err(format!(
                "record {} must be little-endian f32, got {} {}",
                record.index, record.byte_order, record.dtype
            ));
        }
        if record.shape.contains(&0) || record.token_count == 0 {
            return Err(format!(
                "record {} has an empty tensor shape/count",
                record.index
            ));
        }
        if record.shape[0] != record.token_count {
            return Err(format!(
                "record {} tensor row count {} does not match token_count {}",
                record.index, record.shape[0], record.token_count
            ));
        }
        if (record.phase == "prefill" && record.start_position != 0)
            || (record.phase == "decode" && record.token_count != 1)
        {
            return Err(format!(
                "record {} has invalid {} position/count semantics",
                record.index, record.phase
            ));
        }
        let token_position = (record.phase == "decode").then_some(record.start_position);
        if !manifest.capture_selection.selects(
            record.stage,
            record.layer,
            &record.phase,
            token_position,
        ) {
            return Err(format!(
                "record {} is outside the declared capture selection",
                record.index
            ));
        }
        if record.stage != ActivationStage::AfterLogits
            && record.shape[1] != manifest.model.hidden_size
        {
            return Err(format!(
                "record {} width {} does not match model hidden size {}",
                record.index, record.shape[1], manifest.model.hidden_size
            ));
        }
        if record.sha256.len() != 64 || !record.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(format!("record {} has an invalid SHA-256", record.index));
        }

        let tensor_path = resolve_record_path(path, &record.path)?;
        let actual_sha = crate::extraction::sha256_file_result(&tensor_path)
            .map_err(|e| format!("failed to hash record {}: {e}", record.index))?;
        if !actual_sha.eq_ignore_ascii_case(&record.sha256) {
            return Err(format!(
                "record {} SHA-256 mismatch: manifest {}, actual {}",
                record.index, record.sha256, actual_sha
            ));
        }
        let tensor_path_str = tensor_path
            .to_str()
            .ok_or_else(|| format!("record {} path is not valid UTF-8", record.index))?;
        let (shape, values) = crate::npy::read_npy_2d(tensor_path_str)
            .map_err(|e| format!("record {} tensor is invalid: {e}", record.index))?;
        if shape.as_slice() != record.shape {
            return Err(format!(
                "record {} tensor shape {:?} disagrees with manifest {:?}",
                record.index, shape, record.shape
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "record {} tensor contains non-finite values",
                record.index
            ));
        }
        let mut sum_sq = 0.0f64;
        let mut abs_max = 0.0f32;
        for value in &values {
            sum_sq += f64::from(*value) * f64::from(*value);
            abs_max = abs_max.max(value.abs());
        }
        let l2_norm = sum_sq.sqrt();
        let l2_tolerance = 8.0 * f64::EPSILON * l2_norm.abs().max(1.0);
        if !record.l2_norm.is_finite()
            || !record.abs_max.is_finite()
            || (record.l2_norm - l2_norm).abs() > l2_tolerance
            || record.abs_max.to_bits() != abs_max.to_bits()
        {
            return Err(format!(
                "record {} tensor statistics disagree with its payload: l2 manifest={} actual={}, abs_max manifest={} actual={}",
                record.index, record.l2_norm, l2_norm, record.abs_max, abs_max
            ));
        }
    }
    Ok(())
}
