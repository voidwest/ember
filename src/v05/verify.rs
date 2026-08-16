//! v0.5 offline bundle verification (contract sections 8, 16).
//!
//! Basic verification requires no internet and no model file. Deep
//! verification additionally checks the model/tokenizer files and
//! execution-plan compatibility against the loaded model.

use crate::v05::hook::SemanticHookSite;
use crate::v05::manifest::{
    sha256_hex, BundleIdentity, BundleManifest, SemanticManifest, BUNDLE_KIND, BUNDLE_SCHEMA_V1,
};
use crate::v05::safetensors::{self, TensorView};
use crate::v05::token_select::{CoverageKind, TokenSelectionRecord};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// One indexed capture tensor (captures/index.jsonl line).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureIndexEntry {
    pub capture_id: String,
    pub input_id: String,
    pub site: SemanticHookSite,
    pub layer: usize,
    pub positions: Vec<usize>,
    /// Payload tensor name; empty for summary-only entries.
    #[serde(default)]
    pub tensor_name: String,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default)]
    pub dtype: String,
    #[serde(default)]
    pub byte_length: usize,
    #[serde(default)]
    pub checksum: String,
    #[serde(default)]
    pub model_sha256: String,
    #[serde(default)]
    pub plan_hash: String,
    #[serde(default)]
    pub hook_route: String,
    #[serde(default)]
    pub fusion: String,
    #[serde(default)]
    pub selection_provenance: serde_json::Value,
    /// Deterministic summary statistics for summary-only captures.
    #[serde(default)]
    pub summary: Option<SummaryEntry>,
}

/// Deterministic summary statistics recorded for a summary-only capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryEntry {
    pub shape: Vec<usize>,
    pub finite_count: usize,
    pub minimum: f32,
    pub maximum: f32,
    pub mean: f64,
    pub l2_norm: f64,
}

/// One verification check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

/// The verification report (also written to `verification.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub bundle_schema: String,
    pub ok: bool,
    pub semantic_hash: String,
    pub payload_hash: String,
    pub checks: Vec<CheckResult>,
    pub warnings: Vec<String>,
    /// Verification timestamp (runtime metadata; excluded from hashes).
    pub timestamp: String,
}

impl VerificationReport {
    fn new(semantic_hash: String, payload_hash: String) -> VerificationReport {
        VerificationReport {
            bundle_schema: BUNDLE_SCHEMA_V1.to_string(),
            ok: true,
            semantic_hash,
            payload_hash,
            checks: Vec::new(),
            warnings: Vec::new(),
            timestamp: String::new(),
        }
    }

    fn record(&mut self, name: &str, ok: bool, detail: String) {
        if !ok {
            self.ok = false;
        }
        self.checks.push(CheckResult {
            name: name.to_string(),
            ok,
            detail,
        });
    }
}

/// A verified bundle loaded for cross-bundle sources.
pub struct LoadedBundle {
    pub root: PathBuf,
    pub semantic_manifest: SemanticManifest,
    pub capture_index: Vec<CaptureIndexEntry>,
    payload_bytes: Vec<u8>,
    tensors: Vec<(String, TensorView)>,
}

impl LoadedBundle {
    /// Load one indexed tensor as f32.
    pub fn tensor_f32_by_name(&self, name: &str) -> Result<Vec<f32>, String> {
        let (_, view) = self
            .tensors
            .iter()
            .find(|(tensor_name, _)| tensor_name == name)
            .ok_or_else(|| format!("tensor '{name}' not found in the payload"))?;
        safetensors::tensor_f32(&self.payload_bytes, view)
    }
}

/// Verify a bundle and load its payloads for source use.
///
/// The source bundle must pass full basic verification before its tensors
/// can back an intervention.
pub fn load_bundle_for_source(root: &Path) -> Result<LoadedBundle, String> {
    let report = verify_bundle(root, &VerifyOptions::default())?;
    if !report.ok {
        let errors: Vec<String> = report
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| format!("{}: {}", check.name, check.detail))
            .collect();
        return Err(format!(
            "source bundle '{}' failed verification: {}",
            root.display(),
            errors.join("; ")
        ));
    }
    let semantic_manifest = read_semantic_manifest(root)?;
    let capture_index = read_capture_index(root)?;
    let payload_path = root.join("captures/tensors.safetensors");
    let payload_bytes = std::fs::read(&payload_path)
        .map_err(|error| format!("cannot read '{}': {error}", payload_path.display()))?;
    let tensors = safetensors::deserialize(&payload_bytes)?;
    Ok(LoadedBundle {
        root: root.to_path_buf(),
        semantic_manifest,
        capture_index,
        payload_bytes,
        tensors,
    })
}

/// Options for bundle verification.
#[derive(Debug, Clone, Default)]
pub struct VerifyOptions {
    /// Deep verification: check model/tokenizer files when supplied.
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
}

/// Verify a bundle fully offline (unless deep verification is requested).
pub fn verify_bundle(root: &Path, options: &VerifyOptions) -> Result<VerificationReport, String> {
    // ---- phase 1: manifest load + schema/basic identity checks ----
    let manifest_path = root.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)
        .map_err(|error| format!("cannot read '{}': {error}", manifest_path.display()))?;
    let manifest: BundleManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("manifest.json is not valid JSON: {error}"))?;
    let mut report = VerificationReport::new(String::new(), String::new());

    report.record(
        "bundle schema",
        manifest.bundle_schema == BUNDLE_SCHEMA_V1,
        format!("schema '{}'", manifest.bundle_schema),
    );
    report.record(
        "bundle kind",
        manifest.kind == BUNDLE_KIND,
        format!("kind '{}'", manifest.kind),
    );
    report.record(
        "bundle complete",
        manifest.status == "complete",
        format!("status '{}'", manifest.status),
    );

    // ---- phase 2: required files + checksum scan ----
    // required files exist
    let required = [
        "semantic-manifest.json",
        "runtime.json",
        "experiment.toml",
        "resolved-experiment.json",
        "model.json",
        "tokenizer.json",
        "execution-plan.json",
        "inputs.jsonl",
        "outputs.jsonl",
        "tokenization.jsonl",
        "captures/tensors.safetensors",
        "captures/index.jsonl",
        "interventions/events.jsonl",
        "traces/events.jsonl",
        "checksums.sha256",
    ];
    let mut missing: Vec<String> = Vec::new();
    for relative in required {
        if !root.join(relative).is_file() {
            missing.push(relative.to_string());
        }
    }
    report.record(
        "required files",
        missing.is_empty(),
        if missing.is_empty() {
            "all present".into()
        } else {
            format!("missing: {}", missing.join(", "))
        },
    );
    if !missing.is_empty() {
        report.ok = false;
        report.timestamp = now_iso8601();
        let _ = write_verification_json(root, &report);
        return Ok(report);
    }

    let semantic_manifest = read_semantic_manifest(root)?;
    report.record(
        "semantic manifest schema",
        semantic_manifest.bundle_schema == BUNDLE_SCHEMA_V1,
        format!("schema '{}'", semantic_manifest.bundle_schema),
    );
    report.record(
        "semantic manifest complete",
        semantic_manifest.complete,
        String::new(),
    );

    // checksums: every file covered by checksums.sha256 must match. Keys
    // come from the untrusted bundle and are path-validated before use:
    // an absolute or `..`-bearing key must fail verification, never escape
    // the bundle root (traversal → arbitrary-file hash oracle / read).
    let checksums = read_checksums(root)?;
    let mut checksum_mismatches: Vec<String> = Vec::new();
    for (relative, expected) in &checksums {
        let Ok(relative) = crate::v05::bundle::validate_relative_path(relative) else {
            checksum_mismatches.push(format!("{relative}: unsafe path in checksums"));
            continue;
        };
        let path = root.join(relative);
        if !path.is_file() {
            checksum_mismatches.push(format!("{relative}: missing file"));
            continue;
        }
        // stream-hash so a large file inside the bundle cannot force a
        // full-buffer read during verification
        let actual = crate::extraction::sha256_file_result(&path)
            .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
        if actual != *expected {
            checksum_mismatches.push(format!("{relative}: checksum mismatch"));
        }
    }
    // verification.json is runtime state; it is not covered by the
    // publish-time checksum file.
    report.record(
        "checksums",
        checksum_mismatches.is_empty(),
        if checksum_mismatches.is_empty() {
            format!("{} files verified", checksums.len())
        } else {
            checksum_mismatches.join("; ")
        },
    );

    // capture index consistency
    let capture_index = read_capture_index(root)?;
    let mut index_errors: Vec<String> = Vec::new();
    let mut seen_ids: BTreeMap<(String, String, SemanticHookSite, usize), usize> = BTreeMap::new();
    for entry in &capture_index {
        *seen_ids
            .entry((
                entry.capture_id.clone(),
                entry.input_id.clone(),
                entry.site,
                entry.layer,
            ))
            .or_insert(0) += 1;
        if entry.summary.is_some() {
            // Summary-only entries carry no payload.
            continue;
        }
        if entry.tensor_name.is_empty() {
            index_errors.push(format!("'{}': empty tensor name", entry.capture_id));
        }
        let expected_bytes: usize = entry
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .and_then(|count| count.checked_mul(dtype_bytes(&entry.dtype)?))
            .unwrap_or(0);
        if expected_bytes != entry.byte_length {
            index_errors.push(format!(
                "'{}': shape {:?} {} implies {expected_bytes} bytes but index says {}",
                entry.capture_id, entry.shape, entry.dtype, entry.byte_length
            ));
        }
    }
    let duplicate_ids: Vec<String> = seen_ids
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|((capture_id, input_id, site, layer), count)| {
            format!("{capture_id}/{input_id}/{site}/layer-{layer} x{count}")
        })
        .collect();
    if !duplicate_ids.is_empty() {
        index_errors.push(format!(
            "duplicate capture ids in index: {}",
            duplicate_ids.join(", ")
        ));
    }
    report.record(
        "capture index",
        index_errors.is_empty(),
        if index_errors.is_empty() {
            format!("{} entries", capture_index.len())
        } else {
            index_errors.join("; ")
        },
    );

    // payload: shapes/dtypes match the index; no unindexed tensors
    let payload_path = root.join("captures/tensors.safetensors");
    let payload_bytes = std::fs::read(&payload_path)
        .map_err(|error| format!("cannot read '{}': {error}", payload_path.display()))?;
    let payload_tensors = safetensors::deserialize(&payload_bytes)?;
    let mut payload_errors: Vec<String> = Vec::new();
    let indexed_names: BTreeMap<&str, &CaptureIndexEntry> = capture_index
        .iter()
        .map(|entry| (entry.tensor_name.as_str(), entry))
        .collect();
    for (name, view) in &payload_tensors {
        let Some(entry) = indexed_names.get(name.as_str()) else {
            payload_errors.push(format!("unindexed tensor '{name}' in payload"));
            continue;
        };
        if view.shape != entry.shape {
            payload_errors.push(format!(
                "'{name}': payload shape {:?} != index shape {:?}",
                view.shape, entry.shape
            ));
        }
        if view.dtype.name() != entry.dtype {
            payload_errors.push(format!(
                "'{name}': payload dtype {} != index dtype {}",
                view.dtype.name(),
                entry.dtype
            ));
        }
        let raw = &payload_bytes[view.data_offsets.0..view.data_offsets.1];
        if sha256_hex(raw) != entry.checksum {
            payload_errors.push(format!("'{name}': tensor checksum mismatch"));
        }
    }
    for entry in &capture_index {
        if entry.summary.is_some() {
            continue;
        }
        if !payload_tensors
            .iter()
            .any(|(name, _)| name == &entry.tensor_name)
        {
            payload_errors.push(format!(
                "indexed tensor '{}' missing from payload",
                entry.tensor_name
            ));
        }
    }
    report.record(
        "tensor payload",
        payload_errors.is_empty(),
        if payload_errors.is_empty() {
            format!("{} tensors verified", payload_tensors.len())
        } else {
            payload_errors.join("; ")
        },
    );

    // token-selection records internally consistent
    let mut selection_errors: Vec<String> = Vec::new();
    for (index, record) in semantic_manifest.token_selections.iter().enumerate() {
        if let Some(error) = selection_consistency(record) {
            selection_errors.push(format!("record {index}: {error}"));
        }
    }
    report.record(
        "token selection records",
        selection_errors.is_empty(),
        if selection_errors.is_empty() {
            format!("{} records", semantic_manifest.token_selections.len())
        } else {
            selection_errors.join("; ")
        },
    );

    // intervention references resolve
    let mut intervention_errors: Vec<String> = Vec::new();
    for (index, intervention) in semantic_manifest.interventions.iter().enumerate() {
        if let Some(crate::v05::intervention::InterventionSource::CaptureFromCurrentRun {
            capture_id,
        }) = &intervention.source
            && !semantic_manifest
                .captures
                .iter()
                .any(|capture| capture.id == *capture_id)
        {
            intervention_errors.push(format!(
                "intervention {index}: source capture '{capture_id}' not declared"
            ));
        }
    }
    report.record(
        "intervention references",
        intervention_errors.is_empty(),
        if intervention_errors.is_empty() {
            format!("{} interventions", semantic_manifest.interventions.len())
        } else {
            intervention_errors.join("; ")
        },
    );

    // execution-plan hash matches the stored plan
    let plan_path = root.join("execution-plan.json");
    let plan_bytes = std::fs::read(&plan_path)
        .map_err(|error| format!("cannot read '{}': {error}", plan_path.display()))?;
    let plan: crate::plan::ExecutionPlan = serde_json::from_slice(&plan_bytes)
        .map_err(|error| format!("execution-plan.json is not valid JSON: {error}"))?;
    let recomputed_plan_hash = crate::plan::plan_hash(&plan);
    let plan_matches = recomputed_plan_hash == plan.plan_hash;
    report.record(
        "execution plan hash",
        plan_matches,
        if plan_matches {
            format!(
                "plan hash {}",
                &plan.plan_hash[..12.min(plan.plan_hash.len())]
            )
        } else {
            format!(
                "stored {} != recomputed {}",
                plan.plan_hash, recomputed_plan_hash
            )
        },
    );

    // semantic hash + payload hash recompute
    let semantic_hash = BundleIdentity::semantic_hash(&semantic_manifest)?;
    let stored_semantic = manifest.semantic_hash.clone();
    report.record(
        "semantic hash",
        semantic_hash == stored_semantic,
        format!(
            "recomputed {} vs stored {}",
            &semantic_hash[..12],
            &stored_semantic[..12.min(stored_semantic.len())]
        ),
    );
    // The payload inventory is the manifest's payloads map plus the
    // semantic manifest's own file (which cannot list itself).
    let mut inventory = semantic_manifest.payloads.clone();
    let semantic_file = std::fs::read(root.join("semantic-manifest.json"))
        .map_err(|error| format!("cannot read semantic-manifest.json: {error}"))?;
    inventory.insert(
        "semantic-manifest.json".to_string(),
        sha256_hex(&semantic_file),
    );
    let payload_hash = BundleIdentity::payload_hash(&inventory)?;
    let stored_payload = manifest.payload_hash.clone();
    report.record(
        "payload hash",
        payload_hash == stored_payload,
        format!(
            "recomputed {} vs stored {}",
            &payload_hash[..12],
            &stored_payload[..12.min(stored_payload.len())]
        ),
    );

    // payload checksums in the semantic manifest must match the files.
    // Same path-validation rule as the checksums pass: untrusted keys can
    // never escape the bundle root.
    let mut payload_errors: Vec<String> = Vec::new();
    for (relative, expected) in &semantic_manifest.payloads {
        let Ok(relative) = crate::v05::bundle::validate_relative_path(relative) else {
            payload_errors.push(format!("{relative}: unsafe path in semantic manifest"));
            continue;
        };
        let path = root.join(relative);
        if !path.is_file() {
            payload_errors.push(format!("{relative}: missing"));
            continue;
        }
        let actual = crate::extraction::sha256_file_result(&path)
            .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
        if actual != *expected {
            payload_errors.push(format!("{relative}: checksum mismatch"));
        }
    }
    report.record(
        "semantic payload checksums",
        payload_errors.is_empty(),
        if payload_errors.is_empty() {
            format!("{} files", semantic_manifest.payloads.len())
        } else {
            payload_errors.join("; ")
        },
    );

    report.semantic_hash = semantic_hash;
    report.payload_hash = payload_hash;

    // ---- phase 3: deep verification (model/tokenizer/plan) ----
    // deep verification
    if let Some(model_path) = &options.model_path {
        deep_model_check(model_path, &semantic_manifest, &mut report);
    }
    if let Some(tokenizer_path) = &options.tokenizer_path {
        let actual = sha256_hex(&std::fs::read(tokenizer_path).map_err(|error| {
            format!(
                "cannot read tokenizer '{}': {error}",
                tokenizer_path.display()
            )
        })?);
        let ok = actual == semantic_manifest.tokenizer.sha256;
        report.record(
            "deep tokenizer sha256",
            ok,
            format!(
                "file {} vs manifest {}",
                &actual[..12],
                &semantic_manifest.tokenizer.sha256[..12]
            ),
        );
    }

    report.timestamp = now_iso8601();
    let _ = write_verification_json(root, &report);
    Ok(report)
}

fn deep_model_check(
    model_path: &Path,
    semantic_manifest: &SemanticManifest,
    report: &mut VerificationReport,
) {
    match crate::extraction::sha256_file_result(model_path) {
        Ok(actual) => {
            let ok = actual == semantic_manifest.model.sha256;
            report.record(
                "deep model sha256",
                ok,
                format!(
                    "file {} vs manifest {}",
                    &actual[..12.min(actual.len())],
                    &semantic_manifest.model.sha256[..12.min(semantic_manifest.model.sha256.len())]
                ),
            );
        }
        Err(error) => {
            report.record(
                "deep model sha256",
                false,
                format!("cannot hash model file: {error}"),
            );
        }
    }
    match read_gguf_summary(model_path) {
        Ok((arch, block_count)) => {
            let arch_ok = arch == semantic_manifest.model.architecture;
            report.record(
                "deep model architecture",
                arch_ok,
                format!(
                    "file '{arch}' vs manifest '{}'",
                    semantic_manifest.model.architecture
                ),
            );
            let layers_ok = block_count == semantic_manifest.model.layer_count;
            report.record(
                "deep model layer count",
                layers_ok,
                format!(
                    "file {block_count} vs manifest {}",
                    semantic_manifest.model.layer_count
                ),
            );
        }
        Err(error) => {
            report.record(
                "deep model metadata",
                false,
                format!("cannot read model metadata: {error}"),
            );
        }
    }
}

/// Minimal GGUF header reader: extracts `general.architecture` and the
/// `*.block_count` metadata key without materializing tensor data.
fn read_gguf_summary(path: &Path) -> Result<(String, usize), String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot open '{}': {error}", path.display()))?;
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let mut magic = [0u8; 4];
    cursor
        .read_exact(&mut magic)
        .map_err(|error| format!("truncated GGUF header: {error}"))?;
    if &magic != b"GGUF" {
        return Err("not a GGUF file (bad magic)".into());
    }
    cursor
        .seek(SeekFrom::Start(8))
        .map_err(|error| format!("seek failed: {error}"))?; // skip version
    let mut count_buf = [0u8; 8];
    cursor
        .read_exact(&mut count_buf)
        .map_err(|error| format!("truncated GGUF header: {error}"))?;
    let _tensor_count = u64::from_le_bytes(count_buf);
    cursor
        .read_exact(&mut count_buf)
        .map_err(|error| format!("truncated GGUF header: {error}"))?;
    let kv_count = u64::from_le_bytes(count_buf);
    let mut architecture: Option<String> = None;
    let mut block_count: Option<usize> = None;
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut cursor)?;
        let value_type = read_u32(&mut cursor)?;
        match value_type {
            4 => {
                // u32
                let value = read_u32(&mut cursor)? as usize;
                if key.ends_with(".block_count") {
                    block_count = Some(value);
                }
            }
            8 => {
                // string
                let value = read_gguf_string(&mut cursor)?;
                if key == "general.architecture" {
                    architecture = Some(value);
                }
            }
            9 => {
                // array: u32 type + u64 count + elements (skipped by
                // element size)
                let element_type = read_u32(&mut cursor)?;
                let element_count = read_u64(&mut cursor)?;
                let element_size = gguf_element_size(element_type)?;
                let skip = element_size
                    .checked_mul(element_count as usize)
                    .ok_or_else(|| "GGUF array size overflow".to_string())?;
                cursor
                    .seek(SeekFrom::Current(skip as i64))
                    .map_err(|error| format!("GGUF array skip failed: {error}"))?;
            }
            10 => {
                // u64
                let value = read_u64(&mut cursor)?;
                if key.ends_with(".block_count") {
                    block_count = Some(value as usize);
                }
            }
            _ => {
                // skip fixed-size scalar
                let size = gguf_element_size(value_type)?;
                cursor
                    .seek(SeekFrom::Current(size as i64))
                    .map_err(|error| format!("GGUF scalar skip failed: {error}"))?;
            }
        }
    }
    let architecture = architecture.ok_or_else(|| "GGUF lacks general.architecture".to_string())?;
    let block_count = block_count.ok_or_else(|| "GGUF lacks *.block_count".to_string())?;
    Ok((architecture, block_count))
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String, String> {
    let len = read_u64(reader)?;
    // Bound the declared length before allocating: a hostile header can
    // claim u64::MAX and a pre-bound vec! panics with capacity overflow.
    const MAX_GGUF_STRING_BYTES: u64 = 1 << 20;
    if len > MAX_GGUF_STRING_BYTES {
        return Err(format!(
            "GGUF string length {len} exceeds the {MAX_GGUF_STRING_BYTES}-byte limit"
        ));
    }
    let mut bytes = vec![0u8; len as usize];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("truncated GGUF string: {error}"))?;
    String::from_utf8(bytes).map_err(|error| format!("GGUF key is not UTF-8: {error}"))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|error| format!("truncated GGUF value: {error}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|error| format!("truncated GGUF value: {error}"))?;
    Ok(u64::from_le_bytes(buf))
}

fn gguf_element_size(value_type: u32) -> Result<usize, String> {
    match value_type {
        0 | 1 => Ok(1),
        2 | 3 => Ok(2),
        4..=7 => Ok(4),
        8 => Err("string type handled separately".into()),
        9 => Err("array type handled separately".into()),
        10..=12 => Ok(8),
        other => Err(format!("unknown GGUF value type {other}")),
    }
}

fn selection_consistency(record: &TokenSelectionRecord) -> Option<String> {
    let seq_len = record.token_ids.len();
    for &index in &record.selected_indices {
        if index >= seq_len {
            return Some(format!(
                "selected index {index} out of range for {seq_len} tokens"
            ));
        }
    }
    if record.byte_offsets.len() != seq_len {
        return Some("byte offset count does not match token count".into());
    }
    for (index, &(start, end)) in record.byte_offsets.iter().enumerate() {
        if start > end {
            return Some(format!("token {index} has a reversed byte offset"));
        }
    }
    if let Some((start, end)) = record.matched_byte_span {
        if start > end || end > record.normalized_text.len() {
            return Some("matched byte span is out of range".into());
        }
        if record.coverage == CoverageKind::None {
            return Some("coverage is none for a recorded span".into());
        }
    }
    None
}

fn dtype_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "F32" => Some(4),
        "F16" => Some(2),
        _ => None,
    }
}

fn read_semantic_manifest(root: &Path) -> Result<SemanticManifest, String> {
    let path = root.join("semantic-manifest.json");
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("semantic-manifest.json is not valid JSON: {error}"))
}

fn read_capture_index(root: &Path) -> Result<Vec<CaptureIndexEntry>, String> {
    let path = root.join("captures/index.jsonl");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    let mut entries = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let entry: CaptureIndexEntry = serde_json::from_str(line).map_err(|error| {
            format!(
                "captures/index.jsonl line {} is invalid: {error}",
                line_index + 1
            )
        })?;
        entries.push(entry);
    }
    Ok(entries)
}

fn read_checksums(root: &Path) -> Result<BTreeMap<String, String>, String> {
    let path = root.join("checksums.sha256");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    let mut checksums = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (sum, relative) = line
            .split_once("  ")
            .ok_or_else(|| format!("checksums.sha256 has a malformed line: {line:?}"))?;
        if sum.len() != 64 {
            return Err(format!(
                "checksums.sha256 has a malformed checksum on line: {line:?}"
            ));
        }
        checksums.insert(relative.to_string(), sum.to_string());
    }
    Ok(checksums)
}

fn write_verification_json(root: &Path, report: &VerificationReport) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("verification.json serialization failed: {error}"))?;
    crate::atomic_file::atomic_write(root.join("verification.json"), &bytes)
        .map_err(|error| format!("cannot write verification.json: {error}"))
}

fn now_iso8601() -> String {
    // Runtime metadata only; not part of any hash.
    format!("epoch-seconds-{}", crate::extraction::unix_timestamp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v05::testutil;
    use crate::v05::testutil::temp_root;

    /// Write the standard test bundle into a fresh temp dir, run `mutate`,
    /// verify, and require exactly the named checks to fail.
    fn assert_verification_failure(
        tag: &str,
        mutate: impl FnOnce(&std::path::Path),
        expect_failed: &[&str],
    ) {
        let root = temp_root(tag);
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        mutate(&root);
        let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
        assert!(!report.ok);
        let names: Vec<&str> = report
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| check.name.as_str())
            .collect();
        for expected in expect_failed {
            assert!(
                names.contains(expected),
                "missing {expected:?} among {names:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn valid_bundle_verifies() {
        let root = temp_root("valid");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        assert_eq!(report.checks.len(), 15);
        // verification.json is written but excluded from hashes
        assert!(root.join("verification.json").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn one_byte_payload_corruption_fails() {
        assert_verification_failure(
            "corrupt",
            |root| {
                let payload = root.join("captures/tensors.safetensors");
                let mut bytes = std::fs::read(&payload).unwrap();
                let last = bytes.len() - 1;
                bytes[last] ^= 0xFF;
                std::fs::write(&payload, bytes).unwrap();
            },
            &["checksums", "tensor payload"],
        );
    }

    #[test]
    fn removed_file_fails() {
        assert_verification_failure(
            "removed",
            |root| {
                std::fs::remove_file(root.join("captures/index.jsonl")).unwrap();
            },
            &["required files"],
        );
    }

    #[test]
    fn altered_manifest_value_fails() {
        assert_verification_failure(
            "altered",
            |root| {
                let path = root.join("semantic-manifest.json");
                let mut value: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                value["experiment"]["name"] = serde_json::json!("tampered");
                std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
            },
            &["semantic hash"],
        );
    }

    #[test]
    fn extra_unindexed_tensor_fails() {
        assert_verification_failure(
            "extra",
            |root| {
                // Append a second tensor to the payload and fix
                // checksums.sha256 so only the unindexed-tensor check can
                // catch it.
                let payload_path = root.join("captures/tensors.safetensors");
                let original = std::fs::read(&payload_path).unwrap();
                let tensors = crate::v05::safetensors::deserialize(&original).unwrap();
                let extra = crate::v05::safetensors::serialize(&[
                    crate::v05::safetensors::TensorData {
                        name: "cap-1/i1/residual-post-mlp/0",
                        dtype: crate::v05::safetensors::TensorDType::F32,
                        shape: &[1, 4],
                        bytes: &[0u8; 16],
                    },
                    crate::v05::safetensors::TensorData {
                        name: "rogue/i1/residual-post-mlp/0",
                        dtype: crate::v05::safetensors::TensorDType::F32,
                        shape: &[1, 4],
                        bytes: &[0u8; 16],
                    },
                ])
                .unwrap();
                let _ = tensors;
                std::fs::write(&payload_path, extra).unwrap();
                // refresh checksums.sha256 to isolate the payload check
                fix_checksums(root);
            },
            &["tensor payload"],
        );
    }

    #[test]
    fn incomplete_bundle_fails() {
        let root = temp_root("incomplete");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("manifest.json"), b"{}").unwrap();
        // A malformed manifest is a hard error; a missing-file bundle
        // yields a failing report. Both must be non-ok.
        let report = verify_bundle(&root, &VerifyOptions::default());
        if let Ok(report) = report {
            assert!(!report.ok);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn deep_model_mismatch_fails() {
        let root = temp_root("deep");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        // A non-model file with the wrong hash fails the deep check.
        let model_path = root.join("fake-model.gguf");
        std::fs::write(&model_path, b"not a model").unwrap();
        let options = VerifyOptions {
            model_path: Some(model_path),
            tokenizer_path: None,
        };
        let report = verify_bundle(&root, &options).unwrap();
        assert!(!report.ok);
        let names: Vec<&str> = report
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| check.name.as_str())
            .collect();
        assert!(names.contains(&"deep model sha256"), "{names:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn traversal_in_checksums_is_rejected() {
        assert_verification_failure(
            "traversal",
            |root| {
                let path = root.join("checksums.sha256");
                let mut text = std::fs::read_to_string(&path).unwrap();
                text.push_str(&format!("{}\n", "00".repeat(32) + "  ../escape.bin"));
                std::fs::write(&path, text).unwrap();
            },
            &["checksums"],
        );
    }

    #[test]
    fn traversal_checksum_fails_even_with_the_correct_hash() {
        // A hostile bundle that lists `../victim` with the victim's real
        // hash must still fail: the path itself is rejected, so the check
        // can never become an arbitrary-file hash oracle.
        assert_verification_failure(
            "traversal-oracle",
            |root| {
                let parent = root.parent().unwrap();
                let victim = parent.join("ember-victim.txt");
                std::fs::write(&victim, b"secret").unwrap();
                let victim_hash = crate::v05::manifest::sha256_hex(b"secret");
                let path = root.join("checksums.sha256");
                let mut text = std::fs::read_to_string(&path).unwrap();
                // the victim lives one level above the bundle root
                text.push_str(&format!("{}\n", victim_hash + "  ../ember-victim.txt"));
                std::fs::write(&path, text).unwrap();
            },
            &["checksums"],
        );
    }

    #[test]
    fn source_bundle_loads_only_when_verified() {
        let root = temp_root("source");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        let loaded = load_bundle_for_source(&root).unwrap();
        let rows = loaded
            .tensor_f32_by_name("cap-1/i1/residual-post-mlp/0")
            .unwrap();
        assert_eq!(rows, testutil::sample_rows());
        // corrupt then refuse to load
        let payload = root.join("captures/tensors.safetensors");
        let mut bytes = std::fs::read(&payload).unwrap();
        bytes[20] ^= 0x01;
        std::fs::write(&payload, bytes).unwrap();
        assert!(load_bundle_for_source(&root).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    fn fix_checksums(root: &std::path::Path) {
        // Rewrite checksums.sha256 from the current files so later checks
        // pass and only the intended check fails.
        let mut lines: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(root).unwrap() {
            let entry = entry.unwrap();
            if !entry.file_type().unwrap().is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let bytes = std::fs::read(entry.path()).unwrap();
            lines.push(format!("{}  {name}", sha256_hex(&bytes)));
        }
        let path = root.join("captures").join("tensors.safetensors");
        let bytes = std::fs::read(&path).unwrap();
        lines.push(format!(
            "{}  captures/tensors.safetensors",
            sha256_hex(&bytes)
        ));
        lines.sort();
        std::fs::write(
            root.join("checksums.sha256"),
            format!("{}\n", lines.join("\n")),
        )
        .unwrap();
    }
}
