//! File-investigation digests behind `ember inspect` and the Python binding.
//!
//! Experimental public Rust API (`docs/api-stability.md` level 2): the
//! [`InspectReport`] shape is the machine-readable contract rendered by
//! `ember inspect --json` and returned as a dict by the `ember` Python
//! module; both consume this module directly, so keep field names and serde
//! attributes aligned with that contract.

use crate::extraction::sha256_file_result;
use crate::llama::Llama;
use crate::loader::{ggml_dtype_name, load_gguf_with_k_strategy, GgufValue};
use crate::plan::{ExecutionMode, ExecutionPlan, HookMode};
use crate::tokenizer::EmberTokenizer;
use anyhow::Context;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Detected file class. Serialized kebab-case (`gguf`, `tokenizer`,
/// `kv-snapshot`, `unknown`) to match the CLI JSON contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FileKind {
    Gguf,
    Tokenizer,
    KvSnapshot,
    Unknown,
}

/// One tensor row in the GGUF digest (capped at 12 rows by [`inspect_path`]).
#[derive(Debug, Clone, Serialize)]
pub struct TensorDigestEntry {
    pub name: String,
    pub dtype: String,
    pub dims: Vec<usize>,
    pub elements: u64,
}

/// Structural digest of a GGUF file.
#[derive(Debug, Clone, Serialize)]
pub struct GgufDigest {
    pub architecture: Option<String>,
    pub metadata_keys: usize,
    pub tensor_count: usize,
    pub total_elements: u64,
    pub dtype_histogram: BTreeMap<String, usize>,
    pub tensors: Vec<TensorDigestEntry>,
    pub k_decisions: BTreeMap<String, String>,
}

/// Tokenizer digest: vocabulary size only.
#[derive(Debug, Clone, Serialize)]
pub struct TokenizerDigest {
    pub vocab_size: usize,
}

/// KV snapshot digest: verification status plus the human summary.
#[derive(Debug, Clone, Serialize)]
pub struct KvSnapshotDigest {
    pub manifest_valid: bool,
    pub summary: String,
}

/// Full `ember inspect` report.
#[derive(Debug, Clone, Serialize)]
pub struct InspectReport {
    pub file: String,
    pub kind: FileKind,
    pub sha256: Option<String>,
    pub gguf: Option<GgufDigest>,
    pub tokenizer: Option<TokenizerDigest>,
    pub kv_snapshot: Option<KvSnapshotDigest>,
    pub notes: Vec<String>,
}

/// Inspect one path and return the structural report.
///
/// `sha256` hashes GGUF/tokenizer files only (KV snapshots hash on load).
/// An unrecognized path is *not* an error: it yields [`FileKind::Unknown`]
/// with a remediation note, matching the CLI.
pub fn inspect_path(path: &Path, sha256: bool) -> anyhow::Result<InspectReport> {
    let kind = detect_kind(path);
    let sha256 = if sha256 && matches!(kind, FileKind::Gguf | FileKind::Tokenizer) {
        Some(
            sha256_file_result(path.to_string_lossy().as_ref())
                .with_context(|| "failed to hash file")?,
        )
    } else {
        None
    };
    let (gguf, tokenizer, kv_snapshot, notes) = match kind {
        FileKind::Gguf => {
            let (digest, gguf_notes) = inspect_gguf(path)?;
            (Some(digest), None, None, gguf_notes)
        }
        FileKind::Tokenizer => {
            let (digest, tok_notes) = inspect_tokenizer(path)?;
            (None, Some(digest), None, tok_notes)
        }
        FileKind::KvSnapshot => {
            let (digest, snap_notes) = inspect_kv_snapshot(path)?;
            (None, None, Some(digest), snap_notes)
        }
        FileKind::Unknown => (
            None,
            None,
            None,
            vec![
                "unrecognized file type; inspect handles .gguf models, tokenizer .json files, and KV snapshot dirs — for run/bundle dirs use `validate-run`, for activation artifacts use `compare-artifacts`"
                    .to_string(),
            ],
        ),
    };
    Ok(InspectReport {
        file: path.display().to_string(),
        kind,
        sha256,
        gguf,
        tokenizer,
        kv_snapshot,
        notes,
    })
}

fn detect_kind(path: &Path) -> FileKind {
    if path.is_dir() {
        for name in [
            crate::kv_snapshot::KV_MANIFEST_FILE,
            crate::kv_snapshot::KV_KEY_FILE,
            crate::kv_snapshot::KV_VALUE_FILE,
        ] {
            if !path.join(name).is_file() {
                return FileKind::Unknown;
            }
        }
        return FileKind::KvSnapshot;
    }
    if path.extension().is_some_and(|ext| ext == "gguf") {
        return FileKind::Gguf;
    }
    if path
        .file_name()
        .is_some_and(|name| name.to_string_lossy().ends_with(".json"))
    {
        return FileKind::Tokenizer;
    }
    if let Ok(mut file) = std::fs::File::open(path) {
        use std::io::Read;
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_ok() && &magic == b"GGUF" {
            return FileKind::Gguf;
        }
    }
    FileKind::Unknown
}

/// Truncated display string for one GGUF metadata value (human digest only).
/// This is intentionally separate from `cli_support::gguf_value_json`, which
/// renders full machine-readable JSON: the digest needs short,
/// possibly-truncated strings, never nested arrays.
fn gguf_value_summary(value: &GgufValue) -> String {
    match value {
        GgufValue::U8(v) => v.to_string(),
        GgufValue::I8(v) => v.to_string(),
        GgufValue::U16(v) => v.to_string(),
        GgufValue::I16(v) => v.to_string(),
        GgufValue::U32(v) => v.to_string(),
        GgufValue::U64(v) => v.to_string(),
        GgufValue::I32(v) => v.to_string(),
        GgufValue::I64(v) => v.to_string(),
        GgufValue::F32(v) => v.to_string(),
        GgufValue::F64(v) => v.to_string(),
        GgufValue::Bool(v) => v.to_string(),
        GgufValue::Str(v) => {
            if v.len() > 80 {
                format!("{}...", &v[..80])
            } else {
                v.clone()
            }
        }
        GgufValue::Array(items) => format!("<array of {}>", items.len()),
        GgufValue::SkippedArray { elements, .. } => {
            format!("<{elements} values, not materialized>")
        }
    }
}

fn inspect_gguf(path: &Path) -> anyhow::Result<(GgufDigest, Vec<String>)> {
    // Structural validation happens inside the hardened loader (T0-T6 trust
    // boundary): magic, counts, string bounds, tensor records, offsets.
    // Inspect never parses GGUF bytes itself.
    let loader = load_gguf_with_k_strategy(path, crate::quant_k::KStrategy::Auto, true)
        .map_err(|error| anyhow::anyhow!("GGUF failed structural validation: {error}"))?;
    let mut notes = Vec::new();
    let architecture = loader
        .metadata
        .get("general.architecture")
        .map(gguf_value_summary);
    if architecture.is_none() {
        notes.push(
            "missing general.architecture metadata; generation and `inspect plan` need it — check the exporter wrote it, or pass --arch explicitly where supported"
                .to_string(),
        );
    }
    let mut dtype_histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut tensors: Vec<TensorDigestEntry> = Vec::new();
    let mut total_elements: u64 = 0;
    let mut names: Vec<&String> = loader.tensor_meta.keys().collect();
    names.sort();
    for name in names {
        let meta = &loader.tensor_meta[name];
        let dtype = ggml_dtype_name(meta.dtype)
            .map(str::to_string)
            .unwrap_or_else(|| format!("unknown({})", meta.dtype));
        if ggml_dtype_name(meta.dtype).is_none() {
            notes.push(format!(
                "tensor '{name}' has unknown dtype code {}; Ember cannot load it — re-export with a supported quantization (f32/f16/q8_0/q4_k/q6_k) or check for file corruption",
                meta.dtype
            ));
        }
        *dtype_histogram.entry(dtype.clone()).or_default() += 1;
        let elements: u64 = meta.dims.iter().map(|&d| d as u64).product();
        total_elements = total_elements.saturating_add(elements);
        tensors.push(TensorDigestEntry {
            name: name.clone(),
            dtype,
            dims: meta.dims.clone(),
            elements,
        });
    }
    let mut k_decisions = BTreeMap::new();
    for (name, decision) in &loader.k_decisions {
        let entry = match &decision.fallback_reason {
            Some(reason) => format!("{:?} (fallback: {reason})", decision.execution),
            None => format!("{:?}", decision.execution),
        };
        k_decisions.insert(name.clone(), entry);
    }
    if tensors.len() > 12 {
        let dropped = tensors.len() - 12;
        tensors.truncate(12);
        notes.push(format!(
            "{dropped} more tensors omitted from digest (see --json for full inventory)"
        ));
    }
    Ok((
        GgufDigest {
            architecture,
            metadata_keys: loader.metadata.len(),
            tensor_count: loader.tensor_meta.len(),
            total_elements,
            dtype_histogram,
            tensors,
            k_decisions,
        },
        notes,
    ))
}

fn inspect_tokenizer(path: &Path) -> anyhow::Result<(TokenizerDigest, Vec<String>)> {
    // Tokenizer parsing goes through the hardened EmberTokenizer boundary.
    let tokenizer = EmberTokenizer::from_file(path).with_context(|| "tokenizer failed to parse")?;
    Ok((
        TokenizerDigest {
            vocab_size: tokenizer.vocab_size(),
        },
        Vec::new(),
    ))
}

fn inspect_kv_snapshot(path: &Path) -> anyhow::Result<(KvSnapshotDigest, Vec<String>)> {
    // Snapshot loading verifies schema, shapes, payload checksums, and
    // manifest identity; a strict three-file directory is enforced.
    let snapshot = crate::kv_snapshot::KvSnapshot::load_dir(path)
        .map_err(|error| anyhow::anyhow!("KV snapshot failed verification: {error}"))?;
    Ok((
        KvSnapshotDigest {
            manifest_valid: true,
            summary: snapshot.to_summary_text(),
        },
        Vec::new(),
    ))
}

/// Execution-plan report for a llama-family GGUF (`ember inspect <file> plan`).
#[derive(Debug, Clone, Serialize)]
pub struct PlanReport {
    /// Resolved generation architecture (`llama` or `qwen3`).
    pub architecture: String,
    /// Canonical execution mode name (`reference` | `planned` | `planned-fused`).
    pub execution: String,
    /// The v0.4 execution plan (same JSON as the CLI `--output` file).
    pub plan: ExecutionPlan,
}

/// Build the v0.4 execution plan for a llama-family GGUF.
///
/// `arch` is `"auto"` or an explicit family (see
/// [`crate::loader::resolve_generation_architecture`]); `execution` is
/// `reference` | `planned` | `planned-fused`. Non-llama-family models are
/// rejected, matching `ember inspect plan`.
pub fn inspect_plan(path: &Path, arch: &str, execution: &str) -> anyhow::Result<PlanReport> {
    let execution_mode = ExecutionMode::from_cli(execution).map_err(anyhow::Error::msg)?;
    let loader = load_gguf_with_k_strategy(path, crate::quant_k::KStrategy::Auto, true)
        .map_err(|error| anyhow::anyhow!("GGUF failed structural validation: {error}"))?;
    let architecture = crate::loader::resolve_generation_architecture(arch, &loader)?;
    anyhow::ensure!(
        architecture == "llama" || architecture == "qwen3",
        "inspect plan supports llama-family models, resolved '{architecture}'; use plain `inspect` for the structural digest, or `kv` for snapshot workflows"
    );
    let model = Llama::from_loader_with_max_seq_len(loader, None)?;
    let max_seq_len = model.config.max_seq_len;
    let plan = model.execution_plan(
        execution_mode,
        HookMode::Disabled,
        &[],
        max_seq_len,
        None,
        None,
    )?;
    Ok(PlanReport {
        architecture,
        execution: execution_mode.name().to_string(),
        plan: (*plan).clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_kind_for_missing_trailing_snapshot_files() {
        let dir = std::env::temp_dir().join(format!(
            "ember-inspect-kind-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("manifest.json"), b"{}").unwrap();
        assert_eq!(detect_kind(&dir), FileKind::Unknown);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn gguf_magic_sniffed_without_extension() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ember-inspect-magic-{}", std::process::id()));
        std::fs::write(&path, b"GGUF\x03\0\0\0rest").unwrap();
        assert_eq!(detect_kind(&path), FileKind::Gguf);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn non_gguf_bytes_are_unknown() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ember-inspect-junk-{}", std::process::id()));
        std::fs::write(&path, b"definitely not a model file").unwrap();
        assert_eq!(detect_kind(&path), FileKind::Unknown);
        std::fs::remove_file(&path).unwrap();
    }
}
