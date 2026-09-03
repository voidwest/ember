use anyhow::Context;
use clap::{Args as ClapArgs, Subcommand};
use ember::extraction::sha256_file_result;
use ember::loader::{ggml_dtype_name, load_gguf_with_k_strategy, GgufValue};
use ember::tokenizer::EmberTokenizer;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(ClapArgs)]
pub(crate) struct InspectCommand {
    /// File to inspect (GGUF model, tokenizer.json, or KV snapshot directory).
    pub file: PathBuf,
    /// Emit machine-readable JSON to stdout (default: human-readable digest).
    #[arg(long)]
    pub json: bool,
    /// Also hash the file with SHA-256 (GGUF/tokenizer; snapshots hash on load).
    #[arg(long)]
    pub sha256: bool,
    #[command(subcommand)]
    pub command: Option<InspectSubcommand>,
}

#[derive(Subcommand)]
pub(crate) enum InspectSubcommand {
    /// Show the v0.4 execution plan for a llama-family GGUF (power-user depth).
    Plan(InspectPlanArgs),
    /// Verify a KV snapshot directory (power-user depth).
    VerifySnapshot(VerifySnapshotArgs),
}

#[derive(ClapArgs)]
pub(crate) struct InspectPlanArgs {
    /// Model architecture override; auto reads general.architecture from GGUF.
    #[arg(long, default_value = "auto", value_parser = ["auto", "llama", "qwen3"])]
    pub arch: String,
    /// Execution mode: reference | planned | planned-fused.
    #[arg(long, default_value = "planned")]
    pub execution: String,
    /// Write the serialized execution-plan.json to this path.
    #[arg(long)]
    pub output: Option<String>,
}

#[derive(ClapArgs)]
pub(crate) struct VerifySnapshotArgs {
    /// Snapshot directory (defaults to the inspected file when it is one).
    #[arg(long)]
    pub snapshot: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum FileKind {
    Gguf,
    Tokenizer,
    KvSnapshot,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
struct TensorDigestEntry {
    name: String,
    dtype: String,
    dims: Vec<usize>,
    elements: u64,
}

#[derive(Debug, Clone, Serialize)]
struct GgufDigest {
    architecture: Option<String>,
    metadata_keys: usize,
    tensor_count: usize,
    total_elements: u64,
    dtype_histogram: BTreeMap<String, usize>,
    tensors: Vec<TensorDigestEntry>,
    k_decisions: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
struct TokenizerDigest {
    vocab_size: usize,
}

#[derive(Debug, Clone, Serialize)]
struct KvSnapshotDigest {
    manifest_valid: bool,
    summary: String,
}

#[derive(Debug, Clone, Serialize)]
struct InspectReport {
    file: String,
    kind: FileKind,
    sha256: Option<String>,
    gguf: Option<GgufDigest>,
    tokenizer: Option<TokenizerDigest>,
    kv_snapshot: Option<KvSnapshotDigest>,
    notes: Vec<String>,
}

fn detect_kind(path: &Path) -> FileKind {
    if path.is_dir() {
        for name in [
            ember::kv_snapshot::KV_MANIFEST_FILE,
            ember::kv_snapshot::KV_KEY_FILE,
            ember::kv_snapshot::KV_VALUE_FILE,
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
    }
}

fn inspect_gguf(path: &Path) -> anyhow::Result<(GgufDigest, Vec<String>)> {
    // Structural validation happens inside the hardened loader (T0-T6 trust
    // boundary): magic, counts, string bounds, tensor records, offsets.
    // Inspect never parses GGUF bytes itself.
    let loader = load_gguf_with_k_strategy(path, ember::quant_k::KStrategy::Auto, true)
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
    let snapshot = ember::kv_snapshot::KvSnapshot::load_dir(path)
        .map_err(|error| anyhow::anyhow!("KV snapshot failed verification: {error}"))?;
    Ok((
        KvSnapshotDigest {
            manifest_valid: true,
            summary: snapshot.to_summary_text(),
        },
        Vec::new(),
    ))
}

fn render_human(report: &InspectReport) -> String {
    let mut lines = Vec::new();
    let kind = match report.kind {
        FileKind::Gguf => "GGUF model",
        FileKind::Tokenizer => "tokenizer",
        FileKind::KvSnapshot => "KV snapshot",
        FileKind::Unknown => "unknown file",
    };
    lines.push(format!("{}: {}", kind, report.file));
    if let Some(sha) = &report.sha256 {
        lines.push(format!("sha256: {sha}"));
    }
    if let Some(gguf) = &report.gguf {
        lines.push(format!(
            "architecture: {}",
            gguf.architecture.as_deref().unwrap_or("unknown")
        ));
        lines.push(format!(
            "metadata keys: {}  tensors: {}  total elements: {}",
            gguf.metadata_keys, gguf.tensor_count, gguf.total_elements
        ));
        let histogram = gguf
            .dtype_histogram
            .iter()
            .map(|(dtype, count)| format!("{dtype}×{count}"))
            .collect::<Vec<_>>()
            .join(" ");
        lines.push(format!("dtypes: {histogram}"));
        for tensor in &gguf.tensors {
            lines.push(format!(
                "  {}  {}  {:?}  ({} elements)",
                tensor.name, tensor.dtype, tensor.dims, tensor.elements
            ));
        }
        let fallbacks = gguf
            .k_decisions
            .iter()
            .filter(|(_, decision)| decision.contains("fallback"))
            .count();
        if fallbacks > 0 {
            lines.push(format!("K-strategy fallbacks: {fallbacks} tensors"));
        }
    }
    if let Some(tokenizer) = &report.tokenizer {
        lines.push(format!("vocab size: {}", tokenizer.vocab_size));
    }
    if let Some(snapshot) = &report.kv_snapshot {
        lines.push(format!("manifest valid: {}", snapshot.manifest_valid));
        lines.push(snapshot.summary.clone());
    }
    for note in &report.notes {
        lines.push(format!("note: {note}"));
    }
    lines.join("\n")
}

pub(crate) fn run_inspect_command(command: &InspectCommand) -> anyhow::Result<()> {
    if let Some(subcommand) = &command.command {
        return run_inspect_subcommand(&command.file, subcommand);
    }
    let kind = detect_kind(&command.file);
    let sha256 = if command.sha256 && matches!(kind, FileKind::Gguf | FileKind::Tokenizer) {
        Some(
            sha256_file_result(command.file.to_string_lossy().as_ref())
                .with_context(|| "failed to hash file")?,
        )
    } else {
        None
    };
    let (gguf, tokenizer, kv_snapshot, notes) = match kind {
        FileKind::Gguf => {
            let (digest, gguf_notes) = inspect_gguf(&command.file)?;
            (Some(digest), None, None, gguf_notes)
        }
        FileKind::Tokenizer => {
            let (digest, tok_notes) = inspect_tokenizer(&command.file)?;
            (None, Some(digest), None, tok_notes)
        }
        FileKind::KvSnapshot => {
            let (digest, snap_notes) = inspect_kv_snapshot(&command.file)?;
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
    let report = InspectReport {
        file: command.file.display().to_string(),
        kind,
        sha256,
        gguf,
        tokenizer,
        kv_snapshot,
        notes,
    };
    if command.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", render_human(&report));
    }
    Ok(())
}

fn run_inspect_subcommand(file: &Path, subcommand: &InspectSubcommand) -> anyhow::Result<()> {
    match subcommand {
        InspectSubcommand::Plan(args) => run_inspect_plan(file, args),
        InspectSubcommand::VerifySnapshot(args) => {
            let snapshot = args.snapshot.clone().unwrap_or_else(|| file.to_path_buf());
            let loaded = ember::kv_snapshot::KvSnapshot::load_dir(&snapshot)
                .map_err(|error| anyhow::anyhow!("KV snapshot failed verification: {error}"))?;
            println!("{}", loaded.to_summary_text());
            Ok(())
        }
    }
}

fn run_inspect_plan(file: &Path, args: &InspectPlanArgs) -> anyhow::Result<()> {
    use crate::cli_support::resolve_generation_architecture;
    use ember::llama::Llama;
    use ember::plan::HookMode;
    let execution =
        ember::plan::ExecutionMode::from_cli(&args.execution).map_err(anyhow::Error::msg)?;
    let loader = load_gguf_with_k_strategy(file, ember::quant_k::KStrategy::Auto, true)
        .map_err(|error| anyhow::anyhow!("GGUF failed structural validation: {error}"))?;
    // Single source of truth for GGUF arch mapping and --arch conflict
    // checking; llama-family gating below mirrors inspect-plan exactly.
    let architecture = resolve_generation_architecture(&args.arch, &loader)?;
    anyhow::ensure!(
        architecture == "llama" || architecture == "qwen3",
        "inspect plan supports llama-family models, resolved '{architecture}'; use plain `inspect` for the structural digest, or `kv` for snapshot workflows"
    );
    let model = Llama::from_loader_with_max_seq_len(loader, None)?;
    let max_seq_len = model.config.max_seq_len;
    let plan = model.execution_plan(execution, HookMode::Disabled, &[], max_seq_len, None, None)?;
    print!("{}", plan.to_summary_text());
    if let Some(output) = &args.output {
        let json = serde_json::to_string_pretty(&*plan)?;
        std::fs::write(output, json)?;
        eprintln!("wrote execution plan to {output}");
    }
    Ok(())
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

    #[test]
    fn human_digest_renders_sections() {
        let report = InspectReport {
            file: "model.gguf".to_string(),
            kind: FileKind::Gguf,
            sha256: Some("ab".repeat(32)),
            gguf: Some(GgufDigest {
                architecture: Some("llama".to_string()),
                metadata_keys: 10,
                tensor_count: 2,
                total_elements: 100,
                dtype_histogram: BTreeMap::from([("q4_k".to_string(), 2)]),
                tensors: vec![TensorDigestEntry {
                    name: "blk.0.attn_q.weight".to_string(),
                    dtype: "q4_k".to_string(),
                    dims: vec![2048, 2048],
                    elements: 100,
                }],
                k_decisions: BTreeMap::new(),
            }),
            tokenizer: None,
            kv_snapshot: None,
            notes: vec!["a note".to_string()],
        };
        let text = render_human(&report);
        assert!(text.contains("GGUF model: model.gguf"));
        assert!(text.contains("architecture: llama"));
        assert!(text.contains("blk.0.attn_q.weight"));
        assert!(text.contains("note: a note"));
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"kind\":\"gguf\""));
    }

    #[test]
    fn remediation_notes_name_a_next_step() {
        // Luminal rule 25: every finding tells the user what to do next.
        let unknown = InspectReport {
            file: "x".to_string(),
            kind: FileKind::Unknown,
            sha256: None,
            gguf: None,
            tokenizer: None,
            kv_snapshot: None,
            notes: vec![
                "unrecognized file type; inspect handles .gguf models, tokenizer .json files, and KV snapshot dirs — for run/bundle dirs use `validate-run`, for activation artifacts use `compare-artifacts`"
                    .to_string(),
            ],
        };
        let text = render_human(&unknown);
        assert!(text.contains("validate-run"));
        assert!(text.contains("compare-artifacts"));
    }
}
