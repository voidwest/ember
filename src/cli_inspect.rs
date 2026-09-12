use clap::{Args as ClapArgs, Subcommand};
use ember::inspect::{inspect_path, FileKind, InspectReport};
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
    let report = inspect_path(&command.file, command.sha256)?;
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
    let report = ember::inspect::inspect_plan(file, &args.arch, &args.execution)?;
    print!("{}", report.plan.to_summary_text());
    print_plan_details(&report.plan);
    if let Some(output) = &args.output {
        let json = serde_json::to_string_pretty(&report.plan)?;
        std::fs::write(output, json)?;
        eprintln!("wrote execution plan to {output}");
    }
    Ok(())
}

/// Print the derived detail the terse summary omits: the host-dependent
/// runtime schedule (kernel/thread selection), scratch-region lifetimes, and
/// the per-tensor kernel/ownership map. None of this is serialized into the
/// plan or its hash.
fn print_plan_details(plan: &ember::plan::ExecutionPlan) {
    println!();
    let schedule = ember::runtime_schedule::RuntimeSchedule::from_plan(plan);
    print!("{}", schedule.to_summary_text());
    println!();
    println!(
        "scratch regions ({} bytes, alignment {}, seq capacity {}):",
        plan.scratch.total_bytes, plan.scratch.alignment, plan.scratch.seq_capacity
    );
    for region in &plan.scratch.regions {
        println!(
            "  {:<28} offset {:>9}  size {:>8}  ops {:>4}..{:<4}{}",
            region.name,
            region.offset,
            region.size,
            region.first_op,
            region.last_op,
            region
                .shared_with
                .as_deref()
                .map(|shared| format!("  shared with {shared}"))
                .unwrap_or_default()
        );
    }
    println!();
    println!("tensors ({}):", plan.tensor_table.len());
    for record in &plan.tensor_table {
        println!(
            "  {:<44} {:<8} {:<24} {:>12} B  {}",
            record.name,
            record.gguf_dtype,
            record.kernel.name(),
            record.resident_bytes,
            if record.mmap { "mmap" } else { "resident" }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ember::inspect::{GgufDigest, TensorDigestEntry};
    use std::collections::BTreeMap;

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
