//! Integration coverage for the library `inspect` report, the `diff`
//! orchestration, and the execution-plan report that `ember inspect`/`diff`
//! and the Python binding both consume.

use ember::diff_outcome::evaluate_diff;
use ember::inspect::{inspect_path, FileKind};
use std::path::PathBuf;
use std::time::Duration;

#[path = "common/gguf.rs"]
mod gguf;
use gguf::*;

/// Minimal valid GGUF v3: one metadata key and one 2x4 f32 tensor.
fn build_minimal_gguf() -> Vec<u8> {
    build_gguf(
        &[TensorSpec {
            name: "test.weight".into(),
            dims: vec![2, 4],
            data: vec![1.0; 8],
        }],
        &[kv_string("general.name", "test")],
    )
}

/// Valid GGUF v3 with `count` one-element f32 tensors and no metadata.
fn build_gguf_with_tensors(count: usize) -> Vec<u8> {
    let tensors: Vec<TensorSpec> = (0..count)
        .map(|index| TensorSpec {
            name: format!("t{index:03}.weight"),
            dims: vec![1],
            data: vec![1.0],
        })
        .collect();
    build_gguf(&tensors, &[])
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ember-inspect-it-{}-{}-{name}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ))
}

#[test]
fn gguf_report_lists_every_tensor() {
    let path = temp_path("many-tensors.gguf");
    std::fs::write(&path, build_gguf_with_tensors(20)).unwrap();
    let report = inspect_path(&path, false).unwrap();
    let gguf = report.gguf.as_ref().expect("gguf digest");
    assert_eq!(gguf.tensor_count, 20);
    assert_eq!(gguf.tensors.len(), 20, "--json promises the full inventory");
    assert_eq!(gguf.tensors[19].name, "t019.weight");
    assert!(
        report.notes.iter().all(|note| !note.contains("omitted")),
        "{:?}",
        report.notes
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_report_carries_structure() {
    let path = temp_path("minimal.gguf");
    std::fs::write(&path, build_minimal_gguf()).unwrap();
    let report = inspect_path(&path, false).unwrap();
    assert_eq!(report.kind, FileKind::Gguf);
    assert!(report.sha256.is_none());
    let gguf = report.gguf.as_ref().expect("gguf digest");
    assert_eq!(gguf.tensor_count, 1);
    assert_eq!(gguf.tensors.len(), 1);
    assert_eq!(gguf.tensors[0].name, "test.weight");
    assert_eq!(gguf.tensors[0].dtype, "f32");
    assert_eq!(gguf.tensors[0].elements, 8);
    assert_eq!(gguf.dtype_histogram.get("f32"), Some(&1));
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn sha256_hashes_gguf_but_not_unknown_files() {
    let path = temp_path("hashed.gguf");
    std::fs::write(&path, build_minimal_gguf()).unwrap();
    let report = inspect_path(&path, true).unwrap();
    let sha = report.sha256.expect("sha256 requested");
    assert_eq!(sha.len(), 64);
    std::fs::remove_file(&path).unwrap();

    // An unrecognized file is not an error and is not hashed.
    let junk = temp_path("junk.bin");
    std::fs::write(&junk, b"not a model").unwrap();
    let report = inspect_path(&junk, true).unwrap();
    assert_eq!(report.kind, FileKind::Unknown);
    assert!(report.sha256.is_none());
    assert_eq!(report.notes.len(), 1);
    std::fs::remove_file(&junk).unwrap();
}

#[test]
fn truncated_gguf_fails_structural_validation() {
    let path = temp_path("broken.gguf");
    // Magic claims GGUF; the body is missing, so the loader must reject it.
    std::fs::write(&path, b"GGUF").unwrap();
    assert!(inspect_path(&path, false).is_err());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn diff_report_shape_without_externals() {
    let path = temp_path("diff.gguf");
    std::fs::write(&path, b"definitely not gguf").unwrap();
    let report = evaluate_diff(&path, &[], Duration::from_secs(1));
    assert_eq!(report.schema, "ember.diff.v1");
    assert_eq!(report.ember.runtime, "ember");
    assert!(report.externals.is_empty());
    assert!(report.agreement.all_agree);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn plan_report_builds_for_tiny_llama() {
    let path = temp_path("tiny-llama-plan.gguf");
    std::fs::write(&path, tiny_llama_gguf()).unwrap();
    let report = ember::inspect::inspect_plan(&path, "auto", "planned").unwrap();
    assert_eq!(report.architecture, "llama");
    assert_eq!(report.execution, "planned");
    assert!(!report.plan.tensor_table.is_empty());
    assert_eq!(
        report.plan.kernel_revision,
        ember::plan::PLAN_KERNEL_REVISION
    );
    // The serialized plan is the CLI `--output` contract the binding returns.
    let json = serde_json::to_value(&report.plan).unwrap();
    assert!(json.get("kernel_revision").is_some(), "plan JSON: {json}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn plan_rejects_conflicting_arch_and_bad_execution() {
    let path = temp_path("tiny-llama-plan-bad.gguf");
    std::fs::write(&path, tiny_llama_gguf()).unwrap();
    // Explicit arch conflicts with the GGUF's declared `llama`.
    assert!(ember::inspect::inspect_plan(&path, "gpt2", "planned").is_err());
    // Unknown execution mode fails before loading.
    assert!(ember::inspect::inspect_plan(&path, "auto", "turbo").is_err());
    std::fs::remove_file(&path).unwrap();
}
