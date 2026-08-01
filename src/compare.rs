//! v0.2 activation-artifact comparison.
//!
//! Compares two capture artifacts record-by-record with deterministic
//! alignment: records are matched on (phase, layer, stage, start_position).
//! Duplicate keys on either side are a hard error — alignment never guesses.
//! Records present on only one side are reported as missing/extra.
//!
//! Determinism: the only field ignored in comparison is `created_at_unix`
//! (explicitly nondeterministic provenance). Everything else either compares
//! exactly or is reported. JSON output is stable for identical inputs.

use crate::artifact::{load_manifest, ActivationManifest, CaptureRecord};
use serde::Serialize;
use std::collections::BTreeMap;

/// Outcome of the tensor-level comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompareStatus {
    /// Every aligned record is bit-identical; no missing or extra records.
    Identical,
    /// At least one aligned record differs, or records are missing/extra.
    Differs,
    /// One side has duplicate alignment keys; comparison refused.
    AlignmentError,
}

/// Run-level field comparison (informational; does not drive status).
#[derive(Debug, Clone, Serialize)]
pub struct RunComparison {
    pub model_sha256_match: Option<bool>,
    pub tokenizer_sha256_match: Option<bool>,
    pub ember_version_match: Option<bool>,
    pub git_commit_match: Option<bool>,
    pub prompt_hash_match: Option<bool>,
    pub input_token_ids_match: bool,
    pub generated_token_ids_match: bool,
    pub model_family_left: String,
    pub model_family_right: String,
}

/// One aligned record comparison.
#[derive(Debug, Clone, Serialize)]
pub struct RecordComparison {
    pub phase: String,
    pub layer: usize,
    pub stage: String,
    pub start_position: usize,
    pub present_left: bool,
    pub present_right: bool,
    pub shape_left: Option<[usize; 2]>,
    pub shape_right: Option<[usize; 2]>,
    pub shape_match: Option<bool>,
    pub dtype_match: Option<bool>,
    pub manifest_sha256_match: Option<bool>,
    /// Bit-exact equality of the loaded f32 values.
    pub exact_equal: bool,
    /// Metrics are `None` when shapes disagree (not comparable element-wise).
    pub max_abs_diff: Option<f64>,
    pub mean_abs_diff: Option<f64>,
    pub rms_diff: Option<f64>,
    pub cosine: Option<f64>,
    pub l2_left: Option<f64>,
    pub l2_right: Option<f64>,
    pub rel_l2_error: Option<f64>,
}

/// The full comparison report. Deterministic for identical inputs.
#[derive(Debug, Clone, Serialize)]
pub struct CompareReport {
    pub schema_version: String,
    pub left: String,
    pub right: String,
    pub status: CompareStatus,
    pub run: RunComparison,
    pub aligned_record_count: usize,
    pub identical_record_count: usize,
    pub differing_record_count: usize,
    pub missing_left: Vec<String>,
    pub missing_right: Vec<String>,
    pub records: Vec<RecordComparison>,
    /// Fields intentionally excluded from comparison.
    pub ignored_fields: [&'static str; 1],
}

/// Alignment key for one record.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RecordKey {
    phase: String,
    layer: usize,
    stage: String,
    start_position: usize,
}

impl RecordKey {
    fn of(record: &CaptureRecord) -> Self {
        Self {
            phase: record.phase.clone(),
            layer: record.layer,
            stage: record.stage.to_string(),
            start_position: record.start_position,
        }
    }
}

/// Compare two v0.2 activation artifacts by manifest path.
pub fn compare_artifacts(left: &str, right: &str) -> Result<CompareReport, String> {
    let left_manifest = load_manifest(left)?;
    let right_manifest = load_manifest(right)?;
    compare_manifests(left, right, &left_manifest, &right_manifest)
}

fn compare_manifests(
    left_path: &str,
    right_path: &str,
    left: &ActivationManifest,
    right: &ActivationManifest,
) -> Result<CompareReport, String> {
    let left_map = index_records(&left.records, "left")?;
    let right_map = index_records(&right.records, "right")?;

    let mut keys: Vec<RecordKey> = left_map.keys().cloned().collect();
    for key in right_map.keys() {
        if !left_map.contains_key(key) {
            keys.push(key.clone());
        }
    }
    keys.sort();

    let run = RunComparison {
        model_sha256_match: compare_option(&left.model.sha256, &right.model.sha256),
        tokenizer_sha256_match: compare_option(
            &left.model.tokenizer_sha256,
            &right.model.tokenizer_sha256,
        ),
        ember_version_match: compare_option(
            &Some(left.ember_version.clone()),
            &Some(right.ember_version.clone()),
        ),
        git_commit_match: compare_option(&left.git_commit, &right.git_commit),
        prompt_hash_match: compare_option(
            &Some(left.run.prompt_hash.clone()),
            &Some(right.run.prompt_hash.clone()),
        ),
        input_token_ids_match: left.run.input_token_ids == right.run.input_token_ids,
        generated_token_ids_match: left.run.generated_token_ids == right.run.generated_token_ids,
        model_family_left: left.model.family.clone(),
        model_family_right: right.model.family.clone(),
    };

    let mut records = Vec::with_capacity(keys.len());
    let mut missing_left = Vec::new();
    let mut missing_right = Vec::new();
    let mut identical = 0usize;
    let mut differing = 0usize;

    for key in &keys {
        let (left_record, right_record) = match (left_map.get(key), right_map.get(key)) {
            (Some(left), Some(right)) => (left, right),
            (Some(_), None) => {
                missing_right.push(format_key(key));
                records.push(record_absent(key, true, false));
                differing += 1;
                continue;
            }
            (None, Some(_)) => {
                missing_left.push(format_key(key));
                records.push(record_absent(key, false, true));
                differing += 1;
                continue;
            }
            (None, None) => unreachable!("keys built from the union of both maps"),
        };

        let comparison = compare_record(
            left_path,
            right_path,
            left,
            right,
            key,
            left_record,
            right_record,
        );
        let fully_identical = comparison.exact_equal
            && comparison.shape_match != Some(false)
            && comparison.dtype_match != Some(false);
        if fully_identical {
            identical += 1;
        } else {
            differing += 1;
        }
        records.push(comparison);
    }

    let status = if missing_left.is_empty() && missing_right.is_empty() && differing == 0 {
        CompareStatus::Identical
    } else {
        CompareStatus::Differs
    };

    Ok(CompareReport {
        schema_version: crate::artifact::ACTIVATION_ARTIFACT_SCHEMA.to_string(),
        left: left_path.to_string(),
        right: right_path.to_string(),
        status,
        run,
        aligned_record_count: records.len(),
        identical_record_count: identical,
        differing_record_count: differing,
        missing_left,
        missing_right,
        records,
        ignored_fields: ["created_at_unix"],
    })
}

fn index_records<'a>(
    records: &'a [CaptureRecord],
    side: &str,
) -> Result<BTreeMap<RecordKey, &'a CaptureRecord>, String> {
    let mut map = BTreeMap::new();
    for record in records {
        let key = RecordKey::of(record);
        if map.insert(key.clone(), record).is_some() {
            return Err(format!(
                "ambiguous record alignment: {side} artifact has duplicate records for {}; \
                 refusing to guess an alignment",
                format_key(&key)
            ));
        }
    }
    Ok(map)
}

fn compare_record(
    left_path: &str,
    right_path: &str,
    _left_manifest: &ActivationManifest,
    _right_manifest: &ActivationManifest,
    key: &RecordKey,
    left: &CaptureRecord,
    right: &CaptureRecord,
) -> RecordComparison {
    let shape_match = Some(left.shape == right.shape);
    let dtype_match = Some(left.dtype == right.dtype);
    let manifest_sha256_match = Some(left.sha256 == right.sha256);

    let left_values = load_record_values(left_path, left);
    let right_values = load_record_values(right_path, right);

    let (shape_left, shape_right) = (Some(left.shape), Some(right.shape));
    let exact_equal = match (&left_values, &right_values) {
        (Ok(left), Ok(right)) => left.values == right.values,
        _ => false,
    };
    let metrics = match (&left_values, &right_values) {
        (Ok(left), Ok(right)) if left.shape == right.shape => {
            Some(tensor_metrics(&left.values, &right.values))
        }
        _ => None,
    };

    RecordComparison {
        phase: key.phase.clone(),
        layer: key.layer,
        stage: key.stage.clone(),
        start_position: key.start_position,
        present_left: true,
        present_right: true,
        shape_left,
        shape_right,
        shape_match,
        dtype_match,
        manifest_sha256_match,
        exact_equal,
        max_abs_diff: metrics.as_ref().map(|m| m.max_abs_diff),
        mean_abs_diff: metrics.as_ref().map(|m| m.mean_abs_diff),
        rms_diff: metrics.as_ref().map(|m| m.rms_diff),
        cosine: metrics.as_ref().map(|m| m.cosine),
        l2_left: metrics.as_ref().map(|m| m.l2_left),
        l2_right: metrics.as_ref().map(|m| m.l2_right),
        rel_l2_error: metrics.as_ref().map(|m| m.rel_l2_error),
    }
}

fn record_absent(key: &RecordKey, present_left: bool, present_right: bool) -> RecordComparison {
    RecordComparison {
        phase: key.phase.clone(),
        layer: key.layer,
        stage: key.stage.clone(),
        start_position: key.start_position,
        present_left,
        present_right,
        shape_left: None,
        shape_right: None,
        shape_match: None,
        dtype_match: None,
        manifest_sha256_match: None,
        exact_equal: false,
        max_abs_diff: None,
        mean_abs_diff: None,
        rms_diff: None,
        cosine: None,
        l2_left: None,
        l2_right: None,
        rel_l2_error: None,
    }
}

struct LoadedRecord {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn load_record_values(manifest_path: &str, record: &CaptureRecord) -> Result<LoadedRecord, String> {
    let base = std::path::Path::new(manifest_path)
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let path = base.join(&record.path);
    let path = path
        .to_str()
        .ok_or_else(|| format!("record path is not valid UTF-8: {}", path.display()))?;
    let (shape, values) =
        crate::npy::read_npy_2d(path).map_err(|e| format!("failed to read '{}': {e}", path))?;
    Ok(LoadedRecord { shape, values })
}

struct TensorMetrics {
    max_abs_diff: f64,
    mean_abs_diff: f64,
    rms_diff: f64,
    cosine: f64,
    l2_left: f64,
    l2_right: f64,
    rel_l2_error: f64,
}

fn tensor_metrics(left: &[f32], right: &[f32]) -> TensorMetrics {
    let mut sum_diff = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_left = 0.0f64;
    let mut sum_sq_right = 0.0f64;
    let mut max_abs_diff = 0.0f64;
    for (a, b) in left.iter().zip(right.iter()) {
        let a = *a as f64;
        let b = *b as f64;
        let diff = a - b;
        let abs = diff.abs();
        if abs > max_abs_diff {
            max_abs_diff = abs;
        }
        sum_diff += abs;
        sum_sq_diff += diff * diff;
        sum_sq_left += a * a;
        sum_sq_right += b * b;
    }
    let count = left.len().max(1) as f64;
    let l2_left = sum_sq_left.sqrt();
    let l2_right = sum_sq_right.sqrt();
    let cosine = if l2_left > 0.0 && l2_right > 0.0 {
        let dot: f64 = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        dot / (l2_left * l2_right)
    } else {
        0.0
    };
    let rel_l2_error = if l2_left > 0.0 {
        sum_sq_diff.sqrt() / l2_left
    } else if sum_sq_diff == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    TensorMetrics {
        max_abs_diff,
        mean_abs_diff: sum_diff / count,
        rms_diff: (sum_sq_diff / count).sqrt(),
        cosine,
        l2_left,
        l2_right,
        rel_l2_error,
    }
}

fn compare_option<T: PartialEq>(left: &Option<T>, right: &Option<T>) -> Option<bool> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left == right),
        _ => None,
    }
}

fn format_key(key: &RecordKey) -> String {
    format!(
        "{} layer{} {} pos{}",
        key.phase, key.layer, key.stage, key.start_position
    )
}

/// Concise human-readable rendering of a comparison report.
pub fn render_human(report: &CompareReport) -> String {
    let mut lines = Vec::new();
    lines.push(format!("status: {:?}", report.status));
    lines.push(format!(
        "records: {} aligned, {} identical, {} differing, {} missing on left, {} missing on right",
        report.aligned_record_count,
        report.identical_record_count,
        report.differing_record_count,
        report.missing_left.len(),
        report.missing_right.len()
    ));
    lines.push(format!(
        "run: model_sha256_match={:?} tokenizer_sha256_match={:?} prompt_hash_match={:?} input_ids_match={} generated_ids_match={}",
        report.run.model_sha256_match,
        report.run.tokenizer_sha256_match,
        report.run.prompt_hash_match,
        report.run.input_token_ids_match,
        report.run.generated_token_ids_match
    ));
    for record in &report.records {
        if !record.present_left || !record.present_right {
            let side = if !record.present_left {
                "missing on left"
            } else {
                "missing on right"
            };
            lines.push(format!(
                "  {} layer{} {} pos{}: {side}",
                record.phase, record.layer, record.stage, record.start_position
            ));
        } else if !record.exact_equal {
            let shape = match (record.shape_match, record.max_abs_diff, record.cosine) {
                (Some(false), _, _) => "SHAPE MISMATCH".to_string(),
                (_, Some(max), Some(cosine)) => {
                    format!("max_abs_diff={max:.6} cosine={cosine:.6}")
                }
                _ => "differing".to_string(),
            };
            lines.push(format!(
                "  {} layer{} {} pos{}: {shape}",
                record.phase, record.layer, record.stage, record.start_position
            ));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::ActivationStage;
    use crate::experiments::{
        CaptureSink, ExecutionContext, ExecutionPhase, GenerationContext, ModelContext,
        ModelFamily, TensorAccess, TracingState,
    };

    struct ArtifactBuilder {
        #[allow(dead_code)]
        dir: std::path::PathBuf,
        sink: CaptureSink,
        model: ModelContext<'static>,
        input_ids: Vec<u32>,
    }

    impl ArtifactBuilder {
        fn new(name: &str, prompt_len: usize) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("ember_compare_test_{}_{name}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let config_path = dir.join("capture.toml");
            std::fs::write(
                &config_path,
                format!(
                    "schema_version = 1\noutput_dir = {:?}\nlayers = [1]\nstages = [\"after-mlp\", \"after-logits\"]\nphase = \"both\"\n",
                    dir.to_str().unwrap()
                ),
            )
            .unwrap();
            let sink = CaptureSink::from_toml_path(
                config_path.to_str().unwrap(),
                "compare test prompt",
                1,
                serde_json::json!({}),
                Some("model-hash-a".to_string()),
                None,
                serde_json::json!({}),
            )
            .unwrap();
            let model = ModelContext::new(ModelFamily::Qwen3, Some("tiny.gguf"), "qwen3", 4, 8);
            let mut sink = sink;
            sink.on_model_loaded(&model).unwrap();
            let input_ids: Vec<u32> = (1..=prompt_len as u32).collect();
            Self {
                dir,
                sink,
                model,
                input_ids,
            }
        }

        fn record_prefill(&mut self, values: Vec<f32>) {
            let seq = self.input_ids.len();
            let execution = ExecutionContext::new(
                self.model,
                ExecutionPhase::Prefill,
                0,
                seq,
                TracingState::Disabled,
            );
            let mut values = values;
            let tensor = TensorAccess::new(seq, 8, &mut values);
            self.sink
                .after_mlp(
                    &execution,
                    1,
                    &tensor,
                    crate::artifact::DispatchPath::Generic,
                )
                .unwrap();
        }

        fn record_decode(&mut self, position: usize, values: Vec<f32>) {
            let execution = ExecutionContext::new(
                self.model,
                ExecutionPhase::Decode,
                position,
                1,
                TracingState::Disabled,
            );
            let mut values = values;
            let tensor = TensorAccess::new(1, 8, &mut values);
            self.sink
                .after_mlp(&execution, 1, &tensor, crate::artifact::DispatchPath::Fast)
                .unwrap();
        }

        fn finalize(mut self) -> std::path::PathBuf {
            let generation = GenerationContext::new(
                self.model,
                self.input_ids.len(),
                1,
                1,
                TracingState::Disabled,
                &self.input_ids,
                &[9],
            );
            self.sink
                .finalize(
                    &generation,
                    crate::artifact::ManifestExperiment {
                        name: "none".to_string(),
                        arguments: serde_json::Value::Null,
                    },
                    Vec::new(),
                )
                .unwrap()
        }
    }

    fn identical_artifacts(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut left = ArtifactBuilder::new(&format!("{name}_left"), 2);
        left.record_prefill(vec![1.0; 16]);
        left.record_decode(2, vec![2.0; 8]);
        let left_path = left.finalize();

        let mut right = ArtifactBuilder::new(&format!("{name}_right"), 2);
        right.record_prefill(vec![1.0; 16]);
        right.record_decode(2, vec![2.0; 8]);
        let right_path = right.finalize();
        (left_path, right_path)
    }

    #[test]
    fn exact_equality_reports_identical() {
        let (left, right) = identical_artifacts("exact");
        let report = compare_artifacts(left.to_str().unwrap(), right.to_str().unwrap()).unwrap();
        assert_eq!(report.status, CompareStatus::Identical);
        assert_eq!(report.aligned_record_count, 2);
        assert_eq!(report.identical_record_count, 2);
        assert_eq!(report.differing_record_count, 0);
        assert!(report.records.iter().all(|r| r.exact_equal));
        assert_eq!(report.run.model_sha256_match, Some(true));
        assert!(report.run.input_token_ids_match);
        std::fs::remove_dir_all(left.parent().unwrap()).ok();
        std::fs::remove_dir_all(right.parent().unwrap()).ok();
    }

    #[test]
    fn one_element_perturbation_detected() {
        let (left, right) = identical_artifacts("perturb");
        // perturb one value in the right prefill record
        let right_manifest = load_manifest(right.to_str().unwrap()).unwrap();
        let prefill = right_manifest
            .records
            .iter()
            .find(|r| r.phase == "prefill" && r.stage == ActivationStage::AfterMlp)
            .unwrap()
            .clone();
        let tensor_path = right.parent().unwrap().join(&prefill.path);
        let (shape, mut values) = crate::npy::read_npy_2d(tensor_path.to_str().unwrap()).unwrap();
        values[0] += 0.5;
        crate::npy::write_npy_2d(
            tensor_path.to_str().unwrap(),
            &values,
            &[shape[0], shape[1]],
        )
        .unwrap();

        let report = compare_artifacts(left.to_str().unwrap(), right.to_str().unwrap()).unwrap();
        assert_eq!(report.status, CompareStatus::Differs);
        let prefill_record = report
            .records
            .iter()
            .find(|r| r.phase == "prefill" && r.stage == "after-mlp")
            .unwrap();
        assert!(!prefill_record.exact_equal);
        assert_eq!(prefill_record.max_abs_diff, Some(0.5));
        let decode_record = report.records.iter().find(|r| r.phase == "decode").unwrap();
        assert!(decode_record.exact_equal);
        std::fs::remove_dir_all(left.parent().unwrap()).ok();
        std::fs::remove_dir_all(right.parent().unwrap()).ok();
    }

    #[test]
    fn shape_mismatch_reported_without_metrics() {
        let mut left = ArtifactBuilder::new("left_shape", 2);
        left.record_prefill(vec![1.0; 16]);
        let left_path = left.finalize();

        let mut right = ArtifactBuilder::new("right_shape", 3);
        right.record_prefill(vec![1.0; 24]);
        let right_path = right.finalize();

        let report =
            compare_artifacts(left_path.to_str().unwrap(), right_path.to_str().unwrap()).unwrap();
        assert_eq!(report.status, CompareStatus::Differs);
        let record = report
            .records
            .iter()
            .find(|r| r.phase == "prefill")
            .unwrap();
        assert_eq!(record.shape_match, Some(false));
        assert_eq!(record.max_abs_diff, None);
        std::fs::remove_dir_all(left_path.parent().unwrap()).ok();
        std::fs::remove_dir_all(right_path.parent().unwrap()).ok();
    }

    #[test]
    fn dtype_mismatch_reported() {
        let (left, right) = identical_artifacts("dtype");
        // hand-edit the right manifest's dtype
        let right_manifest_path = right.clone();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&right_manifest_path).unwrap()).unwrap();
        let mut manifest = manifest;
        manifest["records"][0]["dtype"] = serde_json::json!("f64");
        std::fs::write(
            &right_manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let report = compare_artifacts(left.to_str().unwrap(), right.to_str().unwrap()).unwrap();
        assert_eq!(report.status, CompareStatus::Differs);
        let prefill = report
            .records
            .iter()
            .find(|r| r.phase == "prefill" && r.stage == "after-mlp")
            .unwrap();
        assert_eq!(prefill.dtype_match, Some(false));
        std::fs::remove_dir_all(left.parent().unwrap()).ok();
        std::fs::remove_dir_all(right.parent().unwrap()).ok();
    }

    #[test]
    fn missing_records_reported_on_both_sides() {
        let mut left = ArtifactBuilder::new("left_missing", 2);
        left.record_prefill(vec![1.0; 16]);
        left.record_decode(2, vec![2.0; 8]);
        let left_path = left.finalize();

        let mut right = ArtifactBuilder::new("right_missing", 2);
        right.record_prefill(vec![1.0; 16]);
        right.record_decode(2, vec![2.0; 8]);
        right.record_decode(3, vec![3.0; 8]);
        let right_path = right.finalize();

        let report =
            compare_artifacts(left_path.to_str().unwrap(), right_path.to_str().unwrap()).unwrap();
        assert_eq!(report.status, CompareStatus::Differs);
        assert_eq!(report.missing_left.len(), 1);
        assert!(report.missing_left[0].contains("pos3"));
        assert!(report.missing_right.is_empty());
        std::fs::remove_dir_all(left_path.parent().unwrap()).ok();
        std::fs::remove_dir_all(right_path.parent().unwrap()).ok();
    }

    #[test]
    fn duplicate_keys_refuse_alignment() {
        let (left, _right) = identical_artifacts("dup");
        // hand-build a manifest with a duplicated key
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&left).unwrap()).unwrap();
        let mut manifest = manifest;
        let duplicate = manifest["records"][0].clone();
        manifest["records"].as_array_mut().unwrap().push(duplicate);
        let broken_path = left.parent().unwrap().join("manifest_broken.json");
        std::fs::write(
            &broken_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let (_, right) = identical_artifacts("dup2");
        let error =
            compare_artifacts(broken_path.to_str().unwrap(), right.to_str().unwrap()).unwrap_err();
        assert!(error.contains("ambiguous record alignment"), "{}", error);
        std::fs::remove_dir_all(left.parent().unwrap()).ok();
        std::fs::remove_dir_all(right.parent().unwrap()).ok();
    }

    #[test]
    fn json_output_is_deterministic() {
        let (left, right) = identical_artifacts("determinism");
        let report = compare_artifacts(left.to_str().unwrap(), right.to_str().unwrap()).unwrap();
        let first = serde_json::to_string_pretty(&report).unwrap();
        let second = serde_json::to_string_pretty(&report).unwrap();
        assert_eq!(first, second);
        // created_at_unix appears only as the documented ignored field
        assert_eq!(first.matches("created_at_unix").count(), 1);
        std::fs::remove_dir_all(left.parent().unwrap()).ok();
        std::fs::remove_dir_all(right.parent().unwrap()).ok();
    }
}
