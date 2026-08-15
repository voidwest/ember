//! v0.5 run comparison (contract section 9, Gate G).
//!
//! Comparison is scientific-first: identity, outputs, captures,
//! interventions, then runtime. Tensor metrics are computed per matching
//! capture without loading every tensor simultaneously (one pair at a
//! time).

use crate::v05::hook::SemanticHookSite;
use crate::v05::verify::{load_bundle_for_source, CaptureIndexEntry};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Tensor comparison metrics for one matching capture pair.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TensorMetrics {
    pub shape_equal: bool,
    pub dtype_equal: bool,
    /// Exact equality (bit-identical payloads).
    pub exact: bool,
    pub maximum_absolute_difference: Option<f64>,
    pub mean_absolute_difference: Option<f64>,
    pub relative_l2_difference: Option<f64>,
    pub cosine_similarity: Option<f64>,
    pub finite_value_mismatches: Option<usize>,
}

/// One capture comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureComparison {
    pub capture_id: String,
    pub input_id: String,
    pub site: SemanticHookSite,
    pub layer: usize,
    pub present_in_a: bool,
    pub present_in_b: bool,
    pub metrics: Option<TensorMetrics>,
}

/// Output comparison for one input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputComparison {
    pub input_id: String,
    pub generated_tokens_equal: bool,
    pub generated_text_equal: bool,
    pub final_top1_equal: bool,
    /// First token position (1-based decode step) where token ids differ.
    pub first_divergence_step: Option<usize>,
    pub generated_count_a: usize,
    pub generated_count_b: usize,
}

/// Intervention comparison for one intervention id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionComparison {
    pub intervention_id: String,
    pub operation_equal: bool,
    pub source_equal: bool,
    pub site_equal: bool,
    pub layer_equal: bool,
    pub selected_tokens_equal: bool,
    pub defusion_route_equal: bool,
    pub events_in_a: usize,
    pub events_in_b: usize,
}

/// Identity comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityComparison {
    pub schema_compatible: bool,
    pub semantic_hash_equal: bool,
    pub model_hash_equal: bool,
    pub tokenizer_hash_equal: bool,
    pub execution_mode_equal: bool,
    pub plan_hash_equal: bool,
    pub input_ids_equal: bool,
    pub prompts_equal: bool,
    pub tokenization_equal: bool,
}

/// Runtime comparison (separate from semantic differences).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeComparison {
    pub decode_throughput_tps_a: Option<f64>,
    pub decode_throughput_tps_b: Option<f64>,
    pub prefill_throughput_tps_a: Option<f64>,
    pub prefill_throughput_tps_b: Option<f64>,
    pub first_token_latency_ms_a: Option<f64>,
    pub first_token_latency_ms_b: Option<f64>,
    pub peak_rss_kb_a: Option<u64>,
    pub peak_rss_kb_b: Option<u64>,
    pub scratch_bytes_a: Option<u64>,
    pub scratch_bytes_b: Option<u64>,
}

/// The complete comparison result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareResult {
    pub bundle_a: String,
    pub bundle_b: String,
    pub identity: IdentityComparison,
    pub outputs: Vec<OutputComparison>,
    pub captures: Vec<CaptureComparison>,
    pub interventions: Vec<InterventionComparison>,
    pub runtime: RuntimeComparison,
}

/// Metrics for a runtime.json file (all fields optional).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeJson {
    pub decode_throughput_tps: Option<f64>,
    pub prefill_throughput_tps: Option<f64>,
    pub first_token_latency_ms: Option<f64>,
    pub peak_rss_kb: Option<u64>,
    pub scratch_bytes: Option<u64>,
}

impl RuntimeJson {
    fn read(root: &Path) -> RuntimeJson {
        let path = root.join("runtime.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            return RuntimeJson::default();
        };
        serde_json::from_str(&text).unwrap_or_default()
    }
}

/// Compare two verified bundles.
pub fn compare_bundles(a: &Path, b: &Path) -> Result<CompareResult, String> {
    let bundle_a = load_bundle_for_source(a)?;
    let bundle_b = load_bundle_for_source(b)?;
    let sa = &bundle_a.semantic_manifest;
    let sb = &bundle_b.semantic_manifest;

    // ---- identity ----
    let inputs_a: Vec<&str> = sa.inputs.iter().map(|input| input.id.as_str()).collect();
    let inputs_b: Vec<&str> = sb.inputs.iter().map(|input| input.id.as_str()).collect();
    let identity = IdentityComparison {
        schema_compatible: sa.bundle_schema == sb.bundle_schema
            && sa.experiment_schema == sb.experiment_schema
            && sa.hook_schema == sb.hook_schema,
        semantic_hash_equal: bundle_identity_hash(a)? == bundle_identity_hash(b)?,
        model_hash_equal: sa.model.sha256 == sb.model.sha256,
        tokenizer_hash_equal: sa.tokenizer.sha256 == sb.tokenizer.sha256,
        execution_mode_equal: sa.execution.mode == sb.execution.mode,
        plan_hash_equal: sa.execution.plan_hash == sb.execution.plan_hash,
        input_ids_equal: inputs_a == inputs_b,
        prompts_equal: read_prompt_hashes(a)? == read_prompt_hashes(b)?,
        tokenization_equal: read_tokenization_ids(a)? == read_tokenization_ids(b)?,
    };

    // ---- outputs ----
    let outputs_a = read_outputs(a)?;
    let outputs_b = read_outputs(b)?;
    let mut outputs = Vec::new();
    for (input_a, input_b) in outputs_a.iter().zip(outputs_b.iter()) {
        let tokens_equal = input_a.generated_token_ids == input_b.generated_token_ids;
        let first_divergence = if tokens_equal {
            None
        } else {
            input_a
                .generated_token_ids
                .iter()
                .zip(input_b.generated_token_ids.iter())
                .position(|(x, y)| x != y)
                .map(|index| index + 1)
                .or(Some(
                    input_a
                        .generated_token_ids
                        .len()
                        .min(input_b.generated_token_ids.len())
                        + 1,
                ))
        };
        outputs.push(OutputComparison {
            input_id: input_a.input_id.clone(),
            generated_tokens_equal: tokens_equal,
            generated_text_equal: input_a.generated_text == input_b.generated_text,
            final_top1_equal: input_a.final_top1 == input_b.final_top1,
            first_divergence_step: first_divergence,
            generated_count_a: input_a.generated_token_ids.len(),
            generated_count_b: input_b.generated_token_ids.len(),
        });
    }

    // ---- captures ----
    let captures_a = &bundle_a.capture_index;
    let captures_b = &bundle_b.capture_index;
    let mut captures: Vec<CaptureComparison> = Vec::new();
    let mut seen: std::collections::BTreeSet<(String, String, SemanticHookSite, usize)> =
        std::collections::BTreeSet::new();
    for entry_a in captures_a {
        let key = (
            entry_a.capture_id.clone(),
            entry_a.input_id.clone(),
            entry_a.site,
            entry_a.layer,
        );
        if !seen.insert(key.clone()) {
            continue;
        }
        let entry_b = captures_b.iter().find(|entry_b| {
            entry_b.capture_id == entry_a.capture_id
                && entry_b.input_id == entry_a.input_id
                && entry_b.site == entry_a.site
                && entry_b.layer == entry_a.layer
        });
        let metrics = match entry_b {
            Some(entry_b) => Some(tensor_metrics(&bundle_a, &bundle_b, entry_a, entry_b)?),
            None => None,
        };
        captures.push(CaptureComparison {
            capture_id: entry_a.capture_id.clone(),
            input_id: entry_a.input_id.clone(),
            site: entry_a.site,
            layer: entry_a.layer,
            present_in_a: true,
            present_in_b: entry_b.is_some(),
            metrics,
        });
    }
    // captures present only in b
    for entry_b in captures_b {
        let key = (
            entry_b.capture_id.clone(),
            entry_b.input_id.clone(),
            entry_b.site,
            entry_b.layer,
        );
        if seen.contains(&key) {
            continue;
        }
        captures.push(CaptureComparison {
            capture_id: entry_b.capture_id.clone(),
            input_id: entry_b.input_id.clone(),
            site: entry_b.site,
            layer: entry_b.layer,
            present_in_a: false,
            present_in_b: true,
            metrics: None,
        });
    }

    // ---- interventions ----
    let events_a = read_intervention_events(a)?;
    let events_b = read_intervention_events(b)?;
    let mut interventions: Vec<InterventionComparison> = Vec::new();
    let mut seen_interventions: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
    for event_a in &events_a {
        if !seen_interventions.insert(event_a.intervention_id.clone()) {
            continue;
        }
        let event_b = events_b
            .iter()
            .find(|event_b| event_b.intervention_id == event_a.intervention_id);
        let operation_equal = event_b
            .map(|event_b| event_b.operation == event_a.operation)
            .unwrap_or(false);
        let selected_equal = event_b
            .map(|event_b| {
                event_b.positions == event_a.positions
                    && event_b.layer == event_a.layer
                    && event_b.site == event_a.site
            })
            .unwrap_or(false);
        interventions.push(InterventionComparison {
            intervention_id: event_a.intervention_id.clone(),
            operation_equal,
            source_equal: event_b
                .map(|event_b| event_b.source_kind == event_a.source_kind)
                .unwrap_or(false),
            site_equal: event_b
                .map(|event_b| event_b.site == event_a.site)
                .unwrap_or(false),
            layer_equal: event_b
                .map(|event_b| event_b.layer == event_a.layer)
                .unwrap_or(false),
            selected_tokens_equal: selected_equal,
            defusion_route_equal: plan_fusion_summary(a)? == plan_fusion_summary(b)?,
            events_in_a: events_a
                .iter()
                .filter(|event| event.intervention_id == event_a.intervention_id)
                .count(),
            events_in_b: events_b
                .iter()
                .filter(|event| event.intervention_id == event_a.intervention_id)
                .count(),
        });
    }
    for event_b in &events_b {
        if !seen_interventions.contains(&event_b.intervention_id) {
            interventions.push(InterventionComparison {
                intervention_id: event_b.intervention_id.clone(),
                operation_equal: false,
                source_equal: false,
                site_equal: false,
                layer_equal: false,
                selected_tokens_equal: false,
                defusion_route_equal: plan_fusion_summary(a)? == plan_fusion_summary(b)?,
                events_in_a: 0,
                events_in_b: events_b
                    .iter()
                    .filter(|event| event.intervention_id == event_b.intervention_id)
                    .count(),
            });
        }
    }

    // ---- runtime ----
    let runtime_a = RuntimeJson::read(a);
    let runtime_b = RuntimeJson::read(b);
    let runtime = RuntimeComparison {
        decode_throughput_tps_a: runtime_a.decode_throughput_tps,
        decode_throughput_tps_b: runtime_b.decode_throughput_tps,
        prefill_throughput_tps_a: runtime_a.prefill_throughput_tps,
        prefill_throughput_tps_b: runtime_b.prefill_throughput_tps,
        first_token_latency_ms_a: runtime_a.first_token_latency_ms,
        first_token_latency_ms_b: runtime_b.first_token_latency_ms,
        peak_rss_kb_a: runtime_a.peak_rss_kb,
        peak_rss_kb_b: runtime_b.peak_rss_kb,
        scratch_bytes_a: runtime_a.scratch_bytes,
        scratch_bytes_b: runtime_b.scratch_bytes,
    };

    Ok(CompareResult {
        bundle_a: a.display().to_string(),
        bundle_b: b.display().to_string(),
        identity,
        outputs,
        captures,
        interventions,
        runtime,
    })
}

fn tensor_metrics(
    bundle_a: &crate::v05::verify::LoadedBundle,
    bundle_b: &crate::v05::verify::LoadedBundle,
    entry_a: &CaptureIndexEntry,
    entry_b: &CaptureIndexEntry,
) -> Result<TensorMetrics, String> {
    let shape_equal = entry_a.shape == entry_b.shape;
    let dtype_equal = entry_a.dtype == entry_b.dtype;
    if !shape_equal || !dtype_equal {
        return Ok(TensorMetrics {
            shape_equal,
            dtype_equal,
            exact: false,
            ..TensorMetrics::default()
        });
    }
    let values_a = bundle_a.tensor_f32_by_name(&entry_a.tensor_name)?;
    let values_b = bundle_b.tensor_f32_by_name(&entry_b.tensor_name)?;
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq_a = 0.0f64;
    let mut sum_sq_b = 0.0f64;
    let mut sum_prod = 0.0f64;
    let mut finite_mismatches = 0usize;
    let mut exact = values_a.len() == values_b.len();
    for (x, y) in values_a.iter().zip(values_b.iter()) {
        let dx = (*x as f64 - *y as f64).abs();
        max_abs = max_abs.max(dx);
        sum_abs += dx;
        sum_sq_a += (*x as f64) * (*x as f64);
        sum_sq_b += (*y as f64) * (*y as f64);
        sum_prod += (*x as f64) * (*y as f64);
        if x.to_bits() != y.to_bits() {
            exact = false;
        }
        if x.is_finite() != y.is_finite() {
            finite_mismatches += 1;
        }
    }
    let count = values_a.len().min(values_b.len()) as f64;
    let relative_l2: Option<f64> = if sum_sq_a > 0.0 {
        let diff_sq = sum_sq_a + sum_sq_b - 2.0 * sum_prod;
        Some((diff_sq / sum_sq_a).sqrt())
    } else {
        None
    };
    let cosine = if sum_sq_a > 0.0 && sum_sq_b > 0.0 {
        Some(sum_prod / (sum_sq_a.sqrt() * sum_sq_b.sqrt()))
    } else {
        None
    };
    Ok(TensorMetrics {
        shape_equal: true,
        dtype_equal: true,
        exact,
        maximum_absolute_difference: Some(max_abs),
        mean_absolute_difference: Some(if count > 0.0 { sum_abs / count } else { 0.0 }),
        relative_l2_difference: relative_l2,
        cosine_similarity: cosine,
        finite_value_mismatches: Some(finite_mismatches),
    })
}

fn bundle_identity_hash(root: &Path) -> Result<String, String> {
    let path = root.join("manifest.json");
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    let manifest: crate::v05::manifest::BundleManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("manifest.json is invalid: {error}"))?;
    Ok(manifest.semantic_hash)
}

fn read_prompt_hashes(root: &Path) -> Result<Vec<String>, String> {
    let path = root.join("inputs.jsonl");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    let mut hashes = Vec::new();
    for line in text.lines() {
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("inputs.jsonl line is invalid: {error}"))?;
        let hash = value
            .get("prompt_hash")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "inputs.jsonl line lacks prompt_hash".to_string())?;
        hashes.push(hash.to_string());
    }
    Ok(hashes)
}

fn read_tokenization_ids(root: &Path) -> Result<Vec<Vec<u32>>, String> {
    let path = root.join("tokenization.jsonl");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    let mut ids = Vec::new();
    for line in text.lines() {
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("tokenization.jsonl line is invalid: {error}"))?;
        let token_ids: Vec<u32> = value
            .get("token_ids")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "tokenization.jsonl line lacks token_ids".to_string())?
            .iter()
            .map(|id| id.as_u64().unwrap_or(0) as u32)
            .collect();
        ids.push(token_ids);
    }
    Ok(ids)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct OutputLine {
    input_id: String,
    generated_token_ids: Vec<u32>,
    generated_text: String,
    final_top1: Option<FinalTop1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct FinalTop1 {
    token_id: u32,
    logit: f32,
}

fn read_outputs(root: &Path) -> Result<Vec<OutputLine>, String> {
    let path = root.join("outputs.jsonl");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    let mut outputs = Vec::new();
    for line in text.lines() {
        let output: OutputLine = serde_json::from_str(line)
            .map_err(|error| format!("outputs.jsonl line is invalid: {error}"))?;
        outputs.push(output);
    }
    Ok(outputs)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct InterventionEventLine {
    intervention_id: String,
    input_id: String,
    site: SemanticHookSite,
    layer: usize,
    positions: Vec<usize>,
    operation: String,
    source_kind: Option<String>,
}

fn read_intervention_events(root: &Path) -> Result<Vec<InterventionEventLine>, String> {
    let path = root.join("interventions/events.jsonl");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    let mut events = Vec::new();
    for line in text.lines() {
        let event: InterventionEventLine = serde_json::from_str(line)
            .map_err(|error| format!("interventions/events.jsonl line is invalid: {error}"))?;
        events.push(event);
    }
    Ok(events)
}

/// Fusion/de-fusion summary of a bundle's plan (defusion route equality).
fn plan_fusion_summary(root: &Path) -> Result<Vec<String>, String> {
    let path = root.join("execution-plan.json");
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    let plan: crate::plan::ExecutionPlan = serde_json::from_slice(&bytes)
        .map_err(|error| format!("execution-plan.json is invalid: {error}"))?;
    Ok(plan
        .layers
        .iter()
        .map(|layer| {
            let state = match layer.fusion {
                crate::plan::FusionState::Fused => "fused",
                crate::plan::FusionState::PartiallyFused => "partially-fused",
                crate::plan::FusionState::Unfused => "unfused",
            };
            format!("{}:{state}", layer.layer_index)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v05::testutil;
    use crate::v05::testutil::temp_root;

    #[test]
    fn identical_bundles_report_semantic_identity() {
        let root_a = temp_root("a");
        let root_b = temp_root("b");
        testutil::write_test_bundle(
            &root_a,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        testutil::write_test_bundle(
            &root_b,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        let result = compare_bundles(&root_a, &root_b).unwrap();
        assert!(result.identity.semantic_hash_equal);
        assert!(result.identity.schema_compatible);
        assert!(result.identity.tokenization_equal);
        assert!(result.outputs[0].generated_tokens_equal);
        let capture = &result.captures[0];
        assert_eq!(capture.capture_id, "cap-1");
        let metrics = capture.metrics.as_ref().unwrap();
        assert!(metrics.exact);
        assert_eq!(metrics.maximum_absolute_difference, Some(0.0));
        assert_eq!(metrics.cosine_similarity, Some(1.0));
        assert!(result.interventions.is_empty());
        let _ = std::fs::remove_dir_all(&root_a);
        let _ = std::fs::remove_dir_all(&root_b);
    }

    #[test]
    fn perturbed_payload_produces_correct_metrics() {
        let root_a = temp_root("a");
        let root_b = temp_root("b");
        testutil::write_test_bundle(
            &root_a,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        let mut perturbed = testutil::sample_rows();
        perturbed[0] += 0.5;
        testutil::write_test_bundle(&root_b, &perturbed, &testutil::sample_positions());
        let result = compare_bundles(&root_a, &root_b).unwrap();
        assert!(!result.identity.semantic_hash_equal);
        let metrics = result.captures[0].metrics.as_ref().unwrap();
        assert!(!metrics.exact);
        assert_eq!(metrics.maximum_absolute_difference, Some(0.5));
        assert_eq!(metrics.mean_absolute_difference, Some(0.125));
        let cosine = metrics.cosine_similarity.unwrap();
        assert!(cosine > 0.99 && cosine < 1.0);
        let _ = std::fs::remove_dir_all(&root_a);
        let _ = std::fs::remove_dir_all(&root_b);
    }

    #[test]
    fn json_output_is_deterministic() {
        let root_a = temp_root("a");
        let root_b = temp_root("b");
        testutil::write_test_bundle(
            &root_a,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        testutil::write_test_bundle(
            &root_b,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        let first = serde_json::to_vec(&compare_bundles(&root_a, &root_b).unwrap()).unwrap();
        let second = serde_json::to_vec(&compare_bundles(&root_a, &root_b).unwrap()).unwrap();
        assert_eq!(first, second);
        let _ = std::fs::remove_dir_all(&root_a);
        let _ = std::fs::remove_dir_all(&root_b);
    }
}
