//! v0.5 bundle assembly: turns per-input run results plus model/plan
//! metadata into the complete deterministic bundle file set (contract
//! section 7).

use crate::v05::bundle::BundleWriter;
use crate::v05::manifest::{
    sha256_hex, BundleIdentity, ManifestExecutionMeta, ManifestExperimentMeta, ManifestGenerated,
    ManifestInputMeta, ManifestModelMeta, ManifestTokenizerMeta, SemanticManifest,
    BUNDLE_SCHEMA_V1,
};
use crate::v05::runner::{CaptureSummary, CapturedTensor, InputResult};
use crate::v05::safetensors::{f32_to_f16_bytes, TensorDType, TensorData};
use crate::v05::spec::ExperimentSpecV1;
use crate::v05::token_select::TokenSelectionRecord;
use crate::v05::verify::{CaptureIndexEntry, SummaryEntry};
use serde::Serialize;
use std::collections::BTreeMap;

/// Model identity for the bundle.
#[derive(Debug, Clone)]
pub struct ModelBundleMeta {
    pub sha256: String,
    pub architecture: String,
    pub layer_count: usize,
    pub embed_dim: usize,
    pub vocab_size: usize,
    pub gguf_metadata: serde_json::Value,
}

/// Tokenizer identity for the bundle.
#[derive(Debug, Clone)]
pub struct TokenizerBundleMeta {
    pub sha256: String,
    pub vocab_size: usize,
}

/// Runtime metrics recorded into runtime.json (excluded from hashes).
#[derive(Debug, Clone, Default)]
pub struct RuntimeMetrics {
    pub wall_clock_ms: f64,
    pub prefill_throughput_tps: Option<f64>,
    pub decode_throughput_tps: Option<f64>,
    pub first_token_latency_ms: Option<f64>,
    pub peak_rss_kb: Option<u64>,
    pub threads: usize,
}

/// Everything the assembler needs.
pub struct BundleMaterials {
    pub spec_text: String,
    pub resolved: ExperimentSpecV1,
    pub ember_version: String,
    pub ember_commit: String,
    pub model_meta: ModelBundleMeta,
    pub tokenizer_meta: TokenizerBundleMeta,
    pub plan: crate::plan::ExecutionPlan,
    pub results: Vec<InputResult>,
    pub warnings: Vec<String>,
    pub runtime: RuntimeMetrics,
}

/// The assembled bundle: file set, semantic manifest, runtime JSON.
pub struct AssembledBundle {
    pub files: BTreeMap<String, Vec<u8>>,
    pub semantic_manifest: SemanticManifest,
    pub runtime_json: serde_json::Value,
}

/// Assemble the deterministic bundle content.
pub fn assemble_bundle(materials: &BundleMaterials) -> Result<AssembledBundle, String> {
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();

    // experiment.toml: verbatim user specification.
    files.insert(
        "experiment.toml".to_string(),
        materials.spec_text.as_bytes().to_vec(),
    );

    // resolved-experiment.json
    files.insert(
        "resolved-experiment.json".to_string(),
        pretty_json(&materials.resolved)?,
    );

    // inputs.jsonl / outputs.jsonl / tokenization.jsonl
    let mut inputs_lines = Vec::new();
    let mut outputs_lines = Vec::new();
    let mut tokenization_lines = Vec::new();
    for result in &materials.results {
        inputs_lines.push(serde_json::json!({
            "id": result.input.id,
            "text": result.input.text,
            "prompt_hash": sha256_hex(result.input.text.as_bytes()),
        }));
        outputs_lines.push(serde_json::json!({
            "input_id": result.input.id,
            "generated_token_ids": result.generated_token_ids,
            "generated_text": result.generated_text,
            "final_top1": result.final_top1.map(|(id, logit)| {
                serde_json::json!({"token_id": id, "logit": logit})
            }),
        }));
        tokenization_lines.push(serde_json::json!({
            "input_id": result.input.id,
            "input_text": result.tokenization.text,
            "normalized_text": result.tokenization.normalized_text,
            "token_ids": result.tokenization.token_ids,
            "pieces": result.tokenization.pieces,
            "byte_offsets": result.tokenization.byte_offsets,
        }));
    }
    files.insert("inputs.jsonl".to_string(), jsonl(&inputs_lines));
    files.insert("outputs.jsonl".to_string(), jsonl(&outputs_lines));
    files.insert("tokenization.jsonl".to_string(), jsonl(&tokenization_lines));

    // captures: safetensors payload + index.jsonl
    let (payload_bytes, index_entries, trace_lines) = build_capture_payload(materials)?;
    if !payload_bytes.is_empty() {
        files.insert("captures/tensors.safetensors".to_string(), payload_bytes);
    } else {
        // An empty bundle still needs the payload file for the layout.
        files.insert("captures/tensors.safetensors".to_string(), Vec::new());
    }
    files.insert(
        "captures/index.jsonl".to_string(),
        jsonl(
            &index_entries
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?,
        ),
    );

    // interventions/events.jsonl
    let mut event_lines = Vec::new();
    for result in &materials.results {
        for event in &result.events {
            event_lines.push(serde_json::json!({
                "intervention_id": event.intervention_id,
                "input_id": event.input_id,
                "site": event.site,
                "layer": event.layer,
                "positions": event.positions,
                "operation": event.operation.kind_name(),
                "source_kind": event.source_kind,
                "snapshot_checksum": event.snapshot_checksum,
                "applied": event.applied,
            }));
        }
    }
    files.insert(
        "interventions/events.jsonl".to_string(),
        jsonl(&event_lines),
    );

    // traces/events.jsonl: capture route records + plan fusion summary.
    let mut trace_lines = trace_lines;
    for layer in &materials.plan.layers {
        trace_lines.push(serde_json::json!({
            "event": "layer-fusion",
            "layer": layer.layer_index,
            "fusion": layer.fusion,
            "fusion_reason": layer.fusion_reason,
        }));
    }
    files.insert("traces/events.jsonl".to_string(), jsonl(&trace_lines));

    // model.json / tokenizer.json / execution-plan.json
    files.insert(
        "model.json".to_string(),
        pretty_json(&serde_json::json!({
            "model_sha256": materials.model_meta.sha256,
            "architecture": materials.model_meta.architecture,
            "layer_count": materials.model_meta.layer_count,
            "embed_dim": materials.model_meta.embed_dim,
            "vocab_size": materials.model_meta.vocab_size,
            "quantization": quantization_summary(&materials.plan),
            "gguf_metadata": materials.model_meta.gguf_metadata,
        }))?,
    );
    files.insert(
        "tokenizer.json".to_string(),
        pretty_json(&serde_json::json!({
            "tokenizer_sha256": materials.tokenizer_meta.sha256,
            "vocab_size": materials.tokenizer_meta.vocab_size,
        }))?,
    );
    // The plan's build time is volatile; plan_hash() already zeroes it,
    // and the stored file must not carry it either (Gate E).
    let mut stored_plan = materials.plan.clone();
    stored_plan.provenance.plan_build_time = "unix-0".to_string();
    files.insert(
        "execution-plan.json".to_string(),
        pretty_json(&stored_plan)?,
    );

    // semantic manifest (payload checksums filled below).
    let mut payloads: BTreeMap<String, String> = BTreeMap::new();
    for (relative, bytes) in &files {
        if relative == "resolved-experiment.json" {
            // placement metadata; excluded from semantic identity
            continue;
        }
        payloads.insert(relative.clone(), sha256_hex(bytes));
    }
    let semantic_manifest = SemanticManifest {
        bundle_schema: BUNDLE_SCHEMA_V1.to_string(),
        experiment_schema: crate::v05::spec::EXPERIMENT_SCHEMA_V1.to_string(),
        hook_schema: crate::v05::hook::HOOK_SCHEMA_VERSION,
        plan_schema: crate::plan::PLAN_SCHEMA_VERSION,
        ember_version: materials.ember_version.clone(),
        ember_commit: materials.ember_commit.clone(),
        experiment: ManifestExperimentMeta {
            name: materials.resolved.experiment.name.clone(),
            description: materials.resolved.experiment.description.clone(),
            seed: materials.resolved.experiment.seed,
        },
        model: ManifestModelMeta {
            sha256: materials.model_meta.sha256.clone(),
            architecture: materials.model_meta.architecture.clone(),
            layer_count: materials.model_meta.layer_count,
            embed_dim: materials.model_meta.embed_dim,
            vocab_size: materials.model_meta.vocab_size,
            quantization: quantization_summary(&materials.plan),
        },
        tokenizer: ManifestTokenizerMeta {
            sha256: materials.tokenizer_meta.sha256.clone(),
            vocab_size: materials.tokenizer_meta.vocab_size,
        },
        execution: ManifestExecutionMeta {
            mode: materials.resolved.execution.mode.name().to_string(),
            deterministic: materials.resolved.execution.deterministic,
            plan_hash: materials.plan.plan_hash.clone(),
        },
        inputs: materials
            .resolved
            .inputs
            .iter()
            .map(|input| ManifestInputMeta {
                id: input.id.clone(),
                prompt_hash: sha256_hex(input.text.as_bytes()),
            })
            .collect(),
        token_selections: collect_selection_records(materials),
        captures: materials.resolved.captures.clone(),
        interventions: materials.resolved.interventions.clone(),
        generated: ManifestGenerated {
            token_ids: materials
                .results
                .iter()
                .map(|result| result.generated_token_ids.clone())
                .collect(),
            texts: materials
                .results
                .iter()
                .map(|result| result.generated_text.clone())
                .collect(),
        },
        payloads,
        warnings: materials.warnings.clone(),
        complete: true,
    };
    let semantic_hash = BundleIdentity::semantic_hash(&semantic_manifest)?;

    // runtime.json (excluded from hashes).
    let runtime_json = serde_json::json!({
        "timestamp": format!("epoch-seconds-{}", crate::extraction::unix_timestamp()),
        "hostname": hostname(),
        "os": std::env::consts::OS,
        "cpu_features": materials.plan.cpu.features,
        "threads": materials.runtime.threads,
        "wall_clock_ms": materials.runtime.wall_clock_ms,
        "prefill_throughput_tps": materials.runtime.prefill_throughput_tps,
        "decode_throughput_tps": materials.runtime.decode_throughput_tps,
        "first_token_latency_ms": materials.runtime.first_token_latency_ms,
        "peak_rss_kb": materials.runtime.peak_rss_kb,
        "scratch_bytes": materials.plan.scratch.total_bytes,
        "compiler_version": materials.plan.provenance.rustc_version,
        "process_id": std::process::id(),
        "model_path": materials.resolved.model.path.display().to_string(),
        "tokenizer_path": materials
            .resolved
            .model
            .tokenizer
            .as_ref()
            .map(|path| path.display().to_string()),
    });
    let _ = semantic_hash; // stored in manifest.json by the writer

    Ok(AssembledBundle {
        files,
        semantic_manifest,
        runtime_json,
    })
}

/// Write the assembled bundle atomically and return the identity.
pub fn write_bundle(
    materials: &BundleMaterials,
    retain_incomplete: bool,
) -> Result<(std::path::PathBuf, BundleIdentity), String> {
    let assembled = assemble_bundle(materials)?;
    let mut writer = BundleWriter::new(
        materials.resolved.output.directory.clone(),
        materials.resolved.output.overwrite,
        retain_incomplete,
    );
    for (relative, bytes) in &assembled.files {
        writer.add(relative, bytes.clone());
    }
    writer.finalize(assembled.semantic_manifest, assembled.runtime_json)
}

/// Assembled capture payload: file bytes, index entries, trace lines.
type CapturePayload = (Vec<u8>, Vec<CaptureIndexEntry>, Vec<serde_json::Value>);

fn build_capture_payload(materials: &BundleMaterials) -> Result<CapturePayload, String> {
    // Owned payload buffers (name, bytes, shape, dtype); the safetensors
    // writer borrows them only inside serialize().
    let mut owned_payloads: Vec<(String, Vec<u8>, Vec<usize>, TensorDType)> = Vec::new();
    let mut index_entries: Vec<CaptureIndexEntry> = Vec::new();
    let mut trace_lines: Vec<serde_json::Value> = Vec::new();

    for result in &materials.results {
        for capture in &result.captures {
            let name = tensor_name(capture);
            let (dtype, bytes) = match capture.dtype {
                crate::v05::capture::CaptureDType::F32 => {
                    let bytes: Vec<u8> = capture
                        .rows
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect();
                    (TensorDType::F32, bytes)
                }
                crate::v05::capture::CaptureDType::F16 => {
                    (TensorDType::F16, f32_to_f16_bytes(&capture.rows))
                }
            };
            let shape = vec![capture.positions.len(), capture.columns];
            let (route, fusion) = hook_route(materials, capture);
            let provenance = selection_provenance(materials, result, &capture.capture_id);
            owned_payloads.push((name.clone(), bytes.clone(), shape.clone(), dtype));
            index_entries.push(CaptureIndexEntry {
                capture_id: capture.capture_id.clone(),
                input_id: capture.input_id.clone(),
                site: capture.site,
                layer: capture.layer,
                positions: capture.positions.clone(),
                tensor_name: name,
                shape,
                dtype: dtype.name().to_string(),
                byte_length: bytes.len(),
                checksum: sha256_hex(&bytes),
                model_sha256: materials.model_meta.sha256.clone(),
                plan_hash: materials.plan.plan_hash.clone(),
                hook_route: route.clone(),
                fusion: fusion.clone(),
                selection_provenance: provenance,
                summary: None,
            });
            trace_lines.push(serde_json::json!({
                "event": "capture",
                "capture_id": capture.capture_id,
                "input_id": capture.input_id,
                "site": capture.site,
                "layer": capture.layer,
                "positions": capture.positions,
                "tensor_name": capture_id_leaf(&capture.capture_id),
                "route": route,
                "fusion": fusion,
                "full_tensor": capture.full_tensor,
            }));
        }
        for summary in &result.summaries {
            let (route, fusion) = hook_route(materials, &summary_capture_view(summary));
            index_entries.push(CaptureIndexEntry {
                capture_id: summary.capture_id.clone(),
                input_id: summary.input_id.clone(),
                site: summary.site,
                layer: summary.layer,
                positions: summary.positions.clone(),
                tensor_name: String::new(),
                shape: summary.shape.to_vec(),
                dtype: "F32".to_string(),
                byte_length: 0,
                checksum: String::new(),
                model_sha256: materials.model_meta.sha256.clone(),
                plan_hash: materials.plan.plan_hash.clone(),
                hook_route: route,
                fusion,
                selection_provenance: serde_json::Value::Null,
                summary: Some(SummaryEntry {
                    shape: summary.shape.to_vec(),
                    finite_count: summary.finite_count,
                    minimum: summary.minimum,
                    maximum: summary.maximum,
                    mean: summary.mean,
                    l2_norm: summary.l2_norm,
                }),
            });
        }
    }
    index_entries.sort_by(|a, b| {
        a.capture_id
            .cmp(&b.capture_id)
            .then(a.input_id.cmp(&b.input_id))
            .then(a.layer.cmp(&b.layer))
    });
    let mut payload_tensors: Vec<TensorData<'_>> = Vec::with_capacity(owned_payloads.len());
    for (name, bytes, shape, dtype) in &owned_payloads {
        payload_tensors.push(TensorData {
            name: name.as_str(),
            dtype: *dtype,
            shape,
            bytes,
        });
    }
    let payload = crate::v05::safetensors::serialize(&payload_tensors)?;
    Ok((payload, index_entries, trace_lines))
}

fn tensor_name(capture: &CapturedTensor) -> String {
    format!(
        "{}/{}/{}/{}",
        capture.capture_id, capture.input_id, capture.site, capture.layer
    )
}

fn capture_id_leaf(id: &str) -> String {
    id.replace(['/', '\\'], "_")
}

fn hook_route(materials: &BundleMaterials, capture: &CapturedTensor) -> (String, String) {
    let site = capture.site;
    let layer = capture.layer;
    let plan = &materials.plan;
    let fusion = plan
        .layers
        .get(layer)
        .map(|layer_plan| match layer_plan.fusion {
            crate::plan::FusionState::Fused => "fused".to_string(),
            crate::plan::FusionState::PartiallyFused => "partially-fused".to_string(),
            crate::plan::FusionState::Unfused => "unfused".to_string(),
        })
        .unwrap_or_else(|| "none".to_string());
    let route = if site.is_per_layer() {
        plan.hook_sites
            .sites
            .iter()
            .find(|site_record| {
                site_record.stage == site.stage_id() && site_record.layer == Some(layer)
            })
            .map(|site_record| site_record.route.clone())
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        "unfused".to_string()
    };
    (route, fusion)
}

/// A fake capture view for summary hook-route lookup.
fn summary_capture_view(summary: &CaptureSummary) -> CapturedTensor {
    CapturedTensor {
        capture_id: summary.capture_id.clone(),
        input_id: summary.input_id.clone(),
        site: summary.site,
        layer: summary.layer,
        positions: summary.positions.clone(),
        rows: Vec::new(),
        columns: summary.shape[1],
        full_tensor: false,
        bytes: 0,
        dtype: crate::v05::capture::CaptureDType::F32,
    }
}

fn selection_provenance(
    materials: &BundleMaterials,
    result: &InputResult,
    capture_id: &str,
) -> serde_json::Value {
    let capture = materials
        .resolved
        .captures
        .iter()
        .find(|capture| capture.id == capture_id);
    let Some(capture) = capture else {
        return serde_json::Value::Null;
    };
    let matching: Vec<&TokenSelectionRecord> = result
        .selection_records
        .iter()
        .filter(|record| record.rule == capture.tokens.rule_id())
        .collect();
    match matching.first() {
        Some(record) => serde_json::to_value(record).unwrap_or(serde_json::Value::Null),
        None => serde_json::Value::Null,
    }
}

fn collect_selection_records(materials: &BundleMaterials) -> Vec<TokenSelectionRecord> {
    let mut records = Vec::new();
    for result in &materials.results {
        records.extend(result.selection_records.iter().cloned());
    }
    records
}

fn quantization_summary(plan: &crate::plan::ExecutionPlan) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for tensor in &plan.tensor_table {
        *counts.entry(tensor.gguf_dtype.as_str()).or_insert(0) += 1;
    }
    if counts.is_empty() {
        return "unknown".to_string();
    }
    let mut parts: Vec<String> = counts
        .iter()
        .map(|(dtype, count)| format!("{dtype}:{count}"))
        .collect();
    parts.sort();
    parts.join(",")
}

fn pretty_json<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec_pretty(value).map_err(|error| format!("JSON serialization failed: {error}"))
}

fn jsonl(values: &[serde_json::Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for value in values {
        if let Ok(bytes) = serde_json::to_vec(value) {
            out.extend_from_slice(&bytes);
            out.push(b'\n');
        }
    }
    out
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".to_string())
}
