//! Shared v0.5 test fixtures (test builds only).
//!
//! `write_test_bundle` produces a complete, verifiable `ember.bundle.v1`
//! bundle with one selected-row capture, using the real writer, the real
//! safetensors codec, and a reduced real execution plan fixture.

#![cfg(test)]

use crate::v05::bundle::BundleWriter;
use crate::v05::capture::{CaptureSpec, CaptureStorage, InputSelector, LayerSelector};
use crate::v05::hook::SemanticHookSite;
use crate::v05::manifest::{
    sha256_hex, BundleIdentity, ManifestExecutionMeta, ManifestExperimentMeta, ManifestGenerated,
    ManifestInputMeta, ManifestModelMeta, ManifestTokenizerMeta, SemanticManifest,
    BUNDLE_SCHEMA_V1,
};
use crate::v05::safetensors::{TensorDType, TensorData};
use crate::v05::spec::EXPERIMENT_SCHEMA_V1;
use crate::v05::token_select::{
    CoverageKind, RoundTripStatus, SubtokenSelection, TextNormalization, TokenSelectionRecord,
    TokenSelector,
};
use crate::v05::verify::{CaptureIndexEntry, SummaryEntry};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

/// The fixture model SHA recorded in the test plan.
pub const FIXTURE_MODEL_SHA: &str =
    "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3";
pub const FIXTURE_TOKENIZER_SHA: &str =
    "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b";
const COLUMNS: usize = 4;

/// Mirror of `ExecutionPlan::plan_hash` over the fixture JSON: canonical
/// JSON with `plan_hash` and `provenance.plan_build_time` removed.
pub fn fixture_plan_hash(plan_json: &mut serde_json::Value) -> String {
    if let Some(object) = plan_json.as_object_mut() {
        object.insert("plan_hash".into(), json!(""));
        if let Some(provenance) = object.get_mut("provenance").and_then(|p| p.as_object_mut()) {
            provenance.insert("plan_build_time".into(), json!(""));
        }
    }
    let bytes = serde_json::to_vec(plan_json).expect("fixture plan serializes");
    let hash = sha256_hex(&bytes);
    // write the computed hash back so the stored plan is self-consistent
    if let Some(object) = plan_json.as_object_mut() {
        object.insert("plan_hash".into(), json!(hash));
    }
    hash
}

/// A token-selection record consistent with a one-token input.
pub fn sample_selection_record(selector: TokenSelector) -> TokenSelectionRecord {
    TokenSelectionRecord {
        selector,
        rule: "prompt-final".to_string(),
        input_text: "x".to_string(),
        normalized_text: "x".to_string(),
        token_ids: vec![1],
        pieces: vec!["x".to_string()],
        byte_offsets: vec![(0, 1)],
        matched_byte_span: None,
        selected_indices: vec![0],
        coverage: CoverageKind::Exact,
        boundary_expansion: None,
        ambiguity: crate::v05::token_select::AmbiguityStatus::Resolved,
        round_trip: RoundTripStatus::NotApplicable,
        note: None,
    }
}

/// Write a complete, verifiable bundle with one capture whose rows are
/// `rows` at `positions`. Returns the bundle root.
pub fn write_test_bundle(root: &Path, rows: &[f32], positions: &[usize]) -> std::path::PathBuf {
    let plan_text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/execution-plan-v1.json"
    ))
    .expect("fixture plan present");
    let mut plan_json: serde_json::Value = serde_json::from_str(&plan_text).unwrap();
    let plan_hash = fixture_plan_hash(&mut plan_json);
    let plan_json = serde_json::to_vec_pretty(&plan_json).unwrap();

    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    files.insert(
        "experiment.toml".to_string(),
        br#"schema = "ember.experiment.v1"

[experiment]
name = "fixture"

[model]
path = "model.gguf"

[[inputs]]
id = "i1"
text = "x"

[output]
directory = "out"
"#
        .to_vec(),
    );
    files.insert(
        "resolved-experiment.json".to_string(),
        br#"{"schema":"ember.experiment.v1","experiment":{"name":"fixture","description":"","seed":0},"model":{"path":"model.gguf","expected_sha256":"","tokenizer":null,"tokenizer_expected_sha256":"","arch":"auto"},"execution":{"mode":"reference","threads":0,"deterministic":true},"generation":{"max_new_tokens":0,"temperature":0.0},"inputs":[{"id":"i1","text":"x"}],"captures":[],"interventions":[],"output":{"directory":"out","tensor_format":"safetensors","overwrite":false},"defaults":[]}"#
            .to_vec(),
    );
    files.insert(
        "inputs.jsonl".to_string(),
        br#"{"id":"i1","text":"x","prompt_hash":"2d711642b72679dc010bb2df4b76c638b3dfe1319e3be3e4a65b6d89d2d3c1e6"}
"#
        .to_vec(),
    );
    files.insert(
        "outputs.jsonl".to_string(),
        br#"{"input_id":"i1","generated_token_ids":[1],"generated_text":"x","final_top1":{"token_id":1,"logit":0.0}}
"#
        .to_vec(),
    );
    files.insert(
        "tokenization.jsonl".to_string(),
        br#"{"input_id":"i1","input_text":"x","normalized_text":"x","token_ids":[1],"pieces":["x"],"byte_offsets":[[0,1]]}
"#
        .to_vec(),
    );
    files.insert("interventions/events.jsonl".to_string(), Vec::new());
    files.insert("traces/events.jsonl".to_string(), Vec::new());

    // payload + index
    let tensor_name = "cap-1/i1/residual-post-mlp/0";
    let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shape = vec![positions.len(), COLUMNS];
    let payload = crate::v05::safetensors::serialize(&[TensorData {
        name: tensor_name,
        dtype: TensorDType::F32,
        shape: &shape,
        bytes: &bytes,
    }])
    .unwrap();
    files.insert("captures/tensors.safetensors".to_string(), payload.clone());
    let index_entry = CaptureIndexEntry {
        capture_id: "cap-1".to_string(),
        input_id: "i1".to_string(),
        site: SemanticHookSite::ResidualPostMlp,
        layer: 0,
        positions: positions.to_vec(),
        tensor_name: tensor_name.to_string(),
        shape: shape.clone(),
        dtype: "F32".to_string(),
        byte_length: bytes.len(),
        checksum: sha256_hex(&bytes),
        model_sha256: FIXTURE_MODEL_SHA.to_string(),
        plan_hash: plan_hash.clone(),
        hook_route: "unfused".to_string(),
        fusion: "unfused".to_string(),
        selection_provenance: serde_json::to_value(sample_selection_record(
            TokenSelector::PromptFinal,
        ))
        .unwrap(),
        summary: None,
    };
    files.insert(
        "captures/index.jsonl".to_string(),
        format!("{}\n", serde_json::to_string(&index_entry).unwrap()).into_bytes(),
    );

    files.insert(
        "model.json".to_string(),
        br#"{"model_sha256":"432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3","architecture":"llama","layer_count":1,"embed_dim":4,"vocab_size":16,"quantization":"q8_0:1","gguf_metadata":{}}"#
            .to_vec(),
    );
    files.insert(
        "tokenizer.json".to_string(),
        br#"{"tokenizer_sha256":"6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b","vocab_size":16}"#
            .to_vec(),
    );
    files.insert("execution-plan.json".to_string(), plan_json);

    let mut payloads: BTreeMap<String, String> = BTreeMap::new();
    for (relative, bytes) in &files {
        if relative == "resolved-experiment.json" {
            continue;
        }
        payloads.insert(relative.clone(), sha256_hex(bytes));
    }
    let semantic_manifest = SemanticManifest {
        bundle_schema: BUNDLE_SCHEMA_V1.to_string(),
        experiment_schema: EXPERIMENT_SCHEMA_V1.to_string(),
        hook_schema: 1,
        plan_schema: 1,
        ember_version: "0.5.0-test".to_string(),
        ember_commit: "test".to_string(),
        experiment: ManifestExperimentMeta {
            name: "fixture".to_string(),
            description: String::new(),
            seed: 0,
        },
        model: ManifestModelMeta {
            sha256: FIXTURE_MODEL_SHA.to_string(),
            architecture: "llama".to_string(),
            layer_count: 1,
            embed_dim: COLUMNS,
            vocab_size: 16,
            quantization: "q8_0:1".to_string(),
        },
        tokenizer: ManifestTokenizerMeta {
            sha256: FIXTURE_TOKENIZER_SHA.to_string(),
            vocab_size: 16,
        },
        execution: ManifestExecutionMeta {
            mode: "reference".to_string(),
            deterministic: true,
            plan_hash: plan_hash.clone(),
        },
        inputs: vec![ManifestInputMeta {
            id: "i1".to_string(),
            prompt_hash: "2d711642b72679dc010bb2df4b76c638b3dfe1319e3be3e4a65b6d89d2d3c1e6"
                .to_string(),
        }],
        token_selections: vec![sample_selection_record(TokenSelector::PromptFinal)],
        captures: vec![CaptureSpec {
            id: "cap-1".to_string(),
            site: SemanticHookSite::ResidualPostMlp,
            layers: LayerSelector::All("all".to_string()),
            tokens: TokenSelector::PromptFinal,
            inputs: InputSelector::All("all".to_string()),
            storage: CaptureStorage::SelectedRows,
            dtype: crate::v05::capture::CaptureDType::F32,
        }],
        interventions: Vec::new(),
        generated: ManifestGenerated {
            token_ids: vec![vec![1]],
            texts: vec!["x".to_string()],
        },
        payloads,
        warnings: Vec::new(),
        complete: true,
    };
    let mut writer = BundleWriter::new(root.to_path_buf(), true, false);
    for (relative, bytes) in files {
        writer.add(&relative, bytes);
    }
    let (_, _identity) = writer
        .finalize(semantic_manifest, json!({"hostname": "test"}))
        .expect("fixture bundle writes");
    root.to_path_buf()
}

/// Summary entry used by summary-only index tests.
pub fn sample_summary_entry() -> Option<SummaryEntry> {
    Some(SummaryEntry {
        shape: vec![1, COLUMNS],
        finite_count: COLUMNS,
        minimum: 0.0,
        maximum: 1.0,
        mean: 0.5,
        l2_norm: 1.0,
    })
}

/// The raw capture payload rows used across fixtures.
pub fn sample_rows() -> Vec<f32> {
    vec![1.0, 2.0, 3.0, 4.0]
}

/// Position used in fixtures.
pub fn sample_positions() -> Vec<usize> {
    vec![3]
}

/// Force `subtoken_selection`/`normalization` to be constructed (keeps
/// the types referenced in test builds).
pub fn _use_types() {
    let _ = SubtokenSelection::First;
    let _ = TextNormalization::Nfc;
    let _: BundleIdentity;
    let _ = SemanticHookSite::Logits;
}
