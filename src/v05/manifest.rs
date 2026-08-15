//! v0.5 semantic manifest and bundle identity (contract sections 13, 14).
//!
//! The semantic manifest holds only fields expected to be equal across
//! equivalent reruns. Canonical JSON hashing: serde_json values with
//! sorted object keys (serde_json's default Map is a BTreeMap), stable
//! array order, shortest-round-trip float serialization, UTF-8.

use crate::v05::capture::CaptureSpec;
use crate::v05::intervention::InterventionSpec;
use crate::v05::token_select::TokenSelectionRecord;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Bundle schema version identifier.
pub const BUNDLE_SCHEMA_V1: &str = "ember.bundle.v1";

/// Bundle kind marker.
pub const BUNDLE_KIND: &str = "ember-experiment-bundle";

/// SHA-256 digest as lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Canonical JSON bytes of a serializable value: sorted keys, stable
/// arrays, documented float formatting.
pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    let mut value = serde_json::to_value(value)
        .map_err(|error| format!("canonical serialization failed: {error}"))?;
    crate::plan::sort_value_keys(&mut value);
    serde_json::to_vec(&value).map_err(|error| format!("canonical serialization failed: {error}"))
}

/// SHA-256 over the canonical JSON of `value`.
pub fn canonical_hash<T: Serialize>(value: &T) -> Result<String, String> {
    Ok(sha256_hex(&canonical_json(value)?))
}

/// Experiment identity metadata (semantic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestExperimentMeta {
    pub name: String,
    pub description: String,
    pub seed: u64,
}

/// Model identity metadata (semantic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestModelMeta {
    pub sha256: String,
    pub architecture: String,
    pub layer_count: usize,
    pub embed_dim: usize,
    pub vocab_size: usize,
    /// GGUF quantization summary, e.g. `q4_k` majority.
    pub quantization: String,
}

/// Tokenizer identity metadata (semantic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestTokenizerMeta {
    pub sha256: String,
    pub vocab_size: usize,
}

/// Execution identity metadata (semantic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestExecutionMeta {
    pub mode: String,
    pub deterministic: bool,
    pub plan_hash: String,
}

/// Input identity (semantic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestInputMeta {
    pub id: String,
    /// SHA-256 of the prompt text (UTF-8 bytes).
    pub prompt_hash: String,
}

/// Generated output identity (semantic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestGenerated {
    /// Generated token IDs per input (input order).
    pub token_ids: Vec<Vec<u32>>,
    /// Generated text per input where deterministic (greedy or seeded).
    pub texts: Vec<String>,
}

/// The deterministic semantic manifest (contract section 14).
///
/// Identity hashes live in the top-level `manifest.json`, never here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticManifest {
    pub bundle_schema: String,
    pub experiment_schema: String,
    pub hook_schema: u32,
    pub plan_schema: u32,
    pub ember_version: String,
    pub ember_commit: String,
    pub experiment: ManifestExperimentMeta,
    pub model: ManifestModelMeta,
    pub tokenizer: ManifestTokenizerMeta,
    pub execution: ManifestExecutionMeta,
    pub inputs: Vec<ManifestInputMeta>,
    /// Resolved token-selection records across all inputs (input order).
    pub token_selections: Vec<TokenSelectionRecord>,
    /// Capture definitions.
    pub captures: Vec<CaptureSpec>,
    /// Intervention definitions.
    pub interventions: Vec<InterventionSpec>,
    pub generated: ManifestGenerated,
    /// Payload checksums: relative bundle path -> SHA-256, sorted.
    pub payloads: BTreeMap<String, String>,
    /// Deterministic warnings (fallbacks, overrides).
    pub warnings: Vec<String>,
    /// Set true only when the bundle finalization completed.
    pub complete: bool,
}

/// Bundle identity: semantic hash (scientific semantics) and payload hash
/// (complete deterministic artifact contents).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleIdentity {
    pub semantic_hash: String,
    pub payload_hash: String,
}

impl BundleIdentity {
    /// Compute the semantic hash over the manifest with the given
    /// identity excluded (the identity is stored in `manifest.json`).
    pub fn semantic_hash(manifest: &SemanticManifest) -> Result<String, String> {
        canonical_hash(manifest)
    }

    /// Compute the payload hash over the deterministic file inventory.
    ///
    /// `payloads` maps relative path -> SHA-256 for every deterministic
    /// file (all bundle files except `manifest.json`, `runtime.json`,
    /// `verification.json`, and `checksums.sha256`).
    pub fn payload_hash(payloads: &BTreeMap<String, String>) -> Result<String, String> {
        let mut list: Vec<(String, String)> = payloads
            .iter()
            .map(|(path, sum)| (path.clone(), sum.clone()))
            .collect();
        list.sort();
        canonical_hash(&list)
    }
}

/// Top-level bundle manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleManifest {
    pub bundle_schema: String,
    pub kind: String,
    pub status: String,
    pub semantic_hash: String,
    pub payload_hash: String,
    /// All files in the bundle (relative paths, sorted).
    pub files: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_is_key_sorted_and_stable() {
        let mut a = serde_json::Map::new();
        a.insert("z".into(), serde_json::json!(1));
        a.insert("a".into(), serde_json::json!(2));
        let value_a = serde_json::Value::Object(a);
        let mut b = serde_json::Map::new();
        b.insert("a".into(), serde_json::json!(2));
        b.insert("z".into(), serde_json::json!(1));
        let value_b = serde_json::Value::Object(b);
        assert_eq!(
            canonical_json(&value_a).unwrap(),
            canonical_json(&value_b).unwrap()
        );
        assert_eq!(canonical_json(&value_a).unwrap(), br#"{"a":2,"z":1}"#);
    }

    #[test]
    fn payload_hash_ignores_inventory_order() {
        let mut first = BTreeMap::new();
        first.insert("b".to_string(), "11".to_string());
        first.insert("a".to_string(), "22".to_string());
        let mut second = BTreeMap::new();
        second.insert("a".to_string(), "22".to_string());
        second.insert("b".to_string(), "11".to_string());
        assert_eq!(
            BundleIdentity::payload_hash(&first).unwrap(),
            BundleIdentity::payload_hash(&second).unwrap()
        );
        second.insert("c".to_string(), "33".to_string());
        assert_ne!(
            BundleIdentity::payload_hash(&first).unwrap(),
            BundleIdentity::payload_hash(&second).unwrap()
        );
    }

    #[test]
    fn sha256_hex_is_lowercase_64() {
        let sum = sha256_hex(b"ember");
        assert_eq!(sum.len(), 64);
        assert!(sum
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    }
}
