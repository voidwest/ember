//! v0.5 deterministic experiment bundle writer (contract sections 7, 14).
//!
//! Bundle layout (`ember.bundle.v1`):
//!
//! ```text
//! runs/example/
//! ├── manifest.json            top-level identity + file inventory
//! ├── semantic-manifest.json   deterministic semantics (hashed)
//! ├── runtime.json             machine-dependent metadata (not hashed)
//! ├── experiment.toml          verbatim user specification
//! ├── resolved-experiment.json resolved specification with defaults
//! ├── model.json               model identity + GGUF metadata
//! ├── tokenizer.json           tokenizer identity
//! ├── execution-plan.json      the v0.4 ExecutionPlan
//! ├── inputs.jsonl             input texts
//! ├── outputs.jsonl            generated tokens/text/top-1 per input
//! ├── tokenization.jsonl       tokenizations + selection records
//! ├── captures/tensors.safetensors  payloads
//! ├── captures/index.jsonl     per-tensor index entries
//! ├── interventions/events.jsonl    intervention applications
//! ├── traces/events.jsonl      route/fusion/trace events
//! └── checksums.sha256         SHA-256 of every file
//! ```
//!
//! Everything is written into a sibling staging directory and atomically
//! renamed only after all payloads, checksums, and the manifest are
//! complete.

use crate::v05::manifest::{
    sha256_hex, BundleIdentity, BundleManifest, SemanticManifest, BUNDLE_KIND, BUNDLE_SCHEMA_V1,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Staging-directory guard: removes the staging dir on drop unless
/// explicitly released.
struct StagingGuard(PathBuf, bool);

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if !self.1 {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }
}

/// Collects bundle files in memory and publishes them atomically.
pub struct BundleWriter {
    root: PathBuf,
    overwrite: bool,
    retain_incomplete: bool,
    files: BTreeMap<String, Vec<u8>>,
}

impl BundleWriter {
    pub fn new(root: PathBuf, overwrite: bool, retain_incomplete: bool) -> BundleWriter {
        BundleWriter {
            root,
            overwrite,
            retain_incomplete,
            files: BTreeMap::new(),
        }
    }

    /// Add a deterministic bundle file (relative path, forward slashes).
    pub fn add(&mut self, relative: &str, bytes: Vec<u8>) {
        self.files.insert(relative.to_string(), bytes);
    }

    /// Serialize `value` as canonical JSON into the bundle.
    pub fn add_json<T: serde::Serialize>(
        &mut self,
        relative: &str,
        value: &T,
    ) -> Result<(), String> {
        let value: serde_json::Value =
            serde_json::from_slice(&crate::v05::manifest::canonical_json(value)?)
                .map_err(|error| format!("internal JSON round trip failed: {error}"))?;
        let pretty = serde_json::to_vec_pretty(&value)
            .map_err(|error| format!("internal JSON pretty print failed: {error}"))?;
        self.add(relative, pretty);
        Ok(())
    }

    /// The final destination.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Publish the bundle: write staging, checksums, manifest, rename.
    ///
    /// `semantic_manifest` must already carry its `payloads` checksums
    /// (call `finish_semantic_manifest` first).
    pub fn finalize(
        self,
        semantic_manifest: SemanticManifest,
        runtime_json: serde_json::Value,
    ) -> Result<(PathBuf, BundleIdentity), String> {
        if self.root.as_os_str().is_empty() {
            return Err("bundle output directory must not be empty".into());
        }
        if self.root.exists() && !self.overwrite {
            return Err(format!(
                "bundle output '{}' already exists; refusing to overwrite (set \
                 output.overwrite = true to replace it)",
                self.root.display()
            ));
        }
        let parent = self
            .root
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        std::fs::create_dir_all(&parent)
            .map_err(|error| format!("cannot create '{}': {error}", parent.display()))?;

        let staging = parent.join(format!(
            ".{}.tmp-{}-{}",
            self.root
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "bundle".into()),
            std::process::id(),
            STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        if staging.exists() {
            std::fs::remove_dir_all(&staging).ok();
        }
        std::fs::create_dir(&staging)
            .map_err(|error| format!("cannot create staging '{}': {error}", staging.display()))?;
        let mut guard = StagingGuard(staging.clone(), self.retain_incomplete);

        // 1. write all deterministic files + runtime.json
        for (relative, bytes) in &self.files {
            write_staged(&staging, relative, bytes)?;
        }
        let runtime_bytes = serde_json::to_vec_pretty(&runtime_json)
            .map_err(|error| format!("runtime.json serialization failed: {error}"))?;
        write_staged(&staging, "runtime.json", &runtime_bytes)?;

        // 2. write semantic-manifest.json (a bundle file itself, included
        // in the payload inventory)
        let semantic_bytes = serde_json::to_vec_pretty(&semantic_manifest)
            .map_err(|error| format!("semantic-manifest.json serialization failed: {error}"))?;
        write_staged(&staging, "semantic-manifest.json", &semantic_bytes)?;

        // 3. checksums over everything except manifest.json
        let mut checksums: BTreeMap<String, String> = BTreeMap::new();
        for relative in self.files.keys() {
            checksums.insert(
                relative.clone(),
                sha256_hex(&std::fs::read(staging.join(relative)).map_err(|error| {
                    format!(
                        "failed to read staged '{}': {error}",
                        staging.join(relative).display()
                    )
                })?),
            );
        }
        checksums.insert("runtime.json".to_string(), sha256_hex(&runtime_bytes));
        checksums.insert(
            "semantic-manifest.json".to_string(),
            sha256_hex(&semantic_bytes),
        );

        // 4. identity
        let semantic_hash = BundleIdentity::semantic_hash(&semantic_manifest)?;
        // The payload hash covers the manifest's payload inventory plus
        // the semantic manifest's own file (which cannot list itself);
        // verification recomputes the same inventory.
        let mut payload_inventory = semantic_manifest.payloads.clone();
        payload_inventory.insert(
            "semantic-manifest.json".to_string(),
            sha256_hex(&semantic_bytes),
        );
        let payload_hash = BundleIdentity::payload_hash(&payload_inventory)?;

        // 5. manifest.json (not part of any hash)
        let mut files: Vec<String> = checksums.keys().cloned().collect();
        files.push("manifest.json".to_string());
        files.sort();
        let manifest = BundleManifest {
            bundle_schema: BUNDLE_SCHEMA_V1.to_string(),
            kind: BUNDLE_KIND.to_string(),
            status: "complete".to_string(),
            semantic_hash: semantic_hash.clone(),
            payload_hash: payload_hash.clone(),
            files,
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|error| format!("manifest.json serialization failed: {error}"))?;
        write_staged(&staging, "manifest.json", &manifest_bytes)?;

        // 6. checksums.sha256 including manifest.json
        let mut checksum_lines: Vec<String> = Vec::new();
        for (relative, sum) in checksums {
            checksum_lines.push(format!("{sum}  {relative}"));
        }
        checksum_lines.push(format!("{}  manifest.json", sha256_hex(&manifest_bytes)));
        checksum_lines.sort();
        let checksums_bytes = format!("{}\n", checksum_lines.join("\n")).into_bytes();
        write_staged(&staging, "checksums.sha256", &checksums_bytes)?;

        // 7. atomic publish
        if self.root.exists() {
            std::fs::remove_dir_all(&self.root).map_err(|error| {
                format!(
                    "cannot remove existing bundle '{}': {error}",
                    self.root.display()
                )
            })?;
        }
        std::fs::rename(&staging, &self.root)
            .map_err(|error| format!("cannot publish bundle '{}': {error}", self.root.display()))?;
        guard.1 = true; // staging no longer exists
        Ok((
            self.root,
            BundleIdentity {
                semantic_hash,
                payload_hash,
            },
        ))
    }
}

fn write_staged(staging: &Path, relative: &str, bytes: &[u8]) -> Result<(), String> {
    let relative = validate_relative_path(relative)?;
    let path = staging.join(relative);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create '{}': {error}", parent.display()))?;
    }
    crate::atomic_file::atomic_write(&path, bytes)
        .map_err(|error| format!("cannot write '{}': {error}", path.display()))
}

/// Reject absolute paths, traversal components, and empty segments.
pub(crate) fn validate_relative_path(relative: &str) -> Result<&str, String> {
    if relative.is_empty() {
        return Err("bundle file path must not be empty".into());
    }
    if Path::new(relative).is_absolute() {
        return Err(format!("bundle file path '{relative}' must be relative"));
    }
    for component in Path::new(relative).components() {
        use std::path::Component;
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {
                return Err(format!(
                    "bundle file path '{relative}' contains '.' components"
                ))
            }
            _ => {
                return Err(format!(
                    "bundle file path '{relative}' contains unsafe components (path traversal)"
                ))
            }
        }
    }
    Ok(relative)
}

/// Compute the deterministic payload checksum map for a finished bundle
/// (used by the runner before finalize).
pub fn payload_checksums(files: &BTreeMap<String, Vec<u8>>) -> BTreeMap<String, String> {
    files
        .iter()
        .map(|(relative, bytes)| (relative.clone(), sha256_hex(bytes)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v05::manifest::{
        ManifestExecutionMeta, ManifestExperimentMeta, ManifestGenerated, ManifestInputMeta,
        ManifestModelMeta, ManifestTokenizerMeta,
    };

    fn temp_root() -> PathBuf {
        let parent = crate::v05::testutil::temp_root("bundle");
        std::fs::create_dir_all(&parent).unwrap();
        std::fs::create_dir_all(&parent).unwrap();
        parent.join("bundle")
    }

    fn staging_leftovers(root: &Path) -> Vec<String> {
        let parent = root.parent().unwrap();
        std::fs::read_dir(parent)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect()
    }

    fn sample_manifest(payloads: BTreeMap<String, String>) -> SemanticManifest {
        SemanticManifest {
            bundle_schema: BUNDLE_SCHEMA_V1.into(),
            experiment_schema: "ember.experiment.v1".into(),
            hook_schema: 1,
            plan_schema: 1,
            ember_version: "0.5.0-test".into(),
            ember_commit: "test".into(),
            experiment: ManifestExperimentMeta {
                name: "t".into(),
                description: String::new(),
                seed: 0,
            },
            model: ManifestModelMeta {
                sha256: "aa".repeat(32),
                architecture: "llama".into(),
                layer_count: 1,
                embed_dim: 4,
                vocab_size: 16,
                quantization: "q8_0".into(),
            },
            tokenizer: ManifestTokenizerMeta {
                sha256: "bb".repeat(32),
                vocab_size: 16,
            },
            execution: ManifestExecutionMeta {
                mode: "reference".into(),
                deterministic: true,
                plan_hash: "cc".repeat(32),
            },
            inputs: vec![ManifestInputMeta {
                id: "i1".into(),
                prompt_hash: "dd".repeat(32),
            }],
            token_selections: Vec::new(),
            captures: Vec::new(),
            interventions: Vec::new(),
            generated: ManifestGenerated {
                token_ids: vec![vec![1]],
                texts: vec!["x".into()],
            },
            payloads,
            warnings: Vec::new(),
            complete: true,
        }
    }

    #[test]
    fn publishes_complete_bundle_atomically() {
        let root = temp_root();
        let mut writer = BundleWriter::new(root.clone(), false, false);
        writer.add("inputs.jsonl", br#"{"id":"i1","text":"hello"}"#.to_vec());
        let payloads = payload_checksums(&writer.files);
        let mut semantic = sample_manifest(payloads);
        let semantic_hash = BundleIdentity::semantic_hash(&semantic).unwrap();
        let (published, identity) = writer
            .finalize(semantic.clone(), serde_json::json!({"hostname": "test"}))
            .unwrap();
        assert_eq!(published, root);
        assert_eq!(identity.semantic_hash, semantic_hash);
        assert!(root.join("manifest.json").exists());
        assert!(root.join("checksums.sha256").exists());
        assert!(root.join("runtime.json").exists());
        assert!(root.join("inputs.jsonl").exists());
        // No staging leftovers in the parent.
        assert!(staging_leftovers(&root).is_empty());
        // verification: manifest status complete
        let manifest: BundleManifest =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest.status, "complete");
        assert_eq!(manifest.bundle_schema, BUNDLE_SCHEMA_V1);
        semantic.complete = false;
        assert_ne!(
            BundleIdentity::semantic_hash(&semantic).unwrap(),
            semantic_hash,
            "semantic hash must change with content"
        );
        std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn refuses_overwrite_without_permission() {
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();
        let writer = BundleWriter::new(root.clone(), false, false);
        let payloads = BTreeMap::new();
        let result = writer.finalize(sample_manifest(payloads), serde_json::json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("refusing to overwrite"));
        // with overwrite it succeeds
        let writer = BundleWriter::new(root.clone(), true, false);
        let payloads = BTreeMap::new();
        let (_, identity) = writer
            .finalize(sample_manifest(payloads), serde_json::json!({}))
            .unwrap();
        assert_eq!(identity.semantic_hash.len(), 64);
        std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn retained_incomplete_staging_is_marked() {
        let root = temp_root();
        // A traversal path is rejected at finalize time, before publish.
        let mut writer = BundleWriter::new(root.clone(), false, true);
        writer.add("inputs.jsonl", b"x".to_vec());
        writer.add("../escape.bin", b"evil".to_vec());
        let payloads = payload_checksums(&writer.files);
        let result = writer.finalize(sample_manifest(payloads), serde_json::json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsafe components"));
        assert!(!root.exists(), "a failed bundle must never be published");
        // With retain_incomplete, the staging directory remains, clearly
        // marked with a leading dot and `.tmp-` so `verify` can never
        // mistake it for a bundle.
        let leftovers = staging_leftovers(&root);
        assert_eq!(leftovers.len(), 1, "{leftovers:?}");
        assert!(leftovers[0].starts_with('.'));
        std::fs::remove_dir_all(root.parent().unwrap()).ok();
    }

    #[test]
    fn failed_finalize_cleans_staging_by_default() {
        let root = temp_root();
        let mut writer = BundleWriter::new(root.clone(), false, false);
        writer.add("inputs.jsonl", b"x".to_vec());
        writer.add("../escape.bin", b"evil".to_vec());
        let payloads = payload_checksums(&writer.files);
        let result = writer.finalize(sample_manifest(payloads), serde_json::json!({}));
        assert!(result.is_err());
        assert!(staging_leftovers(&root).is_empty());
        std::fs::remove_dir_all(root.parent().unwrap()).ok();
    }

    #[test]
    fn path_validation_rejects_traversal_and_absolute() {
        assert!(validate_relative_path("captures/index.jsonl").is_ok());
        assert!(validate_relative_path("a/b/c.json").is_ok());
        assert!(validate_relative_path("../x").is_err());
        assert!(validate_relative_path("/abs/x").is_err());
        assert!(validate_relative_path("a/../b").is_err());
        assert!(validate_relative_path("").is_err());
    }
}
