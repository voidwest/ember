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
        if self
            .root
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with('.') && name.contains(".tmp-"))
        {
            return Err("bundle output uses the reserved staging-directory name pattern".into());
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

        let staging = {
            let mut created = None;
            for _ in 0..1000 {
                let candidate = parent.join(format!(
                    ".{}.tmp-{}-{}",
                    self.root
                        .file_name()
                        .map(|name| name.to_string_lossy())
                        .unwrap_or_else(|| "bundle".into()),
                    std::process::id(),
                    STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed),
                ));
                match std::fs::create_dir(&candidate) {
                    Ok(()) => {
                        created = Some(candidate);
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(error) => {
                        return Err(format!(
                            "cannot create staging '{}': {error}",
                            candidate.display()
                        ))
                    }
                }
            }
            created.ok_or("could not reserve a unique staging directory")?
        };
        let mut guard = StagingGuard(staging.clone(), self.retain_incomplete);

        // 1. write all deterministic files + runtime.json. Checksums are
        // taken from the bytes handed to the writer; internal verification
        // (step 7) reads every file back from the staging directory and
        // compares it with these checksums, so nothing on disk is trusted
        // without being re-read.
        let mut stage = StagedFiles::new(&staging);
        let mut checksums: BTreeMap<String, String> = BTreeMap::new();
        for (relative, bytes) in &self.files {
            stage.write(relative, bytes)?;
            checksums.insert(relative.clone(), sha256_hex(bytes));
        }
        let runtime_bytes = serde_json::to_vec_pretty(&runtime_json)
            .map_err(|error| format!("runtime.json serialization failed: {error}"))?;
        stage.write("runtime.json", &runtime_bytes)?;
        checksums.insert("runtime.json".to_string(), sha256_hex(&runtime_bytes));

        // 2. write semantic-manifest.json (a bundle file itself, included
        // in the payload inventory)
        let semantic_bytes = serde_json::to_vec_pretty(&semantic_manifest)
            .map_err(|error| format!("semantic-manifest.json serialization failed: {error}"))?;
        stage.write("semantic-manifest.json", &semantic_bytes)?;
        let semantic_file_hash = sha256_hex(&semantic_bytes);

        // 3. checksums over everything except manifest.json
        checksums.insert(
            "semantic-manifest.json".to_string(),
            semantic_file_hash.clone(),
        );

        // 4. identity
        let semantic_hash = BundleIdentity::semantic_hash(&semantic_manifest)?;
        // The payload hash covers the manifest's payload inventory plus
        // the semantic manifest's own file (which cannot list itself);
        // verification recomputes the same inventory.
        let mut payload_inventory = semantic_manifest.payloads.clone();
        payload_inventory.insert("semantic-manifest.json".to_string(), semantic_file_hash);
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
        stage.write("manifest.json", &manifest_bytes)?;

        // 6. checksums.sha256 including manifest.json
        let mut checksum_lines: Vec<String> = Vec::new();
        for (relative, sum) in checksums {
            checksum_lines.push(format!("{sum}  {relative}"));
        }
        checksum_lines.push(format!("{}  manifest.json", sha256_hex(&manifest_bytes)));
        checksum_lines.sort();
        let checksums_bytes = format!("{}\n", checksum_lines.join("\n")).into_bytes();
        stage.write("checksums.sha256", &checksums_bytes)?;
        // Every staged byte is on stable storage before verification reads
        // it back, and so before the rename can publish it.
        stage.flush_to_stable_storage()?;

        // 7. Verify before touching the destination. A bad replacement must
        // not destroy an existing, valid bundle even with overwrite enabled.
        let verification = crate::v05::verify::verify_staged_bundle(
            &staging,
            &crate::v05::verify::VerifyOptions::default(),
        )
        .map_err(|error| format!("staged bundle verification failed: {error}"))?;
        if !verification.ok {
            let failures = verification
                .checks
                .iter()
                .filter(|check| !check.ok)
                .map(|check| format!("{}: {}", check.name, check.detail))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(format!("staged bundle verification failed: {failures}"));
        }

        // 8. Atomic publication with a no-clobber creation or an atomic swap.
        publish_directory(&staging, &self.root, self.overwrite)?;
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

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn publish_directory(staging: &Path, destination: &Path, overwrite: bool) -> Result<(), String> {
    use rustix::fs::{renameat_with, RenameFlags, CWD};
    match renameat_with(CWD, staging, CWD, destination, RenameFlags::NOREPLACE) {
        Ok(()) => return Ok(()),
        Err(error) if error == rustix::io::Errno::EXIST && overwrite => {}
        Err(error) => {
            return Err(format!(
                "cannot publish bundle '{}': {error}",
                destination.display()
            ))
        }
    }
    let metadata = std::fs::symlink_metadata(destination).map_err(|error| error.to_string())?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("overwrite target must be a directory, not a symlink or file".into());
    }
    renameat_with(CWD, staging, CWD, destination, RenameFlags::EXCHANGE).map_err(|error| {
        format!(
            "cannot atomically replace bundle '{}': {error}",
            destination.display()
        )
    })?;
    // The old bundle now occupies our staging path. Publication has committed;
    // a cleanup failure must not be reported as failure of the new bundle.
    if let Err(error) = std::fs::remove_dir_all(staging) {
        log::warn!(
            "published bundle, but old bundle remains at '{}': {error}",
            staging.display()
        );
    }
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn publish_directory(_: &Path, _: &Path, _: bool) -> Result<(), String> {
    Err("atomic bundle directory publication is supported on Linux and macOS".into())
}

/// Files written into the writer's private staging directory.
///
/// The staging directory is created fresh by this process and becomes
/// visible under the bundle's name only through the final directory rename,
/// so the per-file temporary-file-and-rename step [`crate::atomic_file`]
/// uses for files written in place adds nothing here: each file is created
/// with `create_new` and written once.
///
/// Durability matches `atomic_file` (whose `sync_all` is `F_FULLFSYNC` on
/// macOS): every byte is on stable storage before publication. On macOS each
/// file is `fsync`ed as it is written, which hands its data to the drive, and
/// one `F_FULLFSYNC` after the last file flushes the drive's entire write
/// cache, covering every file fsynced before it. A full flush per file cost
/// several milliseconds each, most of a sweep's bundle-writing time.
/// Elsewhere every file gets `sync_all`, as before.
struct StagedFiles<'a> {
    staging: &'a Path,
    last: Option<std::fs::File>,
}

impl<'a> StagedFiles<'a> {
    fn new(staging: &'a Path) -> Self {
        StagedFiles {
            staging,
            last: None,
        }
    }

    fn write(&mut self, relative: &str, bytes: &[u8]) -> Result<(), String> {
        use std::io::Write as _;
        let relative = validate_relative_path(relative)?;
        let path = self.staging.join(relative);
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create '{}': {error}", parent.display()))?;
        }
        let failed = |error: std::io::Error| format!("cannot write '{}': {error}", path.display());
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(failed)?;
        file.write_all(bytes).map_err(failed)?;
        sync_to_device(&file).map_err(failed)?;
        self.last = Some(file);
        Ok(())
    }

    fn flush_to_stable_storage(&mut self) -> Result<(), String> {
        match self.last.take() {
            Some(file) => flush_device_cache(&file)
                .map_err(|error| format!("cannot flush the staged bundle to storage: {error}")),
            None => Ok(()),
        }
    }
}

#[cfg(target_os = "macos")]
fn sync_to_device(file: &std::fs::File) -> std::io::Result<()> {
    rustix::fs::fsync(file).map_err(std::io::Error::from)
}

#[cfg(target_os = "macos")]
fn flush_device_cache(file: &std::fs::File) -> std::io::Result<()> {
    rustix::fs::fcntl_fullfsync(file).map_err(std::io::Error::from)
}

#[cfg(not(target_os = "macos"))]
fn sync_to_device(file: &std::fs::File) -> std::io::Result<()> {
    file.sync_all()
}

#[cfg(not(target_os = "macos"))]
fn flush_device_cache(_: &std::fs::File) -> std::io::Result<()> {
    Ok(())
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

    fn valid_writer(
        root: &Path,
        overwrite: bool,
        retain: bool,
    ) -> (BundleWriter, SemanticManifest) {
        let (files, semantic) = crate::v05::testutil::test_bundle_materials(
            &crate::v05::testutil::sample_rows(),
            &crate::v05::testutil::sample_positions(),
        );
        let mut writer = BundleWriter::new(root.to_path_buf(), overwrite, retain);
        for (path, bytes) in files {
            writer.add(&path, bytes);
        }
        (writer, semantic)
    }

    #[test]
    fn invalid_replacement_preserves_existing_bundle() {
        let root = temp_root();
        let (writer, semantic) = valid_writer(&root, false, false);
        writer.finalize(semantic, serde_json::json!({})).unwrap();
        let original = std::fs::read(root.join("manifest.json")).unwrap();
        let (mut writer, semantic) = valid_writer(&root, true, false);
        writer.files.remove("outputs.jsonl");
        let error = writer
            .finalize(semantic, serde_json::json!({}))
            .unwrap_err();
        assert!(
            error.contains("staged bundle verification failed"),
            "{error}"
        );
        assert_eq!(std::fs::read(root.join("manifest.json")).unwrap(), original);
        assert!(
            crate::v05::verify::verify_bundle(&root, &Default::default())
                .unwrap()
                .ok
        );
        assert!(staging_leftovers(&root).is_empty());
        std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn publication_rechecks_no_clobber_at_the_rename_boundary() {
        let root = temp_root();
        let stage = root.with_file_name("stage");
        std::fs::create_dir(&stage).unwrap();
        std::fs::write(stage.join("new"), b"new").unwrap();
        // Simulate another writer creating the destination after finalize's
        // initial existence check. Even an empty directory must survive.
        std::fs::create_dir(&root).unwrap();
        assert!(publish_directory(&stage, &root, false).is_err());
        assert!(root.is_dir());
        assert!(!root.join("new").exists());
        assert!(stage.join("new").exists());
        std::fs::write(root.join("old"), b"old").unwrap();
        publish_directory(&stage, &root, true).unwrap();
        assert_eq!(std::fs::read(root.join("new")).unwrap(), b"new");
        assert!(!root.join("old").exists());
        assert!(!stage.exists());
        std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn failed_publication_keeps_both_source_and_destination() {
        let root = temp_root();
        let stage = root.with_file_name("stage");
        std::fs::create_dir(&stage).unwrap();
        std::fs::write(stage.join("new"), b"new").unwrap();
        std::fs::write(&root, b"existing file").unwrap();
        assert!(publish_directory(&stage, &root, true).is_err());
        assert_eq!(std::fs::read(&root).unwrap(), b"existing file");
        assert_eq!(std::fs::read(stage.join("new")).unwrap(), b"new");
        std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn retained_failed_verification_is_never_a_published_bundle() {
        let root = temp_root();
        let (writer, mut semantic) = valid_writer(&root, false, true);
        semantic.hook_schema = 99;
        assert!(writer.finalize(semantic, serde_json::json!({})).is_err());
        assert!(!root.exists());
        let leftovers = staging_leftovers(&root);
        assert_eq!(leftovers.len(), 1);
        let stage = root.parent().unwrap().join(&leftovers[0]);
        let error = crate::v05::verify::verify_bundle(&stage, &Default::default()).unwrap_err();
        assert!(error.contains("staging directory"));
        std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn publishes_complete_bundle_atomically() {
        let root = temp_root();
        let (writer, mut semantic) = valid_writer(&root, false, false);
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
    fn published_files_and_checksums_match_the_written_bytes() {
        // Checksums are computed from the in-memory bytes; they must equal
        // the hashes of the files as published on disk.
        let root = temp_root();
        let (writer, semantic) = valid_writer(&root, false, false);
        let added = writer.files.clone();
        writer.finalize(semantic, serde_json::json!({})).unwrap();
        for (relative, bytes) in &added {
            assert_eq!(
                &std::fs::read(root.join(relative)).unwrap(),
                bytes,
                "{relative}"
            );
        }
        let checksums = std::fs::read_to_string(root.join("checksums.sha256")).unwrap();
        let mut listed = 0;
        for line in checksums.lines() {
            let (sum, relative) = line.split_once("  ").unwrap();
            assert_eq!(
                sum,
                sha256_hex(&std::fs::read(root.join(relative)).unwrap()),
                "{relative}"
            );
            listed += 1;
        }
        // every file but checksums.sha256 itself
        assert_eq!(listed, added.len() + 3);
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
        let (writer, semantic) = valid_writer(&root, true, false);
        let (_, identity) = writer.finalize(semantic, serde_json::json!({})).unwrap();
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
