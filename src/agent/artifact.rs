//! Run-local artifact records (Track O).
//!
//! Research workflows produce files. Phase 1 keeps a trace-local manifest:
//! every produced file gets an id, a sanitized path under the run's
//! artifact directory, a SHA-256, and producer provenance. No artifact
//! database — the trace and the returned [`AgentRunSummary`](crate::agent::AgentRunSummary)
//! carry everything.

use anyhow::{ensure, Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::ids::short_hash;

/// Provenance record for one file produced during a run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArtifactRecord {
    pub artifact_id: String,
    /// Path of the written file (absolute after store creation).
    pub path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
    /// RFC 2046-style media type (`text/plain`, `application/json`, ...).
    pub media_type: String,
    /// Tool that produced the artifact.
    pub producer_tool: String,
    /// Agent step that produced it (`model-0`, `tool-1`, ...).
    pub step_id: String,
    pub run_id: String,
}

/// Filesystem-backed artifact manifest for one run.
///
/// Files are written atomically (temp + rename) into `root`; identical
/// content produces distinct artifacts (sequence-numbered ids) because two
/// writes are two events even when their bytes match.
pub struct ArtifactStore {
    root: PathBuf,
    run_id: String,
    records: Vec<ArtifactRecord>,
    next_seq: u64,
}

impl ArtifactStore {
    /// Create the store directory (parents included) for `run_id`.
    pub fn open(root: impl Into<PathBuf>, run_id: &str) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)
            .with_context(|| format!("failed to create artifact dir {}", root.display()))?;
        Ok(Self {
            root,
            run_id: run_id.to_string(),
            records: Vec::new(),
            next_seq: 0,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Re-key future records to the owning run's id.
    pub fn set_run_id(&mut self, run_id: &str) {
        self.run_id = run_id.to_string();
    }

    pub fn records(&self) -> &[ArtifactRecord] {
        &self.records
    }

    /// Sanitize a producer-supplied file name: a single path component of
    /// `[A-Za-z0-9._-]`, no traversal, non-empty, bounded length.
    pub fn sanitize_name(name: &str) -> Result<String> {
        ensure!(!name.is_empty(), "artifact name must not be empty");
        ensure!(
            name.len() <= 128,
            "artifact name exceeds 128 bytes: {} bytes",
            name.len()
        );
        ensure!(
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')),
            "artifact name `{name}` must contain only [A-Za-z0-9._-]"
        );
        ensure!(
            name != "." && name != "..",
            "artifact name `{name}` is not a usable file name"
        );
        Ok(name.to_string())
    }

    /// Persist `content` as a new artifact and record its provenance.
    ///
    /// Fails closed on any filesystem error; nothing half-written remains
    /// visible (sibling temp file + rename).
    pub fn write(
        &mut self,
        name: &str,
        media_type: &str,
        content: &[u8],
        producer_tool: &str,
        step_id: &str,
    ) -> Result<ArtifactRecord> {
        let name = Self::sanitize_name(name)?;
        let sha256 = crate::extraction::sha256_bytes(content);
        let seq = self.next_seq;
        self.next_seq += 1;
        let artifact_id = format!("{seq:04}-{}", short_hash(content));
        let path = self.root.join(format!("{artifact_id}-{name}"));

        let tmp = self
            .root
            .join(format!(".{}.tmp-{}-{seq}", artifact_id, std::process::id()));
        {
            let mut f = std::fs::File::create(&tmp)
                .with_context(|| format!("failed to stage {}", tmp.display()))?;
            f.write_all(content)
                .and_then(|_| f.flush())
                .and_then(|_| f.sync_all())
                .with_context(|| format!("failed to write {}", tmp.display()))?;
        }
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("failed to publish {}", path.display()))?;

        let record = ArtifactRecord {
            artifact_id: artifact_id.clone(),
            path,
            sha256,
            size_bytes: content.len() as u64,
            media_type: media_type.to_string(),
            producer_tool: producer_tool.to_string(),
            step_id: step_id.to_string(),
            run_id: self.run_id.clone(),
        };
        self.records.push(record.clone());
        Ok(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_artifacts_with_hash_and_provenance() {
        let dir = std::env::temp_dir().join(format!(
            "ember-agent-art-test-{}-{}",
            std::process::id(),
            std::time::Instant::now().elapsed().as_nanos()
        ));
        let mut store = ArtifactStore::open(&dir, "run-x").unwrap();
        let rec = store
            .write(
                "note.txt",
                "text/plain",
                b"hello",
                "write_artifact",
                "tool-1",
            )
            .unwrap();
        assert_eq!(rec.size_bytes, 5);
        assert_eq!(rec.sha256, crate::extraction::sha256_bytes(b"hello"));
        assert!(rec.path.is_file());
        assert_eq!(store.records().len(), 1);
        let bytes = std::fs::read(&rec.path).unwrap();
        assert_eq!(bytes, b"hello");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_traversal_and_bad_names_fail_closed() {
        for bad in ["../evil", "a/b", "", ".", "..", "sp ace", "\u{1f4a9}"] {
            assert!(
                ArtifactStore::sanitize_name(bad).is_err(),
                "expected rejection of {bad:?}"
            );
        }
        assert!(ArtifactStore::sanitize_name("result-note_v2.txt").is_ok());
    }

    #[test]
    fn identical_content_gets_distinct_ids() {
        let dir = std::env::temp_dir().join(format!(
            "ember-agent-art-test-{}-dup",
            std::time::Instant::now().elapsed().as_nanos()
        ));
        let mut store = ArtifactStore::open(&dir, "run-dup").unwrap();
        let a = store.write("a.txt", "text/plain", b"x", "t", "s").unwrap();
        let b = store.write("b.txt", "text/plain", b"x", "t", "s").unwrap();
        assert_ne!(a.artifact_id, b.artifact_id);
        assert_eq!(a.sha256, b.sha256);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
