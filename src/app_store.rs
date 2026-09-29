//! Persisted console state: run history, model recency, and the draft.
//!
//! # Why this exists
//!
//! The console kept run history in a `Vec` on the view. That is fine for a
//! session and useless as a product: a dense Runs table wants a timestamp, a
//! model, an intervention, a layer, a duration and token counts, and Home wants
//! to offer the experiment you left half-written. Neither can be derived after
//! a restart, and neither should be guessed at from whatever the loader happens
//! to have open.
//!
//! # Shape
//!
//! One file, one envelope, one schema id -- the same arrangement as
//! [`crate::trace`], for the same reason. The file is written atomically
//! ([`crate::atomic_file::atomic_write`]) because a run record is written from
//! inside the worker-completion path, where a torn write loses the one thing
//! the user just spent minutes producing.
//!
//! # Reading is fail-closed
//!
//! A missing file is an empty store. A *malformed* file is an error, not an
//! empty store. Silently resetting to "no runs" would turn a corrupted file or
//! a future schema into a UI that looks like the user never ran anything, and
//! the next write would then overwrite the only copy of their history. The
//! reader reports what it found and leaves the file alone.

use crate::atomic_file::atomic_write;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Named schema identifier. Bump the suffix on any incompatible change.
pub const STORE_SCHEMA: &str = "ember.appstate.v1";

/// Major version, compared by the reader so a newer file is refused rather
/// than misread.
pub const STORE_SCHEMA_MAJOR: u32 = 1;

/// The intervention configuration of a completed run, as typed into the form.
/// Enough to branch from a past run without the original session; the prompt
/// and model live on the record itself. Optional on [`RunRecord`] so stores
/// written before this field existed still load unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordConfig {
    pub model_path: String,
    pub execution: String,
    pub site: String,
    pub layer: String,
    pub op: String,
    pub value: String,
    pub source: String,
    pub source_layer: String,
    pub token: String,
    pub span: String,
    pub max_tokens: String,
}

/// One layer's divergence in a saved result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordLayer {
    pub layer: usize,
    pub relative_l2: Option<f64>,
    pub cosine: Option<f64>,
}

/// One generated position in a saved result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordToken {
    pub position: usize,
    pub baseline: Option<String>,
    pub intervention: Option<String>,
    pub differs: bool,
}

/// What a finished run showed, kept so History can reopen the comparison and
/// not just list that it happened. Optional and defaulted on [`RunRecord`], so
/// stores written before it existed still load; those rows simply have no Open.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordResult {
    pub baseline_text: String,
    pub intervention_text: String,
    pub layers: Vec<RecordLayer>,
    pub tokens: Vec<RecordToken>,
    pub first_layer_divergence: Option<usize>,
    pub peak_layer: Option<usize>,
    pub peak_relative_l2: Option<f64>,
    pub tokens_equal: bool,
}

/// One completed run, as a record rather than a line of history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    /// Monotonic per-store counter, assigned when the run completed.
    pub number: u64,
    /// Unix seconds. Stored absolute so a store can be sorted and filtered
    /// without knowing the local zone, and formatted at the edge.
    pub finished_at: i64,
    /// Model identifier as the run saw it, usually the filename stem.
    pub model: String,
    /// Human-readable intervention, e.g. "Scale ×0.5".
    pub intervention: String,
    /// The exact hook the run committed to, e.g. `ember.hook.v1 · after-mlp`.
    /// Kept because a record that cannot be reproduced is not a record.
    pub hook: String,
    /// Layer index, when the intervention is per-layer.
    pub layer: Option<u32>,
    /// Wall-clock milliseconds for the run, when measured.
    pub duration_ms: Option<u64>,
    /// Generated tokens for the baseline side.
    pub baseline_tokens: Option<u32>,
    /// Generated tokens for the intervention side.
    pub intervention_tokens: Option<u32>,
    /// First decode step at which the two sides diverged, if any.
    pub diverged_at_step: Option<u32>,
    /// Whether the two sides produced identical text.
    pub outputs_equal: bool,
    /// Whether the run's own verification passed.
    pub verified: bool,
    /// User-set. Sorting puts pinned records first.
    #[serde(default)]
    pub pinned: bool,
    /// The prompt text, so a run can be reopened without retyping it.
    pub prompt: String,
    /// The full form configuration, so a run can be branched from. Absent on
    /// records written before branching existed.
    #[serde(default)]
    pub config: Option<RecordConfig>,
    /// The comparison the run produced, so it can be reopened from History.
    /// Absent on records written before results were kept.
    #[serde(default)]
    pub result: Option<RecordResult>,
}

impl RunRecord {
    /// Sort key for the Runs view: pinned first, then most recent first.
    ///
    /// Inverted booleans sort ascending, so `false` (pinned) lands ahead of
    /// `true`; the timestamp is negated for the same reason. Written as a
    /// comparator-shaped key so `sort_by_key` can use it directly.
    pub fn sort_key(&self) -> (bool, i64) {
        (!self.pinned, -self.finished_at)
    }
}

/// Model recency, so Models can show when something was last used without
/// re-reading every GGUF header on launch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRecord {
    /// Full path, the identity key. A model that moved is a different record.
    pub path: String,
    /// Unix seconds of last use; `None` until the model is actually used.
    pub last_used_at: Option<i64>,
}

/// The experiment the user was in the middle of, so Home can offer to resume.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Draft {
    /// Monotonic revision, so a stale write cannot clobber a newer one.
    #[serde(default)]
    pub revision: u64,
    pub prompt: String,
    pub model_path: String,
    /// Raw form values as typed, keyed by field name, so restoring a draft does
    /// not need to know the meaning of each one.
    #[serde(default)]
    pub fields: std::collections::BTreeMap<String, String>,
    pub step: String,
    pub updated_at: i64,
}

/// The whole file. `schema` is the compatibility anchor; `schema_version` is
/// retained for symmetry with the trace format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppStore {
    pub schema: String,
    pub schema_version: u32,
    /// Oldest first. Bounded by [`MAX_RUNS`] on write.
    #[serde(default)]
    pub runs: Vec<RunRecord>,
    #[serde(default)]
    pub models: Vec<ModelRecord>,
    #[serde(default)]
    pub draft: Option<Draft>,
}

impl Default for AppStore {
    fn default() -> Self {
        Self {
            schema: STORE_SCHEMA.to_string(),
            schema_version: STORE_SCHEMA_MAJOR,
            runs: Vec::new(),
            models: Vec::new(),
            draft: None,
        }
    }
}

/// Retention bound. The history is a convenience, not an archive; a store that
/// grows without limit is a store that eventually stops being written.
pub const MAX_RUNS: usize = 500;

impl AppStore {
    /// Record a completed run, newest first, bounded.
    pub fn push_run(&mut self, run: RunRecord) {
        self.runs.insert(0, run);
        self.runs.truncate(MAX_RUNS);
    }

    /// Note that a model was used. Inserts on first sight.
    pub fn touch_model(&mut self, path: &str, at: i64) {
        match self.models.iter_mut().find(|m| m.path == path) {
            Some(existing) => existing.last_used_at = Some(at),
            None => self.models.push(ModelRecord {
                path: path.to_string(),
                last_used_at: Some(at),
            }),
        }
    }

    /// When a model was last used, if it is known.
    pub fn model_last_used(&self, path: &str) -> Option<i64> {
        self.models
            .iter()
            .find(|m| m.path == path)
            .and_then(|m| m.last_used_at)
    }

    /// Runs in display order: pinned first, then most recent.
    pub fn runs_ordered(&self) -> Vec<&RunRecord> {
        let mut ordered: Vec<&RunRecord> = self.runs.iter().collect();
        ordered.sort_by_key(|run| run.sort_key());
        ordered
    }

    /// Delete one run by its number. A history you cannot prune is a log you
    /// eventually stop opening; `false` simply means the row was already gone.
    pub fn remove_run(&mut self, number: u64) -> bool {
        let before = self.runs.len();
        self.runs.retain(|run| run.number != number);
        before != self.runs.len()
    }

    /// Flip a run's pinned state. Ordering reads `pinned` on every render, so
    /// this is the whole feature; `false` means no such run.
    pub fn toggle_pin(&mut self, number: u64) -> bool {
        match self.runs.iter_mut().find(|run| run.number == number) {
            Some(run) => {
                run.pinned = !run.pinned;
                true
            }
            None => false,
        }
    }

    /// Serialise with the schema stamped, ready for [`write`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut stamped = self.clone();
        stamped.schema = STORE_SCHEMA.to_string();
        stamped.schema_version = STORE_SCHEMA_MAJOR;
        // serde_json is already a dependency via the trace and agent formats.
        serde_json::to_vec_pretty(&stamped).expect("AppStore is always serialisable")
    }

    /// Write atomically, creating the parent directory if needed.
    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        atomic_write(path, &self.to_bytes())
    }
}

/// Why a store could not be read.
///
/// A missing file is not this. It is an empty store.
#[derive(Debug)]
pub enum StoreError {
    /// The bytes are not a store, or not one this build understands.
    Incompatible { path: PathBuf, detail: String },
    /// The file is a store but could not be read from disk.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Incompatible { path, detail } => {
                write!(
                    f,
                    "{} is not a readable app store: {detail}",
                    path.display()
                )
            }
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for StoreError {}

/// Read a store, or an empty one if the file does not exist.
///
/// A file that exists but cannot be understood is an error. Returning defaults
/// there would let the next write destroy the user's history, and would report
/// a corrupted store as a fresh install.
pub fn load(path: impl AsRef<Path>) -> Result<AppStore, StoreError> {
    let path = path.as_ref();
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(AppStore::default());
        }
        Err(source) => {
            return Err(StoreError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let parsed: AppStore =
        serde_json::from_slice(&bytes).map_err(|error| StoreError::Incompatible {
            path: path.to_path_buf(),
            detail: error.to_string(),
        })?;
    if parsed.schema != STORE_SCHEMA {
        return Err(StoreError::Incompatible {
            path: path.to_path_buf(),
            detail: format!("schema {:?}, expected {STORE_SCHEMA:?}", parsed.schema),
        });
    }
    if parsed.schema_version != STORE_SCHEMA_MAJOR {
        return Err(StoreError::Incompatible {
            path: path.to_path_buf(),
            detail: format!(
                "schema_version {}, expected {STORE_SCHEMA_MAJOR}",
                parsed.schema_version
            ),
        });
    }
    Ok(parsed)
}

/// Where the store lives. Same resolution order as the appearance setting:
/// `$XDG_CONFIG_HOME`, then `~/.config`, then the temp directory so a sandboxed
/// or read-only home still works.
pub fn store_path() -> PathBuf {
    if let Some(root) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(root).join("ember/app-state.json");
    }
    if let Some(root) = std::env::var_os("HOME") {
        return PathBuf::from(root).join(".config/ember/app-state.json");
    }
    std::env::temp_dir().join("ember-app-state.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(number: u64, finished_at: i64, pinned: bool) -> RunRecord {
        RunRecord {
            number,
            finished_at,
            model: "Llama-3.2-1B-Instruct-Q8_0".into(),
            intervention: "Scale ×0.5".into(),
            hook: "ember.hook.v1 · after-mlp".into(),
            layer: Some(8),
            duration_ms: Some(1_240),
            baseline_tokens: Some(48),
            intervention_tokens: Some(48),
            diverged_at_step: Some(17),
            outputs_equal: false,
            verified: true,
            pinned,
            prompt: "اكتب جملة قصيرة عن المدينة المنورة".into(),
            config: None,
            result: None,
        }
    }

    fn temp_path(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ember-store-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("app-state.json")
    }

    #[test]
    fn store_written_before_configurations_still_loads() {
        let path = temp_path("pre-config");
        // Exactly the shape this app wrote before RunRecord grew `config`:
        // the field must default to None, not fail the read.
        let legacy = serde_json::json!({
            "schema": STORE_SCHEMA,
            "schema_version": STORE_SCHEMA_MAJOR,
            "runs": [{
                "number": 1,
                "finished_at": 1_700_000_000,
                "model": "Llama-3.2-1B-Instruct-Q8_0",
                "intervention": "Scale ×0.5",
                "hook": "ember.hook.v1 · after-mlp",
                "layer": 8,
                "duration_ms": 1_240,
                "baseline_tokens": 48,
                "intervention_tokens": 48,
                "diverged_at_step": null,
                "outputs_equal": false,
                "verified": true,
                "pinned": false,
                "prompt": "p"
            }],
            "models": [],
            "draft": null,
        });
        std::fs::create_dir_all(path.parent().expect("temp path has a parent"))
            .expect("create temp store directory");
        std::fs::write(&path, serde_json::to_vec(&legacy).expect("serialize legacy store"))
            .expect("write legacy store");
        let store = load(path).expect("a pre-configuration store still loads");
        assert_eq!(store.runs.len(), 1);
        assert!(store.runs[0].config.is_none());
        assert!(
            store.runs[0].result.is_none(),
            "a record from before results were kept has no result, and loads"
        );
    }

    #[test]
    fn a_kept_result_survives_a_write_and_read() {
        let path = temp_path("result-roundtrip");
        let mut store = AppStore::default();
        let mut run = run(1, 1_700_000_000, false);
        run.result = Some(RecordResult {
            baseline_text: "Paris.".into(),
            intervention_text: "fog".into(),
            layers: vec![RecordLayer { layer: 7, relative_l2: Some(1.25), cosine: None }],
            tokens: vec![RecordToken {
                position: 1,
                baseline: Some("Paris".into()),
                intervention: Some("fog".into()),
                differs: true,
            }],
            first_layer_divergence: Some(7),
            peak_layer: Some(7),
            peak_relative_l2: Some(1.25),
            tokens_equal: false,
        });
        store.push_run(run.clone());
        store.write(path.clone()).expect("write");
        let loaded = load(path).expect("read");
        assert_eq!(loaded.runs[0].result, run.result);
    }

    #[test]
    fn missing_file_is_an_empty_store_not_an_error() {
        let store = load(temp_path("missing")).expect("a missing file is a fresh install");
        assert!(store.runs.is_empty());
        assert!(store.models.is_empty());
        assert!(store.draft.is_none());
    }

    #[test]
    fn round_trips_through_disk() {
        let path = temp_path("roundtrip");
        let mut store = AppStore::default();
        store.push_run(run(1, 1_700_000_000, false));
        store.touch_model("/models/llama-q8_0.gguf", 1_700_000_001);
        store.draft = Some(Draft {
            revision: 4,
            prompt: "half written".into(),
            model_path: "/models/llama-q8_0.gguf".into(),
            fields: Default::default(),
            step: "intervention".into(),
            updated_at: 1_700_000_002,
        });
        store.write(&path).expect("write");

        let read = load(&path).expect("read back");
        assert_eq!(read.runs, store.runs);
        assert_eq!(read.models, store.models);
        assert_eq!(read.draft, store.draft);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn newer_schema_is_refused_rather_than_misread() {
        let path = temp_path("future");
        let mut store = AppStore::default();
        store.push_run(run(1, 1, false));
        store.write(&path).expect("write");

        // Someone on a future build wrote this. Reading it as today's shape
        // would drop fields on the next save.
        let text = std::fs::read_to_string(&path).unwrap();
        let bumped = text.replace(
            &format!("\"schema_version\": {STORE_SCHEMA_MAJOR}"),
            "\"schema_version\": 99",
        );
        std::fs::write(&path, bumped).expect("rewrite");

        let error = load(&path).expect_err("a future schema must not be read as today's");
        assert!(
            matches!(error, StoreError::Incompatible { .. }),
            "expected Incompatible, got {error:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn corrupt_file_is_an_error_so_the_next_write_cannot_destroy_it() {
        let path = temp_path("corrupt");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{ this is not json").expect("write");
        let error = load(&path).expect_err("corrupt bytes must not read as an empty store");
        assert!(matches!(error, StoreError::Incompatible { .. }));
        // The file is still there, so the user can recover it.
        assert_eq!(std::fs::read(&path).unwrap(), b"{ this is not json");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pinned_runs_sort_above_recent_ones() {
        let mut store = AppStore::default();
        store.push_run(run(1, 100, false));
        store.push_run(run(2, 200, false));
        store.push_run(run(3, 50, true));
        let order: Vec<u64> = store.runs_ordered().iter().map(|r| r.number).collect();
        assert_eq!(order, vec![3, 2, 1], "pinned first, then newest first");
    }

    #[test]
    fn retention_is_bounded() {
        let mut store = AppStore::default();
        for index in 0..(MAX_RUNS + 25) {
            store.push_run(run(index as u64, index as i64, false));
        }
        assert_eq!(store.runs.len(), MAX_RUNS);
        assert_eq!(store.runs[0].number, (MAX_RUNS + 24) as u64);
    }

    #[test]
    fn model_recency_is_inserted_once_then_updated() {
        let mut store = AppStore::default();
        store.touch_model("/a.gguf", 10);
        store.touch_model("/a.gguf", 20);
        store.touch_model("/b.gguf", 5);
        assert_eq!(store.models.len(), 2);
        assert_eq!(store.model_last_used("/a.gguf"), Some(20));
        assert_eq!(store.model_last_used("/b.gguf"), Some(5));
        assert_eq!(store.model_last_used("/missing.gguf"), None);
    }

    #[test]
    fn a_store_without_optional_sections_still_reads() {
        // Runs-only file, written by an earlier build of the same major schema.
        let path = temp_path("minimal");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            format!("{{\"schema\":\"{STORE_SCHEMA}\",\"schema_version\":{STORE_SCHEMA_MAJOR}}}"),
        )
        .expect("write");
        let store = load(&path).expect("optional sections default");
        assert!(store.runs.is_empty());
        assert!(store.draft.is_none());
        let _ = std::fs::remove_file(&path);
    }
}
