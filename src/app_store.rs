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
//! [`crate::trace`], for the same reason. The file is `app-state.v2.json` in
//! [`config_dir`] ([`store_path`]). The file is written atomically
//! ([`crate::atomic_file::atomic_write`]) because a run record is written from
//! inside the worker-completion path, where a torn write loses the one thing
//! the user just spent minutes producing. Two console windows can share the
//! file, so the console writes through [`AppStore::save_merged`], which locks,
//! re-reads and merges rather than replacing another instance's runs.
//!
//! # Reading is fail-closed
//!
//! A missing file is an empty store. A *malformed* file is an error, not an
//! empty store. Silently resetting to "no runs" would turn a corrupted file or
//! a future schema into a UI that looks like the user never ran anything, and
//! the next write would then overwrite the only copy of their history. The
//! reader reports what it found and leaves the file alone.
//!
//! # The legacy file
//!
//! Builds before the v2 file kept history in `app-state.json`
//! ([`legacy_store_path`]). They cannot be changed, and they read that file,
//! drop every field they do not know (configurations, results, the run
//! counter, tombstones) and write it back -- the oldest ones even replace an
//! unreadable file with an empty store. So this build never writes the legacy
//! file: it only reads it, on every [`open`], and imports runs it has not seen
//! yet ([`AppStore::import_legacy`]). The first launch of this build is the
//! migration; runs made later with an old build still appear; and an old build
//! can only ever damage the file it owns, never this one.

use crate::atomic_file::atomic_write;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Named schema identifier. Bump the suffix on any incompatible change.
pub const STORE_SCHEMA: &str = "ember.appstate.v1";

/// Major version, compared by the reader so a newer file is refused rather
/// than misread.
pub const STORE_SCHEMA_MAJOR: u32 = 1;

/// Additive revision within [`STORE_SCHEMA_MAJOR`]. Fields added under the
/// same major are optional, so an older build can read a newer file -- but
/// serde drops fields it does not know, so that build must not write the file
/// back. The reader accepts a higher minor; [`AppStore::written_by_newer_build`]
/// tells the caller to treat the session as read-only.
///
/// History: 0 had no `config`, `result` or `last_run_number`; 1 has all three;
/// 2 adds `deleted_runs` and `legacy_imported`.
pub const STORE_SCHEMA_MINOR: u32 = 2;

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
    /// Monotonic revision. When two instances save different drafts, the
    /// merge in [`AppStore::save_merged`] keeps the higher revision, so a
    /// stale write cannot clobber a newer one.
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

/// Identity of a run across instances: a number alone can be issued twice by
/// two instances that started from the same file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RunKey {
    pub number: u64,
    pub finished_at: i64,
}

/// The whole file. `schema` is the compatibility anchor; `schema_version` is
/// retained for symmetry with the trace format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppStore {
    pub schema: String,
    pub schema_version: u32,
    /// See [`STORE_SCHEMA_MINOR`]. Defaulted: files from before it existed
    /// are minor 0.
    #[serde(default)]
    pub schema_minor: u32,
    /// Oldest first. Bounded by [`MAX_RUNS`] on write.
    #[serde(default)]
    pub runs: Vec<RunRecord>,
    #[serde(default)]
    pub models: Vec<ModelRecord>,
    #[serde(default)]
    pub draft: Option<Draft>,
    /// Highest run number ever issued. Kept apart from `runs` because deleting
    /// the newest run, or the [`MAX_RUNS`] bound dropping old ones, must not
    /// let a later run take a number that already named a different run.
    /// Defaulted so older stores load; [`AppStore::next_run_number`] falls back
    /// to the highest stored number for them.
    #[serde(default)]
    pub last_run_number: u64,
    /// Tombstones: runs deleted here or in any other instance that shared
    /// the file, so a merge does not bring them back from either side.
    /// Persisted, unioned by [`AppStore::merged_with`], and bounded to the
    /// [`MAX_TOMBSTONES`] most recent.
    #[serde(default)]
    pub deleted_runs: BTreeSet<RunKey>,
    /// Legacy-file runs already imported, keyed as the legacy file names
    /// them. An import can renumber a run, so its key here is the only way
    /// to recognise it the next time the legacy file is read -- and a run
    /// deleted after import must not be imported again. Bounded like
    /// `deleted_runs`.
    #[serde(default)]
    pub legacy_imported: BTreeSet<RunKey>,
    /// The highest draft revision this session cleared, so a merge does not
    /// resurrect that draft -- and does not erase one saved later elsewhere.
    #[serde(skip)]
    cleared_draft: Option<u64>,
    /// Changes whenever `runs` does through this API ([`AppStore::generation`]).
    /// Not persisted and not part of equality.
    #[serde(skip, default = "fresh_generation")]
    generation: u64,
}

/// Every store content change takes a value from one process-wide counter, so
/// a store replaced wholesale (a merge result, a reload) can never present a
/// generation some earlier store already had.
static GENERATIONS: AtomicU64 = AtomicU64::new(1);

fn fresh_generation() -> u64 {
    GENERATIONS.fetch_add(1, Ordering::Relaxed)
}

/// Equality of content. The generation is bookkeeping for views, so two
/// stores holding the same history compare equal whatever it says.
impl PartialEq for AppStore {
    fn eq(&self, other: &Self) -> bool {
        self.schema == other.schema
            && self.schema_version == other.schema_version
            && self.schema_minor == other.schema_minor
            && self.runs == other.runs
            && self.models == other.models
            && self.draft == other.draft
            && self.last_run_number == other.last_run_number
            && self.deleted_runs == other.deleted_runs
            && self.legacy_imported == other.legacy_imported
            && self.cleared_draft == other.cleared_draft
    }
}

impl Default for AppStore {
    fn default() -> Self {
        Self {
            schema: STORE_SCHEMA.to_string(),
            schema_version: STORE_SCHEMA_MAJOR,
            schema_minor: STORE_SCHEMA_MINOR,
            runs: Vec::new(),
            models: Vec::new(),
            draft: None,
            last_run_number: 0,
            deleted_runs: BTreeSet::new(),
            legacy_imported: BTreeSet::new(),
            cleared_draft: None,
            generation: fresh_generation(),
        }
    }
}

/// Retention bound. The history is a convenience, not an archive; a store that
/// grows without limit is a store that eventually stops being written.
pub const MAX_RUNS: usize = 500;

/// Bound on `deleted_runs` and `legacy_imported`: twice the run bound, the
/// most recent kept. A run older than that has already fallen out of every
/// store through [`MAX_RUNS`], so forgetting its tombstone cannot revive it.
pub const MAX_TOMBSTONES: usize = 2 * MAX_RUNS;

/// Keep the [`MAX_TOMBSTONES`] most recent keys.
fn bound_keys(keys: &mut BTreeSet<RunKey>) {
    if keys.len() <= MAX_TOMBSTONES {
        return;
    }
    let mut by_recency: Vec<RunKey> = keys.iter().copied().collect();
    by_recency.sort_by_key(|key| std::cmp::Reverse((key.finished_at, key.number)));
    by_recency.truncate(MAX_TOMBSTONES);
    *keys = by_recency.into_iter().collect();
}

/// Union models by path, keeping the latest use.
fn merge_models(into: &mut Vec<ModelRecord>, from: &[ModelRecord]) {
    for model in from {
        match into.iter_mut().find(|m| m.path == model.path) {
            Some(existing) => {
                existing.last_used_at = existing.last_used_at.max(model.last_used_at);
            }
            None => into.push(model.clone()),
        }
    }
}

/// A run whose number a merge changed, because another instance had already
/// issued that number for a different run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Renumbered {
    pub from: u64,
    pub to: u64,
    pub finished_at: i64,
}

/// The result of merging this session's store with the file on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct Merged {
    /// What was written, and what the session should continue from.
    pub store: AppStore,
    /// Runs of this session that were renumbered, so references held
    /// elsewhere (an open run, a pending confirmation) can follow them.
    pub renumbered: Vec<Renumbered>,
}

fn run_key(run: &RunRecord) -> RunKey {
    RunKey {
        number: run.number,
        finished_at: run.finished_at,
    }
}

impl AppStore {
    /// Changes whenever the runs change through this API (push, delete, pin,
    /// merge, renumbering, import) and on every load, so a view that mirrors
    /// the runs can skip rebuilding when it has not. Code that edits `runs`
    /// directly must call [`AppStore::touch`].
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Mark the runs as changed; see [`AppStore::generation`].
    pub fn touch(&mut self) {
        self.generation = fresh_generation();
    }

    /// Issue the number for a new run: one past anything this store has ever
    /// issued or still holds. Open, Pin and Delete address runs by number, so a
    /// reused number would make them act on the wrong row.
    pub fn next_run_number(&mut self) -> u64 {
        let highest_stored = self.runs.iter().map(|run| run.number).max().unwrap_or(0);
        self.last_run_number = self.last_run_number.max(highest_stored) + 1;
        self.last_run_number
    }

    /// Record a completed run, newest first, bounded.
    pub fn push_run(&mut self, run: RunRecord) {
        self.runs.insert(0, run);
        self.runs.truncate(MAX_RUNS);
        self.touch();
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
    ///
    /// The deletion is recorded as a tombstone in the file, so neither a
    /// merge with the file nor another instance's merge brings it back.
    pub fn remove_run(&mut self, number: u64) -> bool {
        let before = self.runs.len();
        let removed = &mut self.deleted_runs;
        self.runs.retain(|run| {
            let keep = run.number != number;
            if !keep {
                removed.insert(run_key(run));
            }
            keep
        });
        bound_keys(&mut self.deleted_runs);
        let changed = before != self.runs.len();
        if changed {
            self.touch();
        }
        changed
    }

    /// Drop the draft, remembering its revision so a merge with the file on
    /// disk does not bring it back.
    pub fn clear_draft(&mut self) {
        if let Some(draft) = self.draft.take() {
            self.cleared_draft = Some(self.cleared_draft.unwrap_or(0).max(draft.revision));
        }
    }

    /// The revision for a newly saved draft: above anything this session has
    /// held or cleared.
    pub fn next_draft_revision(&self) -> u64 {
        let held = self.draft.as_ref().map_or(0, |draft| draft.revision);
        held.max(self.cleared_draft.unwrap_or(0)) + 1
    }

    /// Whether a newer build wrote this file. Fields it added would be lost
    /// on a write from this build, so such a store is read-only here.
    pub fn written_by_newer_build(&self) -> bool {
        self.schema_minor > STORE_SCHEMA_MINOR
    }

    /// Merge this session's store (`self`) with what another instance may
    /// have written (`disk`).
    ///
    /// - Runs are a union keyed by `(number, finished_at)`; for a run both
    ///   sides hold, this side's version wins (a pin, say).
    /// - Tombstones are a union, and a tombstoned run is dropped from both
    ///   sides: a run deleted in any instance stays deleted.
    /// - The legacy import ledger is a union.
    /// - A run of ours whose number the other side already issued to a
    ///   different run is renumbered past every number either side knows.
    /// - Models are a union by path, keeping the latest use.
    /// - The draft is ours, unless the other side's has a higher revision.
    /// - `last_run_number` is the maximum; runs are bounded by [`MAX_RUNS`].
    pub fn merged_with(&self, disk: &AppStore) -> Merged {
        let mut merged = self.clone();
        merged
            .deleted_runs
            .extend(disk.deleted_runs.iter().copied());
        merged
            .legacy_imported
            .extend(disk.legacy_imported.iter().copied());
        let tombstones = &merged.deleted_runs;
        merged
            .runs
            .retain(|run| !tombstones.contains(&run_key(run)));
        let ours: HashSet<RunKey> = self.runs.iter().map(run_key).collect();
        let theirs: HashSet<RunKey> = disk.runs.iter().map(run_key).collect();
        for run in &disk.runs {
            let key = run_key(run);
            if !ours.contains(&key) && !tombstones.contains(&key) {
                merged.runs.push(run.clone());
            }
        }
        // Numbers the other side issued to runs that are not ours.
        let taken: HashSet<u64> = disk
            .runs
            .iter()
            .filter(|run| !ours.contains(&run_key(run)))
            .map(|run| run.number)
            .collect();
        // Newest first, the order `push_run` keeps.
        merged
            .runs
            .sort_by_key(|run| std::cmp::Reverse((run.finished_at, run.number)));
        let mut next = merged
            .runs
            .iter()
            .map(|run| run.number)
            .max()
            .unwrap_or(0)
            .max(self.last_run_number)
            .max(disk.last_run_number);
        let mut renumbered = Vec::new();
        // Oldest first, so renumbered runs keep their relative order.
        for run in merged.runs.iter_mut().rev() {
            let key = run_key(run);
            if ours.contains(&key) && !theirs.contains(&key) && taken.contains(&run.number) {
                next += 1;
                renumbered.push(Renumbered {
                    from: run.number,
                    to: next,
                    finished_at: run.finished_at,
                });
                run.number = next;
            }
        }
        merged.last_run_number = next;
        merged.runs.truncate(MAX_RUNS);
        bound_keys(&mut merged.deleted_runs);
        bound_keys(&mut merged.legacy_imported);

        merge_models(&mut merged.models, &disk.models);

        merged.draft = match (&self.draft, &disk.draft) {
            (Some(ours), Some(theirs)) if theirs.revision > ours.revision => Some(theirs.clone()),
            (Some(ours), _) => Some(ours.clone()),
            // No draft here: keep theirs unless it is one this session cleared.
            (None, Some(theirs)) if self.cleared_draft.is_none_or(|r| theirs.revision > r) => {
                Some(theirs.clone())
            }
            (None, _) => None,
        };
        merged.schema_minor = STORE_SCHEMA_MINOR;
        merged.touch();
        Merged {
            store: merged,
            renumbered,
        }
    }

    /// Follow renumbering reported by a merge whose result this session did
    /// not adopt wholesale (its store had moved on meanwhile), so the next
    /// merge recognises those runs as the ones already on disk.
    pub fn apply_renumbering(&mut self, renumbered: &[Renumbered]) {
        for change in renumbered {
            let from = RunKey {
                number: change.from,
                finished_at: change.finished_at,
            };
            for run in &mut self.runs {
                if run_key(run) == from {
                    run.number = change.to;
                }
            }
            // A run deleted here after the merge renumbered it on disk: the
            // tombstone has to name the number the file now uses.
            if self.deleted_runs.contains(&from) {
                self.deleted_runs.insert(RunKey {
                    number: change.to,
                    finished_at: change.finished_at,
                });
            }
            self.last_run_number = self.last_run_number.max(change.to);
        }
        if !renumbered.is_empty() {
            self.touch();
        }
    }

    /// Import runs from a store written by an older build (the legacy file),
    /// without ever writing that file.
    ///
    /// Each legacy run is imported once: its legacy key goes into
    /// `legacy_imported`, which later reads consult, so a run renumbered on
    /// import is not imported again and one deleted after import stays
    /// deleted. A run whose number already names a different record here
    /// (live or tombstoned) is renumbered past everything this store has
    /// issued. Models are unioned; the draft is taken only when `migrating`
    /// (no store of this format existed yet) and none is held. Returns how
    /// many runs were imported.
    pub fn import_legacy(&mut self, legacy: &AppStore, migrating: bool) -> usize {
        self.last_run_number = self.last_run_number.max(legacy.last_run_number);
        let mut imported = 0;
        // Oldest first, so renumbered runs keep their relative order.
        for run in legacy.runs.iter().rev() {
            let key = run_key(run);
            if self.legacy_imported.contains(&key) || self.deleted_runs.contains(&key) {
                continue;
            }
            self.legacy_imported.insert(key);
            if self.runs.iter().any(|held| run_key(held) == key) {
                continue;
            }
            let mut run = run.clone();
            let collides = self.runs.iter().any(|held| held.number == run.number)
                || self
                    .deleted_runs
                    .iter()
                    .any(|tomb| tomb.number == run.number);
            if collides {
                run.number = self.next_run_number();
            }
            self.runs.push(run);
            imported += 1;
        }
        self.runs
            .sort_by_key(|run| std::cmp::Reverse((run.finished_at, run.number)));
        self.runs.truncate(MAX_RUNS);
        bound_keys(&mut self.legacy_imported);
        merge_models(&mut self.models, &legacy.models);
        if migrating && self.draft.is_none() && self.cleared_draft.is_none() {
            self.draft = legacy.draft.clone();
        }
        self.touch();
        imported
    }

    /// Write this session's store without clobbering another instance's.
    ///
    /// Takes an advisory exclusive lock on a sibling `<file>.lock`, re-reads
    /// the file under it, merges ([`AppStore::merged_with`]) and writes
    /// atomically. Returns the merged store, which the caller should adopt so
    /// its in-memory state matches the file.
    ///
    /// Refuses, leaving the file alone, when it cannot be read or was written
    /// by a newer build.
    pub fn save_merged(&self, path: impl AsRef<Path>) -> std::io::Result<Merged> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let _lock = lock_store(path)?;
        let disk = load(path).map_err(|error| std::io::Error::other(error.to_string()))?;
        if disk.written_by_newer_build() {
            return Err(std::io::Error::other(format!(
                "{} was written by a newer Ember (schema minor {}); not overwriting it",
                path.display(),
                disk.schema_minor
            )));
        }
        let merged = self.merged_with(&disk);
        atomic_write(path, &merged.store.to_bytes())?;
        Ok(merged)
    }

    /// Flip a run's pinned state. Ordering reads `pinned` on every render, so
    /// this is the whole feature; `false` means no such run.
    pub fn toggle_pin(&mut self, number: u64) -> bool {
        match self.runs.iter_mut().find(|run| run.number == number) {
            Some(run) => {
                run.pinned = !run.pinned;
                self.touch();
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
        stamped.schema_minor = STORE_SCHEMA_MINOR;
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

/// Held while a store is read, merged and written; dropping it unlocks.
struct StoreLock {
    _file: std::fs::File,
}

/// Take the advisory lock on `<store>.lock`, blocking until it is free.
///
/// The lock lives in a sibling file because the store itself is replaced by
/// rename on every write, and a lock on a replaced inode excludes nobody. On
/// platforms without `flock` this only creates the file.
fn lock_store(path: &Path) -> std::io::Result<StoreLock> {
    let mut name = path
        .file_name()
        .ok_or_else(|| std::io::Error::other(format!("{} has no filename", path.display())))?
        .to_os_string();
    name.push(".lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path.with_file_name(name))?;
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    rustix::fs::flock(&file, rustix::fs::FlockOperation::LockExclusive)?;
    Ok(StoreLock { _file: file })
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

/// Ember's per-user configuration directory: `$XDG_CONFIG_HOME/ember`, then
/// `~/.config/ember`, then the temp directory so a sandboxed or read-only home
/// still works. Shared by the store and the console's appearance settings.
pub fn config_dir() -> PathBuf {
    if let Some(root) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(root).join("ember");
    }
    if let Some(root) = std::env::var_os("HOME") {
        return PathBuf::from(root).join(".config/ember");
    }
    std::env::temp_dir().join("ember")
}

/// Where the store lives, inside [`config_dir`]. A different name from the
/// legacy file on purpose: see the module docs.
pub fn store_path() -> PathBuf {
    config_dir().join("app-state.v2.json")
}

/// Where builds before the v2 file kept history, inside [`config_dir`].
/// Read-only for this build.
pub fn legacy_store_path() -> PathBuf {
    config_dir().join("app-state.json")
}

/// A store opened with the legacy file folded in.
#[derive(Debug)]
pub struct Opened {
    pub store: AppStore,
    /// Legacy runs imported by this open.
    pub imported: usize,
    /// The legacy file exists but could not be read. It is left alone and
    /// does not stop the store from being used.
    pub legacy_error: Option<StoreError>,
}

/// Open the store at `path`, importing runs from the legacy file at
/// `legacy` ([`AppStore::import_legacy`]).
///
/// Holds the store's lock while reading, importing and -- when anything was
/// imported -- writing `path`, so two instances starting together import a
/// legacy run once rather than each under a different number. The legacy file
/// is only ever read. A store from a newer build is imported into in memory
/// only and not written. Fails, like [`load`], when `path` cannot be read.
pub fn open(path: impl AsRef<Path>, legacy: impl AsRef<Path>) -> Result<Opened, StoreError> {
    let path = path.as_ref();
    let legacy = legacy.as_ref();
    // Without a lock (an unwritable config directory) the import still
    // happens in memory; the first save reports why nothing can be written.
    let lock = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| lock_store(path))
        .ok();
    let migrating = !path.exists();
    let mut store = load(path)?;
    let (imported, legacy_error) = match load(legacy) {
        Ok(old) => (store.import_legacy(&old, migrating), None),
        Err(error) => (0, Some(error)),
    };
    if imported > 0 && lock.is_some() && !store.written_by_newer_build() {
        // A failed write loses nothing: the legacy file is still there and
        // the next open imports again.
        let _ = atomic_write(path, &store.to_bytes());
    }
    drop(lock);
    Ok(Opened {
        store,
        imported,
        legacy_error,
    })
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
        dir.join("app-state.v2.json")
    }

    /// The legacy file beside a [`temp_path`] store.
    fn legacy_beside(path: &Path) -> PathBuf {
        path.with_file_name("app-state.json")
    }

    /// Write `store` as an old build would have: no field it did not know.
    fn write_as_old_build(path: &Path, store: &AppStore) {
        let mut value = serde_json::to_value(store).unwrap();
        let object = value.as_object_mut().unwrap();
        for newer in [
            "schema_minor",
            "last_run_number",
            "deleted_runs",
            "legacy_imported",
        ] {
            object.remove(newer);
        }
        for run in object["runs"].as_array_mut().unwrap() {
            let run = run.as_object_mut().unwrap();
            run.remove("config");
            run.remove("result");
        }
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    }

    fn keys(store: &AppStore) -> Vec<(u64, i64)> {
        store
            .runs
            .iter()
            .map(|run| (run.number, run.finished_at))
            .collect()
    }

    #[test]
    fn a_run_deleted_in_another_instance_stays_deleted() {
        let path = temp_path("tombstones");
        let mut seeded = AppStore::default();
        seeded.push_run(run(1, 100, false));
        seeded.push_run(run(2, 200, false));
        seeded.write(&path).unwrap();

        // Both instances loaded the file with both runs.
        let mut first = load(&path).unwrap();
        let mut second = load(&path).unwrap();
        assert!(first.remove_run(1));
        first.save_merged(&path).unwrap();

        // The second never deleted run 1, still holds it, and saves.
        assert!(second.toggle_pin(2));
        let merged = second.save_merged(&path).unwrap();
        assert_eq!(
            keys(&merged.store),
            vec![(2, 200)],
            "the other instance's delete wins over our stale copy"
        );
        let on_disk = load(&path).unwrap();
        assert_eq!(keys(&on_disk), vec![(2, 200)]);
        assert!(on_disk.runs[0].pinned);
        assert!(on_disk.deleted_runs.contains(&RunKey {
            number: 1,
            finished_at: 100
        }));

        // And a third instance that still held it cannot bring it back.
        let mut stale = AppStore::default();
        stale.push_run(run(1, 100, false));
        assert_eq!(keys(&stale.merged_with(&on_disk).store), vec![(2, 200)]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn tombstones_are_bounded_to_the_most_recent() {
        let mut store = AppStore::default();
        for index in 0..(MAX_TOMBSTONES as u64 + 10) {
            store.push_run(run(index + 1, index as i64, false));
            assert!(store.remove_run(index + 1));
        }
        assert_eq!(store.deleted_runs.len(), MAX_TOMBSTONES);
        assert!(!store.deleted_runs.iter().any(|key| key.finished_at < 10));
        let merged = store.merged_with(&store.clone()).store;
        assert_eq!(merged.deleted_runs.len(), MAX_TOMBSTONES);
    }

    #[test]
    fn the_first_open_migrates_the_legacy_file_and_never_writes_it() {
        let path = temp_path("migrate");
        let legacy = legacy_beside(&path);
        let mut old = AppStore::default();
        old.push_run(run(1, 100, false));
        old.push_run(run(2, 200, true));
        old.touch_model("/old.gguf", 50);
        old.draft = Some(draft(2, "old draft"));
        write_as_old_build(&legacy, &old);
        let legacy_bytes = std::fs::read(&legacy).unwrap();

        let opened = open(&path, &legacy).unwrap();
        assert_eq!(opened.imported, 2);
        assert!(opened.legacy_error.is_none());
        assert_eq!(keys(&opened.store), vec![(2, 200), (1, 100)]);
        assert!(opened.store.runs[0].pinned);
        assert_eq!(opened.store.model_last_used("/old.gguf"), Some(50));
        assert_eq!(opened.store.draft.as_ref().unwrap().prompt, "old draft");
        // The migration is written to this build's own file ...
        assert_eq!(keys(&load(&path).unwrap()), vec![(2, 200), (1, 100)]);
        // ... and the legacy file is exactly as the old build left it.
        assert_eq!(std::fs::read(&legacy).unwrap(), legacy_bytes);

        // Opening again imports nothing new.
        let again = open(&path, &legacy).unwrap();
        assert_eq!(again.imported, 0);
        assert_eq!(keys(&again.store), vec![(2, 200), (1, 100)]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn runs_from_an_old_build_keep_appearing_renumbered_once() {
        let path = temp_path("legacy-later");
        let legacy = legacy_beside(&path);
        let mut old = AppStore::default();
        old.push_run(run(1, 100, false));
        write_as_old_build(&legacy, &old);

        // This build migrates, then records run 2 of its own, with a result.
        let mut session = open(&path, &legacy).unwrap().store;
        let number = session.next_run_number();
        let mut ours = run(number, 300, false);
        ours.result = Some(RecordResult {
            baseline_text: "kept".into(),
            intervention_text: "kept".into(),
            layers: Vec::new(),
            tokens: Vec::new(),
            first_layer_divergence: None,
            peak_layer: None,
            peak_relative_l2: None,
            tokens_equal: true,
        });
        session.push_run(ours);
        session.save_merged(&path).unwrap();

        // An old build, meanwhile, runs its own run 2 and rewrites its file.
        old.push_run(run(2, 250, false));
        write_as_old_build(&legacy, &old);

        let opened = open(&path, &legacy).unwrap();
        assert_eq!(opened.imported, 1);
        // Its run 2 collided with ours and was renumbered; ours is intact.
        assert_eq!(keys(&opened.store), vec![(2, 300), (3, 250), (1, 100)]);
        assert!(opened.store.runs[0].result.is_some());
        // Reading the legacy file again does not import it a second time.
        let reopened = open(&path, &legacy).unwrap();
        assert_eq!(reopened.imported, 0);
        assert_eq!(keys(&reopened.store), keys(&opened.store));

        // Deleting an imported run sticks, though the legacy file still has it.
        let mut session = reopened.store;
        assert!(session.remove_run(3));
        session.save_merged(&path).unwrap();
        let after_delete = open(&path, &legacy).unwrap();
        assert_eq!(keys(&after_delete.store), vec![(2, 300), (1, 100)]);

        // An old build that emptied its file cannot touch this one.
        write_as_old_build(&legacy, &AppStore::default());
        std::fs::write(&legacy, b"").unwrap();
        let survived = open(&path, &legacy).unwrap();
        assert!(survived.legacy_error.is_some());
        assert_eq!(keys(&survived.store), vec![(2, 300), (1, 100)]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn store_paths_are_separate() {
        assert_ne!(store_path(), legacy_store_path());
        assert_eq!(store_path().parent(), legacy_store_path().parent());
    }

    #[test]
    fn the_generation_moves_with_the_runs_only() {
        let mut store = AppStore::default();
        let start = store.generation();
        store.touch_model("/a.gguf", 1);
        store.draft = Some(draft(1, "d"));
        assert_eq!(store.generation(), start, "not a run change");
        store.push_run(run(1, 1, false));
        let pushed = store.generation();
        assert_ne!(pushed, start);
        assert!(!store.remove_run(9));
        assert_eq!(store.generation(), pushed, "nothing removed");
        assert!(store.toggle_pin(1));
        assert_ne!(store.generation(), pushed);
        let clone = store.clone();
        assert_eq!(clone, store);
        let merged = store.merged_with(&clone).store;
        assert_eq!(merged.runs, store.runs);
        assert_ne!(merged.generation(), store.generation());
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
        std::fs::write(
            &path,
            serde_json::to_vec(&legacy).expect("serialize legacy store"),
        )
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
            layers: vec![RecordLayer {
                layer: 7,
                relative_l2: Some(1.25),
                cosine: None,
            }],
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

    #[test]
    fn run_numbers_continue_from_a_legacy_store() {
        // A store written before `last_run_number` existed: numbering resumes
        // from the highest stored run instead of restarting at 1.
        let mut store = AppStore::default();
        for number in [3, 9, 4] {
            store.push_run(run(number, 0, false));
        }
        assert_eq!(store.next_run_number(), 10);
        assert_eq!(store.next_run_number(), 11);
    }

    #[test]
    fn deleting_the_newest_run_does_not_free_its_number() {
        let mut store = AppStore::default();
        for _ in 0..3 {
            let number = store.next_run_number();
            store.push_run(run(number, 0, false));
        }
        assert!(store.remove_run(3));
        // Survives a round trip through disk, i.e. across launches.
        let path = temp_path("run-counter");
        store.write(&path).unwrap();
        let mut reloaded = load(&path).unwrap();
        assert_eq!(reloaded.next_run_number(), 4, "3 named a deleted run");
        let _ = std::fs::remove_file(&path);
    }

    fn draft(revision: u64, prompt: &str) -> Draft {
        Draft {
            revision,
            prompt: prompt.into(),
            model_path: "/m.gguf".into(),
            fields: Default::default(),
            step: "prompt".into(),
            updated_at: 0,
        }
    }

    #[test]
    fn two_instances_keep_each_others_runs() {
        let path = temp_path("two-instances");
        // Both instances start from the same (empty) file.
        let mut first = load(&path).unwrap();
        let mut second = load(&path).unwrap();
        let number = first.next_run_number();
        first.push_run(run(number, 100, false));
        let number = second.next_run_number();
        second.push_run(run(number, 200, false));
        assert_eq!(number, 1, "both instances issued run 1");

        first.save_merged(&path).unwrap();
        let merged = second.save_merged(&path).unwrap();

        let on_disk = load(&path).unwrap();
        assert_eq!(
            on_disk.runs.len(),
            2,
            "the second write kept the first's run"
        );
        assert_eq!(merged.store.runs, on_disk.runs);
        // The second instance's run 1 collided with the first's and moved.
        assert_eq!(
            merged.renumbered,
            vec![Renumbered {
                from: 1,
                to: 2,
                finished_at: 200
            }]
        );
        let numbers: Vec<(u64, i64)> = on_disk
            .runs
            .iter()
            .map(|run| (run.number, run.finished_at))
            .collect();
        assert_eq!(numbers, vec![(2, 200), (1, 100)]);
        assert_eq!(on_disk.last_run_number, 2);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn our_version_of_a_shared_run_wins_and_deletions_stick() {
        let path = temp_path("pins-deletes");
        let mut seeded = AppStore::default();
        seeded.push_run(run(1, 100, false));
        seeded.push_run(run(2, 200, false));
        seeded.write(&path).unwrap();

        let mut session = load(&path).unwrap();
        assert!(session.toggle_pin(2));
        assert!(session.remove_run(1));
        let merged = session.save_merged(&path).unwrap();

        let on_disk = load(&path).unwrap();
        assert_eq!(
            on_disk.runs.len(),
            1,
            "run 1 was deleted and must not return"
        );
        assert!(on_disk.runs[0].pinned, "our pin wins over the file's copy");
        assert_eq!(merged.store.runs, on_disk.runs);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn models_union_by_latest_use_and_the_higher_draft_revision_wins() {
        let mut ours = AppStore::default();
        ours.touch_model("/a.gguf", 10);
        ours.touch_model("/b.gguf", 50);
        ours.draft = Some(draft(3, "ours"));
        let mut theirs = AppStore::default();
        theirs.touch_model("/a.gguf", 30);
        theirs.touch_model("/c.gguf", 5);
        theirs.draft = Some(draft(4, "theirs"));
        theirs.last_run_number = 17;

        let merged = ours.merged_with(&theirs).store;
        assert_eq!(merged.model_last_used("/a.gguf"), Some(30));
        assert_eq!(merged.model_last_used("/b.gguf"), Some(50));
        assert_eq!(merged.model_last_used("/c.gguf"), Some(5));
        assert_eq!(merged.draft.as_ref().unwrap().prompt, "theirs");
        assert_eq!(merged.last_run_number, 17);

        // A stale draft from elsewhere does not replace a newer one here.
        ours.draft = Some(draft(9, "ours"));
        assert_eq!(
            ours.merged_with(&theirs).store.draft.unwrap().prompt,
            "ours"
        );
    }

    #[test]
    fn a_cleared_draft_stays_cleared_but_a_newer_one_elsewhere_survives() {
        let mut ours = AppStore {
            draft: Some(draft(4, "mine")),
            ..AppStore::default()
        };
        ours.clear_draft();
        let theirs = AppStore {
            draft: Some(draft(4, "mine")),
            ..AppStore::default()
        };
        assert!(ours.merged_with(&theirs).store.draft.is_none());
        let newer = AppStore {
            draft: Some(draft(5, "newer")),
            ..AppStore::default()
        };
        assert_eq!(
            ours.merged_with(&newer).store.draft.unwrap().prompt,
            "newer"
        );
        // An instance that never held a draft does not erase one.
        assert!(AppStore::default()
            .merged_with(&newer)
            .store
            .draft
            .is_some());
        assert_eq!(ours.next_draft_revision(), 5);
    }

    #[test]
    fn a_merge_respects_the_retention_bound() {
        let mut ours = AppStore::default();
        let mut theirs = AppStore::default();
        for index in 0..MAX_RUNS as u64 {
            ours.push_run(run(index + 1, index as i64 * 2, false));
            theirs.push_run(run(index + 1_000, index as i64 * 2 + 1, false));
        }
        let merged = ours.merged_with(&theirs).store;
        assert_eq!(merged.runs.len(), MAX_RUNS);
        assert_eq!(merged.runs[0].finished_at, (MAX_RUNS as i64 - 1) * 2 + 1);
    }

    #[test]
    fn a_newer_minor_is_readable_but_never_overwritten() {
        let path = temp_path("newer-minor");
        let mut store = AppStore::default();
        store.push_run(run(1, 1, false));
        store.write(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let bumped = text.replace(
            &format!("\"schema_minor\": {STORE_SCHEMA_MINOR}"),
            "\"schema_minor\": 99, \"field_from_the_future\": true",
        );
        assert_ne!(text, bumped);
        std::fs::write(&path, &bumped).unwrap();

        let loaded = load(&path).expect("a newer minor still reads");
        assert_eq!(loaded.runs.len(), 1);
        assert!(loaded.written_by_newer_build());
        assert!(loaded.save_merged(&path).is_err());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            bumped,
            "the newer build's fields survive"
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_legacy_file_is_minor_zero_and_is_upgraded_on_write() {
        let path = temp_path("minor-zero");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            format!("{{\"schema\":\"{STORE_SCHEMA}\",\"schema_version\":{STORE_SCHEMA_MAJOR}}}"),
        )
        .unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.schema_minor, 0);
        assert!(!loaded.written_by_newer_build());
        loaded.save_merged(&path).unwrap();
        assert_eq!(load(&path).unwrap().schema_minor, STORE_SCHEMA_MINOR);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn a_write_waits_for_the_lock() {
        let path = temp_path("lock");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let held = lock_store(&path).unwrap();
        let writer = {
            let path = path.clone();
            std::thread::spawn(move || {
                let mut store = AppStore::default();
                store.push_run(run(1, 1, false));
                store.save_merged(&path).unwrap();
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(
            !path.exists(),
            "the writer went ahead while the lock was held"
        );
        drop(held);
        writer.join().unwrap();
        assert_eq!(load(&path).unwrap().runs.len(), 1);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
