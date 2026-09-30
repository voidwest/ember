//! v0.5 offline bundle verification (contract sections 8, 16).
//!
//! Basic verification requires no internet and no model file. Deep
//! verification additionally checks the model/tokenizer files and
//! execution-plan compatibility against the loaded model.

use crate::v05::hook::SemanticHookSite;
use crate::v05::manifest::{
    sha256_hex, BundleIdentity, BundleManifest, SemanticManifest, BUNDLE_KIND, BUNDLE_SCHEMA_V1,
};
use crate::v05::safetensors::{self, TensorView};
use crate::v05::token_select::{CoverageKind, TokenSelectionRecord};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// One indexed capture tensor (captures/index.jsonl line).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureIndexEntry {
    pub capture_id: String,
    pub input_id: String,
    pub site: SemanticHookSite,
    pub layer: usize,
    pub positions: Vec<usize>,
    /// Payload tensor name; empty for summary-only entries.
    #[serde(default)]
    pub tensor_name: String,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default)]
    pub dtype: String,
    #[serde(default)]
    pub byte_length: usize,
    #[serde(default)]
    pub checksum: String,
    #[serde(default)]
    pub model_sha256: String,
    #[serde(default)]
    pub plan_hash: String,
    #[serde(default)]
    pub hook_route: String,
    #[serde(default)]
    pub fusion: String,
    #[serde(default)]
    pub selection_provenance: serde_json::Value,
    /// Deterministic summary statistics for summary-only captures.
    #[serde(default)]
    pub summary: Option<SummaryEntry>,
}

/// Deterministic summary statistics recorded for a summary-only capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryEntry {
    pub shape: Vec<usize>,
    pub finite_count: usize,
    pub minimum: f32,
    pub maximum: f32,
    pub mean: f64,
    pub l2_norm: f64,
}

/// One verification check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

/// The verification report.
///
/// Verification never writes into the bundle it checks; callers that want a
/// report on disk write it themselves (`ember experiment verify
/// --write-report <path>`), outside the bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub bundle_schema: String,
    pub ok: bool,
    pub semantic_hash: String,
    pub payload_hash: String,
    pub checks: Vec<CheckResult>,
    pub warnings: Vec<String>,
    /// Verification timestamp (runtime metadata; excluded from hashes).
    pub timestamp: String,
}

impl VerificationReport {
    fn new(semantic_hash: String, payload_hash: String) -> VerificationReport {
        VerificationReport {
            bundle_schema: BUNDLE_SCHEMA_V1.to_string(),
            ok: true,
            semantic_hash,
            payload_hash,
            checks: Vec::new(),
            warnings: Vec::new(),
            timestamp: String::new(),
        }
    }

    fn record(&mut self, name: &str, ok: bool, detail: String) {
        if !ok {
            self.ok = false;
        }
        self.checks.push(CheckResult {
            name: name.to_string(),
            ok,
            detail,
        });
    }

    /// Add a check computed outside the bundle verifier (for example an
    /// external anchor such as a signed evidence envelope). A failed check
    /// fails the report.
    pub fn add_check(&mut self, name: &str, ok: bool, detail: String) {
        self.record(name, ok, detail);
    }

    fn finished(mut self) -> VerificationReport {
        self.timestamp = now_iso8601();
        self
    }
}

/// A verified bundle loaded for cross-bundle sources, comparison and
/// reproduction.
///
/// Every document verification parsed is kept here as the exact bytes that
/// were hashed, so callers never re-read a file from disk after it was
/// checked (a bundle modified between verification and use cannot substitute
/// different content).
pub struct LoadedBundle {
    pub root: PathBuf,
    /// The top-level `manifest.json`.
    pub manifest: BundleManifest,
    pub semantic_manifest: SemanticManifest,
    pub capture_index: Vec<CaptureIndexEntry>,
    /// Recomputed semantic hash (verification proved it equals the stored
    /// one; for legacy encodings it is the preserved historical hash).
    pub semantic_hash: String,
    /// Recomputed payload hash.
    pub payload_hash: String,
    files: BTreeMap<String, Vec<u8>>,
    payload_bytes: Vec<u8>,
    tensors: Vec<(String, TensorView)>,
}

impl LoadedBundle {
    /// Load one indexed tensor as f32.
    pub fn tensor_f32_by_name(&self, name: &str) -> Result<Vec<f32>, String> {
        let (_, view) = self
            .tensors
            .iter()
            .find(|(tensor_name, _)| tensor_name == name)
            .ok_or_else(|| format!("tensor '{name}' not found in the payload"))?;
        safetensors::tensor_f32(&self.payload_bytes, view)
    }

    /// The verified bytes of one bundle document (for example
    /// `outputs.jsonl` or `resolved-experiment.json`). Only documents the
    /// verifier reads are retained; the tensor payload is available through
    /// [`LoadedBundle::tensor_f32_by_name`].
    pub fn file(&self, relative: &str) -> Option<&[u8]> {
        self.files.get(relative).map(Vec::as_slice)
    }

    /// Like [`LoadedBundle::file`], with an error naming the bundle.
    pub fn required_file(&self, relative: &str) -> Result<&[u8], String> {
        self.file(relative).ok_or_else(|| {
            format!(
                "bundle '{}' has no verified '{relative}'",
                self.root.display()
            )
        })
    }
}

/// Verify a bundle and load its payloads for source use.
///
/// The source bundle must pass full basic verification before its tensors
/// can back an intervention.
pub fn load_bundle_for_source(root: &Path) -> Result<LoadedBundle, String> {
    load_verified_bundle(root, &VerifyOptions::default())
}

/// Verify a bundle with `options` (for example an expected semantic hash)
/// and return its verified contents. Fails unless every check passes.
pub fn load_verified_bundle(root: &Path, options: &VerifyOptions) -> Result<LoadedBundle, String> {
    check_not_staging(root)?;
    let (report, contents) = verify_and_load(root, options)?;
    if !report.ok {
        let errors: Vec<String> = report
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| format!("{}: {}", check.name, check.detail))
            .collect();
        return Err(format!(
            "bundle '{}' failed verification: {}",
            root.display(),
            errors.join("; ")
        ));
    }
    let contents =
        contents.ok_or_else(|| format!("bundle '{}' was not fully read", root.display()))?;
    Ok(LoadedBundle {
        root: root.to_path_buf(),
        manifest: contents.manifest,
        semantic_manifest: contents.semantic_manifest,
        capture_index: contents.capture_index,
        semantic_hash: report.semantic_hash,
        payload_hash: report.payload_hash,
        files: contents.files,
        payload_bytes: contents.payload_bytes,
        tensors: contents.tensors,
    })
}

/// The resolved experiment a reproduction may run, bound to the bundle's
/// verified identity.
///
/// `resolved-experiment.json` sits outside the semantic hash (it records
/// placement such as the output directory), so it is not trusted as is. It
/// is accepted only when it equals what the hashed `experiment.toml`
/// resolves to, apart from the run-time choices a user can override without
/// editing the spec (execution mode and thread count, output directory)
/// and the informational list of applied defaults, and when everything it shares with the semantic manifest agrees:
/// experiment metadata, execution mode and determinism, inputs, captures and
/// interventions. The tokenizer and model paths it names come from the
/// bundle, so callers must still pin them to the recorded SHA-256 values.
pub fn bound_resolved_experiment(
    bundle: &LoadedBundle,
) -> Result<crate::v05::spec::ExperimentSpecV1, String> {
    let spec_text = std::str::from_utf8(bundle.required_file("experiment.toml")?)
        .map_err(|error| format!("bundle experiment.toml is not UTF-8: {error}"))?;
    bind_resolved_experiment(
        spec_text,
        bundle.required_file("resolved-experiment.json")?,
        &bundle.semantic_manifest,
    )
}

fn bind_resolved_experiment(
    spec_text: &str,
    resolved_bytes: &[u8],
    semantic: &SemanticManifest,
) -> Result<crate::v05::spec::ExperimentSpecV1, String> {
    use crate::v05::spec::{ExperimentSpecV1, RawExperimentSpec};
    let stored: ExperimentSpecV1 = serde_json::from_slice(resolved_bytes)
        .map_err(|error| format!("bundle resolved-experiment.json is malformed: {error}"))?;
    let mut derived = RawExperimentSpec::from_toml_str(spec_text)
        .and_then(|raw| raw.resolve())
        .map_err(|error| format!("bundle experiment.toml does not resolve: {error}"))?;
    derived.execution.mode = stored.execution.mode;
    derived.execution.threads = stored.execution.threads;
    derived.output.directory = stored.output.directory.clone();
    // The list of defaults applied is a record of how resolution went, not
    // an input to execution, and its contents vary across Ember releases.
    derived.defaults = stored.defaults.clone();
    if derived != stored {
        let derived_value = serde_json::to_value(&derived).map_err(|error| error.to_string())?;
        let stored_value = serde_json::to_value(&stored).map_err(|error| error.to_string())?;
        let differing: Vec<String> = derived_value
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(key, value)| stored_value.get(key.as_str()) != Some(*value))
            .map(|(key, _)| key.clone())
            .collect();
        return Err(format!(
            "resolved-experiment.json does not match the hashed experiment.toml (differs in: \
             {}); refusing to run an unbound specification",
            differing.join(", ")
        ));
    }
    let mut mismatches = Vec::new();
    if stored.experiment.name != semantic.experiment.name
        || stored.experiment.description != semantic.experiment.description
        || stored.experiment.seed != semantic.experiment.seed
    {
        mismatches.push("experiment");
    }
    if stored.execution.mode.name() != semantic.execution.mode
        || stored.execution.deterministic != semantic.execution.deterministic
    {
        mismatches.push("execution");
    }
    let inputs_match = stored.inputs.len() == semantic.inputs.len()
        && stored
            .inputs
            .iter()
            .zip(&semantic.inputs)
            .all(|(input, meta)| {
                input.id == meta.id && sha256_hex(input.text.as_bytes()) == meta.prompt_hash
            });
    if !inputs_match {
        mismatches.push("inputs");
    }
    if stored.captures != semantic.captures {
        mismatches.push("captures");
    }
    if stored.interventions != semantic.interventions {
        mismatches.push("interventions");
    }
    if !mismatches.is_empty() {
        return Err(format!(
            "resolved-experiment.json disagrees with the semantic manifest in: {}",
            mismatches.join(", ")
        ));
    }
    Ok(stored)
}

/// Options for bundle verification.
#[derive(Debug, Clone, Default)]
pub struct VerifyOptions {
    /// Deep verification: check model/tokenizer files when supplied.
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    /// External anchor: the semantic hash the caller obtained from a trusted
    /// source. Without an anchor, "verified" means only that the bundle is
    /// self-consistent, since every hash it checks is bundle-authored.
    pub expected_semantic_hash: Option<String>,
}

/// Files the verifier reads fully into memory. Each is read exactly once;
/// the same bytes are hashed, parsed, and returned in [`LoadedBundle`].
const REQUIRED_FILES: [&str; 15] = [
    "semantic-manifest.json",
    "runtime.json",
    "experiment.toml",
    "resolved-experiment.json",
    "model.json",
    "tokenizer.json",
    "execution-plan.json",
    "inputs.jsonl",
    "outputs.jsonl",
    "tokenization.jsonl",
    "captures/tensors.safetensors",
    "captures/index.jsonl",
    "interventions/events.jsonl",
    "traces/events.jsonl",
    "checksums.sha256",
];

const PAYLOAD_FILE: &str = "captures/tensors.safetensors";

/// Bundle files under this prefix are optional deterministic artifacts
/// (directions, analysis reports), read and checked when present.
const ARTIFACTS_PREFIX: &str = "artifacts/";

/// A report file that older Ember versions wrote into the bundle itself.
/// It is tolerated in the inventory but never read: its contents cannot
/// influence verification.
const LEGACY_REPORT_FILE: &str = "verification.json";

/// Verify a bundle fully offline (unless deep verification is requested).
pub fn verify_bundle(root: &Path, options: &VerifyOptions) -> Result<VerificationReport, String> {
    check_not_staging(root)?;
    verify_staged_bundle(root, options)
}

fn check_not_staging(root: &Path) -> Result<(), String> {
    if root
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.') && name.contains(".tmp-"))
    {
        return Err("incomplete staging directory is not a published bundle".into());
    }
    Ok(())
}

/// Used only by the writer before atomic publication. All content checks are
/// identical to public verification; only the staging-name guard is bypassed.
pub(crate) fn verify_staged_bundle(
    root: &Path,
    options: &VerifyOptions,
) -> Result<VerificationReport, String> {
    verify_and_load(root, options).map(|(report, _)| report)
}

/// Everything verification read, kept for [`LoadedBundle`].
struct VerifiedContents {
    manifest: BundleManifest,
    semantic_manifest: SemanticManifest,
    capture_index: Vec<CaptureIndexEntry>,
    files: BTreeMap<String, Vec<u8>>,
    payload_bytes: Vec<u8>,
    tensors: Vec<(String, TensorView)>,
}

/// SHA-256 of bundle files, each computed once: from the in-memory bytes
/// for documents the verifier parses, and by streaming for any other listed
/// file (which is only ever hashed, never interpreted).
struct FileHashes<'a> {
    root: &'a Path,
    present: &'a BTreeSet<String>,
    files: &'a BTreeMap<String, Vec<u8>>,
    cache: BTreeMap<String, String>,
}

impl FileHashes<'_> {
    /// `Ok(None)` when the file is absent.
    fn hash(&mut self, relative: &str) -> Result<Option<String>, String> {
        if let Some(hash) = self.cache.get(relative) {
            return Ok(Some(hash.clone()));
        }
        let hash = if let Some(bytes) = self.files.get(relative) {
            sha256_hex(bytes)
        } else if self.present.contains(relative) {
            hash_regular_file(self.root, relative)?
        } else {
            return Ok(None);
        };
        self.cache.insert(relative.to_string(), hash.clone());
        Ok(Some(hash))
    }
}

fn verify_and_load(
    root: &Path,
    options: &VerifyOptions,
) -> Result<(VerificationReport, Option<VerifiedContents>), String> {
    // Check entry types before opening even manifest.json: verification must
    // never follow a bundle-supplied symlink or block on a special file.
    let actual_files = regular_bundle_files(root)?;
    // ---- phase 1: manifest load + schema/basic identity checks ----
    let manifest_bytes = read_regular_file(root, "manifest.json")?;
    let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("manifest.json is not valid JSON: {error}"))?;
    let mut report = VerificationReport::new(String::new(), String::new());

    report.record(
        "bundle schema",
        manifest.bundle_schema == BUNDLE_SCHEMA_V1,
        format!("schema '{}'", manifest.bundle_schema),
    );
    report.record(
        "bundle kind",
        manifest.kind == BUNDLE_KIND,
        format!("kind '{}'", manifest.kind),
    );
    report.record(
        "bundle complete",
        manifest.status == "complete",
        format!("status '{}'", manifest.status),
    );
    let mut listed = BTreeSet::new();
    let mut inventory_errors = Vec::new();
    for name in &manifest.files {
        if crate::v05::bundle::validate_relative_path(name).is_err() {
            inventory_errors.push(format!("unsafe manifest path: {name}"));
        } else if name == LEGACY_REPORT_FILE {
            inventory_errors.push(format!("{LEGACY_REPORT_FILE} must not be a listed file"));
        } else if !listed.insert(name.clone()) {
            inventory_errors.push(format!("duplicate manifest path: {name}"));
        }
    }
    let mut expected_files = listed.clone();
    expected_files.insert("checksums.sha256".into());
    if actual_files.contains(LEGACY_REPORT_FILE) {
        // Written into bundles by Ember versions before 1.0. It is runtime
        // state outside every checksum, so it is ignored, never trusted.
        expected_files.insert(LEGACY_REPORT_FILE.into());
        report.warnings.push(format!(
            "{LEGACY_REPORT_FILE} in the bundle is ignored: it is unchecked runtime state from an \
             earlier verification and has no bearing on this result"
        ));
    }
    for name in actual_files.symmetric_difference(&expected_files) {
        inventory_errors.push(format!("file inventory mismatch: {name}"));
    }
    report.record(
        "bundle file inventory",
        inventory_errors.is_empty(),
        inventory_errors.join("; "),
    );

    // ---- phase 2: required files + checksum scan ----
    let missing: Vec<&str> = REQUIRED_FILES
        .iter()
        .copied()
        .filter(|relative| !actual_files.contains(*relative))
        .collect();
    report.record(
        "required files",
        missing.is_empty(),
        if missing.is_empty() {
            "all present".into()
        } else {
            format!("missing: {}", missing.join(", "))
        },
    );
    if !missing.is_empty() {
        report.ok = false;
        return Ok((report.finished(), None));
    }
    // Read every parsed document exactly once.
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for relative in REQUIRED_FILES {
        files.insert(relative.to_string(), read_regular_file(root, relative)?);
    }
    files.insert("manifest.json".to_string(), manifest_bytes);
    // Artifacts (directions, analysis reports) are interpreted below, so
    // they are read once here like the required documents.
    for relative in listed
        .iter()
        .filter(|name| name.starts_with(ARTIFACTS_PREFIX))
    {
        if actual_files.contains(relative) {
            files.insert(relative.clone(), read_regular_file(root, relative)?);
        }
    }

    let semantic_manifest = parse_semantic_manifest(&files["semantic-manifest.json"])?;
    report.record(
        "semantic manifest schema",
        semantic_manifest.bundle_schema == BUNDLE_SCHEMA_V1,
        format!("schema '{}'", semantic_manifest.bundle_schema),
    );
    report.record(
        "semantic manifest complete",
        semantic_manifest.complete,
        String::new(),
    );
    // A self-consistent hash cannot make an unknown contract interpretable.
    // Check independently versioned semantics before reading their payloads.
    report.record(
        "experiment schema",
        semantic_manifest.experiment_schema == crate::v05::spec::EXPERIMENT_SCHEMA_V1,
        semantic_manifest.experiment_schema.clone(),
    );
    report.record(
        "hook schema",
        semantic_manifest.hook_schema == crate::v05::hook::HOOK_SCHEMA_VERSION,
        semantic_manifest.hook_schema.to_string(),
    );
    report.record(
        "plan schema",
        semantic_manifest.plan_schema == crate::plan::PLAN_SCHEMA_VERSION,
        semantic_manifest.plan_schema.to_string(),
    );
    if !report.ok {
        return Ok((report.finished(), None));
    }

    // checksums: every file covered by checksums.sha256 must match. Keys
    // come from the untrusted bundle and are path-validated before use:
    // an absolute or `..`-bearing key must fail verification, never escape
    // the bundle root (traversal → arbitrary-file hash oracle / read).
    let checksums = parse_checksums(&files["checksums.sha256"])?;
    let mut hashes = FileHashes {
        root,
        present: &actual_files,
        files: &files,
        cache: BTreeMap::new(),
    };
    let mut checksum_mismatches: Vec<String> = Vec::new();
    for name in &listed {
        if !checksums.contains_key(name) {
            checksum_mismatches.push(format!("{name}: missing checksum"));
        }
    }
    for (relative, expected) in &checksums {
        let Ok(relative) = crate::v05::bundle::validate_relative_path(relative) else {
            checksum_mismatches.push(format!("{relative}: unsafe path in checksums"));
            continue;
        };
        if relative == LEGACY_REPORT_FILE || relative == "checksums.sha256" {
            checksum_mismatches.push(format!("{relative}: may not be checksummed"));
            continue;
        }
        match hashes.hash(relative)? {
            None => checksum_mismatches.push(format!("{relative}: missing file")),
            Some(actual) if actual != *expected => {
                checksum_mismatches.push(format!("{relative}: checksum mismatch"))
            }
            Some(_) => {}
        }
    }
    report.record(
        "checksums",
        checksum_mismatches.is_empty(),
        if checksum_mismatches.is_empty() {
            format!("{} files verified", checksums.len())
        } else {
            checksum_mismatches.join("; ")
        },
    );

    // capture index consistency
    let capture_index = parse_capture_index(&files["captures/index.jsonl"])?;
    let mut index_errors: Vec<String> = Vec::new();
    let mut seen_ids: BTreeMap<(String, String, SemanticHookSite, usize), usize> = BTreeMap::new();
    for entry in &capture_index {
        *seen_ids
            .entry((
                entry.capture_id.clone(),
                entry.input_id.clone(),
                entry.site,
                entry.layer,
            ))
            .or_insert(0) += 1;
        if entry.summary.is_some() {
            // Summary-only entries carry no payload.
            continue;
        }
        if entry.tensor_name.is_empty() {
            index_errors.push(format!("'{}': empty tensor name", entry.capture_id));
        }
        let expected_bytes: usize = entry
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .and_then(|count| count.checked_mul(dtype_bytes(&entry.dtype)?))
            .unwrap_or(0);
        if expected_bytes != entry.byte_length {
            index_errors.push(format!(
                "'{}': shape {:?} {} implies {expected_bytes} bytes but index says {}",
                entry.capture_id, entry.shape, entry.dtype, entry.byte_length
            ));
        }
    }
    let duplicate_ids: Vec<String> = seen_ids
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|((capture_id, input_id, site, layer), count)| {
            format!("{capture_id}/{input_id}/{site}/layer-{layer} x{count}")
        })
        .collect();
    if !duplicate_ids.is_empty() {
        index_errors.push(format!(
            "duplicate capture ids in index: {}",
            duplicate_ids.join(", ")
        ));
    }
    report.record(
        "capture index",
        index_errors.is_empty(),
        if index_errors.is_empty() {
            format!("{} entries", capture_index.len())
        } else {
            index_errors.join("; ")
        },
    );

    // payload: shapes/dtypes match the index; no unindexed tensors. The
    // bytes checked here are the bytes checksummed above and returned to
    // callers.
    let payload_bytes = &files[PAYLOAD_FILE];
    let payload_tensors = safetensors::deserialize(payload_bytes)?;
    let mut payload_errors: Vec<String> = Vec::new();
    // Two index entries naming one tensor would let the map below keep only
    // the last, so the other entry's shape, dtype and checksum would never be
    // compared with anything.
    let mut tensor_names = std::collections::BTreeSet::new();
    for entry in capture_index.iter().filter(|entry| entry.summary.is_none()) {
        if !tensor_names.insert(entry.tensor_name.as_str()) {
            payload_errors.push(format!(
                "tensor '{}' is indexed more than once",
                entry.tensor_name
            ));
        }
    }
    let indexed_names: BTreeMap<&str, &CaptureIndexEntry> = capture_index
        .iter()
        .map(|entry| (entry.tensor_name.as_str(), entry))
        .collect();
    for (name, view) in &payload_tensors {
        let Some(entry) = indexed_names.get(name.as_str()) else {
            payload_errors.push(format!("unindexed tensor '{name}' in payload"));
            continue;
        };
        if view.shape != entry.shape {
            payload_errors.push(format!(
                "'{name}': payload shape {:?} != index shape {:?}",
                view.shape, entry.shape
            ));
        }
        if view.dtype.name() != entry.dtype {
            payload_errors.push(format!(
                "'{name}': payload dtype {} != index dtype {}",
                view.dtype.name(),
                entry.dtype
            ));
        }
        let raw = &payload_bytes[view.data_offsets.0..view.data_offsets.1];
        if sha256_hex(raw) != entry.checksum {
            payload_errors.push(format!("'{name}': tensor checksum mismatch"));
        }
    }
    for entry in &capture_index {
        if entry.summary.is_some() {
            continue;
        }
        if !payload_tensors
            .iter()
            .any(|(name, _)| name == &entry.tensor_name)
        {
            payload_errors.push(format!(
                "indexed tensor '{}' missing from payload",
                entry.tensor_name
            ));
        }
    }
    report.record(
        "tensor payload",
        payload_errors.is_empty(),
        if payload_errors.is_empty() {
            format!("{} tensors verified", payload_tensors.len())
        } else {
            payload_errors.join("; ")
        },
    );

    // token-selection records internally consistent
    let mut selection_errors: Vec<String> = Vec::new();
    for (index, record) in semantic_manifest.token_selections.iter().enumerate() {
        if let Some(error) = selection_consistency(record) {
            selection_errors.push(format!("record {index}: {error}"));
        }
    }
    report.record(
        "token selection records",
        selection_errors.is_empty(),
        if selection_errors.is_empty() {
            format!("{} records", semantic_manifest.token_selections.len())
        } else {
            selection_errors.join("; ")
        },
    );

    // intervention references resolve
    let mut intervention_errors: Vec<String> = Vec::new();
    for (index, intervention) in semantic_manifest.interventions.iter().enumerate() {
        if let Some(crate::v05::intervention::InterventionSource::CaptureFromCurrentRun {
            capture_id,
        }) = &intervention.source
            && !semantic_manifest
                .captures
                .iter()
                .any(|capture| capture.id == *capture_id)
        {
            intervention_errors.push(format!(
                "intervention {index}: source capture '{capture_id}' not declared"
            ));
        }
    }
    report.record(
        "intervention references",
        intervention_errors.is_empty(),
        if intervention_errors.is_empty() {
            format!("{} interventions", semantic_manifest.interventions.len())
        } else {
            intervention_errors.join("; ")
        },
    );

    // direction artifacts agree with the spec and their tensors
    let listed_artifacts: Vec<String> = listed
        .iter()
        .filter(|name| name.starts_with(ARTIFACTS_PREFIX))
        .cloned()
        .collect();
    let has_directions = semantic_manifest.interventions.iter().any(|intervention| {
        intervention
            .source
            .as_ref()
            .is_some_and(|source| source.is_resolved_direction())
    }) || listed_artifacts
        .iter()
        .any(|name| name.starts_with(crate::v05::steering::DIRECTION_DIR));
    if has_directions {
        let direction_errors = crate::v05::steering::verify_direction_artifacts(
            &semantic_manifest,
            &listed_artifacts,
            &|relative| files.get(relative).cloned(),
        );
        report.record(
            "direction artifacts",
            direction_errors.is_empty(),
            if direction_errors.is_empty() {
                "every resolved direction matches its record and the spec".to_string()
            } else {
                direction_errors.join("; ")
            },
        );
    }

    // execution-plan hash matches the stored plan
    let plan: crate::plan::ExecutionPlan = serde_json::from_slice(&files["execution-plan.json"])
        .map_err(|error| format!("execution-plan.json is not valid JSON: {error}"))?;
    report.record(
        "execution plan schema",
        plan.schema_version == crate::plan::PLAN_SCHEMA_VERSION
            && plan.schema_version == semantic_manifest.plan_schema,
        format!(
            "stored {}, semantic {}",
            plan.schema_version, semantic_manifest.plan_schema
        ),
    );
    report.record(
        "execution plan identity",
        plan.plan_hash == semantic_manifest.execution.plan_hash,
        format!(
            "stored {}, semantic {}",
            plan.plan_hash, semantic_manifest.execution.plan_hash
        ),
    );
    let recomputed_plan_hash = crate::plan::plan_hash(&plan);
    let plan_matches = recomputed_plan_hash == plan.plan_hash;
    report.record(
        "execution plan hash",
        plan_matches,
        if plan_matches {
            format!("plan hash {}", short_hash(&plan.plan_hash))
        } else {
            format!(
                "stored {} != recomputed {}",
                plan.plan_hash, recomputed_plan_hash
            )
        },
    );

    // semantic hash + payload hash recompute
    let canonical_semantic_hash = BundleIdentity::semantic_hash(&semantic_manifest)?;
    let stored_semantic = manifest.semantic_hash.clone();
    let semantic_file = &files["semantic-manifest.json"];
    // Released 0.5/0.6 writers could inherit serde_json/preserve_order and
    // hash insertion order despite documenting sorted keys. Preserve those
    // identities explicitly; never rewrite historical payload bytes.
    let legacy_version = matches!(
        semantic_manifest.ember_version.as_str(),
        "0.5.0"
            | "0.5.1"
            | "0.6.0"
            | "0.6.1"
            | "0.6.2"
            | "0.6.3"
            | "0.6.4"
            | "0.6.5"
            | "0.6.6"
            | "0.6.7"
            | "0.6.8"
    );
    let legacy_hash = if legacy_version && canonical_semantic_hash != stored_semantic {
        let value: serde_json::Value =
            serde_json::from_slice(semantic_file).map_err(|error| error.to_string())?;
        Some(sha256_hex(
            &serde_json::to_vec(&value).map_err(|error| error.to_string())?,
        ))
    } else {
        None
    };
    let used_legacy = legacy_hash.as_ref() == Some(&stored_semantic);
    let semantic_hash = if used_legacy {
        stored_semantic.clone()
    } else {
        canonical_semantic_hash
    };
    report.record(
        "semantic hash",
        semantic_hash == stored_semantic,
        format!(
            "recomputed {} vs stored {} ({})",
            short_hash(&semantic_hash),
            short_hash(&stored_semantic),
            if used_legacy {
                "legacy insertion-order JSON"
            } else {
                "sorted-key JSON"
            },
        ),
    );
    // The payload inventory is the manifest's payloads map plus the
    // semantic manifest's own file (which cannot list itself).
    let mut inventory = semantic_manifest.payloads.clone();
    inventory.insert(
        "semantic-manifest.json".to_string(),
        sha256_hex(semantic_file),
    );
    let payload_hash = BundleIdentity::payload_hash(&inventory)?;
    let stored_payload = manifest.payload_hash.clone();
    report.record(
        "payload hash",
        payload_hash == stored_payload,
        format!(
            "recomputed {} vs stored {}",
            short_hash(&payload_hash),
            short_hash(&stored_payload)
        ),
    );

    // payload checksums in the semantic manifest must match the files.
    // Same path-validation rule as the checksums pass: untrusted keys can
    // never escape the bundle root.
    let mut payload_errors: Vec<String> = Vec::new();
    for (relative, expected) in &semantic_manifest.payloads {
        let Ok(relative) = crate::v05::bundle::validate_relative_path(relative) else {
            payload_errors.push(format!("{relative}: unsafe path in semantic manifest"));
            continue;
        };
        if relative == LEGACY_REPORT_FILE {
            payload_errors.push(format!("{relative}: not a payload"));
            continue;
        }
        match hashes.hash(relative)? {
            None => payload_errors.push(format!("{relative}: missing")),
            Some(actual) if actual != *expected => {
                payload_errors.push(format!("{relative}: checksum mismatch"))
            }
            Some(_) => {}
        }
    }
    report.record(
        "semantic payload checksums",
        payload_errors.is_empty(),
        if payload_errors.is_empty() {
            format!("{} files", semantic_manifest.payloads.len())
        } else {
            payload_errors.join("; ")
        },
    );

    // external anchor: the only check here whose expected value does not
    // come from the bundle itself.
    if let Some(expected) = &options.expected_semantic_hash {
        let expected = expected.trim().to_ascii_lowercase();
        report.record(
            "semantic hash anchor",
            expected == semantic_hash,
            format!(
                "expected {} vs recomputed {}",
                short_hash(&expected),
                short_hash(&semantic_hash)
            ),
        );
    }

    report.semantic_hash = semantic_hash;
    report.payload_hash = payload_hash;

    // ---- phase 3: deep verification (model/tokenizer/plan) ----
    if let Some(model_path) = &options.model_path {
        deep_model_check(model_path, &semantic_manifest, &mut report);
    }
    if let Some(tokenizer_path) = &options.tokenizer_path {
        let actual = sha256_hex(&std::fs::read(tokenizer_path).map_err(|error| {
            format!(
                "cannot read tokenizer '{}': {error}",
                tokenizer_path.display()
            )
        })?);
        let ok = actual == semantic_manifest.tokenizer.sha256;
        report.record(
            "deep tokenizer sha256",
            ok,
            format!(
                "file {} vs manifest {}",
                short_hash(&actual),
                short_hash(&semantic_manifest.tokenizer.sha256)
            ),
        );
    }

    let payload_bytes = files.remove(PAYLOAD_FILE).unwrap_or_default();
    let manifest_value = manifest;
    Ok((
        report.finished(),
        Some(VerifiedContents {
            manifest: manifest_value,
            semantic_manifest,
            capture_index,
            files,
            payload_bytes,
            tensors: payload_tensors,
        }),
    ))
}

/// Read one bundle file, refusing anything but a regular file and refusing
/// a file that was swapped for another entry between the type check and the
/// open.
fn read_regular_file(root: &Path, relative: &str) -> Result<Vec<u8>, String> {
    let path = &root.join(relative);
    let mut file = open_regular_file(root, relative)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    Ok(bytes)
}

/// Stream-hash a bundle file that the verifier never parses.
fn hash_regular_file(root: &Path, relative: &str) -> Result<String, String> {
    use sha2::Digest as _;
    let path = &root.join(relative);
    let mut file = open_regular_file(root, relative)?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    Ok(crate::v05::manifest::hex(&hasher.finalize()))
}

fn open_regular_file(root: &Path, relative: &str) -> Result<std::fs::File, String> {
    let path = &root.join(relative);
    #[cfg(test)]
    tests::note_open(path);
    // The inventory walk rejected symlinks, but that was earlier: a folder
    // such as `captures/` replaced by a symlink since then would make the open
    // below follow it out of the bundle. Re-check every folder on the way.
    let mut folder = root.to_path_buf();
    if let Some(parent) = Path::new(relative).parent() {
        for component in parent.components() {
            folder.push(component);
            let metadata = std::fs::symlink_metadata(&folder)
                .map_err(|error| format!("cannot read '{}': {error}", folder.display()))?;
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                return Err(format!(
                    "bundle contains a symlink or special file: {}",
                    folder.display()
                ));
            }
        }
    }
    let before = std::fs::symlink_metadata(path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    if !before.file_type().is_file() {
        return Err(format!(
            "bundle contains a symlink or special file: {}",
            path.display()
        ));
    }
    let file = std::fs::File::open(path)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        let opened = file
            .metadata()
            .map_err(|error| format!("cannot read '{}': {error}", path.display()))?;
        if opened.dev() != before.dev() || opened.ino() != before.ino() {
            return Err(format!("'{}' changed while being opened", path.display()));
        }
    }
    Ok(file)
}

fn regular_bundle_files(root: &Path) -> Result<BTreeSet<String>, String> {
    let metadata =
        std::fs::symlink_metadata(root).map_err(|error| format!("bundle root: {error}"))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("bundle root must be a directory, not a symlink".into());
    }
    let mut pending = vec![(root.to_path_buf(), 0usize)];
    let mut files = BTreeSet::new();
    let mut entries_seen = 0usize;
    while let Some((directory, depth)) = pending.pop() {
        if depth > 64 {
            return Err("bundle directory nesting exceeds 64 levels".into());
        }
        for entry in std::fs::read_dir(directory).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            entries_seen += 1;
            if entries_seen > 100_000 {
                return Err("bundle inventory exceeds 100000 entries".into());
            }
            let kind = entry.file_type().map_err(|error| error.to_string())?;
            if kind.is_symlink() || (!kind.is_file() && !kind.is_dir()) {
                return Err(format!(
                    "bundle contains a symlink or special file: {}",
                    entry.path().display()
                ));
            }
            if kind.is_dir() {
                pending.push((entry.path(), depth + 1));
            } else {
                let path = entry.path();
                let relative = path.strip_prefix(root).map_err(|error| error.to_string())?;
                let name = relative.to_str().ok_or("bundle file name is not UTF-8")?;
                files.insert(name.replace(std::path::MAIN_SEPARATOR, "/"));
            }
        }
    }
    Ok(files)
}

fn deep_model_check(
    model_path: &Path,
    semantic_manifest: &SemanticManifest,
    report: &mut VerificationReport,
) {
    match crate::extraction::sha256_file_result(model_path) {
        Ok(actual) => {
            let ok = actual == semantic_manifest.model.sha256;
            report.record(
                "deep model sha256",
                ok,
                format!(
                    "file {} vs manifest {}",
                    short_hash(&actual),
                    short_hash(&semantic_manifest.model.sha256)
                ),
            );
        }
        Err(error) => {
            report.record(
                "deep model sha256",
                false,
                format!("cannot hash model file: {error}"),
            );
        }
    }
    match read_gguf_summary(model_path) {
        Ok((arch, block_count)) => {
            let arch_ok = arch == semantic_manifest.model.architecture;
            report.record(
                "deep model architecture",
                arch_ok,
                format!(
                    "file '{arch}' vs manifest '{}'",
                    semantic_manifest.model.architecture
                ),
            );
            let layers_ok = block_count == semantic_manifest.model.layer_count;
            report.record(
                "deep model layer count",
                layers_ok,
                format!(
                    "file {block_count} vs manifest {}",
                    semantic_manifest.model.layer_count
                ),
            );
        }
        Err(error) => {
            report.record(
                "deep model metadata",
                false,
                format!("cannot read model metadata: {error}"),
            );
        }
    }
}

/// Minimal GGUF header reader: extracts `general.architecture` and the
/// `*.block_count` metadata key without materializing tensor data.
fn read_gguf_summary(path: &Path) -> Result<(String, usize), String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("cannot open '{}': {error}", path.display()))?;
    let file_len = file.metadata().map_err(|error| error.to_string())?.len();
    let mut cursor = std::io::BufReader::new(file);
    let mut magic = [0u8; 4];
    cursor
        .read_exact(&mut magic)
        .map_err(|error| format!("truncated GGUF header: {error}"))?;
    if &magic != b"GGUF" {
        return Err("not a GGUF file (bad magic)".into());
    }
    let version = read_u32(&mut cursor)?;
    if !matches!(version, 2 | 3) {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let mut count_buf = [0u8; 8];
    cursor
        .read_exact(&mut count_buf)
        .map_err(|error| format!("truncated GGUF header: {error}"))?;
    let _tensor_count = u64::from_le_bytes(count_buf);
    cursor
        .read_exact(&mut count_buf)
        .map_err(|error| format!("truncated GGUF header: {error}"))?;
    let kv_count = u64::from_le_bytes(count_buf);
    let mut architecture: Option<String> = None;
    let mut block_counts = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut cursor)?;
        let value_type = read_u32(&mut cursor)?;
        match value_type {
            4 => {
                // u32
                let value = read_u32(&mut cursor)? as usize;
                if key.ends_with(".block_count") {
                    block_counts.insert(key, value);
                }
            }
            8 => {
                // string
                if key == "general.architecture" {
                    architecture = Some(read_gguf_string(&mut cursor)?);
                } else {
                    let length = read_u64(&mut cursor)?;
                    skip_gguf_bytes(&mut cursor, length, file_len)?;
                }
            }
            9 => {
                // array: u32 type + u64 count + elements (skipped by
                // element size)
                let element_type = read_u32(&mut cursor)?;
                let element_count = read_u64(&mut cursor)?;
                if element_type == 8 {
                    // String arrays (notably tokenizer tokens/merges) have a
                    // length prefix per element, not a fixed element size.
                    let remaining = file_len.saturating_sub(
                        cursor
                            .stream_position()
                            .map_err(|error| error.to_string())?,
                    );
                    if element_count > remaining / 8 {
                        return Err("truncated GGUF string array".into());
                    }
                    for _ in 0..element_count {
                        let length = read_u64(&mut cursor)?;
                        skip_gguf_bytes(&mut cursor, length, file_len)?;
                    }
                } else {
                    let size = gguf_element_size(element_type)? as u64;
                    let skip = size
                        .checked_mul(element_count)
                        .ok_or_else(|| "GGUF array size overflow".to_string())?;
                    skip_gguf_bytes(&mut cursor, skip, file_len)?;
                }
            }
            10 => {
                // u64
                let value = read_u64(&mut cursor)?;
                if key.ends_with(".block_count") {
                    block_counts.insert(
                        key,
                        usize::try_from(value).map_err(|_| "GGUF block count overflow")?,
                    );
                }
            }
            _ => {
                // skip fixed-size scalar
                let size = gguf_element_size(value_type)?;
                skip_gguf_bytes(&mut cursor, size as u64, file_len)?;
            }
        }
    }
    let architecture = architecture.ok_or_else(|| "GGUF lacks general.architecture".to_string())?;
    let block_key = format!("{architecture}.block_count");
    let block_count = block_counts
        .remove(&block_key)
        .ok_or_else(|| format!("GGUF lacks {block_key}"))?;
    Ok((architecture, block_count))
}

fn skip_gguf_bytes<R: Seek>(reader: &mut R, count: u64, file_len: u64) -> Result<(), String> {
    let position = reader
        .stream_position()
        .map_err(|error| error.to_string())?;
    let end = position
        .checked_add(count)
        .filter(|end| *end <= file_len)
        .ok_or_else(|| "truncated GGUF metadata value".to_string())?;
    reader
        .seek(SeekFrom::Start(end))
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String, String> {
    let len = read_u64(reader)?;
    // Bound the declared length before allocating: a hostile header can
    // claim u64::MAX and a pre-bound vec! panics with capacity overflow.
    const MAX_GGUF_STRING_BYTES: u64 = 1 << 20;
    if len > MAX_GGUF_STRING_BYTES {
        return Err(format!(
            "GGUF string length {len} exceeds the {MAX_GGUF_STRING_BYTES}-byte limit"
        ));
    }
    let mut bytes = vec![0u8; len as usize];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("truncated GGUF string: {error}"))?;
    String::from_utf8(bytes).map_err(|error| format!("GGUF key is not UTF-8: {error}"))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|error| format!("truncated GGUF value: {error}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|error| format!("truncated GGUF value: {error}"))?;
    Ok(u64::from_le_bytes(buf))
}

fn gguf_element_size(value_type: u32) -> Result<usize, String> {
    match value_type {
        0 | 1 | 7 => Ok(1),
        2 | 3 => Ok(2),
        4..=6 => Ok(4),
        8 => Err("string type handled separately".into()),
        9 => Err("array type handled separately".into()),
        10..=12 => Ok(8),
        other => Err(format!("unknown GGUF value type {other}")),
    }
}

fn selection_consistency(record: &TokenSelectionRecord) -> Option<String> {
    let seq_len = record.token_ids.len();
    if let crate::v05::token_select::TokenSelector::GeneratedStep { step } = record.selector {
        // Tokenization describes the prompt; generated rows use absolute
        // sequence positions beyond it. Step one evaluates the first generated
        // token, at prompt_len, rather than selecting a prompt token.
        let expected = step
            .checked_sub(1)
            .and_then(|offset| seq_len.checked_add(offset));
        let Some(position) = expected else {
            return Some("generated step is zero or its absolute position overflows".into());
        };
        if record.selected_indices != [position] {
            return Some(format!(
                "generated step {step} must select absolute position {position}"
            ));
        }
    } else {
        for &index in &record.selected_indices {
            if index >= seq_len {
                return Some(format!(
                    "selected index {index} out of range for {seq_len} tokens"
                ));
            }
        }
    }
    if record.byte_offsets.len() != seq_len {
        return Some("byte offset count does not match token count".into());
    }
    for (index, &(start, end)) in record.byte_offsets.iter().enumerate() {
        if start > end {
            return Some(format!("token {index} has a reversed byte offset"));
        }
    }
    if let Some((start, end)) = record.matched_byte_span {
        if start > end || end > record.normalized_text.len() {
            return Some("matched byte span is out of range".into());
        }
        if record.coverage == CoverageKind::None {
            return Some("coverage is none for a recorded span".into());
        }
    }
    None
}

fn dtype_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "F32" => Some(4),
        "F16" => Some(2),
        _ => None,
    }
}

/// The first 12 characters of a hash for a report line. Hashes read from a
/// bundle are untrusted: they may be short or not ASCII, and a byte slice
/// would then panic in the middle of verification.
fn short_hash(hash: &str) -> &str {
    hash.char_indices()
        .nth(12)
        .map_or(hash, |(end, _)| &hash[..end])
}

fn parse_semantic_manifest(bytes: &[u8]) -> Result<SemanticManifest, String> {
    serde_json::from_slice(bytes)
        .map_err(|error| format!("semantic-manifest.json is not valid JSON: {error}"))
}

fn parse_capture_index(bytes: &[u8]) -> Result<Vec<CaptureIndexEntry>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("captures/index.jsonl is not UTF-8: {error}"))?;
    let mut entries = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let entry: CaptureIndexEntry = serde_json::from_str(line).map_err(|error| {
            format!(
                "captures/index.jsonl line {} is invalid: {error}",
                line_index + 1
            )
        })?;
        entries.push(entry);
    }
    Ok(entries)
}

fn parse_checksums(bytes: &[u8]) -> Result<BTreeMap<String, String>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("checksums.sha256 is not UTF-8: {error}"))?;
    let mut checksums = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (sum, relative) = line
            .split_once("  ")
            .ok_or_else(|| format!("checksums.sha256 has a malformed line: {line:?}"))?;
        if sum.len() != 64 {
            return Err(format!(
                "checksums.sha256 has a malformed checksum on line: {line:?}"
            ));
        }
        if checksums
            .insert(relative.to_string(), sum.to_string())
            .is_some()
        {
            return Err(format!("checksums.sha256 repeats path {relative:?}"));
        }
    }
    Ok(checksums)
}

fn now_iso8601() -> String {
    // Runtime metadata only; not part of any hash.
    format!("epoch-seconds-{}", crate::extraction::unix_timestamp())
}

#[cfg(test)]
mod tests {
    #[test]
    fn short_hash_never_panics_on_untrusted_hashes() {
        assert_eq!(super::short_hash("0123456789abcdef"), "0123456789ab");
        assert_eq!(super::short_hash(""), "");
        assert_eq!(super::short_hash("abc"), "abc");
        // A multi-byte character straddling byte 12.
        assert_eq!(super::short_hash("0123456789aé-tail"), "0123456789aé");
    }

    #[test]
    fn generated_selection_uses_absolute_decode_positions() {
        use crate::v05::token_select::TokenSelector;
        let mut record =
            crate::v05::testutil::sample_selection_record(TokenSelector::GeneratedStep { step: 1 });
        record.selected_indices = vec![1];
        assert!(super::selection_consistency(&record).is_none());
        for invalid in [vec![], vec![0], vec![2], vec![1, 1]] {
            record.selected_indices = invalid;
            assert!(super::selection_consistency(&record).is_some());
        }
        record.selector = TokenSelector::GeneratedStep { step: 3 };
        record.selected_indices = vec![3];
        assert!(super::selection_consistency(&record).is_none());
        record.selector = TokenSelector::GeneratedStep { step: 0 };
        assert!(super::selection_consistency(&record).is_some());
        record.token_ids.push(2);
        record.selector = TokenSelector::GeneratedStep { step: usize::MAX };
        assert!(super::selection_consistency(&record).is_some());
        record.selector = TokenSelector::PromptFinal;
        assert!(super::selection_consistency(&record).is_some());
    }

    use super::*;
    use crate::v05::testutil;
    use crate::v05::testutil::temp_root;

    thread_local! {
        /// Bundle file opens per path on this test thread.
        static OPENS: std::cell::RefCell<BTreeMap<PathBuf, usize>> =
            const { std::cell::RefCell::new(BTreeMap::new()) };
    }

    pub(super) fn note_open(path: &Path) {
        OPENS.with(|opens| *opens.borrow_mut().entry(path.to_path_buf()).or_insert(0) += 1);
    }

    fn take_opens() -> BTreeMap<PathBuf, usize> {
        OPENS.with(|opens| std::mem::take(&mut *opens.borrow_mut()))
    }

    fn read_semantic_manifest(root: &Path) -> Result<SemanticManifest, String> {
        parse_semantic_manifest(&std::fs::read(root.join("semantic-manifest.json")).unwrap())
    }

    /// Write the standard test bundle into a fresh temp dir, run `mutate`,
    /// verify, and require exactly the named checks to fail.
    fn assert_verification_failure(
        tag: &str,
        mutate: impl FnOnce(&std::path::Path),
        expect_failed: &[&str],
    ) {
        let root = temp_root(tag);
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        mutate(&root);
        let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
        assert!(!report.ok);
        let names: Vec<&str> = report
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| check.name.as_str())
            .collect();
        for expected in expect_failed {
            assert!(
                names.contains(expected),
                "missing {expected:?} among {names:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn valid_bundle_verifies() {
        let root = temp_root("valid");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        assert_eq!(report.checks.len(), 21);
        // Verification never writes into the bundle it checks.
        assert!(!root.join("verification.json").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verification_leaves_a_read_only_bundle_untouched() {
        let root = temp_root("read-only");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        let before: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o555)).unwrap();
        }
        let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        assert!(report.ok, "{:?}", report.checks);
        let after: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(before, after);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_forged_verification_json_cannot_change_the_verdict() {
        let forged = serde_json::json!({
            "bundle_schema": BUNDLE_SCHEMA_V1, "ok": true, "semantic_hash": "0",
            "payload_hash": "0", "checks": [], "warnings": [], "timestamp": "x",
        });
        // A valid bundle with a stale report still verifies, with a warning.
        let root = temp_root("forged-report");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        std::fs::write(root.join("verification.json"), forged.to_string()).unwrap();
        let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("verification.json")));
        // A corrupted bundle with a report claiming success still fails.
        let path = root.join("outputs.jsonl");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.push(b'\n');
        std::fs::write(&path, bytes).unwrap();
        assert!(!verify_bundle(&root, &VerifyOptions::default()).unwrap().ok);
        std::fs::remove_dir_all(root).unwrap();
        // Listing the report as a bundle file is not a way to smuggle it in.
        assert_verification_failure(
            "listed-report",
            |root| {
                std::fs::write(root.join("verification.json"), forged.to_string()).unwrap();
                let path = root.join("manifest.json");
                let mut manifest: BundleManifest =
                    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                manifest.files.push("verification.json".into());
                std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            },
            &["bundle file inventory"],
        );
    }

    #[test]
    fn each_bundle_file_is_opened_once_and_the_checked_bytes_are_returned() {
        let root = temp_root("read-once");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        take_opens();
        let loaded = load_bundle_for_source(&root).unwrap();
        let opens = take_opens();
        let repeated: Vec<_> = opens.iter().filter(|(_, count)| **count > 1).collect();
        assert!(repeated.is_empty(), "{repeated:?}");
        assert!(opens.contains_key(&root.join("captures/tensors.safetensors")));
        // Replacing files after verification changes nothing that was loaded.
        let outputs = loaded.file("outputs.jsonl").unwrap().to_vec();
        std::fs::write(root.join("outputs.jsonl"), b"tampered\n").unwrap();
        std::fs::write(root.join("captures/tensors.safetensors"), b"x").unwrap();
        assert_eq!(loaded.file("outputs.jsonl").unwrap(), outputs.as_slice());
        assert_eq!(
            loaded
                .tensor_f32_by_name("cap-1/i1/residual-post-mlp/0")
                .unwrap(),
            testutil::sample_rows()
        );
        assert_eq!(loaded.semantic_hash, loaded.manifest.semantic_hash);
        std::fs::remove_dir_all(root).unwrap();
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/bundle-v050")
                .join(name),
        )
        .unwrap()
    }

    #[test]
    fn resolved_experiment_is_bound_to_the_hashed_spec_and_manifest() {
        let spec = String::from_utf8(fixture("experiment.toml")).unwrap();
        let resolved = fixture("resolved-experiment.json");
        let semantic = parse_semantic_manifest(&fixture("semantic-manifest.json")).unwrap();
        // The frozen v0.5 bundle (run with an --output override) binds.
        let bound = bind_resolved_experiment(&spec, &resolved, &semantic).unwrap();
        assert_eq!(bound.generation.max_new_tokens, 8);
        let edit = |change: &dyn Fn(&mut serde_json::Value)| {
            let mut value: serde_json::Value = serde_json::from_slice(&resolved).unwrap();
            change(&mut value);
            bind_resolved_experiment(&spec, &serde_json::to_vec(&value).unwrap(), &semantic)
        };
        // Fields outside the semantic hash that change what runs.
        for change in [
            &(|v: &mut serde_json::Value| v["generation"]["max_new_tokens"] = 4096.into())
                as &dyn Fn(&mut serde_json::Value),
            &|v| v["generation"]["temperature"] = 1.5.into(),
            &|v| v["model"]["tokenizer"] = "/elsewhere/tokenizer.json".into(),
            &|v| v["model"]["tokenizer_expected_sha256"] = "".into(),
            &|v| v["output"]["overwrite"] = true.into(),
        ] {
            let error = edit(change).unwrap_err();
            assert!(error.contains("hashed experiment.toml"), "{error}");
        }
        // A mode override is legitimate only when the manifest agrees.
        let error = edit(&|v| v["execution"]["mode"] = "planned".into()).unwrap_err();
        assert!(error.contains("semantic manifest"), "{error}");
        assert!(edit(&|v| v["execution"]["threads"] = 2.into()).is_ok());
        assert!(edit(&|v| v["output"]["directory"] = "runs/x".into()).is_ok());
        // An edited spec that no longer matches the recorded semantics.
        let edited_spec = spec.replace("max_new_tokens = 8", "max_new_tokens = 9");
        assert!(bind_resolved_experiment(&edited_spec, &resolved, &semantic).is_err());
    }

    #[test]
    fn semantic_hash_anchor_rejects_a_resealed_bundle() {
        let root = temp_root("anchor");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        let original = verify_bundle(&root, &VerifyOptions::default()).unwrap();
        let anchored = |expected: &str| VerifyOptions {
            expected_semantic_hash: Some(expected.to_string()),
            ..VerifyOptions::default()
        };
        let report =
            verify_bundle(&root, &anchored(&original.semantic_hash.to_uppercase())).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        assert!(report
            .checks
            .iter()
            .any(|check| check.name == "semantic hash anchor" && check.ok));
        // Edit the science and reseal every bundle-authored hash: the bundle
        // is self-consistent again, so only the external anchor catches it.
        let mut semantic = read_semantic_manifest(&root).unwrap();
        semantic.experiment.name = "edited".into();
        reseal(&root, &mut semantic);
        assert!(verify_bundle(&root, &VerifyOptions::default()).unwrap().ok);
        let report = verify_bundle(&root, &anchored(&original.semantic_hash)).unwrap();
        let failed: Vec<_> = report
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| check.name.as_str())
            .collect();
        assert_eq!(failed, ["semantic hash anchor"]);
        assert!(load_verified_bundle(&root, &anchored(&original.semantic_hash)).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    // Reseal mutated fixtures so a failed schema/identity check cannot be
    // explained by stale hashes. This does not confer trust on their contents.
    fn reseal(root: &Path, semantic: &mut SemanticManifest) {
        for (name, checksum) in &mut semantic.payloads {
            *checksum = sha256_hex(&std::fs::read(root.join(name)).unwrap());
        }
        let bytes = crate::v05::manifest::canonical_json(semantic).unwrap();
        std::fs::write(root.join("semantic-manifest.json"), &bytes).unwrap();
        let path = root.join("manifest.json");
        let mut manifest: BundleManifest =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        manifest.semantic_hash = BundleIdentity::semantic_hash(semantic).unwrap();
        let mut inventory = semantic.payloads.clone();
        inventory.insert("semantic-manifest.json".into(), sha256_hex(&bytes));
        manifest.payload_hash = BundleIdentity::payload_hash(&inventory).unwrap();
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let checksums = manifest
            .files
            .iter()
            .filter(|name| !matches!(name.as_str(), "checksums.sha256" | "verification.json"))
            .map(|name| {
                format!(
                    "{}  {name}\n",
                    sha256_hex(&std::fs::read(root.join(name)).unwrap())
                )
            })
            .collect::<String>();
        std::fs::write(root.join("checksums.sha256"), checksums).unwrap();
    }

    #[test]
    fn unknown_nested_contracts_fail_even_when_hashes_are_valid() {
        for field in ["experiment schema", "hook schema", "plan schema"] {
            let root = temp_root("unknown-contract");
            testutil::write_test_bundle(
                &root,
                &testutil::sample_rows(),
                &testutil::sample_positions(),
            );
            let mut semantic = read_semantic_manifest(&root).unwrap();
            match field {
                "experiment schema" => semantic.experiment_schema = "ember.experiment.v2".into(),
                "hook schema" => semantic.hook_schema = 2,
                "plan schema" => semantic.plan_schema = 2,
                _ => unreachable!(),
            }
            reseal(&root, &mut semantic);
            let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
            let failed: Vec<_> = report
                .checks
                .iter()
                .filter(|check| !check.ok)
                .map(|check| check.name.as_str())
                .collect();
            assert_eq!(failed, [field]);
            assert!(!report.ok);
            assert!(load_bundle_for_source(&root).is_err());
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn stored_plan_must_match_declared_contract_and_identity() {
        for change_schema in [false, true] {
            let root = temp_root("plan-contract");
            testutil::write_test_bundle(
                &root,
                &testutil::sample_rows(),
                &testutil::sample_positions(),
            );
            let mut semantic = read_semantic_manifest(&root).unwrap();
            let expected = if change_schema {
                let path = root.join("execution-plan.json");
                let mut plan: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                plan["schema_version"] = serde_json::json!(2);
                semantic.execution.plan_hash = testutil::fixture_plan_hash(&mut plan);
                std::fs::write(path, serde_json::to_vec(&plan).unwrap()).unwrap();
                "execution plan schema"
            } else {
                semantic.execution.plan_hash = "0".repeat(64);
                "execution plan identity"
            };
            reseal(&root, &mut semantic);
            let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
            let failed: Vec<_> = report
                .checks
                .iter()
                .filter(|check| !check.ok)
                .map(|check| check.name.as_str())
                .collect();
            assert_eq!(failed, [expected]);
            assert!(!report.ok);
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn unlisted_files_and_unchecked_files_fail() {
        assert_verification_failure(
            "extra-file",
            |root| {
                std::fs::write(root.join("unindexed.bin"), b"extra payload").unwrap();
            },
            &["bundle file inventory"],
        );
        assert_verification_failure(
            "missing-checksum",
            |root| {
                let path = root.join("checksums.sha256");
                let text = std::fs::read_to_string(&path).unwrap();
                let filtered = text
                    .lines()
                    .filter(|line| !line.ends_with("  outputs.jsonl"))
                    .collect::<Vec<_>>()
                    .join("\n");
                std::fs::write(path, filtered).unwrap();
            },
            &["checksums"],
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_folder_swapped_for_a_symlink_is_not_followed() {
        let root = temp_root("folder-symlink");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        assert!(open_regular_file(&root, "captures/tensors.safetensors").is_ok());
        // After the inventory walk: move captures/ out and point a symlink at it.
        let outside = root.with_extension("outside");
        let _ = std::fs::remove_dir_all(&outside);
        std::fs::rename(root.join("captures"), &outside).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("captures")).unwrap();
        let error = open_regular_file(&root, "captures/tensors.safetensors").unwrap_err();
        assert!(error.contains("symlink"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    #[cfg(unix)]
    fn symlinks_are_rejected_before_manifest_or_payload_reads() {
        for name in [
            "manifest.json",
            "captures/tensors.safetensors",
            "verification.json",
        ] {
            let root = temp_root("symlink");
            testutil::write_test_bundle(
                &root,
                &testutil::sample_rows(),
                &testutil::sample_positions(),
            );
            let path = root.join(name);
            if path.exists() {
                std::fs::remove_file(&path).unwrap();
            }
            // A dangling target ensures rejection is based on entry type, not
            // on accidentally succeeding while following the target.
            std::os::unix::fs::symlink(root.join("absent-external-target"), &path).unwrap();
            let error = verify_bundle(&root, &VerifyOptions::default()).unwrap_err();
            assert!(error.contains("symlink"), "{error}");
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn duplicate_checksum_paths_fail() {
        let root = temp_root("duplicate-checksum");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        let path = root.join("checksums.sha256");
        let mut text = std::fs::read_to_string(&path).unwrap();
        let duplicate = text.lines().next().unwrap().to_owned();
        text.push_str(&duplicate);
        text.push('\n');
        std::fs::write(path, text).unwrap();
        assert!(verify_bundle(&root, &VerifyOptions::default())
            .unwrap_err()
            .contains("repeats path"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn one_byte_payload_corruption_fails() {
        assert_verification_failure(
            "corrupt",
            |root| {
                let payload = root.join("captures/tensors.safetensors");
                let mut bytes = std::fs::read(&payload).unwrap();
                let last = bytes.len() - 1;
                bytes[last] ^= 0xFF;
                std::fs::write(&payload, bytes).unwrap();
            },
            &["checksums", "tensor payload"],
        );
    }

    #[test]
    fn removed_file_fails() {
        assert_verification_failure(
            "removed",
            |root| {
                std::fs::remove_file(root.join("captures/index.jsonl")).unwrap();
            },
            &["required files"],
        );
    }

    #[test]
    fn altered_manifest_value_fails() {
        assert_verification_failure(
            "altered",
            |root| {
                let path = root.join("semantic-manifest.json");
                let mut value: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                value["experiment"]["name"] = serde_json::json!("tampered");
                std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
            },
            &["semantic hash"],
        );
    }

    #[test]
    fn extra_unindexed_tensor_fails() {
        assert_verification_failure(
            "extra",
            |root| {
                // Append a second tensor to the payload and fix
                // checksums.sha256 so only the unindexed-tensor check can
                // catch it.
                let payload_path = root.join("captures/tensors.safetensors");
                let original = std::fs::read(&payload_path).unwrap();
                let tensors = crate::v05::safetensors::deserialize(&original).unwrap();
                let extra = crate::v05::safetensors::serialize(&[
                    crate::v05::safetensors::TensorData {
                        name: "cap-1/i1/residual-post-mlp/0",
                        dtype: crate::v05::safetensors::TensorDType::F32,
                        shape: &[1, 4],
                        bytes: &[0u8; 16],
                    },
                    crate::v05::safetensors::TensorData {
                        name: "rogue/i1/residual-post-mlp/0",
                        dtype: crate::v05::safetensors::TensorDType::F32,
                        shape: &[1, 4],
                        bytes: &[0u8; 16],
                    },
                ])
                .unwrap();
                let _ = tensors;
                std::fs::write(&payload_path, extra).unwrap();
                // refresh checksums.sha256 to isolate the payload check
                fix_checksums(root);
            },
            &["tensor payload"],
        );
    }

    #[test]
    fn index_entries_sharing_a_tensor_name_fail() {
        assert_verification_failure(
            "duplicate-tensor-name",
            |root| {
                // A second entry for another capture that points at the same
                // tensor: only one of the two could ever be compared.
                let path = root.join("captures/index.jsonl");
                let text = std::fs::read_to_string(&path).unwrap();
                let first = text.lines().next().unwrap();
                let mut entry: serde_json::Value = serde_json::from_str(first).unwrap();
                entry["capture_id"] = "shadow".into();
                entry["shape"] = serde_json::json!([9, 9]);
                // Placed first, so the genuine entry after it would win a
                // last-one-wins lookup and this one would go unchecked.
                std::fs::write(&path, format!("{entry}\n{text}")).unwrap();
                let mut semantic = read_semantic_manifest(root).unwrap();
                reseal(root, &mut semantic);
            },
            &["tensor payload"],
        );
    }

    #[test]
    fn incomplete_bundle_fails() {
        let root = temp_root("incomplete");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("manifest.json"), b"{}").unwrap();
        // A malformed manifest is a hard error; a missing-file bundle
        // yields a failing report. Both must be non-ok.
        let report = verify_bundle(&root, &VerifyOptions::default());
        if let Ok(report) = report {
            assert!(!report.ok);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn deep_metadata_reads_string_arrays_and_boolean_values() {
        fn string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        let root = temp_root("metadata.gguf");
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&5u64.to_le_bytes());
        string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        string(&mut bytes, "hello");
        string(&mut bytes, "كتاب");
        string(&mut bytes, "tokenizer.ggml.add_bos_token");
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.push(1);
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        string(&mut bytes, "llama");
        string(&mut bytes, "llama.block_count");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&16u32.to_le_bytes());
        // Another architecture's count must not override the selected one.
        string(&mut bytes, "other.block_count");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&99u32.to_le_bytes());
        std::fs::write(&root, &bytes).unwrap();
        assert_eq!(read_gguf_summary(&root).unwrap(), ("llama".into(), 16));
        bytes.pop();
        std::fs::write(&root, &bytes).unwrap();
        assert!(read_gguf_summary(&root).is_err());
        std::fs::remove_file(root).unwrap();
    }

    #[test]
    fn deep_metadata_rejects_skips_past_eof_and_overflow() {
        let mut reader = std::io::Cursor::new(vec![0u8; 8]);
        assert!(skip_gguf_bytes(&mut reader, 9, 8).is_err());
        reader.set_position(4);
        assert!(skip_gguf_bytes(&mut reader, u64::MAX, 8).is_err());
    }

    #[test]
    fn deep_model_mismatch_fails() {
        let root = temp_root("deep");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        // A non-model file with the wrong hash fails the deep check.
        let model_path = temp_root("fake-model.gguf");
        std::fs::write(&model_path, b"not a model").unwrap();
        let options = VerifyOptions {
            model_path: Some(model_path.clone()),
            ..VerifyOptions::default()
        };
        let report = verify_bundle(&root, &options).unwrap();
        assert!(!report.ok);
        let names: Vec<&str> = report
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| check.name.as_str())
            .collect();
        assert!(names.contains(&"deep model sha256"), "{names:?}");
        std::fs::remove_file(model_path).unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn traversal_in_checksums_is_rejected() {
        assert_verification_failure(
            "traversal",
            |root| {
                let path = root.join("checksums.sha256");
                let mut text = std::fs::read_to_string(&path).unwrap();
                text.push_str(&format!("{}\n", "00".repeat(32) + "  ../escape.bin"));
                std::fs::write(&path, text).unwrap();
            },
            &["checksums"],
        );
    }

    #[test]
    fn traversal_checksum_fails_even_with_the_correct_hash() {
        // A hostile bundle that lists `../victim` with the victim's real
        // hash must still fail: the path itself is rejected, so the check
        // can never become an arbitrary-file hash oracle.
        assert_verification_failure(
            "traversal-oracle",
            |root| {
                let parent = root.parent().unwrap();
                let victim = parent.join("ember-victim.txt");
                std::fs::write(&victim, b"secret").unwrap();
                let victim_hash = crate::v05::manifest::sha256_hex(b"secret");
                let path = root.join("checksums.sha256");
                let mut text = std::fs::read_to_string(&path).unwrap();
                // the victim lives one level above the bundle root
                text.push_str(&format!("{}\n", victim_hash + "  ../ember-victim.txt"));
                std::fs::write(&path, text).unwrap();
            },
            &["checksums"],
        );
    }

    #[test]
    fn source_bundle_loads_only_when_verified() {
        let root = temp_root("source");
        testutil::write_test_bundle(
            &root,
            &testutil::sample_rows(),
            &testutil::sample_positions(),
        );
        let loaded = load_bundle_for_source(&root).unwrap();
        let rows = loaded
            .tensor_f32_by_name("cap-1/i1/residual-post-mlp/0")
            .unwrap();
        assert_eq!(rows, testutil::sample_rows());
        // corrupt then refuse to load
        let payload = root.join("captures/tensors.safetensors");
        let mut bytes = std::fs::read(&payload).unwrap();
        bytes[20] ^= 0x01;
        std::fs::write(&payload, bytes).unwrap();
        assert!(load_bundle_for_source(&root).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    fn fix_checksums(root: &std::path::Path) {
        // Rewrite checksums.sha256 from the current files so later checks
        // pass and only the intended check fails.
        let mut lines: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(root).unwrap() {
            let entry = entry.unwrap();
            if !entry.file_type().unwrap().is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let bytes = std::fs::read(entry.path()).unwrap();
            lines.push(format!("{}  {name}", sha256_hex(&bytes)));
        }
        let path = root.join("captures").join("tensors.safetensors");
        let bytes = std::fs::read(&path).unwrap();
        lines.push(format!(
            "{}  captures/tensors.safetensors",
            sha256_hex(&bytes)
        ));
        lines.sort();
        std::fs::write(
            root.join("checksums.sha256"),
            format!("{}\n", lines.join("\n")),
        )
        .unwrap();
    }
}
