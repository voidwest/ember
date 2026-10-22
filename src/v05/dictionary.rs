//! Feature dictionaries (ADR-0003): sparse autoencoders in the SAELens
//! format, and the encode and decode that captures and interventions use in
//! feature space.
//!
//! A dictionary maps a site row `x` (`d_in` wide) to `d_sae` feature
//! activations `f` and back. Captures record the top features of their rows
//! (`ember.features.v1`). Interventions change selected features and add only
//! the decoded change to the row, `x' = x + (f' - f)_S · W_dec[S]`, so the
//! reconstruction error stays in place and an edit that changes nothing is
//! bit-identical.
//!
//! Encode and decode accumulate in f64 in index order and round once to f32,
//! so a capture and an intervention see the same feature values.

use crate::v05::hook::SemanticHookSite;
use crate::v05::manifest::sha256_hex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const DICTIONARY_SCHEMA_V1: &str = "ember.dictionary.v1";
pub const FEATURES_SCHEMA_V1: &str = "ember.features.v1";
pub const DICTIONARY_DIR: &str = "artifacts/dictionaries";
pub const FEATURES_DIR: &str = "artifacts/features";
/// The SAELens weights file inside a dictionary folder.
pub const WEIGHTS_FILE: &str = "sae_weights.safetensors";
/// The SAELens configuration file inside a dictionary folder.
pub const CONFIG_FILE: &str = "cfg.json";
/// The largest weights file Ember reads, 2 GiB: a d_sae 65536 dictionary at
/// d_in 2048 in f32 is 1 GiB. Loading holds the file and its f32 copy at
/// once, so memory peaks near twice the file size (three times for F16).
pub const MAX_WEIGHTS_BYTES: u64 = 2 << 30;
const MAX_CONFIG_BYTES: u64 = 1 << 20;
/// The largest `top_k` a feature capture may ask for.
pub const MAX_TOP_K: usize = 4096;

/// One `[[dictionaries]]` entry of an experiment spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DictionarySpec {
    /// Unique dictionary id within the experiment.
    pub id: String,
    /// The SAELens folder: `cfg.json` and `sae_weights.safetensors`.
    pub path: PathBuf,
    /// SHA-256 of `sae_weights.safetensors` (64 lowercase hex characters).
    pub sha256: String,
    /// SHA-256 of `cfg.json`, which selects the architecture: the weights
    /// alone do not say how to encode.
    pub config_sha256: String,
    /// The site the dictionary reads. It must be a `d_model`-wide
    /// per-layer site.
    pub site: SemanticHookSite,
    /// The one layer the dictionary was trained at.
    pub layer: usize,
}

/// The `features` field of an intervention: the operation acts on these
/// features of `dictionary`, not on the raw row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureSelection {
    pub dictionary: String,
    /// Feature indices, ascending and without repeats.
    pub ids: Vec<usize>,
}

/// The `features` field of a capture: also record the `top_k` strongest
/// features of every stored row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureCapture {
    pub dictionary: String,
    pub top_k: usize,
}

/// The SAELens architectures that version 1 reads. A later version can add
/// one (TopK, for example), so the enum is not exhaustive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Architecture {
    /// `f = relu(pre)`.
    Standard,
    /// `f = pre` where `pre > threshold` and `pre > 0`, else 0.
    Jumprelu,
}

/// The sites a dictionary can read: the per-layer sites of the model's
/// width.
pub fn supports_site(site: SemanticHookSite) -> bool {
    matches!(
        site,
        SemanticHookSite::ResidualPreAttention
            | SemanticHookSite::AttentionOutput
            | SemanticHookSite::MlpOutput
            | SemanticHookSite::ResidualPostMlp
    )
}

/// The fields of a SAELens `cfg.json` that Ember reads: the ones that change
/// the forward pass, which [`Dictionary::from_parts`] checks, and two names
/// that it records. The other fields describe training and are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct SaeLensConfig {
    pub architecture: String,
    pub d_in: usize,
    pub d_sae: usize,
    /// The activation: `activation_fn` (SAELens 5) or `activation_fn_str`
    /// (earlier releases), with `activation_fn_kwargs`.
    #[serde(default)]
    pub activation_fn: Option<String>,
    #[serde(default)]
    pub activation_fn_str: Option<String>,
    #[serde(default)]
    pub activation_fn_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// Required: SAELens releases disagree on its default.
    #[serde(default)]
    pub apply_b_dec_to_input: Option<bool>,
    #[serde(default)]
    pub normalize_activations: Option<String>,
    #[serde(default)]
    pub finetuning_scaling_factor: Option<bool>,
    #[serde(default)]
    pub hook_head_index: Option<usize>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub hook_name: Option<String>,
    /// SAELens 6 keeps the names here.
    #[serde(default)]
    pub metadata: Option<SaeLensMetadata>,
}

/// The `metadata` table of a SAELens 6 `cfg.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct SaeLensMetadata {
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub hook_name: Option<String>,
}

/// Provenance of one dictionary (`artifacts/dictionaries/<id>.json`). The
/// bundle records the dictionary; it does not copy its weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DictionaryRecord {
    pub schema: String,
    pub id: String,
    /// The folder as the spec gives it.
    pub path: PathBuf,
    pub sha256: String,
    pub config_sha256: String,
    pub format: String,
    pub architecture: Architecture,
    pub apply_b_dec_to_input: bool,
    pub d_in: usize,
    pub d_sae: usize,
    pub site: SemanticHookSite,
    pub layer: usize,
    /// `model_name` and `hook_name` from `cfg.json`, as found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hook_name: Option<String>,
}

/// A loaded dictionary: the record of the spec entry that names it, and
/// weights that every entry naming the same file and hash shares.
#[derive(Debug, Clone)]
pub struct Dictionary {
    pub record: DictionaryRecord,
    weights: Arc<Weights>,
}

/// The tensors of a SAELens dictionary, in f32.
#[derive(Debug)]
pub struct Weights {
    /// `[d_in, d_sae]`, row-major.
    w_enc: Vec<f32>,
    b_enc: Vec<f32>,
    /// `[d_sae, d_in]`, row-major.
    w_dec: Vec<f32>,
    b_dec: Vec<f32>,
    /// JumpReLU thresholds.
    threshold: Option<Vec<f32>>,
}

impl Dictionary {
    /// The width of the rows the dictionary reads.
    pub fn d_in(&self) -> usize {
        self.record.d_in
    }

    /// The number of features.
    pub fn d_sae(&self) -> usize {
        self.record.d_sae
    }

    /// The same weights under another spec entry (the same file and hash,
    /// another id, site or layer), without copying them.
    pub fn for_spec(&self, spec: &DictionarySpec) -> Dictionary {
        let mut record = self.record.clone();
        record.id = spec.id.clone();
        record.path = spec.path.clone();
        record.site = spec.site;
        record.layer = spec.layer;
        Dictionary {
            record,
            weights: Arc::clone(&self.weights),
        }
    }

    /// Read, hash-check and parse the dictionary that `spec` names.
    pub fn load(spec: &DictionarySpec) -> Result<Dictionary, String> {
        let config_path = spec.path.join(CONFIG_FILE);
        let weights_path = spec.path.join(WEIGHTS_FILE);
        let config = read_capped(&config_path, MAX_CONFIG_BYTES)?;
        let weights = read_capped(&weights_path, MAX_WEIGHTS_BYTES)?;
        for (path, bytes, pinned) in [
            (&weights_path, &weights, &spec.sha256),
            (&config_path, &config, &spec.config_sha256),
        ] {
            let actual = sha256_hex(bytes);
            if actual != *pinned {
                return Err(format!(
                    "dictionary '{}': {} hashes to {actual} but the spec pins {pinned}",
                    spec.id,
                    path.display(),
                ));
            }
        }
        let config: SaeLensConfig = serde_json::from_slice(&config).map_err(|error| {
            format!(
                "dictionary '{}': {}: {error}",
                spec.id,
                config_path.display()
            )
        })?;
        Dictionary::from_parts(spec, &config, &weights)
            .map_err(|error| format!("dictionary '{}': {error}", spec.id))
    }

    /// Build a dictionary from a parsed `cfg.json` and the weights file's
    /// bytes, refusing every configuration that version 1 does not support.
    pub fn from_parts(
        spec: &DictionarySpec,
        config: &SaeLensConfig,
        weights: &[u8],
    ) -> Result<Dictionary, String> {
        let architecture = match config.architecture.as_str() {
            "standard" => Architecture::Standard,
            "jumprelu" => Architecture::Jumprelu,
            other => {
                return Err(format!(
                    "SAELens architecture '{other}' is not supported; use standard or jumprelu"
                ));
            }
        };
        for activation in [&config.activation_fn, &config.activation_fn_str]
            .into_iter()
            .flatten()
        {
            if activation != "relu" {
                return Err(format!(
                    "activation '{activation}' is not supported; only relu is"
                ));
            }
        }
        if config
            .activation_fn_kwargs
            .as_ref()
            .is_some_and(|kwargs| !kwargs.is_empty())
        {
            return Err("activation_fn_kwargs must be empty (relu takes none)".into());
        }
        if let Some(normalize) = &config.normalize_activations
            && normalize != "none"
        {
            return Err(format!(
                "normalize_activations '{normalize}' is not supported; only none is"
            ));
        }
        if config.finetuning_scaling_factor == Some(true) {
            return Err("finetuning_scaling_factor is not supported".into());
        }
        if config.hook_head_index.is_some() {
            return Err(
                "hook_head_index is not supported: the dictionary must read whole rows".into(),
            );
        }
        let apply_b_dec_to_input = config
            .apply_b_dec_to_input
            .ok_or("cfg.json must state apply_b_dec_to_input")?;
        let (d_in, d_sae) = (config.d_in, config.d_sae);
        if d_in == 0 || d_sae == 0 {
            return Err("d_in and d_sae must be positive".into());
        }
        let tensors = crate::v05::safetensors::deserialize(weights)?;
        let tensor = |name: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
            let (_, view) = tensors
                .iter()
                .find(|(tensor_name, _)| tensor_name == name)
                .ok_or_else(|| format!("{WEIGHTS_FILE} has no tensor '{name}'"))?;
            if view.shape != shape {
                return Err(format!(
                    "tensor '{name}' has shape {:?}; cfg.json implies {shape:?}",
                    view.shape
                ));
            }
            let values = crate::v05::safetensors::tensor_f32(weights, view)?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!("tensor '{name}' holds a non-finite value"));
            }
            Ok(values)
        };
        let threshold = match architecture {
            Architecture::Jumprelu => Some(tensor("threshold", &[d_sae])?),
            Architecture::Standard => None,
        };
        Ok(Dictionary {
            record: DictionaryRecord {
                schema: DICTIONARY_SCHEMA_V1.into(),
                id: spec.id.clone(),
                path: spec.path.clone(),
                sha256: spec.sha256.clone(),
                config_sha256: spec.config_sha256.clone(),
                format: "saelens".into(),
                architecture,
                apply_b_dec_to_input,
                d_in,
                d_sae,
                site: spec.site,
                layer: spec.layer,
                model_name: config.model_name.clone().or_else(|| {
                    config
                        .metadata
                        .as_ref()
                        .and_then(|meta| meta.model_name.clone())
                }),
                hook_name: config.hook_name.clone().or_else(|| {
                    config
                        .metadata
                        .as_ref()
                        .and_then(|meta| meta.hook_name.clone())
                }),
            },
            weights: Arc::new(Weights {
                w_enc: tensor("W_enc", &[d_in, d_sae])?,
                b_enc: tensor("b_enc", &[d_sae])?,
                w_dec: tensor("W_dec", &[d_sae, d_in])?,
                b_dec: tensor("b_dec", &[d_in])?,
                threshold,
            }),
        })
    }

    /// Refuse feature indices outside the dictionary.
    pub fn check_ids(&self, ids: &[usize]) -> Result<(), String> {
        match ids.iter().find(|&&id| id >= self.d_sae()) {
            Some(id) => Err(format!(
                "feature {id} is out of range: dictionary '{}' has {} features",
                self.record.id,
                self.d_sae()
            )),
            None => Ok(()),
        }
    }

    /// The encoder input value of column `j`.
    fn input(&self, row: &[f32], j: usize) -> f64 {
        if self.record.apply_b_dec_to_input {
            f64::from(row[j]) - f64::from(self.weights.b_dec[j])
        } else {
            f64::from(row[j])
        }
    }

    /// The activation of feature `i` with pre-activation `pre` (already
    /// rounded to f32).
    fn activate(&self, i: usize, pre: f32) -> f32 {
        let active = pre > 0.0
            && self
                .weights
                .threshold
                .as_ref()
                .is_none_or(|threshold| pre > threshold[i]);
        if active {
            pre
        } else {
            0.0
        }
    }

    /// The activations of features `ids` for `row` (`d_in` wide).
    ///
    /// The pre-activation of feature `i` is `b_enc[i]` plus the sum over
    /// `j` of `input_j * W_enc[j, i]`, accumulated in f64 in ascending `j`
    /// and rounded once: the same order as [`Dictionary::encode`].
    pub fn encode_selected(&self, row: &[f32], ids: &[usize]) -> Vec<f32> {
        // A large selection reads W_enc in order through the full encode,
        // which sums in the same order and so gives the same values.
        if ids.len() >= 64 && ids.len() * 8 > self.d_sae() {
            let all = self.encode(row);
            return ids.iter().map(|&i| all[i]).collect();
        }
        self.encode_strided(row, ids)
    }

    /// [`Dictionary::encode_selected`] one feature at a time.
    fn encode_strided(&self, row: &[f32], ids: &[usize]) -> Vec<f32> {
        let d_sae = self.d_sae();
        ids.iter()
            .map(|&i| {
                let mut pre = f64::from(self.weights.b_enc[i]);
                for j in 0..self.d_in() {
                    pre += self.input(row, j) * f64::from(self.weights.w_enc[j * d_sae + i]);
                }
                self.activate(i, pre as f32)
            })
            .collect()
    }

    /// Every feature activation of `row`.
    pub fn encode(&self, row: &[f32]) -> Vec<f32> {
        let d_sae = self.d_sae();
        let mut pre: Vec<f64> = self.weights.b_enc.iter().map(|&b| f64::from(b)).collect();
        for j in 0..self.d_in() {
            let input = self.input(row, j);
            let weights = &self.weights.w_enc[j * d_sae..(j + 1) * d_sae];
            for (acc, &weight) in pre.iter_mut().zip(weights) {
                *acc += input * f64::from(weight);
            }
        }
        pre.iter()
            .enumerate()
            .map(|(i, &pre)| self.activate(i, pre as f32))
            .collect()
    }

    /// Add `(after[k] - before[k]) * W_dec[ids[k]]` to `row`: the change of
    /// each feature is exact in f64, and each column adds an f64 sum in
    /// `ids` order once and rounds. An unchanged feature contributes nothing,
    /// and a change of nothing leaves `row` bit-identical.
    pub fn add_decoded(&self, row: &mut [f32], ids: &[usize], before: &[f32], after: &[f32]) {
        let delta: Vec<f64> = after
            .iter()
            .zip(before)
            .map(|(&after, &before)| f64::from(after) - f64::from(before))
            .collect();
        if delta.iter().all(|&change| change == 0.0) {
            return;
        }
        let d_in = self.d_in();
        for (j, value) in row.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for (&i, &change) in ids.iter().zip(&delta) {
                sum += change * f64::from(self.weights.w_dec[i * d_in + j]);
            }
            *value = (f64::from(*value) + sum) as f32;
        }
    }

    /// The reconstruction `b_dec + f · W_dec`, in f64.
    pub fn reconstruct(&self, features: &[f32]) -> Vec<f64> {
        let d_in = self.d_in();
        let mut out: Vec<f64> = self.weights.b_dec.iter().map(|&b| f64::from(b)).collect();
        for (i, &value) in features.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let weights = &self.weights.w_dec[i * d_in..(i + 1) * d_in];
            for (acc, &weight) in out.iter_mut().zip(weights) {
                *acc += f64::from(value) * f64::from(weight);
            }
        }
        out
    }

    /// The feature-capture record of one row.
    pub fn feature_row(
        &self,
        row: &[f32],
        top_k: usize,
        input_id: &str,
        layer: usize,
        position: usize,
    ) -> FeatureRow {
        let features = self.encode(row);
        let reconstruction = self.reconstruct(&features);
        let norm = row_norm(row);
        let error = row
            .iter()
            .zip(&reconstruction)
            .map(|(&x, &x_hat)| (f64::from(x) - x_hat).powi(2))
            .sum::<f64>()
            .sqrt();
        let mut active: Vec<(usize, f32)> = features
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, value)| value > 0.0)
            .collect();
        let l0 = active.len();
        // Strongest first; equal values by the lower index.
        active.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        active.truncate(top_k);
        FeatureRow {
            input_id: input_id.to_string(),
            layer,
            position,
            norm,
            relative_error: if norm > 0.0 { error / norm } else { 0.0 },
            l0,
            features: active,
        }
    }
}

/// `sqrt(sum x_j^2)` in f64 in index order: the norm a feature row records
/// and the one verify recomputes from the stored row.
pub fn row_norm(row: &[f32]) -> f64 {
    row.iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt()
}

fn read_capped(path: &Path, cap: u64) -> Result<Vec<u8>, String> {
    crate::bounded_read::read_regular_file(path, cap)
        .map_err(|error| format!("cannot read '{}': {error}", path.display()))?
        .ok_or_else(|| format!("'{}' does not exist", path.display()))
}

/// One stored row of a feature capture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureRow {
    pub input_id: String,
    pub layer: usize,
    pub position: usize,
    /// The norm of the raw row ([`row_norm`]).
    pub norm: f64,
    /// `|x - x̂| / |x|` (0 for a zero row).
    pub relative_error: f64,
    /// The number of active features.
    pub l0: usize,
    /// The strongest active features, `(index, value)`, strongest first.
    pub features: Vec<(usize, f32)>,
}

/// The feature record of one capture (`artifacts/features/<capture>.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureArtifact {
    pub schema: String,
    pub capture_id: String,
    pub dictionary: String,
    pub dictionary_sha256: String,
    pub top_k: usize,
    pub rows: Vec<FeatureRow>,
}

/// The bundle files of a run's dictionaries and feature captures: one
/// record per dictionary and one feature artifact per feature capture. The
/// rows of a feature artifact follow the spec's input order, then the
/// layer, then the stored position: the order verify reads them in.
pub fn artifact_files(
    spec: &crate::v05::spec::ExperimentSpecV1,
    dictionaries: &[Arc<Dictionary>],
    captures: &[&crate::v05::runner::CapturedTensor],
) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut files = BTreeMap::new();
    for dictionary in dictionaries {
        files.insert(
            format!("{DICTIONARY_DIR}/{}.json", dictionary.record.id),
            to_json_bytes(&dictionary.record)?,
        );
    }
    let input_order = |id: &str| spec.inputs.iter().position(|input| input.id == id);
    for capture in &spec.captures {
        let Some(features) = &capture.features else {
            continue;
        };
        let dictionary = dictionaries
            .iter()
            .find(|dictionary| dictionary.record.id == features.dictionary)
            .ok_or_else(|| {
                format!(
                    "capture '{}': dictionary '{}' was not loaded",
                    capture.id, features.dictionary
                )
            })?;
        let mut tensors: Vec<_> = captures
            .iter()
            .filter(|tensor| tensor.capture_id == capture.id)
            .collect();
        tensors.sort_by_key(|tensor| (input_order(&tensor.input_id), tensor.layer));
        let mut rows = Vec::new();
        for tensor in tensors {
            for (index, &position) in tensor.positions.iter().enumerate() {
                let values = &tensor.rows[index * tensor.columns..(index + 1) * tensor.columns];
                rows.push(dictionary.feature_row(
                    values,
                    features.top_k,
                    &tensor.input_id,
                    tensor.layer,
                    position,
                ));
            }
        }
        let artifact = FeatureArtifact {
            schema: FEATURES_SCHEMA_V1.into(),
            capture_id: capture.id.clone(),
            dictionary: features.dictionary.clone(),
            dictionary_sha256: dictionary.record.sha256.clone(),
            top_k: features.top_k,
            rows,
        };
        files.insert(
            format!("{FEATURES_DIR}/{}.json", capture.id),
            to_json_bytes(&artifact)?,
        );
    }
    Ok(files)
}

/// The JSON bytes of `value`, pretty-printed with a final newline, as the
/// other artifact records are written.
pub fn to_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Check the dictionary records of a bundle against its spec: one record
/// per `[[dictionaries]]` entry, equal in every field the spec states, and
/// no other record.
pub fn verify_dictionary_records(
    specs: &[DictionarySpec],
    listed: &[String],
    file: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Vec<String> {
    let mut errors = Vec::new();
    for spec in specs {
        let name = format!("{DICTIONARY_DIR}/{}.json", spec.id);
        let Some(bytes) = file(&name) else {
            errors.push(format!("dictionary '{}' has no record at {name}", spec.id));
            continue;
        };
        let record: DictionaryRecord = match serde_json::from_slice(&bytes) {
            Ok(record) => record,
            Err(error) => {
                errors.push(format!("{name} does not parse: {error}"));
                continue;
            }
        };
        if record.schema != DICTIONARY_SCHEMA_V1 {
            errors.push(format!("{name} has schema '{}'", record.schema));
        }
        if record.id != spec.id
            || record.path != spec.path
            || record.sha256 != spec.sha256
            || record.config_sha256 != spec.config_sha256
            || record.site != spec.site
            || record.layer != spec.layer
        {
            errors.push(format!(
                "{name} differs from the spec's dictionary '{}'",
                spec.id
            ));
        }
    }
    for name in listed {
        if let Some(rest) = name.strip_prefix(&format!("{DICTIONARY_DIR}/"))
            && !specs.iter().any(|spec| rest == format!("{}.json", spec.id))
        {
            errors.push(format!("unexpected dictionary record '{name}'"));
        }
    }
    errors
}

/// One stored row of a capture, as verify reads it from the payload.
pub struct StoredRow<'a> {
    pub input_id: &'a str,
    pub layer: usize,
    pub position: usize,
    pub values: &'a [f32],
}

/// Check the feature artifact of one capture against the capture's stored
/// rows and its dictionary's record.
pub fn verify_feature_artifact(
    capture_id: &str,
    spec: &FeatureCapture,
    dictionary: Option<&DictionaryRecord>,
    stored: &[StoredRow<'_>],
    bytes: &[u8],
) -> Vec<String> {
    let name = format!("{FEATURES_DIR}/{capture_id}.json");
    let artifact: FeatureArtifact = match serde_json::from_slice(bytes) {
        Ok(artifact) => artifact,
        Err(error) => return vec![format!("{name} does not parse: {error}")],
    };
    let mut errors = Vec::new();
    if artifact.schema != FEATURES_SCHEMA_V1 {
        errors.push(format!("{name} has schema '{}'", artifact.schema));
    }
    if artifact.capture_id != capture_id
        || artifact.dictionary != spec.dictionary
        || artifact.top_k != spec.top_k
    {
        errors.push(format!(
            "{name} differs from capture '{capture_id}' in the spec"
        ));
    }
    let d_sae = match dictionary {
        Some(record) => {
            if artifact.dictionary_sha256 != record.sha256 {
                errors.push(format!("{name} names a different dictionary hash"));
            }
            Some(record.d_sae)
        }
        None => {
            errors.push(format!(
                "{name}: dictionary '{}' has no record",
                spec.dictionary
            ));
            None
        }
    };
    if artifact.rows.len() != stored.len() {
        errors.push(format!(
            "{name} has {} rows; the capture stores {}",
            artifact.rows.len(),
            stored.len()
        ));
        return errors;
    }
    for (row, stored) in artifact.rows.iter().zip(stored) {
        let at = format!(
            "{name} row (input {}, layer {}, position {})",
            row.input_id, row.layer, row.position
        );
        if row.input_id != stored.input_id
            || row.layer != stored.layer
            || row.position != stored.position
        {
            errors.push(format!("{at} does not match the capture's stored row"));
            continue;
        }
        // To within JSON precision: serde_json's default float parser can
        // read a written f64 back one unit in the last place off.
        let norm = row_norm(stored.values);
        if (row.norm - norm).abs() > 1e-12 * norm {
            errors.push(format!("{at}: the norm differs from the stored row"));
        }
        if !row.relative_error.is_finite() || row.relative_error < 0.0 {
            errors.push(format!(
                "{at}: the relative error is not a finite value >= 0"
            ));
        }
        if row.features.len() != row.l0.min(spec.top_k) {
            errors.push(format!(
                "{at}: {} features listed for l0 {} and top_k {}",
                row.features.len(),
                row.l0,
                spec.top_k
            ));
        }
        if d_sae.is_some_and(|d_sae| row.l0 > d_sae) {
            errors.push(format!("{at}: l0 exceeds the dictionary's size"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for (index, &(feature, value)) in row.features.iter().enumerate() {
            if d_sae.is_some_and(|d_sae| feature >= d_sae) || !seen.insert(feature) {
                errors.push(format!(
                    "{at}: feature {feature} is out of range or repeated"
                ));
            }
            if !(value.is_finite() && value > 0.0) {
                errors.push(format!(
                    "{at}: feature {feature} has a value that is not > 0"
                ));
            }
            if let Some(&(previous_feature, previous)) = index
                .checked_sub(1)
                .and_then(|previous| row.features.get(previous))
                && (previous < value || (previous == value && previous_feature > feature))
            {
                errors.push(format!("{at}: the features are not strongest first"));
            }
        }
    }
    errors
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::v05::safetensors::{serialize, TensorDType, TensorData};

    /// A small dictionary's weights file: `d_in` 3, `d_sae` 4, with simple
    /// values that make each feature easy to predict.
    pub(crate) fn tiny_weights(jumprelu: bool) -> Vec<u8> {
        // Feature i reads column i % 3; feature 3 reads the sum of all.
        let w_enc: Vec<f32> = vec![
            1.0, 0.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, 1.0,
        ];
        let b_enc = [0.0f32, -0.5, 0.0, -1.0];
        let w_dec: Vec<f32> = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.5, 0.5, 0.5,
        ];
        let b_dec = [0.1f32, 0.0, -0.1];
        let threshold = [0.25f32, 0.0, 2.0, 0.0];
        let bytes = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        let (enc, benc, dec, bdec, thr) = (
            bytes(&w_enc),
            bytes(&b_enc),
            bytes(&w_dec),
            bytes(&b_dec),
            bytes(&threshold),
        );
        let mut tensors = vec![
            TensorData {
                name: "W_enc",
                dtype: TensorDType::F32,
                shape: &[3, 4],
                bytes: &enc,
            },
            TensorData {
                name: "b_enc",
                dtype: TensorDType::F32,
                shape: &[4],
                bytes: &benc,
            },
            TensorData {
                name: "W_dec",
                dtype: TensorDType::F32,
                shape: &[4, 3],
                bytes: &dec,
            },
            TensorData {
                name: "b_dec",
                dtype: TensorDType::F32,
                shape: &[3],
                bytes: &bdec,
            },
        ];
        if jumprelu {
            tensors.push(TensorData {
                name: "threshold",
                dtype: TensorDType::F32,
                shape: &[4],
                bytes: &thr,
            });
        }
        serialize(&tensors).unwrap()
    }

    pub(crate) fn tiny_config(architecture: &str) -> String {
        format!(
            r#"{{"architecture": "{architecture}", "d_in": 3, "d_sae": 4,
                "activation_fn": "relu", "apply_b_dec_to_input": false,
                "normalize_activations": "none", "model_name": "tiny",
                "hook_name": "blocks.0.hook_resid_post", "lr": 0.0001}}"#
        )
    }

    pub(crate) fn tiny_spec(weights: &[u8]) -> DictionarySpec {
        DictionarySpec {
            id: "tiny".into(),
            path: PathBuf::from("/nonexistent/tiny"),
            sha256: sha256_hex(weights),
            config_sha256: sha256_hex(tiny_config("jumprelu").as_bytes()),
            site: SemanticHookSite::ResidualPostMlp,
            layer: 0,
        }
    }

    fn tiny(architecture: &str) -> Dictionary {
        let weights = tiny_weights(architecture == "jumprelu");
        let config: SaeLensConfig = serde_json::from_str(&tiny_config(architecture)).unwrap();
        Dictionary::from_parts(&tiny_spec(&weights), &config, &weights).unwrap()
    }

    #[test]
    fn encode_follows_relu_and_jumprelu() {
        let row = [1.0f32, 0.25, 3.0];
        // pre = [1.0, -0.25, 3.0, 3.25]
        assert_eq!(tiny("standard").encode(&row), vec![1.0, 0.0, 3.0, 3.25]);
        // threshold [0.25, 0, 2, 0]: feature 2 passes (3 > 2), feature 1 is
        // negative, and feature 0 passes (1 > 0.25).
        assert_eq!(tiny("jumprelu").encode(&row), vec![1.0, 0.0, 3.0, 3.25]);
        // Below a threshold, a positive pre-activation is still inactive.
        assert_eq!(
            tiny("jumprelu").encode(&[0.2, 0.0, 1.0]),
            vec![0.0, 0.0, 0.0, 0.2]
        );
        assert_eq!(
            tiny("standard").encode(&[0.2, 0.0, 1.0]),
            vec![0.2, 0.0, 1.0, 0.2]
        );
    }

    #[test]
    fn a_selected_encode_equals_the_full_encode() {
        let dictionary = tiny("jumprelu");
        let row = [0.7f32, -1.3, 2.9];
        let full = dictionary.encode(&row);
        assert_eq!(
            dictionary.encode_selected(&row, &[0, 2, 3]),
            vec![full[0], full[2], full[3]]
        );
    }

    #[test]
    fn both_selected_encode_paths_give_the_same_values() {
        // d_in 5, d_sae 200: large enough for the full-encode path.
        let (d_in, d_sae) = (5usize, 200usize);
        let values = |count: usize, seed: usize| -> Vec<u8> {
            (0..count)
                .flat_map(|k| ((((k * 37 + seed) % 23) as f32 - 11.0) * 0.07).to_le_bytes())
                .collect()
        };
        let (enc, benc, dec, bdec) = (
            values(d_in * d_sae, 1),
            values(d_sae, 2),
            values(d_sae * d_in, 3),
            values(d_in, 4),
        );
        let weights = serialize(&[
            TensorData {
                name: "W_enc",
                dtype: TensorDType::F32,
                shape: &[d_in, d_sae],
                bytes: &enc,
            },
            TensorData {
                name: "b_enc",
                dtype: TensorDType::F32,
                shape: &[d_sae],
                bytes: &benc,
            },
            TensorData {
                name: "W_dec",
                dtype: TensorDType::F32,
                shape: &[d_sae, d_in],
                bytes: &dec,
            },
            TensorData {
                name: "b_dec",
                dtype: TensorDType::F32,
                shape: &[d_in],
                bytes: &bdec,
            },
        ])
        .unwrap();
        let config: SaeLensConfig = serde_json::from_str(
            &tiny_config("standard")
                .replace("\"d_in\": 3", "\"d_in\": 5")
                .replace("\"d_sae\": 4", "\"d_sae\": 200"),
        )
        .unwrap();
        let dictionary = Dictionary::from_parts(&tiny_spec(&weights), &config, &weights).unwrap();
        let row = [0.3f32, -1.1, 2.4, 0.05, -0.7];
        let ids: Vec<usize> = (0..d_sae).step_by(2).collect();
        let full = dictionary.encode_selected(&row, &ids);
        assert_eq!(full, dictionary.encode_strided(&row, &ids));
        assert!(full.iter().any(|&value| value > 0.0));
    }

    #[test]
    fn a_decoded_edit_adds_only_the_change() {
        let dictionary = tiny("standard");
        let mut row = [1.0f32, 0.25, 3.0];
        let before = row;
        dictionary.add_decoded(&mut row, &[1, 3], &[0.0, 3.25], &[0.0, 3.25]);
        assert_eq!(
            row.map(f32::to_bits),
            before.map(f32::to_bits),
            "no change, no rounding"
        );
        // Removing feature 3 (value 3.25) subtracts 3.25 * [0.5, 0.5, 0.5].
        dictionary.add_decoded(&mut row, &[3], &[3.25], &[0.0]);
        assert_eq!(row, [1.0 - 1.625, 0.25 - 1.625, 3.0 - 1.625]);
    }

    #[test]
    fn a_feature_row_lists_the_strongest_features_and_the_fit() {
        let dictionary = tiny("standard");
        let row = [1.0f32, 0.25, 3.0];
        let record = dictionary.feature_row(&row, 2, "a", 0, 5);
        assert_eq!(record.l0, 3);
        assert_eq!(record.features, vec![(3, 3.25), (2, 3.0)]);
        assert_eq!(record.norm, row_norm(&row));
        // x̂ = b_dec + 1.0 e0 + 3.0 e2 + 3.25 * 0.5 (1, 1, 1)
        let x_hat = [
            f64::from(0.1f32) + 1.0 + 1.625,
            1.625,
            f64::from(-0.1f32) + 3.0 + 1.625,
        ];
        let error = row
            .iter()
            .zip(x_hat)
            .map(|(&x, x_hat)| (f64::from(x) - x_hat).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((record.relative_error - error / record.norm).abs() < 1e-12);
    }

    #[test]
    fn unsupported_configurations_are_refused() {
        let weights = tiny_weights(false);
        let spec = tiny_spec(&weights);
        let refuse = |config: &str| {
            let config: SaeLensConfig = serde_json::from_str(config).unwrap();
            Dictionary::from_parts(&spec, &config, &weights).unwrap_err()
        };
        assert!(refuse(&tiny_config("topk")).contains("architecture"));
        assert!(refuse(
            &tiny_config("standard").replace("\"none\"", "\"expected_average_only_in\"")
        )
        .contains("normalize_activations"));
        assert!(
            refuse(&tiny_config("standard").replace("\"relu\"", "\"tanh-relu\""))
                .contains("activation")
        );
        // SAELens before 5 names the activation activation_fn_str; a TopK SAE
        // says so there, with its k.
        let topk = tiny_config("standard").replace(
            "\"activation_fn\": \"relu\"",
            "\"activation_fn_str\": \"topk\", \"activation_fn_kwargs\": {\"k\": 32}",
        );
        assert!(refuse(&topk).contains("topk"));
        let kwargs = tiny_config("standard").replace(
            "\"activation_fn\": \"relu\"",
            "\"activation_fn_kwargs\": {\"k\": 32}",
        );
        assert!(refuse(&kwargs).contains("activation_fn_kwargs"));
        let scaled = tiny_config("standard")
            .replace("\"lr\"", "\"finetuning_scaling_factor\": true, \"lr\"");
        assert!(refuse(&scaled).contains("finetuning_scaling_factor"));
        let head = tiny_config("standard").replace("\"lr\"", "\"hook_head_index\": 3, \"lr\"");
        assert!(refuse(&head).contains("hook_head_index"));
        let unstated = tiny_config("standard").replace("\"apply_b_dec_to_input\": false,", "");
        assert!(refuse(&unstated).contains("apply_b_dec_to_input"));
        assert!(
            refuse(&tiny_config("standard").replace("\"d_in\": 3", "\"d_in\": 5"))
                .contains("shape")
        );
        // A JumpReLU config needs the threshold tensor.
        assert!(refuse(&tiny_config("jumprelu")).contains("threshold"));
    }

    #[test]
    fn loading_checks_the_pinned_hash() {
        let directory =
            std::env::temp_dir().join(format!("ember-dictionary-load-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let weights = tiny_weights(true);
        std::fs::write(directory.join(WEIGHTS_FILE), &weights).unwrap();
        std::fs::write(directory.join(CONFIG_FILE), tiny_config("jumprelu")).unwrap();
        let mut spec = tiny_spec(&weights);
        spec.path = directory.clone();
        let dictionary = Dictionary::load(&spec).unwrap();
        assert_eq!(dictionary.record.architecture, Architecture::Jumprelu);
        assert_eq!(
            dictionary.record.hook_name.as_deref(),
            Some("blocks.0.hook_resid_post")
        );
        // Both files are pinned: cfg.json chooses how the weights encode.
        let mut weights_changed = spec.clone();
        weights_changed.sha256 = "0".repeat(64);
        assert!(Dictionary::load(&weights_changed)
            .unwrap_err()
            .contains(WEIGHTS_FILE));
        std::fs::write(directory.join(CONFIG_FILE), tiny_config("standard")).unwrap();
        assert!(Dictionary::load(&spec).unwrap_err().contains(CONFIG_FILE));
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn verify_checks_a_feature_artifact_against_the_stored_rows() {
        let dictionary = tiny("standard");
        let spec = FeatureCapture {
            dictionary: "tiny".into(),
            top_k: 2,
        };
        let rows = [[1.0f32, 0.25, 3.0], [0.0, 2.0, 0.0]];
        let artifact = FeatureArtifact {
            schema: FEATURES_SCHEMA_V1.into(),
            capture_id: "cap".into(),
            dictionary: "tiny".into(),
            dictionary_sha256: dictionary.record.sha256.clone(),
            top_k: 2,
            rows: rows
                .iter()
                .enumerate()
                .map(|(position, row)| dictionary.feature_row(row, 2, "a", 0, position))
                .collect(),
        };
        let stored: Vec<StoredRow<'_>> = rows
            .iter()
            .enumerate()
            .map(|(position, row)| StoredRow {
                input_id: "a",
                layer: 0,
                position,
                values: row,
            })
            .collect();
        let check = |artifact: &FeatureArtifact| {
            verify_feature_artifact(
                "cap",
                &spec,
                Some(&dictionary.record),
                &stored,
                &to_json_bytes(artifact).unwrap(),
            )
        };
        assert_eq!(check(&artifact), Vec::<String>::new());

        let mut tampered = artifact.clone();
        tampered.rows[0].norm += 1e-9;
        assert!(check(&tampered)[0].contains("norm"));
        // One unit in the last place, as a JSON round trip can give, passes.
        let mut nudged = artifact.clone();
        nudged.rows[0].norm = f64::from_bits(nudged.rows[0].norm.to_bits() + 1);
        assert_eq!(check(&nudged), Vec::<String>::new());
        let mut tampered = artifact.clone();
        tampered.rows[0].features.swap(0, 1);
        assert!(check(&tampered)[0].contains("strongest first"));
        let mut tampered = artifact.clone();
        tampered.rows[1].position = 7;
        assert!(check(&tampered)[0].contains("does not match"));
        let mut tampered = artifact;
        tampered.dictionary_sha256 = "0".repeat(64);
        assert!(check(&tampered)[0].contains("hash"));
    }

    #[test]
    fn verify_checks_dictionary_records_against_the_spec() {
        let dictionary = tiny("jumprelu");
        let spec = tiny_spec(&tiny_weights(true));
        let name = format!("{DICTIONARY_DIR}/tiny.json");
        let bytes = to_json_bytes(&dictionary.record).unwrap();
        let wanted = name.clone();
        let file = move |relative: &str| (relative == wanted).then(|| bytes.clone());
        assert!(verify_dictionary_records(
            std::slice::from_ref(&spec),
            std::slice::from_ref(&name),
            &file
        )
        .is_empty());
        let mut moved = spec.clone();
        moved.layer = 3;
        assert!(
            verify_dictionary_records(&[moved], std::slice::from_ref(&name), &file)[0]
                .contains("differs")
        );
        let extra = format!("{DICTIONARY_DIR}/other.json");
        assert!(verify_dictionary_records(&[spec], &[name, extra], &file)[0].contains("unexpected"));
    }
}
