//! Layer sweeps (`[sweep]` in an `ember.experiment.v1` specification).
//!
//! A sweep spec is an ordinary experiment spec plus a `[sweep]` table:
//!
//! ```toml
//! [sweep]
//! layers = "all"              # or [2, 4, 8], or { start = 0, end = 16, step = 2 }
//! positions = [3, 7]          # optional absolute token positions
//! interventions = ["replace"] # optional; default: every intervention
//! ```
//!
//! It describes one experiment per *point* (layer, or layer x position): the
//! swept interventions move to `layers = [L]` (and, with `positions`, to
//! `tokens = { kind = "absolute-token", index = P }`); everything else is
//! unchanged. A *baseline* experiment is the spec without interventions.
//!
//! `alphas = [0.0, 2.0, 4.0]` sweeps the `alpha` of the swept interventions
//! (`steer`, `interpolate`) as well, alone or crossed with layers (and
//! positions): `layers` may then be omitted, and the swept interventions stay
//! at their declared layers. Point ids gain an `alpha-<value>` part.
//!
//! Output layout: every point and the baseline is one ordinary, independently
//! verifiable and reproducible `ember.bundle.v1` bundle whose `experiment.toml`
//! is the derived spec (with a first-line comment naming the sweep spec's
//! SHA-256). The sweep directory adds a sweep manifest:
//!
//! ```text
//! <sweep>/sweep.toml          the sweep spec, byte for byte
//! <sweep>/sweep.json          ember.sweep.v1: identities + per-point metrics
//! <sweep>/sweep.csv           the metrics as a table
//! <sweep>/sweep-runtime.json  timings and prefix-reuse paths (not hashed)
//! <sweep>/baseline/           bundle
//! <sweep>/points/layer-07/    bundle (one per point)
//! ```
//!
//! One bundle per point, rather than one bundle with per-point outputs, keeps
//! the bundle schema, `verify`, `compare`, `inspect` and `reproduce` exactly
//! as they are for single experiments: a point bundle is bit-identical to
//! running its derived spec alone. `sweep.json` binds them: its `sweep_hash`
//! covers the sweep spec hash, every bundle's semantic and payload hash, and
//! the per-point metrics, all of which verification recomputes from the
//! bundles.

use crate::v05::capture::LayerSelector;
use crate::v05::compare::compare_loaded;
use crate::v05::intervention::InterventionSource;
use crate::v05::manifest::sha256_hex;
use crate::v05::spec::{ExperimentSpecV1, RawExperimentSpec, SpecError};
use crate::v05::verify::{load_verified_bundle, CheckResult, LoadedBundle, VerifyOptions};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

/// Sweep manifest schema identifier.
pub const SWEEP_SCHEMA_V1: &str = "ember.sweep.v1";
/// File names inside a sweep directory.
pub const SWEEP_SPEC_FILE: &str = "sweep.toml";
pub const SWEEP_MANIFEST_FILE: &str = "sweep.json";
pub const SWEEP_CSV_FILE: &str = "sweep.csv";
pub const SWEEP_RUNTIME_FILE: &str = "sweep-runtime.json";

/// The `[sweep]` table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawSweepSpec {
    /// Layers the swept interventions move to: `"all"`, a list, or a range.
    /// Optional when `alphas` is given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<LayerSelector>,
    /// `alpha` values for the swept interventions (`steer`, `interpolate`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alphas: Option<Vec<f64>>,
    /// Optional absolute token positions; each (layer, position) pair is a
    /// point and the swept interventions' tokens become `absolute-token`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<Vec<usize>>,
    /// Intervention ids to sweep (default: every intervention).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interventions: Option<Vec<String>>,
}

/// A parsed and validated sweep specification.
#[derive(Debug, Clone)]
pub struct SweepDefinition {
    /// The sweep spec text, byte for byte.
    pub text: String,
    pub spec_sha256: String,
    pub name: String,
    /// `None`: the swept interventions keep their declared layers.
    pub layers: Option<LayerSelector>,
    pub positions: Option<Vec<usize>>,
    /// Swept `alpha` values, ascending.
    pub alphas: Option<Vec<f64>>,
    /// Swept intervention ids, in declaration order.
    pub interventions: Vec<String>,
    /// The spec without `[sweep]`, resolved (interventions at their
    /// declared layers). Model, execution and generation come from here.
    pub template: ExperimentSpecV1,
    document: toml::Table,
}

/// One derived experiment (a point or the baseline).
#[derive(Debug, Clone)]
pub struct DerivedSpec {
    /// `baseline` or `layer-07` / `layer-07-pos-3` / `layer-07-alpha-2`.
    pub id: String,
    pub layer: Option<usize>,
    pub position: Option<usize>,
    pub alpha: Option<f64>,
    /// The derived spec text (becomes the bundle's `experiment.toml`).
    pub text: String,
    pub resolved: ExperimentSpecV1,
    /// Bundle directory relative to the sweep directory.
    pub relative_dir: String,
}

fn toml_error(path: &str, error: impl std::fmt::Display) -> SpecError {
    SpecError::at(path, format!("malformed sweep specification: {error}"))
}

impl SweepDefinition {
    /// Parse a sweep spec strictly and validate it without model metadata.
    pub fn parse(text: &str) -> Result<SweepDefinition, SpecError> {
        let raw = RawExperimentSpec::from_toml_str(text)?;
        let sweep = raw
            .sweep
            .clone()
            .ok_or_else(|| SpecError::at("sweep", "the specification has no [sweep] table"))?;
        let mut document: toml::Table =
            toml::from_str(text).map_err(|e| toml_error("<toml>", e))?;
        document.remove("sweep");
        let template_text = toml::to_string(&document).map_err(|e| toml_error("<toml>", e))?;
        let template = RawExperimentSpec::from_toml_str(&template_text)?.resolve()?;

        if template.attribution.is_some() || template.probe.is_some() {
            return Err(SpecError::at(
                "sweep",
                "an [attribution] workflow cannot be swept; run it on its own",
            ));
        }
        if template.interventions.is_empty() {
            return Err(SpecError::at(
                "sweep",
                "a sweep moves interventions across layers; declare at least one intervention",
            ));
        }
        let swept: Vec<String> = match &sweep.interventions {
            None => template
                .interventions
                .iter()
                .map(|i| i.id.clone())
                .collect(),
            Some(ids) => {
                if ids.is_empty() {
                    return Err(SpecError::at("sweep.interventions", "the list is empty"));
                }
                for (index, id) in ids.iter().enumerate() {
                    if ids[..index].contains(id) {
                        return Err(SpecError::at(
                            format!("sweep.interventions[{index}]"),
                            format!("intervention {id:?} is listed twice"),
                        ));
                    }
                    if !template.interventions.iter().any(|i| &i.id == id) {
                        return Err(SpecError::at(
                            format!("sweep.interventions[{index}]"),
                            format!("no intervention has id {id:?}"),
                        ));
                    }
                }
                // Declaration order, whatever order the list uses.
                template
                    .interventions
                    .iter()
                    .filter(|i| ids.contains(&i.id))
                    .map(|i| i.id.clone())
                    .collect()
            }
        };
        if sweep.layers.is_none() && sweep.alphas.is_none() {
            return Err(SpecError::at(
                "sweep",
                "a sweep needs `layers`, `alphas`, or both",
            ));
        }
        if sweep.layers.is_none() && sweep.positions.is_some() {
            return Err(SpecError::at(
                "sweep.positions",
                "positions are swept together with layers; add `layers`",
            ));
        }
        let alphas = match sweep.alphas {
            None => None,
            Some(list) if list.is_empty() => {
                return Err(SpecError::at("sweep.alphas", "the alpha list is empty"))
            }
            Some(mut list) => {
                for (index, alpha) in list.iter().enumerate() {
                    let survives = serde_json::to_string(alpha)
                        .ok()
                        .and_then(|text| serde_json::from_str::<f64>(&text).ok())
                        .is_some_and(|back| back.to_bits() == alpha.to_bits());
                    if !alpha.is_finite() || !survives {
                        return Err(SpecError::at(
                            format!("sweep.alphas[{index}]"),
                            format!("alpha {alpha} must be finite and a short decimal"),
                        ));
                    }
                }
                list.sort_by(f64::total_cmp);
                list.dedup_by(|a, b| a.to_bits() == b.to_bits());
                Some(list)
            }
        };
        for (index, intervention) in template.interventions.iter().enumerate() {
            if !swept.contains(&intervention.id) {
                continue;
            }
            let path = format!("interventions[{index}]");
            if alphas.is_some() && intervention.operation.alpha().is_none() {
                return Err(SpecError::at(
                    format!("{path}.operation"),
                    format!(
                        "an alpha sweep sets `alpha`; operation {} has none (use steer or \
                         interpolate, or leave this intervention out of sweep.interventions)",
                        intervention.operation.kind_name()
                    ),
                ));
            }
            if sweep.layers.is_none() {
                continue;
            }
            if !intervention.site.is_per_layer() {
                return Err(SpecError::at(
                    format!("{path}.site"),
                    format!(
                        "site {} has no layer; a layer sweep moves per-layer sites only",
                        intervention.site
                    ),
                ));
            }
            match &intervention.source {
                Some(InterventionSource::CaptureFromBundle { .. }) => {
                    return Err(SpecError::at(
                        format!("{path}.source"),
                        "a capture-from-bundle source names one fixed layer and cannot move \
                         across a layer sweep",
                    ));
                }
                Some(InterventionSource::CaptureFromCurrentRun { capture_id }) => {
                    let capture = template
                        .captures
                        .iter()
                        .find(|capture| &capture.id == capture_id);
                    if capture.is_some_and(|capture| capture.site != intervention.site) {
                        return Err(SpecError::at(
                            format!("{path}.source"),
                            format!(
                                "source capture {capture_id:?} must be at the swept site {} so \
                                 it has a value at every swept layer",
                                intervention.site
                            ),
                        ));
                    }
                }
                _ => {}
            }
        }
        match &sweep.layers {
            None => {}
            Some(LayerSelector::All(value)) if value != "all" => {
                return Err(SpecError::at(
                    "sweep.layers",
                    format!("expected the string \"all\", found {value:?}"),
                ))
            }
            Some(LayerSelector::List(list)) if list.is_empty() => {
                return Err(SpecError::at("sweep.layers", "the layer list is empty"))
            }
            Some(LayerSelector::Range(range)) if range.step == 0 || range.start >= range.end => {
                return Err(SpecError::at(
                    "sweep.layers",
                    "a layer range needs start < end and step >= 1",
                ))
            }
            _ => {}
        }
        let positions = match sweep.positions {
            None => None,
            Some(list) if list.is_empty() => {
                return Err(SpecError::at(
                    "sweep.positions",
                    "the position list is empty",
                ))
            }
            Some(mut list) => {
                list.sort_unstable();
                list.dedup();
                Some(list)
            }
        };
        Ok(SweepDefinition {
            text: text.to_string(),
            spec_sha256: sha256_hex(text.as_bytes()),
            name: template.experiment.name.clone(),
            layers: sweep.layers,
            positions,
            alphas,
            interventions: swept,
            template,
            document,
        })
    }

    /// The layers of this sweep for an `n_layers`-layer model (empty when
    /// the sweep does not move layers).
    pub fn resolve_layers(&self, n_layers: usize) -> Result<Vec<usize>, SpecError> {
        match &self.layers {
            None => Ok(Vec::new()),
            Some(layers) => layers
                .resolve(n_layers)
                .map_err(|error| SpecError::at("sweep.layers", error)),
        }
    }

    fn output_root(&self) -> String {
        self.template
            .output
            .directory
            .to_string_lossy()
            .trim_end_matches('/')
            .to_string()
    }

    fn derive(
        &self,
        id: &str,
        what: &str,
        mut document: toml::Table,
        relative_dir: String,
    ) -> Result<DerivedSpec, SpecError> {
        let set = |document: &mut toml::Table, table: &str, key: &str, value: toml::Value| {
            if let Some(toml::Value::Table(section)) = document.get_mut(table) {
                section.insert(key.to_string(), value);
            }
        };
        set(
            &mut document,
            "experiment",
            "name",
            toml::Value::String(format!("{}.{id}", self.name)),
        );
        set(
            &mut document,
            "output",
            "directory",
            toml::Value::String(format!("{}/{relative_dir}", self.output_root())),
        );
        let body = toml::to_string(&document).map_err(|e| toml_error("<toml>", e))?;
        let text = format!(
            "# Derived from sweep '{}' (sweep spec sha256 {}): {what}.\n{body}",
            self.name, self.spec_sha256
        );
        let resolved = RawExperimentSpec::from_toml_str(&text)?.resolve()?;
        Ok(DerivedSpec {
            id: id.to_string(),
            layer: None,
            position: None,
            alpha: None,
            text,
            resolved,
            relative_dir,
        })
    }

    /// The baseline: the spec with every intervention removed.
    pub fn baseline(&self) -> Result<DerivedSpec, SpecError> {
        let mut document = self.document.clone();
        document.remove("interventions");
        self.derive(
            "baseline",
            "baseline (no interventions)",
            document,
            "baseline".to_string(),
        )
    }

    /// Every point for an `n_layers`-layer model, layers ascending then
    /// positions ascending.
    pub fn points(&self, n_layers: usize) -> Result<Vec<DerivedSpec>, SpecError> {
        let resolved_layers = self.resolve_layers(n_layers)?;
        let layers: Vec<Option<usize>> = if self.layers.is_some() {
            resolved_layers.iter().copied().map(Some).collect()
        } else {
            vec![None]
        };
        let alphas: Vec<Option<f64>> = match &self.alphas {
            None => vec![None],
            Some(list) => list.iter().copied().map(Some).collect(),
        };
        // A current-run source must have a value at every swept layer.
        for intervention in &self.template.interventions {
            if !self.interventions.contains(&intervention.id) {
                continue;
            }
            if let Some(InterventionSource::CaptureFromCurrentRun { capture_id }) =
                &intervention.source
                && let Some(capture) = self.template.captures.iter().find(|c| &c.id == capture_id)
            {
                let covered = capture
                    .layers
                    .resolve(n_layers)
                    .map_err(|error| SpecError::at("captures", error))?;
                if let Some(missing) = resolved_layers
                    .iter()
                    .find(|layer| !covered.contains(layer))
                {
                    return Err(SpecError::at(
                        "sweep.layers",
                        format!(
                            "intervention {:?} takes its source from capture {capture_id:?}, \
                             which does not capture layer {missing}",
                            intervention.id
                        ),
                    ));
                }
            }
        }
        let width = n_layers.saturating_sub(1).to_string().len().max(2);
        let positions: Vec<Option<usize>> = match &self.positions {
            None => vec![None],
            Some(list) => list.iter().copied().map(Some).collect(),
        };
        let mut out = Vec::with_capacity(layers.len() * positions.len() * alphas.len());
        for &layer in &layers {
            for &position in &positions {
                for &alpha in &alphas {
                    let mut parts = Vec::new();
                    if let Some(layer) = layer {
                        parts.push(format!("layer-{layer:0width$}"));
                    }
                    if let Some(position) = position {
                        parts.push(format!("pos-{position}"));
                    }
                    if let Some(alpha) = alpha {
                        parts.push(alpha_id(alpha));
                    }
                    let id = parts.join("-");
                    let mut document = self.document.clone();
                    if let Some(toml::Value::Array(interventions)) =
                        document.get_mut("interventions")
                    {
                        for entry in interventions.iter_mut() {
                            let toml::Value::Table(table) = entry else {
                                continue;
                            };
                            let swept = table
                                .get("id")
                                .and_then(toml::Value::as_str)
                                .is_some_and(|id| self.interventions.iter().any(|s| s == id));
                            if !swept {
                                continue;
                            }
                            if let Some(layer) = layer {
                                table.insert(
                                    "layers".into(),
                                    toml::Value::Array(vec![toml::Value::Integer(layer as i64)]),
                                );
                            }
                            if let (Some(alpha), Some(toml::Value::Table(operation))) =
                                (alpha, table.get_mut("operation"))
                            {
                                operation.insert("alpha".into(), toml::Value::Float(alpha));
                            }
                            if let Some(position) = position {
                                let mut tokens = toml::Table::new();
                                tokens.insert("kind".into(), "absolute-token".into());
                                tokens
                                    .insert("index".into(), toml::Value::Integer(position as i64));
                                table.insert("tokens".into(), toml::Value::Table(tokens));
                            }
                        }
                    }
                    let mut coordinates = Vec::new();
                    if let Some(layer) = layer {
                        coordinates.push(format!("layer {layer}"));
                    }
                    if let Some(position) = position {
                        coordinates.push(format!("token {position}"));
                    }
                    if let Some(alpha) = alpha {
                        coordinates.push(format!("alpha {alpha}"));
                    }
                    let what = format!(
                        "point {id} (swept interventions at {})",
                        coordinates.join(", ")
                    );
                    let mut derived = self.derive(&id, &what, document, format!("points/{id}"))?;
                    derived.layer = layer;
                    derived.position = position;
                    derived.alpha = alpha;
                    out.push(derived);
                }
            }
        }
        Ok(out)
    }
}

/// The point-id part of an alpha: `alpha-2`, `alpha-0.5`, `alpha-neg-1.5`.
fn alpha_id(alpha: f64) -> String {
    if alpha < 0.0 {
        format!("alpha-neg-{}", -alpha)
    } else {
        format!("alpha-{}", alpha.abs())
    }
}

/// A bundle reference inside a sweep directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepBundleRef {
    /// Relative to the sweep directory.
    pub bundle: String,
    pub semantic_hash: String,
    pub payload_hash: String,
}

/// Per-input effect of one point, relative to the baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepInputMetrics {
    pub input_id: String,
    pub generated_tokens_equal: bool,
    pub generated_text_equal: bool,
    /// First 1-based decode step whose token differs from the baseline.
    pub first_divergent_step: Option<usize>,
    /// Largest relative L2 difference over every capture both bundles hold.
    pub peak_relative_l2: Option<f64>,
    pub peak_capture_id: Option<String>,
    pub peak_site: Option<String>,
    pub peak_layer: Option<usize>,
    pub captures_compared: usize,
    pub captures_exact: usize,
}

/// One point of a sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepPointRecord {
    pub id: String,
    /// `None` when the sweep does not move layers.
    pub layer: Option<usize>,
    pub position: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    #[serde(flatten)]
    pub bundle: SweepBundleRef,
    pub inputs: Vec<SweepInputMetrics>,
}

/// `sweep.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepManifest {
    pub schema: String,
    pub experiment: String,
    pub spec_file: String,
    pub spec_sha256: String,
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    pub layer_count: usize,
    pub layers: Vec<usize>,
    pub positions: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alphas: Option<Vec<f64>>,
    pub interventions: Vec<String>,
    pub baseline: SweepBundleRef,
    pub points: Vec<SweepPointRecord>,
    /// SHA-256 over this manifest's canonical JSON with this field empty.
    pub sweep_hash: String,
}

impl SweepManifest {
    /// The identity of the sweep: every field but `sweep_hash` itself.
    pub fn compute_hash(&self) -> String {
        let mut copy = self.clone();
        copy.sweep_hash = String::new();
        let bytes = serde_json::to_vec(&copy).expect("sweep manifest serializes");
        sha256_hex(&bytes)
    }

    /// `sweep.csv`: one row per (point, input).
    pub fn to_csv(&self) -> String {
        fn opt<T: std::fmt::Display>(value: &Option<T>) -> String {
            value.as_ref().map(ToString::to_string).unwrap_or_default()
        }
        // The alpha column exists only for alpha sweeps, so earlier sweeps
        // keep their exact table.
        let with_alpha = self.alphas.is_some();
        let mut out = String::from(if with_alpha {
            "point,layer,position,alpha,"
        } else {
            "point,layer,position,"
        });
        out.push_str(
            "input_id,first_divergent_step,generated_text_equal,\
             peak_relative_l2,peak_capture_id,peak_site,peak_layer,captures_exact,\
             captures_compared,semantic_hash\n",
        );
        for point in &self.points {
            for input in &point.inputs {
                let alpha = if with_alpha {
                    format!("{},", opt(&point.alpha))
                } else {
                    String::new()
                };
                out.push_str(&format!(
                    "{},{},{},{alpha}{},{},{},{},{},{},{},{},{},{}\n",
                    point.id,
                    opt(&point.layer),
                    opt(&point.position),
                    input.input_id,
                    opt(&input.first_divergent_step),
                    input.generated_text_equal,
                    opt(&input.peak_relative_l2),
                    opt(&input.peak_capture_id),
                    opt(&input.peak_site),
                    opt(&input.peak_layer),
                    input.captures_exact,
                    input.captures_compared,
                    point.bundle.semantic_hash,
                ));
            }
        }
        out
    }
}

/// A metric value that survives a trip through `sweep.json` unchanged.
///
/// serde_json's default float parser is best-effort: it is exact only when
/// the decimal significand fits 53 bits and the decimal exponent is within
/// +-22 (one exactly representable power of ten), and can be an ulp off
/// otherwise, so a full-precision metric could fail its own verification.
/// Rounding to 9 significant digits, and never finer than 1e-22, keeps the
/// shortest representation serde_json writes inside that exact range (the
/// shortest form never has more digits or a finer last digit than the
/// rounded decimal). Magnitudes below 1e-22 round to zero.
pub fn json_stable(value: f64) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    let magnitude = value.abs().log10().floor() as i32;
    let text = if 8 - magnitude > 22 {
        format!("{value:.22}")
    } else {
        format!("{value:.8e}")
    };
    text.parse().unwrap_or(value)
}

/// Metrics of `point` against `baseline`, per input (both verified).
pub fn point_metrics(
    baseline: &LoadedBundle,
    point: &LoadedBundle,
) -> Result<Vec<SweepInputMetrics>, String> {
    let comparison = compare_loaded(baseline, point)?;
    Ok(comparison
        .outputs
        .iter()
        .map(|output| {
            let mut compared = 0usize;
            let mut exact = 0usize;
            let mut peak: Option<(f64, &crate::v05::compare::CaptureComparison)> = None;
            for capture in comparison.captures.iter().filter(|capture| {
                capture.input_id == output.input_id && capture.present_in_a && capture.present_in_b
            }) {
                let Some(metrics) = &capture.metrics else {
                    continue;
                };
                compared += 1;
                if metrics.exact {
                    exact += 1;
                }
                if let Some(value) = metrics.relative_l2_difference.filter(|v| v.is_finite())
                    && peak.is_none_or(|(best, _)| value > best)
                {
                    peak = Some((value, capture));
                }
            }
            // Name a location only for a real difference.
            let located = peak.filter(|(value, _)| *value > 0.0);
            SweepInputMetrics {
                input_id: output.input_id.clone(),
                generated_tokens_equal: output.generated_tokens_equal,
                generated_text_equal: output.generated_text_equal,
                first_divergent_step: output.first_divergence_step,
                peak_relative_l2: peak.map(|(value, _)| json_stable(value)),
                peak_capture_id: located.map(|(_, capture)| capture.capture_id.clone()),
                peak_site: located.map(|(_, capture)| capture.site.to_string()),
                peak_layer: located.map(|(_, capture)| capture.layer),
                captures_compared: compared,
                captures_exact: exact,
            }
        })
        .collect())
}

/// Resolve a manifest-relative bundle path, refusing anything that could
/// leave the sweep directory.
fn inside(dir: &Path, relative: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if relative.is_empty()
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "bundle path {relative:?} is not a plain relative path"
        ));
    }
    Ok(dir.join(path))
}

/// Result of verifying a sweep directory.
#[derive(Debug, Clone, Serialize)]
pub struct SweepVerification {
    pub schema: String,
    pub ok: bool,
    pub sweep_hash: String,
    pub bundles: usize,
    pub checks: Vec<CheckResult>,
}

impl SweepVerification {
    fn check(&mut self, name: impl Into<String>, ok: bool, detail: impl Into<String>) {
        self.ok &= ok;
        self.checks.push(CheckResult {
            name: name.into(),
            ok,
            detail: detail.into(),
        });
    }
}

/// Whether `dir` holds a sweep manifest.
pub fn is_sweep_dir(dir: &Path) -> bool {
    dir.join(SWEEP_MANIFEST_FILE).is_file()
}

/// Read a sweep-level file, refusing symlinks and special files like the
/// bundle verifier does: what is verified must be the directory's own file.
fn read_sweep_file(path: &Path) -> std::io::Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a regular file",
        ));
    }
    std::fs::read(path)
}

/// Read and parse `sweep.json`.
pub fn read_manifest(dir: &Path) -> Result<SweepManifest, String> {
    let bytes = read_sweep_file(&dir.join(SWEEP_MANIFEST_FILE))
        .map_err(|error| format!("cannot read {SWEEP_MANIFEST_FILE}: {error}"))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("malformed {SWEEP_MANIFEST_FILE}: {error}"))
}

/// Verify a sweep directory: the manifest and its hash, the sweep spec, every
/// bundle (fully, with `options` for deep model/tokenizer checks), that each
/// point bundle is exactly the experiment the sweep spec derives for it, and
/// that the recorded metrics are what the bundles give.
pub fn verify_sweep(dir: &Path, options: &VerifyOptions) -> Result<SweepVerification, String> {
    verify_and_load_sweep(dir, options).map(|(report, _)| report)
}

/// [`verify_sweep`], also returning the manifest that was verified so
/// callers never re-read `sweep.json` after verification.
fn verify_and_load_sweep(
    dir: &Path,
    options: &VerifyOptions,
) -> Result<(SweepVerification, SweepManifest), String> {
    let manifest = read_manifest(dir)?;
    let mut report = SweepVerification {
        schema: SWEEP_SCHEMA_V1.to_string(),
        ok: true,
        sweep_hash: manifest.sweep_hash.clone(),
        bundles: 0,
        checks: Vec::new(),
    };
    report.check(
        "sweep schema",
        manifest.schema == SWEEP_SCHEMA_V1,
        format!("{} (expected {SWEEP_SCHEMA_V1})", manifest.schema),
    );
    let recomputed = manifest.compute_hash();
    report.check(
        "sweep hash",
        recomputed == manifest.sweep_hash,
        format!("recomputed {recomputed}"),
    );
    if let Some(expected) = &options.expected_semantic_hash {
        report.check(
            "sweep hash anchor",
            expected == &manifest.sweep_hash,
            format!("expected {expected}"),
        );
    }

    // The sweep spec and what it derives.
    let spec_path = inside(dir, &manifest.spec_file)?;
    let spec_text = read_sweep_file(&spec_path)
        .and_then(|bytes| {
            String::from_utf8(bytes)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
        })
        .map_err(|error| format!("cannot read {}: {error}", manifest.spec_file))?;
    let spec_sha = sha256_hex(spec_text.as_bytes());
    report.check(
        "sweep spec hash",
        spec_sha == manifest.spec_sha256,
        format!("{} hashes to {spec_sha}", manifest.spec_file),
    );
    let definition = SweepDefinition::parse(&spec_text).map_err(|error| error.to_string())?;
    let baseline_spec = definition.baseline().map_err(|error| error.to_string())?;
    // `layer_count` sizes the derived point list, so bound it before use. No
    // model the loader accepts has more layers.
    if manifest.layer_count > crate::loader::limits::MAX_LAYERS {
        report.check(
            "layer count",
            false,
            format!(
                "{} layers exceeds the {}-layer model limit",
                manifest.layer_count,
                crate::loader::limits::MAX_LAYERS
            ),
        );
        return Ok((report, manifest));
    }
    let derived = definition
        .points(manifest.layer_count)
        .map_err(|error| error.to_string())?;
    let layers = definition
        .resolve_layers(manifest.layer_count)
        .map_err(|error| error.to_string())?;
    report.check(
        "sweep definition",
        definition.name == manifest.experiment
            && layers == manifest.layers
            && definition.positions == manifest.positions
            && definition.alphas == manifest.alphas
            && definition.interventions == manifest.interventions,
        "experiment name, layers, positions, alphas and swept interventions match the spec",
    );
    let derived_ids: Vec<&str> = derived.iter().map(|point| point.id.as_str()).collect();
    let recorded_ids: Vec<&str> = manifest
        .points
        .iter()
        .map(|point| point.id.as_str())
        .collect();
    report.check(
        "sweep points",
        derived_ids == recorded_ids,
        format!(
            "{} recorded, {} derived from the spec",
            recorded_ids.len(),
            derived_ids.len()
        ),
    );

    // Bundles.
    let load = |reference: &SweepBundleRef| -> Result<LoadedBundle, String> {
        let path = inside(dir, &reference.bundle)?;
        let bundle_options = VerifyOptions {
            expected_semantic_hash: Some(reference.semantic_hash.clone()),
            ..options.clone()
        };
        load_verified_bundle(&path, &bundle_options)
    };
    let baseline = match load(&manifest.baseline) {
        Ok(bundle) => bundle,
        Err(error) => {
            report.check("baseline bundle", false, error);
            return Ok((report, manifest));
        }
    };
    report.bundles += 1;
    let bundle_checks = |report: &mut SweepVerification,
                         label: &str,
                         bundle: &LoadedBundle,
                         reference: &SweepBundleRef,
                         expected_text: &str| {
        report.check(
            format!("{label} bundle"),
            bundle.payload_hash == reference.payload_hash
                && bundle.semantic_manifest.model.sha256 == manifest.model_sha256
                && bundle.semantic_manifest.tokenizer.sha256 == manifest.tokenizer_sha256,
            format!(
                "verified; semantic {} payload {}",
                bundle.semantic_hash, bundle.payload_hash
            ),
        );
        let text = bundle
            .file("experiment.toml")
            .and_then(|bytes| std::str::from_utf8(bytes).ok());
        report.check(
            format!("{label} derivation"),
            text == Some(expected_text),
            "experiment.toml is the spec the sweep derives for it",
        );
    };
    bundle_checks(
        &mut report,
        "baseline",
        &baseline,
        &manifest.baseline,
        &baseline_spec.text,
    );
    report.check(
        "layer count",
        baseline.semantic_manifest.model.layer_count == manifest.layer_count,
        format!(
            "the model has {} layers",
            baseline.semantic_manifest.model.layer_count
        ),
    );
    for record in &manifest.points {
        let label = format!("point {}", record.id);
        let bundle = match load(&record.bundle) {
            Ok(bundle) => bundle,
            Err(error) => {
                report.check(format!("{label} bundle"), false, error);
                continue;
            }
        };
        report.bundles += 1;
        let expected = derived
            .iter()
            .find(|point| point.id == record.id)
            .map(|point| {
                (
                    point.text.as_str(),
                    point.layer,
                    point.position,
                    point.alpha,
                )
            });
        match expected {
            Some((text, layer, position, alpha)) => {
                bundle_checks(&mut report, &label, &bundle, &record.bundle, text);
                report.check(
                    format!("{label} coordinates"),
                    layer == record.layer
                        && position == record.position
                        && alpha.map(f64::to_bits) == record.alpha.map(f64::to_bits),
                    format!(
                        "layer {:?} position {:?} alpha {:?}",
                        record.layer, record.position, record.alpha
                    ),
                );
            }
            None => report.check(
                format!("{label} derivation"),
                false,
                "the sweep spec does not derive this point",
            ),
        }
        match point_metrics(&baseline, &bundle) {
            Ok(metrics) => report.check(
                format!("{label} metrics"),
                metrics == record.inputs,
                "recorded metrics equal the metrics recomputed from the bundles",
            ),
            Err(error) => report.check(format!("{label} metrics"), false, error),
        }
    }
    // The CSV is a view of the manifest.
    match read_sweep_file(&dir.join(SWEEP_CSV_FILE)) {
        Ok(csv) => report.check(
            "sweep csv",
            csv == manifest.to_csv().as_bytes(),
            "sweep.csv is the manifest's table",
        ),
        Err(error) => report.check("sweep csv", false, error.to_string()),
    }
    Ok((report, manifest))
}

/// One point (or the baseline) compared across two sweeps.
#[derive(Debug, Clone, Serialize)]
pub struct SweepPointComparison {
    pub id: String,
    pub present_in_a: bool,
    pub present_in_b: bool,
    pub semantic_hash_equal: bool,
    pub payload_hash_equal: bool,
    pub metrics_equal: bool,
}

/// Two verified sweeps compared point by point.
#[derive(Debug, Clone, Serialize)]
pub struct SweepComparison {
    pub sweep_a: String,
    pub sweep_b: String,
    pub sweep_hash_equal: bool,
    pub spec_equal: bool,
    pub points: Vec<SweepPointComparison>,
    /// `exact` when every point (and the baseline) is present in both with
    /// equal semantic hashes; `different` otherwise.
    pub verdict: String,
}

/// Compare two sweep directories (each verified first).
pub fn compare_sweeps(a: &Path, b: &Path) -> Result<SweepComparison, String> {
    compare_anchored_sweeps(a, b, None, None)
}

/// [`compare_sweeps`] with optional expected sweep hashes, checked in the
/// same verification pass whose manifests are compared.
pub fn compare_anchored_sweeps(
    a: &Path,
    b: &Path,
    expect_a: Option<&str>,
    expect_b: Option<&str>,
) -> Result<SweepComparison, String> {
    let mut manifests = Vec::with_capacity(2);
    for (dir, expected) in [(a, expect_a), (b, expect_b)] {
        let options = VerifyOptions {
            expected_semantic_hash: expected.map(str::to_string),
            ..VerifyOptions::default()
        };
        let (report, manifest) = verify_and_load_sweep(dir, &options)?;
        manifests.push(manifest);
        if !report.ok {
            let failed: Vec<String> = report
                .checks
                .iter()
                .filter(|check| !check.ok)
                .map(|check| format!("{}: {}", check.name, check.detail))
                .collect();
            return Err(format!(
                "sweep '{}' failed verification: {}",
                dir.display(),
                failed.join("; ")
            ));
        }
    }
    let (mb, ma) = (manifests.pop().expect("two"), manifests.pop().expect("two"));
    let mut points = vec![SweepPointComparison {
        id: "baseline".into(),
        present_in_a: true,
        present_in_b: true,
        semantic_hash_equal: ma.baseline.semantic_hash == mb.baseline.semantic_hash,
        payload_hash_equal: ma.baseline.payload_hash == mb.baseline.payload_hash,
        metrics_equal: true,
    }];
    let mut ids: Vec<&str> = ma.points.iter().map(|p| p.id.as_str()).collect();
    for point in &mb.points {
        if !ids.contains(&point.id.as_str()) {
            ids.push(&point.id);
        }
    }
    for id in ids {
        let pa = ma.points.iter().find(|p| p.id == id);
        let pb = mb.points.iter().find(|p| p.id == id);
        points.push(SweepPointComparison {
            id: id.to_string(),
            present_in_a: pa.is_some(),
            present_in_b: pb.is_some(),
            semantic_hash_equal: matches!((pa, pb), (Some(x), Some(y)) if x.bundle.semantic_hash == y.bundle.semantic_hash),
            payload_hash_equal: matches!((pa, pb), (Some(x), Some(y)) if x.bundle.payload_hash == y.bundle.payload_hash),
            metrics_equal: matches!((pa, pb), (Some(x), Some(y)) if x.inputs == y.inputs),
        });
    }
    let exact = points
        .iter()
        .all(|p| p.present_in_a && p.present_in_b && p.semantic_hash_equal);
    Ok(SweepComparison {
        sweep_a: a.display().to_string(),
        sweep_b: b.display().to_string(),
        sweep_hash_equal: ma.sweep_hash == mb.sweep_hash,
        spec_equal: ma.spec_sha256 == mb.spec_sha256,
        points,
        verdict: if exact { "exact" } else { "different" }.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: &str = r#"
schema = "ember.experiment.v1"

[experiment]
name = "sweep-test"

[model]
path = "m.gguf"

[[inputs]]
id = "a"
text = "one two three"

[[captures]]
id = "rows"
site = "residual-post-mlp"
layers = "all"
[captures.tokens]
kind = "prompt-final"

[[interventions]]
id = "replace"
site = "residual-post-mlp"
layers = [7]
operation = { kind = "replace" }
source = { kind = "capture-from-current-run", capture_id = "rows" }
[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "fixed"
site = "logits"
operation = { kind = "scale", factor = 1.0 }
[interventions.tokens]
kind = "prompt-final"

[output]
directory = "runs/sweep"

[sweep]
layers = { start = 0, end = 12, step = 4 }
interventions = ["replace"]
"#;

    #[test]
    fn points_move_only_the_swept_interventions() {
        let definition = SweepDefinition::parse(SPEC).unwrap();
        assert_eq!(definition.interventions, vec!["replace".to_string()]);
        let points = definition.points(12).unwrap();
        let ids: Vec<&str> = points.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec!["layer-00", "layer-04", "layer-08"]);
        let point = &points[1].resolved;
        assert_eq!(point.experiment.name, "sweep-test.layer-04");
        assert_eq!(
            point.output.directory,
            PathBuf::from("runs/sweep/points/layer-04")
        );
        let replace = point
            .interventions
            .iter()
            .find(|i| i.id == "replace")
            .unwrap();
        assert_eq!(replace.layers, LayerSelector::List(vec![4]));
        let fixed = point
            .interventions
            .iter()
            .find(|i| i.id == "fixed")
            .unwrap();
        assert!(!fixed.site.is_per_layer());
        assert!(points[1]
            .text
            .starts_with("# Derived from sweep 'sweep-test'"));
        // Derivation is deterministic.
        assert_eq!(definition.points(12).unwrap()[1].text, points[1].text);
        let baseline = definition.baseline().unwrap();
        assert!(baseline.resolved.interventions.is_empty());
        assert_eq!(baseline.resolved.captures.len(), 1);
    }

    #[test]
    fn positions_multiply_points_and_retarget_tokens() {
        let spec = SPEC.replace(
            "interventions = [\"replace\"]",
            "interventions = [\"replace\"]\npositions = [2, 0, 2]",
        );
        let definition = SweepDefinition::parse(&spec).unwrap();
        let points = definition.points(12).unwrap();
        assert_eq!(points.len(), 6);
        assert_eq!(points[1].id, "layer-00-pos-2");
        let replace = points[1]
            .resolved
            .interventions
            .iter()
            .find(|i| i.id == "replace")
            .unwrap();
        assert_eq!(
            replace.tokens,
            crate::v05::token_select::TokenSelector::AbsoluteToken { index: 2 }
        );
    }

    #[test]
    fn invalid_sweeps_fail_closed() {
        let cases = [
            (
                SPEC.replace(
                    "interventions = [\"replace\"]",
                    "interventions = [\"fixed\"]",
                ),
                "site",
            ),
            (
                SPEC.replace(
                    "interventions = [\"replace\"]",
                    "interventions = [\"nope\"]",
                ),
                "no intervention",
            ),
            (
                SPEC.replace("layers = { start = 0, end = 12, step = 4 }", "layers = []"),
                "empty",
            ),
            (
                SPEC.replace("[sweep]", "[sweep]\nbogus = 1"),
                "unknown field",
            ),
            (
                SPEC.replace(
                    "interventions = [\"replace\"]",
                    "interventions = [\"replace\"]\npositions = []",
                ),
                "position list",
            ),
        ];
        for (spec, needle) in cases {
            let error = SweepDefinition::parse(&spec).unwrap_err().to_string();
            assert!(error.contains(needle), "{needle:?} not in {error:?}");
        }
        // A sweep spec never resolves as a single experiment.
        let error = RawExperimentSpec::from_toml_str(SPEC)
            .unwrap()
            .resolve()
            .unwrap_err();
        assert_eq!(error.path, "sweep");
        // Source coverage is checked against the model's layer count.
        let narrow = SPEC.replace("layers = \"all\"", "layers = [0, 4]");
        let error = SweepDefinition::parse(&narrow)
            .unwrap()
            .points(12)
            .unwrap_err();
        assert!(
            error.message.contains("does not capture layer 8"),
            "{error}"
        );
    }

    #[test]
    fn stable_metrics_round_trip_through_json_exactly() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..20_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mantissa = (state >> 11) as f64 / (1u64 << 53) as f64;
            let exponent = ((state >> 3) % 44) as i32 - 30;
            let value = json_stable(mantissa * 10f64.powi(exponent));
            let text = serde_json::to_string(&value).unwrap();
            let back: f64 = serde_json::from_str(&text).unwrap();
            assert_eq!(back.to_bits(), value.to_bits(), "{text}");
            assert_eq!(json_stable(value).to_bits(), value.to_bits());
        }
    }

    #[test]
    fn manifest_hash_and_csv_are_deterministic() {
        let reference = SweepBundleRef {
            bundle: "baseline".into(),
            semantic_hash: "aa".into(),
            payload_hash: "bb".into(),
        };
        let mut manifest = SweepManifest {
            schema: SWEEP_SCHEMA_V1.into(),
            experiment: "x".into(),
            spec_file: SWEEP_SPEC_FILE.into(),
            spec_sha256: "cc".into(),
            model_sha256: "dd".into(),
            tokenizer_sha256: "ee".into(),
            layer_count: 2,
            layers: vec![1],
            positions: None,
            interventions: vec!["i".into()],
            baseline: reference.clone(),
            alphas: None,
            points: vec![SweepPointRecord {
                id: "layer-01".into(),
                layer: Some(1),
                position: None,
                alpha: None,
                bundle: SweepBundleRef {
                    bundle: "points/layer-01".into(),
                    ..reference
                },
                inputs: vec![SweepInputMetrics {
                    input_id: "a".into(),
                    generated_tokens_equal: false,
                    generated_text_equal: false,
                    first_divergent_step: Some(3),
                    peak_relative_l2: Some(0.125),
                    peak_capture_id: Some("rows".into()),
                    peak_site: Some("residual-post-mlp".into()),
                    peak_layer: Some(1),
                    captures_compared: 2,
                    captures_exact: 1,
                }],
            }],
            sweep_hash: String::new(),
        };
        manifest.sweep_hash = manifest.compute_hash();
        let json = serde_json::to_string(&manifest).unwrap();
        let back: SweepManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, manifest);
        assert_eq!(back.compute_hash(), manifest.sweep_hash);
        assert!(manifest
            .to_csv()
            .contains("layer-01,1,,a,3,false,0.125,rows,residual-post-mlp,1,1,2,aa\n"));
        assert!(inside(Path::new("/s"), "../x").is_err());
        assert!(inside(Path::new("/s"), "/abs").is_err());
        assert!(inside(Path::new("/s"), "points/layer-01").is_ok());
    }
}
