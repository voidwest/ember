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
//! `heads = "all"` (or a list or range) sweeps the attention heads of
//! `attention-heads` interventions: each point sets their `heads = [H]`,
//! alone or crossed with layers. Point ids gain a `head-<H>` part.
//!
//! `[sweep.effect]` adds a scalar metric for every input and its summary
//! over the inputs for each point (`crate::v05::effect`).
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
//! <sweep>/sweep-effect.csv    the effect summary (only with [sweep.effect])
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
use crate::v05::effect::{
    apply_effect, EffectDefinition, RawEffectSpec, SweepEffectRecord, SweepEffectSummary,
    SWEEP_EFFECT_CSV_FILE,
};
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
/// The most features one sweep may set, one point each: the width of the
/// largest dictionary that `MAX_WEIGHTS_BYTES` admits.
pub const MAX_SWEPT_FEATURES: usize = 65_536;

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
    /// Attention heads the swept `attention-heads` interventions move to,
    /// one per point: `"all"`, a list, or a range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heads: Option<LayerSelector>,
    /// Features of a dictionary (ADR-0003) the swept interventions act on,
    /// one per point: each point gives them `features = { dictionary, ids =
    /// [F] }`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub features: Option<RawFeatureSweep>,
    /// Intervention ids to sweep (default: every intervention).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interventions: Option<Vec<String>>,
    /// Move the layer of every swept `capture-from-bundle` source with the
    /// swept layer, so each point reads the source row at its own layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub move_bundle_sources: Option<bool>,
    /// Effect statistics across the inputs (`crate::v05::effect`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<RawEffectSpec>,
}

/// `[sweep] features`: a dictionary and the feature ids to sweep, as a list
/// or a `{ start, end, step }` range (end exclusive). There is no `"all"`: a
/// dictionary can hold tens of thousands of features, one bundle each.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawFeatureSweep {
    pub dictionary: String,
    pub ids: LayerSelector,
}

/// The features a sweep moves its interventions through (`sweep.json`
/// records it as it is).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureSweep {
    pub dictionary: String,
    /// Ascending, without repeats.
    pub ids: Vec<usize>,
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
    /// `None`: the swept interventions keep their declared heads.
    pub heads: Option<LayerSelector>,
    /// `None`: the swept interventions act on whole rows, as declared.
    pub features: Option<FeatureSweep>,
    /// Swept `alpha` values, ascending.
    pub alphas: Option<Vec<f64>>,
    /// Swept intervention ids, in declaration order.
    pub interventions: Vec<String>,
    /// Swept `capture-from-bundle` sources read the point's layer.
    pub move_bundle_sources: bool,
    /// Effect statistics across the inputs, when `[sweep.effect]` is given.
    pub effect: Option<EffectDefinition>,
    /// The spec without `[sweep]`, resolved (interventions at their
    /// declared layers). Model, execution and generation come from here.
    pub template: ExperimentSpecV1,
    document: toml::Table,
}

/// One derived experiment (a point or the baseline).
#[derive(Debug, Clone)]
pub struct DerivedSpec {
    /// `baseline` or `layer-07` / `layer-07-pos-3` / `layer-07-alpha-2` /
    /// `layer-07-head-03`.
    pub id: String,
    pub layer: Option<usize>,
    pub head: Option<usize>,
    pub feature: Option<usize>,
    pub position: Option<usize>,
    pub alpha: Option<f64>,
    /// The derived spec text (becomes the bundle's `experiment.toml`).
    pub text: String,
    pub resolved: ExperimentSpecV1,
    /// Bundle directory relative to the sweep directory.
    pub relative_dir: String,
}

/// Check a swept layer or head selector without model metadata.
fn check_selector(path: &str, selector: &LayerSelector, unit: &str) -> Result<(), SpecError> {
    match selector {
        LayerSelector::All(value) if value != "all" => Err(SpecError::at(
            path,
            format!("expected the string \"all\", found {value:?}"),
        )),
        LayerSelector::List(list) if list.is_empty() => {
            Err(SpecError::at(path, format!("the {unit} list is empty")))
        }
        LayerSelector::Range(range) if range.step == 0 || range.start >= range.end => {
            Err(SpecError::at(
                path,
                format!("a {unit} range needs start < end and step >= 1"),
            ))
        }
        _ => Ok(()),
    }
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
        if sweep.layers.is_none()
            && sweep.heads.is_none()
            && sweep.features.is_none()
            && sweep.alphas.is_none()
        {
            return Err(SpecError::at(
                "sweep",
                "a sweep needs `layers`, `heads`, `features`, `alphas`, or a combination",
            ));
        }
        let features = match &sweep.features {
            None => None,
            Some(raw) => Some(Self::parse_features(raw, &sweep, &template, &swept)?),
        };
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
        let move_bundle_sources = sweep.move_bundle_sources.unwrap_or(false);
        if move_bundle_sources {
            if sweep.layers.is_none() {
                return Err(SpecError::at(
                    "sweep.move_bundle_sources",
                    "bundle sources move with swept layers; add `layers`",
                ));
            }
            // A source moves to the point's layer but not to its position, so
            // every position would receive the source's own token row.
            if sweep.positions.is_some() {
                return Err(SpecError::at(
                    "sweep.move_bundle_sources",
                    "bundle sources move with layers only; a position sweep would patch \
                     every position with the source's own token row",
                ));
            }
            let moves_any = template.interventions.iter().any(|intervention| {
                swept.contains(&intervention.id)
                    && matches!(
                        intervention.source,
                        Some(InterventionSource::CaptureFromBundle { .. })
                    )
            });
            if !moves_any {
                return Err(SpecError::at(
                    "sweep.move_bundle_sources",
                    "no swept intervention reads a capture-from-bundle source",
                ));
            }
        }
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
                Some(InterventionSource::CaptureFromBundle { .. }) if !move_bundle_sources => {
                    return Err(SpecError::at(
                        format!("{path}.source"),
                        "a capture-from-bundle source names one fixed layer and cannot move \
                         across a layer sweep (set sweep.move_bundle_sources = true to read \
                         the source at each point's layer)",
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
        if let Some(heads) = &sweep.heads {
            check_selector("sweep.heads", heads, "head")?;
            for (index, intervention) in template.interventions.iter().enumerate() {
                if !swept.contains(&intervention.id) {
                    continue;
                }
                if intervention.site != crate::v05::hook::SemanticHookSite::AttentionHeads {
                    return Err(SpecError::at(
                        format!("interventions[{index}].site"),
                        format!(
                            "a head sweep moves attention-heads interventions; site {} has no \
                             heads (leave this intervention out of sweep.interventions)",
                            intervention.site
                        ),
                    ));
                }
                if intervention.heads.is_some() {
                    return Err(SpecError::at(
                        format!("interventions[{index}].heads"),
                        "a head sweep sets heads at each point; remove heads here",
                    ));
                }
                // Each point patches head H from head H of its source, so a
                // source capture must hold every head.
                if let Some(InterventionSource::CaptureFromCurrentRun { capture_id }) =
                    &intervention.source
                    && template
                        .captures
                        .iter()
                        .any(|capture| &capture.id == capture_id && capture.heads.is_some())
                {
                    return Err(SpecError::at(
                        format!("interventions[{index}].source"),
                        format!(
                            "a head sweep patches each head from the same head of its source; \
                             capture {capture_id:?} stores selected heads only (capture every \
                             head)"
                        ),
                    ));
                }
            }
        }
        if let Some(layers) = &sweep.layers {
            check_selector("sweep.layers", layers, "layer")?;
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
        let effect = sweep
            .effect
            .map(|raw| EffectDefinition::parse(raw, &template))
            .transpose()?;
        Ok(SweepDefinition {
            text: text.to_string(),
            spec_sha256: sha256_hex(text.as_bytes()),
            name: template.experiment.name.clone(),
            layers: sweep.layers,
            positions,
            heads: sweep.heads,
            features,
            alphas,
            interventions: swept,
            move_bundle_sources,
            effect,
            template,
            document,
        })
    }

    /// Check `[sweep] features` against the template: a dictionary is
    /// trained at one site and layer, so the swept interventions stay there
    /// and the sweep moves nothing else.
    fn parse_features(
        raw: &RawFeatureSweep,
        sweep: &RawSweepSpec,
        template: &ExperimentSpecV1,
        swept: &[String],
    ) -> Result<FeatureSweep, SpecError> {
        if sweep.layers.is_some() || sweep.heads.is_some() || sweep.positions.is_some() {
            return Err(SpecError::at(
                "sweep.features",
                "a feature sweep keeps the swept interventions at the dictionary's site and \
                 layer; it cannot also sweep layers, heads or positions",
            ));
        }
        let dictionary = template
            .dictionaries
            .iter()
            .find(|dictionary| dictionary.id == raw.dictionary)
            .ok_or_else(|| {
                SpecError::at(
                    "sweep.features.dictionary",
                    format!("no [[dictionaries]] entry has id {:?}", raw.dictionary),
                )
            })?;
        let ids = match &raw.ids {
            LayerSelector::All(_) => {
                return Err(SpecError::at(
                    "sweep.features.ids",
                    "list the feature ids, or give a { start, end, step } range: a sweep of \
                     every feature would write one bundle per feature",
                ))
            }
            LayerSelector::List(list) => {
                if list.is_empty() {
                    return Err(SpecError::at(
                        "sweep.features.ids",
                        "the feature list is empty",
                    ));
                }
                let mut list = list.clone();
                list.sort_unstable();
                list.dedup();
                list
            }
            LayerSelector::Range(range) => {
                if range.step == 0 || range.start >= range.end {
                    return Err(SpecError::at(
                        "sweep.features.ids",
                        "a feature range needs start < end and step >= 1",
                    ));
                }
                // Count the range before allocating it: an end near
                // usize::MAX must fail here, not exhaust memory.
                let count = (range.end - range.start - 1) / range.step + 1;
                if count > MAX_SWEPT_FEATURES {
                    return Err(SpecError::at(
                        "sweep.features.ids",
                        format!(
                            "the range selects {count} features; a sweep sets at most \
                             {MAX_SWEPT_FEATURES}"
                        ),
                    ));
                }
                (range.start..range.end).step_by(range.step).collect()
            }
        };
        if ids.len() > MAX_SWEPT_FEATURES {
            return Err(SpecError::at(
                "sweep.features.ids",
                format!(
                    "the list names {} features; a sweep sets at most {MAX_SWEPT_FEATURES}",
                    ids.len()
                ),
            ));
        }
        for (index, intervention) in template.interventions.iter().enumerate() {
            if !swept.contains(&intervention.id) {
                continue;
            }
            let path = format!("interventions[{index}]");
            if intervention.features.is_some() {
                return Err(SpecError::at(
                    format!("{path}.features"),
                    "a feature sweep sets features at each point; remove features here",
                ));
            }
            use crate::v05::intervention::InterventionOperation as Op;
            if matches!(
                intervention.operation,
                Op::Steer { .. } | Op::AblateProjection | Op::RestoreOriginal
            ) {
                return Err(SpecError::at(
                    "sweep.features",
                    format!(
                        "intervention '{}': {} does not act on features; use zero, scale, \
                         replace, add-delta or interpolate",
                        intervention.id,
                        intervention.operation.kind_name()
                    ),
                ));
            }
            let at_dictionary = intervention.site == dictionary.site
                && match &intervention.layers {
                    LayerSelector::List(layers) => layers.as_slice() == [dictionary.layer],
                    LayerSelector::Range(range) => {
                        range.start == dictionary.layer && range.end == dictionary.layer + 1
                    }
                    LayerSelector::All(_) => false,
                };
            if !at_dictionary {
                return Err(SpecError::at(
                    path,
                    format!(
                        "a feature sweep acts at dictionary {:?}'s site and layer: set site = \
                         \"{}\" and layers = [{}]",
                        dictionary.id, dictionary.site, dictionary.layer
                    ),
                ));
            }
        }
        Ok(FeatureSweep {
            dictionary: raw.dictionary.clone(),
            ids,
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

    /// With `move_bundle_sources`: check, before anything runs, that every
    /// moved source bundle verifies and holds its capture at every swept
    /// layer. Each point loads its source again when it runs.
    pub fn check_moved_bundle_sources(&self, n_layers: usize) -> Result<(), String> {
        if !self.move_bundle_sources {
            return Ok(());
        }
        let layers = self
            .resolve_layers(n_layers)
            .map_err(|error| error.to_string())?;
        let mut loaded: Vec<(PathBuf, LoadedBundle)> = Vec::new();
        for intervention in &self.template.interventions {
            let Some(InterventionSource::CaptureFromBundle {
                bundle_path,
                capture_id,
                input_id,
                ..
            }) = &intervention.source
            else {
                continue;
            };
            if !self.interventions.contains(&intervention.id) {
                continue;
            }
            let index = match loaded.iter().position(|(path, _)| path == bundle_path) {
                Some(index) => index,
                None => {
                    let bundle = crate::v05::verify::load_bundle_for_source(bundle_path)
                        .map_err(|error| format!("intervention {:?}: {error}", intervention.id))?;
                    loaded.push((bundle_path.clone(), bundle));
                    loaded.len() - 1
                }
            };
            let bundle = &loaded[index].1;
            if let Some(missing) = layers.iter().find(|&&layer| {
                !bundle.capture_index.iter().any(|entry| {
                    entry.capture_id == *capture_id
                        && entry.input_id == *input_id
                        && entry.site == intervention.site
                        && entry.layer == layer
                })
            }) {
                return Err(format!(
                    "intervention {:?}: source bundle '{}' has no capture {capture_id:?} of \
                     input {input_id:?} at {} layer {missing}; a sweep with \
                     move_bundle_sources reads the source at every swept layer",
                    intervention.id,
                    bundle_path.display(),
                    intervention.site
                ));
            }
        }
        Ok(())
    }

    /// How many points `points` derives, without deriving them (`None` when
    /// the count overflows).
    pub fn point_count(&self, n_layers: usize, n_heads: usize) -> Result<Option<usize>, SpecError> {
        let layers = match self.layers {
            Some(_) => self.resolve_layers(n_layers)?.len(),
            None => 1,
        };
        let heads = match self.heads {
            Some(_) => self.resolve_heads(n_heads)?.len(),
            None => 1,
        };
        let features = self
            .features
            .as_ref()
            .map_or(1, |features| features.ids.len());
        let positions = self.positions.as_ref().map_or(1, Vec::len);
        let alphas = self.alphas.as_ref().map_or(1, Vec::len);
        Ok(layers
            .checked_mul(heads)
            .and_then(|count| count.checked_mul(features))
            .and_then(|count| count.checked_mul(positions))
            .and_then(|count| count.checked_mul(alphas)))
    }

    /// For head sweeps: check, before anything runs, that every swept
    /// `capture-from-bundle` source holds every head (`n_heads * head_dim`
    /// columns), so each point patches head H from head H of the source.
    pub fn check_head_bundle_sources(&self, n_heads: usize, head_dim: usize) -> Result<(), String> {
        if self.heads.is_none() {
            return Ok(());
        }
        let width = n_heads * head_dim;
        for intervention in &self.template.interventions {
            let Some(InterventionSource::CaptureFromBundle {
                bundle_path,
                capture_id,
                input_id,
                ..
            }) = &intervention.source
            else {
                continue;
            };
            if !self.interventions.contains(&intervention.id) {
                continue;
            }
            let bundle = crate::v05::verify::load_bundle_for_source(bundle_path)
                .map_err(|error| format!("intervention {:?}: {error}", intervention.id))?;
            if let Some(entry) = bundle.capture_index.iter().find(|entry| {
                entry.capture_id == *capture_id
                    && entry.input_id == *input_id
                    && entry.site == intervention.site
                    && entry.shape.last() != Some(&width)
            }) {
                return Err(format!(
                    "intervention {:?}: source capture {capture_id:?} of input {input_id:?} has \
                     shape {:?}; a head sweep needs every head ({width} columns) so each point \
                     patches head H from head H",
                    intervention.id, entry.shape
                ));
            }
        }
        Ok(())
    }

    /// The heads of this sweep for an `n_heads`-head model (empty when the
    /// sweep does not move heads).
    pub fn resolve_heads(&self, n_heads: usize) -> Result<Vec<usize>, SpecError> {
        match &self.heads {
            None => Ok(Vec::new()),
            Some(heads) => heads
                .resolve(n_heads)
                .map_err(|error| SpecError::at("sweep.heads", error.replace("layer", "head"))),
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
            head: None,
            feature: None,
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

    /// Every point for an `n_layers`-layer, `n_heads`-head model: layers
    /// ascending, then heads, then features, then positions, then alphas.
    pub fn points(&self, n_layers: usize, n_heads: usize) -> Result<Vec<DerivedSpec>, SpecError> {
        let resolved_layers = self.resolve_layers(n_layers)?;
        let layers: Vec<Option<usize>> = if self.layers.is_some() {
            resolved_layers.iter().copied().map(Some).collect()
        } else {
            vec![None]
        };
        let heads: Vec<Option<usize>> = if self.heads.is_some() {
            self.resolve_heads(n_heads)?.into_iter().map(Some).collect()
        } else {
            vec![None]
        };
        let head_width = n_heads.saturating_sub(1).to_string().len().max(2);
        let features: Vec<Option<usize>> = match &self.features {
            None => vec![None],
            Some(features) => features.ids.iter().copied().map(Some).collect(),
        };
        let feature_width = self
            .features
            .as_ref()
            .and_then(|features| features.ids.last())
            .map_or(2, |&last| last.to_string().len().max(2));
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
        let mut out = Vec::with_capacity(
            layers.len() * heads.len() * features.len() * positions.len() * alphas.len(),
        );
        let grid = layers.iter().flat_map(|&layer| {
            heads.iter().flat_map({
                let features = &features;
                move |&head| features.iter().map(move |&feature| (layer, head, feature))
            })
        });
        for (layer, head, feature) in grid {
            for &position in &positions {
                for &alpha in &alphas {
                    let mut parts = Vec::new();
                    if let Some(layer) = layer {
                        parts.push(format!("layer-{layer:0width$}"));
                    }
                    if let Some(head) = head {
                        parts.push(format!("head-{head:0head_width$}"));
                    }
                    if let Some(feature) = feature {
                        parts.push(format!("feature-{feature:0feature_width$}"));
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
                                if self.move_bundle_sources
                                    && let Some(toml::Value::Table(source)) =
                                        table.get_mut("source")
                                    && source.get("kind").and_then(toml::Value::as_str)
                                        == Some("capture-from-bundle")
                                {
                                    source
                                        .insert("layer".into(), toml::Value::Integer(layer as i64));
                                }
                            }
                            if let Some(head) = head {
                                table.insert(
                                    "heads".into(),
                                    toml::Value::Array(vec![toml::Value::Integer(head as i64)]),
                                );
                            }
                            if let (Some(feature), Some(features)) = (feature, &self.features) {
                                let mut selection = toml::Table::new();
                                selection.insert(
                                    "dictionary".into(),
                                    toml::Value::String(features.dictionary.clone()),
                                );
                                selection.insert(
                                    "ids".into(),
                                    toml::Value::Array(vec![toml::Value::Integer(feature as i64)]),
                                );
                                table.insert("features".into(), toml::Value::Table(selection));
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
                    if let Some(head) = head {
                        coordinates.push(format!("head {head}"));
                    }
                    if let Some(feature) = feature {
                        coordinates.push(format!("feature {feature}"));
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
                    derived.head = head;
                    derived.feature = feature;
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
    /// With `[sweep.effect]`: the metric on the baseline and on the point,
    /// and their difference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_metric: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_metric: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<f64>,
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
    /// The point's head, for head sweeps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<usize>,
    /// The point's feature, for feature sweeps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature: Option<usize>,
    #[serde(flatten)]
    pub bundle: SweepBundleRef,
    pub inputs: Vec<SweepInputMetrics>,
    /// With `[sweep.effect]`: the effect over every input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<SweepEffectSummary>,
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
    /// For head sweeps: the swept heads and the model's head count (which
    /// `heads = "all"` resolves against).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heads: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_count: Option<usize>,
    /// For feature sweeps: the dictionary and the swept feature ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub features: Option<FeatureSweep>,
    pub interventions: Vec<String>,
    /// With `[sweep.effect]`: the metric, interval settings and resolved
    /// target/foil tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<SweepEffectRecord>,
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
        // The head column likewise exists only for head sweeps, and the
        // feature column only for feature sweeps.
        let with_head = self.heads.is_some();
        let with_feature = self.features.is_some();
        let mut out = String::from(if with_alpha {
            "point,layer,position,alpha,"
        } else {
            "point,layer,position,"
        });
        if with_head {
            out.push_str("head,");
        }
        if with_feature {
            out.push_str("feature,");
        }
        out.push_str(
            "input_id,first_divergent_step,generated_text_equal,\
             peak_relative_l2,peak_capture_id,peak_site,peak_layer,captures_exact,\
             captures_compared,semantic_hash",
        );
        // Effect columns likewise exist only with `[sweep.effect]`.
        let with_effect = self.effect.is_some();
        out.push_str(if with_effect {
            ",baseline_metric,point_metric,effect\n"
        } else {
            "\n"
        });
        for point in &self.points {
            for input in &point.inputs {
                let mut alpha = if with_alpha {
                    format!("{},", opt(&point.alpha))
                } else {
                    String::new()
                };
                if with_head {
                    alpha.push_str(&format!("{},", opt(&point.head)));
                }
                if with_feature {
                    alpha.push_str(&format!("{},", opt(&point.feature)));
                }
                let effect = if with_effect {
                    format!(
                        ",{},{},{}",
                        opt(&input.baseline_metric),
                        opt(&input.point_metric),
                        opt(&input.effect)
                    )
                } else {
                    String::new()
                };
                out.push_str(&format!(
                    "{},{},{},{alpha}{},{},{},{},{},{},{},{},{},{}{effect}\n",
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

    /// `sweep-effect.csv`: one row per point (only with `[sweep.effect]`).
    pub fn to_effect_csv(&self) -> Option<String> {
        fn opt<T: std::fmt::Display>(value: &Option<T>) -> String {
            value.as_ref().map(ToString::to_string).unwrap_or_default()
        }
        self.effect.as_ref()?;
        // A head column only for head sweeps and a feature column only for
        // feature sweeps, so earlier tables stay as they were.
        let with_head = self.heads.is_some();
        let with_feature = self.features.is_some();
        let mut out = String::from(if with_head {
            "point,layer,position,alpha,head,"
        } else {
            "point,layer,position,alpha,"
        });
        if with_feature {
            out.push_str("feature,");
        }
        out.push_str(
            "n,mean,sd,standard_error,ci_low,ci_high,positive,negative,zero,sign_test_p\n",
        );
        for point in &self.points {
            let Some(effect) = &point.effect else {
                continue;
            };
            let mut head = if with_head {
                format!("{},", opt(&point.head))
            } else {
                String::new()
            };
            if with_feature {
                head.push_str(&format!("{},", opt(&point.feature)));
            }
            out.push_str(&format!(
                "{},{},{},{},{head}{},{},{},{},{},{},{},{},{},{}\n",
                point.id,
                opt(&point.layer),
                opt(&point.position),
                opt(&point.alpha),
                effect.n,
                effect.mean,
                opt(&effect.sd),
                opt(&effect.standard_error),
                opt(&effect.ci_low),
                opt(&effect.ci_high),
                effect.positive,
                effect.negative,
                effect.zero,
                opt(&effect.sign_test_p),
            ));
        }
        Some(out)
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
                baseline_metric: None,
                point_metric: None,
                effect: None,
            }
        })
        .collect())
}

/// [`point_metrics`] plus, with an effect record, the per-input effect and
/// its summary over the inputs.
pub fn point_metrics_with_effect(
    baseline: &LoadedBundle,
    point: &LoadedBundle,
    effect: Option<&SweepEffectRecord>,
) -> Result<(Vec<SweepInputMetrics>, Option<SweepEffectSummary>), String> {
    let mut metrics = point_metrics(baseline, point)?;
    let summary = effect
        .map(|record| apply_effect(record, baseline, point, &mut metrics))
        .transpose()?;
    Ok((metrics, summary))
}

/// The attention head count in a verified bundle's model metadata
/// (`<architecture>.attention.head_count`).
fn model_head_count(bundle: &LoadedBundle) -> Option<usize> {
    let model: serde_json::Value = serde_json::from_slice(bundle.file("model.json")?).ok()?;
    let key = format!("{}.attention.head_count", model["architecture"].as_str()?);
    usize::try_from(model["gguf_metadata"][key].as_u64()?).ok()
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

mod verify;
pub use verify::{
    compare_anchored_sweeps, compare_sweeps, is_sweep_dir, read_manifest, verify_sweep,
    SweepComparison, SweepPointComparison, SweepVerification,
};

#[cfg(test)]
mod tests;
