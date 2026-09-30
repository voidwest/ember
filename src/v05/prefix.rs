//! Shared-prefix reuse planning for experiments that run together.
//!
//! When a baseline (the *base* run) and an intervention run (a *variant*)
//! share model, prompt, generation settings and execution mode, their
//! prefills are identical up to the first block an intervention can change.
//! The driver computes that prefix once, during the base run, and starts the
//! variant's prefill at the boundary
//! ([`crate::experiments::forward_last_logits_resumed_with_experiment`]).
//!
//! This module decides, from the resolved specifications alone, where the
//! boundary lies or why reuse must not happen. The runtime adds its own
//! checks (prompt length, identical prompt tokens, cache capacity) and falls
//! back to a full recompute when any fails. Which path ran is recorded in
//! `runtime.json`, never in the semantic identity: a resumed bundle is
//! bit-identical to a fully recomputed one.
//!
//! The boundary for one input is the smallest layer `k` such that no
//! prefill-phase intervention of either run acts before block `k`:
//! a per-layer site at layer `L` gives `k = L` (the residual stream entering
//! block `L` precedes every site of that block), and a final-norm or logits
//! site gives `k = n_layers`. Interventions addressed to generated steps
//! act only in decode and never move the boundary. Captures of the variant
//! that fire before the boundary are recorded during the base run by an
//! observer instance of the variant's experiment and handed to the variant.

use crate::v05::hook::SemanticHookSite;
use crate::v05::spec::ExperimentSpecV1;
use serde::Serialize;
use std::collections::BTreeSet;

/// Why a variant input ran the way it did (recorded in `runtime.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "path")]
pub enum PrefixPath {
    /// The variant's prefill started at `resume_layer` from the base run's
    /// recorded state.
    Resumed { resume_layer: usize },
    /// The variant computed everything itself.
    FullRecompute { reason: String },
}

/// Per-input record of the path a run took.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PrefixInputRecord {
    pub input_id: String,
    #[serde(flatten)]
    pub path: PrefixPath,
}

/// The `prefix_reuse` object of `runtime.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PrefixReuseRecord {
    /// `base` (computed the shared prefix), `co-baseline` (observed the
    /// base run's entire generation without intervening), or `variant`.
    pub role: &'static str,
    /// For a variant: the path each input took. For a base: the variants
    /// it served. For a co-baseline: the base experiment it observed.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<PrefixInputRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl PrefixReuseRecord {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// The spec-level decision for one input of a variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReuseDecision {
    /// Reuse the base prefix up to (not including) block `first_layer`.
    Resume { first_layer: usize },
    /// Recompute the variant's input in full.
    FullRecompute { reason: String },
}

/// The earliest block a prefill-phase intervention of `spec` acts in for
/// `input_id`: `Some(layer)` for a per-layer site, `Some(n_layers)` for a
/// final-norm/logits site, `None` when no intervention acts during this
/// input's prefill.
pub fn first_prefill_intervention_layer(
    spec: &ExperimentSpecV1,
    input_id: &str,
    n_layers: usize,
) -> Result<Option<usize>, String> {
    let input_ids: Vec<String> = spec.inputs.iter().map(|input| input.id.clone()).collect();
    let mut first: Option<usize> = None;
    for intervention in &spec.interventions {
        if intervention.tokens.is_generated() {
            continue;
        }
        if !intervention
            .inputs
            .resolve(&input_ids)?
            .iter()
            .any(|id| id == input_id)
        {
            continue;
        }
        let layer = if intervention.site.is_per_layer() {
            match intervention.layers.resolve(n_layers)?.into_iter().min() {
                Some(layer) => layer,
                None => continue,
            }
        } else {
            n_layers
        };
        first = Some(first.map_or(layer, |current| current.min(layer)));
    }
    Ok(first)
}

/// The canonical plan stage keys of every generated-step site the spec
/// hooks (the set that decides the decode execution plan).
pub fn generated_step_stage_keys(
    spec: &ExperimentSpecV1,
    n_layers: usize,
) -> Result<BTreeSet<String>, String> {
    let mut keys = BTreeSet::new();
    let mut add = |site: SemanticHookSite,
                   layers: &crate::v05::capture::LayerSelector|
     -> Result<(), String> {
        if site.is_per_layer() {
            for layer in layers.resolve(n_layers)? {
                keys.insert(format!("{}@{layer}", site.stage_id()));
            }
        } else {
            keys.insert(site.stage_id().to_string());
        }
        Ok(())
    };
    for capture in &spec.captures {
        if capture.tokens.is_generated() {
            add(capture.site, &capture.layers)?;
        }
    }
    for intervention in &spec.interventions {
        if intervention.tokens.is_generated() {
            add(intervention.site, &intervention.layers)?;
        }
    }
    Ok(keys)
}

/// Settings that must agree for two runs to share any computation: model,
/// execution mode and threads, generation limits, seed, and the input.
fn shared_settings_mismatch(
    base: &ExperimentSpecV1,
    other: &ExperimentSpecV1,
    input_index: usize,
) -> Option<String> {
    if base.model.path != other.model.path {
        return Some("the runs use different model files".into());
    }
    if base.execution.mode != other.execution.mode {
        return Some("the runs use different execution modes".into());
    }
    if base.execution.threads != other.execution.threads {
        return Some("the runs use different thread counts".into());
    }
    if base.generation.max_new_tokens != other.generation.max_new_tokens
        || base.generation.temperature.to_bits() != other.generation.temperature.to_bits()
        || base.experiment.seed != other.experiment.seed
    {
        return Some("the runs use different generation settings".into());
    }
    match (base.inputs.get(input_index), other.inputs.get(input_index)) {
        (Some(a), Some(b)) if a == b => None,
        _ => Some("the base run has no identical input at this index".into()),
    }
}

/// Decide whether input `input_index` of `variant` can start from the
/// prefix `base` computes, and where.
pub fn resume_decision(
    base: &ExperimentSpecV1,
    variant: &ExperimentSpecV1,
    input_index: usize,
    n_layers: usize,
) -> ReuseDecision {
    if let Some(reason) = shared_settings_mismatch(base, variant, input_index) {
        return ReuseDecision::FullRecompute { reason };
    }
    let input_id = &variant.inputs[input_index].id;
    let boundary = |spec: &ExperimentSpecV1| {
        first_prefill_intervention_layer(spec, input_id, n_layers)
            .map(|layer| layer.unwrap_or(n_layers))
    };
    let (base_layer, variant_layer) = match (boundary(base), boundary(variant)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(error), _) | (_, Err(error)) => {
            return ReuseDecision::FullRecompute {
                reason: format!("intervention addressing did not resolve: {error}"),
            }
        }
    };
    let first_layer = base_layer.min(variant_layer).min(n_layers);
    if first_layer == 0 {
        return ReuseDecision::FullRecompute {
            reason: "an intervention acts in block 0, so no computation precedes it".into(),
        };
    }
    ReuseDecision::Resume { first_layer }
}

/// Whether `other` can observe the base run's whole generation instead of
/// running its own: neither run intervenes, the shared settings and every
/// input agree, and both hook the same generated-step sites (so the decode
/// plan the model executes is the one `other` would have executed alone).
pub fn co_baseline_mismatch(
    base: &ExperimentSpecV1,
    other: &ExperimentSpecV1,
    n_layers: usize,
) -> Option<String> {
    if !base.interventions.is_empty() || !other.interventions.is_empty() {
        return Some("an intervening run cannot share a whole generation".into());
    }
    if base.inputs != other.inputs {
        return Some("the runs have different inputs".into());
    }
    for index in 0..base.inputs.len() {
        if let Some(reason) = shared_settings_mismatch(base, other, index) {
            return Some(reason);
        }
    }
    match (
        generated_step_stage_keys(base, n_layers),
        generated_step_stage_keys(other, n_layers),
    ) {
        (Ok(a), Ok(b)) if a == b => None,
        (Ok(_), Ok(_)) => Some("the runs hook different generated-step sites".into()),
        (Err(error), _) | (_, Err(error)) => Some(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v05::spec::RawExperimentSpec;

    fn spec(interventions: &str) -> ExperimentSpecV1 {
        let text = format!(
            r#"
schema = "ember.experiment.v1"
[experiment]
name = "prefix"
seed = 1
[model]
path = "m.gguf"
[generation]
max_new_tokens = 2
temperature = 0.0
[[inputs]]
id = "a"
text = "one two"
[[inputs]]
id = "b"
text = "three four"
{interventions}
[output]
directory = "out"
"#
        );
        RawExperimentSpec::from_toml_str(&text)
            .unwrap()
            .resolve()
            .unwrap()
    }

    #[test]
    fn boundary_is_the_earliest_prefill_intervention_for_the_input() {
        let base = spec("");
        let variant = spec(
            r#"
[[interventions]]
id = "late"
site = "mlp-output"
layers = [5, 3]
operation = { kind = "zero" }
[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "decode-only"
site = "residual-pre-attention"
layers = [0]
operation = { kind = "zero" }
[interventions.tokens]
kind = "generated-step"
step = 1

[[interventions]]
id = "only-b"
site = "attention-output"
layers = [1]
inputs = ["b"]
operation = { kind = "zero" }
[interventions.tokens]
kind = "prompt-final"
"#,
        );
        assert_eq!(
            resume_decision(&base, &variant, 0, 8),
            ReuseDecision::Resume { first_layer: 3 }
        );
        assert_eq!(
            resume_decision(&base, &variant, 1, 8),
            ReuseDecision::Resume { first_layer: 1 }
        );
    }

    #[test]
    fn logits_sites_and_decode_only_runs_share_every_block() {
        let base = spec("");
        let logits = spec(
            r#"
[[interventions]]
id = "logits"
site = "logits"
operation = { kind = "scale", factor = 2.0 }
[interventions.tokens]
kind = "prompt-final"
"#,
        );
        assert_eq!(
            resume_decision(&base, &logits, 0, 4),
            ReuseDecision::Resume { first_layer: 4 }
        );
        assert_eq!(
            resume_decision(&base, &base, 1, 4),
            ReuseDecision::Resume { first_layer: 4 }
        );
    }

    #[test]
    fn layer_zero_and_mismatched_settings_recompute() {
        let base = spec("");
        let layer0 = spec(
            r#"
[[interventions]]
id = "l0"
site = "residual-post-mlp"
layers = [0]
operation = { kind = "zero" }
[interventions.tokens]
kind = "prompt-final"
"#,
        );
        assert!(matches!(
            resume_decision(&base, &layer0, 0, 4),
            ReuseDecision::FullRecompute { .. }
        ));
        let mut longer = base.clone();
        longer.generation.max_new_tokens = 3;
        assert!(matches!(
            resume_decision(&base, &longer, 0, 4),
            ReuseDecision::FullRecompute { .. }
        ));
        let mut other_prompt = base.clone();
        other_prompt.inputs[0].text = "different".into();
        assert!(matches!(
            resume_decision(&base, &other_prompt, 0, 4),
            ReuseDecision::FullRecompute { .. }
        ));
        assert_eq!(
            resume_decision(&base, &other_prompt, 1, 4),
            ReuseDecision::Resume { first_layer: 4 }
        );
        // A base that intervenes bounds the prefix as well.
        assert!(matches!(
            resume_decision(&layer0, &base, 0, 4),
            ReuseDecision::FullRecompute { .. }
        ));
    }

    #[test]
    fn co_baselines_require_no_interventions_and_equal_decode_sites() {
        let base = spec("");
        assert_eq!(co_baseline_mismatch(&base, &base, 4), None);
        let intervening = spec(
            r#"
[[interventions]]
id = "l1"
site = "residual-post-mlp"
layers = [1]
operation = { kind = "zero" }
[interventions.tokens]
kind = "prompt-final"
"#,
        );
        assert!(co_baseline_mismatch(&base, &intervening, 4).is_some());
    }
}
