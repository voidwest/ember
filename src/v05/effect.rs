//! Effect statistics across the inputs of a sweep (`[sweep.effect]`).
//!
//! A sweep compares every point with the baseline, input by input. With an
//! effect table it also measures one scalar per input and summarizes the
//! change over the whole prompt set:
//!
//! ```toml
//! [sweep.effect]
//! capture = "answer"        # a capture at site `logits` (one row per input)
//! confidence = 0.95         # optional (default 0.95)
//! resamples = 10000         # optional bootstrap resamples (default 10000)
//! seed = 0                  # optional bootstrap seed (default 0)
//!
//! [[sweep.effect.targets]]  # one per input, every input
//! input = "france"
//! target = " Paris"         # token text (exactly one token) or a token id
//! foil = " Rome"
//! ```
//!
//! The metric is `m = logit(target) - logit(foil)` in the capture's row, and
//! the effect of a point on an input is `m(point) - m(baseline)`. Per point,
//! the summary gives the mean effect, the sample standard deviation and
//! standard error, a bootstrap percentile interval for the mean, the sign
//! counts, and a two-sided exact sign test.
//!
//! Every value is computed with IEEE additions, multiplications, divisions
//! and square roots only, in a fixed order, from the per-input effects as
//! `sweep.json` records them, so `verify` recomputes the summary bit for bit
//! on any machine. The bootstrap uses SplitMix64 and the same seed at every
//! point, so all points resample the same input indices.

use crate::v05::attribution::{MetricToken, TokenRef};
use crate::v05::capture::CaptureStorage;
use crate::v05::hook::SemanticHookSite;
use crate::v05::spec::{ExperimentSpecV1, SpecError};
use crate::v05::sweep::{json_stable, SweepInputMetrics};
use crate::v05::verify::LoadedBundle;
use serde::{Deserialize, Serialize};

/// File name of the per-point summary table inside a sweep directory.
pub const SWEEP_EFFECT_CSV_FILE: &str = "sweep-effect.csv";
/// The metric, as `sweep.json` records it.
pub const EFFECT_METRIC: &str =
    "logit(target) - logit(foil) in the capture row; effect = point - baseline";
/// The interval method, as `sweep.json` records it.
pub const EFFECT_INTERVAL: &str =
    "bootstrap percentile interval of the mean (SplitMix64, linear interpolation)";

const DEFAULT_CONFIDENCE: f64 = 0.95;
const DEFAULT_RESAMPLES: usize = 10_000;
const MIN_RESAMPLES: usize = 100;
const MAX_RESAMPLES: usize = 100_000;

/// The `[sweep.effect]` table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawEffectSpec {
    /// Id of a capture at site `logits`.
    pub capture: String,
    /// One target/foil pair per input.
    pub targets: Vec<RawEffectTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resamples: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// One `[[sweep.effect.targets]]` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawEffectTarget {
    pub input: String,
    pub target: TokenRef,
    pub foil: TokenRef,
}

/// A validated effect table. Targets are in input declaration order.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectDefinition {
    pub capture: String,
    pub targets: Vec<RawEffectTarget>,
    pub confidence: f64,
    pub resamples: usize,
    pub seed: u64,
}

/// One resolved target/foil pair (recorded in `sweep.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepEffectTarget {
    pub input_id: String,
    pub target: MetricToken,
    pub foil: MetricToken,
}

/// The `effect` object of `sweep.json`: what was measured and how.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepEffectRecord {
    pub metric: String,
    pub interval: String,
    pub capture: String,
    pub confidence: f64,
    pub resamples: usize,
    pub seed: u64,
    pub targets: Vec<SweepEffectTarget>,
}

/// The effect of one point over every input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepEffectSummary {
    /// Number of inputs.
    pub n: usize,
    pub mean: f64,
    /// Sample standard deviation (`n - 1`); `None` for one input.
    pub sd: Option<f64>,
    /// `sd / sqrt(n)`; `None` for one input.
    pub standard_error: Option<f64>,
    /// Bootstrap interval of the mean; `None` for one input.
    pub ci_low: Option<f64>,
    pub ci_high: Option<f64>,
    /// Inputs whose effect is above, below, or exactly zero.
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
    /// Two-sided exact sign test over the non-zero effects; `None` when
    /// every effect is zero.
    pub sign_test_p: Option<f64>,
}

impl EffectDefinition {
    /// Validate `[sweep.effect]` against the sweep's template experiment.
    pub fn parse(
        raw: RawEffectSpec,
        template: &ExperimentSpecV1,
    ) -> Result<EffectDefinition, SpecError> {
        let capture = template
            .captures
            .iter()
            .find(|capture| capture.id == raw.capture)
            .ok_or_else(|| {
                SpecError::at(
                    "sweep.effect.capture",
                    format!("no capture has id {:?}", raw.capture),
                )
            })?;
        if capture.site != SemanticHookSite::Logits {
            return Err(SpecError::at(
                "sweep.effect.capture",
                format!(
                    "capture {:?} is at site {}; the effect reads a logits row (site logits)",
                    capture.id, capture.site
                ),
            ));
        }
        if capture.storage == CaptureStorage::SummaryOnly {
            return Err(SpecError::at(
                "sweep.effect.capture",
                format!(
                    "capture {:?} stores a summary only; the effect needs the logits row",
                    capture.id
                ),
            ));
        }
        let input_ids: Vec<String> = template.inputs.iter().map(|i| i.id.clone()).collect();
        let captured = capture
            .inputs
            .resolve(&input_ids)
            .map_err(|error| SpecError::at("sweep.effect.capture", error))?;
        for (index, target) in raw.targets.iter().enumerate() {
            let path = format!("sweep.effect.targets[{index}]");
            if !input_ids.contains(&target.input) {
                return Err(SpecError::at(
                    format!("{path}.input"),
                    format!("no input has id {:?}", target.input),
                ));
            }
            if raw.targets[..index].iter().any(|t| t.input == target.input) {
                return Err(SpecError::at(
                    format!("{path}.input"),
                    format!("input {:?} has two targets", target.input),
                ));
            }
            if target.target == target.foil {
                return Err(SpecError::at(
                    format!("{path}.foil"),
                    "the target and foil tokens must differ",
                ));
            }
            for (field, token) in [("target", &target.target), ("foil", &target.foil)] {
                if matches!(token, TokenRef::Text(text) if text.is_empty()) {
                    return Err(SpecError::at(
                        format!("{path}.{field}"),
                        "token text is empty",
                    ));
                }
            }
            if !captured.contains(&target.input) {
                return Err(SpecError::at(
                    format!("{path}.input"),
                    format!(
                        "capture {:?} does not apply to input {:?}",
                        capture.id, target.input
                    ),
                ));
            }
        }
        // Every input counts: a prompt set with a silently missing input
        // would summarize a different set than the sweep ran.
        if let Some(missing) = input_ids
            .iter()
            .find(|id| !raw.targets.iter().any(|t| &t.input == *id))
        {
            return Err(SpecError::at(
                "sweep.effect.targets",
                format!("input {missing:?} has no target; give every input a target and foil"),
            ));
        }
        let confidence = raw.confidence.unwrap_or(DEFAULT_CONFIDENCE);
        check_confidence(confidence)
            .map_err(|error| SpecError::at("sweep.effect.confidence", error))?;
        let resamples = raw.resamples.unwrap_or(DEFAULT_RESAMPLES);
        check_resamples(resamples)
            .map_err(|error| SpecError::at("sweep.effect.resamples", error))?;
        let targets = input_ids
            .iter()
            .filter_map(|id| raw.targets.iter().find(|t| &t.input == id).cloned())
            .collect();
        Ok(EffectDefinition {
            capture: raw.capture,
            targets,
            confidence,
            resamples,
            seed: raw.seed.unwrap_or(0),
        })
    }

    /// Build the `sweep.json` record from targets resolved by the caller
    /// (`resolve` maps a token reference to its id and piece).
    pub fn record(
        &self,
        mut resolve: impl FnMut(&TokenRef) -> Result<MetricToken, String>,
    ) -> Result<SweepEffectRecord, String> {
        let mut targets = Vec::with_capacity(self.targets.len());
        for target in &self.targets {
            let resolved = SweepEffectTarget {
                input_id: target.input.clone(),
                target: resolve(&target.target)?,
                foil: resolve(&target.foil)?,
            };
            if resolved.target.token_id == resolved.foil.token_id {
                return Err(format!(
                    "effect: the target and foil of input {:?} are the same token {}",
                    target.input, resolved.target.token_id
                ));
            }
            targets.push(resolved);
        }
        Ok(SweepEffectRecord {
            metric: EFFECT_METRIC.to_string(),
            interval: EFFECT_INTERVAL.to_string(),
            capture: self.capture.clone(),
            confidence: self.confidence,
            resamples: self.resamples,
            seed: self.seed,
            targets,
        })
    }

    /// Whether `record` is what this definition describes. Token ids given
    /// in the spec are compared; token text is checked by `text_token`
    /// when it can re-encode (it returns `None` without a tokenizer).
    /// Returns the outcome and the number of token texts left unchecked.
    pub fn matches(
        &self,
        record: &SweepEffectRecord,
        mut text_token: impl FnMut(&str) -> Option<Result<u32, String>>,
    ) -> (bool, usize) {
        let mut unchecked = 0usize;
        let mut ok = record.metric == EFFECT_METRIC
            && record.interval == EFFECT_INTERVAL
            && record.capture == self.capture
            && record.confidence.to_bits() == self.confidence.to_bits()
            && record.resamples == self.resamples
            && record.seed == self.seed
            && record.targets.len() == self.targets.len();
        for (declared, recorded) in self.targets.iter().zip(&record.targets) {
            ok &= declared.input == recorded.input_id
                && recorded.target.token_id != recorded.foil.token_id;
            for (reference, token) in [
                (&declared.target, &recorded.target),
                (&declared.foil, &recorded.foil),
            ] {
                match reference {
                    TokenRef::Id(id) => ok &= *id == token.token_id,
                    TokenRef::Text(text) => match text_token(text) {
                        Some(Ok(id)) => ok &= id == token.token_id,
                        Some(Err(_)) => ok = false,
                        None => unchecked += 1,
                    },
                }
            }
        }
        (ok, unchecked)
    }
}

/// A confidence level that is inside (0, 1) and survives JSON exactly.
fn check_confidence(confidence: f64) -> Result<(), String> {
    let survives = serde_json::to_string(&confidence)
        .ok()
        .and_then(|text| serde_json::from_str::<f64>(&text).ok())
        .is_some_and(|back| back.to_bits() == confidence.to_bits());
    if confidence > 0.0 && confidence < 1.0 && survives {
        Ok(())
    } else {
        Err(format!(
            "confidence {confidence} must be a short decimal between 0 and 1"
        ))
    }
}

/// A bootstrap size inside the supported bounds.
fn check_resamples(resamples: usize) -> Result<(), String> {
    if (MIN_RESAMPLES..=MAX_RESAMPLES).contains(&resamples) {
        Ok(())
    } else {
        Err(format!(
            "resamples {resamples} must be between {MIN_RESAMPLES} and {MAX_RESAMPLES}"
        ))
    }
}

/// `logit(target) - logit(foil)` in the logits row `capture` recorded for
/// `input_id` in a verified bundle.
pub fn logit_difference_in_bundle(
    bundle: &LoadedBundle,
    capture: &str,
    input_id: &str,
    target: u32,
    foil: u32,
) -> Result<f64, String> {
    let mut entries = bundle
        .capture_index
        .iter()
        .filter(|entry| entry.capture_id == capture && entry.input_id == input_id);
    let entry = entries.next().ok_or_else(|| {
        format!(
            "bundle '{}' has no capture {capture:?} for input {input_id:?} (it did not fire)",
            bundle.root.display()
        )
    })?;
    if entries.next().is_some() {
        return Err(format!(
            "capture {capture:?} has more than one entry for input {input_id:?}"
        ));
    }
    if entry.site != SemanticHookSite::Logits || entry.tensor_name.is_empty() {
        return Err(format!(
            "capture {capture:?} for input {input_id:?} is not a stored logits row"
        ));
    }
    let [1, width] = entry.shape[..] else {
        return Err(format!(
            "capture {capture:?} for input {input_id:?} has shape {:?}; the effect needs \
             exactly one logits row",
            entry.shape
        ));
    };
    let row = bundle.tensor_f32_by_name(&entry.tensor_name)?;
    let at = |id: u32| {
        row.get(id as usize)
            .filter(|_| (id as usize) < width)
            .map(|value| f64::from(*value))
            .ok_or_else(|| format!("token {id} is outside the {width}-wide logits row"))
    };
    Ok(at(target)? - at(foil)?)
}

/// Fill the effect fields of `metrics` (one entry per input, from
/// `point_metrics`) and summarize them.
///
/// `record` can come from an untrusted `sweep.json`, so its interval
/// settings are checked again before they size any allocation.
pub fn apply_effect(
    record: &SweepEffectRecord,
    baseline: &LoadedBundle,
    point: &LoadedBundle,
    metrics: &mut [SweepInputMetrics],
) -> Result<SweepEffectSummary, String> {
    check_confidence(record.confidence).map_err(|error| format!("effect: {error}"))?;
    check_resamples(record.resamples).map_err(|error| format!("effect: {error}"))?;
    let mut effects = Vec::with_capacity(record.targets.len());
    for target in &record.targets {
        let entry = metrics
            .iter_mut()
            .find(|entry| entry.input_id == target.input_id)
            .ok_or_else(|| format!("the sweep has no input {:?}", target.input_id))?;
        let value = |bundle: &LoadedBundle| {
            logit_difference_in_bundle(
                bundle,
                &record.capture,
                &target.input_id,
                target.target.token_id,
                target.foil.token_id,
            )
        };
        let (before, after) = (value(baseline)?, value(point)?);
        // JSON cannot hold a non-finite value, and the summary needs finite
        // effects.
        if !before.is_finite() || !after.is_finite() {
            return Err(format!(
                "effect: input {:?} has a non-finite metric (baseline {before}, point {after})",
                target.input_id
            ));
        }
        let effect = json_stable(after - before);
        entry.baseline_metric = Some(json_stable(before));
        entry.point_metric = Some(json_stable(after));
        entry.effect = Some(effect);
        effects.push(effect);
    }
    if let Some(entry) = metrics.iter().find(|entry| entry.effect.is_none()) {
        return Err(format!("input {:?} has no effect target", entry.input_id));
    }
    Ok(summarize(
        &effects,
        record.confidence,
        record.resamples,
        record.seed,
    ))
}

/// SplitMix64 (Steele, Lea and Flood 2014): a fixed, portable generator,
/// so a bootstrap interval is the same on every machine.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// An index below `n` (multiply-shift; the bias is below `n / 2^64`).
    fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next()) * n as u128) >> 64) as usize
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Linear interpolation between order statistics (Hyndman-Fan type 7).
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let h = sorted.len().saturating_sub(1) as f64 * p.clamp(0.0, 1.0);
    let low = h.floor() as usize;
    let fraction = h - low as f64;
    match sorted.get(low + 1) {
        Some(next) => sorted[low] + fraction * (next - sorted[low]),
        None => sorted[low],
    }
}

/// `2^exponent` for a normal exponent, built exactly.
fn pow2(exponent: i64) -> f64 {
    debug_assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// Two-sided exact sign test: `min(1, 2 P(X <= min(positive, negative)))`
/// for `X ~ Binomial(positive + negative, 1/2)`.
///
/// The binomial terms are built by the recurrence
/// `C(m, i) = C(m, i - 1) (m - i + 1) / i` and rescaled by exact powers of
/// two, so the result uses only correctly rounded operations.
pub fn sign_test_p(positive: usize, negative: usize) -> Option<f64> {
    let m = (positive + negative) as u64;
    if m == 0 {
        return None;
    }
    let k = positive.min(negative) as u64;
    let (mut term, mut sum, mut scale) = (1.0f64, 1.0f64, 0i64);
    for i in 1..=k {
        term = term * (m - i + 1) as f64 / i as f64;
        sum += term;
        if sum >= pow2(512) {
            term *= pow2(-512);
            sum *= pow2(-512);
            scale += 512;
        }
    }
    // sum * 2^(scale - m), applied in exact steps.
    let mut exponent = scale - m as i64;
    let mut tail = sum;
    while exponent < -1000 {
        tail *= pow2(-1000);
        exponent += 1000;
    }
    tail *= pow2(exponent);
    Some(json_stable((2.0 * tail).min(1.0)))
}

/// Summarize per-input effects (see the module documentation).
pub fn summarize(
    effects: &[f64],
    confidence: f64,
    resamples: usize,
    seed: u64,
) -> SweepEffectSummary {
    let n = effects.len();
    let positive = effects.iter().filter(|&&e| e > 0.0).count();
    let negative = effects.iter().filter(|&&e| e < 0.0).count();
    let zero = n - positive - negative;
    let average = if n == 0 { 0.0 } else { mean(effects) };
    let (mut sd, mut standard_error, mut ci_low, mut ci_high) = (None, None, None, None);
    if n >= 2 && resamples > 0 {
        let squares: f64 = effects.iter().map(|e| (e - average) * (e - average)).sum();
        let deviation = (squares / (n - 1) as f64).sqrt();
        sd = Some(json_stable(deviation));
        standard_error = Some(json_stable(deviation / (n as f64).sqrt()));
        let mut rng = SplitMix64(seed);
        let mut means: Vec<f64> = (0..resamples)
            .map(|_| {
                let total: f64 = (0..n).map(|_| effects[rng.below(n)]).sum();
                total / n as f64
            })
            .collect();
        means.sort_by(f64::total_cmp);
        let tail = (1.0 - confidence) / 2.0;
        ci_low = Some(json_stable(quantile(&means, tail)));
        ci_high = Some(json_stable(quantile(&means, 1.0 - tail)));
    }
    SweepEffectSummary {
        n,
        mean: json_stable(average),
        sd,
        standard_error,
        ci_low,
        ci_high,
        positive,
        negative,
        zero,
        sign_test_p: sign_test_p(positive, negative),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_test_matches_exact_binomial_values() {
        // 6 of 6 positive: 2 * (1/2)^6.
        assert_eq!(sign_test_p(6, 0), Some(0.03125));
        // 5 vs 1: 2 * (1 + 6) / 64.
        assert_eq!(sign_test_p(5, 1), Some(0.21875));
        // Balanced counts cap at 1.
        assert_eq!(sign_test_p(3, 3), Some(1.0));
        assert_eq!(sign_test_p(0, 0), None);
        // 2000 vs 0 underflows to zero without a panic or a NaN.
        assert_eq!(sign_test_p(2000, 0), Some(0.0));
        // 1100 vs 900: the rescaled sum stays finite. The normal
        // approximation gives about 8.4e-6.
        let p = sign_test_p(1100, 900).unwrap();
        assert!(p > 5e-6 && p < 1.5e-5, "{p}");
    }

    #[test]
    fn summary_of_known_values() {
        let summary = summarize(&[1.0, 2.0, 3.0, 4.0], 0.95, 2000, 7);
        assert_eq!(summary.n, 4);
        assert_eq!(summary.mean, 2.5);
        // sd = sqrt(5/3), se = sd / 2.
        assert_eq!(summary.sd, Some(json_stable((5.0f64 / 3.0).sqrt())));
        assert_eq!(
            summary.standard_error,
            Some(json_stable((5.0f64 / 3.0).sqrt() / 2.0))
        );
        let (low, high) = (summary.ci_low.unwrap(), summary.ci_high.unwrap());
        assert!((1.0..2.5).contains(&low) && high > 2.5 && high <= 4.0);
        assert_eq!(
            (summary.positive, summary.negative, summary.zero),
            (4, 0, 0)
        );
        assert_eq!(summary.sign_test_p, Some(0.125));
    }

    #[test]
    fn summary_is_deterministic_and_seeded() {
        let effects = [0.5, -1.25, 3.0, 2.0, 0.0, 1.5];
        let a = summarize(&effects, 0.9, 5000, 1);
        assert_eq!(a, summarize(&effects, 0.9, 5000, 1));
        assert_eq!((a.positive, a.negative, a.zero), (4, 1, 1));
        // The zero effect is dropped from the sign test: 4 vs 1 of 5.
        assert_eq!(a.sign_test_p, Some(0.375));
    }

    #[test]
    fn splitmix64_matches_the_reference_sequence() {
        // The published SplitMix64 output for state 0.
        let mut rng = SplitMix64(0);
        assert_eq!(rng.next(), 0xE220_A839_7B1D_CDAF);
        let mut other = SplitMix64(1);
        assert_ne!(SplitMix64(0).next(), other.next());
        let mut bounded = SplitMix64(9);
        assert!((0..1000).all(|_| bounded.below(7) < 7));
    }

    #[test]
    fn single_input_has_no_spread_or_interval() {
        let summary = summarize(&[-2.0], 0.95, 100, 0);
        assert_eq!(summary.mean, -2.0);
        assert_eq!(
            (
                summary.sd,
                summary.standard_error,
                summary.ci_low,
                summary.ci_high
            ),
            (None, None, None, None)
        );
        assert_eq!(summary.sign_test_p, Some(1.0));
    }

    #[test]
    fn degenerate_interval_settings_do_not_panic() {
        let summary = summarize(&[1.0, 2.0], 0.95, 0, 0);
        assert_eq!((summary.ci_low, summary.ci_high), (None, None));
        let summary = summarize(&[1.0, 2.0], 1.5, 10, 0);
        assert!(summary.ci_high.is_some());
        assert!(check_resamples(0).is_err() && check_resamples(1 << 40).is_err());
        assert!(check_confidence(1.5).is_err() && check_confidence(f64::NAN).is_err());
        assert!(check_confidence(0.95).is_ok());
    }

    #[test]
    fn constant_effects_give_a_point_interval() {
        let summary = summarize(&[1.5; 5], 0.95, 1000, 3);
        assert_eq!(summary.sd, Some(0.0));
        assert_eq!((summary.ci_low, summary.ci_high), (Some(1.5), Some(1.5)));
    }

    const SPEC: &str = r#"
schema = "ember.experiment.v1"

[experiment]
name = "effect-test"

[model]
path = "m.gguf"

[[inputs]]
id = "a"
text = "one two three"

[[inputs]]
id = "b"
text = "four five"

[[captures]]
id = "answer"
site = "logits"
[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "rows"
site = "residual-post-mlp"
layers = "all"
[captures.tokens]
kind = "prompt-final"

[[interventions]]
id = "zero"
site = "mlp-output"
layers = [1]
operation = { kind = "zero" }
[interventions.tokens]
kind = "prompt-final"

[output]
directory = "runs/effect"

[sweep]
layers = "all"

[sweep.effect]
capture = "answer"

[[sweep.effect.targets]]
input = "b"
target = " x"
foil = 7

[[sweep.effect.targets]]
input = "a"
target = 5
foil = 6
"#;

    fn parse(text: &str) -> Result<EffectDefinition, String> {
        crate::v05::sweep::SweepDefinition::parse(text)
            .map(|definition| definition.effect.expect("an effect table"))
            .map_err(|error| error.to_string())
    }

    #[test]
    fn effect_table_resolves_defaults_and_input_order() {
        let effect = parse(SPEC).unwrap();
        assert_eq!(effect.capture, "answer");
        assert_eq!(
            (effect.confidence, effect.resamples, effect.seed),
            (0.95, 10_000, 0)
        );
        let inputs: Vec<&str> = effect.targets.iter().map(|t| t.input.as_str()).collect();
        assert_eq!(inputs, ["a", "b"]);
        // The derived experiments do not carry the effect table.
        let definition = crate::v05::sweep::SweepDefinition::parse(SPEC).unwrap();
        let baseline = definition.baseline().unwrap().text;
        assert!(
            !baseline.contains("[sweep") && !baseline.contains("targets"),
            "{baseline}"
        );
    }

    #[test]
    fn invalid_effect_tables_fail_closed() {
        let cases: [(&str, &str, &str); 9] = [
            ("capture = \"answer\"", "capture = \"rows\"", "site logits"),
            ("capture = \"answer\"", "capture = \"nope\"", "no capture"),
            ("input = \"b\"", "input = \"a\"", "two targets"),
            ("input = \"b\"", "input = \"z\"", "no input"),
            ("target = 5", "target = 6", "must differ"),
            ("target = \" x\"", "target = \"\"", "empty"),
            (
                "capture = \"answer\"",
                "capture = \"answer\"\nconfidence = 1.0",
                "between 0 and 1",
            ),
            (
                "capture = \"answer\"",
                "capture = \"answer\"\nresamples = 10",
                "resamples 10",
            ),
            (
                "capture = \"answer\"",
                "capture = \"answer\"\nmetric = \"other\"",
                "unknown field",
            ),
        ];
        for (from, to, expected) in cases {
            let text = SPEC.replacen(from, to, 1);
            assert_ne!(text, SPEC, "{from}");
            let error = parse(&text).unwrap_err();
            assert!(error.contains(expected), "{to}: {error}");
        }
        // Every input needs a target.
        let one = SPEC.replace(
            "\n[[sweep.effect.targets]]\ninput = \"a\"\ntarget = 5\nfoil = 6\n",
            "",
        );
        assert!(parse(&one)
            .unwrap_err()
            .contains("input \"a\" has no target"));
        // The capture must apply to every input and store the row.
        let narrow = SPEC.replacen(
            "id = \"answer\"\nsite = \"logits\"",
            "id = \"answer\"\nsite = \"logits\"\ninputs = [\"a\"]",
            1,
        );
        assert!(parse(&narrow)
            .unwrap_err()
            .contains("does not apply to input"));
        let summary = SPEC.replacen(
            "id = \"answer\"\nsite = \"logits\"",
            "id = \"answer\"\nsite = \"logits\"\nstorage = \"summary-only\"",
            1,
        );
        assert!(parse(&summary).unwrap_err().contains("summary only"));
    }

    #[test]
    fn matches_checks_ids_and_reports_unchecked_text() {
        let effect = parse(SPEC).unwrap();
        let token = |id| MetricToken {
            token_id: id,
            piece: String::new(),
        };
        let mut record = effect
            .record(|reference| match reference {
                TokenRef::Id(id) => Ok(token(*id)),
                TokenRef::Text(_) => Ok(token(9)),
            })
            .unwrap();
        assert_eq!(effect.matches(&record, |_| None), (true, 1));
        assert_eq!(effect.matches(&record, |_| Some(Ok(9))), (true, 0));
        assert_eq!(effect.matches(&record, |_| Some(Ok(8))), (false, 0));
        record.targets[0].foil.token_id = 4;
        assert!(!effect.matches(&record, |_| None).0);
        // A text that resolves to the foil's id is refused at record time.
        let error = effect
            .record(|reference| match reference {
                TokenRef::Id(id) => Ok(token(*id)),
                TokenRef::Text(_) => Ok(token(7)),
            })
            .unwrap_err();
        assert!(error.contains("same token 7"), "{error}");
    }

    #[test]
    fn quantile_interpolates_between_order_statistics() {
        let sorted = [0.0, 10.0, 20.0, 30.0, 40.0];
        assert_eq!(quantile(&sorted, 0.0), 0.0);
        assert_eq!(quantile(&sorted, 1.0), 40.0);
        assert_eq!(quantile(&sorted, 0.5), 20.0);
        assert_eq!(quantile(&sorted, 0.125), 5.0);
    }
}
