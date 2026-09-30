//! Probe bridge (`[probe]` in an `ember.experiment.v1` spec): "the probe can
//! read it; can the model use it?"
//!
//! A linear probe direction at one (site, layer) is either read from a
//! SHA-256-pinned file or trained in the run as a closed-form ridge
//! regression on labelled prompts (targets +1/-1, centred, dual form
//! `w = Xc^T (Xc Xc^T + lambda I)^-1 yc`, solved by Cholesky in f64: no
//! seed, no iteration, bit-for-bit repeatable). Its accuracy is measured on
//! the training prompts and on held-out labelled prompts.
//!
//! The same direction then drives interventions on the spec's inputs (the
//! held-out behavioural prompts): `ablate-projection` and/or `steer` at each
//! `alpha` (unit-normalized), at the probe's site and layer. For every
//! variant and input the report records the target token's logit and
//! probability at the final prompt position against the unintervened
//! baseline, and whether (and where) the generated text changed. Probe
//! accuracy and causal effect sit side by side in one hashed artifact.

use crate::v05::hook::SemanticHookSite;
use crate::v05::intervention::SteerNormalization;
use crate::v05::manifest::sha256_hex;
use crate::v05::token_select::TokenSelector;
use serde::{Deserialize, Serialize};

pub const PROBE_SCHEMA_V1: &str = "ember.probe-bridge.v1";
pub const PROBE_JSON: &str = "artifacts/probe/probe.json";
pub const PROBE_CSV: &str = "artifacts/probe/effects.csv";
pub const PROBE_DIRECTION: &str = "artifacts/probe/direction.safetensors";

/// One labelled probe example.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelledPrompt {
    pub text: String,
    /// 0 or 1.
    pub label: u8,
}

/// A pinned probe direction file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeFile {
    pub path: std::path::PathBuf,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor: Option<String>,
}

fn default_tokens() -> TokenSelector {
    TokenSelector::PromptFinal
}
fn default_lambda() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}
fn default_normalize() -> SteerNormalization {
    SteerNormalization::Unit
}

/// The `[probe]` table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeSpec {
    /// Per-layer site and layer the probe reads and the interventions act at.
    pub site: SemanticHookSite,
    pub layer: usize,
    /// Rows averaged per labelled prompt (default `prompt-final`).
    #[serde(default = "default_tokens")]
    pub tokens: TokenSelector,
    /// Training examples (the probe is trained in the run when no `file`).
    #[serde(default)]
    pub train: Vec<LabelledPrompt>,
    /// Held-out labelled examples for probe accuracy.
    #[serde(default)]
    pub test: Vec<LabelledPrompt>,
    /// Ridge penalty (> 0).
    #[serde(default = "default_lambda")]
    pub ridge_lambda: f64,
    /// A pinned direction instead of training (then `train`, if given, only
    /// fits the decision threshold).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<ProbeFile>,
    /// Intervene by removing the projection on the direction.
    #[serde(default = "default_true")]
    pub ablate: bool,
    /// Intervene by steering along the direction at each alpha.
    #[serde(default)]
    pub steer_alphas: Vec<f32>,
    #[serde(default = "default_normalize")]
    pub steer_normalize: SteerNormalization,
    /// Rows of the behavioural inputs the interventions act on.
    #[serde(default = "default_tokens")]
    pub intervene_tokens: TokenSelector,
    /// Token whose logit and probability are measured (text of exactly one
    /// token, or an id).
    pub target: crate::v05::attribution::TokenRef,
}

impl ProbeSpec {
    /// Validation without a model.
    pub fn validate(&self, input_texts: &[&str]) -> Result<(), (String, String)> {
        let at = |field: &str, message: String| Err((format!("probe.{field}"), message));
        if !self.site.is_per_layer() {
            return at(
                "site",
                format!(
                    "site {} has no layer; probes read per-layer sites",
                    self.site
                ),
            );
        }
        if self.tokens.is_generated() || self.intervene_tokens.is_generated() {
            return at(
                "tokens",
                "probe rows and interventions are prompt rows; generated-step selectors are \
                 not supported"
                    .into(),
            );
        }
        for (field, examples) in [("train", &self.train), ("test", &self.test)] {
            for (index, example) in examples.iter().enumerate() {
                if example.label > 1 {
                    return at(
                        &format!("{field}[{index}].label"),
                        "labels are 0 or 1".into(),
                    );
                }
                if example.text.is_empty() {
                    return at(
                        &format!("{field}[{index}].text"),
                        "text must not be empty".into(),
                    );
                }
            }
        }
        match &self.file {
            None => {
                let positives = self.train.iter().filter(|e| e.label == 1).count();
                if positives == 0 || positives == self.train.len() {
                    return at(
                        "train",
                        "training needs examples of both labels (or give a probe `file`)".into(),
                    );
                }
            }
            Some(file) => {
                if file.path.as_os_str().is_empty()
                    || file.sha256.len() != 64
                    || !file
                        .sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return at(
                        "file",
                        "a probe file needs a path and a 64-hex lowercase sha256".into(),
                    );
                }
            }
        }
        if !(self.ridge_lambda.is_finite() && self.ridge_lambda > 0.0) {
            return at(
                "ridge_lambda",
                "the ridge penalty must be finite and > 0".into(),
            );
        }
        if !self.ablate && self.steer_alphas.is_empty() {
            return at(
                "ablate",
                "nothing to measure: enable `ablate` or give `steer_alphas`".into(),
            );
        }
        if self.steer_alphas.iter().any(|alpha| !alpha.is_finite()) {
            return at("steer_alphas", "alphas must be finite".into());
        }
        if matches!(&self.target, crate::v05::attribution::TokenRef::Text(t) if t.is_empty()) {
            return at("target", "token text must not be empty".into());
        }
        for example in self.train.iter().chain(&self.test) {
            if input_texts.contains(&example.text.as_str()) {
                return at(
                    "train",
                    format!(
                        "{:?} is also a behavioural input; the inputs must be held out from \
                         the probe's examples",
                        example.text
                    ),
                );
            }
        }
        Ok(())
    }

    /// The intervention variants, in report order.
    pub fn variants(&self) -> Vec<ProbeVariant> {
        let mut out = Vec::new();
        if self.ablate {
            out.push(ProbeVariant::Ablate);
        }
        for &alpha in &self.steer_alphas {
            out.push(ProbeVariant::Steer(alpha));
        }
        out
    }
}

/// One intervention along the probe direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProbeVariant {
    Ablate,
    Steer(f32),
}

impl ProbeVariant {
    pub fn id(self) -> String {
        match self {
            ProbeVariant::Ablate => "ablate".into(),
            ProbeVariant::Steer(alpha) => format!("steer{alpha:+}"),
        }
    }
}

/// Solve `A x = b` for symmetric positive definite `A` (`n x n`,
/// row-major) by Cholesky decomposition.
pub fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if a.len() != n * n || b.len() != n {
        return Err("cholesky: dimension mismatch".into());
    }
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err("cholesky: matrix is not positive definite".into());
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in i + 1..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    Ok(x)
}

/// A fitted linear probe: `score(x) = w . x + b`, label 1 when `> 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearProbe {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LinearProbe {
    pub fn score(&self, row: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(row)
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias
    }

    /// Fraction of examples classified correctly.
    pub fn accuracy(&self, rows: &[Vec<f64>], labels: &[u8]) -> Option<f64> {
        if rows.is_empty() {
            return None;
        }
        let correct = rows
            .iter()
            .zip(labels)
            .filter(|(row, label)| (self.score(row) > 0.0) == (**label == 1))
            .count();
        Some(correct as f64 / rows.len() as f64)
    }
}

/// Closed-form ridge regression on `+1/-1` targets with an unpenalized
/// intercept (centred data), in the dual so the cost is `n^2 d + n^3`.
pub fn fit_ridge(rows: &[Vec<f64>], labels: &[u8], lambda: f64) -> Result<LinearProbe, String> {
    let n = rows.len();
    let d = rows.first().map(Vec::len).ok_or("no training rows")?;
    if labels.len() != n || rows.iter().any(|row| row.len() != d) {
        return Err("training rows and labels disagree in shape".into());
    }
    let targets: Vec<f64> = labels
        .iter()
        .map(|&l| if l == 1 { 1.0 } else { -1.0 })
        .collect();
    let mut mean = vec![0.0f64; d];
    for row in rows {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let target_mean = targets.iter().sum::<f64>() / n as f64;
    let centred: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| row.iter().zip(&mean).map(|(v, m)| v - m).collect())
        .collect();
    let mut gram = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = centred[i].iter().zip(&centred[j]).map(|(a, b)| a * b).sum();
            gram[i * n + j] = dot;
            gram[j * n + i] = dot;
        }
        gram[i * n + i] += lambda;
    }
    let yc: Vec<f64> = targets.iter().map(|t| t - target_mean).collect();
    let alpha = cholesky_solve(&gram, &yc, n)?;
    let mut weights = vec![0.0f64; d];
    for (a, row) in alpha.iter().zip(&centred) {
        for (w, v) in weights.iter_mut().zip(row) {
            *w += a * v;
        }
    }
    let bias = target_mean - weights.iter().zip(&mean).map(|(w, m)| w * m).sum::<f64>();
    Ok(LinearProbe { weights, bias })
}

/// For a given direction, the threshold halfway between the class means of
/// the projections (used for pinned probe files with labelled examples).
pub fn midpoint_bias(weights: &[f64], rows: &[Vec<f64>], labels: &[u8]) -> Option<f64> {
    let project = |row: &Vec<f64>| weights.iter().zip(row).map(|(w, x)| w * x).sum::<f64>();
    let mean = |label: u8| {
        let values: Vec<f64> = rows
            .iter()
            .zip(labels)
            .filter(|(_, l)| **l == label)
            .map(|(row, _)| project(row))
            .collect();
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    Some(-(mean(0)? + mean(1)?) / 2.0)
}

/// Log-softmax probability of `token` in a logits row, in f64.
pub fn token_probability(logits: &[f32], token: usize) -> Option<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&v| (f64::from(v) - max).exp()).sum();
    logits.get(token).map(|&v| (f64::from(v) - max).exp() / sum)
}

/// The probe half of the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeRecord {
    pub site: SemanticHookSite,
    pub layer: usize,
    /// `trained-ridge` or `vector-file`.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ridge_lambda: Option<f64>,
    pub train_examples: usize,
    pub test_examples: usize,
    pub train_accuracy: Option<f64>,
    pub test_accuracy: Option<f64>,
    pub bias: Option<f64>,
    pub direction_norm: f64,
    /// SHA-256 of the direction's little-endian f32 bytes (the tensor in
    /// `direction.safetensors`).
    pub direction_checksum: String,
}

/// One (variant, input) measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeEffect {
    pub variant: String,
    pub input_id: String,
    pub target_logit: f64,
    pub target_probability: f64,
    pub delta_logit: f64,
    pub delta_probability: f64,
    pub generated_text: String,
    pub text_changed: bool,
    /// First 1-based generated step whose token differs from the baseline.
    pub first_divergent_step: Option<usize>,
}

/// Summary per variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariantSummary {
    pub variant: String,
    pub mean_delta_logit: f64,
    pub mean_delta_probability: f64,
    pub texts_changed: usize,
    pub inputs: usize,
}

/// `artifacts/probe/probe.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeReport {
    pub schema: String,
    pub probe: ProbeRecord,
    pub target: crate::v05::attribution::MetricToken,
    pub intervene_tokens: TokenSelector,
    /// `baseline` first, then each variant, each over every input.
    pub effects: Vec<ProbeEffect>,
    pub summary: Vec<VariantSummary>,
}

fn stable(value: f64) -> f64 {
    crate::v05::attribution::json_stable(value)
}

/// Build the per-variant summary from the effects (baseline excluded).
pub fn summarize_effects(effects: &[ProbeEffect]) -> Vec<VariantSummary> {
    let mut variants: Vec<String> = Vec::new();
    for effect in effects {
        if effect.variant != "baseline" && !variants.contains(&effect.variant) {
            variants.push(effect.variant.clone());
        }
    }
    variants
        .into_iter()
        .map(|variant| {
            let rows: Vec<&ProbeEffect> = effects.iter().filter(|e| e.variant == variant).collect();
            let n = rows.len().max(1) as f64;
            VariantSummary {
                mean_delta_logit: stable(rows.iter().map(|e| e.delta_logit).sum::<f64>() / n),
                mean_delta_probability: stable(
                    rows.iter().map(|e| e.delta_probability).sum::<f64>() / n,
                ),
                texts_changed: rows.iter().filter(|e| e.text_changed).count(),
                inputs: rows.len(),
                variant,
            }
        })
        .collect()
}

impl ProbeReport {
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "variant,input_id,target_logit,target_probability,delta_logit,delta_probability,\
             text_changed,first_divergent_step\n",
        );
        for e in &self.effects {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                e.variant,
                e.input_id,
                e.target_logit,
                e.target_probability,
                e.delta_logit,
                e.delta_probability,
                e.text_changed,
                e.first_divergent_step
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            ));
        }
        out
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, String> {
        let mut value = serde_json::to_value(self).map_err(|error| error.to_string())?;
        crate::plan::sort_value_keys(&mut value);
        let mut bytes = serde_json::to_vec_pretty(&value).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

/// Round an effect's floats so they survive JSON exactly.
pub fn stable_effect(mut effect: ProbeEffect) -> ProbeEffect {
    effect.target_logit = stable(effect.target_logit);
    effect.target_probability = stable(effect.target_probability);
    effect.delta_logit = stable(effect.delta_logit);
    effect.delta_probability = stable(effect.delta_probability);
    effect
}

/// Check the probe artifacts of a bundle against the spec and themselves.
pub fn verify_probe_artifacts(
    spec: &ProbeSpec,
    input_ids: &[String],
    file: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let Some(json) = file(PROBE_JSON) else {
        return vec![format!("{PROBE_JSON} is missing")];
    };
    let report: ProbeReport = match serde_json::from_slice(&json) {
        Ok(report) => report,
        Err(error) => return vec![format!("{PROBE_JSON} is malformed: {error}")],
    };
    if report.schema != PROBE_SCHEMA_V1 {
        errors.push(format!("unknown probe schema {:?}", report.schema));
    }
    if report.probe.site != spec.site || report.probe.layer != spec.layer {
        errors.push("the report's site/layer differ from the spec's".into());
    }
    let expected_source = if spec.file.is_some() {
        "vector-file"
    } else {
        "trained-ridge"
    };
    if report.probe.source != expected_source
        || report.probe.file_sha256.as_deref() != spec.file.as_ref().map(|f| f.sha256.as_str())
        || report.probe.train_examples != spec.train.len()
        || report.probe.test_examples != spec.test.len()
    {
        errors.push("the probe record does not describe the spec's probe".into());
    }
    let mut expected_variants = vec!["baseline".to_string()];
    expected_variants.extend(spec.variants().into_iter().map(ProbeVariant::id));
    let expected: Vec<(String, String)> = expected_variants
        .iter()
        .flat_map(|v| input_ids.iter().map(move |i| (v.clone(), i.clone())))
        .collect();
    let recorded: Vec<(String, String)> = report
        .effects
        .iter()
        .map(|e| (e.variant.clone(), e.input_id.clone()))
        .collect();
    if expected != recorded {
        errors.push("effects do not cover exactly the baseline and each variant per input".into());
    }
    if summarize_effects(&report.effects) != report.summary {
        errors.push("the summary differs from the recorded effects".into());
    }
    match file(PROBE_DIRECTION)
        .ok_or_else(|| format!("{PROBE_DIRECTION} is missing"))
        .and_then(|bytes| crate::v05::steering::parse_safetensors_direction(&bytes, None))
    {
        Ok(matrix) => {
            let bytes: Vec<u8> = matrix.values.iter().flat_map(|v| v.to_le_bytes()).collect();
            if sha256_hex(&bytes) != report.probe.direction_checksum {
                errors.push("the direction tensor differs from its recorded checksum".into());
            }
        }
        Err(error) => errors.push(error),
    }
    match file(PROBE_CSV) {
        Some(csv) if csv == report.to_csv().as_bytes() => {}
        Some(_) => errors.push(format!("{PROBE_CSV} is not the report's table")),
        None => errors.push(format!("{PROBE_CSV} is missing")),
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_a_known_system() {
        let a = [4.0, 2.0, 2.0, 3.0];
        let x = cholesky_solve(&a, &[2.0, 1.0], 2).unwrap();
        assert!((x[0] - 0.5).abs() < 1e-12 && x[1].abs() < 1e-12);
        assert!(cholesky_solve(&[0.0, 0.0, 0.0, 0.0], &[1.0, 1.0], 2).is_err());
    }

    #[test]
    fn ridge_separates_a_linear_concept_deterministically() {
        // Label = sign of coordinate 2, plus nuisance coordinates.
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for i in 0..20 {
            let label = (i % 2) as u8;
            let signal = if label == 1 { 1.0 } else { -1.0 };
            rows.push(vec![
                (i as f64 * 0.37).sin(),
                (i as f64 * 1.3).cos(),
                signal + 0.1 * (i as f64).sin(),
                0.5,
            ]);
            labels.push(label);
        }
        let probe = fit_ridge(&rows, &labels, 0.1).unwrap();
        assert_eq!(probe.accuracy(&rows, &labels), Some(1.0));
        let largest = probe
            .weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        assert_eq!(largest, 2);
        assert_eq!(fit_ridge(&rows, &labels, 0.1).unwrap(), probe);
        let bias = midpoint_bias(&probe.weights, &rows, &labels).unwrap();
        let shifted = LinearProbe {
            weights: probe.weights.clone(),
            bias,
        };
        assert_eq!(shifted.accuracy(&rows, &labels), Some(1.0));
    }

    #[test]
    fn probabilities_are_a_softmax() {
        let p = token_probability(&[0.0, 0.0], 1).unwrap();
        assert!((p - 0.5).abs() < 1e-12);
        assert!(token_probability(&[0.0], 3).is_none());
    }
}
