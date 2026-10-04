//! Attribution patching without autograd (`[attribution]` in an
//! `ember.experiment.v1` spec).
//!
//! Given a *clean* and a *corrupted* prompt of equal token length and a
//! metric `m = logit(target) - logit(foil)` at the final prompt position,
//! every candidate site `(site, layer, position)` is scored by the first-order
//! effect of patching its clean activation into the corrupted run, restricted
//! to the *direct path*:
//!
//! ```text
//! estimate = (a_clean - a_corrupted) . r       at the final position
//! estimate = 0                                  at every other position
//! r = d m / d x_final = J_norm(x)^T (g * (W_U[target] - W_U[foil]))
//! ```
//!
//! `x` is the corrupted run's residual stream entering the final RMS norm at
//! the final position and `J_norm` is the exact Jacobian of
//! `y = g * x / sqrt(mean(x^2) + eps)` at `x`:
//! `r = (g*u)/s - x * sum(g*u*x) / (d * s^3)` with `s = sqrt(mean(x^2)+eps)`.
//! A patched difference at `attention-output`/`mlp-output` enters the
//! residual stream by addition, and at `residual-pre-attention` /
//! `residual-post-mlp` it replaces the stream; either way its direct effect on
//! the final residual is the difference itself, so `estimate` is exactly
//! attribution patching (gradient times activation difference) with every
//! path through later blocks removed. It needs one clean and one corrupted
//! forward pass for all candidates.
//!
//! Limits, by construction: indirect effects (anything a later attention or
//! MLP block computes from the patched value) are ignored, so the estimate is
//! exact only for the last block's projection outputs up to the final norm's
//! curvature; and positions other than the final one reach the final logits
//! only through attention, so their estimate is zero. The workflow therefore
//! verifies its top candidates with real activation patches (`replace` with
//! the clean row) and reports the rank correlation between the estimate and
//! the measured effect.

use crate::v05::capture::LayerSelector;
use crate::v05::hook::SemanticHookSite;
use serde::{Deserialize, Serialize};

/// Attribution report schema.
pub const ATTRIBUTION_SCHEMA_V1: &str = "ember.attribution.v1";
/// Bundle paths of the attribution artifacts.
pub const ATTRIBUTION_JSON: &str = "artifacts/attribution/attribution.json";
pub const ATTRIBUTION_CSV: &str = "artifacts/attribution/candidates.csv";

/// A token named by id or by text that must encode to exactly one token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TokenRef {
    Id(u32),
    Text(String),
}

/// Candidate positions: `"all"`, `"final"`, or a list of absolute
/// positions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PositionSelector {
    Named(String),
    List(Vec<usize>),
}

impl PositionSelector {
    /// Resolve against a sequence of `len` tokens.
    pub fn resolve(&self, len: usize) -> Result<Vec<usize>, String> {
        match self {
            PositionSelector::Named(name) if name == "all" => Ok((0..len).collect()),
            PositionSelector::Named(name) if name == "final" => Ok(vec![len.saturating_sub(1)]),
            PositionSelector::Named(other) => Err(format!(
                "positions: expected \"all\", \"final\" or a list, found {other:?}"
            )),
            PositionSelector::List(list) => {
                let mut list = list.clone();
                list.sort_unstable();
                list.dedup();
                if let Some(bad) = list.iter().find(|&&p| p >= len) {
                    return Err(format!("position {bad} is outside the {len}-token prompt"));
                }
                Ok(list)
            }
        }
    }
}

fn default_sites() -> Vec<SemanticHookSite> {
    vec![
        SemanticHookSite::ResidualPreAttention,
        SemanticHookSite::AttentionOutput,
        SemanticHookSite::MlpOutput,
    ]
}

fn default_layers() -> LayerSelector {
    LayerSelector::All("all".into())
}

fn default_positions() -> PositionSelector {
    PositionSelector::Named("all".into())
}

fn default_top_k() -> usize {
    10
}

/// The `[attribution]` table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttributionSpec {
    /// Input id of the clean prompt.
    pub clean: String,
    /// Input id of the corrupted prompt (same token count).
    pub corrupted: String,
    /// Metric: `logit(target) - logit(foil)` at the final prompt position.
    pub target: TokenRef,
    pub foil: TokenRef,
    /// Per-layer sites to rank (default: residual-pre-attention,
    /// attention-output, mlp-output).
    #[serde(default = "default_sites")]
    pub sites: Vec<SemanticHookSite>,
    #[serde(default = "default_layers")]
    pub layers: LayerSelector,
    #[serde(default = "default_positions")]
    pub positions: PositionSelector,
    /// How many top-ranked candidates to verify with real patches.
    #[serde(default = "default_top_k")]
    pub verify_top_k: usize,
}

impl AttributionSpec {
    /// Validation that needs no model: the inputs exist and differ, sites
    /// are per-layer and distinct, selectors are well formed.
    pub fn validate(&self, input_ids: &[&str]) -> Result<(), (String, String)> {
        let at = |field: &str, message: String| Err((format!("attribution.{field}"), message));
        for (field, id) in [("clean", &self.clean), ("corrupted", &self.corrupted)] {
            if !input_ids.contains(&id.as_str()) {
                return at(field, format!("no input has id {id:?}"));
            }
        }
        if self.clean == self.corrupted {
            return at(
                "corrupted",
                "the clean and corrupted inputs must differ".into(),
            );
        }
        if self.target == self.foil {
            return at("foil", "the target and foil tokens must differ".into());
        }
        for (field, token) in [("target", &self.target), ("foil", &self.foil)] {
            if matches!(token, TokenRef::Text(text) if text.is_empty()) {
                return at(field, "token text must not be empty".into());
            }
        }
        if self.sites.is_empty() {
            return at("sites", "list at least one site".into());
        }
        for (index, site) in self.sites.iter().enumerate() {
            if !site.is_per_layer() {
                return at(
                    &format!("sites[{index}]"),
                    format!("site {site} has no layer; attribution ranks per-layer sites"),
                );
            }
            if self.sites[..index].contains(site) {
                return at(
                    &format!("sites[{index}]"),
                    format!("site {site} is listed twice"),
                );
            }
        }
        match &self.positions {
            PositionSelector::Named(name) if name != "all" && name != "final" => {
                return at(
                    "positions",
                    format!("expected \"all\", \"final\" or a list, found {name:?}"),
                )
            }
            PositionSelector::List(list) if list.is_empty() => {
                return at("positions", "the position list is empty".into())
            }
            _ => {}
        }
        if let LayerSelector::List(list) = &self.layers
            && list.is_empty()
        {
            return at("layers", "the layer list is empty".into());
        }
        if self.verify_top_k == 0 {
            return at(
                "verify_top_k",
                "verify at least one candidate with a real patch".into(),
            );
        }
        Ok(())
    }
}

/// Exact gradient of `u . norm(x)` with respect to `x` for the RMS norm
/// `norm(x) = g * x / sqrt(mean(x^2) + eps)`, computed in f64.
pub fn rms_norm_readout(x: &[f32], g: &[f32], u: &[f32], eps: f32) -> Result<Vec<f64>, String> {
    let d = x.len();
    if d == 0 || g.len() != d || u.len() != d {
        return Err(format!(
            "readout widths differ: residual {d}, norm weight {}, unembedding {}",
            g.len(),
            u.len()
        ));
    }
    let mean_sq = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / d as f64;
    let s = (mean_sq + f64::from(eps)).sqrt();
    let gu: Vec<f64> = g
        .iter()
        .zip(u)
        .map(|(&g, &u)| f64::from(g) * f64::from(u))
        .collect();
    let projection: f64 = gu.iter().zip(x).map(|(a, &x)| a * f64::from(x)).sum();
    let coefficient = projection / (d as f64 * s * s * s);
    Ok(gu
        .iter()
        .zip(x)
        .map(|(a, &x)| a / s - f64::from(x) * coefficient)
        .collect())
}

/// `delta . r` in f64.
pub fn project(delta: &[f64], readout: &[f64]) -> f64 {
    delta.iter().zip(readout).map(|(a, b)| a * b).sum()
}

/// Round to 9 significant digits so a value survives JSON exactly (the
/// same rule as sweep metrics).
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

/// Average ranks (1-based), ties sharing their mean rank.
fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut out = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end + 1) as f64 / 2.0;
        for &index in &order[start..end] {
            out[index] = rank;
        }
        start = end;
    }
    out
}

/// Pearson correlation; `None` for fewer than two points or no variance.
pub fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 2 {
        return None;
    }
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    (saa > 0.0 && sbb > 0.0).then(|| sab / (saa * sbb).sqrt())
}

/// Spearman rank correlation (Pearson over average ranks).
pub fn spearman(a: &[f64], b: &[f64]) -> Option<f64> {
    pearson(&ranks(a), &ranks(b))
}

/// A resolved token of the metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricToken {
    pub token_id: u32,
    pub piece: String,
}

/// One ranked candidate site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttributionCandidate {
    /// 1-based rank by |estimate| (ties: site order, layer, position).
    pub rank: usize,
    pub site: SemanticHookSite,
    pub layer: usize,
    pub position: usize,
    /// The corrupted prompt's token at `position`.
    pub token: String,
    /// |a_clean - a_corrupted| at this site.
    pub delta_norm: f64,
    /// Whether the candidate has a direct path to the metric (final
    /// position); otherwise the estimate is zero by construction.
    pub direct_path: bool,
    /// Direct-path first-order estimate of patching clean -> corrupted.
    pub estimate: f64,
    /// Measured effect of the real patch (verified candidates only):
    /// `m(patched) - m(corrupted)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual: Option<f64>,
    /// `actual / (m(clean) - m(corrupted))`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovered_fraction: Option<f64>,
}

/// `artifacts/attribution/attribution.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttributionReport {
    pub schema: String,
    pub clean_input: String,
    pub corrupted_input: String,
    pub target: MetricToken,
    pub foil: MetricToken,
    pub metric: String,
    pub approximation: String,
    pub sequence_length: usize,
    /// `m` on the clean and corrupted prompts.
    pub clean_metric: f64,
    pub corrupted_metric: f64,
    /// |r|, the norm of the direct-path readout direction.
    pub readout_norm: f64,
    /// Every candidate, in rank order.
    pub candidates: Vec<AttributionCandidate>,
    /// Number of candidates verified with a real patch (the top ranks).
    pub verified: usize,
    /// Correlations between `estimate` and `actual` over the verified
    /// candidates (`None` with fewer than two or no variance).
    pub spearman: Option<f64>,
    pub pearson: Option<f64>,
    /// Fraction of verified candidates whose estimate and actual effect
    /// have the same sign (zeros count as their own sign).
    pub sign_agreement: Option<f64>,
}

pub const METRIC_DESCRIPTION: &str = "logit(target) - logit(foil) at the final prompt position";
pub const APPROXIMATION_DESCRIPTION: &str =
    "direct-path attribution patching: (a_clean - a_corrupted) . r at the final position, \
     r = exact gradient of the metric through the final RMS norm at the corrupted final \
     residual; 0 at other positions; indirect paths through later blocks ignored";

fn site_rank(site: SemanticHookSite) -> usize {
    SemanticHookSite::ALL
        .iter()
        .position(|candidate| *candidate == site)
        .unwrap_or(usize::MAX)
}

/// Order candidates by |estimate| descending (ties: larger delta norm, then
/// site order, layer, position) and number them.
pub fn rank_candidates(candidates: &mut [AttributionCandidate]) {
    candidates.sort_by(|a, b| {
        b.estimate
            .abs()
            .total_cmp(&a.estimate.abs())
            .then(b.delta_norm.total_cmp(&a.delta_norm))
            .then(site_rank(a.site).cmp(&site_rank(b.site)))
            .then(a.layer.cmp(&b.layer))
            .then(a.position.cmp(&b.position))
    });
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.rank = index + 1;
    }
}

/// Correlation summary over the verified candidates.
pub fn summarize(
    candidates: &[AttributionCandidate],
) -> (usize, Option<f64>, Option<f64>, Option<f64>) {
    let pairs: Vec<(f64, f64)> = candidates
        .iter()
        .filter_map(|c| c.actual.map(|actual| (c.estimate, actual)))
        .collect();
    let (estimates, actuals): (Vec<f64>, Vec<f64>) = pairs.iter().copied().unzip();
    let sign = |v: f64| {
        if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        }
    };
    let agreement = (!pairs.is_empty()).then(|| {
        json_stable(
            pairs.iter().filter(|(e, a)| sign(*e) == sign(*a)).count() as f64 / pairs.len() as f64,
        )
    });
    (
        pairs.len(),
        spearman(&estimates, &actuals).map(json_stable),
        pearson(&estimates, &actuals).map(json_stable),
        agreement,
    )
}

impl AttributionReport {
    /// `candidates.csv`: one row per candidate in rank order.
    pub fn to_csv(&self) -> String {
        fn opt(value: Option<f64>) -> String {
            value.map(|v| v.to_string()).unwrap_or_default()
        }
        let mut out = String::from(
            "rank,site,layer,position,token,delta_norm,direct_path,estimate,actual,\
             recovered_fraction\n",
        );
        for c in &self.candidates {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{}\n",
                c.rank,
                c.site,
                c.layer,
                c.position,
                csv_field(&c.token),
                c.delta_norm,
                c.direct_path,
                c.estimate,
                opt(c.actual),
                opt(c.recovered_fraction),
            ));
        }
        out
    }

    /// The JSON artifact bytes (sorted keys, pretty, newline-terminated).
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, String> {
        let mut bytes = crate::v05::run::pretty_json(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Internal consistency: ranks follow the ranking rule, the summary is
    /// what the recorded values give, and verified candidates are the top
    /// ranks.
    pub fn check_consistency(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != ATTRIBUTION_SCHEMA_V1 {
            errors.push(format!("unknown attribution schema {:?}", self.schema));
        }
        let mut reranked = self.candidates.clone();
        rank_candidates(&mut reranked);
        if reranked != self.candidates {
            errors.push("candidates are not in rank order".into());
        }
        let summary = summarize(&self.candidates);
        if summary
            != (
                self.verified,
                self.spearman,
                self.pearson,
                self.sign_agreement,
            )
        {
            errors.push("the correlation summary differs from the recorded values".into());
        }
        if self
            .candidates
            .iter()
            .enumerate()
            .any(|(index, c)| c.actual.is_some() != (index < self.verified))
        {
            errors.push("verified candidates are not exactly the top ranks".into());
        }
        errors
    }
}

fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

/// Verify the attribution artifacts of a bundle: the JSON parses, is
/// internally consistent, names the bundle's inputs, and the CSV is its
/// table.
pub fn verify_attribution_artifacts(
    input_ids: &[String],
    file: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Vec<String> {
    let Some(json) = file(ATTRIBUTION_JSON) else {
        return vec![format!("{ATTRIBUTION_JSON} is missing")];
    };
    let report: AttributionReport = match serde_json::from_slice(&json) {
        Ok(report) => report,
        Err(error) => return vec![format!("{ATTRIBUTION_JSON} is malformed: {error}")],
    };
    let mut errors = report.check_consistency();
    for id in [&report.clean_input, &report.corrupted_input] {
        if !input_ids.contains(id) {
            errors.push(format!(
                "the report names input {id:?}, which the bundle lacks"
            ));
        }
    }
    match file(ATTRIBUTION_CSV) {
        Some(csv) if csv == report.to_csv().as_bytes() => {}
        Some(_) => errors.push(format!("{ATTRIBUTION_CSV} is not the report's table")),
        None => errors.push(format!("{ATTRIBUTION_CSV} is missing")),
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readout_matches_a_finite_difference() {
        let x = [0.5f32, -1.25, 2.0, 0.75];
        let g = [1.1f32, 0.9, 1.3, 0.7];
        let u = [0.2f32, -0.4, 0.1, 0.6];
        let eps = 1e-5f32;
        let metric = |x: &[f64]| {
            let s = (x.iter().map(|v| v * v).sum::<f64>() / 4.0 + f64::from(eps)).sqrt();
            x.iter()
                .zip(g.iter().zip(&u))
                .map(|(x, (&g, &u))| f64::from(g) * x / s * f64::from(u))
                .sum::<f64>()
        };
        let r = rms_norm_readout(&x, &g, &u, eps).unwrap();
        let base: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        for i in 0..4 {
            let mut plus = base.clone();
            plus[i] += 1e-6;
            let mut minus = base.clone();
            minus[i] -= 1e-6;
            let numeric = (metric(&plus) - metric(&minus)) / 2e-6;
            assert!((numeric - r[i]).abs() < 1e-7, "{i}: {numeric} vs {}", r[i]);
        }
    }

    #[test]
    fn rank_correlations() {
        assert_eq!(spearman(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]), Some(1.0));
        assert_eq!(spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), Some(-1.0));
        assert_eq!(spearman(&[1.0, 1.0], &[1.0, 2.0]), None);
        assert_eq!(ranks(&[5.0, 1.0, 5.0]), vec![2.5, 1.0, 2.5]);
        let rho = spearman(&[1.0, 2.0, 3.0, 4.0], &[1.0, 3.0, 2.0, 4.0]).unwrap();
        assert!((rho - 0.8).abs() < 1e-12);
    }

    #[test]
    fn positions_and_validation() {
        assert_eq!(
            PositionSelector::Named("final".into()).resolve(5).unwrap(),
            vec![4]
        );
        assert_eq!(
            PositionSelector::List(vec![3, 1, 3]).resolve(5).unwrap(),
            vec![1, 3]
        );
        assert!(PositionSelector::List(vec![5]).resolve(5).is_err());
        let spec = AttributionSpec {
            clean: "c".into(),
            corrupted: "x".into(),
            target: TokenRef::Text(" a".into()),
            foil: TokenRef::Id(3),
            sites: default_sites(),
            layers: default_layers(),
            positions: default_positions(),
            verify_top_k: 5,
        };
        assert!(spec.validate(&["c", "x"]).is_ok());
        assert!(spec.validate(&["c"]).is_err());
        let mut bad = spec.clone();
        bad.sites = vec![SemanticHookSite::Logits];
        assert!(bad.validate(&["c", "x"]).is_err());
        let mut bad = spec.clone();
        bad.foil = TokenRef::Text(" a".into());
        assert!(bad.validate(&["c", "x"]).is_err());
    }
}
