//! Driver of the attribution-patching workflow (`ember::v05::attribution`):
//! one capture pass over the clean and corrupted prompts, the direct-path
//! estimate for every candidate, then real `replace` patches for the top
//! candidates. The report and its table become bundle artifacts.

use crate::cli_experiment::{activate_spec, new_input_experiment, run_input, PreparedRun};
use ember::v05::attribution::{
    json_stable, project, rank_candidates, rms_norm_readout, summarize, AttributionCandidate,
    AttributionReport, AttributionSpec, MetricToken, TokenRef, APPROXIMATION_DESCRIPTION,
    ATTRIBUTION_CSV, ATTRIBUTION_JSON, ATTRIBUTION_SCHEMA_V1, METRIC_DESCRIPTION,
};
use ember::v05::capture::{
    CaptureDType, CaptureSpec, CaptureStorage, InputSelector, LayerSelector,
};
use ember::v05::hook::SemanticHookSite;
use ember::v05::intervention::{
    CompatibilityPolicy, InterventionOperation, InterventionSource, InterventionSpec, ShapePolicy,
};
use ember::v05::runner::InputResult;
use ember::v05::spec::{ExperimentSpecV1, InputSpec};
use ember::v05::token_select::{tokenize_for_selection, TextNormalization, TokenSelector};
use std::collections::BTreeMap;

/// Run a derived, prefill-only experiment on the prepared session and
/// return its per-input results (no bundle is written).
pub(crate) fn run_derived(
    prepared: &PreparedRun,
    spec: &ExperimentSpecV1,
) -> anyhow::Result<Vec<InputResult>> {
    let active = activate_spec(prepared, spec)?;
    let mut results = Vec::with_capacity(spec.inputs.len());
    for index in 0..spec.inputs.len() {
        let experiment = new_input_experiment(prepared, spec, Some(&active), index)?;
        results.push(run_input(
            prepared,
            spec,
            &active,
            &experiment,
            None,
            None,
            None,
        )?);
    }
    Ok(results)
}

/// A capture of every selected row (or the full prefill tensor).
pub(crate) fn capture(
    id: &str,
    site: SemanticHookSite,
    layers: LayerSelector,
    tokens: TokenSelector,
    storage: CaptureStorage,
) -> CaptureSpec {
    CaptureSpec {
        id: id.into(),
        site,
        layers,
        tokens,
        inputs: InputSelector::All("all".into()),
        storage,
        dtype: CaptureDType::F32,
    }
}

/// A prefill-only copy of `resolved` with the given inputs and captures and
/// no interventions.
pub(crate) fn derived_spec(
    resolved: &ExperimentSpecV1,
    inputs: Vec<InputSpec>,
    captures: Vec<CaptureSpec>,
) -> ExperimentSpecV1 {
    let mut spec = resolved.clone();
    spec.inputs = inputs;
    spec.captures = captures;
    spec.interventions = Vec::new();
    spec.attribution = None;
    spec.generation.max_new_tokens = 0;
    spec
}

/// Resolve a metric token: an id inside the vocabulary, or text that
/// encodes to exactly one token.
pub(crate) fn resolve_token(
    prepared: &PreparedRun,
    token: &TokenRef,
) -> anyhow::Result<MetricToken> {
    let vocab = prepared.model.config.vocab_size;
    let id = match token {
        TokenRef::Id(id) => *id,
        TokenRef::Text(text) => {
            let ids = prepared.tokenizer.encode_no_special(text)?;
            match ids.as_slice() {
                [id] => *id,
                _ => anyhow::bail!(
                    "token text {text:?} encodes to {} tokens {ids:?}; name a single token (or \
                     give its id)",
                    ids.len()
                ),
            }
        }
    };
    anyhow::ensure!(
        (id as usize) < vocab,
        "token {id} is outside the model's {vocab}-token vocabulary"
    );
    Ok(MetricToken {
        token_id: id,
        piece: prepared.tokenizer.token_piece(id).unwrap_or_default(),
    })
}

/// `logit(target) - logit(foil)` from a logits capture row.
pub(crate) fn logit_difference(
    result: &InputResult,
    capture_id: &str,
    target: u32,
    foil: u32,
) -> anyhow::Result<f64> {
    let row = &result
        .captures
        .iter()
        .find(|c| c.capture_id == capture_id)
        .ok_or_else(|| anyhow::anyhow!("the logits capture did not fire"))?
        .rows;
    let at = |id: u32| {
        row.get(id as usize)
            .map(|value| f64::from(*value))
            .ok_or_else(|| anyhow::anyhow!("token {id} is outside the logits row"))
    };
    Ok(at(target)? - at(foil)?)
}

const LOGITS: &str = "attribution-logits";
const FINAL: &str = "attribution-final-residual";

/// Run the attribution workflow and return its artifact files.
pub(crate) fn run_attribution(
    prepared: &PreparedRun,
    resolved: &ExperimentSpecV1,
    spec: &AttributionSpec,
) -> anyhow::Result<BTreeMap<String, Vec<u8>>> {
    let report = attribution_report(prepared, resolved, spec)?;
    let mut files = BTreeMap::new();
    files.insert(
        ATTRIBUTION_JSON.to_string(),
        report.to_json_bytes().map_err(anyhow::Error::msg)?,
    );
    files.insert(ATTRIBUTION_CSV.to_string(), report.to_csv().into_bytes());
    Ok(files)
}

pub(crate) fn attribution_report(
    prepared: &PreparedRun,
    resolved: &ExperimentSpecV1,
    spec: &AttributionSpec,
) -> anyhow::Result<AttributionReport> {
    let input = |id: &str| {
        resolved
            .inputs
            .iter()
            .find(|input| input.id == id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("attribution input {id:?} is not declared"))
    };
    let (clean, corrupted) = (input(&spec.clean)?, input(&spec.corrupted)?);
    let tokenize = |text: &str| {
        tokenize_for_selection(&prepared.tokenizer, text, TextNormalization::None)
            .map_err(anyhow::Error::msg)
    };
    let (clean_tokens, corrupted_tokens) = (tokenize(&clean.text)?, tokenize(&corrupted.text)?);
    let n = corrupted_tokens.token_ids.len();
    anyhow::ensure!(
        clean_tokens.token_ids.len() == n && n > 0,
        "attribution: the clean prompt has {} tokens and the corrupted prompt {n}; positions are \
         aligned index by index, so they must have the same length",
        clean_tokens.token_ids.len()
    );
    let target = resolve_token(prepared, &spec.target)?;
    let foil = resolve_token(prepared, &spec.foil)?;
    anyhow::ensure!(
        target.token_id != foil.token_id,
        "attribution: target and foil resolve to the same token {}",
        target.token_id
    );
    let n_layers = prepared.n_layers;
    let layers = spec
        .layers
        .resolve(n_layers)
        .map_err(|error| anyhow::anyhow!("attribution.layers: {error}"))?;
    let positions = spec
        .positions
        .resolve(n)
        .map_err(|error| anyhow::anyhow!("attribution.positions: {error}"))?;

    // 1. One capture pass: every candidate site in full, the corrupted
    //    residual entering the final norm, and the final logits.
    let mut captures: Vec<CaptureSpec> = spec
        .sites
        .iter()
        .enumerate()
        .map(|(index, site)| {
            capture(
                &format!("attribution-site-{index}"),
                *site,
                LayerSelector::List(layers.clone()),
                TokenSelector::PromptFinal,
                CaptureStorage::FullTensor,
            )
        })
        .collect();
    captures.push(capture(
        FINAL,
        SemanticHookSite::ResidualPostMlp,
        LayerSelector::List(vec![n_layers - 1]),
        TokenSelector::PromptFinal,
        CaptureStorage::SelectedRows,
    ));
    captures.push(capture(
        LOGITS,
        SemanticHookSite::Logits,
        LayerSelector::All("all".into()),
        TokenSelector::PromptFinal,
        CaptureStorage::SelectedRows,
    ));
    let pass = derived_spec(resolved, vec![clean.clone(), corrupted.clone()], captures);
    let results = run_derived(prepared, &pass)?;
    let (clean_result, corrupted_result) = (&results[0], &results[1]);
    let clean_metric = logit_difference(clean_result, LOGITS, target.token_id, foil.token_id)?;
    let corrupted_metric =
        logit_difference(corrupted_result, LOGITS, target.token_id, foil.token_id)?;

    // 2. The direct-path readout at the corrupted final residual.
    let final_residual = &corrupted_result
        .captures
        .iter()
        .find(|c| c.capture_id == FINAL)
        .ok_or_else(|| anyhow::anyhow!("the final residual capture did not fire"))?
        .rows;
    let lens = ember::v05::lens::ModelLens::new(&prepared.model);
    let (norm_weight, eps) = lens.final_norm_weight();
    let target_row = lens
        .unembedding_row(target.token_id)
        .map_err(anyhow::Error::msg)?;
    let foil_row = lens
        .unembedding_row(foil.token_id)
        .map_err(anyhow::Error::msg)?;
    let u: Vec<f32> = target_row
        .iter()
        .zip(&foil_row)
        .map(|(a, b)| a - b)
        .collect();
    let readout =
        rms_norm_readout(final_residual, &norm_weight, &u, eps).map_err(anyhow::Error::msg)?;
    let readout_norm = readout.iter().map(|v| v * v).sum::<f64>().sqrt();

    // 3. Every candidate's estimate.
    let full_rows = |result: &InputResult, id: &str, layer: usize| -> anyhow::Result<Vec<f32>> {
        let capture = result
            .captures
            .iter()
            .find(|c| c.capture_id == id && c.layer == layer)
            .ok_or_else(|| anyhow::anyhow!("capture {id} layer {layer} did not fire"))?;
        anyhow::ensure!(
            capture.positions == (0..n).collect::<Vec<_>>(),
            "capture {id} layer {layer} does not cover the prompt"
        );
        Ok(capture.rows.clone())
    };
    let columns = prepared.embed_dim;
    let mut candidates = Vec::new();
    let mut clean_rows: BTreeMap<(usize, usize, usize), Vec<f32>> = BTreeMap::new();
    for (site_index, site) in spec.sites.iter().enumerate() {
        let id = format!("attribution-site-{site_index}");
        for &layer in &layers {
            let clean_full = full_rows(clean_result, &id, layer)?;
            let corrupted_full = full_rows(corrupted_result, &id, layer)?;
            for &position in &positions {
                let range = position * columns..(position + 1) * columns;
                let delta: Vec<f64> = clean_full[range.clone()]
                    .iter()
                    .zip(&corrupted_full[range.clone()])
                    .map(|(a, b)| f64::from(*a) - f64::from(*b))
                    .collect();
                let direct_path = position + 1 == n;
                let estimate = if direct_path {
                    project(&delta, &readout)
                } else {
                    0.0
                };
                clean_rows.insert((site_index, layer, position), clean_full[range].to_vec());
                candidates.push(AttributionCandidate {
                    rank: 0,
                    site: *site,
                    layer,
                    position,
                    token: corrupted_tokens.pieces[position].clone(),
                    delta_norm: json_stable(delta.iter().map(|v| v * v).sum::<f64>().sqrt()),
                    direct_path,
                    estimate: json_stable(estimate),
                    actual: None,
                    recovered_fraction: None,
                });
            }
        }
    }
    rank_candidates(&mut candidates);

    // 4. Real patches for the top candidates: the corrupted run with the
    //    clean row written at the candidate site.
    let gap = clean_metric - corrupted_metric;
    let verify = spec.verify_top_k.min(candidates.len());
    for candidate in candidates.iter_mut().take(verify) {
        let site_index = spec
            .sites
            .iter()
            .position(|site| *site == candidate.site)
            .expect("candidate site is listed");
        let row = clean_rows[&(site_index, candidate.layer, candidate.position)].clone();
        let mut patch = derived_spec(
            resolved,
            vec![corrupted.clone()],
            vec![capture(
                LOGITS,
                SemanticHookSite::Logits,
                LayerSelector::All("all".into()),
                TokenSelector::PromptFinal,
                CaptureStorage::SelectedRows,
            )],
        );
        patch.interventions = vec![InterventionSpec {
            id: "attribution-patch".into(),
            site: candidate.site,
            layers: LayerSelector::List(vec![candidate.layer]),
            tokens: TokenSelector::AbsoluteToken {
                index: candidate.position,
            },
            inputs: InputSelector::All("all".into()),
            operation: InterventionOperation::Replace,
            source: Some(InterventionSource::InlineVector { values: row }),
            shape_policy: ShapePolicy::Strict,
            compatibility: CompatibilityPolicy::default(),
        }];
        let patched = run_derived(prepared, &patch)?;
        let metric = logit_difference(&patched[0], LOGITS, target.token_id, foil.token_id)?;
        let actual = json_stable(metric - corrupted_metric);
        candidate.actual = Some(actual);
        candidate.recovered_fraction = (gap != 0.0).then(|| json_stable(actual / gap));
    }
    let (verified, spearman, pearson, sign_agreement) = summarize(&candidates);
    Ok(AttributionReport {
        schema: ATTRIBUTION_SCHEMA_V1.into(),
        clean_input: clean.id,
        corrupted_input: corrupted.id,
        target,
        foil,
        metric: METRIC_DESCRIPTION.into(),
        approximation: APPROXIMATION_DESCRIPTION.into(),
        sequence_length: n,
        clean_metric: json_stable(clean_metric),
        corrupted_metric: json_stable(corrupted_metric),
        readout_norm: json_stable(readout_norm),
        candidates,
        verified,
        spearman,
        pearson,
        sign_agreement,
    })
}

fn fmt_opt(value: Option<f64>, precision: usize) -> String {
    value
        .map(|v| format!("{v:.precision$}"))
        .unwrap_or_else(|| "-".into())
}

/// Print the attribution table of a report.
pub(crate) fn print_attribution(report: &AttributionReport, rows: usize) {
    println!(
        "attribution: {} -> {} ({} tokens), metric logit({:?}) - logit({:?})",
        report.clean_input,
        report.corrupted_input,
        report.sequence_length,
        report.target.piece,
        report.foil.piece
    );
    println!(
        "  clean {:.4}, corrupted {:.4}, gap {:.4}; |readout| {:.4}",
        report.clean_metric,
        report.corrupted_metric,
        report.clean_metric - report.corrupted_metric,
        report.readout_norm
    );
    println!(
        "  {:>4}  {:<24} {:>5} {:>4}  {:<14} {:>10} {:>10} {:>9}",
        "rank", "site", "layer", "pos", "token", "estimate", "actual", "recovered"
    );
    for c in report.candidates.iter().take(rows) {
        println!(
            "  {:>4}  {:<24} {:>5} {:>4}  {:<14} {:>10.4} {:>10} {:>9}",
            c.rank,
            c.site.to_string(),
            c.layer,
            c.position,
            format!("{:?}", c.token),
            c.estimate,
            fmt_opt(c.actual, 4),
            fmt_opt(c.recovered_fraction, 3),
        );
    }
    println!(
        "  verified {} of {} candidates with real patches: spearman {}, pearson {}, sign \
         agreement {}",
        report.verified,
        report.candidates.len(),
        fmt_opt(report.spearman, 3),
        fmt_opt(report.pearson, 3),
        fmt_opt(report.sign_agreement, 3)
    );
}

/// After `experiment run`: print the analysis tables a bundle carries.
pub(crate) fn print_bundle_reports(bundle: &std::path::Path) -> anyhow::Result<()> {
    let path = bundle.join(ATTRIBUTION_JSON);
    if path.is_file() {
        let report: AttributionReport = serde_json::from_slice(&std::fs::read(&path)?)?;
        print_attribution(&report, report.verified.max(10));
        println!("  report: {} (and candidates.csv)", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_experiment::{execute_prepared, prepare_run};
    use crate::experiment_testutil::{resolve, spec_text, tiny_model};
    use ember::quant_k::KStrategy;
    use ember::v05::verify::{load_bundle_for_source, verify_bundle, VerifyOptions};

    fn body(corrupted: &str, extra: &str) -> String {
        format!(
            r#"
[[inputs]]
id = "clean"
text = "w3 w17 w5 w40 w9"

[[inputs]]
id = "corrupted"
text = "{corrupted}"

[attribution]
clean = "clean"
corrupted = "corrupted"
target = 10
foil = 11
verify_top_k = 60
{extra}
"#
        )
    }

    #[test]
    fn attribution_recovers_the_known_structure_of_a_single_token_corruption() {
        // The prompts differ only at position 2 (w5 -> w6); the word-level
        // tokenizer adds no BOS, so positions are word indices.
        let model = tiny_model("attribution", 4, 64, false);
        let text = spec_text(&model, "reference", 0, &body("w3 w17 w6 w40 w9", ""));
        let resolved = resolve(&text);
        let mut prepared = prepare_run(&resolved, KStrategy::Auto, false).unwrap();
        let out = model.dir.join("bundle");
        let (path, _, report, _) =
            execute_prepared(&mut prepared, &resolved, &text, &out, false, None).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        assert!(report
            .checks
            .iter()
            .any(|check| check.name == "attribution report" && check.ok));
        let bundle = load_bundle_for_source(&path).unwrap();
        let attribution: AttributionReport =
            serde_json::from_slice(bundle.file(ATTRIBUTION_JSON).unwrap()).unwrap();
        assert_eq!(attribution.candidates.len(), 3 * 4 * 5);
        assert_eq!(attribution.verified, 60);
        let gap = attribution.clean_metric - attribution.corrupted_metric;
        assert!(gap.abs() > 1e-3, "the corruption must move the metric");
        let find = |site: SemanticHookSite, layer: usize, position: usize| {
            attribution
                .candidates
                .iter()
                .find(|c| c.site == site && c.layer == layer && c.position == position)
                .unwrap()
        };
        for c in &attribution.candidates {
            // Causal attention: nothing before the corrupted token differs,
            // and patching it changes nothing.
            if c.position < 2 {
                assert_eq!(c.delta_norm, 0.0, "{c:?}");
                assert_eq!(c.actual, Some(0.0), "{c:?}");
            }
            // Only the final position has a direct path.
            assert_eq!(c.direct_path, c.position == 4);
            if !c.direct_path {
                assert_eq!(c.estimate, 0.0);
            }
        }
        // The last block's MLP output at an earlier position never reaches
        // the final logits: exactly zero effect.
        for position in 0..4 {
            assert_eq!(
                find(SemanticHookSite::MlpOutput, 3, position).actual,
                Some(0.0)
            );
        }
        // Restoring the corrupted token's embedding restores everything.
        let embedding = find(SemanticHookSite::ResidualPreAttention, 0, 2);
        let recovered = embedding.recovered_fraction.unwrap();
        assert!((recovered - 1.0).abs() < 1e-6, "{embedding:?}");
        // For the last block the direct path is the only path: the
        // estimate is first-order exact, off only by the final norm's
        // curvature.
        let last = find(SemanticHookSite::MlpOutput, 3, 4);
        let actual = last.actual.unwrap();
        assert!(
            (last.estimate - actual).abs() <= 0.25 * actual.abs().max(1e-3),
            "{last:?}"
        );
        assert!(attribution.spearman.is_some());
        let csv = std::str::from_utf8(bundle.file(ATTRIBUTION_CSV).unwrap()).unwrap();
        assert_eq!(csv, attribution.to_csv());
        assert_eq!(csv.lines().count(), 61);

        // Tampered artifacts fail verification.
        let tampered = model.dir.join("tampered");
        std::fs::create_dir_all(&tampered).unwrap();
        copy_tree(&path, &tampered);
        let json = tampered.join(ATTRIBUTION_JSON);
        let mut forged = attribution.clone();
        forged.spearman = Some(0.999);
        std::fs::write(&json, forged.to_json_bytes().unwrap()).unwrap();
        let report = verify_bundle(&tampered, &VerifyOptions::default()).unwrap();
        assert!(!report.ok);

        // Same spec, fresh session: identical bundle identity.
        let mut fresh = prepare_run(&resolved, KStrategy::Auto, false).unwrap();
        let (_, a, _, _) = execute_prepared(
            &mut fresh,
            &resolved,
            &text,
            &model.dir.join("again"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            a.semantic_hash, bundle.semantic_hash,
            "attribution is deterministic"
        );
    }

    fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                std::fs::create_dir_all(&target).unwrap();
                copy_tree(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    #[test]
    fn attribution_fails_closed() {
        let model = tiny_model("attribution-invalid", 2, 64, false);
        // Different token counts are refused at run time.
        let text = spec_text(&model, "reference", 0, &body("w3 w17 w6 w40", ""));
        let resolved = resolve(&text);
        let mut prepared = prepare_run(&resolved, KStrategy::Auto, false).unwrap();
        let error = execute_prepared(
            &mut prepared,
            &resolved,
            &text,
            &model.dir.join("x"),
            false,
            None,
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("same length"), "{error:#}");
        let parse = |body: String| {
            ember::v05::spec::RawExperimentSpec::from_toml_str(&spec_text(
                &model,
                "reference",
                0,
                &body,
            ))
            .unwrap()
            .resolve()
            .unwrap_err()
            .to_string()
        };
        let error = parse(body("w1", "sites = [\"logits\"]"));
        assert!(error.contains("attribution.sites[0]"), "{error}");
        let error =
            parse(body("w1", "").replace("corrupted = \"corrupted\"", "corrupted = \"missing\""));
        assert!(error.contains("attribution.corrupted"), "{error}");
        let with_intervention = format!(
            "{}\n[[interventions]]\nid = \"z\"\nsite = \"mlp-output\"\nlayers = [0]\noperation = {{ kind = \"zero\" }}\n[interventions.tokens]\nkind = \"prompt-final\"\n",
            body("w1", "")
        );
        assert!(parse(with_intervention).contains("interventions"));
        // Target text must be a single token.
        let two = body("w3 w17 w6 w40 w9", "").replace("target = 10", "target = \"w1 w2\"");
        let text = spec_text(&model, "reference", 0, &two);
        let resolved = resolve(&text);
        let error = execute_prepared(
            &mut prepared,
            &resolved,
            &text,
            &model.dir.join("y"),
            false,
            None,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("encodes to 2 tokens"),
            "{error:#}"
        );
    }
}
