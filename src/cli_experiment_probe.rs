//! Driver of the probe bridge (`ember::v05::probe`): obtain the probe
//! direction (pinned file or in-run ridge fit), measure its accuracy, then
//! intervene along it on the spec's inputs and measure the behavioural
//! change against the unintervened baseline.

use crate::cli_experiment::PreparedRun;
use crate::cli_experiment_attribution::{capture, derived_spec, resolve_token, run_derived};
use ember::v05::capture::{CaptureStorage, InputSelector, LayerSelector};
use ember::v05::hook::SemanticHookSite;
use ember::v05::intervention::{
    CompatibilityPolicy, InterventionOperation, InterventionSource, InterventionSpec, ShapePolicy,
};
use ember::v05::manifest::sha256_hex;
use ember::v05::probe::{
    fit_ridge, midpoint_bias, stable_effect, summarize_effects, token_probability, LinearProbe,
    ProbeEffect, ProbeRecord, ProbeReport, ProbeSpec, ProbeVariant, PROBE_CSV, PROBE_DIRECTION,
    PROBE_JSON, PROBE_SCHEMA_V1,
};
use ember::v05::runner::InputResult;
use ember::v05::spec::ExperimentSpecV1;
use ember::v05::token_select::TokenSelector;
use std::collections::BTreeMap;

const LOGITS: &str = "probe-logits";

fn stable(value: f64) -> f64 {
    ember::v05::attribution::json_stable(value)
}

/// Run the probe bridge and return its artifact files.
pub(crate) fn run_probe_bridge(
    prepared: &PreparedRun,
    resolved: &ExperimentSpecV1,
    spec: &ProbeSpec,
) -> anyhow::Result<BTreeMap<String, Vec<u8>>> {
    let (report, direction) = probe_report(prepared, resolved, spec)?;
    let bytes: Vec<u8> = direction.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor = ember::v05::safetensors::serialize(&[ember::v05::safetensors::TensorData {
        name: "direction",
        dtype: ember::v05::safetensors::TensorDType::F32,
        shape: &[direction.len()],
        bytes: &bytes,
    }])
    .map_err(anyhow::Error::msg)?;
    let mut files = BTreeMap::new();
    files.insert(
        PROBE_JSON.to_string(),
        report.to_json_bytes().map_err(anyhow::Error::msg)?,
    );
    files.insert(PROBE_CSV.to_string(), report.to_csv().into_bytes());
    files.insert(PROBE_DIRECTION.to_string(), tensor);
    Ok(files)
}

/// The probe's direction and fitted threshold, from a file or a ridge fit.
fn probe_direction(
    prepared: &PreparedRun,
    spec: &ProbeSpec,
    train: &[Vec<f64>],
    train_labels: &[u8],
) -> anyhow::Result<(Vec<f32>, Option<f64>)> {
    match &spec.file {
        None => {
            let probe =
                fit_ridge(train, train_labels, spec.ridge_lambda).map_err(anyhow::Error::msg)?;
            Ok((
                probe.weights.iter().map(|&w| w as f32).collect(),
                Some(probe.bias),
            ))
        }
        Some(file) => {
            let matrix = ember::v05::steering::read_direction_file(
                &file.path,
                &file.sha256,
                file.tensor.as_deref(),
            )
            .map_err(|error| anyhow::anyhow!("probe.file: {error}"))?;
            let mut by_layer = ember::v05::steering::directions_for_layers(
                &matrix,
                &[spec.layer],
                prepared.n_layers,
                prepared.embed_dim,
                true,
            )
            .map_err(|error| anyhow::anyhow!("probe.file: {error}"))?;
            let direction = by_layer.remove(&spec.layer).expect("mapped layer");
            let weights: Vec<f64> = direction.iter().map(|&v| f64::from(v)).collect();
            Ok((direction, midpoint_bias(&weights, train, train_labels)))
        }
    }
}

fn logits_row(result: &InputResult) -> anyhow::Result<&[f32]> {
    Ok(&result
        .captures
        .iter()
        .find(|c| c.capture_id == LOGITS)
        .ok_or_else(|| anyhow::anyhow!("the logits capture did not fire"))?
        .rows)
}

fn probe_report(
    prepared: &PreparedRun,
    resolved: &ExperimentSpecV1,
    spec: &ProbeSpec,
) -> anyhow::Result<(ProbeReport, Vec<f32>)> {
    anyhow::ensure!(
        spec.layer < prepared.n_layers,
        "probe.layer {} is out of range for a {}-layer model",
        spec.layer,
        prepared.n_layers
    );
    let target = resolve_token(prepared, &spec.target)?;

    // 1. Probe rows for every labelled example.
    let examples: Vec<&ember::v05::probe::LabelledPrompt> =
        spec.train.iter().chain(&spec.test).collect();
    let rows: Vec<Vec<f64>> = if examples.is_empty() {
        Vec::new()
    } else {
        crate::cli_experiment_steering::prompt_means(
            prepared,
            resolved,
            spec.site,
            &LayerSelector::List(vec![spec.layer]),
            &spec.tokens,
            examples.iter().map(|example| &example.text),
            "probe",
        )?
        .into_iter()
        .map(|mut by_layer| {
            by_layer
                .remove(&spec.layer)
                .ok_or_else(|| anyhow::anyhow!("probe: a labelled prompt recorded no row"))
        })
        .collect::<anyhow::Result<_>>()?
    };
    let (train_rows, test_rows) = rows.split_at(spec.train.len());
    let train_labels: Vec<u8> = spec.train.iter().map(|e| e.label).collect();
    let test_labels: Vec<u8> = spec.test.iter().map(|e| e.label).collect();

    // 2. The direction, its threshold and accuracy.
    let (direction, bias) = probe_direction(prepared, spec, train_rows, &train_labels)?;
    let norm = direction
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    anyhow::ensure!(norm > 0.0, "probe: the direction is zero");
    let classifier = bias.map(|bias| LinearProbe {
        weights: direction.iter().map(|&v| f64::from(v)).collect(),
        bias,
    });
    let accuracy = |rows: &[Vec<f64>], labels: &[u8]| {
        classifier
            .as_ref()
            .and_then(|probe| probe.accuracy(rows, labels))
            .map(stable)
    };
    let direction_bytes: Vec<u8> = direction.iter().flat_map(|v| v.to_le_bytes()).collect();
    let record = ProbeRecord {
        site: spec.site,
        layer: spec.layer,
        source: if spec.file.is_some() {
            "vector-file"
        } else {
            "trained-ridge"
        }
        .into(),
        file_sha256: spec.file.as_ref().map(|file| file.sha256.clone()),
        ridge_lambda: spec.file.is_none().then_some(spec.ridge_lambda),
        train_examples: spec.train.len(),
        test_examples: spec.test.len(),
        train_accuracy: accuracy(train_rows, &train_labels),
        test_accuracy: accuracy(test_rows, &test_labels),
        bias: bias.map(stable),
        direction_norm: stable(norm),
        direction_checksum: sha256_hex(&direction_bytes),
    };

    // 3. Behaviour: the baseline and every variant over the spec's inputs,
    //    with their own generation settings.
    let behaviour = |intervention: Option<InterventionSpec>| -> anyhow::Result<Vec<InputResult>> {
        let mut derived = derived_spec(
            resolved,
            resolved.inputs.clone(),
            vec![capture(
                LOGITS,
                SemanticHookSite::Logits,
                LayerSelector::All("all".into()),
                TokenSelector::PromptFinal,
                CaptureStorage::SelectedRows,
            )],
        );
        derived.generation = resolved.generation.clone();
        derived.interventions = intervention.into_iter().collect();
        run_derived(prepared, &derived)
    };
    let intervention = |operation| InterventionSpec {
        id: "probe-intervention".into(),
        site: spec.site,
        layers: LayerSelector::List(vec![spec.layer]),
        tokens: spec.intervene_tokens.clone(),
        inputs: InputSelector::All("all".into()),
        operation,
        source: Some(InterventionSource::InlineVector {
            values: direction.clone(),
        }),
        shape_policy: ShapePolicy::Strict,
        compatibility: CompatibilityPolicy::default(),
    };
    let token = target.token_id as usize;
    let baseline = behaviour(None)?;
    let mut base_values = Vec::new();
    let mut effects = Vec::new();
    for result in &baseline {
        let logits = logits_row(result)?;
        let logit =
            f64::from(*logits.get(token).ok_or_else(|| {
                anyhow::anyhow!("target token {token} is outside the logits row")
            })?);
        let probability = token_probability(logits, token).unwrap_or(0.0);
        base_values.push((logit, probability));
        effects.push(stable_effect(ProbeEffect {
            variant: "baseline".into(),
            input_id: result.input.id.clone(),
            target_logit: logit,
            target_probability: probability,
            delta_logit: 0.0,
            delta_probability: 0.0,
            generated_text: result.generated_text.clone(),
            text_changed: false,
            first_divergent_step: None,
        }));
    }
    for variant in spec.variants() {
        let operation = match variant {
            ProbeVariant::Ablate => InterventionOperation::AblateProjection,
            ProbeVariant::Steer(alpha) => InterventionOperation::Steer {
                alpha,
                normalize: spec.steer_normalize,
            },
        };
        let results = behaviour(Some(intervention(operation)))?;
        for ((result, base), (base_logit, base_probability)) in
            results.iter().zip(&baseline).zip(&base_values)
        {
            let logits = logits_row(result)?;
            let logit = f64::from(logits[token]);
            let probability = token_probability(logits, token).unwrap_or(0.0);
            let first_divergent_step = result
                .generated_token_ids
                .iter()
                .zip(&base.generated_token_ids)
                .position(|(a, b)| a != b)
                .or_else(|| {
                    (result.generated_token_ids.len() != base.generated_token_ids.len()).then(
                        || {
                            result
                                .generated_token_ids
                                .len()
                                .min(base.generated_token_ids.len())
                        },
                    )
                })
                .map(|step| step + 1);
            effects.push(stable_effect(ProbeEffect {
                variant: variant.id(),
                input_id: result.input.id.clone(),
                target_logit: logit,
                target_probability: probability,
                delta_logit: logit - base_logit,
                delta_probability: probability - base_probability,
                generated_text: result.generated_text.clone(),
                text_changed: result.generated_text != base.generated_text,
                first_divergent_step,
            }));
        }
    }
    let summary = summarize_effects(&effects);
    Ok((
        ProbeReport {
            schema: PROBE_SCHEMA_V1.into(),
            probe: record,
            target,
            intervene_tokens: spec.intervene_tokens.clone(),
            effects,
            summary,
        },
        direction,
    ))
}

fn pct(value: Option<f64>) -> String {
    value
        .map(|v| format!("{:.1}%", v * 100.0))
        .unwrap_or_else(|| "-".into())
}

/// Print the probe bridge summary.
pub(crate) fn print_probe(report: &ProbeReport) {
    let probe = &report.probe;
    println!(
        "probe bridge: {} probe at {} layer {} ({} train / {} test examples)",
        probe.source, probe.site, probe.layer, probe.train_examples, probe.test_examples
    );
    println!(
        "  can the probe read it?  train accuracy {}, held-out accuracy {}",
        pct(probe.train_accuracy),
        pct(probe.test_accuracy)
    );
    println!(
        "  can the model use it?   target {:?}: effect of intervening along the direction",
        report.target.piece
    );
    println!(
        "  {:<14} {:>16} {:>18} {:>14}",
        "variant", "mean d-logit", "mean d-prob", "texts changed"
    );
    for summary in &report.summary {
        println!(
            "  {:<14} {:>16.4} {:>18.6} {:>11}/{}",
            summary.variant,
            summary.mean_delta_logit,
            summary.mean_delta_probability,
            summary.texts_changed,
            summary.inputs
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_experiment::{execute_prepared, prepare_run};
    use crate::experiment_testutil::{resolve, spec_text, tiny_model, TinyModel};
    use ember::quant_k::KStrategy;
    use ember::v05::intervention::SteerNormalization;
    use ember::v05::verify::{load_bundle_for_source, verify_bundle, VerifyOptions};

    const INPUTS: &str = r#"
[[inputs]]
id = "h1"
text = "w20 w21 w1"

[[inputs]]
id = "h2"
text = "w22 w23 w44"
"#;

    /// Label 1: the prompt ends in w1..w5; label 0: it ends in w40..w44.
    fn examples(prefix: &str, count: usize) -> String {
        (0..count)
            .map(|i| {
                let (word, label) = if i % 2 == 0 {
                    (1 + i % 5, 1)
                } else {
                    (40 + i % 5, 0)
                };
                format!(
                    "{{ text = \"w{} w{} w{word}\", label = {label} }}",
                    prefix.len() + i,
                    10 + i
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn probe_body(layer: usize, source: &str, extra: &str) -> String {
        format!(
            r#"{INPUTS}
[probe]
site = "residual-post-mlp"
layer = {layer}
{source}
test = [{}]
ridge_lambda = 0.5
ablate = true
steer_alphas = [0.0, 3.0]
target = 7
{extra}
"#,
            examples("tt", 6)
        )
    }

    fn run(
        prepared: &mut crate::cli_experiment::PreparedRun,
        model: &TinyModel,
        text: &str,
        name: &str,
    ) -> (std::path::PathBuf, ProbeReport, Vec<f32>) {
        let resolved = resolve(text);
        let (path, _, report, _) = execute_prepared(
            prepared,
            &resolved,
            text,
            &model.dir.join(name),
            false,
            None,
        )
        .unwrap();
        assert!(report.ok, "{:?}", report.checks);
        assert!(report
            .checks
            .iter()
            .any(|check| check.name == "probe bridge report" && check.ok));
        let bundle = load_bundle_for_source(&path).unwrap();
        let probe: ProbeReport = serde_json::from_slice(bundle.file(PROBE_JSON).unwrap()).unwrap();
        let direction = ember::v05::steering::parse_safetensors_direction(
            bundle.file(PROBE_DIRECTION).unwrap(),
            None,
        )
        .unwrap()
        .values;
        (path, probe, direction)
    }

    #[test]
    fn trained_probe_bridge_measures_accuracy_and_causal_effect() {
        let model = tiny_model("probe-bridge", 4, 64, false);
        let text = spec_text(
            &model,
            "reference",
            3,
            &probe_body(1, &format!("train = [{}]", examples("t", 12)), ""),
        );
        let mut prepared = prepare_run(&resolve(&text), KStrategy::Auto, false).unwrap();
        let (path, report, direction) = run(&mut prepared, &model, &text, "trained");
        assert_eq!(report.probe.source, "trained-ridge");
        // 12 examples in 64 dimensions: ridge interpolates the training set.
        assert_eq!(report.probe.train_accuracy, Some(1.0));
        let test = report.probe.test_accuracy.unwrap();
        assert!((0.0..=1.0).contains(&test));
        let variants: Vec<&str> = report.summary.iter().map(|s| s.variant.as_str()).collect();
        assert_eq!(variants, ["ablate", "steer+0", "steer+3"]);
        // alpha 0 changes nothing, bit for bit.
        let zero = &report.summary[1];
        assert_eq!(
            (zero.mean_delta_logit, zero.texts_changed),
            (0.0, 0),
            "{zero:?}"
        );
        // Deterministic: the same spec again gives the same bundle.
        let resolved = resolve(&text);
        let (_, again, _, _) = execute_prepared(
            &mut prepared,
            &resolved,
            &text,
            &model.dir.join("again"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            again.semantic_hash,
            load_bundle_for_source(&path).unwrap().semantic_hash
        );

        // The same direction from a pinned file: identical effects.
        let npy = model.dir.join("probe.npy");
        let bytes = ember::v05::steering::write_npy(&[64], &direction);
        std::fs::write(&npy, &bytes).unwrap();
        let file_text = spec_text(
            &model,
            "reference",
            3,
            &probe_body(
                1,
                &format!(
                    "train = [{}]\nfile = {{ path = {:?}, sha256 = \"{}\" }}",
                    examples("t", 12),
                    npy.display().to_string(),
                    sha256_hex(&bytes)
                ),
                "",
            ),
        );
        let (_, from_file, _) = run(&mut prepared, &model, &file_text, "file");
        assert_eq!(from_file.probe.source, "vector-file");
        assert_eq!(
            from_file.probe.direction_checksum,
            report.probe.direction_checksum
        );
        assert_eq!(from_file.effects, report.effects);
        assert!(from_file.probe.train_accuracy.is_some());

        // Tampering with the direction fails verification.
        let tampered = model.dir.join("tampered");
        std::fs::create_dir_all(tampered.join("artifacts/probe")).unwrap();
        copy_tree(&path, &tampered);
        let forged = ember::v05::safetensors::serialize(&[ember::v05::safetensors::TensorData {
            name: "direction",
            dtype: ember::v05::safetensors::TensorDType::F32,
            shape: &[64],
            bytes: &[0u8; 256],
        }])
        .unwrap();
        std::fs::write(tampered.join(PROBE_DIRECTION), forged).unwrap();
        assert!(
            !verify_bundle(&tampered, &VerifyOptions::default())
                .unwrap()
                .ok
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
    fn steering_the_last_block_changes_the_target_logit_analytically() {
        // At the last block, steering the prompt-final row changes the final
        // logits by exactly final_norm + LM head of the steered row.
        let model = tiny_model("probe-analytic", 4, 64, false);
        let direction: Vec<f32> = (0..64).map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.5).collect();
        let npy = model.dir.join("d.npy");
        let bytes = ember::v05::steering::write_npy(&[64], &direction);
        std::fs::write(&npy, &bytes).unwrap();
        let text = spec_text(
            &model,
            "reference",
            0,
            &probe_body(
                3,
                &format!(
                    "file = {{ path = {:?}, sha256 = \"{}\" }}",
                    npy.display().to_string(),
                    sha256_hex(&bytes)
                ),
                "",
            ),
        );
        let mut prepared = prepare_run(&resolve(&text), KStrategy::Auto, false).unwrap();
        let (_, report, _) = run(&mut prepared, &model, &text, "analytic");
        assert_eq!(report.probe.train_accuracy, None);
        assert!(report.probe.test_accuracy.is_none());
        // The final residual of each input, from an ordinary capture run.
        let capture_text = spec_text(
            &model,
            "reference",
            0,
            &format!(
                "{INPUTS}\n[[captures]]\nid = \"x\"\nsite = \"residual-post-mlp\"\nlayers = [3]\n[captures.tokens]\nkind = \"prompt-final\"\n"
            ),
        );
        let resolved = resolve(&capture_text);
        let (_, _, _, results) = execute_prepared(
            &mut prepared,
            &resolved,
            &capture_text,
            &model.dir.join("x"),
            false,
            None,
        )
        .unwrap();
        let lens = ember::v05::lens::ModelLens::new(&prepared.model);
        for (index, result) in results.iter().enumerate() {
            let x = result.captures[0].rows.clone();
            let base = ember::v05::lens::LensHead::project(&lens, &x).unwrap()[7];
            let mut steered = x.clone();
            ember::v05::steering::steer_row(
                &mut steered,
                &direction,
                3.0,
                SteerNormalization::Unit,
            )
            .unwrap();
            let after = ember::v05::lens::LensHead::project(&lens, &steered).unwrap()[7];
            let effect = report
                .effects
                .iter()
                .find(|e| e.variant == "steer+3" && e.input_id == result.input.id)
                .unwrap();
            let expected = f64::from(after) - f64::from(base);
            assert!(
                (effect.delta_logit - expected).abs() < 1e-3,
                "input {index}: {} vs {expected}",
                effect.delta_logit
            );
        }
    }

    #[test]
    fn probe_specs_fail_closed() {
        let model = tiny_model("probe-invalid", 2, 64, false);
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
        let one_class = probe_body(
            1,
            "train = [{ text = \"w1\", label = 1 }, { text = \"w2\", label = 1 }]",
            "",
        );
        assert!(parse(one_class).contains("both labels"));
        let overlap = probe_body(
            1,
            "train = [{ text = \"w20 w21 w1\", label = 1 }, { text = \"w2\", label = 0 }]",
            "",
        );
        assert!(parse(overlap).contains("held out"));
        let nothing = probe_body(1, &format!("train = [{}]", examples("t", 4)), "")
            .replace("ablate = true", "ablate = false")
            .replace("steer_alphas = [0.0, 3.0]", "steer_alphas = []");
        assert!(parse(nothing).contains("nothing to measure"));
        let site = probe_body(1, &format!("train = [{}]", examples("t", 4)), "")
            .replace("site = \"residual-post-mlp\"", "site = \"logits\"");
        assert!(parse(site).contains("probe.site"));
    }
}
