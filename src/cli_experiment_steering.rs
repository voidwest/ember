//! Driver side of direction interventions (`ember::v05::steering`): load
//! pinned direction files and compute contrastive directions by running the
//! model over the contrastive prompts, before the spec's inputs execute.

use crate::cli_experiment::{
    activate_spec, new_input_experiment, run_input, DirectionLayers, PreparedRun,
};
use ember::v05::capture::{CaptureDType, CaptureSpec, CaptureStorage, InputSelector};
use ember::v05::intervention::{InterventionSource, InterventionSpec};
use ember::v05::spec::{ExperimentSpecV1, InputSpec};
use ember::v05::steering::{
    directions_for_layers, mean_difference, mean_rows, read_direction_file, ResolvedDirection,
};

/// The layers an intervention acts at (layer 0 for head sites).
fn intervention_layers(
    intervention: &InterventionSpec,
    n_layers: usize,
) -> anyhow::Result<Vec<usize>> {
    if intervention.site.is_per_layer() {
        intervention
            .layers
            .resolve(n_layers)
            .map_err(anyhow::Error::msg)
    } else {
        Ok(vec![0])
    }
}

/// Width of the tensor at the intervention's site.
fn site_columns(prepared: &PreparedRun, intervention: &InterventionSpec) -> usize {
    if intervention.site == ember::v05::hook::SemanticHookSite::Logits {
        prepared.model.config.vocab_size
    } else {
        prepared.embed_dim
    }
}

/// Resolve every `vector-file` and `contrastive` direction of `resolved`.
pub(crate) fn resolve_directions(
    prepared: &PreparedRun,
    resolved: &ExperimentSpecV1,
) -> anyhow::Result<Vec<ResolvedDirection>> {
    let mut out = Vec::new();
    for intervention in &resolved.interventions {
        let Some(source) = intervention
            .source
            .as_ref()
            .filter(|source| source.is_resolved_direction())
        else {
            continue;
        };
        let layers = intervention_layers(intervention, prepared.n_layers)?;
        let columns = site_columns(prepared, intervention);
        let vectors = match source {
            InterventionSource::VectorFile {
                path,
                sha256,
                tensor,
            } => {
                let matrix =
                    read_direction_file(path, sha256, tensor.as_deref()).map_err(|error| {
                        anyhow::anyhow!("intervention '{}': {error}", intervention.id)
                    })?;
                directions_for_layers(
                    &matrix,
                    &layers,
                    prepared.n_layers,
                    columns,
                    intervention.site.is_per_layer(),
                )
                .map_err(|error| anyhow::anyhow!("intervention '{}': {error}", intervention.id))?
            }
            InterventionSource::Contrastive { .. } => {
                contrastive_direction(prepared, resolved, intervention, source)?
            }
            _ => unreachable!("filtered to resolved directions"),
        };
        out.push(ResolvedDirection {
            intervention_id: intervention.id.clone(),
            site: intervention.site,
            source_kind: source.kind_name().to_string(),
            layers: vectors,
        });
    }
    Ok(out)
}

/// `mean(capture | positive) - mean(capture | negative)` at the
/// intervention's site and layers, from capture-only runs of the prompts
/// under the spec's model, execution mode and thread count.
fn contrastive_direction(
    prepared: &PreparedRun,
    resolved: &ExperimentSpecV1,
    intervention: &InterventionSpec,
    source: &InterventionSource,
) -> anyhow::Result<DirectionLayers> {
    let InterventionSource::Contrastive {
        positive,
        negative,
        tokens,
    } = source
    else {
        anyhow::bail!("not a contrastive source");
    };
    let key = serde_json::to_string(&serde_json::json!({
        "site": intervention.site,
        "layers": intervention_layers(intervention, prepared.n_layers)?,
        "source": source,
        "mode": resolved.execution.mode.name(),
        "threads": resolved.execution.threads,
        "model": prepared.model_sha,
    }))?;
    if let Some(cached) = prepared
        .direction_cache
        .lock()
        .expect("direction cache lock")
        .get(&key)
    {
        return Ok(cached.clone());
    }
    let means = prompt_means(
        prepared,
        resolved,
        intervention.site,
        &intervention.layers,
        tokens,
        positive.iter().chain(negative.iter()),
        &format!("intervention '{}'", intervention.id),
    )?;
    let (positive_means, negative_means) = means.split_at(positive.len());
    let layers = intervention_layers(intervention, prepared.n_layers)?;
    let mut out = DirectionLayers::new();
    for layer in layers {
        let collect = |means: &[std::collections::BTreeMap<usize, Vec<f64>>]| {
            means
                .iter()
                .map(|by_layer| by_layer.get(&layer).cloned())
                .collect::<Option<Vec<Vec<f64>>>>()
        };
        let (Some(a), Some(b)) = (collect(positive_means), collect(negative_means)) else {
            anyhow::bail!(
                "intervention '{}': a contrastive prompt recorded no row at layer {layer}",
                intervention.id
            );
        };
        let direction = mean_difference(&a, &b)
            .map_err(|error| anyhow::anyhow!("intervention '{}': {error}", intervention.id))?;
        out.insert(layer, direction);
    }
    prepared
        .direction_cache
        .lock()
        .expect("direction cache lock")
        .insert(key, out.clone());
    Ok(out)
}

/// Per-prompt, per-layer mean of the rows `tokens` selects at `site` and
/// `layers`, from a capture-only prefill of each prompt (`label` names the
/// caller in errors).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prompt_means<'a>(
    prepared: &PreparedRun,
    resolved: &ExperimentSpecV1,
    site: ember::v05::hook::SemanticHookSite,
    layers: &ember::v05::capture::LayerSelector,
    tokens: &ember::v05::token_select::TokenSelector,
    prompts: impl Iterator<Item = &'a String>,
    label: &str,
) -> anyhow::Result<Vec<std::collections::BTreeMap<usize, Vec<f64>>>> {
    let mut derived = resolved.clone();
    derived.inputs = prompts
        .enumerate()
        .map(|(index, text)| InputSpec {
            id: format!("prompt-{index:04}"),
            text: text.clone(),
        })
        .collect();
    derived.captures = vec![CaptureSpec {
        id: "direction".into(),
        site,
        layers: if site.is_per_layer() {
            layers.clone()
        } else {
            ember::v05::capture::LayerSelector::All("all".into())
        },
        tokens: tokens.clone(),
        inputs: InputSelector::All("all".into()),
        storage: CaptureStorage::SelectedRows,
        dtype: CaptureDType::F32,
    }];
    derived.interventions = Vec::new();
    derived.attribution = None;
    derived.probe = None;
    derived.generation.max_new_tokens = 0;
    let active = activate_spec(prepared, &derived)?;
    let mut out = Vec::with_capacity(derived.inputs.len());
    for index in 0..derived.inputs.len() {
        let experiment = new_input_experiment(prepared, &derived, Some(&active), index)?;
        let result = run_input(prepared, &derived, &active, &experiment, None, None, None)?;
        let mut by_layer = std::collections::BTreeMap::new();
        for capture in &result.captures {
            let mean = mean_rows(&capture.rows, capture.columns).map_err(|error| {
                anyhow::anyhow!("{label}: prompt {}: {error}", derived.inputs[index].id)
            })?;
            by_layer.insert(capture.layer, mean);
        }
        out.push(by_layer);
    }
    Ok(out)
}

/// `experiment validate`: read, hash-check and parse every direction file
/// the spec pins (the dimensions are checked against the model at run
/// time). Returns how many were checked.
pub(crate) fn check_direction_files(resolved: &ExperimentSpecV1) -> anyhow::Result<usize> {
    let mut checked = 0;
    for (index, intervention) in resolved.interventions.iter().enumerate() {
        if let Some(InterventionSource::VectorFile {
            path,
            sha256,
            tensor,
        }) = &intervention.source
        {
            read_direction_file(path, sha256, tensor.as_deref())
                .map_err(|error| anyhow::anyhow!("interventions[{index}].source: {error}"))?;
            checked += 1;
        }
    }
    Ok(checked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_experiment::{execute_prepared, prepare_run};
    use crate::experiment_testutil::{resolve, spec_text, tiny_model, TinyModel};
    use ember::quant_k::KStrategy;
    use ember::v05::hook::SemanticHookSite;
    use ember::v05::intervention::SteerNormalization;
    use ember::v05::runner::InputResult;
    use ember::v05::steering::steer_row;
    use ember::v05::verify::{load_bundle_for_source, verify_bundle, VerifyOptions};

    const INPUTS: &str = r#"
[[inputs]]
id = "a"
text = "w3 w17 w5 w40 w9"

[[inputs]]
id = "b"
text = "w8 w1 w33"
"#;

    /// The residual stream leaving block 1 (pre-intervention, since a
    /// capture fires before an intervention at the same site), entering
    /// block 2 (post-intervention), the last block's output and the logits.
    const CAPTURES: &str = r#"
[[captures]]
id = "post1"
site = "residual-post-mlp"
layers = [1, 3]
[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "pre2"
site = "residual-pre-attention"
layers = [2]
[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "logits"
site = "logits"
[captures.tokens]
kind = "prompt-final"
"#;

    fn direction() -> Vec<f32> {
        (0..64)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.125)
            .collect()
    }

    fn steer_body(site: &str, layer: usize, alpha: f32, normalize: &str, source: &str) -> String {
        format!(
            r#"{INPUTS}{CAPTURES}
[[interventions]]
id = "steer"
site = "{site}"
layers = [{layer}]
operation = {{ kind = "steer", alpha = {alpha:?}, normalize = "{normalize}" }}
source = {source}
[interventions.tokens]
kind = "prompt-final"
"#
        )
    }

    fn inline_source() -> String {
        format!("{{ kind = \"inline-vector\", values = {:?} }}", direction())
    }

    fn run(
        prepared: &mut crate::cli_experiment::PreparedRun,
        model: &TinyModel,
        text: &str,
        name: &str,
    ) -> anyhow::Result<(std::path::PathBuf, Vec<InputResult>)> {
        let resolved = resolve(text);
        let out = model.dir.join(name);
        let (path, _, report, results) =
            execute_prepared(prepared, &resolved, text, &out, false, None)?;
        assert!(report.ok, "{:?}", report.checks);
        Ok((path, results))
    }

    fn rows(result: &InputResult, capture: &str, site: SemanticHookSite, layer: usize) -> Vec<f32> {
        result
            .captures
            .iter()
            .find(|c| c.capture_id == capture && c.site == site && c.layer == layer)
            .unwrap_or_else(|| panic!("capture {capture} at {site} {layer}"))
            .rows
            .clone()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn steering_adds_exactly_alpha_times_the_direction() {
        let model = tiny_model("steer-analytic", 4, 64, false);
        let base_text = spec_text(&model, "reference", 0, &format!("{INPUTS}{CAPTURES}"));
        let mut prepared = prepare_run(&resolve(&base_text), KStrategy::Auto, false).unwrap();
        let (_, baseline) = run(&mut prepared, &model, &base_text, "baseline").unwrap();
        for normalize in [
            SteerNormalization::None,
            SteerNormalization::Unit,
            SteerNormalization::MatchResidualNorm,
        ] {
            let name = serde_json::to_value(normalize).unwrap();
            let name = name.as_str().unwrap();
            let text = spec_text(
                &model,
                "reference",
                0,
                &steer_body("residual-post-mlp", 1, 2.5, name, &inline_source()),
            );
            let (_, steered) = run(&mut prepared, &model, &text, name).unwrap();
            for (base, result) in baseline.iter().zip(&steered) {
                // Upstream of the change nothing moves.
                let before = rows(result, "post1", SemanticHookSite::ResidualPostMlp, 1);
                assert_eq!(
                    bits(&before),
                    bits(&rows(base, "post1", SemanticHookSite::ResidualPostMlp, 1))
                );
                // The stream entering block 2 is exactly x + alpha * c * d.
                let mut expected = before.clone();
                steer_row(&mut expected, &direction(), 2.5, normalize).unwrap();
                let after = rows(result, "pre2", SemanticHookSite::ResidualPreAttention, 2);
                assert_eq!(bits(&after), bits(&expected), "{name}");
                assert_ne!(bits(&after), bits(&before));
            }
        }

        // At the last block the effect on the logits is analytic: the final
        // norm and LM head applied to the steered row.
        let text = spec_text(
            &model,
            "reference",
            0,
            &steer_body("residual-post-mlp", 3, -1.5, "none", &inline_source()),
        );
        let (_, steered) = run(&mut prepared, &model, &text, "last-block").unwrap();
        let lens = ember::v05::lens::ModelLens::new(&prepared.model);
        for result in &steered {
            let mut row = rows(result, "post1", SemanticHookSite::ResidualPostMlp, 3);
            steer_row(&mut row, &direction(), -1.5, SteerNormalization::None).unwrap();
            let expected = ember::v05::lens::LensHead::project(&lens, &row).unwrap();
            let logits = rows(result, "logits", SemanticHookSite::Logits, 0);
            let worst = expected
                .iter()
                .zip(&logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 1e-4,
                "logits differ from the analytic value by {worst}"
            );
        }
    }

    #[test]
    fn alpha_zero_is_bit_identical_to_the_baseline() {
        let model = tiny_model("steer-alpha0", 4, 64, true);
        for mode in ["reference", "planned", "planned-fused"] {
            let base_text = spec_text(&model, mode, 3, &format!("{INPUTS}{CAPTURES}"));
            let mut prepared = prepare_run(&resolve(&base_text), KStrategy::Auto, false).unwrap();
            let (_, baseline) =
                run(&mut prepared, &model, &base_text, &format!("{mode}-base")).unwrap();
            for normalize in ["none", "unit", "match-residual-norm"] {
                let text = spec_text(
                    &model,
                    mode,
                    3,
                    &steer_body("residual-post-mlp", 1, 0.0, normalize, &inline_source()),
                );
                let (_, zero) =
                    run(&mut prepared, &model, &text, &format!("{mode}-{normalize}")).unwrap();
                for (a, b) in baseline.iter().zip(&zero) {
                    assert_eq!(a.generated_token_ids, b.generated_token_ids, "{mode}");
                    assert_eq!(a.captures.len(), b.captures.len());
                    for (x, y) in a.captures.iter().zip(&b.captures) {
                        assert_eq!(x.capture_id, y.capture_id);
                        assert_eq!(bits(&x.rows), bits(&y.rows), "{mode} {}", x.capture_id);
                    }
                    assert_eq!(b.events.len(), 1);
                }
            }
        }
    }

    fn file_source(path: &std::path::Path, sha: &str) -> String {
        format!(
            "{{ kind = \"vector-file\", path = {:?}, sha256 = \"{sha}\" }}",
            path.display().to_string()
        )
    }

    #[test]
    fn vector_files_are_pinned_checked_and_bundled() {
        let model = tiny_model("steer-file", 4, 64, false);
        let npy = model.dir.join("direction.npy");
        let bytes = ember::v05::steering::write_npy(&[64], &direction());
        std::fs::write(&npy, &bytes).unwrap();
        let sha = ember::v05::manifest::sha256_hex(&bytes);
        let text = spec_text(
            &model,
            "reference",
            1,
            &steer_body(
                "residual-post-mlp",
                1,
                2.5,
                "none",
                &file_source(&npy, &sha),
            ),
        );
        let resolved = resolve(&text);
        assert_eq!(check_direction_files(&resolved).unwrap(), 1);
        let mut prepared = prepare_run(&resolved, KStrategy::Auto, false).unwrap();
        let (path, from_file) = run(&mut prepared, &model, &text, "from-file").unwrap();
        // The same direction inline gives the same rows.
        let inline_text = spec_text(
            &model,
            "reference",
            1,
            &steer_body("residual-post-mlp", 1, 2.5, "none", &inline_source()),
        );
        let (_, inline) = run(&mut prepared, &model, &inline_text, "inline").unwrap();
        for (a, b) in from_file.iter().zip(&inline) {
            for (x, y) in a.captures.iter().zip(&b.captures) {
                assert_eq!(bits(&x.rows), bits(&y.rows));
            }
        }
        // The bundle carries the direction and verifies it.
        let bundle = load_bundle_for_source(&path).unwrap();
        let tensors = ember::v05::steering::read_direction_tensors(
            bundle
                .file("artifacts/directions/steer.safetensors")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(tensors[&1], direction());
        let report = verify_bundle(&path, &VerifyOptions::default()).unwrap();
        assert!(report
            .checks
            .iter()
            .any(|check| check.name == "direction artifacts" && check.ok));
        // Tampering with the artifact fails verification.
        let tampered = model.dir.join("tampered");
        copy_dir(&path, &tampered);
        let record = tampered.join("artifacts/directions/steer.json");
        let edited = std::fs::read_to_string(&record)
            .unwrap()
            .replace("\"layer\": 1", "\"layer\": 2");
        std::fs::write(&record, edited).unwrap();
        assert!(
            !verify_bundle(&tampered, &VerifyOptions::default())
                .unwrap()
                .ok
        );

        // A safetensors file with a per-layer matrix: row L at layer L.
        let per_layer: Vec<f32> = (0..4)
            .flat_map(|layer| direction().into_iter().map(move |v| v * layer as f32))
            .collect();
        let st_payload: Vec<u8> = per_layer.iter().flat_map(|v| v.to_le_bytes()).collect();
        let st_bytes = ember::v05::safetensors::serialize(&[ember::v05::safetensors::TensorData {
            name: "probe",
            dtype: ember::v05::safetensors::TensorDType::F32,
            shape: &[4, 64],
            bytes: &st_payload,
        }])
        .unwrap();
        let st = model.dir.join("direction.safetensors");
        std::fs::write(&st, &st_bytes).unwrap();
        let st_sha = ember::v05::manifest::sha256_hex(&st_bytes);
        let st_text = spec_text(
            &model,
            "reference",
            0,
            &steer_body(
                "residual-post-mlp",
                2,
                1.0,
                "none",
                &file_source(&st, &st_sha),
            ),
        );
        let (path, _) = run(&mut prepared, &model, &st_text, "safetensors").unwrap();
        let bundle = load_bundle_for_source(&path).unwrap();
        let tensors = ember::v05::steering::read_direction_tensors(
            bundle
                .file("artifacts/directions/steer.safetensors")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(tensors[&2], per_layer[128..192].to_vec());

        // Fail closed: a wrong pin, a wrong width, a non-direction op.
        let wrong_sha = spec_text(
            &model,
            "reference",
            0,
            &steer_body(
                "residual-post-mlp",
                1,
                1.0,
                "none",
                &file_source(&npy, &"0".repeat(64)),
            ),
        );
        let error = check_direction_files(&resolve(&wrong_sha)).unwrap_err();
        assert!(error.to_string().contains("hashes to"), "{error}");
        let narrow = model.dir.join("narrow.npy");
        let narrow_bytes = ember::v05::steering::write_npy(&[32], &[1.0; 32]);
        std::fs::write(&narrow, &narrow_bytes).unwrap();
        let narrow_text = spec_text(
            &model,
            "reference",
            0,
            &steer_body(
                "residual-post-mlp",
                1,
                1.0,
                "none",
                &file_source(&narrow, &ember::v05::manifest::sha256_hex(&narrow_bytes)),
            ),
        );
        let error = run(&mut prepared, &model, &narrow_text, "narrow").unwrap_err();
        assert!(
            format!("{error:#}").contains("dimension mismatch"),
            "{error:#}"
        );
        let add = steer_body(
            "residual-post-mlp",
            1,
            1.0,
            "none",
            &file_source(&npy, &sha),
        )
        .replace(
            "operation = { kind = \"steer\", alpha = 1.0, normalize = \"none\" }",
            "operation = { kind = \"add-delta\" }",
        );
        let error = ember::v05::spec::RawExperimentSpec::from_toml_str(&spec_text(
            &model,
            "reference",
            0,
            &add,
        ))
        .unwrap()
        .resolve()
        .unwrap_err();
        assert!(error.message.contains("direction operation"), "{error}");
    }

    fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    #[test]
    fn contrastive_direction_is_the_difference_of_prompt_means() {
        let model = tiny_model("steer-contrast", 4, 64, false);
        let source = "{ kind = \"contrastive\", positive = [\"w1 w2 w3\", \"w4 w5\"], negative = [\"w9 w8 w7\"] }";
        let text = spec_text(
            &model,
            "reference",
            0,
            &steer_body("residual-post-mlp", 1, 1.0, "none", source),
        );
        let mut prepared = prepare_run(&resolve(&text), KStrategy::Auto, false).unwrap();
        let (path, _) = run(&mut prepared, &model, &text, "contrastive").unwrap();
        // The prompts' own captures, from an ordinary run.
        let prompts = format!(
            r#"
[[inputs]]
id = "p0"
text = "w1 w2 w3"
[[inputs]]
id = "p1"
text = "w4 w5"
[[inputs]]
id = "n0"
text = "w9 w8 w7"
{CAPTURES}"#
        );
        let (_, results) = run(
            &mut prepared,
            &model,
            &spec_text(&model, "reference", 0, &prompts),
            "prompts",
        )
        .unwrap();
        let row = |index: usize| -> Vec<f64> {
            rows(
                &results[index],
                "post1",
                SemanticHookSite::ResidualPostMlp,
                1,
            )
            .into_iter()
            .map(f64::from)
            .collect()
        };
        let expected: Vec<f32> = (0..64)
            .map(|i| ((row(0)[i] + row(1)[i]) / 2.0 - row(2)[i]) as f32)
            .collect();
        let bundle = load_bundle_for_source(&path).unwrap();
        let tensors = ember::v05::steering::read_direction_tensors(
            bundle
                .file("artifacts/directions/steer.safetensors")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(bits(&tensors[&1]), bits(&expected));
        // The record names the prompts by hash.
        let record: ember::v05::steering::DirectionRecord =
            serde_json::from_slice(bundle.file("artifacts/directions/steer.json").unwrap())
                .unwrap();
        assert_eq!(record.source_kind, "contrastive");
        assert_eq!(record.positive_prompt_sha256.len(), 2);
        // The same spec again, in a fresh session: identical identity.
        let resolved = resolve(&text);
        let (_, first, _, _) = execute_prepared(
            &mut prepared,
            &resolved,
            &text,
            &model.dir.join("again-1"),
            false,
            None,
        )
        .unwrap();
        let mut fresh = prepare_run(&resolved, KStrategy::Auto, false).unwrap();
        let (_, second, _, _) = execute_prepared(
            &mut fresh,
            &resolved,
            &text,
            &model.dir.join("again-2"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn alpha_sweeps_cross_layers_and_verify() {
        let model = tiny_model("steer-sweep", 4, 64, true);
        let body = steer_body("residual-post-mlp", 1, 1.0, "unit", &inline_source());
        let text = format!(
            "{}\n[sweep]\nlayers = [1, 2]\nalphas = [4.0, 0.0]\n",
            spec_text(&model, "reference", 2, &body)
        );
        let spec = model.dir.join("sweep.toml");
        std::fs::write(&spec, &text).unwrap();
        let out = model.dir.join("sweep-out");
        let args = crate::cli_experiment::RunArgs {
            spec: spec.clone(),
            execution: None,
            threads: None,
            output: Some(out.clone()),
            retain_incomplete: false,
            variants: Vec::new(),
            sign_key: None,
            no_sign: true,
            json: false,
        };
        crate::cli_experiment_sweep::run_sweep(&args, &text, KStrategy::Auto, false).unwrap();
        let report = ember::v05::sweep::verify_sweep(&out, &VerifyOptions::default()).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        let manifest = ember::v05::sweep::read_manifest(&out).unwrap();
        let ids: Vec<&str> = manifest.points.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "layer-01-alpha-0",
                "layer-01-alpha-4",
                "layer-02-alpha-0",
                "layer-02-alpha-4"
            ]
        );
        for point in &manifest.points {
            let unchanged = point.inputs.iter().all(|input| {
                input.first_divergent_step.is_none()
                    && input.captures_exact == input.captures_compared
            });
            assert_eq!(unchanged, point.alpha == Some(0.0), "{}", point.id);
        }
        assert!(std::fs::read_to_string(out.join("sweep.csv"))
            .unwrap()
            .starts_with("point,layer,position,alpha,"));

        // Alpha alone: the intervention stays at its declared layer.
        let alone = format!(
            "{}\n[sweep]\nalphas = [-2.0, 2.0]\n",
            spec_text(&model, "reference", 2, &body)
        );
        let definition = ember::v05::sweep::SweepDefinition::parse(&alone).unwrap();
        let points = definition.points(4).unwrap();
        assert_eq!(points[0].id, "alpha-neg-2");
        assert_eq!(
            points[0].resolved.interventions[0].operation,
            ember::v05::intervention::InterventionOperation::Steer {
                alpha: -2.0,
                normalize: SteerNormalization::Unit
            }
        );
        // An alpha sweep needs operations that have an alpha.
        let zero = body
            .replace(
                "operation = { kind = \"steer\", alpha = 1.0, normalize = \"unit\" }",
                "operation = { kind = \"zero\" }",
            )
            .replace(&format!("source = {}\n", inline_source()), "");
        let error = ember::v05::sweep::SweepDefinition::parse(&format!(
            "{}\n[sweep]\nalphas = [1.0]\n",
            spec_text(&model, "reference", 2, &zero)
        ))
        .unwrap_err();
        assert!(error.message.contains("alpha"), "{error}");
    }
}
