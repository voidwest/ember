//! Driver side of feature dictionaries (`ember::v05::dictionary`, ADR-0003):
//! load each dictionary once per session and check it against the model and
//! the feature ids that name it, before the spec's inputs execute.

use crate::cli_experiment::PreparedRun;
use std::sync::Arc;

/// Load the spec's dictionaries (each folder and hash once per session) and
/// check them against the model and the feature ids that name them.
pub(crate) fn load_dictionaries(
    prepared: &PreparedRun,
    resolved: &ember::v05::spec::ExperimentSpecV1,
) -> anyhow::Result<Vec<Arc<ember::v05::dictionary::Dictionary>>> {
    use ember::v05::dictionary::Dictionary;
    let mut out = Vec::new();
    for spec in &resolved.dictionaries {
        let key = (
            spec.path.clone(),
            spec.sha256.clone(),
            spec.config_sha256.clone(),
        );
        let cached = prepared
            .dictionary_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("dictionary cache lock poisoned"))?
            .get(&key)
            .cloned();
        let loaded = match cached {
            Some(loaded) => loaded,
            None => {
                let loaded = Arc::new(Dictionary::load(spec).map_err(anyhow::Error::msg)?);
                prepared
                    .dictionary_cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("dictionary cache lock poisoned"))?
                    .insert(key, Arc::clone(&loaded));
                loaded
            }
        };
        let dictionary = Arc::new(loaded.for_spec(spec));
        if dictionary.d_in() != prepared.embed_dim {
            anyhow::bail!(
                "dictionary '{}' reads {} columns; the model's width is {}",
                spec.id,
                dictionary.d_in(),
                prepared.embed_dim
            );
        }
        if spec.layer >= prepared.n_layers {
            anyhow::bail!(
                "dictionary '{}' is at layer {}; the model has {} layers",
                spec.id,
                spec.layer,
                prepared.n_layers
            );
        }
        for intervention in &resolved.interventions {
            if let Some(features) = &intervention.features
                && features.dictionary == spec.id
            {
                dictionary.check_ids(&features.ids).map_err(|error| {
                    anyhow::anyhow!("intervention '{}': {error}", intervention.id)
                })?;
            }
        }
        out.push(dictionary);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::cli_experiment::{execute_prepared, prepare_run, PreparedRun};
    use crate::experiment_testutil::{
        copy_dir, resolve, resolve_err, spec_text, tiny_model, TinyModel,
    };
    use ember::quant_k::KStrategy;
    use ember::v05::dictionary::{
        Dictionary, DictionarySpec, FeatureArtifact, CONFIG_FILE, WEIGHTS_FILE,
    };
    use ember::v05::hook::SemanticHookSite;
    use ember::v05::runner::InputResult;
    use ember::v05::safetensors::{serialize, TensorDType, TensorData};
    use ember::v05::verify::{load_bundle_for_source, verify_bundle, VerifyOptions};
    use std::path::{Path, PathBuf};

    const EMBED: usize = 64;
    const D_SAE: usize = 16;

    const INPUTS: &str = r#"
[[inputs]]
id = "a"
text = "w3 w17 w5 w40 w9"

[[inputs]]
id = "b"
text = "w8 w1 w33"
"#;

    /// A JumpReLU dictionary for the tiny model's residual stream, written as
    /// a SAELens folder. Returns the folder and the weights' SHA-256.
    fn write_dictionary(dir: &Path, d_in: usize) -> (PathBuf, String) {
        let folder = dir.join("dictionary");
        std::fs::create_dir_all(&folder).unwrap();
        let w_enc: Vec<f32> = (0..d_in * D_SAE)
            .map(|k| (((k / D_SAE) * 7 + (k % D_SAE) * 3) % 11) as f32 * 0.05 - 0.25)
            .collect();
        let b_enc = vec![-0.05f32; D_SAE];
        let w_dec: Vec<f32> = (0..D_SAE * d_in)
            .map(|k| (((k / d_in) * 5 + k % d_in) % 9) as f32 * 0.03 - 0.12)
            .collect();
        let b_dec: Vec<f32> = (0..d_in).map(|j| (j % 3) as f32 * 0.01).collect();
        let threshold = vec![0.02f32; D_SAE];
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
        let weights = serialize(&[
            TensorData {
                name: "W_enc",
                dtype: TensorDType::F32,
                shape: &[d_in, D_SAE],
                bytes: &enc,
            },
            TensorData {
                name: "b_enc",
                dtype: TensorDType::F32,
                shape: &[D_SAE],
                bytes: &benc,
            },
            TensorData {
                name: "W_dec",
                dtype: TensorDType::F32,
                shape: &[D_SAE, d_in],
                bytes: &dec,
            },
            TensorData {
                name: "b_dec",
                dtype: TensorDType::F32,
                shape: &[d_in],
                bytes: &bdec,
            },
            TensorData {
                name: "threshold",
                dtype: TensorDType::F32,
                shape: &[D_SAE],
                bytes: &thr,
            },
        ])
        .unwrap();
        std::fs::write(folder.join(WEIGHTS_FILE), &weights).unwrap();
        std::fs::write(
            folder.join(CONFIG_FILE),
            format!(
                r#"{{"architecture": "jumprelu", "d_in": {d_in}, "d_sae": {D_SAE},
                    "activation_fn": "relu", "apply_b_dec_to_input": false,
                    "normalize_activations": "none", "model_name": "tiny",
                    "hook_name": "blocks.1.hook_resid_post"}}"#
            ),
        )
        .unwrap();
        (folder, ember::v05::manifest::sha256_hex(&weights))
    }

    fn dictionary_section(folder: &Path, sha: &str, layer: usize) -> String {
        format!(
            r#"
[[dictionaries]]
id = "d"
path = "{}"
sha256 = "{sha}"
config_sha256 = "{}"
site = "residual-post-mlp"
layer = {layer}
"#,
            folder.display(),
            ember::v05::manifest::sha256_hex(&std::fs::read(folder.join(CONFIG_FILE)).unwrap())
        )
    }

    /// `post1` is the residual stream leaving block 1 before an edit there
    /// (a capture fires first), with its features; `pre2` is the stream
    /// entering block 2, after the edit.
    fn captures() -> &'static str {
        r#"
[[captures]]
id = "post1"
site = "residual-post-mlp"
layers = [1]
features = { dictionary = "d", top_k = 4 }
[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "pre2"
site = "residual-pre-attention"
layers = [2]
[captures.tokens]
kind = "prompt-final"
"#
    }

    fn edit(operation: &str, ids: &[usize], source: Option<&str>) -> String {
        format!(
            r#"
[[interventions]]
id = "edit"
site = "residual-post-mlp"
layers = [1]
operation = {operation}
features = {{ dictionary = "d", ids = {ids:?} }}
{}
[interventions.tokens]
kind = "prompt-final"
"#,
            source
                .map(|source| format!("source = {source}"))
                .unwrap_or_default()
        )
    }

    fn run(
        prepared: &mut PreparedRun,
        model: &TinyModel,
        text: &str,
        name: &str,
    ) -> anyhow::Result<(PathBuf, Vec<InputResult>)> {
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

    fn load(folder: &Path, sha: &str) -> Dictionary {
        Dictionary::load(&DictionarySpec {
            id: "d".into(),
            path: folder.to_path_buf(),
            sha256: sha.into(),
            config_sha256: ember::v05::manifest::sha256_hex(
                &std::fs::read(folder.join(CONFIG_FILE)).unwrap(),
            ),
            site: SemanticHookSite::ResidualPostMlp,
            layer: 1,
        })
        .unwrap()
    }

    #[test]
    fn a_feature_capture_records_the_strongest_features_of_each_row() {
        let model = tiny_model("dictionary-capture", 4, EMBED, false);
        let (folder, sha) = write_dictionary(&model.dir, EMBED);
        let body = format!(
            "{INPUTS}{}{}",
            dictionary_section(&folder, &sha, 1),
            captures()
        );
        let text = spec_text(&model, "reference", 0, &body);
        let mut prepared = prepare_run(&resolve(&text), KStrategy::Auto, false).unwrap();
        let (bundle, results) = run(&mut prepared, &model, &text, "capture").unwrap();
        let dictionary = load(&folder, &sha);

        let loaded = load_bundle_for_source(&bundle).unwrap();
        let artifact: FeatureArtifact =
            serde_json::from_slice(loaded.file("artifacts/features/post1.json").unwrap()).unwrap();
        assert_eq!(artifact.rows.len(), 2, "one prompt-final row per input");
        for (row, (result, input)) in artifact.rows.iter().zip(results.iter().zip(["a", "b"])) {
            let values = rows(result, "post1", SemanticHookSite::ResidualPostMlp, 1);
            let expected = dictionary.feature_row(&values, 4, input, 1, row.position);
            assert_eq!(*row, expected);
        }
        assert!(
            artifact.rows.iter().any(|row| row.l0 > 0),
            "the synthetic dictionary must have active features to test"
        );
        let record: ember::v05::dictionary::DictionaryRecord =
            serde_json::from_slice(loaded.file("artifacts/dictionaries/d.json").unwrap()).unwrap();
        assert_eq!(record.sha256, sha);
        assert_eq!(
            record.hook_name.as_deref(),
            Some("blocks.1.hook_resid_post")
        );

        // verify catches a changed feature artifact or dictionary record.
        for (file, from, to) in [
            ("artifacts/features/post1.json", "\"l0\": ", "\"l0\": 9"),
            (
                "artifacts/dictionaries/d.json",
                "\"layer\": 1",
                "\"layer\": 2",
            ),
        ] {
            let tampered = model.dir.join(format!("tampered-{}", file.len()));
            copy_dir(&bundle, &tampered);
            let path = tampered.join(file);
            let text = std::fs::read_to_string(&path).unwrap();
            std::fs::write(&path, text.replacen(from, to, 1)).unwrap();
            let report = verify_bundle(&tampered, &VerifyOptions::default()).unwrap();
            assert!(!report.ok, "{file}");
            // Where the check runs, it fails on its own (not only through the
            // file's checksum).
            if let Some(check) = report
                .checks
                .iter()
                .find(|check| check.name == "feature dictionaries")
            {
                assert!(!check.ok, "{file}: {}", check.detail);
            }
        }
        let report = verify_bundle(&bundle, &VerifyOptions::default()).unwrap();
        assert!(report
            .checks
            .iter()
            .any(|check| check.name == "feature dictionaries" && check.ok));
    }

    #[test]
    fn a_feature_edit_adds_only_the_decoded_change() {
        let model = tiny_model("dictionary-edit", 4, EMBED, false);
        let (folder, sha) = write_dictionary(&model.dir, EMBED);
        let head = format!(
            "{INPUTS}{}{}",
            dictionary_section(&folder, &sha, 1),
            captures()
        );
        let base_text = spec_text(&model, "reference", 0, &head);
        let mut prepared = prepare_run(&resolve(&base_text), KStrategy::Auto, false).unwrap();
        let (_, baseline) = run(&mut prepared, &model, &base_text, "baseline").unwrap();
        let dictionary = load(&folder, &sha);
        let ids: Vec<usize> = (0..D_SAE).collect();

        // An edit that changes nothing is bit-identical, downstream included.
        let text = spec_text(
            &model,
            "reference",
            0,
            &format!(
                "{head}{}",
                edit(r#"{ kind = "scale", factor = 1.0 }"#, &ids, None)
            ),
        );
        let (_, same) = run(&mut prepared, &model, &text, "identity").unwrap();
        for (base, result) in baseline.iter().zip(&same) {
            let pre2 = SemanticHookSite::ResidualPreAttention;
            assert_eq!(
                bits(&rows(result, "pre2", pre2, 2)),
                bits(&rows(base, "pre2", pre2, 2))
            );
            assert_eq!(result.generated_token_ids, base.generated_token_ids);
        }

        // Zero every feature, then set feature 3 to 2.5: the row changes by
        // exactly (f' - f) · W_dec, and the reconstruction error stays.
        for (operation, source, after) in [
            (r#"{ kind = "zero" }"#, None, vec![0.0f32; D_SAE]),
            (
                r#"{ kind = "replace" }"#,
                Some(format!("{{ kind = \"inline-vector\", values = {:?} }}", {
                    let mut values = vec![0.0f32; D_SAE];
                    values[3] = 2.5;
                    values
                })),
                {
                    let mut values = vec![0.0f32; D_SAE];
                    values[3] = 2.5;
                    values
                },
            ),
        ] {
            let text = spec_text(
                &model,
                "reference",
                0,
                &format!("{head}{}", edit(operation, &ids, source.as_deref())),
            );
            let (_, edited) = run(
                &mut prepared,
                &model,
                &text,
                operation.len().to_string().as_str(),
            )
            .unwrap();
            for result in &edited {
                let before = rows(result, "post1", SemanticHookSite::ResidualPostMlp, 1);
                let mut expected = before.clone();
                let features = dictionary.encode_selected(&before, &ids);
                dictionary.add_decoded(&mut expected, &ids, &features, &after);
                let actual = rows(result, "pre2", SemanticHookSite::ResidualPreAttention, 2);
                assert_eq!(bits(&actual), bits(&expected), "{operation}");
            }
        }
    }

    #[test]
    fn dictionaries_that_do_not_fit_the_model_or_the_spec_are_refused() {
        let model = tiny_model("dictionary-refuse", 4, EMBED, false);
        let (folder, sha) = write_dictionary(&model.dir, EMBED);
        let section = dictionary_section(&folder, &sha, 1);
        let refuse = |body: &str| {
            let text = spec_text(&model, "reference", 0, &format!("{INPUTS}{body}"));
            let mut prepared = prepare_run(&resolve(&text), KStrategy::Auto, false).unwrap();
            let resolved = resolve(&text);
            execute_prepared(
                &mut prepared,
                &resolved,
                &text,
                &model.dir.join("refused"),
                false,
                None,
            )
            .expect_err("the run is refused")
            .to_string()
        };
        assert!(refuse(&dictionary_section(&folder, &"0".repeat(64), 1)).contains("hashes to"));
        // A second entry for the same files is checked on its own, not taken
        // from the session cache under the first entry's config hash.
        let second = section.replacen("id = \"d\"", "id = \"d2\"", 1).replace(
            &ember::v05::manifest::sha256_hex(&std::fs::read(folder.join(CONFIG_FILE)).unwrap()),
            &"0".repeat(64),
        );
        assert!(refuse(&format!("{section}{second}")).contains("cfg.json hashes to"));
        assert!(refuse(&dictionary_section(&folder, &sha, 4)).contains("layers"));
        let (narrow, narrow_sha) = write_dictionary(&model.dir.join("narrow"), 32);
        assert!(refuse(&dictionary_section(&narrow, &narrow_sha, 1)).contains("width"));
        assert!(refuse(&format!(
            "{section}{}",
            edit(r#"{ kind = "zero" }"#, &[D_SAE], None)
        ))
        .contains("out of range"));

        // The spec itself must name the dictionary's one site and layer.
        let resolve_err = |body: &str| resolve_err(&model, &format!("{INPUTS}{body}"));
        let edit_all_layers = edit(r#"{ kind = "zero" }"#, &[1], None).replace("layers = [1]", "");
        assert!(resolve_err(&format!("{section}{edit_all_layers}")).contains("layers = [1]"));
        let at_mlp = edit(r#"{ kind = "zero" }"#, &[1], None)
            .replace("site = \"residual-post-mlp\"", "site = \"mlp-output\"");
        assert!(resolve_err(&format!("{section}{at_mlp}")).contains("reads residual-post-mlp"));
        let unsorted = edit(r#"{ kind = "zero" }"#, &[3, 1], None);
        assert!(resolve_err(&format!("{section}{unsorted}")).contains("ascending"));
        let steer = edit(
            r#"{ kind = "steer", alpha = 1.0 }"#,
            &[1],
            Some("{ kind = \"inline-vector\", values = [1.0] }"),
        );
        assert!(resolve_err(&format!("{section}{steer}")).contains("does not act on features"));
        let f16 = captures().replacen("layers = [1]", "layers = [1]\ndtype = \"f16\"", 1);
        assert!(resolve_err(&format!("{section}{f16}")).contains("dtype f32"));
        assert!(resolve_err(captures()).contains("no [[dictionaries]] entry"));
    }
}
