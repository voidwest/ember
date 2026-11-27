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
    let points = definition.points(12, 4).unwrap();
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
    assert_eq!(definition.points(12, 4).unwrap()[1].text, points[1].text);
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
    let points = definition.points(12, 4).unwrap();
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
        .points(12, 4)
        .unwrap_err();
    assert!(
        error.message.contains("does not capture layer 8"),
        "{error}"
    );
}

/// SPEC with the swept intervention at `attention-heads`.
fn head_spec(sweep: &str) -> String {
    SPEC.replace(
            "id = \"replace\"\nsite = \"residual-post-mlp\"\nlayers = [7]\noperation = { kind = \"replace\" }\nsource = { kind = \"capture-from-current-run\", capture_id = \"rows\" }",
            "id = \"replace\"\nsite = \"attention-heads\"\nlayers = [7]\noperation = { kind = \"zero\" }",
        )
        .replace(
            "[sweep]\nlayers = { start = 0, end = 12, step = 4 }\ninterventions = [\"replace\"]\n",
            &format!("[sweep]\n{sweep}interventions = [\"replace\"]\n"),
        )
}

/// SPEC with a dictionary at the swept intervention's site and layer
/// (residual-post-mlp, layer 7) and a feature sweep in place of the
/// layer sweep.
fn feature_spec(sweep: &str) -> String {
    let hash = "0".repeat(64);
    SPEC.replace(
        "[output]\n",
        &format!(
            "[[dictionaries]]\nid = \"d\"\npath = \"dict\"\nsha256 = \"{hash}\"\n\
                 config_sha256 = \"{hash}\"\nsite = \"residual-post-mlp\"\nlayer = 7\n\n[output]\n"
        ),
    )
    .replace(
        "[sweep]\nlayers = { start = 0, end = 12, step = 4 }\ninterventions = [\"replace\"]\n",
        &format!("[sweep]\n{sweep}interventions = [\"replace\"]\n"),
    )
}

#[test]
fn feature_sweeps_set_one_feature_per_point() {
    let text = feature_spec("features = { dictionary = \"d\", ids = [40, 3, 40, 12] }\n");
    let definition = SweepDefinition::parse(&text).unwrap();
    let features = definition.features.clone().unwrap();
    assert_eq!(features.ids, vec![3, 12, 40], "ascending, without repeats");
    assert_eq!(definition.point_count(12, 4).unwrap(), Some(3));
    let points = definition.points(12, 4).unwrap();
    let ids: Vec<&str> = points.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, vec!["feature-03", "feature-12", "feature-40"]);
    let point = &points[1];
    assert_eq!(point.feature, Some(12));
    assert_eq!(point.layer, None);
    let find = |id: &str| {
        point
            .resolved
            .interventions
            .iter()
            .find(|i| i.id == id)
            .unwrap()
            .clone()
    };
    let swept = find("replace");
    assert_eq!(
        swept.features,
        Some(crate::v05::dictionary::FeatureSelection {
            dictionary: "d".into(),
            ids: vec![12],
        })
    );
    assert_eq!(
        swept.layers,
        LayerSelector::List(vec![7]),
        "it stays at the dictionary's layer"
    );
    assert_eq!(
        find("fixed").features,
        None,
        "an unswept intervention is unchanged"
    );

    let range = feature_spec(
        "features = { dictionary = \"d\", ids = { start = 0, end = 10, step = 5 } }\n",
    );
    let range = SweepDefinition::parse(&range).unwrap();
    assert_eq!(range.features.unwrap().ids, vec![0, 5]);
}

#[test]
fn feature_sweeps_refuse_what_a_dictionary_cannot_do() {
    let refuse = |text: &str| SweepDefinition::parse(text).unwrap_err().to_string();
    let features = "features = { dictionary = \"d\", ids = [1] }\n";
    assert!(refuse(&feature_spec(
        "features = { dictionary = \"d\", ids = \"all\" }\n"
    ))
    .contains("list the feature ids"));
    assert!(refuse(&feature_spec(
        "features = { dictionary = \"d\", ids = [] }\n"
    ))
    .contains("empty"));
    assert!(
        refuse(&feature_spec(&format!("layers = [7]\n{features}"))).contains("cannot also sweep")
    );
    assert!(refuse(&feature_spec(
        "features = { dictionary = \"x\", ids = [1] }\n"
    ))
    .contains("no [[dictionaries]] entry"));
    let declared = feature_spec(features).replace(
        "source = { kind = \"capture-from-current-run\", capture_id = \"rows\" }",
        "source = { kind = \"capture-from-current-run\", capture_id = \"rows\" }\n\
             features = { dictionary = \"d\", ids = [2] }",
    );
    assert!(refuse(&declared).contains("remove features here"));
    let elsewhere = feature_spec(features).replacen("layers = [7]", "layers = [3]", 1);
    assert!(refuse(&elsewhere).contains("layers = [7]"));
    // A huge range fails before it allocates.
    let huge = format!(
        "features = {{ dictionary = \"d\", ids = {{ start = 0, end = {}, step = 1 }} }}\n",
        i64::MAX
    );
    assert!(refuse(&feature_spec(&huge)).contains("a sweep sets at most 65536"));
    let ceiling = "features = { dictionary = \"d\", ids = { start = 0, end = 65536, step = 1 } }\n";
    assert!(SweepDefinition::parse(&feature_spec(ceiling)).is_ok());
    let steer = feature_spec(features).replace(
        "operation = { kind = \"replace\" }\nsource = { kind = \"capture-from-current-run\", \
             capture_id = \"rows\" }",
        "operation = { kind = \"steer\", alpha = 2.0 }\n\
             source = { kind = \"inline-vector\", values = [1.0] }",
    );
    assert_ne!(steer, feature_spec(features));
    let error = refuse(&steer);
    assert!(error.contains("steer does not act on features"), "{error}");
}

#[test]
fn head_sweeps_set_one_head_per_point() {
    let text = head_spec("layers = [3, 7]\nheads = \"all\"\n");
    assert_ne!(text, SPEC);
    let definition = SweepDefinition::parse(&text).unwrap();
    let points = definition.points(12, 4).unwrap();
    let ids: Vec<&str> = points.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(
        ids[..5],
        [
            "layer-03-head-00",
            "layer-03-head-01",
            "layer-03-head-02",
            "layer-03-head-03",
            "layer-07-head-00"
        ]
    );
    assert_eq!(points.len(), 8);
    for point in &points {
        let swept = point
            .resolved
            .interventions
            .iter()
            .find(|i| i.id == "replace")
            .unwrap();
        assert_eq!(swept.heads, Some(vec![point.head.unwrap()]));
        assert_eq!(
            swept.layers,
            LayerSelector::List(vec![point.layer.unwrap()])
        );
        // The unswept intervention keeps everything.
        let fixed = point
            .resolved
            .interventions
            .iter()
            .find(|i| i.id == "fixed")
            .unwrap();
        assert_eq!(fixed.heads, None);
    }
    // Heads alone keep the declared layer; a list and a range resolve.
    let alone = SweepDefinition::parse(&head_spec("heads = [1, 3]\n")).unwrap();
    let points = alone.points(12, 4).unwrap();
    assert_eq!(
        points.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
        ["head-01", "head-03"]
    );
    assert_eq!(alone.resolve_heads(4).unwrap(), vec![1, 3]);
    let range =
        SweepDefinition::parse(&head_spec("heads = { start = 0, end = 4, step = 2 }\n")).unwrap();
    assert_eq!(range.resolve_heads(4).unwrap(), vec![0, 2]);
    let error = alone.resolve_heads(2).unwrap_err().to_string();
    assert!(
        error.contains("head 3 is out of range for a 2-head model"),
        "{error}"
    );
}

#[test]
fn invalid_head_sweeps_fail_closed() {
    // Only attention-heads interventions have heads to move.
    let error = SweepDefinition::parse(&SPEC.replace(
        "layers = { start = 0, end = 12, step = 4 }",
        "heads = \"all\"",
    ))
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("a head sweep moves attention-heads"),
        "{error}"
    );
    // The sweep sets heads, so the template must not.
    let declared = head_spec("heads = \"all\"\n").replace(
        "site = \"attention-heads\"\nlayers = [7]",
        "site = \"attention-heads\"\nlayers = [7]\nheads = [0]",
    );
    let error = SweepDefinition::parse(&declared).unwrap_err().to_string();
    assert!(error.contains("remove heads here"), "{error}");
    let error = SweepDefinition::parse(&head_spec("heads = []\n"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("the head list is empty"), "{error}");
    // A current-run source capture of selected heads would patch every
    // head from those heads.
    let narrow = head_spec("heads = \"all\"\n")
            .replace(
                "operation = { kind = \"zero\" }",
                "operation = { kind = \"replace\" }\nsource = { kind = \"capture-from-current-run\", capture_id = \"z\" }",
            )
            .replace(
                "[[interventions]]\nid = \"replace\"",
                "[[captures]]\nid = \"z\"\nsite = \"attention-heads\"\nheads = [2]\nlayers = \"all\"\n[captures.tokens]\nkind = \"prompt-final\"\n\n[[interventions]]\nid = \"replace\"",
            );
    let error = SweepDefinition::parse(&narrow).unwrap_err().to_string();
    assert!(error.contains("stores selected heads only"), "{error}");
    let error = SweepDefinition::parse(&head_spec("heads = \"every\"\n"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("expected the string"), "{error}");
}

#[test]
fn bundle_sources_move_only_when_asked() {
    let bundle_spec = SPEC.replace(
        "source = { kind = \"capture-from-current-run\", capture_id = \"rows\" }",
        "source = { kind = \"capture-from-bundle\", bundle_path = \"runs/clean\", \
             capture_id = \"rows\", input_id = \"a\", layer = 7 }",
    );
    assert_ne!(bundle_spec, SPEC);
    let error = SweepDefinition::parse(&bundle_spec)
        .unwrap_err()
        .to_string();
    assert!(error.contains("move_bundle_sources = true"), "{error}");

    let moving = format!("{bundle_spec}move_bundle_sources = true\n");
    let definition = SweepDefinition::parse(&moving).unwrap();
    assert!(definition.move_bundle_sources);
    for point in definition.points(12, 4).unwrap() {
        let layer = point.layer.unwrap();
        let replace = point
            .resolved
            .interventions
            .iter()
            .find(|i| i.id == "replace")
            .unwrap();
        match &replace.source {
            Some(InterventionSource::CaptureFromBundle {
                layer: source_layer,
                ..
            }) => assert_eq!(*source_layer, layer, "{}", point.id),
            other => panic!("unexpected source {other:?}"),
        }
    }

    // Sources do not move with positions, so the two are refused together.
    let positions = format!("{moving}positions = [1, 2]\n");
    let error = SweepDefinition::parse(&positions).unwrap_err().to_string();
    assert!(error.contains("position sweep"), "{error}");

    // The option needs layers and a swept bundle source.
    let no_source = format!("{SPEC}move_bundle_sources = true\n");
    let error = SweepDefinition::parse(&no_source).unwrap_err().to_string();
    assert!(error.contains("no swept intervention reads"), "{error}");
    let no_layers = moving.replace(
        "layers = { start = 0, end = 12, step = 4 }",
        "alphas = [1.0]",
    );
    let error = SweepDefinition::parse(&no_layers).unwrap_err().to_string();
    assert!(error.contains("add `layers`"), "{error}");
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
        heads: None,
        head_count: None,
        features: None,
        points: vec![SweepPointRecord {
            id: "layer-01".into(),
            layer: Some(1),
            position: None,
            alpha: None,
            head: None,
            feature: None,
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
                baseline_metric: None,
                point_metric: None,
                effect: None,
            }],
            effect: None,
        }],
        effect: None,
        sweep_hash: String::new(),
    };
    manifest.sweep_hash = manifest.compute_hash();
    let json = serde_json::to_string(&manifest).unwrap();
    // A sweep without [sweep.effect] serializes exactly as before the
    // effect fields existed, so earlier sweep hashes still verify.
    for key in ["effect", "baseline_metric", "point_metric"] {
        assert!(!json.contains(key), "{key} leaked into {json}");
    }
    for key in ["\"head\"", "\"heads\"", "\"head_count\""] {
        assert!(!json.contains(key), "{key} leaked into {json}");
    }
    let back: SweepManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(back, manifest);
    assert_eq!(back.compute_hash(), manifest.sweep_hash);
    assert!(manifest
        .to_csv()
        .contains("layer-01,1,,a,3,false,0.125,rows,residual-post-mlp,1,1,2,aa\n"));
    assert!(manifest.to_effect_csv().is_none());

    // With an effect record: extra CSV columns and the per-point table.
    let token = |id| crate::v05::attribution::MetricToken {
        token_id: id,
        piece: format!("t{id}"),
    };
    manifest.effect = Some(SweepEffectRecord {
        metric: crate::v05::effect::EFFECT_METRIC.into(),
        interval: crate::v05::effect::EFFECT_INTERVAL.into(),
        capture: "answer".into(),
        confidence: 0.95,
        resamples: 100,
        seed: 0,
        targets: vec![crate::v05::effect::SweepEffectTarget {
            input_id: "a".into(),
            target: token(1),
            foil: token(2),
        }],
    });
    let input = &mut manifest.points[0].inputs[0];
    input.baseline_metric = Some(2.5);
    input.point_metric = Some(1.0);
    input.effect = Some(-1.5);
    manifest.points[0].effect = Some(crate::v05::effect::summarize(&[-1.5], 0.95, 100, 0));
    let json = serde_json::to_string(&manifest).unwrap();
    let back: SweepManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(back, manifest);
    assert!(manifest.to_csv().contains(",aa,2.5,1,-1.5\n"));
    assert_eq!(
        manifest.to_effect_csv().unwrap(),
        "point,layer,position,alpha,n,mean,sd,standard_error,ci_low,ci_high,positive,\
             negative,zero,sign_test_p\nlayer-01,1,,,1,-1.5,,,,,0,1,0,1\n"
    );
    assert!(inside(Path::new("/s"), "../x").is_err());
    assert!(inside(Path::new("/s"), "/abs").is_err());
    assert!(inside(Path::new("/s"), "points/layer-01").is_ok());
}
