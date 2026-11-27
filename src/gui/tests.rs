use super::server::{
    discover_models_in, host_is_loopback, origin_is_loopback_or_absent,
    origin_matches_host_or_absent, security_gate, GateInput, SlotLimiter,
};
use super::*;

#[test]
fn model_discovery_skips_hidden_library_and_build_directories() {
    let root = std::env::temp_dir().join(format!("ember-discover-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for dir in [
        "models/llama",
        "Library/Caches",
        ".hidden",
        "target/debug",
        "node_modules/x",
        ".git/objects",
    ] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
        std::fs::write(root.join(dir).join("m.gguf"), b"").unwrap();
    }
    let found = discover_models_in(&root);
    assert_eq!(found.len(), 1, "found {found:?}");
    assert!(found[0].ends_with("models/llama/m.gguf"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn request_limiter_reserves_a_slot_before_spawning() {
    use std::sync::mpsc;
    use std::time::Duration;

    let limiter = Arc::new(SlotLimiter::new());
    let mut occupied: Vec<_> = (0..8).map(|_| limiter.acquire()).collect();
    let (started_tx, started_rx) = mpsc::channel();
    let (spawned_tx, spawned_rx) = mpsc::channel();
    let producer_limiter = Arc::clone(&limiter);
    let producer = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        let worker = producer_limiter.spawn(|| {});
        spawned_tx.send(worker).unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let blocked = matches!(
        spawned_rx.recv_timeout(Duration::from_millis(50)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
    assert_eq!(*limiter.active.lock().unwrap(), 8);
    drop(occupied.pop());
    if blocked {
        spawned_rx
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .join()
            .unwrap();
    }
    producer.join().unwrap();
    drop(occupied);
    assert_eq!(*limiter.active.lock().unwrap(), 0);
    assert!(
        blocked,
        "spawning must wait until a worker slot is available"
    );
}

#[test]
fn request_limiter_releases_a_slot_when_a_worker_panics() {
    let limiter = Arc::new(SlotLimiter::new());
    assert!(limiter
        .spawn(|| panic!("test handler panic"))
        .join()
        .is_err());
    assert_eq!(*limiter.active.lock().unwrap(), 0);
    limiter.spawn(|| {}).join().unwrap();
    assert_eq!(*limiter.active.lock().unwrap(), 0);
}

#[test]
fn host_allowlist_accepts_loopback_forms_only() {
    for ok in [
        "localhost",
        "localhost:8337",
        "127.0.0.1",
        "127.0.0.1:8337",
        "[::1]:8337",
        "::1",
    ] {
        assert!(host_is_loopback(ok), "should accept {ok:?}");
    }
    for bad in [
        "evil.example",
        "evil.example:8337",
        "192.168.1.10:8337",
        "127.0.0.1.evil.example",
    ] {
        assert!(!host_is_loopback(bad), "should reject {bad:?}");
    }
}

#[test]
fn cross_origin_posts_are_identified() {
    assert!(origin_is_loopback_or_absent(None), "no Origin is allowed");
    for ok in [
        "http://localhost:8337",
        "http://127.0.0.1:8337",
        "http://[::1]:8337",
    ] {
        assert!(origin_is_loopback_or_absent(Some(ok)), "accept {ok:?}");
    }
    for bad in [
        "https://evil.example",
        "http://evil.example:8337",
        "http://127.0.0.1.evil.example",
    ] {
        assert!(!origin_is_loopback_or_absent(Some(bad)), "reject {bad:?}");
    }
}

#[test]
fn network_origin_must_match_the_served_host() {
    let host = Some("192.168.1.10:8337");
    assert!(origin_matches_host_or_absent(None, host), "no Origin");
    assert!(origin_matches_host_or_absent(
        Some("http://192.168.1.10:8337"),
        host
    ));
    assert!(origin_matches_host_or_absent(
        Some("http://Lab-Box.local:8337"),
        Some("lab-box.local:8337")
    ));
    for bad in [
        "https://evil.example",
        "http://192.168.1.10:9000",
        "http://192.168.1.10:8337.evil.example",
        "null",
    ] {
        assert!(
            !origin_matches_host_or_absent(Some(bad), host),
            "reject {bad:?}"
        );
    }
    assert!(
        !origin_matches_host_or_absent(Some("http://192.168.1.10:8337"), None),
        "an Origin with no Host to compare against is refused"
    );
}

fn gate<'a>(
    is_root: bool,
    is_post: bool,
    host: &'a str,
    origin: Option<&'a str>,
    token: Option<&'a str>,
) -> GateInput<'a> {
    GateInput {
        is_root,
        is_post,
        host: Some(host),
        origin,
        token,
    }
}

#[test]
fn loopback_gate_pins_host_token_and_origin() {
    let local = "127.0.0.1:8337";
    assert_eq!(
        security_gate(true, "tok", &gate(true, false, local, None, None)),
        Ok(()),
        "the page itself needs no token"
    );
    assert_eq!(
        security_gate(true, "tok", &gate(true, false, "evil.example", None, None)),
        Err("forbidden: bad Host header"),
        "a rebound Host cannot even read the page"
    );
    assert_eq!(
        security_gate(true, "tok", &gate(false, false, local, None, None)),
        Err("forbidden: missing or invalid token")
    );
    assert_eq!(
        security_gate(true, "tok", &gate(false, true, local, None, Some("tok"))),
        Ok(())
    );
    assert_eq!(
        security_gate(
            true,
            "tok",
            &gate(
                false,
                true,
                local,
                Some("https://evil.example"),
                Some("tok")
            )
        ),
        Err("forbidden: cross-origin request")
    );
}

#[test]
fn network_gate_still_requires_the_token_and_same_origin_posts() {
    let lan = "192.168.1.10:8337";
    assert_eq!(
        security_gate(false, "tok", &gate(true, false, lan, None, None)),
        Ok(()),
        "a network bind serves the page to non-loopback hosts"
    );
    for token in [None, Some("wrong")] {
        assert_eq!(
            security_gate(false, "tok", &gate(false, true, lan, None, token)),
            Err("forbidden: missing or invalid token"),
            "a network bind must not drop the token check ({token:?})"
        );
    }
    assert_eq!(
        security_gate(
            false,
            "tok",
            &gate(
                false,
                true,
                lan,
                Some("http://192.168.1.10:8337"),
                Some("tok")
            )
        ),
        Ok(())
    );
    assert_eq!(
        security_gate(
            false,
            "tok",
            &gate(false, true, lan, Some("https://evil.example"), Some("tok"))
        ),
        Err("forbidden: cross-origin request")
    );
    assert_eq!(
        security_gate(false, "tok", &gate(false, false, lan, None, Some("tok"))),
        Ok(()),
        "GETs carry no Origin requirement"
    );
}

fn base_request() -> RunRequest {
    RunRequest {
        model_path: "model.gguf".to_string(),
        prompt: "\u{0627}\u{0643}\u{062a}\u{0628} \u{062c}\u{0645}\u{0644}\u{0629}".to_string(),
        max_new_tokens: 16,
        execution: "reference".to_string(),
        site: "after-mlp".to_string(),
        layer: Some(7),
        operation: "scale".to_string(),
        factor: Some(0.5),
        alpha: None,
        source: "capture".to_string(),
        source_layer: None,
        token_kind: "prompt-final".to_string(),
        span_text: None,
    }
}

fn cfg_of(req: &RunRequest) -> RunConfig {
    parse_run_request(req).expect("request parses")
}

/// Two bundles with equal computed content (their spec texts name
/// different output directories, so their hashes differ).
fn assert_same_content(a: &str, b: &str) {
    let result = ember::v05::compare::compare_bundles(Path::new(a), Path::new(b)).unwrap();
    assert!(result
        .outputs
        .iter()
        .all(|output| output.generated_tokens_equal && output.final_top1_equal));
    assert!(!result.captures.is_empty());
    assert!(result.captures.iter().all(|capture| capture
        .metrics
        .as_ref()
        .is_some_and(|metrics| metrics.exact)));
}

/// Sweep points (one shared pass, then a resumed intervention per
/// layer) produce exactly what separate pairs produce, on a synthetic
/// model sized for the repository tokenizer.
#[test]
fn sweep_points_equal_separate_pairs() {
    let model =
        crate::experiment_testutil::tiny_model_with_vocab("gui-sweep", 4, 64, true, 128_256);
    let mut req = base_request();
    req.model_path = model.model.display().to_string();
    req.prompt = "The capital of France is".to_string();
    req.max_new_tokens = 3;
    req.site = "after-mlp".to_string();
    req.operation = "scale".to_string();
    req.factor = Some(-2.0);
    let mut session = GuiSession::new(KStrategy::Auto, false);
    session.output_root = model.dir.join("runs").display().to_string();
    let layers = [0usize, 1, 2, 3];
    for (index, &layer) in layers.iter().enumerate() {
        req.layer = Some(layer);
        let cfg = cfg_of(&req);
        let point = session.run_sweep_point(&cfg, &layers[index..]).unwrap();
        assert!(point.verification.ok);
        // After the first point the pass is cached for the rest.
        assert_eq!(
            session.sweep_cache.is_some(),
            index + 1 < layers.len(),
            "layer {layer}"
        );
        let pair = session.run_baseline_intervention(&cfg).unwrap();
        assert_same_content(&point.baseline.bundle_dir, &pair.baseline.bundle_dir);
        assert_same_content(
            &point.intervention.bundle_dir,
            &pair.intervention.bundle_dir,
        );
        assert_eq!(point.comparison, pair.comparison, "layer {layer}");
        assert_eq!(point.baseline_key, pair.baseline_key);
        // Both decode their runs in one batch; standalone runs decode
        // alone and must compute exactly the same bundles.
        let (baseline, _) = run_one(&mut session, &cfg, RunKind::Baseline).unwrap();
        let (intervention, _) = run_one(&mut session, &cfg, RunKind::Intervention).unwrap();
        assert_same_content(&point.baseline.bundle_dir, &baseline.bundle_dir);
        assert_same_content(&point.intervention.bundle_dir, &intervention.bundle_dir);
        for (batched, standalone) in [
            (&point.baseline, &baseline),
            (&point.intervention, &intervention),
            (&pair.baseline, &baseline),
            (&pair.intervention, &intervention),
        ] {
            assert_eq!(batched.generated_token_ids, standalone.generated_token_ids);
            assert_eq!(batched.events, standalone.events);
            let runtime: serde_json::Value = serde_json::from_slice(
                &std::fs::read(Path::new(&batched.bundle_dir).join("runtime.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(runtime["decode_batch"]["path"], "batched", "layer {layer}");
        }
        let runtime: serde_json::Value = serde_json::from_slice(
            &std::fs::read(Path::new(&point.intervention.bundle_dir).join("runtime.json")).unwrap(),
        )
        .unwrap();
        let path = &runtime["prefix_reuse"]["inputs"][0]["path"];
        assert_eq!(
            path,
            if layer == 0 {
                "full-recompute"
            } else {
                "resumed"
            },
            "layer {layer}"
        );
    }
    // A sweep the cache does not cover rebuilds the pass.
    req.layer = Some(2);
    req.factor = Some(0.5);
    let other = session.run_sweep_point(&cfg_of(&req), &[2, 3]).unwrap();
    assert!(other.verification.ok);
}

/// The native console's 16-layer sweep, as the worker runs it: one pair
/// per layer (the previous path) versus sweep points. Needs a real
/// model: `EMBER_TIMING_MODEL=model.gguf cargo test --release --bin
/// ember gui_sweep_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing harness; needs EMBER_TIMING_MODEL"]
fn gui_sweep_timing() {
    let Ok(model) = std::env::var("EMBER_TIMING_MODEL") else {
        return;
    };
    let mut req = base_request();
    req.model_path = model;
    req.prompt = "The capital of France is".to_string();
    req.max_new_tokens = 24;
    req.site = "after-layer".to_string();
    req.operation = "scale".to_string();
    req.factor = Some(0.0);
    let mut session = GuiSession::new(KStrategy::Auto, false);
    session.ensure_prepared(&req.model_path).unwrap();
    let layers: Vec<usize> = (0..session.prepared.as_ref().unwrap().n_layers).collect();
    let mut pairs = Vec::new();
    let started = std::time::Instant::now();
    for &layer in &layers {
        req.layer = Some(layer);
        pairs.push(session.run_baseline_intervention(&cfg_of(&req)).unwrap());
    }
    let pairs_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = std::time::Instant::now();
    let mut points = Vec::new();
    for (index, &layer) in layers.iter().enumerate() {
        req.layer = Some(layer);
        points.push(
            session
                .run_sweep_point(&cfg_of(&req), &layers[index..])
                .unwrap(),
        );
    }
    let points_ms = started.elapsed().as_secs_f64() * 1000.0;
    for (pair, point) in pairs.iter().zip(&points) {
        eprintln!(
            "  pair {:.0} ms (baseline gen {:.0}, intervention gen {:.0}) | point {:.0} ms \
             (intervention gen {:.0})",
            pair.elapsed_ms_total,
            pair.baseline.wall_ms,
            pair.intervention.wall_ms,
            point.elapsed_ms_total,
            point.intervention.wall_ms
        );
        assert_same_content(&pair.baseline.bundle_dir, &point.baseline.bundle_dir);
        assert_same_content(
            &pair.intervention.bundle_dir,
            &point.intervention.bundle_dir,
        );
        assert_eq!(pair.comparison, point.comparison);
    }
    eprintln!(
        "gui sweep ({} layers, 24 tokens, after-layer scale 0): one pair per layer {:.1} s, \
         sweep points {:.1} s (first point {:.1} s)",
        layers.len(),
        pairs_ms / 1000.0,
        points_ms / 1000.0,
        points[0].elapsed_ms_total / 1000.0
    );
}

/// Wall-time of a GUI-style baseline+intervention pair, computed as two
/// full runs (the previous path) and with the shared prefix. Needs a
/// real model: `EMBER_TIMING_MODEL=model.gguf cargo test --release
/// --bin ember gui_pair_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing harness; needs EMBER_TIMING_MODEL"]
fn gui_pair_timing() {
    let Ok(model) = std::env::var("EMBER_TIMING_MODEL") else {
        return;
    };
    let mut req = base_request();
    req.model_path = model;
    req.prompt = "The capital of France is".to_string();
    req.max_new_tokens = 24;
    req.site = "after-mlp".to_string();
    req.layer = Some(8);
    let cfg = cfg_of(&req);
    let mut session = GuiSession::new(KStrategy::Auto, false);
    session.ensure_prepared(&cfg.model_path).unwrap();
    let mut full = Vec::new();
    let mut shared = Vec::new();
    for round in 0..6 {
        let started = std::time::Instant::now();
        let (baseline, _) = run_one(&mut session, &cfg, RunKind::Baseline).unwrap();
        let (intervention, _) = run_one(&mut session, &cfg, RunKind::Intervention).unwrap();
        let full_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = std::time::Instant::now();
        let pair = session.run_baseline_intervention(&cfg).unwrap();
        let shared_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert_same_content(&baseline.bundle_dir, &pair.baseline.bundle_dir);
        assert_same_content(&intervention.bundle_dir, &pair.intervention.bundle_dir);
        if round > 0 {
            full.push(full_ms);
            shared.push(shared_ms);
        }
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    eprintln!(
        "gui pair (24 tokens, after-mlp layer 8): two full runs {:.0} ms, shared prefix \
         {:.0} ms (median of {})",
        median(&mut full),
        median(&mut shared),
        full.len()
    );
}

#[test]
fn the_spec_view_reads_back_what_the_console_writes() {
    for (operation, source) in [
        ("replace", "capture"),
        ("replace", "zero"),
        ("zero", "capture"),
        ("scale", "capture"),
        ("interpolate", "zero"),
        ("add-delta", "capture"),
    ] {
        let mut req = base_request();
        req.operation = operation.to_string();
        req.source = source.to_string();
        req.factor = (operation == "scale").then_some(0.25);
        req.alpha = (operation == "interpolate").then_some(0.4);
        let text = intervention_spec_text(&cfg_of(&req)).expect("spec");
        let fields = spec_fields(&text).expect("fields");
        assert_eq!(fields.operation, operation);
        assert_eq!(fields.site, req.site);
        assert_eq!(fields.layer, req.layer);
        assert_eq!(fields.prompt, req.prompt);
        let takes_source = matches!(operation, "replace" | "interpolate" | "add-delta");
        assert_eq!(fields.source.as_deref(), takes_source.then_some(source));
        assert_eq!(
            fields.value,
            match operation {
                "scale" => Some(0.25),
                "interpolate" => Some(0.4),
                _ => None,
            }
        );
        assert_eq!(check_same_experiment(&text, &text), Ok(()));
    }
}

#[test]
fn the_spec_view_says_what_the_form_cannot_hold() {
    let text = intervention_spec_text(&cfg_of(&base_request())).expect("spec");
    let error = |edited: &str| {
        spec_fields(edited)
            .and_then(|_| check_same_experiment(edited, &text))
            .unwrap_err()
    };
    // Metadata the console owns is not a difference.
    let renamed = text.replace("name = \"gui-intervention\"", "name = \"mine\"");
    assert_ne!(renamed, text);
    assert_eq!(check_same_experiment(&renamed, &text), Ok(()));
    // A field the form has no control for is named.
    let seeded = text.replace("seed = 0", "seed = 7");
    assert_ne!(seeded, text);
    assert!(
        error(&seeded).contains("experiment.seed"),
        "{}",
        error(&seeded)
    );
    let second_input = text.replacen(
        "[[inputs]]",
        "[[inputs]]\nid = \"prompt-2\"\ntext = \"more\"\n\n[[inputs]]",
        1,
    );
    assert!(error(&second_input).contains("one prompt"));
    let absolute = text.replace(
        "kind = \"prompt-final\"",
        "kind = \"absolute-token\"\nindex = 1",
    );
    assert_ne!(absolute, text);
    assert!(
        error(&absolute).contains("absolute-token"),
        "{}",
        error(&absolute)
    );
    // The form's picker does not offer restore-original.
    let restore = text
        .replace("kind = \"scale\"", "kind = \"restore-original\"")
        .replace("factor = 0.5\n", "");
    assert_ne!(restore, text);
    assert!(
        error(&restore).contains("uses restore-original"),
        "{}",
        error(&restore)
    );
    // A spec that does not parse says why.
    assert!(spec_fields("schema = 1").is_err());
}

#[test]
fn every_operation_builds_valid_specs() {
    for (operation, extra) in [
        ("replace", "capture"),
        ("zero", "capture"),
        ("scale", "capture"),
        ("interpolate", "capture"),
        ("add-delta", "capture"),
    ] {
        let mut req = base_request();
        req.operation = operation.to_string();
        req.source = extra.to_string();
        if operation == "scale" {
            req.factor = Some(0.25);
        }
        if operation == "interpolate" {
            req.alpha = Some(0.4);
        }
        let cfg = cfg_of(&req);
        for kind in [RunKind::Baseline, RunKind::Intervention, RunKind::Restore] {
            let (spec_text, resolved) =
                build_and_resolve_spec(&cfg, kind, "runs/gui/_test").expect("spec validates");
            // The raw form must be parseable by the strict raw parser.
            let reparsed = RawExperimentSpec::from_toml_str(&spec_text).expect("raw round trip");
            // Reproduction binds resolved-experiment.json to the hashed
            // spec text, so the text must resolve to exactly this spec.
            assert_eq!(reparsed.resolve().expect("raw re-resolves"), resolved);
            assert_eq!(
                resolved.interventions.len(),
                usize::from(kind == RunKind::Intervention || kind == RunKind::Restore)
            );
            if kind == RunKind::Intervention {
                assert_eq!(resolved.captures.len(), 2);
                let trace = resolved
                    .captures
                    .iter()
                    .find(|capture| capture.id == GUI_LAYER_TRACE_CAPTURE_ID)
                    .expect("native workbench layer trace");
                assert_eq!(trace.site, SemanticHookSite::ResidualPostMlp);
                assert!(matches!(trace.layers, LayerSelector::All(_)));
                assert_eq!(trace.storage, CaptureStorage::SelectedRows);
            }
        }
    }
}

#[test]
fn non_per_layer_sites_use_all_layers() {
    for site in ["before-logits", "after-logits"] {
        let mut req = base_request();
        req.site = site.to_string();
        req.layer = None;
        let cfg = cfg_of(&req);
        let (_, resolved) =
            build_and_resolve_spec(&cfg, RunKind::Intervention, "runs/gui/_test").unwrap();
        let layers = resolved.interventions[0].layers.clone();
        assert!(matches!(layers, LayerSelector::All(_)));
    }
}

/// The browser page shows the layer control for exactly these stages,
/// so the per-head stage must be among them and the head-boundary
/// stages must not; a run at the per-head stage builds a valid spec.
#[test]
fn per_layer_stages_include_the_attention_heads_stage() {
    let per_layer = per_layer_hook_stages();
    assert_eq!(
        per_layer,
        [
            "before-layer",
            "before-o-proj",
            "after-attention",
            "after-mlp",
            "after-layer"
        ]
    );
    assert!(hook_stages().iter().all(
        |stage| per_layer.contains(stage) || ["before-logits", "after-logits"].contains(stage)
    ));
    let mut req = base_request();
    req.site = "before-o-proj".to_string();
    let cfg = cfg_of(&req);
    let (_, resolved) =
        build_and_resolve_spec(&cfg, RunKind::Intervention, "runs/gui/_test").unwrap();
    assert_eq!(
        resolved.interventions[0].site,
        ember::v05::hook::SemanticHookSite::AttentionHeads
    );
}

#[test]
fn layer_required_for_per_layer_sites() {
    let mut req = base_request();
    req.layer = None;
    assert!(parse_run_request(&req).is_err());
}

#[test]
fn source_layer_ordering_is_enforced() {
    let mut req = base_request();
    req.operation = "replace".to_string();
    req.source_layer = Some(9);
    assert!(parse_run_request(&req).is_err()); // 9 > 7
    req.source_layer = Some(6);
    assert!(parse_run_request(&req).is_ok()); // 6 <= 7
}

#[test]
fn zero_source_only_for_replace_and_interpolate() {
    for op in ["zero", "scale", "add-delta"] {
        let mut req = base_request();
        req.operation = op.to_string();
        req.source = "zero".to_string();
        assert!(parse_run_request(&req).is_err(), "{op} + zero source");
    }
    for op in ["replace", "interpolate"] {
        let mut req = base_request();
        req.operation = op.to_string();
        req.source = "zero".to_string();
        assert!(parse_run_request(&req).is_ok(), "{op} + zero source");
    }
}

#[test]
fn matched_span_requires_text() {
    let mut req = base_request();
    req.token_kind = "matched-span".to_string();
    req.span_text = Some("   ".to_string());
    assert!(parse_run_request(&req).is_err());
    req.span_text = Some("\u{0643}\u{062a}\u{0627}\u{0628}".to_string());
    let cfg = parse_run_request(&req).unwrap();
    let (_, resolved) =
        build_and_resolve_spec(&cfg, RunKind::Intervention, "runs/gui/_test").unwrap();
    assert!(matches!(
        resolved.captures[0].tokens,
        TokenSelector::MatchedTextSpan { .. }
    ));
}

#[test]
fn restore_spec_uses_restore_original() {
    let cfg = cfg_of(&base_request());
    let (_, resolved) = build_and_resolve_spec(&cfg, RunKind::Restore, "runs/gui/_test").unwrap();
    assert_eq!(resolved.interventions.len(), 1);
    assert!(matches!(
        resolved.interventions[0].operation,
        InterventionOperation::RestoreOriginal
    ));
    assert!(resolved.captures.is_empty());
}

#[test]
fn comparison_keys_cover_shared_configuration() {
    // The comparison key deliberately ignores the intervention
    // operation (restore is compared against the same site/selection).
    let a = base_request();
    let mut same_op = base_request();
    same_op.operation = "zero".to_string();
    assert_eq!(
        cfg_of(&a).comparison_key(),
        cfg_of(&same_op).comparison_key()
    );
    // But it does cover site, layer, and token selection.
    let mut other_site = base_request();
    other_site.site = "after-attention".to_string();
    assert_ne!(
        cfg_of(&a).comparison_key(),
        cfg_of(&other_site).comparison_key()
    );
    let mut other_layer = base_request();
    other_layer.layer = Some(11);
    assert_ne!(
        cfg_of(&a).comparison_key(),
        cfg_of(&other_layer).comparison_key()
    );
    let mut other_tokens = base_request();
    other_tokens.token_kind = "matched-span".to_string();
    other_tokens.span_text = Some("\u{0643}".to_string());
    assert_ne!(
        cfg_of(&a).comparison_key(),
        cfg_of(&other_tokens).comparison_key()
    );
}

fn comparison_fixture(
    captures: Vec<ember::v05::compare::CaptureComparison>,
    output: Option<ember::v05::compare::OutputComparison>,
) -> ember::v05::compare::CompareResult {
    use ember::v05::compare::{CompareResult, IdentityComparison, RuntimeComparison};
    CompareResult {
        bundle_a: "baseline".to_string(),
        bundle_b: "intervention".to_string(),
        identity: IdentityComparison {
            schema_compatible: true,
            semantic_hash_equal: false,
            model_hash_equal: true,
            tokenizer_hash_equal: true,
            execution_mode_equal: true,
            plan_hash_equal: true,
            input_ids_equal: true,
            prompts_equal: true,
            tokenization_equal: true,
        },
        outputs: output.into_iter().collect(),
        captures,
        interventions: Vec::new(),
        runtime: RuntimeComparison {
            decode_throughput_tps_a: None,
            decode_throughput_tps_b: None,
            prefill_throughput_tps_a: None,
            prefill_throughput_tps_b: None,
            first_token_latency_ms_a: None,
            first_token_latency_ms_b: None,
            peak_rss_kb_a: None,
            peak_rss_kb_b: None,
            scratch_bytes_a: None,
            scratch_bytes_b: None,
        },
    }
}

fn output_fixture(
    equal: bool,
    first_divergence_step: Option<usize>,
) -> ember::v05::compare::OutputComparison {
    ember::v05::compare::OutputComparison {
        input_id: "i1".to_string(),
        generated_tokens_equal: equal,
        generated_text_equal: equal,
        final_top1_equal: equal,
        first_divergence_step,
        generated_count_a: 3,
        generated_count_b: 3,
    }
}

fn capture_fixture(
    layer: usize,
    metrics: Option<ember::v05::compare::TensorMetrics>,
) -> ember::v05::compare::CaptureComparison {
    ember::v05::compare::CaptureComparison {
        capture_id: GUI_LAYER_TRACE_CAPTURE_ID.to_string(),
        input_id: "i1".to_string(),
        site: SemanticHookSite::ResidualPostMlp,
        layer,
        present_in_a: true,
        present_in_b: metrics.is_some(),
        metrics,
    }
}

#[test]
fn result_metrics_handle_empty_aborted_comparison() {
    let raw = comparison_fixture(Vec::new(), None);
    let result = derive_experiment_comparison(&raw, &[], &[], &[], &[]);
    assert!(result.layers.is_empty());
    assert!(result.tokens.is_empty());
    assert_eq!(result.first_token_divergence, None);
    assert!(!result.generated_text_equal);
    assert_eq!(result.landmarks.first_layer_divergence, None);
    assert_eq!(result.landmarks.peak_layer, None);
    assert_eq!(result.landmarks.stable_token_tail_step, None);
    assert!(result.layer_token_grid.is_none());
}

#[test]
fn result_metrics_preserve_sparse_layers_and_reject_non_finite_values() {
    use ember::v05::compare::TensorMetrics;
    let captures = vec![
        capture_fixture(
            9,
            Some(TensorMetrics {
                shape_equal: true,
                dtype_equal: true,
                exact: true,
                ..TensorMetrics::default()
            }),
        ),
        capture_fixture(4, None),
        capture_fixture(
            1,
            Some(TensorMetrics {
                shape_equal: true,
                dtype_equal: true,
                exact: false,
                relative_l2_difference: Some(f64::NAN),
                cosine_similarity: Some(f64::INFINITY),
                maximum_absolute_difference: Some(f64::NEG_INFINITY),
                ..TensorMetrics::default()
            }),
        ),
    ];
    let raw = comparison_fixture(captures, Some(output_fixture(true, None)));
    let result = derive_experiment_comparison(&raw, &[], &[], &[], &[]);
    assert_eq!(
        result
            .layers
            .iter()
            .map(|metric| metric.layer)
            .collect::<Vec<_>>(),
        vec![1, 4, 9]
    );
    assert_eq!(result.layers[0].relative_l2_difference, None);
    assert_eq!(result.layers[0].cosine_distance, None);
    assert_eq!(result.layers[0].maximum_absolute_difference, None);
    assert_eq!(result.layers[1].relative_l2_difference, None);
    assert_eq!(result.layers[2].relative_l2_difference, Some(0.0));
    assert_eq!(result.layers[2].cosine_distance, Some(0.0));
    assert_eq!(result.landmarks.first_layer_divergence, None);
    assert_eq!(result.landmarks.peak_relative_l2, None);
}

#[test]
fn identical_single_layer_pair_has_zero_divergence() {
    let metrics = ember::v05::compare::TensorMetrics {
        shape_equal: true,
        dtype_equal: true,
        exact: true,
        relative_l2_difference: Some(0.0),
        cosine_similarity: Some(1.0),
        maximum_absolute_difference: Some(0.0),
        ..Default::default()
    };
    let raw = comparison_fixture(
        vec![capture_fixture(0, Some(metrics))],
        Some(output_fixture(true, None)),
    );
    let texts = vec![" المدينة".to_string()];
    let result = derive_experiment_comparison(&raw, &[42], &texts, &[42], &texts);
    assert_eq!(result.layers.len(), 1);
    assert_eq!(result.layers[0].relative_l2_difference, Some(0.0));
    assert_eq!(result.tokens[0].baseline_text.as_deref(), Some(" المدينة"));
    assert!(!result.tokens[0].differs);
    assert!(result.generated_text_equal);
    assert_eq!(result.landmarks.first_layer_divergence, None);
    assert_eq!(result.landmarks.peak_relative_l2, None);
    assert_eq!(result.landmarks.stable_token_tail_step, None);
}

#[test]
fn token_pairing_handles_long_and_unequal_traces() {
    let baseline: Vec<u32> = (0..5_000).collect();
    let mut intervention = baseline.clone();
    intervention[4_321] = 99_999;
    intervention.push(5_001);
    let baseline_text: Vec<String> = baseline.iter().map(ToString::to_string).collect();
    let intervention_text: Vec<String> = intervention.iter().map(ToString::to_string).collect();
    let raw = comparison_fixture(Vec::new(), Some(output_fixture(false, Some(4_322))));
    let result = derive_experiment_comparison(
        &raw,
        &baseline,
        &baseline_text,
        &intervention,
        &intervention_text,
    );
    assert_eq!(result.tokens.len(), 5_001);
    assert!(result.tokens[4_321].differs);
    assert!(result.tokens[5_000].differs);
    assert_eq!(result.first_token_divergence, Some(4_322));
    assert_eq!(result.landmarks.stable_token_tail_step, None);
}

#[test]
fn result_landmarks_find_first_layer_peak_and_stable_token_tail() {
    use ember::v05::compare::TensorMetrics;
    let metric = |relative_l2_difference: f64| TensorMetrics {
        shape_equal: true,
        dtype_equal: true,
        exact: false,
        relative_l2_difference: Some(relative_l2_difference),
        cosine_similarity: Some(0.9),
        maximum_absolute_difference: Some(relative_l2_difference),
        ..TensorMetrics::default()
    };
    let raw = comparison_fixture(
        vec![
            capture_fixture(2, Some(metric(0.0))),
            capture_fixture(5, Some(metric(0.125))),
            capture_fixture(9, Some(metric(0.75))),
        ],
        Some(output_fixture(false, Some(2))),
    );
    let result = derive_experiment_comparison(&raw, &[10, 20, 30, 40], &[], &[10, 99, 30, 40], &[]);
    assert_eq!(result.landmarks.first_layer_divergence, Some(5));
    assert_eq!(result.landmarks.peak_layer, Some(9));
    assert_eq!(result.landmarks.peak_relative_l2, Some(0.75));
    assert_eq!(result.landmarks.stable_token_tail_step, Some(3));
}

#[test]
fn stable_token_tail_requires_equal_lengths_and_nonempty_suffix() {
    let raw = comparison_fixture(Vec::new(), Some(output_fixture(false, Some(2))));
    let unequal = derive_experiment_comparison(&raw, &[1, 2], &[], &[1, 9, 2], &[]);
    assert_eq!(unequal.landmarks.stable_token_tail_step, None);

    let final_step_differs = derive_experiment_comparison(&raw, &[1, 2, 3], &[], &[1, 2, 9], &[]);
    assert_eq!(final_step_differs.landmarks.stable_token_tail_step, None);
}
