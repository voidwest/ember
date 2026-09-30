//! Render-test harness (`gui-tests`): fixture data, the example probe, the
//! live flow and the screenshot renderer.

use super::*;

/// Whether the render harness should fill the store with representative runs.
///
/// Gated behind an env var rather than on `cfg!(feature = "gui-tests")` alone,
/// because a feature-gated fixture would also fire in the kit tests, where a
/// populated store would mask the empty-state behaviour those tests check.
pub(super) fn seed_runs_requested() -> bool {
    std::env::var_os("EMBER_GUI_TEST_SEED_RUNS").is_some()
}

/// Representative history for screenshot review.
///
/// Every screenshot so far has been an empty state, which is why column widths,
/// truncation and density kept going unjudged until someone opened a real
/// window. These are the shapes that actually occur: a long model name, an
/// intervention with a layer, one run that did not verify, a pinned record, and
/// timestamps spread far enough apart to exercise "2m ago" versus "3d ago".
///
/// Timestamps are relative to load time so the relative formatting stays
/// honest whenever the fixtures are regenerated.
#[cfg(feature = "gui-tests")]
pub(super) fn seed_store() -> AppStore {
    /// One fixture row. Named fields on purpose: the previous version was a
    /// twelve-element tuple ending in three bare booleans, and I set `pinned`
    /// on six of the seven rows without noticing, which made the pin marker
    /// meaningless and the ordering look like a plain newest-first list.
    struct Row {
        model: &'static str,
        intervention: &'static str,
        hook: &'static str,
        layer: Option<u32>,
        duration_ms: u64,
        baseline_tokens: u32,
        intervention_tokens: u32,
        diverged_at_step: Option<u32>,
        outputs_equal: bool,
        verified: bool,
        pinned: bool,
        age_seconds: i64,
    }
    // The shapes that actually occur: a long model name, a layer, a run that
    // did not verify, a run whose token counts diverged, and timestamps spread
    // from minutes to a fortnight so the relative formatting is exercised.
    //
    // One run is pinned, and it is deliberately not the newest, so the render
    // shows pinning reordering rather than coinciding with recency.
    let rows = [
        Row {
            model: "Llama-3.2-1B-Instruct-Q8_0",
            intervention: "Scale \u{d7}0.5",
            hook: "After MLP block",
            layer: Some(8),
            duration_ms: 1_240,
            baseline_tokens: 48,
            intervention_tokens: 48,
            diverged_at_step: Some(17),
            outputs_equal: false,
            verified: true,
            pinned: false,
            age_seconds: 120,
        },
        Row {
            model: "Qwen2.5-1.5B-Instruct-Q8_0",
            intervention: "Zero",
            hook: "Before output head",
            layer: None,
            duration_ms: 890,
            baseline_tokens: 48,
            intervention_tokens: 31,
            diverged_at_step: Some(4),
            outputs_equal: false,
            verified: true,
            pinned: false,
            age_seconds: 2_400,
        },
        Row {
            model: "gemma-3-270m-it-Q4_K_M",
            intervention: "Copy from layer",
            hook: "After MLP block",
            layer: Some(3),
            duration_ms: 410,
            baseline_tokens: 32,
            intervention_tokens: 32,
            diverged_at_step: Some(2),
            outputs_equal: false,
            verified: true,
            pinned: true,
            age_seconds: 86_400,
        },
        Row {
            model: "Llama-3.2-1B-Instruct-Q8_0",
            intervention: "Add a learned difference",
            hook: "After attention block",
            layer: Some(14),
            duration_ms: 3_010,
            baseline_tokens: 48,
            intervention_tokens: 48,
            diverged_at_step: None,
            outputs_equal: true,
            verified: false,
            pinned: false,
            age_seconds: 9_600,
        },
        Row {
            model: "Llama-3.2-1B-Instruct-Q8_0",
            intervention: "Blend representations",
            hook: "Final prompt token",
            layer: None,
            duration_ms: 1_980,
            baseline_tokens: 48,
            intervention_tokens: 48,
            diverged_at_step: Some(29),
            outputs_equal: false,
            verified: true,
            pinned: false,
            age_seconds: 172_800,
        },
        Row {
            model: "Qwen2.5-0.5B-Instruct-Q8_0",
            intervention: "Scale \u{d7}0.75",
            hook: "Before output head",
            layer: None,
            duration_ms: 640,
            baseline_tokens: 24,
            intervention_tokens: 24,
            diverged_at_step: None,
            outputs_equal: true,
            verified: true,
            pinned: false,
            age_seconds: 604_800,
        },
        Row {
            model: "Llama-3.2-1B-Instruct-Q8_0",
            intervention: "Zero",
            hook: "After attention block",
            layer: Some(21),
            duration_ms: 2_260,
            baseline_tokens: 48,
            intervention_tokens: 12,
            diverged_at_step: Some(1),
            outputs_equal: false,
            verified: true,
            pinned: false,
            age_seconds: 1_209_600,
        },
    ];

    let now = unix_now();
    let mut store = AppStore::default();
    for (index, row) in rows.iter().enumerate() {
        store.push_run(RunRecord {
            number: (rows.len() - index) as u64,
            finished_at: now - row.age_seconds,
            model: row.model.to_string(),
            intervention: row.intervention.to_string(),
            hook: row.hook.to_string(),
            layer: row.layer,
            duration_ms: Some(row.duration_ms),
            baseline_tokens: Some(row.baseline_tokens),
            intervention_tokens: Some(row.intervention_tokens),
            diverged_at_step: row.diverged_at_step,
            outputs_equal: row.outputs_equal,
            verified: row.verified,
            pinned: row.pinned,
            prompt: "\u{0627}\u{0643}\u{062a}\u{0628} \u{062c}\u{0645}\u{0644}\u{0629}".to_string(),
            result: (index <= 1).then(|| app_store::RecordResult {
                baseline_text: "Paris. The Eiffel Tower is located in Paris.".to_string(),
                intervention_text: "covered in a thick layer of fog.".to_string(),
                layers: (0..16)
                    .map(|layer| app_store::RecordLayer {
                        layer,
                        relative_l2: Some(if layer < 7 { 0.0 } else { 1.1 }),
                        cosine: Some(if layer < 7 { 0.0 } else { 0.9 }),
                    })
                    .collect(),
                tokens: Vec::new(),
                first_layer_divergence: Some(7),
                peak_layer: Some(10),
                peak_relative_l2: Some(1.187),
                tokens_equal: false,
            }),
            // Only recent rows carry a configuration: the Reuse action must
            // be shown and hidden in the same render.
            config: (index == 0).then(|| app_store::RecordConfig {
                model_path: format!("/models/{}.gguf", row.model),
                execution: "reference".to_string(),
                site: "after-mlp".to_string(),
                layer: row.layer.map(|layer| layer.to_string()).unwrap_or_default(),
                op: "scale".to_string(),
                value: "0.5".to_string(),
                source: "live".to_string(),
                source_layer: String::new(),
                token: "prompt-final".to_string(),
                span: String::new(),
                max_tokens: "48".to_string(),
            }),
        });
        store.touch_model(
            &format!("/models/{}.gguf", row.model),
            now - row.age_seconds,
        );
    }
    // An experiment in progress, so Home's "continue where you left off" row
    // is reviewed at real density rather than as a guess.
    store.draft = Some(app_store::Draft {
        revision: 3,
        prompt: "\u{0627}\u{0643}\u{062a}\u{0628} \u{062c}\u{0645}\u{0644}\u{0629} \u{0642}\u{0635}\u{064a}\u{0631}\u{0629} \u{0639}\u{0646} \u{0627}\u{0644}\u{0645}\u{062f}\u{064a}\u{0646}\u{0629} \u{0627}\u{0644}\u{0645}\u{0646}\u{0648}\u{0631}\u{0629}".to_string(),
        model_path: "/models/Llama-3.2-1B-Instruct-Q8_0.gguf".to_string(),
        fields: [
            ("op".to_string(), "scale".to_string()),
            ("site".to_string(), "after-mlp".to_string()),
            ("layer".to_string(), "8".to_string()),
            ("value".to_string(), "0.5".to_string()),
            ("max_tokens".to_string(), "48".to_string()),
            ("token".to_string(), "prompt-final".to_string()),
        ]
        .into_iter()
        .collect(),
        step: "intervention".to_string(),
        updated_at: now - 240,
    });
    store
}

#[cfg(not(feature = "gui-tests"))]
pub(super) fn seed_store() -> AppStore {
    AppStore::default()
}

/// A finished comparison, for reviewing the Review page at real density.
///
/// The empty state is what every screenshot had shown, so the pane layout, the
/// metadata row and the bidi handling of a mixed-direction completion were all
/// being judged against a page with nothing on it. Two outputs that share a
/// prefix and then diverge is the interesting case: it is the one where the
/// token arrow, the divergence note and the side-by-side panes all have
/// something to say.
#[cfg(feature = "gui-tests")]
pub(super) fn seed_comparison() -> (RunOutput, RunOutput, ExperimentComparison) {
    let shared = "The city of Madinah is one of the oldest continuously inhabited places in the world, known for the Prophet's Mosque.";
    let diverged = "The city of Madinah is among the oldest inhabited places in the world, famed for the Prophet's Mosque and its courtyards.";
    let arabic = "\u{0627}\u{0643}\u{062a}\u{0628}\u{060c}\u{0627}\u{0644}\u{0645}\u{062f}\u{064a}\u{0646}\u{0629} \u{0627}\u{062d}\u{062f} \u{0623}\u{0642}\u{062f}\u{0645}\u{0627}\u{0641}\u{0627}\u{064a} \u{0641}\u{064a} \u{0627}\u{0644}\u{0639}\u{0627}\u{0644}\u{0645}.";
    let make = |text: String, tokens: usize, wall_ms: f64| RunOutput {
        text,
        generated_token_ids: vec![1; tokens],
        generated_token_texts: vec![String::new(); tokens],
        prompt_tokens: 24,
        generated_tokens: tokens,
        bundle_dir: "/tmp/ember-render-fixture/baseline".to_string(),
        semantic_hash: "9f2c1a7b4e6d0358".to_string(),
        payload_hash: "3a91f0c2".to_string(),
        wall_ms,
        decode_tps: Some(tokens as f64 / (wall_ms / 1000.0)),
        events: Vec::new(),
    };
    let baseline = make(shared.to_string(), 48, 1_240.0);
    let intervention = make(format!("{shared} {diverged}"), 61, 1_980.0);
    let comparison = ExperimentComparison {
        layers: Vec::new(),
        tokens: Vec::new(),
        first_token_divergence: Some(38),
        generated_tokens_equal: false,
        generated_text_equal: false,
        landmarks: crate::gui::DivergenceLandmarks {
            first_layer_divergence: Some(7),
            peak_layer: Some(14),
            peak_relative_l2: Some(0.214),
            stable_token_tail_step: None,
        },
        layer_token_grid: None,
    };
    let _ = arabic;
    (baseline, intervention, comparison)
}

// Without the gui-tests feature there is no render harness to feed, and the
// only call site is behind the same cfg, so this stub exists purely to keep the
// non-test build compiling.
#[cfg(not(feature = "gui-tests"))]
#[expect(dead_code, reason = "only the gui-tests build calls the real one")]
fn seed_comparison() -> (RunOutput, RunOutput, ExperimentComparison) {
    unimplemented!("render fixture is only available under gui-tests")
}

/// Run a batch of interventions on one loaded model and report which change
/// the words. Output is plain text on stderr; nothing is rendered or saved.
#[cfg(all(target_os = "macos", feature = "gui-tests"))]
pub(super) fn probe_examples(model: String) -> anyhow::Result<()> {
    let platform = gpui_kit::platform::current_platform(true);
    let mut context = HeadlessAppContext::with_platform(
        platform.text_system(),
        Arc::new(icons::Assets),
        gpui_kit::platform::current_headless_renderer,
    );
    context.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, rx) = spawn_worker(ember::quant_k::KStrategy::Auto, false);
    let mut console = None;
    let handle = context.open_window(size(px(1200.), px(800.)), |window, cx| {
        let view = cx.new(|cx| Console::new(tx, rx, true, window, cx));
        console = Some(view.clone());
        cx.new(|cx| gpui_kit::component::Root::new(view, window, cx))
    })?;
    let console = console.unwrap();
    let prompt = std::env::var("EMBER_GUI_TEST_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    // (op, site, layer, value)
    let configs: Vec<(&str, &str, &str, &str)> = vec![
        ("zero", "after-layer", "15", "1.0"),
        ("zero", "after-layer", "14", "1.0"),
        ("zero", "after-layer", "12", "1.0"),
        ("zero", "after-layer", "8", "1.0"),
        ("zero", "after-mlp", "12", "1.0"),
        ("scale", "after-layer", "14", "8.0"),
        ("scale", "after-layer", "10", "4.0"),
        ("scale", "after-mlp", "12", "10.0"),
        ("scale", "after-mlp", "8", "-2.0"),
        ("scale", "after-layer", "6", "0.0"),
    ];
    for (op, site, layer, value) in configs {
        context.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, cx| {
                console.model_path = model.clone();
                console
                    .inputs
                    .model
                    .update(cx, |input, cx| input.set_value(model.clone(), cx));
                console.op = op.into();
                console.site = site.into();
                console.layer = layer.into();
                console.value = value.into();
                console.token = "prompt-final".into();
                console.prompt = prompt.clone();
                for (input, text) in [
                    (console.inputs.layer.clone(), layer.to_string()),
                    (console.inputs.value.clone(), value.to_string()),
                    (console.inputs.prompt.clone(), prompt.clone()),
                ] {
                    console.set_input_value(input, text, cx);
                }
                console.comparison = None;
                console.baseline = None;
                console.error = None;
                console.run();
            });
        })?;
        let started = std::time::Instant::now();
        loop {
            context.advance_clock(Duration::from_millis(50));
            context.run_until_parked();
            std::thread::sleep(Duration::from_millis(50));
            context.update_window(handle.into(), |_, _, cx| {
                console.update(cx, |console, cx| {
                    console.drain_replies(cx);
                });
            })?;
            let (done, error) = console.read_with(&context, |c, _| {
                (
                    c.comparison.is_some() && c.status == Status::Idle,
                    c.error.clone(),
                )
            });
            if let Some(error) = error {
                eprintln!("PROBE {op} {site} L{layer} x{value}: ERROR {error}");
                break;
            }
            if done {
                let line = console.read_with(&context, |c, _| {
                    let comparison = c.comparison.as_ref().unwrap();
                    format!(
                        "changed={} first_step={:?} peak={:?}@{:?}\n    base: {:?}\n    intv: {:?}",
                        !comparison.generated_text_equal,
                        comparison.first_token_divergence,
                        comparison.landmarks.peak_relative_l2,
                        comparison.landmarks.peak_layer,
                        c.baseline.as_ref().map(|b| b
                            .text
                            .trim()
                            .chars()
                            .take(70)
                            .collect::<String>()),
                        c.intervention.as_ref().map(|b| b
                            .text
                            .trim()
                            .chars()
                            .take(70)
                            .collect::<String>()),
                    )
                });
                eprintln!("PROBE {op} {site} L{layer} x{value}: {line}");
                break;
            }
            anyhow::ensure!(
                started.elapsed() < Duration::from_secs(300),
                "probe run timed out"
            );
        }
    }
    Ok(())
}

/// The first-run path, end to end: Home, an example, the three steps, a real
/// run on a real model with a live worker, then the result, Copy summary and
/// Runs. A frame is saved whenever the run's status changes, so the progress
/// steps are seen as they advance rather than inferred.
#[cfg(all(target_os = "macos", feature = "gui-tests"))]
pub(super) fn render_live_flow(directory: &std::path::Path, model: String) -> anyhow::Result<()> {
    use gpui_kit::test::TestWindowExt as _;
    std::fs::create_dir_all(directory)?;
    let platform = gpui_kit::platform::current_platform(true);
    let mut context = HeadlessAppContext::with_platform(
        platform.text_system(),
        Arc::new(icons::Assets),
        gpui_kit::platform::current_headless_renderer,
    );
    context.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, rx) = spawn_worker(ember::quant_k::KStrategy::Auto, false);
    let mut console = None;
    let handle = context.open_window(size(px(1728.), px(1092.)), |window, cx| {
        let view = cx.new(|cx| Console::new(tx, rx, true, window, cx));
        console = Some(view.clone());
        cx.new(|cx| gpui_kit::component::Root::new(view, window, cx))
    })?;
    let console = console.unwrap();
    let mut frame = 0usize;
    let mut shot = |context: &mut HeadlessAppContext, tag: &str| -> anyhow::Result<()> {
        context.update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
        })?;
        context.run_until_parked();
        context.update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
        })?;
        frame += 1;
        context
            .capture_screenshot(handle.into())?
            .save(directory.join(format!("live-{frame:02}-{tag}.png")))?;
        Ok(())
    };
    let click = |context: &mut HeadlessAppContext, id: &str| -> anyhow::Result<()> {
        let id = SharedString::from(id.to_string());
        context.update_window(handle.into(), |_, window, cx| window.click(id, cx))?;
        Ok(())
    };
    context.update_window(handle.into(), |_, _, cx| {
        console.update(cx, |console, cx| {
            console.appearance = AppearanceMode::Dark;
            console.model_path = model.clone();
            console
                .inputs
                .model
                .update(cx, |input, cx| input.set_value(model.clone(), cx));
            console.sync_kit_theme(cx);
            cx.notify();
        });
    })?;
    shot(&mut context, "home")?;
    click(&mut context, "home-example")?;
    anyhow::ensure!(
        console.read_with(&context, |c, _| c.view) == View::Experiment,
        "Try an example did not land on the workspace"
    );
    shot(&mut context, "workspace-before-run")?;
    click(&mut context, "setup-run")?;

    let started = std::time::Instant::now();
    let mut last = console.read_with(&context, |c, _| c.status);
    shot(&mut context, &format!("run-{last:?}").to_lowercase())?;
    let mut ticks = 0u32;
    loop {
        context.advance_clock(Duration::from_millis(50));
        context.run_until_parked();
        std::thread::sleep(Duration::from_millis(50));
        context.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, cx| {
                if console.drain_replies(cx) {
                    cx.notify();
                }
            });
        })?;
        let (status, done, error) = console.read_with(&context, |c, _| {
            (
                c.status,
                c.baseline.is_some() && c.status == Status::Idle,
                c.error.clone(),
            )
        });
        ticks += 1;
        if status != last {
            shot(&mut context, &format!("run-{status:?}").to_lowercase())?;
            last = status;
        } else if ticks.is_multiple_of(40) && status != Status::Idle {
            shot(
                &mut context,
                &format!("run-{status:?}-still").to_lowercase(),
            )?;
        }
        if let Some(error) = error {
            anyhow::bail!("the run reported an error: {error}");
        }
        if done {
            break;
        }
        anyhow::ensure!(
            started.elapsed() < Duration::from_secs(400),
            "run timed out"
        );
    }
    eprintln!(
        "live run finished in {:.1}s",
        started.elapsed().as_secs_f32()
    );
    shot(&mut context, "result-overview")?;
    click(&mut context, "review-copy")?;
    let copied = context
        .update(|cx| cx.read_from_clipboard())
        .and_then(|item| item.text());
    eprintln!(
        "copied summary: {}",
        copied
            .as_deref()
            .map_or("NOTHING".to_string(), |text| format!(
                "{} bytes",
                text.len()
            ))
    );
    if let Some(text) = &copied {
        std::fs::write(directory.join("copied-summary.md"), text)?;
    }
    shot(&mut context, "result-copied")?;
    click(&mut context, "result:layers")?;
    shot(&mut context, "result-layers")?;
    click(&mut context, "result:tokens")?;
    shot(&mut context, "result-tokens")?;
    // The repeated-experiment loop: pin this result, change one setting, see
    // that the result now says it is stale, run again, and compare.
    click(&mut context, "review-pin")?;
    anyhow::ensure!(
        console.read_with(&context, |c, _| c.reference.is_some()),
        "Pin as reference did not pin"
    );
    context.update_window(handle.into(), |_, _, cx| {
        console.update(cx, |console, cx| {
            console.select_combo(ComboId::Op, "zero", cx);
            console.select_combo(ComboId::Site, "after-mlp", cx);
            console.layer = "12".into();
            let layer = console.inputs.layer.clone();
            console.set_input_value(layer, "12".into(), cx);
            cx.notify();
        });
    })?;
    anyhow::ensure!(
        console.read_with(&context, |c, _| c.results_stale()),
        "editing a setting did not mark the result stale"
    );
    shot(&mut context, "stale-after-edit")?;
    click(&mut context, "setup-run")?;
    let started_again = std::time::Instant::now();
    loop {
        context.advance_clock(Duration::from_millis(50));
        context.run_until_parked();
        std::thread::sleep(Duration::from_millis(50));
        context.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, cx| {
                if console.drain_replies(cx) {
                    cx.notify();
                }
            });
        })?;
        let (done, error) = console.read_with(&context, |c, _| {
            (c.status == Status::Idle && c.result_context.as_ref().is_some_and(|x| x.op == "zero"), c.error.clone())
        });
        if let Some(error) = error {
            anyhow::bail!("the second run reported an error: {error}");
        }
        if done {
            break;
        }
        anyhow::ensure!(started_again.elapsed() < Duration::from_secs(400), "second run timed out");
    }
    eprintln!("second run (same loaded model) finished in {:.1}s", started_again.elapsed().as_secs_f32());
    shot(&mut context, "compare-with-reference")?;
    click(&mut context, "result:layers")?;
    shot(&mut context, "compare-layers")?;
    // A sweep: the same change at every layer, then open one point.
    context.update_window(handle.into(), |_, _, cx| {
        console.update(cx, |console, cx| {
            console.select_combo(ComboId::Op, "scale", cx);
            console.value = "0.0".into();
            let value = console.inputs.value.clone();
            console.set_input_value(value, "0.0".into(), cx);
            console.select_combo(ComboId::Site, "after-layer", cx);
            cx.notify();
        });
    })?;
    click(&mut context, "setup-sweep")?;
    let sweep_started = std::time::Instant::now();
    let mut sweep_shots = 0;
    loop {
        context.advance_clock(Duration::from_millis(50));
        context.run_until_parked();
        std::thread::sleep(Duration::from_millis(50));
        context.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, cx| {
                if console.drain_replies(cx) {
                    cx.notify();
                }
            });
        })?;
        let (finished, progress, error) = console.read_with(&context, |c, _| {
            (
                c.sweep.as_ref().is_some_and(|s| s.finished),
                c.sweep.as_ref().map_or(0, |s| s.points.len()),
                c.error.clone(),
            )
        });
        if let Some(error) = error {
            anyhow::bail!("the sweep reported an error: {error}");
        }
        if progress >= 6 && sweep_shots == 0 {
            shot(&mut context, "sweep-midway")?;
            sweep_shots = 1;
        }
        if finished {
            break;
        }
        anyhow::ensure!(sweep_started.elapsed() < Duration::from_secs(600), "sweep timed out");
    }
    let points = console.read_with(&context, |c, _| c.sweep.as_ref().map_or(0, |s| s.points.len()));
    eprintln!("sweep of {points} layers finished in {:.1}s", sweep_started.elapsed().as_secs_f32());
    anyhow::ensure!(points >= 14, "the sweep produced too few points: {points}");
    shot(&mut context, "sweep-result")?;
    click(&mut context, "sweep-open:8")?;
    anyhow::ensure!(
        console.read_with(&context, |c, _| c.result_view) == ResultView::Overview,
        "opening a sweep point did not show its result"
    );
    shot(&mut context, "sweep-point-opened")?;
    click(&mut context, "result:sweep")?;
    click(&mut context, &format!("nav:{}", View::Runs.key()))?;
    shot(&mut context, "runs")?;
    click(&mut context, "run-compare:1")?;
    anyhow::ensure!(
        console.read_with(&context, |c, _| c.reference.as_ref().is_some_and(|r| r.label.starts_with("Run #1"))),
        "Compare on a saved run did not pin it"
    );
    anyhow::ensure!(
        console.read_with(&context, |c, _| c.view) == View::Experiment,
        "Compare did not return to the workspace"
    );
    shot(&mut context, "compare-from-runs")?;
    click(&mut context, &format!("nav:{}", View::Runs.key()))?;
    click(&mut context, "run-open:1")?;
    anyhow::ensure!(
        console.read_with(&context, |c, _| c.saved_run) == Some(1),
        "Open on the saved run did not reopen it"
    );
    shot(&mut context, "reopened-from-history")?;
    click(&mut context, &format!("nav:{}", View::Home.key()))?;
    shot(&mut context, "home-after")?;
    let draft = console.read_with(&context, |c, _| c.store.draft.is_some());
    eprintln!(
        "draft after a completed run: {}",
        if draft {
            "PRESENT (unexpected)"
        } else {
            "none"
        }
    );
    anyhow::ensure!(!draft, "a finished run left a draft to resume");
    Ok(())
}

/// Offscreen test scenes use the production Console render tree, CoreText, and
/// Metal. This exercises layout without automating another desktop application.
#[cfg(all(target_os = "macos", feature = "gui-tests"))]
pub(super) fn render_test_artifacts(directory: &std::path::Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(directory)?;
    // The harness drives the real console, whose persistence writes history
    // and workspace flags. Point it at a scratch directory first so fixture
    // runs never land in the developer's own ~/.config/ember. Nothing else has
    // started a thread yet, which is what makes changing the environment sound.
    if std::env::var_os("EMBER_GUI_TEST_KEEP_CONFIG").is_none() {
        let scratch =
            std::env::temp_dir().join(format!("ember-render-config-{}", std::process::id()));
        // SAFETY: called at the top of the harness, before any worker thread
        // exists, so no other thread can be reading the environment.
        unsafe { std::env::set_var("XDG_CONFIG_HOME", scratch) };
    }
    let platform = gpui_kit::platform::current_platform(true);
    let mut context = HeadlessAppContext::with_platform(
        platform.text_system(),
        Arc::new(icons::Assets),
        gpui_kit::platform::current_headless_renderer,
    );
    context.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let mut real_replies: Option<(WorkerReply, WorkerReply)> = None;
    // `EMBER_GUI_TEST_WINDOW=1728x1092` renders one window at that logical
    // size instead of the two defaults -- for previewing the console at a
    // demo machine's full-screen size without touching the visual tests.
    let sizes: Vec<(&'static str, f32, f32)> = std::env::var("EMBER_GUI_TEST_WINDOW")
        .ok()
        .and_then(|spec| {
            let (width, height) = spec.split_once('x')?;
            Some(vec![(
                "display",
                width.trim().parse().ok()?,
                height.trim().parse().ok()?,
            )])
        })
        .unwrap_or_else(|| vec![("standard", 1180., 720.), ("minimum", 980., 620.)]);
    for (name, width, height) in sizes {
        let (tx, _worker) = mpsc::channel();
        let (reply, rx) = mpsc::channel();
        let mut console = None;
        let handle = context.open_window(size(px(width), px(height)), |window, cx| {
            let view = cx.new(|cx| {
                let mut console = Console::new(tx, Arc::new(Mutex::new(rx)), false, window, cx);
                // Synchronously, so the Models scene is the same every time.
                console.models_discovered(scan_models(), cx);
                console
            });
            console = Some(view.clone());
            cx.new(|cx| gpui_kit::component::Root::new(view, window, cx))
        })?;
        let console = console.unwrap();
        for (appearance, mode) in [
            ("light", AppearanceMode::Light),
            ("dark", AppearanceMode::Dark),
        ] {
            // The app has a shell with several destinations, so the baseline
            // has to cover the shell and the experiment workspace, not just
            // the old single screen.
            for (view, step, inspector) in [
                (View::Home, WorkspaceStep::Prompt, false),
                (View::Models, WorkspaceStep::Prompt, false),
                (View::Runs, WorkspaceStep::Prompt, false),
                (View::Settings, WorkspaceStep::Prompt, false),
                (View::Experiment, WorkspaceStep::Prompt, false),
                (View::Experiment, WorkspaceStep::Intervention, false),
                (View::Experiment, WorkspaceStep::Review, true),
            ] {
                context.update_window(handle.into(), |_, window, cx| {
                    console.update(cx, |console, cx| {
                        console.appearance = mode;
                        console.view = view;
                        console.step = step;
                        // The seed comparison persists on the console once the
                        // Review scene has run, and scenes share one console
                        // per window. Without this reset every scene after the
                        // first Review reports "measured" on steps that have
                        // not measured anything, and the fixtures lie.
                        if !(view == View::Experiment && step == WorkspaceStep::Review) {
                            console.baseline = None;
                            console.intervention = None;
                            console.comparison = None;
                            console.layer_series = Arc::from([]);
                            console.verification = None;
                            console.restore = None;
                            console.result_context = None;
                            console.last_metrics = None;
                        }
                        console.sync_kit_theme(cx);
                        cx.notify();
                    });
                    window.draw(cx).clear(cx);
                })?;
                context.run_until_parked();
                context.update_window(handle.into(), |_, window, cx| {
                    window.draw(cx).clear(cx);
                })?;
                let file = match view {
                    View::Home => "home".to_string(),
                    View::Models => "models".to_string(),
                    View::Runs => "runs".to_string(),
                    View::Settings => "settings".to_string(),
                    View::Experiment => {
                        format!(
                            "experiment-{}{}",
                            step.number(),
                            if inspector { "-inspector" } else { "" }
                        )
                    }
                };
                context
                    .capture_screenshot(handle.into())?
                    .save(directory.join(format!("{name}-{appearance}-{file}.png")))?;
            }
            // The command palette overlays every page; capture it once per
            // theme so the dim backdrop and the floating panel are reviewed
            // against both canvases.
            context.update_window(handle.into(), |_, window, cx| {
                console.update(cx, |console, cx| {
                    console.palette_open = true;
                    cx.notify();
                });
                window.draw(cx).clear(cx);
            })?;
            context.run_until_parked();
            context.update_window(handle.into(), |_, window, cx| {
                window.draw(cx).clear(cx);
            })?;
            context
                .capture_screenshot(handle.into())?
                .save(directory.join(format!("{name}-{appearance}-palette.png")))?;
            context.update_window(handle.into(), |_, _, cx| {
                console.update(cx, |console, _| console.palette_open = false);
            })?;
            // Interaction states: hover and keyboard focus are invisible in
            // the resting-state scenes, and they are where a console reads as
            // polished or as a form.
            {
                use gpui_kit::test::TestWindowExt as _;
                let hover_nav = format!("nav:{}", View::Models.key());
                for (state, view, step, hover, tabs) in [
                    (
                        "hover-nav",
                        View::Home,
                        WorkspaceStep::Prompt,
                        Some(hover_nav),
                        0,
                    ),
                    (
                        "hover-tile",
                        View::Experiment,
                        WorkspaceStep::Intervention,
                        Some("setup-examples".to_string()),
                        0,
                    ),
                    (
                        "hover-preset",
                        View::Experiment,
                        WorkspaceStep::Prompt,
                        Some("setup-run".to_string()),
                        0,
                    ),
                    (
                        "focus-tab",
                        View::Experiment,
                        WorkspaceStep::Prompt,
                        None,
                        3,
                    ),
                    (
                        "focus-click",
                        View::Experiment,
                        WorkspaceStep::Prompt,
                        Some("generation-length:24".to_string()),
                        0,
                    ),
                ] {
                    let scrolled = context.update_window(handle.into(), |_, window, cx| {
                        console.update(cx, |console, cx| {
                            console.appearance = mode;
                            console.view = view;
                            console.step = step;
                            console.sync_kit_theme(cx);
                            cx.notify();
                        });
                        window.draw(cx).clear(cx);
                        let mut scrolled = 0;
                        if let Some(id) = &hover {
                            if state == "focus-click" {
                                // The target sits below the fold at the
                                // standard size, and the kit refuses to click
                                // what is not visible: scroll it into view
                                // first. A click focuses the button; the kit
                                // paints its ring whenever it holds focus.
                                let target = SharedString::from(id.clone());
                                while !window.find(target.clone()).visible()
                                    && scrolled < HARNESS_MAX_SCROLL_STEPS
                                {
                                    window.scroll("workspace-scroll", harness_scroll(-1.0), cx);
                                    window.draw(cx).clear(cx);
                                    scrolled += 1;
                                }
                                // "Visible" includes clipped at the edge; one
                                // more step brings the whole control in frame.
                                if scrolled > 0 {
                                    window.scroll("workspace-scroll", harness_scroll(-1.0), cx);
                                    window.draw(cx).clear(cx);
                                    scrolled += 1;
                                }
                                window.click(target, cx);
                            } else {
                                window.hover(SharedString::from(id.clone()), cx);
                            }
                        }
                        for _ in 0..tabs {
                            if let Ok(stroke) = Keystroke::parse("tab") {
                                window.dispatch_keystroke(stroke, cx);
                            }
                            window.draw(cx).clear(cx);
                        }
                        window.draw(cx).clear(cx);
                        scrolled
                    })?;
                    context.run_until_parked();
                    context.update_window(handle.into(), |_, window, cx| {
                        window.draw(cx).clear(cx);
                    })?;
                    context
                        .capture_screenshot(handle.into())?
                        .save(directory.join(format!("{name}-{appearance}-state-{state}.png")))?;
                    // Undo the scroll so later scenes start at the top.
                    if scrolled > 0 {
                        context.update_window(handle.into(), |_, window, cx| {
                            let steps = scrolled as f32;
                            window.scroll("workspace-scroll", harness_scroll(steps), cx);
                            window.draw(cx).clear(cx);
                        })?;
                    }
                }
            }
        }
        // A run in flight: the progress steps a first-time user waits on.
        for (label_name, status) in [("loading", Status::Preparing), ("running", Status::Running)] {
            context.update_window(handle.into(), |_, window, cx| {
                console.update(cx, |console, cx| {
                    console.appearance = AppearanceMode::Dark;
                    console.view = View::Experiment;
                    console.step = WorkspaceStep::Review;
                    console.status = status;
                    console.sync_kit_theme(cx);
                });
                window.draw(cx).clear(cx);
            })?;
            context.run_until_parked();
            context.update_window(handle.into(), |_, window, cx| {
                window.draw(cx).clear(cx);
            })?;
            context
                .capture_screenshot(handle.into())?
                .save(directory.join(format!("{name}-progress-{label_name}.png")))?;
        }
        context.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, _| console.status = Status::Idle);
        })?;
        // The built-in sample result, which needs no model.
        for (appearance, mode) in [
            ("light", AppearanceMode::Light),
            ("dark", AppearanceMode::Dark),
        ] {
            for result in [ResultView::Overview, ResultView::Layers, ResultView::Tokens] {
                context.update_window(handle.into(), |_, window, cx| {
                    console.update(cx, |console, cx| {
                        console.appearance = mode;
                        if !console.sample {
                            console.show_sample(cx);
                        }
                        console.result_view = result;
                        console.sync_kit_theme(cx);
                    });
                    window.draw(cx).clear(cx);
                })?;
                context.run_until_parked();
                context.update_window(handle.into(), |_, window, cx| {
                    window.draw(cx).clear(cx);
                })?;
                context
                    .capture_screenshot(handle.into())?
                    .save(directory.join(format!(
                        "{name}-{appearance}-sample-{}.png",
                        result.label().replace(' ', "-")
                    )))?;
            }
        }
        context.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, _| {
                console.sample = false;
                console.baseline = None;
                console.intervention = None;
                console.comparison = None;
                console.layer_series = Arc::from([]);
                console.result_context = None;
            });
        })?;
        if let Ok(model) = std::env::var("EMBER_GUI_TEST_MODEL") {
            let config = context.update_window(handle.into(), |_, _, cx| {
                console.update(cx, |console, _| {
                    console.model_path = model.clone();
                    console.prompt = "The capital of France is".into();
                    console.max_tokens = "4".into();
                    console.pending_context = Some(console.form_values());
                    parse_run_request(&console.form_values().build_run_request().unwrap()).unwrap()
                })
            })?;
            if real_replies.is_none() {
                eprintln!("Rendering real-model result scenes: preparing and running four tokens");
                let (worker, receiver) = spawn_worker(ember::quant_k::KStrategy::Auto, false);
                worker.send(WorkerMsg::Prepare(model))?;
                let prepared = receiver
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(300))?;
                if let WorkerReply::Prepared(info) = &prepared {
                    info.as_ref()
                        .as_ref()
                        .map_err(|error| anyhow::anyhow!(error.clone()))?;
                }
                worker.send(WorkerMsg::Run(config))?;
                let completed = receiver
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(300))?;
                if let WorkerReply::RunDone(result) = &completed {
                    let bundle = result
                        .as_ref()
                        .as_ref()
                        .map_err(|error| anyhow::anyhow!(error.clone()))?;
                    anyhow::ensure!(
                        bundle.verification.ok,
                        "render fixture bundle failed verification"
                    );
                    eprintln!("Render fixture: {}", bundle.baseline.bundle_dir);
                }
                real_replies = Some((prepared, completed));
            }
            let (prepared, completed) = real_replies.as_ref().unwrap();
            reply.send(prepared.clone())?;
            reply.send(completed.clone())?;
            context.update_window(handle.into(), |_, _, cx| {
                console.update(cx, |console, cx| {
                    console.drain_replies(cx);
                    cx.notify();
                });
            })?;
            for (appearance, mode) in [
                ("light", AppearanceMode::Light),
                ("dark", AppearanceMode::Dark),
            ] {
                for result in ResultView::ALL {
                    context.update_window(handle.into(), |_, window, cx| {
                        console.update(cx, |console, cx| {
                            console.appearance = mode;
                            console.step = WorkspaceStep::Review;
                            console.result_view = result;
                            console.sync_kit_theme(cx);
                            cx.notify();
                        });
                        window.draw(cx).clear(cx);
                    })?;
                    context.run_until_parked();
                    context.update_window(handle.into(), |_, window, cx| {
                        window.draw(cx).clear(cx);
                    })?;
                    for _ in 0..20 {
                        context.advance_clock(Duration::from_millis(50));
                        context.run_until_parked();
                        context.update_window(handle.into(), |_, window, cx| {
                            window.draw(cx).clear(cx);
                        })?;
                    }
                    context
                        .capture_screenshot(handle.into())?
                        .save(directory.join(format!(
                            "{name}-{appearance}-result-{}.png",
                            result.label().replace(' ', "-")
                        )))?;
                }
            }
            // Presentation mode over the results, once per window size.
            for result in [ResultView::Overview, ResultView::Layers] {
                context.update_window(handle.into(), |_, window, cx| {
                    console.update(cx, |console, cx| {
                        console.appearance = AppearanceMode::Dark;
                        console.result_view = result;
                        if console.presentation.is_none() {
                            console.toggle_presentation(cx);
                        }
                    });
                    window.draw(cx).clear(cx);
                })?;
                context.run_until_parked();
                context.update_window(handle.into(), |_, window, cx| {
                    window.draw(cx).clear(cx);
                })?;
                context
                    .capture_screenshot(handle.into())?
                    .save(directory.join(format!(
                        "{name}-presentation-{}.png",
                        result.label().replace(' ', "-")
                    )))?;
            }
            context.update_window(handle.into(), |_, _, cx| {
                console.update(cx, |console, cx| {
                    if console.presentation.is_some() {
                        console.toggle_presentation(cx);
                    }
                });
            })?;
        }
    }
    Ok(())
}

/// Bound on scroll steps while bringing a harness target into view.
#[cfg(all(target_os = "macos", feature = "gui-tests"))]
const HARNESS_MAX_SCROLL_STEPS: usize = 20;

/// One harness scroll step of 120px per unit; negative scrolls content up.
#[cfg(all(target_os = "macos", feature = "gui-tests"))]
fn harness_scroll(steps: f32) -> gpui_kit::ScrollDelta {
    gpui_kit::ScrollDelta::Pixels(point(px(0.0), px(120.0 * steps)))
}
