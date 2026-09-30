//! Headless UI tests (`gui-tests`) driving the real console.

use super::{Console, Preset, View, WorkspaceStep, FONT_ARABIC, FONT_MONO, FONT_SANS};
use gpui_kit::component::Root;
use gpui_kit::test::{TestAppContextExt, TestWindowExt};
use gpui_kit::{AppContext, SharedString, TestAppContext};
use std::{
    borrow::Cow,
    sync::{mpsc, Arc, Mutex},
};

#[gpui_kit::test]
async fn page_bottom_is_reachable_by_scrolling(cx: &mut TestAppContext) {
    cx.update(gpui_kit::init);
    let (tx, _worker) = mpsc::channel();
    let (_reply, rx) = mpsc::channel();
    let handle = cx.add_window(move |window, cx| {
        let console = cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(rx)), false, window, cx));
        console.update(cx, |console, _| {
            console.view = View::Experiment;
            console.step = WorkspaceStep::Prompt;
        });
        Root::new(console, window, cx)
    });
    // The standard window size, where the page is taller than the viewport.
    cx.simulate_window_resize(
        handle.into(),
        gpui_kit::size(gpui_kit::px(1180.0), gpui_kit::px(720.0)),
    );
    cx.run_until_parked();
    cx.wait_for(
        handle.into(),
        std::time::Duration::from_secs(2),
        |window, _| {
            window
                .try_find(SharedString::from("setup-scroll"))
                .is_some()
        },
    )
    .await;
    // Scroll far past the end; the container clamps at its maximum offset.
    cx.update_window(handle.into(), |_, window, cx| {
        for _ in 0..40 {
            window.scroll(
                SharedString::from("setup-scroll"),
                gpui_kit::ScrollDelta::Pixels(gpui_kit::point(
                    gpui_kit::px(0.0),
                    gpui_kit::px(-400.0),
                )),
                cx,
            );
            window.draw(cx).clear(cx);
        }
    })
    .expect("the window update runs");
    cx.run_until_parked();
    let (control_bottom, statusbar_top) = cx
        .update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            let control = window
                .find(SharedString::from("generation-length:24"))
                .bounds();
            let action = window.find(SharedString::from("setup-run")).bounds();
            let bottom: f32 = (control.origin.y + control.size.height).into();
            let top: f32 = action.origin.y.into();
            (bottom, top)
        })
        .expect("the window update runs");
    assert!(
        control_bottom <= statusbar_top,
        "fully scrolled, the last control ends at {control_bottom}px, under the status bar at {statusbar_top}px"
    );
}

#[gpui_kit::test]
fn primary_action_is_gated_for_both_mouse_and_keyboard(cx: &mut TestAppContext) {
    cx.update(gpui_kit::init);
    let (tx, _worker) = mpsc::channel();
    let (_reply, rx) = mpsc::channel();

    // Console::new needs a real Window, so the entity is built inside the
    // window closure and handed back out through a cell for assertions.
    let cell: Arc<Mutex<Option<gpui_kit::Entity<Console>>>> = Arc::default();
    let sink = cell.clone();
    let handle = cx.add_window(move |window, cx| {
        let console = cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(rx)), false, window, cx));
        *sink.lock().expect("cell unlocked") = Some(console.clone());
        Root::new(console, window, cx)
    });
    cx.update_window(handle.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
    })
    .unwrap();
    let console = cell
        .lock()
        .expect("cell unlocked")
        .clone()
        .expect("window construction registers the Console");

    // Idle: the action is eligible and advances the workspace.
    let idle = cx.update(|cx| {
        let console = console.read(cx);
        (console.action_enabled(), console.busy(), console.step)
    });
    assert!(idle.0, "an idle console must allow the primary action");
    assert!(!idle.1, "an idle console must not report busy");

    // The workspace has no steps to advance: an eligible action runs, which
    // with no model loaded starts by preparing one.
    cx.update(|cx| console.update(cx, |console, _| console.advance_or_run()));
    let status_after = cx.update(|cx| console.read(cx).status);
    assert_eq!(
        status_after,
        super::Status::Preparing,
        "an eligible action must start the run"
    );
    cx.update(|cx| console.update(cx, |console, _| console.status = super::Status::Idle));

    // Busy: a run is in flight, so the button is disabled and the
    // shortcut must be refused rather than advancing the step anyway.
    cx.update(|cx| {
        console.update(cx, |console, _| {
            console.status = super::Status::Running;
        })
    });
    let busy = cx.update(|cx| {
        let console = console.read(cx);
        (console.action_enabled(), console.busy(), console.step)
    });
    assert!(busy.1, "the console must report busy while running");
    assert!(
        !busy.0,
        "a busy console must disable the primary action (and its shortcut)"
    );

    cx.update(|cx| console.update(cx, |console, _| console.advance_or_run()));
    let after_busy = cx.update(|cx| console.read(cx).step);
    assert_eq!(
        busy.2, after_busy,
        "the keyboard path must not advance the workspace while busy"
    );
}

#[gpui_kit::test]
fn populated_chart_registers_handlers_during_paint(cx: &mut TestAppContext) {
    use gpui_kit::{Context, IntoElement, Render, Window};
    struct ChartFixture(gpui_kit::Entity<Console>);
    impl Render for ChartFixture {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            super::chart::layer_divergence_chart(
                self.0.clone(),
                Arc::from(vec![crate::gui::LayerMetric {
                    layer: 8,
                    relative_l2_difference: Some(0.5),
                    cosine_distance: Some(0.01),
                    maximum_absolute_difference: Some(1.0),
                    exact: false,
                }]),
                None,
                super::chart::LayerMarks {
                    intervention: Some(8),
                    ..Default::default()
                },
                160.,
                &super::theme::light(),
            )
        }
    }
    cx.update(gpui_kit::init);
    let (tx, _worker) = mpsc::channel();
    let (_reply, rx) = mpsc::channel();
    let handle = cx.add_window(|window, cx| {
        let console = cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(rx)), false, window, cx));
        let chart = cx.new(|_| ChartFixture(console));
        Root::new(chart, window, cx)
    });
    cx.update_window(handle.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
    })
    .unwrap();
}

#[gpui_kit::test]
async fn opening_a_saved_run_reopens_its_comparison(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, _worker_rx) = mpsc::channel();
    let (_reply_tx, reply_rx) = mpsc::channel();
    let mut view = None;
    let handle = cx.add_window(|window, cx| {
        let console =
            cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
        view = Some(console.clone());
        Root::new(console, window, cx)
    });
    let console = view.unwrap();
    cx.update_window(handle.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
        console.update(cx, |console, cx| {
            let mut run = super::seed_store().runs[0].clone();
            run.number = 77;
            run.result = Some(super::app_store::RecordResult {
                baseline_text: "Paris.".into(),
                intervention_text: "fog".into(),
                layers: vec![super::app_store::RecordLayer {
                    layer: 7,
                    relative_l2: Some(1.25),
                    cosine: Some(0.5),
                }],
                tokens: Vec::new(),
                first_layer_divergence: Some(7),
                peak_layer: Some(7),
                peak_relative_l2: Some(1.25),
                tokens_equal: false,
            });
            run.config = Some(super::app_store::RecordConfig {
                model_path: "fixture.gguf".into(),
                execution: "reference".into(),
                site: "after-layer".into(),
                layer: "6".into(),
                op: "scale".into(),
                value: "0.0".into(),
                source: "capture".into(),
                source_layer: "0".into(),
                token: "prompt-final".into(),
                span: String::new(),
                max_tokens: "24".into(),
            });
            console.store.push_run(run);
            console.open_run(77, cx);
        });
        let console = console.read(cx);
        assert_eq!(console.saved_run, Some(77));
        assert!(console.sample, "a reopened run is not a live result");
        assert_eq!(console.step, WorkspaceStep::Review);
        assert_eq!(console.layer_series.len(), 1);
        assert_eq!(console.selected_layer, Some(7));
        assert_eq!(console.layer, "6");
        let markdown = console
            .result_markdown()
            .expect("a reopened run can be copied");
        assert!(markdown.contains("Run #77, reopened from history"));
    })
    .unwrap();
}

async fn console_window(
    cx: &mut TestAppContext,
) -> (gpui_kit::AnyWindowHandle, gpui_kit::Entity<Console>) {
    cx.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, _worker_rx) = mpsc::channel();
    let (_reply_tx, reply_rx) = mpsc::channel();
    let mut view = None;
    let handle = cx.add_window(|window, cx| {
        let console =
            cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
        view = Some(console.clone());
        Root::new(console, window, cx)
    });
    (handle.into(), view.unwrap())
}

#[gpui_kit::test]
async fn deleting_a_run_from_the_table_takes_a_confirming_click(cx: &mut TestAppContext) {
    let (handle, console) = console_window(cx).await;
    cx.update_window(handle, |_, window, cx| {
        console.update(cx, |console, cx| {
            let mut run = super::seed_store().runs[0].clone();
            run.number = 5;
            console.store.push_run(run);
            console.goto(View::Runs, cx);
        });
        window.draw(cx).clear(cx);
        window.click(SharedString::from("run-delete:5"), cx);
        assert_eq!(
            console.read(cx).store.runs.len(),
            1,
            "one click on Delete must not delete"
        );
        window.render_frame(cx);
        window.click(SharedString::from("run-delete-cancel:5"), cx);
        window.render_frame(cx);
        window.click(SharedString::from("run-delete:5"), cx);
        window.render_frame(cx);
        window.click(SharedString::from("run-delete-confirm:5"), cx);
        assert!(console.read(cx).store.runs.is_empty(), "confirmed, it goes");
    })
    .unwrap();
}

/// The Runs page syncs its table on every render. That sync used to clone
/// every record, results and all; now a render with no history change must
/// not rebuild the rows at all.
#[gpui_kit::test]
async fn a_runs_render_without_a_store_change_rebuilds_nothing(cx: &mut TestAppContext) {
    let (handle, console) = console_window(cx).await;
    cx.update_window(handle, |_, window, cx| {
        console.update(cx, |console, cx| {
            let mut run = super::seed_store().runs[0].clone();
            run.number = 5;
            console.store.push_run(run);
            console.goto(View::Runs, cx);
        });
        window.draw(cx).clear(cx);
        let rebuilds = |cx: &mut gpui_kit::App| {
            let table = console.read(cx).runs_table.clone().expect("Runs was drawn");
            table.read(cx).delegate().rebuilds
        };
        let first = rebuilds(cx);
        for _ in 0..3 {
            console.update(cx, |_, cx| cx.notify());
            window.render_frame(cx);
        }
        assert_eq!(rebuilds(cx), first, "unchanged history, no rebuild");
        window.click(SharedString::from("run-pin:5"), cx);
        window.render_frame(cx);
        assert_eq!(rebuilds(cx), first + 1, "a pin rebuilds once");
        assert!(console.read(cx).store.runs[0].pinned);
    })
    .unwrap();
}

#[gpui_kit::test]
async fn the_poll_loop_ends_when_the_console_is_released(cx: &mut TestAppContext) {
    use std::future::Future;
    let (handle, console) = console_window(cx).await;
    let mut task = console.update(cx, |console, cx| console.poll_task(cx));
    drop(console);
    cx.run_until_parked();
    cx.update_window(handle, |_, window, _| window.remove_window())
        .unwrap();
    cx.run_until_parked();
    cx.executor()
        .advance_clock(std::time::Duration::from_secs(1));
    cx.run_until_parked();
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(
        std::pin::Pin::new(&mut task).poll(&mut context).is_ready(),
        "the loop outlived the console"
    );
}

#[gpui_kit::test]
async fn discovered_models_fill_an_empty_selection_only(cx: &mut TestAppContext) {
    let (handle, console) = console_window(cx).await;
    cx.update_window(handle, |_, window, cx| {
        console.update(cx, |console, cx| {
            assert!(console.model_options.is_empty(), "no scan on the UI thread");
            console.models_discovered(
                vec![
                    ("/m/a-Q8_0.gguf".into(), Some(2048)),
                    ("/m/b.gguf".into(), None),
                ],
                cx,
            );
            assert_eq!(console.model_path, "/m/a-Q8_0.gguf");
            assert_eq!(console.model_sizes["/m/a-Q8_0.gguf"], Some(2048));
            console.model_path = "/mine.gguf".into();
            console.models_discovered(vec![("/m/b.gguf".into(), None)], cx);
            assert_eq!(console.model_path, "/mine.gguf", "a user's choice stays");
            console.goto(View::Models, cx);
        });
        // The Models page renders from the cached sizes.
        window.draw(cx).clear(cx);
    })
    .unwrap();
}

#[gpui_kit::test]
async fn a_failed_history_write_reaches_the_status_bar(cx: &mut TestAppContext) {
    let (handle, console) = console_window(cx).await;
    let dir = std::env::temp_dir().join(format!("ember-kit-store-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let good = dir.join("app-state.v2.json");
    std::fs::write(dir.join("file"), b"").unwrap();
    let bad = dir.join("file").join("app-state.v2.json");
    cx.update_window(handle, |_, _, cx| {
        console.update(cx, |console, _| {
            // A readable store: the write lands, merged, off the UI thread.
            console.store_path = Some(good.clone());
            let number = console.store.next_run_number();
            let mut run = super::seed_store().runs[0].clone();
            run.number = number;
            console.store.push_run(run);
            console.persist();
            console.flush_store();
            assert!(console.drain_store_writes());
            assert!(console.store_error.is_none());
            assert_eq!(super::app_store::load(&good).unwrap().runs.len(), 1);

            // An unwritable one: the failure comes back as a message.
            console.store_writer = None;
            console.store_path = Some(bad.clone());
            console.persist();
            console.flush_store();
            assert!(console.drain_store_writes());
            assert!(console
                .store_error
                .as_deref()
                .is_some_and(|error| error.contains("could not save run history")));
        });
    })
    .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_menu_bar_has_the_expected_menus_and_none_are_empty() {
    let menus = super::app_menus();
    let names: Vec<_> = menus.iter().map(|menu| menu.name.to_string()).collect();
    assert_eq!(names, ["Ember", "Edit", "Experiment", "View", "Help"]);
    assert!(menus.iter().all(|menu| !menu.items.is_empty()));
}

#[gpui_kit::test]
async fn menu_commands_reach_the_console(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, _worker_rx) = mpsc::channel();
    let (_reply_tx, reply_rx) = mpsc::channel();
    let mut view = None;
    let handle = cx.add_window(|window, cx| {
        let console =
            cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
        view = Some(console.clone());
        Root::new(console, window, cx)
    });
    let console = view.unwrap();
    cx.update(|cx| super::register_menu_actions(console.downgrade(), handle.into(), cx));
    cx.update_window(handle.into(), |_, window, cx| window.draw(cx).clear(cx))
        .unwrap();
    let before = console.read_with(cx, |c, _| c.sidebar_open);
    cx.dispatch_action(handle.into(), super::HideShowSidebar);
    assert_ne!(console.read_with(cx, |c, _| c.sidebar_open), before);
    cx.dispatch_action(handle.into(), super::OpenSettings);
    assert_eq!(console.read_with(cx, |c, _| c.view), View::Settings);
    cx.dispatch_action(handle.into(), super::OpenSampleResult);
    assert!(console.read_with(cx, |c, _| c.sample));
    cx.dispatch_action(handle.into(), super::OpenPalette);
    cx.run_until_parked();
    assert!(console.read_with(cx, |c, _| c.palette_open));
}

#[gpui_kit::test]
async fn editing_a_setting_marks_the_result_stale_and_a_result_can_be_pinned(
    cx: &mut TestAppContext,
) {
    cx.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(super::FONT_SANS),
                Cow::Borrowed(super::FONT_MONO),
                Cow::Borrowed(super::FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, _worker_rx) = mpsc::channel();
    let (_reply_tx, reply_rx) = mpsc::channel();
    let mut view = None;
    let handle = cx.add_window(|window, cx| {
        let console =
            cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
        view = Some(console.clone());
        Root::new(console, window, cx)
    });
    let console = view.unwrap();
    cx.update_window(handle.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
        console.update(cx, |console, cx| {
            console.show_sample(cx);
            // Treat it as a live result of these settings.
            console.sample = false;
            assert!(
                !console.results_stale(),
                "a fresh result matches its settings"
            );
            console.layer = "9".into();
            assert!(
                console.results_stale(),
                "editing a setting must mark the result stale"
            );
            console.layer = "8".into();
            assert!(!console.results_stale(), "restoring the setting clears it");

            assert!(console.reference.is_none());
            console.pin_reference(cx);
            let reference = console.reference.as_ref().expect("pinned");
            assert!(reference.label.contains("Change strength"));
            assert!(!reference.layers.is_empty());
            console.clear_reference(cx);
            assert!(console.reference.is_none());
        });
    })
    .unwrap();
}

#[gpui_kit::test]
async fn sample_result_opens_review_and_copies_markdown(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, _worker_rx) = mpsc::channel();
    let (_reply_tx, reply_rx) = mpsc::channel();
    let mut view = None;
    let handle = cx.add_window(|window, cx| {
        let console =
            cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
        view = Some(console.clone());
        Root::new(console, window, cx)
    });
    let console = view.unwrap();
    cx.update_window(handle.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
        window.click(SharedString::from("home-sample"), cx);
        {
            let console = console.read(cx);
            assert!(console.sample, "the sample flag marks illustrative data");
            assert_eq!(console.step, WorkspaceStep::Review);
            assert!(console.baseline.is_some() && console.comparison.is_some());
        }
        window.render_frame(cx);
        window.click(SharedString::from("review-copy"), cx);
        assert!(console.read(cx).copied);
    })
    .unwrap();
    let copied = cx
        .read_from_clipboard()
        .and_then(|item| item.text())
        .expect("the summary is on the clipboard");
    assert!(copied.starts_with("# Ember experiment"));
    assert!(
        copied.contains("Sample result"),
        "a sample must say so in the copy"
    );
    assert!(copied.contains("| 8 | 0.2960 | 0.0450 |"));
    assert!(copied.contains("First internal divergence:** layer 8"));
    // A real run must not inherit the sample's label or leave it copied.
    cx.update_window(handle.into(), |_, _, cx| {
        console.update(cx, |console, _| {
            console.send_run(
                crate::gui::parse_run_request(&console.build_run_request().unwrap()).unwrap(),
            );
            assert!(!console.sample);
        });
    })
    .unwrap();
}

#[gpui_kit::test]
async fn kit_navigation_and_presets_update_experiment_state(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, worker_rx) = mpsc::channel();
    let (reply_tx, reply_rx) = mpsc::channel();
    let mut view = None;
    let handle = cx.add_window(|window, cx| {
        let console =
            cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
        console.update(cx, |console, cx| {
            console.model_path = "fixture.gguf".into();
            console
                .inputs
                .model
                .update(cx, |input, cx| input.set_value("fixture.gguf", cx));
        });
        view = Some(console.clone());
        Root::new(console, window, cx)
    });
    let console = view.unwrap();
    cx.update_window(handle.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
        window.click(SharedString::from("nav:experiments"), cx);
        console.update(cx, |console, cx| {
            console.select_combo(super::ComboId::Op, "zero", cx)
        });
        assert_eq!(console.read(cx).op, "zero");
        console.update(cx, |console, cx| {
            console.apply_preset(Preset::ArabicMorphology, cx)
        });
        window.render_frame(cx);
        let console = console.read(cx);
        assert!(!console.prompt.is_ascii());
        assert!(console.build_run_request().is_ok());
    })
    .unwrap();
    cx.update_window(handle.into(), |_, window, cx| {
        let input = console.read(cx).inputs.prompt.clone();
        input.update(cx, |input, cx| input.focus_for_test(window, cx));
        window.press(
            if cfg!(target_os = "macos") {
                "cmd-a"
            } else {
                "ctrl-a"
            },
            cx,
        );
        window.input("مرحبا Ember\nاختبار", cx);
    })
    .unwrap();
    cx.update_window(handle.into(), |_, window, cx| {
        assert_eq!(console.read(cx).prompt, "مرحبا Ember\nاختبار");
        window.press(
            if cfg!(target_os = "macos") {
                "cmd-z"
            } else {
                "ctrl-z"
            },
            cx,
        );
    })
    .unwrap();
    cx.update_window(handle.into(), |_, window, cx| {
        assert_ne!(console.read(cx).prompt, "مرحبا Ember\nاختبار");
        window.click(SharedString::from("nav:experiments"), cx);
        // The advanced controls sit in the setup pane behind a disclosure.
        assert!(
            window
                .try_find(SharedString::from("execution-picker"))
                .is_none(),
            "the advanced controls start collapsed"
        );
        // The control sits below the fold of a short window: scroll it in.
        window.scroll(
            SharedString::from("setup-scroll"),
            gpui_kit::ScrollDelta::Pixels(gpui_kit::point(
                gpui_kit::px(0.0),
                gpui_kit::px(-2000.0),
            )),
            cx,
        );
        window.draw(cx).clear(cx);
        window.click(SharedString::from("setup-advanced"), cx);
        window.click(SharedString::from("picker:Site"), cx);
    })
    .unwrap();
    cx.run_until_parked();
    cx.update_window(handle.into(), |_, window, cx| {
        window.input("before-logits", cx);
    })
    .unwrap();
    cx.wait_for(
        handle.into(),
        std::time::Duration::from_secs(1),
        |window, _| {
            window
                .try_find(SharedString::from("choice:before-logits"))
                // Human-facing wording, not the frozen hook identifier:
                // the picker shows "Before output head" while the value
                // committed to the spec stays "before-logits".
                .is_some_and(|item| item.label() == Some("Before output head"))
        },
    )
    .await;
    cx.update_window(handle.into(), |_, window, cx| {
        window.press("down", cx);
        window.press("enter", cx);
    })
    .unwrap();
    cx.run_until_parked();
    cx.update_window(handle.into(), |_, window, cx| {
            assert_eq!(console.read(cx).site, "before-logits");
            window.click(SharedString::from("setup-run"), cx);
            assert_eq!(console.read(cx).status, super::Status::Preparing);
            assert!(matches!(worker_rx.try_recv().unwrap(), super::WorkerMsg::Prepare(path) if path == "fixture.gguf"));
            let loading = SharedString::from("setup-run");
            assert!(window.find(loading.clone()).visible());
            window.click(loading, cx);
            assert!(worker_rx.try_recv().is_err(), "disabled run must not submit another job");
            reply_tx.send(super::WorkerReply::Prepared(Box::new(Err("fixture load failure".into())))).unwrap();
            console.update(cx, |console, cx| { console.drain_replies(cx); cx.notify(); });
            window.render_frame(cx);
            assert_eq!(console.read(cx).status, super::Status::Idle);
            assert_eq!(console.read(cx).error.as_deref(), Some("fixture load failure"));
            window.click(SharedString::from("setup-run"), cx);
            assert!(matches!(worker_rx.try_recv().unwrap(), super::WorkerMsg::Prepare(_)));
        }).unwrap();
}

/// A fake loaded model, so a run goes straight to the worker.
fn fake_session() -> crate::gui::SessionInfo {
    crate::gui::SessionInfo {
        model_path: "fixture.gguf".into(),
        model_name: "fixture".into(),
        architecture: "llama".into(),
        n_layers: 16,
        embed_dim: 64,
        vocab_size: 128,
        model_sha: String::new(),
        tokenizer_sha: String::new(),
        load_ms: 0.0,
    }
}

/// A verified-looking pair, for replies the tests inject.
fn fake_bundle() -> crate::gui::RunBundle {
    let (baseline, intervention, comparison, _) = super::sample_result();
    crate::gui::RunBundle {
        baseline,
        intervention,
        comparison,
        verification: ember::v05::verify::VerificationReport {
            bundle_schema: String::new(),
            ok: true,
            semantic_hash: String::new(),
            payload_hash: String::new(),
            checks: Vec::new(),
            warnings: Vec::new(),
            timestamp: String::new(),
        },
        elapsed_ms_total: 1.0,
        elapsed_ms_baseline: 1.0,
        baseline_key: String::new(),
    }
}

type WorkerEnds = (
    gpui_kit::AnyWindowHandle,
    gpui_kit::Entity<Console>,
    mpsc::Receiver<super::WorkerMsg>,
    mpsc::Sender<super::WorkerReply>,
);

/// A console whose worker channel the test holds, with a model "loaded".
async fn console_with_worker(cx: &mut TestAppContext) -> WorkerEnds {
    cx.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, worker_rx) = mpsc::channel();
    let (reply_tx, reply_rx) = mpsc::channel();
    let mut view = None;
    let handle = cx.add_window(|window, cx| {
        let console =
            cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
        console.update(cx, |console, cx| {
            console.model_path = "fixture.gguf".into();
            console
                .inputs
                .model
                .update(cx, |input, cx| input.set_value("fixture.gguf", cx));
            console.session = Some(fake_session());
            console.view = View::Experiment;
        });
        view = Some(console.clone());
        Root::new(console, window, cx)
    });
    (handle.into(), view.unwrap(), worker_rx, reply_tx)
}

#[gpui_kit::test]
async fn cancelling_a_run_stops_it_and_records_nothing(cx: &mut TestAppContext) {
    let (handle, console, worker_rx, reply_tx) = console_with_worker(cx).await;
    cx.update_window(handle, |_, window, cx| {
        window.draw(cx).clear(cx);
        assert!(
            window
                .try_find(SharedString::from("setup-cancel"))
                .is_none(),
            "nothing to cancel while idle"
        );
        window.click(SharedString::from("setup-run"), cx);
        assert_eq!(console.read(cx).status, super::Status::Running);
        let Ok(super::WorkerMsg::Run(_, token)) = worker_rx.try_recv() else {
            panic!("the run went to the worker with a cancel token");
        };
        assert!(!token.is_cancelled());
        window.render_frame(cx);

        // The button cancels: the token fires at once, and the console waits
        // for the worker to confirm rather than claiming it stopped.
        window.click(SharedString::from("setup-cancel"), cx);
        assert!(token.is_cancelled(), "Cancel fires the run's token");
        assert_eq!(console.read(cx).status, super::Status::Cancelling);
        assert!(
            !console.read(cx).action_enabled(),
            "no new run until it stops"
        );
        reply_tx.send(super::WorkerReply::Cancelled).unwrap();
        console.update(cx, |console, cx| {
            console.drain_replies(cx);
        });
        window.render_frame(cx);
        {
            let console = console.read(cx);
            assert_eq!(console.status, super::Status::Idle);
            assert!(console.cancelled, "the page says it was cancelled");
            assert!(console.error.is_none(), "a cancel is not an error");
            assert!(console.store.runs.is_empty(), "nothing recorded");
            assert!(console.session.is_some(), "the model stays loaded");
            assert!(console.baseline.is_none());
        }
        assert!(window
            .try_find(SharedString::from("run-cancelled"))
            .is_some());

        // Esc does the same, and a result that raced the cancel is dropped.
        window.click(SharedString::from("setup-run"), cx);
        let Ok(super::WorkerMsg::Run(_, token)) = worker_rx.try_recv() else {
            panic!("second run");
        };
        assert!(!console.read(cx).cancelled, "a new run clears the notice");
        window.press("escape", cx);
        assert!(token.is_cancelled(), "Esc cancels the run in flight");
        reply_tx
            .send(super::WorkerReply::RunDone(Box::new(Ok(fake_bundle()))))
            .unwrap();
        console.update(cx, |console, cx| {
            console.drain_replies(cx);
        });
        let console = console.read(cx);
        assert_eq!(console.status, super::Status::Idle);
        assert!(console.store.runs.is_empty(), "a raced result is not kept");
        assert!(console.baseline.is_none(), "nor shown");
    })
    .unwrap();

    // The palette command reaches the same path.
    cx.update_window(handle, |_, _, cx| {
        console.update(cx, |console, cx| {
            console.run();
            let Ok(super::WorkerMsg::Run(_, token)) = worker_rx.try_recv() else {
                panic!("third run");
            };
            console.palette_open = true;
            console.palette_query = "cancel".into();
            console.palette_index = 0;
            assert_eq!(
                console.palette_candidates().first(),
                Some(&super::palette::Command::CancelRun)
            );
            console.palette_execute(cx);
            assert!(token.is_cancelled());
        });
    })
    .unwrap();
}

#[gpui_kit::test]
async fn two_saved_runs_can_be_compared_and_old_records_say_they_cannot(cx: &mut TestAppContext) {
    let (handle, console) = console_window(cx).await;
    cx.update_window(handle, |_, window, cx| {
        console.update(cx, |console, cx| {
            // Seeded: the two newest runs (#7, #6) kept results, #5 did not.
            console.store = super::seed_store();
            console.goto(View::Runs, cx);
        });
        window.draw(cx).clear(cx);
        assert!(window.try_find(SharedString::from("compare-bar")).is_none());
        window.click(SharedString::from("run-select:7"), cx);
        window.render_frame(cx);
        assert_eq!(console.read(cx).compare_picks, [7]);
        assert!(window.try_find(SharedString::from("compare-bar")).is_some());

        // A record without a result is refused, with the reason.
        window.click(SharedString::from("run-select:5"), cx);
        window.render_frame(cx);
        let blocker = console.read(cx).compare_blocker().expect("refused");
        assert!(blocker.contains("Run #5") && blocker.contains("can't be compared"));
        window.click(SharedString::from("runs-compare-open"), cx);
        assert!(console.read(cx).comparing.is_none(), "refused");

        // Deselect it and pick a comparable one instead.
        window.click(SharedString::from("run-select:5"), cx);
        window.render_frame(cx);
        window.click(SharedString::from("run-select:6"), cx);
        window.render_frame(cx);
        assert_eq!(console.read(cx).compare_picks, [7, 6]);
        assert!(console.read(cx).compare_blocker().is_none());
        window.click(SharedString::from("runs-compare-open"), cx);
        assert_eq!(console.read(cx).comparing, Some((7, 6)));
        window.render_frame(cx);
        assert!(window
            .try_find(SharedString::from("compare-metrics"))
            .is_some());
        assert!(window
            .try_find(SharedString::from("compare-tokens"))
            .is_some());

        // Swap flips the sides; Back returns to the table with the picks kept.
        window.click(SharedString::from("compare-swap"), cx);
        assert_eq!(console.read(cx).comparing, Some((6, 7)));
        window.render_frame(cx);
        window.click(SharedString::from("compare-back"), cx);
        window.render_frame(cx);
        assert!(console.read(cx).comparing.is_none());
        assert!(window
            .try_find(SharedString::from("run-select:7"))
            .is_some());
    })
    .unwrap();
}

#[gpui_kit::test]
async fn any_history_row_exports_markdown_and_its_bundle(cx: &mut TestAppContext) {
    let (handle, console, worker_rx, reply_tx) = console_with_worker(cx).await;
    let root = std::env::temp_dir().join(format!("ember-kit-export-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (kept_base, kept_int) = (root.join("kept-base"), root.join("kept-int"));
    std::fs::create_dir_all(&kept_base).unwrap();
    std::fs::create_dir_all(&kept_int).unwrap();
    let bundles = |base: &std::path::Path, int: &std::path::Path| super::app_store::RecordBundles {
        baseline: base.display().to_string(),
        intervention: int.display().to_string(),
    };
    cx.update_window(handle, |_, window, cx| {
        console.update(cx, |console, cx| {
            // Seeded: #7 kept result and config, #6 a result only, #5 neither.
            console.store = super::seed_store();
            let seven = console
                .store
                .runs
                .iter_mut()
                .find(|r| r.number == 7)
                .unwrap();
            seven.bundles = Some(bundles(&kept_base, &kept_int));
            let six = console
                .store
                .runs
                .iter_mut()
                .find(|r| r.number == 6)
                .unwrap();
            six.bundles = Some(bundles(&root.join("gone-a"), &root.join("gone-b")));
            console.goto(View::Runs, cx);
        });
        window.draw(cx).clear(cx);

        // A run whose bundle is on disk: Markdown, reveal, verify command.
        window.click(SharedString::from("run-export:7"), cx);
        window.render_frame(cx);
        assert!(window
            .try_find(SharedString::from("export-reveal"))
            .is_some());
        assert!(window
            .try_find(SharedString::from("export-rebundle"))
            .is_none());
        window.click(SharedString::from("export-markdown"), cx);
    })
    .unwrap();
    let copied = cx.read_from_clipboard().and_then(|i| i.text()).unwrap();
    assert!(copied.starts_with("# Ember experiment"));
    assert!(copied.contains("Run #7, from history"));
    assert!(copied.contains("## Divergence by layer"));
    assert!(copied.contains(&format!(
        "ember experiment verify '{}'",
        kept_base.display()
    )));
    cx.update_window(handle, |_, window, cx| {
        window.click(SharedString::from("export-verify"), cx);
    })
    .unwrap();
    let command = cx.read_from_clipboard().and_then(|i| i.text()).unwrap();
    assert!(command.contains(&kept_int.display().to_string()));

    cx.update_window(handle, |_, window, cx| {
        // An old record with neither result nor configuration: it still
        // exports Markdown, and says it cannot be re-run.
        window.click(SharedString::from("run-export:5"), cx);
        window.render_frame(cx);
        assert!(window
            .try_find(SharedString::from("export-rebundle"))
            .is_none());
        assert!(window
            .try_find(SharedString::from("export-reveal"))
            .is_none());
        window.click(SharedString::from("export-markdown"), cx);
    })
    .unwrap();
    let copied = cx.read_from_clipboard().and_then(|i| i.text()).unwrap();
    assert!(copied.contains("Run #5, from history"));

    // #7's bundle goes missing; it kept its configuration, so it re-runs.
    std::fs::remove_dir_all(&kept_int).unwrap();
    let before = cx.update(|cx| console.read(cx).store.runs.len());
    cx.update_window(handle, |_, window, cx| {
        window.click(SharedString::from("run-export:7"), cx);
        window.render_frame(cx);
        assert!(window
            .try_find(SharedString::from("export-reveal"))
            .is_none());
        window.click(SharedString::from("export-rebundle"), cx);
        assert_eq!(console.read(cx).status, super::Status::Running);
        assert_eq!(console.read(cx).rebundle, Some(7));
        let Ok(super::WorkerMsg::Run(config, _)) = worker_rx.try_recv() else {
            panic!("the stored configuration went to the worker");
        };
        assert_eq!(config.model_path, "/models/Llama-3.2-1B-Instruct-Q8_0.gguf");
        window.render_frame(cx);
        assert!(window
            .try_find(SharedString::from("export-rebundle-cancel"))
            .is_some());
    })
    .unwrap();
    let (fresh_base, fresh_int) = (root.join("fresh-base"), root.join("fresh-int"));
    std::fs::create_dir_all(&fresh_base).unwrap();
    std::fs::create_dir_all(&fresh_int).unwrap();
    let mut bundle = fake_bundle();
    bundle.baseline.bundle_dir = fresh_base.display().to_string();
    bundle.intervention.bundle_dir = fresh_int.display().to_string();
    reply_tx
        .send(super::WorkerReply::RunDone(Box::new(Ok(bundle))))
        .unwrap();
    cx.update_window(handle, |_, window, cx| {
        console.update(cx, |console, cx| {
            console.drain_replies(cx);
        });
        window.render_frame(cx);
        let console = console.read(cx);
        assert_eq!(console.status, super::Status::Idle);
        assert_eq!(console.store.runs.len(), before, "no new history row");
        assert!(console.baseline.is_none(), "nothing changes on screen");
        let seven = console.store.runs.iter().find(|r| r.number == 7).unwrap();
        let kept = seven.bundles.as_ref().unwrap();
        assert!(kept.exist(), "the record points at the new bundles");
        assert!(kept.intervention.ends_with("fresh-int"));
        assert!(console
            .export_note
            .as_deref()
            .is_some_and(|note| note.contains("wrote a verified bundle")));
        assert!(window
            .try_find(SharedString::from("export-reveal"))
            .is_some());
    })
    .unwrap();
    let _ = std::fs::remove_dir_all(&root);
}
