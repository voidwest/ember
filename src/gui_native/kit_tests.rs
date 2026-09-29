//! Headless UI tests (`gui-tests`) driving the real console.

use super::{Console, Preset, View, WorkspaceStep, FONT_ARABIC, FONT_MONO, FONT_SANS};
use gpui_kit::component::Root;
use gpui_kit::test::{TestAppContextExt, TestWindowExt};
use gpui_kit::{AppContext, SharedString, TestAppContext};
use std::{
    borrow::Cow,
    sync::{mpsc, Arc, Mutex},
};

/// The inspector declares 300px. This asserts the layout engine agrees.
///
/// It was rendering at roughly 118px at the standard 1180pt window, clipping
/// the model name, the hook and the advanced disclosure mid-word on every
/// experiment screen. Four hypotheses were tried and ruled out by reading
/// screenshots: `min_w(0)` on the aside, `flex_shrink_0` on it, `w_full()`
/// on the workspace column, and `overflow_x_hidden` on the scroll container.
/// Inferring layout from a picture of it does not work.
///
/// **What this does and does not prove.** It passes, so the declaration is
/// honoured and the 300px is not being ignored outright. But the test window
/// is wide enough that the row never has to overflow, which is exactly the
/// condition the render fails under -- so this does not reproduce the bug.
/// Closing it needs this test to drive a narrow window, and then whatever it
/// reports is the number to fix against.
///
/// The aside also needed `.test_support()`. Without it its id was only a
/// scope inside its children's paths, so a test could find the controls
/// inside it but never the box itself -- which is why a layout bug this
/// visible had no test standing behind it.
/// The bottom of a page must be reachable. The workspace used `h_full()`
/// inside a column that also holds the stepper, so it came out one stepper
/// too tall: scrolled all the way down, the last control (generation length
/// on Prompt) still sat under the status bar, clipped.
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
                .try_find(SharedString::from("workspace-scroll"))
                .is_some()
        },
    )
    .await;
    // Scroll far past the end; the container clamps at its maximum offset.
    cx.update_window(handle.into(), |_, window, cx| {
        for _ in 0..40 {
            window.scroll(
                SharedString::from("workspace-scroll"),
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
            let action = window
                .find(SharedString::from("btn:Continue: Intervention"))
                .bounds();
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
async fn inspector_keeps_its_declared_width(cx: &mut TestAppContext) {
    cx.update(gpui_kit::init);
    let (tx, _worker) = mpsc::channel();
    let (_reply, rx) = mpsc::channel();
    let cell: Arc<Mutex<Option<gpui_kit::Entity<Console>>>> = Arc::default();
    let sink = cell.clone();
    let handle = cx.add_window(move |window, cx| {
        let console = cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(rx)), false, window, cx));
        console.update(cx, |console, _| {
            console.view = View::Experiment;
            console.inspector_open = true;
        });
        *sink.lock().expect("cell unlocked") = Some(console.clone());
        Root::new(console, window, cx)
    });
    cx.run_until_parked();
    cx.update_window(handle.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
    })
    .unwrap();
    // The aside appears once the row that owns it has been laid out, which
    // is a later frame than the first. `find` on an earlier frame reports a
    // miss even though the id is registered, so wait for it rather than
    // reading the first frame and concluding the inspector is not there.
    cx.wait_for(
        handle.into(),
        std::time::Duration::from_secs(2),
        |window, _| window.try_find(SharedString::from("inspector")).is_some(),
    )
    .await;

    let width = cx
        .update_window(handle.into(), |_, window, _| {
            window
                .find(SharedString::from("inspector"))
                .bounds()
                .size
                .width
        })
        .expect("the window update runs");
    // Compared in f32: `Pixels` has no `abs` in this version.
    let rendered: f32 = width.into();
    assert!(
        (rendered - super::views::INSPECTOR_WIDTH).abs() < 2.0,
        "inspector rendered at {rendered}px, not the {}px it declares",
        super::views::INSPECTOR_WIDTH
    );
}

/// The primary action must be gated identically by the button and by the
/// Ctrl+Enter shortcut.
///
/// The keyboard path previously advanced the workspace with no gate at
/// all, so a visibly disabled button could still be driven from the
/// keyboard -- and the UI advertises "Ctrl+Enter" right next to it.
/// While a run is in flight the button is disabled; the shortcut must
/// refuse too.
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

    let step_before = idle.2;
    cx.update(|cx| console.update(cx, |console, _| console.advance_or_run()));
    let step_after = cx.update(|cx| console.read(cx).step);
    assert_ne!(
        step_before, step_after,
        "an eligible action must advance the workspace"
    );

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
                Some(8),
                None,
                None,
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
        window.click(SharedString::from("step:intervention"), cx);
        assert_eq!(console.read(cx).step, WorkspaceStep::Intervention);
        window.click(SharedString::from("operation-card:zero"), cx);
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
        window.click(SharedString::from("step:prompt"), cx);
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
        window.click(SharedString::from("step:intervention"), cx);
        // The advanced controls live in the inspector, and the inspector
        // starts closed -- so this has to open the inspector first. This
        // test used to find `advanced-toggle` without doing that, which
        // was only possible because a second, unclosable copy of the
        // inspector was being rendered alongside the real one.
        assert!(
            window
                .try_find(SharedString::from("advanced-toggle"))
                .is_none(),
            "advanced controls must not be reachable while the inspector is closed"
        );
        window.click(SharedString::from("inspector-toggle"), cx);
        window.click(SharedString::from("advanced-toggle"), cx);
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
            window.click(SharedString::from("step:results"), cx);
            window.click(SharedString::from("btn:Run experiment"), cx);
            assert_eq!(console.read(cx).status, super::Status::Preparing);
            assert!(matches!(worker_rx.try_recv().unwrap(), super::WorkerMsg::Prepare(path) if path == "fixture.gguf"));
            let loading = SharedString::from("btn:Loading model…");
            assert!(window.find(loading.clone()).visible());
            window.click(loading, cx);
            assert!(worker_rx.try_recv().is_err(), "disabled run must not submit another job");
            reply_tx.send(super::WorkerReply::Prepared(Box::new(Err("fixture load failure".into())))).unwrap();
            console.update(cx, |console, cx| { console.drain_replies(cx); cx.notify(); });
            window.render_frame(cx);
            assert_eq!(console.read(cx).status, super::Status::Idle);
            assert_eq!(console.read(cx).error.as_deref(), Some("fixture load failure"));
            window.click(SharedString::from("btn:Run experiment"), cx);
            assert!(matches!(worker_rx.try_recv().unwrap(), super::WorkerMsg::Prepare(_)));
        }).unwrap();
}
