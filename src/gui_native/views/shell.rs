use super::*;
use crate::gui_native::*;
use gpui_kit::component::TitleBar;

// -- shell: top bar, navigation, palette, inspector, status bar -----------
impl Console {
    /// The palette as an overlay surface: dim backdrop, one floating panel,
    /// the query field, the candidates, the keys. Escape and the key that
    /// opened it dismiss it; nothing else on the page responds while it is
    /// open. Go to run lists saved runs in the same panel.
    pub(super) fn palette_overlay(&self, colors: &Colors, cx: &mut Context<Self>) -> Stateful<Div> {
        let lists_runs = self.palette_lists_runs();
        let row = |id: String, title: String, hint: String, shortcut: Option<Kbd>, index: usize| {
            Button::new(SharedString::from(id))
                .ghost()
                .w_full()
                .h(px(32.0))
                .justify_start()
                .rounded(px(Radius::SM))
                .accessibility_label(format!("{title}, {hint}"))
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap(px(Space::MD))
                        .w_full()
                        .px_2()
                        .overflow_hidden()
                        .child(label(title, Type::LABEL, colors.text))
                        .child(div().w_full())
                        .child(label(hint, Type::MICRO, colors.text_faint))
                        .children(shortcut),
                )
                .on_click(cx.listener(move |console, _: &ClickEvent, window, cx| {
                    console.palette_index = index;
                    console.palette_execute(cx);
                    if !console.palette_open {
                        console.focus_handle.focus(window, cx);
                    }
                }))
        };
        let rows = if lists_runs {
            self.run_candidates()
                .into_iter()
                .filter_map(|number| self.store.runs.iter().find(|run| run.number == number))
                .enumerate()
                .map(|(index, run)| {
                    row(
                        format!("palette-run:{}", run.number),
                        run_palette_label(run),
                        format!("{}  \u{00b7}  {}", run.intervention, run.model),
                        None,
                        index,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            self.palette_candidates()
                .into_iter()
                .enumerate()
                .map(|(index, command)| {
                    row(
                        format!("palette-cmd:{:?}", command),
                        command.label().to_string(),
                        command.hint().to_string(),
                        palette_shortcut(command),
                        index,
                    )
                })
                .collect::<Vec<_>>()
        };
        let selected = self.palette_index.min(rows.len().saturating_sub(1));
        let empty = rows.is_empty().then(|| {
            let note = if !lists_runs {
                "No command matches."
            } else if self.palette_query.trim().is_empty() {
                "No saved run with a result yet."
            } else {
                "No saved run matches."
            };
            div()
                .px_2()
                .py_2()
                .child(label(note, Type::LABEL, colors.text_faint))
        });
        let rows = rows
            .into_iter()
            .enumerate()
            .map(|(index, row)| row.selected(index == selected))
            .collect::<Vec<_>>();
        let legend = if lists_runs {
            "up down navigate \u{00b7} enter open \u{00b7} > commands \u{00b7} esc close"
        } else {
            "up down navigate \u{00b7} enter run \u{00b7} esc close \u{00b7} cmd-p go to run"
        };

        div()
            .id("palette-backdrop")
            .absolute()
            .inset_0()
            .bg(Hsla::from(rgb(0x000000)).opacity(0.35))
            .flex()
            .flex_col()
            .items_center()
            .pt(px(96.0))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|console, _: &MouseDownEvent, window, cx| {
                    console.palette_open = false;
                    console.focus_handle.focus(window, cx);
                    cx.notify();
                }),
            )
            .child(
                div()
                    .id("palette-panel")
                    .w(px(600.0))
                    .max_h(px(520.0))
                    .flex()
                    .flex_col()
                    .bg(colors.surface_raised)
                    .border_1()
                    .border_color(colors.border)
                    .rounded(px(Radius::LG))
                    .p(px(Space::SM))
                    // Clicks inside the panel are not backdrop clicks.
                    .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                    // Go to run keeps its name visible once the placeholder
                    // gives way to the query.
                    .when(lists_runs, |panel| {
                        panel.child(div().px_2().pt_1().child(label(
                            "Go to run",
                            Type::LABEL,
                            colors.text_muted,
                        )))
                    })
                    // The query field and the key legend stay put; only the
                    // candidate list scrolls.
                    .child(div().px_2().pt_1().pb_2().child(text_input(
                        colors,
                        self.palette_input.clone(),
                        FONT_SANS_NAME,
                        Type::BODY,
                        None,
                        cx,
                    )))
                    .child(
                        div()
                            .id("palette-rows")
                            .flex_1()
                            .min_h(px(0.0))
                            .overflow_y_scroll()
                            .flex()
                            .flex_col()
                            .children(rows)
                            .children(empty),
                    )
                    .child(
                        div()
                            .px_2()
                            .py_2()
                            .child(mono(legend, Type::MICRO, colors.text_faint)),
                    ),
            )
    }

    pub(in crate::gui_native) fn picker(
        &self,
        _colors: &Colors,
        _id: &'static str,
        combo: ComboId,
        selected: &str,
        options: &[String],
        cx: &mut Context<Self>,
    ) -> Div {
        let picker = &self
            .pickers
            .iter()
            .find(|(id, _)| *id == combo)
            .expect("all domain selectors are initialized")
            .1;
        picker.update(cx, |picker, _| picker.sync(options, selected));
        div().w_full().child(picker.clone())
    }

    /// Slim top bar: identity, context, model state.
    ///
    /// All typography, no icons. The wordmark carries the app identity, the
    /// middle is a breadcrumb that is only as specific as the view makes it,
    /// and the right side is the model's live state plus the two view toggles.
    pub(super) fn topbar(&self, colors: &Colors, cx: &mut Context<Self>) -> AnyElement {
        let toggle = cx.listener(|console, _: &ClickEvent, _w, cx| {
            console.cycle_appearance(cx);
        });
        let model_state = if self.session.is_some() {
            "ready"
        } else if self.status == Status::Preparing {
            "loading"
        } else {
            "not loaded"
        };
        let model_summary = if self.model_path.trim().is_empty() {
            "no model selected".to_string()
        } else {
            format!(
                "{} · {}",
                truncate_chars(&model_display_name(&self.model_path), 24),
                model_state
            )
        };
        // The window's only top strip. It is the kit's TitleBar, so it moves the
        // window and zooms on a double click, and on macOS it leaves room for
        // the traffic lights. No wordmark here: the sidebar header carries it.
        let bar = div()
            .flex()
            .flex_row()
            .items_center()
            .gap(px(Space::MD))
            .pl(px(Space::MD))
            .pr(px(Space::LG))
            .w_full()
            .h_full()
            // A flex_1 row will happily paint text over its siblings when the
            // content cannot shrink, so this one truncates rather than trusting
            // min_w(0) alone. The step is the context that is true on every
            // screen; the model is the experiment's business and the inspector
            // already carries it, where it cannot go stale.
            .child(
                div()
                    .flex_1()
                    .min_w(px(0.0))
                    .overflow_hidden()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(Space::SM))
                    .child(label(self.view.label(), Type::LABEL, colors.text_faint))
                    // The breadcrumb names the state of the work, not a step:
                    // the workspace has none.
                    .when(self.view == View::Experiment, |row| {
                        let state = if self.busy() {
                            "Running"
                        } else if self.saved_run.is_some() {
                            "Saved run"
                        } else if self.result_view == ResultView::Sweep {
                            "Sweep"
                        } else if self.opened_note.is_some() {
                            "Sweep point"
                        } else if self.sample {
                            "Sample result"
                        } else if self.results_stale() {
                            "Settings changed"
                        } else if self.baseline.is_some() {
                            "Results"
                        } else {
                            "New"
                        };
                        row.child(label("/", Type::LABEL, colors.border_strong))
                            .child(label(state, Type::LABEL, colors.text_muted))
                    }),
            )
            // The workspace's setup pane already shows the model; elsewhere it
            // is useful context, so it stays on every other page.
            .when(self.view != View::Experiment, |bar| {
                bar.child(
                    div()
                        .flex_none()
                        .max_w(px(theme::scaled(320.0)))
                        .overflow_hidden()
                        .child(mono(model_summary, Type::META, colors.text_faint))
                        .whitespace_nowrap(),
                )
            })
            .child(
                Button::new("theme-toggle")
                    .ghost()
                    .small()
                    .label(self.appearance.label())
                    .tooltip("Appearance")
                    .accessibility_label(format!(
                        "Appearance: {}. Switch appearance.",
                        self.appearance.label()
                    ))
                    .on_click(toggle),
            );
        TitleBar::new()
            .h(px(theme::scaled(44.0)))
            .bg(colors.canvas)
            .border_color(colors.border)
            .child(bar)
            .into_any_element()
    }

    /// Left navigation rail: quiet text rows, no icons.
    ///
    /// The label is the interface. Selection is a surface fill at one
    /// luminance step -- never the accent, which is spent on the intervention
    /// and the primary action -- so the rail reads like an editor's side pane
    /// rather than a dashboard menu.
    pub(super) fn nav_rail(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let mut column = div()
            .flex()
            .flex_col()
            .gap(px(Space::XS))
            .w(px(200.0))
            .flex_none()
            .px(px(Space::MD))
            .py_3()
            .bg(colors.sidebar)
            .border_r_1()
            .border_color(colors.border);
        // A workspace header and grouped sections, the way Notion and Obsidian
        // organise a sidebar: the wordmark on top, the working pages first,
        // a "Library" of things you accumulate, and Settings pinned below.
        column = column.child(
            div()
                .flex()
                .flex_col()
                .gap(px(2.0))
                .px_2()
                .pb(px(Space::MD))
                .child(label("Ember", Type::SUBSECTION, colors.text))
                .child(label("Experiment console", Type::META, colors.text_faint)),
        );
        for view in View::ALL {
            let active = self.view == view;
            if view == View::Models {
                column = column.child(
                    div()
                        .px_2()
                        .pt(px(Space::MD))
                        .pb(px(Space::XS))
                        .child(label("Library", Type::META, colors.text_faint)),
                );
            }
            if view == View::Settings {
                column = column.child(div().flex_1());
            }
            column = column.child(
                Button::new(SharedString::from(format!("nav:{}", view.key())))
                    .ghost()
                    .w_full()
                    .h(px(36.0))
                    .justify_start()
                    .selected(active)
                    // Left-aligned like an editor's file tree; the kit centres
                    // a plain `.label()` regardless of `justify_start`.
                    .child(div().w_full().text_left().child(view.label()))
                    .tooltip(view.hint())
                    .accessibility_label(format!("{}, {}", view.label(), view.hint()))
                    .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                        console.goto(view, cx);
                    })),
            );
        }
        column
    }

    /// Workflow stepper. Reads as tabs, not a wizard diagram.
    /// One tab bar, used for both the workflow steps and the result views.
    ///
    /// gpui-kit ships a `TabBar` with exactly this `.underline()` treatment, and
    /// the design guides say to reach for the component. It cannot be used here
    /// for a concrete reason, checked in the 0.6.6 and 0.9.0 source rather than assumed:
    /// `Tab::ix` sets `self.base = self.base.id(ix)` -- a tab's element id is
    /// its *positional index*, assigned internally, and the docs page documents
    /// a `.id("custom-id")` method that does not exist. So a tab has no stable
    /// address, which means no test can click "go to Prompt" and nothing can
    /// label a tab for a screen reader beyond the visible text.
    ///
    /// Both call sites need those, so both are built here. The alternative --
    /// the library tab bar on one screen and a hand-rolled imitation on the
    /// other -- is worse than either alone.
    pub(super) fn tab_row(
        &self,
        colors: &Colors,
        cx: &mut Context<Self>,
        prefix: &'static str,
        // (element key, visible label, what a screen reader should hear)
        tabs: &[(&'static str, &'static str, String)],
        active: usize,
    ) -> Div {
        let mut row = div().flex().flex_row().gap(px(Space::SM));
        for (index, (key, text, accessible)) in tabs.iter().enumerate() {
            let is_active = index == active;
            let selected = *key;
            row = row.child(
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .child(
                        Button::new(SharedString::from(format!("{prefix}:{key}")))
                            .ghost()
                            .compact()
                            .label(*text)
                            .tooltip(*text)
                            .accessibility_label(accessible)
                            .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                                console.select_tab(prefix, selected);
                                cx.notify();
                            })),
                    )
                    // The underline is the whole selected state: a 2px accent
                    // bar under the active tab and nothing under the others.
                    // A grey rail under every tab made the row read as a rule
                    // with exceptions instead of a position.
                    .child(
                        div()
                            .w_full()
                            .h(px(2.0))
                            .mt(px(-Space::XS))
                            .rounded_full()
                            .when(is_active, |bar| bar.bg(colors.accent)),
                    ),
            );
        }
        row
    }

    /// Home: a landing surface with something to do, not a form in waiting.
    pub(super) fn section_header(&self, colors: &Colors, title: &'static str, hint: &str) -> Div {
        div()
            .flex()
            .flex_col()
            .gap(px(Space::XS))
            .child(label(title, Type::SECTION, colors.text))
            .child(label(hint.to_string(), Type::BODY, colors.text_muted))
    }

    pub(super) fn main_panel(&mut self, colors: &Colors, cx: &mut Context<Self>) -> Stateful<Div> {
        // The render fixture needs a finished comparison to review the Review
        // page against. Applied here rather than in the constructor because the
        // constructor cannot know which step the harness will photograph.
        #[cfg(feature = "gui-tests")]
        if self.baseline.is_none() && seed_runs_requested() && self.step == WorkspaceStep::Review {
            let (baseline, intervention, comparison) = seed_comparison();
            self.baseline = Some(baseline);
            self.intervention = Some(intervention);
            self.comparison = Some(comparison);
        }
        // The results column. Same content the Review step used to hold; the
        // difference is that it now sits beside the setup instead of behind a
        // step, and shows an empty state before the first run.
        let has_results = self.baseline.is_some() && self.intervention.is_some();
        let results = if has_results || self.busy() {
            self.review_step(colors, cx).into_any_element()
        } else {
            self.results_empty(colors, cx).into_any_element()
        };
        // A new result eases in rather than snapping: a short fade, keyed to
        // the result so it plays once per result and not on every repaint.
        // Off in the render harness, which photographs single frames.
        let results = if cfg!(feature = "gui-tests") {
            results
        } else {
            div()
                .w_full()
                .child(results)
                .with_animation(
                    SharedString::from(format!("results-in-{}", self.result_epoch)),
                    Animation::new(std::time::Duration::from_millis(180))
                        .with_easing(ease_out_quint()),
                    |element, delta| element.opacity(delta),
                )
                .into_any_element()
        };

        div()
            .id("workspace")
            .flex()
            .flex_row()
            .flex_1()
            .min_w(px(0.0))
            .min_h(px(0.0))
            .child(self.setup_pane(colors, cx))
            .child(
                div()
                    .id("workspace-scroll")
                    // Registers the container itself so the render harness
                    // can scroll it; a no-op outside `test-support`.
                    .test_support()
                    .flex_1()
                    .min_w(px(if self.spec_open {
                        super::workspace::SPEC_RESULTS_MIN_WIDTH
                    } else {
                        0.0
                    }))
                    .h_full()
                    // Vertical scroll only: a wide page must not push the
                    // inspector off the window (see the width tests).
                    .overflow_y_scroll()
                    .overflow_x_hidden()
                    .px_6()
                    .py_5()
                    .child(
                        div()
                            .w_full()
                            .min_w(px(0.0))
                            .flex()
                            .flex_col()
                            .gap(px(Space::LG))
                            .child(self.feedback_banners(colors))
                            .child(results),
                    ),
            )
    }

    pub(super) fn statusbar_needed(&self) -> bool {
        self.store_error.is_some() || (self.view == View::Experiment && self.busy())
    }

    pub(super) fn statusbar(&self, colors: &Colors, _cx: &mut Context<Self>) -> Div {
        // One quiet line. Storage failures outrank everything else here (a run
        // history the user believes was saved but was not is the worst silent
        // state this app can reach), then validation, then the run state.
        let (store_line, store_color) = match &self.store_error {
            Some(error) => (Some(error.clone()), colors.err),
            None => (None, colors.text_faint),
        };
        // Home, Models, Runs and Settings have no experiment to advance, so the
        // status line reports the surface instead of offering a run button that
        // would do nothing sensible.
        if self.view != View::Experiment {
            let (text, color) = store_line
                .map(|text| (text, store_color))
                .unwrap_or_else(|| (self.view.hint().to_string(), colors.text_faint));
            return div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(Space::MD))
                .px_4()
                .h(px(32.0))
                .w_full()
                .bg(colors.canvas)
                .border_t_1()
                .border_color(colors.border)
                .child(status_dot(colors.ok, false))
                .child(label(text, Type::META, color))
                .child(div().w_full());
        }
        let (dot, status_text) = match self.status {
            Status::Idle => (colors.ok, "Ready to run"),
            Status::Preparing => (colors.warn, "Loading the model…"),
            Status::Running => (colors.busy, "Running baseline and intervention…"),
            Status::Restoring => (colors.busy, "Checking exact restoration…"),
            Status::Cancelling => (colors.warn, "Cancelling the run…"),
        };
        let (line, line_color) = if let Some((text, color)) = store_line.map(|t| (t, store_color)) {
            (text, color)
        } else if let Some(error) = self.validation_error() {
            (error, colors.warn)
        } else {
            (
                match self.status {
                    Status::Idle => "Ready · baseline + intervention · seed 0".to_string(),
                    _ => status_text.to_string(),
                },
                colors.text,
            )
        };
        div()
            .flex()
            .flex_row()
            .items_center()
            .gap(px(Space::MD))
            .px(px(Space::LG))
            .h(px(theme::scaled(42.0)))
            .w_full()
            .bg(colors.canvas)
            .border_t_1()
            .border_color(colors.border)
            .child(status_dot(dot, self.busy()))
            .child(div().min_w(px(0.0)).flex_1().overflow_hidden().child(label(
                line,
                Type::LABEL,
                line_color,
            )))
            // Named for the platform: the binding accepts Control and Command
            // alike, but the hint should read the way the keyboard does.
            .children(
                Keystroke::parse(if cfg!(target_os = "macos") {
                    "cmd-enter"
                } else {
                    "ctrl-enter"
                })
                .ok()
                .map(Kbd::new),
            )
    }
}
