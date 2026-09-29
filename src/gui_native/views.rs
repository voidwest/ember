//! View builders: the shell, the pages, the experiment steps and the result
//! panels, each an `impl Console` block of its own.

use super::*;

// -- shell: top bar, navigation, palette, inspector, status bar -----------
impl Console {
    /// The palette as an overlay surface: dim backdrop, one floating panel,
    /// the query field, the candidates, the keys. Escape and Cmd+K dismiss
    /// it; nothing else on the page responds while it is open.
    fn palette_overlay(&self, colors: &Colors, cx: &mut Context<Self>) -> Stateful<Div> {
        let candidates = self.palette_candidates();
        let selected = self.palette_index.min(candidates.len().saturating_sub(1));
        let rows = candidates
            .into_iter()
            .enumerate()
            .map(|(index, command)| {
                Button::new(SharedString::from(format!("palette-cmd:{:?}", command)))
                    .ghost()
                    .w_full()
                    .h(px(32.0))
                    .justify_start()
                    .rounded(px(Radius::SM))
                    .selected(index == selected)
                    .accessibility_label(format!("{}, {}", command.label(), command.hint()))
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(px(Space::MD))
                            .w_full()
                            .px_2()
                            .overflow_hidden()
                            .child(label(command.label(), Type::LABEL, colors.text))
                            .child(div().w_full())
                            .child(label(command.hint(), Type::MICRO, colors.text_faint))
                            .children(palette_shortcut(command)),
                    )
                    .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                        console.palette_index = index;
                        console.palette_execute(cx);
                    }))
            })
            .collect::<Vec<_>>();

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
                cx.listener(|console, _: &MouseDownEvent, _, cx| {
                    console.palette_open = false;
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
                            .children(rows),
                    )
                    .child(div().px_2().py_2().child(mono(
                        "up down navigate \u{00b7} enter run \u{00b7} esc close \u{00b7} cmd-k palette",
                        Type::MICRO,
                        colors.text_faint,
                    ))),
            )
    }

    pub(super) fn picker(
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
    fn topbar(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
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
        div()
            .flex()
            .flex_row()
            .items_center()
            .gap(px(Space::MD))
            .px_4()
            .h(px(theme::scaled(46.0)))
            .w_full()
            .bg(colors.canvas)
            .border_b_1()
            .border_color(colors.border)
            .child(
                Button::new("topbar-home")
                    .ghost()
                    .small()
                    .label("ember")
                    .tooltip("Home")
                    .accessibility_label("Ember, go to Home")
                    .on_click(cx.listener(|console, _: &ClickEvent, _, cx| {
                        console.goto(View::Home, cx);
                    })),
            )
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
            .child(
                div()
                    .flex_none()
                    .max_w(px(theme::scaled(320.0)))
                    .overflow_hidden()
                    .child(mono(model_summary, Type::META, colors.text_faint))
                    .whitespace_nowrap(),
            )
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
            )
            // No button when there is no room: a toggle that changes nothing
            // on screen reads as broken.
            .when(
                self.view == View::Experiment && self.inspector_fits,
                |bar| {
                    bar.child(
                        Button::new("inspector-toggle")
                            .ghost()
                            .small()
                            .selected(self.inspector_open)
                            .label("Inspector")
                            .tooltip("Toggle the inspector")
                            .accessibility_label(format!(
                                "Inspector. {}",
                                if self.inspector_open {
                                    "Hide the inspector"
                                } else {
                                    "Show the inspector"
                                }
                            ))
                            .on_click(cx.listener(|console, _: &ClickEvent, _, cx| {
                                console.toggle_inspector(cx);
                            })),
                    )
                },
            )
    }

    /// Left navigation rail: quiet text rows, no icons.
    ///
    /// The label is the interface. Selection is a surface fill at one
    /// luminance step -- never the accent, which is spent on the intervention
    /// and the primary action -- so the rail reads like an editor's side pane
    /// rather than a dashboard menu.
    fn nav_rail(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
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
    /// for a concrete reason, checked in the 0.6.6 source rather than assumed:
    /// `Tab::ix` sets `self.base = self.base.id(ix)` -- a tab's element id is
    /// its *positional index*, assigned internally, and the docs page documents
    /// a `.id("custom-id")` method that does not exist. So a tab has no stable
    /// address, which means no test can click "go to Prompt" and nothing can
    /// label a tab for a screen reader beyond the visible text.
    ///
    /// Both call sites need those, so both are built here. The alternative --
    /// the library tab bar on one screen and a hand-rolled imitation on the
    /// other -- is worse than either alone.
    fn tab_row(
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
    fn section_header(&self, colors: &Colors, title: &'static str, hint: &str) -> Div {
        div()
            .flex()
            .flex_col()
            .gap(px(Space::XS))
            .child(label(title, Type::SECTION, colors.text))
            .child(label(hint.to_string(), Type::BODY, colors.text_muted))
    }

    fn advanced_inspector(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let context = self.visible_experiment_context();
        let model_name = model_display_name(&context.model_path);
        let prompt_excerpt = truncate_chars(&context.prompt, 120);
        let target = if per_layer(&context.site) {
            format!(
                "Layer {}\n{}\n{}",
                context.layer,
                site_label(&context.site),
                token_label(&context.token)
            )
        } else {
            format!(
                "{}\n{}",
                site_label(&context.site),
                token_label(&context.token)
            )
        };
        let intervention = match context.op.as_str() {
            "scale" => format!("Scale ×{}", context.value),
            "zero" => "Set activation to zero".to_string(),
            "replace" => format!("Replace from layer {}", context.source_layer),
            "interpolate" => format!("Interpolate α={}", context.value),
            "add-delta" => format!("Add delta from layer {}", context.source_layer),
            _ => context.op.clone(),
        };
        let active_metric = self
            .hovered_layer
            .or(self.selected_layer)
            .and_then(|layer| {
                self.layer_series
                    .iter()
                    .find(|metric| metric.layer == layer)
            });
        let active_metric_label = if self.hovered_layer.is_some() {
            "Hovered"
        } else {
            "Selected"
        };
        let advanced = self.advanced_open.then(|| {
            div()
                .flex()
                .flex_col()
                .gap(px(Space::MD))
                .pt_2()
                .child(field(
                    colors,
                    "Execution engine",
                    self.picker(
                        colors,
                        "execution-picker",
                        ComboId::Execution,
                        &self.execution,
                        &self.execution_options,
                        cx,
                    ),
                ))
                .child(field(
                    colors,
                    "Exact token limit",
                    text_input(
                        colors,
                        self.inputs.max_tokens.clone(),
                        FONT_MONO_NAME,
                        Type::BODY,
                        None,
                        cx,
                    ),
                ))
                .child(field(
                    colors,
                    "Raw model path",
                    text_input(
                        colors,
                        self.inputs.model.clone(),
                        FONT_MONO_NAME,
                        Type::LABEL,
                        Some(52.0),
                        cx,
                    ),
                ))
                .child(
                    div()
                        .p(px(Space::SM))
                        .bg(colors.surface_raised)
                        .rounded(px(Radius::MD))
                        .flex()
                        .flex_col()
                        .gap(px(Space::XS))
                        .child(mono(
                            format!("hook     {}", self.site),
                            Type::LABEL,
                            colors.text_faint,
                        ))
                        .child(mono(
                            format!("operation {}", self.op),
                            Type::LABEL,
                            colors.text_faint,
                        ))
                        .child(mono(
                            format!("tokens    {}", self.token),
                            Type::LABEL,
                            colors.text_faint,
                        )),
                )
        });
        // Notion-style property row: the name on the left, the value beside it.
        let prop = |name: &'static str, value: Div| -> Div {
            div()
                .flex()
                .flex_row()
                .items_start()
                .gap(px(Space::MD))
                .child(div().w(px(80.0)).flex_none().child(label(
                    name,
                    Type::META,
                    colors.text_faint,
                )))
                .child(div().flex_1().min_w(px(0.0)).child(value))
        };
        let toggle = cx.listener(|console, _: &ClickEvent, _window, cx| {
            console.advanced_open = !console.advanced_open;
            cx.notify();
        });

        div()
            .flex()
            .flex_col()
            .gap(px(Space::MD))
            .child(prop(
                "Model",
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label(model_name, Type::BODY, colors.text))
                    .child(match &self.session {
                        Some(session) => mono(
                            format!(
                                "{} · {} layers · {}d",
                                session.architecture, session.n_layers, session.embed_dim
                            ),
                            Type::META,
                            colors.text_muted,
                        ),
                        None => mono("not loaded", Type::META, colors.text_faint),
                    }),
            ))
            .child(prop(
                "Input",
                div().flex().flex_col().gap(px(Space::XS)).child(multiline(
                    &prompt_excerpt,
                    Type::LABEL,
                    colors.text,
                    FONT_ARABIC_NAME,
                )),
            ))
            .child(prop(
                "Target",
                div().flex().flex_col().gap(px(Space::XS)).child(multiline(
                    &target,
                    Type::LABEL,
                    colors.text,
                    FONT_SANS_NAME,
                )),
            ))
            .child(prop(
                "Intervention",
                div().flex().flex_col().gap(px(Space::XS)).child(label(
                    intervention,
                    Type::BODY,
                    colors.accent,
                )),
            ))
            .child(prop(
                "Generation",
                div().flex().flex_col().gap(px(Space::XS)).child(mono(
                    format!(
                        "≤{} tokens · seed 0\n{}",
                        context.max_tokens, context.execution
                    ),
                    Type::META,
                    colors.text,
                )),
            ))
            .children(self.intervention.as_ref().map(|output| {
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(rule_h(colors))
                    .child(prop(
                        "Run",
                        div().flex().flex_col().gap(px(Space::XS)).child(mono(
                            format!(
                                "{} total\n{} generated\n{}",
                                self.last_metrics.as_ref().map_or_else(
                                    || "—".to_string(),
                                    |(_, elapsed, _)| fmt_ms(*elapsed)
                                ),
                                output.generated_tokens,
                                fmt_tps(output.decode_tps)
                            ),
                            Type::LABEL,
                            colors.text,
                        )),
                    ))
            }))
            .children(active_metric.map(|metric| {
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(rule_h(colors))
                    .child(prop(
                        active_metric_label,
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(Space::XS))
                            .child(mono(
                                format!("layer {}", metric.layer),
                                Type::LABEL,
                                colors.text,
                            ))
                            .child(mono(
                                metric.relative_l2_difference.map_or_else(
                                    || "relative L2  —".to_string(),
                                    |value| format!("relative L2  {value:.6}"),
                                ),
                                Type::LABEL,
                                colors.accent,
                            ))
                            .child(mono(
                                metric.cosine_distance.map_or_else(
                                    || "cosine distance  —".to_string(),
                                    |value| format!("cosine distance  {value:.6}"),
                                ),
                                Type::LABEL,
                                colors.text_muted,
                            )),
                    ))
            }))
            .child(rule_h(colors))
            // A ghost button with only a text label reads as static copy in a
            // wide empty column, so the accessible name states the position
            // and the label changes with the state.
            .child(
                Button::new("advanced-toggle")
                    .ghost()
                    .w_full()
                    .label(if self.advanced_open {
                        "Hide advanced controls"
                    } else {
                        "Show advanced controls"
                    })
                    .accessibility_label(if self.advanced_open {
                        "Hide advanced controls (currently expanded)"
                    } else {
                        "Show advanced controls (currently collapsed)"
                    })
                    .on_click(toggle),
            )
            .children(advanced)
    }

    fn main_panel(&mut self, colors: &Colors, cx: &mut Context<Self>) -> Stateful<Div> {
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
                    .min_w(px(0.0))
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

    fn statusbar(&self, colors: &Colors, _cx: &mut Context<Self>) -> Div {
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

// -- pages: Home, Models, Runs, Settings --------------------------------
impl Console {
    /// Models: one row per discovered file, name first.
    ///
    /// This page was a single card whose entire content was the absolute path
    /// `~/Projects/ember/Llama-3.2-1B-Instruct-Q8_0.gguf`. The design
    /// brief is explicit that a page must not be centred on filesystem paths and
    /// that each model should read as a proper object, and a path is neither:
    /// it is the one string on the screen that means nothing to anyone reading
    /// over your shoulder.
    ///
    /// So the name leads, the path demotes to a secondary line that truncates
    /// from the left -- the end of a path is the informative end, and cutting
    /// the start is what keeps the filename visible. Size comes from the
    /// filesystem, quant from the filename, and last-used from the store, so
    /// nothing here is invented. Rows are selectable objects: selecting one
    /// reveals its actions.
    fn models_view(&mut self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let loaded = self.session.is_some();
        let current = self.model_path.trim();
        let paths: Vec<String> = self.model_options.clone();
        let selected = self.selected_model.clone();

        let mut rows = div().flex().flex_col();
        if paths.is_empty() {
            rows = rows.child(
                div()
                    .py(px(Space::XXL))
                    .child(label("No models found", Type::SUBSECTION, colors.text))
                    .child(label(
                        "Point Ember at a directory of GGUF files to get started.",
                        Type::LABEL,
                        colors.text_faint,
                    )),
            );
        }
        for path in paths.iter() {
            let name = model_display_name(path);
            let is_current = path.as_str() == current;
            let is_selected = selected.as_deref() == Some(path.as_str());
            let size = self.model_sizes.get(path).copied().flatten();
            let last_used = self.store.model_last_used(path);
            let row_button = Button::new(SharedString::from(format!("model-row:{path}")))
                .ghost()
                .w_full()
                .h_auto()
                .justify_start()
                // Name plus path lands the row at ~52px: scannable without
                // the page turning into a wall of vertical gaps.
                .py(px(Space::XS))
                .px_2()
                .rounded(px(Radius::SM))
                .selected(is_selected)
                .accessibility_label(format!("Model {name}"))
                .child(
                    div()
                        .w_full()
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap(px(Space::LG))
                        // Name, then the metadata that makes it a model rather
                        // than a filename: quant, size, and whether it is the
                        // one currently in play.
                        .child(
                            div()
                                .flex_1()
                                .min_w(px(0.0))
                                .overflow_hidden()
                                .flex()
                                .flex_col()
                                .gap(px(Space::XS))
                                .child(
                                    div()
                                        .flex()
                                        .flex_row()
                                        .items_center()
                                        .gap(px(Space::SM))
                                        .when(is_current, |row| {
                                            row.child(status_dot(colors.accent, false))
                                        })
                                        .child(label(name, Type::BODY, colors.text)),
                                )
                                // Left-truncated: the tail of a path identifies
                                // the file, so that is the end worth keeping.
                                .child(mono(
                                    format!("…{}", truncate_path_start(path, 64)),
                                    Type::META,
                                    colors.text_faint,
                                )),
                        )
                        .child(div().w(px(80.0)).flex_none().child(mono(
                            quant_of(path).to_string(),
                            Type::META,
                            colors.text_muted,
                        )))
                        .child(
                            div().w(px(88.0)).flex_none().child(mono(
                                size.map(fmt_bytes)
                                    .unwrap_or_else(|| "\u{2014}".to_string()),
                                Type::META,
                                colors.text_muted,
                            )),
                        )
                        .child(div().w(px(100.0)).flex_none().child(label(
                            if is_current && loaded {
                                "Loaded"
                            } else if is_current {
                                "Selected"
                            } else {
                                "Available"
                            },
                            Type::LABEL,
                            if is_current {
                                colors.accent
                            } else {
                                colors.text_faint
                            },
                        )))
                        .child(
                            div().w(px(64.0)).flex_none().child(label(
                                last_used
                                    .map(relative_time)
                                    .unwrap_or_else(|| "Never".into()),
                                Type::META,
                                colors.text_faint,
                            )),
                        ),
                )
                .on_click(cx.listener({
                    let path = path.clone();
                    move |console, _: &ClickEvent, _, cx| {
                        console.selected_model = Some(path.clone());
                        cx.notify();
                    }
                }));
            let row = div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(Space::SM))
                .child(row_button)
                // The action lane exists only for the selected row: two quiet
                // text commands, always reachable through selection rather
                // than hover.
                .when(is_selected, |lane| {
                    lane.child(
                        div()
                            .flex()
                            .flex_row()
                            .flex_none()
                            .gap(px(Space::XS))
                            .child(text_button(
                                SharedString::from(format!("model-load:{path}")),
                                "Load",
                                cx.listener({
                                    let path = path.clone();
                                    move |console, _: &ClickEvent, _, cx| {
                                        console.select_combo(ComboId::Model, &path, cx);
                                        console.load();
                                        cx.notify();
                                    }
                                }),
                            ))
                            .child(text_button(
                                SharedString::from(format!("model-reveal:{path}")),
                                "Reveal in Finder",
                                cx.listener({
                                    let path = path.clone();
                                    move |console, _: &ClickEvent, _, _| {
                                        if let Err(error) = reveal_in_finder(&path) {
                                            console.error =
                                                Some(format!("could not reveal model: {error}"));
                                        }
                                    }
                                }),
                            )),
                    )
                });
            rows = rows.child(row);
        }

        div()
            .flex()
            .flex_col()
            .gap(px(Space::XL))
            .w_full()
            .max_w(px(1200.0))
            .px_6()
            .pt_8()
            .child(self.section_header(
                colors,
                "Models",
                &format!(
                    "{} local GGUF file{}. Nothing leaves this machine.",
                    paths.len(),
                    if paths.len() == 1 { "" } else { "s" }
                ),
            ))
            .child(rows)
    }

    /// Runs: the kit's `DataTable` over the store.
    ///
    /// Previously a stack of bordered cards, then a hand-rolled seven-column
    /// grid. The grid had the right columns and none of the properties that
    /// make a history usable: it could not sort, could not be navigated by
    /// keyboard, and at the minimum window width it clipped the last two
    /// columns with no way to reach them.
    ///
    /// The table state is created here rather than in the constructor because
    /// `TableState::new` needs a `Window`. It is built once and then synced:
    /// the delegate holds a snapshot, so a run finishing while Runs is on screen
    /// has to push new rows in rather than expect the table to notice. The
    /// sync is cheap: rows are rebuilt only when the store's generation moved.
    fn runs_view(&mut self, colors: &Colors, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let row_count = self.store.runs.len();
        let store = &self.store;
        // Row actions mutate the store through the owning console, so the
        // delegate needs a handle that does not keep the console alive.
        let console = cx.entity().downgrade();
        let table = match &self.runs_table {
            Some(state) => {
                let console = console.clone();
                state.update(cx, |state, cx| {
                    state.delegate_mut().sync(store, *colors, console, cx);
                });
                state.clone()
            }
            None => {
                let state =
                    cx.new(|cx| TableState::new(RunsDelegate::new(store, *colors), window, cx));
                self.runs_table = Some(state.clone());
                state
            }
        };

        let mut body: Div = div().flex().flex_col();
        if row_count == 0 {
            body = body.child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .py(px(Space::XXL))
                    .child(label("No runs yet", Type::SUBSECTION, colors.text))
                    .child(label(
                        "Experiments you run will appear here.",
                        Type::LABEL,
                        colors.text_faint,
                    ))
                    .child(div().w(px(160.0)).child(btn_secondary(
                        colors,
                        "New experiment",
                        Some(cx.listener(|console, _: &ClickEvent, _window, cx| {
                            console.goto(View::Experiment, cx);
                            console.step = WorkspaceStep::Prompt;
                            cx.notify();
                        })),
                    ))),
            );
        } else {
            // Header plus one row band per record, measured off a render rather
            // than guessed: a fixed 420px left a void under a short history, and
            // asking for less than the rows need silently dropped the last one.
            //
            // `flex_none` matters as much as the number. In a flex column this
            // wrapper was being compressed to fit the page, so it rendered
            // shorter than it was asked for -- 194pt for a 230pt request -- and
            // the shortfall came straight out of the last row.
            let height = 34.0 + row_count as f32 * 32.0;
            body = body.child(
                div()
                    .id("runs-table-scroll")
                    .w_full()
                    .flex_none()
                    .h(px(height))
                    // The table is the region that overflows: at the minimum
                    // window the eight columns do not all fit, and the action
                    // lane must be reachable by scrolling this region rather
                    // than by shrinking the window's other panes.
                    .overflow_x_scroll()
                    .child(DataTable::new(&table)),
            );
        }

        div()
            .flex()
            .flex_col()
            .gap(px(Space::XL))
            .w_full()
            .px_5()
            .pt_6()
            .child(self.section_header(
                colors,
                "Runs",
                &format!(
                    "{row_count} recorded experiment{}.",
                    if row_count == 1 { "" } else { "s" }
                ),
            ))
            .child(body)
            .when(row_count > 0, |page| {
                page.child(label(
                    "Reuse loads a run's settings into a new experiment. Pin keeps a run at the top of the list.",
                    Type::LABEL,
                    colors.text_faint,
                ))
            })
    }

    /// One segment of the theme segmented control on the Settings page.
    fn appearance_button(
        &self,
        mode: AppearanceMode,
        title: &'static str,
        cx: &mut Context<Self>,
    ) -> Button {
        Button::new(SharedString::from(format!(
            "settings-appearance:{}",
            mode.label()
        )))
        .ghost()
        .flex_1()
        .h(px(28.0))
        .selected(self.appearance == mode)
        .label(title)
        .accessibility_label(format!("Appearance: {title}"))
        .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
            if console.appearance != mode {
                console.appearance = mode;
                console.appearance.persist();
                console.sync_kit_theme(cx);
                cx.notify();
            }
        }))
    }

    /// Settings as grouped rows, not cards. Every row is either a working
    /// control or a real fact about where state lives; nothing here is a
    /// toggle that toggles nothing.
    fn settings_view(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let store_path = app_store::store_path().display().to_string();

        div()
            .flex()
            .flex_col()
            .gap(px(Space::XXL))
            .w_full()
            .max_w(px(1040.0))
            .px_6()
            .pt_8()
            .child(self.section_header(
                colors,
                "Settings",
                "Appearance and workspace defaults for this console.",
            ))
            .child(group(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(section_label(colors, "Appearance"))
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(px(Space::LG))
                            .child(
                                div()
                                    .w(px(240.0))
                                    .flex_none()
                                    .child(label("Theme", Type::BODY, colors.text)),
                            )
                            .child(
                                div()
                                    .flex()
                                    .w(px(280.0))
                                    .gap(px(Space::XS))
                                    .p(px(Space::XS))
                                    .bg(colors.sidebar)
                                    .rounded(px(Radius::MD))
                                    .child(self.appearance_button(
                                        AppearanceMode::System,
                                        "System",
                                        cx,
                                    ))
                                    .child(self.appearance_button(
                                        AppearanceMode::Dark,
                                        "Dark",
                                        cx,
                                    ))
                                    .child(self.appearance_button(
                                        AppearanceMode::Light,
                                        "Light",
                                        cx,
                                    )),
                            )
                            .child(label(
                                // Says what is true for the current mode.
                                match self.appearance {
                                    AppearanceMode::System => "Following your operating system.",
                                    AppearanceMode::Dark => "Always dark.",
                                    AppearanceMode::Light => "Always light.",
                                },
                                Type::LABEL,
                                colors.text_faint,
                            )),
                    ),
            ))
            .child(group(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(section_label(colors, "Workspace"))
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(px(Space::LG))
                            .child(
                                div()
                                    .w(px(240.0))
                                    .flex_none()
                                    .child(label("Workspace state", Type::BODY, colors.text)),
                            )
                            .child(label(
                                "The inspector, sidebar and in-progress experiment are remembered between launches.",
                                Type::LABEL,
                                colors.text_faint,
                            )),
                    ),
            ))
            .child(group(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(section_label(colors, "Storage"))
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(px(Space::LG))
                            .child(
                                div()
                                    .w(px(240.0))
                                    .flex_none()
                                    .child(label("Run history", Type::BODY, colors.text)),
                            )
                            .child(mono(store_path, Type::MICRO, colors.text_faint)),
                    ),
            ))
            .child(group(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(section_label(colors, "Keyboard"))
                    .children(shortcut_rows().into_iter().map(|(keys, what)| {
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(px(Space::LG))
                            .child(div().w(px(320.0)).flex_none().child(label(
                                what,
                                Type::BODY,
                                colors.text,
                            )))
                            .child(mono(keys, Type::META, colors.text_muted))
                    })),
            ))
    }

    fn home_view(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let start = cx.listener(|console, _: &ClickEvent, _, cx| {
            console.goto(View::Experiment, cx);
            console.step = WorkspaceStep::Prompt;
            cx.notify();
        });
        let runs = cx.listener(|console, _: &ClickEvent, _, cx| {
            console.goto(View::Runs, cx);
        });
        let models = cx.listener(|console, _: &ClickEvent, _, cx| {
            console.goto(View::Models, cx);
        });
        let resume = cx.listener(|console, _: &ClickEvent, _, cx| {
            console.restore_draft(cx);
        });
        let sample = cx.listener(|console, _: &ClickEvent, _, cx| {
            console.show_sample(cx);
        });
        let example = cx.listener(|console, _: &ClickEvent, _, cx| {
            console.goto(View::Experiment, cx);
            console.apply_preset(Preset::SilenceEarly, cx);
            console.step = WorkspaceStep::Prompt;
            cx.notify();
        });
        // Ember's whole method in three sentences. No dashboard, no fake
        // numbers: just the model a newcomer needs before the first click.
        let how_step = |number: &'static str, title: &'static str, body: &'static str| {
            div()
                .flex_1()
                .min_w(px(240.0))
                .flex()
                .flex_col()
                .gap(px(Space::XS))
                .child(label(number, Type::META, colors.accent))
                .child(label(title, Type::BODY, colors.text))
                .child(label(body, Type::LABEL, colors.text_muted))
        };
        let how_it_works = div()
            .flex()
            .flex_col()
            .gap(px(Space::MD))
            .child(label("How Ember works", Type::LABEL, colors.text_faint))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .flex_wrap()
                    .gap(px(Space::XL))
                    .child(how_step(
                        "1",
                        "Ask",
                        "Pick a local model and write a prompt. Ember runs it once, untouched, as the baseline.",
                    ))
                    .child(how_step(
                        "2",
                        "Change one thing",
                        "Choose a spot inside the model and zero, scale or swap what it computes there.",
                    ))
                    .child(how_step(
                        "3",
                        "Compare",
                        "Ember reruns with your change and shows whether the answer moved, and where inside the model it began.",
                    )),
            );

        // The draft section is the honest version of "continue where you left
        // off": it exists only when there is something to resume, and it names
        // the experiment it will reopen.
        let draft_section = self.store.draft.as_ref().map(|draft| {
            let op = draft
                .fields
                .get("op")
                .map(|op| operation_label(op).to_string())
                .unwrap_or_else(|| "Experiment in progress".to_string());
            let site = draft.fields.get("site").map(|site| site_label(site));
            let layer = draft
                .fields
                .get("layer")
                .and_then(|layer| layer.parse::<u32>().ok())
                .map(|layer| format!("L{layer}"));
            let summary = [Some(op), site.map(str::to_string), layer]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join("  ·  ");
            div()
                .flex()
                .flex_col()
                .gap(px(Space::SM))
                .child(label(
                    "Continue where you left off",
                    Type::LABEL,
                    colors.text_faint,
                ))
                .child(
                    Button::new("home-resume")
                        .ghost()
                        .w_full()
                        .justify_start()
                        .h_auto()
                        .py(px(Space::MD))
                        .px(px(Space::MD))
                        .border_1()
                        .border_color(colors.border)
                        .rounded(px(Radius::LG))
                        .accessibility_label("Resume the saved experiment")
                        .child(
                            div()
                                .w_full()
                                .flex()
                                .flex_row()
                                .items_center()
                                .gap(px(Space::MD))
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w(px(0.0))
                                        .overflow_hidden()
                                        .flex()
                                        .flex_col()
                                        .gap(px(Space::XS))
                                        .child(label(summary, Type::BODY, colors.text))
                                        .child(
                                            div()
                                                .flex()
                                                .flex_row()
                                                .gap(px(Space::MD))
                                                .child(mono(
                                                    model_display_name(&draft.model_path),
                                                    Type::META,
                                                    colors.text_muted,
                                                ))
                                                .child(label(
                                                    format!(
                                                        "{} characters",
                                                        draft.prompt.chars().count()
                                                    ),
                                                    Type::META,
                                                    colors.text_faint,
                                                ))
                                                .child(label(
                                                    relative_time(draft.updated_at),
                                                    Type::META,
                                                    colors.text_faint,
                                                )),
                                        ),
                                )
                                .child(label("Resume", Type::LABEL, colors.accent)),
                        )
                        .on_click(resume),
                )
        });

        // Left-aligned children on purpose: the rows carry the full width, and
        // the trailing command hugs its label instead of floating centred.
        let mut recent = div().flex().flex_col().items_start().gap(px(Space::SM));
        recent = recent.child(label("Recent runs", Type::LABEL, colors.text_faint));
        // Home shows the same records the Runs table does, newest first, without
        // the columns: the point of Home is "what did I do last", not "compare
        // two runs". Each row states what was changed, because a bare "Run #7"
        // is not something you can recognise your own work by.
        let recent_runs = self.store.runs_ordered();
        if recent_runs.is_empty() {
            recent = recent
                .child(label("No recent runs", Type::SUBSECTION, colors.text))
                .child(label(
                    "Your completed experiments will appear here.",
                    Type::LABEL,
                    colors.text_faint,
                ));
        } else {
            for run in recent_runs.iter().take(4) {
                let number = run.number;
                recent = recent.child(
                    Button::new(SharedString::from(format!("home-run:{number}")))
                        .ghost()
                        .w_full()
                        .justify_start()
                        .rounded(px(Radius::SM))
                        .accessibility_label(format!(
                            "Run #{}: {}. Open the run history.",
                            run.number, run.intervention
                        ))
                        .child(
                            div()
                                .w_full()
                                .flex()
                                .flex_row()
                                .items_center()
                                .gap(px(Space::SM))
                                .py(px(Space::SM))
                                .when(run.pinned, |row| {
                                    row.child(status_dot(colors.accent, false))
                                })
                                .child(label(
                                    format!("#{}", run.number),
                                    Type::LABEL,
                                    colors.text_faint,
                                ))
                                .child(div().flex_1().min_w(px(0.0)).overflow_hidden().child(
                                    label(
                                        truncate_chars(&run.intervention, 22),
                                        Type::LABEL,
                                        colors.text,
                                    ),
                                ))
                                .child(label(
                                    model_display_name(&run.model),
                                    Type::META,
                                    colors.text_muted,
                                ))
                                .child(label(
                                    relative_time(run.finished_at),
                                    Type::META,
                                    colors.text_faint,
                                )),
                        )
                        .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                            console.goto(View::Runs, cx);
                        })),
                );
            }
            recent = recent.child(
                Button::new("home-all-runs")
                    .ghost()
                    .compact()
                    .label("See all runs")
                    .accessibility_label("See all runs")
                    .on_click(runs),
            );
        }
        div()
            .flex()
            .flex_col()
            .gap(px(Space::XXL))
            .w_full()
            .max_w(px(1040.0))
            .px_6()
            .pt_8()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("What do you want to do?", Type::SECTION, colors.text))
                    .child(label(
                        "Run a controlled experiment, or pick up where you left off.",
                        Type::BODY,
                        colors.text_muted,
                    )),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(Space::MD))
                    .child(
                        Button::new("home-start")
                            .primary()
                            .label("New experiment")
                            .accessibility_label("Start a new experiment")
                            .on_click(start),
                    )
                    // Only when nothing is waiting to be resumed: an example
                    // rewrites the form, and the next save would replace the
                    // saved draft the Resume card offers.
                    .children(self.store.draft.is_none().then(|| {
                        Button::new("home-example")
                            .label("Try an example")
                            .tooltip("Load a ready-made experiment and step through it")
                            .accessibility_label("Try a ready-made example experiment")
                            .on_click(example)
                    }))
                    .child(
                        Button::new("home-sample")
                            .label("See a sample result")
                            .tooltip("Read a finished comparison without running anything")
                            .accessibility_label("Open a sample result")
                            .on_click(sample),
                    )
                    .child(
                        Button::new("home-models")
                            .label("Manage models")
                            .on_click(models),
                    ),
            )
            .children(draft_section)
            .child(how_it_works)
            .child(recent)
    }
}

// -- experiment steps -----------------------------------------------------
impl Console {


    /// Starting points for a new experiment.
    ///
    /// These were in the left rail, which made them look like a mode switch.
    /// They are choices for starting work, so they belong on the Prompt step
    /// where the work starts.
    /// The examples on offer, in the order they are shown.
    pub(super) fn example_entries() -> Vec<(Preset, &'static str, &'static str)> {
        vec![
            (
                Preset::SilenceEarly,
                "Silence an early layer",
                "Watch the answer drift",
            ),
            (
                Preset::ZeroMiddle,
                "Zero a middle layer",
                "Causal ablation, cleanly scoped",
            ),
            (
                Preset::ScaleLate,
                "Weaken a late layer",
                "Near-output layer, at 50%",
            ),
            (
                Preset::CopyEarlier,
                "Copy an earlier layer",
                "Substitute an earlier capture",
            ),
            (
                Preset::ArabicMorphology,
                "Arabic morphology",
                "Matched spans, Arabic prompt",
            ),
        ]
    }

    pub(super) fn presets_block(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let presets = Self::example_entries();
        // Two columns, laid out as explicit rows: a wrapping flex with a fixed
        // half width left the right column short of the edge, and flex_1 let a
        // lone last tile stretch across the whole row. An odd count gets an
        // empty spacer instead.
        let mut tiles: Vec<AnyElement> = presets
            .into_iter()
            .map(|(preset, title, hint)| {
                div()
                    .flex_1()
                    .min_w(px(0.0))
                    .child(self.preset_card(colors, preset, title, hint, cx))
                    .into_any_element()
            })
            .collect();
        if tiles.len() % 2 == 1 {
            tiles.push(div().flex_1().into_any_element());
        }
        let mut grid = div().flex().flex_col().gap(px(Space::SM));
        while !tiles.is_empty() {
            let rest = tiles.split_off(2);
            let row = std::mem::replace(&mut tiles, rest);
            grid = grid.child(div().flex().flex_row().gap(px(Space::SM)).children(row));
        }
        grid
    }

    pub(super) fn preset_card(
        &self,
        colors: &Colors,
        preset: Preset,
        title: &'static str,
        hint: &'static str,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        Button::new(SharedString::from(format!("preset:{title}")))
            .w_full()
            .h_auto()
            .ghost()
            .compact()
            .py(px(Space::SM))
            .px(px(Space::MD))
            .border_1()
            .border_color(colors.border)
            .rounded(px(Radius::MD))
            .accessibility_label(title)
            .child(
                div()
                    .w_full()
                    .flex()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label(title, Type::BODY, colors.text))
                    // The hint wraps instead of ellipsizing: two short lines
                    // read faster than one clipped one.
                    .child(label(hint.to_string(), Type::META, colors.text_faint)),
            )
            .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                console.apply_preset(preset, cx);
            }))
            .into_any_element()
    }

    /// One segment of the generation-length segmented control. The container
    /// owns the silhouette; each segment is a full-height fill inside it.
    fn generation_option(
        &self,
        colors: &Colors,
        value: usize,
        title: &'static str,
        cx: &mut Context<Self>,
    ) -> Button {
        let selected = self.max_tokens == value.to_string();
        Button::new(SharedString::from(format!("generation-length:{value}")))
            .flex_1()
            .h(px(28.0))
            .ghost()
            .selected(selected)
            .accessibility_label(if selected {
                format!("{title}, selected")
            } else {
                format!("{title}: up to {value} tokens")
            })
            .tooltip(format!("Up to {value} tokens"))
            .child(label(
                title,
                Type::LABEL,
                if selected {
                    colors.text
                } else {
                    colors.text_muted
                },
            ))
            .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                console.set_max_tokens(value, cx);
            }))
    }

    /// Generation length as one segmented control: a single quiet container,
    /// the selected segment filled, no per-segment borders.
    pub(super) fn generation_control(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        div()
            .w_full()
            .flex()
            .gap(px(Space::XS))
            .p(px(Space::XS))
            .bg(colors.sidebar)
            .rounded(px(Radius::MD))
            .max_w(px(360.0))
            .child(self.generation_option(colors, 24, "Short", cx))
            .child(self.generation_option(colors, 48, "Medium", cx))
            .child(self.generation_option(colors, 96, "Long", cx))
    }


    fn feedback_banners(&self, colors: &Colors) -> Div {
        // Errors only. An unchanged output is a result, not a caution, so it
        // lives on the result status row instead of ever appearing here.
        let error = self.error.as_ref().map(|error| {
            div()
                .w_full()
                .px_3()
                .py_2()
                .bg(colors.err_box_bg)
                .rounded(px(Radius::MD))
                .child(label(error.clone(), Type::LABEL, colors.err))
        });
        div().flex().flex_col().gap(px(Space::SM)).children(error)
    }


    pub(super) fn layer_stepper(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let n_layers = self.session.as_ref().map(|session| session.n_layers);
        let current = self.layer.parse::<usize>().unwrap_or(0);
        let position = n_layers.map_or("Load a model to see its layer range", |count| {
            let ratio = current as f32 / count.max(1) as f32;
            if ratio < 0.34 {
                "Early in the model"
            } else if ratio < 0.67 {
                "Middle of the model"
            } else {
                "Late in the model"
            }
        });
        let limit = n_layers
            .map(|count| format!(" of {}", count.saturating_sub(1)))
            .unwrap_or_default();

        div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(Space::SM))
                    .child(
                        Button::new("layer-minus")
                            .label("−")
                            .accessibility_label("Decrease layer")
                            .on_click(cx.listener(|console, _: &ClickEvent, _, cx| {
                                console.adjust_layer(-1, cx);
                            })),
                    )
                    .child(div().w(px(76.0)).child(text_input(
                        colors,
                        self.inputs.layer.clone(),
                        FONT_MONO_NAME,
                        Type::META,
                        None,
                        cx,
                    )))
                    .child(label(limit, Type::LABEL, colors.text_muted))
                    .child(
                        Button::new("layer-plus")
                            .label("+")
                            .accessibility_label("Increase layer")
                            .on_click(cx.listener(|console, _: &ClickEvent, _, cx| {
                                console.adjust_layer(1, cx);
                            })),
                    ),
            )
            .child(label(position, Type::LABEL, colors.text_faint))
    }


    /// What a run is doing right now, as steps. A first run loads the model
    /// before it computes anything, and a silent wait there looks like a hang.
    fn run_progress(&self, colors: &Colors) -> Div {
        // (label, state) with state 0 = pending, 1 = active, 2 = done
        let steps: Vec<(&'static str, u8)> = match self.status {
            Status::Preparing => vec![
                ("Load the model", 1),
                ("Run the baseline and the intervention", 0),
                ("Compare the results", 0),
            ],
            Status::Running => vec![
                ("Load the model", 2),
                ("Run the baseline and the intervention", 1),
                ("Compare the results", 0),
            ],
            Status::Restoring => vec![("Replay the baseline and check it matches exactly", 1)],
            Status::Idle => Vec::new(),
        };
        let note = match self.status {
            Status::Preparing => {
                "The first run loads the model, so it takes longer. Later runs start straight away."
            }
            Status::Running => {
                "Ember is running your prompt twice, once untouched and once with your change."
            }
            _ => "",
        };
        div()
            .w_full()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .px(px(Space::MD))
            .py(px(Space::MD))
            .rounded(px(Radius::MD))
            .border_l_2()
            .border_color(colors.busy)
            .bg(colors.accent_soft)
            .children(steps.into_iter().map(|(text, state)| {
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(Space::SM))
                    .child(status_dot(
                        match state {
                            2 => colors.ok,
                            1 => colors.busy,
                            _ => colors.border_strong,
                        },
                        state == 1,
                    ))
                    .child(label(
                        text,
                        Type::BODY,
                        if state == 0 {
                            colors.text_faint
                        } else {
                            colors.text
                        },
                    ))
            }))
            .when(!note.is_empty(), |panel| {
                panel.child(label(note, Type::LABEL, colors.text_muted))
            })
    }
}

// -- results --------------------------------------------------------------
impl Console {
    /// The one-line outcome of a completed run, split the way the experiment
    /// actually splits: did the generated text change, and did the internal
    /// representation. An unchanged output is a result, not a caution, so
    /// nothing here uses warning styling.
    fn result_summary(&self, colors: &Colors) -> Div {
        match (&self.baseline, &self.intervention) {
            (Some(baseline), Some(intervention)) => {
                let (unchanged, diverged_layer, first_step) = match &self.comparison {
                    Some(comparison) => (
                        comparison.generated_text_equal,
                        comparison.landmarks.first_layer_divergence,
                        comparison.first_token_divergence,
                    ),
                    None => (baseline.text == intervention.text, None, None),
                };
                let (dot, line) = match (unchanged, diverged_layer) {
                    (true, Some(layer)) => (
                        colors.warn,
                        format!("Output unchanged \u{00b7} internal state diverged at Layer {layer}"),
                    ),
                    (true, None) => (
                        colors.text_faint,
                        "Output unchanged \u{00b7} no internal change observed".to_string(),
                    ),
                    (false, _) => (
                        colors.accent,
                        first_step.map_or_else(
                            || "Output changed".to_string(),
                            |step| format!("Output changed \u{00b7} first differs at step {step}"),
                        ),
                    ),
                };
                // The verdict says what happened; the sentence under it says
                // what that means, because a result nobody can interpret is
                // just a number.
                let meaning = match (unchanged, diverged_layer) {
                    (true, Some(_)) => "The intervention disturbed the model's internal state, but the words it wrote came out the same. The change was absorbed before it reached the output.",
                    (true, None) => "Nothing measurable changed. Try a stronger change, or an earlier layer, to see an effect.",
                    (false, _) => "The intervention changed what the model wrote. The Layers tab shows where inside the model that change began.",
                };
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::SM))
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(px(Space::SM))
                            .px_3()
                            .py_2()
                            .rounded(px(Radius::MD))
                            .bg(colors.surface_raised)
                            .child(status_dot(dot, false))
                            .child(label(line, Type::SUBSECTION, colors.text)),
                    )
                    .child(div().px_3().child(label(meaning, Type::LABEL, colors.text_muted)))
            }
            // While a run is in flight the progress steps say so; see
            // `run_progress`.
            _ if self.busy() => div(),
            _ => div()
                .px_3()
                .py_2()
                .rounded(px(Radius::MD))
                .bg(colors.surface_raised)
                .child(label(
                    "Review the experiment summary, then run it to produce a controlled comparison.",
                    Type::LABEL,
                    colors.text_muted,
                )),
        }
    }

    fn result_landmarks(&self, colors: &Colors) -> Div {
        let Some(comparison) = &self.comparison else {
            return div();
        };
        let landmarks = &comparison.landmarks;
        let (text_value, text_detail) = if comparison.generated_text_equal {
            (
                "Unchanged".to_string(),
                "exact token-ID match across both runs".to_string(),
            )
        } else {
            (
                "Changed".to_string(),
                comparison.first_token_divergence.map_or_else(
                    || "generated text differs".to_string(),
                    |step| format!("first differs at decode step {step}"),
                ),
            )
        };
        let (first_value, first_detail) = landmarks.first_layer_divergence.map_or_else(
            || {
                (
                    "None observed".to_string(),
                    "no captured layer diverged".to_string(),
                )
            },
            |layer| {
                (
                    format!("Layer {layer}"),
                    "first non-zero captured layer".to_string(),
                )
            },
        );
        let (peak_value, peak_detail) = match (landmarks.peak_relative_l2, landmarks.peak_layer) {
            (Some(value), Some(layer)) => {
                let magnitude = if value.abs() < 0.001 {
                    format!("{value:.2e}")
                } else {
                    format!("{value:.3}")
                };
                (
                    format!("{magnitude} @ L{layer}"),
                    "relative L2 difference".to_string(),
                )
            }
            _ => (
                "None observed".to_string(),
                "relative L2 difference".to_string(),
            ),
        };
        let (tail_value, tail_detail) = if comparison.generated_tokens_equal {
            ("Identical".to_string(), "exact token-ID suffix".to_string())
        } else {
            landmarks.stable_token_tail_step.map_or_else(
                || {
                    (
                        "Not observed".to_string(),
                        "exact token-ID suffix".to_string(),
                    )
                },
                |step| {
                    (
                        format!("From step {step}"),
                        "exact token-ID suffix".to_string(),
                    )
                },
            )
        };
        // The restore leg is the one fact that answers "can I trust this
        // run's evidence", so it earns its color: green verified, red failed,
        // quiet when it has not been run.
        let (restore_value, restore_color, restore_detail) = match &self.restore {
            Some(restore) if !restore.comparable => (
                "Not comparable".to_string(),
                colors.text_muted,
                "configuration changed since the run".to_string(),
            ),
            Some(restore) if restore.matches => (
                "Verified".to_string(),
                colors.ok,
                "bit-exact baseline replay".to_string(),
            ),
            Some(_) => (
                "Differs".to_string(),
                colors.err,
                "restore replay differs from baseline".to_string(),
            ),
            None => (
                "Not run".to_string(),
                colors.text_faint,
                "verify the baseline replays bit-exactly".to_string(),
            ),
        };
        let landmark = |title: &'static str, value: String, detail: String, value_color: Rgba| {
            // Hovering a term explains it: the numbers are the point of the
            // page, and their names are the part a newcomer cannot guess.
            let meaning = match title {
                "Text output" => "Whether the words the model wrote are identical with and without your change.",
                "First internal divergence" => "The earliest layer where the model's internal activations differ from the baseline run.",
                "Peak divergence" => "The largest difference at any layer, as relative L2: how big the change is compared with the activation itself. 0 means identical.",
                "Token tail" => "Whether the generated token IDs match exactly from some step to the end.",
                _ => "Replays the baseline after the intervention to prove the model can be restored bit for bit.",
            };
            div()
                .flex_1()
                .min_w(px(150.0))
                .flex()
                .flex_col()
                .gap(px(Space::XS))
                .child(
                    label(title, Type::META, colors.text_faint)
                        .id(ElementId::Name(SharedString::from(format!(
                            "landmark:{title}"
                        ))))
                        .tooltip(move |window, cx| Tooltip::new(meaning).build(window, cx)),
                )
                .child(label(value, Type::VALUE, value_color))
                .child(label(detail, Type::META, colors.text_muted))
        };

        // Five facts, one band of space, no boxes: the row is the densest
        // reading on the page, so it stays typography and spacing only.
        div()
            .w_full()
            .flex()
            .flex_wrap()
            .items_start()
            .gap(px(Space::XL))
            .py_1()
            .child(landmark(
                "Text output",
                text_value,
                text_detail,
                colors.text,
            ))
            .child(landmark(
                "First internal divergence",
                first_value,
                first_detail,
                colors.text,
            ))
            .child(landmark(
                "Peak divergence",
                peak_value,
                peak_detail,
                colors.text,
            ))
            .child(landmark("Token tail", tail_value, tail_detail, colors.text))
            .child(landmark(
                "Restore",
                restore_value,
                restore_detail,
                restore_color,
            ))
    }

    /// Result sub-views: Overview, Layers, Tokens, Trace.
    fn result_tabs(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let tabs: Vec<(&'static str, &'static str, String)> = ResultView::ALL
            .iter()
            .map(|view| {
                (
                    view.key(),
                    view.label(),
                    format!("{} of 4 result views", view.label()),
                )
            })
            .collect();
        let active = ResultView::ALL
            .iter()
            .position(|view| *view == self.result_view)
            .unwrap_or(0);
        div()
            .w_full()
            .child(self.tab_row(colors, cx, "result", &tabs, active))
    }

    fn intervention_layer_for_result(&self) -> Option<usize> {
        let context = self.result_context.as_ref()?;
        per_layer(&context.site)
            .then(|| context.layer.parse::<usize>().ok())
            .flatten()
    }

    fn layer_chart_panel(&self, colors: &Colors, height: f32, cx: &mut Context<Self>) -> Div {
        let csv = {
            let mut text = String::from(
                "layer,relative_l2_difference,cosine_distance,maximum_absolute_difference,exact\n",
            );
            for metric in self.layer_series.iter() {
                text.push_str(&format!(
                    "{},{},{},{},{}\n",
                    metric.layer,
                    metric
                        .relative_l2_difference
                        .map_or_else(String::new, |value| value.to_string()),
                    metric
                        .cosine_distance
                        .map_or_else(String::new, |value| value.to_string()),
                    metric
                        .maximum_absolute_difference
                        .map_or_else(String::new, |value| value.to_string()),
                    metric.exact
                ));
            }
            text
        };
        let export = text_button(
            "copy-layer-csv",
            "Copy CSV",
            cx.listener(move |_console, _: &ClickEvent, _window, cx| {
                cx.write_to_clipboard(ClipboardItem::new_string(csv.clone()));
            }),
        );
        panel(
            colors,
            div()
                .flex()
                .flex_col()
                .gap(px(Space::SM))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(Space::XS))
                                .child(label(
                                    "Representation divergence",
                                    Type::LABEL,
                                    colors.text_faint,
                                ))
                                .child(label(
                                    "At which layers does the intervention diverge from baseline?",
                                    Type::BODY,
                                    colors.text,
                                )),
                        )
                        .child(div().w_full())
                        .child(export),
                )
                .child(chart::layer_divergence_chart(
                    cx.entity(),
                    self.layer_series.clone(),
                    self.reference.as_ref().map(|reference| reference.layers.clone()),
                    self.intervention_layer_for_result(),
                    self.selected_layer,
                    self.hovered_layer,
                    height,
                    colors,
                )),
        )
    }

    fn paired_outputs(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        div()
            .flex()
            .gap(px(Space::MD))
            .child(self.output_panel(colors, "Baseline", self.baseline.as_ref(), self.status, cx))
            .child(self.output_panel(
                colors,
                "Intervention",
                self.intervention.as_ref(),
                self.status,
                cx,
            ))
    }

    fn token_comparison_panel(&self, colors: &Colors) -> Div {
        let Some(comparison) = &self.comparison else {
            return panel(
                colors,
                label(
                    "No token comparison is available.",
                    Type::LABEL,
                    colors.text_faint,
                ),
            );
        };
        let count = comparison.tokens.len();
        let center = comparison
            .first_token_divergence
            .unwrap_or(1)
            .saturating_sub(1);
        let start = if count > 160 {
            center.saturating_sub(48).min(count.saturating_sub(160))
        } else {
            0
        };
        let end = (start + 160).min(count);
        let token_cells = comparison.tokens[start..end]
            .iter()
            .map(|token| {
                let baseline = token
                    .baseline_text
                    .as_deref()
                    .map(isolate_bidi)
                    .unwrap_or_else(|| "—".to_string());
                let intervention = token
                    .intervention_text
                    .as_deref()
                    .map(isolate_bidi)
                    .unwrap_or_else(|| "—".to_string());
                div()
                    .w(px(92.0))
                    .flex_none()
                    .p(px(Space::SM))
                    .flex()
                    .flex_col()
                    .gap(px(Space::XS))
                    .bg(if token.differs {
                        colors.accent_soft
                    } else {
                        colors.surface_raised
                    })
                    .border_1()
                    .border_color(if token.differs {
                        colors.accent
                    } else {
                        colors.border
                    })
                    .rounded(px(Radius::MD))
                    .child(mono(
                        format!("step {}", token.position),
                        Type::MICRO,
                        colors.text_faint,
                    ))
                    .child(label(baseline, Type::BODY, colors.text))
                    .child(rule_h(colors))
                    .child(label(intervention, Type::BODY, colors.text))
                    .into_any_element()
            })
            .collect::<Vec<_>>();
        let divergence = comparison.first_token_divergence;
        let changed = comparison
            .tokens
            .iter()
            .filter(|token| token.differs)
            .count();
        // The verdict first, in words a beginner can act on; the decode-step
        // arithmetic stays as the secondary line.
        let (verdict, verdict_color, verdict_detail) = if comparison.generated_tokens_equal {
            (
                "Outputs match exactly",
                colors.text,
                format!("{count} tokens \u{00b7} token IDs identical"),
            )
        } else {
            (
                "Outputs diverge",
                colors.accent,
                divergence.map_or_else(
                    || format!("{changed} of {count} tokens differ"),
                    |step| {
                        format!("first at step {step} \u{00b7} {changed} of {count} tokens differ")
                    },
                ),
            )
        };
        panel(
            colors,
            div()
                .flex()
                .flex_col()
                .gap(px(Space::SM))
                .child(label(
                    "Token-level output comparison",
                    Type::LABEL,
                    colors.text_faint,
                ))
                .child(label(verdict, Type::BODY, verdict_color))
                .child(label(verdict_detail, Type::META, colors.text_muted))
                .child(
                    div()
                        .flex()
                        .gap(px(Space::SM))
                        .child(chip("Baseline", colors.text_muted))
                        .child(chip("Intervention", colors.accent))
                        .child(div().w_full())
                        .children((count > end).then(|| {
                            mono(
                                format!("showing {}–{} of {count}", start + 1, end),
                                Type::MICRO,
                                colors.text_faint,
                            )
                        })),
                )
                .child(
                    div()
                        .id("token-trace-scroll")
                        .w_full()
                        .overflow_x_scroll()
                        .flex()
                        .gap(px(Space::XS))
                        .pb_2()
                        .children(token_cells),
                ),
        )
    }

    fn raw_trace_panel(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let output = self.intervention.as_ref();
        // The pretty printer re-indents the same JSON value; it does not
        // change what is in it. The copy action copies this exact text.
        let events: Vec<String> = output
            .map(|output| {
                output
                    .events
                    .iter()
                    .filter_map(|event| serde_json::to_string_pretty(event).ok())
                    .collect()
            })
            .unwrap_or_default();
        let trace_text = events.join("\n\n");
        let copy = (!events.is_empty()).then(|| {
            text_button(
                "copy-trace",
                "Copy",
                cx.listener(move |_console, _: &ClickEvent, _window, cx| {
                    cx.write_to_clipboard(ClipboardItem::new_string(trace_text.clone()));
                }),
            )
        });
        let (verified_chip, verified_color) = match &self.verification {
            Some(verification) if verification.ok => ("Verified", colors.ok),
            Some(_) => ("Not verified", colors.err),
            None => ("Not checked", colors.text_faint),
        };
        let mut body = div().flex().flex_col().gap(px(Space::XS));
        if events.is_empty() {
            body = body.child(label(
                "No trace events were recorded for this run.",
                Type::LABEL,
                colors.text_faint,
            ));
        } else {
            for event in &events {
                body = body.child(mono(event.clone(), Type::META, colors.text_muted));
            }
        }
        panel(
            colors,
            div()
                .flex()
                .flex_col()
                .gap(px(Space::SM))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(Space::SM))
                        .child(label(
                            "Raw intervention trace",
                            Type::LABEL,
                            colors.text_faint,
                        ))
                        .child(chip(verified_chip, verified_color))
                        .child(div().w_full())
                        .children(copy),
                )
                .children(output.map(|output| {
                    mono(
                        format!("bundle  {}", output.bundle_dir),
                        Type::META,
                        colors.text_faint,
                    )
                }))
                .child(body),
        )
    }

    fn review_step(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let has_results = self.baseline.is_some() && self.intervention.is_some();
        let result_body = if !has_results {
            div()
                .flex()
                .flex_col()
                .gap(px(Space::MD))
                .child(self.result_summary(colors))
                .child(self.paired_outputs(colors, cx))
                .into_any_element()
        } else {
            match self.result_view {
                ResultView::Overview => div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(self.result_summary(colors))
                    .child(self.result_landmarks(colors))
                    .children(self.reference_panel(colors, cx))
                    .child(self.paired_outputs(colors, cx))
                    .child(self.layer_chart_panel(colors, 210.0, cx))
                    .into_any_element(),
                ResultView::Layers => div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .children(self.reference_panel(colors, cx))
                    .child(self.layer_chart_panel(colors, 380.0, cx))
                    .into_any_element(),
                ResultView::Tokens => div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(self.paired_outputs(colors, cx))
                    .child(self.token_comparison_panel(colors))
                    .into_any_element(),
                ResultView::Trace => div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(self.raw_trace_panel(colors, cx))
                    .child(self.verification_panel(colors))
                    .into_any_element(),
            }
        };

        div()
            .flex()
            .flex_col()
            .gap(px(Space::LG))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_end()
                    .child(
                        div()
                            .flex_1()
                            // The title never wraps: it keeps its natural
                            // width and the action buttons drop below it
                            // when the row is too narrow for both.
                            .min_w(px(340.0))
                            .flex()
                            .flex_col()
                            .gap(px(Space::XS))
                            .child(label("Results", Type::TITLE, colors.text).whitespace_nowrap())
                            .child(label(
                                "The baseline and your change run on the same prompt with deterministic settings.",
                                Type::BODY,
                                colors.text_muted,
                            )),
                    )
                    .gap(px(Space::MD))
                    // The branch loop, one click after a run: replay the
                    // exact configuration, or duplicate it and change one
                    // field. Quiet text commands; the bottom bar stays the
                    // primary runner.
                    .when(has_results, |header| {
                        header.child(text_button(
                            "review-copy",
                            if self.copied { "Copied" } else { "Copy summary" },
                            cx.listener(|console, _: &ClickEvent, _window, cx| {
                                if let Some(markdown) = console.result_markdown() {
                                    cx.write_to_clipboard(ClipboardItem::new_string(markdown));
                                    console.copied = true;
                                    cx.notify();
                                }
                            }),
                        ))
                    })
                    // A sample offers neither; a run reopened from History offers
                    // both, since the banner says Duplicate branches from it.
                    .when(has_results && (!self.sample || self.saved_run.is_some()), |header| {
                        header
                            .child(text_button(
                                "review-pin",
                                if self.reference.is_some() { "Re-pin reference" } else { "Pin as reference" },
                                cx.listener(|console, _: &ClickEvent, _window, cx| {
                                    console.pin_reference(cx);
                                }),
                            ))
                            .child(text_button(
                                "review-rerun",
                                "Rerun",
                                cx.listener(|console, _: &ClickEvent, _window, cx| {
                                    console.rerun(cx);
                                }),
                            ))
                    })
                    .when(self.last_config.is_some(), |header| {
                        header.child(
                            div()
                                .w(px(132.0))
                                .flex_shrink_0()
                                .child(btn_secondary(
                                    colors,
                                    if self.status == Status::Restoring {
                                        "Verifying…"
                                    } else {
                                        "Verify restore"
                                    },
                                    (!self.busy()).then(|| {
                                        cx.listener(
                                            |console, _: &ClickEvent, _window, cx| {
                                                console.restore();
                                                cx.notify();
                                            },
                                        )
                                    }),
                                )),
                        )
                    }),
            )
            .when(self.busy(), |page| page.child(self.run_progress(colors)))
            .when(has_results && !self.busy() && self.results_stale(), |page| {
                page.child(self.stale_notice(colors))
            })
            .when(has_results && self.sample, |page| {
                page.child(
                    div()
                        .w_full()
                        .flex()
                        .flex_col()
                        .gap(px(Space::XS))
                        .px(px(Space::MD))
                        .py(px(Space::SM))
                        .rounded(px(Radius::MD))
                        .border_l_2()
                        .border_color(colors.accent)
                        .bg(colors.accent_soft)
                        .child(label(
                            match self.saved_run {
                                Some(number) => format!("Saved run #{number}"),
                                None => "Sample result".to_string(),
                            },
                            Type::LABEL,
                            colors.accent,
                        ))
                        .child(label(
                            match self.saved_run {
                                Some(_) => "Reopened from your history. Duplicate branches from it, or run it again to check it still reproduces.",
                                None => "Illustrative data, so you can see what a finished comparison looks like. Run your own experiment and it replaces this.",
                            },
                            Type::LABEL,
                            colors.text,
                        )),
                )
            })
            .when(has_results, |page| page.child(div().pt(px(Space::SM)).child(self.result_tabs(colors, cx))))
            .child(result_body)
            .when(has_results && self.result_view != ResultView::Trace, |page| {
                page.child(self.verification_panel(colors))
            })
    }

    fn output_panel(
        &self,
        colors: &Colors,
        title: &'static str,
        output: Option<&RunOutput>,
        status: Status,
        cx: &mut Context<Self>,
    ) -> Div {
        let (badge_text, badge_color) = match (output, status) {
            (Some(_), _) => ("ok", colors.ok),
            (None, Status::Running) => ("run", colors.warn),
            (None, _) => ("\u{2014}", colors.text_faint),
        };
        let copy_button = output.map(|output| {
            let text = output.text.clone();
            text_button(
                "copy-output",
                "Copy",
                cx.listener(move |_console, _: &ClickEvent, _window, cx| {
                    cx.write_to_clipboard(ClipboardItem::new_string(text.clone()));
                }),
            )
        });
        let divergence_note = self.comparison.as_ref().map(|comparison| {
            if comparison.generated_tokens_equal {
                "Text output unchanged".to_string()
            } else if title == "Baseline" {
                comparison.first_token_divergence.map_or_else(
                    || "generated token sequence changed".to_string(),
                    |step| {
                        let prefix = step.saturating_sub(1);
                        format!(
                            "shared prefix  ·  {prefix} decode step{}",
                            if prefix == 1 { "" } else { "s" }
                        )
                    },
                )
            } else {
                comparison.first_token_divergence.map_or_else(
                    || "generated token sequence differs".to_string(),
                    |step| format!("first changed token  ·  step {step}"),
                )
            }
        });
        let body: Div = match output {
            Some(out) if !out.text.is_empty() => {
                let display_text = isolate_bidi(&out.text);
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::SM))
                    // The text is the point of this page, so it gets the room:
                    // a floor and a generous ceiling rather than a 164px cap
                    // that showed three lines and left the pane half empty.
                    .child(
                        div()
                            .id(ElementId::Name(SharedString::from(format!(
                                "output-scroll:{title}"
                            ))))
                            .flex_1()
                            .min_h(px(88.0))
                            .max_h(px(420.0))
                            .overflow_y_scroll()
                            .w_full()
                            .px_3()
                            .py_3()
                            // The raised fill only: the pane already owns the
                            // boundary, and a second hairline inside it read
                            // as a box inside a box.
                            .bg(colors.surface_raised)
                            .rounded(px(Radius::MD))
                            .child(multiline(
                                &display_text,
                                Type::OUTPUT,
                                colors.text,
                                FONT_ARABIC_NAME,
                            )),
                    )
                    .children(divergence_note.map(|note| {
                        mono(
                            note,
                            Type::LABEL,
                            if title == "Intervention" {
                                colors.accent
                            } else {
                                colors.text_muted
                            },
                        )
                    }))
                    // One metadata row. This was three, the last of them a
                    // full bundle path sitting in the middle of the default
                    // view -- the page the brief says must not be centred on
                    // filesystem locations. The path and the semantic hash are
                    // reproducibility facts, so they stay reachable on the
                    // pane's tooltip rather than being deleted.
                    .child(
                        div()
                            .id(ElementId::Name(SharedString::from(format!(
                                "output-meta:{title}"
                            ))))
                            // Repro details on hover rather than in the middle
                            // of the default view. `tooltip` on a plain `Div`
                            // does not exist -- it is an interactive-element
                            // method, so the row needs an id first, and the kit
                            // owns the overlay rather than a hand-rolled one.
                            .tooltip({
                                // Owned, because the overlay outlives this
                                // frame and cannot borrow the run output.
                                let detail = format!(
                                    "prompt {} tok  ·  bundle {}  ·  {}",
                                    out.prompt_tokens,
                                    short_id(&out.semantic_hash),
                                    out.bundle_dir
                                );
                                move |window, cx| Tooltip::new(detail.clone()).build(window, cx)
                            })
                            .child(mono(
                                if out.bundle_dir == "history" {
                                    format!("{} tok  ·  saved", out.generated_tokens)
                                } else {
                                    format!(
                                        "{} tok  ·  {}  ·  {}",
                                        out.generated_tokens,
                                        fmt_ms(out.wall_ms),
                                        fmt_tps(out.decode_tps)
                                    )
                                },
                                Type::META,
                                colors.text_faint,
                            )),
                    )
            }
            Some(_out) => div().child(label("(empty output)", Type::SUBSECTION, colors.text_faint)),
            None => div().child(label(
                "no run yet \u{2014} outputs appear here",
                Type::META,
                colors.text_faint,
            )),
        };
        // Not `panel()`: a pane is a grouped object with real boundaries, so it
        // keeps a surface, but the header sits outside the scrolling body so it
        // does not leave with the text.
        div()
            .flex()
            .flex_col()
            .w(relative(0.5))
            .min_w(px(0.0))
            .min_h(px(150.0))
            .overflow_hidden()
            .rounded(px(Radius::LG))
            .border_1()
            .border_color(colors.border)
            .bg(colors.surface)
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(Space::SM))
                    .flex_none()
                    .px(px(Space::LG))
                    .pt(px(Space::MD))
                    .pb(px(Space::SM))
                    .border_b_1()
                    .border_color(colors.border)
                    .child(label(title, Type::SUBSECTION, colors.text))
                    .child(div().w_full())
                    .children(copy_button)
                    .child(chip(badge_text, badge_color)),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_h(px(0.0))
                    .gap(px(Space::SM))
                    .p(px(Space::MD))
                    .child(body),
            )
    }

    fn verification_panel(&self, colors: &Colors) -> Div {
        let (badge, badge_color) = match (&self.verification, self.status) {
            (Some(verification), _) if verification.ok => ("Verified", colors.ok),
            (Some(_), _) => ("Verification failed", colors.err),
            (None, Status::Running) => ("Running", colors.warn),
            (None, Status::Restoring) => ("Restoring", colors.warn),
            (None, _) => ("Not run", colors.text_faint),
        };
        let mut lines: Vec<String> = Vec::new();
        if let Some(restore) = &self.restore {
            if !restore.comparable {
                lines.push("restore: baseline not comparable (configuration changed)".to_string());
            } else if restore.matches {
                lines.push("restore: bit-exact".to_string());
            } else {
                lines.push("restore: differs from baseline".to_string());
            }
        } else if self.verification.is_some() {
            lines.push("restore: not run".to_string());
        }
        if let Some(verification) = &self.verification {
            let failed = verification.failed();
            if failed.is_empty() {
                lines.push(format!(
                    "bundle self-check {}/{} passed",
                    verification.checks.len(),
                    verification.checks.len()
                ));
            } else {
                for (name, _, detail) in failed {
                    lines.push(format!("check failed: {name} \u{2014} {detail}"));
                }
            }
            for warning in &verification.warnings {
                lines.push(format!("warning: {warning}"));
            }
        }
        let detail = if lines.is_empty() {
            div().child(label(
                "bundle self-verification and the restore-original leg report here.",
                Type::LABEL,
                colors.text_faint,
            ))
        } else {
            div().flex().flex_col().gap(px(Space::XS)).children(
                lines
                    .iter()
                    .map(|line| {
                        mono(line.clone(), Type::LABEL, colors.text_muted).into_any_element()
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let metrics = match &self.last_metrics {
            Some((bundle, elapsed, tps)) => format!(
                "bundle {} \u{00b7} {} \u{00b7} {}",
                short_id(bundle),
                fmt_ms(*elapsed),
                fmt_tps(*tps)
            ),
            None => String::new(),
        };
        panel(
            colors,
            div()
                .flex()
                .flex_col()
                .gap(px(Space::XS))
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .child(chip(badge, badge_color))
                        .child(div().w_full())
                        .child(mono(metrics, Type::LABEL, colors.text_faint)),
                )
                .child(detail),
        )
    }
}

/// The keyboard shortcuts, in one place so Settings cannot drift from the
/// bindings in the key handler and the palette.
fn shortcut_rows() -> Vec<(String, &'static str)> {
    let cmd = if cfg!(target_os = "macos") {
        "Cmd"
    } else {
        "Ctrl"
    };
    vec![
        (format!("{cmd}+K"), "Command palette"),
        (format!("{cmd}+Enter"), "Continue, or run the experiment"),
        (format!("{cmd}+N"), "New experiment"),
        (format!("{cmd}+R"), "Rerun the last run"),
        (format!("{cmd}+1 / 2 / 3"), "Prompt / Intervention / Review"),
        (
            format!("{cmd}+1 - 4 on Review"),
            "Overview / Layers / Tokens / Raw trace",
        ),
        (format!("{cmd}+B"), "Show or hide the sidebar"),
        (format!("{cmd}+Shift+I"), "Show or hide the inspector"),
    ]
}

/// Window widths below which the inspector, then the sidebar, fold away.
/// Column width of the Prompt and Intervention forms.
/// The inspector reads as a property list (name left, value right), which
/// needs a little more room than a stacked label did.
pub(super) const INSPECTOR_WIDTH: f32 = 344.0;
const INSPECTOR_MIN_WINDOW: f32 = 1560.0;
const SIDEBAR_MIN_WINDOW: f32 = 1240.0;

impl Render for Console {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Narrow windows give up chrome in a fixed order -- inspector first,
        // then the sidebar -- so the workspace keeps the room its headings
        // need. The stored flags are untouched: widening the window brings
        // both panes back exactly as the user left them.
        let width = f32::from(window.viewport_size().width);
        self.inspector_fits = width >= INSPECTOR_MIN_WINDOW;
        let show_inspector = self.inspector_open && self.inspector_fits;
        let show_sidebar = self.sidebar_open && width >= SIDEBAR_MIN_WINDOW;
        let colors = self.colors();
        let topbar = self.topbar(&colors, cx);
        let statusbar = self.statusbar(&colors, cx);

        // The content region depends on the destination. Only the experiment
        // view carries the workflow stepper and the contextual inspector; the
        // other views are plain pages.
        let content: AnyElement = match self.view {
            View::Home => self.home_view(&colors, cx).into_any_element(),
            View::Models => self.models_view(&colors, cx).into_any_element(),
            View::Runs => self.runs_view(&colors, window, cx).into_any_element(),
            View::Settings => self.settings_view(&colors, cx).into_any_element(),
            View::Experiment => {
                let inspector = self.advanced_inspector(&colors, cx);
                // The column is the growing region; the inspector is a fixed
                // aside beside it. Note the direction is set once here --
                // re-calling .flex() on this element would silently override
                // flex_col with a row and squeeze the workspace out.
                let column = div()
                    .flex()
                    .flex_col()
                    // No `w_full()` here. `width: 100%` overrides the
                    // `flex-basis: 0%` that `flex_1` sets, so the workspace
                    // claimed the whole row and the 300px aside was left with
                    // whatever was over -- about 118px, clipping the model name,
                    // the hook and the advanced disclosure mid-word. `flex_1`
                    // already means "take the space that is not spoken for".
                    .flex_1()
                    .min_h(px(0.0))
                    // The workspace is the pane that must give way. Without an
                    // explicit min-width it keeps its intrinsic width and shoves
                    // the inspector past the right edge of the window.
                    .min_w(px(0.0))
                    .child(self.main_panel(&colors, cx));
                if show_inspector {
                    // The inspector is an aside beside the workspace, not a
                    // band below it: the row is the thing that places them.
                    //
                    // No `w_full()` here, and an explicit `min_w(0)`: width
                    // 100% overrides the flex-basis `flex_1` sets, and without
                    // a min-width the row's shrink is clamped at its
                    // min-content -- so an intrinsically wide page pushed the
                    // 300px aside past the window edge, leaving roughly the
                    // first 100px visible. Same trap the workspace column
                    // comment below describes, one level up.
                    div()
                        .flex()
                        .flex_row()
                        .flex_1()
                        .min_w(px(0.0))
                        .min_h(px(0.0))
                        .child(column)
                        .child(
                            div()
                                .id(ElementId::Name(SharedString::from("inspector")))
                                // Observe the aside. Without this its id is only
                                // a scope in its children's paths, so a test can
                                // find the controls inside it but not the box
                                // itself -- which is why the width bug below had
                                // no test to catch it. A no-op outside the
                                // `test-support` feature.
                                .test_support()
                                .w(px(INSPECTOR_WIDTH))
                                .flex_none()
                                // `flex_none` alone was not enough: the row
                                // still took the shortfall out of the aside, and
                                // a 300px inspector was rendering at about 120px
                                // with the model name, the hook and the advanced
                                // disclosure all clipped mid-word. The shrink is
                                // pinned explicitly so the workspace column is
                                // the only thing that gives way.
                                .flex_shrink_0()
                                .h_full()
                                .overflow_y_scroll()
                                .bg(colors.surface)
                                .border_l_1()
                                .border_color(colors.border)
                                .p(px(Space::LG))
                                .child(inspector),
                        )
                        .into_any_element()
                } else {
                    column.into_any_element()
                }
            }
        };

        let body = div()
            .flex()
            .flex_row()
            .w_full()
            .flex_1()
            .min_h(px(0.0))
            .when(show_sidebar, |row| row.child(self.nav_rail(&colors, cx)))
            .child(content);

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(colors.canvas)
            .relative()
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(|console, event: &KeyDownEvent, window, cx| {
                console.picker_key(event, window, cx);
            }))
            .child(topbar.flex_none())
            .child(body)
            .child(statusbar.flex_none())
            .when(self.palette_open, |shell| {
                shell.child(self.palette_overlay(&colors, cx))
            })
    }
}

// ---------------------------------------------------------------------------
// formatting helpers
// ---------------------------------------------------------------------------

fn short_id(hash: &str) -> String {
    if hash.len() > 6 {
        format!("{}\u{2026}", &hash[..6])
    } else {
        hash.to_string()
    }
}

/// The shortcut hint a palette command carries, when it has one. Parsed from
/// the same spelling the keybindings use, so the hint cannot drift from the
/// binding it advertises.
fn palette_shortcut(command: Command) -> Option<Kbd> {
    let stroke = match command {
        Command::NewExperiment => "cmd-n",
        Command::RerunExperiment => "cmd-r",
        Command::RunExperiment => "cmd-enter",
        Command::ToggleSidebar => "cmd-b",
        Command::ToggleInspector => "cmd-shift-i",
        _ => return None,
    };
    Some(Kbd::new(Keystroke::parse(stroke).ok()?))
}

fn fmt_ms(ms: f64) -> String {
    format!("{:.2} s", ms / 1000.0)
}

fn fmt_tps(tps: Option<f64>) -> String {
    match tps {
        Some(tps) => format!("{tps:.1} tok/s"),
        None => "\u{2014}".to_string(),
    }
}

/// Show a file in the platform file manager, selected.
///
/// One command per platform, spawned detached: the file manager is not ours
/// to wait on, and a failure is a banner, not a crash.
fn reveal_in_finder(path: &str) -> Result<(), String> {
    let result = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
            .arg("-R")
            .arg(path)
            .spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("explorer")
            .arg(format!("/select,{path}"))
            .spawn()
    } else {
        let parent = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("/"));
        std::process::Command::new("xdg-open").arg(parent).spawn()
    };
    result.map(|_| ()).map_err(|error| error.to_string())
}

fn isolate_bidi(text: &str) -> String {
    format!("\u{2068}{text}\u{2069}")
}
