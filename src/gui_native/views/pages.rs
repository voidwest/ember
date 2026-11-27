use super::results::shortcut_rows;
use super::*;
use crate::gui_native::*;

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
    pub(super) fn models_view(&mut self, colors: &Colors, cx: &mut Context<Self>) -> Div {
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
                            quant_of(path),
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
    pub(super) fn runs_view(
        &mut self,
        colors: &Colors,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        if self.comparing.is_some() {
            return div()
                .id("compare-scroll")
                .test_support()
                .flex_1()
                .min_w(px(0.0))
                .h_full()
                .overflow_y_scroll()
                .overflow_x_hidden()
                .px_5()
                .pt_6()
                .pb_6()
                .child(self.compare_view(colors, cx))
                .into_any_element();
        }
        let row_count = self.store.runs.len();
        let store = &self.store;
        // Row actions mutate the store through the owning console, so the
        // delegate needs a handle that does not keep the console alive.
        let console = cx.entity().downgrade();
        let table = match &self.runs_table {
            Some(state) => {
                let console = console;
                let picks = &self.compare_picks;
                state.update(cx, |state, cx| {
                    state
                        .delegate_mut()
                        .sync(store, picks, *colors, console, cx);
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

        // The table's region may shrink to whatever the page has left: the
        // kit table is a virtual list, so it builds only the rows it shows
        // and scrolls the rest. Everything else on the page keeps its size.
        let mut body: Div = div().flex().flex_col().min_h(px(0.0));
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
            // than guessed: a fixed 420px left a void under a short history.
            // That is the most the region asks for, not what it gets: with
            // 500 runs it used to be 16,000px tall, the table's virtual list
            // saw every row as visible, and every frame built all of them.
            // Now the region shrinks to the page (and the table scrolls), so
            // a frame builds only the rows on screen. A short history still
            // gets exactly its height, with no void under it.
            let height = 34.0 + row_count as f32 * 32.0;
            body = body.child(
                div()
                    .id("runs-table-scroll")
                    .w_full()
                    .h(px(height))
                    .flex_shrink(1.0)
                    .min_h(px(0.0))
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
            // Never wider than the window: the table scrolls sideways in its
            // own region instead of widening the page (and every bar on it).
            .flex_1()
            .min_w(px(0.0))
            // Exactly the window's height, so the table region is bounded.
            .h_full()
            .min_h(px(0.0))
            .px_5()
            .pt_6()
            .pb_4()
            .child(
                self.section_header(
                    colors,
                    "Runs",
                    &format!(
                        "{row_count} recorded experiment{}.",
                        if row_count == 1 { "" } else { "s" }
                    ),
                )
                .flex_none(),
            )
            .children(
                self.compare_bar(colors, cx)
                    .map(|bar| div().flex_none().child(bar)),
            )
            .children(
                self.export_bar(colors, cx)
                    .map(|bar| div().flex_none().child(bar)),
            )
            .child(body)
            .when(row_count > 0, |page| {
                page.child(label(
                    "Select two runs to compare them side by side. Export copies a run as Markdown or gets its verifiable bundle. Open shows a run's result, Pin makes it the reference for your next run, and Reuse loads its settings. Star keeps a run at the top.",
                    Type::LABEL,
                    colors.text_faint,
                ).flex_none())
            })
            .into_any_element()
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
    pub(super) fn settings_view(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let store_path = app_store::store_path().map_or_else(
            || "(not saved)".to_string(),
            |path| path.display().to_string(),
        );

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
                                "The sidebar and the in-progress experiment are remembered between launches.",
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

    pub(super) fn home_view(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
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
