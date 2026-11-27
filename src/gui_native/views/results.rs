use super::*;
use crate::gui_native::*;

// -- results --------------------------------------------------------------
impl Console {
    /// The one-line outcome of a completed run, split the way the experiment
    /// actually splits: did the generated text change, and did the internal
    /// representation. An unchanged output is a result, not a caution, so
    /// nothing here uses warning styling.
    pub(super) fn result_summary(&self, colors: &Colors) -> Div {
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

    pub(super) fn result_landmarks(&self, colors: &Colors) -> Div {
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
    pub(super) fn result_tabs(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let views = self.result_views();
        let count = views.len();
        let tabs: Vec<(&'static str, &'static str, String)> = views
            .iter()
            .map(|view| {
                (
                    view.key(),
                    view.label(),
                    format!("{} of {count} result views", view.label()),
                )
            })
            .collect();
        let active = views
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

    pub(super) fn layer_chart_panel(
        &self,
        colors: &Colors,
        height: f32,
        cx: &mut Context<Self>,
    ) -> Div {
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
            cx.listener(move |console, _: &ClickEvent, _window, cx| {
                cx.write_to_clipboard(ClipboardItem::new_string(csv.clone()));
                console.show_toast("Copied the layer table as CSV", cx);
            }),
        );
        // A drawing of the chart for slides and papers, in the current theme.
        let svg = {
            let change = self
                .result_context
                .as_ref()
                .map(super::workspace::change_summary)
                .unwrap_or_default();
            chart::layers_svg(
                &self.layer_series,
                self.reference
                    .as_ref()
                    .map(|reference| &reference.layers[..]),
                self.intervention_layer_for_result(),
                chart::SvgLabels {
                    title: "Representation divergence by layer",
                    subtitle: &change,
                    x_axis: "Transformer layer",
                    y_axis: "Relative L2 difference from baseline",
                },
                colors,
            )
        };
        let save_svg = text_button(
            "save-layer-svg",
            "Save SVG",
            cx.listener(move |console, _: &ClickEvent, _window, cx| {
                console.save_svg("ember-layers.svg", svg.clone(), cx);
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
                        .child(save_svg)
                        .child(export),
                )
                .child(chart::layer_divergence_chart(
                    cx.entity(),
                    self.layer_series.clone(),
                    self.reference
                        .as_ref()
                        .map(|reference| reference.layers.clone()),
                    chart::LayerMarks {
                        intervention: self.intervention_layer_for_result(),
                        selected: self.selected_layer,
                        hovered: self.hovered_layer,
                    },
                    height,
                    colors,
                )),
        )
    }

    pub(super) fn paired_outputs(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
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

    pub(super) fn token_comparison_panel(&self, colors: &Colors) -> Div {
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

    pub(super) fn raw_trace_panel(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
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

    pub(super) fn review_step(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
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
                ResultView::Sweep => self.sweep_panel(colors, cx).into_any_element(),
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
                // One row: the result tabs on the left, the actions on the
                // right. A page title above them ("Results") said nothing the
                // tabs do not; the row wraps the actions below when narrow.
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(340.0))
                            .when(has_results, |row| row.child(self.result_tabs(colors, cx))),
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
                                    console.show_toast("Copied the summary as Markdown", cx);
                                }
                            }),
                        ))
                    })
                    // A sample offers neither; a run reopened from History offers
                    // both, since the banner says Duplicate branches from it.
                    // The run's verified bundle, when it is on disk: a live
                    // result, or a reopened run whose bundle was kept.
                    .when(
                        has_results
                            && self
                                .intervention
                                .as_ref()
                                .is_some_and(|output| self.bundle_dir_on_disk(&output.bundle_dir)),
                        |header| {
                            let dir = self
                                .intervention
                                .as_ref()
                                .map(|output| output.bundle_dir.clone())
                                .unwrap_or_default();
                            header.child(text_button(
                                "review-bundle",
                                "Reveal bundle",
                                cx.listener(move |console, _: &ClickEvent, _window, cx| {
                                    if let Err(error) = reveal_in_finder(&dir) {
                                        console.error =
                                            Some(format!("Could not reveal the bundle: {error}"));
                                        cx.notify();
                                    }
                                }),
                            ))
                        },
                    )
                    .when(has_results && (!self.sample || self.saved_run.is_some()), |header| {
                        header
                            .child(text_button(
                                "review-pin",
                                if self.reference.is_some() { "Re-pin reference" } else { "Pin as reference" },
                                cx.listener(|console, _: &ClickEvent, _window, cx| {
                                    console.pin_reference(cx);
                                    console.show_toast("Pinned as the reference for the next run", cx);
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
            .children(self.sweep_progress(colors, cx))
            .when(self.busy(), |page| page.child(self.run_progress(colors)))
            .when(has_results && !self.busy() && self.results_stale(), |page| {
                page.child(self.stale_notice(colors))
            })
            .when(has_results && self.sample && self.result_view != ResultView::Sweep, |page| {
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
                            match (self.saved_run, &self.opened_note) {
                                (Some(number), _) => format!("Saved run #{number}"),
                                (None, Some((title, _))) => title.clone(),
                                (None, None) => "Sample result".to_string(),
                            },
                            Type::LABEL,
                            colors.accent,
                        ))
                        .child(label(
                            match (self.saved_run, &self.opened_note) {
                                (Some(_), _) => "Reopened from your history. The setup holds its settings: change one and run, or run it again to check it still reproduces.".to_string(),
                                (None, Some((_, text))) => text.clone(),
                                (None, None) => "Illustrative data, so you can see what a finished comparison looks like. Run your own experiment and it replaces this.".to_string(),
                            },
                            Type::LABEL,
                            colors.text,
                        )),
                )
            })
            .child(result_body)
            .when(
                has_results
                    && self.result_view != ResultView::Trace
                    && self.result_view != ResultView::Sweep,
                |page| page.child(self.verification_panel(colors)),
            )
    }

    pub(super) fn output_panel(
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
                // The inline diff: this side's text from the point where the
                // two outputs part.
                let other = if title == "Baseline" {
                    self.intervention.as_ref()
                } else {
                    self.baseline.as_ref()
                };
                let style = HighlightStyle {
                    background_color: Some(
                        if title == "Intervention" {
                            colors.accent_soft
                        } else {
                            colors.neutral_soft
                        }
                        .into(),
                    ),
                    ..Default::default()
                };
                let (display_text, mark) = match self.comparison.as_ref().zip(other) {
                    Some((_, other)) => {
                        let from = divergence_offset(&out.text, &other.text);
                        let (text, range) = marked_display(&out.text, from);
                        (text, range.map(|range| (range, style)))
                    }
                    None => (isolate_bidi(&out.text), None),
                };
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
                            .child(multiline_marked(
                                &display_text,
                                Type::OUTPUT,
                                colors.text,
                                FONT_ARABIC_NAME,
                                mark,
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
            // While the run is in flight the pane shows the shape of the text
            // to come rather than a sentence about it.
            None if self.busy() => div()
                .pt(px(Space::XS))
                .child(super::skeleton::skeleton_text(
                    &format!("output-skeleton-{title}"),
                    3,
                    14.0,
                    colors.selection.into(),
                )),
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

    pub(super) fn verification_panel(&self, colors: &Colors) -> Div {
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
pub(super) fn shortcut_rows() -> Vec<(String, &'static str)> {
    let cmd = if cfg!(target_os = "macos") {
        "Cmd"
    } else {
        "Ctrl"
    };
    vec![
        (
            format!("{cmd}+K  or  {cmd}+Shift+P"),
            "Command palette (fuzzy: \"tsb\" finds Toggle sidebar)",
        ),
        (format!("{cmd}+P"), "Go to a saved run; type > for commands"),
        (format!("{cmd}+Enter"), "Continue, or run the experiment"),
        (format!("{cmd}+N"), "New experiment"),
        (format!("{cmd}+R"), "Rerun the last run"),
        (
            format!("{cmd}+1 - 4 on Review"),
            "Overview / Layers / Tokens / Raw trace",
        ),
        (
            "Ctrl+Tab  /  Ctrl+Shift+Tab".to_string(),
            "Next / previous result tab",
        ),
        (format!("{cmd}+B"), "Show or hide the sidebar"),
        (format!("{cmd}+,"), "Settings"),
        ("Esc".to_string(), "Cancel the run, or close the palette"),
    ]
}
