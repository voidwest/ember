//! The experiment workspace: setup on the left, results on the right.
//!
//! The first version of this page was a three-step wizard (Prompt, Intervention,
//! Review). That reads well the first time and badly the tenth: the work is
//! "change one thing, run, compare, tweak, run again", and a wizard puts the
//! change and its result on different pages. Here the setup is always visible
//! and editable beside the result it produced, a result says when the settings
//! have moved on since it was made, and one result can be pinned as a
//! reference so the next run is judged against something.

use super::components::*;
use super::form::FormValues;
use super::theme::{Radius, Space, Type};
use super::{
    model_display_name, operation_explainer, operation_label, per_layer, site_contract_name,
    site_label, token_label, Colors, ComboId, Console, ExperimentComparison, Status,
};
use crate::gui::LayerMetric;
use gpui_kit::component::{
    button::{Button, ButtonVariants},
    Disableable,
};
use gpui_kit::prelude::*;
use gpui_kit::*;
use std::sync::Arc;

/// Width of the setup pane.
pub(super) const SETUP_WIDTH: f32 = 372.0;

/// A result kept to judge later runs against. Only what the comparison shows:
/// the per-layer series, the landmarks and the intervention's words.
#[derive(Clone)]
pub(super) struct Reference {
    pub label: String,
    pub layers: Arc<[LayerMetric]>,
    pub first_layer: Option<usize>,
    pub peak: Option<(f64, usize)>,
    pub text_equal: bool,
    pub intervention_text: String,
}

/// One line saying what a run changed, for labels and comparisons.
pub(super) fn change_summary(values: &FormValues) -> String {
    let mut out = operation_label(&values.op).to_string();
    match values.op.as_str() {
        "scale" | "interpolate" => out.push_str(&format!(" \u{d7}{}", values.value)),
        _ => {}
    }
    if per_layer(&values.site) {
        out.push_str(&format!(" \u{00b7} layer {}", values.layer));
    }
    out.push_str(&format!(" \u{00b7} {}", site_label(&values.site)));
    out
}

impl Console {
    /// Whether the settings on screen differ from the ones that produced the
    /// result on screen. A sample or a reopened run is never "stale": it is not
    /// a live result of these settings.
    pub(super) fn results_stale(&self) -> bool {
        !self.sample
            && self.baseline.is_some()
            && self
                .result_context
                .as_ref()
                .is_some_and(|context| *context != self.form_values())
    }

    /// Pin the result on screen as the reference for later runs.
    pub(super) fn pin_reference(&mut self, cx: &mut Context<Self>) {
        let (Some(comparison), Some(intervention), Some(context)) = (
            self.comparison.as_ref(),
            self.intervention.as_ref(),
            self.result_context.as_ref(),
        ) else {
            return;
        };
        self.reference = Some(Reference {
            label: change_summary(context),
            layers: self.layer_series.clone(),
            first_layer: comparison.landmarks.first_layer_divergence,
            peak: comparison
                .landmarks
                .peak_relative_l2
                .zip(comparison.landmarks.peak_layer),
            text_equal: comparison.generated_text_equal,
            intervention_text: intervention.text.clone(),
        });
        cx.notify();
    }

    /// Pin a saved run as the reference and return to the workspace, so the
    /// next run is judged against it. Needs a run that kept its result.
    pub(super) fn compare_with_run(&mut self, number: u64, cx: &mut Context<Self>) {
        let Some(record) = self
            .store
            .runs
            .iter()
            .find(|run| run.number == number)
            .cloned()
        else {
            return;
        };
        let (Some(result), Some(config)) = (record.result, record.config) else {
            return;
        };
        let values = FormValues {
            model_path: config.model_path,
            prompt: record.prompt,
            max_tokens: config.max_tokens,
            execution: config.execution,
            site: config.site,
            layer: config.layer,
            op: config.op,
            value: config.value,
            source: config.source,
            source_layer: config.source_layer,
            token: config.token,
            span: config.span,
        };
        self.reference = Some(Reference {
            label: format!("Run #{number} \u{00b7} {}", change_summary(&values)),
            layers: result
                .layers
                .iter()
                .map(|layer| LayerMetric {
                    layer: layer.layer,
                    relative_l2_difference: layer.relative_l2,
                    cosine_distance: layer.cosine,
                    maximum_absolute_difference: None,
                    exact: layer.relative_l2 == Some(0.0),
                })
                .collect(),
            first_layer: result.first_layer_divergence,
            peak: result.peak_relative_l2.zip(result.peak_layer),
            text_equal: record.outputs_equal,
            intervention_text: result.intervention_text,
        });
        self.goto(super::View::Experiment, cx);
        cx.notify();
    }

    /// Make sure the console's shortcuts reach it after a mouse-started
    /// run. The key handler lives on the window's root and only hears keys
    /// while something inside it has focus; a click on a button leaves focus
    /// nowhere, and Esc would then have nothing to cancel.
    pub(super) fn claim_keyboard(&self, window: &mut Window, cx: &mut Context<Self>) {
        if window.focused(cx).is_none() {
            self.focus_handle.focus(window, cx);
        }
    }

    pub(super) fn clear_reference(&mut self, cx: &mut Context<Self>) {
        self.reference = None;
        cx.notify();
    }

    /// The setup pane: everything that defines a run, in the order you decide
    /// it, with the run button under it. Nothing here is a step; every control
    /// is live at all times.
    pub(super) fn setup_pane(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let needs_source = matches!(self.op.as_str(), "replace" | "interpolate" | "add-delta");
        let needs_value = matches!(self.op.as_str(), "scale" | "interpolate");
        let model_ready = self.session.is_some();

        let heading = |text: &'static str| label(text, Type::META, colors.text_faint);

        // Once a model is loaded the section is a single line -- the model is
        // not what you edit from run to run -- with a way back to the picker.
        let model_summary = div()
            .flex()
            .flex_col()
            .gap(px(Space::XS))
            .child(heading("Model"))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(Space::SM))
                    .child(div().flex_1().min_w(px(0.0)).overflow_hidden().child(label(
                        model_display_name(&self.model_path),
                        Type::BODY,
                        colors.text,
                    )))
                    .child(chip("Ready", colors.ok))
                    .child(text_button(
                        "setup-model-change",
                        "Change",
                        cx.listener(|console, _: &ClickEvent, _window, cx| {
                            console.model_open = true;
                            cx.notify();
                        }),
                    )),
            )
            .children(self.session.as_ref().map(|info| {
                label(
                    format!("{} \u{00b7} {} layers", info.architecture, info.n_layers),
                    Type::META,
                    colors.text_faint,
                )
            }));
        let collapse_model = model_ready && !self.model_open;

        let model = div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .child(heading("Model"))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(Space::SM))
                    .child(div().flex_1().min_w(px(0.0)).child(self.picker(
                        colors,
                        "model-picker",
                        ComboId::Model,
                        &self.model_path,
                        &self.model_options,
                        cx,
                    )))
                    .child(chip(
                        if model_ready { "Ready" } else { "Not loaded" },
                        if model_ready {
                            colors.ok
                        } else {
                            colors.text_faint
                        },
                    )),
            )
            .child(label(
                match &self.session {
                    Some(info) => {
                        format!("{} \u{00b7} {} layers", info.architecture, info.n_layers)
                    }
                    None => "Loads automatically on the first run.".to_string(),
                },
                Type::META,
                colors.text_faint,
            ))
            .when(
                self.advanced_open || self.model_options.is_empty(),
                |section| {
                    section.child(text_input(
                        colors,
                        self.inputs.model.clone(),
                        super::FONT_MONO_NAME,
                        Type::META,
                        None,
                        cx,
                    ))
                },
            );

        let prompt = div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .child(heading("Prompt"))
            .child(text_input(
                colors,
                self.inputs.prompt.clone(),
                super::FONT_ARABIC_NAME,
                Type::BODY,
                Some(104.0),
                cx,
            ));

        let change = div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .child(heading("Change"))
            .child(self.picker(
                colors,
                "op-picker",
                ComboId::Op,
                &self.op,
                &["replace", "zero", "scale", "interpolate", "add-delta"].map(str::to_string),
                cx,
            ))
            .child(label(
                operation_explainer(&self.op),
                Type::META,
                colors.text_muted,
            ));

        let where_ = div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .child(heading("Where"))
            .child(self.picker(
                colors,
                "site-picker",
                ComboId::Site,
                &self.site,
                &self.site_options,
                cx,
            ))
            .child(mono(
                format!("ember.hook.v1 \u{00b7} {}", site_contract_name(&self.site)),
                Type::MICRO,
                colors.text_faint,
            ))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_start()
                    .gap(px(Space::MD))
                    .when(per_layer(&self.site), |row| {
                        row.child(div().flex_1().min_w(px(0.0)).child(field(
                            colors,
                            "Layer",
                            self.layer_stepper(colors, cx),
                        )))
                    })
                    .when(needs_value, |row| {
                        row.child(div().w(px(104.0)).flex_none().child(field(
                            colors,
                            if self.op == "interpolate" {
                                "Blend 0\u{2013}1"
                            } else {
                                "Strength"
                            },
                            text_input(
                                colors,
                                self.inputs.value.clone(),
                                super::FONT_MONO_NAME,
                                Type::META,
                                None,
                                cx,
                            ),
                        )))
                    }),
            )
            .when(needs_source, |section| {
                section
                    .child(field(
                        colors,
                        "Source",
                        self.picker(
                            colors,
                            "source-picker",
                            ComboId::Source,
                            &self.source,
                            &self.source_options,
                            cx,
                        ),
                    ))
                    .when(self.source == "capture", |section| {
                        section.child(field(
                            colors,
                            "Source layer",
                            text_input(
                                colors,
                                self.inputs.source_layer.clone(),
                                super::FONT_MONO_NAME,
                                Type::META,
                                None,
                                cx,
                            ),
                        ))
                    })
            });

        let target = div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .child(heading("Target tokens"))
            .child(self.picker(
                colors,
                "token-picker",
                ComboId::Token,
                &self.token,
                &self.token_options,
                cx,
            ))
            .when(self.token == "matched-span", |section| {
                section.child(field(
                    colors,
                    "Phrase to target",
                    text_input(
                        colors,
                        self.inputs.span.clone(),
                        super::FONT_ARABIC_NAME,
                        Type::META,
                        None,
                        cx,
                    ),
                ))
            });

        let advanced = div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .child(heading("Advanced"))
                    .child(div().flex_1())
                    .child(text_button(
                        "setup-advanced",
                        if self.advanced_open { "Hide" } else { "Show" },
                        cx.listener(|console, _: &ClickEvent, _window, cx| {
                            console.advanced_open = !console.advanced_open;
                            cx.notify();
                        }),
                    )),
            )
            .when(self.advanced_open, |section| {
                section
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
                            super::FONT_MONO_NAME,
                            Type::META,
                            None,
                            cx,
                        ),
                    ))
                    .child(mono(
                        format!(
                            "hook {} \u{00b7} operation {} \u{00b7} tokens {}",
                            self.site, self.op, self.token
                        ),
                        Type::MICRO,
                        colors.text_faint,
                    ))
            });

        let length = div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .child(heading("Generation length"))
            .child(self.generation_control(colors, cx));

        let run_label = match self.status {
            Status::Preparing => "Loading model\u{2026}",
            Status::Running => "Running\u{2026}",
            Status::Restoring => "Verifying restore\u{2026}",
            Status::Cancelling => "Cancelling\u{2026}",
            Status::Idle if self.baseline.is_some() && !self.sample => "Run again",
            Status::Idle => "Run experiment",
        };
        let can_run = self.action_enabled();
        let footer = div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .px(px(Space::LG))
            .py(px(Space::MD))
            .border_t_1()
            .border_color(colors.border)
            .bg(colors.canvas)
            .children(
                self.validation_error()
                    .map(|error| label(error, Type::META, colors.warn)),
            )
            .child(
                Button::new("setup-run")
                    .primary()
                    .w_full()
                    .label(run_label)
                    .disabled(!can_run)
                    .accessibility_label(run_label)
                    .on_click(cx.listener(|console, _: &ClickEvent, window, cx| {
                        console.claim_keyboard(window, cx);
                        console.run_now();
                        cx.notify();
                    })),
            )
            // Stopping a run sits under the button that started it, for as
            // long as there is something to stop.
            .when(
                self.can_cancel() || self.status == Status::Cancelling,
                |footer| {
                    footer.child(
                        Button::new("setup-cancel")
                            .w_full()
                            .label(if self.status == Status::Cancelling {
                                "Cancelling\u{2026}"
                            } else {
                                "Cancel run"
                            })
                            .tooltip("Stop the run in flight (Esc). Nothing is recorded.")
                            .accessibility_label("Cancel the run in flight")
                            .disabled(self.status == Status::Cancelling)
                            .on_click(cx.listener(|console, _: &ClickEvent, _window, cx| {
                                console.cancel_run(cx);
                            })),
                    )
                },
            )
            .children(self.can_sweep().then(|| {
                if self.sweep_running() {
                    Button::new("setup-sweep-stop")
                        .w_full()
                        .label("Stop sweep")
                        .accessibility_label("Stop the sweep after the run in flight")
                        .on_click(cx.listener(|console, _: &ClickEvent, _window, cx| {
                            console.stop_sweep(cx);
                        }))
                } else {
                    let layers = self.session.as_ref().map(|session| session.n_layers);
                    Button::new("setup-sweep")
                        .w_full()
                        .label(match layers {
                            Some(count) => format!("Sweep all {count} layers"),
                            None => "Sweep all layers".to_string(),
                        })
                        .tooltip("Run this experiment at every layer and plot the effect of each")
                        .accessibility_label("Sweep the change across every layer")
                        .disabled(!can_run)
                        .on_click(cx.listener(|console, _: &ClickEvent, window, cx| {
                            console.claim_keyboard(window, cx);
                            console.start_sweep(cx);
                            cx.notify();
                        }))
                }
            }));

        let rule = || rule_h(colors);
        div()
            .flex()
            .flex_col()
            .w(px(SETUP_WIDTH))
            .flex_none()
            .min_h(px(0.0))
            .h_full()
            .bg(colors.surface)
            .border_r_1()
            .border_color(colors.border)
            .child(
                div()
                    .id("setup-scroll")
                    .test_support()
                    .flex_1()
                    .min_h(px(0.0))
                    .overflow_y_scroll()
                    .px(px(Space::LG))
                    .py(px(Space::LG))
                    .flex()
                    .flex_col()
                    .gap(px(Space::LG))
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .child(label("Setup", Type::SUBSECTION, colors.text))
                            .child(div().flex_1())
                            .child(text_button(
                                "setup-examples",
                                "Examples",
                                cx.listener(|console, _: &ClickEvent, _window, cx| {
                                    console.examples_open = !console.examples_open;
                                    cx.notify();
                                }),
                            )),
                    )
                    .when(self.examples_open, |pane| {
                        pane.child(self.examples_list(colors, cx))
                    })
                    .child(if collapse_model { model_summary } else { model })
                    .child(rule())
                    .child(prompt)
                    .child(length)
                    .child(rule())
                    .child(change)
                    .child(where_)
                    .child(target)
                    .child(rule())
                    .child(advanced),
            )
            .child(footer)
    }

    /// Examples as a single compact column, for the setup pane.
    fn examples_list(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let mut list = div().flex().flex_col().gap(px(Space::XS));
        for (preset, title, hint) in Self::example_entries() {
            list = list.child(self.preset_card(colors, preset, title, hint, cx));
        }
        list
    }

    /// The empty state of the results pane: what this page is for, and a way
    /// in that does not need a model.
    pub(super) fn results_empty(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        div()
            .flex()
            .flex_col()
            .gap(px(Space::LG))
            .child(label(
                "Nothing has run yet",
                Type::SECTION,
                colors.text,
            ))
            .child(label(
                "Set up an experiment on the left and press Run. The baseline and your change run on the same prompt, and the difference lands here. Change a setting and run again to compare.",
                Type::BODY,
                colors.text_muted,
            ))
            .child(label(
                "Start from an example",
                Type::LABEL,
                colors.text_faint,
            ))
            .child(self.presets_block(colors, cx))
            .child(
                div().flex().flex_row().child(text_button(
                    "empty-sample",
                    "or read a sample result",
                    cx.listener(|console, _: &ClickEvent, _window, cx| {
                        console.show_sample(cx);
                    }),
                )),
            )
    }

    /// Shown above a result whose settings have since been edited.
    pub(super) fn stale_notice(&self, colors: &Colors) -> Div {
        div()
            .w_full()
            .flex()
            .flex_row()
            .items_center()
            .gap(px(Space::SM))
            .px(px(Space::MD))
            .py(px(Space::SM))
            .rounded(px(Radius::MD))
            .border_l_2()
            .border_color(colors.warn)
            .bg(colors.surface)
            .child(label(
                "Settings changed since this result.",
                Type::LABEL,
                colors.text,
            ))
            .child(label(
                "It is still the result of the previous settings. Run again to update it.",
                Type::LABEL,
                colors.text_muted,
            ))
    }

    /// The pinned reference against the result on screen.
    pub(super) fn reference_panel(&self, colors: &Colors, cx: &mut Context<Self>) -> Option<Div> {
        let reference = self.reference.as_ref()?;
        let comparison: &ExperimentComparison = self.comparison.as_ref()?;
        let now_peak = comparison
            .landmarks
            .peak_relative_l2
            .zip(comparison.landmarks.peak_layer);
        let peak_text = |peak: Option<(f64, usize)>| {
            peak.map_or_else(
                || "none".to_string(),
                |(value, layer)| format!("{value:.3} @ L{layer}"),
            )
        };
        let layer_text = |layer: Option<usize>| {
            layer.map_or_else(|| "none".to_string(), |layer| format!("layer {layer}"))
        };
        let words = |equal: bool| if equal { "unchanged" } else { "changed" };
        let delta = match (reference.peak, now_peak) {
            (Some((before, _)), Some((after, _))) if before > 0.0 => {
                let change = (after - before) / before * 100.0;
                format!("{:+.0}% peak divergence", change)
            }
            _ => String::new(),
        };
        let cell = |text: String, color: Rgba| {
            div()
                .flex_1()
                .min_w(px(0.0))
                .child(label(text, Type::LABEL, color))
        };
        let head = |text: &'static str| {
            div()
                .flex_1()
                .min_w(px(0.0))
                .child(label(text, Type::META, colors.text_faint))
        };
        let row = |name: &'static str, before: String, after: String| {
            div()
                .flex()
                .flex_row()
                .gap(px(Space::MD))
                .child(div().w(px(150.0)).flex_none().child(label(
                    name,
                    Type::LABEL,
                    colors.text_muted,
                )))
                .child(cell(before, colors.text_muted))
                .child(cell(after, colors.text))
        };
        Some(
            div()
                .w_full()
                .flex()
                .flex_col()
                .gap(px(Space::SM))
                .px(px(Space::MD))
                .py(px(Space::MD))
                .rounded(px(Radius::MD))
                .bg(colors.surface)
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .child(label(
                            "Compared with pinned reference",
                            Type::LABEL,
                            colors.text,
                        ))
                        .child(div().flex_1())
                        .children(
                            (!delta.is_empty()).then(|| label(delta, Type::LABEL, colors.accent)),
                        )
                        .child(text_button(
                            "reference-clear",
                            "Clear",
                            cx.listener(|console, _: &ClickEvent, _window, cx| {
                                console.clear_reference(cx);
                            }),
                        )),
                )
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .gap(px(Space::MD))
                        .child(div().w(px(150.0)).flex_none())
                        .child(head("Pinned"))
                        .child(head("This run")),
                )
                .child(row(
                    "Change",
                    reference.label.clone(),
                    self.result_context
                        .as_ref()
                        .map(change_summary)
                        .unwrap_or_default(),
                ))
                .child(row(
                    "Text output",
                    words(reference.text_equal).to_string(),
                    words(comparison.generated_text_equal).to_string(),
                ))
                .child(row(
                    "First divergence",
                    layer_text(reference.first_layer),
                    layer_text(comparison.landmarks.first_layer_divergence),
                ))
                .child(row(
                    "Peak divergence",
                    peak_text(reference.peak),
                    peak_text(now_peak),
                ))
                .child(label(
                    format!(
                        "Pinned run wrote: {}",
                        super::truncate_chars(reference.intervention_text.trim(), 140)
                    ),
                    Type::META,
                    colors.text_faint,
                )),
        )
    }
}

/// Used by the model line in the setup pane and by tests.
#[allow(dead_code)]
pub(super) fn model_line(path: &str) -> String {
    model_display_name(path)
}

#[allow(dead_code)]
pub(super) fn token_line(token: &str) -> &'static str {
    token_label(token)
}
