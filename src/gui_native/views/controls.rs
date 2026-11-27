use crate::gui_native::*;

// -- experiment steps -----------------------------------------------------
impl Console {
    /// Starting points for a new experiment.
    ///
    /// These were in the left rail, which made them look like a mode switch.
    /// They are choices for starting work, so they belong on the Prompt step
    /// where the work starts.
    /// The examples on offer, in the order they are shown.
    pub(in crate::gui_native) fn example_entries() -> Vec<(Preset, &'static str, &'static str)> {
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

    pub(in crate::gui_native) fn presets_block(
        &self,
        colors: &Colors,
        cx: &mut Context<Self>,
    ) -> Div {
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

    pub(in crate::gui_native) fn preset_card(
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
    pub(in crate::gui_native) fn generation_control(
        &self,
        colors: &Colors,
        cx: &mut Context<Self>,
    ) -> Div {
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

    pub(super) fn feedback_banners(&self, colors: &Colors) -> Div {
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
        // A cancelled run is a choice, not a failure: it is said plainly and
        // in neutral colours, with what it did and did not leave behind.
        let cancelled = (self.cancelled && !self.busy()).then(|| {
            div()
                .id("run-cancelled")
                .test_support()
                .w_full()
                .flex()
                .flex_col()
                .gap(px(Space::XS))
                .px(px(Space::MD))
                .py(px(Space::SM))
                .rounded(px(Radius::MD))
                .border_l_2()
                .border_color(colors.border_strong)
                .bg(colors.surface)
                .child(label("Run cancelled", Type::LABEL, colors.text))
                .child(label(
                    "Nothing was recorded and no bundle was kept. The model is still loaded, so the next run starts straight away.",
                    Type::LABEL,
                    colors.text_muted,
                ))
        });
        div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .children(error)
            .children(cancelled)
    }

    pub(in crate::gui_native) fn layer_stepper(
        &self,
        colors: &Colors,
        cx: &mut Context<Self>,
    ) -> Div {
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
                    .child(
                        Button::new("layer-plus")
                            .label("+")
                            .accessibility_label("Increase layer")
                            .on_click(cx.listener(|console, _: &ClickEvent, _, cx| {
                                console.adjust_layer(1, cx);
                            })),
                    )
                    .child(label(limit, Type::LABEL, colors.text_muted)),
            )
            // Before a model is loaded there is no range to show; once it is,
            // the track below says where the layer sits and this line goes.
            .when(n_layers.is_none(), |stepper| {
                stepper.child(label(position, Type::META, colors.text_faint))
            })
    }

    /// What a run is doing right now, as steps. A first run loads the model
    /// before it computes anything, and a silent wait there looks like a hang.
    pub(super) fn run_progress(&self, colors: &Colors) -> Div {
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
            Status::Cancelling => vec![("Stop at the next decode step", 1)],
            Status::Idle => Vec::new(),
        };
        let note = match self.status {
            Status::Preparing => {
                "The first run loads the model, so it takes longer. Later runs start straight away."
            }
            Status::Running => {
                "Ember is running your prompt twice, once untouched and once with your change. Esc cancels."
            }
            Status::Cancelling => {
                "Nothing from this run will be recorded, and the model stays loaded."
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
