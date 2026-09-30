//! Sweeps: run one experiment at every layer and see the whole curve.
//!
//! The question people actually ask is not "what happens at layer 8?" but
//! "where does it matter?". Answering it by hand is sixteen edits and sixteen
//! runs; with the model already loaded each run takes about a second, so the
//! console does it in turn. Points are kept in memory, not written to history:
//! sixteen near-identical rows would bury the runs the user chose to make.

use super::components::*;
use super::form::FormValues;
use super::theme::{Radius, Space, Type};
use super::workspace::change_summary;
use super::{per_layer, Colors, Console, ExperimentComparison, RunOutput, Status};
use crate::gui::LayerMetric;
use gpui_kit::prelude::*;
use gpui_kit::*;
use std::sync::Arc;

/// One finished run of a sweep, with everything needed to open it.
#[derive(Clone)]
pub(super) struct SweepPoint {
    pub layer: usize,
    pub baseline: RunOutput,
    pub intervention: RunOutput,
    pub comparison: ExperimentComparison,
    pub values: FormValues,
}

impl SweepPoint {
    fn peak(&self) -> f64 {
        self.comparison.landmarks.peak_relative_l2.unwrap_or(0.0)
    }
}

/// A sweep in progress or finished.
pub(super) struct Sweep {
    /// Layers still to run, in order, and how many there were at the start.
    pub queue: Vec<usize>,
    pub total: usize,
    pub points: Vec<SweepPoint>,
    /// The layer the form held before the sweep, restored at the end.
    pub layer_before: String,
    /// Stop after the run in flight.
    pub stop: bool,
    pub finished: bool,
}

impl Console {
    /// Whether a sweep is worth offering: the change must act on one layer.
    pub(super) fn can_sweep(&self) -> bool {
        per_layer(&self.site) && self.comparison_not_sample_blocking()
    }

    fn comparison_not_sample_blocking(&self) -> bool {
        true
    }

    pub(super) fn sweep_running(&self) -> bool {
        self.sweep.as_ref().is_some_and(|sweep| !sweep.finished)
    }

    /// Start a sweep over every layer of the loaded model, or -- when no model
    /// is loaded yet -- load it first and start when it is ready.
    pub(super) fn start_sweep(&mut self, cx: &mut Context<Self>) {
        if self.busy() || !per_layer(&self.site) {
            return;
        }
        if self.session.is_none() {
            self.pending_sweep = true;
            self.run_prepare_for_sweep();
            return;
        }
        let layers = self.session.as_ref().map_or(0, |session| session.n_layers);
        if layers == 0 {
            return;
        }
        self.pending_sweep = false;
        self.sweep = Some(Sweep {
            queue: (0..layers).collect(),
            total: layers,
            points: Vec::new(),
            layer_before: self.layer.clone(),
            stop: false,
            finished: false,
        });
        self.error = None;
        self.advance_sweep(cx);
    }

    fn run_prepare_for_sweep(&mut self) {
        self.status = Status::Preparing;
        self.error = None;
        let _ = self
            .worker_tx
            .send(super::WorkerMsg::Prepare(self.model_path.trim().to_string()));
    }

    /// Ask the running sweep to stop after the run in flight.
    pub(super) fn stop_sweep(&mut self, cx: &mut Context<Self>) {
        if let Some(sweep) = self.sweep.as_mut() {
            sweep.stop = true;
        }
        cx.notify();
    }

    /// Run the next layer that the form accepts, or finish.
    pub(super) fn advance_sweep(&mut self, cx: &mut Context<Self>) {
        loop {
            let Some(sweep) = self.sweep.as_mut() else {
                return;
            };
            if sweep.stop || sweep.queue.is_empty() {
                return self.finish_sweep(cx);
            }
            let layer = sweep.queue.remove(0);
            self.layer = layer.to_string();
            let input = self.inputs.layer.clone();
            self.set_input_value(input, layer.to_string(), cx);
            // A layer the form rejects (a source layer that is not earlier
            // than the target, say) is skipped rather than aborting the sweep.
            if self.validation_error().is_some() {
                if let Some(sweep) = self.sweep.as_mut() {
                    sweep.total = sweep.total.saturating_sub(1);
                }
                continue;
            }
            self.run();
            return;
        }
    }

    /// Record the run that just finished as a sweep point. Called from the
    /// run-completion handler while a sweep is active.
    pub(super) fn sweep_point_done(&mut self, cx: &mut Context<Self>) {
        let (Some(baseline), Some(intervention), Some(comparison), Some(values)) = (
            self.baseline.clone(),
            self.intervention.clone(),
            self.comparison.clone(),
            self.result_context.clone(),
        ) else {
            return;
        };
        let layer = values.layer.parse::<usize>().unwrap_or(0);
        if let Some(sweep) = self.sweep.as_mut() {
            sweep.points.push(SweepPoint {
                layer,
                baseline,
                intervention,
                comparison,
                values,
            });
        }
        self.advance_sweep(cx);
    }

    fn finish_sweep(&mut self, cx: &mut Context<Self>) {
        let Some(sweep) = self.sweep.as_mut() else {
            return;
        };
        sweep.finished = true;
        let layer_before = sweep.layer_before.clone();
        // Open the point that moved the model most: it is the one worth
        // reading first. With no points (stopped immediately) there is
        // nothing to open.
        let best = sweep
            .points
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.peak().total_cmp(&right.1.peak()))
            .map(|(index, _)| index);
        match best {
            Some(index) => {
                self.open_sweep_point(index, cx);
                self.result_view = super::ResultView::Sweep;
            }
            None => {
                self.layer = layer_before.clone();
                let input = self.inputs.layer.clone();
                self.set_input_value(input, layer_before, cx);
                self.sweep = None;
            }
        }
        cx.notify();
    }

    /// Put one sweep point on the page as a result of its own.
    pub(super) fn open_sweep_point(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(point) = self
            .sweep
            .as_ref()
            .and_then(|sweep| sweep.points.get(index))
            .cloned()
        else {
            return;
        };
        let layer = point.layer;
        self.show_result(
            point.baseline,
            point.intervention,
            point.comparison,
            point.values,
            None,
            cx,
        );
        self.opened_note = Some((
            format!("Sweep point \u{00b7} layer {layer}"),
            "One run from the sweep. Open the Sweep tab to go back to the curve; change any setting and Run to continue from here.".to_string(),
        ));
        self.result_view = super::ResultView::Overview;
    }

    /// The sweep as one metric series: the peak divergence each run reached,
    /// keyed by the layer that was changed.
    fn sweep_series(&self) -> Arc<[LayerMetric]> {
        let Some(sweep) = self.sweep.as_ref() else {
            return Arc::from([]);
        };
        let mut points: Vec<&SweepPoint> = sweep.points.iter().collect();
        points.sort_by_key(|point| point.layer);
        points
            .iter()
            .map(|point| LayerMetric {
                layer: point.layer,
                relative_l2_difference: Some(point.peak()),
                cosine_distance: None,
                maximum_absolute_difference: None,
                exact: point.peak() == 0.0,
            })
            .collect()
    }

    fn sweep_csv(&self) -> String {
        let mut out = String::from("changed_layer,peak_relative_l2,words_changed\n");
        if let Some(sweep) = self.sweep.as_ref() {
            let mut points: Vec<&SweepPoint> = sweep.points.iter().collect();
            points.sort_by_key(|point| point.layer);
            for point in points {
                out.push_str(&format!(
                    "{},{},{}\n",
                    point.layer,
                    point.peak(),
                    !point.comparison.generated_text_equal
                ));
            }
        }
        out
    }

    /// The Sweep tab: the curve, a sentence saying what it shows, and one row
    /// per layer that opens that run.
    pub(super) fn sweep_panel(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let Some(sweep) = self.sweep.as_ref() else {
            return div();
        };
        let mut points: Vec<&SweepPoint> = sweep.points.iter().collect();
        points.sort_by_key(|point| point.layer);
        let changed = points
            .iter()
            .filter(|point| !point.comparison.generated_text_equal)
            .count();
        let strongest = points
            .iter()
            .max_by(|left, right| left.peak().total_cmp(&right.peak()));
        let summary = match strongest {
            Some(best) => format!(
                "The words changed at {changed} of {} layers. The largest effect came from changing layer {} (peak divergence {:.3}).",
                points.len(),
                best.layer,
                best.peak()
            ),
            None => "No runs finished.".to_string(),
        };
        let sample_values = points.first().map(|point| &point.values);
        let csv = self.sweep_csv();
        let entity = cx.entity();
        let rows = points.iter().map(|point| {
            let index = sweep
                .points
                .iter()
                .position(|candidate| candidate.layer == point.layer)
                .unwrap_or(0);
            let words_changed = !point.comparison.generated_text_equal;
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(Space::MD))
                .child(div().w(px(84.0)).flex_none().child(label(
                    format!("Layer {}", point.layer),
                    Type::LABEL,
                    colors.text,
                )))
                .child(div().w(px(120.0)).flex_none().child(mono(
                    format!("peak {:.3}", point.peak()),
                    Type::META,
                    colors.text_muted,
                )))
                .child(div().flex_1().child(label(
                    if words_changed { "words changed" } else { "words unchanged" },
                    Type::LABEL,
                    if words_changed { colors.accent } else { colors.text_faint },
                )))
                .child(text_button(
                    SharedString::from(format!("sweep-open:{}", point.layer)),
                    "Open",
                    cx.listener(move |console, _: &ClickEvent, _window, cx| {
                        console.open_sweep_point(index, cx);
                    }),
                ))
        });
        let chart_series = self.sweep_series();
        div()
            .flex()
            .flex_col()
            .gap(px(Space::MD))
            .child(
                div()
                    .w_full()
                    .flex()
                    .flex_col()
                    .gap(px(Space::SM))
                    .px(px(Space::LG))
                    .py(px(Space::MD))
                    .rounded(px(Radius::LG))
                    .bg(colors.surface)
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .child(label(
                                "Sweep across layers",
                                Type::SUBSECTION,
                                colors.text,
                            ))
                            .child(div().flex_1())
                            .child(text_button(
                                "sweep-csv",
                                "Copy CSV",
                                cx.listener(move |_console, _: &ClickEvent, _window, cx| {
                                    cx.write_to_clipboard(ClipboardItem::new_string(csv.clone()));
                                }),
                            )),
                    )
                    .children(sample_values.map(|values| {
                        label(
                            format!(
                                "{} \u{00b7} run once per layer",
                                change_summary(values).split(" \u{00b7} layer").next().unwrap_or("")
                            ),
                            Type::META,
                            colors.text_faint,
                        )
                    }))
                    .child(label(summary, Type::BODY, colors.text_muted))
                    .child(super::chart::layer_divergence_chart(
                        entity,
                        chart_series,
                        None,
                        None,
                        self.selected_layer,
                        self.hovered_layer,
                        260.0,
                        colors,
                    )),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::SM))
                    .px(px(Space::LG))
                    .py(px(Space::MD))
                    .rounded(px(Radius::LG))
                    .bg(colors.surface)
                    .child(label(
                        "Each row is one run. Open shows its full comparison.",
                        Type::META,
                        colors.text_faint,
                    ))
                    .children(rows),
            )
    }

    /// Progress line for a running sweep, shown above the run steps.
    pub(super) fn sweep_progress(&self, colors: &Colors, cx: &mut Context<Self>) -> Option<Div> {
        let sweep = self.sweep.as_ref().filter(|sweep| !sweep.finished)?;
        let done = sweep.points.len();
        Some(
            div()
                .w_full()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(Space::MD))
                .child(label(
                    format!(
                        "Sweep: {} of {} layers{}",
                        (done + 1).min(sweep.total.max(1)),
                        sweep.total,
                        if sweep.stop { " \u{00b7} stopping after this run" } else { "" }
                    ),
                    Type::LABEL,
                    colors.text,
                ))
                .child(div().flex_1())
                .child(text_button(
                    "sweep-stop",
                    "Stop",
                    cx.listener(|console, _: &ClickEvent, _window, cx| {
                        console.stop_sweep(cx);
                    }),
                )),
        )
    }
}
