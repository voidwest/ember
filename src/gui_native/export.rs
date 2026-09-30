//! Exporting runs: Markdown for a notebook or a message, and the verified
//! bundle for anyone who wants to check the numbers with the CLI.
//!
//! A run's Markdown is the same document the Review page's Copy summary
//! produces, so a history row and the result on screen export identically.
//! Its bundle is the pair of directories the run wrote. They belong to the
//! user and may have been moved or deleted since; when they are gone and the
//! record kept its configuration, the run can be re-run through the same
//! bundle writer (`execute_prepared` -> `write_bundle`, which self-verifies),
//! and the record is pointed at the new bundles. `ember experiment verify`
//! accepts either.

use super::components::*;
use super::form::FormValues;
use super::history::record_view;
use super::theme::{Radius, Space, Type};
use super::{
    model_display_name, operation_label, parse_run_request, site_label, token_label, Colors,
    Console, ExperimentComparison, RunOutput, Status,
};
use crate::gui::LayerMetric;
use ember::app_store::{RecordBundles, RunRecord};
use gpui_kit::component::{button::Button, Disableable, Sizable};
use gpui_kit::prelude::*;
use gpui_kit::*;

/// The comparison as Markdown. `note` is a quoted line under the title (a
/// reopened run, the sample); the bundle section appears when the
/// intervention's bundle directory is on disk.
pub(super) fn experiment_markdown(
    note: Option<&str>,
    baseline: &RunOutput,
    intervention: &RunOutput,
    comparison: &ExperimentComparison,
    context: &FormValues,
    layer_series: &[LayerMetric],
) -> String {
    let mut out = String::new();
    out.push_str("# Ember experiment\n\n");
    if let Some(note) = note {
        out.push_str(&format!("> {note}\n\n"));
    }
    out.push_str(&format!(
        "- **Model:** {}\n",
        model_display_name(&context.model_path)
    ));
    out.push_str(&format!("- **Prompt:** {}\n", context.prompt.trim()));
    out.push_str(&format!(
        "- **Change:** {} at layer {} ({}), affecting {}\n",
        operation_label(&context.op),
        context.layer,
        site_label(&context.site),
        token_label(&context.token),
    ));
    out.push_str(&format!(
        "- **Generation:** up to {} tokens, seed 0\n\n",
        context.max_tokens
    ));
    out.push_str("## Result\n\n");
    out.push_str(&format!(
        "- **Text output:** {}\n",
        if comparison.generated_text_equal {
            "unchanged"
        } else {
            "changed"
        }
    ));
    if let Some(layer) = comparison.landmarks.first_layer_divergence {
        out.push_str(&format!("- **First internal divergence:** layer {layer}\n"));
    }
    if let (Some(value), Some(layer)) = (
        comparison.landmarks.peak_relative_l2,
        comparison.landmarks.peak_layer,
    ) {
        out.push_str(&format!(
            "- **Peak divergence:** {value:.3} (relative L2) at layer {layer}\n"
        ));
    }
    out.push_str(&format!("\n**Baseline:** {}\n\n", baseline.text.trim()));
    out.push_str(&format!("**Intervention:** {}\n", intervention.text.trim()));
    if !layer_series.is_empty() {
        out.push_str("\n## Divergence by layer\n\n| layer | relative L2 | cosine distance |\n|---|---|---|\n");
        for metric in layer_series {
            let cell = |value: Option<f64>| {
                value.map_or_else(|| "n/a".to_string(), |value| format!("{value:.4}"))
            };
            out.push_str(&format!(
                "| {} | {} | {} |\n",
                metric.layer,
                cell(metric.relative_l2_difference),
                cell(metric.cosine_distance)
            ));
        }
    }
    if std::path::Path::new(&intervention.bundle_dir).is_dir() {
        out.push_str(&bundle_markdown(&RecordBundles {
            baseline: baseline.bundle_dir.clone(),
            intervention: intervention.bundle_dir.clone(),
        }));
    }
    out
}

fn bundle_markdown(bundles: &RecordBundles) -> String {
    format!(
        "\n## Bundles\n\n- Baseline: `{}`\n- Intervention: `{}`\n\nCheck them with:\n\n```sh\n{}\n```\n",
        bundles.baseline,
        bundles.intervention,
        verify_command(bundles)
    )
}

/// Quote a path for a POSIX shell.
fn shell_quote(path: &str) -> String {
    format!("'{}'", path.replace('\'', "'\\''"))
}

/// The CLI command that verifies both of a run's bundles.
pub(super) fn verify_command(bundles: &RecordBundles) -> String {
    format!(
        "ember experiment verify {} && ember experiment verify {}",
        shell_quote(&bundles.baseline),
        shell_quote(&bundles.intervention)
    )
}

/// Markdown for any history record. A record that kept its result exports
/// the full comparison; an older one exports what it recorded.
pub(super) fn record_markdown(record: &RunRecord) -> String {
    let note = format!("Run #{}, from history.", record.number);
    let mut out = match record_view(record) {
        Some((baseline, intervention, comparison, values)) => experiment_markdown(
            Some(&note),
            &baseline,
            &intervention,
            &comparison,
            &values,
            &comparison.layers,
        ),
        None => {
            let mut out = format!("# Ember experiment\n\n> {note}\n\n");
            out.push_str(&format!("- **Model:** {}\n", record.model));
            out.push_str(&format!("- **Prompt:** {}\n", record.prompt.trim()));
            out.push_str(&format!(
                "- **Change:** {}{} ({})\n\n",
                record.intervention,
                record
                    .layer
                    .map_or_else(String::new, |layer| format!(" at layer {layer}")),
                record.hook
            ));
            out.push_str("## Result\n\n");
            out.push_str(&format!(
                "- **Text output:** {}\n",
                if record.outputs_equal {
                    "unchanged"
                } else {
                    "changed"
                }
            ));
            if let (Some(baseline), Some(intervention)) =
                (record.baseline_tokens, record.intervention_tokens)
            {
                out.push_str(&format!(
                    "- **Tokens:** {baseline} baseline, {intervention} intervention\n"
                ));
            }
            if let Some(step) = record.diverged_at_step {
                out.push_str(&format!("- **Words first differ:** step {step}\n"));
            }
            out.push_str(&format!(
                "- **Verified:** {}\n",
                if record.verified { "yes" } else { "no" }
            ));
            if let Some(result) = &record.result {
                out.push_str(&format!(
                    "\n**Baseline:** {}\n\n**Intervention:** {}\n",
                    result.baseline_text.trim(),
                    result.intervention_text.trim()
                ));
            }
            out
        }
    };
    // `record_view` puts the bundle into the document only when both
    // directories exist; a record without a full view still names them.
    if !out.contains("## Bundles")
        && let Some(bundles) = record.bundles.as_ref().filter(|bundles| bundles.exist())
    {
        out.push_str(&bundle_markdown(bundles));
    }
    out
}

impl Console {
    fn export_record(&self, number: u64) -> Option<&RunRecord> {
        self.store.runs.iter().find(|run| run.number == number)
    }

    /// Open (or close) the export strip for one history row.
    pub(super) fn toggle_export(&mut self, number: u64, cx: &mut Context<Self>) {
        self.export_run = if self.export_run == Some(number) {
            None
        } else {
            Some(number)
        };
        self.export_note = None;
        cx.notify();
    }

    /// Re-run a stored configuration through the bundle writer, so a run
    /// whose bundles are gone has a verifiable bundle again. Nothing new is
    /// added to history: the record is pointed at the new bundles.
    pub(super) fn rerun_to_bundle(&mut self, number: u64, cx: &mut Context<Self>) {
        if self.busy() {
            return;
        }
        let Some(values) = self
            .export_record(number)
            .and_then(super::history::record_values)
        else {
            return;
        };
        let config = match values
            .build_run_request()
            .and_then(|request| parse_run_request(&request))
        {
            Ok(config) => config,
            Err(error) => {
                self.export_note = Some(format!("Run #{number} cannot be re-run: {error}"));
                cx.notify();
                return;
            }
        };
        // The worker loads the run's model if it is not the resident one;
        // the console's view of the session then no longer holds.
        if self
            .session
            .as_ref()
            .is_some_and(|session| session.model_path != config.model_path)
        {
            self.session = None;
        }
        self.rebundle = Some(number);
        self.export_note = None;
        self.cancelled = false;
        self.error = None;
        self.status = Status::Running;
        let token = ember::cancel::CancelToken::new();
        self.run_cancel = Some(token.clone());
        let _ = self.worker_tx.send(super::WorkerMsg::Run(config, token));
        cx.notify();
    }

    /// A re-run finished: point the record at its new bundles.
    pub(super) fn rebundle_done(&mut self, number: u64, bundle: &crate::gui::RunBundle) {
        let bundles = super::history::record_bundles(bundle);
        let stored = self
            .export_record(number)
            .and_then(|record| record.result.as_ref())
            .map(|result| result.intervention_text.clone());
        let reproduced = stored.as_deref() == Some(bundle.intervention.text.as_str());
        self.store.set_bundles(number, bundles);
        self.persist();
        self.export_note = Some(if !bundle.verification.ok {
            format!("Re-ran run #{number}, but its bundle failed verification.")
        } else if reproduced || stored.is_none() {
            format!("Re-ran run #{number} and wrote a verified bundle.")
        } else {
            format!(
                "Re-ran run #{number} and wrote a verified bundle, but the output differs from the stored one: the bundle records the re-run."
            )
        });
    }

    /// The export strip above the Runs table, for the row whose Export was
    /// clicked.
    pub(super) fn export_bar(&self, colors: &Colors, cx: &mut Context<Self>) -> Option<AnyElement> {
        let number = self.export_run?;
        let record = self.export_record(number)?;
        let rebundling = self.rebundle == Some(number);
        let markdown = record_markdown(record);
        let bundles = record.bundles.clone().filter(RecordBundles::exist);
        let can_rerun = record.config.is_some();

        let mut actions = div()
            .flex()
            .flex_row()
            .flex_wrap()
            .items_center()
            .gap(px(Space::SM))
            .child(
                Button::new("export-markdown")
                    .small()
                    .label("Copy as Markdown")
                    .tooltip("The run's summary, outputs and per-layer numbers as Markdown")
                    .on_click(move |_, _, cx| {
                        cx.write_to_clipboard(ClipboardItem::new_string(markdown.clone()));
                    }),
            );
        let bundle_line: String;
        match &bundles {
            Some(bundles) => {
                let reveal = bundles.intervention.clone();
                let command = verify_command(bundles);
                bundle_line = format!(
                    "Bundles on disk: {}",
                    super::truncate_path_start(&bundles.intervention, 72)
                );
                actions = actions
                    .child(
                        Button::new("export-reveal")
                            .small()
                            .label("Reveal bundle")
                            .tooltip("Show the run's bundle directory in the file manager")
                            .on_click(cx.listener(move |console, _, _, cx| {
                                if let Err(error) = super::views::reveal_in_finder(&reveal) {
                                    console.export_note =
                                        Some(format!("Could not reveal the bundle: {error}"));
                                    cx.notify();
                                }
                            })),
                    )
                    .child(
                        Button::new("export-verify")
                            .small()
                            .label("Copy verify command")
                            .tooltip("ember experiment verify, for both bundles")
                            .on_click(move |_, _, cx| {
                                cx.write_to_clipboard(ClipboardItem::new_string(command.clone()));
                            }),
                    );
            }
            None if rebundling => {
                bundle_line = format!(
                    "{} run #{number} through the bundle writer\u{2026}",
                    if self.status == Status::Cancelling {
                        "Cancelling the re-run of"
                    } else {
                        "Re-running"
                    }
                );
                actions = actions.child(
                    Button::new("export-rebundle-cancel")
                        .small()
                        .label("Cancel")
                        .disabled(self.status == Status::Cancelling)
                        .on_click(cx.listener(|console, _, _, cx| console.cancel_run(cx))),
                );
            }
            None if can_rerun => {
                bundle_line = if record.bundles.is_some() {
                    "This run's bundle is no longer on disk. Re-run it to write a new, verifiable one.".to_string()
                } else {
                    "No bundle was kept for this run. Re-run it to write a verifiable one."
                        .to_string()
                };
                actions = actions.child(
                    Button::new("export-rebundle")
                        .small()
                        .label("Re-run to bundle")
                        .tooltip("Run the stored configuration again through the CLI bundle writer")
                        .disabled(self.busy())
                        .on_click(cx.listener(move |console, _, _, cx| {
                            console.rerun_to_bundle(number, cx);
                        })),
                );
            }
            None => {
                bundle_line = "No bundle was kept for this run, and it kept no configuration, so it can't be re-run to write one.".to_string();
            }
        }
        Some(
            div()
                .id("export-bar")
                .test_support()
                .w_full()
                .flex()
                .flex_col()
                .gap(px(Space::SM))
                .px(px(Space::MD))
                .py(px(Space::SM))
                .rounded(px(Radius::MD))
                .border_l_2()
                .border_color(colors.accent)
                .bg(colors.surface)
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .child(div().flex_1().child(label(
                            format!("Export run #{number}"),
                            Type::LABEL,
                            colors.text,
                        )))
                        .child(text_button(
                            "export-close",
                            "Close",
                            cx.listener(|console, _: &ClickEvent, _window, cx| {
                                console.export_run = None;
                                console.export_note = None;
                                cx.notify();
                            }),
                        )),
                )
                .child(actions)
                .child(label(bundle_line, Type::META, colors.text_muted))
                .children(
                    self.export_note
                        .clone()
                        .map(|note| label(note, Type::META, colors.text)),
                )
                .into_any_element(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{record_markdown, verify_command};
    use ember::app_store::{RecordBundles, RunRecord};

    #[test]
    fn a_record_without_a_result_still_exports_what_it_recorded() {
        let record = RunRecord {
            number: 3,
            finished_at: 0,
            model: "Llama".into(),
            intervention: "Zero".into(),
            hook: "After MLP block".into(),
            layer: Some(4),
            duration_ms: None,
            baseline_tokens: Some(48),
            intervention_tokens: Some(12),
            diverged_at_step: Some(1),
            outputs_equal: false,
            verified: true,
            pinned: false,
            prompt: "The capital of France is".into(),
            config: None,
            result: None,
            bundles: None,
        };
        let markdown = record_markdown(&record);
        assert!(markdown.starts_with("# Ember experiment"));
        assert!(markdown.contains("Run #3, from history"));
        assert!(markdown.contains("Zero at layer 4 (After MLP block)"));
        assert!(markdown.contains("48 baseline, 12 intervention"));
        assert!(!markdown.contains("## Bundles"), "no bundle to name");
    }

    #[test]
    fn the_verify_command_quotes_both_paths() {
        let command = verify_command(&RecordBundles {
            baseline: "/tmp/a b/base".into(),
            intervention: "/tmp/it's".into(),
        });
        assert_eq!(
            command,
            "ember experiment verify '/tmp/a b/base' && ember experiment verify '/tmp/it'\\''s'"
        );
    }
}
