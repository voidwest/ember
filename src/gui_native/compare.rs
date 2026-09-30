//! Comparing two saved runs.
//!
//! Pinning a reference answers "is the next run better than this one?". This
//! answers the question that comes after a week of runs: "what was different
//! between those two?". Any two records that kept their result can be put
//! side by side -- outputs, a token diff, both divergence curves on one chart
//! and a table of what moved. Records from before results were kept have
//! nothing to compare, and the page says so instead of drawing an empty chart.

use super::chart::{layer_divergence_chart_with, LayerMarks};
use super::components::*;
use super::history::{record_series, record_values};
use super::theme::{Radius, Space, Type};
use super::workspace::change_summary;
use super::{truncate_chars, Colors, Console, View};
use ember::app_store::{RecordToken, RunRecord};
use gpui_kit::component::{
    button::{Button, ButtonVariants},
    Disableable, Sizable,
};
use gpui_kit::prelude::*;
use gpui_kit::*;

/// Why a record cannot take part in a comparison, if it cannot.
pub(super) fn not_comparable(record: Option<&RunRecord>, number: u64) -> Option<String> {
    match record {
        None => Some(format!("Run #{number} is no longer in the history.")),
        Some(record) if record.result.is_none() => Some(format!(
            "Run #{number} has no saved result, so it can't be compared. Runs recorded by older versions of Ember kept only their summary."
        )),
        Some(_) => None,
    }
}

/// One position of the token diff between two runs' intervention outputs.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TokenPair {
    pub position: usize,
    pub left: Option<String>,
    pub right: Option<String>,
    pub differs: bool,
}

/// Pair two runs' intervention tokens by position. A position only one run
/// reached counts as a difference; one neither intervention reached (a
/// longer baseline) is left out.
pub(super) fn token_diff(left: &[RecordToken], right: &[RecordToken]) -> Vec<TokenPair> {
    let count = left.len().max(right.len());
    (0..count)
        .map(|index| {
            let left_text = left.get(index).and_then(|token| token.intervention.clone());
            let right_text = right
                .get(index)
                .and_then(|token| token.intervention.clone());
            let position = left
                .get(index)
                .or(right.get(index))
                .map_or(index + 1, |token| token.position);
            TokenPair {
                position,
                differs: left_text != right_text,
                left: left_text,
                right: right_text,
            }
        })
        .filter(|pair| pair.left.is_some() || pair.right.is_some())
        .collect()
}

/// One row of the metrics table: the two values and what changed.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct MetricRow {
    pub name: &'static str,
    pub left: String,
    pub right: String,
    pub delta: String,
}

fn dash() -> String {
    "\u{2014}".to_string()
}

fn opt<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(dash, |value| value.to_string())
}

fn change_line(record: &RunRecord) -> String {
    match record_values(record) {
        Some(values) => change_summary(&values),
        None => match record.layer {
            Some(layer) => format!(
                "{} \u{00b7} layer {layer} \u{00b7} {}",
                record.intervention, record.hook
            ),
            None => format!("{} \u{00b7} {}", record.intervention, record.hook),
        },
    }
}

/// The metrics table. Deltas read "right minus left": the later-selected
/// run against the first one.
pub(super) fn metric_rows(left: &RunRecord, right: &RunRecord) -> Vec<MetricRow> {
    let same = |a: &str, b: &str| if a == b { "same" } else { "different" }.to_string();
    let (Some(lr), Some(rr)) = (left.result.as_ref(), right.result.as_ref()) else {
        return Vec::new();
    };
    let words = |equal: bool| if equal { "unchanged" } else { "changed" }.to_string();
    let tokens = |record: &RunRecord| match (record.baseline_tokens, record.intervention_tokens) {
        (Some(b), Some(i)) if b == i => b.to_string(),
        (Some(b), Some(i)) => format!("{b} \u{2192} {i}"),
        _ => dash(),
    };
    let peak = |value: Option<f64>, layer: Option<usize>| match (value, layer) {
        (Some(value), Some(layer)) => format!("{value:.3} @ L{layer}"),
        (Some(value), None) => format!("{value:.3}"),
        _ => dash(),
    };
    let peak_delta = match (lr.peak_relative_l2, rr.peak_relative_l2) {
        (Some(a), Some(b)) if a > 0.0 => format!("{:+.3} ({:+.0}%)", b - a, (b - a) / a * 100.0),
        (Some(a), Some(b)) => format!("{:+.3}", b - a),
        _ => dash(),
    };
    let layer_delta = match (lr.first_layer_divergence, rr.first_layer_divergence) {
        (Some(a), Some(b)) if a == b => "same layer".to_string(),
        (Some(a), Some(b)) => format!("{:+} layers", b as i64 - a as i64),
        _ => dash(),
    };
    let duration_delta = match (left.duration_ms, right.duration_ms) {
        (Some(a), Some(b)) => format!("{:+.1}s", (b as f64 - a as f64) / 1000.0),
        _ => dash(),
    };
    let seconds =
        |ms: Option<u64>| ms.map_or_else(dash, |ms| format!("{:.1}s", ms as f64 / 1000.0));
    vec![
        MetricRow {
            name: "Change",
            left: change_line(left),
            right: change_line(right),
            delta: same(&change_line(left), &change_line(right)),
        },
        MetricRow {
            name: "Model",
            left: left.model.clone(),
            right: right.model.clone(),
            delta: same(&left.model, &right.model),
        },
        MetricRow {
            name: "Prompt",
            left: truncate_chars(left.prompt.trim(), 48),
            right: truncate_chars(right.prompt.trim(), 48),
            delta: same(left.prompt.trim(), right.prompt.trim()),
        },
        MetricRow {
            name: "Text output",
            left: words(left.outputs_equal),
            right: words(right.outputs_equal),
            delta: same(&words(left.outputs_equal), &words(right.outputs_equal)),
        },
        MetricRow {
            name: "Tokens",
            left: tokens(left),
            right: tokens(right),
            delta: same(&tokens(left), &tokens(right)),
        },
        MetricRow {
            name: "Words first differ",
            left: left
                .diverged_at_step
                .map_or_else(dash, |step| format!("step {step}")),
            right: right
                .diverged_at_step
                .map_or_else(dash, |step| format!("step {step}")),
            delta: match (left.diverged_at_step, right.diverged_at_step) {
                (Some(a), Some(b)) => format!("{:+} steps", b as i64 - a as i64),
                _ => dash(),
            },
        },
        MetricRow {
            name: "First divergence",
            left: lr
                .first_layer_divergence
                .map_or_else(dash, |l| format!("layer {l}")),
            right: rr
                .first_layer_divergence
                .map_or_else(dash, |l| format!("layer {l}")),
            delta: layer_delta,
        },
        MetricRow {
            name: "Peak divergence",
            left: peak(lr.peak_relative_l2, lr.peak_layer),
            right: peak(rr.peak_relative_l2, rr.peak_layer),
            delta: peak_delta,
        },
        MetricRow {
            name: "Duration",
            left: seconds(left.duration_ms),
            right: seconds(right.duration_ms),
            delta: duration_delta,
        },
        MetricRow {
            name: "Verified",
            left: opt(Some(if left.verified { "yes" } else { "no" })),
            right: opt(Some(if right.verified { "yes" } else { "no" })),
            delta: same(&left.verified.to_string(), &right.verified.to_string()),
        },
    ]
}

/// Tokens as they read in a cell: spaces and newlines made visible.
fn visible(token: &Option<String>) -> String {
    match token {
        Some(text) => text.replace(' ', "\u{00b7}").replace('\n', "\u{23ce}"),
        None => "\u{2205}".to_string(),
    }
}

impl Console {
    fn record(&self, number: u64) -> Option<&RunRecord> {
        self.store.runs.iter().find(|run| run.number == number)
    }

    /// Select or deselect a run for comparison. Two at most: a third pick
    /// replaces the older of the two.
    pub(super) fn toggle_compare_pick(&mut self, number: u64, cx: &mut Context<Self>) {
        if let Some(index) = self.compare_picks.iter().position(|&n| n == number) {
            self.compare_picks.remove(index);
        } else {
            self.compare_picks.push(number);
            if self.compare_picks.len() > 2 {
                self.compare_picks.remove(0);
            }
        }
        cx.notify();
    }

    /// Why the current selection cannot be compared, if it cannot.
    pub(super) fn compare_blocker(&self) -> Option<String> {
        self.compare_picks
            .iter()
            .find_map(|&number| not_comparable(self.record(number), number))
    }

    /// Open the comparison of the two selected runs. Refused (with the
    /// reason on the page) unless both kept their results.
    pub(super) fn open_comparison(&mut self, cx: &mut Context<Self>) {
        let [left, right] = self.compare_picks[..] else {
            return;
        };
        if self.compare_blocker().is_some() {
            return;
        }
        self.comparing = Some((left, right));
        self.goto(View::Runs, cx);
        cx.notify();
    }

    pub(super) fn close_comparison(&mut self, cx: &mut Context<Self>) {
        self.comparing = None;
        cx.notify();
    }

    /// The selection bar above the Runs table.
    pub(super) fn compare_bar(
        &self,
        colors: &Colors,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        if self.compare_picks.is_empty() {
            return None;
        }
        let blocker = self.compare_blocker();
        let picks: Vec<String> = self
            .compare_picks
            .iter()
            .map(|n| format!("Run #{n}"))
            .collect();
        let (text, color) = match (&blocker, picks.len()) {
            (Some(reason), _) => (reason.clone(), colors.warn),
            (None, 1) => (
                format!("{} selected. Select one more run to compare.", picks[0]),
                colors.text,
            ),
            (None, _) => (
                format!("{} and {} selected.", picks[0], picks[1]),
                colors.text,
            ),
        };
        let ready = blocker.is_none() && picks.len() == 2;
        Some(
            div()
                .id("compare-bar")
                .test_support()
                .w_full()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(Space::MD))
                .px(px(Space::MD))
                .py(px(Space::SM))
                .rounded(px(Radius::MD))
                .border_l_2()
                .border_color(if blocker.is_some() {
                    colors.warn
                } else {
                    colors.accent
                })
                .bg(colors.surface)
                .child(
                    div()
                        .flex_1()
                        .min_w(px(0.0))
                        .child(label(text, Type::LABEL, color)),
                )
                .child(
                    Button::new("runs-compare-open")
                        .primary()
                        .small()
                        .label("Compare runs")
                        .disabled(!ready)
                        .tooltip("Put the two selected runs side by side")
                        .on_click(cx.listener(|console, _: &ClickEvent, _window, cx| {
                            console.open_comparison(cx);
                        })),
                )
                .child(text_button(
                    "runs-compare-clear",
                    "Clear",
                    cx.listener(|console, _: &ClickEvent, _window, cx| {
                        console.compare_picks.clear();
                        cx.notify();
                    }),
                ))
                .into_any_element(),
        )
    }

    /// The comparison page, in place of the Runs table.
    pub(super) fn compare_view(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let Some((left_number, right_number)) = self.comparing else {
            return div();
        };
        let (Some(left), Some(right)) = (self.record(left_number), self.record(right_number))
        else {
            return div()
                .flex()
                .flex_col()
                .gap(px(Space::MD))
                .child(label(
                    "One of these runs is no longer in the history.",
                    Type::BODY,
                    colors.text_muted,
                ))
                .child(self.compare_back(cx));
        };
        let (Some(left_result), Some(right_result)) = (&left.result, &right.result) else {
            return div()
                .flex()
                .flex_col()
                .gap(px(Space::MD))
                .children(
                    not_comparable(Some(left), left_number)
                        .or(not_comparable(Some(right), right_number))
                        .map(|reason| label(reason, Type::BODY, colors.warn)),
                )
                .child(self.compare_back(cx));
        };
        let left_name = format!("Run #{left_number}");
        let right_name = format!("Run #{right_number}");

        // Metrics table.
        let cell = |text: String, color: Rgba| {
            div()
                .flex_1()
                .min_w(px(0.0))
                .overflow_hidden()
                .child(label(text, Type::LABEL, color))
        };
        let head = |text: String| {
            div()
                .flex_1()
                .min_w(px(0.0))
                .child(label(text, Type::META, colors.text_faint))
        };
        let rows = metric_rows(left, right).into_iter().map(|row| {
            let unchanged = row.delta == "same" || row.delta == "same layer";
            div()
                .flex()
                .flex_row()
                .gap(px(Space::MD))
                .py(px(Space::XS))
                .border_t_1()
                .border_color(colors.border)
                .child(div().w(px(150.0)).flex_none().child(label(
                    row.name,
                    Type::LABEL,
                    colors.text_muted,
                )))
                .child(cell(row.left, colors.text))
                .child(cell(row.right, colors.text))
                .child(cell(
                    row.delta,
                    if unchanged {
                        colors.text_faint
                    } else {
                        colors.accent
                    },
                ))
        });
        let table = div()
            .id("compare-metrics")
            .test_support()
            .flex()
            .flex_col()
            .px(px(Space::LG))
            .py(px(Space::MD))
            .rounded(px(Radius::LG))
            .bg(colors.surface)
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(Space::MD))
                    .pb(px(Space::XS))
                    .child(div().w(px(150.0)).flex_none())
                    .child(head(left_name.clone()))
                    .child(head(right_name.clone()))
                    .child(head(format!("{right_name} vs {left_name}"))),
            )
            .children(rows);

        // Both curves on one chart: the later pick solid, the first dashed.
        let chart = div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .px(px(Space::LG))
            .py(px(Space::MD))
            .rounded(px(Radius::LG))
            .bg(colors.surface)
            .child(label("Representation divergence", Type::LABEL, colors.text))
            .child(label(
                format!("Solid: {right_name}. Dashed: {left_name}. Relative L2 difference from each run's own baseline, per layer."),
                Type::META,
                colors.text_faint,
            ))
            .child(layer_divergence_chart_with(
                cx.entity(),
                record_series(right_result),
                Some((
                    record_series(left_result),
                    SharedString::from(left_name.clone()),
                )),
                LayerMarks {
                    intervention: None,
                    selected: self.selected_layer,
                    hovered: self.hovered_layer,
                },
                260.0,
                colors,
            ));

        // Side-by-side outputs.
        let same_baseline = left_result.baseline_text.trim() == right_result.baseline_text.trim();
        let output_column = |name: &str, baseline: &str, intervention: &str| {
            div()
                .flex_1()
                .min_w(px(0.0))
                .flex()
                .flex_col()
                .gap(px(Space::SM))
                .px(px(Space::LG))
                .py(px(Space::MD))
                .rounded(px(Radius::LG))
                .bg(colors.surface)
                .child(label(name.to_string(), Type::LABEL, colors.text))
                .child(label("Baseline", Type::META, colors.text_faint))
                .child(multiline(
                    baseline.trim(),
                    Type::BODY,
                    colors.text_muted,
                    super::FONT_ARABIC_NAME,
                ))
                .child(label("Intervention", Type::META, colors.text_faint))
                .child(multiline(
                    intervention.trim(),
                    Type::BODY,
                    colors.text,
                    super::FONT_ARABIC_NAME,
                ))
        };
        let outputs = div()
            .flex()
            .flex_col()
            .gap(px(Space::SM))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(Space::MD))
                    .child(output_column(
                        &left_name,
                        &left_result.baseline_text,
                        &left_result.intervention_text,
                    ))
                    .child(output_column(
                        &right_name,
                        &right_result.baseline_text,
                        &right_result.intervention_text,
                    )),
            )
            .child(label(
                if same_baseline {
                    "Both runs start from the same baseline text, so the interventions are directly comparable."
                } else {
                    "The baselines differ (a different prompt, model or length), so compare each intervention against its own baseline."
                },
                Type::META,
                colors.text_faint,
            ));

        // Token diff of the two interventions.
        let pairs = token_diff(&left_result.tokens, &right_result.tokens);
        let token_panel = {
            let differing = pairs.iter().filter(|pair| pair.differs).count();
            let summary = if pairs.is_empty() {
                "No token-level record for these runs; compare the outputs above.".to_string()
            } else if differing == 0 {
                format!(
                    "The two interventions wrote the same {} tokens.",
                    pairs.len()
                )
            } else {
                let first = pairs
                    .iter()
                    .find(|pair| pair.differs)
                    .map_or(0, |pair| pair.position);
                format!(
                    "The interventions first differ at token {first}; {differing} of {} positions differ. Top: {left_name}. Bottom: {right_name}.",
                    pairs.len()
                )
            };
            const SHOWN: usize = 96;
            let cells = pairs.iter().take(SHOWN).map(|pair| {
                let tint = if pair.differs {
                    colors.accent
                } else {
                    colors.text_muted
                };
                div()
                    .flex()
                    .flex_col()
                    .px(px(Space::XS))
                    .py(px(2.0))
                    .rounded(px(Radius::SM))
                    .when(pair.differs, |cell| {
                        cell.border_1().border_color(colors.accent)
                    })
                    .child(mono(visible(&pair.left), Type::META, tint))
                    .when(pair.differs, |cell| {
                        cell.child(mono(visible(&pair.right), Type::META, colors.text))
                    })
            });
            div()
                .id("compare-tokens")
                .test_support()
                .flex()
                .flex_col()
                .gap(px(Space::SM))
                .px(px(Space::LG))
                .py(px(Space::MD))
                .rounded(px(Radius::LG))
                .bg(colors.surface)
                .child(label("Token diff", Type::LABEL, colors.text))
                .child(label(summary, Type::META, colors.text_muted))
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .flex_wrap()
                        .gap(px(Space::XS))
                        .children(cells),
                )
                .when(pairs.len() > SHOWN, |panel| {
                    panel.child(label(
                        format!("First {SHOWN} of {} positions shown.", pairs.len()),
                        Type::META,
                        colors.text_faint,
                    ))
                })
        };

        div()
            .flex()
            .flex_col()
            .gap(px(Space::LG))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(Space::MD))
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap(px(Space::XS))
                            .child(label(
                                format!("{left_name} vs {right_name}"),
                                Type::SECTION,
                                colors.text,
                            ))
                            .child(label(
                                format!("{} \u{2192} {}", change_line(left), change_line(right)),
                                Type::BODY,
                                colors.text_muted,
                            )),
                    )
                    .child(text_button(
                        "compare-swap",
                        "Swap",
                        cx.listener(|console, _: &ClickEvent, _window, cx| {
                            if let Some((a, b)) = console.comparing {
                                console.comparing = Some((b, a));
                                console.compare_picks = vec![b, a];
                                cx.notify();
                            }
                        }),
                    ))
                    .child(self.compare_back(cx)),
            )
            .child(table)
            .child(outputs)
            .child(token_panel)
            .child(chart)
    }

    fn compare_back(&self, cx: &mut Context<Self>) -> Button {
        text_button(
            "compare-back",
            "Back to runs",
            cx.listener(|console, _: &ClickEvent, _window, cx| {
                console.close_comparison(cx);
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{metric_rows, not_comparable, token_diff};
    use ember::app_store::{RecordResult, RecordToken, RunRecord};

    fn token(position: usize, text: &str) -> RecordToken {
        RecordToken {
            position,
            baseline: Some(text.into()),
            intervention: Some(text.into()),
            differs: false,
        }
    }

    fn record(number: u64, peak: Option<f64>) -> RunRecord {
        RunRecord {
            number,
            finished_at: 0,
            model: "m".into(),
            intervention: "Zero".into(),
            hook: "after-mlp".into(),
            layer: Some(4),
            duration_ms: Some(1_000),
            baseline_tokens: Some(8),
            intervention_tokens: Some(8),
            diverged_at_step: Some(2),
            outputs_equal: false,
            verified: true,
            pinned: false,
            prompt: "p".into(),
            config: None,
            result: peak.map(|peak| RecordResult {
                baseline_text: "a".into(),
                intervention_text: "b".into(),
                layers: Vec::new(),
                tokens: Vec::new(),
                first_layer_divergence: Some(4),
                peak_layer: Some(6),
                peak_relative_l2: Some(peak),
                tokens_equal: false,
            }),
        }
    }

    #[test]
    fn the_token_diff_pairs_by_position_and_counts_a_missing_side() {
        let left = [token(1, " Paris"), token(2, "."), token(3, " It")];
        let right = [token(1, " Paris"), token(2, " is")];
        let pairs = token_diff(&left, &right);
        assert_eq!(pairs.len(), 3);
        assert!(!pairs[0].differs);
        assert!(pairs[1].differs);
        assert!(pairs[2].differs && pairs[2].right.is_none());
    }

    #[test]
    fn metric_deltas_read_right_minus_left() {
        let rows = metric_rows(&record(1, Some(0.5)), &record(2, Some(0.75)));
        let peak = rows
            .iter()
            .find(|row| row.name == "Peak divergence")
            .unwrap();
        assert_eq!(peak.left, "0.500 @ L6");
        assert_eq!(peak.delta, "+0.250 (+50%)");
        let first = rows
            .iter()
            .find(|row| row.name == "First divergence")
            .unwrap();
        assert_eq!(first.delta, "same layer");
    }

    #[test]
    fn a_record_without_a_result_says_it_cannot_be_compared() {
        let old = record(3, None);
        assert!(not_comparable(Some(&old), 3)
            .unwrap()
            .contains("can't be compared"));
        assert!(not_comparable(Some(&record(4, Some(0.1))), 4).is_none());
        assert!(not_comparable(None, 9).is_some());
    }
}
