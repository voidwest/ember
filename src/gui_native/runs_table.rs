//! The Runs table: the kit `DataTable` delegate over run history, and the
//! cell formatting it shares with the other pages.

use super::*;

/// `TableDelegate` over the run history.
///
/// This is the kit's real table: virtualized, sortable, resizable columns,
/// keyboard-navigable. The hand-rolled grid it replaces had the same seven
/// columns in the same order and none of those properties, and at the minimum
/// window width it clipped the last two columns with no way to reach them.
///
/// The delegate owns a *snapshot* of the rows rather than reading the store on
/// demand. `TableDelegate` is addressed through `&mut self` during paint, so
/// borrowing the console's store into it would alias the very thing the table
/// is editing; a snapshot plus an explicit `sync` is the version that stays
/// honest when a run finishes while the table is on screen.
///
/// The snapshot holds [`RunRow`]s -- what the table shows -- not records: a
/// record carries its full result (every token, every layer), and the console
/// syncs on every render. It is rebuilt only when the store's generation moved
/// ([`AppStore::generation`]), so a frame with no history change copies
/// nothing.
pub(super) struct RunsDelegate {
    /// The snapshot, in store order (pinned first, then newest).
    rows: Vec<RunRow>,
    /// The store generation `rows` was built from; `None` until the first sync.
    synced: Option<u64>,
    /// How many times `rows` was rebuilt from the store. Tests read it to
    /// prove a render without a store change rebuilds nothing.
    pub(super) rebuilds: usize,
    /// Rows built by `render_td` since tests last reset it. The table is a
    /// virtual list; tests read this to prove a frame builds only the rows
    /// on screen, not all 500.
    pub(super) rows_built: usize,
    /// Display order: indices into `rows`, sorted by `sort` when one is set.
    order: Vec<usize>,
    /// The column sort the user chose, re-applied to every new snapshot:
    /// the console syncs rows on each render, and a sort that lived only in
    /// the row order was undone by the next frame.
    sort: Option<(usize, ColumnSort)>,
    /// The run whose Delete was clicked once and now asks for confirmation.
    /// Deleting is permanent, so it takes a second, deliberate click.
    confirm_delete: Option<u64>,
    /// Runs selected for comparison, mirrored from the console on sync.
    picks: Vec<u64>,
    colors: Colors,
    /// The owning console, so row actions can mutate the store they
    /// snapshot. Weak: the table must not keep the console alive.
    console: Option<WeakEntity<Console>>,
}

/// One Runs row: exactly what the table displays and sorts on, and which
/// actions the row offers. Open and Reuse look the record up by number when
/// clicked, so the result and configuration never ride along in the snapshot.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RunRow {
    pub(super) number: u64,
    pub(super) finished_at: i64,
    pub(super) model: String,
    pub(super) intervention: String,
    pub(super) duration_ms: Option<u64>,
    pub(super) baseline_tokens: Option<u32>,
    pub(super) intervention_tokens: Option<u32>,
    pub(super) outputs_equal: bool,
    pub(super) verified: bool,
    pub(super) pinned: bool,
    /// The record kept its result and configuration, so Open works.
    pub(super) can_open: bool,
    /// The record kept its configuration, so Reuse works.
    pub(super) can_reuse: bool,
    /// Divergence by layer, for the row's sparkline; `None` when the record
    /// kept no result.
    pub(super) spark: Option<Vec<f32>>,
}

impl From<&RunRecord> for RunRow {
    fn from(run: &RunRecord) -> Self {
        Self {
            number: run.number,
            finished_at: run.finished_at,
            model: run.model.clone(),
            intervention: run.intervention.clone(),
            duration_ms: run.duration_ms,
            baseline_tokens: run.baseline_tokens,
            intervention_tokens: run.intervention_tokens,
            outputs_equal: run.outputs_equal,
            verified: run.verified,
            pinned: run.pinned,
            can_open: run.result.is_some() && run.config.is_some(),
            can_reuse: run.config.is_some(),
            spark: run.result.as_ref().map(|result| {
                result
                    .layers
                    .iter()
                    .map(|layer| layer.relative_l2.unwrap_or(0.0) as f32)
                    .collect()
            }),
        }
    }
}

/// The store's runs as table rows, in display order.
fn rows_of(store: &AppStore) -> Vec<RunRow> {
    store.runs_ordered().into_iter().map(RunRow::from).collect()
}

/// Column identities. Declared once because `Column` is keyed by string and a
/// typo in a key silently sorts the wrong column rather than failing to build.
mod run_col {
    pub const RUN: &str = "run";
    pub const MODEL: &str = "model";
    pub const INTERVENTION: &str = "intervention";
    pub const TOKENS: &str = "tokens";
    pub const RESULT: &str = "result";
    pub const DURATION: &str = "duration";
    pub const WHEN: &str = "when";
    pub const ACTIONS: &str = "actions";
}

impl RunsDelegate {
    pub(super) fn new(store: &AppStore, colors: Colors) -> Self {
        let mut delegate = Self {
            rows: Vec::new(),
            synced: None,
            rebuilds: 0,
            rows_built: 0,
            order: Vec::new(),
            sort: None,
            confirm_delete: None,
            picks: Vec::new(),
            colors,
            console: None,
        };
        delegate.refresh_rows(store);
        delegate
    }

    /// Rebuild the snapshot if the store changed since the last one.
    fn refresh_rows(&mut self, store: &AppStore) {
        if self.synced == Some(store.generation()) {
            return;
        }
        self.synced = Some(store.generation());
        self.rebuilds += 1;
        self.set_rows(rows_of(store));
    }

    /// Replace the snapshot, keeping the active sort.
    fn set_rows(&mut self, rows: Vec<RunRow>) {
        self.rows = rows;
        self.apply_sort();
    }

    /// Remember a column sort and apply it. `ColumnSort::Default` returns to
    /// store order.
    pub(super) fn set_sort(&mut self, col_ix: usize, sort: ColumnSort) {
        self.sort = (!matches!(sort, ColumnSort::Default)).then_some((col_ix, sort));
        self.apply_sort();
    }

    fn apply_sort(&mut self) {
        self.order = (0..self.rows.len()).collect();
        let Some((col_ix, sort)) = self.sort else {
            return;
        };
        let descending = matches!(sort, ColumnSort::Descending);
        let rows = &self.rows;
        // Stable, so ties keep store order.
        self.order.sort_by(|&left, &right| {
            let ordering = compare_runs(&rows[left], &rows[right], col_ix);
            if descending {
                ordering.reverse()
            } else {
                ordering
            }
        });
    }

    /// The run shown at a display row.
    pub(super) fn row(&self, row_ix: usize) -> Option<&RunRow> {
        self.order.get(row_ix).and_then(|&ix| self.rows.get(ix))
    }

    /// A click on a row's Delete. The first click only arms the confirmation;
    /// `true` means this click confirmed it and the run should go.
    fn request_delete(&mut self, number: u64) -> bool {
        if self.confirm_delete == Some(number) {
            self.confirm_delete = None;
            true
        } else {
            self.confirm_delete = Some(number);
            false
        }
    }

    /// Bring the snapshot up to date with the store and repaint. Cheap when
    /// the store has not changed: see [`RunsDelegate::refresh_rows`].
    ///
    /// `TableState::refresh` is a method on the state, and a delegate has no
    /// handle on the state that owns it, so the signal has to be the context.
    ///
    /// Colours come in here too, not just rows. The delegate is built once and
    /// kept, so a palette captured at construction would survive every theme
    /// change afterwards -- which showed up as a table that stayed light while
    /// the rest of the app went dark, and then as text too dim to read.
    pub(super) fn sync(
        &mut self,
        store: &AppStore,
        picks: &[u64],
        colors: Colors,
        console: WeakEntity<Console>,
        cx: &mut Context<TableState<Self>>,
    ) {
        self.refresh_rows(store);
        if self.picks != picks {
            self.picks = picks.to_vec();
        }
        self.colors = colors;
        self.console = Some(console);
        cx.notify();
    }

    /// The text for one cell. Kept beside `render_td` so sorting and rendering
    /// cannot disagree about what a cell says.
    fn cell_text(&self, row_ix: usize, col_ix: usize) -> String {
        let Some(run) = self.row(row_ix) else {
            return String::new();
        };
        match col_ix {
            0 => format!("Run #{}", run.number),
            1 => run.model.clone(),
            2 => run.intervention.clone(),
            3 => token_cell(run),
            4 => result_word(run).to_string(),
            5 => run
                .duration_ms
                .map(|ms| format!("{:.1}s", ms as f64 / 1000.0))
                .unwrap_or_else(|| "\u{2014}".to_string()),
            6 => relative_time(run.finished_at),
            _ => String::new(),
        }
    }
}

impl TableDelegate for RunsDelegate {
    fn columns_count(&self, _cx: &App) -> usize {
        8
    }

    fn rows_count(&self, _cx: &App) -> usize {
        self.rows.len()
    }

    fn column(&self, col_ix: usize, _cx: &App) -> Column {
        // The table supplies its own cell padding and a sort chevron per
        // column, so a width has to cover that overhead as well as the text.
        let column = match col_ix {
            0 => Column::new(run_col::RUN, "Run").width(px(84.0)),
            1 => Column::new(run_col::MODEL, "Model").width(px(224.0)),
            2 => Column::new(run_col::INTERVENTION, "Intervention").width(px(228.0)),
            3 => Column::new(run_col::TOKENS, "Tokens").width(px(88.0)),
            4 => Column::new(run_col::RESULT, "Result").width(px(176.0)),
            5 => Column::new(run_col::DURATION, "Duration").width(px(96.0)),
            6 => Column::new(run_col::WHEN, "When").width(px(80.0)),
            // Wide enough for Select + Open + Pin + Reuse + Star + Delete,
            // the fullest lane a row can carry.
            _ => Column::new(run_col::ACTIONS, "").width(px(430.0)),
        };
        // Every data column sorts: `sortable` is a flagless builder, and a
        // history you cannot re-order is a log file. The action lane does not.
        if col_ix == 7 {
            column
        } else {
            column.sortable().resizable(true)
        }
    }

    fn perform_sort(
        &mut self,
        col_ix: usize,
        sort: ColumnSort,
        _window: &mut Window,
        _cx: &mut Context<TableState<Self>>,
    ) {
        self.set_sort(col_ix, sort);
    }

    fn render_td(
        &mut self,
        row_ix: usize,
        col_ix: usize,
        _window: &mut Window,
        cx: &mut Context<TableState<Self>>,
    ) -> impl IntoElement {
        let colors = self.colors;
        if col_ix == 0 {
            self.rows_built += 1;
        }
        // The action lane: pin and delete, as quiet text commands. They talk
        // to the console through the weak entity the snapshot came from, so
        // the store, the snapshot and the screen stay one system.
        if col_ix == 7 {
            let Some(run) = self.row(row_ix) else {
                return div().into_any_element();
            };
            let number = run.number;
            let pinned = run.pinned;
            // Armed delete: the lane becomes the question, so the confirming
            // click cannot land on a neighbouring command by accident.
            if self.confirm_delete == Some(number) {
                return div()
                    .flex()
                    .flex_row()
                    .gap(px(Space::SM))
                    .child(
                        Button::new(SharedString::from(format!("run-delete-cancel:{number}")))
                            .ghost()
                            .compact()
                            .label("Cancel")
                            .on_click(cx.listener(|table, _, _, cx| {
                                table.delegate_mut().confirm_delete = None;
                                cx.notify();
                            })),
                    )
                    .child(
                        Button::new(SharedString::from(format!("run-delete-confirm:{number}")))
                            .ghost()
                            .compact()
                            .label(format!("Delete run #{number}"))
                            .text_color(colors.err)
                            .tooltip("Remove this run from the history permanently")
                            .on_click(cx.listener(move |table, _, _, cx| {
                                table.delegate_mut().request_delete(number);
                                let console = table.delegate().console.clone();
                                if let Some(console) = console {
                                    let _ = console.update(cx, |console, cx| {
                                        console.store.remove_run(number);
                                        console.persist();
                                        cx.notify();
                                    });
                                }
                                cx.notify();
                            })),
                    )
                    .into_any_element();
            }
            // Reuse appears only where there is a stored configuration to
            // load; older records keep Pin and Delete and nothing that would
            // pretend to work.
            let reuse = run.can_reuse.then(|| self.console.clone());
            let console = self.console.clone();
            let mut lane = div().flex().flex_row().gap(px(Space::SM));
            // Any row can be selected for comparison; one that kept no
            // result is refused by the selection bar with the reason.
            let picked = self.picks.contains(&number);
            let select_console = self.console.clone();
            lane = lane.child(
                Button::new(SharedString::from(format!("run-select:{number}")))
                    .ghost()
                    .compact()
                    .selected(picked)
                    .label(if picked {
                        "\u{2713} Selected"
                    } else {
                        "Select"
                    })
                    .tooltip("Select two runs to compare them side by side")
                    .on_click(move |_, _, cx| {
                        if let Some(console) = select_console.as_ref() {
                            let _ = console.update(cx, |console, cx| {
                                console.toggle_compare_pick(number, cx);
                            });
                        }
                    }),
            );
            // Every row exports: Markdown always, and its bundle when it is
            // on disk or can be re-run.
            let export_console = self.console.clone();
            lane = lane.child(
                Button::new(SharedString::from(format!("run-export:{number}")))
                    .ghost()
                    .compact()
                    .label("Export")
                    .tooltip("Copy as Markdown, or get this run's verifiable bundle")
                    .on_click(move |_, _, cx| {
                        if let Some(console) = export_console.as_ref() {
                            let _ = console.update(cx, |console, cx| {
                                console.toggle_export(number, cx);
                            });
                        }
                    }),
            );
            // Open appears only where the run kept its result.
            if run.can_open {
                let console = self.console.clone();
                lane = lane.child(
                    Button::new(SharedString::from(format!("run-open:{number}")))
                        .ghost()
                        .compact()
                        .label("Open")
                        .tooltip("Reopen this run's comparison")
                        .on_click(move |_, _, cx| {
                            if let Some(console) = console.as_ref() {
                                let _ = console.update(cx, |console, cx| {
                                    console.open_run(number, cx);
                                });
                            }
                        }),
                );
                let console = self.console.clone();
                lane = lane.child(
                    Button::new(SharedString::from(format!("run-compare:{number}")))
                        .ghost()
                        .compact()
                        .label("Pin")
                        .tooltip("Pin this run as the reference for your next run, and go back to the workspace")
                        .on_click(move |_, _, cx| {
                            if let Some(console) = console.as_ref() {
                                let _ = console.update(cx, |console, cx| {
                                    console.compare_with_run(number, cx);
                                });
                            }
                        }),
                );
            }
            if let Some(console) = reuse {
                lane = lane.child(
                    Button::new(SharedString::from(format!("run-reuse:{number}")))
                        .ghost()
                        .compact()
                        .label("Reuse")
                        .tooltip("Load this run's configuration into a new experiment")
                        .on_click(move |_, _, cx| {
                            if let Some(console) = console.as_ref() {
                                let _ = console.update(cx, |console, cx| {
                                    console.reuse_run(number, cx);
                                });
                            }
                        }),
                );
            }
            return lane
                .child(
                    Button::new(SharedString::from(format!("run-pin:{number}")))
                        .ghost()
                        .compact()
                        .label(if pinned { "Unstar" } else { "Star" })
                        .tooltip(if pinned {
                            "Remove the star"
                        } else {
                            "Star this run to keep it at the top of the history"
                        })
                        .on_click(move |_, _, cx| {
                            if let Some(console) = console.as_ref() {
                                let _ = console.update(cx, |console, cx| {
                                    console.store.toggle_pin(number);
                                    console.persist();
                                    cx.notify();
                                });
                            }
                        }),
                )
                .child(
                    Button::new(SharedString::from(format!("run-delete:{number}")))
                        .ghost()
                        .compact()
                        .label("Delete")
                        .text_color(colors.err)
                        .tooltip("Delete this run from the history (asks to confirm)")
                        .on_click(cx.listener(move |table, _, _, cx| {
                            table.delegate_mut().request_delete(number);
                            cx.notify();
                        })),
                )
                .into_any_element();
        }
        let text = self.cell_text(row_ix, col_ix);
        let failed = self
            .row(row_ix)
            .is_some_and(|run| col_ix == 4 && !run.verified);
        let tint = if failed {
            colors.err
        } else if matches!(col_ix, 0 | 1 | 2 | 4) {
            // Run, model, intervention, result: what you came to read.
            colors.text
        } else {
            // Tokens, duration and when: context for the row beside them.
            colors.text_muted
        };
        let _ = cx;
        // The result reads as a word and, beside it, the run's divergence by
        // layer: which runs did something is visible down the column.
        if col_ix == 4 {
            let spark = self.row(row_ix).and_then(|run| {
                let ink: Hsla = if run.outputs_equal {
                    colors.text_faint.into()
                } else {
                    colors.accent.into()
                };
                run.spark
                    .as_deref()
                    .and_then(|values| super::spark::sparkline(values, ink, 56.0, 16.0))
            });
            return div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(Space::SM))
                .child(
                    div()
                        .w(px(84.0))
                        .flex_none()
                        .child(mono(text, Type::META, tint)),
                )
                .children(spark)
                .into_any_element();
        }
        mono(text, Type::META, tint).into_any_element()
    }

    fn render_th(
        &mut self,
        col_ix: usize,
        _window: &mut Window,
        cx: &mut Context<TableState<Self>>,
    ) -> impl IntoElement {
        let colors = self.colors;
        let name = self.column(col_ix, cx).name;
        div()
            .size_full()
            .flex()
            .items_center()
            .child(label(name, Type::META, colors.text_faint))
    }
}

/// Token cell: a bare count when the two sides agree, an arrow when they do
/// not. Reading `48` next to `48 -> 12` is the cheapest way to see that an
/// intervention shortened the completion.
/// A word, not a colour. `failed` is the only result not obvious from the
/// columns beside it, and the only one the token counts do not imply.
///
/// Free rather than a method on the delegate: it is read from inside the
/// `sort_by` closure, which already holds a mutable borrow of the delegate's
/// rows, so a `&self` method would not compile there.
fn result_word(run: &RunRow) -> &'static str {
    if !run.verified {
        "failed"
    } else if run.outputs_equal {
        "unchanged"
    } else {
        "changed"
    }
}

/// Ordering of two runs by one Runs column, ascending.
fn compare_runs(left: &RunRow, right: &RunRow, col_ix: usize) -> std::cmp::Ordering {
    match col_ix {
        0 => left.number.cmp(&right.number),
        1 => left.model.cmp(&right.model),
        2 => left.intervention.cmp(&right.intervention),
        4 => result_word(left).cmp(result_word(right)),
        5 => left.duration_ms.cmp(&right.duration_ms),
        6 => left.finished_at.cmp(&right.finished_at),
        // Tokens are a pair; sort on the baseline side so the order is
        // total, and the arrow in the cell still shows both.
        _ => left
            .baseline_tokens
            .cmp(&right.baseline_tokens)
            .then(left.intervention_tokens.cmp(&right.intervention_tokens)),
    }
}

fn token_cell(run: &RunRow) -> String {
    match (run.baseline_tokens, run.intervention_tokens) {
        (Some(baseline), Some(intervention)) if baseline == intervention => format!("{baseline}"),
        (Some(baseline), Some(intervention)) => format!("{baseline} \u{2192} {intervention}"),
        _ => "\u{2014}".to_string(),
    }
}

/// Quantisation, read off the filename.
///
/// GGUF files are conventionally named `<model>-<quant>.gguf`, so this is a
/// naming convention rather than a parse of the header. It is a display
/// nicety: a file that does not follow the convention shows an em dash rather
/// than a guess, and the loader remains the authority on what a model actually
/// is. Returns an owned string because the caller renders per frame -- a
/// cached leak here would grow once per painted row, forever.
pub(super) fn quant_of(path: &str) -> String {
    let stem = path.trim_end_matches(".gguf");
    let Some((_, tail)) = stem.rsplit_once('-') else {
        return "\u{2014}".to_string();
    };
    let upper = tail.to_ascii_uppercase();
    // Q4_K_M, Q8_0, Q6_K, F16, BF16 and friends.
    if upper.starts_with('Q')
        && upper[1..]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return upper;
    }
    "\u{2014}".to_string()
}

/// Human-readable file size, binary units, one decimal.
pub(super) fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Keep the informative end of a path, dropping the front.
///
/// `/Users/someone/very/deep/place/model-q8.gguf` should read as
/// `…/place/model-q8.gguf`: the filename is the part that identifies it, and a
/// right-truncation would throw exactly that away.
pub(super) fn truncate_path_start(path: &str, max: usize) -> String {
    if path.chars().count() <= max {
        return path.to_string();
    }
    let keep: String = path.chars().skip(path.chars().count() - max + 1).collect();
    format!("\u{2026}{keep}")
}

/// Coarse relative time for a run row.
///
/// Deliberately not a full date: the table is scanned, and "2m" answers the
/// only question a scan asks. Anything older than a week gets a date, because
/// "6d" stops being useful once you lose the thread of what you were doing.
pub(super) fn relative_time(at: i64) -> String {
    let elapsed = unix_now().saturating_sub(at).max(0);
    match elapsed {
        ..=59 => "just now".to_string(),
        60..=3_599 => format!("{}m", elapsed / 60),
        3_600..=86_399 => format!("{}h", elapsed / 3_600),
        86_400..=604_799 => format!("{}d", elapsed / 86_400),
        _ => {
            let days = elapsed / 86_400;
            if days < 365 {
                format!("{}d", days)
            } else {
                format!("{}y", days / 365)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::theme;
    use super::{AppStore, RunRecord, RunRow, RunsDelegate};

    fn record(number: u64, model: &str) -> RunRecord {
        RunRecord {
            number,
            finished_at: number as i64,
            model: model.into(),
            intervention: "Scale".into(),
            hook: "after-mlp".into(),
            layer: None,
            duration_ms: None,
            baseline_tokens: None,
            intervention_tokens: None,
            diverged_at_step: None,
            outputs_equal: false,
            verified: true,
            pinned: false,
            prompt: String::new(),
            config: None,
            result: None,
            bundles: None,
        }
    }

    fn shown_models(delegate: &RunsDelegate) -> Vec<String> {
        (0..delegate.rows.len())
            .map(|ix| delegate.row(ix).unwrap().model.clone())
            .collect()
    }

    #[test]
    fn a_runs_sort_survives_the_next_snapshot() {
        use gpui_kit::component::table::ColumnSort;
        let rows = || {
            [record(1, "b"), record(2, "c"), record(3, "a")]
                .iter()
                .map(RunRow::from)
                .collect::<Vec<_>>()
        };
        let mut delegate = RunsDelegate::new(&AppStore::default(), theme::light());
        delegate.set_rows(rows());
        delegate.set_sort(1, ColumnSort::Ascending);
        assert_eq!(shown_models(&delegate), ["a", "b", "c"]);
        // The console re-syncs rows on every render.
        delegate.set_rows(rows());
        assert_eq!(shown_models(&delegate), ["a", "b", "c"], "sort kept");
        delegate.set_sort(1, ColumnSort::Descending);
        delegate.set_rows(rows());
        assert_eq!(shown_models(&delegate), ["c", "b", "a"]);
        delegate.set_sort(1, ColumnSort::Default);
        assert_eq!(
            shown_models(&delegate),
            ["b", "c", "a"],
            "store order again"
        );
    }

    #[test]
    fn deleting_a_run_needs_a_second_click_on_the_same_row() {
        let mut delegate = RunsDelegate::new(&AppStore::default(), theme::light());
        assert!(!delegate.request_delete(3), "the first click only asks");
        assert!(
            !delegate.request_delete(4),
            "a click on another row moves the question"
        );
        assert!(
            delegate.request_delete(4),
            "the second click on it confirms"
        );
        assert_eq!(delegate.confirm_delete, None);
    }

    #[test]
    fn rows_are_rebuilt_only_when_the_store_changed() {
        let mut store = AppStore::default();
        store.push_run(record(1, "a"));
        let mut delegate = RunsDelegate::new(&store, theme::light());
        assert_eq!(delegate.rebuilds, 1);
        for _ in 0..10 {
            delegate.refresh_rows(&store);
        }
        assert_eq!(delegate.rebuilds, 1, "an unchanged store copies nothing");
        store.push_run(record(2, "b"));
        delegate.refresh_rows(&store);
        assert_eq!(delegate.rebuilds, 2);
        assert_eq!(shown_models(&delegate), ["b", "a"]);
        assert!(store.toggle_pin(1));
        delegate.refresh_rows(&store);
        assert_eq!(shown_models(&delegate), ["a", "b"], "a pin shows");
    }
}
