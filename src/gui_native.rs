//! Ember v0.6 native experiment console (`ember gui`).
//!
//! A native, single-window console over the exact same v0.5 pipeline as the
//! web console (`ember web-gui`). The UI uses GPUI Kit controls and its
//! platform renderer (Metal on macOS), and
//! every experiment is executed in a worker thread through the shared
//! `GuiSession` core, which in turn calls `prepare_run` / `execute_prepared`
//! — the same code path as `ember experiment run`. No inference logic lives
//! in the UI.
//!
//! GPUI Kit owns text editing and platform input-method integration. Embedded
//! Noto fonts provide offline Latin and Arabic glyph coverage.

use crate::gui::{
    discover_models, parse_run_request, ExperimentComparison, RestoreBundle, RunBundle, RunConfig,
    RunOutput, RunRequest, SessionInfo,
};
use clap::Args as ClapArgs;
use ember::app_store::{self, AppStore, RunRecord};
use ember::quant_k::KStrategy;
use gpui_kit::component::{
    button::{Button, ButtonVariants},
    kbd::Kbd,
    table::{Column, ColumnSort, DataTable, TableDelegate, TableState},
    tooltip::Tooltip,
    Selectable, Sizable,
};
use gpui_kit::prelude::*;
use gpui_kit::*;
use std::borrow::Cow;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

gpui_kit::actions!(
    ember_gui,
    [
        Quit,
        OpenSettings,
        OpenPalette,
        HideShowSidebar,
        HideShowInspector,
        EnterPresentation,
        StartExperiment,
        ReplayLastRun,
        OpenSampleResult,
        ShowShortcuts,
        OpenRepository,
    ]
);

/// Where the project lives; the Help menu opens it.
const REPOSITORY_URL: &str = "https://github.com/voidwest/ember";

mod chart;
mod components;
mod icons;
mod input;
mod palette;
mod picker;
mod theme;

use components::*;
use input::{InputEvent, InputId, InputKind, TextInput};
use palette::Command;
use theme::{AppearanceMode, Colors, Radius, Space, Type};

// ---------------------------------------------------------------------------
// embedded fonts (SIL OFL 1.1, see src/gui_fonts/LICENSE.txt)
// ---------------------------------------------------------------------------

const FONT_SANS: &[u8] = include_bytes!("gui_fonts/NotoSans-Regular.ttf");
const FONT_MONO: &[u8] = include_bytes!("gui_fonts/NotoSansMono-Regular.ttf");
const FONT_ARABIC: &[u8] = include_bytes!("gui_fonts/NotoNaskhArabic-Regular.ttf");
const FONT_SANS_NAME: &str = "Noto Sans";
const FONT_MONO_NAME: &str = "Noto Sans Mono";
/// The bundled Noto Naskh renders with its dots (nuqta) detached, floating a
/// full em above the letters and clipped at the top of every text box, under
/// this text stack -- found in the rendered artifacts, and reproduced at every
/// size and line height. macOS ships Geeza Pro, which shapes the same prompt
/// correctly, so it leads there; the embedded face remains the offline
/// fallback on platforms without a system Arabic font.
#[cfg(target_os = "macos")]
const FONT_ARABIC_NAME: &str = "Geeza Pro";
#[cfg(not(target_os = "macos"))]
const FONT_ARABIC_NAME: &str = "Noto Naskh Arabic";

/// `ember gui` (native) CLI arguments.
#[derive(ClapArgs)]
pub(crate) struct NativeGuiArgs {
    /// Render deterministic visual-test artifacts without opening a desktop window.
    #[cfg(all(target_os = "macos", feature = "gui-tests"))]
    #[arg(long, hide = true)]
    render_test_dir: Option<std::path::PathBuf>,
}

/// The v0.4 hook stage ids (from Ember's own hook definitions), in order.
const STAGES: [&str; 6] = [
    "before-layer",
    "after-attention",
    "after-mlp",
    "after-layer",
    "before-logits",
    "after-logits",
];
const PER_LAYER_STAGES: [&str; 4] = [
    "before-layer",
    "after-attention",
    "after-mlp",
    "after-layer",
];
const EXECUTIONS: [&str; 3] = ["reference", "planned", "planned-fused"];

fn per_layer(site: &str) -> bool {
    PER_LAYER_STAGES.contains(&site)
}

fn operation_label(operation: &str) -> &'static str {
    match operation {
        "zero" => "Remove information",
        "scale" => "Change strength",
        "replace" => "Copy from another layer",
        "interpolate" => "Blend representations",
        "add-delta" => "Add a learned difference",
        _ => "Custom intervention",
    }
}

fn operation_hint(operation: &str) -> &'static str {
    match operation {
        "zero" => "Set the selected representation to zero",
        "scale" => "Make a representation weaker or stronger",
        "replace" => "Substitute a representation captured earlier",
        "interpolate" => "Mix the current and captured representations",
        "add-delta" => "Apply the difference from a captured layer",
        _ => "Configure an exact internal change",
    }
}

/// What the chosen change does, in a sentence a first-year student can act on.
/// The hint on the card says what the option is; this says what will happen.
fn operation_explainer(operation: &str) -> &'static str {
    match operation {
        "zero" => "Erases the model's activation at this point, as if that part of the computation had produced nothing. If the answer breaks, that part mattered.",
        "scale" => "Multiplies the activation by the strength. 1.0 changes nothing, 0.5 halves it, 0 removes it, and above 1 amplifies it.",
        "replace" => "Swaps the activation for one the model produced at another layer, to test whether that information is interchangeable.",
        "interpolate" => "Blends the current activation with a captured one. 0 keeps the original, 1 replaces it completely.",
        "add-delta" => "Adds the difference between two captured activations, nudging the model in the direction that difference points.",
        _ => "Configure an exact internal change.",
    }
}

/// Plain-language name for a hook site.
///
/// The exact contract names stay available in Advanced controls, but the
/// default surface should read as an outcome rather than an implementation
/// detail: "After feed-forward processing" describes a stage of the graph, not
/// something an operator reasons about.
fn site_label(site: &str) -> &'static str {
    match site {
        "before-layer" => "Layer input",
        "after-attention" => "After attention",
        "after-mlp" => "After MLP block",
        "after-layer" => "Layer output",
        "before-logits" => "Before output head",
        "after-logits" => "After output head",
        _ => "Custom location",
    }
}

/// The exact `ember.hook.v1` identifier, for Advanced controls.
fn site_contract_name(site: &str) -> &'static str {
    match site {
        "before-layer" => "before-layer",
        "after-attention" => "after-attention",
        "after-mlp" => "after-mlp",
        "after-layer" => "after-layer",
        "before-logits" => "before-logits",
        "after-logits" => "after-logits",
        _ => "custom",
    }
}

fn token_label(token: &str) -> &'static str {
    match token {
        "prompt-final" => "the final prompt token",
        "matched-span" => "a matching phrase",
        _ => "the selected tokens",
    }
}

fn combo_value_label(combo: ComboId, value: &str) -> String {
    match combo {
        ComboId::Model => model_display_name(value),
        ComboId::Site => site_label(value).to_string(),
        ComboId::Op => operation_label(value).to_string(),
        ComboId::Source => match value {
            "capture" => "Capture from another layer".to_string(),
            "zero" => "Use a zero representation".to_string(),
            _ => value.to_string(),
        },
        ComboId::Token => token_label(value).to_string(),
        ComboId::Execution => match value {
            "reference" => "Reference (most inspectable)".to_string(),
            "planned" => "Planned".to_string(),
            "planned-fused" => "Planned + fused".to_string(),
            _ => value.to_string(),
        },
    }
}

/// Filename stem for a model path, for display chips.
fn model_display_name(path: &str) -> String {
    path.rsplit(['/', '\\'])
        .next()
        .unwrap_or(path)
        .trim_end_matches(".gguf")
        .to_string()
}

fn truncate_chars(text: &str, limit: usize) -> String {
    let mut chars = text.chars();
    let excerpt: String = chars.by_ref().take(limit).collect();
    if chars.next().is_some() {
        format!("{excerpt}…")
    } else {
        excerpt
    }
}

// ---------------------------------------------------------------------------
// worker: owns the resident model session, runs experiments off the UI thread
// ---------------------------------------------------------------------------

enum WorkerMsg {
    Prepare(String),
    Run(RunConfig),
    Restore(RunConfig),
}

#[derive(Debug, Clone)]
enum WorkerReply {
    Prepared(Box<Result<SessionInfo, String>>),
    RunDone(Box<Result<RunBundle, String>>),
    RestoreDone(Box<Result<RestoreBundle, String>>),
}

fn spawn_worker(
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> (
    mpsc::Sender<WorkerMsg>,
    Arc<Mutex<mpsc::Receiver<WorkerReply>>>,
) {
    let (tx, rx) = mpsc::channel();
    let (reply_tx, reply_rx) = mpsc::channel();
    let reply_rx = Arc::new(Mutex::new(reply_rx));
    std::thread::spawn(move || {
        let mut session = crate::gui::GuiSession::new(k_strategy, k_allow_fallback);
        while let Ok(msg) = rx.recv() {
            match msg {
                WorkerMsg::Prepare(path) => {
                    let result = session.ensure_prepared(&path).and_then(|_| {
                        session
                            .info()
                            .ok_or_else(|| "model session is not prepared".to_string())
                    });
                    let _ = reply_tx.send(WorkerReply::Prepared(Box::new(result)));
                }
                WorkerMsg::Run(cfg) => {
                    let _ = reply_tx.send(WorkerReply::RunDone(Box::new(
                        session.run_baseline_intervention(&cfg),
                    )));
                }
                WorkerMsg::Restore(cfg) => {
                    let _ = reply_tx.send(WorkerReply::RestoreDone(Box::new(
                        session.run_restore_leg(&cfg),
                    )));
                }
            }
        }
    });
    (tx, reply_rx)
}

// ---------------------------------------------------------------------------
// app state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Idle,
    Preparing,
    Running,
    Restoring,
}

#[derive(Debug, Clone)]
struct VerificationView {
    ok: bool,
    checks: Vec<(String, bool, String)>,
    warnings: Vec<String>,
}

impl VerificationView {
    fn from_report(report: &ember::v05::verify::VerificationReport) -> Self {
        VerificationView {
            ok: report.ok,
            checks: report
                .checks
                .iter()
                .map(|check| (check.name.clone(), check.ok, check.detail.clone()))
                .collect(),
            warnings: report.warnings.clone(),
        }
    }
    fn failed(&self) -> Vec<&(String, bool, String)> {
        self.checks.iter().filter(|(_, ok, _)| !ok).collect()
    }
}

#[derive(Debug, Clone)]
struct RestoreView {
    matches: bool,
    comparable: bool,
}

/// Which dropdown (picker) is currently open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComboId {
    Model,
    Site,
    Op,
    Source,
    Token,
    Execution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkspaceStep {
    Prompt,
    Intervention,
    Review,
}

/// Top-level destinations in the app shell.
///
/// The console used to be one screen: you landed inside a half-configured
/// experiment whether or not that was what you came for. These are ordinary
/// product destinations, and the experiment is one of them rather than the
/// whole application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    Home,
    Experiment,
    Models,
    Runs,
    Settings,
}

impl View {
    /// Stable identity key. Never derive an `ElementId` from `label()`: the
    /// label is display text and is expected to change (or be translated),
    /// while the id is behavior, so a label change would silently reset the
    /// control's state.
    fn key(self) -> &'static str {
        match self {
            View::Home => "home",
            View::Experiment => "experiments",
            View::Models => "models",
            View::Runs => "runs",
            View::Settings => "settings",
        }
    }

    fn label(self) -> &'static str {
        match self {
            View::Home => "Home",
            View::Experiment => "Experiments",
            View::Models => "Models",
            View::Runs => "Runs",
            View::Settings => "Settings",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            View::Home => "Recent runs and starting points",
            View::Experiment => "Configure and run an experiment",
            View::Models => "Local models and their state",
            View::Runs => "Every run from this session",
            View::Settings => "Appearance and defaults",
        }
    }

    const ALL: [View; 5] = [
        View::Home,
        View::Experiment,
        View::Models,
        View::Runs,
        View::Settings,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResultView {
    Overview,
    Layers,
    Tokens,
    Trace,
}

impl ResultView {
    const ALL: [Self; 4] = [Self::Overview, Self::Layers, Self::Tokens, Self::Trace];

    /// Stable identity key for the result tabs. As with [`View::key`], the
    /// element id is behaviour and the label is display text, so they are kept
    /// separate -- "Raw trace" is a label that can be reworded without
    /// renaming the tab's address.
    fn key(self) -> &'static str {
        match self {
            Self::Overview => "overview",
            Self::Layers => "layers",
            Self::Tokens => "tokens",
            Self::Trace => "trace",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Layers => "Layers",
            Self::Tokens => "Tokens",
            Self::Trace => "Raw trace",
        }
    }
}

impl WorkspaceStep {
    const ALL: [Self; 3] = [Self::Prompt, Self::Intervention, Self::Review];

    /// Stable identity key for the stepper. `number()` is display text.
    fn key(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::Intervention => "intervention",
            Self::Review => "results",
        }
    }

    fn number(self) -> &'static str {
        match self {
            Self::Prompt => "1",
            Self::Intervention => "2",
            Self::Review => "3",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Prompt => "Prompt",
            Self::Intervention => "Intervention",
            Self::Review => "Review & results",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Self::Prompt => "Model and prompt",
            Self::Intervention => "Internal change",
            Self::Review => "Evidence and results",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Preset {
    SilenceEarly,
    ZeroMiddle,
    ScaleLate,
    CopyEarlier,
    ArabicMorphology,
}

/// Seconds since the epoch. Behind a function so a test can pin "now" and
/// assert on ordering without sleeping.
fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

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
struct RunsDelegate {
    rows: Vec<RunRecord>,
    colors: Colors,
    /// The owning console, so row actions can mutate the store they
    /// snapshot. Weak: the table must not keep the console alive.
    console: Option<WeakEntity<Console>>,
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
    fn new(rows: Vec<RunRecord>, colors: Colors) -> Self {
        Self {
            rows,
            colors,
            console: None,
        }
    }

    /// Replace the snapshot and repaint.
    ///
    /// `TableState::refresh` is a method on the state, and a delegate has no
    /// handle on the state that owns it, so the signal has to be the context.
    ///
    /// Colours come in here too, not just rows. The delegate is built once and
    /// kept, so a palette captured at construction would survive every theme
    /// change afterwards -- which showed up as a table that stayed light while
    /// the rest of the app went dark, and then as text too dim to read.
    fn sync(
        &mut self,
        rows: Vec<RunRecord>,
        colors: Colors,
        console: WeakEntity<Console>,
        cx: &mut Context<TableState<Self>>,
    ) {
        self.rows = rows;
        self.colors = colors;
        self.console = Some(console);
        cx.notify();
    }

    /// The text for one cell. Kept beside `render_td` so sorting and rendering
    /// cannot disagree about what a cell says.
    fn cell_text(&self, row_ix: usize, col_ix: usize) -> String {
        let Some(run) = self.rows.get(row_ix) else {
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
            4 => Column::new(run_col::RESULT, "Result").width(px(96.0)),
            5 => Column::new(run_col::DURATION, "Duration").width(px(96.0)),
            6 => Column::new(run_col::WHEN, "When").width(px(80.0)),
            // Wide enough for Reuse + Pin + Delete, the fullest lane a row
            // can carry.
            _ => Column::new(run_col::ACTIONS, "").width(px(264.0)),
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
        let descending = matches!(sort, ColumnSort::Descending);
        self.rows.sort_by(|left, right| {
            let ordering = match col_ix {
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
            };
            if descending {
                ordering.reverse()
            } else {
                ordering
            }
        });
    }

    fn render_td(
        &mut self,
        row_ix: usize,
        col_ix: usize,
        _window: &mut Window,
        cx: &mut Context<TableState<Self>>,
    ) -> impl IntoElement {
        let colors = self.colors;
        // The action lane: pin and delete, as quiet text commands. They talk
        // to the console through the weak entity the snapshot came from, so
        // the store, the snapshot and the screen stay one system.
        if col_ix == 7 {
            let Some(run) = self.rows.get(row_ix) else {
                return div().into_any_element();
            };
            let number = run.number;
            let pinned = run.pinned;
            // Reuse appears only where there is a stored configuration to
            // load; older records keep Pin and Delete and nothing that would
            // pretend to work.
            let reuse = run
                .config
                .clone()
                .map(|config| (config, run.prompt.clone(), self.console.clone()));
            let console = self.console.clone();
            let mut lane = div().flex().flex_row().gap(px(Space::SM));
            // Open appears only where the run kept its result.
            if run.result.is_some() && run.config.is_some() {
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
            }
            if let Some((config, prompt, console)) = reuse {
                lane = lane.child(
                    Button::new(SharedString::from(format!("run-reuse:{number}")))
                        .ghost()
                        .compact()
                        .label("Reuse")
                        .tooltip("Load this run's configuration into a new experiment")
                        .on_click(move |_, _, cx| {
                            if let Some(console) = console.as_ref() {
                                let _ = console.update(cx, |console, cx| {
                                    console.reuse_record(&config, &prompt, cx);
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
                        .label(if pinned { "Unpin" } else { "Pin" })
                        .tooltip(if pinned {
                            "Unpin this run"
                        } else {
                            "Keep this run at the top of the history"
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
                .child({
                    let console = self.console.clone();
                    Button::new(SharedString::from(format!("run-delete:{number}")))
                        .ghost()
                        .compact()
                        .label("Delete")
                        .text_color(colors.err)
                        .tooltip("Delete this run from the history")
                        .on_click(move |_, _, cx| {
                            if let Some(console) = console.as_ref() {
                                let _ = console.update(cx, |console, cx| {
                                    console.store.remove_run(number);
                                    console.persist();
                                    cx.notify();
                                });
                            }
                        })
                })
                .into_any_element();
        }
        let text = self.cell_text(row_ix, col_ix);
        let failed = self
            .rows
            .get(row_ix)
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
fn result_word(run: &RunRecord) -> &'static str {
    if !run.verified {
        "failed"
    } else if run.outputs_equal {
        "unchanged"
    } else {
        "changed"
    }
}

fn token_cell(run: &RunRecord) -> String {
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
fn quant_of(path: &str) -> String {
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
fn fmt_bytes(bytes: u64) -> String {
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
fn truncate_path_start(path: &str, max: usize) -> String {
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
fn relative_time(at: i64) -> String {
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

#[derive(Clone)]
struct Inputs {
    model: Entity<TextInput>,
    layer: Entity<TextInput>,
    value: Entity<TextInput>,
    source_layer: Entity<TextInput>,
    span: Entity<TextInput>,
    max_tokens: Entity<TextInput>,
    prompt: Entity<TextInput>,
}

impl Inputs {
    fn all(&self) -> [Entity<TextInput>; 7] {
        [
            self.model.clone(),
            self.layer.clone(),
            self.value.clone(),
            self.source_layer.clone(),
            self.span.clone(),
            self.max_tokens.clone(),
            self.prompt.clone(),
        ]
    }
}

#[derive(Clone, PartialEq)]
struct FormValues {
    model_path: String,
    prompt: String,
    max_tokens: String,
    execution: String,
    site: String,
    layer: String,
    op: String,
    value: String,
    source: String,
    source_layer: String,
    token: String,
    span: String,
}

impl FormValues {
    fn build_run_request(&self) -> Result<RunRequest, String> {
        let layer = if per_layer(&self.site) {
            Some(
                self.layer
                    .parse::<usize>()
                    .map_err(|_| "layer must be an integer".to_string())?,
            )
        } else {
            None
        };
        let source_layer = if per_layer(&self.site) && self.source == "capture" {
            let target = layer.expect("layer checked above");
            Some(
                self.source_layer
                    .parse::<usize>()
                    .map_err(|_| "source layer must be an integer".to_string())?
                    .min(target.saturating_sub(1)),
            )
        } else {
            None
        };
        Ok(RunRequest {
            model_path: self.model_path.trim().to_string(),
            prompt: self.prompt.clone(),
            max_new_tokens: self
                .max_tokens
                .parse::<usize>()
                .map_err(|_| "max new tokens must be an integer".to_string())?,
            execution: self.execution.clone(),
            site: self.site.clone(),
            layer,
            operation: self.op.clone(),
            factor: if self.op == "scale" {
                Some(
                    self.value
                        .parse::<f32>()
                        .map_err(|_| "scale factor must be a number".to_string())?,
                )
            } else {
                None
            },
            alpha: if self.op == "interpolate" {
                Some(
                    self.value
                        .parse::<f32>()
                        .map_err(|_| "interpolate alpha must be a number".to_string())?,
                )
            } else {
                None
            },
            source: self.source.clone(),
            source_layer,
            token_kind: self.token.clone(),
            span_text: (self.token == "matched-span").then(|| self.span.clone()),
        })
    }
}

struct Console {
    // worker
    worker_tx: mpsc::Sender<WorkerMsg>,
    reply_rx: Arc<Mutex<mpsc::Receiver<WorkerReply>>>,
    // model
    /// Lazily created: the table needs a `Window` to build, which the
    /// constructor does not have. Kept out of `Console::new` for that reason.
    runs_table: Option<Entity<TableState<RunsDelegate>>>,
    model_options: Vec<String>,
    model_path: String,
    /// The model row highlighted on the Models page, if any.
    selected_model: Option<String>,
    // form
    site_options: Vec<String>,
    site: String,
    layer: String,
    op: String,
    value: String,
    source_options: Vec<String>,
    source: String,
    source_layer: String,
    token_options: Vec<String>,
    token: String,
    span: String,
    max_tokens: String,
    execution_options: Vec<String>,
    execution: String,
    prompt: String,
    pickers: Vec<(ComboId, Entity<picker::Picker>)>,
    focus_handle: FocusHandle,
    inputs: Inputs,
    // command palette
    palette_open: bool,
    palette_query: String,
    palette_index: usize,
    palette_input: Entity<TextInput>,
    step: WorkspaceStep,
    view: View,
    inspector_open: bool,
    /// Whether the window is wide enough to show the inspector this frame.
    inspector_fits: bool,
    /// The Review page is showing the built-in illustrative sample, not a run.
    sample: bool,
    /// Which saved run the Review page is showing, when it is one from
    /// History rather than a live result.
    saved_run: Option<u64>,
    /// The result summary was just copied; shown on the button until the
    /// user moves on.
    copied: bool,
    sidebar_open: bool,
    /// Presentation mode, and the panes it hid so leaving restores them.
    presentation: Option<(bool, bool)>,
    advanced_open: bool,
    pending_run: bool,
    pending_context: Option<FormValues>,
    result_context: Option<FormValues>,
    store: AppStore,
    /// Read failure, surfaced rather than swallowed: a store we could not
    /// parse is not the same as a store with no runs.
    store_error: Option<String>,
    run_sequence: u64,
    // theme
    appearance: AppearanceMode,
    system_dark: bool,
    // session + results
    session: Option<SessionInfo>,
    status: Status,
    error: Option<String>,
    baseline: Option<RunOutput>,
    intervention: Option<RunOutput>,
    comparison: Option<ExperimentComparison>,
    layer_series: Arc<[crate::gui::LayerMetric]>,
    result_view: ResultView,
    hovered_layer: Option<usize>,
    selected_layer: Option<usize>,
    verification: Option<VerificationView>,
    restore: Option<RestoreView>,
    last_config: Option<String>,
    last_metrics: Option<(String, f64, Option<f64>)>,
}

impl Console {
    fn new(
        worker_tx: mpsc::Sender<WorkerMsg>,
        reply_rx: Arc<Mutex<mpsc::Receiver<WorkerReply>>>,
        system_dark: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let models = discover_models();
        let model_path = models.first().cloned().unwrap_or_default();
        let layer = "8".to_string();
        let value = "0.5".to_string();
        let source_layer = "0".to_string();
        let span = String::new();
        let max_tokens = "48".to_string();
        let prompt = "\u{627}\u{643}\u{62A}\u{628} \u{62C}\u{645}\u{644}\u{629} \
                      \u{642}\u{635}\u{64A}\u{631}\u{629} \u{639}\u{646} \u{627}\u{644}\u{645}\u{62F}\u{64A}\u{646}\u{629} \
                      \u{627}\u{644}\u{645}\u{646}\u{648}\u{631}\u{629}"
            .to_string();
        let appearance = AppearanceMode::load();
        // A store we cannot parse is reported, not treated as empty: silently
        // showing "no runs" for a damaged file would look like the user never
        // ran anything, and the next write would destroy their history.
        let (store, store_error) = if cfg!(feature = "gui-tests") && seed_runs_requested() {
            (seed_store(), None)
        } else {
            match app_store::load(app_store::store_path()) {
                Ok(store) => (store, None),
                Err(error) => (AppStore::default(), Some(error.to_string())),
            }
        };
        let run_sequence = store.runs.iter().map(|run| run.number).max().unwrap_or(0);
        let colors = if appearance.is_dark(system_dark) {
            theme::dark()
        } else {
            theme::light()
        };
        let inputs = Inputs {
            model: cx.new(|cx| {
                TextInput::new(
                    InputId::ModelPath,
                    InputKind::Text,
                    model_path.clone(),
                    "path to model.gguf",
                    &colors,
                    window,
                    cx,
                )
            }),
            layer: cx.new(|cx| {
                TextInput::new(
                    InputId::Layer,
                    InputKind::Integer,
                    layer.clone(),
                    "0",
                    &colors,
                    window,
                    cx,
                )
            }),
            value: cx.new(|cx| {
                TextInput::new(
                    InputId::Value,
                    InputKind::Decimal,
                    value.clone(),
                    "0.5",
                    &colors,
                    window,
                    cx,
                )
            }),
            source_layer: cx.new(|cx| {
                TextInput::new(
                    InputId::SourceLayer,
                    InputKind::Integer,
                    source_layer.clone(),
                    "0",
                    &colors,
                    window,
                    cx,
                )
            }),
            span: cx.new(|cx| {
                TextInput::new(
                    InputId::Span,
                    InputKind::Text,
                    span.clone(),
                    "كلمة في النص",
                    &colors,
                    window,
                    cx,
                )
            }),
            max_tokens: cx.new(|cx| {
                TextInput::new(
                    InputId::MaxTokens,
                    InputKind::Integer,
                    max_tokens.clone(),
                    "48",
                    &colors,
                    window,
                    cx,
                )
            }),
            prompt: cx.new(|cx| {
                TextInput::new(
                    InputId::Prompt,
                    InputKind::Multiline,
                    prompt.clone(),
                    "Enter a prompt…",
                    &colors,
                    window,
                    cx,
                )
            }),
        };
        for input in inputs.all() {
            cx.subscribe(&input, |console, _input, event: &InputEvent, cx| {
                console.input_changed(event, cx);
            })
            .detach();
        }
        let palette_input = cx.new(|cx| {
            TextInput::new(
                InputId::PaletteQuery,
                InputKind::Text,
                String::new(),
                "Type a command…",
                &colors,
                window,
                cx,
            )
        });
        cx.subscribe(&palette_input, |console, _input, event: &InputEvent, cx| {
            console.input_changed(event, cx);
        })
        .detach();
        let pickers = [
            ComboId::Model,
            ComboId::Site,
            ComboId::Op,
            ComboId::Source,
            ComboId::Token,
            ComboId::Execution,
        ]
        .into_iter()
        .map(|combo| {
            let options = match combo {
                ComboId::Model => models.clone(),
                ComboId::Site => STAGES.iter().map(|s| s.to_string()).collect(),
                ComboId::Op => ["replace", "zero", "scale", "interpolate", "add-delta"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                ComboId::Source => vec!["capture".into(), "zero".into()],
                ComboId::Token => vec!["prompt-final".into(), "matched-span".into()],
                ComboId::Execution => EXECUTIONS.iter().map(|s| s.to_string()).collect(),
            };
            let picker = cx.new(|cx| picker::Picker::new(combo, options, window, cx));
            cx.subscribe(&picker, move |console, _, event: &picker::Picked, cx| {
                console.select_combo(combo, &event.0, cx);
            })
            .detach();
            (combo, picker)
        })
        .collect();
        Console {
            worker_tx,
            reply_rx,
            runs_table: None,
            model_options: models,
            model_path,
            selected_model: None,
            site_options: STAGES.iter().map(|s| s.to_string()).collect(),
            site: "after-mlp".to_string(),
            layer,
            op: "scale".to_string(),
            value,
            source_options: vec!["capture".to_string(), "zero".to_string()],
            source: "capture".to_string(),
            source_layer,
            token_options: vec!["prompt-final".to_string(), "matched-span".to_string()],
            token: "prompt-final".to_string(),
            span,
            max_tokens,
            execution_options: EXECUTIONS.iter().map(|s| s.to_string()).collect(),
            execution: "reference".to_string(),
            prompt,
            pickers,
            focus_handle: cx.focus_handle(),
            inputs,
            palette_open: false,
            palette_query: String::new(),
            palette_index: 0,
            palette_input,
            step: WorkspaceStep::Prompt,
            // Land on Home rather than inside a half-configured experiment.
            view: View::Home,
            // The context rail opens on demand; it used to be permanent and
            // permanently half-empty on the setup steps. Both panes remember
            // their last state across launches -- under gui-tests the files
            // are ignored so the fixtures stay deterministic.
            inspector_fits: true,
            sample: false,
            copied: false,
            saved_run: None,
            inspector_open: if cfg!(feature = "gui-tests") {
                false
            } else {
                theme::load_flag("inspector").unwrap_or(false)
            },
            sidebar_open: if cfg!(feature = "gui-tests") {
                true
            } else {
                theme::load_flag("sidebar").unwrap_or(true)
            },
            presentation: None,
            advanced_open: false,
            pending_run: false,
            pending_context: None,
            result_context: None,
            store,
            store_error,
            // Continue from the history on disk. Restarting at 0 gave every
            // launch's first run the number 1 again, and Open, Pin and Delete
            // address a run by number -- so they hit the wrong rows.
            run_sequence,
            appearance,
            system_dark,
            session: None,
            status: Status::Idle,
            error: None,
            baseline: None,
            intervention: None,
            comparison: None,
            layer_series: Arc::from([]),
            result_view: ResultView::Overview,
            hovered_layer: None,
            selected_layer: None,
            verification: None,
            restore: None,
            last_config: None,
            last_metrics: None,
        }
    }

    fn busy(&self) -> bool {
        self.status != Status::Idle
    }

    /// The active semantic palette, resolved from the persisted appearance mode.
    fn colors(&self) -> Colors {
        if self.appearance.is_dark(self.system_dark) {
            theme::dark()
        } else {
            theme::light()
        }
    }

    fn form_values(&self) -> FormValues {
        FormValues {
            model_path: self.model_path.clone(),
            prompt: self.prompt.clone(),
            max_tokens: self.max_tokens.clone(),
            execution: self.execution.clone(),
            site: self.site.clone(),
            layer: self.layer.clone(),
            op: self.op.clone(),
            value: self.value.clone(),
            source: self.source.clone(),
            source_layer: self.source_layer.clone(),
            token: self.token.clone(),
            span: self.span.clone(),
        }
    }

    /// Whether the primary workspace action may proceed.
    ///
    /// Single source of truth for the gate. It used to be a local in `render`
    /// that only the button consulted, while the Ctrl+Enter handler advanced
    /// the workflow with no gate at all -- so a disabled button could still be
    /// driven by the keyboard.
    fn action_enabled(&self) -> bool {
        !self.busy() && self.validation_error().is_none()
    }

    /// The primary action, shared by the button and the keyboard shortcut.
    ///
    /// Both paths must go through here so they cannot disagree about when the
    /// action is allowed.
    fn advance_or_run(&mut self) {
        if !self.action_enabled() {
            return;
        }
        match self.step {
            WorkspaceStep::Prompt => self.step = WorkspaceStep::Intervention,
            WorkspaceStep::Intervention => self.step = WorkspaceStep::Review,
            WorkspaceStep::Review => self.run(),
        }
    }

    fn validation_error(&self) -> Option<String> {
        self.build_run_request()
            .and_then(|request| parse_run_request(&request).map(|_| ()))
            .err()
    }

    fn visible_experiment_context(&self) -> FormValues {
        if self.status == Status::Running {
            self.pending_context
                .clone()
                .unwrap_or_else(|| self.form_values())
        } else if self.step == WorkspaceStep::Review && self.baseline.is_some() {
            self.result_context
                .clone()
                .unwrap_or_else(|| self.form_values())
        } else {
            self.form_values()
        }
    }

    fn pipeline_node(
        &self,
        colors: &Colors,
        id: &'static str,
        text: String,
        accent: bool,
        step: WorkspaceStep,
        cx: &mut Context<Self>,
    ) -> Button {
        // These are commands, not prose, so they use Button (never Link --
        // Link is for URLs and email). Quiet ghost buttons in mono: the
        // sentence carries the meaning, the hover state carries the
        // affordance, and the intervention is the one segment that keeps
        // colour because it is the thing under study.
        Button::new(SharedString::from(format!("pipeline:{id}")))
            .ghost()
            .small()
            .label(text.clone())
            .tooltip(format!("Go to {}", step.label()))
            .accessibility_label(format!("{}: go to {}", text, step.label()))
            .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                console.step = step;
                cx.notify();
            }))
            .when(accent, |button| {
                button.text_color(colors.accent).font_family(FONT_MONO_NAME)
            })
    }

    fn experiment_pipeline(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let context = self.visible_experiment_context();
        let site_short = match context.site.as_str() {
            "before-layer" => "pre",
            "after-attention" => "attn",
            "after-mlp" => "mlp",
            "after-layer" => "out",
            "before-logits" => "final norm",
            "after-logits" => "logits",
            _ => "site",
        };
        let target = if per_layer(&context.site) {
            format!("L{} {site_short}", context.layer)
        } else {
            site_short.to_string()
        };
        let operation = match context.op.as_str() {
            "scale" => format!("\u{d7}{}", context.value),
            "zero" => "zero".to_string(),
            "replace" => format!("copy L{}", context.source_layer),
            "interpolate" => format!("blend {}", context.value),
            "add-delta" => format!("\u{394} L{}", context.source_layer),
            _ => context.op.clone(),
        };
        let sep = || label("\u{00b7}", Type::LABEL, colors.border_strong).flex_none();

        // A summary of what the current form will do, not a diagram of it:
        // one scan-friendly line. Muted words, mono values, one accent value,
        // and the run parameters right-aligned so the line also answers
        // "how long, how deterministic".
        div()
            .w_full()
            .flex()
            .flex_row()
            .flex_wrap()
            .items_center()
            .gap(px(Space::SM))
            // The strip is the experiment, stated once: it sits on a surface
            // so it reads as the page's subject rather than stray metadata.
            .px(px(Space::MD))
            .py(px(Space::SM))
            .bg(colors.surface)
            .rounded(px(Radius::LG))
            .child(self.pipeline_node(
                colors,
                "baseline",
                "unchanged baseline".to_string(),
                false,
                WorkspaceStep::Review,
                cx,
            ))
            .child(label("\u{2192}", Type::LABEL, colors.border_strong).flex_none())
            .child(self.pipeline_node(
                colors,
                "target",
                target,
                false,
                WorkspaceStep::Intervention,
                cx,
            ))
            .child(self.pipeline_node(
                colors,
                "operation",
                operation,
                true,
                WorkspaceStep::Intervention,
                cx,
            ))
            .child(sep())
            .child(label(
                token_label(&context.token),
                Type::LABEL,
                colors.text_faint,
            ))
            .child(div().flex_1())
            .child(mono(
                format!(
                    "\u{2264}{} tokens \u{00b7} seed 0 \u{00b7} {}",
                    context.max_tokens,
                    if self.saved_run.is_some() {
                        "saved"
                    } else if self.sample {
                        "sample"
                    } else if self.baseline.is_some() && self.status == Status::Idle {
                        "measured"
                    } else {
                        "planned"
                    }
                ),
                Type::META,
                colors.text_faint,
            ))
    }

    /// Build the v0.5 request from the current form fields; the shared
    /// `parse_run_request` gate validates it exactly like the web console.
    fn build_run_request(&self) -> Result<RunRequest, String> {
        self.form_values().build_run_request()
    }

    fn send_run(&mut self, cfg: RunConfig) {
        self.sample = false;
        self.saved_run = None;
        self.status = Status::Running;
        self.step = WorkspaceStep::Review;
        self.pending_context = Some(self.form_values());
        self.error = None;
        let _ = self.worker_tx.send(WorkerMsg::Run(cfg));
    }

    fn send_restore(&mut self, cfg: RunConfig) {
        self.status = Status::Restoring;
        self.error = None;
        let _ = self.worker_tx.send(WorkerMsg::Restore(cfg));
    }

    /// Drain the worker reply channel; returns true when anything changed.
    fn drain_replies(&mut self, cx: &mut Context<Self>) -> bool {
        let replies: Vec<WorkerReply> = {
            let rx = self.reply_rx.lock().expect("reply receiver lock");
            std::iter::from_fn(|| rx.try_recv().ok()).collect()
        };
        if replies.is_empty() {
            return false;
        }
        for reply in replies {
            match reply {
                WorkerReply::Prepared(result) => match *result {
                    Ok(info) => {
                        self.session = Some(info);
                        let n = self.session.as_ref().map(|s| s.n_layers).unwrap_or(1);
                        let target = self
                            .layer
                            .parse::<usize>()
                            .unwrap_or(0)
                            .min(n.saturating_sub(1));
                        self.layer = target.to_string();
                        self.source_layer = self
                            .source_layer
                            .parse::<usize>()
                            .unwrap_or(0)
                            .min(target.saturating_sub(1))
                            .to_string();
                        self.inputs
                            .layer
                            .update(cx, |input, cx| input.set_value(self.layer.clone(), cx));
                        self.inputs.source_layer.update(cx, |input, cx| {
                            input.set_value(self.source_layer.clone(), cx)
                        });
                        self.status = Status::Idle;
                        if self.pending_run {
                            self.pending_run = false;
                            self.run();
                        }
                    }
                    Err(error) => {
                        self.pending_run = false;
                        self.error = Some(error);
                        self.status = Status::Idle;
                    }
                },
                WorkerReply::RunDone(result) => match *result {
                    Ok(bundle) => {
                        self.sample = false;
                        self.saved_run = None;
                        self.copied = false;
                        self.result_context = self.pending_context.take();
                        self.baseline = Some(bundle.baseline.clone());
                        self.intervention = Some(bundle.intervention.clone());
                        self.layer_series = Arc::from(bundle.comparison.layers.clone());
                        self.comparison = Some(bundle.comparison.clone());
                        self.result_view = ResultView::Overview;
                        self.hovered_layer = None;
                        self.selected_layer = bundle
                            .comparison
                            .layers
                            .iter()
                            .filter(|metric| metric.relative_l2_difference.is_some())
                            .max_by(|left, right| {
                                left.relative_l2_difference
                                    .unwrap_or(0.0)
                                    .total_cmp(&right.relative_l2_difference.unwrap_or(0.0))
                            })
                            .map(|metric| metric.layer);
                        self.verification =
                            Some(VerificationView::from_report(&bundle.verification));
                        self.restore = None;
                        self.last_config = Some(bundle.baseline_key.clone());
                        self.last_metrics = Some((
                            bundle.intervention.semantic_hash.clone(),
                            bundle.elapsed_ms_total,
                            bundle.intervention.decode_tps,
                        ));
                        self.run_sequence += 1;
                        let now = unix_now();
                        self.store.push_run(RunRecord {
                            number: self.run_sequence,
                            finished_at: now,
                            model: model_display_name(&self.model_path),
                            intervention: operation_label(&self.op).to_string(),
                            hook: site_label(&self.site).to_string(),
                            layer: per_layer(&self.site)
                                .then(|| self.layer.parse::<u32>().ok())
                                .flatten(),
                            duration_ms: Some(bundle.elapsed_ms_total.max(0.0).round() as u64),
                            baseline_tokens: Some(bundle.baseline.generated_tokens as u32),
                            intervention_tokens: Some(bundle.intervention.generated_tokens as u32),
                            diverged_at_step: bundle
                                .comparison
                                .first_token_divergence
                                .map(|step| step as u32),
                            outputs_equal: bundle.comparison.generated_text_equal,
                            verified: bundle.verification.ok,
                            pinned: false,
                            prompt: self.prompt.clone(),
                            config: Some(app_store::RecordConfig {
                                model_path: self.model_path.clone(),
                                execution: self.execution.clone(),
                                site: self.site.clone(),
                                layer: self.layer.clone(),
                                op: self.op.clone(),
                                value: self.value.clone(),
                                source: self.source.clone(),
                                source_layer: self.source_layer.clone(),
                                token: self.token.clone(),
                                span: self.span.clone(),
                                max_tokens: self.max_tokens.clone(),
                            }),
                            result: Some(record_result(&bundle)),
                        });
                        self.store.touch_model(&self.model_path, now);
                        // The state that produced this run is the resume point.
                        self.save_draft();
                        self.persist();
                        self.status = Status::Idle;
                    }
                    Err(error) => {
                        self.pending_context = None;
                        self.error = Some(error);
                        self.status = Status::Idle;
                    }
                },
                WorkerReply::RestoreDone(result) => match *result {
                    Ok(bundle) => {
                        self.restore = Some(RestoreView {
                            matches: bundle.matches_baseline,
                            comparable: bundle.baseline_comparable,
                        });
                        self.verification =
                            Some(VerificationView::from_report(&bundle.verification));
                        if self.last_metrics.is_some() {
                            self.last_metrics = Some((
                                bundle.output.semantic_hash.clone(),
                                bundle.output.wall_ms,
                                bundle.output.decode_tps,
                            ));
                        }
                        self.status = Status::Idle;
                    }
                    Err(error) => {
                        self.error = Some(error);
                        self.status = Status::Idle;
                    }
                },
            }
        }
        true
    }

    fn load(&mut self) {
        if self.busy() {
            return;
        }
        let path = self.model_path.trim().to_string();
        if path.is_empty() {
            self.error = Some("model path must not be empty".to_string());
            return;
        }
        self.status = Status::Preparing;
        self.pending_run = false;
        self.error = None;
        let _ = self.worker_tx.send(WorkerMsg::Prepare(path));
    }

    fn run(&mut self) {
        if self.busy() {
            return;
        }
        match self.build_run_request() {
            Ok(req) => match parse_run_request(&req) {
                Ok(_cfg) if self.session.is_none() => {
                    self.pending_run = true;
                    self.status = Status::Preparing;
                    self.error = None;
                    let _ = self
                        .worker_tx
                        .send(WorkerMsg::Prepare(self.model_path.trim().to_string()));
                }
                Ok(cfg) => self.send_run(cfg),
                Err(error) => self.error = Some(error),
            },
            Err(error) => self.error = Some(error),
        }
    }

    fn restore(&mut self) {
        if self.busy() {
            return;
        }
        if self.last_config.is_none() {
            self.error =
                Some("run an experiment first; restore verifies against its baseline".to_string());
            return;
        }
        match self.build_run_request() {
            Ok(mut req) => {
                req.operation = "restore-original".to_string();
                req.factor = None;
                req.alpha = None;
                req.source = "capture".to_string();
                req.source_layer = None;
                match parse_run_request(&req) {
                    Ok(cfg) => self.send_restore(cfg),
                    Err(error) => self.error = Some(error),
                }
            }
            Err(error) => self.error = Some(error),
        }
    }

    fn sync_kit_theme(&self, cx: &mut App) {
        use gpui_kit::component::{Theme, ThemeMode};
        Theme::change(
            if self.appearance.is_dark(self.system_dark) {
                ThemeMode::Dark
            } else {
                ThemeMode::Light
            },
            None,
            cx,
        );
        let colors = self.colors();
        let theme = Theme::global_mut(cx);
        theme.font_family = FONT_SANS_NAME.into();
        theme.font_size = px(theme::scaled(14.0));
        theme.mono_font_family = FONT_MONO_NAME.into();
        theme.mono_font_size = px(theme::scaled(13.0));
        // Kit-owned geometry and hairlines follow the console's scale, so an
        // Input, a Select or a table cell does not disagree with a hand-laid
        // surface beside it.
        theme.radius = px(Radius::MD);
        theme.radius_lg = px(Radius::LG);
        theme.colors.border = colors.border.into();
        // Ghost buttons, tabs, nav rows and tiles hover to the kit's accent
        // token. Left at its default the shift was a single shade -- present
        // but not noticeable -- so it is set from the console's own palette:
        // a clear step above the selected-row fill, still neutral (the orange
        // stays reserved for the intervention and the primary action).
        let hover: Hsla = if self.appearance.is_dark(self.system_dark) {
            rgb(0x34302c).into()
        } else {
            rgb(0xddd7cd).into()
        };
        theme.colors.accent = hover;
        theme.colors.accent_foreground = colors.text.into();
        theme.primary = colors.accent.into();
        theme.primary_hover = colors.accent.into();
        theme.primary_active = colors.accent.into();
        theme.primary_foreground = rgb(0xffffff).into();
        theme.ring = colors.accent.into();
        theme.button_primary = colors.accent.into();
        theme.button_primary_hover = colors.accent.into();
        theme.button_primary_active = colors.accent.into();
        theme.button_primary_foreground = rgb(0xffffff).into();
        theme.tokens.button_primary = Hsla::from(colors.accent).into();
        theme.tokens.button_primary_hover = Hsla::from(colors.accent).into();
        theme.tokens.button_primary_active = Hsla::from(colors.accent).into();
        // Ghost controls and selectable rows take their hover and selected
        // fills from these tokens: navigation, tabs and segmented choices all
        // land on the same neutral step, which is what makes selection read as
        // position rather than emphasis.
        theme.tokens.secondary_hover = Hsla::from(colors.surface_hover).into();
        theme.tokens.secondary_active = Hsla::from(colors.selection).into();
        theme.tokens.button_hover = Hsla::from(colors.surface_hover).into();
        theme.tokens.list_hover = Hsla::from(colors.surface_hover).into();
        theme.tokens.list_active = Hsla::from(colors.selection).into();
        Theme::sync_base(cx);
    }

    fn cycle_appearance(&mut self, cx: &mut Context<Self>) {
        self.appearance = self.appearance.next();
        self.appearance.persist();
        self.sync_kit_theme(cx);
        cx.notify();
    }

    fn toggle_inspector(&mut self, cx: &mut Context<Self>) {
        self.inspector_open = !self.inspector_open;
        // Presentation mode owns the panes temporarily; what the user chose
        // before it is what gets restored and remembered.
        if self.presentation.is_none() {
            theme::persist_flag("inspector", self.inspector_open);
        }
        cx.notify();
    }

    fn toggle_sidebar(&mut self, cx: &mut Context<Self>) {
        self.sidebar_open = !self.sidebar_open;
        if self.presentation.is_none() {
            theme::persist_flag("sidebar", self.sidebar_open);
        }
        cx.notify();
    }

    /// The finished comparison as Markdown, for a slide, a lab notebook or a
    /// message: what was asked, what was changed, what happened, and the
    /// per-layer numbers. Everything is read from what is on screen, so the
    /// copy can never disagree with the page.
    fn result_markdown(&self) -> Option<String> {
        let (baseline, intervention, comparison, context) = (
            self.baseline.as_ref()?,
            self.intervention.as_ref()?,
            self.comparison.as_ref()?,
            self.result_context.as_ref()?,
        );
        let mut out = String::new();
        out.push_str("# Ember experiment\n\n");
        if let Some(number) = self.saved_run {
            out.push_str(&format!("> Run #{number}, reopened from history.\n\n"));
        } else if self.sample {
            out.push_str("> Sample result: illustrative data, not a measurement.\n\n");
        }
        out.push_str(&format!("- **Model:** {}\n", model_display_name(&context.model_path)));
        out.push_str(&format!("- **Prompt:** {}\n", context.prompt.trim()));
        out.push_str(&format!(
            "- **Change:** {} at layer {} ({}), affecting {}\n",
            operation_label(&context.op),
            context.layer,
            site_label(&context.site),
            token_label(&context.token),
        ));
        out.push_str(&format!("- **Generation:** up to {} tokens, seed 0\n\n", context.max_tokens));
        out.push_str("## Result\n\n");
        out.push_str(&format!(
            "- **Text output:** {}\n",
            if comparison.generated_text_equal { "unchanged" } else { "changed" }
        ));
        if let Some(layer) = comparison.landmarks.first_layer_divergence {
            out.push_str(&format!("- **First internal divergence:** layer {layer}\n"));
        }
        if let (Some(value), Some(layer)) =
            (comparison.landmarks.peak_relative_l2, comparison.landmarks.peak_layer)
        {
            out.push_str(&format!("- **Peak divergence:** {value:.3} (relative L2) at layer {layer}\n"));
        }
        out.push_str(&format!("\n**Baseline:** {}\n\n", baseline.text.trim()));
        out.push_str(&format!("**Intervention:** {}\n", intervention.text.trim()));
        if !self.layer_series.is_empty() {
            out.push_str("\n## Divergence by layer\n\n| layer | relative L2 | cosine distance |\n|---|---|---|\n");
            for metric in self.layer_series.iter() {
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
        Some(out)
    }

    /// Open Review on the built-in sample: a finished comparison a newcomer
    /// (or a demo with no model on the machine) can read without running
    /// anything. It is labelled as illustrative on the page, and the first
    /// real run replaces it.
    fn show_sample(&mut self, cx: &mut Context<Self>) {
        let (baseline, intervention, comparison, mut values) = sample_result();
        values.model_path = self.model_path.clone();
        self.show_result(baseline, intervention, comparison, values, None, cx);
    }

    /// Reopen a run from History. Only records that kept their result can be
    /// opened; the Runs table shows Open on exactly those.
    fn open_run(&mut self, number: u64, cx: &mut Context<Self>) {
        let Some(record) = self.store.runs.iter().find(|run| run.number == number).cloned() else {
            return;
        };
        let (Some(result), Some(config)) = (record.result.clone(), record.config.clone()) else {
            return;
        };
        let output = |text: String, tokens: Option<u32>| RunOutput {
            text,
            generated_token_ids: (1..=tokens.unwrap_or(0)).collect(),
            generated_token_texts: Vec::new(),
            prompt_tokens: 0,
            generated_tokens: tokens.unwrap_or(0) as usize,
            bundle_dir: "history".to_string(),
            semantic_hash: "0000000000000000".to_string(),
            payload_hash: "00000000".to_string(),
            // History keeps one total for the pair, not per-side timings, so
            // none is shown rather than a made-up split (see output_panel).
            wall_ms: 0.0,
            decode_tps: None,
            events: Vec::new(),
        };
        let baseline = output(result.baseline_text.clone(), record.baseline_tokens);
        let intervention = output(result.intervention_text.clone(), record.intervention_tokens);
        let comparison = ExperimentComparison {
            layers: result
                .layers
                .iter()
                .map(|layer| crate::gui::LayerMetric {
                    layer: layer.layer,
                    relative_l2_difference: layer.relative_l2,
                    cosine_distance: layer.cosine,
                    maximum_absolute_difference: None,
                    exact: layer.relative_l2 == Some(0.0),
                })
                .collect(),
            tokens: result
                .tokens
                .iter()
                .map(|token| crate::gui::TokenMetric {
                    position: token.position,
                    baseline_token_id: None,
                    intervention_token_id: None,
                    baseline_text: token.baseline.clone(),
                    intervention_text: token.intervention.clone(),
                    differs: token.differs,
                })
                .collect(),
            first_token_divergence: record.diverged_at_step.map(|step| step as usize),
            generated_tokens_equal: result.tokens_equal,
            generated_text_equal: record.outputs_equal,
            landmarks: crate::gui::DivergenceLandmarks {
                first_layer_divergence: result.first_layer_divergence,
                peak_layer: result.peak_layer,
                peak_relative_l2: result.peak_relative_l2,
                stable_token_tail_step: None,
            },
            layer_token_grid: None,
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
        self.show_result(baseline, intervention, comparison, values, Some(number), cx);
    }

    /// Put a finished comparison on the Review page without running anything.
    /// Shared by the sample and by History; `saved_run` says which it is.
    fn show_result(
        &mut self,
        baseline: RunOutput,
        intervention: RunOutput,
        comparison: ExperimentComparison,
        values: FormValues,
        saved_run: Option<u64>,
        cx: &mut Context<Self>,
    ) {
        self.apply_form_values(values.clone(), cx);
        self.layer_series = Arc::from(comparison.layers.clone());
        let selected = comparison.landmarks.peak_layer.or(comparison.landmarks.first_layer_divergence);
        self.baseline = Some(baseline);
        self.intervention = Some(intervention);
        self.comparison = Some(comparison);
        self.result_context = Some(values);
        self.verification = None;
        self.restore = None;
        self.selected_layer = selected;
        self.result_view = ResultView::Overview;
        self.sample = true;
        self.saved_run = saved_run;
        self.copied = false;
        self.goto(View::Experiment, cx);
        self.step = WorkspaceStep::Review;
        cx.notify();
    }

    /// Presentation mode: a larger text scale with the sidebar and inspector
    /// out of the way. Nothing else changes, and leaving restores the panes
    /// exactly as they were. The panes are hidden without touching the
    /// persisted flags, so a crash mid-talk does not lose the workspace.
    fn toggle_presentation(&mut self, cx: &mut Context<Self>) {
        match self.presentation.take() {
            Some((sidebar, inspector)) => {
                self.sidebar_open = sidebar;
                self.inspector_open = inspector;
                theme::set_ui_scale(1.0);
            }
            None => {
                self.presentation = Some((self.sidebar_open, self.inspector_open));
                self.sidebar_open = false;
                self.inspector_open = false;
                theme::set_ui_scale(theme::PRESENTATION_SCALE);
            }
        }
        self.sync_kit_theme(cx);
        cx.notify();
    }

    // -- command palette -----------------------------------------------------

    /// Open or close the palette. Opening clears the previous query and puts
    /// focus in the field, so Cmd+K, type, Enter just works.
    fn toggle_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.palette_open = !self.palette_open;
        if self.palette_open {
            self.palette_query.clear();
            self.palette_index = 0;
            self.palette_input.update(cx, |input, cx| {
                input.set_value("", cx);
                input.focus(window, cx);
            });
        }
        cx.notify();
    }

    /// The palette's candidate list: the catalog filtered by the query, case
    /// and word order insensitive enough for one-screen use.
    fn palette_candidates(&self) -> Vec<Command> {
        let query = self.palette_query.trim().to_lowercase();
        Command::ALL
            .into_iter()
            .filter(|command| {
                query.is_empty()
                    || command.label().to_lowercase().contains(&query)
                    || command.hint().to_lowercase().contains(&query)
            })
            .collect()
    }

    fn palette_move(&mut self, delta: isize, cx: &mut Context<Self>) {
        let count = self.palette_candidates().len();
        if count == 0 {
            return;
        }
        let index = self.palette_index as isize + delta;
        self.palette_index = index.rem_euclid(count as isize) as usize;
        cx.notify();
    }

    fn palette_execute(&mut self, cx: &mut Context<Self>) {
        let candidates = self.palette_candidates();
        let Some(command) = candidates.get(self.palette_index).copied() else {
            return;
        };
        self.palette_open = false;
        match command {
            Command::NewExperiment => {
                self.goto(View::Experiment, cx);
                self.step = WorkspaceStep::Prompt;
            }
            Command::LoadModel => {
                self.goto(View::Experiment, cx);
                self.load();
            }
            Command::RunExperiment => {
                self.goto(View::Experiment, cx);
                self.step = WorkspaceStep::Review;
                self.run();
            }
            Command::DuplicateExperiment => self.duplicate_experiment(cx),
            Command::RerunExperiment => {
                if self.result_context.is_some() {
                    self.goto(View::Experiment, cx);
                    self.step = WorkspaceStep::Review;
                    self.rerun(cx);
                }
            }
            Command::GoHome => self.goto(View::Home, cx),
            Command::GoExperiments => self.goto(View::Experiment, cx),
            Command::GoModels => self.goto(View::Models, cx),
            Command::GoRuns => self.goto(View::Runs, cx),
            Command::GoSettings => self.goto(View::Settings, cx),
            Command::GoPrompt => {
                self.goto(View::Experiment, cx);
                self.step = WorkspaceStep::Prompt;
            }
            Command::GoIntervention => {
                self.goto(View::Experiment, cx);
                self.step = WorkspaceStep::Intervention;
            }
            Command::GoReview => {
                self.goto(View::Experiment, cx);
                self.step = WorkspaceStep::Review;
            }
            Command::EditIntervention => {
                self.goto(View::Experiment, cx);
                self.step = WorkspaceStep::Intervention;
            }
            Command::ToggleInspector => self.toggle_inspector(cx),
            Command::ToggleSidebar => self.toggle_sidebar(cx),
            Command::ToggleTheme => self.cycle_appearance(cx),
            Command::TogglePresentation => self.toggle_presentation(cx),
        }
        cx.notify();
    }

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

    fn system_appearance_changed(&mut self, dark: bool, cx: &mut Context<Self>) {
        if self.system_dark == dark {
            return;
        }
        self.system_dark = dark;
        if self.appearance == AppearanceMode::System {
            self.sync_kit_theme(cx);
            cx.notify();
        }
    }

    fn input_changed(&mut self, event: &InputEvent, cx: &mut Context<Self>) {
        match event.id {
            InputId::ModelPath => {
                if self.model_path != event.value {
                    self.session = None;
                }
                self.model_path.clone_from(&event.value);
            }
            InputId::Layer => {
                self.layer.clone_from(&event.value);
                self.clamp_source_layer(cx);
            }
            InputId::Value => self.value.clone_from(&event.value),
            InputId::SourceLayer => self.source_layer.clone_from(&event.value),
            InputId::Span => self.span.clone_from(&event.value),
            InputId::MaxTokens => self.max_tokens.clone_from(&event.value),
            InputId::Prompt => self.prompt.clone_from(&event.value),
            InputId::PaletteQuery => {
                // A new query means a new candidate list; the cursor restarts
                // so Enter always runs what the eye is resting on.
                self.palette_query.clone_from(&event.value);
                self.palette_index = 0;
            }
        }
        cx.notify();
    }

    fn select_combo(&mut self, combo: ComboId, value: &str, cx: &mut Context<Self>) {
        match combo {
            ComboId::Model => {
                self.model_path = value.to_string();
                self.session = None;
                self.inputs
                    .model
                    .update(cx, |input, cx| input.set_value(value.to_string(), cx));
            }
            ComboId::Site => self.site = value.to_string(),
            ComboId::Op => self.op = value.to_string(),
            ComboId::Source => self.source = value.to_string(),
            ComboId::Token => self.token = value.to_string(),
            ComboId::Execution => self.execution = value.to_string(),
        }
        // Selecting a non-per-layer site drops the layer fields.
        if combo == ComboId::Site && !per_layer(&self.site) {
            self.layer = "0".to_string();
            self.source_layer = "0".to_string();
            self.inputs
                .layer
                .update(cx, |input, cx| input.set_value("0", cx));
            self.inputs
                .source_layer
                .update(cx, |input, cx| input.set_value("0", cx));
        }
        cx.notify();
    }

    /// Keep the source layer at or above the target layer (the capture must
    /// fire before the intervention in the same pass).
    fn clamp_source_layer(&mut self, cx: &mut Context<Self>) {
        if let (Ok(target), Ok(source)) =
            (self.layer.parse::<i64>(), self.source_layer.parse::<i64>())
            && source > target
        {
            self.source_layer = (target - 1).max(0).to_string();
            self.inputs.source_layer.update(cx, |input, cx| {
                input.set_value(self.source_layer.clone(), cx)
            });
        }
    }

    fn set_input_value(&mut self, input: Entity<TextInput>, value: String, cx: &mut Context<Self>) {
        input.update(cx, |input, cx| input.set_value(value, cx));
    }

    fn set_max_tokens(&mut self, value: usize, cx: &mut Context<Self>) {
        self.max_tokens = value.to_string();
        self.set_input_value(self.inputs.max_tokens.clone(), self.max_tokens.clone(), cx);
        cx.notify();
    }

    fn adjust_layer(&mut self, delta: isize, cx: &mut Context<Self>) {
        let max = self
            .session
            .as_ref()
            .map(|session| session.n_layers.saturating_sub(1))
            .unwrap_or(63);
        let current = self.layer.parse::<usize>().unwrap_or(0);
        let next = current.saturating_add_signed(delta).min(max);
        self.layer = next.to_string();
        self.set_input_value(self.inputs.layer.clone(), self.layer.clone(), cx);
        self.clamp_source_layer(cx);
        cx.notify();
    }

    fn apply_preset(&mut self, preset: Preset, cx: &mut Context<Self>) {
        let layers = self.session.as_ref().map(|session| session.n_layers);
        match preset {
            Preset::ZeroMiddle => {
                self.op = "zero".to_string();
                self.site = "after-mlp".to_string();
                self.layer = layers.map_or(8, |count| count / 2).to_string();
            }
            Preset::SilenceEarly => {
                // Found by probing a real 16-layer model: silencing an early
                // layer's output at the last prompt token makes the answer
                // drift completely ("Paris..." becomes talk of fog and quiet
                // streets), where zeroing a middle MLP changes nothing at all.
                // The prompt is set too, so the example reads the same on a
                // machine whose form still holds the Arabic default.
                self.op = "scale".to_string();
                self.site = "after-layer".to_string();
                self.value = "0.0".to_string();
                self.token = "prompt-final".to_string();
                self.prompt = "The capital of France is".to_string();
                // Short: the effect shows within a couple of dozen tokens,
                // and a live demo should not wait on a long generation.
                self.max_tokens = "24".to_string();
                self.layer = layers.map_or(6, |count| count * 3 / 8).to_string();
            }
            Preset::ScaleLate => {
                self.op = "scale".to_string();
                self.site = "after-mlp".to_string();
                self.value = "0.5".to_string();
                self.layer = layers
                    .map_or(14, |count| count.saturating_sub(2))
                    .to_string();
            }
            Preset::CopyEarlier => {
                self.op = "replace".to_string();
                self.site = "after-layer".to_string();
                self.source = "capture".to_string();
                let target = layers.map_or(12, |count| count.saturating_sub(2));
                self.layer = target.to_string();
                self.source_layer = target.saturating_sub(4).to_string();
            }
            Preset::ArabicMorphology => {
                self.op = "scale".to_string();
                self.site = "after-mlp".to_string();
                self.value = "0.5".to_string();
                self.token = "matched-span".to_string();
                self.span = "المدينة".to_string();
                self.prompt = "اكتب جملة قصيرة عن المدينة المنورة".to_string();
                self.layer = layers.map_or(8, |count| count / 2).to_string();
            }
        }
        for (input, value) in [
            (self.inputs.layer.clone(), self.layer.clone()),
            (self.inputs.value.clone(), self.value.clone()),
            (self.inputs.source_layer.clone(), self.source_layer.clone()),
            (self.inputs.span.clone(), self.span.clone()),
            (self.inputs.prompt.clone(), self.prompt.clone()),
        ] {
            self.set_input_value(input, value, cx);
        }
        self.step = WorkspaceStep::Intervention;
        self.error = None;
        cx.notify();
    }

    /// Poll the worker reply channel every 80 ms while the app lives. The
    /// worker thread is unchanged from the iced implementation; only the
    /// foreground subscription is replaced by gpui's async executor.
    fn spawn_poll(&mut self, cx: &mut Context<Self>) {
        cx.spawn(|this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let mut cx = cx.clone();
            async move {
                loop {
                    let delay = this
                        .read_with(&cx, |console, _| if console.busy() { 50 } else { 250 })
                        .unwrap_or(250);
                    cx.background_executor()
                        .timer(Duration::from_millis(delay))
                        .await;
                    let _ = this.update(&mut cx, |console, cx| {
                        if console.drain_replies(cx) {
                            cx.notify();
                        }
                    });
                }
            }
        })
        .detach();
    }

    // -- view builders -------------------------------------------------------

    fn picker_key(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let key = event.keystroke.key.as_str();
        let cmd = event.keystroke.modifiers.control || event.keystroke.modifiers.platform;
        let shift = event.keystroke.modifiers.shift;

        // Shell shortcuts. Both panes are workspace state, so the toggles go
        // through the persisting methods rather than flipping the fields.
        if cmd && !shift && key == "b" {
            self.toggle_sidebar(cx);
            return;
        }
        if cmd && shift && key == "i" {
            self.toggle_inspector(cx);
            return;
        }
        if cmd && !shift && key == "n" {
            self.goto(View::Experiment, cx);
            self.step = WorkspaceStep::Prompt;
            cx.notify();
            return;
        }
        if cmd && !shift && key == "r" {
            if self.result_context.is_some() {
                self.goto(View::Experiment, cx);
                self.step = WorkspaceStep::Review;
                self.rerun(cx);
            }
            cx.notify();
            return;
        }

        // The palette is a modal: while it is open it owns the keyboard.
        if cmd && !shift && key == "k" {
            self.toggle_palette(window, cx);
            return;
        }
        if self.palette_open {
            match key {
                "escape" => {
                    self.palette_open = false;
                    cx.notify();
                }
                "up" => self.palette_move(-1, cx),
                "down" => self.palette_move(1, cx),
                "enter" | "return" if !cmd => self.palette_execute(cx),
                _ => {}
            }
            return;
        }

        if cmd {
            let result_view = match key {
                "1" => Some(ResultView::Overview),
                "2" => Some(ResultView::Layers),
                "3" => Some(ResultView::Tokens),
                "4" => Some(ResultView::Trace),
                _ => None,
            };
            // On a finished review the number keys address the result views;
            // anywhere else they are the three workflow steps, matching the
            // order the stepper shows.
            if let Some(view) = result_view
                && self.step == WorkspaceStep::Review
                && self.comparison.is_some()
            {
                self.result_view = view;
                cx.notify();
                return;
            }
            if !shift {
                let step = match key {
                    "1" => Some(WorkspaceStep::Prompt),
                    "2" => Some(WorkspaceStep::Intervention),
                    "3" => Some(WorkspaceStep::Review),
                    _ => None,
                };
                if let Some(step) = step {
                    self.step = step;
                    cx.notify();
                    return;
                }
            }
        }
        if matches!(key, "enter" | "return") && cmd {
            self.advance_or_run();
            cx.notify();
        }
    }

    fn picker(
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
                    .when(self.view == View::Experiment, |row| {
                        row.child(label("/", Type::LABEL, colors.border_strong))
                            .child(label(
                                truncate_chars(self.step.label(), 28),
                                Type::LABEL,
                                colors.text_muted,
                            ))
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
            .when(self.view == View::Experiment && self.inspector_fits, |bar| {
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
            })
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

    /// Write the store, surfacing a failure instead of dropping the record.
    ///
    /// A run that completed and was never written is the one loss this app
    /// cannot explain to the user, so the error is shown rather than swallowed.
    fn persist(&mut self) {
        // Unit tests drive the real console; they must never write the
        // developer's own history.
        if cfg!(test) {
            return;
        }
        if let Err(error) = self.store.write(app_store::store_path()) {
            self.store_error = Some(format!("could not save run history: {error}"));
        }
    }

    /// Navigate to a destination, checkpointing an in-progress experiment on
    /// the way out. One switch point, so every path out of the workspace
    /// saves the draft and none of them can forget to.
    fn goto(&mut self, view: View, cx: &mut Context<Self>) {
        if self.view == View::Experiment && view != View::Experiment {
            self.save_draft();
            self.persist();
        }
        self.view = view;
        cx.notify();
    }

    /// Snapshot the current form as the resume point.
    fn save_draft(&mut self) {
        // The sample rewrites the form; it must not become a draft to resume.
        if self.sample {
            return;
        }
        let values = self.form_values();
        // A form identical to the run that just finished has nothing left to
        // resume -- the run record already holds it, and Reuse reopens it.
        // Keeping a draft anyway made Home offer "Continue where you left
        // off" for an experiment the user had just completed.
        if self.result_context.as_ref() == Some(&values) {
            self.store.draft = None;
            return;
        }
        let revision = self
            .store
            .draft
            .as_ref()
            .map_or(0, |draft| draft.revision + 1);
        let fields = [
            ("max_tokens", values.max_tokens),
            ("execution", values.execution),
            ("site", values.site),
            ("layer", values.layer),
            ("op", values.op),
            ("value", values.value),
            ("source", values.source),
            ("source_layer", values.source_layer),
            ("token", values.token),
            ("span", values.span),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
        self.store.draft = Some(app_store::Draft {
            revision,
            prompt: values.prompt,
            model_path: values.model_path,
            fields,
            step: self.step.key().to_string(),
            updated_at: unix_now(),
        });
    }

    /// Restore the saved draft, if one exists. Form values ride the same
    /// `set_value` path as the pickers so the controls cannot disagree with
    /// the model behind them.
    fn restore_draft(&mut self, cx: &mut Context<Self>) {
        let Some(draft) = self.store.draft.clone() else {
            return;
        };
        let field = |name: &str| draft.fields.get(name).cloned().unwrap_or_default();
        self.apply_form_values(
            FormValues {
                model_path: draft.model_path,
                prompt: draft.prompt,
                max_tokens: field("max_tokens"),
                execution: field("execution"),
                site: field("site"),
                layer: field("layer"),
                op: field("op"),
                value: field("value"),
                source: field("source"),
                source_layer: field("source_layer"),
                token: field("token"),
                span: field("span"),
            },
            cx,
        );
        self.step = WorkspaceStep::ALL
            .iter()
            .find(|step| step.key() == draft.step)
            .copied()
            .unwrap_or(WorkspaceStep::Prompt);
        self.view = View::Experiment;
        cx.notify();
    }

    /// Put a captured form configuration back into the console. The resident
    /// session only resets when the model actually changed, so duplicating a
    /// run on the same model does not pay for a reload.
    fn apply_form_values(&mut self, values: FormValues, cx: &mut Context<Self>) {
        if values.model_path != self.model_path {
            self.session = None;
        }
        self.model_path = values.model_path;
        self.prompt = values.prompt;
        self.max_tokens = values.max_tokens;
        self.execution = values.execution;
        self.site = values.site;
        self.layer = values.layer;
        self.op = values.op;
        self.value = values.value;
        self.source = values.source;
        self.source_layer = values.source_layer;
        self.token = values.token;
        self.span = values.span;
        self.clamp_source_layer(cx);
        for (input, value) in [
            (self.inputs.model.clone(), self.model_path.clone()),
            (self.inputs.prompt.clone(), self.prompt.clone()),
            (self.inputs.max_tokens.clone(), self.max_tokens.clone()),
            (self.inputs.layer.clone(), self.layer.clone()),
            (self.inputs.value.clone(), self.value.clone()),
            (self.inputs.source_layer.clone(), self.source_layer.clone()),
            (self.inputs.span.clone(), self.span.clone()),
        ] {
            self.set_input_value(input, value, cx);
        }
    }

    /// Replay the completed experiment exactly: the form returns to the state
    /// that produced the last run, then the run starts. The bottom bar's
    /// "Run experiment again" runs the current form instead, which is the
    /// change-one-variable path.
    fn rerun(&mut self, cx: &mut Context<Self>) {
        if self.busy() {
            return;
        }
        // "Rerun" replays a completed run. With none, it must do nothing
        // rather than quietly start whatever happens to be in the form.
        let Some(context) = self.result_context.clone() else {
            return;
        };
        self.apply_form_values(context, cx);
        self.run();
    }

    /// Branch from the completed run: the form returns to the state that
    /// produced it, and the user changes one thing. The original run stays in
    /// the history; the next run appends a new record.
    fn duplicate_experiment(&mut self, cx: &mut Context<Self>) {
        let Some(context) = self.result_context.clone() else {
            return;
        };
        self.apply_form_values(context, cx);
        self.save_draft();
        self.goto(View::Experiment, cx);
        self.step = WorkspaceStep::Prompt;
        cx.notify();
    }

    /// The Runs-page path into the same loop: load a stored run's recorded
    /// configuration into the form. Records written before configurations
    /// were stored have nothing to load, so their rows do not offer it.
    fn reuse_record(
        &mut self,
        config: &app_store::RecordConfig,
        prompt: &str,
        cx: &mut Context<Self>,
    ) {
        self.apply_form_values(
            FormValues {
                model_path: config.model_path.clone(),
                prompt: prompt.to_string(),
                max_tokens: config.max_tokens.clone(),
                execution: config.execution.clone(),
                site: config.site.clone(),
                layer: config.layer.clone(),
                op: config.op.clone(),
                value: config.value.clone(),
                source: config.source.clone(),
                source_layer: config.source_layer.clone(),
                token: config.token.clone(),
                span: config.span.clone(),
            },
            cx,
        );
        self.save_draft();
        self.goto(View::Experiment, cx);
        self.step = WorkspaceStep::Prompt;
        cx.notify();
    }

    /// Switch whichever family of tabs `prefix` names. One decision point, so
    /// the two tab bars cannot drift apart the way two implementations did.
    fn select_tab(&mut self, prefix: &str, key: &str) {
        if prefix == "step" {
            if let Some(step) = WorkspaceStep::ALL.iter().find(|step| step.key() == key) {
                self.step = *step;
            }
        } else if let Some(view) = ResultView::ALL.iter().find(|view| view.key() == key) {
            self.result_view = *view;
        }
        self.copied = false;
    }

    /// Workflow steps: Prompt, Intervention, Review.
    fn stepper(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let tabs: Vec<(&'static str, &'static str, String)> = WorkspaceStep::ALL
            .iter()
            .map(|step| {
                (
                    step.key(),
                    step.label(),
                    format!(
                        "Step {} of 3: {}. {}",
                        step.number(),
                        step.label(),
                        step.hint()
                    ),
                )
            })
            .collect();
        let active = WorkspaceStep::ALL
            .iter()
            .position(|step| *step == self.step)
            .unwrap_or(0);
        div()
            .w_full()
            .px_5()
            .pt_4()
            .child(self.tab_row(colors, cx, "step", &tabs, active))
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

    /// Models: one row per discovered file, name first.
    ///
    /// This page was a single card whose entire content was the absolute path
    /// `/Users/west/Projects/ember/Llama-3.2-1B-Instruct-Q8_0.gguf`. The design
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
            let size = std::fs::metadata(path).map(|meta| meta.len()).ok();
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
    /// has to push new rows in rather than expect the table to notice.
    fn runs_view(&mut self, colors: &Colors, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let rows: Vec<RunRecord> = self.store.runs_ordered().into_iter().cloned().collect();
        let row_count = rows.len();
        // Row actions mutate the store through the owning console, so the
        // delegate needs a handle that does not keep the console alive.
        let console = cx.entity().downgrade();
        let table = match &self.runs_table {
            Some(state) => {
                let console = console.clone();
                state.update(cx, |state, cx| {
                    state.delegate_mut().sync(rows, *colors, console, cx);
                });
                state.clone()
            }
            None => {
                let state =
                    cx.new(|cx| TableState::new(RunsDelegate::new(rows, *colors), window, cx));
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
                    .child(
                        div().w(px(160.0)).child(btn_secondary(
                            colors,
                            "New experiment",
                            Some(cx.listener(|console, _: &ClickEvent, _window, cx| {
                                console.goto(View::Experiment, cx);
                                console.step = WorkspaceStep::Prompt;
                                cx.notify();
                            })),
                        )),
                    ),
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
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w(px(0.0))
                                        .overflow_hidden()
                                        .child(label(
                                            truncate_chars(&run.intervention, 22),
                                            Type::LABEL,
                                            colors.text,
                                        )),
                                )
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

    /// Starting points for a new experiment.
    ///
    /// These were in the left rail, which made them look like a mode switch.
    /// They are choices for starting work, so they belong on the Prompt step
    /// where the work starts.
    fn presets_block(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let presets = [
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
        ];
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
            grid = grid.child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(Space::SM))
                    .children(row),
            );
        }
        grid
    }

    fn preset_card(
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
    fn generation_control(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
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

    fn operation_card(
        &self,
        colors: &Colors,
        operation: &'static str,
        cx: &mut Context<Self>,
    ) -> Button {
        Button::new(SharedString::from(format!("operation-card:{operation}")))
            // flex_1, not half the row: two 50% cards plus the gap between
            // them were wider than the row and overhung its right edge.
            .flex_1()
            .min_w(px(0.0))
            .h_auto()
            .py(px(Space::MD))
            .px(px(Space::MD))
            .ghost()
            .selected(self.op == operation)
            // The chosen operation *is* the intervention, so it takes the
            // accent ring -- the one place selection and accent agree.
            .border_1()
            .border_color(if self.op == operation {
                colors.accent
            } else {
                colors.border
            })
            .rounded(px(Radius::MD))
            .accessibility_label(operation_label(operation))
            .child(
                div()
                    .w_full()
                    .flex()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label(operation_label(operation), Type::BODY, colors.text))
                    .child(label(
                        operation_hint(operation),
                        Type::LABEL,
                        colors.text_faint,
                    )),
            )
            .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                console.select_combo(ComboId::Op, operation, cx);
                console.step = WorkspaceStep::Intervention;
            }))
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

    fn prompt_step(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let model_status = match &self.session {
            Some(info) => (
                "Ready",
                format!(
                    "{} · {} layers · loaded in {}",
                    info.architecture,
                    info.n_layers,
                    fmt_load_ms(info.load_ms)
                ),
                colors.ok,
            ),
            None => (
                "Not loaded",
                "The model will load automatically when you run.".to_string(),
                colors.text_faint,
            ),
        };
        let raw_path = (self.advanced_open || self.model_options.is_empty()).then(|| {
            field(
                colors,
                "Model file",
                text_input(
                    colors,
                    self.inputs.model.clone(),
                    FONT_MONO_NAME,
                    Type::BODY,
                    None,
                    cx,
                ),
            )
        });

        div()
            .flex()
            .flex_col()
            .gap(px(Space::XL))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("Prepare the experiment", Type::TITLE, colors.text).whitespace_nowrap())
                    .child(label(
                        "Choose a local GGUF model and give it the prompt you want to study.",
                        Type::BODY,
                        colors.text_muted,
                    )),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(label("Start from an example", Type::LABEL, colors.text_faint))
                    .child(self.presets_block(colors, cx)),
            )
            .child(group(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::SM))
                    .child(section_label(colors, "Model"))
                    // Select, status and action on one line. They were three
                    // stacked rows, which split one object across the page and
                    // left the badge and the button looking like they belonged
                    // to the fields below rather than to the model.
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(px(Space::SM))
                            // A floor, not min_w(0): the picker wraps a library
                            // Select that sizes to its content, and min_w(0)
                            // let the whole control collapse to nothing beside
                            // the chip and the action.
                            .child(div().flex_1().min_w(px(200.0)).child(self.picker(
                                colors,
                                "model-picker",
                                ComboId::Model,
                                &self.model_path,
                                &self.model_options,
                                cx,
                            )))
                            .child(chip(model_status.0, model_status.2))
                            .child(div().flex_none().child(btn_secondary(
                                colors,
                                if self.status == Status::Preparing {
                                    "Loading…"
                                } else {
                                    "Load"
                                },
                                (!self.busy()).then(|| {
                                    cx.listener(|console, _: &ClickEvent, _window, cx| {
                                        console.load();
                                        cx.notify();
                                    })
                                }),
                            ))),
                    )
                    .child(label(model_status.1, Type::LABEL, colors.text_faint))
                    .children(raw_path),
            ))
            .child(group(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::SM))
                    .child(section_label(colors, "Prompt"))
                    // The editor is the most important object on the page: it
                    // gets the tallest region and the raised surface, with a
                    // hairline as its only boundary.
                    .child(text_input(
                        colors,
                        self.inputs.prompt.clone(),
                        FONT_ARABIC_NAME,
                        Type::SUBSECTION,
                        Some(150.0),
                        cx,
                    ))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .child(label(
                                format!("{} characters", self.prompt.chars().count()),
                                Type::META,
                                colors.text_faint,
                            ))
                            .child(div().w_full())
                            .child(label(
                                "Arabic and mixed-direction text supported",
                                Type::META,
                                colors.text_faint,
                            )),
                    ),
            ))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::SM))
                    .child(label("Generation length", Type::LABEL, colors.text_faint))
                    .child(self.generation_control(colors, cx)),
            )
    }

    fn layer_stepper(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
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

    fn intervention_step(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let needs_source = matches!(self.op.as_str(), "replace" | "interpolate" | "add-delta");
        let needs_value = matches!(self.op.as_str(), "scale" | "interpolate");

        let source_controls = needs_source.then(|| {
            div()
                .flex()
                .flex_col()
                .gap(px(Space::MD))
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
                .when(self.source == "capture", |controls| {
                    controls.child(field(
                        colors,
                        "Source layer",
                        text_input(
                            colors,
                            self.inputs.source_layer.clone(),
                            FONT_MONO_NAME,
                            Type::META,
                            None,
                            cx,
                        ),
                    ))
                })
        });
        let value_control = needs_value.then(|| {
            field(
                colors,
                if self.op == "interpolate" {
                    "Blend amount (0–1)"
                } else {
                    "Strength"
                },
                text_input(
                    colors,
                    self.inputs.value.clone(),
                    FONT_MONO_NAME,
                    Type::META,
                    None,
                    cx,
                ),
            )
        });
        let matched_span = (self.token == "matched-span").then(|| {
            field(
                colors,
                "Phrase to target",
                text_input(
                    colors,
                    self.inputs.span.clone(),
                    FONT_ARABIC_NAME,
                    Type::META,
                    None,
                    cx,
                ),
            )
        });

        div()
            .flex()
            .flex_col()
            .gap(px(Space::XL))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("Choose the internal change", Type::TITLE, colors.text).whitespace_nowrap())
                    .child(label(
                        "Start with the research question. Exact hook names remain available in Advanced controls.",
                        Type::BODY,
                        colors.text_muted,
                    )),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::SM))
                    .child(card_title(colors, "What should change?"))
                    .child(
                        div()
                            .flex()
                            .gap(px(Space::SM))
                            .child(self.operation_card(colors, "zero", cx))
                            .child(self.operation_card(colors, "scale", cx)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap(px(Space::SM))
                            .child(self.operation_card(colors, "replace", cx))
                            .child(self.operation_card(colors, "interpolate", cx)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap(px(Space::SM))
                            .child(self.operation_card(colors, "add-delta", cx))
                            .child(div().flex_1()),
                    )
                    .child(
                        div().pt(px(Space::XS)).child(label(
                            operation_explainer(&self.op),
                            Type::LABEL,
                            colors.text_muted,
                        )),
                    ),
            )
            // Where and Target are settings, not objects: spacing and section
            // labels organise them, and the kit Select keeps its own single
            // hairline without a card boundary doubling it.
            .child(panel(colors, group(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(card_title(colors, "Where"))
                    .child(field(
                        colors,
                        "Location in each layer",
                        self.picker(
                            colors,
                            "site-picker",
                            ComboId::Site,
                            &self.site,
                            &self.site_options,
                            cx,
                        ),
                    ))
                    // The picker says "After MLP block"; this line carries the
                    // exact frozen identifier for anyone reproducing a run.
                    // Micro mono: present for the record, silent at a glance.
                    .child(mono(
                        format!("ember.hook.v1 \u{00b7} {}", site_contract_name(&self.site)),
                        Type::MICRO,
                        colors.text_faint,
                    ))
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .flex_wrap()
                            .items_start()
                            .gap(px(Space::LG))
                            .when(per_layer(&self.site), |row| {
                                row.child(
                                    div()
                                        .w(px(260.0))
                                        .flex_none()
                                        .child(field(colors, "Layer", self.layer_stepper(colors, cx))),
                                )
                            })
                            .children(value_control.map(|control| {
                                div().w(px(200.0)).flex_none().child(control)
                            })),
                    )
                    .children(source_controls),
            )))
            .child(panel(colors, group(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(card_title(colors, "Target"))
                    .child(field(
                        colors,
                        "Tokens to affect",
                        self.picker(
                            colors,
                            "token-picker",
                            ComboId::Token,
                            &self.token,
                            &self.token_options,
                            cx,
                        ),
                    ))
                    .children(matched_span),
            )))
    }

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
            || ("None observed".to_string(), "no captured layer diverged".to_string()),
            |layer| (format!("Layer {layer}"), "first non-zero captured layer".to_string()),
        );
        let (peak_value, peak_detail) = match (landmarks.peak_relative_l2, landmarks.peak_layer) {
            (Some(value), Some(layer)) => {
                let magnitude = if value.abs() < 0.001 {
                    format!("{value:.2e}")
                } else {
                    format!("{value:.3}")
                };
                (format!("{magnitude} @ L{layer}"), "relative L2 difference".to_string())
            }
            _ => ("None observed".to_string(), "relative L2 difference".to_string()),
        };
        let (tail_value, tail_detail) = if comparison.generated_tokens_equal {
            ("Identical".to_string(), "exact token-ID suffix".to_string())
        } else {
            landmarks.stable_token_tail_step.map_or_else(
                || ("Not observed".to_string(), "exact token-ID suffix".to_string()),
                |step| (format!("From step {step}"), "exact token-ID suffix".to_string()),
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
                        .id(ElementId::Name(SharedString::from(format!("landmark:{title}"))))
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
            .child(landmark("Text output", text_value, text_detail, colors.text))
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
            Status::Preparing => "The first run loads the model, so it takes longer. Later runs start straight away.",
            Status::Running => "Ember is running your prompt twice, once untouched and once with your change.",
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
                        if state == 0 { colors.text_faint } else { colors.text },
                    ))
            }))
            .when(!note.is_empty(), |panel| {
                panel.child(label(note, Type::LABEL, colors.text_muted))
            })
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
                    .child(self.paired_outputs(colors, cx))
                    .child(self.layer_chart_panel(colors, 210.0, cx))
                    .into_any_element(),
                ResultView::Layers => div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::MD))
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
                            .child(label("Review and compare", Type::TITLE, colors.text).whitespace_nowrap())
                            .child(label(
                                "The baseline and intervention use the same prompt and deterministic settings.",
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
                                "review-rerun",
                                "Rerun",
                                cx.listener(|console, _: &ClickEvent, _window, cx| {
                                    console.rerun(cx);
                                }),
                            ))
                            .child(text_button(
                                "review-duplicate",
                                "Duplicate",
                                cx.listener(|console, _: &ClickEvent, _window, cx| {
                                    console.duplicate_experiment(cx);
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
                .child(
                    div()
                        .w(px(80.0))
                        .flex_none()
                        .child(label(name, Type::META, colors.text_faint)),
                )
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
            .child(prop("Model", div().flex().flex_col().gap(px(Space::XS))
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
            .child(prop("Input", div().flex().flex_col().gap(px(Space::XS))
                    .child(multiline(
                        &prompt_excerpt,
                        Type::LABEL,
                        colors.text,
                        FONT_ARABIC_NAME,
                    )),
            ))
            .child(prop("Target", div().flex().flex_col().gap(px(Space::XS))
                    .child(multiline(&target, Type::LABEL, colors.text, FONT_SANS_NAME)),
            ))
            .child(prop("Intervention", div().flex().flex_col().gap(px(Space::XS))
                    .child(label(intervention, Type::BODY, colors.accent)),
            ))
            .child(prop("Generation", div().flex().flex_col().gap(px(Space::XS))
                    .child(mono(
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
                    .child(prop("Run", div().flex().flex_col().gap(px(Space::XS))
                            .child(mono(
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
                    .child(prop(active_metric_label, div().flex().flex_col().gap(px(Space::XS))
                            .child(mono(format!("layer {}", metric.layer), Type::LABEL, colors.text))
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
        let page = match self.step {
            WorkspaceStep::Prompt => self.prompt_step(colors, cx).into_any_element(),
            WorkspaceStep::Intervention => self.intervention_step(colors, cx).into_any_element(),
            WorkspaceStep::Review => self.review_step(colors, cx).into_any_element(),
        };

        div()
            .id("workspace")
            .flex()
            .flex_row()
            .flex_1()
            .min_w(px(0.0))
            .h_full()
            .child(
                div()
                    .id("workspace-scroll")
                    .flex_1()
                    .min_w(px(0.0))
                    .h_full()
                    // Vertical scroll only. The page's min-content width --
                    // long mono identifiers, a two-pane result row -- was
                    // propagating up through the scroll container, past the
                    // column's `min_w(0)`, into the row that also holds the
                    // 300px aside. The row overflowed and the aside was the one
                    // that got squeezed, which is why it rendered at ~118px with
                    // the model name and the advanced disclosure clipped
                    // mid-word. Clipping here is what makes the column's
                    // `min_w(0)` actually mean something.
                    .overflow_y_scroll()
                    .overflow_x_hidden()
                    .px_5()
                    .pt_1()
                    .pb_5()
                    .child(
                        div()
                            .w_full()
                            .min_w(px(0.0))
                            // Forms are read and filled top to bottom: a
                            // bounded column keeps a one-digit field from
                            // spanning 1700px. Results keep the full width.
                            .when(self.step != WorkspaceStep::Review, |column| {
                                column.max_w(px(FORM_MAX_WIDTH))
                            })
                            .flex()
                            .flex_col()
                            .gap(px(Space::XL))
                            .child(self.feedback_banners(colors))
                            .child(self.experiment_pipeline(colors, cx))
                            .child(page),
                    ),
            )
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

    fn statusbar(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
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
        let (line, line_color) = if let Some((text, color)) = store_line.map(|t| (t, store_color))
        {
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
        let action_enabled = self.action_enabled();
        let action_label = match self.status {
            Status::Preparing => "Loading model…",
            Status::Running => "Running experiment…",
            Status::Restoring => "Verifying restore…",
            Status::Idle => match self.step {
                WorkspaceStep::Prompt => "Continue: Intervention",
                WorkspaceStep::Intervention => "Continue: Review",
                WorkspaceStep::Review if self.saved_run.is_some() => "Run this again",
                WorkspaceStep::Review if self.sample => "Run this for real",
                WorkspaceStep::Review if self.baseline.is_some() => "Run experiment again",
                WorkspaceStep::Review => "Run experiment",
            },
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
            .child(div().w(px(theme::scaled(230.0))).flex_none().child(btn_primary(
                colors,
                action_label,
                action_enabled.then(|| {
                    cx.listener(|console, _: &ClickEvent, _window, cx| {
                        console.advance_or_run();
                        cx.notify();
                    })
                }),
            )))
    }
}

/// The keyboard shortcuts, in one place so Settings cannot drift from the
/// bindings in the key handler and the palette.
fn shortcut_rows() -> Vec<(String, &'static str)> {
    let cmd = if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" };
    vec![
        (format!("{cmd}+K"), "Command palette"),
        (format!("{cmd}+Enter"), "Continue, or run the experiment"),
        (format!("{cmd}+N"), "New experiment"),
        (format!("{cmd}+R"), "Rerun the last run"),
        (format!("{cmd}+1 / 2 / 3"), "Prompt / Intervention / Review"),
        (format!("{cmd}+1 - 4 on Review"), "Overview / Layers / Tokens / Raw trace"),
        (format!("{cmd}+B"), "Show or hide the sidebar"),
        (format!("{cmd}+Shift+I"), "Show or hide the inspector"),
    ]
}

/// Window widths below which the inspector, then the sidebar, fold away.
/// Column width of the Prompt and Intervention forms.
const FORM_MAX_WIDTH: f32 = 940.0;
/// The inspector reads as a property list (name left, value right), which
/// needs a little more room than a stacked label did.
const INSPECTOR_WIDTH: f32 = 344.0;
const INSPECTOR_MIN_WINDOW: f32 = 1200.0;
const SIDEBAR_MIN_WINDOW: f32 = 960.0;

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
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .w_full()
                            .px_5()
                            .child(self.stepper(&colors, cx))
                            .child(div().w_full()),
                    )
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
            .when(show_sidebar, |row| {
                row.child(self.nav_rail(&colors, cx))
            })
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

fn fmt_load_ms(ms: f64) -> String {
    format!("{:.1} s", ms / 1000.0)
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

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------


/// The menu bar. The commands mirror the palette and the key handler, so a
/// menu item and its shortcut always do the same thing; Edit uses the kit's
/// own text actions so Cut, Copy, Paste and Undo reach the focused field.
fn app_menus() -> Vec<Menu> {
    use gpui_kit::component::input as text;
    vec![
        Menu::new("Ember").items([
            MenuItem::action("Settings\u{2026}", OpenSettings),
            MenuItem::separator(),
            MenuItem::action("Quit Ember", Quit),
        ]),
        Menu::new("Edit").items([
            MenuItem::os_action("Undo", text::Undo, OsAction::Undo),
            MenuItem::os_action("Redo", text::Redo, OsAction::Redo),
            MenuItem::separator(),
            MenuItem::os_action("Cut", text::Cut, OsAction::Cut),
            MenuItem::os_action("Copy", text::Copy, OsAction::Copy),
            MenuItem::os_action("Paste", text::Paste, OsAction::Paste),
            MenuItem::os_action("Select All", text::SelectAll, OsAction::SelectAll),
        ]),
        Menu::new("Experiment").items([
            MenuItem::action("New Experiment", StartExperiment),
            MenuItem::action("Rerun Last Run", ReplayLastRun),
            MenuItem::separator(),
            MenuItem::action("Open Sample Result", OpenSampleResult),
        ]),
        Menu::new("View").items([
            MenuItem::action("Command Palette", OpenPalette),
            MenuItem::separator(),
            MenuItem::action("Show or Hide Sidebar", HideShowSidebar),
            MenuItem::action("Show or Hide Inspector", HideShowInspector),
            MenuItem::action("Presentation Mode", EnterPresentation),
        ]),
        Menu::new("Help").items([
            MenuItem::action("Keyboard Shortcuts", ShowShortcuts),
            MenuItem::action("Ember on GitHub", OpenRepository),
        ]),
    ]
}

/// Route menu commands to the console. Registered per window, holding only a
/// weak handle, so a closed window never keeps the console alive.
fn register_menu_actions(
    console: WeakEntity<Console>,
    window: AnyWindowHandle,
    cx: &mut App,
) {
    fn to_console(
        console: &WeakEntity<Console>,
        cx: &mut App,
        f: impl FnOnce(&mut Console, &mut Context<Console>),
    ) {
        let _ = console.update(cx, f);
    }
    macro_rules! route {
        ($action:ty, |$c:ident, $cx:ident| $body:expr) => {{
            let console = console.clone();
            cx.on_action(move |_: &$action, cx: &mut App| {
                to_console(&console, cx, |$c, $cx| $body);
            });
        }};
    }
    route!(OpenSettings, |c, cx| c.goto(View::Settings, cx));
    route!(ShowShortcuts, |c, cx| c.goto(View::Settings, cx));
    route!(HideShowSidebar, |c, cx| c.toggle_sidebar(cx));
    route!(HideShowInspector, |c, cx| c.toggle_inspector(cx));
    route!(EnterPresentation, |c, cx| c.toggle_presentation(cx));
    route!(OpenSampleResult, |c, cx| c.show_sample(cx));
    route!(StartExperiment, |c, cx| {
        c.goto(View::Experiment, cx);
        c.step = WorkspaceStep::Prompt;
        cx.notify();
    });
    route!(ReplayLastRun, |c, cx| {
        if c.result_context.is_some() {
            c.goto(View::Experiment, cx);
            c.step = WorkspaceStep::Review;
            c.rerun(cx);
            cx.notify();
        }
    });
    cx.on_action(|_: &OpenRepository, cx: &mut App| cx.open_url(REPOSITORY_URL));
    // The palette needs the window to focus its field. A menu action runs
    // while that window is already being updated, and a nested update is
    // refused, so the work is deferred to just after the dispatch.
    let palette_console = console.clone();
    cx.on_action(move |_: &OpenPalette, cx: &mut App| {
        let console = palette_console.clone();
        cx.defer(move |cx| {
            let _ = window.update(cx, |_, window, cx| {
                let _ = console.update(cx, |console, cx| console.toggle_palette(window, cx));
            });
        });
    });
}

pub(crate) fn run_gui_command(
    _args: &NativeGuiArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    #[cfg(all(target_os = "macos", feature = "gui-tests"))]
    if let Some(directory) = &_args.render_test_dir {
        // EMBER_GUI_TEST_LIVE drives the real console end to end with a real
        // worker and model instead of rendering fixtures. Run it with
        // XDG_CONFIG_HOME pointing at a scratch directory: it writes history.
        // EMBER_GUI_TEST_PROBE=<model> runs a batch of configurations on one
        // loaded model and prints which ones change the generated text -- for
        // choosing a demo experiment with a visible effect.
        if let (Some(model), true) = (
            std::env::var_os("EMBER_GUI_TEST_PROBE"),
            std::env::var_os("XDG_CONFIG_HOME").is_some(),
        ) {
            return probe_examples(model.to_string_lossy().into_owned());
        }
        if let (Some(model), true) = (
            std::env::var_os("EMBER_GUI_TEST_LIVE"),
            std::env::var_os("XDG_CONFIG_HOME").is_some(),
        ) {
            return render_live_flow(directory, model.to_string_lossy().into_owned());
        }
        return render_test_artifacts(directory);
    }
    let (worker_tx, reply_rx) = spawn_worker(k_strategy, k_allow_fallback);
    eprintln!(
        "EMBER experiment console v{} (native, GPUI Kit)",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  model stays resident; press Ctrl-C to quit.");

    gpui_kit::application()
        .with_assets(icons::Assets)
        .run(move |cx: &mut App| {
            gpui_kit::init(cx);
            cx.on_action(|_: &Quit, cx| cx.quit());
            cx.bind_keys([KeyBinding::new(
                if cfg!(target_os = "macos") {
                    "cmd-q"
                } else {
                    "ctrl-q"
                },
                Quit,
                None,
            )]);
            cx.set_menus(app_menus());
            // Register the embedded fonts before the first window opens so the
            // text system can resolve Noto Sans / Mono / Naskh Arabic offline.
            cx.text_system()
                .add_fonts(vec![
                    Cow::Borrowed(FONT_SANS),
                    Cow::Borrowed(FONT_MONO),
                    Cow::Borrowed(FONT_ARABIC),
                ])
                .expect("register embedded fonts");

            let bounds = Bounds::centered(None, size(px(1180.0), px(720.0)), cx);
            cx.open_window(
                WindowOptions {
                    titlebar: Some(TitlebarOptions {
                        title: Some("EMBER \u{2014} experiment console".into()),
                        ..Default::default()
                    }),
                    window_bounds: Some(WindowBounds::Maximized(bounds)),
                    window_min_size: Some(size(px(980.0), px(620.0))),
                    ..Default::default()
                },
                move |window, cx| {
                    let console = cx.new(|cx| {
                        let system_dark = theme::system_is_dark(window.appearance());
                        let mut console =
                            Console::new(worker_tx, reply_rx, system_dark, window, cx);
                        cx.observe_window_appearance(window, |console, window, cx| {
                            console.system_appearance_changed(
                                theme::system_is_dark(window.appearance()),
                                cx,
                            );
                        })
                        .detach();
                        console.sync_kit_theme(cx);
                        console.spawn_poll(cx);
                        console
                    });
                    register_menu_actions(console.downgrade(), window.window_handle(), cx);
                    cx.new(|cx| gpui_kit::component::Root::new(console, window, cx))
                },
            )
            .expect("the experiment console window failed");
            cx.activate(true);
        });
    Ok(())
}
// ---------------------------------------------------------------------------
// render fixture
// ---------------------------------------------------------------------------

/// Whether the render harness should fill the store with representative runs.
///
/// Gated behind an env var rather than on `cfg!(feature = "gui-tests")` alone,
/// because a feature-gated fixture would also fire in the kit tests, where a
/// populated store would mask the empty-state behaviour those tests check.
fn seed_runs_requested() -> bool {
    std::env::var_os("EMBER_GUI_TEST_SEED_RUNS").is_some()
}

/// Representative history for screenshot review.
///
/// Every screenshot so far has been an empty state, which is why column widths,
/// truncation and density kept going unjudged until someone opened a real
/// window. These are the shapes that actually occur: a long model name, an
/// intervention with a layer, one run that did not verify, a pinned record, and
/// timestamps spread far enough apart to exercise "2m ago" versus "3d ago".
///
/// Timestamps are relative to load time so the relative formatting stays
/// honest whenever the fixtures are regenerated.
#[cfg(feature = "gui-tests")]
fn seed_store() -> AppStore {
    /// One fixture row. Named fields on purpose: the previous version was a
    /// twelve-element tuple ending in three bare booleans, and I set `pinned`
    /// on six of the seven rows without noticing, which made the pin marker
    /// meaningless and the ordering look like a plain newest-first list.
    struct Row {
        model: &'static str,
        intervention: &'static str,
        hook: &'static str,
        layer: Option<u32>,
        duration_ms: u64,
        baseline_tokens: u32,
        intervention_tokens: u32,
        diverged_at_step: Option<u32>,
        outputs_equal: bool,
        verified: bool,
        pinned: bool,
        age_seconds: i64,
    }
    // The shapes that actually occur: a long model name, a layer, a run that
    // did not verify, a run whose token counts diverged, and timestamps spread
    // from minutes to a fortnight so the relative formatting is exercised.
    //
    // One run is pinned, and it is deliberately not the newest, so the render
    // shows pinning reordering rather than coinciding with recency.
    let rows = [
        Row {
            model: "Llama-3.2-1B-Instruct-Q8_0",
            intervention: "Scale \u{d7}0.5",
            hook: "After MLP block",
            layer: Some(8),
            duration_ms: 1_240,
            baseline_tokens: 48,
            intervention_tokens: 48,
            diverged_at_step: Some(17),
            outputs_equal: false,
            verified: true,
            pinned: false,
            age_seconds: 120,
        },
        Row {
            model: "Qwen2.5-1.5B-Instruct-Q8_0",
            intervention: "Zero",
            hook: "Before output head",
            layer: None,
            duration_ms: 890,
            baseline_tokens: 48,
            intervention_tokens: 31,
            diverged_at_step: Some(4),
            outputs_equal: false,
            verified: true,
            pinned: false,
            age_seconds: 2_400,
        },
        Row {
            model: "gemma-3-270m-it-Q4_K_M",
            intervention: "Copy from layer",
            hook: "After MLP block",
            layer: Some(3),
            duration_ms: 410,
            baseline_tokens: 32,
            intervention_tokens: 32,
            diverged_at_step: Some(2),
            outputs_equal: false,
            verified: true,
            pinned: true,
            age_seconds: 86_400,
        },
        Row {
            model: "Llama-3.2-1B-Instruct-Q8_0",
            intervention: "Add a learned difference",
            hook: "After attention block",
            layer: Some(14),
            duration_ms: 3_010,
            baseline_tokens: 48,
            intervention_tokens: 48,
            diverged_at_step: None,
            outputs_equal: true,
            verified: false,
            pinned: false,
            age_seconds: 9_600,
        },
        Row {
            model: "Llama-3.2-1B-Instruct-Q8_0",
            intervention: "Blend representations",
            hook: "Final prompt token",
            layer: None,
            duration_ms: 1_980,
            baseline_tokens: 48,
            intervention_tokens: 48,
            diverged_at_step: Some(29),
            outputs_equal: false,
            verified: true,
            pinned: false,
            age_seconds: 172_800,
        },
        Row {
            model: "Qwen2.5-0.5B-Instruct-Q8_0",
            intervention: "Scale \u{d7}0.75",
            hook: "Before output head",
            layer: None,
            duration_ms: 640,
            baseline_tokens: 24,
            intervention_tokens: 24,
            diverged_at_step: None,
            outputs_equal: true,
            verified: true,
            pinned: false,
            age_seconds: 604_800,
        },
        Row {
            model: "Llama-3.2-1B-Instruct-Q8_0",
            intervention: "Zero",
            hook: "After attention block",
            layer: Some(21),
            duration_ms: 2_260,
            baseline_tokens: 48,
            intervention_tokens: 12,
            diverged_at_step: Some(1),
            outputs_equal: false,
            verified: true,
            pinned: false,
            age_seconds: 1_209_600,
        },
    ];

    let now = unix_now();
    let mut store = AppStore::default();
    for (index, row) in rows.iter().enumerate() {
        store.push_run(RunRecord {
            number: (rows.len() - index) as u64,
            finished_at: now - row.age_seconds,
            model: row.model.to_string(),
            intervention: row.intervention.to_string(),
            hook: row.hook.to_string(),
            layer: row.layer,
            duration_ms: Some(row.duration_ms),
            baseline_tokens: Some(row.baseline_tokens),
            intervention_tokens: Some(row.intervention_tokens),
            diverged_at_step: row.diverged_at_step,
            outputs_equal: row.outputs_equal,
            verified: row.verified,
            pinned: row.pinned,
            prompt: "\u{0627}\u{0643}\u{062a}\u{0628} \u{062c}\u{0645}\u{0644}\u{0629}".to_string(),
            result: (index <= 1).then(|| app_store::RecordResult {
                baseline_text: "Paris. The Eiffel Tower is located in Paris.".to_string(),
                intervention_text: "covered in a thick layer of fog.".to_string(),
                layers: (0..16)
                    .map(|layer| app_store::RecordLayer {
                        layer,
                        relative_l2: Some(if layer < 7 { 0.0 } else { 1.1 }),
                        cosine: Some(if layer < 7 { 0.0 } else { 0.9 }),
                    })
                    .collect(),
                tokens: Vec::new(),
                first_layer_divergence: Some(7),
                peak_layer: Some(10),
                peak_relative_l2: Some(1.187),
                tokens_equal: false,
            }),
            // Only recent rows carry a configuration: the Reuse action must
            // be shown and hidden in the same render.
            config: (index == 0).then(|| app_store::RecordConfig {
                model_path: format!("/models/{}.gguf", row.model),
                execution: "reference".to_string(),
                site: "after-mlp".to_string(),
                layer: row.layer.map(|layer| layer.to_string()).unwrap_or_default(),
                op: "scale".to_string(),
                value: "0.5".to_string(),
                source: "live".to_string(),
                source_layer: String::new(),
                token: "prompt-final".to_string(),
                span: String::new(),
                max_tokens: "48".to_string(),
            }),
        });
        store.touch_model(
            &format!("/models/{}.gguf", row.model),
            now - row.age_seconds,
        );
    }
    // An experiment in progress, so Home's "continue where you left off" row
    // is reviewed at real density rather than as a guess.
    store.draft = Some(app_store::Draft {
        revision: 3,
        prompt: "\u{0627}\u{0643}\u{062a}\u{0628} \u{062c}\u{0645}\u{0644}\u{0629} \u{0642}\u{0635}\u{064a}\u{0631}\u{0629} \u{0639}\u{0646} \u{0627}\u{0644}\u{0645}\u{062f}\u{064a}\u{0646}\u{0629} \u{0627}\u{0644}\u{0645}\u{0646}\u{0648}\u{0631}\u{0629}".to_string(),
        model_path: "/models/Llama-3.2-1B-Instruct-Q8_0.gguf".to_string(),
        fields: [
            ("op".to_string(), "scale".to_string()),
            ("site".to_string(), "after-mlp".to_string()),
            ("layer".to_string(), "8".to_string()),
            ("value".to_string(), "0.5".to_string()),
            ("max_tokens".to_string(), "48".to_string()),
            ("token".to_string(), "prompt-final".to_string()),
        ]
        .into_iter()
        .collect(),
        step: "intervention".to_string(),
        updated_at: now - 240,
    });
    store
}

#[cfg(not(feature = "gui-tests"))]
fn seed_store() -> AppStore {
    AppStore::default()
}

/// What a finished run showed, in the shape History stores.
fn record_result(bundle: &crate::gui::RunBundle) -> app_store::RecordResult {
    let comparison = &bundle.comparison;
    app_store::RecordResult {
        baseline_text: bundle.baseline.text.clone(),
        intervention_text: bundle.intervention.text.clone(),
        layers: comparison
            .layers
            .iter()
            .map(|metric| app_store::RecordLayer {
                layer: metric.layer,
                relative_l2: metric.relative_l2_difference,
                cosine: metric.cosine_distance,
            })
            .collect(),
        tokens: comparison
            .tokens
            .iter()
            .map(|token| app_store::RecordToken {
                position: token.position,
                baseline: token.baseline_text.clone(),
                intervention: token.intervention_text.clone(),
                differs: token.differs,
            })
            .collect(),
        first_layer_divergence: comparison.landmarks.first_layer_divergence,
        peak_layer: comparison.landmarks.peak_layer,
        peak_relative_l2: comparison.landmarks.peak_relative_l2,
        tokens_equal: comparison.generated_tokens_equal,
    }
}

/// The built-in sample: what a finished comparison looks like. The numbers
/// are illustrative -- shaped like a real Llama-3.2-1B run of "scale x0.5 at
/// layer 8" -- and the page says so; nothing here claims to be a measurement.
fn sample_result() -> (RunOutput, RunOutput, ExperimentComparison, FormValues) {
    let text = " Paris. The city is known for its art, its food and its history.";
    let make = |wall_ms: f64| RunOutput {
        text: text.to_string(),
        generated_token_ids: (1..=16).collect(),
        generated_token_texts: vec![String::new(); 16],
        prompt_tokens: 6,
        generated_tokens: 16,
        bundle_dir: "sample".to_string(),
        semantic_hash: "0000000000000000".to_string(),
        payload_hash: "00000000".to_string(),
        wall_ms,
        decode_tps: Some(16.0 / (wall_ms / 1000.0)),
        events: Vec::new(),
    };
    let series: [(f64, f64); 16] = [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.296, 0.045),
        (0.245, 0.031),
        (0.207, 0.024),
        (0.158, 0.015),
        (0.150, 0.013),
        (0.138, 0.011),
        (0.141, 0.011),
        (0.141, 0.011),
    ];
    let layers = series
        .iter()
        .enumerate()
        .map(|(layer, (l2, cosine))| crate::gui::LayerMetric {
            layer,
            relative_l2_difference: Some(*l2),
            cosine_distance: Some(*cosine),
            maximum_absolute_difference: None,
            exact: *l2 == 0.0,
        })
        .collect();
    let pieces = [
        " Paris", ".", " The", " city", " is", " known", " for", " its", " art", ",", " its",
        " food", " and", " its", " history", ".",
    ];
    let tokens = pieces
        .iter()
        .enumerate()
        .map(|(index, piece)| crate::gui::TokenMetric {
            position: index + 1,
            baseline_token_id: Some(index as u32 + 1),
            intervention_token_id: Some(index as u32 + 1),
            baseline_text: Some((*piece).to_string()),
            intervention_text: Some((*piece).to_string()),
            differs: false,
        })
        .collect();
    let comparison = ExperimentComparison {
        layers,
        tokens,
        first_token_divergence: None,
        generated_tokens_equal: true,
        generated_text_equal: true,
        landmarks: crate::gui::DivergenceLandmarks {
            first_layer_divergence: Some(8),
            peak_layer: Some(8),
            peak_relative_l2: Some(0.296),
            stable_token_tail_step: None,
        },
        layer_token_grid: None,
    };
    let values = FormValues {
        model_path: String::new(),
        prompt: "The capital of France is".to_string(),
        max_tokens: "24".to_string(),
        execution: "reference".to_string(),
        site: "after-mlp".to_string(),
        layer: "8".to_string(),
        op: "scale".to_string(),
        value: "0.5".to_string(),
        source: "capture".to_string(),
        source_layer: "0".to_string(),
        token: "prompt-final".to_string(),
        span: String::new(),
    };
    (make(1_180.0), make(1_240.0), comparison, values)
}

/// A finished comparison, for reviewing the Review page at real density.
///
/// The empty state is what every screenshot had shown, so the pane layout, the
/// metadata row and the bidi handling of a mixed-direction completion were all
/// being judged against a page with nothing on it. Two outputs that share a
/// prefix and then diverge is the interesting case: it is the one where the
/// token arrow, the divergence note and the side-by-side panes all have
/// something to say.
#[cfg(feature = "gui-tests")]
fn seed_comparison() -> (RunOutput, RunOutput, ExperimentComparison) {
    let shared = "The city of Madinah is one of the oldest continuously inhabited places in the world, known for the Prophet's Mosque.";
    let diverged = "The city of Madinah is among the oldest inhabited places in the world, famed for the Prophet's Mosque and its courtyards.";
    let arabic = "\u{0627}\u{0643}\u{062a}\u{0628}\u{060c}\u{0627}\u{0644}\u{0645}\u{062f}\u{064a}\u{0646}\u{0629} \u{0627}\u{062d}\u{062f} \u{0623}\u{0642}\u{062f}\u{0645}\u{0627}\u{0641}\u{0627}\u{064a} \u{0641}\u{064a} \u{0627}\u{0644}\u{0639}\u{0627}\u{0644}\u{0645}.";
    let make = |text: String, tokens: usize, wall_ms: f64| RunOutput {
        text,
        generated_token_ids: vec![1; tokens],
        generated_token_texts: vec![String::new(); tokens],
        prompt_tokens: 24,
        generated_tokens: tokens,
        bundle_dir: "/tmp/ember-render-fixture/baseline".to_string(),
        semantic_hash: "9f2c1a7b4e6d0358".to_string(),
        payload_hash: "3a91f0c2".to_string(),
        wall_ms,
        decode_tps: Some(tokens as f64 / (wall_ms / 1000.0)),
        events: Vec::new(),
    };
    let baseline = make(shared.to_string(), 48, 1_240.0);
    let intervention = make(format!("{shared} {diverged}"), 61, 1_980.0);
    let comparison = ExperimentComparison {
        layers: Vec::new(),
        tokens: Vec::new(),
        first_token_divergence: Some(38),
        generated_tokens_equal: false,
        generated_text_equal: false,
        landmarks: crate::gui::DivergenceLandmarks {
            first_layer_divergence: Some(7),
            peak_layer: Some(14),
            peak_relative_l2: Some(0.214),
            stable_token_tail_step: None,
        },
        layer_token_grid: None,
    };
    let _ = arabic;
    (baseline, intervention, comparison)
}

// Without the gui-tests feature there is no render harness to feed, and the
// only call site is behind the same cfg, so this stub exists purely to keep the
// non-test build compiling.
#[cfg(not(feature = "gui-tests"))]
#[expect(dead_code, reason = "only the gui-tests build calls the real one")]
fn seed_comparison() -> (RunOutput, RunOutput, ExperimentComparison) {
    unimplemented!("render fixture is only available under gui-tests")
}

#[cfg(test)]
mod tests {
    use super::{theme, truncate_chars, AppearanceMode, FormValues};
    use crate::gui::parse_run_request;

    fn form() -> FormValues {
        FormValues {
            model_path: "model.gguf".to_string(),
            prompt: "اختبار".to_string(),
            max_tokens: "48".to_string(),
            execution: "reference".to_string(),
            site: "after-mlp".to_string(),
            layer: "8".to_string(),
            op: "scale".to_string(),
            value: "0.5".to_string(),
            source: "capture".to_string(),
            source_layer: "0".to_string(),
            token: "prompt-final".to_string(),
            span: String::new(),
        }
    }

    #[test]
    fn appearance_mode_cycles_and_resolves_system_theme() {
        assert_eq!(AppearanceMode::System.next(), AppearanceMode::Dark);
        assert_eq!(AppearanceMode::Dark.next(), AppearanceMode::Light);
        assert_eq!(AppearanceMode::Light.next(), AppearanceMode::System);
        assert!(AppearanceMode::System.is_dark(true));
        assert!(!AppearanceMode::System.is_dark(false));
        assert!(AppearanceMode::Dark.is_dark(false));
        assert!(!AppearanceMode::Light.is_dark(true));
    }

    #[test]
    fn light_and_dark_palettes_differ_in_core_semantic_roles() {
        let dark = theme::dark();
        let light = theme::light();
        assert_ne!(dark.canvas, light.canvas);
        assert_ne!(dark.sidebar, light.sidebar);
        assert_ne!(dark.surface, light.surface);
        assert_ne!(dark.text, light.text);
        assert_ne!(dark.text_muted, light.text_muted);
        assert_ne!(dark.border, light.border);
        assert_ne!(dark.accent, light.accent);
        assert_ne!(dark.ok, light.ok);
        assert_ne!(dark.err, light.err);
        assert_ne!(dark.warn, light.warn);
        assert_ne!(dark.err_box_bg, light.err_box_bg);
    }

    #[test]
    fn default_form_builds_valid_run_request() {
        let request = form().build_run_request().unwrap();
        let config = parse_run_request(&request).expect("default config validates");
        assert_eq!(config.site.stage_id(), "after-mlp");
        assert_eq!(config.layer, Some(8));
        assert!(matches!(config.operation, crate::gui::GuiOperation::Scale));
    }

    #[test]
    fn non_per_layer_site_drops_layer() {
        let mut form = form();
        form.site = "before-logits".to_string();
        assert!(form.build_run_request().unwrap().layer.is_none());
    }

    #[test]
    fn scale_factor_parses_and_validates() {
        let mut form = form();
        form.value = "abc".to_string();
        assert!(form.build_run_request().is_err());
        form.value = "0.25".to_string();
        let config = parse_run_request(&form.build_run_request().unwrap()).unwrap();
        assert_eq!(config.factor, 0.25);
    }

    #[test]
    fn source_layer_is_clamped_below_target() {
        let mut form = form();
        form.layer = "7".to_string();
        form.source_layer = "9".to_string();
        form.op = "replace".to_string();
        let config = parse_run_request(&form.build_run_request().unwrap()).unwrap();
        assert_eq!(config.source_layer, Some(6));
    }

    #[test]
    #[ignore = "requires EMBER_GUI_TEST_MODEL; writes verified experiment bundles"]
    fn native_worker_runs_and_restores_real_model() {
        use super::{spawn_worker, WorkerMsg, WorkerReply};
        use ember::quant_k::KStrategy;
        use std::time::Duration;
        let model = std::env::var("EMBER_GUI_TEST_MODEL").expect("set EMBER_GUI_TEST_MODEL");
        let mut values = form();
        values.model_path = model.clone();
        values.prompt = "The capital of France is".into();
        values.max_tokens = "4".into();
        let mut request = values.build_run_request().unwrap();
        let config = parse_run_request(&request).unwrap();
        let (tx, rx) = spawn_worker(KStrategy::Auto, false);
        let receive = || {
            rx.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(300))
                .unwrap()
        };
        tx.send(WorkerMsg::Prepare(model)).unwrap();
        let WorkerReply::Prepared(info) = receive() else {
            panic!("expected prepared reply")
        };
        let info = info.unwrap();
        assert!(info.n_layers > 8);
        tx.send(WorkerMsg::Run(config)).unwrap();
        let WorkerReply::RunDone(result) = receive() else {
            panic!("expected run reply")
        };
        let run = result.unwrap();
        assert!(run.verification.ok);
        assert!(!run.baseline.generated_token_ids.is_empty());
        println!(
            "baseline: {}\nintervention: {}",
            run.baseline.text, run.intervention.text
        );
        println!(
            "baseline bundle: {}\nintervention bundle: {}",
            run.baseline.bundle_dir, run.intervention.bundle_dir
        );
        request.operation = "restore-original".into();
        request.factor = None;
        request.alpha = None;
        request.source_layer = None;
        tx.send(WorkerMsg::Restore(parse_run_request(&request).unwrap()))
            .unwrap();
        let WorkerReply::RestoreDone(result) = receive() else {
            panic!("expected restore reply")
        };
        let restored = result.unwrap();
        assert!(restored.verification.ok);
        assert!(restored.baseline_comparable);
        assert!(restored.matches_baseline);
        assert_eq!(
            restored.output.generated_token_ids,
            run.baseline.generated_token_ids
        );
        println!("restoration bundle: {}", restored.output.bundle_dir);
    }

    #[test]
    fn inspector_excerpt_preserves_arabic_characters() {
        assert_eq!(truncate_chars("المدينة المنورة", 7), "المدينة…");
        assert_eq!(truncate_chars("اختبار", 20), "اختبار");
    }
}

#[cfg(all(test, feature = "gui-tests"))]
mod kit_tests {
    use super::{Console, Preset, View, WorkspaceStep, FONT_ARABIC, FONT_MONO, FONT_SANS};
    use gpui_kit::component::Root;
    use gpui_kit::test::{TestAppContextExt, TestWindowExt};
    use gpui_kit::{AppContext, SharedString, TestAppContext};
    use std::{
        borrow::Cow,
        sync::{mpsc, Arc, Mutex},
    };

    /// The primary action must be gated identically by the button and by the
    /// Ctrl+Enter shortcut.
    ///
    /// The keyboard path previously advanced the workspace with no gate at
    /// all, so a visibly disabled button could still be driven from the
    /// keyboard -- and the UI advertises "Ctrl+Enter" right next to it.
    /// While a run is in flight the button is disabled; the shortcut must
    /// refuse too.
    #[gpui_kit::test]
    /// The inspector declares 300px. This asserts the layout engine agrees.
    ///
    /// It was rendering at roughly 118px at the standard 1180pt window, clipping
    /// the model name, the hook and the advanced disclosure mid-word on every
    /// experiment screen. Four hypotheses were tried and ruled out by reading
    /// screenshots: `min_w(0)` on the aside, `flex_shrink_0` on it, `w_full()`
    /// on the workspace column, and `overflow_x_hidden` on the scroll container.
    /// Inferring layout from a picture of it does not work.
    ///
    /// **What this does and does not prove.** It passes, so the declaration is
    /// honoured and the 300px is not being ignored outright. But the test window
    /// is wide enough that the row never has to overflow, which is exactly the
    /// condition the render fails under -- so this does not reproduce the bug.
    /// Closing it needs this test to drive a narrow window, and then whatever it
    /// reports is the number to fix against.
    ///
    /// The aside also needed `.test_support()`. Without it its id was only a
    /// scope inside its children's paths, so a test could find the controls
    /// inside it but never the box itself -- which is why a layout bug this
    /// visible had no test standing behind it.
    #[gpui_kit::test]
    async fn inspector_keeps_its_declared_width(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        let (tx, _worker) = mpsc::channel();
        let (_reply, rx) = mpsc::channel();
        let cell: Arc<Mutex<Option<gpui_kit::Entity<Console>>>> = Arc::default();
        let sink = cell.clone();
        let handle = cx.add_window(move |window, cx| {
            let console =
                cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(rx)), false, window, cx));
            console.update(cx, |console, _| {
                console.view = View::Experiment;
                console.inspector_open = true;
            });
            *sink.lock().expect("cell unlocked") = Some(console.clone());
            Root::new(console, window, cx)
        });
        cx.run_until_parked();
        cx.update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
        })
        .unwrap();
        // The aside appears once the row that owns it has been laid out, which
        // is a later frame than the first. `find` on an earlier frame reports a
        // miss even though the id is registered, so wait for it rather than
        // reading the first frame and concluding the inspector is not there.
        cx.wait_for(
            handle.into(),
            std::time::Duration::from_secs(2),
            |window, _| window.try_find(SharedString::from("inspector")).is_some(),
        )
        .await;

        let width = cx
            .update_window(handle.into(), |_, window, _| {
                window
                    .find(SharedString::from("inspector"))
                    .bounds()
                    .size
                    .width
            })
            .expect("the window update runs");
        // Compared in f32: `Pixels` has no `abs` in this version.
        let rendered: f32 = width.into();
        assert!(
            (rendered - super::INSPECTOR_WIDTH).abs() < 2.0,
            "inspector rendered at {rendered}px, not the {}px it declares",
            super::INSPECTOR_WIDTH
        );
    }

    fn primary_action_is_gated_for_both_mouse_and_keyboard(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        let (tx, _worker) = mpsc::channel();
        let (_reply, rx) = mpsc::channel();

        // Console::new needs a real Window, so the entity is built inside the
        // window closure and handed back out through a cell for assertions.
        let cell: Arc<Mutex<Option<gpui_kit::Entity<Console>>>> = Arc::default();
        let sink = cell.clone();
        let handle = cx.add_window(move |window, cx| {
            let console =
                cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(rx)), false, window, cx));
            *sink.lock().expect("cell unlocked") = Some(console.clone());
            Root::new(console, window, cx)
        });
        cx.update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
        })
        .unwrap();
        let console = cell
            .lock()
            .expect("cell unlocked")
            .clone()
            .expect("window construction registers the Console");

        // Idle: the action is eligible and advances the workspace.
        let idle = cx.update(|cx| {
            let console = console.read(cx);
            (console.action_enabled(), console.busy(), console.step)
        });
        assert!(idle.0, "an idle console must allow the primary action");
        assert!(!idle.1, "an idle console must not report busy");

        let step_before = idle.2;
        cx.update(|cx| console.update(cx, |console, _| console.advance_or_run()));
        let step_after = cx.update(|cx| console.read(cx).step);
        assert_ne!(
            step_before, step_after,
            "an eligible action must advance the workspace"
        );

        // Busy: a run is in flight, so the button is disabled and the
        // shortcut must be refused rather than advancing the step anyway.
        cx.update(|cx| {
            console.update(cx, |console, _| {
                console.status = super::Status::Running;
            })
        });
        let busy = cx.update(|cx| {
            let console = console.read(cx);
            (console.action_enabled(), console.busy(), console.step)
        });
        assert!(busy.1, "the console must report busy while running");
        assert!(
            !busy.0,
            "a busy console must disable the primary action (and its shortcut)"
        );

        cx.update(|cx| console.update(cx, |console, _| console.advance_or_run()));
        let after_busy = cx.update(|cx| console.read(cx).step);
        assert_eq!(
            busy.2, after_busy,
            "the keyboard path must not advance the workspace while busy"
        );
    }

    #[gpui_kit::test]
    fn populated_chart_registers_handlers_during_paint(cx: &mut TestAppContext) {
        use gpui_kit::{Context, IntoElement, Render, Window};
        struct ChartFixture(gpui_kit::Entity<Console>);
        impl Render for ChartFixture {
            fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
                super::chart::layer_divergence_chart(
                    self.0.clone(),
                    Arc::from(vec![crate::gui::LayerMetric {
                        layer: 8,
                        relative_l2_difference: Some(0.5),
                        cosine_distance: Some(0.01),
                        maximum_absolute_difference: Some(1.0),
                        exact: false,
                    }]),
                    Some(8),
                    None,
                    None,
                    160.,
                    &super::theme::light(),
                )
            }
        }
        cx.update(gpui_kit::init);
        let (tx, _worker) = mpsc::channel();
        let (_reply, rx) = mpsc::channel();
        let handle = cx.add_window(|window, cx| {
            let console =
                cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(rx)), false, window, cx));
            let chart = cx.new(|_| ChartFixture(console));
            Root::new(chart, window, cx)
        });
        cx.update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
        })
        .unwrap();
    }

    #[gpui_kit::test]
    async fn opening_a_saved_run_reopens_its_comparison(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_kit::init(cx);
            cx.text_system()
                .add_fonts(vec![
                    Cow::Borrowed(FONT_SANS),
                    Cow::Borrowed(FONT_MONO),
                    Cow::Borrowed(FONT_ARABIC),
                ])
                .unwrap();
        });
        let (tx, _worker_rx) = mpsc::channel();
        let (_reply_tx, reply_rx) = mpsc::channel();
        let mut view = None;
        let handle = cx.add_window(|window, cx| {
            let console =
                cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
            view = Some(console.clone());
            Root::new(console, window, cx)
        });
        let console = view.unwrap();
        cx.update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            console.update(cx, |console, cx| {
                let mut run = super::seed_store().runs[0].clone();
                run.number = 77;
                run.result = Some(super::app_store::RecordResult {
                    baseline_text: "Paris.".into(),
                    intervention_text: "fog".into(),
                    layers: vec![super::app_store::RecordLayer {
                        layer: 7,
                        relative_l2: Some(1.25),
                        cosine: Some(0.5),
                    }],
                    tokens: Vec::new(),
                    first_layer_divergence: Some(7),
                    peak_layer: Some(7),
                    peak_relative_l2: Some(1.25),
                    tokens_equal: false,
                });
                run.config = Some(super::app_store::RecordConfig {
                    model_path: "fixture.gguf".into(),
                    execution: "reference".into(),
                    site: "after-layer".into(),
                    layer: "6".into(),
                    op: "scale".into(),
                    value: "0.0".into(),
                    source: "capture".into(),
                    source_layer: "0".into(),
                    token: "prompt-final".into(),
                    span: String::new(),
                    max_tokens: "24".into(),
                });
                console.store.push_run(run);
                console.open_run(77, cx);
            });
            let console = console.read(cx);
            assert_eq!(console.saved_run, Some(77));
            assert!(console.sample, "a reopened run is not a live result");
            assert_eq!(console.step, WorkspaceStep::Review);
            assert_eq!(console.layer_series.len(), 1);
            assert_eq!(console.selected_layer, Some(7));
            assert_eq!(console.layer, "6");
            let markdown = console.result_markdown().expect("a reopened run can be copied");
            assert!(markdown.contains("Run #77, reopened from history"));
        })
        .unwrap();
    }

    #[test]
    fn the_menu_bar_has_the_expected_menus_and_none_are_empty() {
        let menus = super::app_menus();
        let names: Vec<_> = menus.iter().map(|menu| menu.name.to_string()).collect();
        assert_eq!(names, ["Ember", "Edit", "Experiment", "View", "Help"]);
        assert!(menus.iter().all(|menu| !menu.items.is_empty()));
    }

    #[gpui_kit::test]
    async fn menu_commands_reach_the_console(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_kit::init(cx);
            cx.text_system()
                .add_fonts(vec![
                    Cow::Borrowed(FONT_SANS),
                    Cow::Borrowed(FONT_MONO),
                    Cow::Borrowed(FONT_ARABIC),
                ])
                .unwrap();
        });
        let (tx, _worker_rx) = mpsc::channel();
        let (_reply_tx, reply_rx) = mpsc::channel();
        let mut view = None;
        let handle = cx.add_window(|window, cx| {
            let console =
                cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
            view = Some(console.clone());
            Root::new(console, window, cx)
        });
        let console = view.unwrap();
        cx.update(|cx| super::register_menu_actions(console.downgrade(), handle.into(), cx));
        cx.update_window(handle.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
        let before = console.read_with(cx, |c, _| c.sidebar_open);
        cx.dispatch_action(handle.into(), super::HideShowSidebar);
        assert_ne!(console.read_with(cx, |c, _| c.sidebar_open), before);
        cx.dispatch_action(handle.into(), super::OpenSettings);
        assert_eq!(console.read_with(cx, |c, _| c.view), View::Settings);
        cx.dispatch_action(handle.into(), super::OpenSampleResult);
        assert!(console.read_with(cx, |c, _| c.sample));
        cx.dispatch_action(handle.into(), super::OpenPalette);
        cx.run_until_parked();
        assert!(console.read_with(cx, |c, _| c.palette_open));
    }

    #[gpui_kit::test]
    async fn run_numbers_continue_from_stored_history(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_kit::init(cx);
            cx.text_system()
                .add_fonts(vec![
                    Cow::Borrowed(FONT_SANS),
                    Cow::Borrowed(FONT_MONO),
                    Cow::Borrowed(FONT_ARABIC),
                ])
                .unwrap();
        });
        // `seed_runs_requested` is off in this test, so the console loads the
        // real store path; under cfg(test) persistence is disabled, and a
        // fresh console reads whatever is on disk. Assert the rule directly on
        // the derivation instead of depending on that file.
        let mut store = super::AppStore::default();
        for number in [3_u64, 9, 4] {
            let mut run = super::seed_store().runs[0].clone();
            run.number = number;
            store.runs.push(run);
        }
        let next = store.runs.iter().map(|run| run.number).max().unwrap_or(0) + 1;
        assert_eq!(next, 10, "the next run must not reuse a stored number");
        let numbers: std::collections::BTreeSet<_> =
            store.runs.iter().map(|run| run.number).collect();
        assert_eq!(numbers.len(), store.runs.len());
    }

    #[gpui_kit::test]
    async fn sample_result_opens_review_and_copies_markdown(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_kit::init(cx);
            cx.text_system()
                .add_fonts(vec![
                    Cow::Borrowed(FONT_SANS),
                    Cow::Borrowed(FONT_MONO),
                    Cow::Borrowed(FONT_ARABIC),
                ])
                .unwrap();
        });
        let (tx, _worker_rx) = mpsc::channel();
        let (_reply_tx, reply_rx) = mpsc::channel();
        let mut view = None;
        let handle = cx.add_window(|window, cx| {
            let console =
                cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
            view = Some(console.clone());
            Root::new(console, window, cx)
        });
        let console = view.unwrap();
        cx.update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            window.click(SharedString::from("home-sample"), cx);
            {
                let console = console.read(cx);
                assert!(console.sample, "the sample flag marks illustrative data");
                assert_eq!(console.step, WorkspaceStep::Review);
                assert!(console.baseline.is_some() && console.comparison.is_some());
            }
            window.render_frame(cx);
            window.click(SharedString::from("review-copy"), cx);
            assert!(console.read(cx).copied);
        })
        .unwrap();
        let copied = cx
            .read_from_clipboard()
            .and_then(|item| item.text())
            .expect("the summary is on the clipboard");
        assert!(copied.starts_with("# Ember experiment"));
        assert!(copied.contains("Sample result"), "a sample must say so in the copy");
        assert!(copied.contains("| 8 | 0.2960 | 0.0450 |"));
        assert!(copied.contains("First internal divergence:** layer 8"));
        // A real run must not inherit the sample's label or leave it copied.
        cx.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, _| {
                console.send_run(
                    crate::gui::parse_run_request(&console.build_run_request().unwrap()).unwrap(),
                );
                assert!(!console.sample);
            });
        })
        .unwrap();
    }

    #[gpui_kit::test]
    async fn kit_navigation_and_presets_update_experiment_state(cx: &mut TestAppContext) {
        cx.update(|cx| {
            gpui_kit::init(cx);
            cx.text_system()
                .add_fonts(vec![
                    Cow::Borrowed(FONT_SANS),
                    Cow::Borrowed(FONT_MONO),
                    Cow::Borrowed(FONT_ARABIC),
                ])
                .unwrap();
        });
        let (tx, worker_rx) = mpsc::channel();
        let (reply_tx, reply_rx) = mpsc::channel();
        let mut view = None;
        let handle = cx.add_window(|window, cx| {
            let console =
                cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(reply_rx)), false, window, cx));
            console.update(cx, |console, cx| {
                console.model_path = "fixture.gguf".into();
                console
                    .inputs
                    .model
                    .update(cx, |input, cx| input.set_value("fixture.gguf", cx));
            });
            view = Some(console.clone());
            Root::new(console, window, cx)
        });
        let console = view.unwrap();
        cx.update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            window.click(SharedString::from("nav:experiments"), cx);
            window.click(SharedString::from("step:intervention"), cx);
            assert_eq!(console.read(cx).step, WorkspaceStep::Intervention);
            window.click(SharedString::from("operation-card:zero"), cx);
            assert_eq!(console.read(cx).op, "zero");
            console.update(cx, |console, cx| {
                console.apply_preset(Preset::ArabicMorphology, cx)
            });
            window.render_frame(cx);
            let console = console.read(cx);
            assert!(!console.prompt.is_ascii());
            assert!(console.build_run_request().is_ok());
        })
        .unwrap();
        cx.update_window(handle.into(), |_, window, cx| {
            window.click(SharedString::from("step:prompt"), cx);
            let input = console.read(cx).inputs.prompt.clone();
            input.update(cx, |input, cx| input.focus_for_test(window, cx));
            window.press(
                if cfg!(target_os = "macos") {
                    "cmd-a"
                } else {
                    "ctrl-a"
                },
                cx,
            );
            window.input("مرحبا Ember\nاختبار", cx);
        })
        .unwrap();
        cx.update_window(handle.into(), |_, window, cx| {
            assert_eq!(console.read(cx).prompt, "مرحبا Ember\nاختبار");
            window.press(
                if cfg!(target_os = "macos") {
                    "cmd-z"
                } else {
                    "ctrl-z"
                },
                cx,
            );
        })
        .unwrap();
        cx.update_window(handle.into(), |_, window, cx| {
            assert_ne!(console.read(cx).prompt, "مرحبا Ember\nاختبار");
            window.click(SharedString::from("nav:experiments"), cx);
            window.click(SharedString::from("step:intervention"), cx);
            // The advanced controls live in the inspector, and the inspector
            // starts closed -- so this has to open the inspector first. This
            // test used to find `advanced-toggle` without doing that, which
            // was only possible because a second, unclosable copy of the
            // inspector was being rendered alongside the real one.
            assert!(
                window
                    .try_find(SharedString::from("advanced-toggle"))
                    .is_none(),
                "advanced controls must not be reachable while the inspector is closed"
            );
            window.click(SharedString::from("inspector-toggle"), cx);
            window.click(SharedString::from("advanced-toggle"), cx);
            window.click(SharedString::from("picker:Site"), cx);
        })
        .unwrap();
        cx.run_until_parked();
        cx.update_window(handle.into(), |_, window, cx| {
            window.input("before-logits", cx);
        })
        .unwrap();
        cx.wait_for(
            handle.into(),
            std::time::Duration::from_secs(1),
            |window, _| {
                window
                    .try_find(SharedString::from("choice:before-logits"))
                    // Human-facing wording, not the frozen hook identifier:
                    // the picker shows "Before output head" while the value
                    // committed to the spec stays "before-logits".
                    .is_some_and(|item| item.label() == Some("Before output head"))
            },
        )
        .await;
        cx.update_window(handle.into(), |_, window, cx| {
            window.press("down", cx);
            window.press("enter", cx);
        })
        .unwrap();
        cx.run_until_parked();
        cx.update_window(handle.into(), |_, window, cx| {
            assert_eq!(console.read(cx).site, "before-logits");
            window.click(SharedString::from("step:results"), cx);
            window.click(SharedString::from("btn:Run experiment"), cx);
            assert_eq!(console.read(cx).status, super::Status::Preparing);
            assert!(matches!(worker_rx.try_recv().unwrap(), super::WorkerMsg::Prepare(path) if path == "fixture.gguf"));
            let loading = SharedString::from("btn:Loading model…");
            assert!(window.find(loading.clone()).visible());
            window.click(loading, cx);
            assert!(worker_rx.try_recv().is_err(), "disabled run must not submit another job");
            reply_tx.send(super::WorkerReply::Prepared(Box::new(Err("fixture load failure".into())))).unwrap();
            console.update(cx, |console, cx| { console.drain_replies(cx); cx.notify(); });
            window.render_frame(cx);
            assert_eq!(console.read(cx).status, super::Status::Idle);
            assert_eq!(console.read(cx).error.as_deref(), Some("fixture load failure"));
            window.click(SharedString::from("btn:Run experiment"), cx);
            assert!(matches!(worker_rx.try_recv().unwrap(), super::WorkerMsg::Prepare(_)));
        }).unwrap();
    }
}



/// Run a batch of interventions on one loaded model and report which change
/// the words. Output is plain text on stderr; nothing is rendered or saved.
#[cfg(all(target_os = "macos", feature = "gui-tests"))]
fn probe_examples(model: String) -> anyhow::Result<()> {
    let platform = gpui_kit::platform::current_platform(true);
    let mut context = HeadlessAppContext::with_platform(
        platform.text_system(),
        Arc::new(icons::Assets),
        gpui_kit::platform::current_headless_renderer,
    );
    context.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, rx) = spawn_worker(ember::quant_k::KStrategy::Auto, false);
    let mut console = None;
    let handle = context.open_window(size(px(1200.), px(800.)), |window, cx| {
        let view = cx.new(|cx| Console::new(tx, rx, true, window, cx));
        console = Some(view.clone());
        cx.new(|cx| gpui_kit::component::Root::new(view, window, cx))
    })?;
    let console = console.unwrap();
    let prompt = std::env::var("EMBER_GUI_TEST_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    // (op, site, layer, value)
    let configs: Vec<(&str, &str, &str, &str)> = vec![
        ("zero", "after-layer", "15", "1.0"),
        ("zero", "after-layer", "14", "1.0"),
        ("zero", "after-layer", "12", "1.0"),
        ("zero", "after-layer", "8", "1.0"),
        ("zero", "after-mlp", "12", "1.0"),
        ("scale", "after-layer", "14", "8.0"),
        ("scale", "after-layer", "10", "4.0"),
        ("scale", "after-mlp", "12", "10.0"),
        ("scale", "after-mlp", "8", "-2.0"),
        ("scale", "after-layer", "6", "0.0"),
    ];
    for (op, site, layer, value) in configs {
        context.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, cx| {
                console.model_path = model.clone();
                console
                    .inputs
                    .model
                    .update(cx, |input, cx| input.set_value(model.clone(), cx));
                console.op = op.into();
                console.site = site.into();
                console.layer = layer.into();
                console.value = value.into();
                console.token = "prompt-final".into();
                    console.prompt = prompt.clone();
                for (input, text) in [
                    (console.inputs.layer.clone(), layer.to_string()),
                    (console.inputs.value.clone(), value.to_string()),
                    (console.inputs.prompt.clone(), prompt.clone()),
                ] {
                    console.set_input_value(input, text, cx);
                }
                console.comparison = None;
                console.baseline = None;
                console.error = None;
                console.run();
            });
        })?;
        let started = std::time::Instant::now();
        loop {
            context.advance_clock(Duration::from_millis(50));
            context.run_until_parked();
            std::thread::sleep(Duration::from_millis(50));
            context.update_window(handle.into(), |_, _, cx| {
                console.update(cx, |console, cx| {
                    console.drain_replies(cx);
                });
            })?;
            let (done, error) = console.read_with(&context, |c, _| {
                (c.comparison.is_some() && c.status == Status::Idle, c.error.clone())
            });
            if let Some(error) = error {
                eprintln!("PROBE {op} {site} L{layer} x{value}: ERROR {error}");
                break;
            }
            if done {
                let line = console.read_with(&context, |c, _| {
                    let comparison = c.comparison.as_ref().unwrap();
                    format!(
                        "changed={} first_step={:?} peak={:?}@{:?}\n    base: {:?}\n    intv: {:?}",
                        !comparison.generated_text_equal,
                        comparison.first_token_divergence,
                        comparison.landmarks.peak_relative_l2,
                        comparison.landmarks.peak_layer,
                        c.baseline.as_ref().map(|b| b.text.trim().chars().take(70).collect::<String>()),
                        c.intervention.as_ref().map(|b| b.text.trim().chars().take(70).collect::<String>()),
                    )
                });
                eprintln!("PROBE {op} {site} L{layer} x{value}: {line}");
                break;
            }
            anyhow::ensure!(started.elapsed() < Duration::from_secs(300), "probe run timed out");
        }
    }
    Ok(())
}

/// The first-run path, end to end: Home, an example, the three steps, a real
/// run on a real model with a live worker, then the result, Copy summary and
/// Runs. A frame is saved whenever the run's status changes, so the progress
/// steps are seen as they advance rather than inferred.
#[cfg(all(target_os = "macos", feature = "gui-tests"))]
fn render_live_flow(directory: &std::path::Path, model: String) -> anyhow::Result<()> {
    use gpui_kit::test::TestWindowExt as _;
    std::fs::create_dir_all(directory)?;
    let platform = gpui_kit::platform::current_platform(true);
    let mut context = HeadlessAppContext::with_platform(
        platform.text_system(),
        Arc::new(icons::Assets),
        gpui_kit::platform::current_headless_renderer,
    );
    context.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let (tx, rx) = spawn_worker(ember::quant_k::KStrategy::Auto, false);
    let mut console = None;
    let handle = context.open_window(size(px(1728.), px(1092.)), |window, cx| {
        let view = cx.new(|cx| Console::new(tx, rx, true, window, cx));
        console = Some(view.clone());
        cx.new(|cx| gpui_kit::component::Root::new(view, window, cx))
    })?;
    let console = console.unwrap();
    let mut frame = 0usize;
    let mut shot = |context: &mut HeadlessAppContext, tag: &str| -> anyhow::Result<()> {
        context.update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
        })?;
        context.run_until_parked();
        context.update_window(handle.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
        })?;
        frame += 1;
        context
            .capture_screenshot(handle.into())?
            .save(directory.join(format!("live-{frame:02}-{tag}.png")))?;
        Ok(())
    };
    let click = |context: &mut HeadlessAppContext, id: &str| -> anyhow::Result<()> {
        let id = SharedString::from(id.to_string());
        context.update_window(handle.into(), |_, window, cx| window.click(id, cx))?;
        Ok(())
    };
    context.update_window(handle.into(), |_, _, cx| {
        console.update(cx, |console, cx| {
            console.appearance = AppearanceMode::Dark;
            console.model_path = model.clone();
            console
                .inputs
                .model
                .update(cx, |input, cx| input.set_value(model.clone(), cx));
            console.sync_kit_theme(cx);
            cx.notify();
        });
    })?;
    shot(&mut context, "home")?;
    click(&mut context, "home-example")?;
    anyhow::ensure!(
        console.read_with(&context, |c, _| c.step) == WorkspaceStep::Prompt,
        "Try an example did not land on the Prompt step"
    );
    shot(&mut context, "prompt")?;
    click(&mut context, "btn:Continue: Intervention")?;
    shot(&mut context, "intervention")?;
    click(&mut context, "btn:Continue: Review")?;
    shot(&mut context, "review-before-run")?;
    click(&mut context, "btn:Run experiment")?;

    let started = std::time::Instant::now();
    let mut last = console.read_with(&context, |c, _| c.status);
    shot(&mut context, &format!("run-{last:?}").to_lowercase())?;
    let mut ticks = 0u32;
    loop {
        context.advance_clock(Duration::from_millis(50));
        context.run_until_parked();
        std::thread::sleep(Duration::from_millis(50));
        context.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, cx| {
                if console.drain_replies(cx) {
                    cx.notify();
                }
            });
        })?;
        let (status, done, error) = console.read_with(&context, |c, _| {
            (c.status, c.baseline.is_some() && c.status == Status::Idle, c.error.clone())
        });
        ticks += 1;
        if status != last {
            shot(&mut context, &format!("run-{status:?}").to_lowercase())?;
            last = status;
        } else if ticks % 40 == 0 && status != Status::Idle {
            shot(&mut context, &format!("run-{status:?}-still").to_lowercase())?;
        }
        if let Some(error) = error {
            anyhow::bail!("the run reported an error: {error}");
        }
        if done {
            break;
        }
        anyhow::ensure!(started.elapsed() < Duration::from_secs(400), "run timed out");
    }
    eprintln!("live run finished in {:.1}s", started.elapsed().as_secs_f32());
    shot(&mut context, "result-overview")?;
    click(&mut context, "review-copy")?;
    let copied = context.update(|cx| cx.read_from_clipboard()).and_then(|item| item.text());
    eprintln!(
        "copied summary: {}",
        copied.as_deref().map_or("NOTHING".to_string(), |text| format!("{} bytes", text.len()))
    );
    if let Some(text) = &copied {
        std::fs::write(directory.join("copied-summary.md"), text)?;
    }
    shot(&mut context, "result-copied")?;
    click(&mut context, "result:layers")?;
    shot(&mut context, "result-layers")?;
    click(&mut context, "result:tokens")?;
    shot(&mut context, "result-tokens")?;
    click(&mut context, &format!("nav:{}", View::Runs.key()))?;
    shot(&mut context, "runs")?;
    click(&mut context, "run-open:1")?;
    anyhow::ensure!(
        console.read_with(&context, |c, _| c.saved_run) == Some(1),
        "Open on the saved run did not reopen it"
    );
    shot(&mut context, "reopened-from-history")?;
    click(&mut context, &format!("nav:{}", View::Home.key()))?;
    shot(&mut context, "home-after")?;
    let draft = console.read_with(&context, |c, _| c.store.draft.is_some());
    eprintln!("draft after a completed run: {}", if draft { "PRESENT (unexpected)" } else { "none" });
    anyhow::ensure!(!draft, "a finished run left a draft to resume");
    Ok(())
}

/// Offscreen test scenes use the production Console render tree, CoreText, and
/// Metal. This exercises layout without automating another desktop application.
#[cfg(all(target_os = "macos", feature = "gui-tests"))]
fn render_test_artifacts(directory: &std::path::Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(directory)?;
    // The harness drives the real console, whose persistence writes history
    // and workspace flags. Point it at a scratch directory first so fixture
    // runs never land in the developer's own ~/.config/ember. Nothing else has
    // started a thread yet, which is what makes changing the environment sound.
    if std::env::var_os("EMBER_GUI_TEST_KEEP_CONFIG").is_none() {
        let scratch = std::env::temp_dir().join(format!("ember-render-config-{}", std::process::id()));
        // SAFETY: called at the top of the harness, before any worker thread
        // exists, so no other thread can be reading the environment.
        unsafe { std::env::set_var("XDG_CONFIG_HOME", scratch) };
    }
    let platform = gpui_kit::platform::current_platform(true);
    let mut context = HeadlessAppContext::with_platform(
        platform.text_system(),
        Arc::new(icons::Assets),
        gpui_kit::platform::current_headless_renderer,
    );
    context.update(|cx| {
        gpui_kit::init(cx);
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .unwrap();
    });
    let mut real_replies: Option<(WorkerReply, WorkerReply)> = None;
    // `EMBER_GUI_TEST_WINDOW=1728x1092` renders one window at that logical
    // size instead of the two defaults -- for previewing the console at a
    // demo machine's full-screen size without touching the visual tests.
    let sizes: Vec<(&'static str, f32, f32)> = std::env::var("EMBER_GUI_TEST_WINDOW")
        .ok()
        .and_then(|spec| {
            let (width, height) = spec.split_once('x')?;
            Some(vec![(
                "display",
                width.trim().parse().ok()?,
                height.trim().parse().ok()?,
            )])
        })
        .unwrap_or_else(|| vec![("standard", 1180., 720.), ("minimum", 980., 620.)]);
    for (name, width, height) in sizes {
        let (tx, _worker) = mpsc::channel();
        let (reply, rx) = mpsc::channel();
        let mut console = None;
        let handle = context.open_window(size(px(width), px(height)), |window, cx| {
            let view = cx.new(|cx| Console::new(tx, Arc::new(Mutex::new(rx)), false, window, cx));
            console = Some(view.clone());
            cx.new(|cx| gpui_kit::component::Root::new(view, window, cx))
        })?;
        let console = console.unwrap();
        for (appearance, mode) in [
            ("light", AppearanceMode::Light),
            ("dark", AppearanceMode::Dark),
        ] {
            // The app has a shell with several destinations, so the baseline
            // has to cover the shell and the experiment workspace, not just
            // the old single screen.
            for (view, step, inspector) in [
                (View::Home, WorkspaceStep::Prompt, false),
                (View::Models, WorkspaceStep::Prompt, false),
                (View::Runs, WorkspaceStep::Prompt, false),
                (View::Settings, WorkspaceStep::Prompt, false),
                (View::Experiment, WorkspaceStep::Prompt, false),
                (View::Experiment, WorkspaceStep::Intervention, false),
                (View::Experiment, WorkspaceStep::Review, true),
            ] {
                context.update_window(handle.into(), |_, window, cx| {
                    console.update(cx, |console, cx| {
                        console.appearance = mode;
                        console.view = view;
                        console.step = step;
                        console.inspector_open = inspector;
                        // The seed comparison persists on the console once the
                        // Review scene has run, and scenes share one console
                        // per window. Without this reset every scene after the
                        // first Review reports "measured" on steps that have
                        // not measured anything, and the fixtures lie.
                        if !(view == View::Experiment && step == WorkspaceStep::Review) {
                            console.baseline = None;
                            console.intervention = None;
                            console.comparison = None;
                            console.layer_series = Arc::from([]);
                            console.verification = None;
                            console.restore = None;
                            console.result_context = None;
                            console.last_metrics = None;
                        }
                        console.sync_kit_theme(cx);
                        cx.notify();
                    });
                    window.draw(cx).clear(cx);
                })?;
                context.run_until_parked();
                context.update_window(handle.into(), |_, window, cx| {
                    window.draw(cx).clear(cx);
                })?;
                let file = match view {
                    View::Home => "home".to_string(),
                    View::Models => "models".to_string(),
                    View::Runs => "runs".to_string(),
                    View::Settings => "settings".to_string(),
                    View::Experiment => {
                        format!(
                            "experiment-{}{}",
                            step.number(),
                            if inspector { "-inspector" } else { "" }
                        )
                    }
                };
                context
                    .capture_screenshot(handle.into())?
                    .save(directory.join(format!("{name}-{appearance}-{file}.png")))?;
            }
            // The command palette overlays every page; capture it once per
            // theme so the dim backdrop and the floating panel are reviewed
            // against both canvases.
            context.update_window(handle.into(), |_, window, cx| {
                console.update(cx, |console, cx| {
                    console.palette_open = true;
                    cx.notify();
                });
                window.draw(cx).clear(cx);
            })?;
            context.run_until_parked();
            context.update_window(handle.into(), |_, window, cx| {
                window.draw(cx).clear(cx);
            })?;
            context
                .capture_screenshot(handle.into())?
                .save(directory.join(format!("{name}-{appearance}-palette.png")))?;
            context.update_window(handle.into(), |_, _, cx| {
                console.update(cx, |console, _| console.palette_open = false);
            })?;
            // Interaction states: hover and keyboard focus are invisible in
            // the resting-state scenes, and they are where a console reads as
            // polished or as a form.
            {
                use gpui_kit::test::TestWindowExt as _;
                let hover_nav = format!("nav:{}", View::Models.key());
                for (state, view, step, hover, tabs) in [
                    ("hover-nav", View::Home, WorkspaceStep::Prompt, Some(hover_nav), 0),
                    (
                        "hover-tile",
                        View::Experiment,
                        WorkspaceStep::Intervention,
                        Some("operation-card:zero".to_string()),
                        0,
                    ),
                    (
                        "hover-preset",
                        View::Experiment,
                        WorkspaceStep::Prompt,
                        Some("preset:Zero a middle layer".to_string()),
                        0,
                    ),
                    ("focus-tab", View::Experiment, WorkspaceStep::Prompt, None, 3),
                    ("focus-click", View::Experiment, WorkspaceStep::Prompt, Some("generation-length:24".to_string()), 0),
                ] {
                    context.update_window(handle.into(), |_, window, cx| {
                        console.update(cx, |console, cx| {
                            console.appearance = mode;
                            console.view = view;
                            console.step = step;
                            console.inspector_open = false;
                            console.sync_kit_theme(cx);
                            cx.notify();
                        });
                        window.draw(cx).clear(cx);
                        if let Some(id) = &hover {
                            if state == "focus-click" {
                                // A click focuses the button; the kit paints
                                // its ring whenever it holds focus.
                                window.click(SharedString::from(id.clone()), cx);
                            } else {
                                window.hover(SharedString::from(id.clone()), cx);
                            }
                        }
                        for _ in 0..tabs {
                            if let Ok(stroke) = Keystroke::parse("tab") {
                                window.dispatch_keystroke(stroke, cx);
                            }
                            window.draw(cx).clear(cx);
                        }
                        window.draw(cx).clear(cx);
                    })?;
                    context.run_until_parked();
                    context.update_window(handle.into(), |_, window, cx| {
                        window.draw(cx).clear(cx);
                    })?;
                    context
                        .capture_screenshot(handle.into())?
                        .save(directory.join(format!("{name}-{appearance}-state-{state}.png")))?;
                }
            }
        }
        // A run in flight: the progress steps a first-time user waits on.
        for (label_name, status) in [("loading", Status::Preparing), ("running", Status::Running)] {
            context.update_window(handle.into(), |_, window, cx| {
                console.update(cx, |console, cx| {
                    console.appearance = AppearanceMode::Dark;
                    console.view = View::Experiment;
                    console.step = WorkspaceStep::Review;
                    console.inspector_open = false;
                    console.status = status;
                    console.sync_kit_theme(cx);
                });
                window.draw(cx).clear(cx);
            })?;
            context.run_until_parked();
            context.update_window(handle.into(), |_, window, cx| {
                window.draw(cx).clear(cx);
            })?;
            context
                .capture_screenshot(handle.into())?
                .save(directory.join(format!("{name}-progress-{label_name}.png")))?;
        }
        context.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, _| console.status = Status::Idle);
        })?;
        // The built-in sample result, which needs no model.
        for (appearance, mode) in [
            ("light", AppearanceMode::Light),
            ("dark", AppearanceMode::Dark),
        ] {
            for result in [ResultView::Overview, ResultView::Layers, ResultView::Tokens] {
                context.update_window(handle.into(), |_, window, cx| {
                    console.update(cx, |console, cx| {
                        console.appearance = mode;
                        console.inspector_open = true;
                        if !console.sample {
                            console.show_sample(cx);
                        }
                        console.result_view = result;
                        console.sync_kit_theme(cx);
                    });
                    window.draw(cx).clear(cx);
                })?;
                context.run_until_parked();
                context.update_window(handle.into(), |_, window, cx| {
                    window.draw(cx).clear(cx);
                })?;
                context
                    .capture_screenshot(handle.into())?
                    .save(directory.join(format!(
                        "{name}-{appearance}-sample-{}.png",
                        result.label().replace(' ', "-")
                    )))?;
            }
        }
        context.update_window(handle.into(), |_, _, cx| {
            console.update(cx, |console, _| {
                console.sample = false;
                console.baseline = None;
                console.intervention = None;
                console.comparison = None;
                console.layer_series = Arc::from([]);
                console.result_context = None;
            });
        })?;
        if let Ok(model) = std::env::var("EMBER_GUI_TEST_MODEL") {
            let config = context.update_window(handle.into(), |_, _, cx| {
                console.update(cx, |console, _| {
                    console.model_path = model.clone();
                    console.prompt = "The capital of France is".into();
                    console.max_tokens = "4".into();
                    console.pending_context = Some(console.form_values());
                    parse_run_request(&console.form_values().build_run_request().unwrap()).unwrap()
                })
            })?;
            if real_replies.is_none() {
                eprintln!("Rendering real-model result scenes: preparing and running four tokens");
                let (worker, receiver) = spawn_worker(ember::quant_k::KStrategy::Auto, false);
                worker.send(WorkerMsg::Prepare(model))?;
                let prepared = receiver
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(300))?;
                if let WorkerReply::Prepared(info) = &prepared {
                    info.as_ref()
                        .as_ref()
                        .map_err(|error| anyhow::anyhow!(error.clone()))?;
                }
                worker.send(WorkerMsg::Run(config))?;
                let completed = receiver
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(300))?;
                if let WorkerReply::RunDone(result) = &completed {
                    let bundle = result
                        .as_ref()
                        .as_ref()
                        .map_err(|error| anyhow::anyhow!(error.clone()))?;
                    anyhow::ensure!(
                        bundle.verification.ok,
                        "render fixture bundle failed verification"
                    );
                    eprintln!("Render fixture: {}", bundle.baseline.bundle_dir);
                }
                real_replies = Some((prepared, completed));
            }
            let (prepared, completed) = real_replies.as_ref().unwrap();
            reply.send(prepared.clone())?;
            reply.send(completed.clone())?;
            context.update_window(handle.into(), |_, _, cx| {
                console.update(cx, |console, cx| {
                    console.drain_replies(cx);
                    cx.notify();
                });
            })?;
            for (appearance, mode) in [
                ("light", AppearanceMode::Light),
                ("dark", AppearanceMode::Dark),
            ] {
                for result in ResultView::ALL {
                    context.update_window(handle.into(), |_, window, cx| {
                        console.update(cx, |console, cx| {
                            console.appearance = mode;
                            console.step = WorkspaceStep::Review;
                            console.result_view = result;
                            console.sync_kit_theme(cx);
                            cx.notify();
                        });
                        window.draw(cx).clear(cx);
                    })?;
                    context.run_until_parked();
                    context.update_window(handle.into(), |_, window, cx| {
                        window.draw(cx).clear(cx);
                    })?;
                    for _ in 0..20 {
                        context.advance_clock(Duration::from_millis(50));
                        context.run_until_parked();
                        context.update_window(handle.into(), |_, window, cx| {
                            window.draw(cx).clear(cx);
                        })?;
                    }
                    context
                        .capture_screenshot(handle.into())?
                        .save(directory.join(format!(
                            "{name}-{appearance}-result-{}.png",
                            result.label().replace(' ', "-")
                        )))?;
                }
            }
            // Presentation mode over the results, once per window size.
            for result in [ResultView::Overview, ResultView::Layers] {
                context.update_window(handle.into(), |_, window, cx| {
                    console.update(cx, |console, cx| {
                        console.appearance = AppearanceMode::Dark;
                        console.result_view = result;
                        if console.presentation.is_none() {
                            console.toggle_presentation(cx);
                        }
                    });
                    window.draw(cx).clear(cx);
                })?;
                context.run_until_parked();
                context.update_window(handle.into(), |_, window, cx| {
                    window.draw(cx).clear(cx);
                })?;
                context
                    .capture_screenshot(handle.into())?
                    .save(directory.join(format!(
                        "{name}-presentation-{}.png",
                        result.label().replace(' ', "-")
                    )))?;
            }
            context.update_window(handle.into(), |_, _, cx| {
                console.update(cx, |console, cx| {
                    if console.presentation.is_some() {
                        console.toggle_presentation(cx);
                    }
                });
            })?;
        }
    }
    Ok(())
}
