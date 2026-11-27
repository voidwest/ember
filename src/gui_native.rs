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
        EnterPresentation,
        StartExperiment,
        ReplayLastRun,
        OpenSampleResult,
        ShowShortcuts,
        OpenRepository,
        CancelRun,
    ]
);

/// Where the project lives; the Help menu opens it.
const REPOSITORY_URL: &str = "https://github.com/voidwest/ember";

mod actions;
mod chart;
mod commands;
mod sample;
use sample::*;
mod compare;
mod components;
mod export;
mod form;
mod harness;
mod history;
mod icons;
mod input;
mod menu;
mod palette;
mod picker;
mod runs_table;
mod skeleton;
mod spark;
mod store_writer;
mod sweep;
mod theme;
mod views;
mod worker;
mod workspace;

use components::*;
use form::{FormValues, Inputs};
#[cfg(feature = "gui-tests")]
use harness::seed_comparison;
#[cfg(all(target_os = "macos", feature = "gui-tests"))]
use harness::{probe_examples, render_live_flow, render_test_artifacts};
use harness::{seed_runs_requested, seed_store};
use history::{open_store, record_bundles, record_result};
use input::{InputEvent, InputId, InputKind, TextInput};
use menu::{app_menus, register_menu_actions};
use palette::{Command, PaletteMode};
use runs_table::{fmt_bytes, quant_of, relative_time, truncate_path_start, RunsDelegate};
use sweep::Sweep;
use theme::{AppearanceMode, Colors, Radius, Space, Type};
use worker::{spawn_worker, WorkerMsg, WorkerReply};
use workspace::Reference;

// ---------------------------------------------------------------------------
// embedded fonts (SIL OFL 1.1, see src/gui_fonts/LICENSE.txt)
// ---------------------------------------------------------------------------

const FONT_SANS: &[u8] = include_bytes!("gui_fonts/NotoSans-Regular.ttf");
const FONT_MONO: &[u8] = include_bytes!("gui_fonts/NotoSansMono-Regular.ttf");
const FONT_ARABIC: &[u8] = include_bytes!("gui_fonts/NotoNaskhArabic-Regular.ttf");
/// The interface face. On macOS this is the system font (SF Pro): it ships in
/// every weight, so headings can carry emphasis by weight instead of by size
/// alone, and it is what a native Mac app reads in. Elsewhere the bundled
/// Noto Sans, which is Regular only.
#[cfg(target_os = "macos")]
const FONT_SANS_NAME: &str = ".SystemUIFont";
#[cfg(not(target_os = "macos"))]
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

const EXECUTIONS: [&str; 3] = ["reference", "planned", "planned-fused"];

/// Whether the hook stage id carries a per-layer tensor.
fn per_layer(site: &str) -> bool {
    ember::v05::hook::SemanticHookSite::ALL
        .iter()
        .any(|hook| hook.stage_id() == site && hook.is_per_layer())
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
        "before-o-proj" => "Attention heads",
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
        "before-o-proj" => "before-o-proj",
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

/// The most rows Go to run lists at once.
const GO_TO_RUN_LIMIT: usize = 50;

/// A saved run as Go to run lists it: the number, then the prompt on one
/// line, short enough to scan. The prompt is a bidi isolate, so an Arabic
/// prompt does not turn the row right-to-left and move the number to its end.
fn run_palette_label(run: &RunRecord) -> String {
    let prompt = run.prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    format!(
        "#{}  \u{2068}{}\u{2069}",
        run.number,
        truncate_chars(&prompt, 64)
    )
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
// app state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Idle,
    Preparing,
    Running,
    Restoring,
    /// Cancel was pressed; waiting for the worker to reach its next check
    /// point (a decode step) and confirm nothing was kept.
    Cancelling,
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
    /// Only offered once a sweep has finished.
    Sweep,
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
            Self::Sweep => "sweep",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Layers => "Layers",
            Self::Tokens => "Tokens",
            Self::Trace => "Raw trace",
            Self::Sweep => "Sweep",
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

    /// Only the render harness labels its scenes by step number; it is
    /// built on macOS with `gui-tests` only.
    #[cfg(all(target_os = "macos", feature = "gui-tests"))]
    fn number(self) -> &'static str {
        match self {
            Self::Prompt => "1",
            Self::Intervention => "2",
            Self::Review => "3",
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

struct Console {
    // worker
    worker_tx: mpsc::Sender<WorkerMsg>,
    reply_rx: Arc<Mutex<mpsc::Receiver<WorkerReply>>>,
    // model
    /// Lazily created: the table needs a `Window` to build, which the
    /// constructor does not have. Kept out of `Console::new` for that reason.
    runs_table: Option<Entity<TableState<RunsDelegate>>>,
    model_options: Vec<String>,
    /// File sizes for `model_options`, read once at discovery rather than
    /// with a `stat` per model per frame.
    model_sizes: std::collections::HashMap<String, Option<u64>>,
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
    palette_mode: PaletteMode,
    palette_input: Entity<TextInput>,
    // spec view
    spec_open: bool,
    spec_input: Entity<TextInput>,
    /// The form's spec as the view last showed it.
    spec_shown: String,
    /// The view holds an edit the form could not take; it follows the form
    /// again after Revert or a valid edit.
    spec_held: bool,
    /// Why the view holds its edit, or why the form has no spec.
    spec_note: Option<String>,
    step: WorkspaceStep,
    view: View,
    /// The Review page is showing the built-in illustrative sample, not a run.
    sample: bool,
    /// Which saved run the Review page is showing, when it is one from
    /// History rather than a live result.
    saved_run: Option<u64>,
    /// The result summary was just copied; shown on the button until the
    /// user moves on.
    copied: bool,
    /// A result pinned to compare later runs against.
    reference: Option<Reference>,
    /// A sweep over layers, running or finished.
    sweep: Option<Sweep>,
    /// A sweep was asked for before the model was loaded; start it when ready.
    pending_sweep: bool,
    /// Title and text for a result opened from somewhere other than a run.
    opened_note: Option<(String, String)>,
    /// Bumped whenever a new result lands, to replay the results fade-in.
    result_epoch: u64,
    /// A short confirmation shown bottom-right ("Copied", "Saved"), and the
    /// id that lets a later one replace it without the earlier timer clearing it.
    toast: Option<(String, u64)>,
    /// The compact examples list in the setup pane.
    examples_open: bool,
    /// The Model section of the setup pane, expanded while a model is loaded.
    model_open: bool,
    sidebar_open: bool,
    /// Presentation mode, and the panes it hid so leaving restores them.
    presentation: Option<bool>,
    advanced_open: bool,
    pending_run: bool,
    pending_context: Option<FormValues>,
    result_context: Option<FormValues>,
    store: AppStore,
    /// Read failure, surfaced rather than swallowed: a store we could not
    /// parse is not the same as a store with no runs.
    store_error: Option<String>,
    /// Where history is written back, or `None` when this session must not
    /// write it: the file could not be read (overwriting it would destroy the
    /// history it holds), the store is a fixture, or this is a unit test.
    store_path: Option<std::path::PathBuf>,
    /// Writes history off the UI thread; started on the first save, so a
    /// session that never writes never starts it.
    store_writer: Option<store_writer::StoreWriter>,
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
    /// The token of the run or restore in flight; firing it cancels.
    run_cancel: Option<ember::cancel::CancelToken>,
    /// The last run was cancelled: shown as a notice until the next one.
    cancelled: bool,
    /// Runs selected on the Runs page for comparison, in selection order.
    compare_picks: Vec<u64>,
    /// The two runs the comparison page is showing, when it is open.
    comparing: Option<(u64, u64)>,
    /// The history row whose export strip is open.
    export_run: Option<u64>,
    /// A history run being re-run to write its bundle again.
    rebundle: Option<u64>,
    /// The outcome of the last export action, shown in the strip.
    export_note: Option<String>,
    /// Render caches for filesystem facts about persisted paths: rendering
    /// runs every frame, and a stored path on a slow or unreachable mount
    /// must not stall each one. Keyed by what they were computed from.
    export_cache: std::cell::RefCell<Option<export::ExportCache>>,
    bundle_dir_check: std::cell::RefCell<Option<(String, bool)>>,
}

impl Console {
    fn new(
        worker_tx: mpsc::Sender<WorkerMsg>,
        reply_rx: Arc<Mutex<mpsc::Receiver<WorkerReply>>>,
        system_dark: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        // Filled in by `start_model_discovery`, off the UI thread: scanning
        // the working directory must not hold up the window.
        let models: Vec<String> = Vec::new();
        let model_path = String::new();
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
        let (store, store_error, store_path) =
            if cfg!(feature = "gui-tests") && seed_runs_requested() {
                (seed_store(), None, None)
            } else if cfg!(test) {
                // Unit tests drive the real console; they must neither read nor
                // write the developer's own history.
                (AppStore::default(), None, None)
            } else if let (Some(path), Some(legacy)) =
                (app_store::store_path(), app_store::legacy_store_path())
            {
                open_store(path, &legacy)
            } else {
                (
                    AppStore::default(),
                    Some(
                        "no private configuration directory (HOME unset and the temp directory \
                         is not usable) -- history is not saved this session"
                            .to_string(),
                    ),
                    None,
                )
            };
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
                PaletteMode::Commands.placeholder(),
                &colors,
                window,
                cx,
            )
        });
        cx.subscribe(&palette_input, |console, _input, event: &InputEvent, cx| {
            console.input_changed(event, cx);
        })
        .detach();
        let spec_input = cx.new(|cx| {
            TextInput::new(
                InputId::Spec,
                InputKind::Code,
                String::new(),
                "",
                &colors,
                window,
                cx,
            )
        });
        cx.subscribe(&spec_input, |console, _input, event: &InputEvent, cx| {
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
                ComboId::Site => crate::gui::hook_stages()
                    .into_iter()
                    .map(String::from)
                    .collect(),
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
        // History writes are queued on a background thread; quitting must not
        // drop the last of them.
        cx.on_app_quit(|console: &mut Console, _| {
            console.flush_store();
            async {}
        })
        .detach();
        Console {
            worker_tx,
            reply_rx,
            runs_table: None,
            model_options: models,
            model_sizes: Default::default(),
            model_path,
            selected_model: None,
            site_options: crate::gui::hook_stages()
                .into_iter()
                .map(String::from)
                .collect(),
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
            palette_mode: PaletteMode::Commands,
            palette_input,
            spec_open: false,
            spec_input,
            spec_shown: String::new(),
            spec_held: false,
            spec_note: None,
            step: WorkspaceStep::Prompt,
            // Land on Home rather than inside a half-configured experiment.
            view: View::Home,
            // The context rail opens on demand; it used to be permanent and
            // permanently half-empty on the setup steps. Both panes remember
            // their last state across launches -- under gui-tests the files
            // are ignored so the fixtures stay deterministic.
            sample: false,
            copied: false,
            reference: None,
            sweep: None,
            pending_sweep: false,
            opened_note: None,
            examples_open: false,
            result_epoch: 0,
            toast: None,
            model_open: false,
            saved_run: None,
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
            store_path,
            store_writer: None,
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
            run_cancel: None,
            cancelled: false,
            compare_picks: Vec::new(),
            comparing: None,
            export_run: None,
            rebundle: None,
            export_note: None,
            export_cache: std::cell::RefCell::new(None),
            bundle_dir_check: std::cell::RefCell::new(None),
        }
    }
}

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------

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
                    // A transparent title bar: the traffic lights sit in Ember's
                    // own top bar instead of a separate system strip above it,
                    // and the bar moves the window itself (see `topbar`).
                    titlebar: Some(TitlebarOptions {
                        title: Some("Ember".into()),
                        appears_transparent: true,
                        traffic_light_position: Some(point(px(14.0), px(14.0))),
                    }),
                    app_owns_titlebar_drag: true,
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
                        console.open_where_you_left_off(cx);
                        console.spawn_poll(cx);
                        console.start_model_discovery(cx);
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
#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "gui-tests"))]
mod kit_tests;
