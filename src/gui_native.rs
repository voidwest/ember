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

mod chart;
mod compare;
mod components;
mod form;
mod harness;
mod history;
mod icons;
mod input;
mod menu;
mod palette;
mod picker;
mod runs_table;
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
use history::{open_store, record_result};
use input::{InputEvent, InputId, InputKind, TextInput};
use menu::{app_menus, register_menu_actions};
use palette::Command;
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
    palette_input: Entity<TextInput>,
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
            } else {
                open_store(app_store::store_path(), &app_store::legacy_store_path())
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
            sample: false,
            copied: false,
            reference: None,
            sweep: None,
            pending_sweep: false,
            opened_note: None,
            examples_open: false,
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
        }
    }

    /// Scan for models on a background thread and fill the model list when
    /// the scan finishes.
    fn start_model_discovery(&mut self, cx: &mut Context<Self>) {
        cx.spawn(|this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let mut cx = cx.clone();
            async move {
                let found = cx
                    .background_executor()
                    .spawn(async { scan_models() })
                    .await;
                let _ = this.update(&mut cx, |console, cx| console.models_discovered(found, cx));
            }
        })
        .detach();
    }

    /// Adopt a finished model scan. The first model becomes the selection only
    /// if the user has not already typed or picked one.
    fn models_discovered(&mut self, found: Vec<(String, Option<u64>)>, cx: &mut Context<Self>) {
        self.model_sizes = found.iter().cloned().collect();
        self.model_options = found.into_iter().map(|(path, _)| path).collect();
        if self.model_path.trim().is_empty()
            && let Some(first) = self.model_options.first().cloned()
        {
            self.model_path = first.clone();
            self.set_input_value(self.inputs.model.clone(), first, cx);
        }
        cx.notify();
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
        self.run();
    }

    /// Run with the settings on screen. Shared by the setup pane's button.
    fn run_now(&mut self) {
        self.advance_or_run();
    }

    fn validation_error(&self) -> Option<String> {
        self.build_run_request()
            .and_then(|request| parse_run_request(&request).map(|_| ()))
            .err()
    }

    /// Build the v0.5 request from the current form fields; the shared
    /// `parse_run_request` gate validates it exactly like the web console.
    fn build_run_request(&self) -> Result<RunRequest, String> {
        self.form_values().build_run_request()
    }

    fn send_run(&mut self, cfg: RunConfig) {
        self.sample = false;
        self.saved_run = None;
        self.opened_note = None;
        self.status = Status::Running;
        self.step = WorkspaceStep::Review;
        self.pending_context = Some(self.form_values());
        self.error = None;
        self.cancelled = false;
        let token = ember::cancel::CancelToken::new();
        self.run_cancel = Some(token.clone());
        let _ = self.worker_tx.send(WorkerMsg::Run(cfg, token));
    }

    fn send_restore(&mut self, cfg: RunConfig) {
        self.status = Status::Restoring;
        self.error = None;
        self.cancelled = false;
        let token = ember::cancel::CancelToken::new();
        self.run_cancel = Some(token.clone());
        let _ = self.worker_tx.send(WorkerMsg::Restore(cfg, token));
    }

    /// Whether Cancel has something to stop: a run or restore in flight, or
    /// a model load that a run is waiting on.
    fn can_cancel(&self) -> bool {
        match self.status {
            Status::Running | Status::Restoring => true,
            Status::Preparing => self.pending_run || self.pending_sweep,
            Status::Idle | Status::Cancelling => false,
        }
    }

    /// Stop the run in flight. Shared by the button, Esc, the palette and
    /// the menu.
    ///
    /// The token is checked at every decode step, so the worker stops within
    /// one token; the console waits in `Cancelling` for it to confirm, then
    /// returns to idle with nothing recorded. A model load cannot be
    /// interrupted, but a run waiting on one is dropped and the model is kept
    /// for next time. A running sweep stops too; its finished points stay.
    fn cancel_run(&mut self, cx: &mut Context<Self>) {
        if !self.can_cancel() {
            return;
        }
        if let Some(sweep) = self.sweep.as_mut() {
            sweep.stop = true;
        }
        if self.status == Status::Preparing {
            self.pending_run = false;
            self.pending_sweep = false;
            self.pending_context = None;
            self.cancelled = true;
        } else {
            if let Some(token) = &self.run_cancel {
                token.cancel();
            }
            self.status = Status::Cancelling;
        }
        cx.notify();
    }

    /// The worker confirmed a cancellation: back to idle, nothing kept.
    fn run_cancelled(&mut self, cx: &mut Context<Self>) {
        self.run_cancel = None;
        self.pending_context = None;
        self.pending_run = false;
        self.error = None;
        self.cancelled = true;
        self.status = Status::Idle;
        if self.sweep_running() {
            self.stop_sweep(cx);
            self.advance_sweep(cx);
        }
    }

    /// Drain the worker reply channel; returns true when anything changed.
    fn drain_replies(&mut self, cx: &mut Context<Self>) -> bool {
        let (mut replies, disconnected) = {
            let rx = self.reply_rx.lock().expect("reply receiver lock");
            let mut replies = Vec::new();
            let disconnected = loop {
                match rx.try_recv() {
                    Ok(reply) => replies.push(reply),
                    Err(mpsc::TryRecvError::Empty) => break false,
                    Err(mpsc::TryRecvError::Disconnected) => break true,
                }
            };
            (replies, disconnected)
        };
        // Report a dead worker once, when something was still waiting on it.
        if disconnected && self.busy() {
            replies.push(WorkerReply::Failed(
                "the model worker stopped; restart Ember to run again".to_string(),
            ));
        }
        if replies.is_empty() {
            return false;
        }
        for reply in replies {
            // A reply for a run the user already cancelled is discarded: it
            // finished before the worker reached a check point, and it must
            // not land in history or on screen.
            if self.status == Status::Cancelling {
                match reply {
                    WorkerReply::RunDone(result) => {
                        if let Ok(bundle) = result.as_ref() {
                            worker::discard_run_bundles(bundle);
                        }
                        self.run_cancelled(cx);
                        continue;
                    }
                    WorkerReply::RestoreDone(_) | WorkerReply::Cancelled => {
                        self.run_cancelled(cx);
                        continue;
                    }
                    _ => {}
                }
            }
            match reply {
                WorkerReply::Cancelled => self.run_cancelled(cx),
                WorkerReply::Failed(error) => {
                    self.run_cancel = None;
                    self.pending_run = false;
                    self.pending_context = None;
                    self.session = None;
                    self.error = Some(error);
                    self.status = Status::Idle;
                }
                // A load that finished for a model the user has since moved
                // away from: load the current one instead of claiming it.
                WorkerReply::Prepared(result) if matches!(&*result, Ok(info) if info.model_path != self.model_path.trim()) =>
                {
                    let pending_run = self.pending_run;
                    self.status = Status::Idle;
                    self.load();
                    self.pending_run = pending_run && self.status == Status::Preparing;
                }
                WorkerReply::Prepared(result) => match *result {
                    Ok(info) => {
                        self.session = Some(info);
                        self.model_open = false;
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
                        if self.pending_sweep {
                            self.start_sweep(cx);
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
                        self.run_cancel = None;
                        self.sample = false;
                        self.saved_run = None;
                        self.opened_note = None;
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
                        // Record the form as it was sent, not as it is now: it
                        // stays editable while a run is in flight.
                        let sent = self
                            .result_context
                            .clone()
                            .unwrap_or_else(|| self.form_values());
                        // A sweep's runs are kept on the sweep, not in history:
                        // sixteen near-identical rows would bury the runs the
                        // user chose to make.
                        let in_sweep = self.sweep_running();
                        if !in_sweep {
                            let number = self.store.next_run_number();
                            let now = unix_now();
                            self.store.push_run(RunRecord {
                                number,
                                finished_at: now,
                                model: model_display_name(&sent.model_path),
                                intervention: operation_label(&sent.op).to_string(),
                                hook: site_label(&sent.site).to_string(),
                                layer: per_layer(&sent.site)
                                    .then(|| sent.layer.parse::<u32>().ok())
                                    .flatten(),
                                duration_ms: Some(bundle.elapsed_ms_total.max(0.0).round() as u64),
                                baseline_tokens: Some(bundle.baseline.generated_tokens as u32),
                                intervention_tokens: Some(
                                    bundle.intervention.generated_tokens as u32,
                                ),
                                diverged_at_step: bundle
                                    .comparison
                                    .first_token_divergence
                                    .map(|step| step as u32),
                                outputs_equal: bundle.comparison.generated_text_equal,
                                verified: bundle.verification.ok,
                                pinned: false,
                                prompt: sent.prompt.clone(),
                                config: Some(sent.record_config()),
                                result: Some(record_result(&bundle)),
                            });
                            self.store.touch_model(&sent.model_path, now);
                            // The state that produced this run is the resume point.
                            self.save_draft();
                            self.persist();
                        }
                        self.status = Status::Idle;
                        if in_sweep {
                            self.sweep_point_done(cx);
                        }
                    }
                    Err(error) => {
                        self.run_cancel = None;
                        self.pending_context = None;
                        self.error = Some(error);
                        self.status = Status::Idle;
                        if self.sweep_running() {
                            self.stop_sweep(cx);
                            self.advance_sweep(cx);
                        }
                    }
                },
                WorkerReply::RestoreDone(result) => match *result {
                    Ok(bundle) => {
                        self.run_cancel = None;
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
                    self.cancelled = false;
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
        let selected = comparison
            .landmarks
            .peak_layer
            .or(comparison.landmarks.first_layer_divergence);
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

    /// Presentation mode: a larger text scale with the sidebar
    /// out of the way. Nothing else changes, and leaving restores the panes
    /// exactly as they were. The panes are hidden without touching the
    /// persisted flags, so a crash mid-talk does not lose the workspace.
    fn toggle_presentation(&mut self, cx: &mut Context<Self>) {
        match self.presentation.take() {
            Some(sidebar) => {
                self.sidebar_open = sidebar;
                theme::set_ui_scale(1.0);
            }
            None => {
                self.presentation = Some(self.sidebar_open);
                self.sidebar_open = false;
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
            Command::ToggleSidebar => self.toggle_sidebar(cx),
            Command::ToggleTheme => self.cycle_appearance(cx),
            Command::TogglePresentation => self.toggle_presentation(cx),
            Command::CancelRun => self.cancel_run(cx),
        }
        cx.notify();
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
    /// Launch straight into the workspace for someone who has used Ember
    /// before: their unfinished experiment is restored into the setup, and
    /// Home -- a page about getting started -- is not the first thing they
    /// see. A first launch has no history and no draft, and still opens Home.
    fn open_where_you_left_off(&mut self, cx: &mut Context<Self>) {
        if self.store.draft.is_some() {
            self.restore_draft(cx);
        } else if !self.store.runs.is_empty() {
            self.view = View::Experiment;
        }
    }

    fn spawn_poll(&mut self, cx: &mut Context<Self>) {
        self.poll_task(cx).detach();
    }

    /// The polling loop behind [`Console::spawn_poll`]. It ends when the
    /// console is released: a loop that outlived its entity would keep
    /// waking every 250 ms for nothing, forever.
    fn poll_task(&mut self, cx: &mut Context<Self>) -> Task<()> {
        cx.spawn(|this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let mut cx = cx.clone();
            async move {
                loop {
                    let Ok(delay) =
                        this.read_with(&cx, |console, _| if console.busy() { 50 } else { 250 })
                    else {
                        break;
                    };
                    cx.background_executor()
                        .timer(Duration::from_millis(delay))
                        .await;
                    let updated = this.update(&mut cx, |console, cx| {
                        let replies = console.drain_replies(cx);
                        if console.drain_store_writes() || replies {
                            cx.notify();
                        }
                    });
                    if updated.is_err() {
                        break;
                    }
                }
            }
        })
    }

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
        // Esc stops a run in flight. Only then: elsewhere it is left to the
        // focused control (closing a picker, say).
        if key == "escape" && self.can_cancel() {
            self.cancel_run(cx);
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
            // With a finished result on screen the number keys address its
            // views: Overview, Layers, Tokens, Raw trace.
            if let Some(view) = result_view
                && self.view == View::Experiment
                && self.comparison.is_some()
            {
                self.result_view = view;
                cx.notify();
                return;
            }
        }
        if matches!(key, "enter" | "return") && cmd {
            self.advance_or_run();
            cx.notify();
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
// ---------------------------------------------------------------------------
// model discovery and the built-in sample
// ---------------------------------------------------------------------------

/// Models under the working directory, with their sizes. Blocking; run it
/// off the UI thread.
fn scan_models() -> Vec<(String, Option<u64>)> {
    discover_models()
        .into_iter()
        .map(|path| {
            let size = std::fs::metadata(&path).map(|meta| meta.len()).ok();
            (path, size)
        })
        .collect()
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

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "gui-tests"))]
mod kit_tests;
