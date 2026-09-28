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
use ember::quant_k::KStrategy;
use gpui_kit::component::{
    button::{Button, ButtonVariants},
    Icon, Selectable, Sizable,
};
use gpui_kit::prelude::*;
use gpui_kit::*;
use std::borrow::Cow;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

gpui_kit::actions!(ember_gui, [Quit]);

mod chart;
mod components;
mod icons;
mod input;
mod picker;
mod theme;

use components::*;
use input::{InputEvent, InputId, InputKind, TextInput};
use theme::{AppearanceMode, Colors, Radius, Space, Type};

// ---------------------------------------------------------------------------
// embedded fonts (SIL OFL 1.1, see src/gui_fonts/LICENSE.txt)
// ---------------------------------------------------------------------------

/// Character budget for a preset hint in the narrow left sidebar.
///
/// Chosen against the rendered minimum-width artifact (980px) so the hint
/// ellipsizes rather than being cut mid-word.
const PRESET_HINT_CHARS: usize = 34;

const FONT_SANS: &[u8] = include_bytes!("gui_fonts/NotoSans-Regular.ttf");
const FONT_MONO: &[u8] = include_bytes!("gui_fonts/NotoSansMono-Regular.ttf");
const FONT_ARABIC: &[u8] = include_bytes!("gui_fonts/NotoNaskhArabic-Regular.ttf");
const FONT_SANS_NAME: &str = "Noto Sans";
const FONT_MONO_NAME: &str = "Noto Sans Mono";
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

    fn icon(self) -> &'static str {
        match self {
            View::Home => icons::HOME,
            View::Experiment => icons::EXPERIMENT,
            View::Models => icons::MODEL,
            View::Runs => icons::RUNS,
            View::Settings => icons::SETTINGS,
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
    ZeroMiddle,
    ScaleLate,
    CopyEarlier,
    ArabicMorphology,
}

#[derive(Debug, Clone)]
struct HistoryEntry {
    number: usize,
    summary: String,
    outcome: String,
    ok: bool,
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

#[derive(Clone)]
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
    model_options: Vec<String>,
    model_path: String,
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
    step: WorkspaceStep,
    view: View,
    inspector_open: bool,
    advanced_open: bool,
    pending_run: bool,
    pending_context: Option<FormValues>,
    result_context: Option<FormValues>,
    history: Vec<HistoryEntry>,
    run_sequence: usize,
    // theme
    appearance: AppearanceMode,
    system_dark: bool,
    // session + results
    session: Option<SessionInfo>,
    status: Status,
    error: Option<String>,
    warning: Option<String>,
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
            model_options: models,
            model_path,
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
            step: WorkspaceStep::Prompt,
            // Land on Home rather than inside a half-configured experiment.
            view: View::Home,
            // The context rail opens on demand; it used to be permanent and
            // permanently half-empty on the setup steps.
            inspector_open: false,
            advanced_open: false,
            pending_run: false,
            pending_context: None,
            result_context: None,
            history: Vec::new(),
            run_sequence: 0,
            appearance,
            system_dark,
            session: None,
            status: Status::Idle,
            error: None,
            warning: None,
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
        _colors: &Colors,
        id: &'static str,
        text: String,
        accent: bool,
        step: WorkspaceStep,
        cx: &mut Context<Self>,
    ) -> Button {
        // These are commands, not prose, so they use Button (never Link --
        // Link is for URLs and email). `outline` is not a variant but composes
        // with ghost, which is what gives the node a resting hairline: a plain
        // ghost button on a dark canvas reads as static text, so the control
        // looked clickable while advertising nothing. gpui-kit still owns the
        // hover and focus-ring geometry, which overriding by hand would break.
        Button::new(SharedString::from(format!("pipeline:{id}")))
            .ghost()
            .outline()
            .small()
            .label(text.clone())
            .tooltip(format!("Go to {}", step.label()))
            .accessibility_label(format!("{}: go to {}", text, step.label()))
            // The intervention is the variable under study, so it is the one
            // node that keeps emphasis. Everything else stays quiet.
            .selected(accent)
            .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                console.step = step;
                cx.notify();
            }))
    }

    fn experiment_pipeline(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let context = self.visible_experiment_context();
        let target = if per_layer(&context.site) {
            let site = match context.site.as_str() {
                "before-layer" => "PRE",
                "after-attention" => "ATTN",
                "after-mlp" => "FFN",
                "after-layer" => "RESID",
                _ => "SITE",
            };
            format!("L{} {site}", context.layer)
        } else if context.site == "before-logits" {
            "FINAL NORM".to_string()
        } else {
            "LOGITS".to_string()
        };
        let operation = match context.op.as_str() {
            "scale" => format!("\u{d7}{}", context.value),
            "zero" => "ZERO".to_string(),
            "replace" => format!("COPY L{}", context.source_layer),
            "interpolate" => format!("BLEND {}", context.value),
            "add-delta" => format!("\u{394} L{}", context.source_layer),
            _ => context.op.to_ascii_uppercase(),
        };
        let completed = self.baseline.is_some() && self.status == Status::Idle;
        let sep = || label("\u{00b7}", Type::LABEL, colors.text_faint).flex_none();

        // A summary of what the current form will do, not a diagram of it.
        //
        // This was a bordered card with eight pill nodes, two ruled tracks and
        // a header, which made the experiment graph the loudest object on the
        // screen while the actual task sat underneath it. The information is
        // unchanged and every node is still a click target that jumps to the
        // control it names; only the framing changed -- no container, no
        // background, no node borders, and the intervention track is the one
        // thing that carries colour because it is the thing under study.
        div()
            .w_full()
            .flex_col()
            .gap(px(Space::XS))
            .pb_4()
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(Space::SM))
                    .overflow_hidden()
                    .child(self.pipeline_node(
                        colors,
                        "input",
                        "prompt".to_string(),
                        false,
                        WorkspaceStep::Prompt,
                        cx,
                    ))
                    .child(sep())
                    .child(self.pipeline_node(
                        colors,
                        "baseline",
                        "unchanged".to_string(),
                        false,
                        WorkspaceStep::Review,
                        cx,
                    ))
                    .child(sep())
                    .child(self.pipeline_node(
                        colors,
                        "target",
                        target,
                        false,
                        WorkspaceStep::Intervention,
                        cx,
                    ))
                    .child(sep())
                    .child(self.pipeline_node(
                        colors,
                        "operation",
                        operation,
                        true,
                        WorkspaceStep::Intervention,
                        cx,
                    ))
                    .child(sep())
                    .child(label("compare", Type::META, colors.text_faint))
                    .child(sep())
                    .child(self.pipeline_node(
                        colors,
                        "compare",
                        "both".to_string(),
                        false,
                        WorkspaceStep::Review,
                        cx,
                    )),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(Space::SM))
                    .child(mono(
                        format!(
                            "{}  \u{00b7}  {}",
                            token_label(&context.token),
                            if completed { "measured" } else { "planned" }
                        ),
                        Type::META,
                        colors.text_faint,
                    ))
                    .child(mono(
                        format!("\u{2264}{} tokens  \u{00b7}  seed 0", context.max_tokens),
                        Type::META,
                        colors.text_faint,
                    )),
            )
    }

    /// Build the v0.5 request from the current form fields; the shared
    /// `parse_run_request` gate validates it exactly like the web console.
    fn build_run_request(&self) -> Result<RunRequest, String> {
        self.form_values().build_run_request()
    }

    fn send_run(&mut self, cfg: RunConfig) {
        self.status = Status::Running;
        self.step = WorkspaceStep::Review;
        self.pending_context = Some(self.form_values());
        self.error = None;
        self.warning = None;
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
                        if bundle.baseline.text == bundle.intervention.text {
                            self.warning = Some(
                                "baseline and intervention outputs are identical for this \
                                 configuration."
                                    .to_string(),
                            );
                        }
                        self.run_sequence += 1;
                        self.history.insert(
                            0,
                            HistoryEntry {
                                number: self.run_sequence,
                                summary: format!(
                                    "{} · {}",
                                    operation_label(&self.op),
                                    if per_layer(&self.site) {
                                        format!("layer {}", self.layer)
                                    } else {
                                        site_label(&self.site).to_string()
                                    }
                                ),
                                outcome: if bundle.baseline.text == bundle.intervention.text {
                                    "Output unchanged".to_string()
                                } else {
                                    bundle.comparison.first_token_divergence.map_or_else(
                                        || "Output text changed".to_string(),
                                        |step| format!("Diverged at decode step {step}"),
                                    )
                                },
                                ok: bundle.verification.ok,
                            },
                        );
                        self.history.truncate(6);
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
        theme.font_size = px(13.0);
        theme.mono_font_family = FONT_MONO_NAME.into();
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
        Theme::sync_base(cx);
    }

    fn cycle_appearance(&mut self, cx: &mut Context<Self>) {
        self.appearance = self.appearance.next();
        self.appearance.persist();
        self.sync_kit_theme(cx);
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

    fn picker_key(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) {
        if event.keystroke.modifiers.control || event.keystroke.modifiers.platform {
            let result_view = match event.keystroke.key.as_str() {
                "1" => Some(ResultView::Overview),
                "2" => Some(ResultView::Layers),
                "3" => Some(ResultView::Tokens),
                "4" => Some(ResultView::Trace),
                _ => None,
            };
            if let Some(view) = result_view
                && self.step == WorkspaceStep::Review
                && self.comparison.is_some()
            {
                self.result_view = view;
                cx.notify();
                return;
            }
        }
        if matches!(event.keystroke.key.as_str(), "enter" | "return")
            && (event.keystroke.modifiers.control || event.keystroke.modifiers.platform)
        {
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

    /// Slim top bar: identity, current model, appearance.
    ///
    /// The wordmark stands on its own. There was an accent-coloured "E" badge
    /// here that carried no information and dated the product instantly.
    fn topbar(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let toggle = cx.listener(|console, _: &ClickEvent, _w, cx| {
            console.cycle_appearance(cx);
        });
        div()
            .flex()
            .flex_row()
            .items_center()
            .gap(px(Space::MD))
            .px_4()
            .h(px(46.0))
            .w_full()
            .bg(colors.surface)
            .border_b_1()
            .border_color(colors.border)
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(Space::SM))
                    .child(
                        Button::new("topbar-home")
                            .ghost()
                            .small()
                            .icon(Icon::default().path(icons::BACK))
                            .label("ember")
                            .tooltip("Home")
                            .accessibility_label("Ember, go to Home")
                            .on_click(cx.listener(|console, _: &ClickEvent, _, cx| {
                                console.view = View::Home;
                                cx.notify();
                            })),
                    ),
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
                        row.child(
                            div()
                                .w(px(3.0))
                                .h(px(3.0))
                                .rounded(px(Radius::SM))
                                .bg(colors.border_strong),
                        )
                        .child(label(
                            truncate_chars(self.step.label(), 28),
                            Type::LABEL,
                            colors.text_muted,
                        ))
                    }),
            )
            .child(
                Button::new("theme-toggle")
                    .ghost()
                    .small()
                    .icon(
                        Icon::default().path(if self.appearance.is_dark(self.system_dark) {
                            icons::SUN
                        } else {
                            icons::MOON
                        }),
                    )
                    .tooltip("Appearance")
                    .accessibility_label(format!(
                        "Appearance: {}. Switch appearance.",
                        self.appearance.label()
                    ))
                    .on_click(toggle),
            )
            .when(self.view == View::Experiment, |bar| {
                bar.child(
                    Button::new("inspector-toggle")
                        .ghost()
                        .small()
                        .selected(self.inspector_open)
                        .icon(Icon::default().path(icons::PANEL_RIGHT))
                        .tooltip("Inspector")
                        .accessibility_label(format!(
                            "Inspector. {}",
                            if self.inspector_open {
                                "Hide the inspector"
                            } else {
                                "Show the inspector"
                            }
                        ))
                        .on_click(cx.listener(|console, _: &ClickEvent, _, cx| {
                            console.inspector_open = !console.inspector_open;
                            cx.notify();
                        })),
                )
            })
    }

    /// Left navigation rail: icon plus label, tight spacing.
    fn nav_rail(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let mut column = div()
            .flex()
            .flex_col()
            .gap(px(Space::XS))
            .w(px(184.0))
            .flex_none()
            .px_3()
            .py_3()
            .bg(colors.sidebar)
            .border_r_1()
            .border_color(colors.border);
        for view in View::ALL {
            let active = self.view == view;
            column = column.child(
                Button::new(SharedString::from(format!("nav:{}", view.key())))
                    .ghost()
                    .w_full()
                    .h(px(30.0))
                    .justify_start()
                    .gap(px(Space::SM))
                    .selected(active)
                    .icon(Icon::default().path(view.icon()))
                    .label(view.label())
                    .tooltip(view.hint())
                    .accessibility_label(format!("{}, {}", view.label(), view.hint()))
                    .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                        console.view = view;
                        cx.notify();
                    })),
            );
        }
        column
    }

    /// Workflow stepper. Reads as tabs, not a wizard diagram.
    fn stepper(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let mut row = div().flex().flex_row().gap(px(Space::XS));
        for (index, step) in WorkspaceStep::ALL.iter().enumerate() {
            if index > 0 {
                row = row.child(label("/", Type::LABEL, colors.text_faint));
            }
            row = row.child(
                Button::new(SharedString::from(format!("step:{}", step.key())))
                    .small()
                    .selected(self.step == *step)
                    .label(step.label())
                    .tooltip(step.hint())
                    .accessibility_label(format!("Step {}: {}", step.number(), step.label()))
                    .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                        console.step = *step;
                        cx.notify();
                    })),
            );
        }
        div().w_full().px_5().pt_4().pb_1().child(row)
    }

    /// Home: a landing surface with something to do, not a form in waiting.
    fn section_header(&self, colors: &Colors, title: &'static str, hint: &'static str) -> Div {
        div()
            .flex()
            .flex_col()
            .gap(px(Space::XS))
            .child(label(title, Type::SECTION, colors.text))
            .child(label(hint, Type::BODY, colors.text_muted))
    }

    fn models_view(&self, colors: &Colors, _cx: &mut Context<Self>) -> Div {
        let loaded = self.session.is_some();
        let current = self.model_path.trim();
        div()
            .flex()
            .flex_col()
            .gap(px(Space::XXL))
            .w_full()
            .max_w(px(760.0))
            .px_5()
            .pt_6()
            .child(self.section_header(
                colors,
                "Models",
                "Local GGUF files. Nothing leaves this machine.",
            ))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(Space::SM))
                    .p(px(Space::LG))
                    .rounded(px(8.0))
                    .bg(colors.surface)
                    .border_1()
                    .border_color(colors.border)
                    .child(label("Current", Type::LABEL, colors.text_faint))
                    .child(mono(
                        if current.is_empty() {
                            "No model selected"
                        } else {
                            current
                        },
                        Type::BODY,
                        colors.text,
                    ))
                    .child(label(
                        if loaded {
                            "Loaded and resident"
                        } else {
                            "Not loaded yet"
                        },
                        Type::LABEL,
                        if loaded { colors.ok } else { colors.text_muted },
                    )),
            )
    }

    fn runs_view(&self, colors: &Colors, _cx: &mut Context<Self>) -> Div {
        let mut list = div().flex().flex_col().gap(px(Space::SM));
        if self.history.is_empty() {
            list = list.child(
                div()
                    .p(px(Space::XL))
                    .rounded(px(8.0))
                    .bg(colors.surface)
                    .border_1()
                    .border_color(colors.border)
                    .flex()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("No runs yet", Type::SUBSECTION, colors.text))
                    .child(label(
                        "Experiments you run in this session are listed here.",
                        Type::LABEL,
                        colors.text_faint,
                    )),
            );
        } else {
            for entry in self.history.iter().rev() {
                list = list.child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap(px(Space::MD))
                        .px_3()
                        .py_2()
                        .rounded(px(6.0))
                        .bg(colors.surface)
                        .border_1()
                        .border_color(if entry.ok {
                            colors.border
                        } else {
                            colors.err_box_border
                        })
                        .child(label(format!("Run #{}", entry.number), 11.0, colors.text))
                        .child(label(
                            entry.summary.as_str(),
                            Type::LABEL,
                            colors.text_muted,
                        ))
                        .child(div().w_full())
                        .child(mono(entry.outcome.as_str(), Type::LABEL, colors.text_faint)),
                );
            }
        }
        div()
            .flex()
            .flex_col()
            .gap(px(Space::XXL))
            .w_full()
            .max_w(px(760.0))
            .px_5()
            .pt_6()
            .child(self.section_header(colors, "Runs", "Every experiment run from this session."))
            .child(list)
    }

    fn settings_view(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let toggle = cx.listener(|console, _: &ClickEvent, _, cx| {
            console.cycle_appearance(cx);
        });
        div()
            .flex()
            .flex_col()
            .gap(px(Space::XXL))
            .w_full()
            .max_w(px(760.0))
            .px_5()
            .pt_6()
            .child(self.section_header(
                colors,
                "Settings",
                "Appearance and defaults for this console.",
            ))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(Space::MD))
                    .p(px(Space::LG))
                    .rounded(px(8.0))
                    .bg(colors.surface)
                    .border_1()
                    .border_color(colors.border)
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(Space::XS))
                            .child(label("Appearance", Type::BODY, colors.text))
                            .child(label(
                                "Currently following the system setting when set to System.",
                                Type::LABEL,
                                colors.text_faint,
                            )),
                    )
                    .child(div().w_full())
                    .child(
                        Button::new("settings-appearance")
                            .small()
                            .label(self.appearance.label())
                            .on_click(toggle),
                    ),
            )
    }

    fn home_view(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let start = cx.listener(|console, _: &ClickEvent, _, cx| {
            console.view = View::Experiment;
            console.step = WorkspaceStep::Prompt;
            cx.notify();
        });
        let runs = cx.listener(|console, _: &ClickEvent, _, cx| {
            console.view = View::Runs;
            cx.notify();
        });
        let models = cx.listener(|console, _: &ClickEvent, _, cx| {
            console.view = View::Models;
            cx.notify();
        });
        let mut recent = div().flex().flex_col().gap(px(Space::SM));
        recent = recent.child(label("Recent runs", Type::LABEL, colors.text_faint));
        if self.history.is_empty() {
            recent = recent.child(
                div()
                    .text_color(colors.text_faint)
                    .text_size(px(11.0))
                    .child("No runs yet. Start an experiment and its results will collect here."),
            );
        } else {
            for entry in self.history.iter().rev().take(5) {
                recent = recent.child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap(px(Space::SM))
                        .px_3()
                        .py_2()
                        .rounded(px(6.0))
                        .bg(colors.surface)
                        .border_1()
                        .border_color(colors.border)
                        .child(label(format!("Run #{}", entry.number), 11.0, colors.text))
                        .child(div().w_full())
                        .child(mono(entry.outcome.as_str(), Type::LABEL, colors.text_muted)),
                );
            }
            recent = recent.child(
                Button::new("home-all-runs")
                    .small()
                    .label("See all runs")
                    .on_click(runs),
            );
        }
        div()
            .flex()
            .flex_col()
            .gap(px(Space::XXL))
            .w_full()
            .max_w(px(760.0))
            .px_5()
            .pt_6()
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
                            .icon(Icon::default().path(icons::PLUS))
                            .label("New experiment")
                            .accessibility_label("Start a new experiment")
                            .on_click(start),
                    )
                    .child(
                        Button::new("home-models")
                            .icon(Icon::default().path(icons::MODEL))
                            .label("Manage models")
                            .on_click(models),
                    ),
            )
            .child(recent)
    }

    /// Contextual inspector. Only present while the user has it open, and it
    /// reports the state of the current run rather than repeating it.
    fn inspector(&self, colors: &Colors, _cx: &mut Context<Self>) -> Stateful<Div> {
        div()
            .id(ElementId::Name(SharedString::from("workflow-rail")))
            .flex_col()
            .w(px(216.0))
            .flex_none()
            .h_full()
            .overflow_y_scroll()
            .bg(colors.sidebar)
            .border_r_1()
            .border_color(colors.border)
            .p(px(Space::MD))
            .gap(px(Space::LG))
    }

    /// Starting points for a new experiment.
    ///
    /// These were in the left rail, which made them look like a mode switch.
    /// They are choices for starting work, so they belong on the Prompt step
    /// where the work starts.
    fn presets_block(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let presets = [
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
        // Wrapping grid, not one row: four w_full cards in a row overflowed the
        // workspace and the trailing cards were unreachable.
        div()
            .flex()
            .flex_row()
            .flex_wrap()
            .gap(px(Space::SM))
            .children(presets.into_iter().map(|(preset, title, hint)| {
                div()
                    .flex_1()
                    .min_w(px(300.0))
                    .child(self.preset_card(colors, preset, title, hint, cx))
            }))
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
            .outline()
            .compact()
            .py_1()
            .px_3()
            .accessibility_label(title)
            .child(
                div()
                    .w_full()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label(title, Type::LABEL, colors.text))
                    // The sidebar is narrow, so the hint is hard-clipped
                    // mid-word by the button's content box ("...an Arabic
                    // pi"). Ellipsize explicitly so the reader can see there
                    // is more, using the same helper the prompt excerpt uses.
                    .child(label(
                        truncate_chars(hint, PRESET_HINT_CHARS),
                        Type::META,
                        colors.text_faint,
                    )),
            )
            .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                console.apply_preset(preset, cx);
            }))
            .into_any_element()
    }

    fn generation_option(
        &self,
        colors: &Colors,
        value: usize,
        title: &'static str,
        hint: &'static str,
        cx: &mut Context<Self>,
    ) -> Button {
        // State must be visible, and text contrast must survive it. The hint
        // used text_faint in both states, so the selected segment was a
        // barely-darker gray on a light gray fill. The selected variant
        // promotes the hint to text_muted, which is a real contrast step.
        let selected = self.max_tokens == value.to_string();
        Button::new(SharedString::from(format!("generation-length:{value}")))
            .w(relative(0.333))
            .h_auto()
            .py_2()
            .selected(selected)
            .accessibility_label(if selected {
                format!("{title}, selected")
            } else {
                title.to_string()
            })
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label(title, Type::BODY, colors.text))
                    .child(label(
                        hint,
                        Type::LABEL,
                        if selected {
                            colors.text_muted
                        } else {
                            colors.text_faint
                        },
                    )),
            )
            .on_click(cx.listener(move |console, _: &ClickEvent, _, cx| {
                console.set_max_tokens(value, cx);
            }))
    }

    fn operation_card(
        &self,
        colors: &Colors,
        operation: &'static str,
        cx: &mut Context<Self>,
    ) -> Button {
        Button::new(SharedString::from(format!("operation-card:{operation}")))
            .w(relative(0.5))
            .h_auto()
            .min_h(px(70.0))
            .py_3()
            .selected(self.op == operation)
            .accessibility_label(operation_label(operation))
            .child(
                div()
                    .w_full()
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
        let error = self.error.as_ref().map(|error| {
            div()
                .w_full()
                .px_3()
                .py_2()
                .bg(colors.err_box_bg)
                .border_1()
                .border_color(colors.err_box_border)
                .rounded(px(Radius::MD))
                .flex()
                .items_center()
                .gap(px(Space::SM))
                .child(
                    icons::icon(icons::WARNING)
                        .size(px(15.0))
                        .text_color(colors.err),
                )
                .child(label(error.clone(), Type::LABEL, colors.err))
        });
        let warning = self.warning.as_ref().map(|warning| {
            div()
                .w_full()
                .px_3()
                .py_2()
                .bg(colors.warn_box_bg)
                .border_1()
                .border_color(colors.warn_box_border)
                .rounded(px(Radius::MD))
                .flex()
                .items_center()
                .gap(px(Space::SM))
                .child(
                    icons::icon(icons::WARNING)
                        .size(px(15.0))
                        .text_color(colors.warn),
                )
                .child(label(warning.clone(), Type::LABEL, colors.warn))
        });
        div()
            .flex_col()
            .gap(px(Space::SM))
            .children(error)
            .children(warning)
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
                "MODEL FILE",
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
            .flex_col()
            .gap(px(Space::LG))
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("Prepare the experiment", Type::TITLE, colors.text))
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
                    .mb_5()
                    .child(label("Start from a preset", Type::LABEL, colors.text_faint))
                    .child(self.presets_block(colors, cx)),
            )
            .child(group(
                div()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .child(section_label(colors, "Model"))
                            .child(div().w_full())
                            .child(chip(model_status.0, model_status.2)),
                    )
                    .child(self.picker(
                        colors,
                        "model-picker",
                        ComboId::Model,
                        &self.model_path,
                        &self.model_options,
                        cx,
                    ))
                    .children(raw_path)
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(Space::SM))
                            .child(label(model_status.1, Type::LABEL, colors.text_faint))
                            .child(div().w_full())
                            .child(div().w(px(150.0)).child(btn_secondary(
                                colors,
                                icons::MODEL,
                                if self.status == Status::Preparing {
                                    "LOADING…"
                                } else {
                                    "LOAD MODEL"
                                },
                                (!self.busy()).then(|| {
                                    cx.listener(|console, _: &ClickEvent, _window, cx| {
                                        console.load();
                                        cx.notify();
                                    })
                                }),
                            ))),
                    ),
            ))
            .child(group(
                div()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(section_label(colors, "Prompt"))
                    .child(text_input(
                        colors,
                        self.inputs.prompt.clone(),
                        FONT_ARABIC_NAME,
                        Type::SUBSECTION,
                        Some(160.0),
                        cx,
                    ))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .child(label(
                                format!("{} characters", self.prompt.chars().count()),
                                Type::LABEL,
                                colors.text_faint,
                            ))
                            .child(div().w_full())
                            .child(label(
                                "Arabic and mixed-direction text supported",
                                Type::LABEL,
                                colors.text_faint,
                            )),
                    ),
            ))
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::SM))
                    .child(label("Response length", Type::LABEL, colors.text_faint))
                    .child(
                        div()
                            .flex()
                            .gap(px(Space::SM))
                            .child(self.generation_option(
                                colors,
                                24,
                                "Short",
                                "Up to 24 tokens",
                                cx,
                            ))
                            .child(self.generation_option(
                                colors,
                                48,
                                "Medium",
                                "Up to 48 tokens",
                                cx,
                            ))
                            .child(self.generation_option(
                                colors,
                                96,
                                "Long",
                                "Up to 96 tokens",
                                cx,
                            )),
                    ),
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
                .flex_col()
                .gap(px(Space::MD))
                .child(field(
                    colors,
                    "SOURCE",
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
                        "SOURCE LAYER",
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
                    "BLEND AMOUNT (0–1)"
                } else {
                    "Strength multiplier"
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
                "PHRASE TO TARGET",
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
            .flex_col()
            .gap(px(Space::LG))
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("Choose the internal change", Type::TITLE, colors.text))
                    .child(label(
                        "Start with the research question. Exact hook names remain available in Advanced controls.",
                        Type::BODY,
                        colors.text_muted,
                    )),
            )
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::SM))
                    .child(label("What should change?", Type::LABEL, colors.text_faint))
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
                            .child(div().w(relative(0.5))),
                    ),
            )
            .child(panel(
                colors,
                div()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(section_label(colors, "Where"))
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
                    .child(mono(
                        format!("ember.hook.v1 \u{00b7} {}", site_contract_name(&self.site)),
                        Type::META,
                        colors.text_faint,
                    ))
                    .when(per_layer(&self.site), |content| {
                        content.child(field(colors, "Layer", self.layer_stepper(colors, cx)))
                    })
                    .children(value_control)
                    .children(source_controls),
            ))
            .child(panel(
                colors,
                div()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(section_label(colors, "Target"))
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
            ))
    }

    fn result_summary(&self, colors: &Colors) -> Div {
        match (&self.baseline, &self.intervention, &self.comparison) {
            (Some(baseline), Some(intervention), _)
                if baseline.text == intervention.text => div()
                .px_3()
                .py_2()
                .rounded(px(Radius::MD))
                .bg(colors.warn_box_bg)
                .border_1()
                .border_color(colors.warn_box_border)
                .flex()
                .items_center()
                .gap(px(Space::SM))
                .child(
                    icons::icon(icons::WARNING)
                        .size(px(15.0))
                        .text_color(colors.warn),
                )
                .child(label(
                    "The intervention completed successfully but did not change the generated text.",
                    Type::LABEL,
                    colors.warn,
                )),
            (Some(_), Some(_), Some(comparison)) => div()
                .px_3()
                .py_2()
                .rounded(px(Radius::MD))
                .bg(colors.accent_soft)
                .border_1()
                .border_color(colors.accent)
                .flex()
                .items_center()
                .gap(px(Space::SM))
                .child(
                    icons::icon(icons::CHECK)
                        .size(px(15.0))
                        .text_color(colors.accent),
                )
                .child(label(
                    comparison.first_token_divergence.map_or_else(
                        || "The generated output text changed.".to_string(),
                        |step| format!("Generated behavior diverges at decode step {step}."),
                    ),
                    Type::LABEL,
                    colors.text,
                )),
            _ if self.busy() => div()
                .px_3()
                .py_2()
                .rounded(px(Radius::MD))
                .bg(colors.accent_soft)
                .child(label(
                    "The model is running both the baseline and intervention. Results will appear here.",
                    Type::LABEL,
                    colors.text_muted,
                )),
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
        let first_layer = landmarks
            .first_layer_divergence
            .map_or_else(|| "NONE OBSERVED".to_string(), |layer| format!("L{layer}"));
        let peak = match (landmarks.peak_relative_l2, landmarks.peak_layer) {
            (Some(value), Some(layer)) => format!("{value:.6}  @ L{layer}"),
            _ => "NONE OBSERVED".to_string(),
        };
        let stable_tail = if comparison.generated_tokens_equal {
            "OUTPUTS IDENTICAL".to_string()
        } else {
            landmarks.stable_token_tail_step.map_or_else(
                || "NOT OBSERVED".to_string(),
                |step| format!("FROM STEP {step}"),
            )
        };
        let landmark = |title: &'static str, value: String, detail: &'static str| {
            div()
                .flex_1()
                .min_w(px(0.0))
                .flex_col()
                .gap(px(Space::XS))
                .child(label(title, Type::MICRO, colors.text_faint))
                .child(mono(value, Type::LABEL, colors.text))
                .child(label(detail, Type::META, colors.text_muted))
        };
        let divider = || div().w(px(1.0)).h(px(42.0)).bg(colors.border).flex_none();

        div()
            .w_full()
            .px_3()
            .py_2()
            .flex()
            .items_center()
            .gap(px(Space::MD))
            .bg(colors.surface)
            .border_1()
            .border_color(colors.border)
            .rounded(px(Radius::MD))
            .child(landmark(
                "FIRST INTERNAL DIVERGENCE",
                first_layer,
                "first non-zero captured layer",
            ))
            .child(divider())
            .child(landmark(
                "PEAK REPRESENTATION DIVERGENCE",
                peak,
                "relative L2 difference",
            ))
            .child(divider())
            .child(landmark(
                "STABLE TOKEN TAIL",
                stable_tail,
                "exact token-ID suffix",
            ))
    }

    fn result_tabs(&self, _colors: &Colors, cx: &mut Context<Self>) -> Div {
        use gpui_kit::component::tab::{Tab, TabBar};
        let selected = ResultView::ALL
            .iter()
            .position(|view| *view == self.result_view)
            .unwrap_or(0);
        div().w_full().child(
            TabBar::new("result-tabs")
                .underline()
                .selected_index(selected)
                .children(
                    ResultView::ALL
                        .into_iter()
                        .map(|view| Tab::new().label(view.label())),
                )
                .on_click(cx.listener(|console, index: &usize, _, cx| {
                    console.result_view = ResultView::ALL[*index];
                    cx.notify();
                })),
        )
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
        let export = icon_button(
            colors,
            icons::COPY,
            "Copy layer metrics as CSV",
            cx.listener(move |_console, _: &ClickEvent, _window, cx| {
                cx.write_to_clipboard(ClipboardItem::new_string(csv.clone()));
            }),
        );
        panel(
            colors,
            div()
                .flex_col()
                .gap(px(Space::SM))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .child(
                            div()
                                .flex_col()
                                .gap(px(Space::XS))
                                .child(label(
                                    "REPRESENTATION DIVERGENCE",
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
                        .child(label("COPY CSV", Type::MICRO, colors.text_faint))
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
            .child(self.output_panel(colors, "BASELINE", self.baseline.as_ref(), self.status, cx))
            .child(self.output_panel(
                colors,
                "INTERVENTION",
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
                        format!("STEP {}", token.position),
                        Type::MICRO,
                        colors.text_faint,
                    ))
                    .child(label(baseline, Type::BODY, colors.text))
                    .child(rule_h(colors))
                    .child(label(intervention, Type::BODY, colors.text))
                    .into_any_element()
            })
            .collect::<Vec<_>>();
        let divergence = comparison.first_token_divergence.map_or_else(
            || "Generated token IDs are identical.".to_string(),
            |step| format!("Generated behavior first differs at decode step {step}."),
        );
        panel(
            colors,
            div()
                .flex_col()
                .gap(px(Space::SM))
                .child(label(
                    "TOKEN-LEVEL OUTPUT COMPARISON",
                    Type::LABEL,
                    colors.text_faint,
                ))
                .child(label(divergence, Type::BODY, colors.text))
                .child(
                    div()
                        .flex()
                        .gap(px(Space::SM))
                        .child(chip("BASELINE", colors.text_muted))
                        .child(chip("INTERVENTION", colors.accent))
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

    fn raw_trace_panel(&self, colors: &Colors) -> Div {
        let events = self.intervention.as_ref().map_or_else(Vec::new, |output| {
            output
                .events
                .iter()
                .map(|event| {
                    mono(event.to_string(), Type::META, colors.text_muted).into_any_element()
                })
                .collect::<Vec<_>>()
        });
        panel(
            colors,
            div()
                .flex_col()
                .gap(px(Space::SM))
                .child(label(
                    "RAW INTERVENTION TRACE",
                    Type::LABEL,
                    colors.text_faint,
                ))
                .children(
                    (!events.is_empty())
                        .then(|| div().flex_col().gap(px(Space::XS)).children(events)),
                )
                .children(self.intervention.as_ref().map(|output| {
                    mono(
                        format!("bundle  {}", output.bundle_dir),
                        Type::META,
                        colors.text_faint,
                    )
                })),
        )
    }

    fn review_step(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let has_results = self.baseline.is_some() && self.intervention.is_some();
        let result_body = if !has_results {
            div()
                .flex_col()
                .gap(px(Space::MD))
                .child(self.result_summary(colors))
                .child(self.paired_outputs(colors, cx))
                .into_any_element()
        } else {
            match self.result_view {
                ResultView::Overview => div()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(self.result_summary(colors))
                    .child(self.result_landmarks(colors))
                    .child(self.paired_outputs(colors, cx))
                    .child(self.layer_chart_panel(colors, 190.0, cx))
                    .into_any_element(),
                ResultView::Layers => div()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(self.layer_chart_panel(colors, 350.0, cx))
                    .into_any_element(),
                ResultView::Tokens => div()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(self.paired_outputs(colors, cx))
                    .child(self.token_comparison_panel(colors))
                    .into_any_element(),
                ResultView::Trace => div()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(self.raw_trace_panel(colors))
                    .child(self.verification_panel(colors))
                    .into_any_element(),
            }
        };

        div()
            .flex_col()
            .gap(px(Space::LG))
            .child(
                div()
                    .flex()
                    .items_end()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex_col()
                            .gap(px(Space::XS))
                            .child(label("Review and compare", Type::TITLE, colors.text))
                            .child(label(
                                "The baseline and intervention use the same prompt and deterministic settings.",
                                Type::BODY,
                                colors.text_muted,
                            )),
                    )
                    .gap(px(Space::MD))
                    .when(self.last_config.is_some(), |header| {
                        header.child(
                            div()
                                .w(px(140.0))
                                .flex_shrink_0()
                                .child(btn_secondary(
                                    colors,
                                    icons::RESTORE,
                                    if self.status == Status::Restoring {
                                        "VERIFYING…"
                                    } else {
                                        "VERIFY RESTORE"
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
            .when(has_results, |page| page.child(self.result_tabs(colors, cx)))
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
            "HOVERED POINT"
        } else {
            "SELECTED POINT"
        };
        let advanced = self.advanced_open.then(|| {
            div()
                .flex_col()
                .gap(px(Space::MD))
                .pt_2()
                .child(field(
                    colors,
                    "EXECUTION ENGINE",
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
                    "EXACT TOKEN LIMIT",
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
                    "RAW MODEL PATH",
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
        let toggle = cx.listener(|console, _: &ClickEvent, _window, cx| {
            console.advanced_open = !console.advanced_open;
            cx.notify();
        });

        div()
            .flex_col()
            .gap(px(Space::MD))
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("Model", Type::META, colors.text_faint))
                    .child(label(model_name, Type::LABEL, colors.text))
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
            )
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("Input", Type::META, colors.text_faint))
                    .child(multiline(
                        &prompt_excerpt,
                        Type::LABEL,
                        colors.text,
                        FONT_ARABIC_NAME,
                    )),
            )
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("Target", Type::META, colors.text_faint))
                    .child(multiline(&target, Type::META, colors.text, FONT_SANS_NAME)),
            )
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("Intervention", Type::META, colors.text_faint))
                    .child(label(intervention, Type::LABEL, colors.accent)),
            )
            .child(rule_h(colors))
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label("Generation", Type::META, colors.text_faint))
                    .child(mono(
                        format!(
                            "≤{} tokens · seed 0\n{}",
                            context.max_tokens, context.execution
                        ),
                        Type::META,
                        colors.text,
                    )),
            )
            .children(self.intervention.as_ref().map(|output| {
                div()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(rule_h(colors))
                    .child(
                        div()
                            .flex_col()
                            .gap(px(Space::XS))
                            .child(label("RUN", Type::META, colors.text_faint))
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
                    )
            }))
            .children(active_metric.map(|metric| {
                div()
                    .flex_col()
                    .gap(px(Space::MD))
                    .child(rule_h(colors))
                    .child(
                        div()
                            .flex_col()
                            .gap(px(Space::XS))
                            .child(label(active_metric_label, Type::META, colors.text_faint))
                            .child(mono(format!("layer {}", metric.layer), 10.0, colors.text))
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
                    )
            }))
            .child(rule_h(colors))
            // A ghost button with only a text label reads as static copy in a
            // wide empty column. The design guides require a disclosure
            // control to look like one: a chevron carries the affordance, and
            // the accessible name states the position so the control is not
            // announced as a bare label.
            .child(
                Button::new("advanced-toggle")
                    .ghost()
                    .w_full()
                    .icon(Icon::default().path(icons::CHEVRON_DOWN))
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

    fn main_panel(&self, colors: &Colors, cx: &mut Context<Self>) -> Stateful<Div> {
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
                    .overflow_y_scroll()
                    .p_5()
                    .child(
                        div()
                            .w_full()
                            .max_w(px(980.0))
                            .mx_auto()
                            .flex_col()
                            .gap(px(Space::XL))
                            .child(self.feedback_banners(colors))
                            .child(self.experiment_pipeline(colors, cx))
                            .child(page),
                    ),
            )
            .child(
                div()
                    .id("inspector-scroll")
                    .w(px(224.0))
                    .flex_none()
                    .h_full()
                    .overflow_y_scroll()
                    .bg(colors.surface)
                    .border_l_1()
                    .border_color(colors.border)
                    .p(px(Space::LG))
                    .child(self.advanced_inspector(colors, cx)),
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
            (Some(_), _) => ("OK", colors.ok),
            (None, Status::Running) => ("RUN", colors.warn),
            (None, _) => ("\u{2014}", colors.text_faint),
        };
        let copy_button = output.map(|output| {
            let text = output.text.clone();
            icon_button(
                colors,
                icons::COPY,
                "Copy output",
                cx.listener(move |_console, _: &ClickEvent, _window, cx| {
                    cx.write_to_clipboard(ClipboardItem::new_string(text.clone()));
                }),
            )
        });
        let divergence_note = self.comparison.as_ref().map(|comparison| {
            if comparison.generated_tokens_equal {
                "token IDs match across both runs".to_string()
            } else if title == "BASELINE" {
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
                    .flex_col()
                    .gap(px(Space::SM))
                    .child(
                        div()
                            .id(ElementId::Name(SharedString::from(format!(
                                "output-scroll:{title}"
                            ))))
                            .min_h(px(84.0))
                            .max_h(px(164.0))
                            .overflow_y_scroll()
                            .w_full()
                            .px_2()
                            .py_2()
                            .bg(colors.surface_raised)
                            .border_1()
                            .border_color(colors.border)
                            .rounded(px(Radius::MD))
                            .child(multiline(
                                &display_text,
                                Type::BODY,
                                colors.text,
                                FONT_ARABIC_NAME,
                            )),
                    )
                    .children(divergence_note.map(|note| {
                        mono(
                            note,
                            Type::LABEL,
                            if title == "INTERVENTION" {
                                colors.accent
                            } else {
                                colors.text_muted
                            },
                        )
                    }))
                    .child(mono(
                        format!(
                            "{} tok \u{00b7} {} \u{00b7} {}",
                            out.generated_tokens,
                            fmt_ms(out.wall_ms),
                            fmt_tps(out.decode_tps)
                        ),
                        Type::LABEL,
                        colors.text_muted,
                    ))
                    .child(mono(
                        format!(
                            "prompt {} tok \u{00b7} bundle {}",
                            out.prompt_tokens,
                            short_id(&out.semantic_hash)
                        ),
                        Type::META,
                        colors.text_faint,
                    ))
                    .child(mono(out.bundle_dir.clone(), Type::LABEL, colors.text_faint))
            }
            Some(_out) => div().child(label("(empty output)", Type::SUBSECTION, colors.text_faint)),
            None => div().child(label(
                "no run yet \u{2014} outputs appear here",
                Type::META,
                colors.text_faint,
            )),
        };
        panel(
            colors,
            div()
                .flex_col()
                .gap(px(Space::SM))
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .child(label(title, Type::BODY, colors.text_muted))
                        .child(div().w_full())
                        .children(copy_button)
                        .child(chip(badge_text, badge_color)),
                )
                .child(body),
        )
        .w(relative(0.5))
        .min_h(px(148.0))
        .overflow_hidden()
    }

    fn verification_panel(&self, colors: &Colors) -> Div {
        let (badge, badge_color) = match (&self.verification, self.status) {
            (Some(verification), _) if verification.ok => ("VERIFIED", colors.ok),
            (Some(_), _) => ("VERIFICATION FAILED", colors.err),
            (None, Status::Running) => ("RUNNING", colors.warn),
            (None, Status::Restoring) => ("RESTORING", colors.warn),
            (None, _) => ("NOT RUN", colors.text_faint),
        };
        let mut lines: Vec<String> = Vec::new();
        if let Some(restore) = &self.restore {
            if !restore.comparable {
                lines.push("restore: baseline not comparable (configuration changed)".to_string());
            } else if restore.matches {
                lines.push("restore: BIT-EXACT".to_string());
            } else {
                lines.push("restore: DIFFERS from baseline".to_string());
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
            div().flex_col().gap(px(Space::XS)).children(
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
        // Home, Models, Runs and Settings have no experiment to advance, so the
        // status line reports the surface instead of offering a run button that
        // would do nothing sensible.
        if self.view != View::Experiment {
            return div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(Space::MD))
                .px_4()
                .h(px(40.0))
                .w_full()
                .bg(colors.surface)
                .border_t_1()
                .border_color(colors.border)
                .child(status_dot(colors.ok, false))
                .child(label(self.view.hint(), Type::LABEL, colors.text_muted))
                .child(div().w_full());
        }
        let (dot, status_text) = match self.status {
            Status::Idle => (colors.ok, "Ready to run"),
            Status::Preparing => (colors.warn, "Loading the model…"),
            Status::Running => (colors.busy, "Running baseline and intervention…"),
            Status::Restoring => (colors.busy, "Checking exact restoration…"),
        };
        let validation_error = self.validation_error();
        let action_enabled = self.action_enabled();
        let action_label = match self.status {
            Status::Preparing => "LOADING MODEL…",
            Status::Running => "RUNNING EXPERIMENT…",
            Status::Restoring => "VERIFYING RESTORE…",
            Status::Idle => match self.step {
                WorkspaceStep::Prompt => "CONTINUE: INTERVENTION",
                WorkspaceStep::Intervention => "CONTINUE: REVIEW",
                WorkspaceStep::Review if self.baseline.is_some() => "RUN EXPERIMENT AGAIN",
                WorkspaceStep::Review => "RUN EXPERIMENT",
            },
        };
        let action_icon = if self.step == WorkspaceStep::Review {
            icons::PLAY
        } else {
            icons::CHEVRON_DOWN
        };

        div()
            .flex()
            .flex_row()
            .items_center()
            .gap(px(Space::MD))
            .px_5()
            .h(px(58.0))
            .w_full()
            .bg(colors.surface)
            .border_t_1()
            .border_color(colors.border)
            .child(status_dot(dot, self.busy()))
            .child(
                div()
                    .flex_col()
                    .gap(px(Space::XS))
                    .child(label(status_text, Type::LABEL, colors.text))
                    .child(label(
                        validation_error.unwrap_or_else(|| {
                            "Deterministic baseline + intervention pair · seed 0".to_string()
                        }),
                        Type::META,
                        colors.text_faint,
                    )),
            )
            .child(div().w_full())
            .child(div().w(px(230.0)).child(btn_primary(
                colors,
                action_icon,
                action_label,
                action_enabled.then(|| {
                    cx.listener(|console, _: &ClickEvent, _window, cx| {
                        console.advance_or_run();
                        cx.notify();
                    })
                }),
            )))
            .child(label("Ctrl+Enter", Type::META, colors.text_faint))
    }
}

impl Render for Console {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = self.colors();
        let topbar = self.topbar(&colors, cx);
        let statusbar = self.statusbar(&colors, cx);

        // The content region depends on the destination. Only the experiment
        // view carries the workflow stepper and the contextual inspector; the
        // other views are plain pages.
        let content: AnyElement = match self.view {
            View::Home => self.home_view(&colors, cx).into_any_element(),
            View::Models => self.models_view(&colors, cx).into_any_element(),
            View::Runs => self.runs_view(&colors, cx).into_any_element(),
            View::Settings => self.settings_view(&colors, cx).into_any_element(),
            View::Experiment => {
                let inspector = self.inspector(&colors, cx);
                // The column is the growing region; the inspector is a fixed
                // aside beside it. Note the direction is set once here --
                // re-calling .flex() on this element would silently override
                // flex_col with a row and squeeze the workspace out.
                let column = div()
                    .flex()
                    .flex_col()
                    .w_full()
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
                if self.inspector_open {
                    // The inspector is an aside beside the workspace, not a
                    // band below it: the row is the thing that places them.
                    div()
                        .flex()
                        .flex_row()
                        .w_full()
                        .flex_1()
                        .min_h(px(0.0))
                        .child(column)
                        .child(
                            div()
                                .w(px(300.0))
                                .flex_none()
                                .min_w(px(0.0))
                                .child(div().w_full().child(inspector)),
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
            .child(self.nav_rail(&colors, cx))
            .child(content);

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(colors.canvas)
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(|console, event: &KeyDownEvent, _window, cx| {
                console.picker_key(event, cx);
            }))
            .child(topbar.flex_none())
            .child(body)
            .child(statusbar.flex_none())
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

fn isolate_bidi(text: &str) -> String {
    format!("\u{2068}{text}\u{2069}")
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
            cx.set_menus([Menu::new("Ember").items([MenuItem::action("Quit Ember", Quit)])]);
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
                    cx.new(|cx| gpui_kit::component::Root::new(console, window, cx))
                },
            )
            .expect("the experiment console window failed");
            cx.activate(true);
        });
    Ok(())
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
        assert_ne!(dark.warn_box_bg, light.warn_box_bg);
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
    use super::{Console, Preset, WorkspaceStep, FONT_ARABIC, FONT_MONO, FONT_SANS};
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
            // The site picker is an advanced control and the panel starts
            // collapsed, so open it before reaching for the picker.
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
            window.click(SharedString::from("btn:RUN EXPERIMENT"), cx);
            assert_eq!(console.read(cx).status, super::Status::Preparing);
            assert!(matches!(worker_rx.try_recv().unwrap(), super::WorkerMsg::Prepare(path) if path == "fixture.gguf"));
            let loading = SharedString::from("btn:LOADING MODEL…");
            assert!(window.find(loading.clone()).visible());
            window.click(loading, cx);
            assert!(worker_rx.try_recv().is_err(), "disabled run must not submit another job");
            reply_tx.send(super::WorkerReply::Prepared(Box::new(Err("fixture load failure".into())))).unwrap();
            console.update(cx, |console, cx| { console.drain_replies(cx); cx.notify(); });
            window.render_frame(cx);
            assert_eq!(console.read(cx).status, super::Status::Idle);
            assert_eq!(console.read(cx).error.as_deref(), Some("fixture load failure"));
            window.click(SharedString::from("btn:RUN EXPERIMENT"), cx);
            assert!(matches!(worker_rx.try_recv().unwrap(), super::WorkerMsg::Prepare(_)));
        }).unwrap();
    }
}

/// Offscreen test scenes use the production Console render tree, CoreText, and
/// Metal. This exercises layout without automating another desktop application.
#[cfg(all(target_os = "macos", feature = "gui-tests"))]
fn render_test_artifacts(directory: &std::path::Path) -> anyhow::Result<()> {
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
    let mut real_replies: Option<(WorkerReply, WorkerReply)> = None;
    for (name, width, height) in [("standard", 1180., 720.), ("minimum", 980., 620.)] {
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
        }
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
        }
    }
    Ok(())
}
