//! Ember v0.6 native experiment console (`ember gui`).
//!
//! A native, single-window console over the exact same v0.5 pipeline as the
//! web console (`ember web-gui`). The UI is built with gpui (Zed's
//! GPU-accelerated framework, rendered through blade/Vulkan on Linux) and
//! every experiment is executed in a worker thread through the shared
//! `GuiSession` core, which in turn calls `prepare_run` / `execute_prepared`
//! — the same code path as `ember experiment run`. No inference logic lives
//! in the UI.
//!
//! Arabic input/output is shaped and laid out RTL by cosmic-text (gpui's text
//! engine). The Noto Sans / Noto Sans Mono / Noto Naskh Arabic fonts are
//! embedded so rendering is identical on any machine, fully offline.

use crate::gui::{
    discover_models, parse_run_request, ExperimentComparison, RestoreBundle, RunBundle, RunConfig,
    RunOutput, RunRequest, SessionInfo,
};
use clap::Args as ClapArgs;
use ember::quant_k::KStrategy;
use gpui::prelude::*;
use gpui::*;
use std::borrow::Cow;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod chart;
mod components;
mod icons;
mod input;
mod theme;

use components::*;
use input::{InputEvent, InputId, InputKind, TextInput};
use theme::{AppearanceMode, Colors};

// ---------------------------------------------------------------------------
// embedded fonts (SIL OFL 1.1, see src/gui_fonts/LICENSE.txt)
// ---------------------------------------------------------------------------

const FONT_SANS: &[u8] = include_bytes!("gui_fonts/NotoSans-Regular.ttf");
const FONT_MONO: &[u8] = include_bytes!("gui_fonts/NotoSansMono-Regular.ttf");
const FONT_ARABIC: &[u8] = include_bytes!("gui_fonts/NotoNaskhArabic-Regular.ttf");
const FONT_SANS_NAME: &str = "Noto Sans";
const FONT_MONO_NAME: &str = "Noto Sans Mono";
const FONT_ARABIC_NAME: &str = "Noto Naskh Arabic";

/// `ember gui` (native) CLI arguments.
#[derive(ClapArgs)]
pub(crate) struct NativeGuiArgs {}

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
const OPERATIONS: [&str; 5] = ["replace", "zero", "scale", "interpolate", "add-delta"];
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

fn site_label(site: &str) -> &'static str {
    match site {
        "before-layer" => "Before the transformer layer",
        "after-attention" => "After attention",
        "after-mlp" => "After feed-forward processing",
        "after-layer" => "After the transformer layer",
        "before-logits" => "Before output prediction",
        "after-logits" => "After output prediction",
        _ => "Custom location",
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
        ComboId::Model => value
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(value)
            .trim_end_matches(".gguf")
            .to_string(),
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
            Self::Prompt => "Choose a model and write the input",
            Self::Intervention => "Describe the internal change",
            Self::Review => "Check the setup and compare outputs",
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
    op_options: Vec<String>,
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
    open_combo: Option<ComboId>,
    combo_active: usize,
    model_filter: String,
    menu_generation: u64,
    focus_handle: FocusHandle,
    inputs: Inputs,
    step: WorkspaceStep,
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
        Console {
            worker_tx,
            reply_rx,
            model_options: models,
            model_path,
            site_options: STAGES.iter().map(|s| s.to_string()).collect(),
            site: "after-mlp".to_string(),
            layer,
            op_options: OPERATIONS.iter().map(|s| s.to_string()).collect(),
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
            open_combo: None,
            combo_active: 0,
            model_filter: String::new(),
            menu_generation: 0,
            focus_handle: cx.focus_handle(),
            inputs,
            step: WorkspaceStep::Prompt,
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

    fn model_name(&self) -> String {
        self.session
            .as_ref()
            .map(|info| info.model_name.clone())
            .unwrap_or_else(|| {
                self.model_path
                    .rsplit(['/', '\\'])
                    .next()
                    .unwrap_or(&self.model_path)
                    .trim_end_matches(".gguf")
                    .to_string()
            })
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
    ) -> Stateful<Div> {
        div()
            .id(ElementId::Name(SharedString::from(format!(
                "pipeline:{id}"
            ))))
            .min_w(px(54.0))
            .px_2()
            .py_1()
            .flex()
            .items_center()
            .justify_center()
            .bg(if accent {
                colors.accent_soft
            } else {
                colors.surface_raised
            })
            .border_1()
            .border_color(if accent { colors.accent } else { colors.border })
            .rounded_md()
            .cursor_pointer()
            .hover(|node| node.border_color(colors.border_strong).bg(colors.hover))
            .on_click(cx.listener(move |console, _: &ClickEvent, _window, cx| {
                console.step = step;
                cx.notify();
            }))
            .child(mono(
                text,
                9.75,
                if accent {
                    colors.accent
                } else {
                    colors.text_muted
                },
            ))
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
            "scale" => format!("×{}", context.value),
            "zero" => "ZERO".to_string(),
            "replace" => format!("COPY L{}", context.source_layer),
            "interpolate" => format!("BLEND {}", context.value),
            "add-delta" => format!("Δ L{}", context.source_layer),
            _ => context.op.to_ascii_uppercase(),
        };
        let completed = self.baseline.is_some() && self.status == Status::Idle;
        let arrow = || mono("→", 11.0, colors.text_faint);
        let track_label =
            |text: &'static str, color: Rgba| mono(text, 8.0, color).w(px(78.0)).flex_none();

        div()
            .w_full()
            .p_4()
            .flex_col()
            .gap_3()
            .bg(colors.surface)
            .border_1()
            .border_color(colors.border)
            .rounded(px(10.0))
            .child(
                div()
                    .flex()
                    .items_center()
                    .child(label("EXPERIMENT PIPELINE", 9.25, colors.text_faint))
                    .child(div().w_full())
                    .child(chip(
                        if completed { "MEASURED" } else { "PLANNED" },
                        if completed {
                            colors.ok
                        } else {
                            colors.text_faint
                        },
                    )),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(self.pipeline_node(
                        colors,
                        "input",
                        "INPUT".to_string(),
                        false,
                        WorkspaceStep::Prompt,
                        cx,
                    ))
                    .child(arrow())
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(track_label("BASELINE", colors.text_faint))
                                    .child(self.pipeline_node(
                                        colors,
                                        "baseline",
                                        "ORIGINAL".to_string(),
                                        false,
                                        WorkspaceStep::Review,
                                        cx,
                                    ))
                                    .child(arrow())
                                    .child(self.pipeline_node(
                                        colors,
                                        "baseline-generate",
                                        "GENERATE".to_string(),
                                        false,
                                        WorkspaceStep::Review,
                                        cx,
                                    ))
                                    .child(
                                        div().flex_1().min_w(px(16.0)).h(px(1.0)).bg(colors.border),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(track_label("INTERVENTION", colors.accent))
                                    .child(self.pipeline_node(
                                        colors,
                                        "target",
                                        target,
                                        false,
                                        WorkspaceStep::Intervention,
                                        cx,
                                    ))
                                    .child(arrow())
                                    .child(self.pipeline_node(
                                        colors,
                                        "operation",
                                        operation,
                                        true,
                                        WorkspaceStep::Intervention,
                                        cx,
                                    ))
                                    .child(arrow())
                                    .child(self.pipeline_node(
                                        colors,
                                        "intervention-generate",
                                        "GENERATE".to_string(),
                                        false,
                                        WorkspaceStep::Review,
                                        cx,
                                    ))
                                    .child(
                                        div()
                                            .flex_1()
                                            .min_w(px(16.0))
                                            .h(px(1.0))
                                            .bg(Hsla::from(colors.accent).opacity(0.45)),
                                    ),
                            ),
                    )
                    .child(arrow())
                    .child(self.pipeline_node(
                        colors,
                        "compare",
                        "COMPARE".to_string(),
                        false,
                        WorkspaceStep::Review,
                        cx,
                    )),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .child(mono(
                        format!(
                            "TOKEN  {}",
                            token_label(&context.token).to_ascii_uppercase()
                        ),
                        8.5,
                        colors.text_faint,
                    ))
                    .child(div().w_full())
                    .child(mono(
                        format!("≤{} TOKENS  ·  SEED 0", context.max_tokens),
                        8.5,
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

    fn cycle_appearance(&mut self, cx: &mut Context<Self>) {
        self.appearance = self.appearance.next();
        self.appearance.persist();
        let colors = self.colors();
        for input in self.inputs.all() {
            input.update(cx, |input, cx| input.set_palette(&colors, cx));
        }
        cx.notify();
    }

    fn system_appearance_changed(&mut self, dark: bool, cx: &mut Context<Self>) {
        if self.system_dark == dark {
            return;
        }
        self.system_dark = dark;
        if self.appearance == AppearanceMode::System {
            let colors = self.colors();
            for input in self.inputs.all() {
                input.update(cx, |input, cx| input.set_palette(&colors, cx));
            }
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
        self.open_combo = None;
        self.model_filter.clear();
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

    fn visible_combo_options(&self, combo: ComboId) -> Vec<String> {
        let options = match combo {
            ComboId::Model => &self.model_options,
            ComboId::Site => &self.site_options,
            ComboId::Op => &self.op_options,
            ComboId::Source => &self.source_options,
            ComboId::Token => &self.token_options,
            ComboId::Execution => &self.execution_options,
        };
        if combo != ComboId::Model || self.model_filter.is_empty() {
            return options.clone();
        }
        let query = self.model_filter.to_lowercase();
        options
            .iter()
            .filter(|option| option.to_lowercase().contains(&query))
            .cloned()
            .collect()
    }

    fn picker_key(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) {
        let Some(combo) = self.open_combo else {
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
                match self.step {
                    WorkspaceStep::Prompt => self.step = WorkspaceStep::Intervention,
                    WorkspaceStep::Intervention => self.step = WorkspaceStep::Review,
                    WorkspaceStep::Review => self.run(),
                }
                cx.notify();
            }
            return;
        };
        let options = self.visible_combo_options(combo);
        match event.keystroke.key.as_str() {
            "escape" => {
                self.open_combo = None;
                self.model_filter.clear();
            }
            "up" => {
                self.combo_active = self.combo_active.saturating_sub(1);
            }
            "down" => {
                self.combo_active = (self.combo_active + 1).min(options.len().saturating_sub(1));
            }
            "enter" | "return" => {
                if let Some(value) = options.get(self.combo_active) {
                    self.select_combo(combo, value, cx);
                    return;
                }
            }
            "backspace" if combo == ComboId::Model => {
                self.model_filter.pop();
                self.combo_active = 0;
            }
            _ if combo == ComboId::Model
                && !event.keystroke.modifiers.control
                && !event.keystroke.modifiers.platform =>
            {
                if let Some(text) = event.keystroke.key_char.as_deref() {
                    self.model_filter.push_str(text);
                    self.combo_active = 0;
                }
            }
            _ => {}
        }
        cx.notify();
    }

    fn picker(
        &self,
        colors: &Colors,
        id: &'static str,
        combo: ComboId,
        selected: &str,
        options: &[String],
        cx: &mut Context<Self>,
    ) -> Div {
        let open = self.open_combo == Some(combo);
        let selected_index = options
            .iter()
            .position(|option| option == selected)
            .unwrap_or(0);
        let toggle = cx.listener(move |console, _: &ClickEvent, window, cx| {
            console.focus_handle.focus(window);
            console.open_combo = if console.open_combo == Some(combo) {
                None
            } else {
                console.combo_active = selected_index;
                console.model_filter.clear();
                console.menu_generation = console.menu_generation.wrapping_add(1);
                Some(combo)
            };
            cx.notify();
        });
        let button = div()
            .id(ElementId::Name(SharedString::from(id)))
            .w_full()
            .px_2()
            .py_1()
            .flex()
            .items_center()
            .justify_between()
            .gap_2()
            .h(px(32.0))
            .bg(colors.surface_raised)
            .border_1()
            .border_color(colors.border)
            .rounded_md()
            .cursor_pointer()
            .hover(|style| style.bg(colors.hover).border_color(colors.border_strong))
            .active(|style| style.bg(colors.selected))
            .on_click(toggle)
            .child(label(combo_value_label(combo, selected), 12.0, colors.text))
            .child(
                icons::icon(icons::CHEVRON_DOWN)
                    .size(px(14.0))
                    .text_color(colors.text_faint),
            );

        if open {
            let options = self.visible_combo_options(combo);
            let list = options
                .iter()
                .enumerate()
                .map(|(index, opt)| {
                    let opt = opt.clone();
                    let is_selected = opt == selected;
                    let is_active = index == self.combo_active;
                    let listener = {
                        let opt = opt.clone();
                        cx.listener(move |console, _: &ClickEvent, _w, cx| {
                            console.select_combo(combo, &opt, cx);
                        })
                    };
                    let opt_id = ElementId::Name(SharedString::from(format!("{id}:{opt}")));
                    div()
                        .id(opt_id)
                        .w_full()
                        .px_2()
                        .py_1()
                        .flex()
                        .items_center()
                        .gap_2()
                        .cursor_pointer()
                        .rounded(px(5.0))
                        .when(is_active, |style| style.bg(colors.hover))
                        .when(is_selected, |style| style.bg(colors.selected))
                        .hover(|style| style.bg(colors.hover))
                        .on_click(listener)
                        .when(is_selected, |row| {
                            row.child(
                                icons::icon(icons::CHECK)
                                    .size(px(12.0))
                                    .text_color(colors.accent),
                            )
                        })
                        .child(label(combo_value_label(combo, &opt), 12.0, colors.text))
                        .into_any_element()
                })
                .collect::<Vec<_>>();
            let dismiss = cx.listener(|console, _: &MouseDownEvent, _window, cx| {
                console.open_combo = None;
                console.model_filter.clear();
                cx.notify();
            });
            let search_hint = (combo == ComboId::Model).then(|| {
                div()
                    .px_2()
                    .py_1()
                    .border_b_1()
                    .border_color(colors.border)
                    .child(mono(
                        if self.model_filter.is_empty() {
                            "type to filter models".to_string()
                        } else {
                            format!("filter: {}", self.model_filter)
                        },
                        9.0,
                        colors.text_faint,
                    ))
            });
            let menu = div()
                .id(ElementId::Name(SharedString::from(format!("{id}:list"))))
                .flex_col()
                .w(px(248.0))
                .max_h(px(260.0))
                .overflow_y_scroll()
                .p_1()
                .bg(colors.overlay)
                .border_1()
                .border_color(colors.border_strong)
                .rounded_md()
                .shadow_lg()
                .on_mouse_down_out(dismiss)
                .children(search_hint)
                .child(div().flex_col().children(list))
                .with_animation(
                    ElementId::Name(SharedString::from(format!(
                        "picker-in:{id}:{}",
                        self.menu_generation
                    ))),
                    Animation::new(Duration::from_millis(130)),
                    |menu, delta| menu.opacity(delta).top(px(-3.0 * (1.0 - delta))),
                );
            div().relative().w_full().child(button).child(deferred(
                anchored()
                    .position_mode(AnchoredPositionMode::Local)
                    .anchor(Corner::TopLeft)
                    .offset(point(px(0.0), px(35.0)))
                    .child(menu),
            ))
        } else {
            div().relative().w_full().child(button)
        }
    }

    fn header(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let session_chip = match &self.session {
            Some(info) => format!(
                "{} \u{00b7} {} \u{00b7} {} layers",
                info.model_name, info.architecture, info.n_layers
            ),
            None => format!("{} \u{2014} not loaded", self.model_name()),
        };
        let toggle = cx.listener(|console, _: &ClickEvent, _w, cx| {
            console.cycle_appearance(cx);
        });
        div()
            .flex()
            .flex_row()
            .items_center()
            .gap_3()
            .px_4()
            .h(px(44.0))
            .w_full()
            .bg(colors.surface)
            .border_b_1()
            .border_color(colors.border)
            .child(
                div()
                    .size(px(20.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(colors.accent)
                    .rounded(px(5.0))
                    .child(label("E", 11.0, rgb(0xffffff))),
            )
            .child(
                div()
                    .flex_col()
                    .child(label("ember", 13.0, colors.text))
                    .child(label("causal experiment workbench", 9.0, colors.text_faint)),
            )
            .child(div().w_full())
            .child(
                div()
                    .px_2()
                    .py_1()
                    .bg(colors.surface_raised)
                    .rounded_full()
                    .child(mono(session_chip, 10.0, colors.text_muted)),
            )
            .child(
                div()
                    .id(ElementId::Name(SharedString::from("theme-toggle")))
                    .size(px(28.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .gap_1()
                    .px_2()
                    .w_auto()
                    .bg(colors.surface_raised)
                    .border_1()
                    .border_color(colors.border)
                    .rounded_md()
                    .cursor_pointer()
                    .hover(|s| s.bg(colors.hover).border_color(colors.border_strong))
                    .on_click(toggle)
                    .child(
                        icons::icon(match self.appearance {
                            AppearanceMode::System => icons::MONITOR,
                            AppearanceMode::Dark => icons::MOON,
                            AppearanceMode::Light => icons::SUN,
                        })
                        .size(px(13.0))
                        .text_color(colors.text_muted),
                    )
                    .child(label(self.appearance.label(), 9.0, colors.text_muted)),
            )
    }

    fn sidebar(&self, colors: &Colors, cx: &mut Context<Self>) -> Stateful<Div> {
        let steps = WorkspaceStep::ALL
            .into_iter()
            .map(|step| {
                let selected = self.step == step;
                div()
                    .id(ElementId::Name(SharedString::from(format!(
                        "workflow-step:{}",
                        step.number()
                    ))))
                    .w_full()
                    .px_3()
                    .py_2()
                    .flex()
                    .items_center()
                    .gap_3()
                    .rounded_md()
                    .cursor_pointer()
                    .when(selected, |row| row.bg(colors.selected))
                    .hover(|row| row.bg(colors.hover))
                    .on_click(cx.listener(move |console, _: &ClickEvent, _window, cx| {
                        console.step = step;
                        console.open_combo = None;
                        cx.notify();
                    }))
                    .child(
                        div()
                            .size(px(24.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded_full()
                            .bg(if selected {
                                colors.accent
                            } else {
                                colors.surface_raised
                            })
                            .child(label(
                                step.number(),
                                10.0,
                                if selected {
                                    rgb(0xffffff)
                                } else {
                                    colors.text_muted
                                },
                            )),
                    )
                    .child(
                        div()
                            .flex_col()
                            .gap_1()
                            .child(label(step.label(), 11.0, colors.text))
                            .child(label(step.hint(), 8.5, colors.text_faint)),
                    )
                    .into_any_element()
            })
            .collect::<Vec<_>>();

        let presets = [
            (
                Preset::ZeroMiddle,
                "Zero a middle layer",
                "A clear causal-ablation starting point",
            ),
            (
                Preset::ScaleLate,
                "Weaken a late layer",
                "Test a near-output representation at 50%",
            ),
            (
                Preset::CopyEarlier,
                "Copy an earlier layer",
                "Replace a late state with an earlier capture",
            ),
            (
                Preset::ArabicMorphology,
                "Arabic morphology",
                "Matched-span experiment using an Arabic prompt",
            ),
        ]
        .into_iter()
        .map(|(preset, title, hint)| self.preset_card(colors, preset, title, hint, cx))
        .collect::<Vec<_>>();

        let history: Vec<AnyElement> = if self.history.is_empty() {
            vec![label(
                "Completed experiments will appear here during this session.",
                9.0,
                colors.text_faint,
            )
            .into_any_element()]
        } else {
            self.history
                .iter()
                .map(|entry| {
                    div()
                        .w_full()
                        .px_2()
                        .py_2()
                        .rounded_md()
                        .bg(colors.surface_raised)
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .child(mono(
                                    format!("RUN {:02}", entry.number),
                                    8.5,
                                    colors.text_faint,
                                ))
                                .child(div().w_full())
                                .child(status_dot(
                                    if entry.ok { colors.ok } else { colors.err },
                                    false,
                                )),
                        )
                        .child(label(entry.summary.clone(), 10.0, colors.text))
                        .child(label(
                            entry.outcome.clone(),
                            9.0,
                            if entry.ok { colors.ok } else { colors.err },
                        ))
                        .into_any_element()
                })
                .collect()
        };

        div()
            .id(ElementId::Name(SharedString::from("workflow-rail")))
            .flex_col()
            .h_full()
            .overflow_y_scroll()
            .p_3()
            .gap_4()
            .child(
                div()
                    .flex_col()
                    .gap_1()
                    .child(label("EXPERIMENT WORKFLOW", 9.0, colors.text_faint))
                    .children(steps),
            )
            .child(rule_h(colors))
            .child(
                div()
                    .flex_col()
                    .gap_2()
                    .child(label("START FROM A PRESET", 9.0, colors.text_faint))
                    .children(presets),
            )
            .child(rule_h(colors))
            .child(
                div()
                    .flex_col()
                    .gap_2()
                    .child(label("RECENT RUNS", 9.0, colors.text_faint))
                    .children(history),
            )
    }

    fn preset_card(
        &self,
        colors: &Colors,
        preset: Preset,
        title: &'static str,
        hint: &'static str,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        div()
            .id(ElementId::Name(SharedString::from(format!(
                "preset:{title}"
            ))))
            .w_full()
            .px_2()
            .py_2()
            .flex_col()
            .gap_1()
            .bg(colors.surface)
            .border_1()
            .border_color(colors.border)
            .rounded_md()
            .cursor_pointer()
            .hover(|card| card.bg(colors.hover).border_color(colors.border_strong))
            .on_click(cx.listener(move |console, _: &ClickEvent, _window, cx| {
                console.apply_preset(preset, cx);
            }))
            .child(label(title, 10.0, colors.text))
            .child(label(hint, 8.5, colors.text_faint))
            .into_any_element()
    }

    fn generation_option(
        &self,
        colors: &Colors,
        value: usize,
        title: &'static str,
        hint: &'static str,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let selected = self.max_tokens == value.to_string();
        div()
            .id(ElementId::Name(SharedString::from(format!(
                "generation-length:{value}"
            ))))
            .w(relative(0.333))
            .px_3()
            .py_2()
            .flex_col()
            .gap_1()
            .border_1()
            .border_color(if selected {
                colors.border_strong
            } else {
                colors.border
            })
            .bg(if selected {
                colors.surface
            } else {
                colors.surface_raised
            })
            .rounded_md()
            .cursor_pointer()
            .hover(|card| card.bg(colors.hover).border_color(colors.border_strong))
            .on_click(cx.listener(move |console, _: &ClickEvent, _window, cx| {
                console.set_max_tokens(value, cx);
            }))
            .child(label(title, 11.0, colors.text))
            .child(label(hint, 9.0, colors.text_faint))
    }

    fn operation_card(
        &self,
        colors: &Colors,
        operation: &'static str,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let selected = self.op == operation;
        div()
            .id(ElementId::Name(SharedString::from(format!(
                "operation-card:{operation}"
            ))))
            .w(relative(0.5))
            .min_h(px(70.0))
            .px_3()
            .py_3()
            .flex_col()
            .gap_1()
            .border_1()
            .border_color(if selected {
                colors.accent
            } else {
                colors.border
            })
            .bg(if selected {
                colors.selected
            } else {
                colors.surface
            })
            .rounded(px(9.0))
            .cursor_pointer()
            .hover(|card| card.bg(colors.hover).border_color(colors.border_strong))
            .on_click(cx.listener(move |console, _: &ClickEvent, _window, cx| {
                console.select_combo(ComboId::Op, operation, cx);
                console.step = WorkspaceStep::Intervention;
            }))
            .child(
                div()
                    .flex()
                    .items_center()
                    .child(label(operation_label(operation), 11.0, colors.text))
                    .child(div().w_full())
                    .when(selected, |row| {
                        row.child(
                            icons::icon(icons::CHECK)
                                .size(px(13.0))
                                .text_color(colors.accent),
                        )
                    }),
            )
            .child(label(operation_hint(operation), 9.0, colors.text_faint))
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
                .rounded_md()
                .flex()
                .items_center()
                .gap_2()
                .child(
                    icons::icon(icons::WARNING)
                        .size(px(15.0))
                        .text_color(colors.err),
                )
                .child(label(error.clone(), 10.0, colors.err))
        });
        let warning = self.warning.as_ref().map(|warning| {
            div()
                .w_full()
                .px_3()
                .py_2()
                .bg(colors.warn_box_bg)
                .border_1()
                .border_color(colors.warn_box_border)
                .rounded_md()
                .flex()
                .items_center()
                .gap_2()
                .child(
                    icons::icon(icons::WARNING)
                        .size(px(15.0))
                        .text_color(colors.warn),
                )
                .child(label(warning.clone(), 10.0, colors.warn))
        });
        div().flex_col().gap_2().children(error).children(warning)
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
                    11.0,
                    None,
                    cx,
                ),
            )
        });

        div()
            .flex_col()
            .gap_4()
            .child(
                div()
                    .flex_col()
                    .gap_1()
                    .child(label("Prepare the experiment", 20.0, colors.text))
                    .child(label(
                        "Choose a local GGUF model and give it the prompt you want to study.",
                        11.0,
                        colors.text_muted,
                    )),
            )
            .child(panel(
                colors,
                div()
                    .flex_col()
                    .gap_3()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .child(section_label(colors, "MODEL"))
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
                            .gap_2()
                            .child(label(model_status.1, 9.5, colors.text_faint))
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
            .child(panel(
                colors,
                div()
                    .flex_col()
                    .gap_3()
                    .child(section_label(colors, "PROMPT"))
                    .child(text_input(
                        colors,
                        self.inputs.prompt.clone(),
                        FONT_ARABIC_NAME,
                        15.0,
                        Some(360.0),
                        cx,
                    ))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .child(label(
                                format!("{} characters", self.prompt.chars().count()),
                                9.0,
                                colors.text_faint,
                            ))
                            .child(div().w_full())
                            .child(label(
                                "Arabic and mixed-direction text supported",
                                9.0,
                                colors.text_faint,
                            )),
                    ),
            ))
            .child(
                div()
                    .flex_col()
                    .gap_2()
                    .child(label("RESPONSE LENGTH", 9.0, colors.text_faint))
                    .child(
                        div()
                            .flex()
                            .gap_2()
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
            .gap_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .id("layer-minus")
                            .size(px(32.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded_md()
                            .border_1()
                            .border_color(colors.border)
                            .cursor_pointer()
                            .hover(|button| button.bg(colors.hover))
                            .on_click(cx.listener(|console, _: &ClickEvent, _window, cx| {
                                console.adjust_layer(-1, cx);
                            }))
                            .child(label("−", 16.0, colors.text)),
                    )
                    .child(div().w(px(76.0)).child(text_input(
                        colors,
                        self.inputs.layer.clone(),
                        FONT_MONO_NAME,
                        12.0,
                        None,
                        cx,
                    )))
                    .child(label(limit, 10.0, colors.text_muted))
                    .child(
                        div()
                            .id("layer-plus")
                            .size(px(32.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded_md()
                            .border_1()
                            .border_color(colors.border)
                            .cursor_pointer()
                            .hover(|button| button.bg(colors.hover))
                            .on_click(cx.listener(|console, _: &ClickEvent, _window, cx| {
                                console.adjust_layer(1, cx);
                            }))
                            .child(label("+", 15.0, colors.text)),
                    ),
            )
            .child(label(position, 9.0, colors.text_faint))
    }

    fn intervention_step(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let needs_source = matches!(self.op.as_str(), "replace" | "interpolate" | "add-delta");
        let needs_value = matches!(self.op.as_str(), "scale" | "interpolate");

        let source_controls = needs_source.then(|| {
            div()
                .flex_col()
                .gap_3()
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
                            12.0,
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
                    "STRENGTH MULTIPLIER"
                },
                text_input(
                    colors,
                    self.inputs.value.clone(),
                    FONT_MONO_NAME,
                    12.0,
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
                    12.0,
                    None,
                    cx,
                ),
            )
        });

        div()
            .flex_col()
            .gap_4()
            .child(
                div()
                    .flex_col()
                    .gap_1()
                    .child(label("Choose the internal change", 20.0, colors.text))
                    .child(label(
                        "Start with the research question. Exact hook names remain available in Advanced controls.",
                        11.0,
                        colors.text_muted,
                    )),
            )
            .child(
                div()
                    .flex_col()
                    .gap_2()
                    .child(label("WHAT SHOULD CHANGE?", 9.0, colors.text_faint))
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(self.operation_card(colors, "zero", cx))
                            .child(self.operation_card(colors, "scale", cx)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(self.operation_card(colors, "replace", cx))
                            .child(self.operation_card(colors, "interpolate", cx)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(self.operation_card(colors, "add-delta", cx))
                            .child(div().w(relative(0.5))),
                    ),
            )
            .child(panel(
                colors,
                div()
                    .flex_col()
                    .gap_3()
                    .child(section_label(colors, "WHERE"))
                    .child(field(
                        colors,
                        "LOCATION IN EACH LAYER",
                        self.picker(
                            colors,
                            "site-picker",
                            ComboId::Site,
                            &self.site,
                            &self.site_options,
                            cx,
                        ),
                    ))
                    .when(per_layer(&self.site), |content| {
                        content.child(field(colors, "MODEL LAYER", self.layer_stepper(colors, cx)))
                    })
                    .children(value_control)
                    .children(source_controls),
            ))
            .child(panel(
                colors,
                div()
                    .flex_col()
                    .gap_3()
                    .child(section_label(colors, "TARGET"))
                    .child(field(
                        colors,
                        "TOKENS TO AFFECT",
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
                .rounded_md()
                .bg(colors.warn_box_bg)
                .border_1()
                .border_color(colors.warn_box_border)
                .flex()
                .items_center()
                .gap_2()
                .child(
                    icons::icon(icons::WARNING)
                        .size(px(15.0))
                        .text_color(colors.warn),
                )
                .child(label(
                    "The intervention completed successfully but did not change the generated text.",
                    10.0,
                    colors.warn,
                )),
            (Some(_), Some(_), Some(comparison)) => div()
                .px_3()
                .py_2()
                .rounded_md()
                .bg(colors.accent_soft)
                .border_1()
                .border_color(colors.accent)
                .flex()
                .items_center()
                .gap_2()
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
                    10.0,
                    colors.text,
                )),
            _ if self.busy() => div()
                .px_3()
                .py_2()
                .rounded_md()
                .bg(colors.accent_soft)
                .child(label(
                    "The model is running both the baseline and intervention. Results will appear here.",
                    10.0,
                    colors.text_muted,
                )),
            _ => div()
                .px_3()
                .py_2()
                .rounded_md()
                .bg(colors.surface_raised)
                .child(label(
                    "Review the experiment summary, then run it to produce a controlled comparison.",
                    10.0,
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
                .gap_1()
                .child(label(title, 8.0, colors.text_faint))
                .child(mono(value, 10.5, colors.text))
                .child(label(detail, 8.5, colors.text_muted))
        };
        let divider = || div().w(px(1.0)).h(px(42.0)).bg(colors.border).flex_none();

        div()
            .w_full()
            .px_3()
            .py_2()
            .flex()
            .items_center()
            .gap_3()
            .bg(colors.surface)
            .border_1()
            .border_color(colors.border)
            .rounded_md()
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

    fn result_tabs(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        div()
            .flex()
            .items_center()
            .gap_1()
            .border_b_1()
            .border_color(colors.border)
            .children(ResultView::ALL.into_iter().map(|view| {
                let selected = self.result_view == view;
                div()
                    .id(ElementId::Name(SharedString::from(format!(
                        "result-view:{}",
                        view.label()
                    ))))
                    .px_3()
                    .py_2()
                    .bg(if selected {
                        colors.surface_raised
                    } else {
                        colors.canvas
                    })
                    .border_b_1()
                    .border_color(if selected {
                        colors.border_strong
                    } else {
                        colors.canvas
                    })
                    .cursor_pointer()
                    .hover(|tab| tab.bg(colors.hover))
                    .on_click(cx.listener(move |console, _: &ClickEvent, _window, cx| {
                        console.result_view = view;
                        cx.notify();
                    }))
                    .child(label(
                        view.label(),
                        9.5,
                        if selected {
                            colors.text
                        } else {
                            colors.text_faint
                        },
                    ))
            }))
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
                .gap_2()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .child(
                            div()
                                .flex_col()
                                .gap_1()
                                .child(label("REPRESENTATION DIVERGENCE", 9.0, colors.text_faint))
                                .child(label(
                                    "At which layers does the intervention diverge from baseline?",
                                    11.0,
                                    colors.text,
                                )),
                        )
                        .child(div().w_full())
                        .child(label("COPY CSV", 8.0, colors.text_faint))
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
            .gap_3()
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
                label("No token comparison is available.", 10.0, colors.text_faint),
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
                    .p_2()
                    .flex_col()
                    .gap_1()
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
                    .rounded_md()
                    .child(mono(
                        format!("STEP {}", token.position),
                        7.5,
                        colors.text_faint,
                    ))
                    .child(label(baseline, 11.0, colors.text))
                    .child(rule_h(colors))
                    .child(label(intervention, 11.0, colors.text))
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
                .gap_2()
                .child(label(
                    "TOKEN-LEVEL OUTPUT COMPARISON",
                    9.0,
                    colors.text_faint,
                ))
                .child(label(divergence, 11.0, colors.text))
                .child(
                    div()
                        .flex()
                        .gap_2()
                        .child(chip("BASELINE", colors.text_muted))
                        .child(chip("INTERVENTION", colors.accent))
                        .child(div().w_full())
                        .children((count > end).then(|| {
                            mono(
                                format!("showing {}–{} of {count}", start + 1, end),
                                8.0,
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
                        .gap_1()
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
                .map(|event| mono(event.to_string(), 8.5, colors.text_muted).into_any_element())
                .collect::<Vec<_>>()
        });
        panel(
            colors,
            div()
                .flex_col()
                .gap_2()
                .child(label("RAW INTERVENTION TRACE", 9.0, colors.text_faint))
                .children((!events.is_empty()).then(|| div().flex_col().gap_1().children(events)))
                .children(self.intervention.as_ref().map(|output| {
                    mono(
                        format!("bundle  {}", output.bundle_dir),
                        8.5,
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
                .gap_3()
                .child(self.result_summary(colors))
                .child(self.paired_outputs(colors, cx))
                .into_any_element()
        } else {
            match self.result_view {
                ResultView::Overview => div()
                    .flex_col()
                    .gap_3()
                    .child(self.result_summary(colors))
                    .child(self.result_landmarks(colors))
                    .child(self.paired_outputs(colors, cx))
                    .child(self.layer_chart_panel(colors, 190.0, cx))
                    .into_any_element(),
                ResultView::Layers => div()
                    .flex_col()
                    .gap_3()
                    .child(self.layer_chart_panel(colors, 350.0, cx))
                    .into_any_element(),
                ResultView::Tokens => div()
                    .flex_col()
                    .gap_3()
                    .child(self.paired_outputs(colors, cx))
                    .child(self.token_comparison_panel(colors))
                    .into_any_element(),
                ResultView::Trace => div()
                    .flex_col()
                    .gap_3()
                    .child(self.raw_trace_panel(colors))
                    .child(self.verification_panel(colors))
                    .into_any_element(),
            }
        };

        div()
            .flex_col()
            .gap_4()
            .child(
                div()
                    .flex()
                    .items_end()
                    .child(
                        div()
                            .flex_col()
                            .gap_1()
                            .child(label("Review and compare", 20.0, colors.text))
                            .child(label(
                                "The baseline and intervention use the same prompt and deterministic settings.",
                                11.0,
                                colors.text_muted,
                            )),
                    )
                    .child(div().w_full())
                    .when(self.last_config.is_some(), |header| {
                        header.child(
                            div()
                                .w(px(170.0))
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
        let model_name = context
            .model_path
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(&context.model_path)
            .trim_end_matches(".gguf")
            .to_string();
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
                .gap_3()
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
                        11.0,
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
                        10.0,
                        Some(52.0),
                        cx,
                    ),
                ))
                .child(
                    div()
                        .p_2()
                        .bg(colors.surface_raised)
                        .rounded_md()
                        .flex_col()
                        .gap_1()
                        .child(mono(
                            format!("hook     {}", self.site),
                            9.0,
                            colors.text_faint,
                        ))
                        .child(mono(
                            format!("operation {}", self.op),
                            9.0,
                            colors.text_faint,
                        ))
                        .child(mono(
                            format!("tokens    {}", self.token),
                            9.0,
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
            .gap_3()
            .child(
                div()
                    .flex_col()
                    .gap_1()
                    .child(label("MODEL", 8.5, colors.text_faint))
                    .child(label(model_name, 10.0, colors.text))
                    .child(match &self.session {
                        Some(session) => mono(
                            format!(
                                "{} · {} layers · {}d",
                                session.architecture, session.n_layers, session.embed_dim
                            ),
                            8.5,
                            colors.text_muted,
                        ),
                        None => mono("not loaded", 8.5, colors.text_faint),
                    }),
            )
            .child(rule_h(colors))
            .child(
                div()
                    .flex_col()
                    .gap_1()
                    .child(label("INPUT", 8.5, colors.text_faint))
                    .child(multiline(
                        &prompt_excerpt,
                        11.0,
                        colors.text,
                        FONT_ARABIC_NAME,
                    )),
            )
            .child(rule_h(colors))
            .child(
                div()
                    .flex_col()
                    .gap_1()
                    .child(label("TARGET", 8.5, colors.text_faint))
                    .child(multiline(&target, 10.0, colors.text, FONT_SANS_NAME)),
            )
            .child(rule_h(colors))
            .child(
                div()
                    .flex_col()
                    .gap_1()
                    .child(label("INTERVENTION", 8.5, colors.text_faint))
                    .child(label(intervention, 10.0, colors.accent)),
            )
            .child(rule_h(colors))
            .child(
                div()
                    .flex_col()
                    .gap_1()
                    .child(label("GENERATION", 8.5, colors.text_faint))
                    .child(mono(
                        format!(
                            "≤{} tokens · seed 0\n{}",
                            context.max_tokens, context.execution
                        ),
                        9.0,
                        colors.text,
                    )),
            )
            .children(self.intervention.as_ref().map(|output| {
                div().flex_col().gap_3().child(rule_h(colors)).child(
                    div()
                        .flex_col()
                        .gap_1()
                        .child(label("RUN", 8.5, colors.text_faint))
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
                            9.0,
                            colors.text,
                        )),
                )
            }))
            .children(active_metric.map(|metric| {
                div().flex_col().gap_3().child(rule_h(colors)).child(
                    div()
                        .flex_col()
                        .gap_1()
                        .child(label(active_metric_label, 8.5, colors.text_faint))
                        .child(mono(format!("layer {}", metric.layer), 10.0, colors.text))
                        .child(mono(
                            metric.relative_l2_difference.map_or_else(
                                || "relative L2  —".to_string(),
                                |value| format!("relative L2  {value:.6}"),
                            ),
                            9.0,
                            colors.accent,
                        ))
                        .child(mono(
                            metric.cosine_distance.map_or_else(
                                || "cosine distance  —".to_string(),
                                |value| format!("cosine distance  {value:.6}"),
                            ),
                            9.0,
                            colors.text_muted,
                        )),
                )
            }))
            .child(rule_h(colors))
            .child(
                div()
                    .id("advanced-toggle")
                    .flex()
                    .items_center()
                    .py_2()
                    .cursor_pointer()
                    .hover(|row| row.text_color(colors.accent))
                    .on_click(toggle)
                    .child(label("ADVANCED CONTROLS", 9.0, colors.text_muted))
                    .child(div().w_full())
                    .child(label(
                        if self.advanced_open { "HIDE" } else { "SHOW" },
                        9.0,
                        colors.accent,
                    )),
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
                            .gap_3()
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
                    .p_4()
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
                    .gap_2()
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
                            .rounded_md()
                            .child(multiline(
                                &display_text,
                                14.0,
                                colors.text,
                                FONT_ARABIC_NAME,
                            )),
                    )
                    .children(divergence_note.map(|note| {
                        mono(
                            note,
                            9.0,
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
                        9.5,
                        colors.text_muted,
                    ))
                    .child(mono(
                        format!(
                            "prompt {} tok \u{00b7} bundle {}",
                            out.prompt_tokens,
                            short_id(&out.semantic_hash)
                        ),
                        8.5,
                        colors.text_faint,
                    ))
                    .child(mono(out.bundle_dir.clone(), 9.0, colors.text_faint))
            }
            Some(_out) => div().child(label("(empty output)", 12.0, colors.text_faint)),
            None => div().child(label(
                "no run yet \u{2014} outputs appear here",
                12.0,
                colors.text_faint,
            )),
        };
        panel(
            colors,
            div()
                .flex_col()
                .gap_2()
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .child(label(title, 11.0, colors.text_muted))
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
                10.0,
                colors.text_faint,
            ))
        } else {
            div().flex_col().gap_1().children(
                lines
                    .iter()
                    .map(|line| mono(line.clone(), 10.0, colors.text_muted).into_any_element())
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
                .gap_1()
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .child(chip(badge, badge_color))
                        .child(div().w_full())
                        .child(mono(metrics, 10.0, colors.text_faint)),
                )
                .child(detail),
        )
    }

    fn statusbar(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let (dot, status_text) = match self.status {
            Status::Idle => (colors.ok, "Ready to run"),
            Status::Preparing => (colors.warn, "Loading the model…"),
            Status::Running => (colors.busy, "Running baseline and intervention…"),
            Status::Restoring => (colors.busy, "Checking exact restoration…"),
        };
        let validation_error = self.validation_error();
        let action_enabled = !self.busy() && validation_error.is_none();
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
            .gap_3()
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
                    .gap_1()
                    .child(label(status_text, 10.5, colors.text))
                    .child(label(
                        validation_error.unwrap_or_else(|| {
                            "Deterministic baseline + intervention pair · seed 0".to_string()
                        }),
                        8.5,
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
                        match console.step {
                            WorkspaceStep::Prompt => console.step = WorkspaceStep::Intervention,
                            WorkspaceStep::Intervention => console.step = WorkspaceStep::Review,
                            WorkspaceStep::Review => console.run(),
                        }
                        cx.notify();
                    })
                }),
            )))
            .child(label("Ctrl+Enter", 8.5, colors.text_faint))
    }
}

impl Render for Console {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = self.colors();
        let header = self.header(&colors, cx);
        let body = div()
            .flex()
            .flex_row()
            .w_full()
            .flex_1()
            .min_h(px(0.0))
            .child(
                div()
                    .w(px(196.0))
                    .flex_none()
                    .h_full()
                    .bg(colors.sidebar)
                    .border_r_1()
                    .border_color(colors.border)
                    .child(self.sidebar(&colors, cx)),
            )
            .child(self.main_panel(&colors, cx));
        let statusbar = self.statusbar(&colors, cx);

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(colors.canvas)
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(|console, event: &KeyDownEvent, _window, cx| {
                console.picker_key(event, cx);
            }))
            .child(header.flex_none())
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
    let (worker_tx, reply_rx) = spawn_worker(k_strategy, k_allow_fallback);
    eprintln!(
        "EMBER experiment console v{} (native, gpui)",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  model stays resident; press Ctrl-C to quit.");

    Application::new()
        .with_assets(icons::Assets)
        .run(move |cx: &mut App| {
            input::bind_keys(cx);
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
                    cx.new(|cx| {
                        let system_dark = theme::system_is_dark(window.appearance());
                        let mut console = Console::new(worker_tx, reply_rx, system_dark, cx);
                        cx.observe_window_appearance(window, |console, window, cx| {
                            console.system_appearance_changed(
                                theme::system_is_dark(window.appearance()),
                                cx,
                            );
                        })
                        .detach();
                        console.spawn_poll(cx);
                        console
                    })
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
    fn inspector_excerpt_preserves_arabic_characters() {
        assert_eq!(truncate_chars("المدينة المنورة", 7), "المدينة…");
        assert_eq!(truncate_chars("اختبار", 20), "اختبار");
    }
}
