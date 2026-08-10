//! Ember v0.6 native experiment console (`ember gui`).
//!
//! A native, single-window console over the exact same v0.5 pipeline as the
//! web console (`ember web-gui`). The UI is built with iced (tiny-skia
//! software rendering — no GPU or system-webview dependency) and every
//! experiment is executed in a worker thread through the shared
//! `GuiSession` core, which in turn calls `prepare_run` / `execute_prepared`
//! — the same code path as `ember experiment run`. No inference logic lives
//! in the UI.
//!
//! Arabic input/output is shaped and laid out RTL by cosmic-text (iced's
//! text engine). The Noto Sans / Noto Sans Mono / Noto Naskh Arabic fonts
//! are embedded so rendering is identical on any machine, fully offline.

use crate::gui::{
    discover_models, parse_run_request, RestoreBundle, RunBundle, RunConfig, RunOutput, RunRequest,
    SessionInfo,
};
use anyhow::Context;
use clap::Args as ClapArgs;
use ember::quant_k::KStrategy;
use iced::gradient::Linear;
use iced::widget::{
    button, column, combo_box, container, overlay::menu, row, rule, scrollable, space, text,
    text_editor, text_input, Column,
};
use iced::{
    border, Alignment, Background, Color, Element, Font, Gradient, Length, Shadow, Subscription,
    Task, Theme, Vector,
};
use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
    Prepared(Result<SessionInfo, String>),
    RunDone(Result<RunBundle, String>),
    RestoreDone(Result<RestoreBundle, String>),
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
                    let _ = reply_tx.send(WorkerReply::Prepared(result));
                }
                WorkerMsg::Run(cfg) => {
                    let _ = reply_tx.send(WorkerReply::RunDone(
                        session.run_baseline_intervention(&cfg),
                    ));
                }
                WorkerMsg::Restore(cfg) => {
                    let _ = reply_tx.send(WorkerReply::RestoreDone(session.run_restore_leg(&cfg)));
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

struct Console {
    // worker
    worker_tx: mpsc::Sender<WorkerMsg>,
    reply_rx: Arc<Mutex<mpsc::Receiver<WorkerReply>>>,
    // model
    model_combo: combo_box::State<String>,
    model_path: String,
    // form
    site_combo: combo_box::State<String>,
    site: String,
    layer: String,
    op_combo: combo_box::State<String>,
    op: String,
    value: String,
    source_combo: combo_box::State<String>,
    source: String,
    source_layer: String,
    token_combo: combo_box::State<String>,
    token: String,
    span: String,
    max_tokens: String,
    execution_combo: combo_box::State<String>,
    execution: String,
    prompt: text_editor::Content<iced::Renderer>,
    // theme
    dark: bool,
    // session + results
    session: Option<SessionInfo>,
    status: Status,
    error: Option<String>,
    warning: Option<String>,
    baseline: Option<RunOutput>,
    intervention: Option<RunOutput>,
    verification: Option<VerificationView>,
    restore: Option<RestoreView>,
    last_config: Option<String>,
    last_metrics: Option<(String, f64, Option<f64>)>,
}

impl Console {
    fn new(
        worker_tx: mpsc::Sender<WorkerMsg>,
        reply_rx: Arc<Mutex<mpsc::Receiver<WorkerReply>>>,
    ) -> Self {
        let models = discover_models();
        let model_path = models.first().cloned().unwrap_or_default();
        let mut model_combo = combo_box::State::new(models.clone());
        for model in &models {
            model_combo.push(model.clone());
        }
        let mut site_combo = combo_box::State::new(STAGES.iter().map(|s| s.to_string()).collect());
        for stage in STAGES {
            site_combo.push(stage.to_string());
        }
        let mut op_combo =
            combo_box::State::new(OPERATIONS.iter().map(|s| s.to_string()).collect());
        for op in OPERATIONS {
            op_combo.push(op.to_string());
        }
        let mut source_combo =
            combo_box::State::new(vec!["capture".to_string(), "zero".to_string()]);
        source_combo.push("capture".to_string());
        source_combo.push("zero".to_string());
        let mut token_combo =
            combo_box::State::new(vec!["prompt-final".to_string(), "matched-span".to_string()]);
        token_combo.push("prompt-final".to_string());
        token_combo.push("matched-span".to_string());
        let mut execution_combo =
            combo_box::State::new(EXECUTIONS.iter().map(|s| s.to_string()).collect());
        for exec in EXECUTIONS {
            execution_combo.push(exec.to_string());
        }
        Console {
            worker_tx,
            reply_rx,
            model_combo,
            model_path,
            site_combo,
            site: "after-mlp".to_string(),
            layer: "0".to_string(),
            op_combo,
            op: "scale".to_string(),
            value: "0.5".to_string(),
            source_combo,
            source: "capture".to_string(),
            source_layer: "0".to_string(),
            token_combo,
            token: "prompt-final".to_string(),
            span: String::new(),
            max_tokens: "48".to_string(),
            execution_combo,
            execution: "reference".to_string(),
            prompt: text_editor::Content::with_text(
                "\u{627}\u{643}\u{62A}\u{628} \u{62C}\u{645}\u{644}\u{629} \
                 \u{642}\u{635}\u{64A}\u{631}\u{629} \u{639}\u{646} \u{627}\u{644}\u{645}\u{62F}\u{64A}\u{646}\u{629} \
                 \u{627}\u{644}\u{645}\u{646}\u{648}\u{631}\u{629}",
            ),
            dark: true,
            session: None,
            status: Status::Idle,
            error: None,
            warning: None,
            baseline: None,
            intervention: None,
            verification: None,
            restore: None,
            last_config: None,
            last_metrics: None,
        }
    }

    fn busy(&self) -> bool {
        self.status != Status::Idle
    }

    /// The active palette (dark by default; toggled from the header).
    fn colors(&self) -> &'static Colors {
        if self.dark {
            &DARK
        } else {
            &LIGHT
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

    /// Build the v0.5 request from the current form fields; the shared
    /// `parse_run_request` gate validates it exactly like the web console.
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
            // The capture fires before the intervention in the same pass, so
            // clamp the source layer to at most the layer above the target.
            // Mirrors the web console's syncSourceLayer; keeps the request
            // valid even when the two fields are edited out of order.
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
            prompt: self.prompt.text(),
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
            span_text: if self.token == "matched-span" {
                Some(self.span.clone())
            } else {
                None
            },
        })
    }

    fn send_run(&mut self, cfg: RunConfig) {
        self.status = Status::Running;
        self.error = None;
        self.warning = None;
        let _ = self.worker_tx.send(WorkerMsg::Run(cfg));
    }

    fn send_restore(&mut self, cfg: RunConfig) {
        self.status = Status::Restoring;
        self.error = None;
        let _ = self.worker_tx.send(WorkerMsg::Restore(cfg));
    }

    fn drain_replies(&mut self) {
        let replies: Vec<WorkerReply> = {
            let rx = self.reply_rx.lock().expect("reply receiver lock");
            std::iter::from_fn(|| rx.try_recv().ok()).collect()
        };
        for reply in replies {
            match reply {
                WorkerReply::Prepared(result) => match result {
                    Ok(info) => {
                        self.session = Some(info);
                        let n = self.session.as_ref().map(|s| s.n_layers).unwrap_or(1);
                        self.layer = n.saturating_sub(2).to_string();
                        self.source_layer = n.saturating_sub(3).to_string();
                        self.status = Status::Idle;
                    }
                    Err(error) => {
                        self.error = Some(error);
                        self.status = Status::Idle;
                    }
                },
                WorkerReply::RunDone(result) => match result {
                    Ok(bundle) => {
                        self.baseline = Some(bundle.baseline.clone());
                        self.intervention = Some(bundle.intervention.clone());
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
                        self.status = Status::Idle;
                    }
                    Err(error) => {
                        self.error = Some(error);
                        self.status = Status::Idle;
                    }
                },
                WorkerReply::RestoreDone(result) => match result {
                    Ok(bundle) => {
                        self.restore = Some(RestoreView {
                            matches: bundle.matches_baseline,
                            comparable: bundle.baseline_comparable,
                        });
                        self.verification =
                            Some(VerificationView::from_report(&bundle.verification));
                        if let Some((_, _, _)) = &self.last_metrics {
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
    }
}

// ---------------------------------------------------------------------------
// messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Message {
    ModelPathChanged(String),
    ModelSelected(String),
    Load,
    PromptEdited(text_editor::Action),
    SiteSelected(String),
    LayerChanged(String),
    OpSelected(String),
    ValueChanged(String),
    SourceSelected(String),
    SourceLayerChanged(String),
    TokenSelected(String),
    SpanChanged(String),
    MaxTokensChanged(String),
    ExecutionSelected(String),
    Run,
    Restore,
    Poll,
    ToggleTheme,
}

fn update(state: &mut Console, message: Message) -> Task<Message> {
    match message {
        Message::ModelPathChanged(value) => state.model_path = value,
        Message::ModelSelected(model) => state.model_path = model,
        Message::Load => {
            if state.busy() {
                return Task::none();
            }
            let path = state.model_path.trim().to_string();
            if path.is_empty() {
                state.error = Some("model path must not be empty".to_string());
                return Task::none();
            }
            state.status = Status::Preparing;
            state.error = None;
            let _ = state.worker_tx.send(WorkerMsg::Prepare(path));
        }
        Message::PromptEdited(action) => state.prompt.perform(action),
        Message::SiteSelected(site) => {
            state.site = site;
            if !per_layer(&state.site) {
                state.layer = "0".to_string();
                state.source_layer = "0".to_string();
            }
        }
        Message::LayerChanged(layer) => {
            state.layer = layer;
            // keep the source layer at or above the target layer (the
            // capture must fire before the intervention in the same pass)
            if let (Ok(target), Ok(source)) = (
                state.layer.parse::<i64>(),
                state.source_layer.parse::<i64>(),
            ) {
                if source > target {
                    state.source_layer = (target - 1).max(0).to_string();
                }
            }
        }
        Message::OpSelected(op) => state.op = op,
        Message::ValueChanged(value) => state.value = value,
        Message::SourceSelected(source) => state.source = source,
        Message::SourceLayerChanged(layer) => state.source_layer = layer,
        Message::TokenSelected(token) => state.token = token,
        Message::SpanChanged(span) => state.span = span,
        Message::MaxTokensChanged(tokens) => state.max_tokens = tokens,
        Message::ExecutionSelected(execution) => state.execution = execution,
        Message::Run => {
            if state.busy() {
                return Task::none();
            }
            match state.build_run_request() {
                Ok(req) => match parse_run_request(&req) {
                    Ok(cfg) => state.send_run(cfg),
                    Err(error) => state.error = Some(error),
                },
                Err(error) => state.error = Some(error),
            }
        }
        Message::Restore => {
            if state.busy() {
                return Task::none();
            }
            if state.last_config.is_none() {
                state.error = Some(
                    "run an experiment first; restore verifies against its baseline".to_string(),
                );
                return Task::none();
            }
            match state.build_run_request() {
                Ok(mut req) => {
                    req.operation = "restore-original".to_string();
                    req.factor = None;
                    req.alpha = None;
                    req.source = "capture".to_string();
                    req.source_layer = None;
                    match parse_run_request(&req) {
                        Ok(cfg) => state.send_restore(cfg),
                        Err(error) => state.error = Some(error),
                    }
                }
                Err(error) => state.error = Some(error),
            }
        }
        Message::Poll => state.drain_replies(),
        Message::ToggleTheme => state.dark = !state.dark,
    }
    Task::none()
}

fn subscription(state: &Console) -> Subscription<Message> {
    if state.busy() {
        iced::time::every(Duration::from_millis(80)).map(|_| Message::Poll)
    } else {
        Subscription::none()
    }
}

// ---------------------------------------------------------------------------
// view
// ---------------------------------------------------------------------------

// The console palette — "ember" styling: a warm ember-orange accent over deep
// cool charcoal surfaces, with soft shadows and rounded corners throughout.
// The header and status bar stay dark in both themes (brand constants), while
// everything else follows the in-window theme toggle.

const HEADER_TOP: Color = Color::from_rgb8(0x0d, 0x0f, 0x14);
const HEADER_BOTTOM: Color = Color::from_rgb8(0x16, 0x1b, 0x23);
const HEADER_TEXT: Color = Color::from_rgb8(0xe8, 0xea, 0xee);
const HEADER_DIM: Color = Color::from_rgb8(0x93, 0x9c, 0xad);
const HEADER_CHIP_TEXT: Color = Color::from_rgb8(0xc3, 0xca, 0xd6);

struct Colors {
    bg: Color,
    panel: Color,
    panel_alt: Color,
    text: Color,
    dim: Color,
    faint: Color,
    border: Color,
    accent: Color,
    accent_hi: Color,
    accent_lo: Color,
    accent_soft: Color,
    ok: Color,
    err: Color,
    warn: Color,
    err_box_bg: Color,
    err_box_border: Color,
    warn_box_bg: Color,
    warn_box_border: Color,
    shadow: Color,
}

/// Light console theme: dark text on white surfaces, ember accent.
const LIGHT: Colors = Colors {
    bg: Color::from_rgb8(0xf1, 0xf2, 0xf6),
    panel: Color::from_rgb8(0xff, 0xff, 0xff),
    panel_alt: Color::from_rgb8(0xf5, 0xf6, 0xfa),
    text: Color::from_rgb8(0x1a, 0x1d, 0x26),
    dim: Color::from_rgb8(0x4d, 0x55, 0x66),
    faint: Color::from_rgb8(0x8a, 0x92, 0xa4),
    border: Color::from_rgb8(0xda, 0xde, 0xe7),
    accent: Color::from_rgb8(0xc8, 0x3e, 0x0a),
    accent_hi: Color::from_rgb8(0xdd, 0x4f, 0x16),
    accent_lo: Color::from_rgb8(0xa8, 0x32, 0x06),
    accent_soft: Color::from_rgba8(0xc8, 0x3e, 0x0a, 0.10),
    ok: Color::from_rgb8(0x14, 0x7d, 0x3c),
    err: Color::from_rgb8(0xb3, 0x26, 0x1e),
    warn: Color::from_rgb8(0x9a, 0x67, 0x00),
    err_box_bg: Color::from_rgb8(0xfd, 0xed, 0xeb),
    err_box_border: Color::from_rgb8(0xdf, 0xa5, 0xa0),
    warn_box_bg: Color::from_rgb8(0xfd, 0xf4, 0xda),
    warn_box_border: Color::from_rgb8(0xd8, 0xb3, 0x59),
    shadow: Color::from_rgba8(0x17, 0x1c, 0x28, 0.10),
};

/// Dark console theme: light text on deep charcoal surfaces, ember accent.
const DARK: Colors = Colors {
    bg: Color::from_rgb8(0x10, 0x13, 0x18),
    panel: Color::from_rgb8(0x1a, 0x1e, 0x26),
    panel_alt: Color::from_rgb8(0x21, 0x26, 0x30),
    text: Color::from_rgb8(0xec, 0xee, 0xf2),
    dim: Color::from_rgb8(0xa7, 0xae, 0xbd),
    faint: Color::from_rgb8(0x67, 0x70, 0x83),
    border: Color::from_rgb8(0x2b, 0x32, 0x3e),
    accent: Color::from_rgb8(0xff, 0x77, 0x35),
    accent_hi: Color::from_rgb8(0xff, 0x90, 0x59),
    accent_lo: Color::from_rgb8(0xdf, 0x5c, 0x1c),
    accent_soft: Color::from_rgba8(0xff, 0x77, 0x35, 0.15),
    ok: Color::from_rgb8(0x4a, 0xd1, 0x8c),
    err: Color::from_rgb8(0xf1, 0x6b, 0x62),
    warn: Color::from_rgb8(0xe9, 0xb4, 0x4e),
    err_box_bg: Color::from_rgb8(0x35, 0x20, 0x1e),
    err_box_border: Color::from_rgb8(0x7c, 0x44, 0x3e),
    warn_box_bg: Color::from_rgb8(0x35, 0x2c, 0x1a),
    warn_box_border: Color::from_rgb8(0x7c, 0x60, 0x2c),
    shadow: Color::from_rgba8(0x00, 0x00, 0x00, 0.35),
};

fn mono(text: impl Into<String>) -> iced::widget::Text<'static> {
    iced::widget::text(text.into()).font(Font::with_name(FONT_MONO_NAME))
}

/// The ember brand gradient: a bright accent melting into its darker sibling.
fn ember_gradient(hi: Color, lo: Color) -> Gradient {
    // angle PI runs top -> bottom (see Radians::to_distance)
    Gradient::Linear(Linear::new(PI).add_stop(0.0, hi).add_stop(1.0, lo))
}

/// A soft drop shadow using the theme's shadow color.
fn panel_shadow(colors: &Colors, blur: f32, offset: f32) -> Shadow {
    Shadow {
        color: colors.shadow,
        offset: Vector::new(0.0, offset),
        blur_radius: blur,
    }
}

fn rule_h(colors: &'static Colors) -> Element<'static, Message> {
    rule::horizontal(1)
        .style(move |_| rule::Style {
            color: colors.border,
            radius: 0.0.into(),
            fill_mode: rule::FillMode::Full,
            snap: false,
        })
        .into()
}

fn field<'a>(
    colors: &Colors,
    title: &'static str,
    control: impl Into<Element<'a, Message>>,
) -> Element<'a, Message> {
    column![text(title).size(9).color(colors.faint), control.into()]
        .spacing(3)
        .width(Length::Fill)
        .into()
}

/// A card: rounded panel with a hairline border and a soft shadow.
fn panel<'a>(
    colors: &'static Colors,
    content: impl Into<Element<'a, Message>>,
) -> container::Container<'a, Message> {
    container(content.into())
        .width(Length::Fill)
        .padding(12)
        .style(move |_: &Theme| container::Style {
            background: Some(Background::Color(colors.panel)),
            border: border::rounded(10).width(1).color(colors.border),
            shadow: panel_shadow(colors, 8.0, 2.0),
            ..Default::default()
        })
}

/// A small uppercase group label with an ember tick.
fn section_label(colors: &'static Colors, label: &'static str) -> Element<'static, Message> {
    row![
        container(space().width(3).height(11)).style(move |_| container::Style {
            background: Some(Background::Color(colors.accent)),
            border: border::rounded(2),
            ..Default::default()
        }),
        space().width(6),
        text(label).size(9).color(colors.dim),
    ]
    .align_y(Alignment::Center)
    .into()
}

fn input_style(
    colors: &'static Colors,
) -> impl Fn(&Theme, text_input::Status) -> text_input::Style + 'static {
    move |_theme, status| {
        let (background, border_color, width) = match status {
            text_input::Status::Focused { .. } => (colors.panel, colors.accent, 1.5),
            text_input::Status::Hovered => (colors.panel_alt, colors.accent_hi, 1.0),
            _ => (colors.panel_alt, colors.border, 1.0),
        };
        text_input::Style {
            background: Background::Color(background),
            border: border::rounded(6).width(width).color(border_color),
            icon: colors.faint,
            placeholder: colors.faint,
            value: colors.text,
            selection: colors.accent,
        }
    }
}

fn editor_style(
    colors: &'static Colors,
) -> impl Fn(&Theme, text_editor::Status) -> text_editor::Style + 'static {
    move |_theme, status| {
        let (background, border_color, width) = match status {
            text_editor::Status::Focused { .. } => (colors.panel, colors.accent, 1.5),
            text_editor::Status::Hovered => (colors.panel_alt, colors.accent_hi, 1.0),
            _ => (colors.panel_alt, colors.border, 1.0),
        };
        text_editor::Style {
            background: Background::Color(background),
            border: border::rounded(6).width(width).color(border_color),
            placeholder: colors.faint,
            value: colors.text,
            selection: colors.accent,
        }
    }
}

fn menu_style(colors: &'static Colors) -> impl Fn(&Theme) -> menu::Style + 'static {
    move |_theme| menu::Style {
        background: Background::Color(colors.panel),
        border: border::rounded(6).width(1).color(colors.border),
        text_color: colors.text,
        selected_text_color: colors.accent,
        selected_background: Background::Color(colors.accent_soft),
        shadow: panel_shadow(colors, 10.0, 3.0),
    }
}

/// A combo box with the console's input + menu styling.
fn combo<'a>(
    colors: &'static Colors,
    state: &'a combo_box::State<String>,
    placeholder: &'static str,
    selected: Option<&'a String>,
    on_selected: fn(String) -> Message,
    on_input: Option<fn(String) -> Message>,
) -> Element<'a, Message> {
    let base = combo_box(state, placeholder, selected, on_selected)
        .input_style(input_style(colors))
        .menu_style(menu_style(colors));
    match on_input {
        Some(f) => base.on_input(f).into(),
        None => base.into(),
    }
}

fn scroll_style(
    colors: &'static Colors,
) -> impl Fn(&Theme, scrollable::Status) -> scrollable::Style + 'static {
    move |_theme, _status| {
        let rail = scrollable::Rail {
            background: None,
            border: border::rounded(4),
            scroller: scrollable::Scroller {
                background: Background::Color(colors.faint),
                border: border::rounded(4),
            },
        };
        scrollable::Style {
            container: container::Style::default(),
            vertical_rail: rail,
            horizontal_rail: rail,
            gap: None,
            auto_scroll: scrollable::AutoScroll {
                background: Background::Color(colors.accent_soft),
                border: border::rounded(6).width(1).color(colors.accent),
                shadow: panel_shadow(colors, 6.0, 2.0),
                icon: colors.accent,
            },
        }
    }
}

/// A rounded status pill (tinted background + matching text).
fn chip<'a>(label: &'a str, color: Color) -> Element<'a, Message> {
    container(text(label).size(9).color(color))
        .padding([2, 8])
        .style(move |_| container::Style {
            background: Some(Background::Color(color.scale_alpha(0.13))),
            border: border::rounded(999).width(1).color(color.scale_alpha(0.40)),
            ..Default::default()
        })
        .into()
}

/// A small colored status dot.
fn status_dot(color: Color) -> Element<'static, Message> {
    container(space().width(8).height(8))
        .style(move |_| container::Style {
            background: Some(Background::Color(color)),
            border: border::rounded(4).width(1).color(color.scale_alpha(0.45)),
            ..Default::default()
        })
        .into()
}

/// Full-width ember gradient primary button.
fn btn_primary<'a>(
    colors: &'static Colors,
    label: &'a str,
    message: Option<Message>,
) -> iced::widget::Button<'a, Message> {
    button(text(label).size(12))
        .on_press_maybe(message)
        .width(Length::Fill)
        .padding([10, 16])
        .style(move |_theme, status| {
            let (hi, lo) = match status {
                button::Status::Hovered => (colors.accent_hi, colors.accent),
                button::Status::Pressed => (colors.accent_lo, colors.accent_lo),
                _ => (colors.accent, colors.accent_lo),
            };
            button::Style {
                background: Some(Background::Gradient(ember_gradient(hi, lo))),
                text_color: Color::WHITE,
                border: border::rounded(7),
                shadow: panel_shadow(colors, 6.0, 2.0),
                ..Default::default()
            }
        })
}

/// Full-width secondary (outlined) button.
fn btn_secondary<'a>(
    colors: &'static Colors,
    label: &'a str,
    message: Option<Message>,
) -> iced::widget::Button<'a, Message> {
    button(text(label).size(12))
        .on_press_maybe(message)
        .width(Length::Fill)
        .padding([10, 16])
        .style(move |_theme, status| {
            let (bg, border_color, fg) = match status {
                button::Status::Hovered => (colors.panel_alt, colors.accent, colors.text),
                button::Status::Pressed => (colors.panel, colors.accent_lo, colors.dim),
                _ => (colors.panel, colors.border, colors.text),
            };
            button::Style {
                background: Some(Background::Color(bg)),
                text_color: fg,
                border: border::rounded(7).width(1).color(border_color),
                shadow: panel_shadow(colors, 4.0, 1.0),
                ..Default::default()
            }
        })
}

fn header<'a>(state: &'a Console) -> Element<'a, Message> {
    let colors = state.colors();
    let session_chip = match &state.session {
        Some(info) => format!(
            "model: {} · {} · {}L",
            info.model_name, info.architecture, info.n_layers
        ),
        None => format!("model: {} — not loaded", state.model_name()),
    };
    column![
        container(
            row![
                container(text("E").size(14).color(Color::WHITE))
                    .width(26)
                    .height(26)
                    .align_x(Alignment::Center)
                    .align_y(Alignment::Center)
                    .style(move |_| container::Style {
                        background: Some(Background::Gradient(ember_gradient(
                            colors.accent_hi,
                            colors.accent_lo
                        ))),
                        border: border::rounded(7),
                        shadow: panel_shadow(colors, 4.0, 1.0),
                        ..Default::default()
                    }),
                column![
                    text("EMBER").size(15).color(HEADER_TEXT),
                    text("EXPERIMENT CONSOLE · v0.6").size(9).color(HEADER_DIM),
                ]
                .spacing(0),
                space().width(Length::Fill).height(Length::Shrink),
                container(mono(session_chip).size(11).color(HEADER_CHIP_TEXT))
                    .padding([4, 10])
                    .style(move |_| container::Style {
                        background: Some(Background::Color(Color::from_rgb8(0x18, 0x1d, 0x26))),
                        border: border::rounded(999)
                            .width(1)
                            .color(Color::from_rgb8(0x2c, 0x33, 0x40)),
                        ..Default::default()
                    }),
                button(text(if state.dark { "☀" } else { "☾" }).size(12))
                    .on_press(Message::ToggleTheme)
                    .padding([4, 10])
                    .style(move |_theme, status| {
                        let (bg, fg, border_color) = match status {
                            button::Status::Hovered => (
                                Color::from_rgb8(0x2a, 0x31, 0x3d),
                                Color::WHITE,
                                Color::from_rgb8(0x3a, 0x43, 0x52),
                            ),
                            _ => (
                                Color::from_rgb8(0x1c, 0x21, 0x2b),
                                Color::from_rgb8(0xc3, 0xca, 0xd6),
                                Color::from_rgb8(0x2c, 0x33, 0x40),
                            ),
                        };
                        button::Style {
                            background: Some(Background::Color(bg)),
                            text_color: fg,
                            border: border::rounded(6).width(1).color(border_color),
                            ..Default::default()
                        }
                    }),
            ]
            .align_y(Alignment::Center)
            .padding([10, 14])
            .spacing(12),
        )
        .width(Length::Fill)
        .style(|_| container::Style {
            background: Some(Background::Gradient(ember_gradient(
                HEADER_TOP,
                HEADER_BOTTOM
            ))),
            ..Default::default()
        }),
        container(space().width(Length::Fill).height(3))
            .width(Length::Fill)
            .style(move |_| container::Style {
                background: Some(Background::Gradient(Gradient::Linear(
                    Linear::new(FRAC_PI_2)
                        .add_stop(0.0, colors.accent_hi)
                        .add_stop(1.0, colors.accent),
                ))),
                ..Default::default()
            }),
    ]
    .into()
}

fn sidebar<'a>(state: &'a Console) -> Element<'a, Message> {
    let colors = state.colors();
    let n_layers = state.session.as_ref().map(|s| s.n_layers).unwrap_or(0);
    let layer_hint = if per_layer(&state.site) && n_layers > 0 {
        format!("0 \u{2013} {} \u{00b7} {n_layers} layers", n_layers - 1)
    } else {
        "no per-layer site".to_string()
    };

    let needs_source = matches!(state.op.as_str(), "replace" | "interpolate" | "add-delta");
    let needs_value = matches!(state.op.as_str(), "scale" | "interpolate");
    let value_label = if state.op == "interpolate" {
        "ALPHA (0\u{2013}1)"
    } else {
        "VALUE"
    };
    let source_is_capture = state.source == "capture";

    let mut children: Vec<Element<'a, Message>> = Vec::new();

    // ---- MODEL ----
    children.push(section_label(colors, "MODEL"));
    children.push(field(
        colors,
        "MODEL",
        combo(
            colors,
            &state.model_combo,
            "select a model\u{2026}",
            if state.model_path.is_empty() {
                None
            } else {
                Some(&state.model_path)
            },
            Message::ModelSelected,
            Some(Message::ModelPathChanged),
        ),
    ));
    children.push(
        text_input("path to model.gguf", &state.model_path)
            .on_input(Message::ModelPathChanged)
            .font(Font::with_name(FONT_MONO_NAME))
            .size(12)
            .padding(6)
            .style(input_style(colors))
            .into(),
    );
    children.push(
        btn_secondary(
            colors,
            if state.status == Status::Preparing {
                "LOADING\u{2026}"
            } else {
                "LOAD"
            },
            (!state.busy()).then_some(Message::Load),
        )
        .into(),
    );
    match &state.session {
        Some(info) => children.push(
            mono(format!(
                "loaded {} \u{00b7} {} layers \u{00b7} {}d \u{00b7} {}",
                info.architecture,
                info.n_layers,
                info.embed_dim,
                fmt_load_ms(info.load_ms)
            ))
            .size(10)
            .color(colors.ok)
            .into(),
        ),
        None => children.push(
            text("no model loaded \u{2014} pick a .gguf, then LOAD")
                .size(10)
                .color(colors.faint)
                .into(),
        ),
    }
    children.push(rule_h(colors));

    // ---- HOOK & INTERVENTION ----
    children.push(section_label(colors, "HOOK & INTERVENTION"));
    children.push(field(
        colors,
        "HOOK STAGE",
        combo(
            colors,
            &state.site_combo,
            "hook\u{2026}",
            Some(&state.site),
            Message::SiteSelected,
            None,
        ),
    ));
    children.push(field(
        colors,
        "LAYER",
        text_input("0", &state.layer)
            .on_input(Message::LayerChanged)
            .font(Font::with_name(FONT_MONO_NAME))
            .width(Length::Fill)
            .padding(6)
            .style(input_style(colors)),
    ));
    children.push(text(layer_hint).size(9).color(colors.faint).into());
    children.push(field(
        colors,
        "INTERVENTION",
        combo(
            colors,
            &state.op_combo,
            "intervention\u{2026}",
            Some(&state.op),
            Message::OpSelected,
            None,
        ),
    ));
    if needs_value {
        children.push(
            column![
                field(
                    colors,
                    value_label,
                    text_input("0.5", &state.value)
                        .on_input(Message::ValueChanged)
                        .padding(6)
                        .style(input_style(colors)),
                ),
                text(if state.op == "interpolate" {
                    "blend toward the source"
                } else {
                    "multiplicative factor"
                })
                .size(9)
                .color(colors.faint),
            ]
            .spacing(2)
            .into(),
        );
    }
    if needs_source {
        children.push(field(
            colors,
            "SOURCE",
            combo(
                colors,
                &state.source_combo,
                "source\u{2026}",
                Some(&state.source),
                Message::SourceSelected,
                None,
            ),
        ));
        if source_is_capture {
            children.push(
                column![
                    field(
                        colors,
                        "SOURCE LAYER",
                        text_input("0", &state.source_layer)
                            .on_input(Message::SourceLayerChanged)
                            .font(Font::with_name(FONT_MONO_NAME))
                            .padding(6)
                            .style(input_style(colors)),
                    ),
                    text("capture fires before the intervention (same pass)")
                        .size(9)
                        .color(colors.faint),
                ]
                .spacing(2)
                .into(),
            );
        }
    }
    children.push(rule_h(colors));

    // ---- TARGET ----
    children.push(section_label(colors, "TARGET"));
    children.push(field(
        colors,
        "TARGET TOKENS",
        combo(
            colors,
            &state.token_combo,
            "tokens\u{2026}",
            Some(&state.token),
            Message::TokenSelected,
            None,
        ),
    ));
    if state.token == "matched-span" {
        children.push(
            text_input(
                "\u{643}\u{644}\u{645}\u{629} \u{641}\u{64A} \u{627}\u{644}\u{646}\u{635}",
                &state.span,
            )
            .on_input(Message::SpanChanged)
            .font(Font::with_name(FONT_ARABIC_NAME))
            .padding(6)
            .style(input_style(colors))
            .into(),
        );
    }
    children.push(rule_h(colors));

    // ---- ACTIONS ----
    children.push(section_label(colors, "ACTIONS"));
    children.push(
        btn_primary(
            colors,
            match state.status {
                Status::Running => "RUNNING\u{2026}",
                Status::Preparing => "LOADING MODEL\u{2026}",
                Status::Restoring => "RESTORING\u{2026}",
                Status::Idle => "RUN EXPERIMENT",
            },
            (!state.busy()).then_some(Message::Run),
        )
        .into(),
    );
    children.push(
        btn_secondary(
            colors,
            "VERIFY RESTORE",
            (!state.busy()).then_some(Message::Restore),
        )
        .into(),
    );
    children.push(
        text("change layer or intervention \u{2192} RUN \u{2192} compare \u{2192} VERIFY RESTORE")
            .size(9)
            .color(colors.faint)
            .into(),
    );

    scrollable(Column::with_children(children).spacing(10).padding(14))
        .style(scroll_style(colors))
        .into()
}

fn main_panel<'a>(state: &'a Console) -> Element<'a, Message> {
    let colors = state.colors();
    let prompt_editor = text_editor(&state.prompt)
        .on_action(Message::PromptEdited)
        .height(84)
        .padding(10)
        .font(Font::with_name(FONT_ARABIC_NAME))
        .size(14)
        .style(editor_style(colors));
    let prompt_meta = row![
        field(
            colors,
            "MAX TOKENS",
            text_input("48", &state.max_tokens)
                .on_input(Message::MaxTokensChanged)
                .width(76)
                .padding(6)
                .style(input_style(colors)),
        ),
        field(
            colors,
            "EXECUTION",
            combo(
                colors,
                &state.execution_combo,
                "execution\u{2026}",
                Some(&state.execution),
                Message::ExecutionSelected,
                None,
            ),
        ),
        space().width(Length::Fill).height(Length::Shrink),
        text("greedy \u{00b7} deterministic \u{00b7} temp 0.0")
            .size(9)
            .color(colors.faint),
    ]
    .align_y(Alignment::End)
    .spacing(10);

    let error = state
        .error
        .as_ref()
        .map(|error| {
            container(mono(error.clone()).size(11).color(colors.err))
                .width(Length::Fill)
                .padding([9, 12])
                .style(move |_| container::Style {
                    background: Some(Background::Color(colors.err_box_bg)),
                    border: border::rounded(8).width(1).color(colors.err_box_border),
                    ..Default::default()
                })
        })
        .unwrap_or_else(|| container(Column::new()).height(0));

    let warning = state
        .warning
        .as_ref()
        .map(|warning| {
            container(mono(warning.clone()).size(10).color(colors.warn))
                .width(Length::Fill)
                .padding([9, 12])
                .style(move |_| container::Style {
                    background: Some(Background::Color(colors.warn_box_bg)),
                    border: border::rounded(8).width(1).color(colors.warn_box_border),
                    ..Default::default()
                })
        })
        .unwrap_or_else(|| container(Column::new()).height(0));

    let outputs = row![
        output_panel(colors, "BASELINE", state.baseline.as_ref(), state.status),
        output_panel(
            colors,
            "INTERVENTION",
            state.intervention.as_ref(),
            state.status
        ),
    ]
    .spacing(12);

    let verification = verification_panel(state);

    scrollable(
        column![
            panel(
                colors,
                column![section_label(colors, "PROMPT"), prompt_editor, prompt_meta,].spacing(8),
            ),
            error,
            warning,
            outputs,
            verification,
        ]
        .spacing(12),
    )
    .width(Length::Fill)
    .style(scroll_style(colors))
    .into()
}

fn output_panel<'a>(
    colors: &'static Colors,
    title: &'static str,
    output: Option<&'a RunOutput>,
    status: Status,
) -> Element<'a, Message> {
    let (badge_text, badge_color) = match (output, status) {
        (Some(_), _) => ("OK", colors.ok),
        (None, Status::Running) => ("RUN", colors.warn),
        (None, Status::Preparing) => ("\u{2014}", colors.faint),
        (None, Status::Restoring) => ("\u{2014}", colors.faint),
        (None, Status::Idle) => ("\u{2014}", colors.faint),
    };
    let body: Element<'a, Message> = match output {
        Some(out) if !out.text.is_empty() => column![
            container(
                text(out.text.clone())
                    .font(Font::with_name(FONT_ARABIC_NAME))
                    .size(14)
                    .width(Length::Fill),
            )
            .width(Length::Fill)
            .padding(10)
            .style(move |_| container::Style {
                background: Some(Background::Color(colors.panel_alt)),
                border: border::rounded(8).width(1).color(colors.border),
                ..Default::default()
            }),
            row![
                mono(format!(
                    "{} tok \u{00b7} {} \u{00b7} {}",
                    out.generated_tokens,
                    fmt_ms(out.wall_ms),
                    fmt_tps(out.decode_tps)
                ))
                .size(10)
                .color(colors.dim),
                space().width(Length::Fill).height(Length::Shrink),
                mono(format!(
                    "prompt {} tok \u{00b7} bundle {}",
                    out.prompt_tokens,
                    short_id(&out.semantic_hash)
                ))
                .size(9)
                .color(colors.faint),
            ]
            .align_y(Alignment::Center),
            mono(out.bundle_dir.clone()).size(9).color(colors.faint),
        ]
        .spacing(6)
        .into(),
        Some(_out) => text("(empty output)").size(12).color(colors.faint).into(),
        None => text("no run yet \u{2014} outputs appear here")
            .size(12)
            .color(colors.faint)
            .into(),
    };
    panel(
        colors,
        column![
            row![
                text(title).size(11).color(colors.dim),
                space().width(Length::Fill).height(Length::Shrink),
                chip(badge_text, badge_color),
            ]
            .align_y(Alignment::Center),
            body,
        ]
        .spacing(8),
    )
    .into()
}

fn verification_panel<'a>(state: &'a Console) -> Element<'a, Message> {
    let colors = state.colors();
    let (badge, badge_color) = match (&state.verification, state.status) {
        (Some(verification), _) if verification.ok => ("VERIFIED", colors.ok),
        (Some(_), _) => ("VERIFICATION FAILED", colors.err),
        (None, Status::Running) => ("RUNNING", colors.warn),
        (None, Status::Restoring) => ("RESTORING", colors.warn),
        (None, _) => ("NOT RUN", colors.faint),
    };
    let mut lines: Vec<String> = Vec::new();
    if let Some(restore) = &state.restore {
        if !restore.comparable {
            lines.push("restore: baseline not comparable (configuration changed)".to_string());
        } else if restore.matches {
            lines.push("restore: BIT-EXACT".to_string());
        } else {
            lines.push("restore: DIFFERS from baseline".to_string());
        }
    } else if state.verification.is_some() {
        lines.push("restore: not run".to_string());
    }
    if let Some(verification) = &state.verification {
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
    let detail: Element<'a, Message> = if lines.is_empty() {
        text("bundle self-verification and the restore-original leg report here.")
            .size(10)
            .color(colors.faint)
            .into()
    } else {
        Column::with_children(
            lines
                .iter()
                .map(|line| mono(line.clone()).size(10).color(colors.dim).into())
                .collect::<Vec<Element<'static, Message>>>(),
        )
        .spacing(2)
        .into()
    };
    let metrics = match &state.last_metrics {
        Some((bundle, elapsed, tps)) => {
            format!(
                "bundle {} \u{00b7} {} \u{00b7} {}",
                short_id(bundle),
                fmt_ms(*elapsed),
                fmt_tps(*tps)
            )
        }
        None => String::new(),
    };
    panel(
        colors,
        column![
            row![
                chip(badge, badge_color),
                space().width(Length::Fill).height(Length::Shrink),
                mono(metrics).size(10).color(colors.faint),
            ]
            .align_y(Alignment::Center),
            detail,
        ]
        .spacing(6),
    )
    .into()
}

fn statusbar<'a>(state: &'a Console) -> Element<'a, Message> {
    let colors = state.colors();
    let (dot, status_text) = match state.status {
        Status::Idle => (Color::from_rgb8(0x6e, 0x77, 0x8a), "idle"),
        Status::Preparing => (colors.warn, "loading model\u{2026}"),
        Status::Running => (colors.accent, "running experiment\u{2026}"),
        Status::Restoring => (
            Color::from_rgb8(0x6b, 0xa7, 0xff),
            "verifying restore\u{2026}",
        ),
    };
    let layer_hook = if per_layer(&state.site) {
        format!("L{} \u{00b7} {}", state.layer, state.site)
    } else {
        state.site.clone()
    };
    let intervention = match state.op.as_str() {
        "scale" => format!("scale \u{00d7}{}", state.value),
        "interpolate" => format!("interpolate \u{03b1}={}", state.value),
        op => op.to_string(),
    };
    let metrics = state
        .last_metrics
        .as_ref()
        .map(|(bundle, elapsed, tps)| {
            format!(
                "bundle {} \u{00b7} {} \u{00b7} {}",
                short_id(bundle),
                fmt_ms(*elapsed),
                fmt_tps(*tps)
            )
        })
        .unwrap_or_default();
    let divider = container(space().width(1).height(12)).style(|_| container::Style {
        background: Some(Background::Color(Color::from_rgb8(0x2a, 0x30, 0x3c))),
        ..Default::default()
    });
    container(
        row![
            status_dot(dot),
            text(status_text).size(10).color(HEADER_CHIP_TEXT),
            divider,
            mono(format!("model {}", state.model_name()))
                .size(10)
                .color(HEADER_TEXT),
            mono(format!("layer/hook {layer_hook}"))
                .size(10)
                .color(HEADER_TEXT),
            mono(format!("intervention {intervention}"))
                .size(10)
                .color(HEADER_TEXT),
            space().width(Length::Fill).height(Length::Shrink),
            mono(metrics)
                .size(10)
                .color(Color::from_rgb8(0x57, 0xd6, 0x8d)),
        ]
        .align_y(Alignment::Center)
        .spacing(12)
        .padding([7, 14]),
    )
    .width(Length::Fill)
    .style(|_| container::Style {
        background: Some(Background::Color(HEADER_BOTTOM)),
        ..Default::default()
    })
    .into()
}

fn view(state: &Console) -> Element<'_, Message> {
    let colors = state.colors();
    let body = row![
        container(sidebar(state))
            .width(272)
            .height(Length::Fill)
            .style(move |_| container::Style {
                background: Some(Background::Color(colors.panel)),
                ..Default::default()
            }),
        rule::vertical(1).style(move |_| rule::Style {
            color: colors.border,
            radius: 0.0.into(),
            fill_mode: rule::FillMode::Full,
            snap: false,
        }),
        container(main_panel(state))
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(12),
    ]
    .width(Length::Fill)
    .height(Length::Fill);
    column![header(state), body, statusbar(state)].into()
}

// ---------------------------------------------------------------------------
// formatting helpers
// ---------------------------------------------------------------------------

fn short_id(hash: &str) -> String {
    if hash.len() > 6 {
        format!("{}…", &hash[..6])
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

fn theme(state: &Console) -> Theme {
    let colors = state.colors();
    Theme::custom(
        if state.dark {
            "ember-dark"
        } else {
            "ember-light"
        },
        iced::theme::Palette {
            background: colors.bg,
            text: colors.text,
            primary: colors.accent,
            success: colors.ok,
            danger: colors.err,
            warning: colors.warn,
        },
    )
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
        "EMBER experiment console v{} (native, iced)",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  model stays resident; press Ctrl-C to quit.");
    let boot = {
        let worker_tx = worker_tx.clone();
        let reply_rx = Arc::clone(&reply_rx);
        move || {
            (
                Console::new(worker_tx.clone(), Arc::clone(&reply_rx)),
                Task::none(),
            )
        }
    };
    iced::application(boot, update, view)
        .title(|_: &Console| "EMBER \u{2014} experiment console".to_string())
        .window_size(iced::Size::new(1240.0, 860.0))
        .theme(theme)
        .default_font(Font::with_name(FONT_SANS_NAME))
        .font(FONT_SANS)
        .font(FONT_MONO)
        .font(FONT_ARABIC)
        .subscription(subscription)
        .run()
        .context("the experiment console window failed")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn console() -> Console {
        let (tx, _rx) = mpsc::channel();
        let (_rtx, rrx) = mpsc::channel();
        Console::new(tx, Arc::new(Mutex::new(rrx)))
    }

    #[test]
    fn console_defaults_to_dark_and_toggle_switches_palette() {
        let state = console();
        assert!(state.dark, "the console starts in dark mode");
        assert_eq!(state.colors().bg, DARK.bg);
        let mut state = console();
        let _ = update(&mut state, Message::ToggleTheme);
        assert!(!state.dark);
        assert_eq!(state.colors().bg, LIGHT.bg);
        assert_eq!(state.colors().panel, LIGHT.panel);
        let _ = update(&mut state, Message::ToggleTheme);
        assert!(state.dark);
    }

    #[test]
    fn light_and_dark_palettes_differ_in_every_role() {
        // Every color role must differ between the two themes, otherwise a
        // theme switch would silently leave text or surfaces unreadable.
        let dark = DARK;
        let light = LIGHT;
        assert_ne!(dark.bg, light.bg);
        assert_ne!(dark.panel, light.panel);
        assert_ne!(dark.text, light.text);
        assert_ne!(dark.dim, light.dim);
        assert_ne!(dark.faint, light.faint);
        assert_ne!(dark.border, light.border);
        assert_ne!(dark.accent, light.accent);
        assert_ne!(dark.ok, light.ok);
        assert_ne!(dark.err, light.err);
        assert_ne!(dark.warn, light.warn);
        assert_ne!(dark.err_box_bg, light.err_box_bg);
        assert_ne!(dark.warn_box_bg, light.warn_box_bg);
        // contrast sanity: light theme has dark text on light surfaces,
        // dark theme has light text on dark surfaces.
        assert!(light.bg.r > light.text.r);
        assert!(dark.text.r > dark.bg.r);
    }

    #[test]
    fn default_form_builds_valid_run_request() {
        let state = console();
        let req = state.build_run_request().expect("default form is valid");
        let cfg = parse_run_request(&req).expect("default config validates");
        assert_eq!(cfg.site.stage_id(), "after-mlp");
        assert_eq!(cfg.layer, Some(0));
        assert!(matches!(cfg.operation, crate::gui::GuiOperation::Scale));
    }

    #[test]
    fn non_per_layer_site_drops_layer() {
        let mut state = console();
        state.site = "before-logits".to_string();
        let req = state.build_run_request().expect("valid");
        assert!(req.layer.is_none());
    }

    #[test]
    fn scale_factor_parses_and_validates() {
        let mut state = console();
        state.op = "scale".to_string();
        state.value = "abc".to_string();
        assert!(state.build_run_request().is_err());
        state.value = "0.25".to_string();
        let cfg = parse_run_request(&state.build_run_request().unwrap()).unwrap();
        assert_eq!(cfg.factor, 0.25);
    }

    #[test]
    fn source_layer_clamped_to_target() {
        let mut state = console();
        state.layer = "7".to_string();
        state.source_layer = "9".to_string();
        state.source = "capture".to_string();
        state.op = "replace".to_string();
        let cfg = parse_run_request(&state.build_run_request().unwrap()).unwrap();
        assert_eq!(cfg.source_layer, Some(6));
    }
}
