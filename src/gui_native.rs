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
    discover_models, parse_run_request, RestoreBundle, RunBundle, RunConfig, RunOutput, RunRequest,
    SessionInfo,
};
use clap::Args as ClapArgs;
use ember::quant_k::KStrategy;
use gpui::prelude::*;
use gpui::*;
use std::borrow::Cow;
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

/// Which text field is receiving keystrokes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputTarget {
    ModelPath,
    Layer,
    Value,
    SourceLayer,
    Span,
    MaxTokens,
    Prompt,
}

/// The console palette — "ember" styling: a warm ember-orange accent over deep
/// cool charcoal surfaces, with soft shadows and rounded corners throughout.
/// The header and status bar stay dark in both themes (brand constants), while
/// everything else follows the in-window theme toggle.
#[derive(Clone, Copy)]
struct Colors {
    bg: Rgba,
    panel: Rgba,
    panel_alt: Rgba,
    text: Rgba,
    dim: Rgba,
    faint: Rgba,
    border: Rgba,
    accent: Rgba,
    accent_hi: Rgba,
    accent_lo: Rgba,
    accent_soft: Rgba,
    ok: Rgba,
    err: Rgba,
    warn: Rgba,
    err_box_bg: Rgba,
    err_box_border: Rgba,
    warn_box_bg: Rgba,
    warn_box_border: Rgba,
}

/// Light console theme: dark text on white surfaces, ember accent.
fn light() -> Colors {
    Colors {
        bg: rgb(0xf4f5f7),
        panel: rgb(0xffffff),
        panel_alt: rgb(0xeef0f3),
        text: rgb(0x1b1e23),
        dim: rgb(0x505764),
        faint: rgb(0x838b97),
        border: rgb(0xe3e6eb),
        accent: rgb(0xdc5c20),
        accent_hi: rgb(0xef6a2b),
        accent_lo: rgb(0xc04a16),
        accent_soft: rgba(0xdc5c201a),
        ok: rgb(0x1f8a52),
        err: rgb(0xd0433a),
        warn: rgb(0xb07a16),
        err_box_bg: rgb(0xfceceb),
        err_box_border: rgb(0xe5b8b4),
        warn_box_bg: rgb(0xfaf3df),
        warn_box_border: rgb(0xe0c98f),
    }
}

/// Dark console theme: light text on deep charcoal surfaces, ember accent.
fn dark() -> Colors {
    Colors {
        bg: rgb(0x0a0c10),
        panel: rgb(0x161920),
        panel_alt: rgb(0x1d212a),
        text: rgb(0xe7e9ed),
        dim: rgb(0x9aa2ae),
        faint: rgb(0x6a7380),
        border: rgb(0x232833),
        accent: rgb(0xf06b2f),
        accent_hi: rgb(0xff7f45),
        accent_lo: rgb(0xd95b24),
        accent_soft: rgba(0xf06b2f26),
        ok: rgb(0x4cc38a),
        err: rgb(0xf0685c),
        warn: rgb(0xe8b34b),
        err_box_bg: rgb(0x2a1a18),
        err_box_border: rgb(0x5c332e),
        warn_box_bg: rgb(0x2a2418),
        warn_box_border: rgb(0x5c4a2a),
    }
}

/// Focus handles for the seven editable text fields, created once on first
/// render so focus state survives re-renders.
#[derive(Clone)]
struct FocusHandles {
    model: FocusHandle,
    layer: FocusHandle,
    value: FocusHandle,
    source_layer: FocusHandle,
    span: FocusHandle,
    max_tokens: FocusHandle,
    prompt: FocusHandle,
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
    focus: Option<FocusHandles>,
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
        Console {
            worker_tx,
            reply_rx,
            model_options: models,
            model_path,
            site_options: STAGES.iter().map(|s| s.to_string()).collect(),
            site: "after-mlp".to_string(),
            layer: "0".to_string(),
            op_options: OPERATIONS.iter().map(|s| s.to_string()).collect(),
            op: "scale".to_string(),
            value: "0.5".to_string(),
            source_options: vec!["capture".to_string(), "zero".to_string()],
            source: "capture".to_string(),
            source_layer: "0".to_string(),
            token_options: vec!["prompt-final".to_string(), "matched-span".to_string()],
            token: "prompt-final".to_string(),
            span: String::new(),
            max_tokens: "48".to_string(),
            execution_options: EXECUTIONS.iter().map(|s| s.to_string()).collect(),
            execution: "reference".to_string(),
            prompt: "\u{627}\u{643}\u{62A}\u{628} \u{62C}\u{645}\u{644}\u{629} \
                     \u{642}\u{635}\u{64A}\u{631}\u{629} \u{639}\u{646} \u{627}\u{644}\u{645}\u{62F}\u{64A}\u{646}\u{629} \
                     \u{627}\u{644}\u{645}\u{646}\u{648}\u{631}\u{629}"
                .to_string(),
            open_combo: None,
            focus: None,
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
    fn colors(&self) -> Colors {
        if self.dark {
            dark()
        } else {
            light()
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

    /// Drain the worker reply channel; returns true when anything changed.
    fn drain_replies(&mut self) -> bool {
        let replies: Vec<WorkerReply> = {
            let rx = self.reply_rx.lock().expect("reply receiver lock");
            std::iter::from_fn(|| rx.try_recv().ok()).collect()
        };
        if replies.is_empty() {
            return false;
        }
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
        self.error = None;
        let _ = self.worker_tx.send(WorkerMsg::Prepare(path));
    }

    fn run(&mut self) {
        if self.busy() {
            return;
        }
        match self.build_run_request() {
            Ok(req) => match parse_run_request(&req) {
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

    fn toggle_theme(&mut self) {
        self.dark = !self.dark;
    }

    fn select_combo(&mut self, combo: ComboId, value: &str, cx: &mut Context<Self>) {
        match combo {
            ComboId::Model => self.model_path = value.to_string(),
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
        }
        self.open_combo = None;
        cx.notify();
    }

    /// Keep the source layer at or above the target layer (the capture must
    /// fire before the intervention in the same pass).
    fn clamp_source_layer(&mut self) {
        if let (Ok(target), Ok(source)) =
            (self.layer.parse::<i64>(), self.source_layer.parse::<i64>())
            && source > target
        {
            self.source_layer = (target - 1).max(0).to_string();
        }
    }

    /// Handle a keystroke for one of the editable text fields.
    fn input_key(&mut self, target: InputTarget, e: &KeyDownEvent) {
        // Ignore shortcuts; only plain text entry and backspace are handled.
        if e.keystroke.modifiers.control {
            return;
        }
        let field: &mut String = match target {
            InputTarget::ModelPath => &mut self.model_path,
            InputTarget::Layer => &mut self.layer,
            InputTarget::Value => &mut self.value,
            InputTarget::SourceLayer => &mut self.source_layer,
            InputTarget::Span => &mut self.span,
            InputTarget::MaxTokens => &mut self.max_tokens,
            InputTarget::Prompt => &mut self.prompt,
        };
        match e.keystroke.key.as_str() {
            "backspace" => {
                field.pop();
            }
            "enter" | "return" => {
                if target == InputTarget::Prompt {
                    field.push('\n');
                }
            }
            "space" => field.push(' '),
            "escape" | "tab" => {}
            _ => {
                if let Some(ch) = e.keystroke.key_char.as_deref() {
                    field.push_str(ch);
                } else if e.keystroke.key.chars().count() == 1 {
                    field.push_str(&e.keystroke.key);
                }
            }
        }
        if target == InputTarget::Layer {
            self.clamp_source_layer();
        }
    }

    /// Poll the worker reply channel every 80 ms while the app lives. The
    /// worker thread is unchanged from the iced implementation; only the
    /// foreground subscription is replaced by gpui's async executor.
    fn spawn_poll(&mut self, cx: &mut Context<Self>) {
        cx.spawn(|this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let mut cx = cx.clone();
            async move {
                loop {
                    cx.background_executor()
                        .timer(Duration::from_millis(80))
                        .await;
                    let _ = this.update(&mut cx, |console, cx| {
                        if console.drain_replies() {
                            cx.notify();
                        }
                    });
                }
            }
        })
        .detach();
    }

    // -- view builders -------------------------------------------------------

    fn picker(
        &self,
        colors: &Colors,
        id: &'static str,
        combo: ComboId,
        selected: &str,
        options: &[String],
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let open = self.open_combo == Some(combo);
        let toggle = cx.listener(move |console, _: &ClickEvent, _w, cx| {
            console.open_combo = if console.open_combo == Some(combo) {
                None
            } else {
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
            .bg(colors.panel_alt)
            .border_1()
            .border_color(colors.border)
            .rounded_md()
            .cursor_pointer()
            .on_click(toggle)
            .child(label(selected.to_string(), 12.0, colors.text))
            .child(label("\u{25be}", 10.0, colors.faint));

        if open {
            let list = options
                .iter()
                .map(|opt| {
                    let opt = opt.clone();
                    let is_selected = opt == selected;
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
                        .cursor_pointer()
                        .when(is_selected, |s| s.bg(colors.accent_soft))
                        .hover(|s| s.bg(colors.panel_alt))
                        .on_click(listener)
                        .child(label(opt, 12.0, colors.text))
                        .into_any_element()
                })
                .collect::<Vec<_>>();
            div()
                .id(ElementId::Name(SharedString::from(format!("{id}:list"))))
                .flex_col()
                .w_full()
                .bg(colors.panel)
                .rounded_md()
                .child(div().flex_col().children(list))
        } else {
            button
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
            console.toggle_theme();
            cx.notify();
        });
        div()
            .flex()
            .flex_row()
            .items_center()
            .gap_3()
            .px_4()
            .h(px(44.0))
            .w_full()
            .bg(colors.panel)
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
                    .child(label("experiment console", 9.0, colors.faint)),
            )
            .child(div().w_full())
            .child(
                div()
                    .px_2()
                    .py_1()
                    .bg(colors.panel_alt)
                    .rounded_full()
                    .child(mono(session_chip, 10.0, colors.dim)),
            )
            .child(
                div()
                    .id(ElementId::Name(SharedString::from("theme-toggle")))
                    .size(px(28.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(colors.panel_alt)
                    .border_1()
                    .border_color(colors.border)
                    .rounded_md()
                    .cursor_pointer()
                    .hover(|s| s.border_color(colors.faint))
                    .on_click(toggle)
                    .child(label(
                        if self.dark { "\u{2600}" } else { "\u{263e}" },
                        12.0,
                        colors.dim,
                    )),
            )
    }

    fn sidebar(
        &self,
        colors: &Colors,
        focus: &FocusHandles,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let n_layers = self.session.as_ref().map(|s| s.n_layers).unwrap_or(0);
        let layer_hint = if per_layer(&self.site) && n_layers > 0 {
            format!("0 \u{2013} {} \u{00b7} {n_layers} layers", n_layers - 1)
        } else {
            "no per-layer site".to_string()
        };

        let needs_source = matches!(self.op.as_str(), "replace" | "interpolate" | "add-delta");
        let needs_value = matches!(self.op.as_str(), "scale" | "interpolate");
        let value_label = if self.op == "interpolate" {
            "ALPHA (0\u{2013}1)"
        } else {
            "VALUE"
        };
        let source_is_capture = self.source == "capture";

        let mut children: Vec<AnyElement> = Vec::new();

        // ---- MODEL ----
        children.push(section_label(colors, "MODEL").into_any_element());
        children.push(
            field(
                colors,
                "MODEL",
                self.picker(
                    colors,
                    "model-picker",
                    ComboId::Model,
                    &self.model_path,
                    &self.model_options,
                    cx,
                ),
            )
            .into_any_element(),
        );
        children.push(
            text_field(
                colors,
                "model-path",
                &self.model_path,
                "path to model.gguf",
                FONT_MONO_NAME,
                12.0,
                None,
                &focus.model,
                cx.listener(|console, e: &KeyDownEvent, _w, cx| {
                    console.input_key(InputTarget::ModelPath, e);
                    cx.notify();
                }),
            )
            .into_any_element(),
        );
        children.push(
            btn_secondary(
                colors,
                if self.status == Status::Preparing {
                    "LOADING\u{2026}"
                } else {
                    "LOAD"
                },
                (!self.busy()).then(|| {
                    cx.listener(|console, _: &ClickEvent, _w, cx| {
                        console.load();
                        cx.notify();
                    })
                }),
            )
            .into_any_element(),
        );
        match &self.session {
            Some(info) => children.push(
                mono(
                    format!(
                        "loaded {} \u{00b7} {} layers \u{00b7} {}d \u{00b7} {}",
                        info.architecture,
                        info.n_layers,
                        info.embed_dim,
                        fmt_load_ms(info.load_ms)
                    ),
                    10.0,
                    colors.ok,
                )
                .into_any_element(),
            ),
            None => children.push(
                label(
                    "no model loaded \u{2014} pick a .gguf, then LOAD",
                    10.0,
                    colors.faint,
                )
                .into_any_element(),
            ),
        }
        children.push(rule_h(colors).into_any_element());

        // ---- HOOK & INTERVENTION ----
        children.push(section_label(colors, "HOOK & INTERVENTION").into_any_element());
        children.push(
            field(
                colors,
                "HOOK STAGE",
                self.picker(
                    colors,
                    "site-picker",
                    ComboId::Site,
                    &self.site,
                    &self.site_options,
                    cx,
                ),
            )
            .into_any_element(),
        );
        children.push(
            field(
                colors,
                "LAYER",
                text_field(
                    colors,
                    "layer",
                    &self.layer,
                    "0",
                    FONT_MONO_NAME,
                    12.0,
                    None,
                    &focus.layer,
                    cx.listener(|console, e: &KeyDownEvent, _w, cx| {
                        console.input_key(InputTarget::Layer, e);
                        cx.notify();
                    }),
                ),
            )
            .into_any_element(),
        );
        children.push(label(layer_hint.clone(), 9.0, colors.faint).into_any_element());
        children.push(
            field(
                colors,
                "INTERVENTION",
                self.picker(
                    colors,
                    "op-picker",
                    ComboId::Op,
                    &self.op,
                    &self.op_options,
                    cx,
                ),
            )
            .into_any_element(),
        );
        if needs_value {
            children.push(
                div()
                    .flex_col()
                    .gap_1()
                    .child(field(
                        colors,
                        value_label,
                        text_field(
                            colors,
                            "value",
                            &self.value,
                            "0.5",
                            FONT_SANS_NAME,
                            12.0,
                            None,
                            &focus.value,
                            cx.listener(|console, e: &KeyDownEvent, _w, cx| {
                                console.input_key(InputTarget::Value, e);
                                cx.notify();
                            }),
                        ),
                    ))
                    .child(label(
                        if self.op == "interpolate" {
                            "blend toward the source"
                        } else {
                            "multiplicative factor"
                        },
                        9.0,
                        colors.faint,
                    ))
                    .into_any_element(),
            );
        }
        if needs_source {
            children.push(
                field(
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
                )
                .into_any_element(),
            );
            if source_is_capture {
                children.push(
                    div()
                        .flex_col()
                        .gap_1()
                        .child(field(
                            colors,
                            "SOURCE LAYER",
                            text_field(
                                colors,
                                "source-layer",
                                &self.source_layer,
                                "0",
                                FONT_MONO_NAME,
                                12.0,
                                None,
                                &focus.source_layer,
                                cx.listener(|console, e: &KeyDownEvent, _w, cx| {
                                    console.input_key(InputTarget::SourceLayer, e);
                                    cx.notify();
                                }),
                            ),
                        ))
                        .child(label(
                            "capture fires before the intervention (same pass)",
                            9.0,
                            colors.faint,
                        ))
                        .into_any_element(),
                );
            }
        }
        children.push(rule_h(colors).into_any_element());

        // ---- TARGET ----
        children.push(section_label(colors, "TARGET").into_any_element());
        children.push(
            field(
                colors,
                "TARGET TOKENS",
                self.picker(
                    colors,
                    "token-picker",
                    ComboId::Token,
                    &self.token,
                    &self.token_options,
                    cx,
                ),
            )
            .into_any_element(),
        );
        if self.token == "matched-span" {
            children.push(
                text_field(
                    colors,
                    "span",
                    &self.span,
                    "\u{643}\u{644}\u{645}\u{629} \u{641}\u{64A} \u{627}\u{644}\u{646}\u{635}",
                    FONT_ARABIC_NAME,
                    12.0,
                    None,
                    &focus.span,
                    cx.listener(|console, e: &KeyDownEvent, _w, cx| {
                        console.input_key(InputTarget::Span, e);
                        cx.notify();
                    }),
                )
                .into_any_element(),
            );
        }
        children.push(rule_h(colors).into_any_element());

        // ---- ACTIONS ----
        children.push(section_label(colors, "ACTIONS").into_any_element());
        children.push(
            btn_primary(
                colors,
                match self.status {
                    Status::Running => "RUNNING\u{2026}",
                    Status::Preparing => "LOADING MODEL\u{2026}",
                    Status::Restoring => "RESTORING\u{2026}",
                    Status::Idle => "RUN EXPERIMENT",
                },
                (!self.busy()).then(|| {
                    cx.listener(|console, _: &ClickEvent, _w, cx| {
                        console.run();
                        cx.notify();
                    })
                }),
            )
            .into_any_element(),
        );
        children.push(
            btn_secondary(
                colors,
                "VERIFY RESTORE",
                (!self.busy()).then(|| {
                    cx.listener(|console, _: &ClickEvent, _w, cx| {
                        console.restore();
                        cx.notify();
                    })
                }),
            )
            .into_any_element(),
        );
        children.push(
            label(
                "change layer or intervention \u{2192} RUN \u{2192} compare \u{2192} VERIFY RESTORE",
                9.0,
                colors.faint,
            )
            .into_any_element(),
        );

        div()
            .id(ElementId::Name(SharedString::from("sidebar")))
            .flex_col()
            .gap_3()
            .p_4()
            .h_full()
            .overflow_scroll()
            .children(children)
    }

    fn main_panel(
        &self,
        colors: &Colors,
        focus: &FocusHandles,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let prompt_editor = text_field(
            colors,
            "prompt",
            &self.prompt,
            "prompt\u{2026}",
            FONT_ARABIC_NAME,
            14.0,
            Some(84.0),
            &focus.prompt,
            cx.listener(|console, e: &KeyDownEvent, _w, cx| {
                console.input_key(InputTarget::Prompt, e);
                cx.notify();
            }),
        );
        let prompt_meta = div()
            .flex()
            .flex_row()
            .items_end()
            .gap_2()
            .child(field(
                colors,
                "MAX TOKENS",
                text_field(
                    colors,
                    "max-tokens",
                    &self.max_tokens,
                    "48",
                    FONT_MONO_NAME,
                    12.0,
                    None,
                    &focus.max_tokens,
                    cx.listener(|console, e: &KeyDownEvent, _w, cx| {
                        console.input_key(InputTarget::MaxTokens, e);
                        cx.notify();
                    }),
                ),
            ))
            .child(field(
                colors,
                "EXECUTION",
                self.picker(
                    colors,
                    "execution-picker",
                    ComboId::Execution,
                    &self.execution,
                    &self.execution_options,
                    cx,
                ),
            ))
            .child(div().w_full())
            .child(label(
                "greedy \u{00b7} deterministic \u{00b7} temp 0.0",
                9.0,
                colors.faint,
            ));

        let error = match &self.error {
            Some(error) => div()
                .w_full()
                .px_3()
                .py_2()
                .bg(colors.err_box_bg)
                .border_1()
                .border_color(colors.err_box_border)
                .rounded_md()
                .child(mono(error.clone(), 11.0, colors.err))
                .into_any_element(),
            None => div().h(px(0.0)).into_any_element(),
        };
        let warning = match &self.warning {
            Some(warning) => div()
                .w_full()
                .px_3()
                .py_2()
                .bg(colors.warn_box_bg)
                .border_1()
                .border_color(colors.warn_box_border)
                .rounded_md()
                .child(mono(warning.clone(), 10.0, colors.warn))
                .into_any_element(),
            None => div().h(px(0.0)).into_any_element(),
        };

        let outputs = div()
            .flex()
            .flex_row()
            .gap_3()
            .child(self.output_panel(colors, "BASELINE", self.baseline.as_ref(), self.status))
            .child(self.output_panel(
                colors,
                "INTERVENTION",
                self.intervention.as_ref(),
                self.status,
            ));

        div()
            .id(ElementId::Name(SharedString::from("main-panel")))
            .flex_col()
            .gap_4()
            .w_full()
            .h_full()
            .p_4()
            .overflow_scroll()
            .child(panel(
                colors,
                div()
                    .flex_col()
                    .gap_2()
                    .child(section_label(colors, "PROMPT"))
                    .child(prompt_editor)
                    .child(prompt_meta),
            ))
            .child(error)
            .child(warning)
            .child(outputs)
            .child(self.verification_panel(colors))
    }

    fn output_panel(
        &self,
        colors: &Colors,
        title: &'static str,
        output: Option<&RunOutput>,
        status: Status,
    ) -> Div {
        let (badge_text, badge_color) = match (output, status) {
            (Some(_), _) => ("OK", colors.ok),
            (None, Status::Running) => ("RUN", colors.warn),
            (None, _) => ("\u{2014}", colors.faint),
        };
        let body: Div = match output {
            Some(out) if !out.text.is_empty() => div()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .w_full()
                        .px_2()
                        .py_2()
                        .bg(colors.panel_alt)
                        .border_1()
                        .border_color(colors.border)
                        .rounded_md()
                        .child(multiline(&out.text, 14.0, colors.text, FONT_ARABIC_NAME)),
                )
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .child(mono(
                            format!(
                                "{} tok \u{00b7} {} \u{00b7} {}",
                                out.generated_tokens,
                                fmt_ms(out.wall_ms),
                                fmt_tps(out.decode_tps)
                            ),
                            10.0,
                            colors.dim,
                        ))
                        .child(div().w_full())
                        .child(mono(
                            format!(
                                "prompt {} tok \u{00b7} bundle {}",
                                out.prompt_tokens,
                                short_id(&out.semantic_hash)
                            ),
                            9.0,
                            colors.faint,
                        )),
                )
                .child(mono(out.bundle_dir.clone(), 9.0, colors.faint)),
            Some(_out) => div().child(label("(empty output)", 12.0, colors.faint)),
            None => div().child(label(
                "no run yet \u{2014} outputs appear here",
                12.0,
                colors.faint,
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
                        .child(label(title, 11.0, colors.dim))
                        .child(div().w_full())
                        .child(chip(badge_text, badge_color)),
                )
                .child(body),
        )
    }

    fn verification_panel(&self, colors: &Colors) -> Div {
        let (badge, badge_color) = match (&self.verification, self.status) {
            (Some(verification), _) if verification.ok => ("VERIFIED", colors.ok),
            (Some(_), _) => ("VERIFICATION FAILED", colors.err),
            (None, Status::Running) => ("RUNNING", colors.warn),
            (None, Status::Restoring) => ("RESTORING", colors.warn),
            (None, _) => ("NOT RUN", colors.faint),
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
                colors.faint,
            ))
        } else {
            div().flex_col().gap_1().children(
                lines
                    .iter()
                    .map(|line| mono(line.clone(), 10.0, colors.dim).into_any_element())
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
                        .child(mono(metrics, 10.0, colors.faint)),
                )
                .child(detail),
        )
    }

    fn statusbar(&self, colors: &Colors) -> Div {
        let (dot, status_text) = match self.status {
            Status::Idle => (rgb(0x6e778a), "idle"),
            Status::Preparing => (colors.warn, "loading model\u{2026}"),
            Status::Running => (colors.accent, "running experiment\u{2026}"),
            Status::Restoring => (rgb(0x6ba7ff), "verifying restore\u{2026}"),
        };
        let layer_hook = if per_layer(&self.site) {
            format!("L{} \u{00b7} {}", self.layer, self.site)
        } else {
            self.site.clone()
        };
        let intervention = match self.op.as_str() {
            "scale" => format!("scale \u{00d7}{}", self.value),
            "interpolate" => format!("interpolate \u{03b1}={}", self.value),
            op => op.to_string(),
        };
        let metrics = self
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

        div()
            .flex()
            .flex_row()
            .items_center()
            .gap_3()
            .px_4()
            .h(px(34.0))
            .w_full()
            .bg(colors.panel)
            .border_t_1()
            .border_color(colors.border)
            .child(status_dot(dot))
            .child(label(status_text.to_string(), 10.0, colors.dim))
            .child(div().w(px(1.0)).h(px(12.0)).bg(colors.border))
            .child(mono(
                format!("model {}", self.model_name()),
                10.0,
                colors.dim,
            ))
            .child(mono(format!("layer/hook {layer_hook}"), 10.0, colors.dim))
            .child(mono(
                format!("intervention {intervention}"),
                10.0,
                colors.dim,
            ))
            .child(div().w_full())
            .child(mono(metrics, 10.0, colors.ok))
    }
}

impl Render for Console {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = self.colors();
        let focus = self
            .focus
            .get_or_insert_with(|| FocusHandles {
                model: cx.focus_handle(),
                layer: cx.focus_handle(),
                value: cx.focus_handle(),
                source_layer: cx.focus_handle(),
                span: cx.focus_handle(),
                max_tokens: cx.focus_handle(),
                prompt: cx.focus_handle(),
            })
            .clone();

        let header = self.header(&colors, cx);
        let body = div()
            .flex()
            .flex_row()
            .w_full()
            .h_full()
            .child(
                div()
                    .w(px(272.0))
                    .h_full()
                    .bg(colors.bg)
                    .border_r_1()
                    .border_color(colors.border)
                    .child(self.sidebar(&colors, &focus, cx)),
            )
            .child(self.main_panel(&colors, &focus, cx));
        let statusbar = self.statusbar(&colors);

        div()
            .flex_col()
            .w_full()
            .h_full()
            .bg(colors.bg)
            .child(header)
            .child(body)
            .child(statusbar)
    }
}

// ---------------------------------------------------------------------------
// view helpers (pure presentation)
// ---------------------------------------------------------------------------

fn label(content: impl Into<SharedString>, size: f32, color: Rgba) -> Div {
    let content = content.into();
    div().child(content).text_size(px(size)).text_color(color)
}

fn mono(content: impl Into<SharedString>, size: f32, color: Rgba) -> Div {
    let content = content.into();
    div()
        .child(content)
        .font_family(FONT_MONO_NAME)
        .text_size(px(size))
        .text_color(color)
}

/// Render a possibly-multi-line string as a stack of single-line divs, so
/// newlines are preserved regardless of the element's white-space handling.
fn multiline(content: &str, size: f32, color: Rgba, font: &'static str) -> Div {
    div().flex_col().children(
        content
            .split('\n')
            .map(|line| {
                div()
                    .child(line.to_string())
                    .font_family(font)
                    .text_size(px(size))
                    .text_color(color)
                    .into_any_element()
            })
            .collect::<Vec<_>>(),
    )
}

/// A small uppercase group label with a slim ember tick.
fn section_label(colors: &Colors, label_text: &'static str) -> Div {
    div()
        .flex()
        .items_center()
        .gap_2()
        .child(
            div()
                .w(px(2.0))
                .h(px(10.0))
                .bg(colors.accent)
                .rounded_full(),
        )
        .child(label(label_text, 10.0, colors.faint))
}

fn field(colors: &Colors, title: &'static str, control: impl IntoElement) -> Div {
    div()
        .flex_col()
        .gap_1()
        .w_full()
        .child(label(title, 10.0, colors.faint))
        .child(control)
}

/// A raised surface: borderless, separated from the canvas by tone alone.
fn panel(colors: &Colors, content: impl IntoElement) -> Div {
    div()
        .w_full()
        .p_4()
        .bg(colors.panel)
        .rounded(px(10.0))
        .child(content)
}

fn rule_h(colors: &Colors) -> Div {
    div().w_full().h(px(1.0)).bg(colors.border)
}

/// A rounded status pill (tinted background + matching text).
fn chip(label_text: &str, color: Rgba) -> Div {
    let hsla = Hsla::from(color);
    div()
        .px_2()
        .py_1()
        .bg(hsla.opacity(0.13))
        .border_1()
        .border_color(hsla.opacity(0.40))
        .rounded_full()
        .child(label(label_text.to_string(), 10.0, color))
}

/// A small colored status dot.
fn status_dot(color: Rgba) -> Div {
    let hsla = Hsla::from(color);
    div()
        .size(px(8.0))
        .bg(color)
        .border_1()
        .border_color(hsla.opacity(0.45))
        .rounded_full()
}

/// Full-width ember primary button.
fn btn_primary(
    colors: &Colors,
    label_text: &str,
    on_click: Option<impl Fn(&ClickEvent, &mut Window, &mut App) + 'static>,
) -> Stateful<Div> {
    let mut button = div()
        .id(ElementId::Name(SharedString::from(format!(
            "btn:{label_text}"
        ))))
        .w_full()
        .h(px(32.0))
        .px_4()
        .flex()
        .items_center()
        .justify_center()
        .bg(colors.accent)
        .rounded_md()
        .cursor_pointer()
        .hover(|s| s.bg(colors.accent_hi))
        .active(|s| s.bg(colors.accent_lo))
        .child(label(label_text.to_string(), 12.0, rgb(0xffffff)));
    if let Some(on_click) = on_click {
        button = button.on_click(on_click);
    }
    button
}

/// Full-width secondary (outlined) button.
fn btn_secondary(
    colors: &Colors,
    label_text: &str,
    on_click: Option<impl Fn(&ClickEvent, &mut Window, &mut App) + 'static>,
) -> Stateful<Div> {
    let mut button = div()
        .id(ElementId::Name(SharedString::from(format!(
            "btn:{label_text}"
        ))))
        .w_full()
        .h(px(32.0))
        .px_4()
        .flex()
        .items_center()
        .justify_center()
        .bg(colors.panel)
        .border_1()
        .border_color(colors.border)
        .rounded_md()
        .cursor_pointer()
        .hover(|s| s.bg(colors.panel_alt).border_color(colors.accent))
        .child(label(label_text.to_string(), 12.0, colors.text));
    if let Some(on_click) = on_click {
        button = button.on_click(on_click);
    }
    button
}

/// A single-line (or fixed-height multi-line) text field.
#[allow(clippy::too_many_arguments)] // private helper; font/size/height are positional presentation knobs
fn text_field(
    colors: &Colors,
    id: &'static str,
    value: &str,
    placeholder: &'static str,
    font: &'static str,
    size: f32,
    height: Option<f32>,
    focus: &FocusHandle,
    on_key_down: impl Fn(&KeyDownEvent, &mut Window, &mut App) + 'static,
) -> Stateful<Div> {
    let (shown, color) = if value.is_empty() {
        (placeholder.to_string(), colors.faint)
    } else {
        (value.to_string(), colors.text)
    };
    let focus_handle = focus.clone();
    let mut field = div()
        .id(ElementId::Name(SharedString::from(id)))
        .w_full()
        .px_2()
        .py_1()
        .bg(colors.panel_alt)
        .border_1()
        .border_color(colors.border)
        .rounded_md()
        .track_focus(focus)
        .cursor_text()
        .on_key_down(on_key_down)
        .on_click(move |_e, window, _cx| {
            focus_handle.focus(window);
        })
        .child(multiline(&shown, size, color, font));
    if let Some(height) = height {
        field = field.h(px(height)).overflow_scroll();
    }
    field
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

    Application::new().run(move |cx: &mut App| {
        // Register the embedded fonts before the first window opens so the
        // text system can resolve Noto Sans / Mono / Naskh Arabic offline.
        cx.text_system()
            .add_fonts(vec![
                Cow::Borrowed(FONT_SANS),
                Cow::Borrowed(FONT_MONO),
                Cow::Borrowed(FONT_ARABIC),
            ])
            .expect("register embedded fonts");

        let bounds = Bounds::centered(None, size(px(1240.0), px(860.0)), cx);
        cx.open_window(
            WindowOptions {
                titlebar: Some(TitlebarOptions {
                    title: Some("EMBER \u{2014} experiment console".into()),
                    ..Default::default()
                }),
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            move |_window, cx| {
                cx.new(|cx| {
                    let mut console = Console::new(worker_tx, reply_rx);
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
    use super::{dark, light, Console};
    use crate::gui::parse_run_request;
    use std::sync::{mpsc, Arc, Mutex};

    fn console() -> Console {
        let (tx, _rx) = mpsc::channel();
        let (_rtx, rrx) = mpsc::channel();
        Console::new(tx, Arc::new(Mutex::new(rrx)))
    }

    #[test]
    fn console_defaults_to_dark_and_toggle_switches_palette() {
        let mut state = console();
        assert!(state.dark, "the console starts in dark mode");
        assert_eq!(state.colors().bg, dark().bg);
        state.toggle_theme();
        assert!(!state.dark);
        assert_eq!(state.colors().bg, light().bg);
        assert_eq!(state.colors().panel, light().panel);
        state.toggle_theme();
        assert!(state.dark);
    }

    #[test]
    fn light_and_dark_palettes_differ_in_every_role() {
        let dark = dark();
        let light = light();
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
