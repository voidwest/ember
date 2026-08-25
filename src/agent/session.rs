//! The agent run state machine (Tracks D/E/G/X).
//!
//! One explicit loop, no recursion:
//!
//! ```text
//! Start -> commit system+user -> ModelTurn(step)
//!       -> ParseAction
//!            FinalText            -> Finalize -> RunCompleted
//!            ToolCall             -> Validate -> (auto-approve) -> Execute
//!                                    -> ToolResult -> ReinjectIntoSession
//!                                    -> ModelTurn(step+1)   [limits checked]
//!            MalformedToolCall    -> structured rejection reinjected
//!                                    -> ModelTurn(step+1)   [limits checked]
//! ```
//!
//! Commit semantics (Track G), exactly:
//!
//! - a finished generation commits scaffold + content + terminal policy
//!   tokens as ONE message (the engine's transaction contract);
//! - a tool result commits only after execution returns; if the run is
//!   cancelled between execution and commit, the side effect happened but
//!   the result is NOT in the session (`tool_result_uncommitted`);
//! - validation failures are DATA: they are traced and reinjected so the
//!   model can recover; they never abort the run by themselves;
//! - cancellation mid-generation leaves nothing committed (engine rolls
//!   back); cancellation between steps leaves a clean prefix.
//!
//! Hard limits (Track E): steps, tool calls, wall time, per-tool timeout,
//! per-turn output tokens, tool-result bytes. Firing a limit terminates
//! cleanly with a structured reason and preserves all committed state.

use anyhow::{Context as _, Result};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::extraction::sha256_bytes;

use super::artifact::ArtifactStore;
use super::ids;
use super::model::{ChatModelEngine, GenerationParams};
use super::protocol::{AssistantAction, ToolCallProtocol, ToolResultMessage};
use super::tool::{CancelFlag, ToolInvocation, ToolRegistry};
use super::trace::{self, TraceRecorder};

/// Hard runtime limits for one agent run (Track E).
#[derive(Debug, Clone)]
pub struct AgentLimits {
    /// Maximum MODEL turns (each may contain at most one tool call).
    pub max_steps: usize,
    /// Maximum successful tool EXECUTIONS per run.
    pub max_tool_calls: usize,
    /// Whole-run wall-clock budget. `None` disables (not recommended).
    pub max_wall_time: Option<Duration>,
    /// Per-tool-execution deadline (watchdog-enforced).
    pub tool_timeout: Duration,
    /// Output token cap per model turn.
    pub max_output_tokens_per_turn: usize,
    /// Reinjection cap: payloads above this size enter the session
    /// truncated (with an explicit marker), keeping context bounded.
    pub max_tool_result_bytes: usize,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_tool_calls: 16,
            max_wall_time: Some(Duration::from_secs(600)),
            tool_timeout: Duration::from_secs(60),
            max_output_tokens_per_turn: 256,
            max_tool_result_bytes: 32 * 1024,
        }
    }
}

/// Static configuration of a session (protocol + sampling knobs).
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Base instruction placed before the tool definitions.
    pub system_prompt: Option<String>,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    /// Sampling seed (temperature > 0); deterministic when set.
    pub seed: Option<u64>,
    /// KV capacity for the conversation (context budget).
    pub kv_capacity: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: None,
            kv_capacity: 8192,
        }
    }
}

/// Why/whether a run is still meaningful. Serialized into summaries.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RunStatus {
    /// Model produced a final answer.
    Completed,
    /// Cancelled via the control flag (state preserved, see ledger).
    Cancelled,
    /// A hard limit fired; committed state is valid up to the limit.
    LimitReached(LimitKind),
    /// Infrastructure failure (generation error, context overflow...).
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LimitKind {
    MaxSteps,
    MaxToolCalls,
    WallTime,
}

impl LimitKind {
    pub fn as_str(self) -> &'static str {
        match self {
            LimitKind::MaxSteps => "max_steps",
            LimitKind::MaxToolCalls => "max_tool_calls",
            LimitKind::WallTime => "max_wall_time",
        }
    }
}

/// Explicit representation of everything committed to the session
/// (Track G). The ledger IS the answer to "what state changed?".
#[derive(Debug, Clone)]
pub enum CommittedMessage {
    System {
        span: (usize, usize),
        rendered: String,
    },
    User {
        span: (usize, usize),
        content: String,
    },
    AssistantFinal {
        span: (usize, usize),
        text: String,
    },
    AssistantToolCall {
        span: (usize, usize),
        text_before_call: String,
        tool_name: String,
    },
    ToolResult {
        span: (usize, usize),
        call_name: String,
        ok: bool,
    },
}

impl CommittedMessage {
    pub fn span(&self) -> (usize, usize) {
        match self {
            CommittedMessage::System { span, .. }
            | CommittedMessage::User { span, .. }
            | CommittedMessage::AssistantFinal { span, .. }
            | CommittedMessage::AssistantToolCall { span, .. }
            | CommittedMessage::ToolResult { span, .. } => *span,
        }
    }

    pub fn role_label(&self) -> &'static str {
        match self {
            CommittedMessage::System { .. } => "system",
            CommittedMessage::User { .. } => "user",
            CommittedMessage::AssistantFinal { .. } => "assistant_final",
            CommittedMessage::AssistantToolCall { .. } => "assistant_tool_call",
            CommittedMessage::ToolResult { .. } => "tool_result",
        }
    }
}

/// Result summary returned by every run (and mirrored into the trace).
#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentRunSummary {
    pub run_id: String,
    pub status: RunStatus,
    pub final_text: Option<String>,
    pub steps_executed: usize,
    pub tool_calls_executed: usize,
    pub rejected_calls: usize,
    pub total_model_ms: f64,
    pub total_tool_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub prompt_tokens_committed: usize,
    pub output_tokens: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<super::artifact::ArtifactRecord>,
    pub ledger_roles: Vec<&'static str>,
    pub trace_write_errors: u64,
}

/// One configured agent session over a resident engine.
///
/// Construct once per loaded model; call [`AgentSession::run`] per task.
/// Each `run` gets a fresh run_id, artifact store view, and trace.
pub struct AgentSession<'e> {
    engine: &'e mut dyn ChatModelEngine,
    registry: ToolRegistry,
    protocol: Arc<dyn ToolCallProtocol>,
    config: AgentConfig,
    limits: AgentLimits,

    // per-run state
    run_id: String,
    trace: Option<TraceRecorder>,
    artifacts: Arc<Mutex<ArtifactStore>>,
    ledger: Vec<CommittedMessage>,
    system_committed: bool,
    prompt_tokens_committed: usize,

    // totals
    steps_executed: usize,
    tool_calls_executed: usize,
    rejected_calls: usize,
    total_model_ms: f64,
    total_tool_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    output_tokens: usize,
}

/// Trace/artifact plumbing handed to a single run.
pub struct RunResources {
    pub trace: Option<TraceRecorder>,
    pub artifacts: Arc<Mutex<ArtifactStore>>,
}

impl<'e> AgentSession<'e> {
    pub fn new(
        engine: &'e mut dyn ChatModelEngine,
        protocol: Arc<dyn ToolCallProtocol>,
        registry: ToolRegistry,
        config: AgentConfig,
        limits: AgentLimits,
    ) -> Self {
        Self {
            engine,
            registry,
            protocol,
            config,
            limits,
            run_id: String::new(),
            trace: None,
            artifacts: Arc::new(Mutex::new(
                ArtifactStore::open(std::env::temp_dir(), "orphan").expect("temp artifact dir"),
            )),
            ledger: Vec::new(),
            system_committed: false,
            prompt_tokens_committed: 0,
            steps_executed: 0,
            tool_calls_executed: 0,
            rejected_calls: 0,
            total_model_ms: 0.0,
            total_tool_ms: 0.0,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            output_tokens: 0,
        }
    }

    pub fn ledger(&self) -> &[CommittedMessage] {
        &self.ledger
    }

    /// Take the run's trace recorder out (tests / post-run inspection).
    pub fn take_trace(&mut self) -> Option<TraceRecorder> {
        self.trace.take()
    }

    /// Read-only view of the current trace events (memory sink).
    pub fn trace_events(&self) -> Vec<serde_json::Value> {
        self.trace
            .as_ref()
            .map_or(Vec::new(), |t| t.events().to_vec())
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    fn tr(&mut self) -> &mut TraceRecorder {
        self.trace.as_mut().expect("run in progress")
    }

    // ------------------------------------------------------------------
    // main entry point
    // ------------------------------------------------------------------

    /// Run one user task through the loop (system prompt committed on the
    /// first run only). Returns a full summary; the trace carries every
    /// event along the way.
    pub fn run(
        &mut self,
        control: &CancelFlag,
        user_content: &str,
        resources: RunResources,
    ) -> Result<AgentRunSummary> {
        let run_id = ids::new_run_id();
        self.run_id = run_id.clone();
        self.trace = resources.trace;
        self.artifacts = resources.artifacts;
        if let Some(t) = self.trace.as_mut() {
            t.set_run_id(&run_id);
        }
        if let Ok(mut store) = self.artifacts.lock() {
            store.set_run_id(&run_id);
        }
        self.reset_run_totals();

        let started = Instant::now();
        let wall_deadline = self.limits.max_wall_time.map(|d| started + d);

        if let Some(t) = self.trace.as_mut() {
            t.emit(
                trace::ev::RUN_STARTED,
                "run",
                None,
                serde_json::json!({
                    "max_steps": self.limits.max_steps,
                    "max_tool_calls": self.limits.max_tool_calls,
                    "wall_time_ms": self.limits.max_wall_time.map(|d| d.as_millis()),
                    "tool_timeout_ms": self.limits.tool_timeout.as_millis(),
                }),
            );
            self.emit_provenance();
        }

        // -- commit system + user --------------------------------------
        let setup = self.commit_preamble(user_content);
        if let Err(e) = setup {
            return Ok(self.finish_failed(control, format!("session setup failed: {e:#}")));
        }

        let outcome = self.run_loop(control, wall_deadline);

        let summary = match outcome {
            LoopOutcome::Completed(final_text) => {
                if let Some(t) = self.trace.as_mut() {
                    let mut data = serde_json::json!({
                        "steps": self.steps_executed,
                        "tool_calls": self.tool_calls_executed,
                    });
                    if t.config().trace_generated_text {
                        data["final_text"] = serde_json::json!(final_text);
                    } else {
                        data["final_text_sha256"] =
                            serde_json::json!(sha256_bytes(final_text.as_bytes()));
                    }
                    t.emit(trace::ev::RUN_COMPLETED, "finalize", None, data);
                }
                self.summary(RunStatus::Completed, Some(final_text))
            }
            LoopOutcome::Cancelled(context) => {
                if let Some(t) = self.trace.as_mut() {
                    t.emit(
                        trace::ev::RUN_CANCELLED,
                        "finalize",
                        None,
                        serde_json::json!({ "at": context }),
                    );
                }
                self.summary(RunStatus::Cancelled, None)
            }
            LoopOutcome::Limit(kind) => {
                if let Some(t) = self.trace.as_mut() {
                    t.emit(
                        trace::ev::RUN_TERMINATED,
                        "finalize",
                        None,
                        serde_json::json!({ "limit_hit": kind.as_str(), "status": "limit_reached" }),
                    );
                }
                self.summary(RunStatus::LimitReached(kind), None)
            }
            LoopOutcome::Failed(message) => {
                if let Some(t) = self.trace.as_mut() {
                    t.emit(
                        trace::ev::RUN_FAILED,
                        "finalize",
                        None,
                        serde_json::json!({ "error": message }),
                    );
                }
                self.summary(RunStatus::Failed(message.clone()), None)
            }
        };

        // keep the recorder for inspection until the next run replaces it
        Ok(summary)
    }

    fn reset_run_totals(&mut self) {
        self.steps_executed = 0;
        self.tool_calls_executed = 0;
        self.rejected_calls = 0;
        self.total_model_ms = 0.0;
        self.total_tool_ms = 0.0;
        self.prefill_ms = 0.0;
        self.decode_ms = 0.0;
        self.output_tokens = 0;
    }

    // ------------------------------------------------------------------
    // loop internals
    // ------------------------------------------------------------------

    fn emit_provenance(&mut self) {
        let schemas: Vec<serde_json::Value> = self
            .registry
            .schemas()
            .iter()
            .map(|s| s.to_json_schema())
            .collect();
        let data = serde_json::json!({
            "ember_version": env!("CARGO_PKG_VERSION"),
            "git_commit": option_env!("EMBER_GIT_COMMIT"),
            "rustc": option_env!("EMBER_RUSTC_VERSION"),
            "target": option_env!("EMBER_TARGET"),
            "model": self.engine.identity(),
            "protocol_id": self.protocol.id(),
            "tools": schemas,
            "config": {
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "kv_capacity": self.config.kv_capacity,
            },
        });
        self.tr().emit(trace::ev::PROVENANCE, "run", None, data);
    }

    fn commit_preamble(&mut self, user_content: &str) -> Result<()> {
        if !self.system_committed {
            let rendered = self.protocol.render_system_message(
                self.config.system_prompt.as_deref(),
                &self.registry.schemas(),
            );
            let span = self
                .engine
                .commit_message(&rendered)
                .context("committing system message")?;
            self.ledger
                .push(CommittedMessage::System { span, rendered });
            self.system_committed = true;
        }

        let rendered_user = self.protocol.render_user_message(user_content);
        let span = self
            .engine
            .commit_message(&rendered_user)
            .context("committing user message")?;
        if let Some(t) = self.trace.as_mut() {
            t.emit_prompt(trace::ev::MESSAGE_COMMITTED, "user", "user", user_content);
            t.emit(
                trace::ev::SESSION_STATE_CHANGED,
                "user",
                None,
                serde_json::json!({
                    "role": "user",
                    "span": [span.0, span.1],
                    "committed_len": self.engine.committed_len(),
                }),
            );
        }
        self.prompt_tokens_committed += span.1 - span.0;
        self.ledger.push(CommittedMessage::User {
            span,
            content: user_content.to_string(),
        });
        Ok(())
    }

    fn run_loop(&mut self, control: &CancelFlag, wall_deadline: Option<Instant>) -> LoopOutcome {
        let mut step = 0usize;
        loop {
            // -- boundary checks (cancel first, always) -----------------
            if control.is_cancelled() {
                return LoopOutcome::Cancelled("step-boundary".to_string());
            }
            if let Some(deadline) = wall_deadline
                && Instant::now() >= deadline
            {
                return LoopOutcome::Limit(LimitKind::WallTime);
            }
            if step >= self.limits.max_steps {
                return LoopOutcome::Limit(LimitKind::MaxSteps);
            }
            let step_id = format!("model-{step}");

            // -- model turn ---------------------------------------------
            let prefix_rendered = self.protocol.render_assistant_prefix();
            let suffix_rendered = self.protocol.render_assistant_suffix();
            let params = GenerationParams {
                max_new_tokens: self.limits.max_output_tokens_per_turn,
                temperature: self.config.temperature,
                top_k: self.config.top_k,
                top_p: self.config.top_p,
                seed: self.config.seed,
                stop_strings: self.protocol.stop_strings(),
                extra_eos_tokens: self.protocol.extra_eos_tokens(),
            };

            if let Some(t) = self.trace.as_mut() {
                t.emit(
                    trace::ev::MODEL_CALL_STARTED,
                    &step_id,
                    None,
                    serde_json::json!({ "step_index": step }),
                );
            }
            let step_id_for_tokens = step_id.clone();
            let mut token_cb = |id: u32, piece: &str| {
                if let Some(t) = self.trace.as_mut() {
                    t.emit_token(&step_id_for_tokens, id, piece);
                }
            };
            let gen_start = Instant::now();
            let turn = match self.engine.generate_turn(
                &prefix_rendered,
                &suffix_rendered,
                &params,
                control,
                &mut token_cb,
            ) {
                Ok(turn) => turn,
                Err(e) => return LoopOutcome::Failed(format!("generation failed: {e:#}")),
            };
            let gen_ms = gen_start.elapsed().as_secs_f64() * 1e3;
            self.total_model_ms += gen_ms;
            self.prefill_ms += turn.prefill_ms;
            self.decode_ms += turn.decode_ms;
            self.output_tokens += turn.committed_ids.len();
            self.steps_executed += 1;
            step += 1;

            if turn.cancelled {
                return LoopOutcome::Cancelled("model-generation".to_string());
            }

            {
                let mut data = serde_json::json!({
                    "input_tokens_prefilled": turn.prompt_tokens_prefilled,
                    "output_tokens_committed": turn.committed_ids.len(),
                    "decode_evaluations": turn.decode_evaluations,
                    "prefill_ms": turn.prefill_ms,
                    "decode_ms": turn.decode_ms,
                    "tok_per_s": turn.tokens_per_second(),
                    "stop_reason": turn.stop.as_ref().map(|s| match s {
                        super::model::TurnStop::Eos => "eos".to_string(),
                        super::model::TurnStop::StopString(s2) => format!("stop_string:{s2}"),
                        super::model::TurnStop::MaxTokens => "max_tokens".to_string(),
                    }),
                    "wall_ms": gen_ms,
                });
                if let Some(t) = self.trace.as_mut() {
                    if t.config().trace_generated_text {
                        data["text"] = serde_json::json!(turn.text);
                    } else {
                        data["text_sha256"] = serde_json::json!(sha256_bytes(turn.text.as_bytes()));
                    }
                    t.emit(trace::ev::MODEL_CALL_FINISHED, &step_id, None, data);
                }
            }

            // -- parse action -------------------------------------------
            let action = self.protocol.parse_assistant_output(&turn.text);

            match action {
                AssistantAction::FinalText(text) => {
                    let span = (
                        self.engine
                            .committed_len()
                            .saturating_sub(turn.committed_ids.len()),
                        self.engine.committed_len(),
                    );
                    if let Some(t) = self.trace.as_mut() {
                        t.emit(
                            trace::ev::ASSISTANT_ACTION_PARSED,
                            &step_id,
                            Some("parse"),
                            serde_json::json!({ "action": "final_text" }),
                        );
                        t.emit(
                            trace::ev::SESSION_STATE_CHANGED,
                            &step_id,
                            None,
                            serde_json::json!({
                                "role": "assistant_final",
                                "span": [span.0, span.1],
                            }),
                        );
                    }
                    self.ledger.push(CommittedMessage::AssistantFinal {
                        span,
                        text: text.clone(),
                    });
                    return LoopOutcome::Completed(text);
                }
                AssistantAction::MalformedToolCall { excerpt, reason } => {
                    self.rejected_calls += 1;
                    if let Some(t) = self.trace.as_mut() {
                        t.emit(
                            trace::ev::ASSISTANT_ACTION_PARSED,
                            &step_id,
                            Some("parse"),
                            serde_json::json!({ "action": "malformed_tool_call", "reason": reason, "excerpt": excerpt }),
                        );
                        t.emit(
                            trace::ev::TOOL_CALL_REJECTED,
                            &step_id,
                            Some("validate"),
                            serde_json::json!({ "kind": "malformed_tool_call", "reason": reason }),
                        );
                    }
                    let feedback = ToolResultMessage::from_text(
                        "<malformed>",
                        false,
                        &format!("your tool call could not be parsed: {reason}"),
                    );
                    if let Err(LoopOutcome::Failed(m)) = self.commit_feedback(&feedback) {
                        return LoopOutcome::Failed(m);
                    }
                    continue;
                }
                AssistantAction::ToolCall(raw_call) => {
                    if raw_call.additional_calls_ignored > 0
                        && let Some(t) = self.trace.as_mut()
                    {
                        t.emit(
                            trace::ev::ASSISTANT_ACTION_PARSED,
                            &step_id,
                            Some("parse"),
                            serde_json::json!({
                                "action": "tool_call",
                                "note": "additional calls ignored (phase-1 single-call limit)",
                                "ignored": raw_call.additional_calls_ignored,
                            }),
                        );
                    }
                    // ledger: assistant requested a tool (its message is committed)
                    let call_span = (
                        self.engine
                            .committed_len()
                            .saturating_sub(turn.committed_ids.len()),
                        self.engine.committed_len(),
                    );
                    self.ledger.push(CommittedMessage::AssistantToolCall {
                        span: call_span,
                        text_before_call: String::new(),
                        tool_name: raw_call.name.clone(),
                    });

                    // -- validate ---------------------------------------
                    let validation = self.validate_call(&raw_call.name, &raw_call.arguments_json);
                    let call = match validation {
                        Ok(call) => {
                            if let Some(t) = self.trace.as_mut() {
                                t.emit(
                                    trace::ev::TOOL_CALL_VALIDATED,
                                    &step_id,
                                    Some("validate"),
                                    serde_json::json!({
                                        "tool": call.schema.name,
                                        "arguments": call.validated.to_json(),
                                        "approval": "auto",
                                        "effect": call.schema.effect.as_str(),
                                    }),
                                );
                            }
                            call
                        }
                        Err(rejection) => {
                            self.rejected_calls += 1;
                            if let Some(t) = self.trace.as_mut() {
                                t.emit(
                                    trace::ev::TOOL_CALL_REJECTED,
                                    &step_id,
                                    Some("validate"),
                                    serde_json::json!({
                                        "kind": rejection.kind_str(),
                                        "tool": raw_call.name,
                                        "reason": rejection.reason(),
                                    }),
                                );
                            }
                            let feedback = ToolResultMessage::from_text(
                                &raw_call.name,
                                false,
                                &format!(
                                    "tool call rejected [{}]: {}",
                                    rejection.kind_str(),
                                    rejection.reason()
                                ),
                            );
                            if let Err(LoopOutcome::Failed(m)) = self.commit_feedback(&feedback) {
                                return LoopOutcome::Failed(m);
                            }
                            continue;
                        }
                    };

                    let ValidatedCall {
                        schema,
                        validated,
                        tool,
                    } = call;
                    // -- hard limit before execution ---------------------
                    if self.tool_calls_executed >= self.limits.max_tool_calls {
                        return LoopOutcome::Limit(LimitKind::MaxToolCalls);
                    }

                    // -- execute (auto-approved deterministic registry) --
                    let tool_step = format!("tool-{}", self.tool_calls_executed);
                    if let Some(t) = self.trace.as_mut() {
                        t.emit(
                            trace::ev::TOOL_EXECUTION_STARTED,
                            &tool_step,
                            Some("execute"),
                            serde_json::json!({
                                "tool": schema.name,
                                "call_seq": self.tool_calls_executed + 1,
                            }),
                        );
                    }
                    let deadline = Instant::now() + self.limits.tool_timeout;
                    let invocation = ToolInvocation {
                        tool: Arc::clone(&tool),
                        args: validated,
                        run_id: self.run_id.clone(),
                        step_id: tool_step.clone(),
                        call_seq: self.tool_calls_executed as u64 + 1,
                        deadline,
                        cancel: control.clone(),
                        artifacts: Arc::clone(&self.artifacts),
                    };
                    let exec_start = Instant::now();
                    let outcome = invocation.execute();
                    let exec_ms = exec_start.elapsed().as_secs_f64() * 1e3;
                    self.total_tool_ms += exec_ms;
                    self.tool_calls_executed += 1;

                    let (ok, payload_text, failure_kind, artifact_ids) = match outcome {
                        Ok(out) => {
                            let text = out.payload.serialize_compact();
                            (true, text, None::<String>, out.artifact_ids)
                        }
                        Err(failure) => {
                            let kind = failure.kind.as_str().to_string();
                            (false, failure.message.clone(), Some(kind), Vec::new())
                        }
                    };
                    let new_artifacts = self.artifact_records_for(&artifact_ids);
                    if let Some(t) = self.trace.as_mut() {
                        let mut payload_mode = t.tool_payload_data(&payload_text);
                        let mut data = serde_json::json!({
                            "tool": schema.name,
                            "ok": ok,
                            "duration_ms": exec_ms,
                            "failure_kind": failure_kind,
                            "artifact_ids": artifact_ids,
                        });
                        if let (Some(obj), Some(dst)) =
                            (payload_mode.as_object_mut(), data.as_object_mut())
                        {
                            for (k, v) in obj {
                                dst.insert(k.clone(), v.clone());
                            }
                        }
                        t.emit(
                            trace::ev::TOOL_EXECUTION_FINISHED,
                            &tool_step,
                            Some("execute"),
                            data,
                        );
                        for record in &new_artifacts {
                            t.emit(
                                trace::ev::ARTIFACT_WRITTEN,
                                &tool_step,
                                None,
                                serde_json::to_value(record).unwrap_or(serde_json::Value::Null),
                            );
                        }
                    }

                    // cancellation AFTER execution: side effect happened;
                    // result stays OUT of the session (documented seam).
                    if control.is_cancelled() {
                        if let Some(t) = self.trace.as_mut() {
                            t.emit(
                                trace::ev::TOOL_RESULT_UNCOMMITTED,
                                &tool_step,
                                None,
                                serde_json::json!({
                                    "tool": schema.name,
                                    "reason": "cancelled after execution; external side effect already occurred",
                                }),
                            );
                        }
                        return LoopOutcome::Cancelled("after-tool-execution".to_string());
                    }

                    // -- reinject into the SAME session ------------------
                    let content_value: serde_json::Value = serde_json::from_str(&payload_text)
                        .unwrap_or_else(|_| serde_json::Value::String(payload_text.clone()));
                    let bounded = self.bound_payload(content_value);
                    let msg = ToolResultMessage {
                        call_name: &schema.name,
                        ok,
                        content_json: bounded.value,
                    };
                    let rendered = self.protocol.render_tool_result_message(&msg);
                    let commit_res = self
                        .engine
                        .commit_message(&rendered)
                        .context("committing tool result");
                    match commit_res {
                        Ok(span) => {
                            self.prompt_tokens_committed += span.1 - span.0;
                            self.ledger.push(CommittedMessage::ToolResult {
                                span,
                                call_name: schema.name.clone(),
                                ok,
                            });
                            if let Some(t) = self.trace.as_mut() {
                                t.emit(
                                    trace::ev::TOOL_RESULT_COMMITTED,
                                    &tool_step,
                                    None,
                                    serde_json::json!({
                                        "tool": schema.name,
                                        "ok": ok,
                                        "truncated": bounded.truncated,
                                        "committed_len": self.engine.committed_len(),
                                    }),
                                );
                                t.emit(
                                    trace::ev::SESSION_STATE_CHANGED,
                                    &tool_step,
                                    None,
                                    serde_json::json!({
                                        "role": "tool_result",
                                        "span": [span.0, span.1],
                                    }),
                                );
                            }
                        }
                        Err(e) => return LoopOutcome::Failed(format!("reinjection failed: {e:#}")),
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------

    fn validate_call(
        &self,
        name: &str,
        arguments_json: &str,
    ) -> std::result::Result<ValidatedCall, CallRejection> {
        use super::schema::{ArgumentError, ValidatedArguments};
        let Some(tool) = self.registry.get(name) else {
            return Err(CallRejection {
                kind: "unknown_tool",
                reason: format!(
                    "unknown tool `{name}`; available: {:?}",
                    self.registry.names().collect::<Vec<_>>()
                ),
            });
        };
        let schema = tool.schema();
        let validated = match ValidatedArguments::parse(&schema, arguments_json) {
            Ok(v) => v,
            Err(err) => {
                return Err(match err {
                    ArgumentError::MalformedJson(m) => CallRejection {
                        kind: "malformed_json",
                        reason: m.to_string(),
                    },
                    ArgumentError::Schema(errors) => CallRejection {
                        kind: "invalid_arguments",
                        reason: errors.to_string(),
                    },
                });
            }
        };
        Ok(ValidatedCall {
            schema,
            validated,
            tool: Arc::clone(tool),
        })
    }

    /// Feed a rejection/error payload back to the model as a protocol
    /// result message. Failure here is a hard infrastructure error.
    fn commit_feedback(&mut self, msg: &ToolResultMessage<'_>) -> Result<(), LoopOutcome> {
        let rendered = self.protocol.render_tool_result_message(msg);
        match self.engine.commit_message(&rendered) {
            Ok(_) => Ok(()),
            Err(e) => Err(LoopOutcome::Failed(format!("reinjection failed: {e:#}"))),
        }
    }

    fn bound_payload(&self, value: serde_json::Value) -> BoundedPayload {
        let serialized = serde_json::to_string(&value).unwrap_or_default();
        if serialized.len() <= self.limits.max_tool_result_bytes {
            return BoundedPayload {
                value,
                truncated: false,
            };
        }
        let cut: String = serialized
            .chars()
            .take(self.limits.max_tool_result_bytes)
            .collect();
        BoundedPayload {
            value: serde_json::json!({
                "truncated_result": true,
                "original_bytes": serialized.len(),
                "excerpt": cut,
            }),
            truncated: true,
        }
    }

    fn artifact_records_for(&self, ids: &[String]) -> Vec<super::artifact::ArtifactRecord> {
        let guard = match self.artifacts.lock() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        guard
            .records()
            .iter()
            .filter(|r| ids.contains(&r.artifact_id))
            .cloned()
            .collect()
    }

    fn summary(&self, status: RunStatus, final_text: Option<String>) -> AgentRunSummary {
        let artifacts = self
            .artifacts
            .lock()
            .map(|g| g.records().to_vec())
            .unwrap_or_default();
        AgentRunSummary {
            run_id: self.run_id.clone(),
            status,
            final_text,
            steps_executed: self.steps_executed,
            tool_calls_executed: self.tool_calls_executed,
            rejected_calls: self.rejected_calls,
            total_model_ms: self.total_model_ms,
            total_tool_ms: self.total_tool_ms,
            prefill_ms: self.prefill_ms,
            decode_ms: self.decode_ms,
            prompt_tokens_committed: self.prompt_tokens_committed,
            output_tokens: self.output_tokens,
            artifacts,
            ledger_roles: self.ledger.iter().map(|m| m.role_label()).collect(),
            trace_write_errors: self.trace.as_ref().map_or(0, |t| t.write_errors()),
        }
    }

    fn finish_failed(&mut self, _control: &CancelFlag, message: String) -> AgentRunSummary {
        if let Some(t) = self.trace.as_mut() {
            t.emit(
                trace::ev::RUN_FAILED,
                "finalize",
                None,
                serde_json::json!({ "error": message }),
            );
        }
        self.summary(RunStatus::Failed(message), None)
    }
}

enum LoopOutcome {
    Completed(String),
    Cancelled(String),
    Limit(LimitKind),
    Failed(String),
}

struct ValidatedCall {
    schema: super::schema::ToolSchema,
    validated: super::schema::ValidatedArguments,
    tool: Arc<dyn super::tool::Tool>,
}

struct CallRejection {
    kind: &'static str,
    reason: String,
}

impl CallRejection {
    fn kind_str(&self) -> &'static str {
        self.kind
    }
    fn reason(&self) -> &str {
        &self.reason
    }
}

struct BoundedPayload {
    value: serde_json::Value,
    truncated: bool,
}
