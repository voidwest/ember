//! The tool runtime (Tracks A2/A3/B): validated execution, structured
//! results, and the frozen registry.
//!
//! Pipeline invariant: raw model text never reaches a tool. Execution
//! takes [`ValidatedArguments`] only, and every outcome — success,
//! tool-reported failure, timeout, or panic — is a structured value the
//! loop can trace and reinject. Tool panics are caught (`catch_unwind`)
//! and become `ToolFailureKind::Panicked`; a tool can never take the
//! agent run down.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::artifact::ArtifactStore;
use super::schema::{ToolSchema, ValidatedArguments};

/// Cooperative cancellation handle. Reuses Ember's existing
/// [`crate::multimodal::session::GenerationControl`] (a shared atomic
/// flag checked at safe checkpoints) so agent runs and generation share
/// one cancellation mechanism.
pub type CancelFlag = crate::multimodal::session::GenerationControl;

/// Everything a tool may touch at execution time.
///
/// `artifacts` is shared because a timed-out tool's worker thread keeps
/// running detached; the store must stay reachable behind a mutex.
pub struct ToolContext<'a> {
    pub run_id: &'a str,
    pub step_id: &'a str,
    /// 1-based sequence of this call within the run.
    pub call_seq: u64,
    /// Hard deadline for this invocation (cooperative + watchdog).
    pub deadline: Instant,
    pub cancel: &'a CancelFlag,
    pub artifacts: &'a Mutex<ArtifactStore>,
}

impl ToolContext<'_> {
    /// Remaining time before this invocation's deadline expires.
    pub fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}

/// What a successful tool hands back. Text is the common case; JSON is
/// carried structurally so future renderers can format it per protocol
/// (and so Track W multimodal content parts have a seam to attach to).
#[derive(Debug, Clone)]
pub enum ToolPayload {
    Text(String),
    Json(serde_json::Value),
}

impl ToolPayload {
    /// Compact serialization used for tracing digests and size limits.
    pub fn serialize_compact(&self) -> String {
        match self {
            ToolPayload::Text(t) => t.clone(),
            ToolPayload::Json(v) => {
                serde_json::to_string(v).unwrap_or_else(|_| "\"<unserializable>\"".to_string())
            }
        }
    }

    pub fn byte_len(&self) -> usize {
        match self {
            ToolPayload::Text(t) => t.len(),
            ToolPayload::Json(v) => serde_json::to_vec(v).map(|b| b.len()).unwrap_or(0),
        }
    }
}

/// Successful execution output plus artifact references (Track A3).
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub payload: ToolPayload,
    pub artifact_ids: Vec<String>,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            payload: ToolPayload::Text(text.into()),
            artifact_ids: Vec::new(),
        }
    }

    pub fn json(value: serde_json::Value) -> Self {
        Self {
            payload: ToolPayload::Json(value),
            artifact_ids: Vec::new(),
        }
    }
}

/// Structured failure of one execution. This is NOT an infrastructure
/// error: it is data the loop traces and feeds back to the model.
#[derive(Debug, Clone)]
pub struct ToolFailure {
    pub kind: ToolFailureKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFailureKind {
    /// The tool ran and reported failure (bad key, failed lookup, ...).
    Execution,
    /// Watchdog deadline expired; the worker thread is abandoned.
    Timeout,
    /// The tool panicked; the panic was caught and contained.
    Panicked,
}

impl ToolFailureKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ToolFailureKind::Execution => "execution_failed",
            ToolFailureKind::Timeout => "timed_out",
            ToolFailureKind::Panicked => "panicked",
        }
    }
}

impl fmt::Display for ToolFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.as_str(), self.message)
    }
}

impl std::error::Error for ToolFailure {}

/// Outcome of one execution attempt.
pub type ToolOutcome = std::result::Result<ToolOutput, ToolFailure>;

/// A deterministic unit the agent can call. Implementations must be
/// Send + Sync ('static capture) so execution can run either inline or on
/// a watchdog worker thread for timeout enforcement. Synchronous by
/// design: Phase 1 does not introduce async into the runtime.
pub trait Tool: Send + Sync {
    fn schema(&self) -> ToolSchema;

    fn execute(&self, args: &ValidatedArguments, ctx: &ToolContext<'_>) -> ToolOutcome;
}

/// One fully-specified execution: everything needed to run a validated
/// call, including on a detached watchdog worker ('static by ownership).
pub struct ToolInvocation {
    pub tool: Arc<dyn Tool>,
    pub args: ValidatedArguments,
    pub run_id: String,
    pub step_id: String,
    /// 1-based sequence of this call within the run.
    pub call_seq: u64,
    /// Hard deadline for this invocation.
    pub deadline: Instant,
    pub cancel: CancelFlag,
    pub artifacts: Arc<Mutex<ArtifactStore>>,
}

impl ToolInvocation {
    fn context(&self) -> ToolContext<'_> {
        ToolContext {
            run_id: &self.run_id,
            step_id: &self.step_id,
            call_seq: self.call_seq,
            deadline: self.deadline,
            cancel: &self.cancel,
            artifacts: &self.artifacts,
        }
    }

    /// Execute the call (see [`execute_tool`] for the semantics).
    pub fn execute(&self) -> ToolOutcome {
        execute_tool(self)
    }
}

/// Execute one validated call.
///
/// - deadline <= now (or expiry while waiting): [`ToolFailureKind::Timeout`];
///   the tool body keeps running on a detached worker and its eventual
///   result is discarded — documented cooperative-preemption caveat.
/// - panics are contained and reported as [`ToolFailureKind::Panicked`].
/// - cancellation is NOT checked here on purpose: if the run is cancelled
///   while a tool executes, the side effect still happened and the caller
///   decides how to record it.
pub fn execute_tool(invocation: &ToolInvocation) -> ToolOutcome {
    let remaining = invocation
        .deadline
        .saturating_duration_since(Instant::now());
    let name = invocation.tool.schema().name;
    if remaining.is_zero() {
        return Err(ToolFailure {
            kind: ToolFailureKind::Timeout,
            message: format!("deadline already expired before `{name}` started"),
        });
    }

    // Fast path: run inline when no watchdog is needed (effectively
    // unbounded budget).
    if remaining >= Duration::from_secs(3600) {
        let ctx = invocation.context();
        return catch_panics(&name, || invocation.tool.execute(&invocation.args, &ctx));
    }

    // Watchdog path: bounded wait on a worker thread. Every captured value
    // is 'static + Send by construction (`Tool: Send + Sync`,
    // `ValidatedArguments: Send`, `CancelFlag: Send + Sync`,
    // `Arc<Mutex<ArtifactStore>>`); the store stays shared so a
    // late-detached worker cannot lose artifacts.
    let worker_invocation = ToolInvocation {
        tool: Arc::clone(&invocation.tool),
        args: invocation.args.clone(),
        run_id: invocation.run_id.clone(),
        step_id: invocation.step_id.clone(),
        call_seq: invocation.call_seq,
        deadline: invocation.deadline,
        cancel: invocation.cancel.clone(),
        artifacts: Arc::clone(&invocation.artifacts),
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name(format!("ember-tool-{name}"))
        .spawn(move || {
            let ctx = worker_invocation.context();
            let outcome = catch_panics(&worker_invocation.tool.schema().name, || {
                worker_invocation
                    .tool
                    .execute(&worker_invocation.args, &ctx)
            });
            let _ = tx.send(outcome);
        });
    match handle {
        Ok(_worker) => match rx.recv_timeout(remaining) {
            Ok(outcome) => outcome,
            Err(_) => {
                // Detach: the worker finishes whenever it finishes; its
                // result goes nowhere. This is the documented cost of
                // synchronous tools under a hard deadline.
                Err(ToolFailure {
                    kind: ToolFailureKind::Timeout,
                    message: format!("`{name}` exceeded its {}ms deadline", remaining.as_millis()),
                })
            }
        },
        Err(e) => Err(ToolFailure {
            kind: ToolFailureKind::Execution,
            message: format!("failed to spawn tool worker: {e}"),
        }),
    }
}

fn catch_panics(name: &str, f: impl FnOnce() -> ToolOutcome) -> ToolOutcome {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(outcome) => outcome,
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            Err(ToolFailure {
                kind: ToolFailureKind::Panicked,
                message: format!("tool `{name}` panicked: {message}"),
            })
        }
    }
}

/// Frozen ownership of the tools available to a run (Track B).
///
/// Registration happens exclusively through [`ToolRegistryBuilder`];
/// duplicate names are rejected at registration and re-checked at build.
/// A built registry has no mutators — it is immutable for the lifetime of
/// a run, and its schema snapshot is recorded in the trace provenance.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: std::collections::BTreeMap<String, Arc<dyn Tool>>,
}

#[derive(Debug)]
pub struct DuplicateTool {
    pub name: String,
}

impl fmt::Display for DuplicateTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "duplicate tool registration: `{}`", self.name)
    }
}

impl std::error::Error for DuplicateTool {}

/// Unknown-tool lookup. Fails closed by construction (`get` returns
/// `None`; the loop turns that into a structured rejection).
#[derive(Debug, Clone)]
pub struct UnknownTool {
    pub name: String,
}

impl fmt::Display for UnknownTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown tool `{}`", self.name)
    }
}

impl std::error::Error for UnknownTool {}

#[derive(Default)]
pub struct ToolRegistryBuilder {
    tools: std::collections::BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistryBuilder {
    /// Register one tool; duplicate names fail here already.
    pub fn register(mut self, tool: Arc<dyn Tool>) -> Result<Self, DuplicateTool> {
        let schema = tool.schema();
        if self.tools.contains_key(&schema.name) {
            return Err(DuplicateTool { name: schema.name });
        }
        self.tools.insert(schema.name.clone(), tool);
        Ok(self)
    }

    pub fn build(self) -> Result<ToolRegistry, DuplicateTool> {
        Ok(ToolRegistry { tools: self.tools })
    }
}

impl ToolRegistry {
    pub fn builder() -> ToolRegistryBuilder {
        ToolRegistryBuilder::default()
    }

    /// Empty registry: the model can only answer directly.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Schemas in stable (sorted) order — this exact snapshot goes into
    /// the trace provenance and the model prompt.
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|t| t.schema()).collect()
    }
}
