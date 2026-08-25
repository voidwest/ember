//! The agentic execution layer + research tracing subsystem (Phase 1).
//!
//! Ember acting, not merely generating: a small, rigorous, embeddable
//! runtime for
//!
//! ```text
//! model -> tool decision -> structured call -> validation -> execution
//!       -> result reinjection -> continuation -> final answer
//! ```
//!
//! with an auditable research trace over every step.
//!
//! Module map:
//!
//! - [`schema`] — tool argument schemas (JSON-Schema-compatible subset)
//! - [`tool`] — Tool trait, validated execution, frozen registry
//! - [`protocol`] — model-family tool-call codecs behind one boundary
//! - [`model`] — session engine seam (`ChatModelEngine`) + real impl
//! - [`trace`] — structured events/spans, crash-tolerant JSONL writer
//! - [`artifact`] — hashed artifact provenance records
//! - [`tools`] — deterministic built-in tools
//! - [`session`] — the explicit agent state machine with hard limits
//! - [`testkit`] — scripted model for hermetic tests
//!
//! Architectural invariant: this layer sits ABOVE inference. Nothing here
//! touches attention, KV kernels, tokenization primitives, encoders, or
//! tensor math; it drives the existing session/runtime abstractions only.

pub mod artifact;
pub mod ids;
pub mod inspect;
pub mod model;
pub mod protocol;
pub mod schema;
pub mod session;
pub mod testkit;
pub mod tool;
pub mod tools;
pub mod trace;

pub use artifact::{ArtifactRecord, ArtifactStore};
pub use inspect::{inspect_file, timeline, validate_trace_invariants, TraceSummary};
pub use model::{
    ChatModelEngine, GeneratedTurn, GenerationParams, LlamaChatModel, ModelIdentity, TurnStop,
};
pub use protocol::{
    AssistantAction, EmberJsonToolProtocol, LlamaToolProtocol, Qwen25ToolProtocol, RawToolCall,
    ToolCallProtocol, ToolResultMessage,
};
pub use schema::{
    ArgumentError, ArgumentErrors, JsonType, MalformedJson, ParamSchema, ToolEffect, ToolSchema,
    ValidatedArguments, ValidationError, ValidationErrorKind,
};
pub use session::{
    AgentConfig, AgentLimits, AgentRunSummary, AgentSession, CommittedMessage, LimitKind,
    RunResources, RunStatus,
};
pub use tool::{
    execute_tool, CancelFlag, DuplicateTool, Tool, ToolContext, ToolFailure, ToolFailureKind,
    ToolInvocation, ToolOutcome, ToolOutput, ToolPayload, ToolRegistry, ToolRegistryBuilder,
    UnknownTool,
};
pub use tools::{
    CalculatorTool, EchoTool, FailTool, LookupFixtureTool, ReadTextFileTool, SearchTextTool,
    SlowTool, WriteArtifactTool,
};
pub use trace::{parse_trace_file, ToolTraceMode, TraceConfig, TraceRecorder, TRACE_SCHEMA};
