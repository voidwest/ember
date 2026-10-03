//! Hermetic agent-runtime tests (Tracks S/T): the full loop exercised
//! through a scripted model — no GGUF, no network, no timing races in the
//! core paths. Every test validates the trace with
//! [`ember::agent::validate_trace_invariants`].

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ember::agent::protocol::EmberJsonToolProtocol;
use ember::agent::testkit::{ScriptedModel, ScriptedTurn};
use ember::agent::tools::{
    CalculatorTool, EchoTool, FailTool, LookupFixtureTool, SlowTool, WriteArtifactTool,
};
use ember::agent::{
    validate_trace_invariants, AgentConfig, AgentLimits, AgentRunSummary, AgentSession,
    ArtifactStore, CancelFlag, RunResources, Tool, ToolContext, ToolOutcome, ToolOutput,
    ToolRegistry, ToolSchema, TraceConfig, TraceRecorder,
};

// -- helpers ---------------------------------------------------------------

fn memory_resources() -> RunResources {
    RunResources {
        trace: Some(TraceRecorder::open(TraceConfig::default(), "pending").expect("trace")),
        artifacts: Arc::new(Mutex::new(
            ArtifactStore::open(std::env::temp_dir(), "pending").expect("artifact dir"),
        )),
    }
}

fn json_protocol() -> Arc<dyn ember::agent::ToolCallProtocol> {
    Arc::new(EmberJsonToolProtocol::default())
}

fn registry_of(tools: Vec<Arc<dyn Tool>>) -> ToolRegistry {
    tools
        .into_iter()
        .fold(ToolRegistry::builder(), |builder, tool| {
            builder.register(tool).unwrap()
        })
        .build()
        .unwrap()
}

fn basic_registry() -> ToolRegistry {
    let fixtures = BTreeMap::from([
        ("alpha".to_string(), "42".to_string()),
        ("beta".to_string(), "43".to_string()),
    ]);
    registry_of(vec![
        Arc::new(CalculatorTool),
        Arc::new(LookupFixtureTool::from_map(fixtures)),
        Arc::new(EchoTool),
    ])
}

/// One task through a fresh session with in-memory resources; returns the
/// summary and the trace events.
fn run_with(
    engine: &mut ScriptedModel,
    registry: ToolRegistry,
    config: AgentConfig,
    limits: AgentLimits,
    prompt: &str,
) -> (AgentRunSummary, Vec<serde_json::Value>) {
    let mut session = AgentSession::new(engine, json_protocol(), registry, config, limits);
    let summary = session
        .run(&CancelFlag::new(), prompt, memory_resources())
        .unwrap();
    (summary, session.trace_events())
}

fn run(
    engine: &mut ScriptedModel,
    registry: ToolRegistry,
    prompt: &str,
) -> (AgentRunSummary, Vec<serde_json::Value>) {
    run_with(
        engine,
        registry,
        AgentConfig::default(),
        AgentLimits::default(),
        prompt,
    )
}

fn generic_call(tool: &str, args: &str) -> String {
    format!(r#"{{"type":"tool_call","name":"{tool}","arguments":{args}}}"#)
}

/// Tool fixture that cancels from inside the invocation, avoiding a
/// scheduler-dependent race while proving post-execution cancellation keeps
/// a completed side effect out of the committed conversation.
struct CancellingSlowTool {
    cancel: CancelFlag,
}

impl Tool for CancellingSlowTool {
    fn schema(&self) -> ToolSchema {
        SlowTool::new(20).schema()
    }

    fn execute(
        &self,
        _args: &ember::agent::ValidatedArguments,
        _ctx: &ToolContext<'_>,
    ) -> ToolOutcome {
        self.cancel.cancel();
        Ok(ToolOutput::json(serde_json::json!({ "side_effect": true })))
    }
}

/// A deliberately panicking tool (panic containment proof).
struct PanicProbe;

impl Tool for PanicProbe {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("panic_probe", "always panics").effect(ember::agent::ToolEffect::ReadOnly)
    }

    fn execute(
        &self,
        _args: &ember::agent::ValidatedArguments,
        _ctx: &ToolContext<'_>,
    ) -> ToolOutcome {
        panic!("boom from tool");
    }
}

// -- Track T: the mandatory scripted one-tool round trip --------------------

#[test]
fn scripted_one_tool_round_trip_is_exact() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("lookup", r#"{"key":"alpha"}"#)),
        ScriptedTurn::output("The value is 42."),
    ]);
    let (summary, events) = run(&mut engine, basic_registry(), "What is alpha?");

    // exact final answer
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.final_text.as_deref(), Some("The value is 42."));
    // exactly one tool call, with the exact arguments
    assert_eq!(summary.tool_calls_executed, 1);
    let committed = &engine.committed_messages;
    let tool_result = committed
        .iter()
        .find(|(role, text)| role == "message" && text.contains(r#""type":"tool_result""#))
        .map(|(_, t)| t.clone())
        .expect("tool result reinjected");
    assert!(tool_result.contains(r#""name":"lookup""#));
    assert!(tool_result.contains(r#""value":"42""#), "{tool_result}");
    assert!(tool_result.contains(r#""ok":true"#));
    // ledger shape: system, user, assistant_tool_call, tool_result, assistant_final
    assert_eq!(
        summary.ledger_roles,
        vec![
            "system",
            "user",
            "assistant_tool_call",
            "tool_result",
            "assistant_final"
        ]
    );
    // trace complete and ordered
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
    assert_eq!(events[0]["event_type"], "run_started");
    assert_eq!(events.last().unwrap()["event_type"], "run_completed");
    assert_eq!(events[0]["run_id"], events.last().unwrap()["run_id"]);
}

#[test]
fn final_answer_without_tools_completes() {
    let mut engine = ScriptedModel::new(vec![ScriptedTurn::output("Just an answer.")]);
    let (summary, events) = run(&mut engine, ToolRegistry::empty(), "hi");
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.tool_calls_executed, 0);
    assert_eq!(summary.final_text.as_deref(), Some("Just an answer."));
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

#[test]
fn multi_step_sequential_tool_calls_work() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("lookup", r#"{"key":"alpha"}"#)),
        ScriptedTurn::output(generic_call(
            "calculate",
            r#"{"operation":"add","a":40,"b":2}"#,
        )),
        ScriptedTurn::output("alpha is 42; confirmed by calculation."),
    ]);
    let (summary, events) = run(&mut engine, basic_registry(), "compute alpha plus check");
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.steps_executed, 3);
    assert_eq!(summary.tool_calls_executed, 2);
    let committed = &engine.committed_messages;
    assert!(committed.iter().any(|(_, t)| t.contains(r#""value":"42""#)));
    assert!(committed.iter().any(|(_, t)| t.contains(r#""result":42"#)));
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

// -- failure paths -----------------------------------------------------------

#[test]
fn tool_failure_is_structured_and_recoverable() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("fail", r#"{"message":"kaput"}"#)),
        ScriptedTurn::output("Recovered."),
    ]);
    let (summary, events) = run(&mut engine, registry_of(vec![Arc::new(FailTool)]), "go");
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.final_text.as_deref(), Some("Recovered."));
    // the error payload entered the session marked ok=false
    let committed = &engine.committed_messages;
    let feedback = committed
        .iter()
        .find(|(_, t)| t.contains("kaput") && t.contains("tool_result"))
        .expect("failure fed back");
    assert!(feedback.1.contains(r#""ok":false"#), "{}", feedback.1);

    assert!(events
        .iter()
        .any(|e| e["event_type"] == "tool_execution_finished" && e["data"]["ok"] == false));
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

#[test]
fn unknown_tool_fails_closed_but_run_recovers() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("does_not_exist", r#"{}"#)),
        ScriptedTurn::output("Understood; no such tool."),
    ]);
    let (summary, events) = run(&mut engine, basic_registry(), "try it");
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.rejected_calls, 1);
    assert_eq!(summary.tool_calls_executed, 0);

    let rejected = events
        .iter()
        .find(|e| e["event_type"] == "tool_call_rejected")
        .expect("rejection recorded");
    assert_eq!(rejected["data"]["kind"], "unknown_tool");
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

#[test]
fn malformed_tool_calls_are_rejected_structured_never_silent_text() {
    // generic protocol: a typed call whose arguments are not JSON at all,
    // and a typed object missing its name field
    for raw in [
        r#"{"type":"tool_call","name":"echo","arguments":{"text" }"#,
        r#"{"type":"tool_call","arguments":{"a":1}}"#,
    ] {
        let mut engine = ScriptedModel::new(vec![
            ScriptedTurn::output(raw),
            ScriptedTurn::output("fine now"),
        ]);
        let (summary, events) = run(&mut engine, basic_registry(), "x");
        assert_eq!(summary.status, ember::agent::RunStatus::Completed, "{raw}");
        assert_eq!(summary.rejected_calls, 1, "{raw}");
        assert!(
            events
                .iter()
                .any(|e| e["event_type"] == "assistant_action_parsed"
                    && e["data"]["action"] == "malformed_tool_call"),
            "{raw}"
        );
        let rejected = events
            .iter()
            .find(|e| e["event_type"] == "tool_call_rejected")
            .expect("rejection recorded");
        assert_eq!(rejected["data"]["kind"], "malformed_tool_call", "{raw}");
        assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
    }
}

#[test]
fn invalid_arguments_against_schema_are_rejected_with_all_problems() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("calculate", r#"{"operation":"mod","a":"x"}"#)),
        ScriptedTurn::output("got it"),
    ]);
    let (summary, events) = run(&mut engine, basic_registry(), "x");
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    let rejected = events
        .iter()
        .find(|e| e["event_type"] == "tool_call_rejected")
        .expect("rejection recorded");
    assert_eq!(rejected["data"]["kind"], "invalid_arguments");
    // reason carries BOTH violations (enum + wrong type)
    let reason = rejected["data"]["reason"].as_str().unwrap();
    assert!(reason.contains("expected one of"));
    assert!(reason.contains("expected number"));
}

// -- limits ------------------------------------------------------------------

#[test]
fn max_steps_terminates_cleanly() {
    let script = (0..10)
        .map(|_| ScriptedTurn::output(generic_call("echo", r#"{"text":"loop"}"#)))
        .collect::<Vec<_>>();
    let mut engine = ScriptedModel::new(script);
    let (summary, events) = run_with(
        &mut engine,
        registry_of(vec![Arc::new(EchoTool)]),
        AgentConfig::default(),
        AgentLimits {
            max_steps: 3,
            ..Default::default()
        },
        "loop forever",
    );
    assert_eq!(
        summary.status,
        ember::agent::RunStatus::LimitReached(ember::agent::LimitKind::MaxSteps)
    );
    assert_eq!(summary.steps_executed, 3);
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

#[test]
fn max_tool_calls_fires_across_steps_and_between_calls_of_one_step() {
    let echo = |text: &str| generic_call("echo", &format!(r#"{{"text":"{text}"}}"#));
    for (script, max_steps) in [
        // one call per step: the limit fires before the third execution
        (
            (0..5)
                .map(|_| ScriptedTurn::output(echo("x")))
                .collect::<Vec<_>>(),
            8,
        ),
        // three calls in one step: the limit fires between calls
        (
            vec![ScriptedTurn::output(format!(
                "{} {} {}",
                echo("1"),
                echo("2"),
                echo("3")
            ))],
            2,
        ),
    ] {
        let mut engine = ScriptedModel::new(script);
        let (summary, _) = run_with(
            &mut engine,
            registry_of(vec![Arc::new(EchoTool)]),
            AgentConfig::default(),
            AgentLimits {
                max_steps,
                max_tool_calls: 2,
                ..Default::default()
            },
            "go",
        );
        assert_eq!(
            summary.status,
            ember::agent::RunStatus::LimitReached(ember::agent::LimitKind::MaxToolCalls),
            "max_steps={max_steps}"
        );
        assert_eq!(summary.tool_calls_executed, 2, "max_steps={max_steps}");
    }
}

#[test]
fn zero_wall_time_budget_fires_immediately_with_state_intact() {
    let mut engine = ScriptedModel::new(vec![ScriptedTurn::output("never reached")]);
    let (summary, _) = run_with(
        &mut engine,
        basic_registry(),
        AgentConfig::default(),
        AgentLimits {
            max_wall_time: Some(Duration::ZERO),
            ..Default::default()
        },
        "hi",
    );
    assert_eq!(
        summary.status,
        ember::agent::RunStatus::LimitReached(ember::agent::LimitKind::WallTime)
    );
    // user turn is still validly committed
    assert_eq!(summary.ledger_roles.first(), Some(&"system"));
}

// -- cancellation ------------------------------------------------------------

#[test]
fn cancellation_mid_generation_commits_nothing() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output("partial").cancel_after(0),
        ScriptedTurn::output("unreached"),
    ]);
    let before = engine.committed_messages.len();
    let (summary, events) = run(&mut engine, basic_registry(), "say something");
    assert_eq!(summary.status, ember::agent::RunStatus::Cancelled);
    assert!(summary.final_text.is_none());
    // ledger: system + user only; no assistant content committed
    assert_eq!(summary.ledger_roles, vec!["system", "user"]);
    // engine saw preamble + speculative prefix but NO assistant turn
    let after = engine.committed_messages.len();
    assert_eq!(after, before + 3); // system, user, prefix
    assert!(!engine
        .committed_messages
        .iter()
        .any(|(role, _)| role == "assistant_turn"));
    assert_eq!(events.last().unwrap()["event_type"], "run_cancelled");
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

#[test]
fn cancellation_during_tool_execution_keeps_side_effect_visible_and_session_clean() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("slow", r#"{"milliseconds":400}"#)),
        ScriptedTurn::output("unreached"),
    ]);
    let control = CancelFlag::new();
    let registry = registry_of(vec![Arc::new(CancellingSlowTool {
        cancel: control.clone(),
    })]);
    let (summary, events) = {
        let mut session = AgentSession::new(
            &mut engine,
            json_protocol(),
            registry,
            AgentConfig::default(),
            AgentLimits {
                tool_timeout: Duration::from_secs(10),
                ..Default::default()
            },
        );
        let s = session.run(&control, "sleep", memory_resources()).unwrap();
        (s, session.trace_events())
    };
    assert_eq!(summary.status, ember::agent::RunStatus::Cancelled);
    assert_eq!(summary.tool_calls_executed, 1);
    // the result did NOT enter the conversation
    assert!(!engine
        .committed_messages
        .iter()
        .any(|(_, t)| t.contains(r#""type":"tool_result""#)));
    assert!(events.iter().any(|e| {
        e["event_type"] == "tool_execution_finished"
            && e["data"]["payload_excerpt"] == r#"{"side_effect":true}"#
    }));
    assert!(events
        .iter()
        .any(|e| e["event_type"] == "tool_result_uncommitted"));
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

// -- timeouts / panics ---------------------------------------------------------

#[test]
fn tool_timeout_is_structured_and_run_continues() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("slow", r#"{"milliseconds":2000}"#)),
        ScriptedTurn::output("moved on"),
    ]);
    let (summary, events) = run_with(
        &mut engine,
        registry_of(vec![Arc::new(SlowTool::new(50))]),
        AgentConfig::default(),
        AgentLimits {
            tool_timeout: Duration::from_millis(150),
            ..Default::default()
        },
        "slow please",
    );
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    let finished = events
        .iter()
        .find(|e| e["event_type"] == "tool_execution_finished")
        .expect("execution recorded");
    assert_eq!(finished["data"]["failure_kind"], "timed_out");
    // The timeout payload was fed back to the model verbatim. Either
    // reporter is legitimate: the watchdog ("exceeded its ...ms deadline")
    // or the tool's own cooperative deadline check ("hit its deadline") —
    // under heavy host load the cooperative path can win the race.
    assert!(
        engine
            .committed_messages
            .iter()
            .any(|(_, t)| t.contains("tool_result") && t.contains("deadline")),
        "timeout feedback must be reinjected"
    );
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

#[test]
fn tool_panic_is_contained_as_a_structured_failure() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("panic_probe", "{}")),
        ScriptedTurn::output("survived"),
    ]);
    let (summary, events) = run(
        &mut engine,
        registry_of(vec![Arc::new(PanicProbe)]),
        "boom?",
    );
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.final_text.as_deref(), Some("survived"));
    let finished = events
        .iter()
        .find(|e| e["event_type"] == "tool_execution_finished")
        .expect("recorded");
    assert_eq!(finished["data"]["failure_kind"], "panicked");
}

// -- artifacts + trace persistence -------------------------------------------

#[test]
fn artifact_writes_are_hashed_and_traced() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call(
            "write_artifact",
            r#"{"name":"result-note.md","content":"finding 42x"}"#,
        )),
        ScriptedTurn::output("saved."),
    ]);
    let (summary, events) = run(
        &mut engine,
        registry_of(vec![Arc::new(WriteArtifactTool)]),
        "save it",
    );
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.artifacts.len(), 1);
    let artifact = &summary.artifacts[0];
    assert_eq!(
        artifact.sha256,
        ember::extraction::sha256_bytes(b"finding 42x")
    );
    assert_eq!(artifact.size_bytes, 11);
    assert!(artifact.path.is_file());
    let written = events
        .iter()
        .find(|e| e["event_type"] == "artifact_written")
        .expect("artifact event");
    assert_eq!(written["data"]["producer_tool"], "write_artifact");
    assert_eq!(written["data"]["step_id"], "tool-0");
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

#[test]
fn jsonl_trace_file_is_incremental_and_prefix_readable() {
    let dir = std::env::temp_dir().join(format!(
        "ember-agent-trace-it-{}",
        std::time::Instant::now().elapsed().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("run.jsonl");

    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("echo", r#"{"text":"hi"}"#)),
        ScriptedTurn::output("done"),
    ]);
    let mut session = AgentSession::new(
        &mut engine,
        json_protocol(),
        registry_of(vec![Arc::new(EchoTool)]),
        AgentConfig::default(),
        AgentLimits::default(),
    );
    // mid-run snapshot: file must already contain the events so far
    let resources = RunResources {
        trace: Some(
            TraceRecorder::open(
                TraceConfig {
                    output_path: Some(path.clone()),
                    ..Default::default()
                },
                "pending",
            )
            .unwrap(),
        ),
        artifacts: Arc::new(Mutex::new(ArtifactStore::open(&dir, "pending").unwrap())),
    };
    let control = CancelFlag::new();
    let _ = session.run(&control, "hello", resources).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(raw.lines().count() > 5, "events flushed incrementally");

    // torn-file simulation: chop the last line in half; the prefix parses
    let mut lines: Vec<&str> = raw.lines().collect();
    assert!(!lines.is_empty());
    let last = lines.pop().unwrap();
    let cut = &last[..last.len() / 2];
    let torn = format!(
        "{}{}\n{}",
        lines.join("\n"),
        if lines.is_empty() { "" } else { "\n" },
        cut
    );
    std::fs::write(&path, torn).unwrap();
    let (events, skipped) = ember::agent::parse_trace_file(&path).unwrap();
    assert!(!skipped.is_empty(), "torn line reported");
    assert_eq!(
        validate_trace_invariants(&events).len(),
        1,
        "only the missing terminal event"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// -- registry ------------------------------------------------------------------

#[test]
fn duplicate_registration_rejected_at_builder() {
    let err = ToolRegistry::builder()
        .register(Arc::new(EchoTool))
        .unwrap()
        .register(Arc::new(EchoTool))
        .err()
        .expect("duplicate rejected");
    assert_eq!(err.name, "echo");
}

#[test]
fn registry_enumerates_schemas_stably() {
    let schemas = basic_registry().schemas();
    let names: Vec<_> = schemas.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["calculate", "echo", "lookup"]);
}

// -- Phase 2: multi-call steps ------------------------------------------------

#[test]
fn one_step_can_request_multiple_tools_executed_in_order() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(format!(
            "{} {}",
            generic_call("lookup", r#"{"key":"alpha"}"#),
            generic_call("calculate", r#"{"operation":"multiply","a":6,"b":7}"#)
        )),
        ScriptedTurn::output("alpha=42 and 6x7=42; consistent."),
    ]);
    let (summary, events) = run(&mut engine, basic_registry(), "cross-check");
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.tool_calls_executed, 2);
    assert_eq!(summary.steps_executed, 2); // two model turns total
                                           // BOTH results entered the conversation, in request order
    let committed: Vec<&str> = engine
        .committed_messages
        .iter()
        .filter(|(r, t)| r == "message" && t.contains("tool_result"))
        .map(|(_, t)| t.as_str())
        .collect();
    assert_eq!(committed.len(), 2);
    assert!(committed[0].contains(r#""value":"42""#));
    assert!(committed[1].contains(r#""result":42"#));
    // per-step parse event records the count
    assert!(events
        .iter()
        .any(|e| e["event_type"] == "assistant_action_parsed"
            && e["data"]["action"] == "tool_calls"
            && e["data"]["count"] == 2));
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

// -- Phase 2: approval gating (Track H) ----------------------------------------

/// Declares ExternalSideEffect so policies have something to bite on.
struct ExternalProbe;

impl Tool for ExternalProbe {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("external_probe", "declares external effects")
            .effect(ember::agent::ToolEffect::ExternalSideEffect)
    }

    fn execute(
        &self,
        _args: &ember::agent::ValidatedArguments,
        _ctx: &ToolContext<'_>,
    ) -> ToolOutcome {
        Ok(ember::agent::ToolOutput::text("side effect done"))
    }
}

#[test]
fn default_policy_denies_declared_external_effects_and_the_model_recovers() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("external_probe", "{}")),
        ScriptedTurn::output("understood; staying local."),
    ]);
    // AgentConfig::default() = DenyExternalSideEffect
    let (summary, events) = run(
        &mut engine,
        registry_of(vec![Arc::new(ExternalProbe)]),
        "go external",
    );
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.rejected_calls, 1);
    assert_eq!(summary.tool_calls_executed, 0, "nothing executed");
    let rejected = events
        .iter()
        .find(|e| e["event_type"] == "tool_call_rejected")
        .expect("rejection recorded");
    assert_eq!(rejected["data"]["kind"], "denied_by_policy");
    // denial was fed back to the model
    assert!(engine
        .committed_messages
        .iter()
        .any(|(_, t)| t.contains("denied by approval policy")));
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

#[test]
fn auto_policy_executes_external_declaring_tools() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("external_probe", "{}")),
        ScriptedTurn::output("done."),
    ]);
    let (summary, _) = run_with(
        &mut engine,
        registry_of(vec![Arc::new(ExternalProbe)]),
        AgentConfig {
            approval: ember::agent::ApprovalPolicy::Auto,
            ..Default::default()
        },
        AgentLimits::default(),
        "go",
    );
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.tool_calls_executed, 1);
    assert_eq!(summary.rejected_calls, 0);
}

// -- Phase 2: trace diff / replay / HTML ---------------------------------------

fn two_runs(script: Vec<ScriptedTurn>) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let run_once = || {
        let mut engine = ScriptedModel::new(script.clone());
        run(&mut engine, basic_registry(), "same input").1
    };
    (run_once(), run_once())
}

#[test]
fn trace_diff_catches_skeleton_divergence() {
    let (a, b) = two_runs(vec![ScriptedTurn::output("plain answer")]);
    let (a2, b2) = two_runs(vec![
        ScriptedTurn::output(generic_call("lookup", r#"{"key":"alpha"}"#)),
        ScriptedTurn::output("The value is 42."),
    ]);
    // same-script pairs must be identical, with and without a tool call
    let (_, _, same) = ember::agent::inspect::diff(&a, &b);
    assert!(same.is_identical(), "{:?}", same.differences);
    let (_, _, same_with_tool) = ember::agent::inspect::diff(&a2, &b2);
    assert!(
        same_with_tool.is_identical(),
        "{:?}",
        same_with_tool.differences
    );
    let (_, _, different) = ember::agent::inspect::diff(&a, &a2);
    assert!(!different.is_identical());
    assert!(different
        .differences
        .iter()
        .any(|d| d.contains("event skeleton diverges")));
}

#[test]
fn replay_reexecutes_recorded_calls_and_verifies_digests() {
    use ember::agent::inspect::replay;
    let dir = std::env::temp_dir().join(format!(
        "ember-replay-it-{}",
        std::time::Instant::now().elapsed().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let script = vec![
        ScriptedTurn::output(format!(
            "{} {} {}",
            generic_call("lookup", r#"{"key":"alpha"}"#),
            generic_call(
                "write_artifact",
                r#"{"name":"note.md","content":"replay me"}"#
            ),
            generic_call("fail", r#"{"message":"expected failure"}"#)
        )),
        ScriptedTurn::output("done"),
    ];
    let fixtures = std::collections::BTreeMap::from([("alpha".to_string(), "42".to_string())]);
    let registry = registry_of(vec![
        Arc::new(LookupFixtureTool::from_map(fixtures)),
        Arc::new(WriteArtifactTool),
        Arc::new(FailTool),
    ]);
    let events = {
        let mut engine = ScriptedModel::new(script);
        let resources = RunResources {
            trace: Some(TraceRecorder::open(TraceConfig::default(), "pending").unwrap()),
            artifacts: Arc::new(Mutex::new(ArtifactStore::open(&dir, "pending").unwrap())),
        };
        let mut session = AgentSession::new(
            &mut engine,
            json_protocol(),
            registry.clone(),
            AgentConfig::default(),
            AgentLimits::default(),
        );
        session.run(&CancelFlag::new(), "go", resources).unwrap();
        session.trace_events()
    };
    let report = replay(&events, &registry).unwrap();
    assert_eq!(report.outcomes.len(), 3);
    // lookup + write_artifact verify (artifact ids/paths excluded from the
    // stable digest); the failed call is skipped, not counted as mismatch
    assert_eq!(
        report
            .outcomes
            .iter()
            .filter(|o| o.matches == Some(true))
            .count(),
        2,
        "{:?}",
        report.outcomes
    );
    assert_eq!(report.skipped(), 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn html_report_renders_self_contained_summary() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("echo", r#"{"text":"hi"}"#)),
        ScriptedTurn::output("done"),
    ]);
    let (_, events) = run(&mut engine, registry_of(vec![Arc::new(EchoTool)]), "hi");
    let summary = ember::agent::inspect::summarize(&events);
    let html = ember::agent::inspect::render_html(&events, &summary);
    assert!(html.starts_with("<!doctype html>"));
    assert!(html.contains(&summary.run_id));
    assert!(html.contains("tool_execution_finished"));
    assert!(html.contains(".bar"));
    assert!(!html.contains("<script"), "no JS, no external assets");
}
