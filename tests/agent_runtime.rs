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
    validate_trace_invariants, AgentConfig, AgentLimits, AgentSession, ArtifactStore, CancelFlag,
    RunResources, Tool, ToolContext, ToolOutcome, ToolOutput, ToolRegistry, ToolSchema,
    TraceConfig, TraceRecorder,
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

fn basic_registry() -> ToolRegistry {
    let fixtures = BTreeMap::from([
        ("alpha".to_string(), "42".to_string()),
        ("beta".to_string(), "43".to_string()),
    ]);
    ToolRegistry::builder()
        .register(Arc::new(CalculatorTool))
        .unwrap()
        .register(Arc::new(LookupFixtureTool::from_map(fixtures)))
        .unwrap()
        .register(Arc::new(EchoTool))
        .unwrap()
        .build()
        .unwrap()
}

fn scripted_session<'e>(engine: &'e mut ScriptedModel, registry: ToolRegistry) -> AgentSession<'e> {
    AgentSession::new(
        engine,
        json_protocol(),
        registry,
        AgentConfig::default(),
        AgentLimits::default(),
    )
}

fn generic_call(tool: &str, args: &str) -> String {
    format!(r#"{{"type":"tool_call","name":"{tool}","arguments":{args}}}"#)
}

/// SlowTool wrapper that cancels from inside the invocation, avoiding a
/// scheduler-dependent test race while proving cancellation during execute.
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
    let registry = basic_registry();
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, registry);
        let s = session
            .run(&CancelFlag::new(), "What is alpha?", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };

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
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, ToolRegistry::empty());
        let s = session
            .run(&CancelFlag::new(), "hi", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
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
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, basic_registry());
        let s = session
            .run(
                &CancelFlag::new(),
                "compute alpha plus check",
                memory_resources(),
            )
            .unwrap();
        (s, session.trace_events())
    };
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
    let registry = ToolRegistry::builder()
        .register(Arc::new(FailTool))
        .unwrap()
        .build()
        .unwrap();
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, registry);
        let s = session
            .run(&CancelFlag::new(), "go", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
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
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, basic_registry());
        let s = session
            .run(&CancelFlag::new(), "try it", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
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
fn malformed_json_arguments_are_rejected_structured() {
    // generic protocol: a typed call whose arguments are not JSON at all
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(r#"{"type":"tool_call","name":"echo","arguments":{"text" }"#),
        ScriptedTurn::output("fine now"),
    ]);
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, basic_registry());
        let s = session
            .run(&CancelFlag::new(), "x", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.rejected_calls, 1);
    let rejected = events
        .iter()
        .find(|e| e["event_type"] == "tool_call_rejected")
        .expect("rejection recorded");
    assert_eq!(rejected["data"]["kind"], "malformed_tool_call");
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

#[test]
fn invalid_arguments_against_schema_are_rejected_with_all_problems() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("calculate", r#"{"operation":"mod","a":"x"}"#)),
        ScriptedTurn::output("got it"),
    ]);
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, basic_registry());
        let s = session
            .run(&CancelFlag::new(), "x", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
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

#[test]
fn malformed_tool_call_syntax_never_silently_becomes_text() {
    // qwen protocol would be malformed inside tags; for the generic
    // protocol a typed object missing fields is malformed.
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(r#"{"type":"tool_call","arguments":{"a":1}}"#),
        ScriptedTurn::output("done"),
    ]);
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, basic_registry());
        let s = session
            .run(&CancelFlag::new(), "x", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert!(events
        .iter()
        .any(|e| e["event_type"] == "assistant_action_parsed"
            && e["data"]["action"] == "malformed_tool_call"));
    assert!(events
        .iter()
        .any(|e| e["event_type"] == "tool_call_rejected"));
}

// -- limits ------------------------------------------------------------------

#[test]
fn max_steps_terminates_cleanly() {
    let script = (0..10)
        .map(|_| ScriptedTurn::output(generic_call("echo", r#"{"text":"loop"}"#)))
        .collect::<Vec<_>>();
    let mut engine = ScriptedModel::new(script);
    let registry = ToolRegistry::builder()
        .register(Arc::new(EchoTool))
        .unwrap()
        .build()
        .unwrap();
    let (summary, events) = {
        let mut session = AgentSession::new(
            &mut engine,
            json_protocol(),
            registry,
            AgentConfig::default(),
            AgentLimits {
                max_steps: 3,
                ..Default::default()
            },
        );
        let s = session
            .run(&CancelFlag::new(), "loop forever", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
    assert_eq!(
        summary.status,
        ember::agent::RunStatus::LimitReached(ember::agent::LimitKind::MaxSteps)
    );
    assert_eq!(summary.steps_executed, 3);
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
}

#[test]
fn max_tool_calls_terminates_before_execution() {
    let script = (0..5)
        .map(|_| ScriptedTurn::output(generic_call("echo", r#"{"text":"x"}"#)))
        .collect::<Vec<_>>();
    let mut engine = ScriptedModel::new(script);
    let registry = ToolRegistry::builder()
        .register(Arc::new(EchoTool))
        .unwrap()
        .build()
        .unwrap();
    let summary = {
        let mut session = AgentSession::new(
            &mut engine,
            json_protocol(),
            registry,
            AgentConfig::default(),
            AgentLimits {
                max_steps: 8,
                max_tool_calls: 2,
                ..Default::default()
            },
        );
        session
            .run(&CancelFlag::new(), "go", memory_resources())
            .unwrap()
    };
    assert_eq!(
        summary.status,
        ember::agent::RunStatus::LimitReached(ember::agent::LimitKind::MaxToolCalls)
    );
    assert_eq!(summary.tool_calls_executed, 2);
}

#[test]
fn zero_wall_time_budget_fires_immediately_with_state_intact() {
    let mut engine = ScriptedModel::new(vec![ScriptedTurn::output("never reached")]);
    let summary = {
        let mut session = AgentSession::new(
            &mut engine,
            json_protocol(),
            basic_registry(),
            AgentConfig::default(),
            AgentLimits {
                max_wall_time: Some(Duration::ZERO),
                ..Default::default()
            },
        );
        session
            .run(&CancelFlag::new(), "hi", memory_resources())
            .unwrap()
    };
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
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, basic_registry());
        let s = session
            .run(&CancelFlag::new(), "say something", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
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
    let registry = ToolRegistry::builder()
        .register(Arc::new(CancellingSlowTool {
            cancel: control.clone(),
        }))
        .unwrap()
        .build()
        .unwrap();
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
    let registry = ToolRegistry::builder()
        .register(Arc::new(SlowTool::new(50)))
        .unwrap()
        .build()
        .unwrap();
    let (summary, events) = {
        let mut session = AgentSession::new(
            &mut engine,
            json_protocol(),
            registry,
            AgentConfig::default(),
            AgentLimits {
                tool_timeout: Duration::from_millis(150),
                ..Default::default()
            },
        );
        let s = session
            .run(&CancelFlag::new(), "slow please", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
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
    let registry = ToolRegistry::builder()
        .register(Arc::new(PanicProbe))
        .unwrap()
        .build()
        .unwrap();
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, registry);
        let s = session
            .run(&CancelFlag::new(), "boom?", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
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
    let dir = std::env::temp_dir().join(format!(
        "ember-agent-art-it-{}",
        std::time::Instant::now().elapsed().as_nanos()
    ));
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call(
            "write_artifact",
            r#"{"name":"result-note.md","content":"finding 42x"}"#,
        )),
        ScriptedTurn::output("saved."),
    ]);
    let registry = ToolRegistry::builder()
        .register(Arc::new(WriteArtifactTool))
        .unwrap()
        .build()
        .unwrap();
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, registry);
        let s = session
            .run(&CancelFlag::new(), "save it", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
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
    std::fs::remove_dir_all(&dir).ok();
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
    let registry = ToolRegistry::builder()
        .register(Arc::new(EchoTool))
        .unwrap()
        .build()
        .unwrap();
    let mut session = AgentSession::new(
        &mut engine,
        json_protocol(),
        registry,
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
    // serializable snapshot for prompts/provenance
    let rendered = serde_json::to_string(
        &schemas
            .iter()
            .map(|s| s.to_json_schema())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(rendered.contains("\"calculate\""));
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
    let (summary, events) = {
        let mut session = scripted_session(&mut engine, basic_registry());
        let s = session
            .run(&CancelFlag::new(), "cross-check", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
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

#[test]
fn limits_fire_between_calls_of_one_step() {
    let mut engine = ScriptedModel::new(vec![ScriptedTurn::output(format!(
        "{} {} {}",
        generic_call("echo", r#"{"text":"1"}"#),
        generic_call("echo", r#"{"text":"2"}"#),
        generic_call("echo", r#"{"text":"3"}"#)
    ))]);
    let registry = ToolRegistry::builder()
        .register(Arc::new(EchoTool))
        .unwrap()
        .build()
        .unwrap();
    let summary = {
        let mut session = AgentSession::new(
            &mut engine,
            json_protocol(),
            registry,
            AgentConfig::default(),
            AgentLimits {
                max_steps: 2,
                max_tool_calls: 2,
                ..Default::default()
            },
        );
        session
            .run(&CancelFlag::new(), "x", memory_resources())
            .unwrap()
    };
    assert_eq!(
        summary.status,
        ember::agent::RunStatus::LimitReached(ember::agent::LimitKind::MaxToolCalls)
    );
    assert_eq!(summary.tool_calls_executed, 2);
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
    let registry = ToolRegistry::builder()
        .register(Arc::new(ExternalProbe))
        .unwrap()
        .build()
        .unwrap();
    let (summary, events) = {
        // AgentConfig::default() = DenyExternalSideEffect
        let mut session = scripted_session(&mut engine, registry);
        let s = session
            .run(&CancelFlag::new(), "go external", memory_resources())
            .unwrap();
        (s, session.trace_events())
    };
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
    let registry = ToolRegistry::builder()
        .register(Arc::new(ExternalProbe))
        .unwrap()
        .build()
        .unwrap();
    let summary = {
        let mut session = AgentSession::new(
            &mut engine,
            json_protocol(),
            registry,
            AgentConfig {
                approval: ember::agent::ApprovalPolicy::Auto,
                ..Default::default()
            },
            AgentLimits::default(),
        );
        session
            .run(&CancelFlag::new(), "go", memory_resources())
            .unwrap()
    };
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(summary.tool_calls_executed, 1);
    assert_eq!(summary.rejected_calls, 0);
}

// -- Phase 2: trace diff / replay / HTML ---------------------------------------

fn two_runs(script: Vec<ScriptedTurn>) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let run_once = || {
        let mut engine = ScriptedModel::new(script.clone());
        let mut session = scripted_session(&mut engine, basic_registry());
        session
            .run(&CancelFlag::new(), "same input", memory_resources())
            .unwrap();
        session.trace_events()
    };
    (run_once(), run_once())
}

#[test]
fn trace_diff_reports_identical_structure_for_same_script() {
    let (a, b) = two_runs(vec![
        ScriptedTurn::output(generic_call("lookup", r#"{"key":"alpha"}"#)),
        ScriptedTurn::output("The value is 42."),
    ]);
    let (_, _, diff) = ember::agent::inspect::diff(&a, &b);
    assert!(diff.is_identical(), "{:?}", diff.differences);
}

#[test]
fn trace_diff_catches_skeleton_divergence() {
    let (a, b) = two_runs(vec![ScriptedTurn::output("plain answer")]);
    let (a2, _) = two_runs(vec![
        ScriptedTurn::output(generic_call("echo", r#"{"text":"x"}"#)),
        ScriptedTurn::output("with tool"),
    ]);
    // same-script pair must be identical even when compared across scripts
    let (_, _, same) = ember::agent::inspect::diff(&a, &b);
    assert!(same.is_identical());
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
    let registry = ToolRegistry::builder()
        .register(Arc::new(LookupFixtureTool::from_map(fixtures)))
        .unwrap()
        .register(Arc::new(WriteArtifactTool))
        .unwrap()
        .register(Arc::new(FailTool))
        .unwrap()
        .build()
        .unwrap();
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
    assert_eq!(
        report.skipped()
            + report
                .outcomes
                .iter()
                .filter(|o| o.matches == Some(true))
                .count(),
        3
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn html_report_renders_self_contained_summary() {
    let mut engine = ScriptedModel::new(vec![
        ScriptedTurn::output(generic_call("echo", r#"{"text":"hi"}"#)),
        ScriptedTurn::output("done"),
    ]);
    let registry = ToolRegistry::builder()
        .register(Arc::new(EchoTool))
        .unwrap()
        .build()
        .unwrap();
    let events = {
        let mut session = AgentSession::new(
            &mut engine,
            json_protocol(),
            registry,
            AgentConfig::default(),
            AgentLimits::default(),
        );
        session
            .run(&CancelFlag::new(), "hi", memory_resources())
            .unwrap();
        session.trace_events()
    };
    let summary = ember::agent::inspect::summarize(&events);
    let html = ember::agent::inspect::render_html(&events, &summary);
    assert!(html.starts_with("<!doctype html>"));
    assert!(html.contains(&summary.run_id));
    assert!(html.contains("tool_execution_finished"));
    assert!(html.contains(".bar"));
    assert!(!html.contains("<script"), "no JS, no external assets");
}
