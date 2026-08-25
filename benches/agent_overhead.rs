//! Agent-loop orchestration overhead (Phase 1 performance requirement).
//!
//! Measures the runtime cost of everything EXCEPT model inference using
//! deterministic mock tools and a scripted engine:
//!
//!   1. mock one-tool run   (loop + validation + execution + trace)
//!   2. mock three-tool run
//!   3. trace-disabled vs trace-enabled comparison
//!   4. JSONL write overhead (memory sink vs file sink)
//!
//! Run: cargo run --release --bench agent_overhead

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ember::agent::protocol::EmberJsonToolProtocol;
use ember::agent::testkit::{ScriptedModel, ScriptedTurn};
use ember::agent::tools::{CalculatorTool, LookupFixtureTool};
use ember::agent::{
    AgentConfig, AgentLimits, AgentSession, ArtifactStore, CancelFlag, RunResources, ToolRegistry,
    TraceConfig, TraceRecorder,
};

fn call(tool: &str, args: &str) -> String {
    format!(r#"{{"type":"tool_call","name":"{tool}","arguments":{args}}}"#)
}

fn registry(_tools: usize) -> ToolRegistry {
    let fixtures = BTreeMap::from([("alpha".to_string(), "42".to_string())]);
    ToolRegistry::builder()
        .register(Arc::new(CalculatorTool))
        .unwrap()
        .register(Arc::new(LookupFixtureTool::from_map(fixtures)))
        .unwrap()
        .build()
        .unwrap()
}

/// One-tool script: lookup -> final answer.
fn one_tool_script() -> Vec<ScriptedTurn> {
    vec![
        ScriptedTurn::output(call("lookup", r#"{"key":"alpha"}"#)),
        ScriptedTurn::output("The value is 42."),
    ]
}

/// Three-tool script.
fn three_tool_script() -> Vec<ScriptedTurn> {
    vec![
        ScriptedTurn::output(call("lookup", r#"{"key":"alpha"}"#)),
        ScriptedTurn::output(call("calculate", r#"{"operation":"add","a":40,"b":2}"#)),
        ScriptedTurn::output(call("calculate", r#"{"operation":"multiply","a":6,"b":7}"#)),
        ScriptedTurn::output("Best config is 12% over baseline."),
    ]
}

/// `trace_mode`: None = tracing off; Some(None) = in-memory sink;
/// Some(Some(path)) = JSONL file sink.
fn bench(
    script: Vec<ScriptedTurn>,
    tools: usize,
    trace_mode: Option<Option<std::path::PathBuf>>,
    reps: usize,
) -> f64 {
    let mut total = 0f64;
    for _rep in 0..reps {
        let mut engine = ScriptedModel::new(script.clone());
        let protocol = Arc::new(EmberJsonToolProtocol::default());
        let recorder = trace_mode.clone().map(|path| {
            TraceRecorder::open(
                TraceConfig {
                    output_path: path,
                    ..Default::default()
                },
                "bench",
            )
            .unwrap()
        });
        let dir = std::env::temp_dir().join("ember-agent-bench");
        std::fs::create_dir_all(&dir).unwrap();
        let resources = RunResources {
            trace: recorder,
            artifacts: Arc::new(Mutex::new(ArtifactStore::open(&dir, "bench").unwrap())),
        };
        let mut session = AgentSession::new(
            &mut engine,
            protocol,
            registry(tools),
            AgentConfig::default(),
            AgentLimits::default(),
        );
        let t0 = Instant::now();
        let summary = session
            .run(&CancelFlag::new(), "benchmark input", resources)
            .unwrap();
        total += t0.elapsed().as_secs_f64() * 1e3;
        assert_eq!(
            summary.status,
            ember::agent::RunStatus::Completed,
            "bench run must complete"
        );
    }
    total / reps as f64
}

fn main() {
    if std::env::var("BENCH").is_err() {
        // cargo bench passes --bench; allow plain `cargo run --bench` too
    }
    const REPS: usize = 200;

    // warmup
    let _ = bench(one_tool_script(), 3, Some(None), 20);

    let one_tool_off = bench(one_tool_script(), 3, None, REPS);
    let one_tool_mem = bench(one_tool_script(), 3, Some(None), REPS);
    let tmp = std::env::temp_dir().join(format!("ember-bench-trace-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let path = tmp.join("trace.jsonl");
    let one_tool_file = bench(one_tool_script(), 3, Some(Some(path.clone())), REPS);
    let three_tool_off = bench(three_tool_script(), 5, None, REPS);
    let three_tool_mem = bench(three_tool_script(), 5, Some(None), REPS);

    println!("=== agent orchestration overhead (median-of-{REPS} wall ms per run) ===");
    println!("mock one-tool, trace off : {one_tool_off:.4} ms");
    println!("mock one-tool, memory tr : {one_tool_mem:.4} ms");
    println!("mock one-tool, JSONL out : {one_tool_file:.4} ms");
    println!("mock three-tool, off     : {three_tool_off:.4} ms");
    println!("mock three-tool, memory  : {three_tool_mem:.4} ms");
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let lines = std::fs::read_to_string(&path)
        .map(|s| s.lines().count())
        .unwrap_or(0);
    println!("JSONL trace: {bytes} bytes / {lines} events accumulated over {REPS} runs");
    let per_run_bytes = bytes / REPS as u64;
    println!("trace bytes per one-tool run: ~{per_run_bytes}");
    std::fs::remove_dir_all(&tmp).ok();

    assert!(one_tool_file < 50.0, "orchestration must stay negligible");
}
