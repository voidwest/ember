//! Real-GGUF agent validation (Track U).
//!
//! Drives the full loop — llama3 protocol, weather_fixture lookup,
//! reinjection, final answer — against REAL weights. Skips silently
//! unless:
//!
//! ```text
//! EMBER_AGENT_E2E=1
//! EMBER_AGENT_MODEL       path to a llama-family instruct GGUF (Q8_0)
//! EMBER_AGENT_TOKENIZER   matching tokenizer.json
//! ```
//!
//! Hermetic `cargo test` runs skip them; the Phase 1 report records the
//! executed run against Llama-3.2-1B-Instruct-Q8_0.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ember::agent::protocol::LlamaToolProtocol;
use ember::agent::tools::{CalculatorTool, LookupFixtureTool};
use ember::agent::{
    validate_trace_invariants, AgentConfig, AgentLimits, AgentSession, ArtifactStore, CancelFlag,
    RunResources, ToolRegistry, TraceConfig, TraceRecorder,
};

struct Fixture {
    model: PathBuf,
    tokenizer: PathBuf,
}

fn fixture() -> Option<Fixture> {
    if std::env::var("EMBER_AGENT_E2E").ok().as_deref() != Some("1") {
        return None;
    }
    let get = |k: &str| -> Option<PathBuf> {
        std::env::var(k)
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    Some(Fixture {
        model: get("EMBER_AGENT_MODEL")?,
        tokenizer: get("EMBER_AGENT_TOKENIZER")?,
    })
}

#[test]
fn real_model_uses_the_weather_fixture_tool() {
    let Some(fx) = fixture() else {
        eprintln!("skipping: set EMBER_AGENT_E2E=1 (+ EMBER_AGENT_MODEL/TOKENIZER)");
        return;
    };

    eprintln!("loading {} …", fx.model.display());
    use ember::loader::load_gguf_with_k_strategy;
    let loader =
        load_gguf_with_k_strategy(&fx.model, ember::quant_k::KStrategy::Auto, true).unwrap();
    let declared = match loader.metadata.get("general.architecture") {
        Some(ember::loader::GgufValue::Str(v)) => v.clone(),
        _ => panic!("missing architecture"),
    };
    let quantization = loader
        .metadata
        .get("general.file_type")
        .and_then(|v| match v {
            ember::loader::GgufValue::U32(d) => {
                ember::loader::ggml_dtype_name(*d).map(str::to_string)
            }
            _ => None,
        });
    let model_sha256 = ember::extraction::sha256_file_result(&fx.model).unwrap();
    let tokenizer_sha256 = ember::extraction::sha256_file_result(&fx.tokenizer).unwrap();

    let model = ember::llama::Llama::from_loader(loader).unwrap();
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(&fx.tokenizer).unwrap();
    let identity = ember::agent::ModelIdentity {
        model_path: fx.model.display().to_string(),
        model_sha256: Some(model_sha256),
        architecture: declared.clone(),
        quantization,
        n_layers: model.config.n_layers,
        embed_dim: model.config.embed_dim,
        vocab_size: model.config.vocab_size,
        tokenizer_sha256: Some(tokenizer_sha256),
        context_len: 4096,
    };

    let backend = ember::backend::CpuBackend;
    let mut engine =
        ember::agent::LlamaChatModel::new(&model, &backend, &tokenizer, 4096, identity);

    let fixtures = BTreeMap::from([
        ("riyadh".to_string(), "41 C".to_string()),
        ("lisbon".to_string(), "27 C".to_string()),
    ]);
    let registry = ToolRegistry::builder()
        .register(Arc::new(LookupFixtureTool::from_map(fixtures)))
        .unwrap()
        .register(Arc::new(CalculatorTool))
        .unwrap()
        .build()
        .unwrap();

    let dir = std::env::temp_dir().join(format!(
        "ember-agent-e2e-{}",
        std::time::Instant::now().elapsed().as_nanos()
    ));
    let trace_path = dir.join("e2e-trace.jsonl");
    let recorder = TraceRecorder::open(
        TraceConfig {
            output_path: Some(trace_path.clone()),
            ..Default::default()
        },
        "pending",
    )
    .unwrap();

    let mut session = AgentSession::new(
        &mut engine,
        Arc::new(LlamaToolProtocol::default()),
        registry,
        AgentConfig::default(),
        AgentLimits {
            max_steps: 4,
            max_output_tokens_per_turn: 128,
            ..Default::default()
        },
    );

    let summary = session
        .run(
            &CancelFlag::new(),
            "Use the available tool to tell me the fixture temperature in Riyadh. \
             Call the lookup tool with key riyadh, then report exactly what it returned.",
            RunResources {
                trace: Some(recorder),
                artifacts: Arc::new(Mutex::new(ArtifactStore::open(&dir, "pending").unwrap())),
            },
        )
        .unwrap();

    eprintln!(
        "status={:?} steps={} tools={} rejected={} final={:?}",
        summary.status,
        summary.steps_executed,
        summary.tool_calls_executed,
        summary.rejected_calls,
        summary.final_text
    );
    assert_eq!(summary.status, ember::agent::RunStatus::Completed);
    assert_eq!(
        summary.tool_calls_executed, 1,
        "fixture must force a real tool call"
    );
    assert!(summary.rejected_calls == 0, "unexpected rejections");
    let text = summary.final_text.expect("final answer");
    assert!(
        text.contains("41"),
        "answer must reflect the FIXTURE value (41 C), got: {text}"
    );

    // trace on disk validates and reconstructs the timeline
    let (events, skipped) = ember::agent::parse_trace_file(&trace_path).unwrap();
    assert!(skipped.is_empty());
    assert_eq!(validate_trace_invariants(&events), Vec::<String>::new());
    let provenance = events
        .iter()
        .find(|e| e["event_type"] == "provenance")
        .expect("provenance recorded");
    assert_eq!(provenance["data"]["protocol_id"], "llama3-python-tag-v1");
    assert!(
        !provenance["data"]["model"]["quantization"].is_null(),
        "quantization provenance captured"
    );
    std::fs::remove_dir_all(&dir).ok();
}
