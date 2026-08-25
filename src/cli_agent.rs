//! `ember agent` + `ember trace inspect` (Track R): CLI over the Phase 1
//! agentic runtime. Thin by contract — every behavior lives behind
//! `ember::agent`, exactly as applications would drive it.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use clap::{Args as ClapArgs, Subcommand};

use ember::agent::{
    AgentConfig, AgentLimits, AgentSession, ArtifactStore, EmberJsonToolProtocol, LlamaChatModel,
    LlamaToolProtocol, LookupFixtureTool, ModelIdentity, Qwen25ToolProtocol, RunResources,
    ToolCallProtocol, ToolRegistry, TraceConfig, TraceRecorder,
};

#[derive(ClapArgs)]
pub(crate) struct AgentCommand {
    #[command(subcommand)]
    pub command: AgentSubcommand,
}

#[derive(Subcommand)]
pub(crate) enum AgentSubcommand {
    /// Run one agent task over a local GGUF model with deterministic tools
    Run(AgentRunArgs),
    /// Research demo: inspect local experiment summaries, compute the best
    /// configuration's relative improvement, and write a result artifact
    Demo(AgentDemoArgs),
}

#[derive(clap::Args)]
pub(crate) struct AgentRunArgs {
    /// path to the GGUF model
    #[arg(short, long)]
    pub model: String,

    /// tool-call protocol: qwen25 | llama3 | generic-json
    #[arg(long, default_value = "llama3", value_parser = ["qwen25", "llama3", "generic-json"])]
    pub protocol: String,

    /// the user task
    #[arg(short, long)]
    pub prompt: String,

    /// optional base system instruction
    #[arg(long)]
    pub system: Option<String>,

    /// built-in tools to register (calculate, lookup, echo, write_artifact)
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "calculate,lookup,echo,write_artifact"
    )]
    pub tools: Vec<String>,

    /// fixture for the lookup tool: key=value (repeatable)
    #[arg(long = "fixture", value_delimiter = ',')]
    pub fixtures: Vec<String>,

    /// JSONL trace output path
    #[arg(long)]
    pub trace_out: Option<String>,

    /// artifact directory (default: alongside the trace or ./agent-artifacts)
    #[arg(long)]
    pub artifacts_dir: Option<String>,

    /// max model turns
    #[arg(long, default_value_t = 8)]
    pub max_steps: usize,

    /// max successful tool executions
    #[arg(long, default_value_t = 16)]
    pub max_tool_calls: usize,

    /// per-turn output token cap
    #[arg(long, default_value_t = 256)]
    pub max_output_tokens: usize,

    /// per-tool execution timeout in seconds
    #[arg(long, default_value_t = 60)]
    pub tool_timeout_secs: u64,

    /// whole-run wall-clock budget in seconds (0 disables)
    #[arg(long, default_value_t = 600)]
    pub wall_time_secs: u64,

    /// sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// deterministic sampling seed (temperature > 0)
    #[arg(long)]
    pub seed: Option<u64>,

    /// record the model file SHA-256 into trace provenance
    #[arg(long, default_value_t = true)]
    pub record_model_sha256: bool,

    /// explicit tokenizer path (default: beside the model, then ./tokenizer.json)
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// omit prompt/generated text from traces (lengths + hashes only)
    #[arg(long, default_value_t = false)]
    pub privacy_off_content: bool,

    /// KV cache capacity for the conversation
    #[arg(long, default_value_t = 8192)]
    pub kv_capacity: usize,

    /// sandbox root enabling the read-only file tools (read_text_file,
    /// search_text); paths stay strictly inside this directory
    #[arg(long)]
    pub sandbox_root: Option<String>,

    /// search tool available when a sandbox root is set
    #[arg(long, default_value_t = true)]
    pub allow_search: bool,

    /// approve tools that declare ExternalSideEffect (built-ins never do)
    #[arg(long, default_value_t = false)]
    pub allow_unsafe_effects: bool,
}

#[derive(clap::Args)]
pub(crate) struct AgentDemoArgs {
    /// path to the GGUF model
    #[arg(short, long)]
    pub model: String,

    /// directory with .txt experiment summaries to inspect
    #[arg(short, long)]
    pub summaries_dir: String,

    /// JSONL trace output path
    #[arg(long, default_value = "agent-demo-trace.jsonl")]
    pub trace_out: String,

    /// artifact directory
    #[arg(long, default_value = "agent-demo-artifacts")]
    pub artifacts_dir: String,

    /// max model turns
    #[arg(long, default_value_t = 8)]
    pub max_steps: usize,

    /// per-turn output token cap
    #[arg(long, default_value_t = 320)]
    pub max_output_tokens: usize,

    /// explicit tokenizer path (default: beside the model)
    #[arg(long)]
    pub tokenizer: Option<String>,
}

#[derive(ClapArgs)]
pub(crate) struct TraceCommand {
    #[command(subcommand)]
    pub command: TraceSubcommand,
}

#[derive(Subcommand)]
pub(crate) enum TraceSubcommand {
    /// Print a compact timeline + aggregates from an agent trace JSONL
    Inspect(TraceInspectArgs),
    /// Structurally compare two traces (status, steps, tools, skeleton)
    Diff(TraceDiffArgs),
    /// Re-execute recorded deterministic tool calls and verify digests
    Replay(TraceReplayArgs),
    /// Write a self-contained HTML report for a trace
    Report(TraceReportArgs),
}

#[derive(ClapArgs)]
pub(crate) struct TraceDiffArgs {
    /// first trace
    #[arg(long)]
    pub a: String,
    /// second trace
    #[arg(long)]
    pub b: String,
    /// exit non-zero when the traces differ (for scripting)
    #[arg(long, default_value_t = false)]
    pub fail_on_diff: bool,
}

#[derive(ClapArgs)]
pub(crate) struct TraceReplayArgs {
    /// trace whose tool calls to replay
    #[arg(long)]
    pub input: String,
    /// built-in tools to register (must cover every recorded call)
    #[arg(long, value_delimiter = ',')]
    pub tools: Vec<String>,
    /// fixture for the lookup tool: key=value (repeatable)
    #[arg(long = "fixture", value_delimiter = ',')]
    pub fixtures: Vec<String>,
    /// sandbox root for file tools
    #[arg(long)]
    pub sandbox_root: Option<String>,
}

#[derive(ClapArgs)]
pub(crate) struct TraceReportArgs {
    /// trace to render
    #[arg(long)]
    pub input: String,
    /// output HTML path
    #[arg(long)]
    pub output: String,
}

#[derive(clap::Args)]
pub(crate) struct TraceInspectArgs {
    /// trace file to inspect
    #[arg(long)]
    pub input: String,

    /// print every event as JSON instead of the timeline
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

fn build_protocol(name: &str) -> Result<Arc<dyn ToolCallProtocol>> {
    Ok(match name {
        "qwen25" => Arc::new(Qwen25ToolProtocol::default()),
        "llama3" => Arc::new(LlamaToolProtocol::default()),
        "generic-json" => Arc::new(EmberJsonToolProtocol::default()),
        other => anyhow::bail!("unknown protocol {other}"),
    })
}

fn build_registry(
    names: &[String],
    fixtures: &[String],
    sandbox_root: Option<&str>,
    allow_search: bool,
) -> Result<ToolRegistry> {
    let mut builder = ToolRegistry::builder();
    for name in names {
        builder = match name.as_str() {
            "calculate" | "calc" => builder.register(Arc::new(
                ember::agent::CalculatorTool,
            )),
            "lookup" => {
                let mut pairs = Vec::new();
                for fx in fixtures {
                    let (k, v) = fx
                        .split_once('=')
                        .with_context(|| format!("fixture `{fx}` must be key=value"))?;
                    pairs.push((k.trim().to_string(), v.trim().to_string()));
                }
                if pairs.is_empty() {
                    pairs.push(("riyadh".to_string(), "41 C".to_string()));
                    pairs.push(("lisbon".to_string(), "27 C".to_string()));
                }
                builder.register(Arc::new(LookupFixtureTool::new(pairs)))
            }
            "echo" => builder.register(Arc::new(ember::agent::EchoTool)),
            "write_artifact" => builder.register(Arc::new(ember::agent::WriteArtifactTool)),
            "image_fixture" => builder.register(Arc::new(ember::agent::ImageFixtureTool)),
            "read_text_file" | "search_text" => {
                let root = sandbox_root.ok_or_else(|| {
                    anyhow::anyhow!("tool `{name}` requires --sandbox-root <dir>")
                })?;
                if name == "read_text_file" {
                    builder.register(Arc::new(ember::agent::ReadTextFileTool::new(root)))
                } else if allow_search {
                    builder.register(Arc::new(ember::agent::SearchTextTool::new(root)))
                } else {
                    Ok(builder)
                }
            }
            other => anyhow::bail!("unknown built-in tool `{other}` (available: calculate, lookup, echo, write_artifact, image_fixture)"),
        }
        .with_context(|| format!("registering tool `{name}`"))?;
    }
    builder.build().map_err(|e| anyhow::anyhow!("{e}"))
}

/// LLAMA_FTYPE enum names (general.file_type), distinct from per-tensor
/// GGML dtypes.
fn llama_ftype_name(ftype: u64) -> Option<&'static str> {
    Some(match ftype {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        4 => "q4_1_some_f16",
        7 => "q8_0",
        8 => "q5_0",
        9 => "q5_1",
        10 => "q2_k",
        11 => "q3_k_s",
        12 => "q3_k_m",
        13 => "q3_k_l",
        14 => "q4_k_s",
        15 => "q4_k_m",
        16 => "q5_k_s",
        17 => "q5_k_m",
        18 => "q6_k",
        19 => "iq2_xxs",
        20 => "iq2_xs",
        21 => "q2_k_s",
        22 => "iq3_xs",
        23 => "iq3_xxs",
        24 => "iq1_s",
        25 => "iq4_nlw",
        26 => "iq3_s",
        27 => "iq3_m",
        28 => "iq2_s",
        29 => "iq2_m",
        30 => "iq4_xs",
        31 => "iq1_m",
        32 => "bf16",
        _ => return None,
    })
}

pub(crate) fn run_agent_command(command: &AgentCommand) -> Result<()> {
    match &command.command {
        AgentSubcommand::Run(args) => run_agent_run(args),
        AgentSubcommand::Demo(args) => run_agent_demo(args),
    }
}

struct Loaded {
    model: ember::llama::Llama<ember::backend::CpuBackend>,
    tokenizer: ember::tokenizer::EmberTokenizer,
    identity: ModelIdentity,
    arch: String,
}

/// Load the GGUF + tokenizer with the production K strategy and assemble
/// provenance identity. Mirrors the established main.rs loading discipline.
fn load_model(
    model_path: &str,
    k_capacity_hint: usize,
    record_hash: bool,
    tokenizer_override: Option<&str>,
) -> Result<Loaded> {
    use ember::loader::load_gguf_with_k_strategy;
    let loader = load_gguf_with_k_strategy(model_path, ember::quant_k::KStrategy::Auto, true)?;
    let declared = match loader.metadata.get("general.architecture") {
        Some(ember::loader::GgufValue::Str(v)) => v.clone(),
        _ => anyhow::bail!("GGUF missing general.architecture"),
    };
    anyhow::ensure!(
        matches!(declared.as_str(), "llama" | "qwen2" | "qwen3"),
        "agent phase 1 supports llama/qwen-family GGUFs; got `{declared}`"
    );
    // general.file_type carries the LLAMA_FTYPE enum (not a GGML dtype)
    let quantization = loader.metadata.get("general.file_type").and_then(|v| {
        let n = match v {
            ember::loader::GgufValue::U32(d) => Some(*d as u64),
            ember::loader::GgufValue::I32(d) => Some(*d as u64),
            ember::loader::GgufValue::U64(d) => Some(*d),
            ember::loader::GgufValue::I64(d) => Some(*d as u64),
            _ => None,
        }?;
        llama_ftype_name(n).map(str::to_string)
    });
    let model_sha256 = if record_hash {
        Some(
            ember::extraction::sha256_file_result(model_path)
                .with_context(|| format!("hashing {model_path}"))?,
        )
    } else {
        None
    };

    // tokenizer resolution: explicit flag > beside the model > repo default
    let model_dir = std::path::Path::new(model_path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let candidates: Vec<String> = [
        [
            "tokenizer.json",
            "tokenizer-qwen2.5.json",
            "tokenizer-qwen3.json",
        ]
        .iter()
        .map(|n| model_dir.join(n).display().to_string())
        .collect::<Vec<_>>(),
        vec![
            "tokenizer.json".to_string(),
            "tokenizer-qwen2.5.json".to_string(),
            "tokenizer-qwen3.json".to_string(),
        ],
    ]
    .concat();
    let tokenizer_path = if let Some(explicit) = tokenizer_override {
        explicit.to_string()
    } else {
        candidates
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .ok_or_else(|| anyhow::anyhow!("no tokenizer found; pass --tokenizer explicitly"))?
            .clone()
    };
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(&tokenizer_path)?;
    let tokenizer_sha256 = ember::extraction::sha256_file_result(&tokenizer_path).ok();

    let model = ember::llama::Llama::from_loader(loader)?;
    let identity = ModelIdentity {
        model_path: model_path.to_string(),
        model_sha256,
        architecture: declared.clone(),
        quantization,
        n_layers: model.config.n_layers,
        embed_dim: model.config.embed_dim,
        vocab_size: model.config.vocab_size,
        tokenizer_sha256,
        context_len: k_capacity_hint.min(model.config.max_seq_len),
    };
    Ok(Loaded {
        model,
        tokenizer,
        identity,
        arch: declared,
    })
}

fn run_agent_run(args: &AgentRunArgs) -> Result<()> {
    eprintln!("loading {} …", args.model);
    let loaded = load_model(
        &args.model,
        args.kv_capacity,
        args.record_model_sha256,
        args.tokenizer.as_deref(),
    )?;
    eprintln!(
        "loaded {} ({} layers, {:?})",
        loaded.arch, loaded.identity.n_layers, loaded.identity.quantization
    );
    run_task(
        &loaded,
        &args.prompt,
        &args.tools,
        &args.fixtures,
        args.sandbox_root.as_deref(),
        args.allow_search,
        args.protocol.as_str(),
        args.trace_out.as_deref(),
        args.artifacts_dir.as_deref(),
        AgentLimits {
            max_steps: args.max_steps,
            max_tool_calls: args.max_tool_calls,
            max_wall_time: (args.wall_time_secs > 0)
                .then(|| Duration::from_secs(args.wall_time_secs)),
            tool_timeout: Duration::from_secs(args.tool_timeout_secs.max(1)),
            max_output_tokens_per_turn: args.max_output_tokens,
            max_tool_result_bytes: 32 * 1024,
        },
        AgentConfig {
            system_prompt: args.system.clone(),
            temperature: args.temperature,
            top_k: None,
            top_p: None,
            seed: args.seed,
            kv_capacity: args.kv_capacity,
            approval: if args.allow_unsafe_effects {
                ember::agent::ApprovalPolicy::Auto
            } else {
                ember::agent::ApprovalPolicy::default()
            },
        },
        !args.privacy_off_content,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_task(
    loaded: &Loaded,
    prompt: &str,
    tools: &[String],
    fixtures: &[String],
    sandbox_root: Option<&str>,
    allow_search: bool,
    protocol: &str,
    trace_out: Option<&str>,
    artifacts_dir: Option<&str>,
    limits: AgentLimits,
    config: AgentConfig,
    trace_content: bool,
) -> Result<()> {
    let backend = ember::backend::CpuBackend;
    let mut engine = LlamaChatModel::new(
        &loaded.model,
        &backend,
        &loaded.tokenizer,
        config.kv_capacity,
        loaded.identity.clone(),
    );
    let protocol = build_protocol(protocol)?;
    let registry = build_registry(tools, fixtures, sandbox_root, allow_search)?;

    let run_id_placeholder = "pending";
    let trace_config = TraceConfig {
        output_path: trace_out.map(PathBuf::from),
        trace_prompts: trace_content,
        trace_generated_text: trace_content,
        ..Default::default()
    };
    let recorder = TraceRecorder::open(trace_config, run_id_placeholder)?;

    let artifact_root = artifacts_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./agent-artifacts"));
    let store = ArtifactStore::open(&artifact_root, run_id_placeholder)?;

    let mut session = AgentSession::new(&mut engine, protocol, registry, config, limits);
    let control = ember::agent::CancelFlag::new();
    let resources = RunResources {
        trace: Some(recorder),
        artifacts: Arc::new(Mutex::new(store)),
    };

    let summary = session.run(&control, prompt, resources)?;

    println!("{}", serde_json::to_string_pretty(&summary)?);
    if let Some(text) = &summary.final_text {
        println!("\n--- final answer ---\n{text}");
    }
    eprintln!(
        "\nstatus={:?} steps={} tool_calls={} rejected={} model_ms={:.1} tool_ms={:.1}",
        summary.status,
        summary.steps_executed,
        summary.tool_calls_executed,
        summary.rejected_calls,
        summary.total_model_ms,
        summary.total_tool_ms,
    );
    Ok(())
}

fn run_agent_demo(args: &AgentDemoArgs) -> Result<()> {
    // deterministic research fixture set: read/search over the summaries
    // dir + calculate + write_artifact.
    let dir = std::fs::read_dir(&args.summaries_dir)
        .with_context(|| format!("reading {}", args.summaries_dir))?;
    let mut files: Vec<String> = dir
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "txt"))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "no .txt summaries found in {}",
        args.summaries_dir
    );

    let loaded = load_model(&args.model, 8192, true, args.tokenizer.as_deref())?;
    eprintln!("demo model loaded ({})", loaded.arch);

    let first_file = files[0].clone();
    let rest_files = &files[1..];
    let prompt = format!(
        "Analyze local experiment summaries step by step.\n\
         Step 1: call read_text_file with path {first_file} to read its score.\n\
         Step 2: call read_text_file on each remaining summary file ({rest_files:?}).\n\
         Step 3: after reading all scores, decide which config has the highest \
         score and which has the lowest. Call calculate exactly once with \
         operation divide, a set to (best_score - worst_score), and b set to worst_score.\n\
         Step 4: call write_artifact with name best-config.md containing the winner, \
         its score, the worst score, and the relative improvement ((a/b)*100 percent).\n\
         Final answer: one sentence naming the best configuration. Use ONLY these \
         tools; do not invent others."
    );

    let registry = ToolRegistry::builder()
        .register(Arc::new(ember::agent::ReadTextFileTool::new(
            &args.summaries_dir,
        )))
        .unwrap()
        .register(Arc::new(ember::agent::SearchTextTool::new(
            &args.summaries_dir,
        )))
        .unwrap()
        .register(Arc::new(ember::agent::CalculatorTool))
        .unwrap()
        .register(Arc::new(ember::agent::WriteArtifactTool))
        .unwrap()
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut engine = LlamaChatModel::new(
        &loaded.model,
        &ember::backend::CpuBackend,
        &loaded.tokenizer,
        8192,
        loaded.identity.clone(),
    );
    let protocol = Arc::new(LlamaToolProtocol::default());
    let recorder = TraceRecorder::open(
        TraceConfig {
            output_path: Some(PathBuf::from(&args.trace_out)),
            ..Default::default()
        },
        "pending",
    )?;
    let store = ArtifactStore::open(&args.artifacts_dir, "pending")?;
    let mut session = AgentSession::new(
        &mut engine,
        protocol,
        registry,
        AgentConfig::default(),
        AgentLimits {
            max_steps: args.max_steps,
            max_output_tokens_per_turn: args.max_output_tokens,
            ..Default::default()
        },
    );
    let summary = session.run(
        &ember::agent::CancelFlag::new(),
        &prompt,
        RunResources {
            trace: Some(recorder),
            artifacts: Arc::new(Mutex::new(store)),
        },
    )?;

    println!("{}", serde_json::to_string_pretty(&summary)?);
    if let Some(t) = &summary.final_text {
        println!("\n--- final answer ---\n{t}");
    }
    eprintln!("trace written to {}", args.trace_out);
    Ok(())
}

pub(crate) fn run_trace_command(command: &TraceCommand) -> Result<()> {
    match &command.command {
        TraceSubcommand::Inspect(args) => run_trace_inspect(args),
        TraceSubcommand::Diff(args) => run_trace_diff(args),
        TraceSubcommand::Replay(args) => run_trace_replay(args),
        TraceSubcommand::Report(args) => run_trace_report(args),
    }
}

fn run_trace_diff(args: &TraceDiffArgs) -> Result<()> {
    let (ea, _) = ember::agent::parse_trace_file(std::path::Path::new(&args.a))?;
    let (eb, _) = ember::agent::parse_trace_file(std::path::Path::new(&args.b))?;
    let (sa, sb, diff) = ember::agent::inspect::diff(&ea, &eb);
    println!(
        "a: {} ({})\nb: {} ({})",
        sa.run_id,
        sa.status.as_deref().unwrap_or("?"),
        sb.run_id,
        sb.status.as_deref().unwrap_or("?"),
    );
    if diff.is_identical() {
        println!("verdict: IDENTICAL (structure + totals)");
    } else {
        println!("verdict: DIFFERS");
        for d in &diff.differences {
            println!("  - {d}");
        }
    }
    println!(
        "\n         a          b\nsteps    {:>8} {:>10}\ntools    {:>8} {:>10}\nrejected {:>8} {:>10}\nartifacts {:>7} {:>10}\nout tok  {:>8} {:>10}\nmodel ms {:>8.0} {:>10.0}",
        sa.model_steps, sb.model_steps,
        sa.tool_calls, sb.tool_calls,
        sa.rejected_calls, sb.rejected_calls,
        sa.artifacts.len(), sb.artifacts.len(),
        sa.output_tokens, sb.output_tokens,
        sa.total_model_ms, sb.total_model_ms,
    );
    if diff.differences.is_empty() {
        Ok(())
    } else if args.fail_on_diff {
        anyhow::bail!("traces differ")
    } else {
        Ok(())
    }
}

fn run_trace_replay(args: &TraceReplayArgs) -> Result<()> {
    let (events, skipped) = ember::agent::parse_trace_file(std::path::Path::new(&args.input))?;
    anyhow::ensure!(
        skipped.is_empty(),
        "refusing to replay a torn trace ({skipped:?})"
    );
    let registry = build_registry(
        &args.tools,
        &args.fixtures,
        args.sandbox_root.as_deref(),
        true,
    )?;
    let report = ember::agent::inspect::replay(&events, &registry)?;
    for o in &report.outcomes {
        match o.matches {
            Some(true) => println!("{:>4} {} ok (digest match)", o.seq, o.tool),
            Some(false) => println!(
                "{:>4} {} MISMATCH recorded={} computed={}",
                o.seq, o.tool, o.recorded_replay_sha256, o.computed_replay_sha256
            ),
            None => println!("{:>4} {} skipped (no digest / failed call)", o.seq, o.tool),
        }
    }
    println!(
        "{} calls verified, {} skipped; verdict: {}",
        report
            .outcomes
            .iter()
            .filter(|o| o.matches == Some(true))
            .count(),
        report.skipped(),
        if report.all_matched() {
            "MATCH"
        } else {
            "MISMATCH"
        },
    );
    if !report.all_matched() {
        anyhow::bail!("replay verification failed");
    }
    Ok(())
}

fn run_trace_report(args: &TraceReportArgs) -> Result<()> {
    let (events, _) = ember::agent::parse_trace_file(std::path::Path::new(&args.input))?;
    let summary = ember::agent::inspect::summarize(&events);
    let html = ember::agent::inspect::render_html(&events, &summary);
    std::fs::write(&args.output, html).with_context(|| format!("writing {}", args.output))?;
    eprintln!("wrote {}", args.output);
    Ok(())
}

fn run_trace_inspect(args: &TraceInspectArgs) -> Result<()> {
    let (summary, timeline_rows, skipped) =
        ember::agent::inspect_file(std::path::Path::new(&args.input))?;
    if args.json {
        let events = ember::agent::parse_trace_file(std::path::Path::new(&args.input))?.0;
        println!("{}", serde_json::to_string_pretty(&events)?);
        return Ok(());
    }
    for line in &skipped {
        eprintln!("warning: skipped {line}");
    }
    println!(
        "run {}\nmodel: {} ({:?}, {}) protocol: {}\nstatus: {}",
        summary.run_id,
        summary.model_path.as_deref().unwrap_or("?"),
        summary.quantization.as_deref().unwrap_or("?"),
        summary.architecture.as_deref().unwrap_or("?"),
        summary.protocol_id.as_deref().unwrap_or("?"),
        summary.status.as_deref().unwrap_or("(incomplete)"),
    );
    println!(
        "steps: {}  tools: {}  rejected: {}  artifacts: {}",
        summary.model_steps,
        summary.tool_calls,
        summary.rejected_calls,
        summary.artifacts.len(),
    );
    println!(
        "model time: {:.1}ms (prefill {:.1} / decode {:.1})  tool time: {:.1}ms  tokens out: {}  tok/s: {}",
        summary.total_model_ms,
        summary.prefill_ms,
        summary.decode_ms,
        summary.total_tool_ms,
        summary.output_tokens,
        summary.tok_per_s.map(|t| format!("{t:.0}")).unwrap_or_else(|| "-".into()),
    );
    if let Some(err) = &summary.error {
        println!("error: {err}");
    }
    println!();
    for row in &timeline_rows {
        println!("{:>9.3}s  {}", row.t_ms / 1000.0, row.label);
    }
    Ok(())
}
