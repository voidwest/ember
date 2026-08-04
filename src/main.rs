mod cli_experiment;
mod cli_support;

mod cli_commands;
mod cli_generation;
mod cli_probe;

use cli_commands::{
    effective_context_limit, run_bench_decode_command, run_bench_lifecycle_command,
    run_compare_artifacts_command, run_extract_command, run_inspect_plan_command,
    run_native_logits_reference_command, run_validate_backends_command, run_validate_run_command,
    validate_experiment_options,
};
use cli_generation::{
    bail_dump_layers_unsupported, demo_mode, dump_last_logits, dump_layers_gemma4,
    interactive_mode, run_single_prompt, run_single_prompt_with_experiment,
};
use cli_probe::{run_probe_jobs, TensorDumpConfig};

use anyhow::Context;
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use cli_support::{
    build_run_manifest, default_tokenizer_for_arch, gguf_metadata_json, parse_max_seq_len,
    parse_temperature, parse_top_k, parse_top_p, resolve_generation_architecture,
    resolve_tokenizer, write_json_file,
};
use ember::backend::Backend;
use ember::backend::CpuBackend;
use ember::experiments::{
    ActivationPatch, ActivationStats, CaptureSink, ExperimentRunner, ModelContext, ModelFamily,
    PatchTarget, ZeroLayerOutput, ZeroLayerOutputSpec,
};
use ember::extraction::{sha256_file_result, ExecutionBackendName};
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::model::Gpt2;
use std::fs;

pub(crate) fn rayon_current_num_threads() -> usize {
    // rayon doesn't expose current thread count directly; check env
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
}

/// a lightweight, cpu-first llm inference engine.
#[derive(Parser)]
#[command(name = "ember", version)]
pub(crate) struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// path to gguf model file
    #[arg(short, long, default_value = "gpt2.Q8_0.gguf")]
    model: String,

    /// path to tokenizer.json
    #[arg(long)]
    tokenizer: Option<String>,

    /// text prompt to complete
    #[arg(short, long, default_value = "The")]
    prompt: String,

    /// number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 20)]
    max_tokens: usize,

    /// cap usable context length below the model metadata value
    #[arg(long, value_parser = parse_max_seq_len)]
    max_seq_len: Option<usize>,

    /// sampling temperature (0 = greedy argmax)
    #[arg(short, long, default_value_t = 0.8, value_parser = parse_temperature)]
    temperature: f32,

    /// top-k sampling: keep only the k highest logits
    #[arg(long, value_parser = parse_top_k)]
    top_k: Option<usize>,

    /// top-p (nucleus) sampling: keep smallest set of tokens with cumulative probability >= p
    #[arg(long, value_parser = parse_top_p)]
    top_p: Option<f32>,

    /// stay in an interactive read-eval-print loop after the first prompt
    #[arg(short, long)]
    interactive: bool,

    /// model architecture override; auto reads general.architecture from GGUF
    #[arg(long, default_value = "auto", value_parser = ["auto", "gpt2", "llama", "qwen3", "gemma4"])]
    arch: String,

    /// run a curated demo that showcases the project with deterministic output and timing
    #[arg(long, conflicts_with = "interactive")]
    demo: bool,

    /// milliseconds to delay between each token in demo mode (0 = instant)
    #[arg(long, default_value_t = 0)]
    delay_ms: u64,

    /// print prefill/decode timing stats to stderr
    #[arg(long)]
    benchmark: bool,

    /// example research intervention, formatted as LAYER:attention|mlp|layer
    #[arg(
        long,
        value_name = "LAYER:STAGE",
        conflicts_with_all = [
            "demo",
            "interactive",
            "dump_logits",
            "dump_layers",
            "probe",
            "activation_stats"
        ]
    )]
    zero_layer_output: Option<ZeroLayerOutputSpec>,

    /// write observation-only activation norms and fingerprints to JSON
    #[arg(
        long,
        value_name = "FILE",
        conflicts_with_all = [
            "demo",
            "interactive",
            "dump_logits",
            "dump_layers",
            "probe",
            "zero_layer_output"
        ]
    )]
    activation_stats: Option<String>,

    /// capture selected activations during generation into a v0.2 artifact
    /// (typed TOML selection; see docs/activation-artifacts.md)
    #[arg(
        long,
        value_name = "FILE",
        conflicts_with_all = ["demo", "interactive", "dump_logits", "dump_layers", "probe"]
    )]
    capture_activations: Option<String>,

    /// v0.2 activation-patch source artifact (manifest.json); requires at
    /// least one --patch-target. Conflicts with other experiments.
    #[arg(
        long,
        value_name = "FILE",
        requires = "patch_target",
        conflicts_with_all = [
            "zero_layer_output",
            "activation_stats",
            "demo",
            "interactive",
            "dump_logits",
            "dump_layers",
            "probe"
        ]
    )]
    activation_patch: Option<String>,

    /// patch target LAYER:STAGE:PHASE[:POSITION] for --activation-patch
    /// (repeatable; stage in before-layer, after-attention, after-mlp,
    /// after-layer, before-logits, after-logits)
    #[arg(long, value_name = "TARGET", requires = "activation_patch")]
    patch_target: Vec<String>,

    /// enable execution tracing (ops = per-operation breakdown)
    #[arg(
        long,
        value_parser = ["ops"],
        conflicts_with_all = ["demo", "interactive", "dump_logits", "dump_layers", "probe"]
    )]
    trace: Option<String>,

    /// write trace JSON to this path (default: stderr)
    #[arg(long, requires = "trace")]
    trace_out: Option<String>,

    /// collect output norms and fingerprints (none = off, summary = L2 norm + fingerprint)
    #[arg(long, default_value = "none", value_parser = ["none", "summary"])]
    trace_values: String,

    /// attach system metadata (CPU, governor, threads, commit) to trace report
    #[arg(long, requires = "trace")]
    trace_run_metadata: bool,

    /// write last-prompt logits for --prompt to a .npy file and exit
    #[arg(long, conflicts_with_all = ["demo", "interactive", "probe"])]
    dump_logits: Option<String>,

    /// dump per-layer hidden states (last prompt token) to a binary file and exit.
    /// format: little-endian f32 flat array, [n_layers * embed_dim], layer-major.
    #[arg(long, conflicts_with_all = ["demo", "interactive", "probe"])]
    dump_layers: Option<String>,

    /// probe mode: extract hidden states from each transformer block
    /// for every stimulus in the stimuli file, and save as .npy.
    #[arg(long, conflicts_with_all = ["demo", "interactive"])]
    probe: bool,

    /// path to stimuli json for probe mode
    #[arg(long, default_value = "stimuli/nonce_root_pattern_surface.json")]
    probe_stimuli: String,

    /// output path for probe activations (.npy)
    #[arg(long, default_value = "data/activations.npy")]
    probe_output: String,

    /// prompt template key to read from each stimulus prompts object
    #[arg(long, default_value = "en_surface_probe")]
    probe_template: String,

    /// comma-separated prompt template keys for batch probe extraction
    #[arg(long)]
    probe_templates: Option<String>,

    /// hidden-state position to probe: last, root, pattern, or prompt_mean
    #[arg(long, default_value = "last")]
    probe_position: String,

    /// comma-separated hidden-state positions for batch probe extraction
    #[arg(long)]
    probe_positions: Option<String>,

    /// output directory for batch probe extraction
    #[arg(long)]
    probe_output_dir: Option<String>,

    /// output filename prefix for batch probe extraction
    #[arg(long, default_value = "probe")]
    probe_output_prefix: String,

    /// number of continuation tokens to generate for probe behavioral scoring
    #[arg(long, default_value_t = 16)]
    probe_generate_tokens: usize,

    /// limit probe extraction to the first N stimuli for smoke tests
    #[arg(long)]
    probe_limit: Option<usize>,

    /// compute and record model file sha256 in probe metadata
    #[arg(long)]
    record_model_sha256: bool,

    /// write parsed GGUF metadata to this JSON path
    #[arg(long)]
    dump_gguf_metadata: Option<String>,

    /// write a reproducibility manifest that pins model, tokenizer, runtime, and environment
    #[arg(long)]
    write_run_manifest: Option<String>,

    /// K-family (Q4_K/Q6_K) execution strategy; `auto` selects
    /// compressed-resident execution for supported dtypes and eager-f32
    /// for dtypes without a native kernel (recorded per tensor)
    #[arg(long, default_value = "auto", value_parser = ["eager-f32", "scalar", "x86", "auto"])]
    k_strategy: String,

    /// allow per-tensor fallback (eager-f32/scalar) when the requested K
    /// strategy has no native path; the fallback is recorded, never silent
    #[arg(long)]
    k_allow_fallback: bool,

    /// v0.4 execution concept: reference (v0.3 generic path), planned
    /// (execution-plan interpreter), or planned-fused (fused plan; lands
    /// with the fusion phase)
    #[arg(long, default_value = "reference")]
    execution: String,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
    /// extract hidden-state artifacts through a selected execution backend
    Extract(ExtractCommand),
    /// write a native logits-only artifact run without generation or hidden states
    NativeLogitsReference(NativeLogitsReferenceCommand),
    /// validate one Ember artifact run directory
    ValidateRun(ValidateRunCommand),
    /// compare native and llama.cpp backend outputs where comparable
    ValidateBackends(ValidateBackendsCommand),

    /// compare two v0.2 activation artifacts record-by-record
    CompareArtifacts(CompareArtifactsCommand),
    /// benchmark model-only single-token decode with llama-bench-compatible timing
    BenchDecode(BenchDecodeCommand),
    /// measure Llama packed-weight lifecycle timing and process residency
    BenchLifecycle(BenchLifecycleCommand),
    /// print the v0.4 execution plan for a llama-family model
    InspectPlan(InspectPlanCommand),

    /// reproducible experiment workflows (v0.5)
    Experiment(cli_experiment::ExperimentCommand),
}

#[derive(ClapArgs)]
pub(crate) struct BenchDecodeCommand {
    /// path to the GGUF model
    #[arg(short, long)]
    model: String,

    /// model architecture
    #[arg(long, value_parser = ["gpt2", "llama", "qwen3", "gemma4"])]
    arch: String,

    /// number of timed single-token evaluations per repetition
    #[arg(short = 'n', long, default_value_t = 128)]
    tokens: usize,

    /// untimed repetitions used to warm model pages and kernels
    #[arg(long, default_value_t = 2)]
    warmups: usize,

    /// measured repetitions
    #[arg(short, long, default_value_t = 5)]
    repetitions: usize,

    /// deterministic token id fed to the model
    #[arg(long, default_value_t = 1)]
    token_id: u32,

    /// optional context-size cap
    #[arg(long, value_parser = parse_max_seq_len)]
    max_seq_len: Option<usize>,

    /// collect fast-path per-operator timing in the benchmark JSON
    #[arg(long)]
    profile_operators: bool,

    /// v0.4 execution concept for the benchmarked decode (reference |
    /// planned | planned-fused)
    #[arg(long, default_value = "reference")]
    execution: String,
}

#[derive(ClapArgs)]
pub(crate) struct InspectPlanCommand {
    /// path to the GGUF model
    #[arg(short, long)]
    model: String,

    /// model architecture
    #[arg(long, value_parser = ["gpt2", "llama", "qwen3", "gemma4"])]
    arch: String,

    /// path to tokenizer.json (optional; recorded in provenance when present)
    #[arg(long)]
    tokenizer: Option<String>,

    /// execution mode: reference | planned | planned-fused
    #[arg(long, default_value = "planned")]
    execution: String,

    /// hook mode: disabled | observe | intervene
    #[arg(long, default_value = "disabled")]
    hook: String,

    /// active hook stages (comma-separated): before-layer, after-attention,
    /// after-mlp, after-layer, before-logits, after-logits
    #[arg(long)]
    hook_stages: Option<String>,

    /// cap usable context length below the model metadata value
    #[arg(long, value_parser = parse_max_seq_len)]
    max_seq_len: Option<usize>,

    /// write the serialized execution-plan.json to this path
    #[arg(long)]
    output: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum LifecycleModeArg {
    /// generic prefill and generic decode
    Control,
    /// pack and evict before generic prefill, then use packed decode
    PackBeforePrefill,
    /// run generic prefill first, then pack and evict before decode
    PackAfterPrefill,
    /// pack and evict before prefill, then re-evict source pages after prefill
    PackBeforePrefillReevict,
    /// pack before prefill but retain the duplicate source residency
    DuplicatePacked,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum PackedSelectionArg {
    /// gate and up projections
    GateUp,
    /// gate, up, and down projections
    Mlp,
    /// Q, K, V, and O projections
    Attention,
    /// gate/up plus Q/K/V/O
    AttentionGateUp,
    /// all currently eligible transformer projections
    All,
}

impl From<PackedSelectionArg> for ember::llama::LlamaPackedSelection {
    fn from(value: PackedSelectionArg) -> Self {
        match value {
            PackedSelectionArg::GateUp => Self::GateUp,
            PackedSelectionArg::Mlp => Self::Mlp,
            PackedSelectionArg::Attention => Self::Attention,
            PackedSelectionArg::AttentionGateUp => Self::AttentionGateUp,
            PackedSelectionArg::All => Self::All,
        }
    }
}

#[derive(ClapArgs)]
pub(crate) struct BenchLifecycleCommand {
    /// path to a Llama-family Q8_0 GGUF model
    #[arg(short, long)]
    model: String,

    /// path to tokenizer.json
    #[arg(long)]
    tokenizer: String,

    /// deterministic prompt used for generic prefill
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,

    /// number of tokens to generate greedily, including the prefill result
    #[arg(short = 'n', long, default_value_t = 64)]
    tokens: usize,

    /// lifecycle ordering under test
    #[arg(long, value_enum)]
    lifecycle: LifecycleModeArg,

    /// projection group that receives the existing packed representation
    #[arg(long, value_enum, default_value_t = PackedSelectionArg::All)]
    selection: PackedSelectionArg,

    /// optional context-size cap
    #[arg(long, value_parser = parse_max_seq_len)]
    max_seq_len: Option<usize>,

    /// retain phase markers but skip procfs reads (measurement perturbation audit only)
    #[arg(long)]
    timing_only: bool,
}

#[derive(ClapArgs)]
pub(crate) struct ExtractCommand {
    /// extraction config path (.toml or .json)
    #[arg(long)]
    config: Option<String>,

    /// backend override; defaults to the backend in the config
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,

    /// llama.cpp-compatible external extractor binary
    #[arg(long)]
    llama_bin: Option<String>,

    /// GGUF model path override or direct-mode model path
    #[arg(long)]
    model: Option<String>,

    /// input samples JSONL path override or direct-mode samples path
    #[arg(long)]
    samples: Option<String>,

    /// output run directory override or direct-mode output path
    #[arg(long)]
    out: Option<String>,

    /// prompt template override; direct mode defaults to "{prompt}"
    #[arg(long)]
    prompt_template: Option<String>,

    /// architecture hint for native direct mode
    #[arg(long)]
    arch: Option<String>,

    /// tokenizer path override
    #[arg(long)]
    tokenizer: Option<String>,

    /// comma-separated layer indices; external mode currently requires this empty
    #[arg(long)]
    layers: Option<String>,

    /// token position / pooling mode
    #[arg(long, value_enum)]
    token_position: Option<TokenPositionArg>,

    /// sample id field in the input JSONL
    #[arg(long)]
    sample_id_field: Option<String>,

    /// word field for word-based position modes
    #[arg(long)]
    word_field: Option<String>,

    /// request optional logits from the backend
    #[arg(long)]
    write_logits: bool,
}

#[derive(ClapArgs)]
pub(crate) struct ValidateBackendsCommand {
    /// path to GGUF model file
    #[arg(long)]
    model: Option<String>,

    /// path to prompt JSONL/text fixture
    #[arg(long)]
    prompts: Option<String>,

    /// comma-separated layers to compare
    #[arg(long)]
    layers: Option<String>,

    /// existing native Ember artifact run directory
    #[arg(long)]
    native_run: Option<String>,

    /// existing llama-cpp-external artifact run directory
    #[arg(long)]
    external_run: Option<String>,
}

#[derive(ClapArgs)]
pub(crate) struct CompareArtifactsCommand {
    /// left v0.2 activation artifact manifest.json
    #[arg(long)]
    left: String,

    /// right v0.2 activation artifact manifest.json
    #[arg(long)]
    right: String,

    /// emit machine-readable JSON to stdout (default: human-readable)
    #[arg(long)]
    json: bool,

    /// write the report JSON to this path
    #[arg(long)]
    output: Option<String>,
}

#[derive(ClapArgs)]
pub(crate) struct ValidateRunCommand {
    /// existing Ember artifact run directory
    run_dir: String,

    /// require at least one hidden-state layer shard
    #[arg(long)]
    require_layers: bool,
}

#[derive(ClapArgs)]
pub(crate) struct NativeLogitsReferenceCommand {
    /// extraction config path (.toml or .json); must use backend = "native"
    #[arg(long)]
    config: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum BackendArg {
    Native,
    LlamaCpp,
    LlamaCppExternal,
}

impl From<BackendArg> for ExecutionBackendName {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Native => Self::Native,
            BackendArg::LlamaCpp => Self::LlamaCpp,
            BackendArg::LlamaCppExternal => Self::LlamaCppExternal,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum TokenPositionArg {
    #[value(name = "prompt_final")]
    PromptFinal,
    #[value(name = "word_final_subtoken")]
    WordFinalSubtoken,
    #[value(name = "word_mean")]
    WordMean,
    #[value(name = "full_prompt_mean")]
    FullPromptMean,
}

impl From<TokenPositionArg> for ember::extraction::TokenPositionMode {
    fn from(value: TokenPositionArg) -> Self {
        match value {
            TokenPositionArg::PromptFinal => Self::PromptFinal,
            TokenPositionArg::WordFinalSubtoken => Self::WordFinalSubtoken,
            TokenPositionArg::WordMean => Self::WordMean,
            TokenPositionArg::FullPromptMean => Self::FullPromptMean,
        }
    }
}

pub(crate) struct RunMetadata {
    gguf_metadata: serde_json::Value,
    model_file_size_bytes: Option<u64>,
    model_sha256: Option<String>,
    tokenizer_sha256: Option<String>,
    run_manifest: serde_json::Value,
}

fn validate_tokenizer_model_contract<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
) -> anyhow::Result<()> {
    let model_vocab_size = model.vocab_size(backend);
    tokenizer.validate_model_vocab(model_vocab_size)?;
    if tokenizer.vocab_size() != model_vocab_size {
        log::warn!(
            "tokenizer exposes {} tokens while the model head has {} rows; padded or reserved model rows may not be decodable",
            tokenizer.vocab_size(),
            model_vocab_size
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = Args::parse();
    validate_experiment_options(&args)?;

    if let Some(command) = &args.command {
        let k_strategy =
            ember::quant_k::KStrategy::from_cli(&args.k_strategy).map_err(anyhow::Error::msg)?;
        return match command {
            Commands::Extract(command) => {
                run_extract_command(command, k_strategy, args.k_allow_fallback)
            }
            Commands::NativeLogitsReference(command) => {
                run_native_logits_reference_command(command, k_strategy, args.k_allow_fallback)
            }
            Commands::ValidateRun(command) => run_validate_run_command(command),
            Commands::ValidateBackends(command) => run_validate_backends_command(command),
            Commands::CompareArtifacts(command) => run_compare_artifacts_command(command),
            Commands::BenchDecode(command) => {
                run_bench_decode_command(command, k_strategy, args.k_allow_fallback)
            }
            Commands::BenchLifecycle(command) => {
                run_bench_lifecycle_command(command, k_strategy, args.k_allow_fallback)
            }
            Commands::InspectPlan(command) => {
                run_inspect_plan_command(command, k_strategy, args.k_allow_fallback)
            }
            Commands::Experiment(command) => match &command.command {
                cli_experiment::ExperimentSubcommand::Validate(command) => {
                    cli_experiment::run_validate_command(command)
                }
                cli_experiment::ExperimentSubcommand::Run(command) => {
                    cli_experiment::run_experiment_command(
                        command,
                        k_strategy,
                        args.k_allow_fallback,
                    )
                }
                cli_experiment::ExperimentSubcommand::Inspect(command) => {
                    cli_experiment::run_inspect_command(command)
                }
                cli_experiment::ExperimentSubcommand::Verify(command) => {
                    cli_experiment::run_verify_command(command)
                }
                cli_experiment::ExperimentSubcommand::Compare(command) => {
                    cli_experiment::run_compare_command(command)
                }
                cli_experiment::ExperimentSubcommand::Reproduce(command) => {
                    cli_experiment::run_reproduce_command(
                        command,
                        k_strategy,
                        args.k_allow_fallback,
                    )
                }
                cli_experiment::ExperimentSubcommand::Tokenize(command) => {
                    cli_experiment::run_tokenize_command(command, k_strategy, args.k_allow_fallback)
                }
            },
        };
    }

    // demo mode: suppress log noise for clean recordable output
    if args.demo {
        log::set_max_level(log::LevelFilter::Off);
    }

    // Dispatch to the selected architecture. Generation, demo, and probe paths
    // are generic over `ForwardModel`; interactive mode is still GPT-2-specific.
    let k_strategy =
        ember::quant_k::KStrategy::from_cli(&args.k_strategy).map_err(anyhow::Error::msg)?;
    let execution =
        ember::plan::ExecutionMode::from_cli(&args.execution).map_err(anyhow::Error::msg)?;
    let loader = load_gguf_with_k_strategy(&args.model, k_strategy, args.k_allow_fallback)?;
    let execution_inventory = ember::artifact::ExecutionInventory::from_loader(&loader);
    args.arch = resolve_generation_architecture(&args.arch, &loader)?;
    validate_experiment_options(&args)?;
    let n_tensors = loader.tensors.len();
    let tokenizer_path = args
        .tokenizer
        .as_deref()
        .unwrap_or_else(|| default_tokenizer_for_arch(&args.arch));
    let resolved_tokenizer = resolve_tokenizer(tokenizer_path);
    let tokenizer_path = resolved_tokenizer.identity();
    let record_model_sha256 = args.record_model_sha256
        || args.write_run_manifest.is_some()
        || args.probe
        || args.dump_logits.is_some()
        || args.dump_layers.is_some()
        || args.capture_activations.is_some()
        || args.activation_patch.is_some();
    let model_sha256 = if record_model_sha256 {
        Some(
            sha256_file_result(&args.model)
                .with_context(|| format!("failed to hash model '{}'", args.model))?,
        )
    } else {
        None
    };
    let tokenizer_sha256 = Some(resolved_tokenizer.sha256()?);
    let gguf_metadata = gguf_metadata_json(&loader);
    let run_manifest = build_run_manifest(
        &args,
        tokenizer_path,
        model_sha256.as_deref(),
        tokenizer_sha256.as_deref(),
        &gguf_metadata,
    );
    let run_metadata = RunMetadata {
        gguf_metadata,
        model_file_size_bytes: fs::metadata(&args.model).ok().map(|m| m.len()),
        model_sha256,
        tokenizer_sha256,
        run_manifest,
    };
    if let Some(path) = &args.write_run_manifest {
        write_json_file(path, &run_metadata.run_manifest)?;
        eprintln!("wrote run manifest to {path}");
    }
    if let Some(path) = &args.dump_gguf_metadata {
        write_json_file(path, &run_metadata.gguf_metadata)?;
        eprintln!("wrote GGUF metadata to {path}");
    }
    let backend = CpuBackend;
    let tokenizer = resolved_tokenizer.load()?;

    match args.arch.as_str() {
        "gpt2" => {
            let model = Gpt2::from_loader(loader)?;
            validate_tokenizer_model_contract(&backend, &model, &tokenizer)?;
            log::info!("loading model from {}", args.model);
            log::info!("loaded {} tensors", n_tensors);
            log::info!("model built");
            log::debug!("wte shape: {:?}", backend.shape(&model.wte));
            log::info!("tokenizer loaded, vocab size: {}", tokenizer.vocab_size());

            if args.demo {
                demo_mode(
                    &backend,
                    &model,
                    &tokenizer,
                    args.max_tokens,
                    &args.model,
                    args.delay_ms,
                    effective_context_limit(&backend, &model, &args),
                )?;
            } else if args.interactive {
                interactive_mode(
                    &backend,
                    &model,
                    &tokenizer,
                    &args.prompt,
                    args.max_tokens,
                    args.temperature,
                    args.top_k,
                    args.top_p,
                    args.max_seq_len,
                )?;
            } else if let Some(path) = &args.dump_logits {
                dump_last_logits(
                    &backend,
                    &model,
                    &tokenizer,
                    TensorDumpConfig {
                        prompt: &args.prompt,
                        output_path: path,
                        max_seq_len: args.max_seq_len,
                        model_path: &args.model,
                        arch: &args.arch,
                        tokenizer_path,
                        run_metadata: &run_metadata,
                    },
                )?;
            } else if args.dump_layers.is_some() {
                bail_dump_layers_unsupported(&args.arch)?;
            } else if args.probe {
                run_probe_jobs(
                    &backend,
                    &model,
                    &tokenizer,
                    &args,
                    tokenizer_path,
                    &run_metadata,
                )?;
            } else {
                run_single_prompt(&backend, &model, &tokenizer, &args)?;
            }
        }
        "llama" | "qwen3" => {
            use ember::llama::Llama;
            let model = Llama::from_loader_with_max_seq_len(loader, args.max_seq_len)?;
            model.set_execution_mode(execution);
            validate_tokenizer_model_contract(&backend, &model, &tokenizer)?;
            log::info!("loading model from {}", args.model);
            log::info!("loaded {} tensors", n_tensors);
            log::info!("model built (llama)");
            log::info!("tokenizer loaded, vocab size: {}", tokenizer.vocab_size());

            if args.demo {
                demo_mode(
                    &backend,
                    &model,
                    &tokenizer,
                    args.max_tokens,
                    &args.model,
                    args.delay_ms,
                    effective_context_limit(&backend, &model, &args),
                )?;
            } else if args.interactive {
                anyhow::bail!("interactive mode not yet supported for llama");
            } else if let Some(path) = &args.dump_logits {
                dump_last_logits(
                    &backend,
                    &model,
                    &tokenizer,
                    TensorDumpConfig {
                        prompt: &args.prompt,
                        output_path: path,
                        max_seq_len: args.max_seq_len,
                        model_path: &args.model,
                        arch: &args.arch,
                        tokenizer_path,
                        run_metadata: &run_metadata,
                    },
                )?;
            } else if args.dump_layers.is_some() {
                bail_dump_layers_unsupported(&args.arch)?;
            } else if args.probe {
                run_probe_jobs(
                    &backend,
                    &model,
                    &tokenizer,
                    &args,
                    tokenizer_path,
                    &run_metadata,
                )?;
            } else if args.zero_layer_output.is_some()
                || args.activation_stats.is_some()
                || args.activation_patch.is_some()
                || args.capture_activations.is_some()
            {
                let family = if args.arch == "qwen3" {
                    ModelFamily::Qwen3
                } else {
                    ModelFamily::Llama
                };
                let model_context = ModelContext::new(
                    family,
                    Some(&args.model),
                    &args.arch,
                    model.n_layers(),
                    model.embed_dim(),
                )
                .with_provenance(
                    run_metadata.model_sha256.as_deref(),
                    run_metadata.tokenizer_sha256.as_deref(),
                );
                let mut runner = build_experiment_runner(
                    &args,
                    &run_metadata,
                    &model_context,
                    &execution_inventory,
                )?;
                let runner = runner
                    .as_mut()
                    .expect("experiment or capture requested in the arm condition");
                run_single_prompt_with_experiment(
                    &backend,
                    &model,
                    &tokenizer,
                    &args,
                    model_context,
                    runner,
                )?;
            } else {
                run_single_prompt(&backend, &model, &tokenizer, &args)?;
            }
        }
        "gemma4" => {
            use ember::gemma4::Gemma4;
            let model = Gemma4::from_loader(loader)?;
            validate_tokenizer_model_contract(&backend, &model, &tokenizer)?;
            log::info!("loading model from {}", args.model);
            log::info!("loaded {} tensors", n_tensors);
            log::info!("model built (gemma4)");
            log::info!("tokenizer loaded, vocab size: {}", tokenizer.vocab_size());

            if args.demo {
                demo_mode(
                    &backend,
                    &model,
                    &tokenizer,
                    args.max_tokens,
                    &args.model,
                    args.delay_ms,
                    effective_context_limit(&backend, &model, &args),
                )?;
            } else if args.interactive {
                anyhow::bail!("interactive mode not yet supported for gemma4");
            } else if let Some(path) = &args.dump_logits {
                dump_last_logits(
                    &backend,
                    &model,
                    &tokenizer,
                    TensorDumpConfig {
                        prompt: &args.prompt,
                        output_path: path,
                        max_seq_len: args.max_seq_len,
                        model_path: &args.model,
                        arch: &args.arch,
                        tokenizer_path,
                        run_metadata: &run_metadata,
                    },
                )?;
            } else if let Some(path) = &args.dump_layers {
                dump_layers_gemma4(
                    &backend,
                    &model,
                    &tokenizer,
                    TensorDumpConfig {
                        prompt: &args.prompt,
                        output_path: path,
                        max_seq_len: args.max_seq_len,
                        model_path: &args.model,
                        arch: &args.arch,
                        tokenizer_path,
                        run_metadata: &run_metadata,
                    },
                )?;
            } else if args.probe {
                run_probe_jobs(
                    &backend,
                    &model,
                    &tokenizer,
                    &args,
                    tokenizer_path,
                    &run_metadata,
                )?;
            } else if args.zero_layer_output.is_some()
                || args.activation_stats.is_some()
                || args.activation_patch.is_some()
                || args.capture_activations.is_some()
            {
                let model_context = ModelContext::new(
                    ModelFamily::Gemma4,
                    Some(&args.model),
                    &args.arch,
                    model.n_layers(),
                    model.embed_dim(),
                )
                .with_provenance(
                    run_metadata.model_sha256.as_deref(),
                    run_metadata.tokenizer_sha256.as_deref(),
                );
                let mut runner = build_experiment_runner(
                    &args,
                    &run_metadata,
                    &model_context,
                    &execution_inventory,
                )?;
                let runner = runner
                    .as_mut()
                    .expect("experiment or capture requested in the arm condition");
                run_single_prompt_with_experiment(
                    &backend,
                    &model,
                    &tokenizer,
                    &args,
                    model_context,
                    runner,
                )?;
            } else {
                run_single_prompt(&backend, &model, &tokenizer, &args)?;
            }
        }
        _ => anyhow::bail!("unknown architecture: {}", args.arch),
    }

    Ok(())
}

/// Build the v0.2 capture sink from `--capture-activations`, if requested.
fn build_capture_sink(
    args: &Args,
    run_metadata: &RunMetadata,
    execution_inventory: &ember::artifact::ExecutionInventory,
) -> anyhow::Result<Option<CaptureSink>> {
    let Some(path) = &args.capture_activations else {
        return Ok(None);
    };
    let sink = CaptureSink::from_toml_path(
        path,
        &args.prompt,
        rayon_current_num_threads(),
        serde_json::to_value(&run_metadata.run_manifest).unwrap_or_else(|_| serde_json::json!({})),
        run_metadata.model_sha256.clone(),
        run_metadata.tokenizer_sha256.clone(),
        run_metadata.gguf_metadata.clone(),
    )
    .map_err(anyhow::Error::msg)?
    .with_execution(execution_inventory.clone());
    eprintln!(
        "research capture active: config={path} output_dir={}",
        sink.selection().output_dir.display()
    );
    Ok(Some(sink))
}

/// Build the experiment runner (one experiment, optional capture) for a
/// generation run, or `None` when neither is requested.
fn build_experiment_runner(
    args: &Args,
    run_metadata: &RunMetadata,
    model_context: &ModelContext<'_>,
    execution_inventory: &ember::artifact::ExecutionInventory,
) -> anyhow::Result<Option<ExperimentRunner>> {
    let mut runner = if let Some(spec) = args.zero_layer_output {
        eprintln!(
            "research experiment active: zero-layer-output layer={} stage={}; execution will be modified",
            spec.layer(),
            spec.stage()
        );
        Some(ExperimentRunner::new(ZeroLayerOutput::new(spec)))
    } else if let Some(path) = &args.activation_stats {
        eprintln!(
            "research experiment active: activation-stats output={path}; execution is observation-only"
        );
        Some(ExperimentRunner::new(ActivationStats::new(path)))
    } else if let Some(source) = &args.activation_patch {
        if args.patch_target.is_empty() {
            anyhow::bail!("--activation-patch requires at least one --patch-target");
        }
        let targets = args
            .patch_target
            .iter()
            .map(|target| target.parse::<PatchTarget>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| anyhow::anyhow!("invalid --patch-target: {error}"))?;
        let experiment = ActivationPatch::new(source, targets)
            .map_err(|error| anyhow::anyhow!("activation-patch: {error}"))?;
        eprintln!(
            "research experiment active: activation-patch source={source}; execution will be modified"
        );
        Some(ExperimentRunner::new(experiment))
    } else {
        None
    };
    if let Some(sink) = build_capture_sink(args, run_metadata, execution_inventory)? {
        runner = Some(match runner {
            Some(runner) => runner.with_capture(sink),
            None => ExperimentRunner::capture_only(sink),
        });
    }
    if let Some(runner) = runner.as_mut() {
        runner.on_model_loaded(model_context)?;
    }
    Ok(runner)
}
