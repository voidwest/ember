use anyhow::Context;
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use ember::backend::Backend;
use ember::backend::CpuBackend;
use ember::extraction::{
    validate_artifact_contract, ArtifactManifest, ExecutionBackendName, ExtractionConfig,
};
use ember::loader::load_gguf;
use ember::loader::GgufLoader;
use ember::loader::GgufValue;
use ember::model::ForwardModel;
use ember::model::Gpt2;
use ember::model_backend::compare_backend_artifacts;
use ember::model_backend::run_extraction_with_backend;
use ember::model_backend::run_llama_cpp_external_backend;
use ember::model_backend::LlamaCppBackend;
use ember::model_backend::NativeModelBackend;
use ember::npy::{write_npy_2d, NpyStreamWriter};
use ember::sampler::sample_token;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// a lightweight, cpu-first llm inference engine.
#[derive(Parser)]
#[command(name = "ember", version)]
struct Args {
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

    /// model architecture: gpt2, llama, qwen3, or gemma4
    #[arg(long, default_value = "gpt2", value_parser = ["gpt2", "llama", "qwen3", "gemma4"])]
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

    /// write last-prompt logits for --prompt to a .npy file and exit
    #[arg(long, conflicts_with_all = ["demo", "interactive", "probe"])]
    dump_logits: Option<String>,

    /// dump per-layer hidden states (last prompt token) to a binary file and exit.
    /// format: f32 flat array, [n_layers * embed_dim], layer-major, native endian.
    #[arg(long, conflicts_with_all = ["demo", "interactive", "probe"])]
    dump_layers: Option<String>,

    /// probe mode: extract hidden states from each transformer block
    /// for every stimulus in the stimuli file, and save as .npy.
    #[arg(long, conflicts_with_all = ["demo", "interactive"])]
    probe: bool,

    /// path to stimuli json for probe mode
    #[arg(long, default_value = "stimuli/nonce_root_pattern.json")]
    probe_stimuli: String,

    /// output path for probe activations (.npy)
    #[arg(long, default_value = "data/activations.npy")]
    probe_output: String,

    /// prompt template key to read from each stimulus prompts object
    #[arg(long, default_value = "en_zero")]
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
}

#[derive(Subcommand)]
enum Commands {
    /// extract hidden-state artifacts through a selected execution backend
    Extract(ExtractCommand),
    /// validate one Ember artifact run directory
    ValidateRun(ValidateRunCommand),
    /// compare native and llama.cpp backend outputs where comparable
    ValidateBackends(ValidateBackendsCommand),
}

#[derive(ClapArgs)]
struct ExtractCommand {
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
struct ValidateBackendsCommand {
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
struct ValidateRunCommand {
    /// existing Ember artifact run directory
    run_dir: String,

    /// require at least one hidden-state layer shard
    #[arg(long)]
    require_layers: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
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
enum TokenPositionArg {
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

struct RunMetadata {
    gguf_metadata: serde_json::Value,
    model_file_size_bytes: Option<u64>,
    model_sha256: Option<String>,
    tokenizer_sha256: Option<String>,
    run_manifest: serde_json::Value,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    if let Some(command) = &args.command {
        return match command {
            Commands::Extract(command) => run_extract_command(command),
            Commands::ValidateRun(command) => run_validate_run_command(command),
            Commands::ValidateBackends(command) => run_validate_backends_command(command),
        };
    }

    // demo mode: suppress log noise for clean recordable output
    if args.demo {
        log::set_max_level(log::LevelFilter::Off);
    }

    // Dispatch to the selected architecture. Generation, demo, and probe paths
    // are generic over `ForwardModel`; interactive mode is still GPT-2-specific.
    let loader = load_gguf(&args.model)?;
    let n_tensors = loader.tensors.len();
    let tokenizer_path = args
        .tokenizer
        .as_deref()
        .unwrap_or_else(|| default_tokenizer_for_arch(&args.arch));
    let record_model_sha256 = args.record_model_sha256
        || args.write_run_manifest.is_some()
        || args.probe
        || args.dump_logits.is_some()
        || args.dump_layers.is_some();
    let model_sha256 = if record_model_sha256 {
        sha256_file(&args.model)
    } else {
        None
    };
    let tokenizer_sha256 = sha256_file(tokenizer_path);
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
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(tokenizer_path)?;

    match args.arch.as_str() {
        "gpt2" => {
            let model = Gpt2::from_loader(loader)?;
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
                    LogitDumpConfig {
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
                run_probe_jobs(&backend, &model, &tokenizer, &args, &run_metadata)?;
            } else {
                run_single_prompt(&backend, &model, &tokenizer, &args)?;
            }
        }
        "llama" | "qwen3" => {
            use ember::model::Llama;
            let model = Llama::from_loader_with_max_seq_len(loader, args.max_seq_len)?;
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
                    LogitDumpConfig {
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
                run_probe_jobs(&backend, &model, &tokenizer, &args, &run_metadata)?;
            } else {
                run_single_prompt(&backend, &model, &tokenizer, &args)?;
            }
        }
        "gemma4" => {
            use ember::gemma4::Gemma4;
            let model = Gemma4::from_loader(loader)?;
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
                    LogitDumpConfig {
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
                    &args.prompt,
                    path,
                    args.max_seq_len,
                )?;
            } else if args.probe {
                run_probe_jobs(&backend, &model, &tokenizer, &args, &run_metadata)?;
            } else {
                run_single_prompt(&backend, &model, &tokenizer, &args)?;
            }
        }
        _ => anyhow::bail!("unknown architecture: {}", args.arch),
    }

    Ok(())
}

fn default_tokenizer_for_arch(arch: &str) -> &'static str {
    match arch {
        "gpt2" => "tokenizer-gpt2.json",
        "llama" => "tokenizer.json",
        "qwen3" => "tokenizer-qwen3.json",
        "gemma4" => "tokenizer-gemma4.json",
        _ => "tokenizer.json",
    }
}

fn run_extract_command(command: &ExtractCommand) -> anyhow::Result<()> {
    let config = build_extraction_config(command)?;
    config.validate()?;

    match config.backend {
        ExecutionBackendName::Native => run_native_extract_command(&config),
        ExecutionBackendName::LlamaCpp => {
            let _backend = LlamaCppBackend::from_config(&config)?;
            anyhow::bail!(
                "llama-cpp backend not implemented for hidden-state extraction yet; \
                 config '{}' is valid, but Ember still needs the external patched/custom \
                 llama.cpp extraction binary integration",
                command.config.as_deref().unwrap_or("<direct>")
            )
        }
        ExecutionBackendName::LlamaCppExternal => run_llama_cpp_external_extract_command(&config),
    }
}

fn run_validate_backends_command(command: &ValidateBackendsCommand) -> anyhow::Result<()> {
    if let (Some(native_run), Some(external_run)) = (&command.native_run, &command.external_run) {
        let report = compare_backend_artifacts(native_run, external_run)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let _ = (&command.model, &command.prompts, &command.layers);
    anyhow::bail!("validate-backends requires --native-run and --external-run for Milestone 3")
}

fn run_validate_run_command(command: &ValidateRunCommand) -> anyhow::Result<()> {
    let summary = validate_artifact_contract(&command.run_dir, !command.require_layers)?;
    let manifest_path = std::path::Path::new(&command.run_dir).join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read manifest: {}", manifest_path.display()))?;
    let manifest_value: serde_json::Value = serde_json::from_str(&manifest_text)
        .with_context(|| format!("failed to parse manifest: {}", manifest_path.display()))?;
    let manifest: ArtifactManifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("failed to parse manifest: {}", manifest_path.display()))?;

    if manifest.backend.name.trim().is_empty() {
        anyhow::bail!("manifest backend.name is empty");
    }
    if manifest.artifact_kind.trim().is_empty() {
        anyhow::bail!("manifest artifact_kind is empty");
    }
    let config_path = std::path::Path::new(&command.run_dir).join(&manifest.config_path);
    if !config_path.is_file() {
        anyhow::bail!("manifest config_path is missing: {}", config_path.display());
    }

    let report_path = std::path::Path::new(&command.run_dir).join(&manifest.report_path);
    let report_text = fs::read_to_string(&report_path)
        .with_context(|| format!("failed to read report: {}", report_path.display()))?;
    let report_value: serde_json::Value = serde_json::from_str(&report_text)
        .with_context(|| format!("failed to parse report: {}", report_path.display()))?;
    validate_report_fields(&manifest, &report_value)?;

    let markers = collect_run_markers(&manifest_value, &report_value);
    validate_run_markers(&summary, &markers)?;

    let output = serde_json::json!({
        "kind": "validate_run",
        "status": "pass",
        "run_dir": summary.run_dir,
        "artifact_kind": manifest.artifact_kind,
        "backend": {
            "name": manifest.backend.name,
            "version": manifest.backend.version,
            "executable": manifest.backend.executable,
        },
        "sample_count": summary.sample_count,
        "layer_count": summary.layer_count,
        "logits_present": summary.logits_present,
        "sample_order_hash": summary.sample_order_hash,
        "markers": markers,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn validate_report_fields(
    manifest: &ArtifactManifest,
    report: &serde_json::Value,
) -> anyhow::Result<()> {
    if report.get("status").and_then(serde_json::Value::as_str) != Some("complete") {
        anyhow::bail!("report status is not complete");
    }
    if let Some(schema_version) = report
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
    {
        if schema_version != u64::from(manifest.schema_version) {
            anyhow::bail!(
                "report schema_version {} does not match manifest schema_version {}",
                schema_version,
                manifest.schema_version
            );
        }
    }
    if let Some(layout) = report.get("layout").and_then(serde_json::Value::as_str) {
        if layout != manifest.layout {
            anyhow::bail!(
                "report layout '{}' does not match manifest layout '{}'",
                layout,
                manifest.layout
            );
        }
    }
    Ok(())
}

fn collect_run_markers(
    manifest: &serde_json::Value,
    report: &serde_json::Value,
) -> serde_json::Map<String, serde_json::Value> {
    let marker_names = [
        "mock",
        "mock_backend",
        "no_inference",
        "real_llama_cpp",
        "real_tokenization",
        "no_generation",
        "no_logits",
        "no_hidden_states",
        "not_research_output",
    ];
    let mut markers = serde_json::Map::new();
    for name in marker_names {
        let mut observed = Vec::new();
        collect_marker_values(manifest, name, &mut observed);
        collect_marker_values(report, name, &mut observed);
        if observed.is_empty() {
            continue;
        }
        let first = observed[0];
        if observed.iter().any(|value| *value != first) {
            markers.insert(
                name.to_string(),
                serde_json::Value::String("conflict".to_string()),
            );
        } else {
            markers.insert(name.to_string(), serde_json::Value::Bool(first));
        }
    }
    markers
}

fn collect_marker_values(value: &serde_json::Value, name: &str, observed: &mut Vec<bool>) {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(marker) = map.get(name) {
                if let Some(bool_value) = marker.as_bool() {
                    observed.push(bool_value);
                }
            }
            for key in [
                "provenance",
                "run_metadata",
                "details",
                "backend",
                "extraction_config",
            ] {
                if let Some(child) = map.get(key) {
                    collect_marker_values(child, name, observed);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_marker_values(item, name, observed);
            }
        }
        _ => {}
    }
}

fn validate_run_markers(
    summary: &ember::extraction::ArtifactValidationSummary,
    markers: &serde_json::Map<String, serde_json::Value>,
) -> anyhow::Result<()> {
    for (name, value) in markers {
        if value.as_bool().is_none() {
            anyhow::bail!("metadata marker '{name}' has conflicting values");
        }
    }
    if marker_is_true(markers, "no_logits") && summary.logits_present {
        anyhow::bail!("metadata marker no_logits=true conflicts with present logits artifact");
    }
    if marker_is_true(markers, "no_hidden_states") && summary.layer_count > 0 {
        anyhow::bail!(
            "metadata marker no_hidden_states=true conflicts with {} layer shard(s)",
            summary.layer_count
        );
    }
    if marker_is_true(markers, "mock") && !marker_is_true(markers, "not_research_output") {
        anyhow::bail!("mock run must be marked not_research_output=true");
    }
    if marker_is_true(markers, "mock_backend") && !marker_is_true(markers, "not_research_output") {
        anyhow::bail!("mock backend run must be marked not_research_output=true");
    }
    Ok(())
}

fn marker_is_true(markers: &serde_json::Map<String, serde_json::Value>, name: &str) -> bool {
    markers
        .get(name)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn build_extraction_config(command: &ExtractCommand) -> anyhow::Result<ExtractionConfig> {
    let mut config = if let Some(path) = &command.config {
        ExtractionConfig::from_path(path)?
    } else {
        ExtractionConfig {
            run_id: None,
            model_path: command
                .model
                .clone()
                .context("extract direct mode requires --model")?,
            architecture: command.arch.clone(),
            tokenizer_path: command.tokenizer.clone(),
            backend: command
                .backend
                .map(ExecutionBackendName::from)
                .unwrap_or(ExecutionBackendName::Native),
            prompt_template: command
                .prompt_template
                .clone()
                .unwrap_or_else(|| "{prompt}".to_string()),
            input_jsonl_path: command
                .samples
                .clone()
                .context("extract direct mode requires --samples")?,
            output_dir: command
                .out
                .clone()
                .context("extract direct mode requires --out")?,
            layers: parse_layers_list(command.layers.as_deref())?,
            token_position: command
                .token_position
                .map(ember::extraction::TokenPositionMode::from)
                .unwrap_or(ember::extraction::TokenPositionMode::PromptFinal),
            word_field: command
                .word_field
                .clone()
                .unwrap_or_else(|| "word".to_string()),
            sample_id_field: command
                .sample_id_field
                .clone()
                .unwrap_or_else(|| "id".to_string()),
            batch_size: 1,
            dtype: ember::extraction::ArtifactDType::F32,
            output_format: ember::extraction::ArtifactOutputFormat::Npy,
            prompt_hashes_only: false,
            write_logits: command.write_logits,
            resume: false,
            max_seq_len: None,
            record_model_sha256: false,
            llama_cpp_binary: command.llama_bin.clone(),
            run_metadata: serde_json::Value::Null,
        }
    };

    if let Some(backend) = command.backend {
        config.backend = ExecutionBackendName::from(backend);
    }
    if let Some(llama_bin) = &command.llama_bin {
        config.llama_cpp_binary = Some(llama_bin.clone());
    }
    if let Some(model) = &command.model {
        config.model_path = model.clone();
    }
    if let Some(samples) = &command.samples {
        config.input_jsonl_path = samples.clone();
    }
    if let Some(out) = &command.out {
        config.output_dir = out.clone();
        config.run_id = None;
    }
    if let Some(template) = &command.prompt_template {
        config.prompt_template = template.clone();
    }
    if let Some(arch) = &command.arch {
        config.architecture = Some(arch.clone());
    }
    if let Some(tokenizer) = &command.tokenizer {
        config.tokenizer_path = Some(tokenizer.clone());
    }
    if command.layers.is_some() {
        config.layers = parse_layers_list(command.layers.as_deref())?;
    }
    if let Some(position) = command.token_position {
        config.token_position = ember::extraction::TokenPositionMode::from(position);
    }
    if let Some(sample_id_field) = &command.sample_id_field {
        config.sample_id_field = sample_id_field.clone();
    }
    if let Some(word_field) = &command.word_field {
        config.word_field = word_field.clone();
    }
    if command.write_logits {
        config.write_logits = true;
    }
    Ok(config)
}

fn run_native_extract_command(config: &ExtractionConfig) -> anyhow::Result<()> {
    let loader = load_gguf(&config.model_path)?;
    let gguf_metadata = gguf_metadata_json(&loader);
    let arch = infer_extraction_architecture(config, &gguf_metadata);
    let tokenizer_path = config
        .tokenizer_path
        .as_deref()
        .unwrap_or_else(|| default_tokenizer_for_arch(&arch));
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(tokenizer_path)?;

    match arch.as_str() {
        "gpt2" => {
            let model = Gpt2::from_loader(loader)?;
            run_native_extract_for_model(model, tokenizer, config, &arch, gguf_metadata)
        }
        "llama" | "qwen3" => {
            use ember::model::Llama;
            let model = Llama::from_loader_with_max_seq_len(loader, config.max_seq_len)?;
            run_native_extract_for_model(model, tokenizer, config, &arch, gguf_metadata)
        }
        "gemma4" => {
            use ember::gemma4::Gemma4;
            let model = Gemma4::from_loader(loader)?;
            run_native_extract_for_model(model, tokenizer, config, &arch, gguf_metadata)
        }
        _ => anyhow::bail!(
            "unsupported native extraction architecture '{}'; set architecture to gpt2, llama, qwen3, or gemma4",
            arch
        ),
    }
}

fn run_native_extract_for_model<M>(
    model: M,
    tokenizer: ember::tokenizer::EmberTokenizer,
    config: &ExtractionConfig,
    arch: &str,
    gguf_metadata: serde_json::Value,
) -> anyhow::Result<()>
where
    M: ForwardModel<CpuBackend>,
    <CpuBackend as Backend>::Error: Send + Sync + 'static,
{
    let mut backend = NativeModelBackend::new(
        model,
        tokenizer,
        &config.model_path,
        Some(arch.to_string()),
        gguf_metadata,
        config.record_model_sha256,
    );
    let output = run_extraction_with_backend(&mut backend, config)?;
    eprintln!(
        "wrote {} sample(s) to {} with {} layer shard(s)",
        output.sample_count,
        output.run_dir,
        output.layer_paths.len()
    );
    eprintln!("manifest: {}", output.manifest_path);
    eprintln!("samples: {}", output.samples_path);
    eprintln!("tokenization: {}", output.tokenization_path);
    eprintln!("positions: {}", output.positions_path);
    eprintln!("checksums: {}", output.checksums_path);
    eprintln!("report: {}", output.report_path);
    Ok(())
}

fn run_llama_cpp_external_extract_command(config: &ExtractionConfig) -> anyhow::Result<()> {
    let output = run_llama_cpp_external_backend(config)?;
    eprintln!(
        "llama-cpp-external wrote {} sample(s) to {}",
        output.sample_count, output.run_dir
    );
    eprintln!("manifest: {}", output.manifest_path);
    eprintln!("samples: {}", output.samples_path);
    eprintln!("tokenization: {}", output.tokenization_path);
    eprintln!("positions: {}", output.positions_path);
    eprintln!("checksums: {}", output.checksums_path);
    eprintln!("report: {}", output.report_path);
    Ok(())
}

fn parse_layers_list(value: Option<&str>) -> anyhow::Result<Vec<usize>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .with_context(|| format!("invalid layer index '{s}'"))
        })
        .collect()
}

fn infer_extraction_architecture(
    config: &ExtractionConfig,
    gguf_metadata: &serde_json::Value,
) -> String {
    config
        .architecture
        .clone()
        .or_else(|| {
            gguf_metadata
                .get("general.architecture")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "gpt2".to_string())
}

fn parse_temperature(value: &str) -> Result<f32, String> {
    let temperature = value
        .parse::<f32>()
        .map_err(|_| format!("invalid temperature '{value}'"))?;
    if temperature.is_finite() && temperature >= 0.0 {
        Ok(temperature)
    } else {
        Err("temperature must be a finite number >= 0".to_string())
    }
}

fn parse_top_k(value: &str) -> Result<usize, String> {
    let top_k = value
        .parse::<usize>()
        .map_err(|_| format!("invalid top-k '{value}'"))?;
    if top_k > 0 {
        Ok(top_k)
    } else {
        Err("top-k must be greater than 0".to_string())
    }
}

fn parse_top_p(value: &str) -> Result<f32, String> {
    let top_p = value
        .parse::<f32>()
        .map_err(|_| format!("invalid top-p '{value}'"))?;
    if top_p.is_finite() && top_p > 0.0 && top_p <= 1.0 {
        Ok(top_p)
    } else {
        Err("top-p must be in the range (0, 1]".to_string())
    }
}

fn parse_max_seq_len(value: &str) -> Result<usize, String> {
    let max_seq_len = value
        .parse::<usize>()
        .map_err(|_| format!("invalid max sequence length '{value}'"))?;
    if max_seq_len > 0 {
        Ok(max_seq_len)
    } else {
        Err("max sequence length must be greater than 0".to_string())
    }
}

fn effective_context_limit<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    args: &Args,
) -> usize {
    match args.max_seq_len {
        Some(cap) => cap.min(model.max_seq_len(backend)),
        None => model.max_seq_len(backend),
    }
}

fn ensure_sequence_fits(
    prompt_len: usize,
    max_tokens: usize,
    context_limit: usize,
) -> anyhow::Result<usize> {
    let requested = prompt_len
        .checked_add(max_tokens)
        .context("requested sequence length overflowed usize")?;
    if requested > context_limit {
        anyhow::bail!(
            "requested sequence length {} exceeds context limit {} (prompt tokens {} + generation tokens {})",
            requested,
            context_limit,
            prompt_len,
            max_tokens
        );
    }
    Ok(requested)
}

fn run_single_prompt<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    args: &Args,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let output = generate(
        backend,
        model,
        tokenizer,
        &args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.benchmark,
        effective_context_limit(backend, model, args),
    )?;
    println!("{}", output);
    Ok(())
}

/// run a curated demo showcasing the project.
///
/// uses greedy sampling (temperature 0) for deterministic, repeatable output -
/// ideal for screen recordings, benchmarks, and project demonstrations.
/// runs through a fixed set of prompts, printing each one with its completion
/// and per-prompt timing, then a summary table.
///
/// when `delay_ms > 0`, tokens are streamed one at a time with a typewriter
/// effect. ansi colors are used for visual distinction (`--color` cli flag or
/// terminal detection can be added to toggle).
fn demo_mode<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    max_tokens: usize,
    model_path: &str,
    delay_ms: u64,
    context_limit: usize,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    // -- ansi style helpers ----------------------------------------------
    // simple string concatenation to avoid macro complexity.
    // each "style" builder returns a formatted string with escape codes.
    const RST: &str = "\x1b[0m";
    const BLD: &str = "\x1b[1m";
    const DIM: &str = "\x1b[2m";
    const CYN: &str = "\x1b[36m";
    const GRN: &str = "\x1b[32m";
    const YLW: &str = "\x1b[33m";

    fn s(ansi: &str, text: &dyn std::fmt::Display) -> String {
        format!("{ansi}{text}{RST}")
    }
    fn s2(a: &str, b: &str, text: &dyn std::fmt::Display) -> String {
        format!("{a}{b}{text}{RST}")
    }

    // eprintln / print without newline helpers so we don't forget io::stdout().flush()
    macro_rules! eprint_flush { ($($arg:tt)*) => {{
        eprint!($($arg)*);
        let _ = io::stderr().flush();
    }}; }
    macro_rules! print_flush { ($($arg:tt)*) => {{
        print!($($arg)*);
        let _ = io::stdout().flush();
    }}; }

    let embed_dim = model.embed_dim();

    // -- header ------------------------------------------------------
    let header_border = s2(
        DIM,
        CYN,
        &"+--------------------------------------------------+",
    );
    let header_line = s2(
        BLD,
        CYN,
        &"|              ember  -  llm inference             |",
    );
    let header_sep = s2(
        DIM,
        CYN,
        &"+--------------------------------------------------+",
    );

    println!("{header_border}");
    println!("{header_line}");
    println!("{header_sep}");

    let kv = |k: &str, v: &dyn std::fmt::Display| {
        println!(
            "{} {} {:>37} {}",
            s2(DIM, CYN, &"|"),
            s(DIM, &k),
            s(BLD, &v),
            s2(DIM, CYN, &"|"),
        );
    };
    kv("model     ", &model_path);
    kv("layers    ", &model.n_layers());
    kv("embed_dim ", &embed_dim);
    kv("vocab     ", &tokenizer.vocab_size());
    kv("sampling  ", &"greedy (temp=0)");

    let header_foot = s2(
        DIM,
        CYN,
        &"+--------------------------------------------------+",
    );
    println!("{header_foot}");

    if delay_ms > 0 {
        println!();
        println!(
            "{}",
            s(
                DIM,
                &format!("  typewriter delay: {delay_ms} ms/token - press ctrl-c to exit")
            ),
        );
    }
    println!();

    let prompts: &[(&str, &str)] = &[
        ("Once upon a time, in a land far away,", "story generation"),
        (
            "The three primary colors of light are",
            "factual completion",
        ),
        (
            "// fibonacci sequence in python\ndef fib(n):",
            "code generation",
        ),
        ("The meaning of life is", "open-ended reasoning"),
    ];

    let spinner_chars = ['|', '/', '-', '\\'];

    let mut total_prefill_ms = 0.0;
    let mut total_decode_ms = 0.0;
    let mut total_prompt_tokens = 0usize;
    let mut total_generated = 0usize;

    for (i, (prompt, category)) in prompts.iter().enumerate() {
        let prompt_tokens = tokenizer.encode(prompt)?;
        let prompt_len = prompt_tokens.len();
        let max_seq_len = ensure_sequence_fits(prompt_len, max_tokens, context_limit)?;

        // -- prefill with spinner ----------------------------------
        let prefill_start = std::time::Instant::now();
        eprint_flush!(
            "{}  {}{}",
            s(CYN, &"*"),
            s(DIM, &"prefilling... "),
            spinner_chars[0],
        );

        let mut cache = model.create_cache(backend, max_seq_len);
        let mut logits =
            model.forward_last_logits_with_cache(backend, &prompt_tokens, &mut cache, 0)?;
        let vocab_size = backend.shape(&logits)[1];

        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        eprint_flush!("\r{}\n", s(GRN, &"prefill complete"));

        // -- decode with typewriter streaming ----------------------
        let decode_start = std::time::Instant::now();
        let mut generated = Vec::with_capacity(max_tokens);
        let eos_ids = tokenizer.eos_token_ids();

        // print prompt card
        println!();
        let pn = i + 1;
        let card_width: usize = 50;
        let top_prefix = format!("+- prompt {pn} - {category} - ");
        let pad_len = card_width.saturating_sub(top_prefix.chars().count() + 1);
        let dashes = "-".repeat(pad_len);
        println!("{}", s2(BLD, CYN, &format!("{top_prefix}{dashes}+")),);
        println!("{}", s(DIM, &"|"));
        println!("{} {}", s(DIM, &"| prompt:    "), s(YLW, &prompt),);
        print_flush!(
            "{} {}",
            s(DIM, &"| completion:"),
            GRN, // start completion on a new line, green
        );

        for step in 0..max_tokens {
            let logit_data = backend.data(&logits);
            let last_logits = &logit_data[..vocab_size];

            let next = argmax_token(last_logits);

            if eos_ids.contains(&(next as u32)) {
                break;
            }

            generated.push(next as u32);

            // stream this single token now, before computing the next.
            // individual subword tokens may decode to replacement characters
            // (U+FFFD) when they're part of a multi-token UTF-8 sequence;
            // filter those out so the typewriter effect stays clean.
            let token_text = tokenizer.decode(&[next as u32])?;
            let cleaned: String = token_text.chars().filter(|c| *c != '\u{FFFD}').collect();
            if !cleaned.is_empty() {
                print_flush!("{}", cleaned);
            }

            if delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }

            logits = model.forward_last_logits_with_cache(
                backend,
                &[next as u32],
                &mut cache,
                prompt_len + step,
            )?;
        }
        // reset color after completion
        println!("{RST}");
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        // -- per-prompt stats -------------------------------------
        println!("{}", s(DIM, &"|"));
        println!(
            "{} {} prompt + {} generated = {} total",
            s(DIM, &"| tokens:    "),
            prompt_len,
            generated.len(),
            prompt_len + generated.len(),
        );
        println!(
            "{} {:.1} ms ({:.0} tok/s)",
            s(DIM, &"| prefill:   "),
            prefill_ms,
            prompt_len as f64 / (prefill_ms / 1000.0)
        );
        println!(
            "{} {:.1} ms ({:.0} tok/s)",
            s(DIM, &"| decode:    "),
            decode_ms,
            generated.len() as f64 / (decode_ms / 1000.0)
        );
        println!(
            "{}",
            s2(
                DIM,
                CYN,
                &"+------------------------------------------------+"
            )
        );
        println!();

        total_prefill_ms += prefill_ms;
        total_decode_ms += decode_ms;
        total_prompt_tokens += prompt_len;
        total_generated += generated.len();

        // brief pause between prompts so the viewer can absorb
        if delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms * 5));
        }
    }

    // -- summary ----------------------------------------------------
    let total_ms = total_prefill_ms + total_decode_ms;
    let total_tokens = total_prompt_tokens + total_generated;

    println!(
        "{}",
        s2(
            BLD,
            YLW,
            &"=========================== summary =========================="
        ),
    );
    println!();
    println!("  prompts:       {}", prompts.len());
    println!(
        "  total tokens:  {} ({} prompt + {} generated)",
        total_tokens, total_prompt_tokens, total_generated
    );
    println!("  total time:    {:.1} ms", total_ms);
    println!(
        "  throughput:    {:.0} tok/s",
        total_tokens as f64 / (total_ms / 1000.0)
    );
    println!(
        "  prefill avg:   {:.1} ms - {:.0} tok/s",
        total_prefill_ms / prompts.len() as f64,
        total_prompt_tokens as f64 / (total_prefill_ms / 1000.0)
    );
    println!(
        "  decode avg:    {:.1} ms - {:.0} tok/s",
        total_decode_ms / prompts.len() as f64,
        total_generated as f64 / (total_decode_ms / 1000.0)
    );
    println!();
    println!(
        "{}",
        s2(
            DIM,
            YLW,
            &"=============================================================="
        ),
    );
    println!();

    // -- end-of-demo flicker ------------------------------------
    // prints a blinking cursor effect that persists for ~2 seconds
    // so the viewer knows the demo is complete and the terminal is
    // still live.
    if delay_ms > 0 {
        print_flush!("{}", s(DIM, &"demo complete. "));
        let cursor_chars = ['|', ' '];
        let flicker_start = std::time::Instant::now();
        let mut flicker_idx = 0usize;
        while flicker_start.elapsed().as_secs() < 2 {
            print_flush!(
                "\r{} {}",
                s(DIM, &"demo complete. "),
                cursor_chars[flicker_idx % 2],
            );
            flicker_idx += 1;
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        // clear the cursor line
        print_flush!("\r{}\r", s(DIM, &"demo complete."));
    }

    Ok(())
}

/// run the full autoregressive generation loop.
///
/// operates in two phases:
/// 1. **prefill** - feeds the entire prompt through the model in one forward pass,
///    populating the kv cache with key/value projections for all prompt tokens.
/// 2. **decode** - generates one token at a time: samples from the last position's
///    logits, appends it, and runs a single-token forward pass reusing the cached
///    k/v from all previous positions. stops when `max_tokens` is reached or a
///    tokenizer-defined eos token is predicted.
///
/// temperature 0.0 uses greedy argmax; any positive value enables temperature
/// scaling with optional top-k and top-p filtering via [`sample_token`].
#[allow(clippy::too_many_arguments)]
fn generate<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    benchmark: bool,
    context_limit: usize,
) -> anyhow::Result<String>
where
    B::Error: Send + Sync + 'static,
{
    let mut rng = rand::thread_rng();

    let mut all_tokens = tokenizer
        .encode(prompt)
        .context("failed to tokenize prompt")?;
    log::info!("prompt has {} tokens", all_tokens.len());

    let prompt_len = all_tokens.len();
    let max_seq_len = ensure_sequence_fits(prompt_len, max_tokens, context_limit)?;

    // -- 1. prefill: run the prompt through the transformer and fill kv cache.
    // Only the last prompt position needs logits for generation, so avoid
    // materializing a full [prompt_len, vocab_size] logits tensor.
    let prefill_start = if benchmark {
        Some(Instant::now())
    } else {
        None
    };
    log::info!("prefilling KV cache for {} tokens", prompt_len);
    let mut cache = model.create_cache(backend, max_seq_len);
    let mut logits = model.forward_last_logits_with_cache(backend, &all_tokens, &mut cache, 0)?;
    let prefill_elapsed = prefill_start.map(|s| s.elapsed());
    let vocab_size = backend.shape(&logits)[1];

    // -- 2. decode loop: one new token at a time --------------------------
    let decode_start = if benchmark {
        Some(Instant::now())
    } else {
        None
    };
    let mut generated = Vec::with_capacity(max_tokens);
    let mut next_token: usize;

    for step in 0..max_tokens {
        let logit_data = backend.data(&logits);
        let last_logits = &logit_data[..vocab_size];

        next_token = if temperature == 0.0 {
            argmax_token(last_logits)
        } else {
            sample_token(last_logits, temperature, top_k, top_p, &mut rng)
        };

        log::debug!("step {}: predicted token {}", step, next_token);

        let eos_ids = tokenizer.eos_token_ids();
        if eos_ids.contains(&(next_token as u32)) {
            log::info!("eos token reached after {} generated tokens", step);
            break;
        }

        all_tokens.push(next_token as u32);
        generated.push(next_token as u32);

        // decode step: forward with just the new token, using cached K/V
        logits = model.forward_last_logits_with_cache(
            backend,
            &[next_token as u32],
            &mut cache,
            prompt_len + step, // absolute position offset
        )?;
    }

    let output = tokenizer.decode(&generated)?;

    if benchmark {
        let prefill_ms = prefill_elapsed.unwrap().as_secs_f64() * 1000.0;
        let decode_ms = decode_start.unwrap().elapsed().as_secs_f64() * 1000.0;
        eprintln!("--- benchmark ---");
        eprintln!(
            "prefill: {} tokens in {:.1}ms -> {:.0} tok/s",
            prompt_len,
            prefill_ms,
            prompt_len as f64 / prefill_elapsed.unwrap().as_secs_f64()
        );
        eprintln!(
            "decode:  {} tokens in {:.1}ms -> {:.0} tok/s",
            generated.len(),
            decode_ms,
            generated.len() as f64 / decode_start.unwrap().elapsed().as_secs_f64()
        );
    }

    if log::log_enabled!(log::Level::Debug) {
        let decoded_prompt = tokenizer.decode(&all_tokens[..prompt_len])?;
        log::debug!("prompt: {:?}", decoded_prompt);
        log::debug!("generated: {:?}", output);
    }

    Ok(output)
}

fn dump_last_logits<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    config: LogitDumpConfig<'_>,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let token_ids = tokenizer
        .encode(config.prompt)
        .context("failed to tokenize prompt")?;
    let (offset_ids, offsets) = tokenizer
        .encode_with_offsets(config.prompt)
        .context("failed to tokenize prompt with offsets")?;
    if offset_ids != token_ids {
        anyhow::bail!("token audit failed: encode and encode_with_offsets emitted different ids");
    }
    if token_ids.is_empty() {
        anyhow::bail!("cannot dump logits for an empty prompt");
    }
    let context_limit = config
        .max_seq_len
        .unwrap_or_else(|| model.max_seq_len(backend));
    ensure_sequence_fits(token_ids.len(), 0, context_limit)?;
    let mut cache = model.create_cache(backend, context_limit);
    let logits = model.forward_last_logits_with_cache(backend, &token_ids, &mut cache, 0)?;
    let shape = backend.shape(&logits);
    if shape.len() != 2 || shape[0] != 1 {
        anyhow::bail!("expected last logits shape [1, vocab], got {:?}", shape);
    }
    write_npy_2d(
        config.output_path,
        backend.data(&logits),
        &[shape[0], shape[1]],
    )?;
    let metadata_path = config.output_path.replace(".npy", "_metadata.json");
    let metadata = serde_json::json!({
        "model_path": config.model_path,
        "architecture": config.arch,
        "tokenizer_path": config.tokenizer_path,
        "tokenizer_sha256": config.run_metadata.tokenizer_sha256,
        "model_file_size_bytes": config.run_metadata.model_file_size_bytes,
        "model_sha256": config.run_metadata.model_sha256,
        "gguf_metadata": config.run_metadata.gguf_metadata,
        "output_path": config.output_path,
        "prompt": config.prompt,
        "context_limit": context_limit,
        "logits_shape": [shape[0], shape[1]],
        "token_audit": token_audit_json(
            config.prompt,
            config.tokenizer_path,
            config.run_metadata.tokenizer_sha256.as_deref(),
            tokenizer.bos_token_id(),
            &token_ids,
            &offsets,
        ),
        "run_manifest": config.run_metadata.run_manifest,
    });
    write_json_file(&metadata_path, &metadata)?;
    eprintln!(
        "saved last logits for {} prompt tokens to {} with shape {:?}",
        token_ids.len(),
        config.output_path,
        shape
    );
    eprintln!("saved logits metadata to {}", metadata_path);
    Ok(())
}

fn bail_dump_layers_unsupported(arch: &str) -> anyhow::Result<()> {
    anyhow::bail!("--dump-layers is only supported with --arch gemma4, got --arch {arch}")
}

/// Dump per-layer hidden states (last prompt token) directly to a binary file.
///
/// ## Binary output format
///
///   dtype:      f32 (native endian)
///   shape:      [n_layers * embed_dim] flat, layer-major
///   layer count: model n_layers
///   hidden size: model embed_dim
///   row order:   layer 0 first, layer (n_layers-1) last
///
/// Boundary: after each block's final residual add and layer_output_scale.
/// Matches llama.cpp per-layer dump point (after `build_cvec` in gemma4.cpp).
fn dump_layers_gemma4<B: Backend>(
    backend: &B,
    model: &ember::gemma4::Gemma4<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    prompt: &str,
    output_path: &str,
    max_seq_len: Option<usize>,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let token_ids = tokenizer
        .encode(prompt)
        .context("failed to tokenize prompt")?;
    if token_ids.is_empty() {
        anyhow::bail!("cannot dump layers for an empty prompt");
    }
    let context_limit = max_seq_len.unwrap_or_else(|| model.max_seq_len(backend));
    ensure_sequence_fits(token_ids.len(), 0, context_limit)?;
    let mut cache = model.create_cache(backend, context_limit);
    let (layer_states, _logits) =
        model.forward_last_logits_with_layer_dump(backend, &token_ids, &mut cache, 0)?;
    let embed_dim = model.config.embed_dim;
    let n_layers = layer_states.len();
    let flat: Vec<f32> = layer_states.into_iter().flatten().collect();
    assert_eq!(flat.len(), n_layers * embed_dim);
    let bytes = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
    std::fs::write(output_path, bytes)?;
    eprintln!(
        "saved {} layers × {} hidden = {} floats to {}",
        n_layers,
        embed_dim,
        flat.len(),
        output_path
    );
    Ok(())
}

/// greedy argmax: return the index of the largest logit value.
#[inline]
fn argmax_token(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tokenizer_tracks_architecture() {
        let gpt2 = Args::try_parse_from(["ember"]).expect("default args should parse");
        assert_eq!(
            gpt2.tokenizer
                .as_deref()
                .unwrap_or_else(|| default_tokenizer_for_arch(&gpt2.arch)),
            "tokenizer-gpt2.json"
        );

        let llama =
            Args::try_parse_from(["ember", "--arch", "llama"]).expect("llama args should parse");
        assert_eq!(
            llama
                .tokenizer
                .as_deref()
                .unwrap_or_else(|| default_tokenizer_for_arch(&llama.arch)),
            "tokenizer.json"
        );

        let gemma4 =
            Args::try_parse_from(["ember", "--arch", "gemma4"]).expect("gemma4 args should parse");
        assert_eq!(
            gemma4
                .tokenizer
                .as_deref()
                .unwrap_or_else(|| default_tokenizer_for_arch(&gemma4.arch)),
            "tokenizer-gemma4.json"
        );

        let qwen3 =
            Args::try_parse_from(["ember", "--arch", "qwen3"]).expect("qwen3 args should parse");
        assert_eq!(
            qwen3
                .tokenizer
                .as_deref()
                .unwrap_or_else(|| default_tokenizer_for_arch(&qwen3.arch)),
            "tokenizer-qwen3.json"
        );
    }

    #[test]
    fn cli_rejects_invalid_sampling_args() {
        assert!(Args::try_parse_from(["ember", "--temperature", "-0.1"]).is_err());
        assert!(Args::try_parse_from(["ember", "--top-k", "0"]).is_err());
        assert!(Args::try_parse_from(["ember", "--top-p", "0"]).is_err());
        assert!(Args::try_parse_from(["ember", "--top-p", "1.1"]).is_err());
        assert!(Args::try_parse_from(["ember", "--max-seq-len", "0"]).is_err());
    }
}

#[allow(clippy::too_many_arguments)]
fn interactive_mode<B: Backend>(
    backend: &B,
    model: &Gpt2<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    _initial_prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    max_seq_len: Option<usize>,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    println!("ember interactive mode. type /quit to exit, /help for commands.");
    println!("max tokens per turn: {}", max_tokens);

    // warm-up with the initial prompt
    print!("> ");
    io::stdout().flush()?;

    loop {
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break; // ctrl-d
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            print!("> ");
            io::stdout().flush()?;
            continue;
        }

        match line.as_str() {
            "/quit" | "/exit" => break,
            "/help" => {
                println!("/help   show this message");
                println!("/quit   exit interactive mode");
                println!("/stats  show model info");
                print!("> ");
                io::stdout().flush()?;
                continue;
            }
            "/stats" => {
                log::info!(
                    "wte shape: {:?}, blocks: {}",
                    backend.shape(&model.wte),
                    model.blocks.len()
                );
                print!("> ");
                io::stdout().flush()?;
                continue;
            }
            prompt => {
                let output = generate(
                    backend,
                    model,
                    tokenizer,
                    prompt,
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    false, // benchmark not meaningful in interactive mode
                    max_seq_len.unwrap_or_else(|| {
                        <Gpt2<B> as ForwardModel<B>>::max_seq_len(model, backend)
                    }),
                )?;
                println!("{}", output);
                print!("> ");
                io::stdout().flush()?;
            }
        }
    }

    Ok(())
}

// -- probe mode -------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum ProbePosition {
    Last,
    Root,
    Pattern,
    PromptMean,
}

struct ProbeJob {
    template: String,
    position: ProbePosition,
    output_path: String,
}

struct ProbeOutput {
    position: ProbePosition,
    output_path: String,
}

struct ProbeGroupConfig<'a> {
    stimuli_path: &'a str,
    template: &'a str,
    outputs: Vec<ProbeOutput>,
    generate_tokens: usize,
    limit: Option<usize>,
    context_limit: usize,
    model_path: &'a str,
    arch: &'a str,
    tokenizer_path: &'a str,
    run_metadata: &'a RunMetadata,
}

struct LogitDumpConfig<'a> {
    prompt: &'a str,
    output_path: &'a str,
    max_seq_len: Option<usize>,
    model_path: &'a str,
    arch: &'a str,
    tokenizer_path: &'a str,
    run_metadata: &'a RunMetadata,
}

impl ProbePosition {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "last" => Ok(Self::Last),
            "root" => Ok(Self::Root),
            "pattern" => Ok(Self::Pattern),
            "prompt_mean" => Ok(Self::PromptMean),
            _ => anyhow::bail!(
                "unknown probe position '{}'; expected last, root, pattern, or prompt_mean",
                value
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Last => "last",
            Self::Root => "root",
            Self::Pattern => "pattern",
            Self::PromptMean => "prompt_mean",
        }
    }
}

fn split_probe_list(value: Option<&String>, fallback: &str) -> Vec<String> {
    value
        .map(|s| s.as_str())
        .unwrap_or(fallback)
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

fn sanitize_probe_path_part(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn build_probe_jobs(args: &Args) -> anyhow::Result<Vec<ProbeJob>> {
    let templates = split_probe_list(args.probe_templates.as_ref(), &args.probe_template);
    let positions = split_probe_list(args.probe_positions.as_ref(), &args.probe_position);
    if templates.is_empty() {
        anyhow::bail!("probe template list is empty");
    }
    if positions.is_empty() {
        anyhow::bail!("probe position list is empty");
    }

    let is_batch = args.probe_templates.is_some()
        || args.probe_positions.is_some()
        || args.probe_output_dir.is_some()
        || templates.len() > 1
        || positions.len() > 1;
    if !is_batch {
        return Ok(vec![ProbeJob {
            template: templates[0].clone(),
            position: ProbePosition::parse(&positions[0])?,
            output_path: args.probe_output.clone(),
        }]);
    }

    let output_dir = args
        .probe_output_dir
        .clone()
        .unwrap_or_else(|| "data/probe_matrix".to_string());
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create probe output directory: {output_dir}"))?;

    let mut jobs = Vec::with_capacity(templates.len() * positions.len());
    let prefix = sanitize_probe_path_part(&args.probe_output_prefix);
    for template in templates {
        let template_part = sanitize_probe_path_part(&template);
        for position_value in &positions {
            let position = ProbePosition::parse(position_value)?;
            let output_path = format!(
                "{}/{}_{}_{}_activations.npy",
                output_dir,
                prefix,
                template_part,
                position.as_str()
            );
            jobs.push(ProbeJob {
                template: template.clone(),
                position,
                output_path,
            });
        }
    }
    Ok(jobs)
}

fn run_probe_jobs<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    args: &Args,
    run_metadata: &RunMetadata,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let jobs = build_probe_jobs(args)?;
    eprintln!("running {} probe extraction job(s)", jobs.len());

    let mut grouped: Vec<(String, Vec<&ProbeJob>)> = Vec::new();
    for job in &jobs {
        if let Some((_, group_jobs)) = grouped
            .iter_mut()
            .find(|(template, _)| template == &job.template)
        {
            group_jobs.push(job);
        } else {
            grouped.push((job.template.clone(), vec![job]));
        }
    }

    let total_groups = grouped.len();
    for (group_idx, (template, group_jobs)) in grouped.into_iter().enumerate() {
        let outputs = group_jobs
            .iter()
            .map(|job| ProbeOutput {
                position: job.position,
                output_path: job.output_path.clone(),
            })
            .collect::<Vec<_>>();
        let positions = outputs
            .iter()
            .map(|output| output.position.as_str())
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "\n=== probe job group {}/{}: template={} positions={} ===",
            group_idx + 1,
            total_groups,
            template,
            positions
        );
        probe_group_mode(
            backend,
            model,
            tokenizer,
            ProbeGroupConfig {
                stimuli_path: &args.probe_stimuli,
                template: &template,
                outputs,
                generate_tokens: args.probe_generate_tokens,
                limit: args.probe_limit,
                context_limit: effective_context_limit(backend, model, args),
                model_path: &args.model,
                arch: &args.arch,
                tokenizer_path: args
                    .tokenizer
                    .as_deref()
                    .unwrap_or_else(|| default_tokenizer_for_arch(&args.arch)),
                run_metadata,
            },
        )?;
    }
    Ok(())
}

fn token_indices_for_offsets(offsets: &[(usize, usize)], start: usize, end: usize) -> Vec<usize> {
    offsets
        .iter()
        .enumerate()
        .filter_map(|(i, &(tok_start, tok_end))| {
            if tok_start != tok_end && tok_start < end && tok_end > start {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

fn non_special_token_indices(offsets: &[(usize, usize)], token_count: usize) -> Vec<usize> {
    let indices: Vec<usize> = offsets
        .iter()
        .enumerate()
        .filter_map(|(i, &(start, end))| if start != end { Some(i) } else { None })
        .collect();
    if indices.is_empty() {
        (0..token_count).collect()
    } else {
        indices
    }
}

fn stimulus_text_field(stimulus: &serde_json::Value, field: &str) -> anyhow::Result<String> {
    stimulus[field]
        .as_str()
        .map(str::to_owned)
        .with_context(|| format!("stimulus missing string field '{}'", field))
}

fn select_probe_indices(
    prompt: &str,
    token_ids: &[u32],
    offsets: &[(usize, usize)],
    stimulus: &serde_json::Value,
    position: ProbePosition,
) -> anyhow::Result<Vec<usize>> {
    match position {
        ProbePosition::Last => {
            let indices = non_special_token_indices(offsets, token_ids.len());
            indices
                .last()
                .copied()
                .map(|i| vec![i])
                .context("cannot select last token from empty prompt")
        }
        ProbePosition::PromptMean => Ok(non_special_token_indices(offsets, token_ids.len())),
        ProbePosition::Root | ProbePosition::Pattern => {
            let field = match position {
                ProbePosition::Root => "root",
                ProbePosition::Pattern => "pattern",
                _ => unreachable!(),
            };
            let needle = stimulus_text_field(stimulus, field)?;
            let start = prompt.find(&needle).with_context(|| {
                format!(
                    "could not locate {} '{}' in selected prompt template",
                    field, needle
                )
            })?;
            let indices = token_indices_for_offsets(offsets, start, start + needle.len());
            if indices.is_empty() {
                anyhow::bail!(
                    "could not map {} '{}' to tokenizer offsets in selected prompt template",
                    field,
                    needle
                );
            }
            Ok(indices)
        }
    }
}

fn normalize_for_match(text: &str) -> String {
    text.trim()
        .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn match_generated_text(generated: &str, expected: &str) -> (bool, bool) {
    let generated_norm = normalize_for_match(generated);
    let expected_norm = normalize_for_match(expected);
    if expected_norm.is_empty() {
        return (false, false);
    }
    (
        generated_norm == expected_norm,
        generated_norm.contains(&expected_norm),
    )
}

fn generate_probe_continuation<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    context_limit: usize,
) -> anyhow::Result<(Vec<u32>, String)>
where
    B::Error: Send + Sync + 'static,
{
    if max_tokens == 0 {
        return Ok((Vec::new(), String::new()));
    }

    let prompt_len = prompt_tokens.len();
    let max_seq_len = ensure_sequence_fits(prompt_len, max_tokens, context_limit)?;
    let mut cache = model.create_cache(backend, max_seq_len);
    let mut logits = model.forward_last_logits_with_cache(backend, prompt_tokens, &mut cache, 0)?;
    let vocab_size = backend.shape(&logits)[1];
    let mut generated = Vec::with_capacity(max_tokens);

    for step in 0..max_tokens {
        let logit_data = backend.data(&logits);
        let last_logits = &logit_data[..vocab_size];
        let next_token = argmax_token(last_logits);

        let eos_ids = tokenizer.eos_token_ids();
        if eos_ids.contains(&(next_token as u32)) {
            break;
        }

        generated.push(next_token as u32);
        logits = model.forward_last_logits_with_cache(
            backend,
            &[next_token as u32],
            &mut cache,
            prompt_len + step,
        )?;
    }

    let generated_text = tokenizer.decode(&generated)?;
    Ok((generated, generated_text))
}

/// probe mode: feed each stimulus prompt through the model and collect pooled
/// per-layer hidden states for one or more selected token positions.
///
/// Writes one 3d .npy file per requested position: `(n_stimuli, n_layers, embed_dim)`.
fn probe_group_mode<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    config: ProbeGroupConfig<'_>,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    if config.outputs.is_empty() {
        anyhow::bail!("probe group has no outputs");
    }

    // -- load stimuli ------------------------------------------
    let stimuli_json = fs::read_to_string(config.stimuli_path)
        .with_context(|| format!("failed to read stimuli file: {}", config.stimuli_path))?;
    let mut stimuli: Vec<serde_json::Value> = serde_json::from_str(&stimuli_json)?;
    if let Some(limit) = config.limit {
        stimuli.truncate(limit);
    }
    eprintln!(
        "loaded {} stimuli from {}",
        stimuli.len(),
        config.stimuli_path
    );

    let n_layers = model.n_layers();
    let embed_dim = model.embed_dim();
    eprintln!("model: {} layers, {} hidden dim", n_layers, embed_dim);

    let shape = [stimuli.len(), n_layers, embed_dim];
    let row_floats = n_layers * embed_dim;
    eprintln!(
        "streaming {} activation file(s): {} floats per row ({:.1} KB per row)",
        config.outputs.len(),
        row_floats,
        row_floats as f64 * 4.0 / 1024.0
    );
    let mut activation_writers = config
        .outputs
        .iter()
        .map(|output| NpyStreamWriter::create(&output.output_path, &shape))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let zero_activation_row = vec![0.0f32; row_floats];

    // -- collect -----------------------------------------------
    let start = Instant::now();
    let mut correctness: Vec<Vec<serde_json::Value>> = config
        .outputs
        .iter()
        .map(|_| Vec::with_capacity(stimuli.len()))
        .collect();
    let mut token_selections: Vec<Vec<serde_json::Value>> = config
        .outputs
        .iter()
        .map(|_| Vec::with_capacity(stimuli.len()))
        .collect();

    // batched extraction: concatenate all stimuli into one sequence
    // with block-diagonal attention masking for independent processing.
    let mut all_token_ids: Vec<u32> = Vec::new();
    let mut block_boundaries: Vec<usize> = Vec::new();
    let mut block_token_counts: Vec<usize> = Vec::new();
    let mut stimulus_info: Vec<(String, serde_json::Value)> = Vec::new();

    for (si, stimulus) in stimuli.iter().enumerate() {
        let prompt = stimulus["prompts"][config.template]
            .as_str()
            .with_context(|| {
                format!(
                    "stimulus {} missing prompt template '{}'",
                    si, config.template
                )
            })?;

        let (token_ids, _offsets) = tokenizer.encode_with_offsets(prompt)?;
        if token_ids.is_empty() {
            eprintln!(
                "  [{}/{}] WARNING: empty tokenization, skipping",
                si + 1,
                stimuli.len()
            );
            // write zero activation row for this stimulus
            for writer in &mut activation_writers {
                writer.write_f32s(&zero_activation_row)?;
            }
            continue;
        }

        block_boundaries.push(all_token_ids.len());
        block_token_counts.push(token_ids.len());
        all_token_ids.extend_from_slice(&token_ids);
        stimulus_info.push((prompt.to_string(), stimulus.clone()));
    }

    if block_boundaries.is_empty() {
        eprintln!("no valid stimuli to process");
        return Ok(());
    }
    let total_tokens = all_token_ids.len();
    let context_limit = config.context_limit;
    eprintln!(
        "batched {} stimuli into {} total tokens ({} blocks), context limit {}",
        stimulus_info.len(),
        total_tokens,
        block_boundaries.len(),
        context_limit
    );

    // split into chunks that fit within the context limit
    let n_outputs = config.outputs.len();
    let n_stimuli = stimulus_info.len();
    let mut chunk_start = 0usize; // stimulus index
    let mut global_stimulus_idx = 0usize;

    while chunk_start < n_stimuli {
        // find how many stimuli fit in this chunk
        let mut chunk_end = chunk_start;
        let mut chunk_tokens = 0usize;
        while chunk_end < n_stimuli {
            let next_tokens = chunk_tokens + block_token_counts[chunk_end];
            if next_tokens > context_limit && chunk_tokens > 0 {
                break;
            }
            chunk_tokens = next_tokens;
            chunk_end += 1;
        }

        let chunk_boundaries = &block_boundaries[chunk_start..chunk_end];
        // remap boundaries relative to chunk start token position
        let chunk_base = chunk_boundaries[0];
        let chunk_token_ids = &all_token_ids[chunk_base..chunk_base + chunk_tokens];
        let remapped_boundaries: Vec<usize> =
            chunk_boundaries.iter().map(|b| b - chunk_base).collect();

        // build probe index groups for this chunk only
        let mut chunk_probe_indices: Vec<Vec<usize>> =
            Vec::with_capacity((chunk_end - chunk_start) * n_outputs);

        for bi in chunk_start..chunk_end {
            let base = block_boundaries[bi];
            let token_slice = &all_token_ids[base..base + block_token_counts[bi]];

            let (_, offsets) = tokenizer.encode_with_offsets(&stimulus_info[bi].0)?;

            for output in &config.outputs {
                let local_indices = select_probe_indices(
                    &stimulus_info[bi].0,
                    token_slice,
                    &offsets,
                    &stimulus_info[bi].1,
                    output.position,
                )?;
                let absolute: Vec<usize> = local_indices
                    .iter()
                    .map(|i| (base - chunk_base) + i)
                    .collect();
                chunk_probe_indices.push(absolute);
            }
        }

        eprintln!(
            "  chunk [{}-{}]: {} stimuli, {} tokens",
            chunk_start,
            chunk_end - 1,
            chunk_end - chunk_start,
            chunk_tokens
        );

        // batched forward pass for this chunk
        let (pooled_states, logits) = model.forward_pooled_with_blocks(
            backend,
            chunk_token_ids,
            &remapped_boundaries,
            &chunk_probe_indices,
        )?;

        // write pooled states and correctness for each stimulus in this chunk
        for (local_bi, bi) in (chunk_start..chunk_end).enumerate() {
            let base = chunk_boundaries[local_bi];
            let n_tokens = block_token_counts[bi];
            let token_slice = &all_token_ids[base..base + n_tokens];

            // extract per-stimulus logits slice (chunk-relative positions)
            let logit_data = backend.data(&logits);
            let logit_shape = backend.shape(&logits);
            let vocab_size = logit_shape[1];
            let rel_base = remapped_boundaries[local_bi];
            let last_row_start = (rel_base + n_tokens - 1) * vocab_size;
            let last_logits = &logit_data[last_row_start..];
            let predicted_id = last_logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(max_i, max_v), (i, &v)| {
                    if v > max_v {
                        (i, v)
                    } else {
                        (max_i, max_v)
                    }
                })
                .0;
            let predicted_text = tokenizer.decode(&[predicted_id as u32])?;
            let (generated_ids, generated_text) = generate_probe_continuation(
                backend,
                model,
                tokenizer,
                token_slice,
                config.generate_tokens,
                config.context_limit,
            )?;
            let stimulus = &stimulus_info[bi].1;
            let expected = stimulus["expected_surface"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let (generated_exact_match, generated_contains_match) =
                match_generated_text(&generated_text, &expected);

            for (oi, output) in config.outputs.iter().enumerate() {
                // extract pooled states: index = local_bi * n_outputs + oi
                let pool_idx = local_bi * n_outputs + oi;
                let pooled_slice = &pooled_states[pool_idx];

                // re-tokenize for offsets
                let prompt_str = stimulus["prompts"][config.template].as_str().unwrap_or("");
                let (_, offsets) = tokenizer.encode_with_offsets(prompt_str)?;
                let probe_indices = select_probe_indices(
                    prompt_str,
                    token_slice,
                    &offsets,
                    stimulus,
                    output.position,
                )?;

                correctness[oi].push(serde_json::json!({
                    "index": bi,
                    "root": stimulus["root"],
                    "pattern": stimulus["pattern"],
                    "expected": expected,
                    "predicted": predicted_text.trim().to_string(),
                    "predicted_id": predicted_id,
                    "next_token_predicted": predicted_text.trim().to_string(),
                    "next_token_id": predicted_id,
                    "generated": generated_text.trim().to_string(),
                    "generated_ids": generated_ids,
                    "generated_exact_match": generated_exact_match,
                    "generated_contains_match": generated_contains_match,
                    "correct": generated_exact_match || generated_contains_match,
                    "probe_template": config.template,
                    "probe_position": output.position.as_str(),
                    "probe_generate_tokens": config.generate_tokens,
                    "probe_token_indices": probe_indices,
                }));
                token_selections[oi].push(serde_json::json!({
                    "index": bi,
                    "token_count": n_tokens,
                    "probe_token_indices": probe_indices,
                }));

                activation_writers[oi].write_f32s(pooled_slice)?;
            }

            global_stimulus_idx += 1;
            if global_stimulus_idx.is_multiple_of(100) || global_stimulus_idx == n_stimuli {
                eprintln!(
                    "  [{:4}/{}] saved in {:.1}s",
                    global_stimulus_idx,
                    n_stimuli,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        chunk_start = chunk_end;
    } // end while chunk_start < n_stimuli

    // -- save --------------------------------------------------
    for (writer, output) in activation_writers.iter_mut().zip(&config.outputs) {
        writer.finish()?;
        eprintln!("saved activations to {}", output.output_path);
    }

    for (oi, output) in config.outputs.iter().enumerate() {
        let correct_count = correctness[oi]
            .iter()
            .filter(|c| {
                c["correct"]
                    .as_bool()
                    .unwrap_or(c["predicted"] == c["expected"])
            })
            .count();
        let correctness_pct = if correctness[oi].is_empty() {
            0.0
        } else {
            correct_count as f64 / correctness[oi].len() as f64 * 100.0
        };
        eprintln!(
            "correctness [{}]: {}/{} ({:.1}%)",
            output.position.as_str(),
            correct_count,
            correctness[oi].len(),
            correctness_pct
        );

        let correctness_path = output.output_path.replace(".npy", "_correctness.json");
        fs::write(
            &correctness_path,
            serde_json::to_string_pretty(&correctness[oi])?,
        )?;
        eprintln!("saved correctness to {}", correctness_path);

        let metadata = serde_json::json!({
            "model_path": config.model_path,
            "architecture": config.arch,
            "tokenizer_path": config.tokenizer_path,
            "tokenizer_sha256": config.run_metadata.tokenizer_sha256,
            "model_file_size_bytes": config.run_metadata.model_file_size_bytes,
            "model_sha256": config.run_metadata.model_sha256,
            "gguf_metadata": config.run_metadata.gguf_metadata,
            "run_manifest": config.run_metadata.run_manifest,
            "stimuli_path": config.stimuli_path,
            "output_path": output.output_path,
            "probe_template": config.template,
            "probe_position": output.position.as_str(),
            "probe_generate_tokens": config.generate_tokens,
            "probe_limit": config.limit,
            "context_limit": config.context_limit,
            "n_stimuli": stimuli.len(),
            "n_layers": n_layers,
            "embed_dim": embed_dim,
            "activation_shape": shape,
            "correctness_path": correctness_path,
            "token_selections": token_selections[oi],
            "run_timestamp_unix": unix_timestamp(),
            "git_commit": git_commit(),
            "batched_probe_extraction": true,
            "batched_probe_positions": config
                .outputs
                .iter()
                .map(|output| output.position.as_str())
                .collect::<Vec<_>>(),
        });
        let metadata_path = output.output_path.replace(".npy", "_metadata.json");
        fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;
        eprintln!("saved metadata to {}", metadata_path);
    }

    eprintln!("done in {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn gguf_metadata_json(loader: &GgufLoader) -> serde_json::Value {
    let mut entries = serde_json::Map::new();
    for (key, value) in &loader.metadata {
        entries.insert(key.clone(), gguf_value_json(value));
    }
    serde_json::Value::Object(entries)
}

fn gguf_value_json(value: &GgufValue) -> serde_json::Value {
    match value {
        GgufValue::U8(v) => serde_json::json!(v),
        GgufValue::U32(v) => serde_json::json!(v),
        GgufValue::U64(v) => serde_json::json!(v),
        GgufValue::I32(v) => serde_json::json!(v),
        GgufValue::F32(v) => serde_json::json!(v),
        GgufValue::Bool(v) => serde_json::json!(v),
        GgufValue::Str(v) => serde_json::json!(v),
        GgufValue::Array(values) => {
            serde_json::Value::Array(values.iter().map(gguf_value_json).collect())
        }
    }
}

fn write_json_file(path: &str, value: &serde_json::Value) -> anyhow::Result<()> {
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn sha256_file(path: &str) -> Option<String> {
    let output = Command::new("sha256sum").arg(path).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    stdout.split_whitespace().next().map(str::to_string)
}

fn build_run_manifest(
    args: &Args,
    tokenizer_path: &str,
    model_sha256: Option<&str>,
    tokenizer_sha256: Option<&str>,
    gguf_metadata: &serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "created_at_unix": unix_timestamp(),
        "command_argv": env::args().collect::<Vec<_>>(),
        "source": {
            "git_commit": git_commit(),
        },
        "compiler": {
            "rustc_version_verbose": command_output("rustc", &["--version", "--verbose"]),
        },
        "runtime": {
            "rayon_num_threads_env": env::var("RAYON_NUM_THREADS").ok(),
            "rayon_current_num_threads": rayon::current_num_threads(),
            "cpu_features_detected": cpu_features_detected(),
        },
        "model": {
            "path": args.model,
            "sha256": model_sha256,
            "file_size_bytes": fs::metadata(&args.model).ok().map(|m| m.len()),
            "architecture": args.arch,
            "gguf_metadata": gguf_metadata,
        },
        "tokenizer": {
            "path": tokenizer_path,
            "sha256": tokenizer_sha256,
        },
        "execution": {
            "max_seq_len": args.max_seq_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "probe": args.probe,
            "probe_stimuli": args.probe_stimuli,
            "probe_template": args.probe_template,
            "probe_templates": args.probe_templates,
            "probe_position": args.probe_position,
            "probe_positions": args.probe_positions,
            "probe_generate_tokens": args.probe_generate_tokens,
            "probe_limit": args.probe_limit,
        },
    })
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|s| s.trim().to_string())
}

fn cpu_features_detected() -> Vec<&'static str> {
    let mut features = Vec::new();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            features.push("sse2");
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            features.push("ssse3");
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            features.push("sse4.1");
        }
        if std::arch::is_x86_feature_detected!("avx") {
            features.push("avx");
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            features.push("fma");
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            features.push("avx512f");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("neon");
        }
        if std::arch::is_aarch64_feature_detected!("fp16") {
            features.push("fp16");
        }
        if std::arch::is_aarch64_feature_detected!("sve") {
            features.push("sve");
        }
    }

    features
}

fn token_audit_json(
    prompt: &str,
    tokenizer_path: &str,
    tokenizer_sha256: Option<&str>,
    bos_token_id: Option<u32>,
    token_ids: &[u32],
    offsets: &[(usize, usize)],
) -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "prompt": prompt,
        "tokenizer_path": tokenizer_path,
        "tokenizer_sha256": tokenizer_sha256,
        "bos_token_id": bos_token_id,
        "token_ids": token_ids,
        "token_count": token_ids.len(),
        "offsets": offsets,
        "encode_with_offsets_matches_encode": true,
    })
}

fn git_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let commit = String::from_utf8(output.stdout).ok()?;
    Some(commit.trim().to_string())
}
