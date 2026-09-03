use anyhow::Context;
use clap::{Args as ClapArgs, Subcommand};
use ember::backend::{Backend, CpuBackend};
use ember::experiments::ModelFamily;
use ember::extraction::{sha256_bytes, sha256_file_result};
use ember::kv_compare::{
    compare_snapshot_to_diagnostic, compare_snapshots, prepare_diagnostic_perturbation,
    KvComparisonOptions, KvDiagnosticPerturbation, KvPerturbComponent, KvPerturbOperation,
    KvSnapshotComparison,
};
use ember::kv_diagnostics::{
    diagnose_continuation, KvContinuationCandidate, KvContinuationDiagnostics,
};
use ember::kv_snapshot::{KvCompatibilityTarget, KvSnapshot, KvSnapshotOrigin};
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::plan::{ExecutionMode, HookMode};
use ember::quant_k::KStrategy;
use ember::tokenizer::EmberTokenizer;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(ClapArgs)]
pub(crate) struct KvCommand {
    #[command(subcommand)]
    pub command: KvSubcommand,
}

#[derive(Subcommand)]
pub(crate) enum KvSubcommand {
    /// Prefill a Llama/Qwen-family prompt and write a verified KV snapshot.
    Export(KvExportCommand),
    /// Print snapshot metadata (and verify it while loading).
    Inspect(KvInspectCommand),
    /// Verify schema, shape, payload checksums, and snapshot identity.
    Verify(KvVerifyCommand),
    /// Compare aligned KV payloads, with optional continuation diagnostics.
    Compare(KvCompareCommand),
    /// Strictly import a same-model snapshot and continue greedy generation.
    Replay(KvReplayCommand),
    /// Run uninterrupted greedy generation and save a full-logit validation trace.
    TraceNative(KvTraceNativeCommand),
}

#[derive(ClapArgs)]
pub(crate) struct KvExportCommand {
    #[arg(short, long)]
    model: String,
    #[arg(long)]
    tokenizer: String,
    #[arg(long, default_value = "auto", value_parser = ["auto", "llama", "qwen3"])]
    arch: String,
    #[arg(long)]
    prompt: String,
    #[arg(short, long)]
    output: PathBuf,
    /// Source cache/table capacity. Defaults to exactly the tokenized prefix.
    #[arg(long)]
    max_seq_len: Option<usize>,
    #[arg(long, default_value = "reference", value_parser = ["reference", "planned"])]
    execution: String,
    #[arg(long)]
    overwrite: bool,
    /// Optional `[1, vocab]` boundary-logit NPY for process-level replay validation.
    #[arg(long, requires = "metrics_output")]
    boundary_logits_output: Option<PathBuf>,
    /// JSON trace metadata/timing; requires `--boundary-logits-output`.
    #[arg(long, requires = "boundary_logits_output")]
    metrics_output: Option<PathBuf>,
}

#[derive(ClapArgs)]
pub(crate) struct KvInspectCommand {
    snapshot: PathBuf,
    /// Emit the manifest as JSON rather than human-readable text.
    #[arg(long)]
    json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct KvVerifyCommand {
    snapshot: PathBuf,
}

#[derive(ClapArgs)]
pub(crate) struct KvCompareCommand {
    /// Native reference snapshot directory.
    reference: PathBuf,
    /// Candidate snapshot directory. Omit only for an in-memory perturbation.
    candidate: Option<PathBuf>,
    /// Emit deterministic machine-readable JSON.
    #[arg(long)]
    json: bool,
    /// Compute directional R2 in addition to cosine/MSE/max-absolute error.
    #[arg(long)]
    r2: bool,
    /// Flag per-head K/V entries whose max-absolute error exceeds this value.
    #[arg(long)]
    max_abs: Option<f64>,
    /// Flag per-head K/V entries whose MSE exceeds this value.
    #[arg(long)]
    max_mse: Option<f64>,
    /// Flag per-head K/V entries whose cosine is below this value.
    #[arg(long)]
    min_cosine: Option<f64>,
    /// Flag per-head K/V entries whose directional R2 is below this value.
    #[arg(long)]
    min_r2: Option<f64>,

    /// Model for same-input attention/logit and independent-greedy diagnostics.
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    tokenizer: Option<String>,
    #[arg(long, value_parser = ["auto", "llama", "qwen3"])]
    arch: Option<String>,
    /// Greedy sequence horizon including the fixed initial resume token (2..=64).
    #[arg(long)]
    continuation_tokens: Option<usize>,
    /// Override the common initial token; otherwise stored resume IDs must match.
    #[arg(long)]
    token_id: Option<u32>,

    /// Zero-based layer for an in-memory diagnostic perturbation.
    #[arg(long, requires_all = ["perturb_head", "perturb_component"])]
    perturb_layer: Option<usize>,
    /// Zero-based KV head for an in-memory diagnostic perturbation.
    #[arg(long, requires_all = ["perturb_layer", "perturb_component"])]
    perturb_head: Option<usize>,
    /// Perturb keys, values, or both.
    #[arg(long, value_parser = ["keys", "values", "both"], requires_all = ["perturb_layer", "perturb_head"])]
    perturb_component: Option<String>,
    /// Write positive f16 zero to the selected prefix head.
    #[arg(long, conflicts_with = "scale")]
    zero: bool,
    /// Scale the selected head using f16->f32 multiply->f16 rounding.
    #[arg(long, conflicts_with = "zero")]
    scale: Option<f32>,
}

#[derive(Debug, Serialize)]
struct KvMeasurementReport {
    schema: String,
    kind: String,
    comparison: KvSnapshotComparison,
    continuation: Option<KvContinuationDiagnostics>,
}

#[derive(ClapArgs)]
pub(crate) struct KvReplayCommand {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(short, long)]
    model: String,
    #[arg(long)]
    tokenizer: String,
    #[arg(long, default_value = "auto", value_parser = ["auto", "llama", "qwen3"])]
    arch: String,
    /// Number of continuation tokens, including the stored/overridden first
    /// token selected from the prefix logits.
    #[arg(short = 'n', long, default_value_t = 20)]
    max_tokens: usize,
    /// Destination cache/table capacity. Defaults to the minimum needed for
    /// the requested continuation.
    #[arg(long)]
    max_seq_len: Option<usize>,
    /// Must equal the snapshot mode. Defaults to the recorded mode.
    #[arg(long, value_parser = ["reference", "planned"])]
    execution: Option<String>,
    /// Override the snapshot's greedy resume token (needed when absent).
    #[arg(long)]
    token_id: Option<u32>,
    /// Optional `[max_tokens - 1, vocab]` NPY of continuation logits.
    #[arg(long, requires = "metrics_output")]
    logits_output: Option<PathBuf>,
    /// JSON trace metadata/timing; requires `--logits-output`.
    #[arg(long, requires = "logits_output")]
    metrics_output: Option<PathBuf>,
    /// Replace existing trace NPY/JSON outputs.
    #[arg(long)]
    overwrite: bool,
}

#[derive(ClapArgs)]
pub(crate) struct KvTraceNativeCommand {
    #[arg(short, long)]
    model: String,
    #[arg(long)]
    tokenizer: String,
    #[arg(long, default_value = "auto", value_parser = ["auto", "llama", "qwen3"])]
    arch: String,
    #[arg(long)]
    prompt: String,
    #[arg(short = 'n', long, default_value_t = 20)]
    max_tokens: usize,
    #[arg(long)]
    max_seq_len: Option<usize>,
    #[arg(long, default_value = "reference", value_parser = ["reference", "planned"])]
    execution: String,
    /// `[max_tokens, vocab]` uninterrupted native logits.
    #[arg(long)]
    logits_output: PathBuf,
    /// JSON tokens, provenance, shapes, hashes, and phase timings.
    #[arg(long)]
    metrics_output: PathBuf,
    /// Replace existing trace NPY/JSON outputs.
    #[arg(long)]
    overwrite: bool,
}

#[derive(Debug, Serialize)]
struct KvReplayTrace {
    trace_schema: String,
    kind: String,
    ember_version: String,
    model_sha256: String,
    tokenizer_sha256: String,
    architecture: String,
    execution_mode: String,
    execution_fingerprint: String,
    execution_plan_hash: Option<String>,
    snapshot_hash: Option<String>,
    prompt_sha256: Option<String>,
    prefix_token_ids_sha256: Option<String>,
    prefix_length: usize,
    cache_capacity: usize,
    max_tokens: usize,
    generated_token_ids: Vec<u32>,
    stored_resume_token_id: Option<u32>,
    effective_resume_token_id: Option<u32>,
    logits_rows: usize,
    logits_global_row_start: Option<usize>,
    predicted_absolute_position_start: Option<usize>,
    forward_evaluations: usize,
    vocab_size: usize,
    logits_sha256: String,
    logits_semantics: String,
    snapshot_bytes: Option<u64>,
    sampling: String,
    eos_policy: String,
    timings_ms: BTreeMap<String, f64>,
}

pub(crate) fn run_kv_command(
    command: &KvCommand,
    k_strategy: KStrategy,
    allow_fallback: bool,
) -> anyhow::Result<()> {
    match &command.command {
        KvSubcommand::Export(command) => run_export(command, k_strategy, allow_fallback),
        KvSubcommand::Inspect(command) => run_inspect(command),
        KvSubcommand::Verify(command) => run_verify(command),
        KvSubcommand::Compare(command) => run_compare(command, k_strategy, allow_fallback),
        KvSubcommand::Replay(command) => run_replay(command, k_strategy, allow_fallback),
        KvSubcommand::TraceNative(command) => run_trace_native(command, k_strategy, allow_fallback),
    }
}

fn run_export(
    command: &KvExportCommand,
    k_strategy: KStrategy,
    allow_fallback: bool,
) -> anyhow::Result<()> {
    if let (Some(logits), Some(metrics)) = (
        command.boundary_logits_output.as_deref(),
        command.metrics_output.as_deref(),
    ) {
        prepare_trace_outputs(logits, metrics, command.overwrite)?;
        ensure_outputs_outside_snapshot(logits, metrics, &command.output)?;
        ensure_trace_outputs_avoid_inputs(
            logits,
            metrics,
            &[Path::new(&command.model), Path::new(&command.tokenizer)],
        )?;
    }
    let execution = ExecutionMode::from_cli(&command.execution).map_err(anyhow::Error::msg)?;
    let mut timings = BTreeMap::new();

    let (model_sha256, tokenizer_sha256) =
        hash_inputs(&command.model, &command.tokenizer, &mut timings)?;

    let tokenizer_start = Instant::now();
    let tokenizer = EmberTokenizer::from_file(&command.tokenizer)?;
    let token_ids = tokenizer.encode(&command.prompt)?;
    timings.insert(
        "tokenizer_load_and_encode".into(),
        elapsed_ms(tokenizer_start),
    );
    anyhow::ensure!(!token_ids.is_empty(), "tokenized prefix is empty");
    let cache_capacity = command.max_seq_len.unwrap_or(token_ids.len());
    anyhow::ensure!(
        cache_capacity >= token_ids.len(),
        "tokenized prefix has {} tokens but max_seq_len is {cache_capacity}",
        token_ids.len()
    );

    let model_start = Instant::now();
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, allow_fallback)?;
    let architecture = crate::cli_support::resolve_generation_architecture(&command.arch, &loader)?;
    validate_loader_architecture(&loader, &architecture)?;
    let model = ember::llama::Llama::from_loader_with_max_seq_len(loader, Some(cache_capacity))?;
    timings.insert("model_load".into(), elapsed_ms(model_start));
    anyhow::ensure!(
        model.config.max_seq_len >= token_ids.len(),
        "tokenized prefix has {} tokens but model context limit is {}",
        token_ids.len(),
        model.config.max_seq_len
    );
    model.set_execution_mode(execution);
    tokenizer.validate_model_vocab(model.config.vocab_size)?;
    let capacity = model.config.max_seq_len.min(cache_capacity);
    model.set_plan_provenance(model_sha256.clone(), tokenizer_sha256.clone(), capacity);
    let plan = model.execution_plan(
        execution,
        HookMode::Disabled,
        &[],
        capacity,
        Some(&model_sha256),
        Some(&tokenizer_sha256),
    )?;
    let target = KvCompatibilityTarget::from_execution_plan(&plan)?;
    ensure_default_live_cache_limit(&target)?;
    let execution_fingerprint = target.execution_fingerprint.clone();
    let execution_plan_hash = target.plan_hash.clone();
    let backend = CpuBackend;
    let mut cache = model.create_cache(&backend, capacity);

    let prefill_start = Instant::now();
    let logits =
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &token_ids, &mut cache, 0)?;
    timings.insert("prefill_inference".into(), elapsed_ms(prefill_start));
    let resume_token_id =
        checked_trace_argmax(&backend, &logits, &tokenizer, model.config.vocab_size)?;

    let snapshot_build_start = Instant::now();
    let snapshot =
        KvSnapshot::export_native(&cache, target, Some(&token_ids), Some(resume_token_id))?;
    timings.insert(
        "snapshot_build_and_verify".into(),
        elapsed_ms(snapshot_build_start),
    );
    let snapshot_write_start = Instant::now();
    snapshot.save_dir(&command.output, command.overwrite)?;
    timings.insert("snapshot_write".into(), elapsed_ms(snapshot_write_start));

    if let (Some(logits_output), Some(metrics_output)) = (
        command.boundary_logits_output.as_deref(),
        command.metrics_output.as_deref(),
    ) {
        let write_start = Instant::now();
        write_logits_npy(
            logits_output,
            logits.data(),
            1,
            model.config.vocab_size,
            command.overwrite,
        )?;
        timings.insert("logits_write".into(), elapsed_ms(write_start));
        let logits_hash_start = Instant::now();
        let logits_sha256 = sha256_file_result(logits_output)?;
        timings.insert("logits_hash".into(), elapsed_ms(logits_hash_start));
        let trace = KvReplayTrace {
            trace_schema: "ember.kv-replay-trace.v1".into(),
            kind: "snapshot-export-boundary".into(),
            ember_version: env!("CARGO_PKG_VERSION").into(),
            model_sha256: model_sha256.clone(),
            tokenizer_sha256: tokenizer_sha256.clone(),
            architecture: snapshot.manifest().architecture.clone(),
            execution_mode: execution.name().into(),
            execution_fingerprint,
            execution_plan_hash,
            snapshot_hash: Some(snapshot.manifest().snapshot_hash.clone()),
            prompt_sha256: Some(sha256_bytes(command.prompt.as_bytes())),
            prefix_token_ids_sha256: snapshot
                .manifest()
                .provenance
                .prefix_token_ids_sha256
                .clone(),
            prefix_length: token_ids.len(),
            cache_capacity: capacity,
            max_tokens: 1,
            generated_token_ids: vec![resume_token_id],
            stored_resume_token_id: Some(resume_token_id),
            effective_resume_token_id: Some(resume_token_id),
            logits_rows: 1,
            logits_global_row_start: Some(0),
            predicted_absolute_position_start: Some(token_ids.len()),
            forward_evaluations: 1,
            vocab_size: model.config.vocab_size,
            logits_sha256,
            logits_semantics:
                "row 0 is the prefix-boundary logits selecting generated_token_ids[0]".into(),
            snapshot_bytes: Some(directory_regular_file_bytes(&command.output)?),
            sampling: "greedy-full-f32-logits-lowest-id-tie".into(),
            eos_policy: "ignore-fixed-count".into(),
            timings_ms: timings,
        };
        write_trace_json(metrics_output, &trace, command.overwrite)?;
    }

    println!("{}", snapshot.to_summary_text());
    eprintln!("wrote KV snapshot to {}", command.output.display());
    Ok(())
}

fn run_inspect(command: &KvInspectCommand) -> anyhow::Result<()> {
    let snapshot = KvSnapshot::load_dir(&command.snapshot)?;
    if command.json {
        println!("{}", serde_json::to_string_pretty(snapshot.manifest())?);
    } else {
        print!("{}", snapshot.to_summary_text());
    }
    Ok(())
}

fn run_verify(command: &KvVerifyCommand) -> anyhow::Result<()> {
    let identity = KvSnapshot::verify_dir(&command.snapshot)?;
    println!("verified {identity}  {}", command.snapshot.display());
    Ok(())
}

fn run_compare(
    command: &KvCompareCommand,
    k_strategy: KStrategy,
    allow_fallback: bool,
) -> anyhow::Result<()> {
    let perturb_requested = command.perturb_layer.is_some()
        || command.perturb_head.is_some()
        || command.perturb_component.is_some()
        || command.zero
        || command.scale.is_some();
    anyhow::ensure!(
        command.candidate.is_some() != perturb_requested,
        "provide exactly one candidate: a candidate snapshot path, or a complete in-memory perturbation"
    );
    let model_option_count = usize::from(command.model.is_some())
        + usize::from(command.tokenizer.is_some())
        + usize::from(command.arch.is_some())
        + usize::from(command.continuation_tokens.is_some());
    anyhow::ensure!(
        model_option_count == 0 || model_option_count == 4,
        "continuation diagnostics require all of --model, --tokenizer, --arch, and --continuation-tokens"
    );
    anyhow::ensure!(
        command.token_id.is_none() || model_option_count == 4,
        "--token-id requires continuation diagnostics"
    );
    if let Some(max_tokens) = command.continuation_tokens {
        anyhow::ensure!(
            (2..=64).contains(&max_tokens),
            "continuation-tokens must be within 2..=64"
        );
    }

    let reference = KvSnapshot::load_dir(&command.reference)?;
    let reference_payload_bytes = reference
        .manifest()
        .keys
        .byte_length
        .checked_add(reference.manifest().values.byte_length)
        .context("reference payload byte count overflow")?;
    let pair_limit = ember::kv_snapshot::DEFAULT_MAX_PAYLOAD_BYTES;
    anyhow::ensure!(
        reference_payload_bytes <= pair_limit,
        "reference payload is {reference_payload_bytes} bytes; comparison pair limit is {pair_limit}"
    );

    let candidate_snapshot = if let Some(path) = command.candidate.as_deref() {
        let remaining = pair_limit - reference_payload_bytes;
        Some(
            KvSnapshot::load_dir_with_limit(path, remaining).with_context(|| {
                format!(
                "candidate snapshot does not fit the {pair_limit}-byte aggregate comparison limit"
            )
            })?,
        )
    } else {
        None
    };
    let alteration = if perturb_requested {
        anyhow::ensure!(
            command.perturb_layer.is_some()
                && command.perturb_head.is_some()
                && command.perturb_component.is_some(),
            "in-memory perturbation requires --perturb-layer, --perturb-head, and --perturb-component"
        );
        anyhow::ensure!(
            command.zero != command.scale.is_some(),
            "in-memory perturbation requires exactly one of --zero or --scale"
        );
        anyhow::ensure!(
            reference_payload_bytes
                .checked_mul(2)
                .context("diagnostic payload pair size overflow")?
                <= pair_limit,
            "reference plus diagnostic copy exceeds the {pair_limit}-byte aggregate comparison limit"
        );
        let component = match command.perturb_component.as_deref() {
            Some("keys") => KvPerturbComponent::Keys,
            Some("values") => KvPerturbComponent::Values,
            Some("both") => KvPerturbComponent::Both,
            Some(other) => anyhow::bail!("unsupported perturbation component '{other}'"),
            None => unreachable!("complete perturbation checked above"),
        };
        let operation = if command.zero {
            KvPerturbOperation::Zero
        } else {
            KvPerturbOperation::Scale {
                factor: command.scale.context("missing perturbation scale")?,
            }
        };
        Some(prepare_diagnostic_perturbation(
            &reference,
            KvDiagnosticPerturbation {
                layer: command
                    .perturb_layer
                    .context("missing perturbation layer")?,
                head: command.perturb_head.context("missing perturbation head")?,
                component,
                operation,
            },
        )?)
    } else {
        None
    };

    let options = KvComparisonOptions {
        include_r2: command.r2 || command.min_r2.is_some(),
        max_abs_threshold: command.max_abs,
        mse_threshold: command.max_mse,
        cosine_threshold: command.min_cosine,
        r2_threshold: command.min_r2,
    };
    let comparison = match (&candidate_snapshot, &alteration) {
        (Some(candidate), None) => compare_snapshots(&reference, candidate, options)?,
        (None, Some(candidate)) => compare_snapshot_to_diagnostic(&reference, candidate, options)?,
        _ => unreachable!("candidate mode checked above"),
    };

    let continuation = if let Some(max_tokens) = command.continuation_tokens {
        let model_path = command
            .model
            .as_deref()
            .context("missing diagnostic model")?;
        let tokenizer_path = command
            .tokenizer
            .as_deref()
            .context("missing diagnostic tokenizer")?;
        let architecture = command
            .arch
            .as_deref()
            .context("missing diagnostic architecture")?;
        let recorded_mode = reference.manifest().provenance.execution_mode.as_str();
        let execution = ExecutionMode::from_cli(recorded_mode).map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            execution != ExecutionMode::PlannedFused,
            "KV continuation diagnostics support reference or planned execution only"
        );
        let capacity = reference
            .manifest()
            .sequence_length
            .checked_add(max_tokens - 1)
            .context("diagnostic context length overflow")?
            .max(1);
        let model_sha256 = sha256_file_result(model_path)
            .with_context(|| format!("failed to hash model '{model_path}'"))?;
        let tokenizer_sha256 = sha256_file_result(tokenizer_path)
            .with_context(|| format!("failed to hash tokenizer '{tokenizer_path}'"))?;
        let loader = load_gguf_with_k_strategy(model_path, k_strategy, allow_fallback)?;
        let architecture =
            crate::cli_support::resolve_generation_architecture(architecture, &loader)?;
        validate_loader_architecture(&loader, &architecture)?;
        let model = ember::llama::Llama::from_loader_with_max_seq_len(loader, Some(capacity))?;
        model.set_execution_mode(execution);
        let tokenizer = EmberTokenizer::from_file(tokenizer_path)?;
        tokenizer.validate_model_vocab(model.config.vocab_size)?;
        let actual_capacity = model.config.max_seq_len.min(capacity);
        model.set_plan_provenance(
            model_sha256.clone(),
            tokenizer_sha256.clone(),
            actual_capacity,
        );
        anyhow::ensure!(
            actual_capacity >= capacity,
            "diagnostics need {capacity} positions but model context limit is {}",
            model.config.max_seq_len
        );
        let plan = model.execution_plan(
            execution,
            HookMode::Disabled,
            &[],
            actual_capacity,
            Some(&model_sha256),
            Some(&tokenizer_sha256),
        )?;
        let target = KvCompatibilityTarget::from_execution_plan(&plan)?;
        let cache_pair_bytes = target
            .live_cache_bytes()?
            .checked_mul(2)
            .context("diagnostic cache pair byte count overflow")?;
        anyhow::ensure!(
            cache_pair_bytes <= pair_limit,
            "two live diagnostic caches require {cache_pair_bytes} bytes; aggregate limit is {pair_limit}"
        );

        let reference_resume = reference.manifest().provenance.resume_token_id;
        let candidate_resume = candidate_snapshot
            .as_ref()
            .map_or(reference_resume, |snapshot| {
                snapshot.manifest().provenance.resume_token_id
            });
        if command.token_id.is_none() {
            anyhow::ensure!(
                reference_resume == candidate_resume,
                "stored resume token IDs differ; pass --token-id for an explicit common seed"
            );
        }
        let initial_token_id = command
            .token_id
            .or(reference_resume)
            .context("reference snapshot has no resume token; pass --token-id")?;
        anyhow::ensure!(
            tokenizer.contains_token_id(initial_token_id),
            "diagnostic token {initial_token_id} is absent from the tokenizer vocabulary"
        );
        let family = match architecture.as_str() {
            "llama" => ModelFamily::Llama,
            "qwen3" => ModelFamily::Qwen3,
            _ => unreachable!("architecture constrained by clap"),
        };
        let candidate = match (&candidate_snapshot, &alteration) {
            (Some(snapshot), None) => KvContinuationCandidate::Snapshot(snapshot),
            (None, Some(alteration)) => KvContinuationCandidate::Diagnostic {
                source: &reference,
                alteration,
            },
            _ => unreachable!("candidate mode checked above"),
        };
        Some(diagnose_continuation(
            &model,
            &CpuBackend,
            &reference,
            candidate,
            &target,
            family,
            initial_token_id,
            max_tokens,
            None,
        )?)
    } else {
        None
    };

    let report = KvMeasurementReport {
        schema: "ember.kv-measurement.v1".into(),
        kind: "kv-state-comparison-and-continuation-diagnostics".into(),
        comparison,
        continuation,
    };
    if command.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_measurement_report(&report);
    }
    Ok(())
}

fn print_measurement_report(report: &KvMeasurementReport) {
    let comparison = &report.comparison;
    println!("KV comparison: {}", comparison.candidate_identity);
    println!(
        "  payload bit-exact: {}  snapshot hash equal: {}",
        comparison.payload_bit_exact, comparison.snapshot_hash_equal
    );
    println!(
        "  global K: cosine={} mse={:.9e} max_abs={:.9e} bit_mismatches={}",
        format_optional_metric(comparison.keys_global.cosine_similarity),
        comparison.keys_global.mse,
        comparison.keys_global.max_abs_error,
        comparison.keys_global.bit_mismatch_count
    );
    println!(
        "  global V: cosine={} mse={:.9e} max_abs={:.9e} bit_mismatches={}",
        format_optional_metric(comparison.values_global.cosine_similarity),
        comparison.values_global.mse,
        comparison.values_global.max_abs_error,
        comparison.values_global.bit_mismatch_count
    );
    for item in &comparison.per_layer_head {
        println!(
            "  L{} H{}  K cos={} mse={:.9e} r2={} max={:.9e}  V cos={} mse={:.9e} r2={} max={:.9e}",
            item.layer,
            item.head,
            format_optional_metric(item.keys.cosine_similarity),
            item.keys.mse,
            format_optional_metric(item.keys.r2),
            item.keys.max_abs_error,
            format_optional_metric(item.values.cosine_similarity),
            item.values.mse,
            format_optional_metric(item.values.r2),
            item.values.max_abs_error,
        );
    }
    if comparison.thresholds_evaluated {
        println!(
            "  thresholds: {} ({} failing layer/head entries)",
            if comparison.thresholds_passed {
                "PASS"
            } else {
                "FAIL"
            },
            comparison.threshold_exceedances.len()
        );
        if let Some(first) = &comparison.first_threshold_exceedance {
            println!(
                "  first exceedance: layer {} head {}: {}",
                first.layer,
                first.head,
                first.reasons.join(", ")
            );
        }
    }
    if let Some(diagnostics) = &report.continuation {
        println!(
            "Continuation: forced top-1 all agree={} final-logit cosine={} greedy match={} first divergence={}",
            diagnostics.forced_top1_all_agree,
            format_optional_metric(diagnostics.final_logit_cosine),
            diagnostics.greedy_sequences_match,
            diagnostics
                .first_generated_token_divergence
                .map_or_else(|| "none through horizon".into(), |value| value.to_string())
        );
        for layer in &diagnostics.attention_by_layer {
            println!(
                "  attention-output L{} cosine={} mse={:.9e} max={:.9e}",
                layer.layer,
                format_optional_metric(layer.metrics.cosine_similarity),
                layer.metrics.mse,
                layer.metrics.max_abs_error
            );
        }
    }
}

fn format_optional_metric(value: Option<f64>) -> String {
    value.map_or_else(|| "undefined".into(), |value| format!("{value:.9}"))
}

fn run_replay(
    command: &KvReplayCommand,
    k_strategy: KStrategy,
    allow_fallback: bool,
) -> anyhow::Result<()> {
    let trace_enabled = command.logits_output.is_some() && command.metrics_output.is_some();
    if let (Some(logits), Some(metrics)) = (
        command.logits_output.as_deref(),
        command.metrics_output.as_deref(),
    ) {
        prepare_trace_outputs(logits, metrics, command.overwrite)?;
        ensure_outputs_outside_snapshot(logits, metrics, &command.snapshot)?;
        ensure_trace_outputs_avoid_inputs(
            logits,
            metrics,
            &[Path::new(&command.model), Path::new(&command.tokenizer)],
        )?;
    }
    let mut timings = BTreeMap::new();
    let snapshot_load_start = Instant::now();
    let snapshot = KvSnapshot::load_dir(&command.snapshot)?;
    anyhow::ensure!(
        snapshot.manifest().provenance.origin == KvSnapshotOrigin::Native,
        "ordinary KV replay accepts native snapshots only; use `kv compare` for explicit altered-state diagnostics"
    );
    if trace_enabled {
        timings.insert(
            "snapshot_load_and_verify".into(),
            elapsed_ms(snapshot_load_start),
        );
    }
    let recorded_mode = snapshot.manifest().provenance.execution_mode.as_str();
    let execution_name = command.execution.as_deref().unwrap_or(recorded_mode);
    let execution = ExecutionMode::from_cli(execution_name).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        execution_name == recorded_mode,
        "requested execution mode '{execution_name}' does not equal snapshot mode '{recorded_mode}'"
    );
    anyhow::ensure!(
        execution != ExecutionMode::PlannedFused,
        "KV replay does not yet expose planned-fused mode"
    );

    let (model_sha256, tokenizer_sha256) =
        hash_inputs(&command.model, &command.tokenizer, &mut timings)?;
    let minimum_capacity = snapshot
        .manifest()
        .sequence_length
        .checked_add(command.max_tokens.saturating_sub(1))
        .ok_or_else(|| anyhow::anyhow!("requested replay context length overflow"))?
        .max(1);
    let requested_capacity = command.max_seq_len.unwrap_or(minimum_capacity);
    anyhow::ensure!(
        requested_capacity >= minimum_capacity,
        "max_seq_len {requested_capacity} is too small; replay needs at least {minimum_capacity}"
    );

    let model_start = Instant::now();
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, allow_fallback)?;
    let architecture = crate::cli_support::resolve_generation_architecture(&command.arch, &loader)?;
    validate_loader_architecture(&loader, &architecture)?;
    let model =
        ember::llama::Llama::from_loader_with_max_seq_len(loader, Some(requested_capacity))?;
    if trace_enabled {
        timings.insert("model_load".into(), elapsed_ms(model_start));
    }
    anyhow::ensure!(
        model.config.max_seq_len >= minimum_capacity,
        "requested replay needs {minimum_capacity} positions but model context limit is {}",
        model.config.max_seq_len
    );
    model.set_execution_mode(execution);
    let tokenizer_start = Instant::now();
    let tokenizer = EmberTokenizer::from_file(&command.tokenizer)?;
    tokenizer.validate_model_vocab(model.config.vocab_size)?;
    if trace_enabled {
        timings.insert("tokenizer_load".into(), elapsed_ms(tokenizer_start));
    }
    let capacity = model.config.max_seq_len.min(requested_capacity);
    model.set_plan_provenance(model_sha256.clone(), tokenizer_sha256.clone(), capacity);
    let plan = model.execution_plan(
        execution,
        HookMode::Disabled,
        &[],
        capacity,
        Some(&model_sha256),
        Some(&tokenizer_sha256),
    )?;
    let target = KvCompatibilityTarget::from_execution_plan(&plan)?;
    let execution_fingerprint = target.execution_fingerprint.clone();
    let execution_plan_hash = target.plan_hash.clone();
    let report = snapshot.compatibility_report(&target);
    if !report.compatible {
        anyhow::bail!(
            "snapshot is incompatible with replay target:\n- {}",
            report.reasons.join("\n- ")
        );
    }
    let import_start = Instant::now();
    let mut cache = snapshot.import_cache(&target)?;
    if trace_enabled {
        timings.insert("snapshot_import".into(), elapsed_ms(import_start));
    }
    let stored_first = snapshot.manifest().provenance.resume_token_id;
    let first = command.token_id.or(stored_first);
    if command.max_tokens > 0 {
        anyhow::ensure!(
            first.is_some(),
            "snapshot has no resume token; pass --token-id"
        );
    }
    let logits_rows = command.max_tokens.saturating_sub(1);
    let mut logits_writer = if let Some(logits_output) = command.logits_output.as_deref() {
        Some(create_logits_writer(
            logits_output,
            logits_rows,
            model.config.vocab_size,
        )?)
    } else {
        None
    };
    let backend = CpuBackend;
    let mut generated = Vec::new();
    generated
        .try_reserve_exact(command.max_tokens)
        .context("cannot allocate replay token IDs")?;
    let mut decode_inference_ms = 0.0;
    let mut logits_write_ms = 0.0;
    if let Some(mut current) = first.filter(|_| command.max_tokens > 0) {
        anyhow::ensure!(
            (current as usize) < model.config.vocab_size,
            "resume token {current} is outside model vocabulary"
        );
        anyhow::ensure!(
            tokenizer.contains_token_id(current),
            "resume token {current} is absent from the tokenizer vocabulary"
        );
        generated.push(current);
        for _ in 1..command.max_tokens {
            let start_pos = cache.cursor();
            let decode_start = trace_enabled.then(Instant::now);
            let logits = ForwardModel::forward_last_logits_with_cache(
                &model,
                &backend,
                &[current],
                &mut cache,
                start_pos,
            )?;
            if let Some(start) = decode_start {
                decode_inference_ms += elapsed_ms(start);
                current =
                    checked_trace_argmax(&backend, &logits, &tokenizer, model.config.vocab_size)?;
                let write_start = Instant::now();
                logits_writer
                    .as_mut()
                    .context("missing replay trace writer")?
                    .write_f32s(logits.data())?;
                logits_write_ms += elapsed_ms(write_start);
            } else {
                current = ember::sampler::argmax_token(logits.data()) as u32;
            }
            generated.push(current);
        }
    }

    if let (Some(logits_output), Some(metrics_output)) = (
        command.logits_output.as_deref(),
        command.metrics_output.as_deref(),
    ) {
        timings.insert("decode_inference".into(), decode_inference_ms);
        let write_start = Instant::now();
        finish_logits_writer(
            logits_writer
                .as_mut()
                .context("missing replay trace writer")?,
            command.overwrite,
        )?;
        logits_write_ms += elapsed_ms(write_start);
        timings.insert("logits_write".into(), logits_write_ms);
        let logits_hash_start = Instant::now();
        let logits_sha256 = sha256_file_result(logits_output)?;
        timings.insert("logits_hash".into(), elapsed_ms(logits_hash_start));
        let has_rows = logits_rows > 0;
        let trace = KvReplayTrace {
            trace_schema: "ember.kv-replay-trace.v1".into(),
            kind: "snapshot-replay-continuation".into(),
            ember_version: env!("CARGO_PKG_VERSION").into(),
            model_sha256: model_sha256.clone(),
            tokenizer_sha256: tokenizer_sha256.clone(),
            architecture: snapshot.manifest().architecture.clone(),
            execution_mode: execution.name().into(),
            execution_fingerprint,
            execution_plan_hash,
            snapshot_hash: Some(snapshot.manifest().snapshot_hash.clone()),
            prompt_sha256: None,
            prefix_token_ids_sha256: snapshot
                .manifest()
                .provenance
                .prefix_token_ids_sha256
                .clone(),
            prefix_length: snapshot.manifest().sequence_length,
            cache_capacity: capacity,
            max_tokens: command.max_tokens,
            generated_token_ids: generated.clone(),
            stored_resume_token_id: stored_first,
            effective_resume_token_id: first.filter(|_| command.max_tokens > 0),
            logits_rows,
            logits_global_row_start: has_rows.then_some(1),
            predicted_absolute_position_start: has_rows
                .then_some(snapshot.manifest().sequence_length + 1),
            forward_evaluations: logits_rows,
            vocab_size: model.config.vocab_size,
            logits_sha256,
            logits_semantics:
                "row i selects generated_token_ids[i + 1] after evaluating generated_token_ids[i]"
                    .into(),
            snapshot_bytes: Some(directory_regular_file_bytes(&command.snapshot)?),
            sampling: "greedy-full-f32-logits-lowest-id-tie".into(),
            eos_policy: "ignore-fixed-count".into(),
            timings_ms: timings,
        };
        write_trace_json(metrics_output, &trace, command.overwrite)?;
    }

    println!("{}", tokenizer.decode(&generated)?);
    eprintln!(
        "replayed {} continuation tokens from prefix length {} (snapshot {})",
        generated.len(),
        snapshot.manifest().sequence_length,
        snapshot.manifest().snapshot_hash
    );
    Ok(())
}

fn run_trace_native(
    command: &KvTraceNativeCommand,
    k_strategy: KStrategy,
    allow_fallback: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        command.max_tokens > 0,
        "trace-native requires at least one generated token"
    );
    prepare_trace_outputs(
        &command.logits_output,
        &command.metrics_output,
        command.overwrite,
    )?;
    ensure_trace_outputs_avoid_inputs(
        &command.logits_output,
        &command.metrics_output,
        &[Path::new(&command.model), Path::new(&command.tokenizer)],
    )?;
    let execution = ExecutionMode::from_cli(&command.execution).map_err(anyhow::Error::msg)?;
    let mut timings = BTreeMap::new();

    let (model_sha256, tokenizer_sha256) =
        hash_inputs(&command.model, &command.tokenizer, &mut timings)?;

    let tokenizer_start = Instant::now();
    let tokenizer = EmberTokenizer::from_file(&command.tokenizer)?;
    let token_ids = tokenizer.encode(&command.prompt)?;
    timings.insert(
        "tokenizer_load_and_encode".into(),
        elapsed_ms(tokenizer_start),
    );
    anyhow::ensure!(!token_ids.is_empty(), "tokenized prefix is empty");
    let minimum_capacity = token_ids
        .len()
        .checked_add(command.max_tokens - 1)
        .ok_or_else(|| anyhow::anyhow!("native trace context length overflow"))?;
    let requested_capacity = command.max_seq_len.unwrap_or(minimum_capacity);
    anyhow::ensure!(
        requested_capacity >= minimum_capacity,
        "max_seq_len {requested_capacity} is too small; native trace needs at least {minimum_capacity}"
    );

    let model_start = Instant::now();
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, allow_fallback)?;
    let architecture = crate::cli_support::resolve_generation_architecture(&command.arch, &loader)?;
    validate_loader_architecture(&loader, &architecture)?;
    let model =
        ember::llama::Llama::from_loader_with_max_seq_len(loader, Some(requested_capacity))?;
    timings.insert("model_load".into(), elapsed_ms(model_start));
    anyhow::ensure!(
        model.config.max_seq_len >= minimum_capacity,
        "native trace needs {minimum_capacity} positions but model context limit is {}",
        model.config.max_seq_len
    );
    model.set_execution_mode(execution);
    tokenizer.validate_model_vocab(model.config.vocab_size)?;
    let capacity = model.config.max_seq_len.min(requested_capacity);
    model.set_plan_provenance(model_sha256.clone(), tokenizer_sha256.clone(), capacity);
    let plan = model.execution_plan(
        execution,
        HookMode::Disabled,
        &[],
        capacity,
        Some(&model_sha256),
        Some(&tokenizer_sha256),
    )?;
    let target = KvCompatibilityTarget::from_execution_plan(&plan)?;
    ensure_default_live_cache_limit(&target)?;
    let execution_fingerprint = target.execution_fingerprint.clone();
    let execution_plan_hash = target.plan_hash.clone();
    let backend = CpuBackend;
    let mut cache = model.create_cache(&backend, capacity);
    let mut logits_writer = create_logits_writer(
        &command.logits_output,
        command.max_tokens,
        model.config.vocab_size,
    )?;
    let mut generated = Vec::new();
    generated
        .try_reserve_exact(command.max_tokens)
        .context("cannot allocate native trace token IDs")?;
    let mut logits_write_ms = 0.0;

    let prefill_start = Instant::now();
    let mut logits =
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &token_ids, &mut cache, 0)?;
    timings.insert("prefill_inference".into(), elapsed_ms(prefill_start));
    let mut current = checked_trace_argmax(&backend, &logits, &tokenizer, model.config.vocab_size)?;
    let write_start = Instant::now();
    logits_writer.write_f32s(logits.data())?;
    logits_write_ms += elapsed_ms(write_start);
    generated.push(current);

    let mut decode_inference_ms = 0.0;
    for _ in 1..command.max_tokens {
        let start_pos = cache.cursor();
        let decode_start = Instant::now();
        logits = ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &[current],
            &mut cache,
            start_pos,
        )?;
        decode_inference_ms += elapsed_ms(decode_start);
        current = checked_trace_argmax(&backend, &logits, &tokenizer, model.config.vocab_size)?;
        let write_start = Instant::now();
        logits_writer.write_f32s(logits.data())?;
        logits_write_ms += elapsed_ms(write_start);
        generated.push(current);
    }
    timings.insert("decode_inference".into(), decode_inference_ms);
    let write_start = Instant::now();
    finish_logits_writer(&mut logits_writer, command.overwrite)?;
    logits_write_ms += elapsed_ms(write_start);
    timings.insert("logits_write".into(), logits_write_ms);
    let logits_hash_start = Instant::now();
    let logits_sha256 = sha256_file_result(&command.logits_output)?;
    timings.insert("logits_hash".into(), elapsed_ms(logits_hash_start));
    let trace = KvReplayTrace {
        trace_schema: "ember.kv-replay-trace.v1".into(),
        kind: "native-uninterrupted".into(),
        ember_version: env!("CARGO_PKG_VERSION").into(),
        model_sha256,
        tokenizer_sha256,
        architecture: target.architecture,
        execution_mode: execution.name().into(),
        execution_fingerprint,
        execution_plan_hash,
        snapshot_hash: None,
        prompt_sha256: Some(sha256_bytes(command.prompt.as_bytes())),
        prefix_token_ids_sha256: Some(ember::kv_snapshot::hash_token_ids(&token_ids)),
        prefix_length: token_ids.len(),
        cache_capacity: capacity,
        max_tokens: command.max_tokens,
        generated_token_ids: generated.clone(),
        stored_resume_token_id: None,
        effective_resume_token_id: generated.first().copied(),
        logits_rows: command.max_tokens,
        logits_global_row_start: Some(0),
        predicted_absolute_position_start: Some(token_ids.len()),
        forward_evaluations: command.max_tokens,
        vocab_size: model.config.vocab_size,
        logits_sha256,
        logits_semantics: "row i selects generated_token_ids[i]; row 0 is the prefix boundary"
            .into(),
        snapshot_bytes: None,
        sampling: "greedy-full-f32-logits-lowest-id-tie".into(),
        eos_policy: "ignore-fixed-count".into(),
        timings_ms: timings,
    };
    write_trace_json(&command.metrics_output, &trace, command.overwrite)?;
    println!("{}", tokenizer.decode(&generated)?);
    eprintln!(
        "wrote native KV replay validation trace: {} logits rows to {}",
        command.max_tokens,
        command.logits_output.display()
    );
    Ok(())
}

fn ensure_default_live_cache_limit(target: &KvCompatibilityTarget) -> anyhow::Result<()> {
    let live_bytes = target.live_cache_bytes()?;
    anyhow::ensure!(
        live_bytes <= ember::kv_snapshot::DEFAULT_MAX_PAYLOAD_BYTES,
        "live KV cache requires {live_bytes} bytes; default safety limit is {}",
        ember::kv_snapshot::DEFAULT_MAX_PAYLOAD_BYTES
    );
    Ok(())
}

fn checked_trace_argmax(
    backend: &CpuBackend,
    logits: &ember::tensor::CpuTensor,
    tokenizer: &EmberTokenizer,
    expected_vocab_size: usize,
) -> anyhow::Result<u32> {
    let shape = backend.shape(logits);
    anyhow::ensure!(
        shape == [1, expected_vocab_size],
        "expected trace logits shape [1, {expected_vocab_size}], got {shape:?}"
    );
    let data = backend.data(logits);
    anyhow::ensure!(
        data.len() == expected_vocab_size,
        "trace logits contain {} values, expected {expected_vocab_size}",
        data.len()
    );
    if let Some((index, value)) = data
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("trace logits contain non-finite value {value} at vocabulary index {index}");
    }
    let token = ember::sampler::argmax_token(data) as u32;
    anyhow::ensure!(
        tokenizer.contains_token_id(token),
        "greedy token {token} is absent from the tokenizer vocabulary"
    );
    Ok(token)
}

/// Hash the model and tokenizer files, recording wall time under
/// "hash_inputs" in the shared timing map.
fn hash_inputs(
    model: &str,
    tokenizer: &str,
    timings: &mut BTreeMap<String, f64>,
) -> anyhow::Result<(String, String)> {
    let hash_start = Instant::now();
    let model_sha256 =
        sha256_file_result(model).with_context(|| format!("failed to hash model '{model}'"))?;
    let tokenizer_sha256 = sha256_file_result(tokenizer)
        .with_context(|| format!("failed to hash tokenizer '{tokenizer}'"))?;
    timings.insert("hash_inputs".into(), elapsed_ms(hash_start));
    Ok((model_sha256, tokenizer_sha256))
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn prepare_trace_outputs(logits: &Path, metrics: &Path, overwrite: bool) -> anyhow::Result<()> {
    anyhow::ensure!(
        resolved_absolute(logits)? != resolved_absolute(metrics)?,
        "logits and metrics outputs must be different paths"
    );
    if !overwrite {
        anyhow::ensure!(
            !logits.exists(),
            "trace logits output '{}' already exists; pass --overwrite to replace it",
            logits.display()
        );
        anyhow::ensure!(
            !metrics.exists(),
            "trace metrics output '{}' already exists; pass --overwrite to replace it",
            metrics.display()
        );
    }
    Ok(())
}

fn ensure_outputs_outside_snapshot(
    logits: &Path,
    metrics: &Path,
    snapshot: &Path,
) -> anyhow::Result<()> {
    let snapshot = resolved_absolute(snapshot)?;
    for output in [logits, metrics] {
        let output = resolved_absolute(output)?;
        anyhow::ensure!(
            !output.starts_with(&snapshot),
            "trace output '{}' must be outside the strict three-file snapshot directory '{}'",
            output.display(),
            snapshot.display()
        );
    }
    Ok(())
}

fn resolved_absolute(path: &Path) -> anyhow::Result<PathBuf> {
    let absolute = ember::kv_snapshot::normalize_path(path)?;
    let mut existing = absolute.as_path();
    let mut suffix = Vec::new();
    while !existing.exists() {
        let name = existing.file_name().ok_or_else(|| {
            anyhow::anyhow!("output path '{}' has no existing ancestor", path.display())
        })?;
        suffix.push(name.to_os_string());
        existing = existing
            .parent()
            .ok_or_else(|| anyhow::anyhow!("output path '{}' has no parent", path.display()))?;
    }
    let mut resolved = existing.canonicalize()?;
    for component in suffix.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn ensure_trace_outputs_avoid_inputs(
    logits: &Path,
    metrics: &Path,
    inputs: &[&Path],
) -> anyhow::Result<()> {
    let logits = resolved_absolute(logits)?;
    let metrics = resolved_absolute(metrics)?;
    for input in inputs {
        let input = resolved_absolute(input)?;
        anyhow::ensure!(
            logits != input && metrics != input,
            "trace outputs must not replace input '{}'",
            input.display()
        );
    }
    let executable = resolved_absolute(&std::env::current_exe()?)?;
    anyhow::ensure!(
        logits != executable && metrics != executable,
        "trace outputs must not replace the running Ember executable"
    );
    Ok(())
}

fn create_logits_writer(
    path: &Path,
    rows: usize,
    vocab_size: usize,
) -> anyhow::Result<ember::npy::NpyStreamWriter> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("logits output path is not UTF-8"))?;
    ember::npy::NpyStreamWriter::create(path_str, &[rows, vocab_size])
}

fn finish_logits_writer(
    writer: &mut ember::npy::NpyStreamWriter,
    overwrite: bool,
) -> anyhow::Result<()> {
    if overwrite {
        writer.finish()
    } else {
        writer.finish_no_replace()
    }
}

fn write_logits_npy(
    path: &Path,
    logits: &[f32],
    rows: usize,
    vocab_size: usize,
    overwrite: bool,
) -> anyhow::Result<()> {
    let mut writer = create_logits_writer(path, rows, vocab_size)?;
    writer.write_f32s(logits)?;
    finish_logits_writer(&mut writer, overwrite)
}

fn write_trace_json(path: &Path, trace: &KvReplayTrace, overwrite: bool) -> anyhow::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(trace)?;
    bytes.push(b'\n');
    if overwrite {
        ember::atomic_file::atomic_write(path, &bytes)?;
    } else {
        ember::atomic_file::atomic_write_new(path, &bytes)?;
    }
    Ok(())
}

fn directory_regular_file_bytes(path: &Path) -> anyhow::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        anyhow::ensure!(
            metadata.is_file(),
            "snapshot entry '{}' is not a regular file",
            entry.path().display()
        );
        total = total
            .checked_add(metadata.len())
            .ok_or_else(|| anyhow::anyhow!("snapshot directory byte count overflow"))?;
    }
    Ok(total)
}

fn validate_loader_architecture(
    loader: &ember::loader::GgufLoader,
    requested: &str,
) -> anyhow::Result<()> {
    use ember::loader::GgufValue;
    let recorded = match loader.metadata.get("general.architecture") {
        Some(GgufValue::Str(value)) => value.as_str(),
        Some(_) => anyhow::bail!("GGUF general.architecture is not a string"),
        None => "llama",
    };
    let matches = match requested {
        "llama" => recorded == "llama",
        "qwen3" => matches!(recorded, "qwen2" | "qwen3"),
        "gpt2" | "gemma4" => {
            anyhow::bail!("kv commands support llama/qwen3 models only; the model is '{requested}'")
        }
        _ => false,
    };
    anyhow::ensure!(
        matches,
        "requested architecture '{requested}' does not match GGUF architecture '{recorded}'"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestKvParser {
        #[command(flatten)]
        kv: KvCommand,
    }

    #[test]
    fn compare_cli_parses_snapshot_and_diagnostic_forms() {
        let parsed = TestKvParser::try_parse_from([
            "test",
            "compare",
            "reference",
            "candidate",
            "--json",
            "--r2",
            "--min-cosine",
            "0.99",
        ])
        .unwrap();
        let KvSubcommand::Compare(command) = parsed.kv.command else {
            panic!("expected compare command");
        };
        assert_eq!(command.reference, PathBuf::from("reference"));
        assert_eq!(command.candidate, Some(PathBuf::from("candidate")));
        assert!(command.json && command.r2);
        assert_eq!(command.min_cosine, Some(0.99));

        let parsed = TestKvParser::try_parse_from([
            "test",
            "compare",
            "reference",
            "--perturb-layer",
            "1",
            "--perturb-head",
            "2",
            "--perturb-component",
            "keys",
            "--zero",
        ])
        .unwrap();
        let KvSubcommand::Compare(command) = parsed.kv.command else {
            panic!("expected compare command");
        };
        assert!(command.candidate.is_none());
        assert!(command.zero);
        assert_eq!(command.perturb_layer, Some(1));
    }

    #[test]
    fn compare_cli_rejects_partial_perturbation_and_bad_architecture() {
        assert!(TestKvParser::try_parse_from([
            "test",
            "compare",
            "reference",
            "--perturb-layer",
            "1",
            "--zero",
        ])
        .is_err());
        assert!(TestKvParser::try_parse_from([
            "test",
            "compare",
            "reference",
            "candidate",
            "--model",
            "model.gguf",
            "--tokenizer",
            "tokenizer.json",
            "--arch",
            "gemma4",
            "--continuation-tokens",
            "4",
        ])
        .is_err());
    }

    #[test]
    fn prefix_token_trace_hash_is_domain_separated_and_little_endian() {
        // The shared hasher lives in kv_snapshot; this test locks in that the
        // contract observed here matches the single source of truth.
        assert_eq!(
            ember::kv_snapshot::hash_token_ids(&[1, 2, u32::MAX]),
            "7ba3fbe5e313572a9a6ee56956380b6a07a48019956aaf835c5d441babe7924e"
        );
    }

    #[test]
    fn trace_outputs_must_be_distinct_and_outside_snapshot() {
        assert!(prepare_trace_outputs(
            Path::new("validation/logits.npy"),
            Path::new("validation/./logits.npy"),
            true,
        )
        .is_err());
        assert!(ensure_outputs_outside_snapshot(
            Path::new("snapshot/boundary.npy"),
            Path::new("validation/export.json"),
            Path::new("snapshot"),
        )
        .is_err());
        assert!(ensure_outputs_outside_snapshot(
            Path::new("validation/boundary.npy"),
            Path::new("validation/export.json"),
            Path::new("snapshot"),
        )
        .is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_trace_parents_cannot_alias_or_enter_snapshot() {
        use std::os::unix::fs::symlink;
        use std::time::{SystemTime, UNIX_EPOCH};
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "ember-kv-trace-paths-{}-{unique}",
            std::process::id()
        ));
        let snapshot = root.join("snapshot");
        std::fs::create_dir_all(&snapshot).unwrap();
        let alias_a = root.join("alias-a");
        let alias_b = root.join("alias-b");
        symlink(&snapshot, &alias_a).unwrap();
        symlink(&snapshot, &alias_b).unwrap();

        assert!(
            prepare_trace_outputs(&alias_a.join("same"), &alias_b.join("same"), true,).is_err()
        );
        assert!(ensure_outputs_outside_snapshot(
            &alias_a.join("boundary.npy"),
            &root.join("export.json"),
            &snapshot,
        )
        .is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
