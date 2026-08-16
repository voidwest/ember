//! Subcommand implementations: extraction, validation, logits reference, benchmarks.
//! Split out of `main.rs` (2026-08-01) to keep the CLI dispatcher thin.

use crate::cli_support::{
    default_tokenizer_for_arch, gguf_metadata_json, parse_layers_list,
    resolve_generation_architecture, validate_token_ids_for_model, write_json_file,
};
use crate::{
    rayon_current_num_threads, Args, BenchDecodeCommand, BenchLifecycleCommand,
    CompareArtifactsCommand, ExtractCommand, InspectPlanCommand, LifecycleModeArg,
    NativeLogitsReferenceCommand, ValidateBackendsCommand, ValidateRunCommand,
};
use anyhow::Context;
use ember::backend::Backend;
use ember::backend::CpuBackend;
use ember::extraction::{
    canonical_config_toml, git_commit, load_input_samples, pooling_for_mode, run_dir,
    sample_order_hash, select_token_positions, sha256_file_result, source_field_for_position,
    source_span_for_position, source_value_for_position, stable_bytes_hash, stable_prompt_hash,
    unix_timestamp, validate_artifact_contract, ArtifactManifest, BackendMetadata,
    ExecutionBackendName, ExtractionConfig, LogitsArtifact, ModelMetadata, PositionArtifactRecord,
    RunDirectoryTransaction, SampleArtifactRecord, TensorContract, TokenizationArtifactRecord,
    TokenizerMetadata, ARTIFACT_CONTRACT_VERSION, ARTIFACT_LAYOUT, CHECKSUMS_FILENAME,
    CONFIG_FILENAME, LOGITS_FILENAME, MANIFEST_FILENAME, POSITIONS_FILENAME, REPORT_FILENAME,
    SAMPLES_FILENAME, TOKENIZATION_FILENAME,
};
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::model::Gpt2;
use ember::model_backend::compare_backend_artifacts;
use ember::model_backend::run_extraction_with_backend;
use ember::model_backend::run_llama_cpp_external_backend;
use ember::model_backend::NativeModelBackend;
use ember::model_backend::{checksum_insert, path_to_string};
use ember::npy::NpyStreamWriter;
use ember::sampler::argmax_token;
use ember::trace;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::time::Instant;

struct DecodeProfileSession {
    active: bool,
}

impl DecodeProfileSession {
    fn start() -> Self {
        ember::decode_profile::start();
        Self { active: true }
    }

    fn finish(&mut self) -> Vec<ember::decode_profile::DecodeOpSummary> {
        self.active = false;
        ember::decode_profile::finish()
    }
}

impl Drop for DecodeProfileSession {
    fn drop(&mut self) {
        if self.active {
            let _ = ember::decode_profile::finish();
        }
    }
}

pub(crate) fn validate_experiment_options(args: &Args) -> anyhow::Result<()> {
    if args.trace.is_none() {
        if args.trace_out.is_some() {
            anyhow::bail!("--trace-out requires --trace ops");
        }
        if args.trace_values != "none" {
            anyhow::bail!("--trace-values summary requires --trace ops");
        }
        if args.trace_run_metadata {
            anyhow::bail!("--trace-run-metadata requires --trace ops");
        }
    }

    if args.command.is_some()
        && (args.demo
            || args.interactive
            || args.probe
            || args.dump_logits.is_some()
            || args.dump_layers.is_some()
            || args.trace.is_some()
            || args.dump_gguf_metadata.is_some()
            || args.write_run_manifest.is_some()
            || args.record_model_sha256)
    {
        anyhow::bail!("top-level generation/output options cannot be combined with a subcommand");
    }

    if !args.probe
        && (args.probe_templates.is_some()
            || args.probe_positions.is_some()
            || args.probe_output_dir.is_some()
            || args.probe_limit.is_some()
            || args.probe_stimuli != "stimuli/nonce_root_pattern_surface.json"
            || args.probe_output != "data/activations.npy"
            || args.probe_template != "en_surface_probe"
            || args.probe_position != "last"
            || args.probe_output_prefix != "probe"
            || args.probe_generate_tokens != 16)
    {
        anyhow::bail!("probe-specific options require --probe");
    }

    let produces_reproducibility_output = args.probe
        || args.dump_logits.is_some()
        || args.dump_layers.is_some()
        || args.write_run_manifest.is_some()
        || args.capture_activations.is_some()
        || args.activation_patch.is_some()
        || args.activation_stats.is_some();
    if args.record_model_sha256 && !produces_reproducibility_output {
        anyhow::bail!(
            "--record-model-sha256 requires an artifact-producing mode such as --probe, --dump-logits, --dump-layers, or --write-run-manifest"
        );
    }

    let option = if args.zero_layer_output.is_some() {
        "--zero-layer-output"
    } else if args.activation_stats.is_some() {
        "--activation-stats"
    } else if args.activation_patch.is_some() {
        "--activation-patch"
    } else if args.capture_activations.is_some() {
        "--capture-activations"
    } else {
        if !args.patch_target.is_empty() {
            anyhow::bail!("--patch-target requires --activation-patch");
        }
        return Ok(());
    };
    if args.command.is_some() {
        anyhow::bail!("{option} is supported only for normal generation");
    }
    if args.arch == "gpt2" {
        anyhow::bail!("{option} is supported for --arch llama, qwen3, and gemma4, not gpt2");
    }
    if args.activation_patch.is_some() && args.patch_target.is_empty() {
        anyhow::bail!("--activation-patch requires at least one --patch-target");
    }
    Ok(())
}

pub(crate) fn run_extract_command(
    command: &ExtractCommand,
    k_strategy: ember::quant_k::KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let config = build_extraction_config(command)?;
    config.validate()?;

    match config.backend {
        ExecutionBackendName::Native => {
            run_native_extract_command(&config, k_strategy, k_allow_fallback)
        }
        ExecutionBackendName::LlamaCpp => {
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

pub(crate) fn run_validate_backends_command(
    command: &ValidateBackendsCommand,
) -> anyhow::Result<()> {
    if let (Some(native_run), Some(external_run)) = (&command.native_run, &command.external_run) {
        let report = compare_backend_artifacts(native_run, external_run)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    anyhow::bail!(
        "validate-backends requires both --native-run and --external-run \
         (two recorded extraction runs to compare)"
    )
}

pub(crate) fn run_compare_artifacts_command(
    command: &CompareArtifactsCommand,
) -> anyhow::Result<()> {
    let report = ember::compare::compare_artifacts(&command.left, &command.right)
        .map_err(|error| anyhow::anyhow!("compare-artifacts: {error}"))?;
    if command.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", ember::compare::render_human(&report));
    }
    if let Some(output) = &command.output {
        write_json_file(output, &serde_json::to_value(&report)?)?;
        eprintln!("wrote comparison report to {output}");
    }
    Ok(())
}

pub(crate) fn run_native_logits_reference_command(
    command: &NativeLogitsReferenceCommand,
    k_strategy: ember::quant_k::KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let config = ExtractionConfig::from_path(&command.config)?;
    config.validate()?;
    if config.backend != ExecutionBackendName::Native {
        anyhow::bail!("native-logits-reference requires backend = \"native\"");
    }
    if !config.layers.is_empty() {
        anyhow::bail!("native-logits-reference is logits-only; set layers = []");
    }
    if !config.write_logits {
        anyhow::bail!("native-logits-reference requires write_logits = true");
    }

    let loader = load_gguf_with_k_strategy(&config.model_path, k_strategy, k_allow_fallback)?;
    let gguf_metadata = gguf_metadata_json(&loader);
    let arch = infer_extraction_architecture(&config, &gguf_metadata)?;
    let tokenizer_path = config
        .tokenizer_path
        .as_deref()
        .unwrap_or_else(|| default_tokenizer_for_arch(&arch));
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(tokenizer_path)?;
    let backend = CpuBackend;

    match arch.as_str() {
        "gpt2" => {
            let model = Gpt2::from_loader(loader)?;
            run_native_logits_reference_for_model(model, &backend, &tokenizer, &config, &arch, gguf_metadata)
        }
        "llama" | "qwen3" => {
            use ember::llama::Llama;
            let model = Llama::from_loader_with_max_seq_len(loader, config.max_seq_len)?;
            run_native_logits_reference_for_model(model, &backend, &tokenizer, &config, &arch, gguf_metadata)
        }
        "gemma4" => {
            use ember::gemma4::Gemma4;
            let model = Gemma4::from_loader(loader)?;
            run_native_logits_reference_for_model(model, &backend, &tokenizer, &config, &arch, gguf_metadata)
        }
        _ => anyhow::bail!(
            "unsupported native logits reference architecture '{}'; set architecture to gpt2, llama, qwen3, or gemma4",
            arch
        ),
    }
}

pub(crate) fn run_native_logits_reference_for_model<M>(
    model: M,
    backend: &CpuBackend,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    config: &ExtractionConfig,
    arch: &str,
    gguf_metadata: serde_json::Value,
) -> anyhow::Result<()>
where
    M: ForwardModel<CpuBackend>,
    <CpuBackend as Backend>::Error: Send + Sync + 'static,
{
    let samples = load_input_samples(config)?;
    let model_vocab_size = model.vocab_size(backend);
    tokenizer.validate_model_vocab(model_vocab_size)?;
    let final_run_dir = run_dir(config);
    let transaction = RunDirectoryTransaction::begin(&final_run_dir)?;
    let run_dir = transaction.staging_path().to_path_buf();

    let config_path = run_dir.join(CONFIG_FILENAME);
    let manifest_path = run_dir.join(MANIFEST_FILENAME);
    let samples_path = run_dir.join(SAMPLES_FILENAME);
    let tokenization_path = run_dir.join(TOKENIZATION_FILENAME);
    let positions_path = run_dir.join(POSITIONS_FILENAME);
    let checksums_path = run_dir.join(CHECKSUMS_FILENAME);
    let report_path = run_dir.join(REPORT_FILENAME);
    let logits_path = run_dir.join(LOGITS_FILENAME);

    let canonical_config = canonical_config_toml(config)?;
    fs::write(&config_path, &canonical_config)
        .with_context(|| format!("failed to write config: {}", config_path.display()))?;
    let config_hash = stable_bytes_hash(canonical_config.as_bytes());

    let mut sample_writer = fs::File::create(&samples_path)
        .with_context(|| format!("failed to create {}", samples_path.display()))?;
    let mut tokenization_writer = fs::File::create(&tokenization_path)
        .with_context(|| format!("failed to create {}", tokenization_path.display()))?;
    let mut positions_writer = fs::File::create(&positions_path)
        .with_context(|| format!("failed to create {}", positions_path.display()))?;

    let mut logits_writer: Option<NpyStreamWriter> = None;
    let mut logits_shape: Option<Vec<usize>> = None;
    let mut order_hash_inputs = Vec::with_capacity(samples.len());
    let model_context_limit = model.max_seq_len(backend);
    let context_limit = config
        .max_seq_len
        .unwrap_or(model_context_limit)
        .min(model_context_limit);

    for (sample_index, sample) in samples.iter().enumerate() {
        let (token_ids, offsets) = tokenizer
            .encode_with_offsets(&sample.prompt)
            .with_context(|| format!("failed to tokenize sample '{}'", sample.sample_id))?;
        if token_ids.is_empty() {
            anyhow::bail!("sample '{}' produced no token IDs", sample.sample_id);
        }
        validate_token_ids_for_model(
            &token_ids,
            model_vocab_size,
            &format!("sample '{}'", sample.sample_id),
        )?;
        let token_count = token_ids.len();
        ensure_sequence_fits(token_ids.len(), 0, context_limit)?;
        let selected_token_positions =
            select_token_positions(&token_ids, &offsets, config, sample.word_byte_span)?;
        if selected_token_positions != vec![token_ids.len() - 1] {
            anyhow::bail!(
                "native logits reference only supports final-token logits; sample '{}' selected {:?}",
                sample.sample_id,
                selected_token_positions
            );
        }

        let mut cache = model.create_cache(backend, context_limit);
        let logits = model.forward_last_logits_with_cache(backend, &token_ids, &mut cache, 0)?;
        validate_logits_tensor(backend, &logits, 1, model_vocab_size, true)?;
        let vocab_size = model_vocab_size;
        if logits_writer.is_none() {
            logits_writer = Some(NpyStreamWriter::create(
                &path_to_string(&logits_path)?,
                &[samples.len(), vocab_size],
            )?);
            logits_shape = Some(vec![samples.len(), vocab_size]);
        }
        logits_writer
            .as_mut()
            .expect("logits writer initialized above")
            .write_f32s(backend.data(&logits))?;

        let prompt_hash = stable_prompt_hash(&sample.prompt);
        order_hash_inputs.push((sample.sample_id.clone(), prompt_hash.clone()));
        serde_json::to_writer(
            &mut sample_writer,
            &SampleArtifactRecord {
                schema_version: ARTIFACT_CONTRACT_VERSION,
                sample_index,
                sample_id: sample.sample_id.clone(),
                input_index: sample.input_index,
                prompt: if config.prompt_hashes_only {
                    None
                } else {
                    Some(sample.prompt.clone())
                },
                prompt_hash: prompt_hash.clone(),
            },
        )?;
        sample_writer.write_all(b"\n")?;
        serde_json::to_writer(
            &mut tokenization_writer,
            &TokenizationArtifactRecord {
                schema_version: ARTIFACT_CONTRACT_VERSION,
                sample_index,
                sample_id: sample.sample_id.clone(),
                token_ids,
                token_count,
                prompt_hash: prompt_hash.clone(),
                offsets,
                offset_unit: "unicode_character_index".to_string(),
            },
        )?;
        tokenization_writer.write_all(b"\n")?;
        serde_json::to_writer(
            &mut positions_writer,
            &PositionArtifactRecord {
                schema_version: ARTIFACT_CONTRACT_VERSION,
                sample_index,
                sample_id: sample.sample_id.clone(),
                position_mode: config.token_position.as_str().to_string(),
                pooling: pooling_for_mode(config.token_position).to_string(),
                selected_token_positions,
                source_field: source_field_for_position(config),
                source_value: source_value_for_position(config, sample.word_value.as_deref()),
                source_byte_span: source_span_for_position(config, sample.word_byte_span)?,
            },
        )?;
        positions_writer.write_all(b"\n")?;
    }

    if let Some(writer) = &mut logits_writer {
        writer.finish()?;
    }
    sample_writer.flush()?;
    tokenization_writer.flush()?;
    positions_writer.flush()?;

    let logits_shape = logits_shape.context("no logits were written")?;
    let provenance = serde_json::json!({
        "real_logits": true,
        "no_generation": true,
        "no_hidden_states": true,
        "not_research_output": true,
        "purpose": "native logits reference smoke test",
    });
    let model_file_size_bytes = Some(
        fs::metadata(&config.model_path)
            .with_context(|| format!("failed to stat model: {}", config.model_path))?
            .len(),
    );
    let model_sha256 = if config.record_model_sha256 {
        Some(sha256_file_result(&config.model_path)?)
    } else {
        None
    };
    let tokenizer_path = config
        .tokenizer_path
        .as_deref()
        .unwrap_or_else(|| default_tokenizer_for_arch(arch));
    let tokenizer_file_size_bytes = fs::metadata(tokenizer_path)
        .with_context(|| format!("failed to stat tokenizer: {tokenizer_path}"))?
        .len();
    let tokenizer_metadata = TokenizerMetadata {
        path: tokenizer_path.to_string(),
        file_size_bytes: tokenizer_file_size_bytes,
        sha256: sha256_file_result(tokenizer_path)?,
    };
    let manifest = ArtifactManifest {
        schema_version: ARTIFACT_CONTRACT_VERSION,
        layout: ARTIFACT_LAYOUT.to_string(),
        artifact_kind: "ember_hidden_states".to_string(),
        created_at_unix: unix_timestamp(),
        run_id: config.run_id.clone(),
        run_dir: path_to_string(&final_run_dir)?,
        config_path: CONFIG_FILENAME.to_string(),
        samples_path: SAMPLES_FILENAME.to_string(),
        tokenization_path: TOKENIZATION_FILENAME.to_string(),
        positions_path: POSITIONS_FILENAME.to_string(),
        checksums_path: CHECKSUMS_FILENAME.to_string(),
        report_path: REPORT_FILENAME.to_string(),
        logits_path: Some(LOGITS_FILENAME.to_string()),
        tensor_contract: TensorContract {
            storage: "layer-sharded-npy".to_string(),
            dtype: config.dtype.as_str().to_string(),
            byte_order: "little-endian".to_string(),
            sample_axis: 0,
            hidden_axis: 1,
            layers: Vec::new(),
            logits: Some(LogitsArtifact {
                path: LOGITS_FILENAME.to_string(),
                shape: logits_shape.clone(),
            }),
        },
        sample_count: samples.len(),
        sample_order_hash: sample_order_hash(&order_hash_inputs),
        config_hash,
        dtype: config.dtype.as_str().to_string(),
        output_format: config.output_format.as_str().to_string(),
        model: ModelMetadata {
            path: config.model_path.clone(),
            architecture: Some(arch.to_string()),
            n_layers: model.n_layers(),
            embed_dim: model.embed_dim(),
            max_seq_len: model_context_limit,
            file_size_bytes: model_file_size_bytes,
            sha256: model_sha256,
            gguf_metadata,
        },
        tokenizer: Some(tokenizer_metadata),
        backend: BackendMetadata {
            name: ExecutionBackendName::Native.as_str().to_string(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
            executable: None,
            commit: git_commit(),
            details: serde_json::json!({
                "compute_backend": "CpuBackend",
                "crate": env!("CARGO_PKG_NAME"),
                "real_logits": true,
                "no_generation": true,
                "no_hidden_states": true,
                "not_research_output": true,
                "purpose": "native logits reference smoke test",
            }),
        },
        extraction_config: config.clone(),
    };
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    let report = serde_json::json!({
        "schema_version": ARTIFACT_CONTRACT_VERSION,
        "layout": ARTIFACT_LAYOUT,
        "status": "complete",
        "sample_count": samples.len(),
        "layer_count": 0,
        "logits_written": true,
        "logits_shape": logits_shape,
        "provenance": provenance,
        "real_logits": true,
        "no_generation": true,
        "no_hidden_states": true,
        "not_research_output": true,
        "purpose": "native logits reference smoke test",
    });
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;

    let mut checksums = BTreeMap::new();
    checksum_insert(&mut checksums, &config_path, CONFIG_FILENAME)?;
    checksum_insert(&mut checksums, &manifest_path, MANIFEST_FILENAME)?;
    checksum_insert(&mut checksums, &samples_path, SAMPLES_FILENAME)?;
    checksum_insert(&mut checksums, &tokenization_path, TOKENIZATION_FILENAME)?;
    checksum_insert(&mut checksums, &positions_path, POSITIONS_FILENAME)?;
    checksum_insert(&mut checksums, &report_path, REPORT_FILENAME)?;
    checksum_insert(&mut checksums, &logits_path, LOGITS_FILENAME)?;
    fs::write(&checksums_path, serde_json::to_string_pretty(&checksums)?)?;

    validate_artifact_contract(&run_dir, true)?;
    let published_run_dir = transaction.commit()?;

    eprintln!(
        "native logits reference wrote {} sample(s) to {}",
        samples.len(),
        published_run_dir.display()
    );
    eprintln!(
        "logits: {}",
        published_run_dir.join(LOGITS_FILENAME).display()
    );
    Ok(())
}

pub(crate) fn run_validate_run_command(command: &ValidateRunCommand) -> anyhow::Result<()> {
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

pub(crate) fn validate_report_fields(
    manifest: &ArtifactManifest,
    report: &serde_json::Value,
) -> anyhow::Result<()> {
    if report.get("status").and_then(serde_json::Value::as_str) != Some("complete") {
        anyhow::bail!("report status is not complete");
    }
    if let Some(schema_version) = report
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        && schema_version != u64::from(manifest.schema_version)
    {
        anyhow::bail!(
            "report schema_version {} does not match manifest schema_version {}",
            schema_version,
            manifest.schema_version
        );
    }
    if let Some(layout) = report.get("layout").and_then(serde_json::Value::as_str)
        && layout != manifest.layout
    {
        anyhow::bail!(
            "report layout '{}' does not match manifest layout '{}'",
            layout,
            manifest.layout
        );
    }
    Ok(())
}

pub(crate) fn collect_run_markers(
    manifest: &serde_json::Value,
    report: &serde_json::Value,
) -> serde_json::Map<String, serde_json::Value> {
    let marker_names = [
        "mock",
        "mock_backend",
        "no_inference",
        "real_llama_cpp",
        "real_tokenization",
        "real_logits",
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

pub(crate) fn collect_marker_values(
    value: &serde_json::Value,
    name: &str,
    observed: &mut Vec<bool>,
) {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(marker) = map.get(name)
                && let Some(bool_value) = marker.as_bool()
            {
                observed.push(bool_value);
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

pub(crate) fn validate_run_markers(
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

pub(crate) fn marker_is_true(
    markers: &serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> bool {
    markers
        .get(name)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

pub(crate) fn build_extraction_config(
    command: &ExtractCommand,
) -> anyhow::Result<ExtractionConfig> {
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

pub(crate) fn run_native_extract_command(
    config: &ExtractionConfig,
    k_strategy: ember::quant_k::KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let loader = load_gguf_with_k_strategy(&config.model_path, k_strategy, k_allow_fallback)?;
    let gguf_metadata = gguf_metadata_json(&loader);
    let arch = infer_extraction_architecture(config, &gguf_metadata)?;
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
            use ember::llama::Llama;
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

pub(crate) fn run_native_extract_for_model<M>(
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
        config
            .tokenizer_path
            .as_deref()
            .unwrap_or_else(|| default_tokenizer_for_arch(arch)),
        &config.model_path,
        Some(arch.to_string()),
        gguf_metadata,
        config.record_model_sha256,
    )?;
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

pub(crate) fn run_llama_cpp_external_extract_command(
    config: &ExtractionConfig,
) -> anyhow::Result<()> {
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

pub(crate) fn infer_extraction_architecture(
    config: &ExtractionConfig,
    gguf_metadata: &serde_json::Value,
) -> anyhow::Result<String> {
    let declared = gguf_metadata
        .get("general.architecture")
        .and_then(serde_json::Value::as_str)
        .context("GGUF is missing string metadata general.architecture")?;
    let detected = match declared {
        "gpt2" => "gpt2",
        "llama" => "llama",
        "qwen2" | "qwen3" => "qwen3",
        "gemma3" | "gemma4" => "gemma4",
        other => anyhow::bail!("unsupported GGUF architecture '{other}'"),
    };
    if let Some(requested) = config.architecture.as_deref() {
        let requested = match requested {
            "qwen2" | "qwen3" => "qwen3",
            "gemma3" | "gemma4" => "gemma4",
            "gpt2" => "gpt2",
            "llama" => "llama",
            other => anyhow::bail!("unsupported extraction architecture '{other}'"),
        };
        if requested != detected {
            anyhow::bail!(
                "extraction architecture '{requested}' conflicts with GGUF general.architecture='{declared}'"
            );
        }
    }
    Ok(detected.to_string())
}

pub(crate) fn effective_context_limit<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    args: &Args,
) -> usize {
    match args.max_seq_len {
        Some(cap) => cap.min(model.max_seq_len(backend)),
        None => model.max_seq_len(backend),
    }
}

pub(crate) fn ensure_sequence_fits(
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

fn validate_logits_tensor<B: Backend>(
    backend: &B,
    logits: &B::Tensor,
    expected_rows: usize,
    expected_vocab_size: usize,
    require_finite: bool,
) -> anyhow::Result<()> {
    let shape = backend.shape(logits);
    if shape != [expected_rows, expected_vocab_size] {
        anyhow::bail!(
            "logits shape mismatch: expected [{expected_rows}, {expected_vocab_size}], got {shape:?}"
        );
    }
    let expected_len = expected_rows
        .checked_mul(expected_vocab_size)
        .context("logits shape product overflow")?;
    let values = backend.data(logits);
    if values.len() != expected_len {
        anyhow::bail!(
            "logits payload has {} values, expected {expected_len}",
            values.len()
        );
    }
    if require_finite
        && let Some((index, value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("logits contain non-finite value {value} at flat index {index}");
    }
    Ok(())
}

pub(crate) fn run_bench_decode_command(
    command: &BenchDecodeCommand,
    k_strategy: ember::quant_k::KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    if command.tokens == 0 {
        anyhow::bail!("--tokens must be greater than 0");
    }
    if command.repetitions == 0 {
        anyhow::bail!("--repetitions must be greater than 0");
    }

    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, k_allow_fallback)?;
    let execution_inventory = ember::artifact::ExecutionInventory::from_loader(&loader);
    let architecture = resolve_generation_architecture(&command.arch, &loader)?;
    if command.profile_operators && !matches!(architecture.as_str(), "llama" | "qwen3") {
        anyhow::bail!(
            "--profile-operators is currently supported only for llama/qwen3 decode paths"
        );
    }
    let backend = CpuBackend;
    match architecture.as_str() {
        "gpt2" => {
            let model = Gpt2::from_loader(loader)?;
            bench_decode_model(&backend, &model, command, k_strategy, &execution_inventory)
        }
        "llama" | "qwen3" => {
            let model =
                ember::llama::Llama::from_loader_with_max_seq_len(loader, command.max_seq_len)?;
            let execution = ember::plan::ExecutionMode::from_cli(&command.execution)
                .map_err(anyhow::Error::msg)?;
            model.set_execution_mode(execution);
            bench_decode_model(&backend, &model, command, k_strategy, &execution_inventory)
        }
        "gemma4" => {
            let model = ember::gemma4::Gemma4::from_loader(loader)?;
            bench_decode_model(&backend, &model, command, k_strategy, &execution_inventory)
        }
        architecture => anyhow::bail!("unsupported architecture: {architecture}"),
    }
}

pub(crate) fn run_inspect_plan_command(
    command: &InspectPlanCommand,
    k_strategy: ember::quant_k::KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    use ember::llama::Llama;
    use ember::plan::{ExecutionMode, HookMode};

    let execution = ExecutionMode::from_cli(&command.execution).map_err(anyhow::Error::msg)?;
    let hook_mode = match command.hook.as_str() {
        "disabled" => HookMode::Disabled,
        "observe" => HookMode::Observe,
        "intervene" => HookMode::Intervene,
        other => anyhow::bail!(
            "unknown --hook value '{other}' (expected disabled | observe | intervene)"
        ),
    };
    let active_stages: Vec<String> = match command.hook_stages.as_deref() {
        None => Vec::new(),
        Some(stages) => stages
            .split(',')
            .map(|stage| stage.trim().to_string())
            .filter(|stage| !stage.is_empty())
            .collect(),
    };
    let stages: Vec<&str> = active_stages.iter().map(String::as_str).collect();

    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, k_allow_fallback)?;
    let architecture = resolve_generation_architecture(&command.arch, &loader)?;
    match architecture.as_str() {
        "llama" | "qwen3" => {
            let model = Llama::from_loader_with_max_seq_len(loader, command.max_seq_len)?;
            let max_seq_len = command.max_seq_len.unwrap_or(model.config.max_seq_len);
            let model_sha = ember::extraction::sha256_file(&command.model);
            let tokenizer_sha = command
                .tokenizer
                .as_deref()
                .and_then(ember::extraction::sha256_file);
            let plan = model.execution_plan(
                execution,
                hook_mode,
                &stages,
                max_seq_len,
                model_sha.as_deref(),
                tokenizer_sha.as_deref(),
            )?;
            print!("{}", plan.to_summary_text());
            if let Some(output) = &command.output {
                let json = serde_json::to_string_pretty(&*plan)?;
                std::fs::write(output, json)?;
                eprintln!("wrote execution plan to {output}");
            }
            Ok(())
        }
        architecture => anyhow::bail!(
            "inspect-plan supports llama-family models (--arch llama/qwen3), got '{architecture}'"
        ),
    }
}

pub(crate) fn run_bench_lifecycle_command(
    command: &BenchLifecycleCommand,
    k_strategy: ember::quant_k::KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    use ember::llama::{LlamaEvictionStats, LlamaPackedSelection, LlamaPackingStats};
    use ember::residency::ResidencyRecorder;

    if !cfg!(target_os = "linux") {
        anyhow::bail!("bench-lifecycle currently requires Linux procfs");
    }
    if command.tokens == 0 {
        anyhow::bail!("--tokens must be greater than 0");
    }

    let lifecycle_name = match command.lifecycle {
        LifecycleModeArg::Control => "control",
        LifecycleModeArg::PackBeforePrefill => "pack_before_prefill",
        LifecycleModeArg::PackAfterPrefill => "pack_after_prefill",
        LifecycleModeArg::PackBeforePrefillReevict => "pack_before_prefill_reevict",
        LifecycleModeArg::DuplicatePacked => "duplicate_packed",
    };
    let selection = LlamaPackedSelection::from(command.selection);
    let pack_before_prefill = matches!(
        command.lifecycle,
        LifecycleModeArg::PackBeforePrefill
            | LifecycleModeArg::PackBeforePrefillReevict
            | LifecycleModeArg::DuplicatePacked
    );
    let pack_after_prefill = matches!(command.lifecycle, LifecycleModeArg::PackAfterPrefill);
    let evict_after_pack = !matches!(
        command.lifecycle,
        LifecycleModeArg::Control | LifecycleModeArg::DuplicatePacked
    );
    let reevict_after_prefill = matches!(
        command.lifecycle,
        LifecycleModeArg::PackBeforePrefillReevict
    );

    let mut recorder = if command.timing_only {
        ResidencyRecorder::timing_only()
    } else {
        ResidencyRecorder::new()
    };
    recorder.capture("process_start")?;

    let model_init_start = Instant::now();
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, k_allow_fallback)?;
    let mut model =
        ember::model::Llama::from_loader_without_packed_decode(loader, command.max_seq_len)?;
    let model_init_ns = model_init_start.elapsed().as_nanos() as u64;
    recorder.capture("model_initialized")?;

    let tokenizer_init_start = Instant::now();
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(&command.tokenizer)?;
    let model_vocab_size = model.config.vocab_size;
    tokenizer.validate_model_vocab(model_vocab_size)?;
    let prompt_tokens = tokenizer.encode(&command.prompt)?;
    validate_token_ids_for_model(&prompt_tokens, model_vocab_size, "lifecycle prompt")?;
    let tokenizer_init_ns = tokenizer_init_start.elapsed().as_nanos() as u64;
    if prompt_tokens.len() < 2 {
        anyhow::bail!(
            "lifecycle prefill requires at least two prompt tokens to force the generic path"
        );
    }
    let required_context = prompt_tokens
        .len()
        .checked_add(command.tokens.saturating_sub(1))
        .context("lifecycle benchmark context length overflowed")?;
    if required_context > model.config.max_seq_len {
        anyhow::bail!(
            "lifecycle benchmark needs context {}, but model limit is {}",
            required_context,
            model.config.max_seq_len
        );
    }
    recorder.capture("tokenizer_initialized")?;

    let mut packing_stats = LlamaPackingStats::default();
    let mut post_pack_eviction_stats = LlamaEvictionStats::default();
    let mut post_prefill_eviction_stats = LlamaEvictionStats::default();
    let mut packing_wall_ns = 0_u64;
    let mut post_pack_eviction_wall_ns = 0_u64;
    let mut post_prefill_eviction_wall_ns = 0_u64;

    if pack_before_prefill {
        recorder.capture("packing_start")?;
        let packing_start = Instant::now();
        packing_stats = model.prepare_packed_decode_selected(selection, false)?;
        packing_wall_ns = packing_start.elapsed().as_nanos() as u64;
        recorder.capture("packing_complete")?;

        recorder.capture("post_pack_eviction_start")?;
        if evict_after_pack {
            let eviction_start = Instant::now();
            post_pack_eviction_stats = model.reevict_packed_decode_sources(selection)?;
            post_pack_eviction_wall_ns = eviction_start.elapsed().as_nanos() as u64;
        }
        recorder.capture("post_pack_eviction_complete")?;
    } else if !pack_after_prefill {
        // Keep the control run's phase accounting explicit and structurally
        // comparable even though no representation is constructed.
        recorder.capture("packing_start")?;
        recorder.capture("packing_complete")?;
        recorder.capture("post_pack_eviction_start")?;
        recorder.capture("post_pack_eviction_complete")?;
    }

    let backend = CpuBackend;
    let mut cache = model.create_cache(&backend, required_context);
    recorder.capture("prefill_start")?;
    let prefill_start = Instant::now();
    let logits = ForwardModel::forward_last_logits_with_cache(
        &model,
        &backend,
        &prompt_tokens,
        &mut cache,
        0,
    )?;
    let prefill_ns = prefill_start.elapsed().as_nanos() as u64;
    recorder.capture("prefill_complete")?;
    validate_logits_tensor(&backend, &logits, 1, model_vocab_size, true)?;

    let first_token_start = Instant::now();
    let vocab_size = model_vocab_size;
    let first_token = u32::try_from(argmax_token(&backend.data(&logits)[..vocab_size]))
        .context("first lifecycle token ID exceeds u32")?;
    if !tokenizer.contains_token_id(first_token) {
        anyhow::bail!(
            "model selected token ID {first_token}, but the lifecycle tokenizer cannot decode it"
        );
    }
    let first_token_selection_ns = first_token_start.elapsed().as_nanos() as u64;
    let mut generated = Vec::with_capacity(command.tokens);
    generated.push(first_token);

    if pack_after_prefill {
        recorder.capture("packing_start")?;
        let packing_start = Instant::now();
        packing_stats = model.prepare_packed_decode_selected(selection, false)?;
        packing_wall_ns = packing_start.elapsed().as_nanos() as u64;
        recorder.capture("packing_complete")?;

        recorder.capture("post_pack_eviction_start")?;
        let eviction_start = Instant::now();
        post_pack_eviction_stats = model.reevict_packed_decode_sources(selection)?;
        post_pack_eviction_wall_ns = eviction_start.elapsed().as_nanos() as u64;
        recorder.capture("post_pack_eviction_complete")?;
    }

    if reevict_after_prefill {
        recorder.capture("post_prefill_reeviction_start")?;
        let eviction_start = Instant::now();
        post_prefill_eviction_stats = model.reevict_packed_decode_sources(selection)?;
        post_prefill_eviction_wall_ns = eviction_start.elapsed().as_nanos() as u64;
        recorder.capture("post_prefill_reeviction_complete")?;
    } else {
        recorder.capture("post_prefill_reeviction_start")?;
        recorder.capture("post_prefill_reeviction_complete")?;
    }

    recorder.capture("decode_start")?;
    let decode_start = Instant::now();
    for step in 0..command.tokens.saturating_sub(1) {
        let logits = ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &[*generated.last().expect("first token exists")],
            &mut cache,
            prompt_tokens.len() + step,
        )?;
        // Keep timed decode limited to its existing argmax scan; shape and
        // payload length remain checked here, while argmax rejects NaNs.
        validate_logits_tensor(&backend, &logits, 1, model_vocab_size, false)?;
        let next_token = u32::try_from(argmax_token(&backend.data(&logits)[..vocab_size]))
            .context("lifecycle token ID exceeds u32")?;
        if !tokenizer.contains_token_id(next_token) {
            anyhow::bail!(
                "model selected token ID {next_token}, but the lifecycle tokenizer cannot decode it"
            );
        }
        generated.push(next_token);
    }
    let decode_ns = decode_start.elapsed().as_nanos() as u64;
    recorder.capture("decode_complete")?;

    let generated_text = tokenizer.decode(&generated)?;
    let generated_bytes = generated
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect::<Vec<_>>();
    let output_hash = stable_bytes_hash(&generated_bytes);
    let (packed_weights, packed_bytes) = model.packed_decode_summary(selection);

    let decode_evaluations = command.tokens.saturating_sub(1);
    let decode_evaluations_per_second = if decode_ns == 0 {
        None
    } else {
        Some(decode_evaluations as f64 * 1_000_000_000.0 / decode_ns as f64)
    };
    let time_to_first_token_work_ns = model_init_ns
        .saturating_add(tokenizer_init_ns)
        .saturating_add(if pack_before_prefill {
            packing_wall_ns.saturating_add(post_pack_eviction_wall_ns)
        } else {
            0
        })
        .saturating_add(prefill_ns)
        .saturating_add(first_token_selection_ns);
    let predecode_work_ns = model_init_ns
        .saturating_add(tokenizer_init_ns)
        .saturating_add(packing_wall_ns)
        .saturating_add(post_pack_eviction_wall_ns)
        .saturating_add(prefill_ns)
        .saturating_add(first_token_selection_ns)
        .saturating_add(post_prefill_eviction_wall_ns);

    drop(cache);
    drop(model);
    drop(tokenizer);
    recorder.capture("process_exit")?;

    let snapshots = recorder.snapshots();
    let first_snapshot = snapshots
        .first()
        .context("missing initial residency snapshot")?;
    let final_snapshot = snapshots
        .last()
        .context("missing final residency snapshot")?;
    let post_prefill = snapshots
        .iter()
        .find(|snapshot| snapshot.phase == "prefill_complete")
        .context("missing post-prefill residency snapshot")?;
    let peak_rss_kib = snapshots
        .iter()
        .map(|snapshot| snapshot.peak_rss_kib)
        .max()
        .unwrap_or(0);
    let measurement_ns = recorder.measurement_ns();
    let whole_process_until_exit_snapshot_ns = final_snapshot.elapsed_ns;
    let measurement_overhead_fraction = if whole_process_until_exit_snapshot_ns == 0 {
        0.0
    } else {
        measurement_ns as f64 / whole_process_until_exit_snapshot_ns as f64
    };

    let output = serde_json::json!({
        "schema_version": 2,
        "benchmark": "packed_lifecycle",
        "model": command.model,
        "model_file_size_bytes": fs::metadata(&command.model).ok().map(|metadata| metadata.len()),
        "tokenizer": command.tokenizer,
        "prompt": command.prompt,
        "prompt_tokens": prompt_tokens,
        "requested_generated_tokens": command.tokens,
        "generated_tokens": generated,
        "generated_text": generated_text,
        "output_hash": output_hash,
        "lifecycle": lifecycle_name,
        "selection": selection,
        "threads": rayon_current_num_threads(),
        "git_commit": git_commit(),
        "timings_ns": {
            "mmap_model_initialization": model_init_ns,
            "tokenizer_initialization_and_prompt_encoding": tokenizer_init_ns,
            "packing": packing_wall_ns,
            "packing_inner_sum": packing_stats.packing_ns,
            "post_pack_eviction": post_pack_eviction_wall_ns,
            "prefill": prefill_ns,
            "first_token_selection": first_token_selection_ns,
            "post_prefill_reeviction": post_prefill_eviction_wall_ns,
            "decode": decode_ns,
            "time_to_first_token_work": time_to_first_token_work_ns,
            "predecode_work": predecode_work_ns,
            "measurement_hooks": measurement_ns,
            "whole_process_until_exit_snapshot": whole_process_until_exit_snapshot_ns,
        },
        "decode_evaluations": decode_evaluations,
        "decode_evaluations_per_second": decode_evaluations_per_second,
        "packing": packing_stats,
        "post_pack_eviction": post_pack_eviction_stats,
        "post_prefill_reeviction": post_prefill_eviction_stats,
        "packed_weights": packed_weights,
        "packed_bytes": packed_bytes,
        "peak_rss_kib": peak_rss_kib,
        "post_prefill": {
            "rss_kib": post_prefill.rss_kib,
            "anonymous_pss_kib": post_prefill.anonymous_pss_kib,
            "file_pss_kib": post_prefill.file_pss_kib,
            "minor_faults": post_prefill.minor_faults,
            "major_faults": post_prefill.major_faults,
        },
        "faults": {
            "minor_total": final_snapshot.minor_faults.saturating_sub(first_snapshot.minor_faults),
            "major_total": final_snapshot.major_faults.saturating_sub(first_snapshot.major_faults),
        },
        "measurement_overhead_fraction": measurement_overhead_fraction,
        "residency_measurement_enabled": recorder.is_enabled(),
        "residency_snapshots": snapshots,
        "run_metadata": trace::collect_run_metadata(rayon_current_num_threads()),
        "timing_notes": [
            "phase timers exclude procfs snapshot time",
            "time_to_first_token_work excludes procfs snapshot time",
            "whole_process_until_exit_snapshot begins after CLI parsing and excludes JSON serialization",
            "the orchestration script records external process wall time"
        ],
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

pub(crate) fn bench_decode_model<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    command: &BenchDecodeCommand,
    k_strategy: ember::quant_k::KStrategy,
    execution_inventory: &ember::artifact::ExecutionInventory,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let model_vocab_size = model.vocab_size(backend);
    if command.token_id as usize >= model_vocab_size {
        anyhow::bail!(
            "benchmark token ID {} is outside model vocabulary size {model_vocab_size}",
            command.token_id
        );
    }
    let required_context = command
        .tokens
        .checked_add(1)
        .context("decode benchmark context length overflowed")?;
    let model_limit = model.max_seq_len(backend);
    let context_limit = command.max_seq_len.unwrap_or(model_limit).min(model_limit);
    if required_context > context_limit {
        anyhow::bail!(
            "decode benchmark needs context {}, but model limit is {}",
            required_context,
            context_limit
        );
    }

    // Allocation accounting (opt-in): per-token caller-thread events/bytes
    // via the counting allocator, plus process-global deltas across the
    // timed loop (includes worker-thread allocations). Only collected on
    // measured repetitions, never warmups.
    let mut token_alloc_events: Vec<usize> = Vec::new();
    let mut token_alloc_bytes: Vec<usize> = Vec::new();
    let mut global_alloc_events: Vec<usize> = Vec::new();
    let mut global_alloc_bytes: Vec<usize> = Vec::new();

    let mut run_once = |profile_operators: bool, track_allocations: bool| -> anyhow::Result<u64> {
        if profile_operators {
            ember::decode_profile::pause();
        }
        let mut cache = model.create_cache(backend, required_context);
        let prefill_logits =
            model.forward_last_logits_with_cache(backend, &[command.token_id], &mut cache, 0)?;
        validate_logits_tensor(backend, &prefill_logits, 1, model_vocab_size, true)?;
        if profile_operators {
            ember::decode_profile::resume();
        }
        let global_events_before = ember::alloc_counter::total_allocations();
        let global_bytes_before = ember::alloc_counter::total_allocated_bytes();
        let start = Instant::now();
        let mut final_logits = None;
        for position in 0..command.tokens {
            let mut forward = |position: usize| {
                model.forward_last_logits_with_cache(
                    backend,
                    &[command.token_id],
                    &mut cache,
                    position + 1,
                )
            };
            let (logits, events, bytes) = if track_allocations {
                ember::alloc_counter::count_allocations_with_bytes(|| forward(position))
            } else {
                (forward(position), 0, 0)
            };
            let logits = logits?;
            std::hint::black_box(backend.data(&logits));
            if track_allocations {
                token_alloc_events.push(events);
                token_alloc_bytes.push(bytes);
            }
            final_logits = Some(logits);
        }
        let elapsed_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let global_events_after = ember::alloc_counter::total_allocations();
        let global_bytes_after = ember::alloc_counter::total_allocated_bytes();
        validate_logits_tensor(
            backend,
            final_logits
                .as_ref()
                .expect("positive token count guarantees final logits"),
            1,
            model_vocab_size,
            true,
        )?;
        if elapsed_ns == 0 {
            anyhow::bail!("decode benchmark timer resolution produced a zero-duration sample");
        }
        let (delta_events, delta_bytes) = (
            global_events_after.saturating_sub(global_events_before),
            global_bytes_after.saturating_sub(global_bytes_before),
        );
        if track_allocations {
            global_alloc_events.push(delta_events);
            global_alloc_bytes.push(delta_bytes);
        }
        Ok(elapsed_ns)
    };

    for _ in 0..command.warmups {
        run_once(false, false)?;
    }
    let mut profile_session = command.profile_operators.then(DecodeProfileSession::start);
    let mut samples_ns = Vec::with_capacity(command.repetitions);
    for _ in 0..command.repetitions {
        samples_ns.push(run_once(command.profile_operators, command.allocations)?);
    }
    let operator_profile = profile_session.as_mut().map(DecodeProfileSession::finish);
    if operator_profile.as_ref().is_some_and(Vec::is_empty) {
        anyhow::bail!(
            "operator profiling produced no events; this model does not use the instrumented packed Q8 decode path or the planned interpreter"
        );
    }
    let allocation_report = if command.allocations {
        let token_count = command.tokens.max(1);
        let median = |values: &[usize]| -> Option<usize> {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            sorted.get(sorted.len() / 2).copied()
        };
        let sum = |values: &[usize]| -> usize { values.iter().sum() };
        let events_total = sum(&token_alloc_events);
        let bytes_total = sum(&token_alloc_bytes);
        let global_events_total = sum(&global_alloc_events);
        let global_bytes_total = sum(&global_alloc_bytes);
        Some(serde_json::json!({
            "schema_version": 1,
            "method": "counting allocator; caller-thread per-token counts via count_allocations_with_bytes, process-global deltas across the timed loop",
            "tokens": command.tokens,
            "caller_thread_alloc_events_total": events_total,
            "caller_thread_alloc_events_per_token": events_total as f64 / token_count as f64,
            "caller_thread_alloc_bytes_total": bytes_total,
            "caller_thread_alloc_bytes_per_token": bytes_total as f64 / token_count as f64,
            "caller_thread_alloc_events_median": median(&token_alloc_events),
            "caller_thread_alloc_bytes_median": median(&token_alloc_bytes),
            "caller_thread_alloc_events_min": token_alloc_events.iter().copied().min(),
            "caller_thread_alloc_events_max": token_alloc_events.iter().copied().max(),
            "caller_thread_alloc_bytes_min": token_alloc_bytes.iter().copied().min(),
            "caller_thread_alloc_bytes_max": token_alloc_bytes.iter().copied().max(),
            "per_token_alloc_events": token_alloc_events,
            "per_token_alloc_bytes": token_alloc_bytes,
            "global_alloc_events_total": global_events_total,
            "global_alloc_events_per_token": global_events_total as f64 / token_count as f64,
            "global_alloc_bytes_total": global_bytes_total,
            "global_alloc_bytes_per_token": global_bytes_total as f64 / token_count as f64,
            "global_alloc_events_median": median(&global_alloc_events),
            "global_alloc_bytes_median": median(&global_alloc_bytes),
        }))
    } else {
        None
    };
    let mut sorted = samples_ns.clone();
    sorted.sort_unstable();
    let median_ns = sorted[sorted.len() / 2];
    let samples_ts = samples_ns
        .iter()
        .map(|duration| command.tokens as f64 * 1_000_000_000.0 / *duration as f64)
        .collect::<Vec<_>>();
    let median_ts = command.tokens as f64 * 1_000_000_000.0 / median_ns as f64;

    let output = serde_json::json!({
        "schema_version": 1,
        "benchmark": "decode",
        "model": command.model,
        "model_file_size_bytes": fs::metadata(&command.model).ok().map(|metadata| metadata.len()),
        "architecture": command.arch,
        "git_commit": git_commit(),
        "k_strategy": k_strategy.name(),
        "k_tensor_count": execution_inventory.summary.tensor_count,
        "k_fallback_count": execution_inventory.summary.fallback_count,
        "k_compressed_bytes": execution_inventory.summary.compressed_bytes,
        "k_expanded_bytes": execution_inventory.summary.expanded_bytes,
        "tokens": command.tokens,
        "warmups": command.warmups,
        "repetitions": command.repetitions,
        "token_id": command.token_id,
        "threads": rayon_current_num_threads(),
        "timing_excludes": ["model_load", "prefill", "tokenization", "sampling"],
        "median_ns": median_ns,
        "median_tokens_per_second": median_ts,
        "samples_ns": samples_ns,
        "samples_tokens_per_second": samples_ts,
        "operator_profile": operator_profile,
        "allocation_report": allocation_report,
        "run_metadata": trace::collect_run_metadata(rayon_current_num_threads()),
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
