use super::*;

/// Load the spec file and resolve it, applying CLI overrides.
/// The spec text when the file declares `[sweep]`, otherwise `None`.
pub(super) fn read_sweep_spec(spec: &std::path::Path) -> anyhow::Result<Option<String>> {
    let text = std::fs::read_to_string(spec)
        .with_context(|| format!("cannot read experiment spec '{}'", spec.display()))?;
    Ok(crate::cli_experiment_sweep::is_sweep_spec(&text).then_some(text))
}

pub(super) fn resolve_spec_file(
    spec: &std::path::Path,
    execution: Option<&str>,
    threads: Option<usize>,
) -> anyhow::Result<(String, ember::v05::spec::ExperimentSpecV1)> {
    let spec_text = std::fs::read_to_string(spec)
        .with_context(|| format!("cannot read experiment spec '{}'", spec.display()))?;
    let raw = RawExperimentSpec::from_toml_str(&spec_text)
        .map_err(|error| anyhow::anyhow!("{}", error))?;
    let mut resolved = raw
        .resolve()
        .map_err(|error| anyhow::anyhow!("{}", error))?;
    if let Some(mode) = execution {
        resolved.execution.mode = ExecutionMode::from_cli(mode).map_err(anyhow::Error::msg)?;
    }
    if let Some(threads) = threads {
        resolved.execution.threads = threads;
    }
    Ok((spec_text, resolved))
}

pub(crate) fn run_validate_command(command: &ValidateArgs) -> anyhow::Result<()> {
    if let Some(text) = read_sweep_spec(&command.spec)? {
        return crate::cli_experiment_sweep::run_validate_sweep(command, &text);
    }
    let (_, resolved) = resolve_spec_file(&command.spec, None, None)?;
    let direction_files = crate::cli_experiment_steering::check_direction_files(&resolved)?;
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "schema": EXPERIMENT_SCHEMA_V1,
                "experiment": resolved.experiment.name,
                "execution_mode": resolved.execution.mode.name(),
                "captures": resolved.captures.len(),
                "interventions": resolved.interventions.len(),
                "direction_files_checked": direction_files,
                "attribution": resolved.attribution.is_some(),
                "probe": resolved.probe.is_some(),
                "inputs": resolved.inputs.len(),
                "defaults": resolved.defaults,
            }))?
        );
    } else {
        println!("specification OK");
        println!("  schema: {}", EXPERIMENT_SCHEMA_V1);
        println!("  experiment: {}", resolved.experiment.name);
        println!("  execution mode: {}", resolved.execution.mode.name());
        println!("  inputs: {}", resolved.inputs.len());
        println!("  captures: {}", resolved.captures.len());
        println!("  interventions: {}", resolved.interventions.len());
        if direction_files > 0 {
            println!("  direction files: {direction_files} (hash and shape checked)");
        }
        if let Some(attribution) = &resolved.attribution {
            println!(
                "  attribution: {} -> {}, {} site(s), verify top {}",
                attribution.clean,
                attribution.corrupted,
                attribution.sites.len(),
                attribution.verify_top_k
            );
        }
        if let Some(probe) = &resolved.probe {
            println!(
                "  probe bridge: {} layer {}, {} train / {} test examples, {} variant(s)",
                probe.site,
                probe.layer,
                probe.train.len(),
                probe.test.len(),
                probe.variants().len()
            );
            if let Some(file) = &probe.file {
                ember::v05::steering::read_direction_file(
                    &file.path,
                    &file.sha256,
                    file.tensor.as_deref(),
                )
                .map_err(|error| anyhow::anyhow!("probe.file: {error}"))?;
                println!("  probe file: hash and shape checked");
            }
        }
        println!("  defaults applied: {}", resolved.defaults.len());
    }
    Ok(())
}

pub(crate) fn run_experiment_command(
    command: &RunArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    if let Some(text) = read_sweep_spec(&command.spec)? {
        return crate::cli_experiment_sweep::run_sweep(
            command,
            &text,
            k_strategy,
            k_allow_fallback,
        );
    }
    let (spec_text, resolved) =
        resolve_spec_file(&command.spec, command.execution.as_deref(), command.threads)?;
    let output_directory = command
        .output
        .clone()
        .unwrap_or_else(|| resolved.output.directory.clone());
    if !command.variants.is_empty() {
        return run_with_variants(
            command,
            &spec_text,
            &resolved,
            &output_directory,
            k_strategy,
            k_allow_fallback,
        );
    }
    let (path, identity, report, _results) = execute_resolved(
        &resolved,
        &spec_text,
        &output_directory,
        k_strategy,
        k_allow_fallback,
        command.retain_incomplete,
    )?;
    if !report.ok {
        anyhow::bail!(
            "bundle self-verification failed: {} check(s) failed",
            report.checks.iter().filter(|check| !check.ok).count()
        );
    }
    if !command.json {
        crate::cli_experiment_attribution::print_bundle_reports(&path)?;
    }
    let sign_key = resolve_sign_key(command);
    let evidence = match &sign_key {
        Some(key) => Some(sign_bundle(&path, &identity, key)?),
        None => None,
    };
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "bundle": path.display().to_string(),
                "semantic_hash": identity.semantic_hash,
                "payload_hash": identity.payload_hash,
                "evidence": evidence.as_ref().map(|(envelope, signer)| serde_json::json!({
                    "path": envelope.display().to_string(),
                    "signer_fingerprint": signer,
                })),
                "verification": report,
            }))?
        );
    } else {
        println!("bundle written to {}", path.display());
        println!("  semantic hash: {}", identity.semantic_hash);
        println!("  payload hash:  {}", identity.payload_hash);
        println!("  verification: {} check(s) passed", report.checks.len());
        if let Some((envelope, signer)) = &evidence {
            println!(
                "  signed evidence: {} (signer {signer})",
                envelope.display()
            );
            println!(
                "    check with: ember experiment verify {} --trusted-key <key.pub>",
                path.display()
            );
        }
    }
    Ok(())
}

/// `experiment run base.toml --variant a.toml --variant b.toml`: one model
/// load, the base computes the shared prefix, each variant resumes from it.
pub(super) fn run_with_variants(
    command: &RunArgs,
    spec_text: &str,
    resolved: &ember::v05::spec::ExperimentSpecV1,
    output_directory: &std::path::Path,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let mut variants = Vec::with_capacity(command.variants.len());
    for path in &command.variants {
        variants.push(resolve_spec_file(
            path,
            command.execution.as_deref(),
            command.threads,
        )?);
    }
    let mut prepared = prepare_run(resolved, k_strategy, k_allow_fallback)?;
    let targets: Vec<RunTarget<'_>> = variants
        .iter()
        .map(|(text, spec)| RunTarget {
            resolved: spec,
            spec_text: text,
            output_directory: &spec.output.directory,
            retain_incomplete: command.retain_incomplete,
        })
        .collect();
    let (base, _, outcomes) = crate::cli_experiment_shared::execute_shared(
        &mut prepared,
        RunTarget {
            resolved,
            spec_text,
            output_directory,
            retain_incomplete: command.retain_incomplete,
        },
        &[],
        &targets,
        None,
    )?;
    let all: Vec<&RunOutcome> = std::iter::once(&base).chain(outcomes.iter()).collect();
    for outcome in &all {
        if !outcome.report.ok {
            anyhow::bail!(
                "bundle {} failed self-verification: {} check(s) failed",
                outcome.path.display(),
                outcome
                    .report
                    .checks
                    .iter()
                    .filter(|check| !check.ok)
                    .count()
            );
        }
    }
    let mut evidence = Vec::new();
    if let Some(key) = resolve_sign_key(command) {
        for outcome in &all {
            evidence.push(Some(sign_bundle(&outcome.path, &outcome.identity, &key)?));
        }
    } else {
        evidence.resize(all.len(), None);
    }
    if command.json {
        let bundles: Vec<serde_json::Value> = all
            .iter()
            .zip(&evidence)
            .map(|(outcome, evidence)| {
                serde_json::json!({
                    "bundle": outcome.path.display().to_string(),
                    "semantic_hash": outcome.identity.semantic_hash,
                    "payload_hash": outcome.identity.payload_hash,
                    "prefix_reuse": outcome.prefix.as_ref().map(|record| record.to_json()),
                    "evidence": evidence.as_ref().map(|(envelope, signer)| serde_json::json!({
                        "path": envelope.display().to_string(),
                        "signer_fingerprint": signer,
                    })),
                    "verification": outcome.report,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"ok": true, "bundles": bundles}))?
        );
    } else {
        for (outcome, evidence) in all.into_iter().zip(&evidence) {
            println!("bundle written to {}", outcome.path.display());
            if let Some((envelope, signer)) = evidence {
                println!(
                    "  signed evidence: {} (signer {signer})",
                    envelope.display()
                );
            }
            println!("  semantic hash: {}", outcome.identity.semantic_hash);
            println!("  payload hash:  {}", outcome.identity.payload_hash);
            println!(
                "  verification: {} check(s) passed",
                outcome.report.checks.len()
            );
            if let Some(record) = &outcome.prefix {
                println!("  prefix: {}", describe_prefix(record));
            }
        }
    }
    Ok(())
}

/// One line on how a bundle was computed.
pub(crate) fn describe_prefix(record: &ember::v05::prefix::PrefixReuseRecord) -> String {
    use ember::v05::prefix::PrefixPath;
    if record.inputs.is_empty() {
        return format!(
            "{}{}",
            record.role,
            record
                .note
                .as_ref()
                .map(|note| format!(" ({note})"))
                .unwrap_or_default()
        );
    }
    let parts: Vec<String> = record
        .inputs
        .iter()
        .map(|input| match &input.path {
            PrefixPath::Resumed { resume_layer } => {
                format!("{} resumed at block {resume_layer}", input.input_id)
            }
            PrefixPath::FullRecompute { reason } => {
                format!("{} recomputed in full ({reason})", input.input_id)
            }
        })
        .collect();
    format!("{}: {}", record.role, parts.join("; "))
}

/// The signing key a run uses: `--sign-key`, else `EMBER_SIGN_KEY`, unless
/// `--no-sign`.
pub(crate) fn resolve_sign_key(command: &RunArgs) -> Option<PathBuf> {
    if command.no_sign {
        None
    } else {
        command.sign_key.clone().or_else(|| {
            std::env::var_os(SIGN_KEY_ENV)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
    }
}

/// Sign `<bundle>/manifest.json` into `<bundle>.evidence.json`. Returns the
/// envelope path and the signer fingerprint.
pub(crate) fn sign_bundle(
    bundle: &std::path::Path,
    identity: &BundleIdentity,
    key: &std::path::Path,
) -> anyhow::Result<(PathBuf, String)> {
    let envelope_path = bundle_evidence_path(bundle);
    // Sign the identity this run produced and verified, not whatever
    // manifest.json holds by the time signing happens.
    let manifest_path = bundle.join("manifest.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&manifest_path)
            .with_context(|| format!("failed to read '{}'", manifest_path.display()))?,
    )
    .with_context(|| format!("'{}' is not valid JSON", manifest_path.display()))?;
    anyhow::ensure!(
        manifest["semantic_hash"].as_str() == Some(identity.semantic_hash.as_str())
            && manifest["payload_hash"].as_str() == Some(identity.payload_hash.as_str()),
        "refusing to sign bundle '{}': its manifest no longer carries the identity this run \
         produced",
        bundle.display()
    );
    let envelope =
        crate::cli_evidence::sign_record_value(&manifest, &key.to_string_lossy(), &envelope_path)
            .with_context(|| format!("failed to sign bundle '{}'", bundle.display()))?;
    let signer = envelope["signer_fingerprint"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    Ok((envelope_path, signer))
}

pub(crate) fn run_inspect_command(command: &InspectArgs) -> anyhow::Result<()> {
    if ember::v05::sweep::is_sweep_dir(&command.bundle) {
        return crate::cli_experiment_sweep::run_inspect_sweep(&command.bundle, command.json);
    }
    let bundle =
        ember::v05::verify::load_bundle_for_source(&command.bundle).map_err(anyhow::Error::msg)?;
    let manifest = bundle.semantic_manifest;
    let index = &bundle.capture_index;
    let summary = serde_json::json!({
        "bundle": command.bundle.display().to_string(),
        "bundle_schema": manifest.bundle_schema,
        "experiment": manifest.experiment.name,
        "model_sha256": manifest.model.sha256,
        "architecture": manifest.model.architecture,
        "execution_mode": manifest.execution.mode,
        "plan_hash": manifest.execution.plan_hash,
        "inputs": manifest.inputs.iter().map(|input| input.id.clone()).collect::<Vec<_>>(),
        "captures": index.len(),
        "interventions": manifest.interventions.len(),
        "generated_token_ids": manifest.generated.token_ids,
        "payloads": manifest.payloads.keys().collect::<Vec<_>>(),
        "warnings": manifest.warnings,
    });
    if command.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("experiment bundle: {}", command.bundle.display());
        println!("  schema: {}", manifest.bundle_schema);
        println!("  experiment: {}", manifest.experiment.name);
        println!(
            "  model: {} ({})",
            manifest
                .model
                .sha256
                .get(..12)
                .unwrap_or(&manifest.model.sha256),
            manifest.model.architecture
        );
        println!(
            "  execution: {} plan {}",
            manifest.execution.mode,
            manifest
                .execution
                .plan_hash
                .get(..12)
                .unwrap_or(&manifest.execution.plan_hash)
        );
        println!(
            "  inputs: {:?}",
            manifest
                .inputs
                .iter()
                .map(|i| i.id.clone())
                .collect::<Vec<_>>()
        );
        println!("  captures: {}", index.len());
        for entry in index.iter().take(10) {
            let tensor = if entry.summary.is_some() {
                "summary".to_string()
            } else {
                entry
                    .shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            };
            println!(
                "    {} @ {} layer {}: {} [{}]",
                entry.capture_id, entry.site, entry.layer, tensor, entry.dtype
            );
        }
        if index.len() > 10 {
            println!("    ... {} more", index.len() - 10);
        }
        println!("  interventions: {}", manifest.interventions.len());
        println!("  warnings: {}", manifest.warnings.len());
    }
    Ok(())
}

/// Clap parser for an expected SHA-256: 64 hex characters, lowercased.
pub(super) fn parse_sha256_hex(value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(value.to_ascii_lowercase())
    } else {
        Err("expected a SHA-256 as 64 hex characters".into())
    }
}

/// Check a signed evidence envelope over a bundle's `manifest.json` against
/// the bundle's recomputed identity. Returns (ok, detail).
pub(super) fn evidence_anchor(
    envelope: &std::path::Path,
    trusted_key: &str,
    semantic_hash: &str,
    payload_hash: &str,
) -> (bool, String) {
    let verified =
        match crate::cli_evidence::verify_envelope_file_with_trusted_key(envelope, trusted_key) {
            Ok(verified) => verified,
            Err(error) => return (false, format!("{error:#}")),
        };
    let input = &verified.input;
    let field = |name: &str| input.get(name).and_then(|value| value.as_str());
    if field("kind") != Some(ember::v05::manifest::BUNDLE_KIND) {
        return (
            false,
            "the signed record is not an experiment bundle manifest.json".into(),
        );
    }
    let semantic_ok = field("semantic_hash") == Some(semantic_hash);
    let payload_ok = field("payload_hash") == Some(payload_hash);
    let signer = verified
        .signer_fingerprint
        .get(..12)
        .unwrap_or(&verified.signer_fingerprint);
    let timestamp = if verified.timestamp_signed {
        ""
    } else {
        "; its signing time is not covered by the v1 signature"
    };
    if semantic_ok && payload_ok {
        (
            true,
            format!(
                "{} from trusted signer {signer} binds this semantic and payload hash{timestamp}",
                verified.schema
            ),
        )
    } else {
        (
            false,
            format!(
                "signed record (trusted signer {signer}) names semantic {} and payload {}, not \
                 this bundle's (semantic {}, payload {})",
                field("semantic_hash").unwrap_or("-"),
                field("payload_hash").unwrap_or("-"),
                semantic_ok,
                payload_ok
            ),
        )
    }
}

/// Load and verify a bundle with its anchors; fails on any failed check.
pub(crate) fn load_anchored_bundle(
    bundle: &std::path::Path,
    anchor: &AnchorArgs,
) -> anyhow::Result<LoadedBundle> {
    let options = VerifyOptions {
        expected_semantic_hash: anchor.expect_semantic_hash.clone(),
        ..VerifyOptions::default()
    };
    let loaded = load_verified_bundle(bundle, &options).map_err(anyhow::Error::msg)?;
    if let Some((envelope, trusted_key)) = anchor.evidence_for(bundle) {
        let (ok, detail) = evidence_anchor(
            &envelope,
            trusted_key,
            &loaded.semantic_hash,
            &loaded.payload_hash,
        );
        anyhow::ensure!(
            ok,
            "bundle '{}' failed its evidence anchor: {detail}",
            bundle.display()
        );
    }
    Ok(loaded)
}

/// Refuse a report path inside the bundle: verification must not add files
/// to the bundle it checks.
/// Fail unless `path` would be written outside `bundle` (a bundle or sweep
/// directory); commands that only read a bundle never write into it.
/// `refusal` is the error message.
pub(crate) fn ensure_outside_bundle(
    bundle: &std::path::Path,
    path: &std::path::Path,
    refusal: &str,
) -> anyhow::Result<()> {
    let bundle_root = bundle
        .canonicalize()
        .with_context(|| format!("cannot resolve '{}'", bundle.display()))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let parent = parent
        .canonicalize()
        .with_context(|| format!("cannot resolve '{}'", parent.display()))?;
    anyhow::ensure!(!parent.starts_with(&bundle_root), "{refusal}");
    Ok(())
}

pub(super) fn write_report_outside(
    bundle: &std::path::Path,
    path: &std::path::Path,
    report: &ember::v05::verify::VerificationReport,
) -> anyhow::Result<()> {
    ensure_outside_bundle(
        bundle,
        path,
        "--write-report must point outside the bundle; verification never modifies the bundle",
    )?;
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    ember::atomic_file::atomic_write(path, &bytes)
        .with_context(|| format!("cannot write report '{}'", path.display()))
}

pub(crate) fn run_verify_command(command: &VerifyArgs) -> anyhow::Result<()> {
    let options = VerifyOptions {
        model_path: command.model.clone(),
        tokenizer_path: command.tokenizer.clone(),
        expected_semantic_hash: command.anchor.expect_semantic_hash.clone(),
    };
    if ember::v05::sweep::is_sweep_dir(&command.bundle) {
        return crate::cli_experiment_sweep::run_verify_sweep(
            &command.bundle,
            &options,
            command.anchor.trusted_key.is_some(),
            command.write_report.as_deref(),
            command.json,
        );
    }
    let mut report = verify_bundle(&command.bundle, &options).map_err(anyhow::Error::msg)?;
    if let Some((envelope, trusted_key)) = command.anchor.evidence_for(&command.bundle) {
        let (ok, detail) = if report.semantic_hash.is_empty() {
            (
                false,
                "the bundle's identity could not be recomputed".to_string(),
            )
        } else {
            evidence_anchor(
                &envelope,
                trusted_key,
                &report.semantic_hash,
                &report.payload_hash,
            )
        };
        report.add_check("evidence anchor", ok, detail);
    }
    let anchored = command.anchor.is_anchored();
    if !anchored {
        report.warnings.push(
            "not anchored: 'verified' means the bundle is self-consistent. Every hash it checks \
             is written by the bundle's producer, so an edited bundle can be resealed; pass \
             --expect-semantic-hash or --expect-evidence/--trusted-key to bind it to an identity \
             obtained elsewhere"
                .into(),
        );
    }
    if let Some(path) = &command.write_report {
        write_report_outside(&command.bundle, path, &report)?;
    }
    if command.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("verification of {}", command.bundle.display());
        for check in &report.checks {
            println!(
                "  [{}] {}: {}",
                if check.ok { "ok" } else { "FAIL" },
                check.name,
                check.detail
            );
        }
        println!("verdict: {}", if report.ok { "verified" } else { "FAILED" });
        if report.ok {
            println!();
            println!("  semantic hash: {}", report.semantic_hash);
            println!("  payload hash:  {}", report.payload_hash);
            if anchored {
                println!("  anchored: the identity matches the externally supplied value");
            } else {
                println!(
                    "  note: 'verified' means self-consistent only; every hash checked is \
                     bundle-authored."
                );
                println!(
                    "        Record the semantic hash somewhere you trust and pass \
                     --expect-semantic-hash <hex>"
                );
                println!(
                    "        (or --expect-evidence <envelope> --trusted-key <key.pub>) to \
                     detect a resealed bundle."
                );
            }
        }
        for warning in report
            .warnings
            .iter()
            .filter(|warning| !warning.starts_with("not anchored"))
        {
            println!("  warning: {warning}");
        }
    }
    if !report.ok {
        return Err(crate::cli_support::VerificationFailed.into());
    }
    Ok(())
}

pub(crate) fn run_compare_command(command: &CompareArgs) -> anyhow::Result<()> {
    if ember::v05::sweep::is_sweep_dir(&command.a) || ember::v05::sweep::is_sweep_dir(&command.b) {
        return crate::cli_experiment_sweep::run_compare_sweeps(
            &command.a,
            &command.b,
            command.expect_a_semantic_hash.as_deref(),
            command.expect_b_semantic_hash.as_deref(),
            command.json,
        );
    }
    let anchor = |expected: &Option<String>| AnchorArgs {
        expect_semantic_hash: expected.clone(),
        ..AnchorArgs::default()
    };
    let bundle_a = load_anchored_bundle(&command.a, &anchor(&command.expect_a_semantic_hash))?;
    let bundle_b = load_anchored_bundle(&command.b, &anchor(&command.expect_b_semantic_hash))?;
    let result = compare_loaded(&bundle_a, &bundle_b).map_err(anyhow::Error::msg)?;
    let host = host_differences(&bundle_a, &bundle_b);
    if command.json {
        let mut value = serde_json::to_value(&result)?;
        value["host_differences"] = serde_json::to_value(&host)?;
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }
    let identity = &result.identity;
    println!(
        "comparing {} vs {}",
        command.a.display(),
        command.b.display()
    );
    println!("identity:");
    println!("  schema compatible: {}", yesno(identity.schema_compatible));
    println!(
        "  semantic hash equal: {}",
        yesno(identity.semantic_hash_equal)
    );
    println!("  model hash equal: {}", yesno(identity.model_hash_equal));
    println!(
        "  tokenizer hash equal: {}",
        yesno(identity.tokenizer_hash_equal)
    );
    println!(
        "  execution mode equal: {}",
        yesno(identity.execution_mode_equal)
    );
    println!("  plan hash equal: {}", yesno(identity.plan_hash_equal));
    println!("  input ids equal: {}", yesno(identity.input_ids_equal));
    println!("  prompts equal: {}", yesno(identity.prompts_equal));
    println!(
        "  tokenization equal: {}",
        yesno(identity.tokenization_equal)
    );
    println!("outputs:");
    for output in &result.outputs {
        println!(
            "  {}: tokens {} text {} top1 {} divergence {}",
            output.input_id,
            yesno(output.generated_tokens_equal),
            yesno(output.generated_text_equal),
            yesno(output.final_top1_equal),
            output
                .first_divergence_step
                .map(|step| format!("step {step}"))
                .unwrap_or_else(|| "none".into())
        );
    }
    println!("captures:");
    for capture in &result.captures {
        if let Some(metrics) = &capture.metrics {
            println!(
                "  {} @ {} layer {}: exact {} max-abs {:.2e} mean-abs {:.2e} rel-l2 {:.2e} cosine {:.4}",
                capture.capture_id,
                capture.site,
                capture.layer,
                yesno(metrics.exact),
                metrics.maximum_absolute_difference.unwrap_or(f64::NAN),
                metrics.mean_absolute_difference.unwrap_or(f64::NAN),
                metrics.relative_l2_difference.unwrap_or(f64::NAN),
                metrics.cosine_similarity.unwrap_or(f64::NAN),
            );
        } else {
            println!(
                "  {} @ {} layer {}: present only in {}",
                capture.capture_id,
                capture.site,
                capture.layer,
                if capture.present_in_a { "a" } else { "b" }
            );
        }
    }
    println!("interventions:");
    for intervention in &result.interventions {
        println!(
            "  {}: operation {} source {} tokens {} defusion-route {} ({} vs {} events)",
            intervention.intervention_id,
            yesno(intervention.operation_equal),
            yesno(intervention.source_equal),
            yesno(intervention.selected_tokens_equal),
            yesno(intervention.defusion_route_equal),
            intervention.events_in_a,
            intervention.events_in_b,
        );
    }
    println!("runtime (not semantic):");
    println!(
        "  decode tps: {} vs {}",
        fmt_opt(result.runtime.decode_throughput_tps_a),
        fmt_opt(result.runtime.decode_throughput_tps_b)
    );
    println!(
        "  peak rss kb: {} vs {}",
        fmt_opt_u64(result.runtime.peak_rss_kb_a),
        fmt_opt_u64(result.runtime.peak_rss_kb_b)
    );
    if results_differ(&result) {
        println!("host (why the numbers may differ):");
        for line in ember::v05::host_profile::report_lines(&host) {
            println!("{line}");
        }
    } else if !host.differences.is_empty() {
        println!(
            "host: {} difference(s), none of which changed the results",
            host.differences.len()
        );
    }
    Ok(())
}

/// Host differences between two bundles, from their runtime.json files.
pub(super) fn host_differences(
    a: &LoadedBundle,
    b: &LoadedBundle,
) -> ember::v05::host_profile::HostDifferenceReport {
    let runtime = |bundle: &LoadedBundle| {
        bundle
            .file("runtime.json")
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(bytes).ok())
    };
    ember::v05::host_profile::explain_host_differences(runtime(a).as_ref(), runtime(b).as_ref())
}

/// Whether any output or capture differs between the compared bundles.
pub(super) fn results_differ(result: &ember::v05::compare::CompareResult) -> bool {
    result
        .outputs
        .iter()
        .any(|output| !output.generated_tokens_equal || !output.generated_text_equal)
        || result.captures.iter().any(|capture| {
            capture
                .metrics
                .as_ref()
                .map(|metrics| !metrics.exact)
                .unwrap_or(true)
        })
}

pub(super) fn yesno(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

pub(super) fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.2}"))
        .unwrap_or_else(|| "-".into())
}

pub(super) fn fmt_opt_u64(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "-".into())
}

/// Worker threads for a resolved spec: its explicit count, or every core.
///
/// Specs also arrive from bundles (`reproduce`), so the count is untrusted
/// and bounded before a pool is built from it.
pub(crate) fn pool_threads(resolved: &ember::v05::spec::ExperimentSpecV1) -> anyhow::Result<usize> {
    const MAX_THREADS: usize = 1024;
    let requested = resolved.execution.threads;
    anyhow::ensure!(
        requested <= MAX_THREADS,
        "execution.threads = {requested} exceeds the limit of {MAX_THREADS}"
    );
    Ok(if requested > 0 {
        requested
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

pub(crate) fn run_reproduce_command(
    command: &ReproduceArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    // Verify (and anchor) the original once; everything below uses the
    // verified bytes, never a second read of the bundle.
    let original = load_anchored_bundle(&command.bundle, &command.anchor)?;
    let manifest = &original.semantic_manifest;
    let spec_text = std::str::from_utf8(
        original
            .required_file("experiment.toml")
            .map_err(anyhow::Error::msg)?,
    )
    .context("bundle experiment.toml is not UTF-8")?
    .to_string();
    // resolved-experiment.json is outside the semantic hash; it is used
    // only after it has been bound to the hashed spec and manifest.
    let mut resolved =
        ember::v05::verify::bound_resolved_experiment(&original).map_err(anyhow::Error::msg)?;

    // Validate the supplied model against the bundle's recorded hash.
    let model_sha = sha256_file_result(&command.model)
        .with_context(|| format!("failed to hash '{}'", command.model.display()))?;
    if model_sha != manifest.model.sha256 {
        anyhow::bail!(
            "model '{}' hashes to {} but the bundle records {}; reproduction requires the \
             identical model file",
            command.model.display(),
            model_sha,
            manifest.model.sha256
        );
    }
    resolved.model.path = command.model.clone();
    resolved.model.expected_sha256 = manifest.model.sha256.clone();
    // The tokenizer path comes from the bundle; whatever file it (or
    // --tokenizer) names must be the tokenizer the bundle recorded.
    if let Some(tokenizer) = &command.tokenizer {
        resolved.model.tokenizer = Some(tokenizer.clone());
    }
    resolved.model.tokenizer_expected_sha256 = manifest.tokenizer.sha256.clone();
    // Nothing from the bundle may widen what this command does to the
    // filesystem: replacing an existing directory needs the user's say-so,
    // not the bundle's.
    resolved.output.overwrite = false;
    let output = command
        .output
        .clone()
        .unwrap_or_else(|| command.bundle.with_extension("reproduced"));

    let (path, identity, report, _results) = execute_resolved(
        &resolved,
        &spec_text,
        &output,
        k_strategy,
        k_allow_fallback,
        command.retain_incomplete,
    )?;
    if !report.ok {
        anyhow::bail!("reproduction bundle failed self-verification");
    }

    // Classify against the original, as verified above.
    let reproduction =
        ember::v05::verify::load_bundle_for_source(&path).map_err(anyhow::Error::msg)?;
    let comparison = compare_loaded(&original, &reproduction).map_err(anyhow::Error::msg)?;
    let tokens_equal = comparison
        .outputs
        .iter()
        .all(|output| output.generated_tokens_equal);
    let captures_declared = !comparison.captures.is_empty();
    let captures_aligned = captures_declared
        && comparison
            .captures
            .iter()
            .all(|capture| capture.present_in_a && capture.present_in_b);
    let captures_exact = captures_aligned
        && comparison
            .captures
            .iter()
            .all(|capture| capture.metrics.as_ref().map(|m| m.exact).unwrap_or(false));
    let captures_within_envelope = captures_aligned
        && comparison.captures.iter().all(|capture| {
            capture
                .metrics
                .as_ref()
                .and_then(|m| m.maximum_absolute_difference)
                .map(|diff| diff <= 1e-4)
                .unwrap_or(false)
        });
    let captures_misaligned = captures_declared && !captures_aligned;
    let top1_equal = comparison
        .outputs
        .iter()
        .all(|output| output.final_top1_equal);
    // Output agreement means nothing unless both ran the same inputs: outputs
    // are paired by position, so a reproduction over a subset or different
    // prompts could otherwise still grade as exact.
    let inputs_equal = comparison.identity.input_ids_equal && comparison.identity.prompts_equal;
    let verdict = if comparison.identity.semantic_hash_equal {
        "exact-semantic"
    } else if !inputs_equal {
        "inputs-differ"
    } else if tokens_equal && (captures_exact || !captures_declared) {
        "exact"
    } else if tokens_equal && (!captures_declared || captures_within_envelope) {
        "output-equivalent"
    } else if captures_misaligned {
        "captures-misaligned"
    } else if top1_equal {
        "top1-equivalent"
    } else {
        "failed"
    };
    let host = host_differences(&original, &reproduction);
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "host_differences": host,
                "verdict": verdict,
                "original": command.bundle.display().to_string(),
                "reproduction": path.display().to_string(),
                "original_semantic_hash": original.semantic_hash,
                "original_anchored": command.anchor.is_anchored(),
                "semantic_hash": identity.semantic_hash,
                "inputs_equal": inputs_equal,
                "tokens_equal": tokens_equal,
                "captures_declared": captures_declared,
                "captures_aligned": captures_aligned,
                "captures_exact": captures_exact,
                "captures_within_envelope": captures_within_envelope,
                "top1_equal": top1_equal,
            }))?
        );
    } else {
        println!("reproduction written to {}", path.display());
        println!("  verdict: {verdict}");
        println!(
            "  tokens equal: {}; captures exact: {}; captures aligned: {}; top1 equal: {}",
            yesno(tokens_equal),
            yesno(captures_exact),
            yesno(captures_aligned),
            yesno(top1_equal)
        );
        println!("  semantic hash: {}", identity.semantic_hash);
        println!(
            "  original semantic hash: {} ({})",
            original.semantic_hash,
            if command.anchor.is_anchored() {
                "anchored"
            } else {
                "not anchored: self-consistent only"
            }
        );
        if !matches!(verdict, "exact-semantic" | "exact") {
            println!("  host (why the numbers may differ):");
            for line in ember::v05::host_profile::report_lines(&host) {
                println!("  {line}");
            }
        }
    }
    if verdict == "failed" || verdict == "captures-misaligned" {
        return Err(crate::cli_support::VerificationFailed.into());
    }
    Ok(())
}

pub(crate) fn run_tokenize_command(
    command: &TokenizeArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let loader = load_gguf_with_k_strategy(&command.model, k_strategy, k_allow_fallback)?;
    let architecture = ember::loader::resolve_generation_architecture(&command.arch, &loader)?;
    let tokenizer_path = command
        .tokenizer
        .clone()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| default_tokenizer_for_arch(&architecture).to_string());
    let resolved_tokenizer = resolve_tokenizer(&tokenizer_path);
    let tokenizer: EmberTokenizer = resolved_tokenizer.load()?;
    let info = tokenize_for_selection(&tokenizer, &command.text, TextNormalization::None)
        .map_err(anyhow::Error::msg)?;
    let selection = match &command.match_span {
        Some(span) => {
            let selector = ember::v05::token_select::TokenSelector::MatchedTextSpan {
                text: span.clone(),
                occurrence: 0,
                subtoken_selection: ember::v05::token_select::SubtokenSelection::All,
                normalization: TextNormalization::None,
            };
            Some(
                ember::v05::token_select::resolve_static_selector(&selector, &info)
                    .map_err(anyhow::Error::msg)?,
            )
        }
        None => None,
    };
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "token_ids": info.token_ids,
                "pieces": info.pieces,
                "byte_offsets": info.byte_offsets,
                "selection": selection,
            }))?
        );
    } else {
        println!("tokenization of {:?}", command.text);
        for (index, ((id, piece), offset)) in info
            .token_ids
            .iter()
            .zip(info.pieces.iter())
            .zip(info.byte_offsets.iter())
            .enumerate()
        {
            println!("  [{index}] id={id:<8} bytes={offset:?} {piece:?}");
        }
        if let Some(selection) = selection {
            println!(
                "match {:?}: span {:?}, selected {:?}, coverage {:?}",
                command.match_span.as_deref().unwrap_or(""),
                selection.matched_byte_span,
                selection.selected_indices,
                selection.coverage
            );
        }
    }
    Ok(())
}
