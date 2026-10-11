//! `ember experiment run|validate|verify|compare|inspect` for layer sweeps
//! (`ember::v05::sweep`).
//!
//! A sweep runs as one shared pass: the baseline computes the prompt prefix
//! once and every point resumes from it at its own layer
//! (`crate::cli_experiment_shared`). On the Q8_0 fast decode path the
//! baseline and every point then decode together: each decode step is one
//! batched forward over all generations still running, which reads every
//! weight once for the whole batch (bit-identical per sequence). Otherwise
//! points run one after another. Running them on concurrent threads would
//! not help: the model forward is not reentrant on one thread (the fused
//! decode path and the greedy logits buffer live in thread-local
//! `RefCell`s), and each forward already uses the whole pool.

use crate::cli_experiment::{
    describe_prefix, prepare_run, RunArgs, RunOutcome, RunTarget, ValidateArgs,
};
use anyhow::Context;
use ember::plan::ExecutionMode;
use ember::quant_k::KStrategy;
use ember::v05::effect::SWEEP_EFFECT_CSV_FILE;
use ember::v05::spec::ExperimentSpecV1;
use ember::v05::sweep::{
    point_metrics_with_effect, verify_sweep, DerivedSpec, SweepBundleRef, SweepDefinition,
    SweepManifest, SweepPointRecord, SweepVerification, SWEEP_CSV_FILE, SWEEP_MANIFEST_FILE,
    SWEEP_RUNTIME_FILE, SWEEP_SCHEMA_V1, SWEEP_SPEC_FILE,
};
use ember::v05::verify::{load_bundle_for_source, VerifyOptions};
use std::path::{Path, PathBuf};

/// Whether a spec text declares `[sweep]` (false for anything unparsable;
/// the ordinary path then reports the parse error).
pub(crate) fn is_sweep_spec(text: &str) -> bool {
    ember::v05::spec::RawExperimentSpec::from_toml_str(text)
        .map(|raw| raw.sweep.is_some())
        .unwrap_or(false)
}

fn apply_overrides(
    spec: &mut ExperimentSpecV1,
    execution: Option<&str>,
    threads: Option<usize>,
) -> anyhow::Result<()> {
    if let Some(mode) = execution {
        spec.execution.mode = ExecutionMode::from_cli(mode).map_err(anyhow::Error::msg)?;
    }
    if let Some(threads) = threads {
        spec.execution.threads = threads;
    }
    Ok(())
}

pub(crate) fn run_validate_sweep(command: &ValidateArgs, text: &str) -> anyhow::Result<()> {
    let definition = SweepDefinition::parse(text).map_err(|error| anyhow::anyhow!("{error}"))?;
    let baseline = definition
        .baseline()
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let layers = serde_json::to_value(&definition.layers)?;
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "schema": ember::v05::spec::EXPERIMENT_SCHEMA_V1,
                "sweep_schema": SWEEP_SCHEMA_V1,
                "experiment": definition.name,
                "execution_mode": definition.template.execution.mode.name(),
                "inputs": definition.template.inputs.len(),
                "captures": definition.template.captures.len(),
                "interventions": definition.template.interventions.len(),
                "swept_interventions": definition.interventions,
                "layers": layers,
                "positions": definition.positions,
                "alphas": definition.alphas,
                "move_bundle_sources": definition.move_bundle_sources,
                "effect": definition.effect.as_ref().map(|effect| serde_json::json!({
                    "capture": effect.capture,
                    "targets": effect.targets.len(),
                    "confidence": effect.confidence,
                    "resamples": effect.resamples,
                    "seed": effect.seed,
                })),
                "baseline_interventions": baseline.resolved.interventions.len(),
            }))?
        );
    } else {
        println!("sweep specification OK");
        println!("  schema: {}", ember::v05::spec::EXPERIMENT_SCHEMA_V1);
        println!("  experiment: {}", definition.name);
        println!(
            "  execution mode: {}",
            definition.template.execution.mode.name()
        );
        println!("  inputs: {}", definition.template.inputs.len());
        println!("  captures: {}", definition.template.captures.len());
        println!(
            "  swept interventions: {}",
            definition.interventions.join(", ")
        );
        if definition.layers.is_some() {
            println!("  layers: {layers} (resolved against the model at run time)");
        } else {
            println!("  layers: as declared (not swept)");
        }
        if let Some(alphas) = &definition.alphas {
            println!("  alphas: {alphas:?}");
        }
        if definition.move_bundle_sources {
            println!("  bundle sources: read at each point's layer (move_bundle_sources)");
        }
        let checked = crate::cli_experiment_steering::check_direction_files(&definition.template)?;
        if checked > 0 {
            println!("  direction files: {checked} (hash and shape checked)");
        }
        if let Some(positions) = &definition.positions {
            println!("  positions: {positions:?}");
        }
        if let Some(effect) = &definition.effect {
            println!(
                "  effect: logit(target) - logit(foil) in capture '{}' for {} input(s); {}% \
                 interval from {} bootstrap resamples (seed {}); tokens resolved at run time",
                effect.capture,
                effect.targets.len(),
                effect.confidence * 100.0,
                effect.resamples,
                effect.seed
            );
        }
        println!("  outputs: one bundle per point plus a baseline, and sweep.json/sweep.csv");
    }
    Ok(())
}

/// Run a sweep spec: one baseline bundle, one bundle per point, the sweep
/// manifest, and a summary.
pub(crate) fn run_sweep(
    command: &RunArgs,
    text: &str,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        command.variants.is_empty(),
        "--variant cannot be combined with a sweep specification"
    );
    let definition = SweepDefinition::parse(text).map_err(|error| anyhow::anyhow!("{error}"))?;
    let execution = command.execution.as_deref();
    let mut template = definition.template.clone();
    apply_overrides(&mut template, execution, command.threads)?;
    let out_dir = command
        .output
        .clone()
        .unwrap_or_else(|| template.output.directory.clone());
    if out_dir.join(SWEEP_MANIFEST_FILE).exists() && !template.output.overwrite {
        anyhow::bail!(
            "sweep output '{}' already exists; refusing to overwrite (set output.overwrite = \
             true to replace it)",
            out_dir.display()
        );
    }

    let load_started = std::time::Instant::now();
    let mut prepared = prepare_run(&template, k_strategy, k_allow_fallback)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let n_layers = prepared.n_layers;
    definition
        .check_moved_bundle_sources(n_layers)
        .map_err(anyhow::Error::msg)?;
    // Resolve the effect tokens before anything runs, so a target that is
    // not one token fails without a wasted sweep.
    let effect = definition
        .effect
        .as_ref()
        .map(|effect| {
            effect.record(|token| {
                crate::cli_experiment_attribution::resolve_token(&prepared, token)
                    .map_err(|error| format!("{error:#}"))
            })
        })
        .transpose()
        .map_err(|error| anyhow::anyhow!("sweep.effect: {error}"))?;
    let mut baseline = definition
        .baseline()
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let mut points = definition
        .points(n_layers)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    for derived in std::iter::once(&mut baseline).chain(points.iter_mut()) {
        apply_overrides(&mut derived.resolved, execution, command.threads)?;
    }
    let directory = |derived: &DerivedSpec| out_dir.join(&derived.relative_dir);
    let baseline_dir = directory(&baseline);
    let point_dirs: Vec<PathBuf> = points.iter().map(directory).collect();
    let targets: Vec<RunTarget<'_>> = points
        .iter()
        .zip(&point_dirs)
        .map(|(point, output_directory)| RunTarget {
            resolved: &point.resolved,
            spec_text: &point.text,
            output_directory,
            retain_incomplete: command.retain_incomplete,
        })
        .collect();
    eprintln!(
        "sweep: {} point(s) on a {}-layer model, baseline computed once",
        points.len(),
        n_layers
    );
    let started = std::time::Instant::now();
    let (base, _, outcomes) = crate::cli_experiment_shared::execute_shared(
        &mut prepared,
        RunTarget {
            resolved: &baseline.resolved,
            spec_text: &baseline.text,
            output_directory: &baseline_dir,
            retain_incomplete: command.retain_incomplete,
        },
        &[],
        &targets,
        None,
    )?;
    let compute_ms = started.elapsed().as_secs_f64() * 1000.0;
    for outcome in std::iter::once(&base).chain(outcomes.iter()) {
        anyhow::ensure!(
            outcome.report.ok,
            "bundle {} failed self-verification",
            outcome.path.display()
        );
    }

    // Metrics from the verified bundles, exactly as `verify` recomputes them.
    let baseline_bundle = load_bundle_for_source(&base.path).map_err(anyhow::Error::msg)?;
    let reference = |outcome: &RunOutcome, relative: &str| SweepBundleRef {
        bundle: relative.to_string(),
        semantic_hash: outcome.identity.semantic_hash.clone(),
        payload_hash: outcome.identity.payload_hash.clone(),
    };
    let mut records = Vec::with_capacity(points.len());
    for (point, outcome) in points.iter().zip(&outcomes) {
        let bundle = load_bundle_for_source(&outcome.path).map_err(anyhow::Error::msg)?;
        let (inputs, point_effect) =
            point_metrics_with_effect(&baseline_bundle, &bundle, effect.as_ref())
                .map_err(anyhow::Error::msg)?;
        records.push(SweepPointRecord {
            id: point.id.clone(),
            layer: point.layer,
            position: point.position,
            alpha: point.alpha,
            bundle: reference(outcome, &point.relative_dir),
            inputs,
            effect: point_effect,
        });
    }
    let mut manifest = SweepManifest {
        schema: SWEEP_SCHEMA_V1.to_string(),
        experiment: definition.name.clone(),
        spec_file: SWEEP_SPEC_FILE.to_string(),
        spec_sha256: definition.spec_sha256.clone(),
        model_sha256: prepared.model_sha.clone(),
        tokenizer_sha256: prepared.tokenizer_sha.clone(),
        layer_count: n_layers,
        layers: definition
            .resolve_layers(n_layers)
            .map_err(|error| anyhow::anyhow!("{error}"))?,
        positions: definition.positions.clone(),
        alphas: definition.alphas.clone(),
        interventions: definition.interventions.clone(),
        effect,
        baseline: reference(&base, &baseline.relative_dir),
        points: records,
        sweep_hash: String::new(),
    };
    manifest.sweep_hash = manifest.compute_hash();

    let runtime = serde_json::json!({
        "timestamp": format!("epoch-seconds-{}", ember::extraction::unix_timestamp()),
        "model_load_ms": load_ms,
        "compute_ms": compute_ms,
        "threads": crate::cli_experiment::pool_threads(&baseline.resolved)?,
        "note": "compute_ms covers the baseline, every point, bundle writing and \
                 self-verification; per-bundle generation time is in each runtime.json",
        "bundles": std::iter::once((&baseline.id, &base))
            .chain(points.iter().map(|point| &point.id).zip(outcomes.iter()))
            .map(|(id, outcome)| serde_json::json!({
                "id": id,
                "prefix_reuse": outcome.prefix.as_ref().map(|record| record.to_json()),
            }))
            .collect::<Vec<_>>(),
    });
    let write = |name: &str, bytes: &[u8]| -> anyhow::Result<()> {
        ember::atomic_file::atomic_write(out_dir.join(name), bytes)
            .with_context(|| format!("cannot write {}", out_dir.join(name).display()))
    };
    write(SWEEP_SPEC_FILE, text.as_bytes())?;
    write(SWEEP_CSV_FILE, manifest.to_csv().as_bytes())?;
    if let Some(table) = manifest.to_effect_csv() {
        write(SWEEP_EFFECT_CSV_FILE, table.as_bytes())?;
    }
    let mut runtime_bytes = serde_json::to_vec_pretty(&runtime)?;
    runtime_bytes.push(b'\n');
    write(SWEEP_RUNTIME_FILE, &runtime_bytes)?;
    let mut signed = Vec::new();
    if let Some(key) = crate::cli_experiment::resolve_sign_key(command) {
        for outcome in std::iter::once(&base).chain(outcomes.iter()) {
            signed.push(crate::cli_experiment::sign_bundle(
                &outcome.path,
                &outcome.identity,
                &key,
            )?);
        }
    }
    // The manifest last: its presence marks a complete sweep.
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    manifest_bytes.push(b'\n');
    write(SWEEP_MANIFEST_FILE, &manifest_bytes)?;

    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "sweep": out_dir.display().to_string(),
                "sweep_hash": manifest.sweep_hash,
                "compute_ms": compute_ms,
                "manifest": manifest,
            }))?
        );
    } else {
        print_table(&manifest);
        println!(
            "sweep written to {} ({} bundle(s), {:.1} s after a {:.1} s model load)",
            out_dir.display(),
            manifest.points.len() + 1,
            compute_ms / 1000.0,
            load_ms / 1000.0
        );
        println!("  sweep hash: {}", manifest.sweep_hash);
        if let Some((_, signer)) = signed.first() {
            println!(
                "  signed evidence: {} bundle(s), <bundle>.evidence.json (signer {signer})",
                signed.len()
            );
        }
        println!("  baseline: {}", describe_prefix_of(&base));
        let resumed = outcomes
            .iter()
            .filter(|outcome| {
                outcome.prefix.as_ref().is_some_and(|record| {
                    record.inputs.iter().all(|input| {
                        matches!(input.path, ember::v05::prefix::PrefixPath::Resumed { .. })
                    })
                })
            })
            .count();
        println!(
            "  prefix reuse: {resumed} of {} point(s) resumed every input from the baseline \
             (details in {SWEEP_RUNTIME_FILE})",
            outcomes.len()
        );
    }
    Ok(())
}

fn describe_prefix_of(outcome: &RunOutcome) -> String {
    outcome
        .prefix
        .as_ref()
        .map(describe_prefix)
        .unwrap_or_else(|| "standalone".into())
}

fn print_table(manifest: &SweepManifest) {
    let over = match (&manifest.alphas, manifest.layers.is_empty()) {
        (Some(alphas), true) => format!("alphas {alphas:?}"),
        (Some(alphas), false) => format!("layers {:?} x alphas {alphas:?}", manifest.layers),
        (None, _) => format!("layers {:?}", manifest.layers),
    };
    println!(
        "sweep '{}': {} point(s), swept {} over {over}",
        manifest.experiment,
        manifest.points.len(),
        manifest.interventions.join(", "),
    );
    println!(
        "  {:<24} {:>5} {:>7} {:<14} {:>14} {:>12}  {:<30} text",
        "point", "layer", "alpha", "input", "first diverges", "peak rel-l2", "at"
    );
    for point in &manifest.points {
        for input in &point.inputs {
            let at = match (&input.peak_capture_id, &input.peak_site, input.peak_layer) {
                (Some(capture), Some(site), Some(layer)) => {
                    format!("{capture} @ {site} L{layer}")
                }
                _ => "-".into(),
            };
            println!(
                "  {:<24} {:>5} {:>7} {:<14} {:>14} {:>12}  {:<30} {}",
                point.id,
                point
                    .layer
                    .map(|layer| layer.to_string())
                    .unwrap_or_else(|| "-".into()),
                point
                    .alpha
                    .map(|alpha| alpha.to_string())
                    .unwrap_or_else(|| "-".into()),
                input.input_id,
                input
                    .first_divergent_step
                    .map(|step| format!("step {step}"))
                    .unwrap_or_else(|| "never".into()),
                input
                    .peak_relative_l2
                    .map(|value| format!("{value:.4e}"))
                    .unwrap_or_else(|| "-".into()),
                at,
                if input.generated_text_equal {
                    "same"
                } else {
                    "changed"
                }
            );
        }
    }
    let Some(effect) = &manifest.effect else {
        return;
    };
    let percent = effect.confidence * 100.0;
    println!(
        "effect: logit(target) - logit(foil) in capture '{}', point minus baseline, over {} \
         input(s); {percent}% bootstrap interval ({} resamples, seed {})",
        effect.capture,
        effect.targets.len(),
        effect.resamples,
        effect.seed
    );
    // The baseline margin gives the scale of the effects.
    let baseline: Vec<f64> = manifest
        .points
        .first()
        .map(|point| {
            point
                .inputs
                .iter()
                .filter_map(|input| input.baseline_metric)
                .collect()
        })
        .unwrap_or_default();
    if !baseline.is_empty() {
        println!(
            "  baseline metric: mean {:.4} over {} input(s)",
            baseline.iter().sum::<f64>() / baseline.len() as f64,
            baseline.len()
        );
    }
    println!(
        "  {:<24} {:>5} {:>7} {:>12} {:>27} {:>9} {:>10}",
        "point", "layer", "alpha", "mean", "interval", "+/-/0", "sign p"
    );
    let number = |value: Option<f64>| {
        value
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "-".into())
    };
    for point in &manifest.points {
        let Some(summary) = &point.effect else {
            continue;
        };
        let interval = match (summary.ci_low, summary.ci_high) {
            (Some(low), Some(high)) => format!("[{low:.4}, {high:.4}]"),
            _ => "-".into(),
        };
        println!(
            "  {:<24} {:>5} {:>7} {:>12} {:>27} {:>9} {:>10}",
            point.id,
            point
                .layer
                .map(|layer| layer.to_string())
                .unwrap_or_else(|| "-".into()),
            point
                .alpha
                .map(|alpha| alpha.to_string())
                .unwrap_or_else(|| "-".into()),
            number(Some(summary.mean)),
            interval,
            format!("{}/{}/{}", summary.positive, summary.negative, summary.zero),
            summary
                .sign_test_p
                .map(|p| if p >= 1e-3 {
                    format!("{p:.4}")
                } else {
                    format!("{p:.2e}")
                })
                .unwrap_or_else(|| "-".into()),
        );
    }
}

fn print_checks(report: &SweepVerification) {
    for check in &report.checks {
        println!(
            "  [{}] {}: {}",
            if check.ok { "ok" } else { "FAIL" },
            check.name,
            check.detail
        );
    }
}

/// `experiment verify <sweep-dir>`.
pub(crate) fn run_verify_sweep(
    dir: &Path,
    options: &VerifyOptions,
    anchored_by_evidence: bool,
    write_report: Option<&Path>,
    json: bool,
) -> anyhow::Result<()> {
    // Refuse rather than ignore a signature anchor: `--trusted-key` alone
    // would otherwise report a sweep "verified" without checking any
    // signature.
    anyhow::ensure!(
        !anchored_by_evidence,
        "--trusted-key/--expect-evidence apply to single bundles; anchor a sweep \
         with --expect-semantic-hash <sweep hash>"
    );
    let report = verify_sweep(dir, options).map_err(anyhow::Error::msg)?;
    if let Some(path) = write_report {
        crate::cli_experiment::ensure_outside_bundle(
            dir,
            path,
            "--write-report must point outside the sweep; verification never modifies it",
        )?;
        let mut bytes = serde_json::to_vec_pretty(&report)?;
        bytes.push(b'\n');
        ember::atomic_file::atomic_write(path, &bytes)
            .with_context(|| format!("cannot write report '{}'", path.display()))?;
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("verification of sweep {}", dir.display());
        print_checks(&report);
        println!(
            "verdict: {} ({} bundle(s))",
            if report.ok { "verified" } else { "FAILED" },
            report.bundles
        );
        if report.ok {
            println!("  sweep hash: {}", report.sweep_hash);
            if options.expected_semantic_hash.is_none() {
                println!(
                    "  note: 'verified' means self-consistent only; pass \
                     --expect-semantic-hash <sweep hash> to bind it to a recorded identity."
                );
            }
        }
    }
    if !report.ok {
        return Err(crate::cli_support::VerificationFailed.into());
    }
    Ok(())
}

/// `experiment compare <sweep-a> <sweep-b>`.
pub(crate) fn run_compare_sweeps(
    a: &Path,
    b: &Path,
    expect_a: Option<&str>,
    expect_b: Option<&str>,
    json: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        ember::v05::sweep::is_sweep_dir(a) && ember::v05::sweep::is_sweep_dir(b),
        "compare a sweep only with another sweep (one of '{}' and '{}' is a single bundle)",
        a.display(),
        b.display()
    );
    let result = ember::v05::sweep::compare_anchored_sweeps(a, b, expect_a, expect_b)
        .map_err(anyhow::Error::msg)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    let yes = |value: bool| if value { "yes" } else { "no" };
    println!("comparing sweep {} vs {}", a.display(), b.display());
    println!("  sweep hash equal: {}", yes(result.sweep_hash_equal));
    println!("  sweep spec equal: {}", yes(result.spec_equal));
    for point in &result.points {
        println!(
            "  {:<18} in a {:<3} in b {:<3} semantic {:<3} payload {:<3} metrics {}",
            point.id,
            yes(point.present_in_a),
            yes(point.present_in_b),
            yes(point.semantic_hash_equal),
            yes(point.payload_hash_equal),
            yes(point.metrics_equal)
        );
    }
    println!("verdict: {}", result.verdict);
    Ok(())
}

/// `experiment inspect <sweep-dir>`.
pub(crate) fn run_inspect_sweep(dir: &Path, json: bool) -> anyhow::Result<()> {
    let report = verify_sweep(dir, &VerifyOptions::default()).map_err(anyhow::Error::msg)?;
    if !report.ok {
        print_checks(&report);
        return Err(crate::cli_support::VerificationFailed.into());
    }
    let manifest = ember::v05::sweep::read_manifest(dir).map_err(anyhow::Error::msg)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
    } else {
        println!("experiment sweep: {}", dir.display());
        println!("  schema: {}", manifest.schema);
        println!(
            "  model: {}",
            manifest
                .model_sha256
                .get(..12)
                .unwrap_or(&manifest.model_sha256)
        );
        println!("  baseline: {}", manifest.baseline.bundle);
        print_table(&manifest);
        println!("  sweep hash: {}", manifest.sweep_hash);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_experiment::execute_prepared;
    use crate::experiment_testutil::{run_args, spec_text, tiny_model};

    const BODY: &str = r#"
[[inputs]]
id = "a"
text = "w3 w17 w5 w40 w9"

[[inputs]]
id = "b"
text = "w8 w1 w33"

[[captures]]
id = "rows"
site = "residual-post-mlp"
layers = "all"
[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "decode"
site = "mlp-output"
layers = [1]
[captures.tokens]
kind = "generated-step"
step = 2

[[interventions]]
id = "scale"
site = "residual-post-mlp"
layers = [0]
operation = { kind = "scale", factor = -3.0 }
[interventions.tokens]
kind = "prompt-final"
"#;

    #[test]
    fn sweep_points_equal_standalone_runs_and_verify() {
        let model = tiny_model("sweep-e2e", 4, 64, true);
        let text = format!(
            "{}\n[sweep]\nlayers = \"all\"\n",
            spec_text(&model, "reference", 3, BODY)
        );
        let spec = model.dir.join("sweep.toml");
        std::fs::write(&spec, &text).unwrap();
        let out = model.dir.join("sweep-out");
        run_sweep(
            &run_args(&spec, Some(out.clone())),
            &text,
            KStrategy::Auto,
            false,
        )
        .unwrap();

        let report = verify_sweep(&out, &VerifyOptions::default()).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        assert_eq!(report.bundles, 5);
        let manifest = ember::v05::sweep::read_manifest(&out).unwrap();
        assert_eq!(manifest.layers, vec![0, 1, 2, 3]);

        // Every point bundle is the bundle its derived spec gives alone.
        let definition = SweepDefinition::parse(&text).unwrap();
        let mut prepared = prepare_run(&definition.template, KStrategy::Auto, false).unwrap();
        let derived = definition.points(4).unwrap();
        for (point, record) in derived.iter().zip(&manifest.points) {
            let (_, identity, _, _) = execute_prepared(
                &mut prepared,
                &point.resolved,
                &point.text,
                &model.dir.join(format!("alone-{}", point.id)),
                false,
                None,
            )
            .unwrap();
            assert_eq!(
                identity.semantic_hash, record.bundle.semantic_hash,
                "{}",
                point.id
            );
            assert_eq!(
                identity.payload_hash, record.bundle.payload_hash,
                "{}",
                point.id
            );
        }
        // Prefix reuse ran for every point but layer 0.
        let runtime: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join(SWEEP_RUNTIME_FILE)).unwrap()).unwrap();
        let paths: Vec<String> = runtime["bundles"]
            .as_array()
            .unwrap()
            .iter()
            .skip(1)
            .map(|bundle| {
                bundle["prefix_reuse"]["inputs"][0]["path"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(paths, ["full-recompute", "resumed", "resumed", "resumed"]);
        // The intervention does something somewhere.
        assert!(manifest.points.iter().any(|point| point
            .inputs
            .iter()
            .any(|input| input.peak_relative_l2.unwrap_or(0.0) > 0.0)));

        // A second run of the same spec compares exact.
        let again = model.dir.join("sweep-again");
        run_sweep(
            &run_args(&spec, Some(again.clone())),
            &text,
            KStrategy::Auto,
            false,
        )
        .unwrap();
        let comparison = ember::v05::sweep::compare_sweeps(&out, &again).unwrap();
        // An anchor is checked in the same verification pass.
        let error =
            ember::v05::sweep::compare_anchored_sweeps(&out, &again, Some(&"0".repeat(64)), None)
                .unwrap_err();
        assert!(error.contains("sweep hash anchor"), "{error}");
        assert_eq!(comparison.verdict, "exact");
        assert!(comparison.sweep_hash_equal);

        // Refuses to overwrite a finished sweep.
        assert!(run_sweep(
            &run_args(&spec, Some(out.clone())),
            &text,
            KStrategy::Auto,
            false
        )
        .is_err());

        // Tampering is caught: the csv, a metric (hash resealed), and one
        // point's bundle swapped for another's (hash resealed).
        std::fs::write(again.join(SWEEP_CSV_FILE), "point\n").unwrap();
        assert!(!verify_sweep(&again, &VerifyOptions::default()).unwrap().ok);
        let reseal = |manifest: &SweepManifest| {
            std::fs::write(out.join(SWEEP_CSV_FILE), manifest.to_csv()).unwrap();
            std::fs::write(
                out.join(SWEEP_MANIFEST_FILE),
                serde_json::to_vec(manifest).unwrap(),
            )
            .unwrap();
            verify_sweep(&out, &VerifyOptions::default()).unwrap()
        };
        let mut forged = manifest.clone();
        forged.points[2].inputs[0].peak_relative_l2 = Some(0.5);
        forged.sweep_hash = forged.compute_hash();
        let report = reseal(&forged);
        assert!(!report.ok);
        assert!(report
            .checks
            .iter()
            .any(|check| !check.ok && check.name == "point layer-02 metrics"));
        // A forged layer count fails a check before it sizes anything.
        let mut huge = manifest.clone();
        huge.layer_count = usize::MAX / 2;
        huge.sweep_hash = huge.compute_hash();
        let report = reseal(&huge);
        assert!(report
            .checks
            .iter()
            .any(|check| !check.ok && check.name == "layer count"));
        let mut swapped = manifest.clone();
        swapped.points[1].bundle = swapped.points[2].bundle.clone();
        swapped.sweep_hash = swapped.compute_hash();
        let report = reseal(&swapped);
        assert!(report
            .checks
            .iter()
            .any(|check| !check.ok && check.name == "point layer-01 derivation"));
        assert!(reseal(&manifest).ok);
    }

    const EFFECT_BODY: &str = r#"
[[inputs]]
id = "a"
text = "w3 w17 w5 w40 w9"

[[inputs]]
id = "b"
text = "w8 w1 w33"

[[inputs]]
id = "c"
text = "w2 w2 w60 w7"

[[captures]]
id = "answer"
site = "logits"
[captures.tokens]
kind = "prompt-final"

[[interventions]]
id = "scale"
site = "residual-post-mlp"
layers = [0]
operation = { kind = "scale", factor = -3.0 }
[interventions.tokens]
kind = "prompt-final"
"#;

    const EFFECT_TABLE: &str = r#"
[sweep]
layers = "all"

[sweep.effect]
capture = "answer"
resamples = 500
seed = 3

[[sweep.effect.targets]]
input = "a"
target = 10
foil = 11

[[sweep.effect.targets]]
input = "c"
target = "w20"
foil = 21

[[sweep.effect.targets]]
input = "b"
target = "w12"
foil = "w13"
"#;

    #[test]
    fn effect_statistics_cover_every_input_and_verify() {
        let model = tiny_model("sweep-effect", 3, 64, false);
        let text = format!(
            "{}{EFFECT_TABLE}",
            spec_text(&model, "reference", 2, EFFECT_BODY)
        );
        let spec = model.dir.join("sweep.toml");
        std::fs::write(&spec, &text).unwrap();
        let out = model.dir.join("sweep-out");
        run_sweep(
            &run_args(&spec, Some(out.clone())),
            &text,
            KStrategy::Auto,
            false,
        )
        .unwrap();

        let report = verify_sweep(&out, &VerifyOptions::default()).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        let effect_check = |report: &SweepVerification| {
            report
                .checks
                .iter()
                .find(|check| check.name == "sweep effect")
                .cloned()
                .unwrap()
        };
        // Two token texts ("w20", and "w12"/"w13" count as two) need a
        // tokenizer to re-encode.
        assert!(effect_check(&report)
            .detail
            .contains("3 token text(s) not re-encoded"));
        let deep = VerifyOptions {
            tokenizer_path: Some(model.tokenizer.clone()),
            ..VerifyOptions::default()
        };
        let report = verify_sweep(&out, &deep).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        assert!(!effect_check(&report).detail.contains("not re-encoded"));
        // Another tokenizer file is refused, not used to re-encode.
        let other = model.dir.join("other-tokenizer.json");
        let mut bytes = std::fs::read(&model.tokenizer).unwrap();
        bytes.push(b'\n');
        std::fs::write(&other, bytes).unwrap();
        let wrong = VerifyOptions {
            tokenizer_path: Some(other),
            ..VerifyOptions::default()
        };
        let check = effect_check(&verify_sweep(&out, &wrong).unwrap());
        assert!(!check.ok && check.detail.contains("hashes to"), "{check:?}");

        let manifest = ember::v05::sweep::read_manifest(&out).unwrap();
        let effect = manifest.effect.as_ref().unwrap();
        // Targets follow input order, whatever order the spec lists them in.
        let resolved: Vec<(&str, u32, u32)> = effect
            .targets
            .iter()
            .map(|t| (t.input_id.as_str(), t.target.token_id, t.foil.token_id))
            .collect();
        assert_eq!(resolved, [("a", 10, 11), ("b", 12, 13), ("c", 20, 21)]);
        assert_eq!((effect.resamples, effect.seed), (500, 3));
        for point in &manifest.points {
            let summary = point.effect.as_ref().unwrap();
            assert_eq!(summary.n, 3);
            assert_eq!(summary.positive + summary.negative + summary.zero, 3);
            let effects: Vec<f64> = point.inputs.iter().map(|i| i.effect.unwrap()).collect();
            assert_eq!(
                *summary,
                ember::v05::effect::summarize(&effects, 0.95, 500, 3),
                "the summary is a function of the recorded per-input effects"
            );
            for input in &point.inputs {
                let (before, after) = (input.baseline_metric.unwrap(), input.point_metric.unwrap());
                assert!((after - before - input.effect.unwrap()).abs() < 1e-6);
            }
        }
        // Scaling the block output by -3 moves the logits somewhere.
        assert!(manifest
            .points
            .iter()
            .any(|point| point.effect.as_ref().unwrap().mean != 0.0));
        let table = std::fs::read_to_string(out.join(SWEEP_EFFECT_CSV_FILE)).unwrap();
        assert_eq!(table.lines().count(), 1 + manifest.points.len());
        assert!(std::fs::read_to_string(out.join(SWEEP_CSV_FILE))
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .ends_with(",baseline_metric,point_metric,effect"));

        // Tampering is caught.
        let reseal = |manifest: &SweepManifest| {
            std::fs::write(out.join(SWEEP_CSV_FILE), manifest.to_csv()).unwrap();
            if let Some(table) = manifest.to_effect_csv() {
                std::fs::write(out.join(SWEEP_EFFECT_CSV_FILE), table).unwrap();
            }
            std::fs::write(
                out.join(SWEEP_MANIFEST_FILE),
                serde_json::to_vec(manifest).unwrap(),
            )
            .unwrap();
            verify_sweep(&out, &VerifyOptions::default()).unwrap()
        };
        let failed = |report: &SweepVerification, name: &str| {
            report.checks.iter().any(|c| !c.ok && c.name == name)
        };
        let mut forged = manifest.clone();
        forged.points[1].effect.as_mut().unwrap().mean += 1.0;
        forged.sweep_hash = forged.compute_hash();
        assert!(failed(&reseal(&forged), "point layer-01 metrics"));
        // A token id the spec names cannot be changed.
        let mut forged = manifest.clone();
        forged.effect.as_mut().unwrap().targets[0].foil.token_id = 12;
        forged.sweep_hash = forged.compute_hash();
        assert!(failed(&reseal(&forged), "sweep effect"));
        // Without the effect record the spec and sweep.json disagree.
        let mut forged = manifest.clone();
        forged.effect = None;
        forged.sweep_hash = forged.compute_hash();
        assert!(failed(&reseal(&forged), "sweep effect"));
        // Forged interval settings fail the checks; they never size an
        // allocation or index the bootstrap.
        for (confidence, resamples) in [(0.95, 0), (1.5, 500), (0.95, 1usize << 40)] {
            let mut forged = manifest.clone();
            let effect = forged.effect.as_mut().unwrap();
            effect.confidence = confidence;
            effect.resamples = resamples;
            forged.sweep_hash = forged.compute_hash();
            let report = reseal(&forged);
            assert!(failed(&report, "sweep effect"), "{confidence} {resamples}");
            assert!(failed(&report, "point layer-00 metrics"));
        }
        assert!(reseal(&manifest).ok);
        std::fs::write(out.join(SWEEP_EFFECT_CSV_FILE), "point\n").unwrap();
        assert!(failed(
            &verify_sweep(&out, &VerifyOptions::default()).unwrap(),
            "sweep effect csv"
        ));
    }

    #[test]
    fn patching_sweeps_read_the_source_at_each_point_layer() {
        let model = tiny_model("sweep-patching", 3, 64, false);
        // The clean run: the residual entering every block at the final
        // token of each input.
        let clean_body = r#"
[[inputs]]
id = "a"
text = "w3 w17 w5 w40 w9"

[[inputs]]
id = "b"
text = "w8 w1 w33"

[[captures]]
id = "rows"
site = "residual-pre-attention"
layers = "all"
[captures.tokens]
kind = "prompt-final"
"#;
        let clean_dir = model.dir.join("clean");
        let clean_text = spec_text(&model, "reference", 0, clean_body).replace(
            "directory = \"unused\"",
            &format!("directory = {clean_dir:?}"),
        );
        let mut prepared = prepare_run(
            &crate::experiment_testutil::resolve(&clean_text),
            KStrategy::Auto,
            false,
        )
        .unwrap();
        execute_prepared(
            &mut prepared,
            &crate::experiment_testutil::resolve(&clean_text),
            &clean_text,
            &clean_dir,
            false,
            None,
        )
        .unwrap();

        // The corrupted inputs differ before the final token, which they
        // share with the clean inputs.
        let source = |input: &str| {
            format!(
                "source = {{ kind = \"capture-from-bundle\", bundle_path = {clean_dir:?}, \
                 capture_id = \"rows\", input_id = \"{input}\", layer = 0 }}"
            )
        };
        let body = format!(
            r#"
[[inputs]]
id = "a"
text = "w3 w18 w6 w41 w9"

[[inputs]]
id = "b"
text = "w7 w2 w33"

[[captures]]
id = "answer"
site = "logits"
[captures.tokens]
kind = "prompt-final"

[[interventions]]
id = "patch-a"
site = "residual-pre-attention"
layers = [0]
inputs = ["a"]
operation = {{ kind = "replace" }}
{}
[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "patch-b"
site = "residual-pre-attention"
layers = [0]
inputs = ["b"]
operation = {{ kind = "replace" }}
{}
[interventions.tokens]
kind = "prompt-final"
"#,
            source("a"),
            source("b")
        );
        let table = r#"
[sweep]
layers = "all"
move_bundle_sources = true

[sweep.effect]
capture = "answer"
resamples = 200

[[sweep.effect.targets]]
input = "a"
target = 10
foil = 11

[[sweep.effect.targets]]
input = "b"
target = 12
foil = 13
"#;
        let text = format!("{}{table}", spec_text(&model, "reference", 0, &body));
        let spec = model.dir.join("sweep.toml");
        std::fs::write(&spec, &text).unwrap();
        let out = model.dir.join("sweep-out");
        run_sweep(
            &run_args(&spec, Some(out.clone())),
            &text,
            KStrategy::Auto,
            false,
        )
        .unwrap();
        let report = verify_sweep(&out, &VerifyOptions::default()).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        let manifest = ember::v05::sweep::read_manifest(&out).unwrap();
        // Layer 0 reads the embedding of the shared final token: a no-op.
        let first = &manifest.points[0];
        assert_eq!(first.id, "layer-00");
        assert!(first.inputs.iter().all(|input| input.effect == Some(0.0)));
        // A later layer carries the clean prefix into the final token.
        assert!(manifest.points[1..]
            .iter()
            .any(|point| point.inputs.iter().any(|input| input.effect != Some(0.0))));
        // Each point's bundle names its own source layer.
        for point in &manifest.points {
            let toml =
                std::fs::read_to_string(out.join(&point.bundle.bundle).join("experiment.toml"))
                    .unwrap();
            let layer = point.layer.unwrap();
            assert_eq!(
                toml.matches(&format!("layer = {layer}")).count(),
                2,
                "{}: {toml}",
                point.id
            );
        }

        // A source bundle without one of the swept layers fails before
        // anything runs.
        let narrow_dir = model.dir.join("clean-narrow");
        let narrow_text = clean_text
            .replace("layers = \"all\"", "layers = [0, 1]")
            .replace(&format!("{clean_dir:?}"), &format!("{narrow_dir:?}"));
        execute_prepared(
            &mut prepared,
            &crate::experiment_testutil::resolve(&narrow_text),
            &narrow_text,
            &narrow_dir,
            false,
            None,
        )
        .unwrap();
        let text = text.replace(&format!("{clean_dir:?}"), &format!("{narrow_dir:?}"));
        let out = model.dir.join("sweep-narrow");
        let error = run_sweep(
            &run_args(&spec, Some(out.clone())),
            &text,
            KStrategy::Auto,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("at residual-pre-attention layer 2"),
            "{error}"
        );
        assert!(!out.exists());
    }

    #[test]
    fn effect_targets_must_be_single_tokens_before_anything_runs() {
        let model = tiny_model("sweep-effect-token", 2, 64, false);
        let text = format!(
            "{}{}",
            spec_text(&model, "reference", 0, EFFECT_BODY),
            EFFECT_TABLE.replace("target = \"w12\"", "target = \"w12 w14\"")
        );
        let spec = model.dir.join("sweep.toml");
        std::fs::write(&spec, &text).unwrap();
        let out = model.dir.join("sweep-out");
        let error = run_sweep(
            &run_args(&spec, Some(out.clone())),
            &text,
            KStrategy::Auto,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("encodes to 2 tokens"), "{error}");
        assert!(!out.exists(), "nothing ran");
    }

    #[test]
    fn position_sweeps_run_every_layer_position_pair() {
        let model = tiny_model("sweep-positions", 4, 64, false);
        let text = format!(
            "{}\n[sweep]\nlayers = [1, 3]\npositions = [0, 2]\n",
            spec_text(&model, "planned", 2, BODY)
        );
        let spec = model.dir.join("sweep.toml");
        std::fs::write(&spec, &text).unwrap();
        let out = model.dir.join("sweep-out");
        run_sweep(
            &run_args(&spec, Some(out.clone())),
            &text,
            KStrategy::Auto,
            false,
        )
        .unwrap();
        let report = verify_sweep(&out, &VerifyOptions::default()).unwrap();
        assert!(report.ok, "{:?}", report.checks);
        let manifest = ember::v05::sweep::read_manifest(&out).unwrap();
        let ids: Vec<&str> = manifest
            .points
            .iter()
            .map(|point| point.id.as_str())
            .collect();
        assert_eq!(
            ids,
            [
                "layer-01-pos-0",
                "layer-01-pos-2",
                "layer-03-pos-0",
                "layer-03-pos-2"
            ]
        );
    }
}
