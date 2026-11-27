use super::*;

/// Result of verifying a sweep directory.
#[derive(Debug, Clone, Serialize)]
pub struct SweepVerification {
    pub schema: String,
    pub ok: bool,
    pub sweep_hash: String,
    pub bundles: usize,
    pub checks: Vec<CheckResult>,
}

impl SweepVerification {
    fn check(&mut self, name: impl Into<String>, ok: bool, detail: impl Into<String>) {
        self.ok &= ok;
        self.checks.push(CheckResult {
            name: name.into(),
            ok,
            detail: detail.into(),
        });
    }
}

/// Whether `dir` holds a sweep manifest.
pub fn is_sweep_dir(dir: &Path) -> bool {
    dir.join(SWEEP_MANIFEST_FILE).is_file()
}

/// Read a sweep-level file, refusing symlinks and special files like the
/// bundle verifier does: what is verified must be the directory's own file.
fn read_sweep_file(path: &Path) -> std::io::Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a regular file",
        ));
    }
    std::fs::read(path)
}

/// Read and parse `sweep.json`.
pub fn read_manifest(dir: &Path) -> Result<SweepManifest, String> {
    let bytes = read_sweep_file(&dir.join(SWEEP_MANIFEST_FILE))
        .map_err(|error| format!("cannot read {SWEEP_MANIFEST_FILE}: {error}"))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("malformed {SWEEP_MANIFEST_FILE}: {error}"))
}

/// Verify a sweep directory: the manifest and its hash, the sweep spec, every
/// bundle (fully, with `options` for deep model/tokenizer checks), that each
/// point bundle is exactly the experiment the sweep spec derives for it, and
/// that the recorded metrics are what the bundles give.
pub fn verify_sweep(dir: &Path, options: &VerifyOptions) -> Result<SweepVerification, String> {
    verify_and_load_sweep(dir, options).map(|(report, _)| report)
}

/// [`verify_sweep`], also returning the manifest that was verified so
/// callers never re-read `sweep.json` after verification.
fn verify_and_load_sweep(
    dir: &Path,
    options: &VerifyOptions,
) -> Result<(SweepVerification, SweepManifest), String> {
    let manifest = read_manifest(dir)?;
    let mut report = SweepVerification {
        schema: SWEEP_SCHEMA_V1.to_string(),
        ok: true,
        sweep_hash: manifest.sweep_hash.clone(),
        bundles: 0,
        checks: Vec::new(),
    };
    report.check(
        "sweep schema",
        manifest.schema == SWEEP_SCHEMA_V1,
        format!("{} (expected {SWEEP_SCHEMA_V1})", manifest.schema),
    );
    let recomputed = manifest.compute_hash();
    report.check(
        "sweep hash",
        recomputed == manifest.sweep_hash,
        format!("recomputed {recomputed}"),
    );
    if let Some(expected) = &options.expected_semantic_hash {
        report.check(
            "sweep hash anchor",
            expected == &manifest.sweep_hash,
            format!("expected {expected}"),
        );
    }

    // The sweep spec and what it derives.
    let spec_path = inside(dir, &manifest.spec_file)?;
    let spec_text = read_sweep_file(&spec_path)
        .and_then(|bytes| {
            String::from_utf8(bytes)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
        })
        .map_err(|error| format!("cannot read {}: {error}", manifest.spec_file))?;
    let spec_sha = sha256_hex(spec_text.as_bytes());
    report.check(
        "sweep spec hash",
        spec_sha == manifest.spec_sha256,
        format!("{} hashes to {spec_sha}", manifest.spec_file),
    );
    let definition = SweepDefinition::parse(&spec_text).map_err(|error| error.to_string())?;
    let baseline_spec = definition.baseline().map_err(|error| error.to_string())?;
    // `layer_count` sizes the derived point list, so bound it before use. No
    // model the loader accepts has more layers.
    if manifest.layer_count > crate::loader::limits::MAX_LAYERS {
        report.check(
            "layer count",
            false,
            format!(
                "{} layers exceeds the {}-layer model limit",
                manifest.layer_count,
                crate::loader::limits::MAX_LAYERS
            ),
        );
        return Ok((report, manifest));
    }
    // `head_count` sizes the head list in the same way; a head sweep must
    // record it, and no other sweep may.
    let head_count = match (definition.heads.is_some(), manifest.head_count) {
        (true, Some(count)) if count <= crate::loader::limits::MAX_HEADS => count,
        (false, None) => 0,
        (_, count) => {
            report.check(
                "head count",
                false,
                format!(
                    "the spec {} heads but {SWEEP_MANIFEST_FILE} records head count {count:?} \
                     (at most {})",
                    if definition.heads.is_some() {
                        "sweeps"
                    } else {
                        "does not sweep"
                    },
                    crate::loader::limits::MAX_HEADS
                ),
            );
            return Ok((report, manifest));
        }
    };
    // Bound the derivation by what the manifest records: forged axes could
    // otherwise derive a huge number of specs before any bundle is read.
    let expected_points = definition
        .point_count(manifest.layer_count, head_count)
        .map_err(|error| error.to_string())?;
    if expected_points != Some(manifest.points.len()) {
        report.check(
            "sweep points",
            false,
            format!(
                "{} recorded, {expected_points:?} derived from the spec",
                manifest.points.len()
            ),
        );
        return Ok((report, manifest));
    }
    let derived = definition
        .points(manifest.layer_count, head_count)
        .map_err(|error| error.to_string())?;
    let layers = definition
        .resolve_layers(manifest.layer_count)
        .map_err(|error| error.to_string())?;
    let heads = definition
        .heads
        .is_some()
        .then(|| definition.resolve_heads(head_count))
        .transpose()
        .map_err(|error| error.to_string())?;
    report.check(
        "sweep definition",
        definition.name == manifest.experiment
            && layers == manifest.layers
            && heads == manifest.heads
            && definition.features == manifest.features
            && definition.positions == manifest.positions
            && definition.alphas == manifest.alphas
            && definition.interventions == manifest.interventions,
        "experiment name, layers, heads, features, positions, alphas and swept interventions \
         match the spec",
    );
    match (&definition.effect, &manifest.effect) {
        (None, None) => {}
        (Some(declared), Some(recorded)) => {
            // Re-encode only with the tokenizer the sweep ran with.
            let tokenizer = options
                .tokenizer_path
                .as_ref()
                .map(|path| {
                    use crate::tokenizer::EmberTokenizer;
                    let bytes = EmberTokenizer::read_file(path)
                        .map_err(|error| format!("cannot read tokenizer: {error:#}"))?;
                    let sha = sha256_hex(&bytes);
                    if sha != manifest.tokenizer_sha256 {
                        return Err(format!(
                            "tokenizer '{}' hashes to {sha}, not the sweep's {}",
                            path.display(),
                            manifest.tokenizer_sha256
                        ));
                    }
                    EmberTokenizer::from_bytes(&bytes)
                        .map_err(|error| format!("cannot load tokenizer: {error:#}"))
                })
                .transpose();
            match tokenizer {
                Err(error) => report.check("sweep effect", false, error),
                Ok(tokenizer) => {
                    let (ok, unchecked) = declared.matches(recorded, |text| {
                        tokenizer.as_ref().map(|tokenizer| {
                            let ids = tokenizer
                                .encode_no_special(text)
                                .map_err(|error| error.to_string())?;
                            match ids[..] {
                                [id] => Ok(id),
                                _ => Err(format!("token text {text:?} is {} tokens", ids.len())),
                            }
                        })
                    });
                    let detail = if unchecked == 0 {
                        "capture, interval settings and target/foil tokens match the spec"
                            .to_string()
                    } else {
                        format!(
                            "capture, interval settings and token ids match the spec; \
                             {unchecked} token text(s) not re-encoded (pass --tokenizer to \
                             check them)"
                        )
                    };
                    report.check("sweep effect", ok, detail);
                }
            }
        }
        (declared, _) => report.check(
            "sweep effect",
            false,
            if declared.is_some() {
                format!(
                    "the spec has [sweep.effect] but {SWEEP_MANIFEST_FILE} has no effect record"
                )
            } else {
                format!(
                    "the spec has no [sweep.effect] but {SWEEP_MANIFEST_FILE} has an effect record"
                )
            },
        ),
    }
    let derived_ids: Vec<&str> = derived.iter().map(|point| point.id.as_str()).collect();
    let recorded_ids: Vec<&str> = manifest
        .points
        .iter()
        .map(|point| point.id.as_str())
        .collect();
    report.check(
        "sweep points",
        derived_ids == recorded_ids,
        format!(
            "{} recorded, {} derived from the spec",
            recorded_ids.len(),
            derived_ids.len()
        ),
    );

    // Bundles.
    let load = |reference: &SweepBundleRef| -> Result<LoadedBundle, String> {
        let path = inside(dir, &reference.bundle)?;
        let bundle_options = VerifyOptions {
            expected_semantic_hash: Some(reference.semantic_hash.clone()),
            ..options.clone()
        };
        load_verified_bundle(&path, &bundle_options)
    };
    let baseline = match load(&manifest.baseline) {
        Ok(bundle) => bundle,
        Err(error) => {
            report.check("baseline bundle", false, error);
            return Ok((report, manifest));
        }
    };
    report.bundles += 1;
    let bundle_checks = |report: &mut SweepVerification,
                         label: &str,
                         bundle: &LoadedBundle,
                         reference: &SweepBundleRef,
                         expected_text: &str| {
        report.check(
            format!("{label} bundle"),
            bundle.payload_hash == reference.payload_hash
                && bundle.semantic_manifest.model.sha256 == manifest.model_sha256
                && bundle.semantic_manifest.tokenizer.sha256 == manifest.tokenizer_sha256,
            format!(
                "verified; semantic {} payload {}",
                bundle.semantic_hash, bundle.payload_hash
            ),
        );
        let text = bundle
            .file("experiment.toml")
            .and_then(|bytes| std::str::from_utf8(bytes).ok());
        report.check(
            format!("{label} derivation"),
            text == Some(expected_text),
            "experiment.toml is the spec the sweep derives for it",
        );
    };
    bundle_checks(
        &mut report,
        "baseline",
        &baseline,
        &manifest.baseline,
        &baseline_spec.text,
    );
    report.check(
        "layer count",
        baseline.semantic_manifest.model.layer_count == manifest.layer_count,
        format!(
            "the model has {} layers",
            baseline.semantic_manifest.model.layer_count
        ),
    );
    // `heads = "all"` resolved against the model's own head count, which
    // the baseline bundle's verified model metadata records.
    if let Some(count) = manifest.head_count {
        let model_heads = model_head_count(&baseline);
        report.check(
            "head count",
            model_heads == Some(count),
            format!("the model has {model_heads:?} attention heads"),
        );
    }
    for record in &manifest.points {
        let label = format!("point {}", record.id);
        let bundle = match load(&record.bundle) {
            Ok(bundle) => bundle,
            Err(error) => {
                report.check(format!("{label} bundle"), false, error);
                continue;
            }
        };
        report.bundles += 1;
        let expected = derived
            .iter()
            .find(|point| point.id == record.id)
            .map(|point| {
                (
                    point.text.as_str(),
                    point.layer,
                    point.head,
                    point.feature,
                    point.position,
                    point.alpha,
                )
            });
        match expected {
            Some((text, layer, head, feature, position, alpha)) => {
                bundle_checks(&mut report, &label, &bundle, &record.bundle, text);
                report.check(
                    format!("{label} coordinates"),
                    layer == record.layer
                        && head == record.head
                        && feature == record.feature
                        && position == record.position
                        && alpha.map(f64::to_bits) == record.alpha.map(f64::to_bits),
                    format!(
                        "layer {:?} head {:?} feature {:?} position {:?} alpha {:?}",
                        record.layer, record.head, record.feature, record.position, record.alpha
                    ),
                );
            }
            None => report.check(
                format!("{label} derivation"),
                false,
                "the sweep spec does not derive this point",
            ),
        }
        match point_metrics_with_effect(&baseline, &bundle, manifest.effect.as_ref()) {
            Ok((metrics, effect)) => report.check(
                format!("{label} metrics"),
                metrics == record.inputs && effect == record.effect,
                "recorded metrics equal the metrics recomputed from the bundles",
            ),
            Err(error) => report.check(format!("{label} metrics"), false, error),
        }
    }
    // The CSV is a view of the manifest.
    match read_sweep_file(&dir.join(SWEEP_CSV_FILE)) {
        Ok(csv) => report.check(
            "sweep csv",
            csv == manifest.to_csv().as_bytes(),
            "sweep.csv is the manifest's table",
        ),
        Err(error) => report.check("sweep csv", false, error.to_string()),
    }
    if let Some(table) = manifest.to_effect_csv() {
        match read_sweep_file(&dir.join(SWEEP_EFFECT_CSV_FILE)) {
            Ok(csv) => report.check(
                "sweep effect csv",
                csv == table.as_bytes(),
                format!("{SWEEP_EFFECT_CSV_FILE} is the manifest's effect table"),
            ),
            Err(error) => report.check("sweep effect csv", false, error.to_string()),
        }
    }
    Ok((report, manifest))
}

/// One point (or the baseline) compared across two sweeps.
#[derive(Debug, Clone, Serialize)]
pub struct SweepPointComparison {
    pub id: String,
    pub present_in_a: bool,
    pub present_in_b: bool,
    pub semantic_hash_equal: bool,
    pub payload_hash_equal: bool,
    pub metrics_equal: bool,
}

/// Two verified sweeps compared point by point.
#[derive(Debug, Clone, Serialize)]
pub struct SweepComparison {
    pub sweep_a: String,
    pub sweep_b: String,
    pub sweep_hash_equal: bool,
    pub spec_equal: bool,
    pub points: Vec<SweepPointComparison>,
    /// `exact` when every point (and the baseline) is present in both with
    /// equal semantic hashes; `different` otherwise.
    pub verdict: String,
}

/// Compare two sweep directories (each verified first).
pub fn compare_sweeps(a: &Path, b: &Path) -> Result<SweepComparison, String> {
    compare_anchored_sweeps(a, b, None, None)
}

/// [`compare_sweeps`] with optional expected sweep hashes, checked in the
/// same verification pass whose manifests are compared.
pub fn compare_anchored_sweeps(
    a: &Path,
    b: &Path,
    expect_a: Option<&str>,
    expect_b: Option<&str>,
) -> Result<SweepComparison, String> {
    let mut manifests = Vec::with_capacity(2);
    for (dir, expected) in [(a, expect_a), (b, expect_b)] {
        let options = VerifyOptions {
            expected_semantic_hash: expected.map(str::to_string),
            ..VerifyOptions::default()
        };
        let (report, manifest) = verify_and_load_sweep(dir, &options)?;
        manifests.push(manifest);
        if !report.ok {
            let failed: Vec<String> = report
                .checks
                .iter()
                .filter(|check| !check.ok)
                .map(|check| format!("{}: {}", check.name, check.detail))
                .collect();
            return Err(format!(
                "sweep '{}' failed verification: {}",
                dir.display(),
                failed.join("; ")
            ));
        }
    }
    let (mb, ma) = (manifests.pop().expect("two"), manifests.pop().expect("two"));
    let mut points = vec![SweepPointComparison {
        id: "baseline".into(),
        present_in_a: true,
        present_in_b: true,
        semantic_hash_equal: ma.baseline.semantic_hash == mb.baseline.semantic_hash,
        payload_hash_equal: ma.baseline.payload_hash == mb.baseline.payload_hash,
        metrics_equal: true,
    }];
    let mut ids: Vec<&str> = ma.points.iter().map(|p| p.id.as_str()).collect();
    for point in &mb.points {
        if !ids.contains(&point.id.as_str()) {
            ids.push(&point.id);
        }
    }
    for id in ids {
        let pa = ma.points.iter().find(|p| p.id == id);
        let pb = mb.points.iter().find(|p| p.id == id);
        points.push(SweepPointComparison {
            id: id.to_string(),
            present_in_a: pa.is_some(),
            present_in_b: pb.is_some(),
            semantic_hash_equal: matches!((pa, pb), (Some(x), Some(y)) if x.bundle.semantic_hash == y.bundle.semantic_hash),
            payload_hash_equal: matches!((pa, pb), (Some(x), Some(y)) if x.bundle.payload_hash == y.bundle.payload_hash),
            metrics_equal: matches!((pa, pb), (Some(x), Some(y)) if x.inputs == y.inputs && x.effect == y.effect),
        });
    }
    let exact = points
        .iter()
        .all(|p| p.present_in_a && p.present_in_b && p.semantic_hash_equal);
    Ok(SweepComparison {
        sweep_a: a.display().to_string(),
        sweep_b: b.display().to_string(),
        sweep_hash_equal: ma.sweep_hash == mb.sweep_hash,
        spec_equal: ma.spec_sha256 == mb.spec_sha256,
        points,
        verdict: if exact { "exact" } else { "different" }.to_string(),
    })
}
