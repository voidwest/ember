//! Frozen bytes from the v0.5 writer, rather than a current-writer round trip.
use ember::v05::verify::{verify_bundle, VerifyOptions};
use std::path::{Path, PathBuf};

fn copy_tree(source: &Path, destination: &Path) {
    std::fs::create_dir_all(destination).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else if target
            .extension()
            .is_some_and(|extension| extension == "gz")
        {
            let mut decoder =
                flate2::read::GzDecoder::new(std::fs::File::open(entry.path()).unwrap());
            let mut output = std::fs::File::create(target.with_extension("")).unwrap();
            std::io::copy(&mut decoder, &mut output).unwrap();
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// A fresh scratch root with the frozen v0.5 fixture bundle copied to
/// `<root>/bundle`; returns `(root, bundle)`.
fn fixture_copy(tag: &str) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "ember-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    let bundle = root.join("bundle");
    copy_tree(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bundle-v050"),
        &bundle,
    );
    (root, bundle)
}

#[test]
fn v050_bundle_keeps_its_original_identity_and_verifies_offline() {
    let (scratch, root) = fixture_copy("v050-compat");
    let plan: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("execution-plan.json")).unwrap()).unwrap();
    assert!(
        plan.get("kernel_revision").is_none(),
        "keep the historical encoding"
    );
    let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
    assert!(report.ok, "{:?}", report.checks);
    assert!(report
        .checks
        .iter()
        .any(|check| check.name == "semantic hash"
            && check.detail.contains("legacy insertion-order JSON")));
    assert_eq!(
        report.semantic_hash,
        "f21442ebae87865c059722cc06a2feb00c22ebb2e99cb6accc84f78d13bd7325"
    );
    assert_eq!(
        report.payload_hash,
        "afcec8ffee90c2d57575c0695c30c82ea2e9d55e7316dfff085ecee3dca70f97"
    );
    // The compatibility exception must not silently become the hash contract
    // for a new producer. Reseal every file so only the encoding check fails.
    use ember::v05::manifest::{sha256_hex, BundleIdentity};
    let semantic_path = root.join("semantic-manifest.json");
    let mut semantic: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&semantic_path).unwrap()).unwrap();
    semantic["ember_version"] = "1.0.0".into();
    let legacy_hash = sha256_hex(&serde_json::to_vec(&semantic).unwrap());
    let bytes = serde_json::to_vec_pretty(&semantic).unwrap();
    std::fs::write(&semantic_path, &bytes).unwrap();
    let mut inventory: std::collections::BTreeMap<String, String> =
        serde_json::from_value(semantic["payloads"].clone()).unwrap();
    inventory.insert("semantic-manifest.json".into(), sha256_hex(&bytes));
    let manifest_path = root.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["semantic_hash"] = legacy_hash.into();
    manifest["payload_hash"] = BundleIdentity::payload_hash(&inventory).unwrap().into();
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let checksums_path = root.join("checksums.sha256");
    let checksums = std::fs::read_to_string(&checksums_path)
        .unwrap()
        .lines()
        .map(|line| {
            let (_, path) = line.split_once("  ").unwrap();
            format!(
                "{}  {path}\n",
                sha256_hex(&std::fs::read(root.join(path)).unwrap())
            )
        })
        .collect::<String>();
    std::fs::write(checksums_path, checksums).unwrap();
    let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
    assert!(!report.ok);
    let failures: Vec<_> = report
        .checks
        .iter()
        .filter(|check| !check.ok)
        .map(|check| check.name.as_str())
        .collect();
    assert_eq!(failures, ["semantic hash"]);
    std::fs::remove_dir_all(scratch).unwrap();
}

#[test]
fn experiment_verify_cli_distinguishes_failed_verdict_from_usage_error() {
    use std::io::Write;
    use std::process::Command;
    let (root, bundle) = fixture_copy("verdict-cli");
    let verify = || {
        Command::new(env!("CARGO_BIN_EXE_ember"))
            .args(["experiment", "verify"])
            .arg(&bundle)
            .arg("--json")
            .output()
            .unwrap()
    };
    let valid = verify();
    assert_eq!(valid.status.code(), Some(0));
    let valid_report: serde_json::Value = serde_json::from_slice(&valid.stdout).unwrap();
    assert_eq!(valid_report["ok"], true);
    std::fs::OpenOptions::new()
        .append(true)
        .open(bundle.join("outputs.jsonl"))
        .unwrap()
        .write_all(b"\n")
        .unwrap();
    let failed = verify();
    assert_eq!(failed.status.code(), Some(3));
    let report: serde_json::Value = serde_json::from_slice(&failed.stdout).unwrap();
    assert_eq!(report["ok"], false);
    assert!(report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|check| check["ok"] == false
            && check["detail"]
                .as_str()
                .unwrap()
                .contains("checksum mismatch")));
    assert!(
        failed.stderr.is_empty(),
        "completed verdict must retain its structured report"
    );
    let usage = Command::new(env!("CARGO_BIN_EXE_ember"))
        .args(["experiment", "verify", "--not-a-valid-flag"])
        .output()
        .unwrap();
    assert_eq!(usage.status.code(), Some(2));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn experiment_verify_anchors_identity_externally_and_never_writes_into_the_bundle() {
    use std::process::Command;
    const SEMANTIC: &str = "f21442ebae87865c059722cc06a2feb00c22ebb2e99cb6accc84f78d13bd7325";
    let (root, bundle) = fixture_copy("anchor-cli");
    let ember = |args: &[&std::ffi::OsStr]| {
        Command::new(env!("CARGO_BIN_EXE_ember"))
            .args(args)
            .output()
            .unwrap()
    };
    let os = |value: &str| std::ffi::OsString::from(value);
    let verify = |extra: &[std::ffi::OsString]| {
        let mut args = vec![os("experiment"), os("verify"), bundle.clone().into()];
        args.extend_from_slice(extra);
        let refs: Vec<&std::ffi::OsStr> = args.iter().map(|arg| arg.as_os_str()).collect();
        ember(&refs)
    };

    // Unanchored: verified, with the hash printed and the caveat stated.
    let plain = verify(&[]);
    assert_eq!(plain.status.code(), Some(0), "{plain:?}");
    let text = String::from_utf8_lossy(&plain.stdout);
    assert!(
        text.contains(&format!("semantic hash: {SEMANTIC}")),
        "{text}"
    );
    assert!(text.contains("self-consistent only"), "{text}");
    assert!(!bundle.join("verification.json").exists());

    // Semantic-hash anchor: right value passes, wrong value is a verdict.
    let good = verify(&[os("--expect-semantic-hash"), os(&SEMANTIC.to_uppercase())]);
    assert_eq!(good.status.code(), Some(0), "{good:?}");
    assert!(String::from_utf8_lossy(&good.stdout).contains("anchored"));
    let wrong = verify(&[
        os("--expect-semantic-hash"),
        os(&"0".repeat(64)),
        os("--json"),
    ]);
    assert_eq!(wrong.status.code(), Some(3), "{wrong:?}");
    let report: serde_json::Value = serde_json::from_slice(&wrong.stdout).unwrap();
    assert!(report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|check| check["name"] == "semantic hash anchor" && check["ok"] == false));
    assert_eq!(
        verify(&[os("--expect-semantic-hash"), os("abc")])
            .status
            .code(),
        Some(2)
    );

    // Evidence anchor: a trusted signature over manifest.json.
    let key = root.join("keys/evidence.key");
    let other = root.join("keys/other.key");
    for path in [&key, &other] {
        let out = ember(&[
            os("evidence").as_os_str(),
            os("init").as_os_str(),
            os("--key").as_os_str(),
            path.as_os_str(),
        ]);
        assert!(out.status.success(), "{out:?}");
    }
    let envelope = root.join("manifest.signed.json");
    let sign = |manifest: &Path, out: &Path| {
        let result = ember(&[
            os("evidence").as_os_str(),
            os("sign").as_os_str(),
            os("--manifest").as_os_str(),
            manifest.as_os_str(),
            os("--key").as_os_str(),
            key.as_os_str(),
            os("--out").as_os_str(),
            out.as_os_str(),
        ]);
        assert!(result.status.success(), "{result:?}");
    };
    sign(&bundle.join("manifest.json"), &envelope);
    let signed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&envelope).unwrap()).unwrap();
    assert_eq!(signed["schema"], "signed-evidence-v2");
    let anchored = verify(&[
        os("--expect-evidence"),
        envelope.clone().into(),
        os("--trusted-key"),
        key.with_extension("pub").into(),
    ]);
    assert_eq!(anchored.status.code(), Some(0), "{anchored:?}");
    let untrusted = verify(&[
        os("--expect-evidence"),
        envelope.clone().into(),
        os("--trusted-key"),
        other.with_extension("pub").into(),
    ]);
    assert_eq!(untrusted.status.code(), Some(3), "{untrusted:?}");
    // A validly signed manifest of a different bundle does not anchor this one.
    let foreign_manifest = root.join("foreign-manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    manifest["semantic_hash"] = "1".repeat(64).into();
    std::fs::write(&foreign_manifest, manifest.to_string()).unwrap();
    let foreign = root.join("foreign.signed.json");
    sign(&foreign_manifest, &foreign);
    let mismatched = verify(&[
        os("--expect-evidence"),
        foreign.into(),
        os("--trusted-key"),
        key.with_extension("pub").into(),
    ]);
    assert_eq!(mismatched.status.code(), Some(3), "{mismatched:?}");
    // The envelope alone proves nothing about who signed it.
    assert_eq!(
        verify(&[os("--expect-evidence"), envelope.clone().into()])
            .status
            .code(),
        Some(2)
    );
    // `--trusted-key` alone looks for the sibling `<bundle>.evidence.json`
    // that `experiment run --sign-key` writes; a missing one is a failed
    // anchor, not a silent pass.
    let missing = verify(&[
        os("--trusted-key"),
        key.with_extension("pub").into(),
        os("--json"),
    ]);
    assert_eq!(missing.status.code(), Some(3), "{missing:?}");
    let sibling = root.join("bundle.evidence.json");
    sign(&bundle.join("manifest.json"), &sibling);
    let discovered = verify(&[os("--trusted-key"), key.with_extension("pub").into()]);
    assert_eq!(discovered.status.code(), Some(0), "{discovered:?}");
    assert!(String::from_utf8_lossy(&discovered.stdout).contains("evidence anchor"));
    let discovered_untrusted = verify(&[os("--trusted-key"), other.with_extension("pub").into()]);
    assert_eq!(
        discovered_untrusted.status.code(),
        Some(3),
        "{discovered_untrusted:?}"
    );

    // Reports go outside the bundle, never into it.
    let report_path = root.join("report.json");
    assert_eq!(
        verify(&[os("--write-report"), report_path.clone().into()])
            .status
            .code(),
        Some(0)
    );
    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(written["semantic_hash"], SEMANTIC);
    let inside = verify(&[
        os("--write-report"),
        bundle.join("verification.json").into(),
    ]);
    assert_ne!(inside.status.code(), Some(0));
    assert!(!bundle.join("verification.json").exists());

    // compare and reproduce refuse an original that misses its anchor
    // before doing anything else.
    let compare = ember(&[
        os("experiment").as_os_str(),
        os("compare").as_os_str(),
        bundle.as_os_str(),
        bundle.as_os_str(),
        os("--expect-b-semantic-hash").as_os_str(),
        os(&"0".repeat(64)).as_os_str(),
    ]);
    assert!(!compare.status.success());
    assert!(String::from_utf8_lossy(&compare.stderr).contains("semantic hash anchor"));
    let reproduce = ember(&[
        os("experiment").as_os_str(),
        os("reproduce").as_os_str(),
        bundle.as_os_str(),
        os("--model").as_os_str(),
        root.join("absent.gguf").as_os_str(),
        os("--expect-semantic-hash").as_os_str(),
        os(&"0".repeat(64)).as_os_str(),
    ]);
    assert!(!reproduce.status.success());
    assert!(
        String::from_utf8_lossy(&reproduce.stderr).contains("semantic hash anchor"),
        "{reproduce:?}"
    );
    std::fs::remove_dir_all(root).unwrap();
}
