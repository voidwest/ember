//! Frozen bytes from the v0.5 writer, rather than a current-writer round trip.
use ember::v05::verify::{verify_bundle, VerifyOptions};
use std::path::Path;

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

#[test]
fn v050_bundle_keeps_its_original_identity_and_verifies_offline() {
    let root = std::env::temp_dir().join(format!(
        "ember-v050-compat-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    copy_tree(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/bundle-v050")
            .as_path(),
        &root,
    );
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
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn experiment_verify_cli_distinguishes_failed_verdict_from_usage_error() {
    use std::io::Write;
    use std::process::Command;
    let root = std::env::temp_dir().join(format!(
        "ember-verdict-cli-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let bundle = root.join("bundle");
    copy_tree(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bundle-v050"),
        &bundle,
    );
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
