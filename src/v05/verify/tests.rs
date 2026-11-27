#[test]
fn short_hash_never_panics_on_untrusted_hashes() {
    assert_eq!(super::short_hash("0123456789abcdef"), "0123456789ab");
    assert_eq!(super::short_hash(""), "");
    assert_eq!(super::short_hash("abc"), "abc");
    // A multi-byte character straddling byte 12.
    assert_eq!(super::short_hash("0123456789aé-tail"), "0123456789aé");
}

#[test]
fn generated_selection_uses_absolute_decode_positions() {
    use crate::v05::token_select::TokenSelector;
    let mut record =
        crate::v05::testutil::sample_selection_record(TokenSelector::GeneratedStep { step: 1 });
    record.selected_indices = vec![1];
    assert!(super::selection_consistency(&record).is_none());
    for invalid in [vec![], vec![0], vec![2], vec![1, 1]] {
        record.selected_indices = invalid;
        assert!(super::selection_consistency(&record).is_some());
    }
    record.selector = TokenSelector::GeneratedStep { step: 3 };
    record.selected_indices = vec![3];
    assert!(super::selection_consistency(&record).is_none());
    record.selector = TokenSelector::GeneratedStep { step: 0 };
    assert!(super::selection_consistency(&record).is_some());
    record.token_ids.push(2);
    record.selector = TokenSelector::GeneratedStep { step: usize::MAX };
    assert!(super::selection_consistency(&record).is_some());
    record.selector = TokenSelector::PromptFinal;
    assert!(super::selection_consistency(&record).is_some());
}

use super::*;
use crate::v05::testutil;
use crate::v05::testutil::temp_root;

thread_local! {
    /// Bundle file opens per path on this test thread.
    static OPENS: std::cell::RefCell<BTreeMap<PathBuf, usize>> =
        const { std::cell::RefCell::new(BTreeMap::new()) };
}

pub(super) fn note_open(path: &Path) {
    OPENS.with(|opens| *opens.borrow_mut().entry(path.to_path_buf()).or_insert(0) += 1);
}

fn take_opens() -> BTreeMap<PathBuf, usize> {
    OPENS.with(|opens| std::mem::take(&mut *opens.borrow_mut()))
}

fn failed_names(report: &VerificationReport) -> Vec<&str> {
    report
        .checks
        .iter()
        .filter(|check| !check.ok)
        .map(|check| check.name.as_str())
        .collect()
}

fn read_semantic_manifest(root: &Path) -> Result<SemanticManifest, String> {
    parse_semantic_manifest(&std::fs::read(root.join("semantic-manifest.json")).unwrap())
}

/// Write the standard test bundle into a fresh temp dir, run `mutate`,
/// verify, and require exactly the named checks to fail.
fn assert_verification_failure(
    tag: &str,
    mutate: impl FnOnce(&std::path::Path),
    expect_failed: &[&str],
) {
    let root = testutil::sample_bundle(tag);
    mutate(&root);
    let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
    assert!(!report.ok);
    let names = failed_names(&report);
    for expected in expect_failed {
        assert!(
            names.contains(expected),
            "missing {expected:?} among {names:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn valid_bundle_verifies() {
    let root = testutil::sample_bundle("valid");
    let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
    assert!(report.ok, "{:?}", report.checks);
    assert_eq!(report.checks.len(), 22);
    // Verification never writes into the bundle it checks.
    assert!(!root.join("verification.json").exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn verification_leaves_a_read_only_bundle_untouched() {
    let root = testutil::sample_bundle("read-only");
    let before: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o555)).unwrap();
    }
    let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    assert!(report.ok, "{:?}", report.checks);
    let after: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(before, after);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_forged_verification_json_cannot_change_the_verdict() {
    let forged = serde_json::json!({
        "bundle_schema": BUNDLE_SCHEMA_V1, "ok": true, "semantic_hash": "0",
        "payload_hash": "0", "checks": [], "warnings": [], "timestamp": "x",
    });
    // A valid bundle with a stale report still verifies, with a warning.
    let root = testutil::sample_bundle("forged-report");
    std::fs::write(root.join("verification.json"), forged.to_string()).unwrap();
    let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
    assert!(report.ok, "{:?}", report.checks);
    assert!(report
        .warnings
        .iter()
        .any(|warning| warning.contains("verification.json")));
    // A corrupted bundle with a report claiming success still fails.
    let path = root.join("outputs.jsonl");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(b'\n');
    std::fs::write(&path, bytes).unwrap();
    assert!(!verify_bundle(&root, &VerifyOptions::default()).unwrap().ok);
    std::fs::remove_dir_all(root).unwrap();
    // Listing the report as a bundle file is not a way to smuggle it in.
    assert_verification_failure(
        "listed-report",
        |root| {
            std::fs::write(root.join("verification.json"), forged.to_string()).unwrap();
            let path = root.join("manifest.json");
            let mut manifest: BundleManifest =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            manifest.files.push("verification.json".into());
            std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        },
        &["bundle file inventory"],
    );
}

#[test]
fn each_bundle_file_is_opened_once_and_the_checked_bytes_are_returned() {
    let root = testutil::sample_bundle("read-once");
    take_opens();
    let loaded = load_bundle_for_source(&root).unwrap();
    let opens = take_opens();
    let repeated: Vec<_> = opens.iter().filter(|(_, count)| **count > 1).collect();
    assert!(repeated.is_empty(), "{repeated:?}");
    assert!(opens.contains_key(&root.join("captures/tensors.safetensors")));
    // Replacing files after verification changes nothing that was loaded.
    let outputs = loaded.file("outputs.jsonl").unwrap().to_vec();
    std::fs::write(root.join("outputs.jsonl"), b"tampered\n").unwrap();
    std::fs::write(root.join("captures/tensors.safetensors"), b"x").unwrap();
    assert_eq!(loaded.file("outputs.jsonl").unwrap(), outputs.as_slice());
    assert_eq!(
        loaded
            .tensor_f32_by_name("cap-1/i1/residual-post-mlp/0")
            .unwrap(),
        testutil::sample_rows()
    );
    assert_eq!(loaded.semantic_hash, loaded.manifest.semantic_hash);
    std::fs::remove_dir_all(root).unwrap();
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/bundle-v050")
            .join(name),
    )
    .unwrap()
}

#[test]
fn resolved_experiment_is_bound_to_the_hashed_spec_and_manifest() {
    let spec = String::from_utf8(fixture("experiment.toml")).unwrap();
    let resolved = fixture("resolved-experiment.json");
    let semantic = parse_semantic_manifest(&fixture("semantic-manifest.json")).unwrap();
    // The frozen v0.5 bundle (run with an --output override) binds.
    let bound = bind_resolved_experiment(&spec, &resolved, &semantic).unwrap();
    assert_eq!(bound.generation.max_new_tokens, 8);
    let edit = |change: &dyn Fn(&mut serde_json::Value)| {
        let mut value: serde_json::Value = serde_json::from_slice(&resolved).unwrap();
        change(&mut value);
        bind_resolved_experiment(&spec, &serde_json::to_vec(&value).unwrap(), &semantic)
    };
    // Fields outside the semantic hash that change what runs.
    for change in [
        &(|v: &mut serde_json::Value| v["generation"]["max_new_tokens"] = 4096.into())
            as &dyn Fn(&mut serde_json::Value),
        &|v| v["generation"]["temperature"] = 1.5.into(),
        &|v| v["model"]["tokenizer"] = "/elsewhere/tokenizer.json".into(),
        &|v| v["model"]["tokenizer_expected_sha256"] = "".into(),
        &|v| v["output"]["overwrite"] = true.into(),
    ] {
        let error = edit(change).unwrap_err();
        assert!(error.contains("hashed experiment.toml"), "{error}");
    }
    // A mode override is legitimate only when the manifest agrees.
    let error = edit(&|v| v["execution"]["mode"] = "planned".into()).unwrap_err();
    assert!(error.contains("semantic manifest"), "{error}");
    assert!(edit(&|v| v["execution"]["threads"] = 2.into()).is_ok());
    assert!(edit(&|v| v["output"]["directory"] = "runs/x".into()).is_ok());
    // An edited spec that no longer matches the recorded semantics.
    let edited_spec = spec.replace("max_new_tokens = 8", "max_new_tokens = 9");
    assert!(bind_resolved_experiment(&edited_spec, &resolved, &semantic).is_err());
}

#[test]
fn semantic_hash_anchor_rejects_a_resealed_bundle() {
    let root = testutil::sample_bundle("anchor");
    let original = verify_bundle(&root, &VerifyOptions::default()).unwrap();
    let anchored = |expected: &str| VerifyOptions {
        expected_semantic_hash: Some(expected.to_string()),
        ..VerifyOptions::default()
    };
    let report = verify_bundle(&root, &anchored(&original.semantic_hash.to_uppercase())).unwrap();
    assert!(report.ok, "{:?}", report.checks);
    assert!(report
        .checks
        .iter()
        .any(|check| check.name == "semantic hash anchor" && check.ok));
    // Edit the science and reseal every bundle-authored hash: the bundle
    // is self-consistent again, so only the external anchor catches it.
    let mut semantic = read_semantic_manifest(&root).unwrap();
    semantic.experiment.name = "edited".into();
    reseal(&root, &mut semantic);
    assert!(verify_bundle(&root, &VerifyOptions::default()).unwrap().ok);
    let report = verify_bundle(&root, &anchored(&original.semantic_hash)).unwrap();
    let failed = failed_names(&report);
    assert_eq!(failed, ["semantic hash anchor"]);
    assert!(load_verified_bundle(&root, &anchored(&original.semantic_hash)).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

// Reseal mutated fixtures so a failed schema/identity check cannot be
// explained by stale hashes. This does not confer trust on their contents.
fn reseal(root: &Path, semantic: &mut SemanticManifest) {
    for (name, checksum) in &mut semantic.payloads {
        *checksum = sha256_hex(&std::fs::read(root.join(name)).unwrap());
    }
    let bytes = crate::v05::manifest::canonical_json(semantic).unwrap();
    std::fs::write(root.join("semantic-manifest.json"), &bytes).unwrap();
    let path = root.join("manifest.json");
    let mut manifest: BundleManifest =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest.semantic_hash = BundleIdentity::semantic_hash(semantic).unwrap();
    let mut inventory = semantic.payloads.clone();
    inventory.insert("semantic-manifest.json".into(), sha256_hex(&bytes));
    manifest.payload_hash = BundleIdentity::payload_hash(&inventory).unwrap();
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let checksums = manifest
        .files
        .iter()
        .filter(|name| !matches!(name.as_str(), "checksums.sha256" | "verification.json"))
        .map(|name| {
            format!(
                "{}  {name}\n",
                sha256_hex(&std::fs::read(root.join(name)).unwrap())
            )
        })
        .collect::<String>();
    std::fs::write(root.join("checksums.sha256"), checksums).unwrap();
}

/// Content the semantic identity does not cover must fail verification,
/// or a semantic-hash or signature anchor would vouch for it.
#[test]
fn content_outside_the_semantic_identity_fails_closed() {
    // An extra listed file, resealed into manifest.json and checksums.
    let root = testutil::sample_bundle("uncovered-file");
    let original = verify_bundle(&root, &VerifyOptions::default()).unwrap();
    assert!(original.ok, "{:?}", original.checks);
    std::fs::write(root.join("notes.md"), "trust me\n").unwrap();
    let path = root.join("manifest.json");
    let mut manifest: BundleManifest =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest.files.push("notes.md".into());
    manifest.files.sort();
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let mut semantic = read_semantic_manifest(&root).unwrap();
    reseal(&root, &mut semantic);
    let anchored = VerifyOptions {
        expected_semantic_hash: Some(original.semantic_hash),
        ..VerifyOptions::default()
    };
    let report = verify_bundle(&root, &anchored).unwrap();
    assert_eq!(failed_names(&report), ["semantic payload checksums"]);
    std::fs::remove_dir_all(root).unwrap();

    // An unknown semantic-manifest field, with every hash resealed.
    let root = testutil::sample_bundle("unknown-field");
    let original = verify_bundle(&root, &VerifyOptions::default()).unwrap();
    let anchored = VerifyOptions {
        expected_semantic_hash: Some(original.semantic_hash),
        ..VerifyOptions::default()
    };
    let mut semantic = read_semantic_manifest(&root).unwrap();
    reseal(&root, &mut semantic);
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("semantic-manifest.json")).unwrap())
            .unwrap();
    value["model"]["note"] = "unchecked".into();
    let bytes = crate::v05::manifest::canonical_json(&value).unwrap();
    std::fs::write(root.join("semantic-manifest.json"), &bytes).unwrap();
    let mut manifest: BundleManifest =
        serde_json::from_slice(&std::fs::read(path_of(&root)).unwrap()).unwrap();
    let mut inventory = semantic.payloads.clone();
    inventory.insert("semantic-manifest.json".into(), sha256_hex(&bytes));
    manifest.payload_hash = BundleIdentity::payload_hash(&inventory).unwrap();
    std::fs::write(path_of(&root), serde_json::to_vec(&manifest).unwrap()).unwrap();
    rewrite_checksums(&root, &manifest);
    let report = verify_bundle(&root, &anchored).unwrap();
    assert_eq!(failed_names(&report), ["semantic manifest fields"]);
    std::fs::remove_dir_all(root).unwrap();
}

fn path_of(root: &Path) -> PathBuf {
    root.join("manifest.json")
}

fn rewrite_checksums(root: &Path, manifest: &BundleManifest) {
    let checksums = manifest
        .files
        .iter()
        .filter(|name| !matches!(name.as_str(), "checksums.sha256" | "verification.json"))
        .map(|name| {
            format!(
                "{}  {name}\n",
                sha256_hex(&std::fs::read(root.join(name)).unwrap())
            )
        })
        .collect::<String>();
    std::fs::write(root.join("checksums.sha256"), checksums).unwrap();
}

#[test]
fn unknown_nested_contracts_fail_even_when_hashes_are_valid() {
    for field in ["experiment schema", "hook schema", "plan schema"] {
        let root = testutil::sample_bundle("unknown-contract");
        let mut semantic = read_semantic_manifest(&root).unwrap();
        match field {
            "experiment schema" => semantic.experiment_schema = "ember.experiment.v2".into(),
            "hook schema" => semantic.hook_schema = 2,
            "plan schema" => semantic.plan_schema = 2,
            _ => unreachable!(),
        }
        reseal(&root, &mut semantic);
        let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
        let failed = failed_names(&report);
        assert_eq!(failed, [field]);
        assert!(!report.ok);
        assert!(load_bundle_for_source(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn stored_plan_must_match_declared_contract_and_identity() {
    for change_schema in [false, true] {
        let root = testutil::sample_bundle("plan-contract");
        let mut semantic = read_semantic_manifest(&root).unwrap();
        let expected = if change_schema {
            let path = root.join("execution-plan.json");
            let mut plan: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            plan["schema_version"] = serde_json::json!(2);
            semantic.execution.plan_hash = testutil::fixture_plan_hash(&mut plan);
            std::fs::write(path, serde_json::to_vec(&plan).unwrap()).unwrap();
            "execution plan schema"
        } else {
            semantic.execution.plan_hash = "0".repeat(64);
            "execution plan identity"
        };
        reseal(&root, &mut semantic);
        let report = verify_bundle(&root, &VerifyOptions::default()).unwrap();
        let failed = failed_names(&report);
        assert_eq!(failed, [expected]);
        assert!(!report.ok);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn unlisted_files_and_unchecked_files_fail() {
    assert_verification_failure(
        "extra-file",
        |root| {
            std::fs::write(root.join("unindexed.bin"), b"extra payload").unwrap();
        },
        &["bundle file inventory"],
    );
    assert_verification_failure(
        "missing-checksum",
        |root| {
            let path = root.join("checksums.sha256");
            let text = std::fs::read_to_string(&path).unwrap();
            let filtered = text
                .lines()
                .filter(|line| !line.ends_with("  outputs.jsonl"))
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(path, filtered).unwrap();
        },
        &["checksums"],
    );
}

#[cfg(unix)]
#[test]
fn a_folder_swapped_for_a_symlink_is_not_followed() {
    let root = testutil::sample_bundle("folder-symlink");
    assert!(open_regular_file(&root, "captures/tensors.safetensors").is_ok());
    // After the inventory walk: move captures/ out and point a symlink at it.
    let outside = root.with_extension("outside");
    let _ = std::fs::remove_dir_all(&outside);
    std::fs::rename(root.join("captures"), &outside).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("captures")).unwrap();
    let error = open_regular_file(&root, "captures/tensors.safetensors").unwrap_err();
    assert!(error.contains("symlink"), "{error}");
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&outside);
}

#[test]
#[cfg(unix)]
fn symlinks_are_rejected_before_manifest_or_payload_reads() {
    for name in [
        "manifest.json",
        "captures/tensors.safetensors",
        "verification.json",
    ] {
        let root = testutil::sample_bundle("symlink");
        let path = root.join(name);
        if path.exists() {
            std::fs::remove_file(&path).unwrap();
        }
        // A dangling target ensures rejection is based on entry type, not
        // on accidentally succeeding while following the target.
        std::os::unix::fs::symlink(root.join("absent-external-target"), &path).unwrap();
        let error = verify_bundle(&root, &VerifyOptions::default()).unwrap_err();
        assert!(error.contains("symlink"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn duplicate_checksum_paths_fail() {
    let root = testutil::sample_bundle("duplicate-checksum");
    let path = root.join("checksums.sha256");
    let mut text = std::fs::read_to_string(&path).unwrap();
    let duplicate = text.lines().next().unwrap().to_owned();
    text.push_str(&duplicate);
    text.push('\n');
    std::fs::write(path, text).unwrap();
    assert!(verify_bundle(&root, &VerifyOptions::default())
        .unwrap_err()
        .contains("repeats path"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn one_byte_payload_corruption_fails() {
    assert_verification_failure(
        "corrupt",
        |root| {
            let payload = root.join("captures/tensors.safetensors");
            let mut bytes = std::fs::read(&payload).unwrap();
            let last = bytes.len() - 1;
            bytes[last] ^= 0xFF;
            std::fs::write(&payload, bytes).unwrap();
        },
        &["checksums", "tensor payload"],
    );
}

#[test]
fn removed_file_fails() {
    assert_verification_failure(
        "removed",
        |root| {
            std::fs::remove_file(root.join("captures/index.jsonl")).unwrap();
        },
        &["required files"],
    );
}

#[test]
fn altered_manifest_value_fails() {
    assert_verification_failure(
        "altered",
        |root| {
            let path = root.join("semantic-manifest.json");
            let mut value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            value["experiment"]["name"] = serde_json::json!("tampered");
            std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        },
        &["semantic hash"],
    );
}

#[test]
fn extra_unindexed_tensor_fails() {
    assert_verification_failure(
        "extra",
        |root| {
            // Append a second tensor to the payload and reseal every
            // hash so only the unindexed-tensor check can catch it.
            let payload_path = root.join("captures/tensors.safetensors");
            let extra = crate::v05::safetensors::serialize(&[
                crate::v05::safetensors::TensorData {
                    name: "cap-1/i1/residual-post-mlp/0",
                    dtype: crate::v05::safetensors::TensorDType::F32,
                    shape: &[1, 4],
                    bytes: &[0u8; 16],
                },
                crate::v05::safetensors::TensorData {
                    name: "rogue/i1/residual-post-mlp/0",
                    dtype: crate::v05::safetensors::TensorDType::F32,
                    shape: &[1, 4],
                    bytes: &[0u8; 16],
                },
            ])
            .unwrap();
            std::fs::write(&payload_path, extra).unwrap();
            let mut semantic = read_semantic_manifest(root).unwrap();
            reseal(root, &mut semantic);
        },
        &["tensor payload"],
    );
}

#[test]
fn index_entries_sharing_a_tensor_name_fail() {
    assert_verification_failure(
        "duplicate-tensor-name",
        |root| {
            // A second entry for another capture that points at the same
            // tensor: only one of the two could ever be compared.
            let path = root.join("captures/index.jsonl");
            let text = std::fs::read_to_string(&path).unwrap();
            let first = text.lines().next().unwrap();
            let mut entry: serde_json::Value = serde_json::from_str(first).unwrap();
            entry["capture_id"] = "shadow".into();
            entry["shape"] = serde_json::json!([9, 9]);
            // Placed first, so the genuine entry after it would win a
            // last-one-wins lookup and this one would go unchecked.
            std::fs::write(&path, format!("{entry}\n{text}")).unwrap();
            let mut semantic = read_semantic_manifest(root).unwrap();
            reseal(root, &mut semantic);
        },
        &["tensor payload"],
    );
}

#[test]
fn incomplete_bundle_fails() {
    let root = temp_root("incomplete");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("manifest.json"), b"{}").unwrap();
    // A malformed manifest is a hard error; a missing-file bundle
    // yields a failing report. Both must be non-ok.
    let report = verify_bundle(&root, &VerifyOptions::default());
    if let Ok(report) = report {
        assert!(!report.ok);
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn deep_metadata_reads_string_arrays_and_boolean_values() {
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    let root = temp_root("metadata.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&5u64.to_le_bytes());
    string(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    string(&mut bytes, "hello");
    string(&mut bytes, "كتاب");
    string(&mut bytes, "tokenizer.ggml.add_bos_token");
    bytes.extend_from_slice(&7u32.to_le_bytes());
    bytes.push(1);
    string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    string(&mut bytes, "llama");
    string(&mut bytes, "llama.block_count");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&16u32.to_le_bytes());
    // Another architecture's count must not override the selected one.
    string(&mut bytes, "other.block_count");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&99u32.to_le_bytes());
    std::fs::write(&root, &bytes).unwrap();
    assert_eq!(read_gguf_summary(&root).unwrap(), ("llama".into(), 16));
    bytes.pop();
    std::fs::write(&root, &bytes).unwrap();
    assert!(read_gguf_summary(&root).is_err());
    std::fs::remove_file(root).unwrap();
}

#[test]
fn deep_metadata_rejects_skips_past_eof_and_overflow() {
    let mut reader = std::io::Cursor::new(vec![0u8; 8]);
    assert!(skip_gguf_bytes(&mut reader, 9, 8).is_err());
    reader.set_position(4);
    assert!(skip_gguf_bytes(&mut reader, u64::MAX, 8).is_err());
}

#[test]
fn deep_model_mismatch_fails() {
    let root = testutil::sample_bundle("deep");
    // A non-model file with the wrong hash fails the deep check.
    let model_path = temp_root("fake-model.gguf");
    std::fs::write(&model_path, b"not a model").unwrap();
    let options = VerifyOptions {
        model_path: Some(model_path.clone()),
        ..VerifyOptions::default()
    };
    let report = verify_bundle(&root, &options).unwrap();
    assert!(!report.ok);
    let names = failed_names(&report);
    assert!(names.contains(&"deep model sha256"), "{names:?}");
    std::fs::remove_file(model_path).unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn traversal_checksum_fails_even_with_the_correct_hash() {
    // A hostile bundle that lists `../victim` with the victim's real
    // hash must still fail: the path itself is rejected, so the check
    // can never become an arbitrary-file hash oracle.
    assert_verification_failure(
        "traversal-oracle",
        |root| {
            let parent = root.parent().unwrap();
            let victim = parent.join("ember-victim.txt");
            std::fs::write(&victim, b"secret").unwrap();
            let victim_hash = crate::v05::manifest::sha256_hex(b"secret");
            let path = root.join("checksums.sha256");
            let mut text = std::fs::read_to_string(&path).unwrap();
            // the victim lives one level above the bundle root
            text.push_str(&format!("{victim_hash}  ../ember-victim.txt\n"));
            std::fs::write(&path, text).unwrap();
        },
        &["checksums"],
    );
}
