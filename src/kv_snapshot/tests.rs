use super::*;

fn target(max_seq: usize) -> KvCompatibilityTarget {
    target_with_head_dim(max_seq, 4)
}

/// A two-layer, two-KV-head llama compatibility target; also the fixture
/// for `kv_compare` tests.
pub(crate) fn target_with_head_dim(max_seq: usize, head_dim: usize) -> KvCompatibilityTarget {
    KvCompatibilityTarget {
        model_sha256: "aa".repeat(32),
        tokenizer_sha256: Some("bb".repeat(32)),
        architecture: "llama".into(),
        max_seq,
        layer_count: 2,
        n_kv_heads: 2,
        head_dim,
        precision: KvPrecision::F16,
        layout: KvLayout::LayerHeadPositionDimensionCompact,
        rope: KvRopeMetadata {
            layout: KvRopeLayout::AdjacentPair,
            dimension_count: head_dim,
            theta: 10_000.0,
            frequency_layout: "uniform-theta".into(),
            frequency_factors: None,
            position_origin: "absolute-zero-based".into(),
            keys_state: "post-rope".into(),
            qk_norm_order: KvQkNormOrder::AfterRope,
            has_q_norm: false,
            has_k_norm: false,
            qk_norm_epsilon: None,
        },
        value_state: "projection-output".into(),
        execution_mode: "planned".into(),
        execution_fingerprint: "dd".repeat(32),
        plan_hash: Some("cc".repeat(32)),
    }
}

fn snapshot(sequence: usize, max_seq: usize) -> KvSnapshot {
    let mut cache = KVCache::new(2, 2, 4, max_seq);
    for position in 0..sequence {
        let keys: Vec<f32> = (0..8)
            .map(|index| position as f32 + index as f32 / 16.0)
            .collect();
        let values: Vec<f32> = (0..8)
            .map(|index| -(position as f32) - index as f32 / 32.0)
            .collect();
        for layer in 0..2 {
            cache.append(layer, position, &keys, &values);
        }
        cache.advance_cursor();
    }
    let tokens: Vec<u32> = (0..sequence as u32).collect();
    KvSnapshot::export_native(&cache, target(max_seq), Some(&tokens), Some(7)).unwrap()
}

fn temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ember-kv-snapshot-{label}-{}-{}",
        std::process::id(),
        STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn execution_fingerprint_ignores_capacity_but_not_execution() {
    use crate::plan::testkit::sample_plan;
    use crate::plan::{ExecutionMode, HookMode};

    let small = sample_plan(ExecutionMode::Planned, HookMode::Disabled).finalize();
    let mut large = small.clone();
    large.kv.max_seq *= 2;
    large.kv.layer_stride *= 2;
    large.kv.head_stride *= 2;
    large.scratch.total_bytes *= 2;
    large.gguf.context_length *= 2;
    large = large.finalize();
    let small_target = KvCompatibilityTarget::from_execution_plan(&small).unwrap();
    let large_target = KvCompatibilityTarget::from_execution_plan(&large).unwrap();
    assert_eq!(
        small_target.execution_fingerprint,
        large_target.execution_fingerprint
    );
    assert_ne!(small_target.plan_hash, large_target.plan_hash);

    let reference = sample_plan(ExecutionMode::Reference, HookMode::Disabled).finalize();
    let reference_target = KvCompatibilityTarget::from_execution_plan(&reference).unwrap();
    assert_ne!(
        small_target.execution_fingerprint,
        reference_target.execution_fingerprint
    );
}

#[test]
fn compact_round_trip_is_bit_exact_and_cursor_is_restored() {
    let snapshot = snapshot(3, 5);
    let imported = snapshot.import_cache(&target(7)).unwrap();
    assert_eq!(imported.cursor(), 3);
    assert_eq!(imported.max_seq_len(), 7);
    let exported =
        KvSnapshot::export_native(&imported, target(7), Some(&[0, 1, 2]), Some(7)).unwrap();
    assert_eq!(snapshot.keys(), exported.keys());
    assert_eq!(snapshot.values(), exported.values());
}

#[test]
fn serialization_and_hash_are_deterministic() {
    let a = snapshot(3, 5);
    let b = snapshot(3, 5);
    assert_eq!(a.manifest(), b.manifest());
    assert_eq!(a.manifest().snapshot_hash, b.manifest().snapshot_hash);
    assert_eq!(a.manifest().keys.sha256, b.manifest().keys.sha256);
    assert_eq!(a.manifest().values.sha256, b.manifest().values.sha256);
    assert_eq!(
        serde_json::to_vec_pretty(a.manifest()).unwrap(),
        serde_json::to_vec_pretty(b.manifest()).unwrap()
    );
}

#[test]
fn manifest_duplicate_keys_are_rejected() {
    let manifest = snapshot(1, 2).manifest().clone();
    let mut bytes = serde_json::to_vec(&manifest).unwrap();
    assert_eq!(bytes.pop(), Some(b'}'));
    bytes.extend_from_slice(br#","schema":"ember.kv-snapshot.v1"}"#);
    let error = KvSnapshotManifest::from_json_bytes(&bytes).unwrap_err();
    assert!(
        error.to_string().contains("duplicate object key"),
        "{error}"
    );
}

#[test]
fn directory_round_trip_and_integrity_verification() {
    let root = temp_dir("roundtrip");
    let original = snapshot(3, 5);
    original.save_dir(&root, false).unwrap();
    let manifest_bytes = std::fs::read(root.join(KV_MANIFEST_FILE)).unwrap();
    let parsed = KvSnapshotManifest::from_json_bytes(&manifest_bytes).unwrap();
    assert_eq!(&parsed, original.manifest());
    let loaded = KvSnapshot::load_dir(&root).unwrap();
    assert_eq!(original.manifest(), loaded.manifest());
    assert_eq!(original.keys(), loaded.keys());
    assert_eq!(original.values(), loaded.values());
    assert_eq!(
        KvSnapshot::verify_dir(&root).unwrap(),
        original.manifest().snapshot_hash
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn truncated_corrupted_and_extra_payloads_fail() {
    for mode in ["truncated", "corrupted", "extra"] {
        let root = temp_dir(mode);
        snapshot(3, 5).save_dir(&root, false).unwrap();
        let path = root.join(KV_KEY_FILE);
        let mut bytes = std::fs::read(&path).unwrap();
        match mode {
            "truncated" => {
                bytes.pop();
            }
            "corrupted" => bytes[0] ^= 0x80,
            "extra" => bytes.push(0),
            _ => unreachable!(),
        }
        std::fs::write(&path, bytes).unwrap();
        let error = KvSnapshot::load_dir(&root).err().unwrap().to_string();
        assert!(
            error.contains("length") || error.contains("SHA-256"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn unexpected_file_is_rejected() {
    let root = temp_dir("extra-file");
    snapshot(1, 2).save_dir(&root, false).unwrap();
    std::fs::write(root.join("../not-inside"), b"unrelated").unwrap();
    std::fs::write(root.join("extra.bin"), b"x").unwrap();
    assert!(KvSnapshot::load_dir(&root).is_err());
    std::fs::remove_file(root.join("../not-inside")).ok();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn malformed_and_overflowing_metadata_fail_before_payload_read() {
    let root = temp_dir("malformed");
    snapshot(1, 2).save_dir(&root, false).unwrap();
    let path = root.join(KV_MANIFEST_FILE);
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["layer_count"] = serde_json::json!(usize::MAX);
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let error = KvSnapshot::load_dir(&root).err().unwrap().to_string();
    assert!(error.contains("layer count") || error.contains("overflow"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn streamed_f16_payload_read_and_hash_match_serialized_bytes() {
    let root = temp_dir("streamed-payload");
    std::fs::create_dir(&root).unwrap();
    let values: Vec<f16> = (0..(PAYLOAD_READ_CHUNK_BYTES / 2 + 1))
        .map(|index| f16::from_f32(index as f32 / 16.0 - 100.0))
        .collect();
    let bytes = f16_to_le_bytes(&values);
    let path = root.join(KV_KEY_FILE);
    std::fs::write(&path, &bytes).unwrap();

    let loaded = read_f16_file(&path, bytes.len() as u64).unwrap();
    assert_eq!(loaded, values);
    assert_eq!(hash_f16_le(&loaded), sha256_hex(&bytes));

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn allocation_limit_is_enforced() {
    let root = temp_dir("allocation-limit");
    snapshot(3, 5).save_dir(&root, false).unwrap();
    assert!(KvSnapshot::load_dir_with_limit(&root, 1).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn destination_live_cache_limit_is_enforced_before_allocation() {
    let snapshot = snapshot(1, 2);
    let candidate = target(100);
    let live_bytes = candidate.live_cache_bytes().unwrap();
    assert_eq!(live_bytes, 6_800);
    assert!(snapshot
        .import_cache_with_limit(&candidate, live_bytes - 1)
        .is_err());
    assert_eq!(
        snapshot
            .import_cache_with_limit(&candidate, live_bytes)
            .unwrap()
            .max_seq_len(),
        100
    );
}

#[test]
fn overwrite_only_replaces_a_verified_snapshot() {
    let root = temp_dir("overwrite-safe");
    std::fs::create_dir(&root).unwrap();
    let marker = root.join("keep.txt");
    std::fs::write(&marker, b"not a snapshot").unwrap();
    assert!(snapshot(1, 2).save_dir(&root, true).is_err());
    assert_eq!(std::fs::read(&marker).unwrap(), b"not a snapshot");
    std::fs::remove_dir_all(&root).unwrap();

    snapshot(1, 2).save_dir(&root, false).unwrap();
    let replacement = snapshot(2, 3);
    replacement.save_dir(&root, true).unwrap();
    assert_eq!(
        KvSnapshot::load_dir(&root).unwrap().manifest(),
        replacement.manifest()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn normalize_path_collapses_dots_and_uses_cwd_for_relative() {
    let cwd = std::env::current_dir().unwrap();
    // Relative path with a self-dot collapses cleanly.
    assert_eq!(
        normalize_path(Path::new("foo/./bar")).unwrap(),
        cwd.join("foo/bar")
    );
    // Parent traversal collapses without touching the filesystem.
    assert_eq!(
        normalize_path(Path::new("foo/../baz")).unwrap(),
        cwd.join("baz")
    );
}

/// Adversarial path-normalization cases. The security contract is that
/// `normalize_path` collapses traversal so the downstream
/// `reject_dangerous_snapshot_output` / `ensure_outputs_outside_snapshot`
/// prefix checks are meaningful. These cases probe the boundaries.
#[test]
fn normalize_path_adversarial_traversal_is_collapsed() {
    let cwd = std::env::current_dir().unwrap();
    // Surplus `..` walk the path back toward (but never past) root. The
    // result is a clean absolute path with no `.`/`..` remnants — so the
    // downstream prefix check against the canonical cwd is meaningful.
    let escaped = format!("{}/a/b/c/../../../../../etc/passwd", cwd.display());
    let normalized = normalize_path(Path::new(&escaped)).unwrap();
    assert!(!normalized.to_string_lossy().contains(".."));
    assert!(!normalized.to_string_lossy().contains("/./"));
    // A non-existent deep base normalizes identically to a real one — the
    // function never touches the filesystem.
    let deep = format!("{}/a/b/c", cwd.display());
    assert_eq!(
        normalize_path(Path::new(&deep)).unwrap(),
        normalize_path(Path::new(&format!("{}/a/b/c/./", cwd.display()))).unwrap()
    );
    // Self-dot is a no-op.
    assert_eq!(
        normalize_path(Path::new(&format!("{}/a/b/c", cwd.display()))).unwrap(),
        normalize_path(Path::new(&format!("{}/a/./b/./c", cwd.display()))).unwrap()
    );
}

#[test]
fn normalize_path_rejects_cwd_and_ancestor_outputs() {
    // The whole point of the normalization: after collapse, the dangerous
    // output check must see the cwd (or an ancestor) and refuse.
    assert!(reject_dangerous_snapshot_output(Path::new(".")).is_err());
    assert!(reject_dangerous_snapshot_output(Path::new("./")).is_err());
    assert!(reject_dangerous_snapshot_output(Path::new("././.")).is_err());
    // A path that traverses up and back to cwd still resolves to cwd.
    assert!(reject_dangerous_snapshot_output(Path::new("foo/..")).is_err());
    // A path that tries to escape then land on an ancestor of cwd.
    let cwd = std::env::current_dir().unwrap();
    let parent = cwd.parent().unwrap();
    let via_traversal = format!("{}/sub/..", parent.display());
    assert!(reject_dangerous_snapshot_output(Path::new(&via_traversal)).is_err());
}

#[test]
fn normalize_path_accepts_non_ancestor_outputs() {
    // Sibling and descendant-of-cwd-but-not-cwd paths are allowed.
    assert!(reject_dangerous_snapshot_output(Path::new("snapshots/out")).is_ok());
    assert!(reject_dangerous_snapshot_output(Path::new("./artifacts/x")).is_ok());
}

fn incompatible(mutator: impl FnOnce(&mut KvCompatibilityTarget), reason: &str) {
    let snapshot = snapshot(3, 5);
    let mut candidate = target(5);
    mutator(&mut candidate);
    let report = snapshot.compatibility_report(&candidate);
    assert!(!report.compatible);
    assert!(
        report.reasons.iter().any(|value| value.contains(reason)),
        "missing '{reason}' in {:?}",
        report.reasons
    );
    assert!(snapshot.import_cache(&candidate).is_err());
}

#[test]
fn strict_compatibility_rejects_every_semantic_mismatch() {
    incompatible(|target| target.model_sha256 = "dd".repeat(32), "model SHA");
    incompatible(
        |target| target.architecture = "qwen2".into(),
        "architecture",
    );
    incompatible(|target| target.head_dim = 6, "head dimension");
    incompatible(|target| target.n_kv_heads = 1, "KV head count");
    incompatible(|target| target.layer_count = 3, "layer count");
    incompatible(|target| target.max_seq = 2, "exceeds target");
    incompatible(
        |target| target.rope.layout = KvRopeLayout::SplitHalf,
        "RoPE layout",
    );
    incompatible(
        |target| target.rope.qk_norm_order = KvQkNormOrder::BeforeRope,
        "QK norm order",
    );
    incompatible(
        |target| target.tokenizer_sha256 = Some("dd".repeat(32)),
        "tokenizer",
    );
    incompatible(
        |target| target.execution_mode = "reference".into(),
        "execution mode",
    );
}

#[test]
fn empty_prefix_and_maximum_boundary_are_supported() {
    let empty = snapshot(0, 3);
    assert!(empty.keys().is_empty());
    assert_eq!(empty.import_cache(&target(3)).unwrap().cursor(), 0);

    let full = snapshot(3, 3);
    let cache = full.import_cache(&target(3)).unwrap();
    assert_eq!(cache.cursor(), cache.max_seq_len());
}

#[test]
fn manifest_tampering_fails_identity_verification() {
    let mut snapshot = snapshot(1, 2);
    snapshot.manifest.provenance.resume_token_id = Some(123);
    assert!(snapshot.verify().is_err());
}

#[test]
fn token_id_hash_is_domain_separated_and_little_endian() {
    // The prefix-token hash recorded by `ember kv` traces: a pinned
    // golden value, so any change to the encoding is caught here.
    assert_eq!(
        hash_token_ids(&[1, 2, u32::MAX]),
        "7ba3fbe5e313572a9a6ee56956380b6a07a48019956aaf835c5d441babe7924e"
    );
    // Empty prefix hashes the separator alone.
    let empty = hash_token_ids(&[]);
    assert_ne!(empty, hash_token_ids(&[0]));
    assert_eq!(empty.len(), 64);
}

#[test]
fn exported_native_prefix_hash_matches_token_id_hasher() {
    // End-to-end: the hash recorded into provenance by export_native must
    // equal the standalone hasher applied to the same token IDs.
    let tokens: Vec<u32> = (0..5).collect();
    assert_eq!(
        snapshot(5, 8).manifest().provenance.prefix_token_ids_sha256,
        Some(hash_token_ids(&tokens))
    );
}

#[cfg(test)]
mod llama_model_tests {
    use crate::backend::CpuBackend;
    use crate::model::ForwardModel;
    use crate::plan::{ExecutionMode, HookMode};

    #[test]
    fn kv_snapshot_same_model_replay_is_bit_exact() {
        use crate::kv_snapshot::{KvCompatibilityTarget, KvSnapshot};

        let model = crate::llama::testkit::q8_llama_without_fast_decode(2);
        model.set_execution_mode(ExecutionMode::Planned);
        let backend = CpuBackend;
        let capacity = 8;
        let prompt = [3u32, 1, 7];
        let model_hash = "aa".repeat(32);
        let tokenizer_hash = "bb".repeat(32);
        let plan = model
            .execution_plan(
                ExecutionMode::Planned,
                HookMode::Disabled,
                &[],
                capacity,
                Some(&model_hash),
                Some(&tokenizer_hash),
            )
            .unwrap();
        let target = KvCompatibilityTarget::from_execution_plan(&plan).unwrap();

        // A: uninterrupted native path.
        let mut native_cache = model.create_cache(&backend, capacity);
        let mut native_logits = ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &prompt,
            &mut native_cache,
            0,
        )
        .unwrap();

        // B: independent prefill, deterministic disk artifact, original cache
        // dropped, then import into fresh owned storage.
        let mut replay_source = model.create_cache(&backend, capacity);
        let replay_prefill_logits = ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &prompt,
            &mut replay_source,
            0,
        )
        .unwrap();
        assert_eq!(native_logits.data(), replay_prefill_logits.data());
        let resume_token = crate::sampler::argmax_token(replay_prefill_logits.data()) as u32;
        let exported = KvSnapshot::export_native(
            &replay_source,
            target.clone(),
            Some(&prompt),
            Some(resume_token),
        )
        .unwrap();
        let snapshot_hash = exported.manifest().snapshot_hash.clone();
        let root = std::env::temp_dir().join(format!(
            "ember-kv-replay-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        exported.save_dir(&root, false).unwrap();
        drop(replay_source);
        let loaded = KvSnapshot::load_dir(&root).unwrap();
        assert_eq!(loaded.manifest().snapshot_hash, snapshot_hash);
        let mut replay_cache = loaded.import_cache(&target).unwrap();
        assert_eq!(replay_cache.cursor(), prompt.len());
        assert_eq!(native_cache.cursor(), replay_cache.cursor());
        let imported_export =
            KvSnapshot::export_native(&replay_cache, target, Some(&prompt), Some(resume_token))
                .unwrap();
        assert_eq!(loaded.keys(), imported_export.keys());
        assert_eq!(loaded.values(), imported_export.values());
        assert_eq!(
            loaded.manifest().snapshot_hash,
            imported_export.manifest().snapshot_hash
        );

        let mut replay_logits = replay_prefill_logits;
        let mut native_tokens = Vec::new();
        let mut replay_tokens = Vec::new();
        const CONTINUATION: usize = 4;
        for step in 0..CONTINUATION {
            assert_eq!(
                native_logits.data(),
                replay_logits.data(),
                "replayed logits differ at continuation step {step}"
            );
            let native_token = crate::sampler::argmax_token(native_logits.data()) as u32;
            let replay_token = crate::sampler::argmax_token(replay_logits.data()) as u32;
            native_tokens.push(native_token);
            replay_tokens.push(replay_token);
            assert_eq!(native_token, replay_token);
            if step + 1 < CONTINUATION {
                let native_start = native_cache.cursor();
                let replay_start = replay_cache.cursor();
                assert_eq!(native_start, replay_start);
                native_logits = ForwardModel::forward_last_logits_with_cache(
                    &model,
                    &backend,
                    &[native_token],
                    &mut native_cache,
                    native_start,
                )
                .unwrap();
                replay_logits = ForwardModel::forward_last_logits_with_cache(
                    &model,
                    &backend,
                    &[replay_token],
                    &mut replay_cache,
                    replay_start,
                )
                .unwrap();
            }
        }
        assert_eq!(native_tokens, replay_tokens);
        assert_eq!(native_cache.cursor(), replay_cache.cursor());

        // Snapshot import must not contaminate subsequent ordinary planned
        // decode with checksum/metadata work or per-token allocations. The
        // only permitted allocations are the existing logits CpuTensor's
        // shape, strides, and data vectors (Gate E's bound of three).
        let next_input = *replay_tokens.last().unwrap();
        let replay_start = replay_cache.cursor();
        assert!(crate::alloc_counter::counting_active());
        let (result, allocations) = crate::alloc_counter::count_allocations(|| {
            ForwardModel::forward_last_logits_with_cache(
                &model,
                &backend,
                &[next_input],
                &mut replay_cache,
                replay_start,
            )
        });
        result.unwrap();
        assert!(
            allocations <= 3,
            "planned decode after snapshot import allocated {allocations} times"
        );
        assert!(model.has_planned_session(), "planned path did not run");
        std::fs::remove_dir_all(root).unwrap();
    }
}
