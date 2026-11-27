use super::*;

/// Machine-readable strict compatibility result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvCompatibilityReport {
    pub compatible: bool,
    pub exact_same_model: bool,
    pub reasons: Vec<String>,
    pub source: KvSnapshotCompatibilityMetadata,
    pub target: KvCompatibilityTarget,
}

/// Snapshot-side metadata included in a compatibility report (no tensor data).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvSnapshotCompatibilityMetadata {
    pub model_sha256: String,
    pub source_model_sha256: String,
    pub tokenizer_sha256: Option<String>,
    pub architecture: String,
    pub sequence_length: usize,
    pub max_seq: usize,
    pub layer_count: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub precision: KvPrecision,
    pub layout: KvLayout,
    pub rope: KvRopeMetadata,
    pub value_state: String,
    pub execution_mode: String,
    pub execution_fingerprint: String,
    pub origin: KvSnapshotOrigin,
}

/// Validate a manifest against a target without allocating a cache.
pub(super) fn validate_compatibility(
    manifest: &KvSnapshotManifest,
    target: &KvCompatibilityTarget,
) -> KvCompatibilityReport {
    let source = KvSnapshotCompatibilityMetadata {
        model_sha256: manifest.model_sha256.clone(),
        source_model_sha256: manifest.provenance.source_model_sha256.clone(),
        tokenizer_sha256: manifest.tokenizer_sha256.clone(),
        architecture: manifest.architecture.clone(),
        sequence_length: manifest.sequence_length,
        max_seq: manifest.max_seq,
        layer_count: manifest.layer_count,
        n_kv_heads: manifest.n_kv_heads,
        head_dim: manifest.head_dim,
        precision: manifest.precision,
        layout: manifest.layout,
        rope: manifest.rope.clone(),
        value_state: manifest.value_state.clone(),
        execution_mode: manifest.provenance.execution_mode.clone(),
        execution_fingerprint: manifest.provenance.execution_fingerprint.clone(),
        origin: manifest.provenance.origin,
    };
    let mut reasons = Vec::new();
    if let Err(error) = target.validate() {
        reasons.push(format!("invalid target metadata: {error}"));
    }
    compare(
        &mut reasons,
        manifest.model_sha256 == target.model_sha256,
        "model SHA-256 mismatch",
    );
    compare(
        &mut reasons,
        manifest.architecture == target.architecture,
        "architecture mismatch",
    );
    compare(
        &mut reasons,
        manifest.precision == target.precision,
        "KV precision mismatch",
    );
    compare(
        &mut reasons,
        manifest.layout == target.layout,
        "KV layout mismatch",
    );
    compare(
        &mut reasons,
        manifest.layer_count == target.layer_count,
        "layer count mismatch",
    );
    compare(
        &mut reasons,
        manifest.n_kv_heads == target.n_kv_heads,
        "KV head count mismatch",
    );
    compare(
        &mut reasons,
        manifest.head_dim == target.head_dim,
        "head dimension mismatch",
    );
    compare(
        &mut reasons,
        manifest.sequence_length <= target.max_seq,
        "snapshot sequence length exceeds target max_seq",
    );
    compare(
        &mut reasons,
        manifest.rope.layout == target.rope.layout,
        "RoPE layout mismatch",
    );
    compare(
        &mut reasons,
        manifest.rope.dimension_count == target.rope.dimension_count,
        "RoPE dimension count mismatch",
    );
    compare(
        &mut reasons,
        manifest.rope.theta.to_bits() == target.rope.theta.to_bits(),
        "RoPE theta mismatch",
    );
    compare(
        &mut reasons,
        manifest.rope.frequency_layout == target.rope.frequency_layout,
        "RoPE frequency layout mismatch",
    );
    compare(
        &mut reasons,
        frequency_factors_agree(&manifest.rope, &target.rope),
        "RoPE frequency factors mismatch",
    );
    compare(
        &mut reasons,
        manifest.rope.position_origin == target.rope.position_origin,
        "RoPE position origin mismatch",
    );
    compare(
        &mut reasons,
        manifest.rope.keys_state == target.rope.keys_state,
        "stored-key state mismatch",
    );
    compare(
        &mut reasons,
        manifest.rope.qk_norm_order == target.rope.qk_norm_order,
        "QK norm order mismatch",
    );
    compare(
        &mut reasons,
        manifest.rope.has_q_norm == target.rope.has_q_norm,
        "Q norm presence mismatch",
    );
    compare(
        &mut reasons,
        manifest.rope.has_k_norm == target.rope.has_k_norm,
        "K norm presence mismatch",
    );
    compare(
        &mut reasons,
        optional_f32_bits(manifest.rope.qk_norm_epsilon)
            == optional_f32_bits(target.rope.qk_norm_epsilon),
        "QK norm epsilon mismatch",
    );
    compare(
        &mut reasons,
        manifest.value_state == target.value_state,
        "value state mismatch",
    );
    compare(
        &mut reasons,
        manifest.provenance.execution_mode == target.execution_mode,
        "execution mode mismatch",
    );
    compare(
        &mut reasons,
        manifest.provenance.execution_fingerprint == target.execution_fingerprint,
        "execution fingerprint mismatch",
    );
    if manifest.provenance.prefix_token_ids_sha256.is_some() {
        match (&manifest.tokenizer_sha256, &target.tokenizer_sha256) {
            (Some(source_tokenizer), Some(target_tokenizer)) => compare(
                &mut reasons,
                source_tokenizer == target_tokenizer,
                "tokenizer SHA-256 mismatch for tokenized prefix",
            ),
            _ => reasons
                .push("tokenized prefix requires known source and target tokenizer SHA-256".into()),
        }
    } else if let (Some(source_tokenizer), Some(target_tokenizer)) =
        (&manifest.tokenizer_sha256, &target.tokenizer_sha256)
    {
        compare(
            &mut reasons,
            source_tokenizer == target_tokenizer,
            "recorded tokenizer SHA-256 mismatch",
        );
    }
    let exact_same_model = manifest.provenance.origin == KvSnapshotOrigin::Native
        && manifest.provenance.source_model_sha256 == target.model_sha256
        && manifest.model_sha256 == target.model_sha256;
    KvCompatibilityReport {
        compatible: reasons.is_empty(),
        exact_same_model,
        reasons,
        source,
        target: target.clone(),
    }
}

pub(super) fn optional_f32_bits(value: Option<f32>) -> Option<u32> {
    value.map(f32::to_bits)
}

pub(super) fn compare(reasons: &mut Vec<String>, condition: bool, reason: &str) {
    if !condition {
        reasons.push(reason.into());
    }
}

pub(super) fn validate_manifest_metadata(manifest: &KvSnapshotManifest) -> anyhow::Result<()> {
    anyhow::ensure!(
        manifest.schema == KV_SNAPSHOT_SCHEMA,
        "unsupported KV snapshot schema '{}'; expected {KV_SNAPSHOT_SCHEMA}",
        manifest.schema
    );
    anyhow::ensure!(
        manifest.kind == KV_SNAPSHOT_KIND,
        "unexpected KV snapshot kind '{}'; expected {KV_SNAPSHOT_KIND}",
        manifest.kind
    );
    anyhow::ensure!(
        manifest.serialization == KV_SNAPSHOT_SERIALIZATION,
        "unsupported KV snapshot serialization '{}'; expected {KV_SNAPSHOT_SERIALIZATION}",
        manifest.serialization
    );
    validate_sha256("snapshot model", &manifest.model_sha256)?;
    if let Some(tokenizer) = &manifest.tokenizer_sha256 {
        validate_sha256("snapshot tokenizer", tokenizer)?;
    }
    anyhow::ensure!(
        !manifest.architecture.is_empty(),
        "snapshot architecture is empty"
    );
    validate_dimensions(
        manifest.max_seq,
        manifest.layer_count,
        manifest.n_kv_heads,
        manifest.head_dim,
    )?;
    anyhow::ensure!(
        manifest.sequence_length <= manifest.max_seq,
        "snapshot sequence length {} exceeds max_seq {}",
        manifest.sequence_length,
        manifest.max_seq
    );
    validate_rope(&manifest.rope, manifest.head_dim)?;
    anyhow::ensure!(
        manifest.value_state == "projection-output",
        "unsupported snapshot value state '{}'",
        manifest.value_state
    );
    anyhow::ensure!(
        matches!(
            manifest.provenance.execution_mode.as_str(),
            "reference" | "planned" | "planned-fused"
        ),
        "unsupported snapshot execution mode '{}'",
        manifest.provenance.execution_mode
    );
    validate_sha256(
        "snapshot provenance source model",
        &manifest.provenance.source_model_sha256,
    )?;
    validate_sha256(
        "snapshot execution fingerprint",
        &manifest.provenance.execution_fingerprint,
    )?;
    if let Some(plan_hash) = &manifest.provenance.execution_plan_hash {
        validate_sha256("snapshot execution plan", plan_hash)?;
    }
    match manifest.provenance.origin {
        KvSnapshotOrigin::Native => {
            anyhow::ensure!(
                manifest.provenance.source_model_sha256 == manifest.model_sha256,
                "native snapshot source model does not equal compatible model"
            );
            anyhow::ensure!(
                manifest.provenance.transform.is_none(),
                "native snapshot must not carry transform provenance"
            );
        }
        KvSnapshotOrigin::Transformed => {
            let transform = manifest.provenance.transform.as_ref().ok_or_else(|| {
                anyhow::anyhow!("transformed snapshot lacks transform provenance")
            })?;
            anyhow::ensure!(!transform.transform_id.is_empty(), "transform id is empty");
            validate_sha256("transform mapper", &transform.mapper_sha256)?;
            anyhow::ensure!(
                !transform.source_layers.is_empty(),
                "transformed snapshot source-layer selection is empty"
            );
            if let Some(target_layer) = transform.target_layer {
                anyhow::ensure!(
                    target_layer < manifest.layer_count,
                    "transform target layer {target_layer} is outside snapshot layer count {}",
                    manifest.layer_count
                );
            }
            anyhow::ensure!(
                !transform.transformation_type.is_empty(),
                "transformation type is empty"
            );
        }
    }
    match (
        manifest.provenance.prefix_token_count,
        &manifest.provenance.prefix_token_ids_sha256,
    ) {
        (Some(count), Some(hash)) => {
            anyhow::ensure!(
                count == manifest.sequence_length,
                "prefix token count does not equal sequence length"
            );
            anyhow::ensure!(
                manifest.tokenizer_sha256.is_some(),
                "tokenized prefix lacks tokenizer SHA-256"
            );
            validate_sha256("prefix token IDs", hash)?;
        }
        (None, None) => {}
        _ => {
            anyhow::bail!("prefix token count and hash must either both be present or both absent")
        }
    }
    validate_sha256("snapshot identity", &manifest.snapshot_hash)?;
    Ok(())
}

pub(super) fn live_cache_bytes(
    layer_count: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) -> anyhow::Result<u64> {
    let elements = [layer_count, n_kv_heads, head_dim, max_seq]
        .into_iter()
        .try_fold(1u64, |count, dimension| {
            count.checked_mul(u64::try_from(dimension).ok()?)
        })
        .context("live KV cache element count overflow")?;
    let kv_bytes = elements
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_mul(2))
        .context("live K/V byte count overflow")?;
    let scratch_bytes = u64::try_from(max_seq)
        .ok()
        .and_then(|length| length.checked_mul(4))
        .context("live KV scratch byte count overflow")?;
    kv_bytes
        .checked_add(scratch_bytes)
        .context("live KV cache byte count overflow")
}

pub(super) fn validate_dimensions(
    max_seq: usize,
    layer_count: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(max_seq > 0, "KV max_seq must be positive");
    anyhow::ensure!(
        max_seq <= MAX_SEQUENCE_LENGTH,
        "KV max_seq {max_seq} exceeds safety limit {MAX_SEQUENCE_LENGTH}"
    );
    anyhow::ensure!(
        layer_count > 0 && layer_count <= MAX_LAYERS,
        "KV layer count {layer_count} is outside 1..={MAX_LAYERS}"
    );
    anyhow::ensure!(
        n_kv_heads > 0 && n_kv_heads <= MAX_KV_HEADS,
        "KV head count {n_kv_heads} is outside 1..={MAX_KV_HEADS}"
    );
    anyhow::ensure!(
        head_dim > 0 && head_dim <= MAX_HEAD_DIM,
        "KV head dimension {head_dim} is outside 1..={MAX_HEAD_DIM}"
    );
    expected_elements(max_seq, layer_count, n_kv_heads, head_dim)?;
    Ok(())
}

pub(super) fn validate_rope(rope: &KvRopeMetadata, head_dim: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        rope.dimension_count > 0
            && rope.dimension_count <= head_dim
            && rope.dimension_count.is_multiple_of(2),
        "RoPE dimension count {} is invalid for head_dim {head_dim}",
        rope.dimension_count
    );
    anyhow::ensure!(
        rope.theta.is_finite() && rope.theta > 0.0,
        "RoPE theta must be finite and positive"
    );
    anyhow::ensure!(
        rope.frequency_layout == "uniform-theta",
        "unsupported RoPE frequency layout '{}'",
        rope.frequency_layout
    );
    if let Some(factors) = &rope.frequency_factors {
        anyhow::ensure!(
            factors.len() == rope.dimension_count / 2,
            "RoPE frequency factors have {} entries; expected {}",
            factors.len(),
            rope.dimension_count / 2
        );
        anyhow::ensure!(
            factors
                .iter()
                .all(|factor| factor.is_finite() && *factor > 0.0),
            "RoPE frequency factors must be finite and positive"
        );
    }
    anyhow::ensure!(
        rope.position_origin == "absolute-zero-based",
        "unsupported RoPE position origin '{}'",
        rope.position_origin
    );
    anyhow::ensure!(
        rope.keys_state == "post-rope",
        "unsupported stored-key state '{}'",
        rope.keys_state
    );
    if rope.has_q_norm || rope.has_k_norm {
        let epsilon = rope
            .qk_norm_epsilon
            .ok_or_else(|| anyhow::anyhow!("QK norm presence requires an epsilon"))?;
        anyhow::ensure!(
            epsilon.is_finite() && epsilon >= 0.0,
            "QK norm epsilon must be finite and non-negative"
        );
    } else {
        anyhow::ensure!(
            rope.qk_norm_epsilon.is_none(),
            "QK norm epsilon is present but Q/K norms are absent"
        );
    }
    Ok(())
}

pub(super) fn expected_elements(
    sequence_length: usize,
    layer_count: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> anyhow::Result<usize> {
    [layer_count, n_kv_heads, sequence_length, head_dim]
        .into_iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(dimension))
        .ok_or_else(|| anyhow::anyhow!("KV payload shape product overflow"))
}

pub(super) fn payload_descriptor_from_f16(
    file: &str,
    elements: usize,
    values: &[f16],
) -> anyhow::Result<KvPayloadDescriptor> {
    ensure_finite_f16(file, values)?;
    let byte_length = u64::try_from(values.len())?
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("{file} byte length overflow"))?;
    Ok(KvPayloadDescriptor {
        file: file.into(),
        elements,
        byte_length,
        sha256: hash_f16_le(values),
    })
}

pub(super) fn verify_payload_descriptor(
    descriptor: &KvPayloadDescriptor,
    expected_file: &str,
    values: &[f16],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        descriptor.file == expected_file,
        "unexpected payload file '{}'; expected {expected_file}",
        descriptor.file
    );
    anyhow::ensure!(
        descriptor.elements == values.len(),
        "{expected_file} element count mismatch"
    );
    ensure_finite_f16(expected_file, values)?;
    let byte_length = u64::try_from(values.len())?
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("{expected_file} byte length overflow"))?;
    anyhow::ensure!(
        descriptor.byte_length == byte_length,
        "{expected_file} byte length mismatch"
    );
    let computed = hash_f16_le(values);
    anyhow::ensure!(
        descriptor.sha256 == computed,
        "{expected_file} SHA-256 mismatch: recorded {}, computed {computed}",
        descriptor.sha256
    );
    Ok(())
}
