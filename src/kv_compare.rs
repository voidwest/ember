//! Deterministic comparison and in-memory diagnostic perturbation of KV state.
//!
//! This module is a measuring instrument. It does not fit, load, execute, or
//! persist a learned cross-model mapper. Diagnostic perturbations never claim
//! to be `ember.kv-snapshot.v1` artifacts.

use crate::extraction::sha256_bytes;
use crate::kv_cache::KVCache;
use crate::kv_snapshot::{KvCompatibilityTarget, KvSnapshot, KvSnapshotOrigin};
use anyhow::Context;
use half::f16;
use serde::{Deserialize, Serialize};

pub const KV_COMPARISON_SCHEMA: &str = "ember.kv-compare.v1";
pub const KV_DIAGNOSTIC_PERTURBATION_SCHEMA: &str = "ember.kv-diagnostic-perturb.v1";

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvComparisonOptions {
    pub include_r2: bool,
    pub max_abs_threshold: Option<f64>,
    pub mse_threshold: Option<f64>,
    pub cosine_threshold: Option<f64>,
    pub r2_threshold: Option<f64>,
}

impl KvComparisonOptions {
    fn validate(self) -> anyhow::Result<Self> {
        if let Some(value) = self.max_abs_threshold {
            anyhow::ensure!(
                value.is_finite() && value >= 0.0,
                "max-absolute-error threshold must be finite and non-negative"
            );
        }
        if let Some(value) = self.mse_threshold {
            anyhow::ensure!(
                value.is_finite() && value >= 0.0,
                "MSE threshold must be finite and non-negative"
            );
        }
        if let Some(value) = self.cosine_threshold {
            anyhow::ensure!(
                value.is_finite() && (-1.0..=1.0).contains(&value),
                "cosine threshold must be finite and within [-1, 1]"
            );
        }
        if let Some(value) = self.r2_threshold {
            anyhow::ensure!(
                value.is_finite() && value <= 1.0,
                "R2 threshold must be finite and at most 1"
            );
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvVectorMetrics {
    pub element_count: usize,
    pub cosine_similarity: Option<f64>,
    pub mse: f64,
    pub r2: Option<f64>,
    pub max_abs_error: f64,
    pub bit_mismatch_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvLayerHeadComparison {
    pub layer: usize,
    pub head: usize,
    pub keys: KvVectorMetrics,
    pub values: KvVectorMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvThresholdExceedance {
    pub layer: usize,
    pub head: usize,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvPerturbComponent {
    Keys,
    Values,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum KvPerturbOperation {
    Zero,
    Scale { factor: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvDiagnosticPerturbation {
    pub layer: usize,
    pub head: usize,
    pub component: KvPerturbComponent,
    pub operation: KvPerturbOperation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvDiagnosticPerturbationReceipt {
    pub schema: String,
    pub diagnostic_id: String,
    pub source_snapshot_hash: String,
    pub perturbation: KvDiagnosticPerturbation,
    pub prefix_positions_affected: usize,
    pub key_elements_affected: usize,
    pub value_elements_affected: usize,
    pub scale_factor_bits: Option<u32>,
    pub arithmetic: String,
}

/// Owned, in-memory candidate state for diagnostics. It is deliberately not a
/// serializable snapshot and cannot be passed to ordinary replay admission.
#[derive(Debug)]
pub struct KvDiagnosticAlteration {
    receipt: KvDiagnosticPerturbationReceipt,
    keys: Vec<f16>,
    values: Vec<f16>,
}

impl KvDiagnosticAlteration {
    #[must_use]
    pub fn receipt(&self) -> &KvDiagnosticPerturbationReceipt {
        &self.receipt
    }

    #[must_use]
    pub fn keys(&self) -> &[f16] {
        &self.keys
    }

    #[must_use]
    pub fn values(&self) -> &[f16] {
        &self.values
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvSnapshotComparison {
    pub schema: String,
    pub kind: String,
    pub metric_contract: String,
    pub status: String,
    pub reference_snapshot_hash: String,
    pub candidate_identity: String,
    pub candidate_snapshot_hash: Option<String>,
    pub diagnostic_perturbation: Option<KvDiagnosticPerturbationReceipt>,
    pub model_sha256: String,
    pub tokenizer_sha256: Option<String>,
    pub prefix_token_ids_sha256: Option<String>,
    pub sequence_length: usize,
    pub reference_capacity: usize,
    pub candidate_capacity: usize,
    pub layer_count: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub reference_origin: KvSnapshotOrigin,
    pub candidate_origin: Option<KvSnapshotOrigin>,
    pub snapshot_hash_equal: bool,
    pub payload_bit_exact: bool,
    pub resume_token_ids_match: bool,
    pub options: KvComparisonOptions,
    pub keys_global: KvVectorMetrics,
    pub values_global: KvVectorMetrics,
    pub per_layer_head: Vec<KvLayerHeadComparison>,
    pub thresholds_evaluated: bool,
    pub thresholds_passed: bool,
    pub threshold_exceedances: Vec<KvThresholdExceedance>,
    pub first_threshold_exceedance: Option<KvThresholdExceedance>,
}

/// Compare two verified snapshots in the same target/prefix coordinate system.
/// Capacity, source model, and native/transformed origin may differ; target
/// execution and prefix alignment may not.
pub fn compare_snapshots(
    reference: &KvSnapshot,
    candidate: &KvSnapshot,
    options: KvComparisonOptions,
) -> anyhow::Result<KvSnapshotComparison> {
    reference.verify()?;
    candidate.verify()?;
    ensure_snapshots_aligned(reference, candidate)?;
    compare_payloads(
        reference,
        candidate.keys(),
        candidate.values(),
        candidate.manifest().max_seq,
        candidate.manifest().snapshot_hash.clone(),
        Some(candidate.manifest().snapshot_hash.clone()),
        None,
        Some(candidate.manifest().provenance.origin),
        reference.manifest().provenance.resume_token_id
            == candidate.manifest().provenance.resume_token_id,
        options,
    )
}

/// Compare a native snapshot with a controlled, in-memory diagnostic edit.
pub fn compare_snapshot_to_diagnostic(
    reference: &KvSnapshot,
    candidate: &KvDiagnosticAlteration,
    options: KvComparisonOptions,
) -> anyhow::Result<KvSnapshotComparison> {
    reference.verify()?;
    anyhow::ensure!(
        candidate.receipt.source_snapshot_hash == reference.manifest().snapshot_hash,
        "diagnostic candidate was derived from a different source snapshot"
    );
    compare_payloads(
        reference,
        candidate.keys(),
        candidate.values(),
        reference.manifest().max_seq,
        candidate.receipt.diagnostic_id.clone(),
        None,
        Some(candidate.receipt.clone()),
        None,
        true,
        options,
    )
}

#[allow(clippy::too_many_arguments)]
fn compare_payloads(
    reference: &KvSnapshot,
    candidate_keys: &[f16],
    candidate_values: &[f16],
    candidate_capacity: usize,
    candidate_identity: String,
    candidate_snapshot_hash: Option<String>,
    diagnostic_perturbation: Option<KvDiagnosticPerturbationReceipt>,
    candidate_origin: Option<KvSnapshotOrigin>,
    resume_token_ids_match: bool,
    options: KvComparisonOptions,
) -> anyhow::Result<KvSnapshotComparison> {
    let mut options = options.validate()?;
    if options.r2_threshold.is_some() {
        options.include_r2 = true;
    }
    anyhow::ensure!(
        reference.manifest().sequence_length > 0,
        "cannot compare empty KV prefixes"
    );
    anyhow::ensure!(
        reference.keys().len() == candidate_keys.len()
            && reference.values().len() == candidate_values.len(),
        "candidate compact payload lengths differ from the reference"
    );
    let manifest = reference.manifest();
    let keys_global = vector_metrics(reference.keys(), candidate_keys, options.include_r2)
        .context("cannot compare global K payload")?;
    let values_global = vector_metrics(reference.values(), candidate_values, options.include_r2)
        .context("cannot compare global V payload")?;
    let payload_bit_exact =
        keys_global.bit_mismatch_count == 0 && values_global.bit_mismatch_count == 0;

    let per_head_elements = manifest
        .sequence_length
        .checked_mul(manifest.head_dim)
        .context("per-head comparison length overflow")?;
    let report_count = manifest
        .layer_count
        .checked_mul(manifest.n_kv_heads)
        .context("layer/head comparison count overflow")?;
    anyhow::ensure!(
        report_count <= 1_000_000,
        "layer/head comparison report would contain {report_count} rows; limit is 1000000"
    );
    let mut per_layer_head = Vec::new();
    per_layer_head
        .try_reserve_exact(report_count)
        .context("cannot allocate layer/head comparison report")?;
    let mut threshold_exceedances = Vec::new();
    for layer in 0..manifest.layer_count {
        for head in 0..manifest.n_kv_heads {
            let ordinal = layer
                .checked_mul(manifest.n_kv_heads)
                .and_then(|value| value.checked_add(head))
                .context("layer/head comparison index overflow")?;
            let start = ordinal
                .checked_mul(per_head_elements)
                .context("layer/head payload offset overflow")?;
            let end = start
                .checked_add(per_head_elements)
                .context("layer/head payload end overflow")?;
            let keys = vector_metrics(
                &reference.keys()[start..end],
                &candidate_keys[start..end],
                options.include_r2,
            )
            .with_context(|| format!("cannot compare K at layer {layer}, head {head}"))?;
            let values = vector_metrics(
                &reference.values()[start..end],
                &candidate_values[start..end],
                options.include_r2,
            )
            .with_context(|| format!("cannot compare V at layer {layer}, head {head}"))?;
            let reasons = threshold_reasons(&keys, &values, options);
            if !reasons.is_empty() {
                threshold_exceedances.push(KvThresholdExceedance {
                    layer,
                    head,
                    reasons,
                });
            }
            per_layer_head.push(KvLayerHeadComparison {
                layer,
                head,
                keys,
                values,
            });
        }
    }

    let snapshot_hash_equal =
        candidate_snapshot_hash.as_deref() == Some(manifest.snapshot_hash.as_str());
    let status = if snapshot_hash_equal {
        "identical"
    } else if payload_bit_exact {
        "payload-identical"
    } else {
        "differs"
    };
    Ok(KvSnapshotComparison {
        schema: KV_COMPARISON_SCHEMA.into(),
        kind: "kv-snapshot-comparison".into(),
        metric_contract: "ember.kv-vector-metrics.v1".into(),
        status: status.into(),
        reference_snapshot_hash: manifest.snapshot_hash.clone(),
        snapshot_hash_equal,
        candidate_identity,
        candidate_snapshot_hash,
        diagnostic_perturbation,
        model_sha256: manifest.model_sha256.clone(),
        tokenizer_sha256: manifest.tokenizer_sha256.clone(),
        prefix_token_ids_sha256: manifest.provenance.prefix_token_ids_sha256.clone(),
        sequence_length: manifest.sequence_length,
        reference_capacity: manifest.max_seq,
        candidate_capacity,
        layer_count: manifest.layer_count,
        n_kv_heads: manifest.n_kv_heads,
        head_dim: manifest.head_dim,
        reference_origin: manifest.provenance.origin,
        candidate_origin,
        payload_bit_exact,
        resume_token_ids_match,
        options,
        keys_global,
        values_global,
        per_layer_head,
        thresholds_evaluated: options.max_abs_threshold.is_some()
            || options.mse_threshold.is_some()
            || options.cosine_threshold.is_some()
            || options.r2_threshold.is_some(),
        thresholds_passed: threshold_exceedances.is_empty(),
        first_threshold_exceedance: threshold_exceedances.first().cloned(),
        threshold_exceedances,
    })
}

pub fn ensure_snapshots_aligned(
    reference: &KvSnapshot,
    candidate: &KvSnapshot,
) -> anyhow::Result<()> {
    let left = reference.manifest();
    let right = candidate.manifest();
    let mut reasons = Vec::new();
    macro_rules! same {
        ($left:expr_2021, $right:expr_2021, $label:literal) => {
            if $left != $right {
                reasons.push($label);
            }
        };
    }
    same!(&left.model_sha256, &right.model_sha256, "model SHA-256");
    same!(
        &left.tokenizer_sha256,
        &right.tokenizer_sha256,
        "tokenizer SHA-256"
    );
    same!(&left.architecture, &right.architecture, "architecture");
    same!(
        left.sequence_length,
        right.sequence_length,
        "sequence length"
    );
    same!(left.layer_count, right.layer_count, "layer count");
    same!(left.n_kv_heads, right.n_kv_heads, "KV head count");
    same!(left.head_dim, right.head_dim, "head dimension");
    same!(left.precision, right.precision, "precision");
    same!(left.layout, right.layout, "layout");
    if !rope_semantics_equal(&left.rope, &right.rope) {
        reasons.push("RoPE/QK semantics");
    }
    same!(&left.value_state, &right.value_state, "value state");
    same!(
        &left.provenance.execution_mode,
        &right.provenance.execution_mode,
        "execution mode"
    );
    same!(
        &left.provenance.execution_fingerprint,
        &right.provenance.execution_fingerprint,
        "execution fingerprint"
    );
    same!(
        left.provenance.prefix_token_count,
        right.provenance.prefix_token_count,
        "prefix token count"
    );
    same!(
        &left.provenance.prefix_token_ids_sha256,
        &right.provenance.prefix_token_ids_sha256,
        "prefix token-ID SHA-256"
    );
    if left.sequence_length > 0
        && (left.provenance.prefix_token_ids_sha256.is_none()
            || right.provenance.prefix_token_ids_sha256.is_none())
    {
        reasons.push("prefix token alignment is unproven");
    }
    anyhow::ensure!(
        reasons.is_empty(),
        "snapshots are not aligned in one target/prefix coordinate system: {}",
        reasons.join(", ")
    );
    anyhow::ensure!(
        reference.keys().len() == candidate.keys().len()
            && reference.values().len() == candidate.values().len(),
        "aligned snapshot payload lengths differ"
    );
    Ok(())
}

fn rope_semantics_equal(
    left: &crate::kv_snapshot::KvRopeMetadata,
    right: &crate::kv_snapshot::KvRopeMetadata,
) -> bool {
    left.layout == right.layout
        && left.dimension_count == right.dimension_count
        && left.theta.to_bits() == right.theta.to_bits()
        && left.frequency_layout == right.frequency_layout
        && left.position_origin == right.position_origin
        && left.keys_state == right.keys_state
        && left.qk_norm_order == right.qk_norm_order
        && left.has_q_norm == right.has_q_norm
        && left.has_k_norm == right.has_k_norm
        && left.qk_norm_epsilon.map(f32::to_bits) == right.qk_norm_epsilon.map(f32::to_bits)
}

fn vector_metrics(
    reference: &[f16],
    candidate: &[f16],
    include_r2: bool,
) -> anyhow::Result<KvVectorMetrics> {
    anyhow::ensure!(
        reference.len() == candidate.len(),
        "vector lengths differ ({} versus {})",
        reference.len(),
        candidate.len()
    );
    if reference.is_empty() {
        return Ok(KvVectorMetrics {
            element_count: 0,
            cosine_similarity: None,
            mse: 0.0,
            r2: None,
            max_abs_error: 0.0,
            bit_mismatch_count: 0,
        });
    }
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut max_abs_error = 0.0f64;
    let mut reference_sum = 0.0f64;
    let mut bit_mismatch_count = 0usize;
    for (index, (left, right)) in reference.iter().zip(candidate).enumerate() {
        let left_value = f64::from(left.to_f32());
        let right_value = f64::from(right.to_f32());
        anyhow::ensure!(
            left_value.is_finite() && right_value.is_finite(),
            "non-finite f16 value at vector index {index}"
        );
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
        let difference = left_value - right_value;
        squared_error += difference * difference;
        max_abs_error = max_abs_error.max(difference.abs());
        reference_sum += left_value;
        bit_mismatch_count += usize::from(left.to_bits() != right.to_bits());
    }
    let cosine_similarity = if squared_error == 0.0 {
        Some(1.0)
    } else if left_norm == 0.0 || right_norm == 0.0 {
        None
    } else {
        Some((dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(-1.0, 1.0))
    };
    let mse = squared_error / reference.len() as f64;
    let r2 = if include_r2 {
        if squared_error == 0.0 {
            Some(1.0)
        } else {
            let mean = reference_sum / reference.len() as f64;
            let total_variance = reference.iter().fold(0.0f64, |sum, value| {
                let value = f64::from(value.to_f32());
                let centered = value - mean;
                sum + centered * centered
            });
            (total_variance > 0.0).then_some(1.0 - squared_error / total_variance)
        }
    } else {
        None
    };
    Ok(KvVectorMetrics {
        element_count: reference.len(),
        cosine_similarity,
        mse,
        r2,
        max_abs_error,
        bit_mismatch_count,
    })
}

fn threshold_reasons(
    keys: &KvVectorMetrics,
    values: &KvVectorMetrics,
    options: KvComparisonOptions,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if let Some(threshold) = options.max_abs_threshold {
        if keys.max_abs_error > threshold {
            reasons.push("K max-absolute error".into());
        }
        if values.max_abs_error > threshold {
            reasons.push("V max-absolute error".into());
        }
    }
    if let Some(threshold) = options.mse_threshold {
        if keys.mse > threshold {
            reasons.push("K MSE".into());
        }
        if values.mse > threshold {
            reasons.push("V MSE".into());
        }
    }
    if let Some(threshold) = options.cosine_threshold {
        if keys.cosine_similarity.is_none_or(|value| value < threshold) {
            reasons.push("K cosine".into());
        }
        if values
            .cosine_similarity
            .is_none_or(|value| value < threshold)
        {
            reasons.push("V cosine".into());
        }
    }
    if let Some(threshold) = options.r2_threshold {
        if keys.r2.is_none_or(|value| value < threshold) {
            reasons.push("K R2".into());
        }
        if values.r2.is_none_or(|value| value < threshold) {
            reasons.push("V R2".into());
        }
    }
    reasons
}

/// Prepare an explicit in-memory perturbation receipt and altered compact
/// payload. The source snapshot remains immutable and re-verifiable.
pub fn prepare_diagnostic_perturbation(
    source: &KvSnapshot,
    perturbation: KvDiagnosticPerturbation,
) -> anyhow::Result<KvDiagnosticAlteration> {
    source.verify()?;
    let manifest = source.manifest();
    anyhow::ensure!(
        manifest.provenance.origin == KvSnapshotOrigin::Native,
        "diagnostic perturbation currently requires a native snapshot"
    );
    anyhow::ensure!(
        manifest.sequence_length > 0,
        "cannot perturb an empty KV prefix"
    );
    anyhow::ensure!(
        perturbation.layer < manifest.layer_count,
        "perturbation layer {} is outside layer count {}",
        perturbation.layer,
        manifest.layer_count
    );
    anyhow::ensure!(
        perturbation.head < manifest.n_kv_heads,
        "perturbation head {} is outside KV head count {}",
        perturbation.head,
        manifest.n_kv_heads
    );
    if let KvPerturbOperation::Scale { factor } = perturbation.operation {
        anyhow::ensure!(factor.is_finite(), "perturbation scale must be finite");
        anyhow::ensure!(
            factor != 0.0,
            "use the explicit zero operation instead of scale 0"
        );
        anyhow::ensure!(factor != 1.0, "perturbation scale 1 is a no-op");
    }
    let mut keys = Vec::new();
    keys.try_reserve_exact(source.keys().len())
        .context("cannot allocate diagnostic K payload")?;
    keys.extend_from_slice(source.keys());
    let mut values = Vec::new();
    values
        .try_reserve_exact(source.values().len())
        .context("cannot allocate diagnostic V payload")?;
    values.extend_from_slice(source.values());
    let per_head_elements = manifest
        .sequence_length
        .checked_mul(manifest.head_dim)
        .context("perturbation head size overflow")?;
    let ordinal = perturbation
        .layer
        .checked_mul(manifest.n_kv_heads)
        .and_then(|value| value.checked_add(perturbation.head))
        .context("perturbation layer/head index overflow")?;
    let start = ordinal
        .checked_mul(per_head_elements)
        .context("perturbation payload offset overflow")?;
    let end = start
        .checked_add(per_head_elements)
        .context("perturbation payload end overflow")?;
    let touches_keys = matches!(
        perturbation.component,
        KvPerturbComponent::Keys | KvPerturbComponent::Both
    );
    let touches_values = matches!(
        perturbation.component,
        KvPerturbComponent::Values | KvPerturbComponent::Both
    );
    if touches_keys {
        perturb_values(&mut keys[start..end], perturbation.operation)?;
    }
    if touches_values {
        perturb_values(&mut values[start..end], perturbation.operation)?;
    }
    let spec_bytes = serde_json::to_vec(&perturbation)?;
    let mut identity_input = KV_DIAGNOSTIC_PERTURBATION_SCHEMA.as_bytes().to_vec();
    identity_input.push(0);
    identity_input.extend_from_slice(manifest.snapshot_hash.as_bytes());
    identity_input.push(0);
    identity_input.extend_from_slice(&spec_bytes);
    let diagnostic_sha256 = sha256_bytes(&identity_input);
    let receipt = KvDiagnosticPerturbationReceipt {
        schema: KV_DIAGNOSTIC_PERTURBATION_SCHEMA.into(),
        diagnostic_id: format!("diagnostic:{diagnostic_sha256}"),
        source_snapshot_hash: manifest.snapshot_hash.clone(),
        perturbation,
        prefix_positions_affected: manifest.sequence_length,
        key_elements_affected: usize::from(touches_keys) * per_head_elements,
        value_elements_affected: usize::from(touches_values) * per_head_elements,
        scale_factor_bits: match perturbation.operation {
            KvPerturbOperation::Zero => None,
            KvPerturbOperation::Scale { factor } => Some(factor.to_bits()),
        },
        arithmetic: match perturbation.operation {
            KvPerturbOperation::Zero => "write positive f16 zero".into(),
            KvPerturbOperation::Scale { .. } => {
                "f16-to-f32 multiply followed by one f16 round".into()
            }
        },
    };
    Ok(KvDiagnosticAlteration {
        receipt,
        keys,
        values,
    })
}

/// Import a fresh, strictly compatible cache and apply an already prepared
/// diagnostic candidate. Ordinary snapshot replay never calls this path.
pub fn import_diagnostic_cache(
    source: &KvSnapshot,
    candidate: &KvDiagnosticAlteration,
    target: &KvCompatibilityTarget,
) -> anyhow::Result<KVCache> {
    anyhow::ensure!(
        candidate.receipt.source_snapshot_hash == source.manifest().snapshot_hash,
        "diagnostic candidate source snapshot does not match import source"
    );
    let mut cache = source.import_cache(target)?;
    cache
        .import_compact_prefix(
            source.manifest().sequence_length,
            candidate.keys(),
            candidate.values(),
        )
        .map_err(anyhow::Error::msg)?;
    Ok(cache)
}

fn perturb_values(values: &mut [f16], operation: KvPerturbOperation) -> anyhow::Result<()> {
    match operation {
        KvPerturbOperation::Zero => values.fill(f16::ZERO),
        KvPerturbOperation::Scale { factor } => {
            for (index, value) in values.iter_mut().enumerate() {
                let scaled = f16::from_f32(value.to_f32() * factor);
                anyhow::ensure!(
                    scaled.is_finite(),
                    "scaled diagnostic f16 value is non-finite at selected-head index {index}"
                );
                *value = scaled;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_snapshot::{KvLayout, KvPrecision, KvQkNormOrder, KvRopeLayout, KvRopeMetadata};

    fn target(max_seq: usize) -> KvCompatibilityTarget {
        KvCompatibilityTarget {
            model_sha256: "11".repeat(32),
            tokenizer_sha256: Some("22".repeat(32)),
            architecture: "llama".into(),
            max_seq,
            layer_count: 2,
            n_kv_heads: 2,
            head_dim: 2,
            precision: KvPrecision::F16,
            layout: KvLayout::LayerHeadPositionDimensionCompact,
            rope: KvRopeMetadata {
                layout: KvRopeLayout::AdjacentPair,
                dimension_count: 2,
                theta: 10_000.0,
                frequency_layout: "uniform-theta".into(),
                position_origin: "absolute-zero-based".into(),
                keys_state: "post-rope".into(),
                qk_norm_order: KvQkNormOrder::AfterRope,
                has_q_norm: false,
                has_k_norm: false,
                qk_norm_epsilon: None,
            },
            value_state: "projection-output".into(),
            execution_mode: "planned".into(),
            execution_fingerprint: "33".repeat(32),
            plan_hash: Some("44".repeat(32)),
        }
    }

    fn snapshot(sequence_length: usize) -> KvSnapshot {
        snapshot_with_capacity(sequence_length, 3)
    }

    fn snapshot_with_capacity(sequence_length: usize, capacity: usize) -> KvSnapshot {
        let mut cache = KVCache::try_new(2, 2, 2, capacity).unwrap();
        for position in 0..sequence_length {
            for layer in 0..2 {
                let base = (1 + position * 10 + layer * 100) as f32;
                let keys = [base, base + 1.0, base + 2.0, base + 3.0];
                let values = [base + 4.0, base + 5.0, base + 6.0, base + 7.0];
                cache.append(layer, position, &keys, &values);
            }
            cache.advance_cursor();
        }
        let tokens = (0..sequence_length as u32).collect::<Vec<_>>();
        KvSnapshot::export_native(&cache, target(capacity), Some(&tokens), Some(7)).unwrap()
    }

    #[test]
    fn capacity_only_difference_is_payload_identical() {
        let reference = snapshot_with_capacity(3, 3);
        let candidate = snapshot_with_capacity(3, 4);
        let report =
            compare_snapshots(&reference, &candidate, KvComparisonOptions::default()).unwrap();
        assert_eq!(report.status, "payload-identical");
        assert!(report.payload_bit_exact);
        assert!(!report.snapshot_hash_equal);
        assert_eq!(
            (report.reference_capacity, report.candidate_capacity),
            (3, 4)
        );
    }

    #[test]
    fn exact_comparison_reports_identity_and_no_threshold() {
        let source = snapshot(3);
        let report = compare_snapshots(
            &source,
            &source,
            KvComparisonOptions {
                include_r2: true,
                ..KvComparisonOptions::default()
            },
        )
        .unwrap();
        assert_eq!(report.schema, KV_COMPARISON_SCHEMA);
        assert_eq!(report.keys_global.cosine_similarity, Some(1.0));
        assert_eq!(report.keys_global.mse, 0.0);
        assert_eq!(report.keys_global.r2, Some(1.0));
        assert_eq!(report.keys_global.bit_mismatch_count, 0);
        assert!(report.payload_bit_exact);
        assert!(report.first_threshold_exceedance.is_none());
        assert_eq!(report.per_layer_head.len(), 4);
        assert_eq!(report.status, "identical");
        let first_json = serde_json::to_string(&report).unwrap();
        let second_json = serde_json::to_string(
            &compare_snapshots(
                &source,
                &source,
                KvComparisonOptions {
                    include_r2: true,
                    ..KvComparisonOptions::default()
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(first_json, second_json);
        assert!(!first_json.contains("NaN"));
    }

    #[test]
    fn one_head_perturbation_is_localized_without_new_snapshot() {
        let source = snapshot(3);
        let altered = prepare_diagnostic_perturbation(
            &source,
            KvDiagnosticPerturbation {
                layer: 1,
                head: 0,
                component: KvPerturbComponent::Keys,
                operation: KvPerturbOperation::Zero,
            },
        )
        .unwrap();
        assert_eq!(
            altered.receipt().source_snapshot_hash,
            source.manifest().snapshot_hash
        );
        assert!(altered.receipt().diagnostic_id.starts_with("diagnostic:"));
        assert_eq!(altered.receipt().prefix_positions_affected, 3);
        assert_eq!(altered.receipt().key_elements_affected, 6);
        assert_eq!(altered.receipt().value_elements_affected, 0);
        assert_eq!(altered.receipt().scale_factor_bits, None);
        assert_eq!(
            source.manifest().provenance.origin,
            KvSnapshotOrigin::Native
        );
        source.verify().unwrap();
        let report = compare_snapshot_to_diagnostic(
            &source,
            &altered,
            KvComparisonOptions {
                max_abs_threshold: Some(0.0),
                ..KvComparisonOptions::default()
            },
        )
        .unwrap();
        assert!(report.candidate_snapshot_hash.is_none());
        assert!(!report.thresholds_passed);
        assert_eq!(report.threshold_exceedances.len(), 1);
        let first = report.first_threshold_exceedance.unwrap();
        assert_eq!((first.layer, first.head), (1, 0));
        for item in report.per_layer_head {
            let affected = item.layer == 1 && item.head == 0;
            assert_eq!(item.keys.bit_mismatch_count > 0, affected);
            assert_eq!(item.values.bit_mismatch_count, 0);
        }
    }

    #[test]
    fn diagnostic_cache_import_preserves_cursor_and_isolates_head() {
        let source = snapshot(3);
        let altered = prepare_diagnostic_perturbation(
            &source,
            KvDiagnosticPerturbation {
                layer: 0,
                head: 1,
                component: KvPerturbComponent::Both,
                operation: KvPerturbOperation::Scale { factor: 0.5 },
            },
        )
        .unwrap();
        let imported = import_diagnostic_cache(&source, &altered, &target(3)).unwrap();
        assert_eq!(imported.cursor(), 3);
        let (keys, values) = imported.export_compact_prefix(3).unwrap();
        assert_eq!(keys, altered.keys());
        assert_eq!(values, altered.values());
    }

    #[test]
    fn incompatible_prefixes_are_rejected_before_metrics() {
        let short = snapshot(2);
        let long = snapshot(3);
        let error = compare_snapshots(&short, &long, KvComparisonOptions::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("sequence length"));
    }

    #[test]
    fn degenerate_reference_metrics_are_null_not_nan() {
        let zero = [f16::ZERO; 4];
        let one = [f16::ONE; 4];
        let metrics = vector_metrics(&zero, &one, true).unwrap();
        assert_eq!(metrics.cosine_similarity, None);
        assert_eq!(metrics.r2, None);
        assert_eq!(metrics.mse, 1.0);
    }

    #[test]
    fn invalid_scale_is_rejected_and_source_is_unchanged() {
        let source = snapshot(3);
        let original_hash = source.manifest().snapshot_hash.clone();
        for factor in [0.0, 1.0, f32::NAN, f32::INFINITY, f32::MAX] {
            assert!(prepare_diagnostic_perturbation(
                &source,
                KvDiagnosticPerturbation {
                    layer: 0,
                    head: 0,
                    component: KvPerturbComponent::Keys,
                    operation: KvPerturbOperation::Scale { factor },
                },
            )
            .is_err());
        }
        assert_eq!(source.manifest().snapshot_hash, original_hash);
        source.verify().unwrap();
    }
}
