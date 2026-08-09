//! Tested stored-key <-> content-space RoPE seam.
//!
//! This is an experimental, allocation-bearing utility and is never called by
//! ordinary inference. It deliberately consumes the same precomputed table
//! generator as Ember inference. Production RoPE call sites are not rerouted:
//! their scalar/SIMD choices are part of existing decode numerics.

use crate::kv_snapshot::{KvQkNormOrder, KvRopeLayout, KvSnapshot, KvSnapshotManifest};
use crate::tensor::compute_rope_freqs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeDirection {
    Forward,
    Inverse,
}

/// Apply one already-selected RoPE cos/sin row to headwise f32 vectors.
///
/// Inverse uses the transpose rotation. It is approximate in f32; callers
/// requiring an exact no-op must preserve/bypass the original stored bytes.
pub fn rotate_key_row_in_place(
    values: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
    layout: KvRopeLayout,
    direction: RopeDirection,
) -> anyhow::Result<()> {
    anyhow::ensure!(n_heads > 0, "RoPE requires at least one head");
    anyhow::ensure!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "RoPE head_dim must be positive and even"
    );
    let expected = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow::anyhow!("RoPE row length overflow"))?;
    anyhow::ensure!(
        values.len() == expected,
        "RoPE row has {} values; expected {expected}",
        values.len()
    );
    let half = head_dim / 2;
    anyhow::ensure!(cos.len() == half, "RoPE cosine row length mismatch");
    anyhow::ensure!(sin.len() == half, "RoPE sine row length mismatch");
    anyhow::ensure!(
        cos.iter().chain(sin).all(|value| value.is_finite()),
        "RoPE table contains a non-finite value"
    );

    for head in 0..n_heads {
        let base = head * head_dim;
        for pair in 0..half {
            let (first, second) = match layout {
                KvRopeLayout::AdjacentPair => (base + 2 * pair, base + 2 * pair + 1),
                KvRopeLayout::SplitHalf => (base + pair, base + pair + half),
            };
            let a = values[first];
            let b = values[second];
            let c = cos[pair];
            let s = sin[pair];
            match direction {
                RopeDirection::Forward => {
                    values[first] = a * c - b * s;
                    values[second] = a * s + b * c;
                }
                RopeDirection::Inverse => {
                    values[first] = a * c + b * s;
                    values[second] = -a * s + b * c;
                }
            }
        }
    }
    Ok(())
}

/// Owned f32 keys in compact `[layer][head][position][dimension]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct KvContentKeys {
    pub layer_count: usize,
    pub n_kv_heads: usize,
    pub sequence_length: usize,
    pub head_dim: usize,
    pub values: Vec<f32>,
}

/// Remove position rotation from stored f16 keys.
///
/// "Content" means post-K-normalization/pre-RoPE. Qwen3/Gemma-style
/// before-RoPE normalization therefore remains in the returned vectors.
/// After-RoPE normalization with a K norm is rejected because snapshot
/// metadata does not contain an invertible normalization operation.
pub fn stored_keys_to_content(snapshot: &KvSnapshot) -> anyhow::Result<KvContentKeys> {
    snapshot.verify()?;
    validate_content_conversion(snapshot.manifest())?;
    let manifest = snapshot.manifest();
    let mut values: Vec<f32> = snapshot.keys().iter().map(|value| value.to_f32()).collect();
    apply_all_positions(&mut values, manifest, RopeDirection::Inverse)?;
    Ok(KvContentKeys {
        layer_count: manifest.layer_count,
        n_kv_heads: manifest.n_kv_heads,
        sequence_length: manifest.sequence_length,
        head_dim: manifest.head_dim,
        values,
    })
}

/// Apply the snapshot model's position rotation to compatible content keys.
///
/// The returned f32 values are ready for a future target snapshot constructor;
/// this function does not quantize or create a transformed snapshot.
pub fn content_keys_to_stored(
    content: &KvContentKeys,
    manifest: &KvSnapshotManifest,
) -> anyhow::Result<Vec<f32>> {
    validate_content_conversion(manifest)?;
    anyhow::ensure!(
        content.layer_count == manifest.layer_count
            && content.n_kv_heads == manifest.n_kv_heads
            && content.sequence_length == manifest.sequence_length
            && content.head_dim == manifest.head_dim,
        "content-key shape does not match target snapshot metadata"
    );
    let expected = manifest
        .layer_count
        .checked_mul(manifest.n_kv_heads)
        .and_then(|value| value.checked_mul(manifest.sequence_length))
        .and_then(|value| value.checked_mul(manifest.head_dim))
        .ok_or_else(|| anyhow::anyhow!("content-key shape product overflow"))?;
    anyhow::ensure!(
        content.values.len() == expected,
        "content-key payload has {} values; expected {expected}",
        content.values.len()
    );
    let mut values = content.values.clone();
    apply_all_positions(&mut values, manifest, RopeDirection::Forward)?;
    Ok(values)
}

fn validate_content_conversion(manifest: &KvSnapshotManifest) -> anyhow::Result<()> {
    anyhow::ensure!(
        manifest.rope.frequency_layout == "uniform-theta",
        "content conversion requires uniform-theta RoPE metadata"
    );
    anyhow::ensure!(
        manifest.rope.dimension_count == manifest.head_dim,
        "content conversion does not yet support partial RoPE dimensions"
    );
    anyhow::ensure!(
        manifest.rope.keys_state == "post-rope",
        "content conversion requires post-RoPE stored keys"
    );
    anyhow::ensure!(
        !(manifest.rope.qk_norm_order == KvQkNormOrder::AfterRope && manifest.rope.has_k_norm),
        "cannot recover content space when K normalization occurs after RoPE"
    );
    Ok(())
}

fn apply_all_positions(
    values: &mut [f32],
    manifest: &KvSnapshotManifest,
    direction: RopeDirection,
) -> anyhow::Result<()> {
    if manifest.sequence_length == 0 {
        return Ok(());
    }
    let (cos, sin) = compute_rope_freqs(
        manifest.sequence_length,
        manifest.head_dim,
        manifest.rope.theta,
        None,
    );
    let half = manifest.head_dim / 2;
    let head_stride = manifest
        .sequence_length
        .checked_mul(manifest.head_dim)
        .ok_or_else(|| anyhow::anyhow!("content-key head stride overflow"))?;
    let layer_stride = manifest
        .n_kv_heads
        .checked_mul(head_stride)
        .ok_or_else(|| anyhow::anyhow!("content-key layer stride overflow"))?;
    for layer in 0..manifest.layer_count {
        for head in 0..manifest.n_kv_heads {
            let head_start = layer * layer_stride + head * head_stride;
            for position in 0..manifest.sequence_length {
                let start = head_start + position * manifest.head_dim;
                let table_start = position * half;
                rotate_key_row_in_place(
                    &mut values[start..start + manifest.head_dim],
                    1,
                    manifest.head_dim,
                    &cos.data()[table_start..table_start + half],
                    &sin.data()[table_start..table_start + half],
                    manifest.rope.layout,
                    direction,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KVCache;
    use crate::kv_snapshot::{KvCompatibilityTarget, KvLayout, KvPrecision, KvRopeMetadata};

    #[test]
    fn hand_vectors_distinguish_layout_and_direction() {
        let cos = [0.0, 1.0];
        let sin = [1.0, 0.0];
        let mut adjacent = vec![1.0, 2.0, 3.0, 4.0];
        rotate_key_row_in_place(
            &mut adjacent,
            1,
            4,
            &cos,
            &sin,
            KvRopeLayout::AdjacentPair,
            RopeDirection::Forward,
        )
        .unwrap();
        assert_eq!(adjacent, [-2.0, 1.0, 3.0, 4.0]);

        let mut split = vec![1.0, 2.0, 3.0, 4.0];
        rotate_key_row_in_place(
            &mut split,
            1,
            4,
            &cos,
            &sin,
            KvRopeLayout::SplitHalf,
            RopeDirection::Forward,
        )
        .unwrap();
        assert_eq!(split, [-3.0, 2.0, 1.0, 4.0]);

        rotate_key_row_in_place(
            &mut split,
            1,
            4,
            &cos,
            &sin,
            KvRopeLayout::SplitHalf,
            RopeDirection::Inverse,
        )
        .unwrap();
        assert_eq!(split, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn multiple_heads_do_not_bleed_across_boundaries() {
        let original = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let mut values = original.clone();
        rotate_key_row_in_place(
            &mut values,
            2,
            4,
            &[0.5, 0.25],
            &[0.75, -0.5],
            KvRopeLayout::AdjacentPair,
            RopeDirection::Forward,
        )
        .unwrap();
        assert_eq!(
            &values[4..],
            values[..4].iter().map(|x| -*x).collect::<Vec<_>>()
        );
    }

    fn snapshot(layout: KvRopeLayout, qk_order: KvQkNormOrder, has_k_norm: bool) -> KvSnapshot {
        let mut cache = KVCache::new(1, 1, 4, 3);
        for position in 0..3 {
            cache.append(
                0,
                position,
                &[1.0 + position as f32, 2.0, 3.0, 4.0],
                &[0.0; 4],
            );
            cache.advance_cursor();
        }
        KvSnapshot::export_native(
            &cache,
            KvCompatibilityTarget {
                model_sha256: "aa".repeat(32),
                tokenizer_sha256: None,
                architecture: "test".into(),
                max_seq: 3,
                layer_count: 1,
                n_kv_heads: 1,
                head_dim: 4,
                precision: KvPrecision::F16,
                layout: KvLayout::LayerHeadPositionDimensionCompact,
                rope: KvRopeMetadata {
                    layout,
                    dimension_count: 4,
                    theta: 10_000.0,
                    frequency_layout: "uniform-theta".into(),
                    position_origin: "absolute-zero-based".into(),
                    keys_state: "post-rope".into(),
                    qk_norm_order: qk_order,
                    has_q_norm: has_k_norm,
                    has_k_norm,
                    qk_norm_epsilon: has_k_norm.then_some(1e-6),
                },
                value_state: "projection-output".into(),
                execution_mode: "reference".into(),
                execution_fingerprint: "cc".repeat(32),
                plan_hash: None,
            },
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn snapshot_content_round_trip_has_tight_f32_error() {
        for layout in [KvRopeLayout::AdjacentPair, KvRopeLayout::SplitHalf] {
            let snapshot = snapshot(layout, KvQkNormOrder::BeforeRope, true);
            let content = stored_keys_to_content(&snapshot).unwrap();
            let restored = content_keys_to_stored(&content, snapshot.manifest()).unwrap();
            for (expected, actual) in snapshot.keys().iter().zip(restored) {
                assert!(
                    (expected.to_f32() - actual).abs() <= 1e-6,
                    "round trip {:?}: {} vs {actual}",
                    layout,
                    expected.to_f32()
                );
            }
        }
    }

    #[test]
    fn after_rope_k_norm_fails_closed() {
        let snapshot = snapshot(KvRopeLayout::AdjacentPair, KvQkNormOrder::AfterRope, true);
        let error = stored_keys_to_content(&snapshot).unwrap_err().to_string();
        assert!(error.contains("normalization occurs after RoPE"));
    }

    #[test]
    fn invalid_shapes_fail_without_panicking() {
        assert!(rotate_key_row_in_place(
            &mut [1.0, 2.0, 3.0],
            1,
            3,
            &[1.0],
            &[0.0],
            KvRopeLayout::AdjacentPair,
            RopeDirection::Forward,
        )
        .is_err());
    }
}
