//! Independently versioned, deterministic KV-prefix snapshots.
//!
//! The live [`KVCache`] remains the allocation used by ordinary inference.
//! This module performs work only when a caller explicitly exports, loads,
//! verifies, or imports a snapshot.

use crate::kv_cache::KVCache;
use crate::plan::ExecutionPlan;
use anyhow::Context;
use half::f16;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Independent snapshot schema. This does not alter `v04-plan/1` or
/// `ember.bundle.v1`.
pub const KV_SNAPSHOT_SCHEMA: &str = "ember.kv-snapshot.v1";
pub const KV_SNAPSHOT_KIND: &str = "kv-prefix";
pub const KV_SNAPSHOT_SERIALIZATION: &str = "manifest-json+f16le-v1";
pub const KV_KEY_FILE: &str = "keys.f16le";
pub const KV_VALUE_FILE: &str = "values.f16le";
pub const KV_MANIFEST_FILE: &str = "manifest.json";

const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_LAYERS: usize = 4096;
const MAX_KV_HEADS: usize = 4096;
const MAX_HEAD_DIM: usize = 65_536;
const MAX_SEQUENCE_LENGTH: usize = 16 * 1024 * 1024;
/// A hard trust boundary for the default loader. Callers that intentionally
/// handle larger artifacts can use [`KvSnapshot::load_dir_with_limit`].
pub const DEFAULT_MAX_PAYLOAD_BYTES: u64 = 16 * 1024 * 1024 * 1024;

static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Physical scalar representation in the snapshot and live cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvPrecision {
    F16,
}

/// Logical tensor order. Snapshot payloads compact away unused live capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvLayout {
    LayerHeadPositionDimensionCompact,
}

/// RoPE coordinate pairing used by the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvRopeLayout {
    AdjacentPair,
    SplitHalf,
}

/// Placement of optional headwise Q/K normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvQkNormOrder {
    BeforeRope,
    AfterRope,
}

/// Whether the payload came directly from the named model or from an
/// explicitly provenance-bearing transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvSnapshotOrigin {
    Native,
    Transformed,
}

/// Exact RoPE and Q/K-normalization semantics needed for compatibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvRopeMetadata {
    pub layout: KvRopeLayout,
    pub dimension_count: usize,
    pub theta: f32,
    /// Currently `uniform-theta`; future architecture-specific tables need a
    /// new independently meaningful identifier and metadata.
    pub frequency_layout: String,
    pub position_origin: String,
    pub keys_state: String,
    pub qk_norm_order: KvQkNormOrder,
    pub has_q_norm: bool,
    pub has_k_norm: bool,
    pub qk_norm_epsilon: Option<f32>,
}

/// A model/runtime cache target, independent of any particular snapshot.
///
/// This can be derived from an immutable [`ExecutionPlan`] for current
/// Llama/Qwen-family execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvCompatibilityTarget {
    pub model_sha256: String,
    pub tokenizer_sha256: Option<String>,
    pub architecture: String,
    pub max_seq: usize,
    pub layer_count: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub precision: KvPrecision,
    pub layout: KvLayout,
    pub rope: KvRopeMetadata,
    /// Semantic representation stored in V. Current Llama/Qwen snapshots use
    /// unmodified projection output; Gemma's normalized V requires a future
    /// per-layer schema.
    pub value_state: String,
    pub execution_mode: String,
    /// Capacity-independent hash of the operations, dispatch, runtime build,
    /// and model semantics that can affect continuation numerics.
    pub execution_fingerprint: String,
    /// Full plan identity retained for provenance. It is not a compatibility
    /// key because plan KV/scratch capacity may safely differ.
    pub plan_hash: Option<String>,
}

impl KvCompatibilityTarget {
    /// Build target metadata from the frozen v0.4 plan without changing that
    /// plan's schema.
    pub fn from_execution_plan(plan: &ExecutionPlan) -> anyhow::Result<Self> {
        let layout = match plan.kv.layout.as_str() {
            "layer-head-pos-dim" => KvLayout::LayerHeadPositionDimensionCompact,
            other => anyhow::bail!("unsupported execution-plan KV layout '{other}'"),
        };
        let precision = match plan.kv.precision.as_str() {
            "f16" => KvPrecision::F16,
            other => anyhow::bail!("unsupported execution-plan KV precision '{other}'"),
        };
        let rope_layout = match plan.rope.layout.as_str() {
            "adjacent-pair" => KvRopeLayout::AdjacentPair,
            "split-half" => KvRopeLayout::SplitHalf,
            other => anyhow::bail!("unsupported execution-plan RoPE layout '{other}'"),
        };
        let qk_norm_order = match plan.rope.qk_norm_order.as_str() {
            "before-rope" => KvQkNormOrder::BeforeRope,
            "after-rope" => KvQkNormOrder::AfterRope,
            other => anyhow::bail!("unsupported execution-plan QK norm order '{other}'"),
        };
        let target = Self {
            model_sha256: plan.model_sha256.clone(),
            tokenizer_sha256: nonempty(plan.tokenizer_sha256.clone()),
            architecture: plan.architecture.clone(),
            max_seq: plan.kv.max_seq,
            layer_count: plan.gguf.block_count,
            n_kv_heads: plan.kv.n_kv_heads,
            head_dim: plan.kv.head_dim,
            precision,
            layout,
            rope: KvRopeMetadata {
                layout: rope_layout,
                dimension_count: plan.gguf.rope_dimension_count,
                theta: plan.gguf.rope_theta,
                frequency_layout: "uniform-theta".into(),
                position_origin: "absolute-zero-based".into(),
                keys_state: "post-rope".into(),
                qk_norm_order,
                has_q_norm: plan.rope.has_q_norm,
                has_k_norm: plan.rope.has_k_norm,
                qk_norm_epsilon: (plan.rope.has_q_norm || plan.rope.has_k_norm).then_some(1e-6),
            },
            value_state: "projection-output".into(),
            execution_mode: plan.provenance.execution_mode.name().into(),
            execution_fingerprint: execution_fingerprint(plan)?,
            plan_hash: nonempty(plan.plan_hash.clone()),
        };
        target.validate()?;
        Ok(target)
    }

    /// Bytes allocated by a live owned cache for this target, including K,
    /// V, and the reusable attention scratch row.
    pub fn live_cache_bytes(&self) -> anyhow::Result<u64> {
        live_cache_bytes(
            self.layer_count,
            self.n_kv_heads,
            self.head_dim,
            self.max_seq,
        )
    }

    fn validate(&self) -> anyhow::Result<()> {
        validate_sha256("target model", &self.model_sha256)?;
        if let Some(tokenizer) = &self.tokenizer_sha256 {
            validate_sha256("target tokenizer", tokenizer)?;
        }
        anyhow::ensure!(
            !self.architecture.is_empty(),
            "target architecture is empty"
        );
        validate_dimensions(
            self.max_seq,
            self.layer_count,
            self.n_kv_heads,
            self.head_dim,
        )?;
        validate_rope(&self.rope, self.head_dim)?;
        anyhow::ensure!(
            self.value_state == "projection-output",
            "unsupported target value state '{}'",
            self.value_state
        );
        anyhow::ensure!(
            matches!(
                self.execution_mode.as_str(),
                "reference" | "planned" | "planned-fused"
            ),
            "unsupported target execution mode '{}'",
            self.execution_mode
        );
        validate_sha256("target execution fingerprint", &self.execution_fingerprint)?;
        if let Some(plan_hash) = &self.plan_hash {
            validate_sha256("target execution plan", plan_hash)?;
        }
        Ok(())
    }
}

/// Reserved transform provenance. Native snapshots leave this absent; no
/// field is populated with a placeholder value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvTransformProvenance {
    pub transform_id: String,
    pub mapper_sha256: String,
    pub source_layers: Vec<usize>,
    pub target_layer: Option<usize>,
    pub transformation_type: String,
}

/// Deterministic provenance relevant to replay and future transfer research.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvSnapshotProvenance {
    pub ember_version: String,
    pub execution_mode: String,
    pub execution_plan_hash: Option<String>,
    pub execution_fingerprint: String,
    pub origin: KvSnapshotOrigin,
    /// Model that originally produced the representations. For a native
    /// snapshot this equals `model_sha256`; a future transformed snapshot can
    /// retain a different source here while naming its compatible target in
    /// `model_sha256`.
    pub source_model_sha256: String,
    pub prefix_token_count: Option<usize>,
    pub prefix_token_ids_sha256: Option<String>,
    /// Optional greedy token selected from the prefix's final logits. This is
    /// replay convenience metadata, not part of the cache tensor state.
    pub resume_token_id: Option<u32>,
    pub transform: Option<KvTransformProvenance>,
}

/// One deterministic binary payload descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvPayloadDescriptor {
    pub file: String,
    pub elements: usize,
    pub byte_length: u64,
    pub sha256: String,
}

/// Small JSON manifest for the separately stored f16 tensor payloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvSnapshotManifest {
    pub schema: String,
    pub kind: String,
    pub serialization: String,
    /// Model whose runtime geometry and weights this payload is compatible
    /// with. Native snapshots also record it as the provenance source.
    pub model_sha256: String,
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
    pub provenance: KvSnapshotProvenance,
    pub keys: KvPayloadDescriptor,
    pub values: KvPayloadDescriptor,
    /// SHA-256 over the canonical manifest with this field empty. Payload
    /// checksums are therefore transitively covered.
    pub snapshot_hash: String,
}

/// A verified, owned snapshot. Payloads never alias a live cache.
pub struct KvSnapshot {
    manifest: KvSnapshotManifest,
    keys: Vec<f16>,
    values: Vec<f16>,
}

impl KvSnapshot {
    /// Copy a completed native prefix from a live cache.
    pub fn export_native(
        cache: &KVCache,
        target: KvCompatibilityTarget,
        prefix_token_ids: Option<&[u32]>,
        resume_token_id: Option<u32>,
    ) -> anyhow::Result<Self> {
        target.validate()?;
        anyhow::ensure!(
            cache.n_layers() == target.layer_count,
            "cache layer count {} does not match target {}",
            cache.n_layers(),
            target.layer_count
        );
        anyhow::ensure!(
            cache.n_kv_heads() == target.n_kv_heads,
            "cache KV head count {} does not match target {}",
            cache.n_kv_heads(),
            target.n_kv_heads
        );
        anyhow::ensure!(
            cache.head_dim() == target.head_dim,
            "cache head dimension {} does not match target {}",
            cache.head_dim(),
            target.head_dim
        );
        anyhow::ensure!(
            cache.max_seq_len() == target.max_seq,
            "cache capacity {} does not match target {}",
            cache.max_seq_len(),
            target.max_seq
        );
        let sequence_length = cache.cursor();
        let (keys, values) = cache
            .export_compact_prefix(sequence_length)
            .map_err(anyhow::Error::msg)?;
        let (prefix_token_count, prefix_token_ids_sha256) = if let Some(tokens) = prefix_token_ids {
            anyhow::ensure!(
                tokens.len() == sequence_length,
                "prefix token count {} does not match cache sequence length {sequence_length}",
                tokens.len()
            );
            anyhow::ensure!(
                target.tokenizer_sha256.is_some(),
                "tokenized-prefix provenance requires a tokenizer SHA-256"
            );
            (Some(tokens.len()), Some(hash_token_ids(tokens)))
        } else {
            (None, None)
        };
        let key_bytes = f16_to_le_bytes(&keys);
        let value_bytes = f16_to_le_bytes(&values);
        let elements = expected_elements(
            sequence_length,
            target.layer_count,
            target.n_kv_heads,
            target.head_dim,
        )?;
        anyhow::ensure!(keys.len() == elements && values.len() == elements);
        let provenance = KvSnapshotProvenance {
            ember_version: env!("CARGO_PKG_VERSION").into(),
            execution_mode: target.execution_mode.clone(),
            execution_plan_hash: target.plan_hash.clone(),
            execution_fingerprint: target.execution_fingerprint.clone(),
            origin: KvSnapshotOrigin::Native,
            source_model_sha256: target.model_sha256.clone(),
            prefix_token_count,
            prefix_token_ids_sha256,
            resume_token_id,
            transform: None,
        };
        let mut manifest = KvSnapshotManifest {
            schema: KV_SNAPSHOT_SCHEMA.into(),
            kind: KV_SNAPSHOT_KIND.into(),
            serialization: KV_SNAPSHOT_SERIALIZATION.into(),
            model_sha256: target.model_sha256,
            tokenizer_sha256: target.tokenizer_sha256,
            architecture: target.architecture,
            sequence_length,
            max_seq: target.max_seq,
            layer_count: target.layer_count,
            n_kv_heads: target.n_kv_heads,
            head_dim: target.head_dim,
            precision: target.precision,
            layout: target.layout,
            rope: target.rope,
            value_state: target.value_state,
            provenance,
            keys: payload_descriptor(KV_KEY_FILE, elements, &key_bytes)?,
            values: payload_descriptor(KV_VALUE_FILE, elements, &value_bytes)?,
            snapshot_hash: String::new(),
        };
        manifest.snapshot_hash = manifest_hash(&manifest)?;
        let snapshot = Self {
            manifest,
            keys,
            values,
        };
        snapshot.verify()?;
        Ok(snapshot)
    }

    pub fn manifest(&self) -> &KvSnapshotManifest {
        &self.manifest
    }

    pub fn keys(&self) -> &[f16] {
        &self.keys
    }

    pub fn values(&self) -> &[f16] {
        &self.values
    }

    /// Recompute all structural, payload, and manifest checks.
    pub fn verify(&self) -> anyhow::Result<()> {
        validate_manifest_metadata(&self.manifest)?;
        let expected = expected_elements(
            self.manifest.sequence_length,
            self.manifest.layer_count,
            self.manifest.n_kv_heads,
            self.manifest.head_dim,
        )?;
        anyhow::ensure!(
            self.keys.len() == expected,
            "key payload has {} elements; expected {expected}",
            self.keys.len()
        );
        anyhow::ensure!(
            self.values.len() == expected,
            "value payload has {} elements; expected {expected}",
            self.values.len()
        );
        verify_payload_descriptor(&self.manifest.keys, KV_KEY_FILE, &self.keys)?;
        verify_payload_descriptor(&self.manifest.values, KV_VALUE_FILE, &self.values)?;
        let computed = manifest_hash(&self.manifest)?;
        anyhow::ensure!(
            computed == self.manifest.snapshot_hash,
            "snapshot manifest hash mismatch: recorded {}, computed {computed}",
            self.manifest.snapshot_hash
        );
        Ok(())
    }

    /// Strict compatibility report. Any reason makes import incompatible;
    /// there is no "close enough" path.
    pub fn compatibility_report(&self, target: &KvCompatibilityTarget) -> KvCompatibilityReport {
        validate_compatibility(&self.manifest, target)
    }

    /// Validate first, then allocate and copy a new live cache whose cursor is
    /// exactly the completed prefix length.
    pub fn import_cache(&self, target: &KvCompatibilityTarget) -> anyhow::Result<KVCache> {
        self.import_cache_with_limit(target, DEFAULT_MAX_PAYLOAD_BYTES)
    }

    /// Import with an explicit upper bound on newly allocated live-cache
    /// bytes. The compact artifact limit and destination-cache limit are
    /// independent because replay may request a larger capacity.
    fn import_cache_with_limit(
        &self,
        target: &KvCompatibilityTarget,
        max_live_cache_bytes: u64,
    ) -> anyhow::Result<KVCache> {
        self.verify()?;
        let report = self.compatibility_report(target);
        anyhow::ensure!(
            report.compatible,
            "incompatible KV snapshot: {}",
            report.reasons.join("; ")
        );
        let live_bytes = target.live_cache_bytes()?;
        anyhow::ensure!(
            live_bytes <= max_live_cache_bytes,
            "destination live KV cache requires {live_bytes} bytes; limit is {max_live_cache_bytes}"
        );
        let mut cache = KVCache::try_new(
            target.layer_count,
            target.n_kv_heads,
            target.head_dim,
            target.max_seq,
        )
        .map_err(anyhow::Error::msg)?;
        cache
            .import_compact_prefix(self.manifest.sequence_length, &self.keys, &self.values)
            .map_err(anyhow::Error::msg)?;
        Ok(cache)
    }

    /// Atomically publish `manifest.json`, `keys.f16le`, and `values.f16le`.
    pub fn save_dir(&self, output: impl AsRef<Path>, overwrite: bool) -> anyhow::Result<PathBuf> {
        self.verify()?;
        let output = output.as_ref();
        anyhow::ensure!(
            !output.as_os_str().is_empty(),
            "snapshot output directory must not be empty"
        );
        reject_dangerous_snapshot_output(output)?;
        reject_symlink_if_present(output)?;
        if output.exists() {
            if !overwrite {
                anyhow::bail!(
                    "snapshot output '{}' already exists; refusing to overwrite",
                    output.display()
                );
            }
            Self::verify_dir(output).with_context(|| {
                format!(
                    "refusing to overwrite '{}': existing directory is not a verified KV snapshot",
                    output.display()
                )
            })?;
        }
        let parent = output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;
        let name = output
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("kv-snapshot");
        let staging = create_staging_directory(parent, name)?;
        let mut guard = StagingGuard::new(staging.clone());

        write_new_file(&staging.join(KV_KEY_FILE), &f16_to_le_bytes(&self.keys))?;
        write_new_file(&staging.join(KV_VALUE_FILE), &f16_to_le_bytes(&self.values))?;
        let mut manifest_bytes = serde_json::to_vec_pretty(&self.manifest)?;
        manifest_bytes.push(b'\n');
        write_new_file(&staging.join(KV_MANIFEST_FILE), &manifest_bytes)?;

        if output.exists() {
            anyhow::ensure!(
                overwrite,
                "snapshot output '{}' appeared during publication; refusing to overwrite",
                output.display()
            );
            reject_symlink_if_present(output)?;
            Self::verify_dir(output).with_context(|| {
                format!(
                    "refusing to overwrite '{}': publication target is no longer a verified KV snapshot",
                    output.display()
                )
            })?;
            std::fs::remove_dir_all(output)?;
        }
        std::fs::rename(&staging, output)?;
        guard.published = true;
        Ok(output.to_path_buf())
    }

    /// Load with the default 16-GiB total payload trust boundary.
    pub fn load_dir(input: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::load_dir_with_limit(input, DEFAULT_MAX_PAYLOAD_BYTES)
    }

    /// Load with an explicit allocation limit. File lengths, shape products,
    /// checksums, and unexpected trailing bytes are checked before import.
    pub fn load_dir_with_limit(
        input: impl AsRef<Path>,
        max_payload_bytes: u64,
    ) -> anyhow::Result<Self> {
        let input = input.as_ref();
        validate_snapshot_directory(input)?;
        let manifest_path = input.join(KV_MANIFEST_FILE);
        let manifest_len = std::fs::metadata(&manifest_path)?.len();
        anyhow::ensure!(
            manifest_len <= MAX_MANIFEST_BYTES,
            "snapshot manifest is {manifest_len} bytes; limit is {MAX_MANIFEST_BYTES}"
        );
        let manifest_bytes = read_exact_file(&manifest_path, manifest_len)?;
        let manifest: KvSnapshotManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| anyhow::anyhow!("malformed KV snapshot manifest: {error}"))?;
        validate_manifest_metadata(&manifest)?;

        let expected = expected_elements(
            manifest.sequence_length,
            manifest.layer_count,
            manifest.n_kv_heads,
            manifest.head_dim,
        )?;
        let expected_bytes = u64::try_from(expected)
            .ok()
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("KV payload byte length overflow"))?;
        for (name, descriptor) in [
            (KV_KEY_FILE, &manifest.keys),
            (KV_VALUE_FILE, &manifest.values),
        ] {
            anyhow::ensure!(descriptor.file == name, "unexpected payload file name");
            anyhow::ensure!(
                descriptor.elements == expected,
                "{name} descriptor elements {} do not match expected {expected}",
                descriptor.elements
            );
            anyhow::ensure!(
                descriptor.byte_length == expected_bytes,
                "{name} descriptor byte length {} does not match expected {expected_bytes}",
                descriptor.byte_length
            );
            let actual = std::fs::metadata(input.join(name))?.len();
            anyhow::ensure!(
                actual == expected_bytes,
                "{name} length {actual} does not match expected {expected_bytes} (truncated or has unexpected extra bytes)"
            );
        }
        let total = expected_bytes
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("total KV payload length overflow"))?;
        anyhow::ensure!(
            total <= max_payload_bytes,
            "snapshot payload is {total} bytes; configured allocation limit is {max_payload_bytes}"
        );
        let key_bytes = read_exact_file(&input.join(KV_KEY_FILE), expected_bytes)?;
        let value_bytes = read_exact_file(&input.join(KV_VALUE_FILE), expected_bytes)?;
        let snapshot = Self {
            manifest,
            keys: f16_from_le_bytes(&key_bytes)?,
            values: f16_from_le_bytes(&value_bytes)?,
        };
        snapshot.verify()?;
        Ok(snapshot)
    }

    /// Verify an on-disk snapshot and return its deterministic identity.
    pub fn verify_dir(input: impl AsRef<Path>) -> anyhow::Result<String> {
        let snapshot = Self::load_dir(input)?;
        Ok(snapshot.manifest.snapshot_hash.clone())
    }

    /// Concise stable text for CLI/GUI inspection.
    pub fn to_summary_text(&self) -> String {
        format!(
            "KV snapshot {}\nhash: {}\nmodel sha256: {}\ntokenizer sha256: {}\narchitecture: {}\nsequence: {} / {}\nshape: {} layers, {} KV heads, head_dim {}\nprecision/layout: {:?} / {:?}\nRoPE: {:?}, theta {}, {}, QK norm {:?} (q={} k={} eps={:?})\nvalues: {}\nexecution: {}  origin: {:?}\nkeys: {} bytes {}\nvalues: {} bytes {}\n",
            self.manifest.schema,
            self.manifest.snapshot_hash,
            self.manifest.model_sha256,
            self.manifest.tokenizer_sha256.as_deref().unwrap_or("unknown"),
            self.manifest.architecture,
            self.manifest.sequence_length,
            self.manifest.max_seq,
            self.manifest.layer_count,
            self.manifest.n_kv_heads,
            self.manifest.head_dim,
            self.manifest.precision,
            self.manifest.layout,
            self.manifest.rope.layout,
            self.manifest.rope.theta,
            self.manifest.rope.keys_state,
            self.manifest.rope.qk_norm_order,
            self.manifest.rope.has_q_norm,
            self.manifest.rope.has_k_norm,
            self.manifest.rope.qk_norm_epsilon,
            self.manifest.value_state,
            self.manifest.provenance.execution_mode,
            self.manifest.provenance.origin,
            self.manifest.keys.byte_length,
            self.manifest.keys.sha256,
            self.manifest.values.byte_length,
            self.manifest.values.sha256,
        )
    }
}

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
fn validate_compatibility(
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

fn optional_f32_bits(value: Option<f32>) -> Option<u32> {
    value.map(f32::to_bits)
}

fn compare(reasons: &mut Vec<String>, condition: bool, reason: &str) {
    if !condition {
        reasons.push(reason.into());
    }
}

fn validate_manifest_metadata(manifest: &KvSnapshotManifest) -> anyhow::Result<()> {
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

fn live_cache_bytes(
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

fn validate_dimensions(
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

fn validate_rope(rope: &KvRopeMetadata, head_dim: usize) -> anyhow::Result<()> {
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

fn expected_elements(
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

fn payload_descriptor(
    file: &str,
    elements: usize,
    bytes: &[u8],
) -> anyhow::Result<KvPayloadDescriptor> {
    Ok(KvPayloadDescriptor {
        file: file.into(),
        elements,
        byte_length: u64::try_from(bytes.len())?,
        sha256: sha256_hex(bytes),
    })
}

fn verify_payload_descriptor(
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
    let bytes = f16_to_le_bytes(values);
    anyhow::ensure!(
        descriptor.byte_length == u64::try_from(bytes.len())?,
        "{expected_file} byte length mismatch"
    );
    let computed = sha256_hex(&bytes);
    anyhow::ensure!(
        descriptor.sha256 == computed,
        "{expected_file} SHA-256 mismatch: recorded {}, computed {computed}",
        descriptor.sha256
    );
    Ok(())
}

fn execution_fingerprint(plan: &ExecutionPlan) -> anyhow::Result<String> {
    let mut value = serde_json::to_value(plan)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("execution plan did not serialize as an object"))?;
    object.remove("plan_hash");
    // Scratch offsets and KV allocation strides/capacity are runtime sizing,
    // not numerical execution semantics. The operation graph still records
    // where KV storage and attention occur.
    object.remove("scratch");
    if let Some(kv) = object
        .get_mut("kv")
        .and_then(serde_json::Value::as_object_mut)
    {
        for field in ["layer_stride", "head_stride", "pos_stride", "max_seq"] {
            kv.insert(field.into(), serde_json::json!(0));
        }
    }
    if let Some(gguf) = object
        .get_mut("gguf")
        .and_then(serde_json::Value::as_object_mut)
    {
        // The loaded runtime may deliberately cap the same model's table and
        // cache below its GGUF context. Positions within the imported prefix
        // use identical tables.
        gguf.insert("context_length".into(), serde_json::json!(0));
    }
    if let Some(provenance) = object
        .get_mut("provenance")
        .and_then(serde_json::Value::as_object_mut)
    {
        provenance.insert(
            "plan_build_time".into(),
            serde_json::Value::String(String::new()),
        );
    }
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

fn manifest_hash(manifest: &KvSnapshotManifest) -> anyhow::Result<String> {
    let mut value = manifest.clone();
    value.snapshot_hash.clear();
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

fn hash_token_ids(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"ember.kv-prefix-token-ids.v1\0");
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hex(&hasher.finalize())
}

fn f16_to_le_bytes(values: &[f16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(2));
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn f16_from_le_bytes(bytes: &[u8]) -> anyhow::Result<Vec<f16>> {
    anyhow::ensure!(
        bytes.len().is_multiple_of(2),
        "f16 payload byte length is odd"
    );
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 2)
        .map_err(|error| anyhow::anyhow!("cannot allocate decoded f16 payload: {error}"))?;
    for pair in bytes.chunks_exact(2) {
        values.push(f16::from_bits(u16::from_le_bytes([pair[0], pair[1]])));
    }
    Ok(values)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn validate_sha256(name: &str, value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} SHA-256 must be 64 lowercase hexadecimal characters"
    );
    Ok(())
}

fn nonempty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn validate_snapshot_directory(path: &Path) -> anyhow::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    anyhow::ensure!(
        metadata.file_type().is_dir(),
        "snapshot '{}' is not a real directory",
        path.display()
    );
    let mut names = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        anyhow::ensure!(
            file_type.is_file(),
            "snapshot entry '{}' is not a regular file",
            entry.path().display()
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("snapshot contains a non-UTF-8 file name"))?;
        names.push(name);
    }
    names.sort();
    let mut expected = vec![
        KV_KEY_FILE.to_string(),
        KV_MANIFEST_FILE.to_string(),
        KV_VALUE_FILE.to_string(),
    ];
    expected.sort();
    anyhow::ensure!(
        names == expected,
        "snapshot directory must contain exactly {expected:?}; found {names:?}"
    );
    Ok(())
}

fn read_exact_file(path: &Path, length: u64) -> anyhow::Result<Vec<u8>> {
    let length_usize = usize::try_from(length)
        .map_err(|_| anyhow::anyhow!("file '{}' is too large for this platform", path.display()))?;
    let mut file = File::open(path)?;
    anyhow::ensure!(
        file.metadata()?.file_type().is_file(),
        "'{}' is not a regular file",
        path.display()
    );
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length_usize).map_err(|error| {
        anyhow::anyhow!(
            "cannot allocate {length} bytes for '{}': {error}",
            path.display()
        )
    })?;
    bytes.resize(length_usize, 0);
    file.read_exact(&mut bytes)?;
    let mut trailing = [0u8; 1];
    anyhow::ensure!(
        file.read(&mut trailing)? == 0,
        "'{}' has unexpected extra bytes",
        path.display()
    );
    Ok(bytes)
}

fn create_staging_directory(parent: &Path, name: &str) -> anyhow::Result<PathBuf> {
    for _ in 0..128 {
        let staging = parent.join(format!(
            ".{name}.tmp-{}-{}",
            std::process::id(),
            STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::create_dir(&staging) {
            Ok(()) => return Ok(staging),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!("could not allocate a unique snapshot staging directory")
}

fn reject_dangerous_snapshot_output(path: &Path) -> anyhow::Result<()> {
    use std::path::Component;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    let cwd = std::env::current_dir()?.canonicalize()?;
    anyhow::ensure!(
        !cwd.starts_with(&normalized),
        "refusing snapshot output '{}' because it is the working directory or one of its ancestors",
        path.display()
    );
    Ok(())
}

fn write_new_file(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn reject_symlink_if_present(path: &Path) -> anyhow::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => anyhow::ensure!(
            !metadata.file_type().is_symlink(),
            "refusing snapshot output symlink '{}'",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

struct StagingGuard {
    path: PathBuf,
    published: bool,
}

impl StagingGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            published: false,
        }
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if !self.published {
            std::fs::remove_dir_all(&self.path).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(max_seq: usize) -> KvCompatibilityTarget {
        KvCompatibilityTarget {
            model_sha256: "aa".repeat(32),
            tokenizer_sha256: Some("bb".repeat(32)),
            architecture: "llama".into(),
            max_seq,
            layer_count: 2,
            n_kv_heads: 2,
            head_dim: 4,
            precision: KvPrecision::F16,
            layout: KvLayout::LayerHeadPositionDimensionCompact,
            rope: KvRopeMetadata {
                layout: KvRopeLayout::AdjacentPair,
                dimension_count: 4,
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
        use crate::plan::tests::sample_plan;
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
    fn directory_round_trip_and_integrity_verification() {
        let root = temp_dir("roundtrip");
        let original = snapshot(3, 5);
        original.save_dir(&root, false).unwrap();
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
    fn working_directory_and_ancestors_are_never_snapshot_outputs() {
        assert!(reject_dangerous_snapshot_output(Path::new(".")).is_err());
        assert!(reject_dangerous_snapshot_output(Path::new("..")).is_err());
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
}
