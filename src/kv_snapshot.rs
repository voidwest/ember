//! Independently versioned, deterministic KV-prefix snapshots.
//!
//! The live [`KVCache`] remains the allocation used by ordinary inference.
//! This module performs work only when a caller explicitly exports, loads,
//! verifies, or imports a snapshot.

use crate::kv_cache::KVCache;
use crate::plan::ExecutionPlan;
use crate::v05::manifest::{hex, sha256_hex};
use anyhow::Context;
use half::f16;
use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
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
/// A hard trust boundary for the default loader. The limit leaves headroom
/// for decoded values and a destination cache on 16-GiB hosts; callers that
/// intentionally handle larger artifacts can use [`KvSnapshot::load_dir_with_limit`].
pub const DEFAULT_MAX_PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;

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
    /// Per-frequency-pair factors (`rope_freqs.weight`, llama3-style RoPE
    /// scaling) applied on top of the uniform-theta frequencies, when the
    /// model has them. Snapshots written before this field existed omit it
    /// even for such models, so `None` means "uniform or not recorded".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_factors: Option<Vec<f32>>,
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

/// Whether two RoPE descriptions agree on their frequency factors. A side
/// that does not record them (snapshots that predate the field) is accepted;
/// the model-identity checks still pin the actual model.
pub(crate) fn frequency_factors_agree(left: &KvRopeMetadata, right: &KvRopeMetadata) -> bool {
    match (&left.frequency_factors, &right.frequency_factors) {
        (Some(left), Some(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        }
        _ => true,
    }
}

impl KvCompatibilityTarget {
    /// Record the model's RoPE frequency factors (see
    /// [`crate::llama::Llama::rope_frequency_factors`]), which the execution
    /// plan does not carry.
    pub fn with_rope_frequency_factors(mut self, factors: Option<&[f32]>) -> Self {
        self.rope.frequency_factors = factors.map(<[f32]>::to_vec);
        self
    }

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
                frequency_factors: None,
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

/// Seed used to walk untyped JSON and reject duplicate object keys before
/// serde maps them with last-wins semantics.
struct DuplicateKeySeed;

impl<'de> DeserializeSeed<'de> for DuplicateKeySeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(DuplicateKeyVisitor)
    }
}

struct DuplicateKeyVisitor;

impl<'de> Visitor<'de> for DuplicateKeyVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut keys = std::collections::HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate object key '{key}'"
                )));
            }
            map.next_value_seed(DuplicateKeySeed)?;
        }
        Ok(())
    }

    fn visit_seq<S>(self, mut sequence: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        while sequence.next_element_seed(DuplicateKeySeed)?.is_some() {}
        Ok(())
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }
}

impl KvSnapshotManifest {
    /// Deserialize and structurally validate an in-memory snapshot manifest.
    ///
    /// This is the same manifest boundary used by [`KvSnapshot::load_dir`],
    /// exposed without filesystem access so hostile JSON can be fuzzed or
    /// validated before payload files are opened. The one-megabyte manifest
    /// limit is enforced here as well as by the directory loader.
    pub fn from_json_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let length =
            u64::try_from(bytes.len()).context("KV snapshot manifest length exceeds u64")?;
        anyhow::ensure!(
            length <= MAX_MANIFEST_BYTES,
            "snapshot manifest is {length} bytes; limit is {MAX_MANIFEST_BYTES}"
        );
        let mut duplicate_check = serde_json::Deserializer::from_slice(bytes);
        duplicate_check
            .deserialize_any(DuplicateKeyVisitor)
            .map_err(|error| anyhow::anyhow!("malformed KV snapshot manifest: {error}"))?;
        duplicate_check
            .end()
            .map_err(|error| anyhow::anyhow!("malformed KV snapshot manifest: {error}"))?;
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|error| anyhow::anyhow!("malformed KV snapshot manifest: {error}"))?;
        validate_manifest_metadata(&manifest)?;
        Ok(manifest)
    }
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
            keys: payload_descriptor_from_f16(KV_KEY_FILE, elements, &keys)?,
            values: payload_descriptor_from_f16(KV_VALUE_FILE, elements, &values)?,
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

        write_f16_new_file(&staging.join(KV_KEY_FILE), &self.keys)?;
        write_f16_new_file(&staging.join(KV_VALUE_FILE), &self.values)?;
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

    /// Load with the default 4-GiB total payload trust boundary.
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
        let manifest = KvSnapshotManifest::from_json_bytes(&manifest_bytes)?;

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
        // Decode one payload at a time directly into its final f16 storage.
        // Keeping the on-disk bytes out of long-lived Vecs avoids retaining
        // both the serialized and decoded copies at once (the limit above is
        // a payload-size limit, not permission to transiently double it).
        let keys = read_f16_file(&input.join(KV_KEY_FILE), expected_bytes)?;
        let values = read_f16_file(&input.join(KV_VALUE_FILE), expected_bytes)?;
        let snapshot = Self {
            manifest,
            keys,
            values,
        };
        snapshot.verify()?;
        Ok(snapshot)
    }

    /// Verify an on-disk snapshot and return its deterministic identity.
    pub fn verify_dir(input: impl AsRef<Path>) -> anyhow::Result<String> {
        let snapshot = Self::load_dir(input)?;
        Ok(snapshot.manifest.snapshot_hash)
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

mod compat;
mod io;
use compat::*;
pub use compat::{KvCompatibilityReport, KvSnapshotCompatibilityMetadata};
use io::*;
pub use io::{hash_token_ids, normalize_path};

#[cfg(test)]
pub(crate) mod tests;
