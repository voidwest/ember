//! v0.4 execution-plan types and deterministic diagnostics.
//!
//! An [`ExecutionPlan`] is an immutable, serializable description of how a
//! loaded model will execute decode: the architecture-specific operation
//! sequence, resolved per-tensor kernel dispatch, scratch layout, KV layout,
//! hook-site resolution, CPU requirements, and provenance. It is built once
//! after model load (see `Llama::execution_plan` in `src/llama.rs`) and
//! interpreted by the planned decode loop.
//!
//! The plan stores stable indices and validated metadata — never raw
//! pointers. Tensor identity is a [`TensorRef`] into the plan's own
//! `tensor_table`; the decode interpreter resolves `(layer, op)` to concrete
//! model fields through the same structural code paths that built the model,
//! so weight access cannot dangle.
//!
//! Contract: `docs/v04-execution-contract.md` (frozen 2026-08-04), section 10.

use crate::v05::manifest::hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Execution-plan schema version (`"v04-plan/1"`).
pub const PLAN_SCHEMA_VERSION: u32 = 1;

/// Numerical/runtime kernel ABI encoded into plan identity. Revision 2 is the
/// canonical Q4_K/Q6_K × Q8_K implementation; revision 1 was the superseded
/// exact-f32 K-quant production path.
pub const PLAN_KERNEL_REVISION: u32 = 2;

fn legacy_plan_kernel_revision() -> u32 {
    1
}

fn is_legacy_plan_kernel_revision(revision: &u32) -> bool {
    *revision == legacy_plan_kernel_revision()
}

// ---------------------------------------------------------------------------
// execution modes
// ---------------------------------------------------------------------------

/// The three separable execution concepts (contract section 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionMode {
    /// The v0.3 generic hooked path with per-tensor dynamic dispatch — the
    /// readable oracle and the parity baseline.
    Reference,
    /// Plan-driven dispatch, identical operation sequence, scratch arena,
    /// no fusion.
    Planned,
    /// The plan with the frozen fusion set, de-fused per active hooks.
    PlannedFused,
}

impl ExecutionMode {
    /// Parse the `--execution` CLI value.
    pub fn from_cli(value: &str) -> Result<Self, String> {
        match value {
            "reference" => Ok(Self::Reference),
            "planned" => Ok(Self::Planned),
            "planned-fused" => Ok(Self::PlannedFused),
            _ => Err(format!(
                "unknown --execution value '{value}' (expected reference | planned | planned-fused)"
            )),
        }
    }

    /// CLI-facing name (round-trips through [`ExecutionMode::from_cli`]).
    pub fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Planned => "planned",
            Self::PlannedFused => "planned-fused",
        }
    }
}

/// Hook activation mode, resolved at plan build (contract section 12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HookMode {
    /// Fast normal path: no capture metadata, no clones, no string lookup,
    /// no registry inspection, no trace serialization.
    Disabled,
    /// Existing capture semantics unchanged.
    Observe,
    /// Existing patch semantics unchanged.
    Intervene,
}

// ---------------------------------------------------------------------------
// kernels and fusion state
// ---------------------------------------------------------------------------

/// Resolved kernel identity per tensor (contract D6: one shared resolution
/// used by both legacy dynamic dispatch and the plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KernelId {
    /// Dense f32 matmul (eager-f32 oracle / documented fallback).
    EagerF32,
    /// Direct f32 embedding-row copy (not a matrix multiplication).
    EmbeddingF32Row,
    /// Q8_0 embedding-row dequantization.
    EmbeddingQ8Row,
    /// Q4_K embedding-row dequantization.
    EmbeddingQ4KRow,
    /// Q6_K embedding-row dequantization.
    EmbeddingQ6KRow,
    /// Q8_0 packed native matmul path (v0.3, unchanged).
    Q8Packed,
    /// Compressed-resident Q4_K scalar kernel.
    KQuantScalarQ4K,
    /// Compressed-resident Q6_K scalar kernel.
    KQuantScalarQ6K,
    /// Compressed-resident Q4_K AVX2/FMA/F16C/SSSE3 kernel.
    KQuantAvx2Q4K,
    /// Compressed-resident Q6_K AVX2/FMA/F16C/SSSE3 kernel.
    KQuantAvx2Q6K,
}

impl KernelId {
    /// Stable short name for provenance and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::EagerF32 => "eager-f32",
            Self::EmbeddingF32Row => "embedding-f32-row",
            Self::EmbeddingQ8Row => "embedding-q8-0-row-dequant",
            Self::EmbeddingQ4KRow => "embedding-q4-k-row-dequant",
            Self::EmbeddingQ6KRow => "embedding-q6-k-row-dequant",
            Self::Q8Packed => "q8-packed",
            Self::KQuantScalarQ4K => "q4-k-q8-k-scalar",
            Self::KQuantScalarQ6K => "q6-k-q8-k-scalar",
            Self::KQuantAvx2Q4K => "q4-k-q8-k-avx2",
            Self::KQuantAvx2Q6K => "q6-k-q8-k-avx2",
        }
    }

    /// Revision-aware diagnostic label. Revision-1 plans used the same enum
    /// variants for the superseded exact-f32/dequant kernels; rendering them
    /// with revision-2 Q8_K names would rewrite history during offline inspect.
    pub fn name_for_revision(self, revision: u32) -> &'static str {
        if revision <= 1 {
            return match self {
                Self::KQuantScalarQ4K => "scalar-q4k",
                Self::KQuantScalarQ6K => "scalar-q6k",
                Self::KQuantAvx2Q4K => "avx2-q4k",
                Self::KQuantAvx2Q6K => "avx2-q6k",
                _ => self.name(),
            };
        }
        self.name()
    }

    /// CPU feature requirement, if any.
    pub fn cpu_feature(self) -> Option<&'static str> {
        match self {
            Self::KQuantAvx2Q4K | Self::KQuantAvx2Q6K => Some("avx2+fma+f16c+ssse3"),
            _ => None,
        }
    }
}

/// Resolve the kernel for a tensor from its GGUF dtype name and the v0.3
/// per-tensor execution decision. This is the single shared resolution
/// function (contract D6); the legacy dispatch and the plan both derive
/// their kernel identity from it.
pub fn resolve_kernel(gguf_dtype: &str, execution: &str) -> KernelId {
    match (gguf_dtype, execution) {
        ("q4_k", "compressed_x86") => KernelId::KQuantAvx2Q4K,
        ("q4_k", "compressed_scalar") => KernelId::KQuantScalarQ4K,
        ("q6_k", "compressed_x86") => KernelId::KQuantAvx2Q6K,
        ("q6_k", "compressed_scalar") => KernelId::KQuantScalarQ6K,
        ("q8_0", _) => KernelId::Q8Packed,
        _ => KernelId::EagerF32,
    }
}

/// Resolve an embedding lookup kernel. Embeddings dequantize one stored row;
/// they do not execute the projection matmul selected by [`resolve_kernel`].
pub fn resolve_embedding_kernel(gguf_dtype: &str, execution: &str) -> Option<KernelId> {
    if execution == "eager_f32" {
        return Some(KernelId::EmbeddingF32Row);
    }
    match (gguf_dtype, execution) {
        ("q8_0", "compressed") => Some(KernelId::EmbeddingQ8Row),
        ("q4_k", "compressed_scalar" | "compressed_x86") => Some(KernelId::EmbeddingQ4KRow),
        ("q6_k", "compressed_scalar" | "compressed_x86") => Some(KernelId::EmbeddingQ6KRow),
        _ => None,
    }
}

/// Per-layer fusion selection (contract section 6 cross-cutting rules).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FusionState {
    /// All applicable fusions active for the layer.
    Fused,
    /// Some fusions de-activated (recorded reason).
    PartiallyFused,
    /// No fusions active for the layer.
    Unfused,
}

/// The frozen fusion set (contract section 6: F1-F5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FusedOpKind {
    /// F1: RMSNorm + quantized linear projection.
    RmsNormLinear,
    /// F2: residual add + RMSNorm.
    ResidualRmsNorm,
    /// F3: Q/K/V projection orchestration with shared normalized input.
    QkvOrchestration,
    /// F4: RoPE within the planned attention path.
    RopeInAttention,
    /// F5: output projection + residual add.
    OutputProjResidual,
}

// ---------------------------------------------------------------------------
// tensor identity
// ---------------------------------------------------------------------------

/// Stable tensor identity: an index into [`ExecutionPlan::tensor_table`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TensorRef {
    pub id: usize,
}

impl TensorRef {
    pub const fn new(id: usize) -> Self {
        Self { id }
    }
}

/// One weight tensor known to the plan (derived from the v0.3 tensor
/// inventory; contract section 5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRecord {
    pub id: usize,
    /// GGUF tensor name, e.g. `blk.3.attn_q.weight`.
    pub name: String,
    /// Logical shape. Linears use the `[out_features, in_features]` K-quant
    /// convention; norm weights are 1-D `[dim]`.
    pub shape: Vec<usize>,
    /// GGUF dtype name: `q4_k`, `q6_k`, `q8_0`, `f32`, `f16`.
    pub gguf_dtype: String,
    /// v0.3 per-tensor execution decision: `eager_f32` | `compressed_scalar`
    /// | `compressed_x86`.
    pub execution: String,
    /// Resolved kernel ([`resolve_kernel`]).
    pub kernel: KernelId,
    /// Compressed resident bytes (or expanded f32 bytes for eager tensors).
    pub resident_bytes: usize,
    /// Whether the storage directly references a read-only file mapping.
    pub mmap: bool,
}

// ---------------------------------------------------------------------------
// scratch arena
// ---------------------------------------------------------------------------

/// One named region of the decode scratch arena.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScratchRegion {
    /// Stable region name; activation regions are named after the tensor id
    /// they hold (e.g. `t12`).
    pub name: String,
    /// Byte offset into the arena.
    pub offset: usize,
    /// Byte size of the region.
    pub size: usize,
    /// Required alignment.
    pub alignment: usize,
    /// First op index (into the layer op sequence at build order) using it.
    pub first_op: usize,
    /// Last op index using it.
    pub last_op: usize,
    /// Name of the region this one shares storage with, when the planner
    /// proved their lifetimes disjoint.
    pub shared_with: Option<String>,
}

/// The scratch arena plan (contract section 11). All offsets are
/// deterministic; regions may share storage only with a recorded proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScratchPlan {
    /// Total arena bytes.
    pub total_bytes: usize,
    /// Arena base alignment (64 bytes; AVX2-safe).
    pub alignment: usize,
    /// Sequence capacity the activation regions are sized for (1 = decode).
    pub seq_capacity: usize,
    /// Region descriptors.
    pub regions: Vec<ScratchRegion>,
    /// tensor id -> region name, for interpreter lookups.
    pub tensor_regions: BTreeMap<usize, String>,
}

/// Deterministic arena diagnostics (contract section 11: total bytes,
/// region names, offsets, alignments, maximum live interval).
#[cfg(test)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArenaReport {
    pub total_bytes: usize,
    pub alignment: usize,
    pub seq_capacity: usize,
    pub region_count: usize,
    /// Largest `last_op - first_op` over all regions.
    pub max_live_interval: usize,
    pub regions: Vec<ScratchRegion>,
}

impl ScratchPlan {
    /// Build the deterministic arena report.
    #[cfg(test)]
    fn arena_report(&self) -> ArenaReport {
        let max_live_interval = self
            .regions
            .iter()
            .map(|region| region.last_op.saturating_sub(region.first_op))
            .max()
            .unwrap_or(0);
        ArenaReport {
            total_bytes: self.total_bytes,
            alignment: self.alignment,
            seq_capacity: self.seq_capacity,
            region_count: self.regions.len(),
            max_live_interval,
            regions: self.regions.clone(),
        }
    }
}

/// Reusable aligned decode scratch arena (contract section 11): allocated
/// once before decode begins, then sliced by region name with pure offset
/// arithmetic. No heap allocation happens in the steady-state token loop.
pub struct DecodeArena {
    storage: Vec<u8>,
    /// Front padding so `storage[pad..]` starts on the alignment boundary.
    pad: usize,
    /// (offset, size, alignment) per region, in plan order.
    regions: Vec<(usize, usize, usize)>,
    total_bytes: usize,
}

impl DecodeArena {
    /// Allocate the arena from a scratch plan. Offsets are already
    /// deterministic and aligned; this adds base-pointer alignment.
    pub fn new(scratch: &ScratchPlan) -> Self {
        let alignment = scratch.alignment.max(4);
        let capacity = scratch.total_bytes + alignment;
        let storage = vec![0u8; capacity];
        let base = storage.as_ptr() as usize;
        let aligned = base.div_ceil(alignment) * alignment;
        let pad = aligned - base;
        let regions = scratch
            .regions
            .iter()
            .map(|r| (r.offset, r.size, r.alignment))
            .collect();
        Self {
            storage,
            pad,
            regions,
            total_bytes: scratch.total_bytes,
        }
    }

    /// Total usable arena bytes.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    fn region(&self, index: usize) -> Result<(usize, usize, usize), String> {
        self.regions
            .get(index)
            .copied()
            .ok_or_else(|| format!("decode arena region index {index} out of range"))
    }

    /// Mutable f32 view of one region.
    pub fn region_f32(&mut self, region: usize) -> Result<&mut [f32], String> {
        let (offset, size, _) = self.region(region)?;
        if size % 4 != 0 {
            return Err(format!("region {region} size {size} is not f32-aligned"));
        }
        let start = self.pad + offset;
        let bytes = &mut self.storage[start..start + size];
        // SAFETY: the arena base (storage[pad..]) is aligned to
        // scratch.alignment (>= 4), region offsets are aligned the same way
        // by the planner, and region sizes are multiples of 4 (they are
        // sized in f32 elements). The slice is therefore a valid aligned
        // f32 slice with no dangling or aliased storage.
        Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, size / 4) })
    }

    /// Disjoint mutable f32 views of several regions, in request order, as a
    /// fixed-size array (no allocation in the decode loop).
    ///
    /// Regions with identical offsets would alias; that is an illegal
    /// aliasing error (the planner must prove disjoint lifetimes before
    /// sharing storage, and shared regions must never be requested
    /// simultaneously).
    pub fn regions_f32<const N: usize>(
        &mut self,
        regions: [usize; N],
    ) -> Result<[&mut [f32]; N], String> {
        let mut entries: [(usize, usize, usize); N] = [(0, 0, 0); N]; // (offset, size, request index)
        for (request, &index) in regions.iter().enumerate() {
            let (offset, size, _) = self.region(index)?;
            if size % 4 != 0 {
                return Err(format!("region {index} size {size} is not f32-aligned"));
            }
            entries[request] = (offset, size, request);
        }
        entries.sort_unstable_by_key(|entry| entry.0);
        for window in entries.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(format!(
                    "illegal aliasing: regions requested at offset {} twice",
                    window[0].0
                ));
            }
        }

        let base = self.pad;
        let mut remaining = &mut self.storage[base..base + self.total_bytes];
        let mut placed: [Option<&mut [f32]>; N] = core::array::from_fn(|_| None);
        let mut cursor = 0usize;
        for (offset, size, request) in entries {
            let gap = offset - cursor;
            let (_, tail) = remaining.split_at_mut(gap);
            let (region_bytes, rest) = tail.split_at_mut(size);
            remaining = rest;
            // SAFETY: same alignment argument as [`DecodeArena::region_f32`];
            // `region_bytes` is a disjoint slice of the storage.
            let slice = unsafe {
                std::slice::from_raw_parts_mut(region_bytes.as_mut_ptr() as *mut f32, size / 4)
            };
            placed[request] = Some(slice);
            cursor = offset + size;
        }
        Ok(placed.map(|slice| slice.expect("every requested region placed")))
    }
}

// ---------------------------------------------------------------------------
// hooks
// ---------------------------------------------------------------------------

/// One resolved hook site (contract section 4/12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookSiteRecord {
    /// Stage id: `before-layer` | `after-attention` | `after-mlp` |
    /// `after-layer` | `before-logits` | `after-logits`.
    pub stage: String,
    /// Layer index for layer-level stages.
    pub layer: Option<usize>,
    /// Tensor id of the observed value when materialized.
    pub tensor: Option<usize>,
    /// Whether the tensor is a real materialized value at the call site.
    pub materialized: bool,
    /// Route that produces it: `unfused` | `fused`.
    pub route: String,
}

/// Hook-site plan: mode plus the resolved active sites (contract section 12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookSitePlan {
    pub mode: HookMode,
    /// Active stage ids (empty when mode is `Disabled`).
    pub active: Vec<String>,
    /// One record per supported site; inactive sites carry
    /// `materialized: false` and no tensor.
    pub sites: Vec<HookSiteRecord>,
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

/// One tensor's resolved dispatch entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelEntry {
    pub tensor: usize,
    pub kernel: KernelId,
    pub cpu_feature: Option<String>,
    /// Documented fallback reason, when this entry deviates from the
    /// requested execution (contract section 8: no silent fallback).
    pub fallback: Option<String>,
}

/// Resolved dispatch plan (contract section 10).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchPlan {
    pub kernel_per_tensor: Vec<KernelEntry>,
    /// `serial` | `row-parallel-rayon` (thread strategy from the model).
    pub thread_strategy: String,
}

// ---------------------------------------------------------------------------
// per-layer ops
// ---------------------------------------------------------------------------

/// A single planned operation. The llama-family decode interpreter matches
/// on the variant and resolves concrete weights/regions via `layer` context
/// and the tensor/region tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case")]
pub enum PlannedOp {
    /// Embedding row lookup. `tensor` is the embedding weight; `out` the
    /// `[seq, embed]` activation.
    Embedding { tensor: TensorRef, out: TensorRef },
    /// RMSNorm. `weight` F32; `input` -> `out`.
    RmsNorm {
        weight: TensorRef,
        input: TensorRef,
        out: TensorRef,
    },
    /// Quantized (or dense) linear projection. `fused_rms_norm` is set for
    /// fusion F1 (the norm weight folded into the projection); `bias` is
    /// set for projections carrying an F32 bias (qwen2.5 q/k/v).
    Matvec {
        weight: TensorRef,
        input: TensorRef,
        out: TensorRef,
        kernel: KernelId,
        fused_rms_norm: Option<TensorRef>,
        bias: Option<TensorRef>,
    },
    /// RoPE (+ optional qk-norm per architecture). Applied in place on
    /// `target`.
    Rope {
        target: TensorRef,
        rope_layout: String,
        qk_norm: Option<TensorRef>,
        qk_norm_order: String,
    },
    /// KV cache store of `k`/`v` at the current cursor.
    KvStore { k: TensorRef, v: TensorRef },
    /// Causal attention over the cache. `q` -> `out`; `score_scratch` names
    /// the arena region reused for attention scores. `rope_q` is set for
    /// fusion F4: the attention applies the Q RoPE (and optional qk-norm)
    /// internally instead of a separate rope op. The K rope is never merged
    /// (the stored K must be roped before the KV store).
    Attention {
        q: TensorRef,
        out: TensorRef,
        score_scratch: String,
        rope_q: bool,
    },
    /// SiLU activation, in place.
    Silu { target: TensorRef },
    /// Elementwise multiply, `a * b` -> `out`.
    Elemul {
        a: TensorRef,
        b: TensorRef,
        out: TensorRef,
    },
    /// `out = a + b`.
    ResidualAdd {
        a: TensorRef,
        b: TensorRef,
        out: TensorRef,
    },
    /// `out = (a + b) + c` — the final residual add when fusion F2 skips
    /// the standalone attention residual (left-associative, bit-identical
    /// to the unfused composition).
    ResidualAdd3 {
        a: TensorRef,
        b: TensorRef,
        c: TensorRef,
        out: TensorRef,
    },
    /// Final RMSNorm (before-logits stage input).
    OutputNorm {
        weight: TensorRef,
        input: TensorRef,
        out: TensorRef,
    },
    /// LM head projection.
    Logits {
        weight: TensorRef,
        input: TensorRef,
        out: TensorRef,
        tied: bool,
    },
    /// A fused operation from the frozen set. `components` are op indices
    /// (within the same layer's op list) folded into the fusion;
    /// `eliminated` lists tensor ids whose standalone materialization the
    /// fusion removes.
    Fused {
        fused: FusedOpKind,
        components: Vec<usize>,
        eliminated: Vec<TensorRef>,
        kernel: KernelId,
    },
    /// Fusion F1+F3: one orchestration pass over the block input computes
    /// the RMSNorm scale, writes the scaled activation into `scaled`, and
    /// runs the Q/K/V projections from it (shared dispatch, one norm pass
    /// instead of one norm plus three separate dispatches).
    FusedQkv {
        input: TensorRef,
        norm: TensorRef,
        scaled: TensorRef,
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
    },
    /// Fusion F5: the output projection accumulates directly into the
    /// residual destination (`out` starts as a copy of `residual`, the
    /// matvec kernel accumulates W·`attn` on top). Requires
    /// `after_attention` inactive for the layer (the o tensor is
    /// eliminated).
    FusedOProjResidual {
        attn: TensorRef,
        residual: TensorRef,
        out: TensorRef,
    },
    /// Fusion F2: `out = rmsnorm(residual_a + residual_b)` computed in one
    /// pass (no standalone residual materialization). Used on the attention
    /// residual when F5 is defused (after_attention active).
    FusedResidualNorm {
        a: TensorRef,
        b: TensorRef,
        weight: TensorRef,
        out: TensorRef,
    },
}

/// One layer's plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPlan {
    pub layer_index: usize,
    pub ops: Vec<PlannedOp>,
    pub fusion: FusionState,
    /// Why the layer is not fully fused (hook-driven de-fusion, fallback).
    pub fusion_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// model/architecture summaries
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufSummary {
    pub arch: String,
    pub block_count: usize,
    pub embedding_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub rope_dimension_count: usize,
    pub context_length: usize,
    pub rope_theta: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeSummary {
    /// `adjacent-pair` | `split-half`.
    pub layout: String,
    /// `before-rope` | `after-rope`.
    pub qk_norm_order: String,
    pub has_q_norm: bool,
    pub has_k_norm: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvLayout {
    pub precision: String,
    pub layout: String,
    /// Live-cache strides in scalar elements (not bytes).
    pub layer_stride: usize,
    pub head_stride: usize,
    pub pos_stride: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub max_seq: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSummary {
    /// Detected features (`avx2`, `fma`, `f16c`, `ssse3`).
    pub features: Vec<String>,
    pub threads: usize,
    /// Features the plan requires for its selected kernels.
    pub required: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanProvenance {
    pub ember_version: String,
    pub git_commit: String,
    pub rustc_version: String,
    /// ISO-8601 build time; zeroed for plan hashing.
    pub plan_build_time: String,
    pub execution_mode: ExecutionMode,
    pub hook_mode: HookMode,
    pub model_sha256: String,
    pub tokenizer_sha256: String,
}

// ---------------------------------------------------------------------------
// the plan
// ---------------------------------------------------------------------------

/// Immutable execution plan (contract section 10).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub schema_version: u32,
    /// Changes whenever a named kernel's numerical/runtime semantics change,
    /// even when the surrounding plan schema remains structurally compatible.
    /// Missing in historical v04-plan/1 payloads and therefore decoded as the
    /// legacy revision 1; the live interpreter rejects that revision.
    #[serde(
        default = "legacy_plan_kernel_revision",
        skip_serializing_if = "is_legacy_plan_kernel_revision"
    )]
    pub kernel_revision: u32,
    pub architecture: String,
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    pub gguf: GgufSummary,
    pub rope: RopeSummary,
    /// Ops executed before the first layer (embedding lookup).
    pub preamble: Vec<PlannedOp>,
    pub layers: Vec<LayerPlan>,
    /// Ops executed after the last layer (final norm, LM head).
    pub final_ops: Vec<PlannedOp>,
    pub tensor_table: Vec<TensorRecord>,
    pub scratch: ScratchPlan,
    pub kv: KvLayout,
    pub hook_sites: HookSitePlan,
    pub dispatch: DispatchPlan,
    pub cpu: CpuSummary,
    pub provenance: PlanProvenance,
    /// SHA-256 over the serialized plan with the timestamp zeroed; computed
    /// by [`ExecutionPlan::finalize`].
    pub plan_hash: String,
}

impl ExecutionPlan {
    /// Compute the deterministic plan hash: SHA-256 over the canonical JSON
    /// serialization with `plan_hash` and `plan_build_time` removed. The
    /// resulting hash is stable across identical plans regardless of build
    /// time; it is written back into the plan.
    pub fn finalize(mut self) -> Self {
        self.plan_hash = plan_hash(&self);
        self
    }

    /// Validate internal consistency. Called at plan build; the interpreter
    /// re-asserts the same invariants at decode time.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema_version == PLAN_SCHEMA_VERSION,
            "unexpected plan schema version {}",
            self.schema_version
        );
        anyhow::ensure!(
            self.kernel_revision == PLAN_KERNEL_REVISION,
            "unexpected kernel revision {}",
            self.kernel_revision
        );
        anyhow::ensure!(
            self.layers.len() == self.gguf.block_count,
            "plan layer count {} does not match gguf block_count {}",
            self.layers.len(),
            self.gguf.block_count
        );
        for (index, record) in self.tensor_table.iter().enumerate() {
            anyhow::ensure!(
                self.tensor_table[..index]
                    .iter()
                    .all(|prior| prior.id != record.id),
                "duplicate tensor table id {} at position {index}",
                record.id
            );
            let resolved = if record.name == "token_embd.weight" {
                resolve_embedding_kernel(&record.gguf_dtype, &record.execution).ok_or_else(
                    || {
                        anyhow::anyhow!(
                            "unsupported embedding dtype '{}' for '{}'",
                            record.gguf_dtype,
                            record.name
                        )
                    },
                )?
            } else {
                resolve_kernel(&record.gguf_dtype, &record.execution)
            };
            anyhow::ensure!(
                record.kernel == resolved,
                "tensor-table kernel {} disagrees with dtype/execution-derived kernel {} for '{}'",
                record.kernel.name(),
                resolved.name(),
                record.name
            );
        }
        let check_ref = |r: TensorRef, what: &str| -> anyhow::Result<()> {
            if self.tensor_table.iter().any(|t| t.id == r.id) {
                return Ok(());
            }
            anyhow::ensure!(
                self.scratch.tensor_regions.contains_key(&r.id),
                "{what} tensor id {} resolves to neither a weight nor a scratch region",
                r.id
            );
            Ok(())
        };
        let check_refs = |refs: &[TensorRef], what: &str| -> anyhow::Result<()> {
            for r in refs {
                check_ref(*r, what)?;
            }
            Ok(())
        };
        let check_op = |op: &PlannedOp| -> anyhow::Result<()> {
            match op {
                PlannedOp::Embedding { tensor, out } => {
                    check_ref(*tensor, "embedding weight")?;
                    check_ref(*out, "embedding output")
                }
                PlannedOp::RmsNorm { weight, input, out } => {
                    check_ref(*weight, "rmsnorm weight")?;
                    check_refs(&[*input, *out], "rmsnorm")
                }
                PlannedOp::Matvec {
                    weight,
                    input,
                    out,
                    kernel,
                    fused_rms_norm,
                    bias,
                } => {
                    check_ref(*weight, "matvec weight")?;
                    let record = self
                        .tensor_table
                        .iter()
                        .find(|record| record.id == weight.id)
                        .ok_or_else(|| {
                            anyhow::anyhow!("matvec weight {} is not resident", weight.id)
                        })?;
                    anyhow::ensure!(
                        *kernel == record.kernel,
                        "matvec kernel {} disagrees with tensor-table kernel {} for '{}'",
                        kernel.name(),
                        record.kernel.name(),
                        record.name
                    );
                    check_refs(&[*input, *out], "matvec")?;
                    if let Some(norm) = fused_rms_norm {
                        check_ref(*norm, "fused rmsnorm weight")?;
                    }
                    if let Some(bias) = bias {
                        check_ref(*bias, "matvec bias")?;
                    }
                    Ok(())
                }
                PlannedOp::Rope {
                    target, qk_norm, ..
                } => {
                    check_ref(*target, "rope target")?;
                    if let Some(norm) = qk_norm {
                        check_ref(*norm, "rope qk-norm weight")?;
                    }
                    Ok(())
                }
                PlannedOp::KvStore { k, v } => check_refs(&[*k, *v], "kv store"),
                PlannedOp::Attention {
                    q,
                    out,
                    score_scratch,
                    ..
                } => {
                    check_refs(&[*q, *out], "attention")?;
                    anyhow::ensure!(
                        self.scratch
                            .regions
                            .iter()
                            .any(|r| r.name == *score_scratch),
                        "attention score scratch region '{score_scratch}' not found"
                    );
                    Ok(())
                }
                PlannedOp::Silu { target } => check_ref(*target, "silu target"),
                PlannedOp::Elemul { a, b, out } => check_refs(&[*a, *b, *out], "elemul"),
                PlannedOp::ResidualAdd { a, b, out } => check_refs(&[*a, *b, *out], "residual add"),
                PlannedOp::ResidualAdd3 { a, b, c, out } => {
                    check_refs(&[*a, *b, *c, *out], "residual add (3 operand)")
                }
                PlannedOp::OutputNorm { weight, input, out } => {
                    check_ref(*weight, "output norm weight")?;
                    check_refs(&[*input, *out], "output norm")
                }
                PlannedOp::Logits {
                    weight, input, out, ..
                } => {
                    check_ref(*weight, "logits weight")?;
                    check_refs(&[*input, *out], "logits")
                }
                PlannedOp::Fused {
                    components,
                    eliminated,
                    ..
                } => {
                    for component in components {
                        anyhow::ensure!(
                            *component < self.operation_count(),
                            "fused op component index {component} out of range"
                        );
                    }
                    for tensor in eliminated {
                        check_ref(*tensor, "fused eliminated tensor")?;
                    }
                    Ok(())
                }
                PlannedOp::FusedQkv {
                    input,
                    norm,
                    scaled,
                    q,
                    k,
                    v,
                    ..
                } => {
                    check_refs(&[*input, *scaled, *q, *k, *v], "fused qkv")?;
                    check_ref(*norm, "fused qkv norm weight")
                }
                PlannedOp::FusedOProjResidual {
                    attn,
                    residual,
                    out,
                    ..
                } => check_refs(&[*attn, *residual, *out], "fused output projection"),
                PlannedOp::FusedResidualNorm { a, b, weight, out } => {
                    check_refs(&[*a, *b, *out], "fused residual norm")?;
                    check_ref(*weight, "fused residual norm weight")
                }
            }
        };
        for op in self.preamble.iter().chain(self.final_ops.iter()) {
            check_op(op)?;
        }
        for (index, layer) in self.layers.iter().enumerate() {
            anyhow::ensure!(
                layer.layer_index == index,
                "layer record at position {index} claims index {}",
                layer.layer_index
            );
            for op in &layer.ops {
                check_op(op)?;
            }
        }
        anyhow::ensure!(
            self.dispatch.kernel_per_tensor.len() == self.tensor_table.len(),
            "dispatch has {} entries for {} resident tensors",
            self.dispatch.kernel_per_tensor.len(),
            self.tensor_table.len()
        );
        for (index, entry) in self.dispatch.kernel_per_tensor.iter().enumerate() {
            anyhow::ensure!(
                self.dispatch.kernel_per_tensor[..index]
                    .iter()
                    .all(|prior| prior.tensor != entry.tensor),
                "duplicate dispatch entry for tensor {}",
                entry.tensor
            );
            let record = self
                .tensor_table
                .iter()
                .find(|record| record.id == entry.tensor)
                .ok_or_else(|| {
                    anyhow::anyhow!("dispatch entry references unknown tensor {}", entry.tensor)
                })?;
            anyhow::ensure!(
                entry.kernel == record.kernel,
                "dispatch kernel {} disagrees with tensor-table kernel {} for '{}'",
                entry.kernel.name(),
                record.kernel.name(),
                record.name
            );
            anyhow::ensure!(
                entry.cpu_feature.as_deref() == entry.kernel.cpu_feature(),
                "dispatch CPU feature {:?} disagrees with kernel requirement {:?} for '{}'",
                entry.cpu_feature,
                entry.kernel.cpu_feature(),
                record.name
            );
        }
        let required_from_dispatch: BTreeSet<&str> = self
            .dispatch
            .kernel_per_tensor
            .iter()
            .filter_map(|entry| entry.kernel.cpu_feature())
            .collect();
        let recorded_required: BTreeSet<&str> =
            self.cpu.required.iter().map(String::as_str).collect();
        anyhow::ensure!(
            self.cpu.required.len() == recorded_required.len(),
            "CPU requirement list contains duplicates"
        );
        anyhow::ensure!(
            recorded_required == required_from_dispatch,
            "CPU requirements {:?} disagree with dispatch-derived requirements {:?}",
            self.cpu.required,
            required_from_dispatch
        );
        for requirement in &recorded_required {
            for feature in requirement.split('+') {
                anyhow::ensure!(
                    self.cpu.features.iter().any(|present| present == feature),
                    "plan requires CPU feature '{feature}' but its detected feature record omits it"
                );
            }
            if *requirement == "avx2+fma+f16c+ssse3" {
                anyhow::ensure!(
                    crate::k_quant_matmul::x86_k_supported(),
                    "plan requires AVX2/FMA/F16C/SSSE3 but the current CPU lacks the tier"
                );
            }
        }
        let expected_thread_strategy = if self.cpu.threads > 1 {
            "column-parallel-rayon"
        } else {
            "serial"
        };
        anyhow::ensure!(
            self.dispatch.thread_strategy == expected_thread_strategy,
            "thread strategy '{}' disagrees with recorded Rayon thread count {}",
            self.dispatch.thread_strategy,
            self.cpu.threads
        );
        for site in &self.hook_sites.sites {
            if let Some(layer) = site.layer {
                anyhow::ensure!(
                    layer < self.layers.len(),
                    "hook site '{}' references out-of-range layer {layer}",
                    site.stage
                );
            }
            if let Some(tensor) = site.tensor {
                anyhow::ensure!(
                    self.tensor_table.iter().any(|t| t.id == tensor)
                        || self.scratch.tensor_regions.contains_key(&tensor),
                    "hook site references unknown tensor {tensor}"
                );
            }
        }
        Ok(())
    }

    /// Deterministic hash input: the JSON value with timestamp and hash
    /// zeroed.
    fn hash_input_json(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).expect("execution plan serializes");
        if let Some(obj) = value.as_object_mut() {
            obj.insert("plan_hash".into(), serde_json::Value::String(String::new()));
            if let Some(provenance) = obj.get_mut("provenance").and_then(|p| p.as_object_mut()) {
                provenance.insert(
                    "plan_build_time".into(),
                    serde_json::Value::String(String::new()),
                );
            }
        }
        value
    }

    /// Total number of planned operations (preamble + layers + final).
    pub fn operation_count(&self) -> usize {
        self.preamble.len()
            + self
                .layers
                .iter()
                .map(|layer| layer.ops.len())
                .sum::<usize>()
            + self.final_ops.len()
    }

    /// Fused op count (for diagnostics).
    fn fused_op_count(&self) -> usize {
        self.preamble
            .iter()
            .chain(self.final_ops.iter())
            .chain(self.layers.iter().flat_map(|layer| layer.ops.iter()))
            .map(|op| match op {
                PlannedOp::FusedQkv { .. } => 2,                // F1 + F3
                PlannedOp::Attention { rope_q: true, .. } => 1, // F4
                PlannedOp::FusedOProjResidual { .. } => 1,      // F5
                PlannedOp::FusedResidualNorm { .. } => 1,       // F2
                PlannedOp::Fused { .. } => 1,
                _ => 0,
            })
            .sum()
    }

    /// Layers not fully fused, with reasons.
    fn defused_layers(&self) -> Vec<(usize, String)> {
        self.layers
            .iter()
            .filter(|layer| layer.fusion != FusionState::Fused)
            .filter_map(|layer| {
                layer
                    .fusion_reason
                    .clone()
                    .map(|reason| (layer.layer_index, reason))
            })
            .collect()
    }

    /// Human-readable summary for `ember inspect-plan`.
    pub fn to_summary_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "execution plan (schema v04-plan/{}, kernel revision {}) hash {}\n",
            self.schema_version, self.kernel_revision, self.plan_hash
        ));
        out.push_str(&format!(
            "architecture: {}  execution: {}  hook mode: {:?}\n",
            self.architecture,
            self.provenance.execution_mode.name(),
            self.provenance.hook_mode
        ));
        out.push_str(&format!(
            "model sha256: {}  tokenizer sha256: {}\n",
            short_sha(&self.model_sha256),
            short_sha(&self.tokenizer_sha256)
        ));
        out.push_str(&format!(
            "gguf: {} layers, embed {}, heads {}/{} kv, ffn {}, vocab {}, ctx {}\n",
            self.gguf.block_count,
            self.gguf.embedding_length,
            self.gguf.head_count,
            self.gguf.head_count_kv,
            self.gguf.ffn_dim,
            self.gguf.vocab_size,
            self.gguf.context_length
        ));
        out.push_str(&format!(
            "rope: {} qk-norm {} (q_norm {} k_norm {})\n",
            self.rope.layout, self.rope.qk_norm_order, self.rope.has_q_norm, self.rope.has_k_norm
        ));
        out.push_str(&format!(
            "operations: {} total ({} preamble, {} final), {} fused, {} defused layers\n",
            self.operation_count(),
            self.preamble.len(),
            self.final_ops.len(),
            self.fused_op_count(),
            self.defused_layers().len()
        ));
        for (layer, reason) in self.defused_layers() {
            out.push_str(&format!("  layer {layer} defused: {reason}\n"));
        }
        out.push_str(&format!(
            "scratch: {} bytes ({} regions, align {})\n",
            self.scratch.total_bytes,
            self.scratch.regions.len(),
            self.scratch.alignment
        ));
        out.push_str(&format!(
            "kv: {} precision, {} layout, head_dim {}, {} kv heads, max_seq {}\n",
            self.kv.precision,
            self.kv.layout,
            self.kv.head_dim,
            self.kv.n_kv_heads,
            self.kv.max_seq
        ));
        out.push_str(&format!(
            "dispatch: {} tensors, thread strategy {}\n",
            self.dispatch.kernel_per_tensor.len(),
            self.dispatch.thread_strategy
        ));
        let mut kernels: BTreeMap<&'static str, usize> = BTreeMap::new();
        for entry in &self.dispatch.kernel_per_tensor {
            *kernels
                .entry(entry.kernel.name_for_revision(self.kernel_revision))
                .or_default() += 1;
        }
        for (kernel, count) in kernels {
            out.push_str(&format!("  kernel {kernel}: {count} tensors\n"));
        }
        out.push_str(&format!(
            "cpu: features {:?}, threads {}, required {:?}\n",
            self.cpu.features, self.cpu.threads, self.cpu.required
        ));
        let fallbacks: Vec<&KernelEntry> = self
            .dispatch
            .kernel_per_tensor
            .iter()
            .filter(|entry| entry.fallback.is_some())
            .collect();
        if fallbacks.is_empty() {
            out.push_str("fallbacks: none\n");
        } else {
            out.push_str(&format!("fallbacks: {} entries\n", fallbacks.len()));
            for entry in fallbacks {
                out.push_str(&format!(
                    "  tensor {} ({}): {}\n",
                    entry.tensor,
                    entry.kernel.name_for_revision(self.kernel_revision),
                    entry.fallback.as_deref().unwrap_or("")
                ));
            }
        }
        out.push_str(&format!(
            "hook sites: mode {:?}, active {:?}\n",
            self.provenance.hook_mode, self.hook_sites.active
        ));
        out.push_str(&format!(
            "ember {} commit {} rustc {}\n",
            self.provenance.ember_version,
            short_sha(&self.provenance.git_commit),
            self.provenance.rustc_version
        ));
        out
    }
}

/// Deterministic SHA-256 over the plan's canonical hash-input JSON
/// (`ExecutionPlan::hash_input_json`, private by design).
pub fn plan_hash(plan: &ExecutionPlan) -> String {
    let mut input = plan.hash_input_json();
    sort_value_keys(&mut input);
    let bytes = serde_json::to_vec(&input).expect("plan hash input serializes");
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    hex(&hasher.finalize())
}

/// Recursively sort every object key in a JSON value so serialization is
/// deterministic regardless of serde_json's `preserve_order` feature. That
/// feature swaps serde_json's default sorted `Map` for an insertion-ordered
/// one and is enabled transitively by some dependencies (gpui); the v0.5
/// canonical-JSON contract requires sorted keys, so sort explicitly here.
pub(crate) fn sort_value_keys(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(String, serde_json::Value)> =
                std::mem::take(map).into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            for (_, child) in entries.iter_mut() {
                sort_value_keys(child);
            }
            *map = entries.into_iter().collect();
        }
        serde_json::Value::Array(items) => {
            for item in items.iter_mut() {
                sort_value_keys(item);
            }
        }
        _ => {}
    }
}

fn short_sha(sha: &str) -> String {
    if sha.len() > 12 {
        sha[..12].to_string()
    } else {
        sha.to_string()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn sample_plan(execution: ExecutionMode, hook: HookMode) -> ExecutionPlan {
        ExecutionPlan {
            schema_version: PLAN_SCHEMA_VERSION,
            kernel_revision: PLAN_KERNEL_REVISION,
            architecture: "llama".into(),
            model_sha256: "aa".repeat(32),
            tokenizer_sha256: "bb".repeat(32),
            gguf: GgufSummary {
                arch: "llama".into(),
                block_count: 1,
                embedding_length: 256,
                head_count: 8,
                head_count_kv: 8,
                ffn_dim: 683,
                vocab_size: 2048,
                rope_dimension_count: 64,
                context_length: 2048,
                rope_theta: 10_000.0,
            },
            rope: RopeSummary {
                layout: "adjacent-pair".into(),
                qk_norm_order: "after-rope".into(),
                has_q_norm: false,
                has_k_norm: false,
            },
            preamble: vec![PlannedOp::Embedding {
                tensor: TensorRef::new(0),
                out: TensorRef::new(10),
            }],
            layers: vec![LayerPlan {
                layer_index: 0,
                ops: vec![
                    PlannedOp::RmsNorm {
                        weight: TensorRef::new(1),
                        input: TensorRef::new(10),
                        out: TensorRef::new(11),
                    },
                    PlannedOp::Matvec {
                        weight: TensorRef::new(2),
                        input: TensorRef::new(11),
                        out: TensorRef::new(12),
                        kernel: KernelId::KQuantScalarQ6K,
                        fused_rms_norm: None,
                        bias: None,
                    },
                    PlannedOp::ResidualAdd {
                        a: TensorRef::new(10),
                        b: TensorRef::new(12),
                        out: TensorRef::new(13),
                    },
                ],
                fusion: FusionState::Unfused,
                fusion_reason: None,
            }],
            final_ops: vec![PlannedOp::OutputNorm {
                weight: TensorRef::new(3),
                input: TensorRef::new(13),
                out: TensorRef::new(14),
            }],
            tensor_table: vec![],
            scratch: ScratchPlan {
                total_bytes: 4096,
                alignment: 64,
                seq_capacity: 1,
                regions: vec![
                    ScratchRegion {
                        name: "t10".into(),
                        offset: 0,
                        size: 1024,
                        alignment: 64,
                        first_op: 0,
                        last_op: 2,
                        shared_with: None,
                    },
                    ScratchRegion {
                        name: "t12".into(),
                        offset: 1024,
                        size: 1024,
                        alignment: 64,
                        first_op: 1,
                        last_op: 1,
                        shared_with: None,
                    },
                ],
                tensor_regions: BTreeMap::from([(10, "t10".into()), (12, "t12".into())]),
            },
            kv: KvLayout {
                precision: "f16".into(),
                layout: "layer-head-pos-dim".into(),
                layer_stride: 8 * 2048 * 64,
                head_stride: 2048 * 64,
                pos_stride: 64,
                head_dim: 64,
                n_kv_heads: 8,
                max_seq: 2048,
            },
            hook_sites: HookSitePlan {
                mode: hook,
                active: vec![],
                sites: vec![HookSiteRecord {
                    stage: "after-attention".into(),
                    layer: Some(0),
                    tensor: Some(12),
                    materialized: true,
                    route: "unfused".into(),
                }],
            },
            dispatch: DispatchPlan {
                kernel_per_tensor: vec![KernelEntry {
                    tensor: 2,
                    kernel: KernelId::KQuantScalarQ6K,
                    cpu_feature: None,
                    fallback: None,
                }],
                thread_strategy: "serial".into(),
            },
            cpu: CpuSummary {
                features: vec!["avx2".into(), "fma".into(), "f16c".into(), "ssse3".into()],
                threads: 8,
                required: vec![],
            },
            provenance: PlanProvenance {
                ember_version: env!("CARGO_PKG_VERSION").into(),
                git_commit: "0000000000000000000000000000000000000000".into(),
                rustc_version: "test".into(),
                plan_build_time: "2026-08-04T00:00:00Z".into(),
                execution_mode: execution,
                hook_mode: hook,
                model_sha256: "aa".repeat(32),
                tokenizer_sha256: "bb".repeat(32),
            },
            plan_hash: String::new(),
        }
    }

    #[test]
    fn serialization_is_deterministic() {
        let a = sample_plan(ExecutionMode::Planned, HookMode::Disabled).finalize();
        let b = sample_plan(ExecutionMode::Planned, HookMode::Disabled).finalize();
        let json_a = serde_json::to_string_pretty(&a).unwrap();
        let json_b = serde_json::to_string_pretty(&b).unwrap();
        assert_eq!(json_a, json_b, "identical plans must serialize identically");
    }

    #[test]
    fn plan_hash_is_stable_across_timestamps() {
        let mut a = sample_plan(ExecutionMode::Planned, HookMode::Disabled);
        let mut b = sample_plan(ExecutionMode::Planned, HookMode::Disabled);
        a.provenance.plan_build_time = "2026-08-04T00:00:00Z".into();
        b.provenance.plan_build_time = "2026-12-31T23:59:59Z".into();
        let a = a.finalize();
        let b = b.finalize();
        assert_eq!(a.plan_hash, b.plan_hash, "hash must ignore build time");
        assert_eq!(a.plan_hash.len(), 64, "sha256 hex length");
    }

    #[test]
    fn plan_hash_changes_with_content() {
        let mut a = sample_plan(ExecutionMode::Planned, HookMode::Disabled);
        let mut b = sample_plan(ExecutionMode::Planned, HookMode::Disabled);
        a.gguf.embedding_length = 512;
        b.gguf.embedding_length = 256;
        assert_ne!(a.finalize().plan_hash, b.finalize().plan_hash);
    }

    #[test]
    fn kernel_revision_is_hashed_and_legacy_serialization_stays_readable() {
        let current = sample_plan(ExecutionMode::Planned, HookMode::Disabled).finalize();
        let mut legacy = sample_plan(ExecutionMode::Planned, HookMode::Disabled);
        legacy.kernel_revision = 1;
        let legacy = legacy.finalize();
        assert_ne!(current.plan_hash, legacy.plan_hash);
        let legacy_json = serde_json::to_string(&legacy).unwrap();
        assert!(!legacy_json.contains("kernel_revision"));
        assert!(legacy.to_summary_text().contains("kernel scalar-q6k"));
        assert!(!legacy.to_summary_text().contains("q6-k-q8-k-scalar"));
        let decoded: ExecutionPlan = serde_json::from_str(&legacy_json).unwrap();
        assert_eq!(decoded.kernel_revision, 1);
        assert_eq!(
            plan_hash(&decoded),
            legacy.plan_hash,
            "omitted legacy field must preserve the historical hash"
        );
        assert!(decoded
            .validate()
            .unwrap_err()
            .to_string()
            .contains("kernel revision"));
        assert!(serde_json::to_string(&current)
            .unwrap()
            .contains("\"kernel_revision\":2"));
    }

    #[test]
    fn execution_mode_cli_parsing() {
        assert_eq!(
            ExecutionMode::from_cli("reference"),
            Ok(ExecutionMode::Reference)
        );
        assert_eq!(
            ExecutionMode::from_cli("planned"),
            Ok(ExecutionMode::Planned)
        );
        assert_eq!(
            ExecutionMode::from_cli("planned-fused"),
            Ok(ExecutionMode::PlannedFused)
        );
        assert!(ExecutionMode::from_cli("fast").is_err());
        for mode in [
            ExecutionMode::Reference,
            ExecutionMode::Planned,
            ExecutionMode::PlannedFused,
        ] {
            assert_eq!(ExecutionMode::from_cli(mode.name()), Ok(mode));
        }
    }

    #[test]
    fn kernel_resolution_matches_v03_strategy_contract() {
        assert_eq!(
            resolve_kernel("q4_k", "compressed_x86"),
            KernelId::KQuantAvx2Q4K
        );
        assert_eq!(
            resolve_kernel("q4_k", "compressed_scalar"),
            KernelId::KQuantScalarQ4K
        );
        assert_eq!(
            resolve_kernel("q6_k", "compressed_x86"),
            KernelId::KQuantAvx2Q6K
        );
        assert_eq!(
            resolve_kernel("q6_k", "compressed_scalar"),
            KernelId::KQuantScalarQ6K
        );
        assert_eq!(resolve_kernel("q8_0", "compressed_x86"), KernelId::Q8Packed);
        assert_eq!(resolve_kernel("q2_k", "eager_f32"), KernelId::EagerF32);
        assert_eq!(resolve_kernel("f32", "eager_f32"), KernelId::EagerF32);
        // fallback: compressed decision on a non-native dtype resolves eager
        assert_eq!(resolve_kernel("q5_k", "compressed_x86"), KernelId::EagerF32);
    }

    #[test]
    fn summary_text_renders() {
        let plan = sample_plan(ExecutionMode::Planned, HookMode::Disabled).finalize();
        let text = plan.to_summary_text();
        assert!(text.contains("execution plan (schema v04-plan/1, kernel revision 2)"));
        assert!(text.contains("operations: 5 total (1 preamble, 1 final)"));
        assert!(text.contains("scratch: 4096 bytes"));
        assert!(text.contains("fallbacks: none"));
    }

    #[test]
    fn arena_report_is_deterministic_and_complete() {
        let plan = sample_plan(ExecutionMode::Planned, HookMode::Disabled).finalize();
        let report = plan.scratch.arena_report();
        assert_eq!(report.total_bytes, 4096);
        assert_eq!(report.region_count, plan.scratch.regions.len());
        assert_eq!(report.alignment, 64);
        assert!(report.max_live_interval > 0);
        // every region has a deterministic offset and a recorded lifetime
        for region in &report.regions {
            assert!(region.offset < report.total_bytes);
            assert!(region.last_op >= region.first_op);
        }
        let json_a = serde_json::to_string(&report).unwrap();
        let json_b = serde_json::to_string(&plan.scratch.arena_report()).unwrap();
        assert_eq!(json_a, json_b, "arena report must be deterministic");
    }
}
