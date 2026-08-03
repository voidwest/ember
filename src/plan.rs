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

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Execution-plan schema version (`"v04-plan/1"`).
pub const PLAN_SCHEMA_VERSION: u32 = 1;

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
    /// Q8_0 packed native path (v0.3, unchanged; not planned in v0.4).
    Q8Packed,
    /// Compressed-resident Q4_K scalar kernel.
    KQuantScalarQ4K,
    /// Compressed-resident Q6_K scalar kernel.
    KQuantScalarQ6K,
    /// Compressed-resident Q4_K AVX2 kernel.
    KQuantAvx2Q4K,
    /// Compressed-resident Q6_K AVX2 kernel.
    KQuantAvx2Q6K,
}

impl KernelId {
    /// Stable short name for provenance and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::EagerF32 => "eager-f32",
            Self::Q8Packed => "q8-packed",
            Self::KQuantScalarQ4K => "scalar-q4k",
            Self::KQuantScalarQ6K => "scalar-q6k",
            Self::KQuantAvx2Q4K => "avx2-q4k",
            Self::KQuantAvx2Q6K => "avx2-q6k",
        }
    }

    /// CPU feature requirement, if any.
    pub fn cpu_feature(self) -> Option<&'static str> {
        match self {
            Self::KQuantAvx2Q4K | Self::KQuantAvx2Q6K => Some("avx2+fma+f16c"),
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
    /// the arena region reused for attention scores.
    Attention {
        q: TensorRef,
        out: TensorRef,
        score_scratch: String,
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
    pub layer_stride: usize,
    pub head_stride: usize,
    pub pos_stride: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub max_seq: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSummary {
    /// Detected features (`avx2`, `fma`, `f16c`).
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
                    fused_rms_norm,
                    bias,
                    ..
                } => {
                    check_ref(*weight, "matvec weight")?;
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
            }
        };
        for op in self.preamble.iter().chain(self.final_ops.iter()) {
            check_op(op)?;
        }
        for layer in &self.layers {
            for op in &layer.ops {
                check_op(op)?;
            }
        }
        for entry in &self.dispatch.kernel_per_tensor {
            anyhow::ensure!(
                self.tensor_table.iter().any(|t| t.id == entry.tensor),
                "dispatch entry references unknown tensor {}",
                entry.tensor
            );
        }
        for site in &self.hook_sites.sites {
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
    pub fn fused_op_count(&self) -> usize {
        self.preamble
            .iter()
            .chain(self.final_ops.iter())
            .chain(self.layers.iter().flat_map(|layer| layer.ops.iter()))
            .filter(|op| matches!(op, PlannedOp::Fused { .. }))
            .count()
    }

    /// Layers not fully fused, with reasons.
    pub fn defused_layers(&self) -> Vec<(usize, String)> {
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
            "execution plan (schema v04-plan/{}) hash {}\n",
            self.schema_version, self.plan_hash
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
            *kernels.entry(entry.kernel.name()).or_default() += 1;
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
                    "  tensor {} ({:?}): {}\n",
                    entry.tensor,
                    entry.kernel,
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

/// Deterministic SHA-256 over [`ExecutionPlan::hash_input_json`].
pub fn plan_hash(plan: &ExecutionPlan) -> String {
    let input = plan.hash_input_json();
    let bytes = serde_json::to_vec(&input).expect("plan hash input serializes");
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    hex(&hasher.finalize())
}

fn short_sha(sha: &str) -> String {
    if sha.len() > 12 {
        sha[..12].to_string()
    } else {
        sha.to_string()
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_plan(execution: ExecutionMode, hook: HookMode) -> ExecutionPlan {
        ExecutionPlan {
            schema_version: PLAN_SCHEMA_VERSION,
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
                layer_stride: 8 * 2048 * 64 * 2,
                head_stride: 2048 * 64 * 2,
                pos_stride: 64 * 2,
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
                features: vec!["avx2".into(), "fma".into(), "f16c".into()],
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
        assert!(text.contains("execution plan (schema v04-plan/1)"));
        assert!(text.contains("operations: 5 total (1 preamble, 1 final)"));
        assert!(text.contains("scratch: 4096 bytes"));
        assert!(text.contains("fallbacks: none"));
    }
}
