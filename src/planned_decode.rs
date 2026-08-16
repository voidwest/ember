//! v0.4 planned-decode interpreter (contract: `docs/v04-execution-contract.md`,
//! sections 9-11).
//!
//! This is the *execution engine* for plan-driven single-token decode: it
//! resolves an [`ExecutionPlan`] into per-layer [`PlannedOp`] sequences
//! (`ResolvedOps`), holds the [`DecodeArena`] session (`PlannedDecodeState`),
//! and implements the planned + fused kernels (`planned_linear_into`,
//! `planned_causal_attention`, the frozen F1-F5 fusion set) plus the
//! `forward_last_logits_planned` entry point.
//!
//! Split out of `src/llama.rs` in the Luminal-review cleanup: the model file
//! keeps model definition, eager forward, and plan *construction* (which is
//! model-introspection code that walks the model's private structure and
//! therefore lives with the model); the interpreter only consumes the
//! finished plan plus the model's public tensors, so it stands alone here.
//!
//! Determinism contract: nothing in this module may depend on wall-clock
//! time, thread scheduling, or hash-map iteration order; plan hashes and
//! Gate-E allocation counts are frozen in `tests/k_parity.rs`.

use crate::backend::{CpuBackend, CpuError};
use crate::experiments::{ActiveHooks, ExecutionContext, LayerHooks, SliceActivation};
use crate::llama::{k_execution_name, Llama, LlamaEmbedding};
use crate::model::{Linear, WeightKindView};
use crate::plan::{
    resolve_embedding_kernel, resolve_kernel, DecodeArena, ExecutionMode, ExecutionPlan, HookMode,
    KernelId, PlannedOp, TensorRef,
};
use crate::tensor::CpuTensor;
use alloc::vec::Vec;

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FusedExecutionCounts {
    pub f1_rmsnorm_linear: usize,
    pub f2_residual_rmsnorm: usize,
    pub f3_qkv_orchestration: usize,
    pub f4_rope_in_attention: usize,
    pub f5_output_proj_residual: usize,
}

#[cfg(test)]
thread_local! {
    static FUSED_EXECUTION_COUNTS: std::cell::Cell<FusedExecutionCounts> =
        const { std::cell::Cell::new(FusedExecutionCounts {
            f1_rmsnorm_linear: 0,
            f2_residual_rmsnorm: 0,
            f3_qkv_orchestration: 0,
            f4_rope_in_attention: 0,
            f5_output_proj_residual: 0,
        }) };
}

#[cfg(test)]
pub(crate) fn reset_fused_execution_counts() {
    FUSED_EXECUTION_COUNTS.with(|counts| counts.set(FusedExecutionCounts::default()));
}

#[cfg(test)]
pub(crate) fn fused_execution_counts() -> FusedExecutionCounts {
    FUSED_EXECUTION_COUNTS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record_fused_execution(update: impl FnOnce(&mut FusedExecutionCounts)) {
    FUSED_EXECUTION_COUNTS.with(|counts| {
        let mut current = counts.get();
        update(&mut current);
        counts.set(current);
    });
}

/// Which norm a resolved RmsNorm op reads (resolved once at session build,
/// never per token).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormRole {
    AttnIn,
    MlpIn,
    Output,
}

/// Which projection a resolved Matvec op reads (resolved once at session
/// build, never per token).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjRole {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
    Head,
}

/// Which RoPE application a resolved Rope op performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RopeRole {
    Q,
    K,
}

/// A plan op with every per-token lookup resolved to a region index and a
/// role discriminant. The decode loop walks these; it performs no string
/// lookup, no registry inspection, and no per-token shape rediscovery.
#[derive(Debug, Clone, Copy)]
enum ResolvedOp {
    Embed {
        out: usize,
        kernel: KernelId,
    },
    RmsNorm {
        role: NormRole,
        input: usize,
        out: usize,
    },
    Matvec {
        role: ProjRole,
        kernel: KernelId,
        input: usize,
        out: usize,
        has_bias: bool,
    },
    Rope {
        role: RopeRole,
        target: usize,
    },
    KvStore {
        k: usize,
        v: usize,
    },
    Attention {
        q: usize,
        out: usize,
        scores: usize,
        rope_q: bool,
    },
    Silu {
        target: usize,
    },
    Elemul {
        a: usize,
        b: usize,
        out: usize,
    },
    ResidualAdd {
        a: usize,
        b: usize,
        out: usize,
    },
    ResidualAdd3 {
        a: usize,
        b: usize,
        c: usize,
        out: usize,
    },
    Logits {
        input: usize,
        out: usize,
        tied: bool,
        kernel: KernelId,
    },
    /// Fusion F1+F3: norm-scale the block input into `scaled`, then run the
    /// Q/K/V projections from it.
    FusedQkv {
        input: usize,
        scaled: usize,
        q: usize,
        k: usize,
        v: usize,
        kernels: [KernelId; 3],
    },
    /// Fusion F5: `out` starts as a copy of `residual`; the matvec kernel
    /// accumulates W·`attn` on top.
    FusedOProjResidual {
        attn: usize,
        residual: usize,
        out: usize,
        kernel: KernelId,
    },
    /// Fusion F2: `out = rmsnorm(a + b)` in one pass.
    FusedResidualNorm {
        a: usize,
        b: usize,
        out: usize,
    },
}

/// Resolved op sequences (preamble / per-layer / final).
#[derive(Debug, Clone)]
struct ResolvedOps {
    preamble: Vec<ResolvedOp>,
    layers: Vec<ResolvedLayerOps>,
    final_ops: Vec<ResolvedOp>,
}

/// One resolved layer: the op sequence plus the arena region index for each
/// semantic hook site (contract section 4/12).
#[derive(Debug, Clone)]
struct ResolvedLayerOps {
    ops: Vec<ResolvedOp>,
    hooks: ResolvedHookSites,
}

/// Arena region indices for the six semantic hook sites. Resolved once at
/// session build so the decode loop fires hooks with no lookup.
#[derive(Debug, Clone, Copy)]
struct ResolvedHookSites {
    before_layer: usize,
    /// `None` when the layer's fusion eliminates the o tensor (F5 active);
    /// the hook is then inactive for this layer.
    after_attention: Option<usize>,
    after_mlp: usize,
    after_layer: usize,
}

/// The mutable half of a planned decode session: the arena plus the
/// resolved op table. Rebuilt only when the plan hash changes (arena size
/// or structure depends on the plan).
pub(crate) struct PlannedDecodeState {
    plan_hash: String,
    arena: DecodeArena,
    ops: ResolvedOps,
}

/// Resolve a plan into region indices and role discriminants. Runs once per
/// session; the decode loop then walks [`ResolvedOps`] with no lookups.
fn build_planned_state(plan: &ExecutionPlan) -> Result<PlannedDecodeState, CpuError> {
    // ---- phase 1: plan integrity (structural validation + hash) ----
    plan.validate()
        .map_err(|error| CpuError::Kernel(format!("invalid execution plan: {error}")))?;
    let recomputed_hash = crate::plan::plan_hash(plan);
    if plan.plan_hash != recomputed_hash {
        return Err(CpuError::Kernel(format!(
            "execution plan hash mismatch: recorded {} recomputed {recomputed_hash}",
            plan.plan_hash
        )));
    }
    // ---- phase 2: scratch arena + tensor-region resolution ----
    let arena = DecodeArena::new(&plan.scratch);
    let region_of = |tensor: TensorRef| -> Result<usize, CpuError> {
        let name = plan.scratch.tensor_regions.get(&tensor.id).ok_or_else(|| {
            CpuError::ShapeMismatch(format!("plan tensor {} has no scratch region", tensor.id))
        })?;
        plan.scratch
            .regions
            .iter()
            .position(|r| &r.name == name)
            .ok_or_else(|| CpuError::ShapeMismatch(format!("plan region '{name}' not found")))
    };
    let region_by_name = |name: &str| -> Result<usize, CpuError> {
        plan.scratch
            .regions
            .iter()
            .position(|r| r.name == name)
            .ok_or_else(|| CpuError::ShapeMismatch(format!("plan region '{name}' not found")))
    };
    let weight_name = |tensor: TensorRef| -> Result<&str, CpuError> {
        plan.tensor_table
            .iter()
            .find(|record| record.id == tensor.id)
            .map(|record| record.name.as_str())
            .ok_or_else(|| {
                CpuError::ShapeMismatch(format!(
                    "plan weight tensor {} not in tensor table",
                    tensor.id
                ))
            })
    };
    let kernel_by_name = |name: &str| -> Result<KernelId, CpuError> {
        plan.tensor_table
            .iter()
            .find(|record| record.name == name)
            .map(|record| record.kernel)
            .ok_or_else(|| {
                CpuError::ShapeMismatch(format!("plan kernel tensor '{name}' not found"))
            })
    };

    let resolve = |op: &PlannedOp,
                   _position: usize,
                   layer_index: Option<usize>|
     -> Result<ResolvedOp, CpuError> {
        match op {
            PlannedOp::Embedding { tensor, out } => Ok(ResolvedOp::Embed {
                out: region_of(*out)?,
                kernel: plan
                    .tensor_table
                    .iter()
                    .find(|record| record.id == tensor.id)
                    .map(|record| record.kernel)
                    .ok_or_else(|| {
                        CpuError::ShapeMismatch(format!(
                            "embedding weight tensor {} not in tensor table",
                            tensor.id
                        ))
                    })?,
            }),
            PlannedOp::RmsNorm { weight, input, out } => {
                let name = weight_name(*weight)?;
                let role = if name.ends_with("attn_norm.weight") {
                    NormRole::AttnIn
                } else if name.ends_with("ffn_norm.weight") {
                    NormRole::MlpIn
                } else if name.ends_with("output_norm.weight") {
                    NormRole::Output
                } else {
                    return Err(CpuError::ShapeMismatch(format!(
                        "unrecognized rms-norm weight '{name}'"
                    )));
                };
                Ok(ResolvedOp::RmsNorm {
                    role,
                    input: region_of(*input)?,
                    out: region_of(*out)?,
                })
            }
            PlannedOp::Matvec {
                weight,
                input,
                out,
                kernel,
                bias,
                ..
            } => {
                let name = weight_name(*weight)?;
                let role = if name.ends_with("attn_q.weight") {
                    ProjRole::Q
                } else if name.ends_with("attn_k.weight") {
                    ProjRole::K
                } else if name.ends_with("attn_v.weight") {
                    ProjRole::V
                } else if name.ends_with("attn_output.weight") {
                    ProjRole::O
                } else if name.ends_with("ffn_gate.weight") {
                    ProjRole::Gate
                } else if name.ends_with("ffn_up.weight") {
                    ProjRole::Up
                } else if name.ends_with("ffn_down.weight") {
                    ProjRole::Down
                } else if name.starts_with("output.weight") {
                    ProjRole::Head
                } else {
                    return Err(CpuError::ShapeMismatch(format!(
                        "unrecognized projection weight '{name}'"
                    )));
                };
                Ok(ResolvedOp::Matvec {
                    role,
                    kernel: *kernel,
                    input: region_of(*input)?,
                    out: region_of(*out)?,
                    has_bias: bias.is_some(),
                })
            }
            PlannedOp::Rope { target, .. } => {
                // the role is read from the target region name (".q" vs
                // ".k") so both the unfused (positions 4/5) and fused
                // (position 1, k only) layer shapes resolve identically
                let target_region = region_of(*target)?;
                let region_name = &plan.scratch.regions[target_region].name;
                let role = if region_name.ends_with(".k") {
                    RopeRole::K
                } else if region_name.ends_with(".q") {
                    RopeRole::Q
                } else {
                    return Err(CpuError::ShapeMismatch(format!(
                        "rope op targets unrecognized region '{region_name}'"
                    )));
                };
                Ok(ResolvedOp::Rope {
                    role,
                    target: target_region,
                })
            }
            PlannedOp::KvStore { k, v } => Ok(ResolvedOp::KvStore {
                k: region_of(*k)?,
                v: region_of(*v)?,
            }),
            PlannedOp::Attention {
                q,
                out,
                score_scratch,
                rope_q,
            } => Ok(ResolvedOp::Attention {
                q: region_of(*q)?,
                out: region_of(*out)?,
                scores: region_by_name(score_scratch)?,
                rope_q: *rope_q,
            }),
            PlannedOp::Silu { target } => Ok(ResolvedOp::Silu {
                target: region_of(*target)?,
            }),
            PlannedOp::Elemul { a, b, out } => Ok(ResolvedOp::Elemul {
                a: region_of(*a)?,
                b: region_of(*b)?,
                out: region_of(*out)?,
            }),
            PlannedOp::ResidualAdd { a, b, out } => Ok(ResolvedOp::ResidualAdd {
                a: region_of(*a)?,
                b: region_of(*b)?,
                out: region_of(*out)?,
            }),
            PlannedOp::ResidualAdd3 { a, b, c, out } => Ok(ResolvedOp::ResidualAdd3 {
                a: region_of(*a)?,
                b: region_of(*b)?,
                c: region_of(*c)?,
                out: region_of(*out)?,
            }),
            PlannedOp::FusedQkv {
                input,
                norm,
                scaled,
                q,
                k,
                v,
            } => {
                let norm_name = weight_name(*norm)?;
                let prefix = norm_name.strip_suffix(".attn_norm.weight").ok_or_else(|| {
                    CpuError::ShapeMismatch(format!(
                        "fused qkv norm has unrecognized name '{norm_name}'"
                    ))
                })?;
                Ok(ResolvedOp::FusedQkv {
                    input: region_of(*input)?,
                    scaled: region_of(*scaled)?,
                    q: region_of(*q)?,
                    k: region_of(*k)?,
                    v: region_of(*v)?,
                    kernels: [
                        kernel_by_name(&format!("{prefix}.attn_q.weight"))?,
                        kernel_by_name(&format!("{prefix}.attn_k.weight"))?,
                        kernel_by_name(&format!("{prefix}.attn_v.weight"))?,
                    ],
                })
            }
            PlannedOp::FusedOProjResidual {
                attn,
                residual,
                out,
            } => {
                let layer_index = layer_index.ok_or_else(|| {
                    CpuError::ShapeMismatch("fused output projection outside a layer".into())
                })?;
                Ok(ResolvedOp::FusedOProjResidual {
                    attn: region_of(*attn)?,
                    residual: region_of(*residual)?,
                    out: region_of(*out)?,
                    kernel: kernel_by_name(&format!("blk.{layer_index}.attn_output.weight"))?,
                })
            }
            PlannedOp::FusedResidualNorm { a, b, out, .. } => Ok(ResolvedOp::FusedResidualNorm {
                a: region_of(*a)?,
                b: region_of(*b)?,
                out: region_of(*out)?,
            }),
            PlannedOp::OutputNorm { weight, input, out } => {
                let name = weight_name(*weight)?;
                if !name.ends_with("output_norm.weight") {
                    return Err(CpuError::ShapeMismatch(format!(
                        "final norm op references '{name}'"
                    )));
                }
                Ok(ResolvedOp::RmsNorm {
                    role: NormRole::Output,
                    input: region_of(*input)?,
                    out: region_of(*out)?,
                })
            }
            PlannedOp::Logits {
                weight,
                input,
                out,
                tied,
            } => Ok(ResolvedOp::Logits {
                input: region_of(*input)?,
                out: region_of(*out)?,
                tied: *tied,
                kernel: plan
                    .tensor_table
                    .iter()
                    .find(|record| record.id == weight.id)
                    .map(|record| record.kernel)
                    .ok_or_else(|| {
                        CpuError::ShapeMismatch(format!(
                            "logits weight tensor {} not in tensor table",
                            weight.id
                        ))
                    })?,
            }),
            PlannedOp::Fused { .. } => Err(CpuError::ShapeMismatch(
                "fused ops are not supported by the v0.4 phase-4 interpreter".into(),
            )),
        }
    };

    let preamble = plan
        .preamble
        .iter()
        .enumerate()
        .map(|(position, op)| resolve(op, position, None))
        .collect::<Result<Vec<_>, _>>()?;
    let layers = plan
        .layers
        .iter()
        .enumerate()
        .map(|(layer_index, layer)| {
            let ops = layer
                .ops
                .iter()
                .enumerate()
                .map(|(position, op)| resolve(op, position, Some(layer_index)))
                .collect::<Result<Vec<_>, _>>()?;
            // semantic hook regions (contract section 4): block input, o
            // projection output (pre-residual, None when F5 fused), down
            // projection output (pre-residual), and the block output.
            let before_layer = match ops.first() {
                Some(ResolvedOp::RmsNorm { input, .. }) => *input,
                Some(ResolvedOp::FusedQkv { input, .. }) => *input,
                _ => {
                    return Err(CpuError::ShapeMismatch(
                        "layer does not begin with the input norm or fused qkv".into(),
                    ));
                }
            };
            let mut after_attention = None;
            let mut after_mlp = None;
            for op in &ops {
                match op {
                    ResolvedOp::Matvec {
                        role: ProjRole::O,
                        out,
                        ..
                    } => after_attention = Some(*out),
                    ResolvedOp::Matvec {
                        role: ProjRole::Down,
                        out,
                        ..
                    } => after_mlp = Some(*out),
                    _ => {}
                }
            }
            let after_layer = match ops.last() {
                Some(ResolvedOp::ResidualAdd { out, .. }) => *out,
                Some(ResolvedOp::ResidualAdd3 { out, .. }) => *out,
                _ => {
                    return Err(CpuError::ShapeMismatch(
                        "layer does not end with the residual add".into(),
                    ));
                }
            };
            Ok(ResolvedLayerOps {
                ops,
                hooks: ResolvedHookSites {
                    before_layer,
                    after_attention,
                    after_mlp: after_mlp.ok_or_else(|| {
                        CpuError::ShapeMismatch("layer has no mlp down matvec".into())
                    })?,
                    after_layer,
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let final_ops = plan
        .final_ops
        .iter()
        .enumerate()
        .map(|(position, op)| resolve(op, position, None))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PlannedDecodeState {
        plan_hash: plan.plan_hash.clone(),
        arena,
        ops: ResolvedOps {
            preamble,
            layers,
            final_ops,
        },
    })
}

/// RMSNorm on flat slices: `x * weight / sqrt(mean(x^2) + eps)` per row.
/// Dense f32 matvec for a `[in, out]` row-major weight (reference F32 path).
fn dense_matvec_into(src: &[f32], weight: &[f32], in_dim: usize, out_dim: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), in_dim);
    debug_assert_eq!(weight.len(), in_dim * out_dim);
    debug_assert_eq!(dst.len(), out_dim);
    for j in 0..out_dim {
        let mut acc = 0.0f32;
        for (i, &x) in src.iter().enumerate() {
            acc += x * weight[i * out_dim + j];
        }
        dst[j] = acc;
    }
}

/// Single-row projection through a Linear weight into `dst`. Kernel dispatch
/// follows the plan's resolved kernel; the underlying implementations are
/// the scalar/AVX2 Q4_K/Q6_K × Q8_K kernels. `parallel`
/// selects the column-parallel decode matvec for large K-quant projections
/// (bit-identical results, contract Gate A). With `accumulate`, the F32
/// dense path adds into `dst` instead of assigning (the quantized kernels
/// accumulate by contract), which the F5 fused output projection relies on.
fn planned_linear_into(
    linear: &Linear<CpuBackend>,
    src: &[f32],
    dst: &mut [f32],
    parallel: bool,
    accumulate: bool,
) -> Result<(), CpuError> {
    match linear.weight_kind() {
        WeightKindView::F32(t) => {
            let in_dim = t.shape()[0];
            let out_dim = t.shape()[1];
            debug_assert_eq!(src.len(), in_dim);
            debug_assert_eq!(dst.len(), out_dim);
            if accumulate {
                let weight = t.data();
                for j in 0..out_dim {
                    let mut acc = 0.0f32;
                    for (i, &x) in src.iter().enumerate() {
                        acc += x * weight[i * out_dim + j];
                    }
                    dst[j] += acc;
                }
            } else {
                dense_matvec_into(src, t.data(), in_dim, out_dim, dst);
            }
            Ok(())
        }
        WeightKindView::Q8_0(w) => {
            if accumulate {
                return Err(CpuError::Kernel(
                    "Q8_0 matmul has assignment semantics and cannot execute fused accumulation"
                        .into(),
                ));
            }
            CpuBackend.matmul_q8_0_into(src, 1, w, dst);
            Ok(())
        }
        WeightKindView::KQuant(w) => {
            let result = if parallel {
                crate::k_matmul::matmul_k_into_parallel(src, 1, w, dst)
            } else {
                crate::k_matmul::matmul_k_into(src, 1, w, dst)
            };
            result.map_err(|message| CpuError::ShapeMismatch(format!("planned matvec: {message}")))
        }
    }
}

/// The kernel the reference dynamic dispatch would choose for a weight —
/// the Gate A equivalence check against the plan's resolved kernel.
fn planned_kernel_for(linear: &Linear<CpuBackend>) -> KernelId {
    match linear.weight_kind() {
        WeightKindView::F32(_) => KernelId::EagerF32,
        WeightKindView::Q8_0(_) => KernelId::Q8Packed,
        WeightKindView::KQuant(w) => {
            let dtype = w.dtype().name();
            resolve_kernel(dtype, k_execution_name(w.execution()))
        }
    }
}

fn planned_embedding_kernel(embedding: &LlamaEmbedding<CpuBackend>) -> KernelId {
    match embedding {
        LlamaEmbedding::F32(_) => KernelId::EmbeddingF32Row,
        LlamaEmbedding::Q8_0(_) => KernelId::EmbeddingQ8Row,
        LlamaEmbedding::KQuant(weight) => {
            resolve_embedding_kernel(weight.dtype().name(), k_execution_name(weight.execution()))
                .expect("resident K embedding has a supported row-dequant dtype")
        }
    }
}

fn ensure_planned_kernel(
    expected: KernelId,
    linear: &Linear<CpuBackend>,
    operation: &str,
) -> Result<(), CpuError> {
    let resident = planned_kernel_for(linear);
    if expected == resident {
        return Ok(());
    }
    Err(CpuError::Kernel(format!(
        "planned kernel {} does not match resident kernel {} for {operation}",
        expected.name(),
        resident.name()
    )))
}

fn planned_scheduler(
    linear: &Linear<CpuBackend>,
    parallel_requested: bool,
) -> crate::decode_profile::DecodeExecutionMode {
    match linear.weight_kind() {
        WeightKindView::KQuant(weight)
            if crate::k_quant_matmul::scheduler_name(1, weight, parallel_requested)
                == "column-parallel-rayon" =>
        {
            crate::decode_profile::DecodeExecutionMode::ColumnParallelRayon
        }
        _ => crate::decode_profile::DecodeExecutionMode::Serial,
    }
}

/// Elementwise `dst = a + b`.
fn add_into(a: &[f32], b: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for i in 0..a.len() {
        dst[i] = a[i] + b[i];
    }
}

/// In-place bias add.
fn add_bias_into(dst: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(dst.len(), bias.len());
    for i in 0..dst.len() {
        dst[i] += bias[i];
    }
}

/// Embedding row lookup into `dst` (bit-identical to the reference path's
/// `assign_row_from_*` helpers).
fn embed_row_into(
    embed: &LlamaEmbedding<CpuBackend>,
    token: u32,
    dst: &mut [f32],
) -> Result<(), CpuError> {
    match embed {
        LlamaEmbedding::F32(table) => {
            let row = token as usize;
            let cols = dst.len();
            if row >= table.shape()[0] {
                return Err(CpuError::ShapeMismatch(format!(
                    "embedding row {row} out of bounds for {} rows",
                    table.shape()[0]
                )));
            }
            dst.copy_from_slice(&table.data()[row * cols..(row + 1) * cols]);
            Ok(())
        }
        LlamaEmbedding::Q8_0(table) => {
            let row = token as usize;
            if row >= table.out_features() {
                return Err(CpuError::ShapeMismatch(format!(
                    "embedding row {row} out of bounds for {} rows",
                    table.out_features()
                )));
            }
            table.dequantize_row(row, dst);
            Ok(())
        }
        LlamaEmbedding::KQuant(table) => {
            let row = token as usize;
            if row >= table.out_features() {
                return Err(CpuError::ShapeMismatch(format!(
                    "embedding row {row} out of bounds for {} rows",
                    table.out_features()
                )));
            }
            table.dequantize_row(row, dst);
            Ok(())
        }
    }
}

/// Single-token causal attention over the f16 cache, mirroring the
/// reference `cached_causal_attention_with_scratch` single-token branch
/// (same score math, same softmax, same weighted V sum) with the per-token
/// overheads removed: no output allocation (arena region), no `to_vec`
/// copies, score region reused across layers, and cache-head slices hoisted
/// so the inner loops iterate contiguous f16 ranges without re-deriving
/// offsets.
#[allow(clippy::too_many_arguments)]
pub(crate) fn planned_causal_attention(
    q: &[f32],
    cached_k: &[half::f16],
    cached_v: &[half::f16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    total_seq_len: usize,
    scores: &mut [f32],
    out: &mut [f32],
) -> Result<(), CpuError> {
    let n_repeat = crate::backend::validate_gqa(n_heads, n_kv_heads)?;
    let scale = (head_dim as f32).sqrt().recip();
    let cache_head_stride = max_seq_len * head_dim;
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    debug_assert!(scores.len() >= n_heads * total_seq_len);
    out.fill(0.0);
    for h in 0..n_heads {
        let kv_h = h / n_repeat;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let k_head = &cached_k[kv_h * cache_head_stride..(kv_h + 1) * cache_head_stride];
        let v_head = &cached_v[kv_h * cache_head_stride..(kv_h + 1) * cache_head_stride];
        let score_row = &mut scores[h * total_seq_len..(h + 1) * total_seq_len];
        // scores: q · k_j for each cached position
        for (j, slot) in score_row.iter_mut().enumerate() {
            let k_j = &k_head[j * head_dim..(j + 1) * head_dim];
            *slot = crate::simd::dot_product_f16(q_head, k_j) * scale;
        }
        crate::backend::softmax_prefix(score_row, total_seq_len);
        let head_out = &mut out[h * head_dim..(h + 1) * head_dim];
        // weighted V sum; the zero-weight skip mirrors the reference and
        // keeps the accumulation bit-identical
        for (j, &weight) in score_row.iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let v_j = &v_head[j * head_dim..(j + 1) * head_dim];
            crate::simd::weighted_add_f16(head_out, v_j, weight);
        }
    }
    Ok(())
}

/// Arena errors as `CpuError` (the arena reports `String` diagnostics).
fn arena_err(message: String) -> CpuError {
    CpuError::ShapeMismatch(format!("decode arena: {message}"))
}

/// One planned op's profile event: created when operator profiling is
/// enabled, records on drop. Normal decode never touches profiling state
/// (creation is guarded by the relaxed flag read).
struct OpTimer {
    layer: usize,
    operator: &'static str,
    input_dimension: usize,
    output_dimension: usize,
    start: std::time::Instant,
    _mode: crate::decode_profile::DecodeExecutionMode,
}

impl OpTimer {
    fn new(
        layer: usize,
        operator: &'static str,
        input_dimension: usize,
        output_dimension: usize,
        mode: crate::decode_profile::DecodeExecutionMode,
    ) -> Option<Self> {
        crate::decode_profile::is_enabled().then(|| Self {
            layer,
            operator,
            input_dimension,
            output_dimension,
            start: std::time::Instant::now(),
            _mode: mode,
        })
    }
}

impl Drop for OpTimer {
    fn drop(&mut self) {
        if !crate::decode_profile::is_enabled() {
            return;
        }
        crate::decode_profile::record(
            self.layer,
            self.operator,
            self.input_dimension,
            self.output_dimension,
            self._mode,
            self.start.elapsed(),
        );
    }
}

/// Split a region-request result into two disjoint slices (request order is
/// preserved by [`DecodeArena::regions_f32`]).
fn two_regions(slices: [&mut [f32]; 2]) -> (&mut [f32], &mut [f32]) {
    let [a, b] = slices;
    (a, b)
}

/// Split a region-request result into three disjoint slices.
fn three_regions(slices: [&mut [f32]; 3]) -> (&mut [f32], &mut [f32], &mut [f32]) {
    let [a, b, c] = slices;
    (a, b, c)
}

/// Split a region-request result into four disjoint slices.
fn four_regions(slices: [&mut [f32]; 4]) -> (&mut [f32], &mut [f32], &mut [f32], &mut [f32]) {
    let [a, b, c, d] = slices;
    (a, b, c, d)
}

/// Fusion F2: `out = rmsnorm(a + b)` in one pass over both inputs (no
/// standalone residual materialization). Bit-identical to the unfused
/// add-then-norm composition (same elementwise adds, same sum order, same
/// scale multiply order).
fn fused_residual_rmsnorm_into(a: &[f32], b: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), weight.len());
    debug_assert_eq!(a.len(), out.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let value = a[i] + b[i];
        sum += value * value;
    }
    let rstd = (sum / a.len() as f32 + eps).sqrt().recip();
    for i in 0..a.len() {
        let value = a[i] + b[i];
        out[i] = value * rstd * weight[i];
    }
}

/// Plan-driven single-token decode: walks the resolved ops against the
/// scratch arena. Zero heap allocation in the steady state. When `hooks` is
/// `Some`, the six semantic hook sites fire against the arena regions at the
/// same call sites as the reference path (contract section 12); the plan is
/// built with the matching hook mode and active stages.
pub(crate) fn forward_last_logits_planned(
    model: &Llama<CpuBackend>,
    token_ids: &[u32],
    cache: &mut crate::kv_cache::KVCache,
    start_pos: usize,
    mut hooks: Option<(&mut ActiveHooks<'_, '_>, &ExecutionContext<'_>)>,
    hook_mode: HookMode,
    active_stages: &[&str],
) -> Result<CpuTensor, CpuError> {
    if token_ids.len() != 1 {
        return Err(CpuError::ShapeMismatch(
            "planned decode requires exactly one token".into(),
        ));
    }
    cache.validate_start_pos(start_pos);
    let runtime_cache_capacity = cache.max_seq_len();
    if hook_mode == HookMode::Disabled && !active_stages.is_empty() {
        return Err(CpuError::Kernel(
            "disabled hook mode cannot carry active semantic sites".into(),
        ));
    }
    let execution_mode = model.execution_mode();
    if !matches!(
        execution_mode,
        ExecutionMode::Planned | ExecutionMode::PlannedFused
    ) {
        return Err(CpuError::ShapeMismatch(format!(
            "planned decode called for execution mode {}",
            execution_mode.name()
        )));
    }
    let (model_sha256, tokenizer_sha256, canonical_capacity) = model.plan_provenance();
    let plan = model
        .execution_plan(
            execution_mode,
            hook_mode,
            active_stages,
            canonical_capacity.unwrap_or(runtime_cache_capacity),
            model_sha256.as_deref(),
            tokenizer_sha256.as_deref(),
        )
        .map_err(|error| {
            CpuError::ShapeMismatch(format!("execution plan build failed: {error}"))
        })?;
    let parallel_matvec = plan.dispatch.thread_strategy == "column-parallel-rayon";

    let mut state = model.decode_state.borrow_mut();
    if state.as_ref().is_none_or(|s| s.plan_hash != plan.plan_hash) {
        *state = Some(build_planned_state(&plan)?);
    }
    let state = state.as_mut().expect("decode state initialized");
    let PlannedDecodeState { arena, ops, .. } = state;

    // preamble: embedding lookup
    for op in &ops.preamble {
        match op {
            ResolvedOp::Embed { out, kernel } => {
                let resident = planned_embedding_kernel(&model.embed_tokens);
                if *kernel != resident {
                    return Err(CpuError::Kernel(format!(
                        "planned kernel {} does not match resident kernel {} for embedding",
                        kernel.name(),
                        resident.name()
                    )));
                }
                let dst = arena.region_f32(*out).map_err(arena_err)?;
                let _timer = OpTimer::new(
                    0,
                    "embedding",
                    1,
                    dst.len(),
                    crate::decode_profile::DecodeExecutionMode::Serial,
                );
                embed_row_into(&model.embed_tokens, token_ids[0], dst)?;
            }
            other => {
                return Err(CpuError::ShapeMismatch(format!(
                    "unexpected preamble op {other:?}"
                )));
            }
        }
    }

    // transformer blocks
    for (layer, block) in model.blocks.iter().enumerate() {
        let layer_ops = &ops.layers[layer];
        // before_layer fires on the block input (contract section 4/12).
        if let Some((hooks, _)) = hooks.as_mut() {
            let [data] = arena
                .regions_f32([layer_ops.hooks.before_layer])
                .map_err(arena_err)?;
            let mut activation = SliceActivation::new(1, model.config.embed_dim, data);
            hooks.before_layer(layer, &mut activation)?;
        }
        for op in &layer_ops.ops {
            match op {
                ResolvedOp::RmsNorm { role, input, out } => {
                    let (x, dst) =
                        two_regions(arena.regions_f32([*input, *out]).map_err(arena_err)?);
                    let operator = match role {
                        NormRole::AttnIn => "attn_norm",
                        NormRole::MlpIn => "ffn_norm",
                        NormRole::Output => "output_norm",
                    };
                    let _timer = OpTimer::new(
                        layer,
                        operator,
                        x.len(),
                        dst.len(),
                        crate::decode_profile::DecodeExecutionMode::Serial,
                    );
                    let weight = match role {
                        NormRole::AttnIn => block.input_layernorm.data(),
                        NormRole::MlpIn => block.post_attention_layernorm.data(),
                        NormRole::Output => {
                            return Err(CpuError::ShapeMismatch(
                                "output norm resolved inside a layer".into(),
                            ));
                        }
                    };
                    crate::simd::rms_norm_into(x, weight, model.config.norm_eps, dst);
                }
                ResolvedOp::Matvec {
                    role,
                    input,
                    out,
                    has_bias,
                    kernel,
                } => {
                    let linear = match role {
                        ProjRole::Q => &block.self_attn.q_proj,
                        ProjRole::K => &block.self_attn.k_proj,
                        ProjRole::V => &block.self_attn.v_proj,
                        ProjRole::O => &block.self_attn.o_proj,
                        ProjRole::Gate => &block.mlp.gate_proj,
                        ProjRole::Up => &block.mlp.up_proj,
                        ProjRole::Down => &block.mlp.down_proj,
                        ProjRole::Head => {
                            return Err(CpuError::ShapeMismatch(
                                "head projection resolved inside a layer".into(),
                            ));
                        }
                    };
                    let (src, dst) =
                        two_regions(arena.regions_f32([*input, *out]).map_err(arena_err)?);
                    let operator = match role {
                        ProjRole::Q => "q",
                        ProjRole::K => "k",
                        ProjRole::V => "v",
                        ProjRole::O => "o",
                        ProjRole::Gate => "gate",
                        ProjRole::Up => "up",
                        ProjRole::Down => "down",
                        ProjRole::Head => "lm_head",
                    };
                    // The serialized plan is authoritative: fail closed in
                    // release builds if the resident weight no longer matches
                    // the kernel identity resolved when the plan was built.
                    ensure_planned_kernel(*kernel, linear, operator)?;
                    let _timer = OpTimer::new(
                        layer,
                        operator,
                        src.len(),
                        dst.len(),
                        planned_scheduler(linear, parallel_matvec),
                    );
                    // The quantized kernels accumulate into dst (must be
                    // zero-initialized). The reference allocates a fresh
                    // zeroed Vec per projection; the arena reuses regions
                    // across tokens, so the destination must be cleared here.
                    dst.fill(0.0);
                    planned_linear_into(linear, src, dst, parallel_matvec, false)?;
                    if *has_bias {
                        if let Some(bias) = linear.bias() {
                            add_bias_into(dst, bias.data());
                        }
                    }
                    // after_attention / after_mlp fire on the pre-residual
                    // projection outputs (contract section 4/12), through the
                    // resolved hook regions so fused plans can point them at
                    // the materialized tensor. after_attention is None when
                    // fusion F5 eliminated the o tensor for this layer.
                    if let Some((hooks, _)) = hooks.as_mut() {
                        match role {
                            ProjRole::O => {
                                if let Some(region) = layer_ops.hooks.after_attention {
                                    let [data] = arena.regions_f32([region]).map_err(arena_err)?;
                                    let mut activation =
                                        SliceActivation::new(1, model.config.embed_dim, data);
                                    hooks.after_attention(layer, &mut activation)?;
                                }
                            }
                            ProjRole::Down => {
                                let [data] = arena
                                    .regions_f32([layer_ops.hooks.after_mlp])
                                    .map_err(arena_err)?;
                                let mut activation =
                                    SliceActivation::new(1, model.config.embed_dim, data);
                                hooks.after_mlp(layer, &mut activation)?;
                            }
                            _ => {}
                        }
                    }
                }
                ResolvedOp::Rope { role, target } => {
                    let (data,) = {
                        let [data] = arena.regions_f32([*target]).map_err(arena_err)?;
                        (data,)
                    };
                    let _timer = OpTimer::new(
                        layer,
                        "rope",
                        data.len(),
                        data.len(),
                        crate::decode_profile::DecodeExecutionMode::Serial,
                    );
                    let (n_heads, qk_norm) = match role {
                        RopeRole::Q => (model.config.n_heads, block.self_attn.q_norm.as_ref()),
                        RopeRole::K => (model.config.n_kv_heads, block.self_attn.k_norm.as_ref()),
                    };
                    block
                        .self_attn
                        .apply_decode_rope_and_qk_norm(data, n_heads, start_pos, qk_norm);
                }
                ResolvedOp::KvStore { k, v } => {
                    let (k_data, v_data) =
                        two_regions(arena.regions_f32([*k, *v]).map_err(arena_err)?);
                    let _timer = OpTimer::new(
                        layer,
                        "kv_store",
                        k_data.len(),
                        v_data.len(),
                        crate::decode_profile::DecodeExecutionMode::Serial,
                    );
                    cache.append_with_layout(
                        layer,
                        start_pos,
                        k_data,
                        v_data,
                        model.config.n_kv_heads,
                        model.config.head_dim,
                    );
                }
                ResolvedOp::Attention {
                    q,
                    out,
                    scores,
                    rope_q,
                } => {
                    let (q_data, out_data, scores_data) =
                        three_regions(arena.regions_f32([*q, *out, *scores]).map_err(arena_err)?);
                    let _timer = OpTimer::new(
                        layer,
                        "attention",
                        q_data.len(),
                        out_data.len(),
                        crate::decode_profile::DecodeExecutionMode::Serial,
                    );
                    // fusion F4: the Q rope (and optional qk-norm) runs
                    // inside the attention op; the K rope stays a separate
                    // op because the stored K must be roped before the store.
                    if *rope_q {
                        #[cfg(test)]
                        record_fused_execution(|counts| counts.f4_rope_in_attention += 1);
                        block.self_attn.apply_decode_rope_and_qk_norm(
                            q_data,
                            model.config.n_heads,
                            start_pos,
                            block.self_attn.q_norm.as_ref(),
                        );
                    }
                    let (cached_k, cached_v) = cache.get(layer);
                    planned_causal_attention(
                        q_data,
                        cached_k,
                        cached_v,
                        model.config.n_heads,
                        model.config.n_kv_heads,
                        model.config.head_dim,
                        cache.max_seq_len(),
                        cache.cursor() + 1,
                        scores_data,
                        out_data,
                    )?;
                }
                ResolvedOp::FusedQkv {
                    input,
                    scaled,
                    q,
                    k,
                    v,
                    kernels,
                } => {
                    #[cfg(test)]
                    record_fused_execution(|counts| {
                        counts.f1_rmsnorm_linear += 1;
                        counts.f3_qkv_orchestration += 1;
                    });
                    // One norm pass over the block input writes the scaled
                    // activation, then the three projections share it. Profile
                    // the norm and each projection separately because their
                    // shape gates can select different schedulers.
                    {
                        let _timer = OpTimer::new(
                            layer,
                            "fused_qkv_norm",
                            model.config.embed_dim,
                            model.config.embed_dim,
                            crate::decode_profile::DecodeExecutionMode::Serial,
                        );
                        let (x, n1) =
                            two_regions(arena.regions_f32([*input, *scaled]).map_err(arena_err)?);
                        crate::simd::rms_norm_into(
                            x,
                            block.input_layernorm.data(),
                            model.config.norm_eps,
                            n1,
                        );
                    }
                    let projections = [
                        (*q, &block.self_attn.q_proj, kernels[0], "fused_q"),
                        (*k, &block.self_attn.k_proj, kernels[1], "fused_k"),
                        (*v, &block.self_attn.v_proj, kernels[2], "fused_v"),
                    ];
                    for (out_region, linear, kernel, operation) in projections {
                        ensure_planned_kernel(kernel, linear, operation)?;
                        let (n1, out) = two_regions(
                            arena
                                .regions_f32([*scaled, out_region])
                                .map_err(arena_err)?,
                        );
                        out.fill(0.0);
                        let _timer = OpTimer::new(
                            layer,
                            operation,
                            n1.len(),
                            out.len(),
                            planned_scheduler(linear, parallel_matvec),
                        );
                        planned_linear_into(linear, n1, out, parallel_matvec, false)?;
                        if let Some(bias) = linear.bias() {
                            add_bias_into(out, bias.data());
                        }
                    }
                }
                ResolvedOp::FusedOProjResidual {
                    attn,
                    residual,
                    out,
                    kernel,
                } => {
                    #[cfg(test)]
                    record_fused_execution(|counts| counts.f5_output_proj_residual += 1);
                    let (attn_data, residual_data, out_data) = three_regions(
                        arena
                            .regions_f32([*attn, *residual, *out])
                            .map_err(arena_err)?,
                    );
                    ensure_planned_kernel(*kernel, &block.self_attn.o_proj, "fused_o")?;
                    let _timer = OpTimer::new(
                        layer,
                        "fused_o_proj",
                        attn_data.len(),
                        out_data.len(),
                        planned_scheduler(&block.self_attn.o_proj, parallel_matvec),
                    );
                    // fusion F5: out starts as the residual; the matvec
                    // kernel accumulates W·attn on top (one pass, no
                    // standalone o tensor).
                    out_data.copy_from_slice(residual_data);
                    planned_linear_into(
                        &block.self_attn.o_proj,
                        attn_data,
                        out_data,
                        parallel_matvec,
                        true,
                    )?;
                    if let Some(bias) = block.self_attn.o_proj.bias() {
                        add_bias_into(out_data, bias.data());
                    }
                }
                ResolvedOp::FusedResidualNorm { a, b, out } => {
                    #[cfg(test)]
                    record_fused_execution(|counts| counts.f2_residual_rmsnorm += 1);
                    let (a_data, b_data, out_data) =
                        three_regions(arena.regions_f32([*a, *b, *out]).map_err(arena_err)?);
                    let _timer = OpTimer::new(
                        layer,
                        "fused_residual_norm",
                        a_data.len(),
                        out_data.len(),
                        crate::decode_profile::DecodeExecutionMode::Serial,
                    );
                    fused_residual_rmsnorm_into(
                        a_data,
                        b_data,
                        block.post_attention_layernorm.data(),
                        model.config.norm_eps,
                        out_data,
                    );
                }
                ResolvedOp::ResidualAdd3 { a, b, c, out } => {
                    let (a_data, b_data, c_data, out_data) =
                        four_regions(arena.regions_f32([*a, *b, *c, *out]).map_err(arena_err)?);
                    let _timer = OpTimer::new(
                        layer,
                        "residual_add3",
                        a_data.len(),
                        out_data.len(),
                        crate::decode_profile::DecodeExecutionMode::Serial,
                    );
                    for i in 0..out_data.len() {
                        out_data[i] = (a_data[i] + b_data[i]) + c_data[i];
                    }
                }
                ResolvedOp::Silu { target } => {
                    let (data,) = {
                        let [data] = arena.regions_f32([*target]).map_err(arena_err)?;
                        (data,)
                    };
                    let _timer = OpTimer::new(
                        layer,
                        "silu",
                        data.len(),
                        data.len(),
                        crate::decode_profile::DecodeExecutionMode::Serial,
                    );
                    // in-place silu: x / (1 + exp(-x)), matching the
                    // reference `CpuTensor::silu` formula
                    for x in data.iter_mut() {
                        *x = *x / (1.0 + (-*x).exp());
                    }
                }
                ResolvedOp::Elemul { a, b, out } => {
                    let (a_data, b_data, out_data) =
                        three_regions(arena.regions_f32([*a, *b, *out]).map_err(arena_err)?);
                    let _timer = OpTimer::new(
                        layer,
                        "elemul",
                        a_data.len(),
                        out_data.len(),
                        crate::decode_profile::DecodeExecutionMode::Serial,
                    );
                    crate::simd::elemul(a_data, b_data, out_data);
                }
                ResolvedOp::ResidualAdd { a, b, out } => {
                    let (a_data, b_data, out_data) =
                        three_regions(arena.regions_f32([*a, *b, *out]).map_err(arena_err)?);
                    let _timer = OpTimer::new(
                        layer,
                        "residual_add",
                        a_data.len(),
                        out_data.len(),
                        crate::decode_profile::DecodeExecutionMode::Serial,
                    );
                    add_into(a_data, b_data, out_data);
                }
                ResolvedOp::Logits { .. } => {
                    return Err(CpuError::ShapeMismatch(
                        "logits op resolved inside a layer".into(),
                    ));
                }
                ResolvedOp::Embed { .. } => {
                    return Err(CpuError::ShapeMismatch(
                        "embedding op resolved inside a layer".into(),
                    ));
                }
            }
        }
        // after_layer fires on the block output (contract section 4/12).
        if let Some((hooks, _)) = hooks.as_mut() {
            let [data] = arena
                .regions_f32([layer_ops.hooks.after_layer])
                .map_err(arena_err)?;
            let mut activation = SliceActivation::new(1, model.config.embed_dim, data);
            hooks.after_layer(layer, &mut activation)?;
        }
    }
    cache.advance_cursor();

    // final ops: output norm + LM head
    let mut logits: Option<CpuTensor> = None;
    for op in &ops.final_ops {
        match op {
            ResolvedOp::RmsNorm {
                role: NormRole::Output,
                input,
                out,
            } => {
                let (x, dst) = two_regions(arena.regions_f32([*input, *out]).map_err(arena_err)?);
                let _timer = OpTimer::new(
                    usize::MAX,
                    "output_norm",
                    x.len(),
                    dst.len(),
                    crate::decode_profile::DecodeExecutionMode::Serial,
                );
                crate::simd::rms_norm_into(x, model.norm.data(), model.config.norm_eps, dst);
                // before_logits fires on the final-norm output.
                if let Some((hooks, _)) = hooks.as_mut() {
                    let mut activation = SliceActivation::new(1, model.config.embed_dim, dst);
                    hooks.before_logits(&mut activation)?;
                }
            }
            ResolvedOp::Logits {
                input,
                out,
                tied,
                kernel,
            } => {
                if *tied != model.head_tied {
                    return Err(CpuError::Kernel(
                        "plan head tie flag diverged from the resident model".into(),
                    ));
                }
                ensure_planned_kernel(*kernel, &model.head, "lm_head")?;
                let (src, dst) = two_regions(arena.regions_f32([*input, *out]).map_err(arena_err)?);
                {
                    let _timer = OpTimer::new(
                        usize::MAX,
                        "lm_head",
                        src.len(),
                        dst.len(),
                        planned_scheduler(&model.head, parallel_matvec),
                    );
                    dst.fill(0.0);
                    planned_linear_into(&model.head, src, dst, parallel_matvec, false)?;
                }
                let mut logits_tensor = {
                    let _materialize_timer = OpTimer::new(
                        usize::MAX,
                        "logits_materialize",
                        dst.len(),
                        dst.len(),
                        crate::decode_profile::DecodeExecutionMode::Serial,
                    );
                    CpuTensor::from_data(vec![1, model.config.vocab_size], dst.to_vec())
                };
                // after_logits fires on the final logits tensor.
                if let Some((hooks, _)) = hooks.as_mut() {
                    hooks.after_logits(&mut logits_tensor)?;
                }
                logits = Some(logits_tensor);
            }
            other => {
                return Err(CpuError::ShapeMismatch(format!(
                    "unexpected final op {other:?}"
                )));
            }
        }
    }
    logits.ok_or_else(|| CpuError::ShapeMismatch("planned final ops produced no logits".into()))
}
