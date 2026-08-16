//! v0.4 execution-plan construction (moved out of `src/llama.rs`).
//!
//! The data model lives in `src/plan.rs` and the interpreter in
//! `src/planned_decode.rs`; this module owns the build side: the
//! `PlanBuilder` table/region bookkeeping and `Llama::execution_plan`,
//! which walks the (mostly `pub`) model structure once and freezes it into
//! an immutable plan. `plan_cache`/`k_decisions` are `pub(crate)` for this
//! module's sake.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::backend::CpuBackend;
use crate::llama::{
    k_execution_name, Llama, LlamaEmbedding, RopeLayout, PLAN_DECODE_SEQ, PLAN_REGION_ALIGN,
};
use crate::model::Linear;
use crate::plan::{
    resolve_embedding_kernel, resolve_kernel, DispatchPlan, ExecutionMode, ExecutionPlan,
    FusionState, HookMode, HookSitePlan, HookSiteRecord, KernelEntry, KernelId, KvLayout,
    LayerPlan, PlanProvenance, PlannedOp, RopeSummary, ScratchPlan, ScratchRegion, TensorRecord,
    TensorRef,
};

struct PlanBuilder {
    next_id: usize,
    tensor_table: Vec<TensorRecord>,
    regions: Vec<ScratchRegion>,
    tensor_regions: BTreeMap<usize, String>,
}

impl PlanBuilder {
    fn new() -> Self {
        Self {
            next_id: 0,
            tensor_table: Vec::new(),
            regions: Vec::new(),
            tensor_regions: BTreeMap::new(),
        }
    }

    /// Register a weight tensor and return its stable id.
    #[allow(clippy::too_many_arguments)]
    fn add_weight(
        &mut self,
        name: &str,
        shape: Vec<usize>,
        gguf_dtype: &str,
        execution: &str,
        kernel: KernelId,
        resident_bytes: usize,
        mmap: bool,
    ) -> TensorRef {
        let id = self.next_id;
        self.next_id += 1;
        self.tensor_table.push(TensorRecord {
            id,
            name: name.to_string(),
            shape,
            gguf_dtype: gguf_dtype.to_string(),
            execution: execution.to_string(),
            kernel,
            resident_bytes,
            mmap,
        });
        TensorRef::new(id)
    }

    /// Register a decode activation (scratch region) and return its id.
    fn add_activation(&mut self, region: &str, elements: usize) -> TensorRef {
        let id = self.next_id;
        self.next_id += 1;
        // checked: an adversarial metadata blow-up must fail with a clean
        // error, never wrap into a smaller (aliasing) arena region
        let size = elements
            .checked_mul(4)
            .expect("activation region byte size overflow (metadata sanity caps)");
        self.regions.push(ScratchRegion {
            name: region.to_string(),
            offset: 0,
            size,
            alignment: PLAN_REGION_ALIGN,
            first_op: usize::MAX,
            last_op: 0,
            shared_with: None,
        });
        self.tensor_regions.insert(id, region.to_string());
        TensorRef::new(id)
    }

    /// Look up the kernel recorded for a weight tensor. Weight ids and
    /// activation ids share one counter, so weights are located by id, not
    /// by table position.
    fn kernel_of(&self, tensor: TensorRef) -> KernelId {
        self.tensor_table
            .iter()
            .find(|record| record.id == tensor.id)
            .expect("matvec weight must be in the tensor table")
            .kernel
    }

    /// Resolve every region referenced by `op` (weights have no region).
    /// Returns owned names so callers are free to mutate the region list.
    fn regions_touched(&self, op: &PlannedOp) -> Vec<String> {
        let mut out = Vec::new();
        let push = |id: TensorRef, out: &mut Vec<String>| {
            if let Some(region) = self.tensor_regions.get(&id.id) {
                out.push(region.clone());
            }
        };
        match op {
            PlannedOp::Embedding { out: dst, .. } => push(*dst, &mut out),
            PlannedOp::RmsNorm {
                input, out: dst, ..
            } => {
                push(*input, &mut out);
                push(*dst, &mut out);
            }
            PlannedOp::Matvec {
                input, out: dst, ..
            } => {
                push(*input, &mut out);
                push(*dst, &mut out);
            }
            PlannedOp::Rope { target, .. } => push(*target, &mut out),
            PlannedOp::KvStore { k, v } => {
                push(*k, &mut out);
                push(*v, &mut out);
            }
            PlannedOp::Attention {
                q,
                out: dst,
                score_scratch,
                ..
            } => {
                push(*q, &mut out);
                push(*dst, &mut out);
                out.push(score_scratch.clone());
            }
            PlannedOp::Silu { target } => push(*target, &mut out),
            PlannedOp::Elemul { a, b, out: dst } => {
                push(*a, &mut out);
                push(*b, &mut out);
                push(*dst, &mut out);
            }
            PlannedOp::ResidualAdd { a, b, out: dst } => {
                push(*a, &mut out);
                push(*b, &mut out);
                push(*dst, &mut out);
            }
            PlannedOp::ResidualAdd3 { a, b, c, out: dst } => {
                push(*a, &mut out);
                push(*b, &mut out);
                push(*c, &mut out);
                push(*dst, &mut out);
            }
            PlannedOp::OutputNorm {
                input, out: dst, ..
            } => {
                push(*input, &mut out);
                push(*dst, &mut out);
            }
            PlannedOp::Logits {
                input, out: dst, ..
            } => {
                push(*input, &mut out);
                push(*dst, &mut out);
            }
            // Fused ops fold their components' regions in via the component
            // ops themselves; the concrete fused variants list their regions
            // directly.
            PlannedOp::Fused { .. } => {}
            PlannedOp::FusedQkv {
                input,
                scaled,
                q,
                k,
                v,
                ..
            } => {
                push(*input, &mut out);
                push(*scaled, &mut out);
                push(*q, &mut out);
                push(*k, &mut out);
                push(*v, &mut out);
            }
            PlannedOp::FusedOProjResidual {
                attn,
                residual,
                out: dst,
                ..
            } => {
                push(*attn, &mut out);
                push(*residual, &mut out);
                push(*dst, &mut out);
            }
            PlannedOp::FusedResidualNorm { a, b, out: dst, .. } => {
                push(*a, &mut out);
                push(*b, &mut out);
                push(*dst, &mut out);
            }
        }
        out
    }

    /// Assign deterministic offsets and fill in region lifetimes from the
    /// op sequences (global op index across preamble/layers/final).
    fn layout(&mut self, preamble: &[PlannedOp], layers: &[LayerPlan], final_ops: &[PlannedOp]) {
        let ops: Vec<&PlannedOp> = preamble
            .iter()
            .chain(layers.iter().flat_map(|layer| layer.ops.iter()))
            .chain(final_ops.iter())
            .collect();
        for (index, op) in ops.iter().enumerate() {
            let touched = self.regions_touched(op);
            for region in touched {
                let region_index = self
                    .regions
                    .iter()
                    .position(|r| r.name == region)
                    .expect("op references a registered region");
                let entry = &mut self.regions[region_index];
                if entry.first_op == usize::MAX {
                    entry.first_op = index;
                }
                entry.last_op = index;
            }
        }
        for region in &mut self.regions {
            if region.first_op == usize::MAX {
                region.first_op = 0;
            }
        }
        let mut cursor = 0usize;
        for region in &mut self.regions {
            cursor = align_up(cursor, region.alignment);
            region.offset = cursor;
            cursor = cursor
                .checked_add(region.size)
                .expect("scratch region layout overflow (metadata sanity caps)");
        }
    }

    fn finish(self) -> (Vec<TensorRecord>, ScratchPlan, DispatchPlan) {
        let total_bytes = align_up(
            self.regions
                .iter()
                .map(|r| r.offset + r.size)
                .max()
                .unwrap_or(0),
            PLAN_REGION_ALIGN,
        );
        let scratch = ScratchPlan {
            total_bytes,
            alignment: PLAN_REGION_ALIGN,
            seq_capacity: PLAN_DECODE_SEQ,
            regions: self.regions,
            tensor_regions: self.tensor_regions,
        };
        let kernel_per_tensor = self
            .tensor_table
            .iter()
            .map(|t| KernelEntry {
                tensor: t.id,
                kernel: t.kernel,
                cpu_feature: t.kernel.cpu_feature().map(str::to_string),
                fallback: None,
            })
            .collect();
        let dispatch = DispatchPlan {
            kernel_per_tensor,
            thread_strategy: if rayon::current_num_threads() > 1 {
                "column-parallel-rayon".to_string()
            } else {
                "serial".to_string()
            },
        };
        (self.tensor_table, scratch, dispatch)
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

impl Llama<CpuBackend> {
    /// Build (or return the cached) execution plan for this model under the
    /// given execution/hook configuration (contract D5). `max_seq_len` is
    /// the runtime KV capacity the scratch score region is sized for.
    pub fn execution_plan(
        &self,
        mode: ExecutionMode,
        hook_mode: HookMode,
        active_stages: &[&str],
        max_seq_len: usize,
        model_sha256: Option<&str>,
        tokenizer_sha256: Option<&str>,
    ) -> anyhow::Result<Arc<ExecutionPlan>> {
        let mut canonical_sites = active_stages
            .iter()
            .map(|stage| stage.to_string())
            .collect::<Vec<_>>();
        canonical_sites.sort();
        canonical_sites.dedup();
        let key = (
            mode,
            hook_mode,
            canonical_sites.clone(),
            max_seq_len,
            rayon::current_num_threads(),
            model_sha256.unwrap_or_default().to_string(),
            tokenizer_sha256.unwrap_or_default().to_string(),
        );
        let cache = self.plan_cache.get_or_init(|| Mutex::new(BTreeMap::new()));
        let mut cache = cache.lock().expect("plan cache poisoned");
        if let Some(plan) = cache.get(&key) {
            return Ok(Arc::clone(plan));
        }
        let canonical_refs = canonical_sites
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let plan = Arc::new(self.build_execution_plan(
            mode,
            hook_mode,
            &canonical_refs,
            max_seq_len,
            model_sha256,
            tokenizer_sha256,
        )?);
        cache.insert(key, Arc::clone(&plan));
        Ok(plan)
    }

    /// Construct the full architecture-specific execution plan.
    #[allow(clippy::too_many_arguments)]
    fn build_execution_plan(
        &self,
        mode: ExecutionMode,
        hook_mode: HookMode,
        active_stages: &[&str],
        max_seq_len: usize,
        model_sha256: Option<&str>,
        tokenizer_sha256: Option<&str>,
    ) -> anyhow::Result<ExecutionPlan> {
        const KNOWN_STAGES: [&str; 6] = [
            "before-layer",
            "after-attention",
            "after-mlp",
            "after-layer",
            "before-logits",
            "after-logits",
        ];
        const LAYER_STAGES: [&str; 4] = [
            "before-layer",
            "after-attention",
            "after-mlp",
            "after-layer",
        ];
        let backend = CpuBackend;
        let config = &self.config;
        let n_layers = self.blocks.len();
        anyhow::ensure!(
            n_layers > 0,
            "cannot plan a model with zero transformer blocks"
        );
        anyhow::ensure!(
            hook_mode != HookMode::Disabled || active_stages.is_empty(),
            "disabled hook mode cannot carry active sites"
        );
        let mut parsed_sites = Vec::with_capacity(active_stages.len());
        for key in active_stages {
            let (stage, layer) = if let Some((stage, suffix)) = key.rsplit_once('@') {
                anyhow::ensure!(
                    LAYER_STAGES.contains(&stage),
                    "hook site '{key}' cannot carry a layer suffix"
                );
                let layer = suffix
                    .parse::<usize>()
                    .map_err(|_| anyhow::anyhow!("invalid hook layer in '{key}'"))?;
                anyhow::ensure!(
                    layer < n_layers,
                    "hook site '{key}' exceeds layer count {n_layers}"
                );
                (stage, Some(layer))
            } else {
                (*key, None)
            };
            anyhow::ensure!(
                KNOWN_STAGES.contains(&stage),
                "unknown hook stage '{stage}' (expected one of {KNOWN_STAGES:?})"
            );
            parsed_sites.push((stage, layer));
        }
        let site_active = |stage: &str, layer: Option<usize>| {
            parsed_sites.iter().any(|&(candidate, target_layer)| {
                candidate == stage && (target_layer.is_none() || target_layer == layer)
            })
        };
        let embed = config.embed_dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;
        let inter = self.blocks[0].mlp.gate_proj.out_features(&backend);
        let vocab = config.vocab_size;

        let (architecture, rope_layout_name, qk_norm_order_name) = match config.rope_layout {
            RopeLayout::AdjacentPair => ("llama", "adjacent-pair", "after-rope"),
            RopeLayout::SplitHalf => ("qwen2", "split-half", "before-rope"),
        };

        // ---- weight table ----
        let mut builder = PlanBuilder::new();

        let embed_weight = match &self.embed_tokens {
            LlamaEmbedding::F32(t) => builder.add_weight(
                "token_embd.weight",
                vec![t.shape()[0], t.shape()[1]],
                "f32",
                "eager_f32",
                KernelId::EmbeddingF32Row,
                t.data().len() * 4,
                false,
            ),
            LlamaEmbedding::Q8_0(w) => builder.add_weight(
                "token_embd.weight",
                vec![w.out_features(), w.in_features()],
                "q8_0",
                "compressed",
                KernelId::EmbeddingQ8Row,
                w.byte_len(),
                w.is_mapped(),
            ),
            LlamaEmbedding::KQuant(w) => {
                let dtype = w.dtype().name();
                let exec = k_execution_name(w.execution());
                builder.add_weight(
                    "token_embd.weight",
                    vec![w.out_features(), w.in_features()],
                    dtype,
                    exec,
                    resolve_embedding_kernel(dtype, exec).expect("native K embedding dtype"),
                    w.byte_len(),
                    w.is_mapped(),
                )
            }
        };

        let mut q_norm_present = false;
        let mut k_norm_present = false;
        let mut last_x2 = None;
        let mut x_in = builder.add_activation("layer0.x_in", embed);
        let x0 = x_in;
        let _scores = builder.add_activation("scores", config.n_heads * max_seq_len);
        let mut layers = Vec::with_capacity(n_layers);
        // (input, o, m, output) tensor ids per layer, for hook-site records.
        let mut hook_tensors: Vec<(TensorRef, TensorRef, TensorRef, TensorRef)> = Vec::new();

        for (l, block) in self.blocks.iter().enumerate() {
            let prefix = format!("blk.{l}");
            let attn = &block.self_attn;
            let mlp = &block.mlp;

            let attn_norm = builder.add_weight(
                &format!("{prefix}.attn_norm.weight"),
                block.input_layernorm.shape().to_vec(),
                "f32",
                "eager_f32",
                KernelId::EagerF32,
                block.input_layernorm.data().len() * 4,
                false,
            );
            let ffn_norm = builder.add_weight(
                &format!("{prefix}.ffn_norm.weight"),
                block.post_attention_layernorm.shape().to_vec(),
                "f32",
                "eager_f32",
                KernelId::EagerF32,
                block.post_attention_layernorm.data().len() * 4,
                false,
            );

            let (q_w, q_b) = plan_linear(
                &attn.q_proj,
                &format!("{prefix}.attn_q.weight"),
                &mut builder,
            );
            let (k_w, k_b) = plan_linear(
                &attn.k_proj,
                &format!("{prefix}.attn_k.weight"),
                &mut builder,
            );
            let (v_w, v_b) = plan_linear(
                &attn.v_proj,
                &format!("{prefix}.attn_v.weight"),
                &mut builder,
            );
            let (o_w, o_b) = plan_linear(
                &attn.o_proj,
                &format!("{prefix}.attn_output.weight"),
                &mut builder,
            );
            let (gate_w, gate_b) = plan_linear(
                &mlp.gate_proj,
                &format!("{prefix}.ffn_gate.weight"),
                &mut builder,
            );
            let (up_w, up_b) = plan_linear(
                &mlp.up_proj,
                &format!("{prefix}.ffn_up.weight"),
                &mut builder,
            );
            let (down_w, down_b) = plan_linear(
                &mlp.down_proj,
                &format!("{prefix}.ffn_down.weight"),
                &mut builder,
            );

            let q_norm = attn.q_norm.as_ref().map(|t| {
                q_norm_present = true;
                builder.add_weight(
                    &format!("{prefix}.attn_q_norm.weight"),
                    t.shape().to_vec(),
                    "f32",
                    "eager_f32",
                    KernelId::EagerF32,
                    t.data().len() * 4,
                    false,
                )
            });
            let k_norm = attn.k_norm.as_ref().map(|t| {
                k_norm_present = true;
                builder.add_weight(
                    &format!("{prefix}.attn_k_norm.weight"),
                    t.shape().to_vec(),
                    "f32",
                    "eager_f32",
                    KernelId::EagerF32,
                    t.data().len() * 4,
                    false,
                )
            });

            // ---- activations (decode seq = 1) ----
            let n1 = builder.add_activation(&format!("layer{l}.n1"), embed);
            let q = builder.add_activation(&format!("layer{l}.q"), q_dim);
            let k = builder.add_activation(&format!("layer{l}.k"), kv_dim);
            let v = builder.add_activation(&format!("layer{l}.v"), kv_dim);
            let attn_out = builder.add_activation(&format!("layer{l}.attn"), q_dim);
            let o = builder.add_activation(&format!("layer{l}.o"), embed);
            let x1 = builder.add_activation(&format!("layer{l}.x1"), embed);
            let n2 = builder.add_activation(&format!("layer{l}.n2"), embed);
            let g = builder.add_activation(&format!("layer{l}.g"), inter);
            let u = builder.add_activation(&format!("layer{l}.u"), inter);
            let gu = builder.add_activation(&format!("layer{l}.gu"), inter);
            let m = builder.add_activation(&format!("layer{l}.m"), embed);
            let x2 = builder.add_activation(&format!("layer{l}.x2"), embed);

            // ---- unfused op sequence (contract section 3) ----
            let mut ops = vec![
                PlannedOp::RmsNorm {
                    weight: attn_norm,
                    input: x_in,
                    out: n1,
                },
                PlannedOp::Matvec {
                    weight: q_w,
                    input: n1,
                    out: q,
                    kernel: builder.kernel_of(q_w),
                    fused_rms_norm: None,
                    bias: q_b,
                },
                PlannedOp::Matvec {
                    weight: k_w,
                    input: n1,
                    out: k,
                    kernel: builder.kernel_of(k_w),
                    fused_rms_norm: None,
                    bias: k_b,
                },
                PlannedOp::Matvec {
                    weight: v_w,
                    input: n1,
                    out: v,
                    kernel: builder.kernel_of(v_w),
                    fused_rms_norm: None,
                    bias: v_b,
                },
                PlannedOp::Rope {
                    target: q,
                    rope_layout: rope_layout_name.to_string(),
                    qk_norm: q_norm,
                    qk_norm_order: qk_norm_order_name.to_string(),
                },
                PlannedOp::Rope {
                    target: k,
                    rope_layout: rope_layout_name.to_string(),
                    qk_norm: k_norm,
                    qk_norm_order: qk_norm_order_name.to_string(),
                },
                PlannedOp::KvStore { k, v },
                PlannedOp::Attention {
                    q,
                    out: attn_out,
                    score_scratch: "scores".to_string(),
                    rope_q: false,
                },
                PlannedOp::Matvec {
                    weight: o_w,
                    input: attn_out,
                    out: o,
                    kernel: builder.kernel_of(o_w),
                    fused_rms_norm: None,
                    bias: o_b,
                },
                PlannedOp::ResidualAdd {
                    a: x_in,
                    b: o,
                    out: x1,
                },
                PlannedOp::RmsNorm {
                    weight: ffn_norm,
                    input: x1,
                    out: n2,
                },
                PlannedOp::Matvec {
                    weight: gate_w,
                    input: n2,
                    out: g,
                    kernel: builder.kernel_of(gate_w),
                    fused_rms_norm: None,
                    bias: gate_b,
                },
                PlannedOp::Silu { target: g },
                PlannedOp::Matvec {
                    weight: up_w,
                    input: n2,
                    out: u,
                    kernel: builder.kernel_of(up_w),
                    fused_rms_norm: None,
                    bias: up_b,
                },
                PlannedOp::Elemul {
                    a: g,
                    b: u,
                    out: gu,
                },
                PlannedOp::Matvec {
                    weight: down_w,
                    input: gu,
                    out: m,
                    kernel: builder.kernel_of(down_w),
                    fused_rms_norm: None,
                    bias: down_b,
                },
                PlannedOp::ResidualAdd {
                    a: x1,
                    b: m,
                    out: x2,
                },
            ];

            // Phase 8: planned-fused applies the frozen fusion set
            // (contract section 6). When after_attention is active, F5 is
            // defused so the o tensor stays materialized at the hook; the
            // mlp norm then runs fused F2 (residual+rmsnorm with the
            // 3-operand final add). Otherwise the layer is fully fused
            // (F1+F3 qkv orchestration, F4 q-rope in attention, F5 output
            // projection with residual accumulate; F2 is subsumed by F5
            // because the residual is already the fused output).
            let (fusion, fusion_reason) = if mode == ExecutionMode::PlannedFused {
                let after_attention_active = site_active("after-attention", Some(l));
                let f5_requires_assignment = builder.kernel_of(o_w) == KernelId::Q8Packed;
                if after_attention_active || f5_requires_assignment {
                    ops = vec![
                        PlannedOp::FusedQkv {
                            input: x_in,
                            norm: attn_norm,
                            scaled: n1,
                            q,
                            k,
                            v,
                        },
                        PlannedOp::Rope {
                            target: k,
                            rope_layout: rope_layout_name.to_string(),
                            qk_norm: k_norm,
                            qk_norm_order: qk_norm_order_name.to_string(),
                        },
                        PlannedOp::KvStore { k, v },
                        PlannedOp::Attention {
                            q,
                            out: attn_out,
                            score_scratch: "scores".to_string(),
                            rope_q: true,
                        },
                        PlannedOp::Matvec {
                            weight: o_w,
                            input: attn_out,
                            out: o,
                            kernel: builder.kernel_of(o_w),
                            fused_rms_norm: None,
                            bias: o_b,
                        },
                        PlannedOp::FusedResidualNorm {
                            a: x_in,
                            b: o,
                            weight: ffn_norm,
                            out: n2,
                        },
                        PlannedOp::Matvec {
                            weight: gate_w,
                            input: n2,
                            out: g,
                            kernel: builder.kernel_of(gate_w),
                            fused_rms_norm: None,
                            bias: gate_b,
                        },
                        PlannedOp::Silu { target: g },
                        PlannedOp::Matvec {
                            weight: up_w,
                            input: n2,
                            out: u,
                            kernel: builder.kernel_of(up_w),
                            fused_rms_norm: None,
                            bias: up_b,
                        },
                        PlannedOp::Elemul {
                            a: g,
                            b: u,
                            out: gu,
                        },
                        PlannedOp::Matvec {
                            weight: down_w,
                            input: gu,
                            out: m,
                            kernel: builder.kernel_of(down_w),
                            fused_rms_norm: None,
                            bias: down_b,
                        },
                        PlannedOp::ResidualAdd3 {
                            a: x_in,
                            b: o,
                            c: m,
                            out: x2,
                        },
                    ];
                    (
                        FusionState::PartiallyFused,
                        Some(if after_attention_active {
                            "F5 defused: after_attention requires the materialized o tensor; F2 active"
                                .to_string()
                        } else {
                            "F5 defused: q8-packed projection has assignment semantics; F2 active"
                                .to_string()
                        }),
                    )
                } else {
                    ops = vec![
                        PlannedOp::FusedQkv {
                            input: x_in,
                            norm: attn_norm,
                            scaled: n1,
                            q,
                            k,
                            v,
                        },
                        PlannedOp::Rope {
                            target: k,
                            rope_layout: rope_layout_name.to_string(),
                            qk_norm: k_norm,
                            qk_norm_order: qk_norm_order_name.to_string(),
                        },
                        PlannedOp::KvStore { k, v },
                        PlannedOp::Attention {
                            q,
                            out: attn_out,
                            score_scratch: "scores".to_string(),
                            rope_q: true,
                        },
                        PlannedOp::FusedOProjResidual {
                            attn: attn_out,
                            residual: x_in,
                            out: x1,
                        },
                        PlannedOp::RmsNorm {
                            weight: ffn_norm,
                            input: x1,
                            out: n2,
                        },
                        PlannedOp::Matvec {
                            weight: gate_w,
                            input: n2,
                            out: g,
                            kernel: builder.kernel_of(gate_w),
                            fused_rms_norm: None,
                            bias: gate_b,
                        },
                        PlannedOp::Silu { target: g },
                        PlannedOp::Matvec {
                            weight: up_w,
                            input: n2,
                            out: u,
                            kernel: builder.kernel_of(up_w),
                            fused_rms_norm: None,
                            bias: up_b,
                        },
                        PlannedOp::Elemul {
                            a: g,
                            b: u,
                            out: gu,
                        },
                        PlannedOp::Matvec {
                            weight: down_w,
                            input: gu,
                            out: m,
                            kernel: builder.kernel_of(down_w),
                            fused_rms_norm: None,
                            bias: down_b,
                        },
                        PlannedOp::ResidualAdd {
                            a: x1,
                            b: m,
                            out: x2,
                        },
                    ];
                    (
                        FusionState::Fused,
                        Some(
                            "F2 subsumed by F5: the fused output projection materializes the residual"
                                .to_string(),
                        ),
                    )
                }
            } else {
                (FusionState::Unfused, None)
            };

            layers.push(LayerPlan {
                layer_index: l,
                ops,
                fusion,
                fusion_reason,
            });
            hook_tensors.push((x_in, o, m, x2));
            last_x2 = Some(x2);
            x_in = x2;
        }

        let x2_last = last_x2.ok_or_else(|| anyhow::anyhow!("no layer output"))?;
        let output_norm = builder.add_weight(
            "output_norm.weight",
            self.norm.shape().to_vec(),
            "f32",
            "eager_f32",
            KernelId::EagerF32,
            self.norm.data().len() * 4,
            false,
        );
        let (head_w, _head_b) = plan_linear(
            &self.head,
            if self.head_tied {
                "output.weight (tied to token_embd.weight)"
            } else {
                "output.weight"
            },
            &mut builder,
        );

        let hf = builder.add_activation("hf", embed);
        let logits = builder.add_activation("logits", vocab);
        let final_ops = vec![
            PlannedOp::OutputNorm {
                weight: output_norm,
                input: x2_last,
                out: hf,
            },
            PlannedOp::Logits {
                weight: head_w,
                input: hf,
                out: logits,
                tied: self.head_tied,
            },
        ];

        // ---- region layout + dispatch ----
        let preamble = vec![PlannedOp::Embedding {
            tensor: embed_weight,
            out: x0,
        }];

        builder.layout(&preamble, &layers, &final_ops);
        let (mut tensor_table, scratch, mut dispatch) = builder.finish();
        for record in &mut tensor_table {
            let decision_name = if record
                .name
                .starts_with("output.weight (tied to token_embd.weight)")
            {
                "token_embd.weight"
            } else {
                record.name.as_str()
            };
            let Some(decision) = self.k_decisions.get(decision_name) else {
                continue;
            };
            let gguf_dtype = crate::loader::ggml_dtype_name(decision.gguf_dtype)
                .unwrap_or("unknown")
                .to_string();
            let execution = k_execution_name(decision.execution).to_string();
            let resolved = if record.name == "token_embd.weight" {
                resolve_embedding_kernel(&gguf_dtype, &execution)
                    .ok_or_else(|| anyhow::anyhow!("unsupported K embedding dtype {gguf_dtype}"))?
            } else {
                resolve_kernel(&gguf_dtype, &execution)
            };
            anyhow::ensure!(
                resolved == record.kernel,
                "loader decision for '{}' resolves to {} but resident model uses {}",
                record.name,
                resolved.name(),
                record.kernel.name()
            );
            record.gguf_dtype = gguf_dtype;
            record.execution = execution;
            let entry = dispatch
                .kernel_per_tensor
                .iter_mut()
                .find(|entry| entry.tensor == record.id)
                .expect("dispatch is one-to-one with the tensor table");
            entry.kernel = resolved;
            entry.cpu_feature = resolved.cpu_feature().map(str::to_string);
            entry.fallback = decision.fallback_reason.clone();
        }

        // ---- hook sites ----
        let mut sites = Vec::new();
        for (l, &(x_in_l, o_l, m_l, x2_l)) in hook_tensors.iter().enumerate() {
            for (stage, tensor) in [
                ("before-layer", x_in_l),
                ("after-attention", o_l),
                ("after-mlp", m_l),
                ("after-layer", x2_l),
            ] {
                let active = site_active(stage, Some(l));
                let f5_eliminated = stage == "after-attention"
                    && mode == ExecutionMode::PlannedFused
                    && layers[l]
                        .ops
                        .iter()
                        .any(|op| matches!(op, PlannedOp::FusedOProjResidual { .. }));
                sites.push(HookSiteRecord {
                    stage: stage.to_string(),
                    layer: Some(l),
                    tensor: (!f5_eliminated).then_some(tensor.id),
                    materialized: active && !f5_eliminated,
                    route: if f5_eliminated {
                        "fused-eliminated"
                    } else {
                        "unfused"
                    }
                    .to_string(),
                });
            }
        }
        for stage in ["before-logits", "after-logits"] {
            sites.push(HookSiteRecord {
                stage: stage.to_string(),
                layer: None,
                tensor: None,
                materialized: site_active(stage, None),
                route: "unfused".to_string(),
            });
        }

        let required: Vec<String> = dispatch
            .kernel_per_tensor
            .iter()
            .filter_map(|entry| entry.cpu_feature.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        #[cfg(target_arch = "x86_64")]
        let features = ["avx2", "fma", "f16c", "ssse3"]
            .into_iter()
            .filter(|feature| match *feature {
                "avx2" => std::is_x86_feature_detected!("avx2"),
                "fma" => std::is_x86_feature_detected!("fma"),
                "f16c" => std::is_x86_feature_detected!("f16c"),
                "ssse3" => std::is_x86_feature_detected!("ssse3"),
                _ => false,
            })
            .map(str::to_string)
            .collect();
        #[cfg(not(target_arch = "x86_64"))]
        let features = Vec::new();

        let plan = ExecutionPlan {
            schema_version: crate::plan::PLAN_SCHEMA_VERSION,
            kernel_revision: crate::plan::PLAN_KERNEL_REVISION,
            architecture: architecture.to_string(),
            model_sha256: model_sha256.unwrap_or_default().to_string(),
            tokenizer_sha256: tokenizer_sha256.unwrap_or_default().to_string(),
            gguf: crate::plan::GgufSummary {
                arch: architecture.to_string(),
                block_count: n_layers,
                embedding_length: embed,
                head_count: config.n_heads,
                head_count_kv: config.n_kv_heads,
                ffn_dim: inter,
                vocab_size: vocab,
                rope_dimension_count: config.head_dim,
                context_length: config.max_seq_len,
                rope_theta: config.rope_theta,
            },
            rope: RopeSummary {
                layout: rope_layout_name.to_string(),
                qk_norm_order: qk_norm_order_name.to_string(),
                has_q_norm: q_norm_present,
                has_k_norm: k_norm_present,
            },
            preamble,
            layers,
            final_ops,
            tensor_table,
            scratch,
            kv: KvLayout {
                precision: "f16".to_string(),
                layout: "layer-head-pos-dim".to_string(),
                layer_stride: config.n_kv_heads * max_seq_len * config.head_dim,
                head_stride: max_seq_len * config.head_dim,
                pos_stride: config.head_dim,
                head_dim: config.head_dim,
                n_kv_heads: config.n_kv_heads,
                max_seq: max_seq_len,
            },
            hook_sites: HookSitePlan {
                mode: hook_mode,
                active: active_stages.iter().map(|s| s.to_string()).collect(),
                sites,
            },
            dispatch,
            cpu: crate::plan::CpuSummary {
                features,
                threads: rayon::current_num_threads(),
                required,
            },
            provenance: PlanProvenance {
                ember_version: env!("CARGO_PKG_VERSION").to_string(),
                git_commit: option_env!("EMBER_GIT_COMMIT")
                    .unwrap_or("unknown")
                    .to_string(),
                rustc_version: option_env!("EMBER_RUSTC_VERSION")
                    .unwrap_or("unknown")
                    .to_string(),
                plan_build_time: format!(
                    "unix-{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0)
                ),
                execution_mode: mode,
                hook_mode,
                model_sha256: model_sha256.unwrap_or_default().to_string(),
                tokenizer_sha256: tokenizer_sha256.unwrap_or_default().to_string(),
            },
            plan_hash: String::new(),
        };
        plan.validate()?;
        Ok(plan.finalize())
    }
}

/// Register a linear weight (+ optional bias) in the plan's tensor table.
fn plan_linear(
    linear: &Linear<CpuBackend>,
    name: &str,
    builder: &mut PlanBuilder,
) -> (TensorRef, Option<TensorRef>) {
    use crate::model::WeightKindView;
    let weight = match linear.weight_kind() {
        WeightKindView::F32(t) => builder.add_weight(
            name,
            vec![t.shape()[1], t.shape()[0]],
            "f32",
            "eager_f32",
            KernelId::EagerF32,
            t.data().len() * 4,
            false,
        ),
        WeightKindView::Q8_0(w) => builder.add_weight(
            name,
            vec![w.out_features(), w.in_features()],
            "q8_0",
            "compressed",
            KernelId::Q8Packed,
            w.byte_len(),
            w.is_mapped(),
        ),
        WeightKindView::KQuant(w) => {
            let dtype = w.dtype().name();
            let exec = k_execution_name(w.execution());
            builder.add_weight(
                name,
                vec![w.out_features(), w.in_features()],
                dtype,
                exec,
                resolve_kernel(dtype, exec),
                w.byte_len(),
                w.is_mapped(),
            )
        }
    };
    let bias = linear.bias().map(|b| {
        builder.add_weight(
            &format!("{name}.bias"),
            b.shape().to_vec(),
            "f32",
            "eager_f32",
            KernelId::EagerF32,
            b.data().len() * 4,
            false,
        )
    });
    (weight, bias)
}
