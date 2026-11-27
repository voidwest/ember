use super::load::{record_profiled_packed, record_profiled_q8};
use super::*;

impl Llama<CpuBackend> {
    pub(super) fn eligible_fast_decode_inter_dim(&self) -> Option<usize> {
        // The allocation-free path is validated for Llama's adjacent-pair
        // RoPE. Real Qwen3 end-to-end coverage showed decode divergence with
        // split-half RoPE, despite synthetic single-layer parity, so Qwen stays
        // on the generic implementation until exact real-model parity exists.
        if self.config.rope_layout != RopeLayout::AdjacentPair {
            return None;
        }

        let q_dim = self.config.n_heads.checked_mul(self.config.head_dim)?;
        let kv_dim = self.config.n_kv_heads.checked_mul(self.config.head_dim)?;
        let embed_dim = self.config.embed_dim;
        let mut inter_dim = None;

        for block in &self.blocks {
            let q = block.self_attn.q_proj.q8_weight_without_bias()?;
            let k = block.self_attn.k_proj.q8_weight_without_bias()?;
            let v = block.self_attn.v_proj.q8_weight_without_bias()?;
            let o = block.self_attn.o_proj.q8_weight_without_bias()?;
            let gate = block.mlp.gate_proj.q8_weight_without_bias()?;
            let up = block.mlp.up_proj.q8_weight_without_bias()?;
            let down = block.mlp.down_proj.q8_weight_without_bias()?;

            if q.in_features() != embed_dim
                || q.out_features() != q_dim
                || k.in_features() != embed_dim
                || k.out_features() != kv_dim
                || v.in_features() != embed_dim
                || v.out_features() != kv_dim
                || o.in_features() != q_dim
                || o.out_features() != embed_dim
                || gate.in_features() != embed_dim
                || up.in_features() != embed_dim
                || gate.out_features() != up.out_features()
                || down.in_features() != gate.out_features()
                || down.out_features() != embed_dim
            {
                return None;
            }

            match inter_dim {
                Some(expected) if expected != gate.out_features() => return None,
                None => inter_dim = Some(gate.out_features()),
                _ => {}
            }
        }

        let head = self.head.q8_weight_without_bias()?;
        if head.in_features() != embed_dim {
            return None;
        }
        inter_dim
    }

    /// Run the allocation-free Q8_0 path for a single decode token.
    ///
    /// `None` means the model uses a mixed/F32 weight layout or tracing is
    /// active, in which case the generic implementation remains authoritative.
    pub(super) fn forward_decode_fast(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Option<Result<CpuTensor, CpuError>> {
        let mut hooks = DisabledHooks;
        self.forward_decode_fast_hooked(backend, token_ids, cache, start_pos, &mut hooks, None)
            .map(|result| {
                result.map(|logits| {
                    logits.expect("fast decode without a caller buffer materializes logits")
                })
            })
    }

    /// Fast-path decode writing the `[vocab]` logits into `out`.
    ///
    /// `None` means the fast path is ineligible and the caller must fall back
    /// to another route; `Some(Ok(()))` means `out` holds the logits.
    pub(super) fn forward_decode_fast_into(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        out: &mut [f32],
    ) -> Option<Result<(), CpuError>> {
        let mut hooks = DisabledHooks;
        self.forward_decode_fast_hooked(backend, token_ids, cache, start_pos, &mut hooks, Some(out))
            .map(|result| {
                result.and_then(|logits| match logits {
                    None => Ok(()),
                    Some(_) => Err(CpuError::Kernel(
                        "fast decode materialized logits despite a caller buffer".into(),
                    )),
                })
            })
    }

    pub(super) fn forward_decode_fast_hooked<H>(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        hooks: &mut H,
        logits_out: Option<&mut [f32]>,
    ) -> Option<Result<Option<CpuTensor>, CpuError>>
    where
        H: for<'a> LayerHooks<SliceActivation<'a>, CpuError>,
    {
        if token_ids.len() != 1 || crate::trace::is_tracing() {
            return None;
        }
        cache.validate_start_pos(start_pos);
        let inter_dim = self.fast_decode_inter_dim?;
        hooks.note_dispatch(DispatchPath::Fast);
        let embed_dim = self.config.embed_dim;
        let q_dim = self.config.n_heads * self.config.head_dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;

        Some(LLAMA_DECODE_WORKSPACE.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            let needs_resize = workspace.as_ref().is_none_or(|current| {
                current.max_rows() != 1
                    || current.embed_dim() != embed_dim
                    || current.inter_dim() != inter_dim
                    || current.q_dim() != q_dim
                    || current.kv_dim() != kv_dim
            });
            if needs_resize {
                *workspace = Some(Workspace::new(
                    1,
                    embed_dim,
                    inter_dim,
                    self.config.n_heads,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                ));
            }
            self.forward_decode_with_workspace(
                backend,
                token_ids[0],
                cache,
                start_pos,
                workspace.as_mut().expect("decode workspace initialized"),
                hooks,
                logits_out,
            )
        }))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "a forward pass takes its model state, input, cache and positions as separate arguments"
    )]
    fn forward_decode_with_workspace<H>(
        &self,
        backend: &CpuBackend,
        token_id: u32,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        workspace: &mut Workspace,
        hooks: &mut H,
        logits_out: Option<&mut [f32]>,
    ) -> Result<Option<CpuTensor>, CpuError>
    where
        H: for<'a> LayerHooks<SliceActivation<'a>, CpuError>,
    {
        let embed_dim = self.config.embed_dim;
        let q_dim = self.config.n_heads * self.config.head_dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;
        let inter_dim = workspace.inter_dim();
        let profile_operators = crate::decode_profile::is_enabled();

        // Profiling helper for the non-matmul ops (norms, rope, kv store,
        // attention, silu, residual adds): guarded by the same
        // `profile_operators` flag as the matmul timings, so the normal
        // path pays nothing extra.
        macro_rules! profile_op {
            ($layer:expr_2021, $name:expr_2021, $in_dim:expr_2021, $out_dim:expr_2021, $body:block) => {{
                profile_op!($layer, $name, $in_dim, $out_dim, "not_applicable", $body)
            }};
            ($layer:expr_2021, $name:expr_2021, $in_dim:expr_2021, $out_dim:expr_2021, $quantization:expr_2021, $body:block) => {{
                let start = if profile_operators {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let result = $body;
                if let Some(start) = start {
                    crate::decode_profile::record_quantized(
                        $layer,
                        $name,
                        $in_dim,
                        $out_dim,
                        crate::decode_profile::DecodeExecutionMode::Serial,
                        $quantization,
                        start.elapsed(),
                    );
                }
                result
            }};
        }

        let Workspace {
            norm_out,
            residual_out,
            q_out,
            k_out,
            v_out,
            attn_out,
            gate_out,
            up_out,
            gated_out,
            mlp_out,
            ..
        } = workspace;
        let x = &mut residual_out[..embed_dim];
        let embedding_quantization = if profile_operators {
            match &self.embed_tokens {
                LlamaEmbedding::F32(_) => "F32",
                LlamaEmbedding::Half(weight) => match weight.dtype() {
                    crate::half_weight::HalfDtype::F16 => "F16",
                    crate::half_weight::HalfDtype::Bf16 => "BF16",
                },
                LlamaEmbedding::Q8_0(_) => "Q8_0",
                LlamaEmbedding::KQuant(weight) => match weight.dtype() {
                    crate::quant_k::KQuantDtype::Q4K => "Q4_K",
                    crate::quant_k::KQuantDtype::Q6K => "Q6_K",
                },
            }
        } else {
            "not_applicable"
        };
        profile_op!(
            usize::MAX,
            "embedding",
            1,
            embed_dim,
            embedding_quantization,
            {
                match &self.embed_tokens {
                    LlamaEmbedding::F32(table) => {
                        if token_id as usize >= table.shape()[0] {
                            return Err(CpuError::ShapeMismatch(format!(
                                "embedding token {} out of bounds for vocabulary {}",
                                token_id,
                                table.shape()[0]
                            )));
                        }
                        let embedding_start = token_id as usize * embed_dim;
                        x.copy_from_slice(
                            &table.data()[embedding_start..embedding_start + embed_dim],
                        );
                    }
                    LlamaEmbedding::Q8_0(table) => {
                        if token_id as usize >= table.out_features() {
                            return Err(CpuError::ShapeMismatch(format!(
                                "embedding token {} out of bounds for vocabulary {}",
                                token_id,
                                table.out_features()
                            )));
                        }
                        table.dequantize_row(token_id as usize, x);
                    }
                    LlamaEmbedding::Half(table) => {
                        if token_id as usize >= table.rows() {
                            return Err(CpuError::ShapeMismatch(format!(
                                "embedding token {} out of bounds for vocabulary {}",
                                token_id,
                                table.rows()
                            )));
                        }
                        table.dequantize_row(token_id as usize, x);
                    }
                    // Fast decode is ineligible for K-quant models (checked at
                    // construction); this arm exists for match exhaustiveness and
                    // must never fire.
                    LlamaEmbedding::KQuant(_) => {
                        return Err(CpuError::ShapeMismatch(
                            "fast decode path is ineligible for K-quant embeddings".into(),
                        ));
                    }
                }
            }
        );

        let norm = &mut norm_out[..embed_dim];
        let q = &mut q_out[..q_dim];
        let k = &mut k_out[..kv_dim];
        let v = &mut v_out[..kv_dim];
        let attention = &mut attn_out[..q_dim];
        let gate = &mut gate_out[..inter_dim];
        let up = &mut up_out[..inter_dim];
        let gated = &mut gated_out[..inter_dim];
        let projected = &mut mlp_out[..embed_dim];

        for (layer, block) in self.blocks.iter().enumerate() {
            {
                let mut hidden = SliceActivation::new(1, embed_dim, x);
                hooks.before_layer(layer, &mut hidden)?;
            }
            profile_op!(layer, "attn_rms_norm", embed_dim, embed_dim, "F32", {
                crate::simd::rms_norm_q8_decode_into(
                    x,
                    block.input_layernorm.data(),
                    block.norm_eps,
                    norm,
                )
            });

            let q_weight = block
                .self_attn
                .q_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let k_weight = block
                .self_attn
                .k_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let v_weight = block
                .self_attn
                .v_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let packed_q = block.self_attn.q_proj.packed_q8_weight_without_bias();
            let packed_k = block.self_attn.k_proj.packed_q8_weight_without_bias();
            let packed_v = block.self_attn.v_proj.packed_q8_weight_without_bias();
            if let (Some(packed_q), Some(packed_k), Some(packed_v)) = (packed_q, packed_k, packed_v)
            {
                if profile_operators {
                    let elapsed = backend.matmul_q8_0_packed_triple_into_timed(
                        norm, packed_q, packed_k, packed_v, q, k, v,
                    );
                    record_profiled_packed(layer, "q", packed_q, elapsed[0]);
                    record_profiled_packed(layer, "k", packed_k, elapsed[1]);
                    record_profiled_packed(layer, "v", packed_v, elapsed[2]);
                } else {
                    backend.matmul_q8_0_packed_triple_into(
                        norm, packed_q, packed_k, packed_v, q, k, v,
                    );
                }
            } else if profile_operators {
                let elapsed = backend
                    .matmul_q8_0_triple_into_timed(norm, q_weight, k_weight, v_weight, q, k, v);
                record_profiled_q8(layer, "q", q_weight, elapsed[0]);
                record_profiled_q8(layer, "k", k_weight, elapsed[1]);
                record_profiled_q8(layer, "v", v_weight, elapsed[2]);
            } else {
                backend.matmul_q8_0_triple_into(norm, 1, q_weight, k_weight, v_weight, q, k, v);
            }

            profile_op!(layer, "rope_q", q_dim, q_dim, {
                block.self_attn.apply_decode_rope_and_qk_norm(
                    q,
                    self.config.n_heads,
                    start_pos,
                    block.self_attn.q_norm.as_ref(),
                )
            });
            profile_op!(layer, "rope_k", kv_dim, kv_dim, {
                block.self_attn.apply_decode_rope_and_qk_norm(
                    k,
                    self.config.n_kv_heads,
                    start_pos,
                    block.self_attn.k_norm.as_ref(),
                )
            });

            let cursor = cache.cursor();
            profile_op!(layer, "kv_store", kv_dim * 2, kv_dim * 2, {
                cache.append(layer, cursor, k, v)
            });
            let attention_spec = CachedAttentionSpec {
                n_heads: self.config.n_heads,
                n_kv_heads: self.config.n_kv_heads,
                head_dim: self.config.head_dim,
                max_seq_len: cache.max_seq_len(),
                total_seq_len: cursor + 1,
            };
            let (cached_k, cached_v, qk_scratch) = cache.get_with_scratch(layer);
            profile_op!(layer, "attention", q_dim, q_dim, {
                backend.cached_causal_attention_into(
                    q,
                    cached_k,
                    cached_v,
                    attention_spec,
                    qk_scratch,
                    attention,
                )?
            });

            {
                let mut heads = SliceActivation::new(1, q_dim, attention);
                hooks.before_o_proj(layer, &mut heads)?;
            }
            let o_weight = block
                .self_attn
                .o_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let packed_o = block.self_attn.o_proj.packed_q8_weight_without_bias();
            if let Some(packed_o) = packed_o {
                if profile_operators {
                    let elapsed =
                        backend.matmul_q8_0_packed_into_timed(attention, packed_o, projected);
                    record_profiled_packed(layer, "o", packed_o, elapsed);
                } else {
                    backend.matmul_q8_0_packed_into(attention, packed_o, projected);
                }
            } else if profile_operators {
                let elapsed = backend.matmul_q8_0_into_timed(attention, o_weight, projected);
                record_profiled_q8(layer, "o", o_weight, elapsed);
            } else {
                backend.matmul_q8_0_into(attention, 1, o_weight, projected);
            }
            {
                let mut attention_output = SliceActivation::new(1, embed_dim, projected);
                hooks.after_attention(layer, &mut attention_output)?;
            }
            profile_op!(layer, "attn_residual_add", embed_dim, embed_dim, {
                crate::simd::add_assign(x, projected)
            });

            profile_op!(layer, "ffn_rms_norm", embed_dim, embed_dim, "F32", {
                crate::simd::rms_norm_q8_decode_into(
                    x,
                    block.post_attention_layernorm.data(),
                    block.norm_eps,
                    norm,
                )
            });
            let gate_weight = block
                .mlp
                .gate_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let up_weight = block
                .mlp
                .up_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let packed_gate = block.mlp.gate_proj.packed_q8_weight_without_bias();
            let packed_up = block.mlp.up_proj.packed_q8_weight_without_bias();
            if let (Some(packed_gate), Some(packed_up)) = (packed_gate, packed_up) {
                if profile_operators {
                    let elapsed = backend.matmul_q8_0_packed_pair_into_timed(
                        norm,
                        packed_gate,
                        packed_up,
                        gate,
                        up,
                    );
                    record_profiled_packed(layer, "gate", packed_gate, elapsed[0]);
                    record_profiled_packed(layer, "up", packed_up, elapsed[1]);
                } else {
                    backend.matmul_q8_0_packed_pair_into(norm, packed_gate, packed_up, gate, up);
                }
            } else if profile_operators {
                let elapsed =
                    backend.matmul_q8_0_pair_into_timed(norm, gate_weight, up_weight, gate, up);
                record_profiled_q8(layer, "gate", gate_weight, elapsed[0]);
                record_profiled_q8(layer, "up", up_weight, elapsed[1]);
            } else {
                backend.matmul_q8_0_pair_into(norm, 1, gate_weight, up_weight, gate, up);
            }
            let down_weight = block
                .mlp
                .down_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let packed_down = block.mlp.down_proj.packed_q8_weight_without_bias();
            if !profile_operators {
                // SiLU, its quantization and the down projection share one
                // pass over the intermediate width on the decode team.
                backend.silu_mul_matmul_q8_0_decode_into(
                    gate,
                    up,
                    gated,
                    down_weight,
                    packed_down,
                    projected,
                );
            } else {
                profile_op!(layer, "silu_mul", inter_dim, inter_dim, {
                    crate::simd::silu_mul_into(gate, up, gated)
                });
                if let Some(packed_down) = packed_down {
                    let elapsed =
                        backend.matmul_q8_0_packed_into_timed(gated, packed_down, projected);
                    record_profiled_packed(layer, "down", packed_down, elapsed);
                } else {
                    let elapsed = backend.matmul_q8_0_into_timed(gated, down_weight, projected);
                    record_profiled_q8(layer, "down", down_weight, elapsed);
                }
            }
            {
                let mut mlp_output = SliceActivation::new(1, embed_dim, projected);
                hooks.after_mlp(layer, &mut mlp_output)?;
            }
            profile_op!(layer, "mlp_residual_add", embed_dim, embed_dim, {
                crate::simd::add_assign(x, projected)
            });
            {
                let mut hidden = SliceActivation::new(1, embed_dim, x);
                hooks.after_layer(layer, &mut hidden)?;
            }
        }
        cache.advance_cursor();

        profile_op!(usize::MAX, "final_rms_norm", embed_dim, embed_dim, "F32", {
            crate::simd::rms_norm_q8_decode_into(x, self.norm.data(), self.config.norm_eps, norm)
        });
        {
            let mut hidden = SliceActivation::new(1, embed_dim, norm);
            hooks.before_logits(&mut hidden)?;
        }
        let head_weight = self
            .head
            .q8_weight_without_bias()
            .expect("fast path eligibility checked");
        let mut owned_logits = Vec::new();
        let wrote_caller_logits = logits_out.is_some();
        let logits: &mut [f32] = match logits_out {
            Some(out) => {
                if out.len() != head_weight.out_features() {
                    return Err(CpuError::ShapeMismatch(format!(
                        "logits buffer has {} values, expected {}",
                        out.len(),
                        head_weight.out_features()
                    )));
                }
                out.fill(0.0);
                out
            }
            None => {
                owned_logits = vec![0.0; head_weight.out_features()];
                &mut owned_logits
            }
        };
        if let Some(interleaved) = self.head.interleaved.as_ref() {
            if profile_operators {
                let elapsed = backend.matmul_q8_0_interleaved_into_timed(norm, interleaved, logits);
                crate::decode_profile::record_quantized(
                    usize::MAX,
                    "lm_head",
                    interleaved.in_features(),
                    interleaved.out_features(),
                    if rayon::current_num_threads() > 1 {
                        crate::decode_profile::DecodeExecutionMode::InterleavedRowParallelRayon
                    } else {
                        crate::decode_profile::DecodeExecutionMode::InterleavedSerial
                    },
                    "Q8_0",
                    elapsed,
                );
            } else {
                backend.matmul_q8_0_interleaved_into(norm, interleaved, logits);
            }
        } else if profile_operators {
            let elapsed = backend.matmul_q8_0_into_timed(norm, head_weight, logits);
            record_profiled_q8(usize::MAX, "lm_head", head_weight, elapsed);
        } else {
            backend.matmul_q8_0_into(norm, 1, head_weight, logits);
        }
        {
            let mut output = SliceActivation::new(1, head_weight.out_features(), logits);
            hooks.after_logits(&mut output)?;
        }
        if wrote_caller_logits {
            return Ok(None);
        }
        Ok(Some(CpuTensor::from_data(
            vec![1, head_weight.out_features()],
            owned_logits,
        )))
    }
}
