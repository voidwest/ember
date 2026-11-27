use super::*;

impl<B: Backend> Gemma4Mlp<B> {
    pub(super) fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        use crate::trace::{self, OpKind};

        let seq_len = backend.shape(x)[0];
        let embed_dim = backend.shape(x)[1];
        let inter_dim = self.gate_proj.out_features(backend);
        let gate_up_span = gemma_trace_span!(
            "gate_up_proj",
            trace::current_layer(),
            OpKind::MatMulQ8_0,
            vec![seq_len, embed_dim],
            trace::bytes_matmul_input(
                seq_len,
                embed_dim,
                self.gate_proj.weight_bytes(backend) + self.up_proj.weight_bytes(backend),
            ),
            trace::flops_matmul(seq_len, inter_dim, embed_dim) * 2,
        );
        let (gate, up) =
            self.gate_proj
                .forward_packed_pair_or_generic(backend, x, &self.up_proj)?;
        if let Some(span) = gate_up_span {
            span.end(
                vec![seq_len, inter_dim * 2],
                trace::bytes_matmul_output(seq_len, inter_dim) * 2,
            );
        }

        let gelu_span = gemma_trace_span!(
            "gelu",
            trace::current_layer(),
            OpKind::Other,
            vec![seq_len, inter_dim],
            trace::bytes_from_shape(&[seq_len, inter_dim]),
            trace::flops_silu(seq_len * inter_dim),
        );
        let gate = gelu_tanh(backend, &gate)?;
        finish_trace_span(gelu_span, backend, &gate);

        let elemul_span = gemma_trace_span!(
            "elemul",
            trace::current_layer(),
            OpKind::Elemul,
            vec![seq_len, inter_dim],
            trace::bytes_from_shape(&[seq_len, inter_dim]) * 2,
            trace::flops_elemul(seq_len * inter_dim),
        );
        let gated = backend.elemul(&gate, &up)?;
        finish_trace_span(elemul_span, backend, &gated);

        let down_span = gemma_trace_span!(
            "down_proj",
            trace::current_layer(),
            OpKind::MatMulQ8_0,
            vec![seq_len, inter_dim],
            trace::bytes_matmul_input(seq_len, inter_dim, self.down_proj.weight_bytes(backend),),
            trace::flops_matmul(seq_len, embed_dim, inter_dim),
        );
        let out = self.down_proj.forward(backend, &gated)?;
        finish_trace_span(down_span, backend, &out);

        Ok(out)
    }
}

impl<B: Backend> Gemma4Attention<B> {
    #[allow(
        clippy::too_many_arguments,
        reason = "a forward pass takes its model state, input, cache and positions as separate arguments"
    )]
    pub(super) fn forward_with_cache(
        &self,
        backend: &B,
        x: &B::Tensor,
        cache: &mut KVCache,
        layer: usize,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        use crate::trace::{self, OpKind};

        let seq_len = backend.shape(x)[0];
        let embed_dim = backend.shape(x)[1];
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let projection_weight_bytes = self.q_proj.weight_bytes(backend)
            + self
                .k_proj
                .as_ref()
                .map_or(0, |projection| projection.weight_bytes(backend))
            + self
                .v_proj
                .as_ref()
                .map_or(0, |projection| projection.weight_bytes(backend));
        let projection_output_dim = if self.shared_source_layer.is_some() {
            q_dim
        } else {
            q_dim + 2 * kv_dim
        };
        let projection_span = gemma_trace_span!(
            if self.shared_source_layer.is_some() {
                "q_proj_shared_kv"
            } else {
                "qkv_proj"
            },
            layer,
            OpKind::MatMulQ8_0,
            vec![seq_len, embed_dim],
            trace::bytes_matmul_input(seq_len, embed_dim, projection_weight_bytes),
            trace::flops_matmul(seq_len, projection_output_dim, embed_dim),
        );
        let (mut q, projected_kv) = if self.shared_source_layer.is_some() {
            (self.q_proj.forward(backend, x)?, None)
        } else {
            let k_proj = self
                .k_proj
                .as_ref()
                .expect("non-shared Gemma 4 layer must have k_proj");
            let v_proj = self
                .v_proj
                .as_ref()
                .expect("non-shared Gemma 4 layer must have v_proj");
            let (q, k, v) = self.q_proj.forward_triple(backend, x, k_proj, v_proj)?;
            (q, Some((k, v)))
        };
        if let Some(span) = projection_span {
            span.end(
                vec![seq_len, projection_output_dim],
                trace::bytes_matmul_output(seq_len, projection_output_dim),
            );
        }

        let rope_q_span = gemma_trace_span!(
            "rope_q_qk_norm",
            layer,
            OpKind::RoPE,
            vec![seq_len, q_dim],
            trace::bytes_from_shape(&[seq_len, q_dim]),
            trace::flops_rope(seq_len, q_dim) + trace::flops_rms_norm(seq_len, q_dim),
        );
        q = apply_rope_and_qk_norm(
            backend,
            q,
            &self.q_norm,
            &self.rope_cos,
            &self.rope_sin,
            start_pos,
            self.n_heads,
            self.head_dim,
            self.norm_eps,
        )?;
        finish_trace_span(rope_q_span, backend, &q);

        let source_layer = if let Some(source_layer) = self.shared_source_layer {
            source_layer
        } else {
            let (k, v) =
                projected_kv.expect("non-shared Gemma 4 layer must project K/V activations");
            let rope_k_span = gemma_trace_span!(
                "rope_k_qk_norm",
                layer,
                OpKind::RoPE,
                vec![seq_len, kv_dim],
                trace::bytes_from_shape(&[seq_len, kv_dim]),
                trace::flops_rope(seq_len, kv_dim) + trace::flops_rms_norm(seq_len, kv_dim),
            );
            let k = apply_rope_and_qk_norm(
                backend,
                k,
                &self.k_norm,
                &self.rope_cos,
                &self.rope_sin,
                start_pos,
                self.n_kv_heads,
                self.head_dim,
                self.norm_eps,
            )?;
            finish_trace_span(rope_k_span, backend, &k);
            // llama.cpp applies a plain per-head RMS norm to V (no learned
            // weights, using f_norm_rms_eps) before storing it in the cache.
            let v = apply_v_rms_norm(backend, v, self.n_kv_heads, self.head_dim, self.norm_eps)?;
            gemma_dump!(layer, "q_post", backend, q);
            gemma_dump!(layer, "k_post", backend, k);
            gemma_dump!(layer, "v_post", backend, v);
            let k_data = backend.data(&k);
            let v_data = backend.data(&v);
            let kv_store_span = gemma_trace_span!(
                "kv_cache_store",
                layer,
                OpKind::Other,
                vec![seq_len, kv_dim],
                (k_data.len() + v_data.len()) * std::mem::size_of::<f32>(),
                0,
            );
            let cursor = cache.cursor();
            for pos in 0..seq_len {
                let offset = pos * kv_dim;
                cache.append_with_layout(
                    layer,
                    cursor + pos,
                    &k_data[offset..offset + kv_dim],
                    &v_data[offset..offset + kv_dim],
                    self.n_kv_heads,
                    self.head_dim,
                );
            }
            if let Some(span) = kv_store_span {
                span.end(vec![], 0);
            }
            layer
        };

        let total_seq_len = cache.cursor() + seq_len;
        let cache_head_dim = cache.layer_head_dim(source_layer);
        let max_seq_len = cache.max_seq_len();
        let attention_span = gemma_trace_span!(
            "attention_score",
            layer,
            OpKind::AttentionScore,
            vec![seq_len, q_dim],
            trace::bytes_from_shape(&[seq_len, q_dim])
                + total_seq_len * kv_dim * std::mem::size_of::<f16>() * 2,
            trace::flops_attention(seq_len, self.n_heads, self.head_dim, total_seq_len),
        );
        let (cached_k, cached_v, qk_scratch) = cache.get_with_scratch(source_layer);
        let out = cached_attention_with_scratch(
            backend,
            &q,
            cached_k,
            cached_v,
            Gemma4CachedAttentionSpec {
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
                head_dim: self.head_dim,
                cache_head_dim,
                max_seq_len,
                total_seq_len,
                sliding_window: if self.layer_type == Gemma4AttentionType::Local {
                    Some(self.sliding_window)
                } else {
                    None
                },
                scale: self.attention_scale,
            },
            qk_scratch,
        )?;
        finish_trace_span(attention_span, backend, &out);
        debug_assert_eq!(backend.shape(&out), &[seq_len, q_dim]);
        gemma_dump!(layer, "attn_result", backend, out);
        let output_span = gemma_trace_span!(
            "o_proj",
            layer,
            OpKind::MatMulQ8_0,
            vec![seq_len, q_dim],
            trace::bytes_matmul_input(seq_len, q_dim, self.o_proj.weight_bytes(backend)),
            trace::flops_matmul(seq_len, embed_dim, q_dim),
        );
        let output = self.o_proj.forward(backend, &out)?;
        finish_trace_span(output_span, backend, &output);
        gemma_dump!(layer, "o_proj", backend, output);
        Ok(output)
    }
}

impl<B: Backend> Gemma4Block<B> {
    /// Forward pass matching llama.cpp's gemma4 graph build exactly:
    ///   1. attn_norm → attention → post_attn_norm → + residual
    ///   2. ffn_norm → FFN → post_ffn_norm → + residual
    ///   3. PLE: inp_gate → gelu → × ple → proj → post_ple_norm → + residual
    ///   4. × layer_output_scale
    pub(super) fn forward_with_cache(
        &self,
        backend: &B,
        x: &B::Tensor,
        ple: Option<&B::Tensor>,
        cache: &mut KVCache,
        layer: usize,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let mut hooks = DisabledHooks;
        self.forward_with_cache_hooked(backend, x, ple, cache, layer, start_pos, &mut hooks)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "a forward pass takes its model state, input, cache and positions as separate arguments"
    )]
    pub(super) fn forward_with_cache_hooked<H>(
        &self,
        backend: &B,
        x: &B::Tensor,
        ple: Option<&B::Tensor>,
        cache: &mut KVCache,
        layer: usize,
        start_pos: usize,
        hooks: &mut H,
    ) -> Result<B::Tensor, B::Error>
    where
        H: LayerHooks<B::Tensor, B::Error>,
    {
        use crate::trace::{self, OpKind};

        trace::set_current_layer(layer);
        let seq_len = backend.shape(x)[0];
        let embed_dim = backend.shape(x)[1];
        let hidden_shape = [seq_len, embed_dim];
        let hidden_bytes = trace::bytes_from_shape(&hidden_shape);

        gemma_dump!(layer, "x_in", backend, *x);

        // 1. Self-attention
        let input_norm_span = gemma_trace_span!(
            "attn_rms_norm",
            layer,
            OpKind::RmsNorm,
            hidden_shape.to_vec(),
            hidden_bytes + trace::bytes_from_shape(backend.shape(&self.input_norm)),
            trace::flops_rms_norm(seq_len, embed_dim),
        );
        let normed = backend.rms_norm(x, &self.input_norm, self.norm_eps)?;
        finish_trace_span(input_norm_span, backend, &normed);
        gemma_dump!(layer, "attn_norm", backend, normed);
        let attn_out = self
            .attn
            .forward_with_cache(backend, &normed, cache, layer, start_pos)?;
        let post_attn_norm_span = gemma_trace_span!(
            "post_attn_rms_norm",
            layer,
            OpKind::RmsNorm,
            hidden_shape.to_vec(),
            hidden_bytes + trace::bytes_from_shape(backend.shape(&self.post_attn_norm)),
            trace::flops_rms_norm(seq_len, embed_dim),
        );
        let mut attn_out = backend.rms_norm(&attn_out, &self.post_attn_norm, self.norm_eps)?;
        finish_trace_span(post_attn_norm_span, backend, &attn_out);
        hooks.after_attention(layer, &mut attn_out)?;
        let attn_add_span = gemma_trace_span!(
            "attn_residual_add",
            layer,
            OpKind::ResidualAdd,
            hidden_shape.to_vec(),
            hidden_bytes * 2,
            trace::flops_residual_add(seq_len * embed_dim),
        );
        let x = backend.add(x, &attn_out)?;
        finish_trace_span(attn_add_span, backend, &x);
        gemma_dump!(layer, "post_attn_add", backend, x);

        // 2. Feed-forward network
        let pre_ffn_norm_span = gemma_trace_span!(
            "ffn_rms_norm",
            layer,
            OpKind::RmsNorm,
            hidden_shape.to_vec(),
            hidden_bytes + trace::bytes_from_shape(backend.shape(&self.pre_ffn_norm)),
            trace::flops_rms_norm(seq_len, embed_dim),
        );
        let normed = backend.rms_norm(&x, &self.pre_ffn_norm, self.norm_eps)?;
        finish_trace_span(pre_ffn_norm_span, backend, &normed);
        let mlp_out = self.mlp.forward(backend, &normed)?;
        gemma_dump!(layer, "ffn", backend, mlp_out);
        let post_ffn_norm_span = gemma_trace_span!(
            "post_ffn_rms_norm",
            layer,
            OpKind::RmsNorm,
            hidden_shape.to_vec(),
            hidden_bytes + trace::bytes_from_shape(backend.shape(&self.post_ffn_norm)),
            trace::flops_rms_norm(seq_len, embed_dim),
        );
        let mut mlp_out = backend.rms_norm(&mlp_out, &self.post_ffn_norm, self.norm_eps)?;
        finish_trace_span(post_ffn_norm_span, backend, &mlp_out);
        hooks.after_mlp(layer, &mut mlp_out)?;
        let ffn_add_span = gemma_trace_span!(
            "ffn_residual_add",
            layer,
            OpKind::ResidualAdd,
            hidden_shape.to_vec(),
            hidden_bytes * 2,
            trace::flops_residual_add(seq_len * embed_dim),
        );
        let mut x = backend.add(&x, &mlp_out)?;
        finish_trace_span(ffn_add_span, backend, &x);
        gemma_dump!(layer, "ffn_add", backend, x);

        // 3. Per-layer embedding (PLE)
        if let Some(ple) = ple {
            let ple_span = gemma_trace_span!(
                "per_layer_embedding",
                layer,
                OpKind::Other,
                hidden_shape.to_vec(),
                hidden_bytes + trace::bytes_from_shape(backend.shape(ple)),
                0,
            );
            x = self.add_ple(backend, &x, ple, layer)?;
            finish_trace_span(ple_span, backend, &x);
            gemma_dump!(layer, "ple_add", backend, x);
        }

        // 4. Layer output scaling
        if let Some(ref scale) = self.layer_output_scale {
            let scale_span = gemma_trace_span!(
                "layer_output_scale",
                layer,
                OpKind::Other,
                hidden_shape.to_vec(),
                hidden_bytes + std::mem::size_of::<f32>(),
                (seq_len * embed_dim) as u64,
            );
            backend.scale_in_place(&mut x, backend.data(scale)[0]);
            finish_trace_span(scale_span, backend, &x);
        }
        Ok(x)
    }

    /// Apply per-layer input following HF pathway:
    /// gate(hidden)→gelu→*PLE→proj→rms_norm→+residual
    fn add_ple(
        &self,
        backend: &B,
        x: &B::Tensor,
        ple: &B::Tensor,
        layer: usize,
    ) -> Result<B::Tensor, B::Error> {
        let gate = self
            .inp_gate
            .as_ref()
            .expect("Gemma 4 PLE requires inp_gate");
        let proj = self
            .ple_proj
            .as_ref()
            .expect("Gemma 4 PLE requires ple_proj");
        let norm = self
            .post_ple_norm
            .as_ref()
            .expect("Gemma 4 PLE requires post_ple_norm");

        // Gate: hidden (1536) → per_layer_dim (256), then GELU (tanh approx matching llama.cpp)
        gemma_dump!(layer, "ple_input", backend, *ple);
        let gated_in = gate.forward(backend, x)?;
        gemma_dump!(layer, "ple_gate_in", backend, gated_in);
        let gated = gelu_tanh(backend, &gated_in)?;
        gemma_dump!(layer, "ple_gelu", backend, gated);
        // Multiply gated hidden by PLE token embedding (both 256-dim)
        let multiplied = backend.elemul(&gated, ple)?;
        gemma_dump!(layer, "ple_mul", backend, multiplied);
        // Project back to hidden dim (256 → 1536)
        let projected = proj.forward(backend, &multiplied)?;
        gemma_dump!(layer, "ple_proj", backend, projected);
        // RMS norm
        let normed = backend.rms_norm(&projected, norm, self.norm_eps)?;
        gemma_dump!(layer, "ple_norm", backend, normed);
        // Add to residual
        backend.add(x, &normed)
    }
}

impl<B: Backend> Gemma4Head<B> {
    pub(super) fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        match self {
            Self::Linear(head) => head.forward(backend, x),
            Self::TiedEmbedding(Gemma4Embedding::F32(table)) => {
                tied_embedding_logits(backend, x, table)
            }
            Self::TiedEmbedding(Gemma4Embedding::Q8_0(table)) => backend.matmul_q8_0(x, table),
        }
    }

    pub(super) fn weight_bytes(&self, backend: &B) -> usize {
        match self {
            Self::Linear(head) => head.weight_bytes(backend),
            Self::TiedEmbedding(Gemma4Embedding::F32(table)) => {
                backend.shape(table).iter().product::<usize>() * std::mem::size_of::<f32>()
            }
            Self::TiedEmbedding(Gemma4Embedding::Q8_0(table)) => table.byte_len(),
        }
    }
}
