use crate::hook_types::DispatchPath;
/// Shared rejection message: gemma4's K-quant path is not implemented.
const NO_K_QUANT: &str = "gemma4 does not support compressed K-quant tensors in v0.3";
/// Gemma widens f16/bf16 tensors on entry (`materialize_half_tensors`), so
/// a `Half` tensor reaching a match is an internal ordering error.
const HALF_NOT_WIDENED: &str = "internal: f16/bf16 tensor reached Gemma construction unwidened";

use crate::backend::{Backend, CpuBackend, CpuError};
use crate::experiments::{
    ActiveHooks, DisabledHooks, ExecutionContext, ExperimentRunner, ExperimentalForwardModel,
    LayerHooks,
};
use crate::kv_cache::KVCache;
use crate::loader::{GgufLoader, GgufValue, LoadedTensor};
use crate::model::{pool_layer_activation, ForwardModel, Linear};
use crate::tensor::{compute_rope_freqs, CpuTensor};
use alloc::vec::Vec;
use half::f16;
use std::sync::Arc;

macro_rules! gemma_trace_span {
    ($($argument:tt)*) => {
        if crate::trace::is_tracing() {
            crate::trace::span($($argument)*)
        } else {
            None
        }
    };
}

fn finish_trace_span<B: Backend>(
    span: Option<crate::trace::TraceSpan>,
    backend: &B,
    output: &B::Tensor,
) {
    if let Some(span) = span {
        let values = crate::trace::values_enabled()
            .then(|| crate::trace::compute_tensor_values(backend.data(output)));
        span.end_with_values(
            backend.shape(output).to_vec(),
            crate::trace::bytes_from_shape(backend.shape(output)),
            values,
        );
    }
}

/// Dump a tensor as raw f32 when EMBER_GEMMA_DUMP is set (forensics).
macro_rules! gemma_dump {
    ($layer:expr, $stage:expr, $backend:expr, $tensor:expr) => {{
        if let Some(dir) = std::env::var_os("EMBER_GEMMA_DUMP") {
            if $layer < 35 {
                let data = $backend.data(&$tensor);
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for v in data.iter() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                let path = std::path::Path::new(&dir).join(format!("L{}_{}.f32", $layer, $stage));
                let _ = std::fs::write(path, &bytes);
            }
        }
    }};
}

mod config;
mod layers;
mod load;
pub use config::{Gemma4AttentionType, Gemma4Config};
#[cfg(test)]
use load::take_gemma4_linear;

struct Gemma4Mlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

struct Gemma4Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Option<Linear<B>>,
    v_proj: Option<Linear<B>>,
    o_proj: Linear<B>,
    q_norm: B::Tensor,
    k_norm: B::Tensor,
    rope_cos: Arc<B::Tensor>,
    rope_sin: Arc<B::Tensor>,
    layer_type: Gemma4AttentionType,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    norm_eps: f32,
    attention_scale: f32,
    shared_source_layer: Option<usize>,
}

struct Gemma4Block<B: Backend> {
    input_norm: B::Tensor,
    attn: Gemma4Attention<B>,
    /// PLE gate: projects hidden 1536→256 (before GELU)
    inp_gate: Option<Linear<B>>,
    /// PLE projection: projects gated PLE 256→1536
    ple_proj: Option<Linear<B>>,
    /// RMS norm after PLE projection
    post_ple_norm: Option<B::Tensor>,
    /// Learned scalar applied to entire block output
    layer_output_scale: Option<B::Tensor>,
    post_attn_norm: B::Tensor,
    pre_ffn_norm: B::Tensor,
    mlp: Gemma4Mlp<B>,
    post_ffn_norm: B::Tensor,
    norm_eps: f32,
}

enum Gemma4Ple<B: Backend> {
    Hidden(B::Tensor),
    PackedQ8 {
        embeddings: crate::quant::QuantizedWeight,
        per_layer_dim: usize,
    },
}

#[derive(Clone)]
enum Gemma4Embedding<B: Backend> {
    F32(Arc<B::Tensor>),
    Q8_0(Arc<crate::quant::QuantizedWeight>),
}

enum Gemma4Head<B: Backend> {
    Linear(Linear<B>),
    TiedEmbedding(Gemma4Embedding<B>),
}

pub struct Gemma4<B: Backend> {
    embed_tokens: Gemma4Embedding<B>,
    blocks: Vec<Gemma4Block<B>>,
    norm: B::Tensor,
    head: Gemma4Head<B>,
    ple: Option<Gemma4Ple<B>>,
    /// Global PLE projection: hidden 1536→8960 (256×35 layers)
    per_layer_model_proj: Option<Linear<B>>,
    /// RMS norm weight for global PLE projection
    per_layer_proj_norm: Option<B::Tensor>,
    pub config: Gemma4Config,
}

impl<B: Backend> ForwardModel<B> for Gemma4<B> {
    fn create_cache(&self, _backend: &B, max_seq_len: usize) -> KVCache {
        use crate::kv_cache::KvLayerLayout;
        let layouts: Vec<_> = self
            .blocks
            .iter()
            .map(|block| {
                let attention = &block.attn;
                match attention.shared_source_layer {
                    Some(source_layer) => KvLayerLayout::Shared { source_layer },
                    None => KvLayerLayout::Owned {
                        n_kv_heads: attention.n_kv_heads,
                        head_dim: attention.head_dim,
                    },
                }
            })
            .collect();
        KVCache::try_new_per_layer(&layouts, max_seq_len)
            .expect("invalid or unallocatable Gemma KV cache geometry")
    }

    fn max_seq_len(&self, _backend: &B) -> usize {
        self.config.max_seq_len
    }

    fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        Gemma4::forward_with_cache(self, backend, token_ids, cache, start_pos)
    }

    fn forward_last_logits_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        Gemma4::forward_last_logits_with_cache(self, backend, token_ids, cache, start_pos)
    }

    fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    fn vocab_size(&self, _backend: &B) -> usize {
        self.config.vocab_size
    }

    fn forward_with_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        Gemma4::forward_with_activations(self, backend, token_ids)
    }

    fn forward_pooled_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        Gemma4::forward_pooled_activations(self, backend, token_ids, token_index_groups)
    }

    fn forward_pooled_hidden_states(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>, B::Error> {
        Gemma4::forward_pooled_hidden_states(self, backend, token_ids, token_index_groups)
    }
}

impl ExperimentalForwardModel for Gemma4<CpuBackend> {
    fn forward_last_logits_with_experiment(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut KVCache,
        start_pos: usize,
        execution: ExecutionContext<'_>,
        runner: &mut ExperimentRunner,
    ) -> Result<CpuTensor, CpuError> {
        // Gemma's attention path does not fire the per-head stage.
        runner.refuse_stage(
            crate::hook_types::ActivationStage::BeforeOProj,
            self.blocks.len(),
            execution.phase,
            "gemma4",
        )?;
        runner.note_dispatch(execution.phase, DispatchPath::Generic);
        let mut hooks = ActiveHooks::new(runner, execution);
        self.forward_last_logits_with_cache_hooked(backend, token_ids, cache, start_pos, &mut hooks)
    }
}

impl<B: Backend> Gemma4<B> {
    pub fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        cache.validate_start_pos(start_pos);
        assert!(
            !token_ids.is_empty(),
            "Gemma 4 forward requires at least one token"
        );
        let mut x = embed_tokens(
            backend,
            &self.embed_tokens,
            token_ids,
            self.config.embed_dim,
        )?;
        let ple = self.ple_vectors(backend, token_ids, &x)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            let layer_ple = ple.as_ref().map(|v| &v[layer]);
            x = block.forward_with_cache(backend, &x, layer_ple, cache, layer, start_pos)?;
        }
        for _ in 0..token_ids.len() {
            cache.advance_cursor();
        }
        let x = backend.rms_norm(&x, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &x)?;
        softcap_logits(backend, logits, self.config.final_logit_softcap)
    }

    pub fn forward_last_logits_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let mut hooks = DisabledHooks;
        self.forward_last_logits_with_cache_hooked(backend, token_ids, cache, start_pos, &mut hooks)
    }

    fn forward_last_logits_with_cache_hooked<H>(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut KVCache,
        start_pos: usize,
        hooks: &mut H,
    ) -> Result<B::Tensor, B::Error>
    where
        H: LayerHooks<B::Tensor, B::Error>,
    {
        use crate::trace::{self, OpKind};

        cache.validate_start_pos(start_pos);
        assert!(
            !token_ids.is_empty(),
            "Gemma 4 forward requires at least one token"
        );
        let seq_len = token_ids.len();
        let embed_dim = self.config.embed_dim;
        let embedding_span = gemma_trace_span!(
            "embedding",
            usize::MAX,
            OpKind::Embedding,
            vec![seq_len, embed_dim],
            trace::bytes_from_shape(&[seq_len, embed_dim]),
            trace::flops_embedding(),
        );
        let mut x = embed_tokens(backend, &self.embed_tokens, token_ids, embed_dim)?;
        finish_trace_span(embedding_span, backend, &x);
        let ple = self.ple_vectors(backend, token_ids, &x)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            hooks.before_layer(layer, &mut x)?;
            let layer_ple = ple.as_ref().map(|v| &v[layer]);
            x = block.forward_with_cache_hooked(
                backend, &x, layer_ple, cache, layer, start_pos, hooks,
            )?;
            hooks.after_layer(layer, &mut x)?;
        }
        for _ in 0..token_ids.len() {
            cache.advance_cursor();
        }
        let last = backend.row_as_2d(&x, token_ids.len() - 1)?;
        let final_norm_span = gemma_trace_span!(
            "final_norm",
            usize::MAX,
            OpKind::RmsNorm,
            vec![1, embed_dim],
            trace::bytes_from_shape(&[1, embed_dim])
                + trace::bytes_from_shape(backend.shape(&self.norm)),
            trace::flops_rms_norm(1, embed_dim),
        );
        let mut last = backend.rms_norm(&last, &self.norm, self.config.norm_eps)?;
        finish_trace_span(final_norm_span, backend, &last);
        hooks.before_logits(&mut last)?;
        let head_span = gemma_trace_span!(
            "lm_head",
            usize::MAX,
            OpKind::MatMulQ8_0,
            vec![1, embed_dim],
            trace::bytes_matmul_input(1, embed_dim, self.head.weight_bytes(backend)),
            trace::flops_matmul(1, self.config.vocab_size, embed_dim),
        );
        let logits = self.head.forward(backend, &last)?;
        finish_trace_span(head_span, backend, &logits);
        let softcap_span = gemma_trace_span!(
            "logit_softcap",
            usize::MAX,
            OpKind::Other,
            backend.shape(&logits).to_vec(),
            trace::bytes_from_shape(backend.shape(&logits)),
            backend.data(&logits).len() as u64,
        );
        let mut logits = softcap_logits(backend, logits, self.config.final_logit_softcap)?;
        finish_trace_span(softcap_span, backend, &logits);
        hooks.after_logits(&mut logits)?;
        Ok(logits)
    }

    /// Run a forward pass and return (per-layer states, logits).
    ///
    /// Per-layer states are taken from the last prompt token after each
    /// block's final residual add and layer_output_scale. This boundary
    /// matches the llama.cpp per-layer dump (after `build_cvec` in gemma4.cpp).
    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    pub fn forward_last_logits_with_layer_dump(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut KVCache,
        start_pos: usize,
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        let mut x = embed_tokens(
            backend,
            &self.embed_tokens,
            token_ids,
            self.config.embed_dim,
        )?;
        let ple = self.ple_vectors(backend, token_ids, &x)?;
        let mut layer_states: Vec<Vec<f32>> = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let layer_ple = ple.as_ref().map(|v| &v[layer]);
            x = block.forward_with_cache(backend, &x, layer_ple, cache, layer, start_pos)?;
            let data = backend.data(&x);
            let off = (token_ids.len() - 1) * self.config.embed_dim;
            layer_states.push(data[off..off + self.config.embed_dim].to_vec());
        }
        for _ in 0..token_ids.len() {
            cache.advance_cursor();
        }
        let last = backend.row_as_2d(&x, token_ids.len() - 1)?;
        let last = backend.rms_norm(&last, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &last)?;
        let logits = softcap_logits(backend, logits, self.config.final_logit_softcap)?;
        Ok((layer_states, logits))
    }

    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    pub fn forward_with_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        let mut x = embed_tokens(
            backend,
            &self.embed_tokens,
            token_ids,
            self.config.embed_dim,
        )?;
        let ple = self.ple_vectors(backend, token_ids, &x)?;
        let mut activations = Vec::with_capacity(self.blocks.len());
        let mut cache = self.create_cache(backend, token_ids.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let layer_ple = ple.as_ref().map(|v| &v[layer]);
            x = block.forward_with_cache(backend, &x, layer_ple, &mut cache, layer, 0)?;
            activations.push(backend.data(&x).to_vec());
        }
        for _ in 0..token_ids.len() {
            cache.advance_cursor();
        }
        let x = backend.rms_norm(&x, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &x)?;
        Ok((
            activations,
            softcap_logits(backend, logits, self.config.final_logit_softcap)?,
        ))
    }

    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    pub fn forward_pooled_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        let (pooled, x) =
            self.forward_pooled_hidden_and_output(backend, token_ids, token_index_groups)?;
        let last = backend.row_as_2d(&x, token_ids.len() - 1)?;
        let last = backend.rms_norm(&last, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &last)?;
        Ok((
            pooled,
            softcap_logits(backend, logits, self.config.final_logit_softcap)?,
        ))
    }

    pub fn forward_pooled_hidden_states(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>, B::Error> {
        self.forward_pooled_hidden_and_output(backend, token_ids, token_index_groups)
            .map(|(pooled, _)| pooled)
    }

    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    fn forward_pooled_hidden_and_output(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        let embed_dim = self.config.embed_dim;
        let mut pooled = token_index_groups
            .iter()
            .map(|_| vec![0.0f32; self.blocks.len() * embed_dim])
            .collect::<Vec<_>>();
        let mut x = embed_tokens(
            backend,
            &self.embed_tokens,
            token_ids,
            self.config.embed_dim,
        )?;
        let ple = self.ple_vectors(backend, token_ids, &x)?;
        let mut cache = self.create_cache(backend, token_ids.len());
        for (li, block) in self.blocks.iter().enumerate() {
            let layer_ple = ple.as_ref().map(|v| &v[li]);
            x = block.forward_with_cache(backend, &x, layer_ple, &mut cache, li, 0)?;
            let data = backend.data(&x);
            for (gi, token_indices) in token_index_groups.iter().enumerate() {
                let offset = li * embed_dim;
                pool_layer_activation(
                    data,
                    token_indices,
                    embed_dim,
                    &mut pooled[gi][offset..offset + embed_dim],
                );
            }
        }
        for _ in 0..token_ids.len() {
            cache.advance_cursor();
        }
        Ok((pooled, x))
    }

    fn ple_vectors(
        &self,
        backend: &B,
        token_ids: &[u32],
        hidden: &B::Tensor,
    ) -> Result<Option<Vec<B::Tensor>>, B::Error> {
        let Some(ref ple) = self.ple else {
            return Ok(None);
        };

        // 1. Look up raw per-layer token embeddings, scaled by sqrt(per_layer_dim)
        let per_layer_dim = match ple {
            Gemma4Ple::Hidden(t) => backend.shape(t)[2],
            Gemma4Ple::PackedQ8 { per_layer_dim, .. } => *per_layer_dim,
        };
        let ple_scale = (per_layer_dim as f32).sqrt();

        let raw_ple = match ple {
            Gemma4Ple::Hidden(ple) => {
                let shape = backend.shape(ple);
                let layer_stride = shape[1] * shape[2];
                let ple_data = backend.data(ple);
                let dim = shape[2];
                let mut out = Vec::with_capacity(self.config.n_layers);
                for layer in 0..self.config.n_layers {
                    let mut data = vec![0.0; token_ids.len() * dim];
                    for (pos, &tok) in token_ids.iter().enumerate() {
                        let token = (tok as usize).min(shape[1] - 1);
                        let src = layer * layer_stride + token * dim;
                        let dst = pos * dim;
                        for d in 0..dim {
                            data[dst + d] = ple_data[src + d] * ple_scale;
                        }
                    }
                    out.push(data);
                }
                out
            }
            Gemma4Ple::PackedQ8 {
                embeddings,
                per_layer_dim,
            } => {
                let packed_dim = embeddings.in_features();
                let expected = self.config.n_layers * *per_layer_dim;
                debug_assert_eq!(packed_dim, expected);
                let mut out =
                    vec![vec![0.0; token_ids.len() * *per_layer_dim]; self.config.n_layers];
                let mut row = vec![0.0; packed_dim];
                for (pos, &tok) in token_ids.iter().enumerate() {
                    let token = (tok as usize).min(embeddings.out_features() - 1);
                    embeddings.dequantize_row(token, &mut row);
                    for (layer, data) in out.iter_mut().enumerate() {
                        let layer_start = layer * *per_layer_dim;
                        let dst = pos * *per_layer_dim;
                        for (output, &value) in data[dst..dst + *per_layer_dim]
                            .iter_mut()
                            .zip(&row[layer_start..layer_start + *per_layer_dim])
                        {
                            *output = value * ple_scale;
                        }
                    }
                }
                out
            }
        };

        // 2. Global projection: hidden (1536) → per_layer_dim * n_layers (8960)
        let proj = if let Some(ref proj_layer) = self.per_layer_model_proj {
            let mut projected = proj_layer.forward(backend, hidden)?;
            let proj_scale = (self.config.embed_dim as f32).sqrt().recip();
            backend.scale_in_place(&mut projected, proj_scale);
            Some(projected)
        } else {
            None
        };

        // 3. Combine global projection with raw PLE (llama.cpp pathway)
        let n_tokens = token_ids.len();
        let mut out = Vec::with_capacity(self.config.n_layers);
        let combine_scale: f32 = 2.0_f32.sqrt().recip(); // 1/sqrt(2)

        for (layer, mut data) in raw_ple.into_iter().enumerate() {
            if let Some(ref proj) = proj {
                let proj_data = backend.data(proj);
                let layer_offset = layer * per_layer_dim;
                let proj_stride = self.config.n_layers * per_layer_dim;
                let norm_weight = self
                    .per_layer_proj_norm
                    .as_ref()
                    .expect("per_layer_proj_norm missing");
                let norm_data = backend.data(norm_weight);
                for t in 0..n_tokens {
                    let mut sq_sum = 0.0f32;
                    let base = t * proj_stride + layer_offset;
                    for d in 0..per_layer_dim {
                        sq_sum += proj_data[base + d] * proj_data[base + d];
                    }
                    let rstd = (sq_sum / per_layer_dim as f32 + self.config.norm_eps)
                        .sqrt()
                        .recip();
                    for d in 0..per_layer_dim {
                        let proj_val = proj_data[base + d] * rstd * norm_data[d];
                        data[t * per_layer_dim + d] =
                            (data[t * per_layer_dim + d] + proj_val) * combine_scale;
                    }
                }
            }

            out.push(backend.load_from_cpu(data, &[n_tokens, per_layer_dim])?);
        }

        Ok(Some(out))
    }
}

fn embed_tokens<B: Backend>(
    backend: &B,
    table: &Gemma4Embedding<B>,
    token_ids: &[u32],
    embed_dim: usize,
) -> Result<B::Tensor, B::Error> {
    let mut x = backend.zeroes(&[token_ids.len(), embed_dim])?;
    for (i, &tok) in token_ids.iter().enumerate() {
        match table {
            Gemma4Embedding::F32(table) => {
                backend.assign_row_from_table(&mut x, i, table, tok as usize)?;
            }
            Gemma4Embedding::Q8_0(table) => {
                backend.assign_row_from_q8_0(&mut x, i, table, tok as usize)?;
            }
        }
    }
    // Matching llama.cpp: scale token embeddings by sqrt(n_embd)
    let emb_scale = (embed_dim as f32).sqrt();
    backend.scale_in_place(&mut x, emb_scale);
    Ok(x)
}

fn tied_embedding_logits<B: Backend>(
    backend: &B,
    x: &B::Tensor,
    table: &B::Tensor,
) -> Result<B::Tensor, B::Error> {
    let x_shape = backend.shape(x);
    let table_shape = backend.shape(table);
    debug_assert_eq!(x_shape.len(), 2);
    debug_assert_eq!(table_shape.len(), 2);
    let seq_len = x_shape[0];
    let embed_dim = x_shape[1];
    let vocab_size = table_shape[0];
    debug_assert_eq!(table_shape[1], embed_dim);

    let x_data = backend.data(x);
    let table_data = backend.data(table);
    let mut out = vec![0.0; seq_len * vocab_size];
    for s in 0..seq_len {
        let x_row = &x_data[s * embed_dim..(s + 1) * embed_dim];
        for tok in 0..vocab_size {
            let emb = &table_data[tok * embed_dim..(tok + 1) * embed_dim];
            out[s * vocab_size + tok] = dot(x_row, emb);
        }
    }
    backend.load_from_cpu(out, &[seq_len, vocab_size])
}

fn gelu_tanh<B: Backend>(backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
    let data = backend
        .data(x)
        .iter()
        .map(|&v| {
            let inner = 0.797_884_6 * (v + 0.044_715 * v * v * v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect();
    backend.load_from_cpu(data, backend.shape(x))
}

fn softcap_logits<B: Backend>(
    backend: &B,
    mut logits: B::Tensor,
    softcap: Option<f32>,
) -> Result<B::Tensor, B::Error> {
    let Some(cap) = softcap else {
        return Ok(logits);
    };
    backend.softcap_in_place(&mut logits, cap)?;
    Ok(logits)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the RoPE tables, norm weight and head shape stay separate arguments"
)]
fn apply_rope_and_qk_norm<B: Backend>(
    backend: &B,
    x: B::Tensor,
    norm: &B::Tensor,
    rope_cos: &B::Tensor,
    rope_sin: &B::Tensor,
    start_pos: usize,
    n_heads: usize,
    head_dim: usize,
    norm_eps: f32,
) -> Result<B::Tensor, B::Error> {
    let seq_len = backend.shape(&x)[0];
    let width = n_heads * head_dim;
    let half = head_dim / 2;
    let mut data = backend.into_cpu_data(x);
    let cos = backend.data(rope_cos);
    let sin = backend.data(rope_sin);
    let norm_data = backend.data(norm);

    for s in 0..seq_len {
        let pos = start_pos + s;
        let cos_row = &cos[pos * half..(pos + 1) * half];
        let sin_row = &sin[pos * half..(pos + 1) * half];
        for h in 0..n_heads {
            let base = s * width + h * head_dim;
            let mut sq_sum = 0.0;
            for d in 0..head_dim {
                sq_sum += data[base + d] * data[base + d];
            }
            let rstd = (sq_sum / head_dim as f32 + norm_eps).sqrt().recip();
            for d in 0..head_dim {
                data[base + d] = data[base + d] * rstd * norm_data[d];
            }

            // NOTE: the bundled llama_cpp reference matches the SPLIT-HALF
            // layout at L0/L1 empirically (master's LLAMA_ROPE_TYPE_NONE is
            // adjacent-pair and does NOT match); keep split-half until the
            // bundled reference version is pinned.
            for d in 0..half {
                let a = data[base + d];
                let b = data[base + d + half];
                data[base + d] = a * cos_row[d] - b * sin_row[d];
                data[base + d + half] = a * sin_row[d] + b * cos_row[d];
            }
        }
    }
    backend.load_from_cpu(data, &[seq_len, width])
}

/// Plain per-head RMS norm (no learned weights): llama.cpp's gemma4 path
/// normalizes V with `ggml_rms_norm(eps)` before caching it.
fn apply_v_rms_norm<B: Backend>(
    backend: &B,
    x: B::Tensor,
    n_heads: usize,
    head_dim: usize,
    norm_eps: f32,
) -> Result<B::Tensor, B::Error> {
    let seq_len = backend.shape(&x)[0];
    let width = n_heads * head_dim;
    let mut data = backend.into_cpu_data(x);
    for s in 0..seq_len {
        for h in 0..n_heads {
            let base = s * width + h * head_dim;
            let mut sq_sum = 0.0f32;
            for d in 0..head_dim {
                sq_sum += data[base + d] * data[base + d];
            }
            let rstd = (sq_sum / head_dim as f32 + norm_eps).sqrt().recip();
            for d in 0..head_dim {
                data[base + d] *= rstd;
            }
        }
    }
    backend.load_from_cpu(data, &[seq_len, width])
}

struct Gemma4CachedAttentionSpec {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cache_head_dim: usize,
    max_seq_len: usize,
    total_seq_len: usize,
    sliding_window: Option<usize>,
    scale: f32,
}

fn cached_attention_with_scratch<B: Backend>(
    backend: &B,
    q: &B::Tensor,
    cached_k: &[f16],
    cached_v: &[f16],
    spec: Gemma4CachedAttentionSpec,
    scores: &mut Vec<f32>,
) -> Result<B::Tensor, B::Error> {
    let seq_len = backend.shape(q)[0];
    let q_width = spec.n_heads * spec.head_dim;
    let n_repeat = spec.n_heads / spec.n_kv_heads;
    let q_data = backend.data(q);
    let mut out = vec![0.0; seq_len * q_width];
    let cache_head_stride = spec.max_seq_len * spec.cache_head_dim;
    crate::backend::cached_attention_dispatch(
        q_data,
        cached_k,
        cached_v,
        &crate::backend::CachedAttentionSpec {
            n_heads: spec.n_heads,
            n_kv_heads: spec.n_kv_heads,
            head_dim: spec.head_dim,
            max_seq_len: spec.max_seq_len,
            total_seq_len: spec.total_seq_len,
        },
        seq_len,
        q_width,
        n_repeat,
        spec.scale,
        cache_head_stride,
        spec.cache_head_dim,
        spec.sliding_window,
        false,
        scores,
        &mut out,
    );
    backend.load_from_cpu(out, &[seq_len, q_width])
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot_product(a, b)
}

#[cfg(test)]
mod tests;
