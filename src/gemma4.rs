use crate::backend::{Backend, CpuBackend};
use crate::kv_cache::KVCache;
use crate::loader::{GgufLoader, GgufValue, LoadedTensor};
use crate::model::{pool_layer_activation, ForwardModel, Linear};
use crate::tensor::{compute_rope_freqs, CpuTensor};
use alloc::vec::Vec;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4AttentionType {
    Local,
    Global,
}

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_local_kv_heads: usize,
    pub n_global_kv_heads: usize,
    pub embed_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub local_head_dim: usize,
    pub global_head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub local_rope_theta: f32,
    pub global_rope_theta: f32,
    pub norm_eps: f32,
    pub attention_scale: f32,
    pub sliding_window: usize,
    pub layer_types: Vec<Gemma4AttentionType>,
    pub final_logit_softcap: Option<f32>,
    pub vocab_size_per_layer_input: Option<usize>,
    pub hidden_size_per_layer_input: Option<usize>,
    pub num_kv_shared_layers: usize,
}

impl Gemma4Config {
    pub fn from_gguf_metadata(loader: &GgufLoader) -> anyhow::Result<Self> {
        if get_bool(loader, "gemma4.enable_moe_block", false)
            || get_bool(loader, "gemma4.feed_forward.expert_count", false)
            || get_bool(loader, "gemma4.expert_used_count", false)
        {
            anyhow::bail!("MoE Gemma 4 models are not supported in v1");
        }

        let n_layers =
            get_u32_any(loader, &["gemma4.block_count", "gemma3.block_count"], 32)? as usize;
        let n_heads = get_u32_any(
            loader,
            &["gemma4.attention.head_count", "gemma3.attention.head_count"],
            32,
        )? as usize;
        let embed_dim = get_u32_any(
            loader,
            &["gemma4.embedding_length", "gemma3.embedding_length"],
            4096,
        )? as usize;
        let intermediate_dim = get_u32_or_first_array_any(
            loader,
            &["gemma4.feed_forward_length", "gemma3.feed_forward_length"],
            embed_dim as u32 * 4,
        )? as usize;
        let vocab_size =
            get_u32_any(loader, &["gemma4.vocab_size", "gemma3.vocab_size"], 256000)? as usize;
        let n_local_kv_heads = get_u32_any(
            loader,
            &[
                "gemma4.attention.head_count_kv",
                "gemma4.attention.local_head_count_kv",
                "gemma3.attention.head_count_kv",
            ],
            n_heads as u32,
        )? as usize;
        let n_global_kv_heads = get_u32_any(
            loader,
            &[
                "gemma4.attention.global_head_count_kv",
                "gemma4.attention.head_count_kv",
                "gemma3.attention.head_count_kv",
            ],
            n_local_kv_heads as u32,
        )? as usize;
        let default_head_dim = embed_dim / n_heads;
        let local_head_dim = get_u32_any(
            loader,
            &[
                "gemma4.attention.key_length_swa",
                "gemma4.attention.key_length",
                "gemma4.attention.local_key_length",
                "gemma3.attention.key_length",
            ],
            default_head_dim as u32,
        )? as usize;
        let global_head_dim = get_u32_any(
            loader,
            &[
                "gemma4.attention.global_key_length",
                "gemma4.attention.key_length",
                "gemma3.attention.key_length",
            ],
            local_head_dim as u32,
        )? as usize;
        let max_seq_len = get_u32_any(
            loader,
            &["gemma4.context_length", "gemma3.context_length"],
            4096,
        )? as usize;
        let rope_theta = get_f32_any(
            loader,
            &["gemma4.rope.freq_base", "gemma3.rope.freq_base"],
            1_000_000.0,
        )?;
        let local_rope_theta = get_f32_any(
            loader,
            &[
                "gemma4.rope.freq_base_swa",
                "gemma4.rope.local_freq_base",
                "gemma4.rope.freq_base",
            ],
            rope_theta,
        )?;
        let global_rope_theta = get_f32_any(
            loader,
            &["gemma4.rope.global_freq_base", "gemma4.rope.freq_base"],
            rope_theta,
        )?;
        let norm_eps = get_f32_any(
            loader,
            &[
                "gemma4.attention.layer_norm_rms_epsilon",
                "gemma4.layer_norm_rms_epsilon",
                "gemma3.attention.layer_norm_rms_epsilon",
            ],
            1e-6,
        )?;
        let query_pre_attn_scalar = get_f32_any(
            loader,
            &[
                "gemma4.attention.query_pre_attn_scalar",
                "gemma4.query_pre_attn_scalar",
                "gemma3.attention.query_pre_attn_scalar",
                "gemma3.query_pre_attn_scalar",
            ],
            local_head_dim as f32,
        )?;
        let attention_scale = query_pre_attn_scalar.sqrt().recip();
        let sliding_window = get_u32_any(
            loader,
            &[
                "gemma4.attention.sliding_window",
                "gemma3.attention.sliding_window",
            ],
            1024,
        )? as usize;
        let layer_types = parse_layer_types(loader, n_layers);
        let final_logit_softcap = get_optional_f32(
            loader,
            &[
                "gemma4.final_logit_softcap",
                "gemma4.final_logit_softcapping",
                "gemma4.attention.final_logit_softcap",
                "gemma3.final_logit_softcap",
            ],
        );
        let vocab_size_per_layer_input = get_optional_u32(
            loader,
            &[
                "gemma4.vocab_size_per_layer_input",
                "gemma3.vocab_size_per_layer_input",
            ],
        )
        .map(|v| v as usize);
        let hidden_size_per_layer_input = get_optional_u32(
            loader,
            &[
                "gemma4.hidden_size_per_layer_input",
                "gemma4.embedding_length_per_layer_input",
                "gemma3.hidden_size_per_layer_input",
            ],
        )
        .map(|v| v as usize);
        let num_kv_shared_layers = get_u32_any(
            loader,
            &[
                "gemma4.attention.shared_kv_layers",
                "gemma4.num_kv_shared_layers",
            ],
            0,
        )? as usize;

        Ok(Self {
            n_layers,
            n_heads,
            n_local_kv_heads,
            n_global_kv_heads,
            embed_dim,
            intermediate_dim,
            vocab_size,
            local_head_dim,
            global_head_dim,
            max_seq_len,
            rope_theta,
            local_rope_theta,
            global_rope_theta,
            norm_eps,
            attention_scale,
            sliding_window,
            layer_types,
            final_logit_softcap,
            vocab_size_per_layer_input,
            hidden_size_per_layer_input,
            num_kv_shared_layers,
        })
    }

    fn layer_type(&self, layer: usize) -> Gemma4AttentionType {
        self.layer_types[layer]
    }

    fn kv_heads_for(&self, layer_type: Gemma4AttentionType) -> usize {
        match layer_type {
            Gemma4AttentionType::Local => self.n_local_kv_heads,
            Gemma4AttentionType::Global => self.n_global_kv_heads,
        }
    }

    fn head_dim_for(&self, layer_type: Gemma4AttentionType) -> usize {
        match layer_type {
            Gemma4AttentionType::Local => self.local_head_dim,
            Gemma4AttentionType::Global => self.global_head_dim,
        }
    }
}

struct Gemma4Mlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> Gemma4Mlp<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let gate = self.gate_proj.forward(backend, x)?;
        let gate = gelu_tanh(backend, &gate)?;
        let up = self.up_proj.forward(backend, x)?;
        let gated = backend.elemul(&gate, &up)?;
        self.down_proj.forward(backend, &gated)
    }
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

impl<B: Backend> Gemma4Attention<B> {
    #[allow(clippy::too_many_arguments)]
    fn forward_with_cache(
        &self,
        backend: &B,
        x: &B::Tensor,
        cache: &mut KVCache,
        layer: usize,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let seq_len = backend.shape(x)[0];
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut q = self.q_proj.forward(backend, x)?;
        q = apply_rope_and_qk_norm(
            backend,
            &q,
            &self.q_norm,
            &self.rope_cos,
            &self.rope_sin,
            start_pos,
            self.n_heads,
            self.head_dim,
            self.norm_eps,
        )?;

        let source_layer = if let Some(source_layer) = self.shared_source_layer {
            source_layer
        } else {
            let k = self
                .k_proj
                .as_ref()
                .expect("non-shared Gemma 4 layer must have k_proj")
                .forward(backend, x)?;
            let k = apply_rope_and_qk_norm(
                backend,
                &k,
                &self.k_norm,
                &self.rope_cos,
                &self.rope_sin,
                start_pos,
                self.n_kv_heads,
                self.head_dim,
                self.norm_eps,
            )?;
            let v = self
                .v_proj
                .as_ref()
                .expect("non-shared Gemma 4 layer must have v_proj")
                .forward(backend, x)?;
            let k_data = backend.data(&k);
            let v_data = backend.data(&v);
            let cursor = cache.cursor();
            for pos in 0..seq_len {
                let offset = pos * kv_dim;
                cache.append_with_head_dim(
                    layer,
                    cursor + pos,
                    &k_data[offset..offset + kv_dim],
                    &v_data[offset..offset + kv_dim],
                    self.head_dim,
                );
            }
            layer
        };

        let total_seq_len = cache.cursor() + seq_len;
        let cache_head_dim = cache.head_dim();
        let max_seq_len = cache.max_seq_len();
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
        debug_assert_eq!(backend.shape(&out), &[seq_len, q_dim]);
        self.o_proj.forward(backend, &out)
    }

    #[allow(dead_code)]
    fn forward_full(
        &self,
        backend: &B,
        x: &B::Tensor,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let seq_len = backend.shape(x)[0];
        let q = self.q_proj.forward(backend, x)?;
        let q = apply_rope_and_qk_norm(
            backend,
            &q,
            &self.q_norm,
            &self.rope_cos,
            &self.rope_sin,
            start_pos,
            self.n_heads,
            self.head_dim,
            self.norm_eps,
        )?;
        let k = self
            .k_proj
            .as_ref()
            .expect("activation capture does not support shared-only Gemma 4 K/V")
            .forward(backend, x)?;
        let k = apply_rope_and_qk_norm(
            backend,
            &k,
            &self.k_norm,
            &self.rope_cos,
            &self.rope_sin,
            start_pos,
            self.n_kv_heads,
            self.head_dim,
            self.norm_eps,
        )?;
        let v = self
            .v_proj
            .as_ref()
            .expect("activation capture does not support shared-only Gemma 4 K/V")
            .forward(backend, x)?;
        let out = full_attention(
            backend,
            &q,
            &k,
            &v,
            Gemma4FullAttentionSpec {
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
                head_dim: self.head_dim,
                sliding_window: if self.layer_type == Gemma4AttentionType::Local {
                    Some(self.sliding_window)
                } else {
                    None
                },
                scale: self.attention_scale,
            },
        )?;
        debug_assert_eq!(
            backend.shape(&out),
            &[seq_len, self.n_heads * self.head_dim]
        );
        self.o_proj.forward(backend, &out)
    }
}

struct Gemma4Block<B: Backend> {
    input_norm: B::Tensor,
    attn: Gemma4Attention<B>,
    ple_proj: Option<Linear<B>>,
    post_attn_norm: B::Tensor,
    pre_ffn_norm: B::Tensor,
    mlp: Gemma4Mlp<B>,
    post_ffn_norm: B::Tensor,
    norm_eps: f32,
}

impl<B: Backend> Gemma4Block<B> {
    fn forward_with_cache(
        &self,
        backend: &B,
        x: &B::Tensor,
        ple: Option<&B::Tensor>,
        cache: &mut KVCache,
        layer: usize,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let x = self.add_ple(backend, x, ple)?;
        let normed = backend.rms_norm(&x, &self.input_norm, self.norm_eps)?;
        let attn_out = self
            .attn
            .forward_with_cache(backend, &normed, cache, layer, start_pos)?;
        let attn_out = backend.rms_norm(&attn_out, &self.post_attn_norm, self.norm_eps)?;
        let x = backend.add(&x, &attn_out)?;
        let normed = backend.rms_norm(&x, &self.pre_ffn_norm, self.norm_eps)?;
        let mlp_out = self.mlp.forward(backend, &normed)?;
        let mlp_out = backend.rms_norm(&mlp_out, &self.post_ffn_norm, self.norm_eps)?;
        backend.add(&x, &mlp_out)
    }

    #[allow(dead_code)]
    fn forward_full(
        &self,
        backend: &B,
        x: &B::Tensor,
        ple: Option<&B::Tensor>,
    ) -> Result<B::Tensor, B::Error> {
        let x = self.add_ple(backend, x, ple)?;
        let normed = backend.rms_norm(&x, &self.input_norm, self.norm_eps)?;
        let attn_out = self.attn.forward_full(backend, &normed, 0)?;
        let attn_out = backend.rms_norm(&attn_out, &self.post_attn_norm, self.norm_eps)?;
        let x = backend.add(&x, &attn_out)?;
        let normed = backend.rms_norm(&x, &self.pre_ffn_norm, self.norm_eps)?;
        let mlp_out = self.mlp.forward(backend, &normed)?;
        let mlp_out = backend.rms_norm(&mlp_out, &self.post_ffn_norm, self.norm_eps)?;
        backend.add(&x, &mlp_out)
    }

    fn add_ple(
        &self,
        backend: &B,
        x: &B::Tensor,
        ple: Option<&B::Tensor>,
    ) -> Result<B::Tensor, B::Error> {
        let Some(ple) = ple else {
            return Ok(x.clone());
        };
        if backend.shape(ple)[1] == backend.shape(x)[1] {
            return backend.add(x, ple);
        }
        let projected = self
            .ple_proj
            .as_ref()
            .expect("Gemma 4 packed PLE requires per-layer projection")
            .forward(backend, ple)?;
        backend.add(x, &projected)
    }
}

enum Gemma4Ple<B: Backend> {
    Hidden(B::Tensor),
    PackedQ8 {
        embeddings: crate::quant::QuantizedWeight,
        per_layer_dim: usize,
    },
}

enum Gemma4Head<B: Backend> {
    Linear(Linear<B>),
    TiedEmbedding(Arc<B::Tensor>),
}

impl<B: Backend> Gemma4Head<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        match self {
            Self::Linear(head) => head.forward(backend, x),
            Self::TiedEmbedding(table) => tied_embedding_logits(backend, x, table),
        }
    }
}

pub struct Gemma4<B: Backend> {
    embed_tokens: Arc<B::Tensor>,
    blocks: Vec<Gemma4Block<B>>,
    norm: B::Tensor,
    head: Gemma4Head<B>,
    ple: Option<Gemma4Ple<B>>,
    config: Gemma4Config,
}

impl<B: Backend> ForwardModel<B> for Gemma4<B> {
    fn create_cache(&self, _backend: &B, max_seq_len: usize) -> KVCache {
        let max_kv_heads = self
            .config
            .n_local_kv_heads
            .max(self.config.n_global_kv_heads);
        let max_head_dim = self.config.local_head_dim.max(self.config.global_head_dim);
        KVCache::new(self.blocks.len(), max_kv_heads, max_head_dim, max_seq_len)
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
}

impl Gemma4<CpuBackend> {
    pub fn from_loader(loader: GgufLoader) -> anyhow::Result<Self> {
        let config = Gemma4Config::from_gguf_metadata(&loader)?;
        log::debug!("gemma4 config: {:?}", config);

        let get_f32 = |name: &str| -> anyhow::Result<CpuTensor> {
            match loader.tensors.get(name) {
                Some(LoadedTensor::F32(t)) => Ok(t.clone()),
                Some(LoadedTensor::Q8_0(qw)) => Ok(qw.dequantize_all()),
                None => anyhow::bail!("Missing tensor: {}", name),
            }
        };
        let get_optional_f32 = |names: &[String]| -> Option<CpuTensor> {
            names
                .iter()
                .find_map(|name| match loader.tensors.get(name) {
                    Some(LoadedTensor::F32(t)) => Some(t.clone()),
                    Some(LoadedTensor::Q8_0(qw)) => Some(qw.dequantize_all()),
                    None => None,
                })
        };
        let get_linear = |name: &str| -> anyhow::Result<Linear<CpuBackend>> {
            match loader.tensors.get(name) {
                Some(LoadedTensor::F32(t)) => Ok(Linear::new(t.clone().transpose(), None)),
                Some(LoadedTensor::Q8_0(qw)) => Ok(Linear::new_q8_0(qw.clone(), None)),
                None => anyhow::bail!("Missing tensor: {}", name),
            }
        };

        let embed_tokens = Arc::new(get_f32("token_embd.weight")?);
        let local_rope = {
            let (cos, sin) = compute_rope_freqs(
                config.max_seq_len,
                config.local_head_dim,
                config.local_rope_theta,
            );
            (Arc::new(cos), Arc::new(sin))
        };
        let global_rope = {
            let (cos, sin) = compute_rope_freqs(
                config.max_seq_len,
                config.global_head_dim,
                config.global_rope_theta,
            );
            (Arc::new(cos), Arc::new(sin))
        };

        let mut blocks = Vec::with_capacity(config.n_layers);
        let mut last_local_source = None;
        let mut last_global_source = None;
        for i in 0..config.n_layers {
            let layer_type = config.layer_type(i);
            let k_name = format!("blk.{}.attn_k.weight", i);
            let v_name = format!("blk.{}.attn_v.weight", i);
            let has_own_kv =
                loader.tensors.contains_key(&k_name) && loader.tensors.contains_key(&v_name);
            let is_shared = !has_own_kv && i + config.num_kv_shared_layers >= config.n_layers;
            let shared_source_layer = if is_shared {
                match layer_type {
                    Gemma4AttentionType::Local => last_local_source,
                    Gemma4AttentionType::Global => last_global_source,
                }
            } else {
                None
            };
            if is_shared && shared_source_layer.is_none() {
                anyhow::bail!(
                    "Gemma 4 shared-KV layer {} has no previous {:?} source layer",
                    i,
                    layer_type
                );
            }

            if !is_shared {
                match layer_type {
                    Gemma4AttentionType::Local => last_local_source = Some(i),
                    Gemma4AttentionType::Global => last_global_source = Some(i),
                }
            }

            let (rope_cos, rope_sin) = match layer_type {
                Gemma4AttentionType::Local => {
                    (Arc::clone(&local_rope.0), Arc::clone(&local_rope.1))
                }
                Gemma4AttentionType::Global => {
                    (Arc::clone(&global_rope.0), Arc::clone(&global_rope.1))
                }
            };
            let n_kv_heads = config.kv_heads_for(layer_type);
            let head_dim = config.head_dim_for(layer_type);

            let attn = Gemma4Attention {
                q_proj: get_linear(&format!("blk.{}.attn_q.weight", i))?,
                k_proj: if shared_source_layer.is_some() {
                    None
                } else {
                    Some(get_linear(&format!("blk.{}.attn_k.weight", i))?)
                },
                v_proj: if shared_source_layer.is_some() {
                    None
                } else {
                    Some(get_linear(&format!("blk.{}.attn_v.weight", i))?)
                },
                o_proj: get_linear(&format!("blk.{}.attn_output.weight", i))?,
                q_norm: get_f32(&format!("blk.{}.attn_q_norm.weight", i))?,
                k_norm: get_f32(&format!("blk.{}.attn_k_norm.weight", i))?,
                rope_cos,
                rope_sin,
                layer_type,
                n_heads: config.n_heads,
                n_kv_heads,
                head_dim,
                sliding_window: config.sliding_window,
                norm_eps: config.norm_eps,
                attention_scale: config.attention_scale,
                shared_source_layer,
            };

            let post_attn_norm = get_optional_f32(&[
                format!("blk.{}.attn_post_norm.weight", i),
                format!("blk.{}.post_attention_norm.weight", i),
            ])
            .ok_or_else(|| anyhow::anyhow!("Missing tensor: blk.{}.attn_post_norm.weight", i))?;
            let post_ffn_norm = get_optional_f32(&[
                format!("blk.{}.ffn_post_norm.weight", i),
                format!("blk.{}.post_ffw_norm.weight", i),
            ])
            .ok_or_else(|| anyhow::anyhow!("Missing tensor: blk.{}.ffn_post_norm.weight", i))?;

            blocks.push(Gemma4Block {
                input_norm: get_f32(&format!("blk.{}.attn_norm.weight", i))?,
                attn,
                ple_proj: match loader.tensors.get(&format!("blk.{}.proj.weight", i)) {
                    Some(_) => Some(get_linear(&format!("blk.{}.proj.weight", i))?),
                    None => None,
                },
                post_attn_norm,
                pre_ffn_norm: get_f32(&format!("blk.{}.ffn_norm.weight", i))?,
                mlp: Gemma4Mlp {
                    gate_proj: get_linear(&format!("blk.{}.ffn_gate.weight", i))?,
                    up_proj: get_linear(&format!("blk.{}.ffn_up.weight", i))?,
                    down_proj: get_linear(&format!("blk.{}.ffn_down.weight", i))?,
                },
                post_ffn_norm,
                norm_eps: config.norm_eps,
            });
        }

        let ple_names = [
            "per_layer_token_embd.weight".to_string(),
            "token_embd_per_layer.weight".to_string(),
            "per_layer_embd.weight".to_string(),
        ];
        let per_layer_dim = config
            .hidden_size_per_layer_input
            .unwrap_or(config.embed_dim);
        let ple = match get_optional_f32_only(&loader, &ple_names) {
            Some(t) => {
                if t.shape().len() != 3
                    || t.shape()[0] != config.n_layers
                    || (t.shape()[2] != config.embed_dim && t.shape()[2] != per_layer_dim)
                {
                    anyhow::bail!(
                        "Gemma 4 PLE tensor must have shape [layers, vocab, {} or hidden={}], got {:?}",
                        per_layer_dim,
                        config.embed_dim,
                        t.shape()
                    );
                }
                Some(Gemma4Ple::Hidden(t))
            }
            None => ple_names
                .iter()
                .find_map(|name| match loader.tensors.get(name) {
                    Some(LoadedTensor::Q8_0(qw)) => Some(Gemma4Ple::PackedQ8 {
                        embeddings: qw.clone(),
                        per_layer_dim,
                    }),
                    _ => None,
                }),
        };
        if ple.is_none()
            && (config.vocab_size_per_layer_input.is_some()
                || config.hidden_size_per_layer_input.is_some())
        {
            log::warn!(
                "Gemma 4 PLE metadata is present but no supported packed PLE tensor was found"
            );
        }

        let head = match loader.tensors.get("output.weight") {
            Some(LoadedTensor::F32(t)) => {
                Gemma4Head::Linear(Linear::new(t.clone().transpose(), None))
            }
            Some(LoadedTensor::Q8_0(qw)) => Gemma4Head::Linear(Linear::new_q8_0(qw.clone(), None)),
            None => Gemma4Head::TiedEmbedding(Arc::clone(&embed_tokens)),
        };

        Ok(Self {
            embed_tokens,
            blocks,
            norm: get_f32("output_norm.weight")?,
            head,
            ple,
            config,
        })
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
        let mut x = embed_tokens(
            backend,
            &self.embed_tokens,
            token_ids,
            self.config.embed_dim,
        )?;
        let ple = self.ple_vectors(backend, token_ids)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            let layer_ple = ple.as_ref().map(|v| &v[layer]);
            x = block.forward_with_cache(backend, &x, layer_ple, cache, layer, start_pos)?;
        }
        for _ in 0..token_ids.len() {
            cache.advance_cursor();
        }
        let x = backend.rms_norm(&x, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &x)?;
        softcap_logits(backend, &logits, self.config.final_logit_softcap)
    }

    pub fn forward_last_logits_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let mut x = embed_tokens(
            backend,
            &self.embed_tokens,
            token_ids,
            self.config.embed_dim,
        )?;
        let ple = self.ple_vectors(backend, token_ids)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            let layer_ple = ple.as_ref().map(|v| &v[layer]);
            x = block.forward_with_cache(backend, &x, layer_ple, cache, layer, start_pos)?;
        }
        for _ in 0..token_ids.len() {
            cache.advance_cursor();
        }

        let last = backend.row_as_2d(&x, token_ids.len() - 1)?;
        let last = backend.rms_norm(&last, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &last)?;
        softcap_logits(backend, &logits, self.config.final_logit_softcap)
    }

    #[allow(clippy::type_complexity)]
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
        let ple = self.ple_vectors(backend, token_ids)?;
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
            softcap_logits(backend, &logits, self.config.final_logit_softcap)?,
        ))
    }

    #[allow(clippy::type_complexity)]
    pub fn forward_pooled_activations(
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
        let ple = self.ple_vectors(backend, token_ids)?;
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
        let x = backend.rms_norm(&x, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &x)?;
        Ok((
            pooled,
            softcap_logits(backend, &logits, self.config.final_logit_softcap)?,
        ))
    }

    fn ple_vectors(
        &self,
        backend: &B,
        token_ids: &[u32],
    ) -> Result<Option<Vec<B::Tensor>>, B::Error> {
        let Some(ref ple) = self.ple else {
            return Ok(None);
        };
        match ple {
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
                        data[dst..dst + dim].copy_from_slice(&ple_data[src..src + dim]);
                    }
                    out.push(backend.load_from_cpu(data, &[token_ids.len(), dim])?);
                }
                Ok(Some(out))
            }
            Gemma4Ple::PackedQ8 {
                embeddings,
                per_layer_dim,
            } => {
                let mut out = Vec::with_capacity(self.config.n_layers);
                let packed_dim = embeddings.in_features();
                let expected = self.config.n_layers * *per_layer_dim;
                debug_assert_eq!(packed_dim, expected);
                for layer in 0..self.config.n_layers {
                    let mut data = vec![0.0; token_ids.len() * *per_layer_dim];
                    let layer_start = layer * *per_layer_dim;
                    let mut row = vec![0.0; packed_dim];
                    for (pos, &tok) in token_ids.iter().enumerate() {
                        let token = (tok as usize).min(embeddings.out_features() - 1);
                        embeddings.dequantize_row(token, &mut row);
                        let dst = pos * *per_layer_dim;
                        data[dst..dst + *per_layer_dim]
                            .copy_from_slice(&row[layer_start..layer_start + *per_layer_dim]);
                    }
                    out.push(backend.load_from_cpu(data, &[token_ids.len(), *per_layer_dim])?);
                }
                Ok(Some(out))
            }
        }
    }
}

fn embed_tokens<B: Backend>(
    backend: &B,
    table: &B::Tensor,
    token_ids: &[u32],
    embed_dim: usize,
) -> Result<B::Tensor, B::Error> {
    let mut x = backend.zeroes(&[token_ids.len(), embed_dim])?;
    for (i, &tok) in token_ids.iter().enumerate() {
        let word_vec = backend.index_select(table, tok as usize)?;
        backend.assign_row(&mut x, i, &word_vec);
    }
    Ok(x)
}

#[allow(dead_code)]
fn add_optional<B: Backend>(
    backend: &B,
    x: &B::Tensor,
    residual: Option<&B::Tensor>,
) -> Result<B::Tensor, B::Error> {
    match residual {
        Some(r) => backend.add(x, r),
        None => Ok(x.clone()),
    }
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
    logits: &B::Tensor,
    softcap: Option<f32>,
) -> Result<B::Tensor, B::Error> {
    let Some(cap) = softcap else {
        return Ok(logits.clone());
    };
    let data = backend
        .data(logits)
        .iter()
        .map(|&v| (v / cap).tanh() * cap)
        .collect();
    backend.load_from_cpu(data, backend.shape(logits))
}

#[allow(clippy::too_many_arguments)]
fn apply_rope_and_qk_norm<B: Backend>(
    backend: &B,
    x: &B::Tensor,
    norm: &B::Tensor,
    rope_cos: &B::Tensor,
    rope_sin: &B::Tensor,
    start_pos: usize,
    n_heads: usize,
    head_dim: usize,
    norm_eps: f32,
) -> Result<B::Tensor, B::Error> {
    let seq_len = backend.shape(x)[0];
    let width = n_heads * head_dim;
    let half = head_dim / 2;
    let mut data = backend.data(x).to_vec();
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

#[allow(dead_code)]
struct Gemma4FullAttentionSpec {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    scale: f32,
}

#[allow(dead_code)]
fn full_attention<B: Backend>(
    backend: &B,
    q: &B::Tensor,
    k: &B::Tensor,
    v: &B::Tensor,
    spec: Gemma4FullAttentionSpec,
) -> Result<B::Tensor, B::Error> {
    let seq_len = backend.shape(q)[0];
    let q_width = spec.n_heads * spec.head_dim;
    let kv_width = spec.n_kv_heads * spec.head_dim;
    let n_repeat = spec.n_heads / spec.n_kv_heads;
    let q_data = backend.data(q);
    let k_data = backend.data(k);
    let v_data = backend.data(v);
    let mut out = vec![0.0; seq_len * q_width];
    let mut scores = vec![f32::NEG_INFINITY; seq_len];

    for h in 0..spec.n_heads {
        let q_head = h * spec.head_dim;
        let kv_head = (h / n_repeat) * spec.head_dim;
        for i in 0..seq_len {
            scores.fill(f32::NEG_INFINITY);
            let min_j = spec
                .sliding_window
                .map(|w| (i + 1).saturating_sub(w))
                .unwrap_or(0);
            for (j, score) in scores.iter_mut().enumerate().take(i + 1).skip(min_j) {
                let q_idx = i * q_width + q_head;
                let k_idx = j * kv_width + kv_head;
                *score = dot(
                    &q_data[q_idx..q_idx + spec.head_dim],
                    &k_data[k_idx..k_idx + spec.head_dim],
                ) * spec.scale;
            }
            softmax_range(&mut scores, min_j, i + 1);
            let out_idx = i * q_width + q_head;
            for (j, &weight) in scores.iter().enumerate().take(i + 1).skip(min_j) {
                if weight == 0.0 {
                    continue;
                }
                let v_idx = j * kv_width + kv_head;
                crate::simd::weighted_add(
                    &mut out[out_idx..out_idx + spec.head_dim],
                    &v_data[v_idx..v_idx + spec.head_dim],
                    weight,
                );
            }
        }
    }
    backend.load_from_cpu(out, &[seq_len, q_width])
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
    cached_k: &[f32],
    cached_v: &[f32],
    spec: Gemma4CachedAttentionSpec,
    scores: &mut Vec<f32>,
) -> Result<B::Tensor, B::Error> {
    let seq_len = backend.shape(q)[0];
    let q_width = spec.n_heads * spec.head_dim;
    let n_repeat = spec.n_heads / spec.n_kv_heads;
    let q_data = backend.data(q);
    let mut out = vec![0.0; seq_len * q_width];
    if scores.capacity() < spec.max_seq_len {
        scores.reserve(spec.max_seq_len - scores.capacity());
    }
    let cache_head_stride = spec.max_seq_len * spec.cache_head_dim;

    for h in 0..spec.n_heads {
        let q_head = h * spec.head_dim;
        let kv_h = h / n_repeat;
        for i in 0..seq_len {
            let max_j = spec.total_seq_len - seq_len + i;
            scores.fill(f32::NEG_INFINITY);
            scores.resize(max_j + 1, f32::NEG_INFINITY);
            let min_j = spec
                .sliding_window
                .map(|w| (max_j + 1).saturating_sub(w))
                .unwrap_or(0);
            let q_idx = i * q_width + q_head;
            for (j, score) in scores.iter_mut().enumerate().take(max_j + 1).skip(min_j) {
                let k_idx = kv_h * cache_head_stride + j * spec.cache_head_dim;
                *score = dot(
                    &q_data[q_idx..q_idx + spec.head_dim],
                    &cached_k[k_idx..k_idx + spec.head_dim],
                ) * spec.scale;
            }
            softmax_range(scores.as_mut_slice(), min_j, max_j + 1);
            let out_idx = i * q_width + q_head;
            for (j, &weight) in scores.iter().enumerate().take(max_j + 1).skip(min_j) {
                if weight == 0.0 {
                    continue;
                }
                let v_idx = kv_h * cache_head_stride + j * spec.cache_head_dim;
                crate::simd::weighted_add(
                    &mut out[out_idx..out_idx + spec.head_dim],
                    &cached_v[v_idx..v_idx + spec.head_dim],
                    weight,
                );
            }
        }
    }
    backend.load_from_cpu(out, &[seq_len, q_width])
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot_product(a, b)
}

fn softmax_range(row: &mut [f32], start: usize, end: usize) {
    let max = row[start..end]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in &mut row[start..end] {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in &mut row[start..end] {
        *v /= sum;
    }
}

fn parse_layer_types(loader: &GgufLoader, n_layers: usize) -> Vec<Gemma4AttentionType> {
    if let Some(GgufValue::Array(values)) = loader
        .metadata
        .get("gemma4.attention.sliding_window_pattern")
    {
        let parsed: Vec<_> = values
            .iter()
            .filter_map(|v| match v {
                GgufValue::Bool(true) => Some(Gemma4AttentionType::Local),
                GgufValue::Bool(false) => Some(Gemma4AttentionType::Global),
                _ => None,
            })
            .collect();
        if parsed.len() == n_layers {
            return parsed;
        }
    }

    for key in [
        "gemma4.attention.layer_types",
        "gemma4.layer_types",
        "gemma3.attention.layer_types",
    ] {
        if let Some(GgufValue::Array(values)) = loader.metadata.get(key) {
            let parsed: Vec<_> = values
                .iter()
                .filter_map(|v| match v {
                    GgufValue::Str(s) if s.contains("global") || s == "full_attention" => {
                        Some(Gemma4AttentionType::Global)
                    }
                    GgufValue::Str(_) => Some(Gemma4AttentionType::Local),
                    GgufValue::U32(1) | GgufValue::I32(1) => Some(Gemma4AttentionType::Global),
                    GgufValue::U32(_) | GgufValue::I32(_) => Some(Gemma4AttentionType::Local),
                    _ => None,
                })
                .collect();
            if parsed.len() == n_layers {
                return parsed;
            }
        }
    }
    (0..n_layers)
        .map(|i| {
            if (i + 1) % 6 == 0 {
                Gemma4AttentionType::Global
            } else {
                Gemma4AttentionType::Local
            }
        })
        .collect()
}

fn get_bool(loader: &GgufLoader, key: &str, default: bool) -> bool {
    match loader.metadata.get(key) {
        Some(GgufValue::Bool(v)) => *v,
        Some(GgufValue::U32(v)) => *v != 0,
        Some(GgufValue::I32(v)) => *v != 0,
        _ => default,
    }
}

fn get_u32_any(loader: &GgufLoader, keys: &[&str], default: u32) -> anyhow::Result<u32> {
    for key in keys {
        match loader.metadata.get(*key) {
            Some(GgufValue::U32(v)) => return Ok(*v),
            Some(GgufValue::U64(v)) => return Ok((*v).try_into()?),
            Some(GgufValue::I32(v)) if *v >= 0 => return Ok(*v as u32),
            _ => {}
        }
    }
    Ok(default)
}

fn get_u32_or_first_array_any(
    loader: &GgufLoader,
    keys: &[&str],
    default: u32,
) -> anyhow::Result<u32> {
    for key in keys {
        match loader.metadata.get(*key) {
            Some(GgufValue::U32(v)) => return Ok(*v),
            Some(GgufValue::U64(v)) => return Ok((*v).try_into()?),
            Some(GgufValue::I32(v)) if *v >= 0 => return Ok(*v as u32),
            Some(GgufValue::Array(values)) => {
                if let Some(v) = values.iter().find_map(|value| match value {
                    GgufValue::U32(v) => Some(*v),
                    GgufValue::U64(v) => (*v).try_into().ok(),
                    GgufValue::I32(v) if *v >= 0 => Some(*v as u32),
                    _ => None,
                }) {
                    return Ok(v);
                }
            }
            _ => {}
        }
    }
    Ok(default)
}

fn get_optional_u32(loader: &GgufLoader, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|key| match loader.metadata.get(*key) {
        Some(GgufValue::U32(v)) => Some(*v),
        Some(GgufValue::U64(v)) => (*v).try_into().ok(),
        Some(GgufValue::I32(v)) if *v >= 0 => Some(*v as u32),
        _ => None,
    })
}

fn get_f32_any(loader: &GgufLoader, keys: &[&str], default: f32) -> anyhow::Result<f32> {
    for key in keys {
        if let Some(GgufValue::F32(v)) = loader.metadata.get(*key) {
            return Ok(*v);
        }
    }
    Ok(default)
}

fn get_optional_f32(loader: &GgufLoader, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| match loader.metadata.get(*key) {
        Some(GgufValue::F32(v)) => Some(*v),
        _ => None,
    })
}

fn get_optional_f32_only(loader: &GgufLoader, names: &[String]) -> Option<CpuTensor> {
    names
        .iter()
        .find_map(|name| match loader.tensors.get(name) {
            Some(LoadedTensor::F32(t)) => Some(t.clone()),
            Some(LoadedTensor::Q8_0(_)) | None => None,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn loader_with(metadata: HashMap<String, GgufValue>) -> GgufLoader {
        GgufLoader {
            metadata,
            tensors: HashMap::new(),
        }
    }

    fn tiny_tensor(shape: &[usize], value: f32) -> LoadedTensor {
        LoadedTensor::F32(CpuTensor::from_data(
            shape.to_vec(),
            vec![value; shape.iter().product()],
        ))
    }

    fn tiny_weight(shape: &[usize]) -> LoadedTensor {
        let rows = shape[0];
        let cols = shape[1];
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows.min(cols) {
            data[i * cols + i] = 0.1;
        }
        LoadedTensor::F32(CpuTensor::from_data(shape.to_vec(), data))
    }

    fn insert_tiny_gemma4_tensors(tensors: &mut HashMap<String, LoadedTensor>) {
        tensors.insert(
            "token_embd.weight".to_string(),
            LoadedTensor::F32(CpuTensor::from_data(
                vec![4, 2],
                vec![0.1, 0.2, 0.2, 0.1, 0.0, 0.3, 0.3, 0.0],
            )),
        );
        tensors.insert("output_norm.weight".to_string(), tiny_tensor(&[2], 1.0));
        tensors.insert("output.weight".to_string(), tiny_weight(&[4, 2]));

        for name in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            tensors.insert(format!("blk.0.{}.weight", name), tiny_weight(&[2, 2]));
        }
        for name in [
            "attn_q_norm",
            "attn_k_norm",
            "attn_norm",
            "attn_post_norm",
            "ffn_norm",
            "ffn_post_norm",
        ] {
            tensors.insert(format!("blk.0.{}.weight", name), tiny_tensor(&[2], 1.0));
        }
    }

    #[test]
    fn gemma4_config_rejects_moe() {
        let mut metadata = HashMap::new();
        metadata.insert("gemma4.enable_moe_block".to_string(), GgufValue::Bool(true));
        let err = Gemma4Config::from_gguf_metadata(&loader_with(metadata)).unwrap_err();
        assert!(err.to_string().contains("MoE Gemma 4"));
    }

    #[test]
    fn gemma4_config_accepts_dense_text_defaults() {
        let mut metadata = HashMap::new();
        metadata.insert("gemma4.block_count".to_string(), GgufValue::U32(2));
        metadata.insert("gemma4.embedding_length".to_string(), GgufValue::U32(8));
        metadata.insert("gemma4.attention.head_count".to_string(), GgufValue::U32(2));
        metadata.insert("gemma4.vocab_size".to_string(), GgufValue::U32(16));
        let cfg = Gemma4Config::from_gguf_metadata(&loader_with(metadata)).unwrap();
        assert_eq!(cfg.n_layers, 2);
        assert_eq!(cfg.embed_dim, 8);
        assert_eq!(cfg.local_head_dim, 4);
    }

    #[test]
    fn softcap_transforms_logits() {
        let backend = CpuBackend;
        let logits = CpuTensor::from_data(vec![1, 3], vec![-100.0, 0.0, 100.0]);
        let capped = softcap_logits(&backend, &logits, Some(30.0)).unwrap();
        assert!(capped.data()[0] > -30.0);
        assert_eq!(capped.data()[1], 0.0);
        assert!(capped.data()[2] < 30.0);
    }

    #[test]
    fn sliding_softmax_limits_attention_range() {
        let backend = CpuBackend;
        let q = CpuTensor::from_data(vec![3, 1], vec![1.0, 1.0, 1.0]);
        let k = CpuTensor::from_data(vec![3, 1], vec![1.0, 1.0, 1.0]);
        let v = CpuTensor::from_data(vec![3, 1], vec![10.0, 20.0, 30.0]);
        let out = full_attention(
            &backend,
            &q,
            &k,
            &v,
            Gemma4FullAttentionSpec {
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 1,
                sliding_window: Some(2),
                scale: 1.0,
            },
        )
        .unwrap();
        assert!((out.data()[2] - 25.0).abs() < 1e-5);
    }

    #[test]
    fn attention_scale_uses_query_pre_attn_scalar() {
        let mut metadata = HashMap::new();
        metadata.insert("gemma4.attention.head_count".into(), GgufValue::U32(1));
        metadata.insert("gemma4.embedding_length".into(), GgufValue::U32(4));
        metadata.insert("gemma4.attention.key_length".into(), GgufValue::U32(4));
        metadata.insert(
            "gemma4.attention.query_pre_attn_scalar".into(),
            GgufValue::F32(16.0),
        );
        let cfg = Gemma4Config::from_gguf_metadata(&loader_with(metadata)).unwrap();
        assert!((cfg.attention_scale - 0.25).abs() < 1e-6);
    }

    #[test]
    fn qk_norm_happens_before_rope_and_uses_config_eps() {
        let backend = CpuBackend;
        let x = CpuTensor::from_data(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let norm = CpuTensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let rope_cos = CpuTensor::from_data(vec![1, 2], vec![0.0, 1.0]);
        let rope_sin = CpuTensor::from_data(vec![1, 2], vec![1.0, 0.0]);
        let out = apply_rope_and_qk_norm(&backend, &x, &norm, &rope_cos, &rope_sin, 0, 1, 4, 0.0)
            .unwrap();
        // RMSNorm first: rstd = 1 / sqrt(mean([1, 4, 9, 16])).
        // RoPE then rotates the first/second half pairs with cos=[0,1],
        // sin=[1,0].
        let rstd = (7.5_f32).sqrt().recip();
        let expected = [-9.0 * rstd, 4.0 * rstd, 1.0 * rstd, 16.0 * rstd];
        for (got, expected) in out.data().iter().zip(expected) {
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn kv_cache_strided_append_keeps_source_layer_values() {
        let mut cache = KVCache::new(2, 1, 4, 3);
        cache.append_with_head_dim(0, 0, &[1.0, 2.0], &[3.0, 4.0], 2);
        let (k, v) = cache.get(0);
        assert_eq!(&k[..4], &[1.0, 2.0, 0.0, 0.0]);
        assert_eq!(&v[..4], &[3.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn tiny_gemma4_loader_runs_forward_pass() {
        let mut metadata = HashMap::new();
        metadata.insert("gemma4.block_count".to_string(), GgufValue::U32(1));
        metadata.insert("gemma4.embedding_length".to_string(), GgufValue::U32(2));
        metadata.insert("gemma4.attention.head_count".to_string(), GgufValue::U32(1));
        metadata.insert(
            "gemma4.attention.head_count_kv".to_string(),
            GgufValue::U32(1),
        );
        metadata.insert("gemma4.attention.key_length".to_string(), GgufValue::U32(2));
        metadata.insert("gemma4.feed_forward_length".to_string(), GgufValue::U32(2));
        metadata.insert("gemma4.vocab_size".to_string(), GgufValue::U32(4));
        metadata.insert("gemma4.context_length".to_string(), GgufValue::U32(8));
        metadata.insert(
            "gemma4.attention.sliding_window".to_string(),
            GgufValue::U32(2),
        );
        metadata.insert(
            "gemma4.final_logit_softcap".to_string(),
            GgufValue::F32(10.0),
        );

        let mut tensors = HashMap::new();
        insert_tiny_gemma4_tensors(&mut tensors);
        let loader = GgufLoader { metadata, tensors };
        let model = Gemma4::from_loader(loader).unwrap();
        let backend = CpuBackend;
        let mut cache = model.create_cache(&backend, 4);
        let logits = model
            .forward_with_cache(&backend, &[0, 1], &mut cache, 0)
            .unwrap();
        assert_eq!(logits.shape(), &[2, 4]);
        assert!(logits.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ple_vectors_slice_by_layer_and_token() {
        let backend = CpuBackend;
        let model = Gemma4 {
            embed_tokens: Arc::new(CpuTensor::zeroes(&[3, 2])),
            blocks: Vec::new(),
            norm: CpuTensor::zeroes(&[2]),
            head: Gemma4Head::Linear(Linear::new(CpuTensor::zeroes(&[2, 3]), None)),
            ple: Some(Gemma4Ple::Hidden(CpuTensor::from_data(
                vec![2, 3, 2],
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                ],
            ))),
            config: Gemma4Config {
                n_layers: 2,
                n_heads: 1,
                n_local_kv_heads: 1,
                n_global_kv_heads: 1,
                embed_dim: 2,
                intermediate_dim: 2,
                vocab_size: 3,
                local_head_dim: 2,
                global_head_dim: 2,
                max_seq_len: 4,
                rope_theta: 10_000.0,
                local_rope_theta: 10_000.0,
                global_rope_theta: 10_000.0,
                norm_eps: 1e-6,
                attention_scale: 1.0,
                sliding_window: 2,
                layer_types: vec![Gemma4AttentionType::Local, Gemma4AttentionType::Local],
                final_logit_softcap: None,
                vocab_size_per_layer_input: Some(3),
                hidden_size_per_layer_input: Some(2),
                num_kv_shared_layers: 0,
            },
        };
        let ple = model.ple_vectors(&backend, &[2, 1]).unwrap().unwrap();
        assert_eq!(ple[0].data(), &[5.0, 6.0, 3.0, 4.0]);
        assert_eq!(ple[1].data(), &[50.0, 60.0, 30.0, 40.0]);
    }
}
