use super::*;

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
        // Validate this before parse_layer_types allocates a vector sized by
        // block_count. The full config validation below repeats the check.
        anyhow::ensure!(
            n_layers <= crate::loader::limits::MAX_LAYERS,
            "Gemma block_count {n_layers} exceeds the {} layer limit",
            crate::loader::limits::MAX_LAYERS
        );
        let n_heads = get_u32_any(
            loader,
            &["gemma4.attention.head_count", "gemma3.attention.head_count"],
            32,
        )? as usize;
        anyhow::ensure!(
            n_heads > 0 && n_heads <= crate::loader::limits::MAX_HEADS,
            "Gemma attention head_count {n_heads} must be in 1..={}",
            crate::loader::limits::MAX_HEADS
        );
        let embed_dim = get_u32_any(
            loader,
            &["gemma4.embedding_length", "gemma3.embedding_length"],
            4096,
        )? as usize;
        // Keep the fallback conversion checked: metadata can set
        // embedding_length to u32::MAX, and `* 4` must not wrap/panic before
        // config validation gets a chance to reject it.
        let default_intermediate_dim = embed_dim
            .checked_mul(4)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(u32::MAX);
        let intermediate_dim = get_u32_or_first_array_any(
            loader,
            &["gemma4.feed_forward_length", "gemma3.feed_forward_length"],
            default_intermediate_dim,
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
        anyhow::ensure!(
            max_seq_len <= crate::loader::limits::MAX_CONTEXT_LEN,
            "Gemma context_length {max_seq_len} exceeds the {} sanity cap",
            crate::loader::limits::MAX_CONTEXT_LEN
        );
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

        let is_gemma4 = loader.metadata.contains_key("gemma4.block_count");
        let attention_scale = if is_gemma4 {
            // llama.cpp gemma4: "Gemma4 uses self.scaling = 1.0 (no pre-attn
            // scaling)". The gemma3-style 1/sqrt(query_pre_attn_scalar)
            // scaling would sharpen every attention distribution 16x on the
            // E2B (which carries no query_pre_attn_scalar key).
            1.0
        } else {
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
            query_pre_attn_scalar.sqrt().recip()
        };
        let sliding_window = get_u32_any(
            loader,
            &[
                "gemma4.attention.sliding_window",
                "gemma3.attention.sliding_window",
            ],
            1024,
        )? as usize;
        let layer_types = parse_layer_types(loader, n_layers)?;
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

        let config = Self {
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
        };
        config.validate()?;
        Ok(config)
    }

    pub(super) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.n_layers > 0,
            "Gemma model must contain at least one layer"
        );
        anyhow::ensure!(
            self.embed_dim > 0,
            "Gemma embedding_length must be greater than zero"
        );
        anyhow::ensure!(
            self.intermediate_dim > 0,
            "Gemma feed_forward_length must be greater than zero"
        );
        anyhow::ensure!(
            self.vocab_size > 0,
            "Gemma vocab_size must be greater than zero"
        );
        anyhow::ensure!(
            self.max_seq_len > 0,
            "Gemma context_length must be greater than zero"
        );
        anyhow::ensure!(
            self.n_local_kv_heads > 0 && self.n_local_kv_heads <= self.n_heads,
            "Gemma local KV head count must be in 1..={}",
            self.n_heads
        );
        anyhow::ensure!(
            self.n_global_kv_heads > 0 && self.n_global_kv_heads <= self.n_heads,
            "Gemma global KV head count must be in 1..={}",
            self.n_heads
        );
        anyhow::ensure!(
            self.n_heads.is_multiple_of(self.n_local_kv_heads)
                && self.n_heads.is_multiple_of(self.n_global_kv_heads),
            "Gemma query head count must be divisible by both KV head counts"
        );
        anyhow::ensure!(
            self.local_head_dim >= 2 && self.local_head_dim.is_multiple_of(2),
            "Gemma local attention head dimension must be positive and even, got {}",
            self.local_head_dim
        );
        anyhow::ensure!(
            self.global_head_dim >= 2 && self.global_head_dim.is_multiple_of(2),
            "Gemma global attention head dimension must be positive and even, got {}",
            self.global_head_dim
        );
        anyhow::ensure!(
            self.sliding_window > 0,
            "Gemma sliding_window must be greater than zero"
        );
        // Metadata magnitudes are attacker-controlled. Check every dimension
        // before the loader allocates RoPE tables, blocks, or KV state.
        use crate::loader::limits as l;
        anyhow::ensure!(
            self.n_layers <= l::MAX_LAYERS,
            "Gemma block_count {} exceeds the {} layer limit",
            self.n_layers,
            l::MAX_LAYERS
        );
        anyhow::ensure!(
            self.n_heads <= l::MAX_HEADS,
            "Gemma attention head_count {} exceeds the {} head limit",
            self.n_heads,
            l::MAX_HEADS
        );
        anyhow::ensure!(
            self.embed_dim <= l::MAX_EMBED_DIM,
            "Gemma embedding_length {} exceeds the {} limit",
            self.embed_dim,
            l::MAX_EMBED_DIM
        );
        anyhow::ensure!(
            self.intermediate_dim <= l::MAX_INTERMEDIATE_DIM,
            "Gemma feed_forward_length {} exceeds the {} limit",
            self.intermediate_dim,
            l::MAX_INTERMEDIATE_DIM
        );
        anyhow::ensure!(
            self.vocab_size <= l::MAX_VOCAB_SIZE,
            "Gemma vocab_size {} exceeds the {} limit",
            self.vocab_size,
            l::MAX_VOCAB_SIZE
        );
        anyhow::ensure!(
            self.max_seq_len <= l::MAX_CONTEXT_LEN,
            "Gemma context_length {} exceeds the {} limit",
            self.max_seq_len,
            l::MAX_CONTEXT_LEN
        );
        anyhow::ensure!(
            self.local_head_dim <= l::MAX_HEAD_DIM && self.global_head_dim <= l::MAX_HEAD_DIM,
            "Gemma attention head dimensions {} and {} exceed the {} limit",
            self.local_head_dim,
            self.global_head_dim,
            l::MAX_HEAD_DIM
        );
        anyhow::ensure!(
            self.sliding_window <= l::MAX_SLIDING_WINDOW,
            "Gemma sliding_window {} exceeds the {} limit",
            self.sliding_window,
            l::MAX_SLIDING_WINDOW
        );
        anyhow::ensure!(
            self.num_kv_shared_layers <= self.n_layers,
            "Gemma shared KV layer count {} exceeds block_count {}",
            self.num_kv_shared_layers,
            self.n_layers
        );
        anyhow::ensure!(
            self.num_kv_shared_layers <= l::MAX_LAYERS,
            "Gemma shared KV layer count {} exceeds the {} layer limit",
            self.num_kv_shared_layers,
            l::MAX_LAYERS
        );
        if let Some(dim) = self.hidden_size_per_layer_input {
            anyhow::ensure!(
                dim <= l::MAX_EMBED_DIM,
                "Gemma hidden_size_per_layer_input {dim} exceeds the {} limit",
                l::MAX_EMBED_DIM
            );
        }
        if let Some(size) = self.vocab_size_per_layer_input {
            anyhow::ensure!(
                size <= l::MAX_VOCAB_SIZE,
                "Gemma vocab_size_per_layer_input {size} exceeds the {} limit",
                l::MAX_VOCAB_SIZE
            );
        }
        anyhow::ensure!(
            self.max_seq_len
                .checked_mul(self.local_head_dim.max(self.global_head_dim))
                .is_some_and(|elements| elements <= l::MAX_ROPE_TABLE_ELEMENTS),
            "Gemma context_length {} * head_dim exceeds the {} RoPE-table element limit",
            self.max_seq_len,
            l::MAX_ROPE_TABLE_ELEMENTS
        );
        for (name, value) in [
            ("rope.freq_base", self.rope_theta),
            ("local rope.freq_base", self.local_rope_theta),
            ("global rope.freq_base", self.global_rope_theta),
            ("attention scale", self.attention_scale),
        ] {
            anyhow::ensure!(
                value.is_finite() && value > 0.0,
                "Gemma {name} must be finite and positive"
            );
        }
        anyhow::ensure!(
            self.norm_eps.is_finite() && self.norm_eps >= 0.0,
            "Gemma RMS epsilon must be finite and non-negative"
        );
        if let Some(cap) = self.final_logit_softcap {
            anyhow::ensure!(
                cap.is_finite() && cap > 0.0,
                "Gemma logit softcap must be finite and positive"
            );
        }
        anyhow::ensure!(
            self.layer_types.len() == self.n_layers,
            "Gemma layer type count {} does not match block_count {}",
            self.layer_types.len(),
            self.n_layers
        );
        Ok(())
    }

    pub(super) fn layer_type(&self, layer: usize) -> Gemma4AttentionType {
        self.layer_types[layer]
    }

    pub(super) fn kv_heads_for(&self, layer_type: Gemma4AttentionType) -> usize {
        match layer_type {
            Gemma4AttentionType::Local => self.n_local_kv_heads,
            Gemma4AttentionType::Global => self.n_global_kv_heads,
        }
    }

    pub(super) fn head_dim_for(&self, layer_type: Gemma4AttentionType) -> usize {
        match layer_type {
            Gemma4AttentionType::Local => self.local_head_dim,
            Gemma4AttentionType::Global => self.global_head_dim,
        }
    }
}

pub(super) fn parse_layer_types(
    loader: &GgufLoader,
    n_layers: usize,
) -> anyhow::Result<Vec<Gemma4AttentionType>> {
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
            return Ok(parsed);
        }
        anyhow::bail!(
            "gemma4.attention.sliding_window_pattern has {} valid entries; expected {n_layers}",
            parsed.len()
        );
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
                return Ok(parsed);
            }
            anyhow::bail!(
                "Gemma layer type metadata '{key}' has {} valid entries; expected {n_layers}",
                parsed.len()
            );
        }
    }
    log::warn!(
        "Gemma GGUF has no explicit layer-type metadata; using the architecture's every-sixth-global default"
    );
    Ok((0..n_layers)
        .map(|i| {
            if (i + 1) % 6 == 0 {
                Gemma4AttentionType::Global
            } else {
                Gemma4AttentionType::Local
            }
        })
        .collect())
}

pub(super) fn get_bool(loader: &GgufLoader, key: &str, default: bool) -> bool {
    match loader.metadata.get(key) {
        Some(GgufValue::Bool(v)) => *v,
        Some(GgufValue::U32(v)) => *v != 0,
        Some(GgufValue::I32(v)) => *v != 0,
        _ => default,
    }
}

pub(super) fn get_u32_any(loader: &GgufLoader, keys: &[&str], default: u32) -> anyhow::Result<u32> {
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

pub(super) fn get_u32_or_first_array_any(
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

pub(super) fn get_optional_u32(loader: &GgufLoader, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|key| match loader.metadata.get(*key) {
        Some(GgufValue::U32(v)) => Some(*v),
        Some(GgufValue::U64(v)) => (*v).try_into().ok(),
        Some(GgufValue::I32(v)) if *v >= 0 => Some(*v as u32),
        _ => None,
    })
}

pub(super) fn get_f32_any(loader: &GgufLoader, keys: &[&str], default: f32) -> anyhow::Result<f32> {
    for key in keys {
        if let Some(GgufValue::F32(v)) = loader.metadata.get(*key) {
            return Ok(*v);
        }
    }
    Ok(default)
}

pub(super) fn get_optional_f32(loader: &GgufLoader, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| match loader.metadata.get(*key) {
        Some(GgufValue::F32(v)) => Some(*v),
        Some(GgufValue::U8(v)) => Some(*v as f32),
        Some(GgufValue::U32(v)) => Some(*v as f32),
        Some(GgufValue::U64(v)) => Some(*v as f32),
        Some(GgufValue::I32(v)) => Some(*v as f32),
        _ => None,
    })
}
