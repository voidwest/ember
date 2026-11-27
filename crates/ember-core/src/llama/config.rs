#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    AdjacentPair,
    SplitHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkNormOrder {
    BeforeRope,
    AfterRope,
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// number of transformer layers
    pub n_layers: usize,
    /// number of query heads
    pub n_heads: usize,
    /// number of key/value heads (gqa: may be < n_heads)
    pub n_kv_heads: usize,
    /// hidden dimension per token
    pub embed_dim: usize,
    /// dimension per attention head (embed_dim / n_heads, often 128)
    pub head_dim: usize,
    /// maximum sequence length the model was trained for
    pub max_seq_len: usize,
    /// base frequency for rotary position embeddings
    /// (10000.0 for llama-2, 500000.0 for llama-3)
    pub rope_theta: f32,
    /// epsilon for rms normalization (typically 1e-5)
    pub norm_eps: f32,
    /// RoPE pairing convention for this architecture.
    pub rope_layout: RopeLayout,
    /// Q/K RMSNorm placement relative to RoPE for architectures that use it.
    pub qk_norm_order: QkNormOrder,
    /// token vocabulary size
    pub vocab_size: usize,
}

/// A present GGUF metadata value as `u32`: any integer type, or an integral
/// float, that converts exactly. Other types and out-of-range values error.
fn metadata_u32(key: &str, value: &crate::loader::GgufValue) -> anyhow::Result<u32> {
    use crate::loader::GgufValue as V;
    let converted = match *value {
        V::U8(v) => Some(u32::from(v)),
        V::I8(v) => u32::try_from(v).ok(),
        V::U16(v) => Some(u32::from(v)),
        V::I16(v) => u32::try_from(v).ok(),
        V::U32(v) => Some(v),
        V::I32(v) => u32::try_from(v).ok(),
        V::U64(v) => u32::try_from(v).ok(),
        V::I64(v) => u32::try_from(v).ok(),
        V::F32(v) => integral_f64_to_u32(f64::from(v)),
        V::F64(v) => integral_f64_to_u32(v),
        _ => anyhow::bail!(
            "GGUF metadata {key} must be numeric, got {}",
            metadata_type_name(value)
        ),
    };
    converted.ok_or_else(|| anyhow::anyhow!("GGUF metadata {key} value {value:?} does not fit u32"))
}

fn integral_f64_to_u32(value: f64) -> Option<u32> {
    (value.is_finite() && value.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&value))
        .then_some(value as u32)
}

/// A present GGUF metadata value as `f32`: `F32` as-is, integers that f32
/// represents exactly (|v| <= 2^24), and finite `F64` within f32 range
/// (rounded to nearest, as llama.cpp does). Other types error.
fn metadata_f32(key: &str, value: &crate::loader::GgufValue) -> anyhow::Result<f32> {
    use crate::loader::GgufValue as V;
    const EXACT: i64 = 1 << f32::MANTISSA_DIGITS;
    let exact_int = |v: i64| (-EXACT..=EXACT).contains(&v).then_some(v as f32);
    let converted = match *value {
        V::F32(v) => Some(v),
        V::F64(v) => (v.is_finite() && v.abs() <= f64::from(f32::MAX)).then_some(v as f32),
        V::U8(v) => Some(f32::from(v)),
        V::I8(v) => Some(f32::from(v)),
        V::U16(v) => Some(f32::from(v)),
        V::I16(v) => Some(f32::from(v)),
        V::U32(v) => exact_int(i64::from(v)),
        V::I32(v) => exact_int(i64::from(v)),
        V::U64(v) => i64::try_from(v).ok().and_then(exact_int),
        V::I64(v) => exact_int(v),
        _ => anyhow::bail!(
            "GGUF metadata {key} must be numeric, got {}",
            metadata_type_name(value)
        ),
    };
    converted.ok_or_else(|| {
        anyhow::anyhow!("GGUF metadata {key} value {value:?} does not fit f32 exactly")
    })
}

fn metadata_type_name(value: &crate::loader::GgufValue) -> &'static str {
    use crate::loader::GgufValue as V;
    match value {
        V::Bool(_) => "bool",
        V::Str(_) => "string",
        V::Array(_) | V::SkippedArray { .. } => "array",
        _ => "number",
    }
}

impl LlamaConfig {
    /// read config from gguf metadata, supporting multiple architectures.
    ///
    /// detects the architecture from `general.architecture` and uses the
    /// appropriate prefix (`llama.*`, `qwen2.*`).  falls back to `llama.*`
    /// when the architecture key is missing for backward compatibility.
    ///
    /// mapped metadata keys (per-architecture prefix):
    ///
    ///   `{prefix}.block_count`                       -> n_layers (default 32)
    ///   `{prefix}.attention.head_count`              -> n_heads (default 32)
    ///   `{prefix}.attention.head_count_kv`           -> n_kv_heads (default n_heads)
    ///   `{prefix}.embedding_length`                  -> embed_dim (default 4096)
    ///   `{prefix}.context_length`                    -> max_seq_len (default 2048)
    ///   `{prefix}.rope.freq_base`                    -> rope_theta (default 10000.0)
    ///   `{prefix}.attention.layer_norm_rms_epsilon`  -> norm_eps (default 1e-5)
    ///   `{prefix}.vocab_size`                        -> vocab_size (default 32000)
    ///
    /// supported architectures: llama, qwen2 (including qwen2.5)
    pub fn from_gguf_metadata(loader: &crate::loader::GgufLoader) -> anyhow::Result<Self> {
        use crate::loader::GgufValue;

        // detect architecture prefix from gguf metadata.
        // llama models use "llama.*", qwen2.5 uses "qwen2.*", etc.
        // fall back to "llama" for backward compatibility.
        let arch_prefix = match loader.metadata.get("general.architecture") {
            Some(GgufValue::Str(s)) => s.as_str(),
            _ => "llama",
        };
        // normalize: qwen2 covers qwen2.5 and qwen3 (same arch family)
        let prefix = match arch_prefix {
            "qwen2" => "qwen2",
            "qwen3" => "qwen3",
            _ => "llama",
        };

        let (rope_layout, qk_norm_order) = match prefix {
            // Qwen-family GGUFs use the split-half RoPE convention and apply
            // Q/K RMSNorm before RoPE. This was validated against llama.cpp
            // with golden-logit prompt ladders.
            "qwen2" | "qwen3" => (RopeLayout::SplitHalf, QkNormOrder::BeforeRope),
            _ => (RopeLayout::AdjacentPair, QkNormOrder::AfterRope),
        };

        // Architecture-specific key first, then the llama fallback. A present
        // key of any numeric GGUF type is accepted when it converts without
        // loss; a present key of another type (or out of range) is an error
        // rather than a silent default. Missing keys keep their defaults.
        let lookup = |key: &str| -> Option<(String, &GgufValue)> {
            let arch_key = format!("{prefix}.{key}");
            if let Some(value) = loader.metadata.get(&arch_key) {
                return Some((arch_key, value));
            }
            let llama_key = format!("llama.{key}");
            loader
                .metadata
                .get(&llama_key)
                .map(|value| (llama_key, value))
        };
        let get_u32 = |key: &str, default: u32| -> anyhow::Result<u32> {
            match lookup(key) {
                None => Ok(default),
                Some((name, value)) => metadata_u32(&name, value),
            }
        };
        let get_f32 = |key: &str, default: f32| -> anyhow::Result<f32> {
            match lookup(key) {
                None => Ok(default),
                Some((name, value)) => metadata_f32(&name, value),
            }
        };

        let n_layers = get_u32("block_count", 32)? as usize;
        let n_heads = get_u32("attention.head_count", 32)? as usize;
        let n_kv_heads = get_u32("attention.head_count_kv", n_heads as u32)? as usize;
        let embed_dim = get_u32("embedding_length", 4096)? as usize;
        // some architectures (qwen3, deepseek, etc.) specify head_dim explicitly
        // in the gguf metadata. fall back to embed_dim / n_heads when absent.
        let default_head_dim = embed_dim.checked_div(n_heads).unwrap_or(0);
        let head_dim = get_u32("attention.key_length", default_head_dim as u32)? as usize;
        let max_seq_len = get_u32("context_length", 2048)? as usize;
        let rope_theta = get_f32("rope.freq_base", 10000.0)?;
        let norm_eps = get_f32("attention.layer_norm_rms_epsilon", 1e-5)?;
        let vocab_size = get_u32("vocab_size", 32000)? as usize;

        Ok(Self {
            n_layers,
            n_heads,
            n_kv_heads,
            embed_dim,
            head_dim,
            max_seq_len,
            rope_theta,
            norm_eps,
            rope_layout,
            qk_norm_order,
            vocab_size,
        })
    }

    pub(super) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.n_layers > 0, "model must contain at least one layer");
        anyhow::ensure!(
            self.n_heads > 0,
            "attention head_count must be greater than zero"
        );
        anyhow::ensure!(
            self.n_kv_heads > 0 && self.n_kv_heads <= self.n_heads,
            "attention head_count_kv {} must be in 1..={}",
            self.n_kv_heads,
            self.n_heads
        );
        anyhow::ensure!(
            self.n_heads.is_multiple_of(self.n_kv_heads),
            "attention head_count {} must be divisible by head_count_kv {}",
            self.n_heads,
            self.n_kv_heads
        );
        anyhow::ensure!(
            self.embed_dim > 0,
            "embedding_length must be greater than zero"
        );
        anyhow::ensure!(
            self.head_dim >= 2 && self.head_dim.is_multiple_of(2),
            "attention key_length must be positive and even, got {}",
            self.head_dim
        );
        anyhow::ensure!(
            self.max_seq_len > 0,
            "context_length must be greater than zero"
        );
        anyhow::ensure!(self.vocab_size > 0, "vocab_size must be greater than zero");
        // Metadata values are attacker-controlled. Keep the existing
        // llama.cpp-style bounds (which are part of the loader contract) and
        // add caps for dimensions that otherwise size model allocations.
        use crate::loader::limits as l;
        anyhow::ensure!(
            self.n_layers <= l::MAX_LAYERS,
            "block_count {} exceeds the {} layer limit",
            self.n_layers,
            l::MAX_LAYERS
        );
        anyhow::ensure!(
            self.embed_dim <= l::MAX_EMBED_DIM,
            "embedding_length {} exceeds the {} limit",
            self.embed_dim,
            l::MAX_EMBED_DIM
        );
        anyhow::ensure!(
            self.max_seq_len <= l::MAX_CONTEXT_LEN,
            "context_length {} exceeds the {} sanity cap",
            self.max_seq_len,
            l::MAX_CONTEXT_LEN
        );
        anyhow::ensure!(
            self.n_heads <= l::MAX_HEADS,
            "attention head_count {} exceeds the {} sanity cap",
            self.n_heads,
            l::MAX_HEADS
        );
        anyhow::ensure!(
            self.head_dim <= l::MAX_HEAD_DIM,
            "attention key_length {} exceeds the {} sanity cap",
            self.head_dim,
            l::MAX_HEAD_DIM
        );
        anyhow::ensure!(
            self.vocab_size <= l::MAX_VOCAB_SIZE,
            "vocab_size {} exceeds the {} sanity cap",
            self.vocab_size,
            l::MAX_VOCAB_SIZE
        );
        anyhow::ensure!(
            self.max_seq_len
                .checked_mul(self.head_dim)
                .is_some_and(|elements| elements <= l::MAX_ROPE_TABLE_ELEMENTS),
            "context_length {} * key_length {} exceeds the {} RoPE-table element limit",
            self.max_seq_len,
            self.head_dim,
            l::MAX_ROPE_TABLE_ELEMENTS
        );
        anyhow::ensure!(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "rope.freq_base must be finite and greater than zero"
        );
        anyhow::ensure!(
            self.norm_eps.is_finite() && self.norm_eps >= 0.0,
            "attention.layer_norm_rms_epsilon must be finite and non-negative"
        );
        Ok(())
    }
}
