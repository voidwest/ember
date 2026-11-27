use super::*;

impl Llama<CpuBackend> {
    /// build a llama model from a gguf loader.
    ///
    /// reads metadata keys under the `llama.*` namespace (as written
    /// by llama.cpp's `llama-arch.cpp`) and maps gguf tensor names
    /// from the llama naming convention.
    ///
    /// expected gguf tensor names per layer:
    ///   `blk.{i}.attn_q.weight`       -> q_proj
    ///   `blk.{i}.attn_k.weight`       -> k_proj
    ///   `blk.{i}.attn_v.weight`       -> v_proj
    ///   `blk.{i}.attn_output.weight`  -> o_proj
    ///   `blk.{i}.ffn_gate.weight`     -> gate_proj
    ///   `blk.{i}.ffn_up.weight`       -> up_proj
    ///   `blk.{i}.ffn_down.weight`     -> down_proj
    ///   `blk.{i}.attn_norm.weight`    -> input_layernorm (rms, no bias)
    ///   `blk.{i}.ffn_norm.weight`     -> post_attention_layernorm (rms, no bias)
    ///
    /// global tensors:
    ///   `token_embd.weight`           -> embed_tokens
    ///   `output_norm.weight`          -> final rms norm (no bias)
    ///   `output.weight`               -> lm_head (linear, no bias)
    ///
    /// design note: f32/f16 linear weights are loaded with their gguf logical
    /// shape and transposed when building `Linear`, matching `Gpt2::from_loader`.
    /// q8_0 weights are loaded into `QuantizedWeight` with the reversed
    /// `[out_features, in_features]` shape expected by the quantized matmul path.
    pub fn from_loader(loader: crate::loader::GgufLoader) -> anyhow::Result<Self> {
        Self::from_loader_with_max_seq_len(loader, None)
    }

    /// build a llama model from a gguf loader, optionally capping runtime
    /// context length and rope table allocation below the GGUF metadata value.
    pub fn from_loader_with_max_seq_len(
        loader: crate::loader::GgufLoader,
        max_seq_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::from_loader_impl(loader, max_seq_len, true, None)
    }

    /// As [`Self::from_loader_with_max_seq_len`], consulting an optional
    /// on-disk cache for the packed Q8_0 VNNI decode layout (see
    /// [`crate::packed_cache`]). Cache problems degrade to in-memory packing.
    pub fn from_loader_with_max_seq_len_cached(
        loader: crate::loader::GgufLoader,
        max_seq_len: Option<usize>,
        packed_cache: Option<&crate::packed_cache::PackedCache>,
    ) -> anyhow::Result<Self> {
        Self::from_loader_impl(loader, max_seq_len, true, packed_cache)
    }

    /// Build a Llama-family model without consulting the automatic packed
    /// decode environment switch. Lifecycle experiments use this constructor
    /// so packing can occur at an explicit phase boundary in the same binary.
    pub fn from_loader_without_packed_decode(
        loader: crate::loader::GgufLoader,
        max_seq_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::from_loader_impl(loader, max_seq_len, false, None)
    }

    fn from_loader_impl(
        mut loader: crate::loader::GgufLoader,
        max_seq_len: Option<usize>,
        allow_automatic_packing: bool,
        packed_cache: Option<&crate::packed_cache::PackedCache>,
    ) -> anyhow::Result<Self> {
        use crate::loader::LoadedTensor;
        use crate::tensor::compute_rope_freqs;

        let k_decisions = std::mem::take(&mut loader.k_decisions)
            .into_iter()
            .collect::<BTreeMap<_, _>>();

        if let Some(architecture) = loader.metadata.get("general.architecture") {
            match architecture {
                crate::loader::GgufValue::Str(architecture) => anyhow::ensure!(
                    matches!(architecture.as_str(), "llama" | "qwen2" | "qwen3"),
                    "unsupported Llama-family GGUF architecture '{architecture}'"
                ),
                _ => anyhow::bail!("GGUF general.architecture must be a string"),
            }
        }

        let mut config = LlamaConfig::from_gguf_metadata(&loader)?;
        // Some Qwen2.5 GGUFs omit the family vocab_size key (a converter
        // quirk seen in the v0.3 validation artifacts). When the key is
        // absent, the token embedding tensor's row count is authoritative
        // for the vocabulary. This only fires for a missing key; a
        // present-but-wrong key stays a hard error downstream.
        let vocab_key_missing = match loader.metadata.get("general.architecture") {
            Some(crate::loader::GgufValue::Str(arch)) if arch == "qwen2" => {
                !loader.metadata.contains_key("qwen2.vocab_size")
            }
            Some(crate::loader::GgufValue::Str(arch)) if arch == "qwen3" => {
                !loader.metadata.contains_key("qwen3.vocab_size")
            }
            _ => false,
        };
        if vocab_key_missing {
            // GGUF token_embd dims are [n_embd, n_vocab] (first dim
            // contiguous); the vocab dimension is the second.
            if let Some(meta) = loader.tensor_meta.get("token_embd.weight")
                && let Some(&rows) = meta.dims.get(1)
                && config.vocab_size != rows
            {
                log::warn!(
                            "qwen2.vocab_size metadata missing; using token_embd rows ({rows}) for vocab_size"
                        );
                config.vocab_size = rows;
            }
        }
        config.validate()?;
        if let Some(max_seq_len) = max_seq_len {
            anyhow::ensure!(
                max_seq_len > 0,
                "max sequence length must be greater than zero"
            );
            config.max_seq_len = config.max_seq_len.min(max_seq_len);
        }
        log::debug!("llama config: {:?}", config);
        loader.check_model_dequantization_budget()?;
        let n_layers = config.n_layers;
        let mut packed_decode_enabled = allow_automatic_packing
            && config.rope_layout == RopeLayout::AdjacentPair
            && std::env::var_os("EMBER_LLAMA_PACKED_Q8").is_none_or(|value| value != "0");
        if packed_decode_enabled
            && loader.q8_0_encoded_bytes()? > crate::loader::limits::MAX_PACKED_DECODE_BYTES
        {
            log::warn!(
                "auto packed decode disabled: Q8_0 payload exceeds the {}-byte budget",
                crate::loader::limits::MAX_PACKED_DECODE_BYTES
            );
            packed_decode_enabled = false;
        }

        // precompute rope tables once, shared across all attention layers.
        // Llama-3.x models converted with llama3-style rope scaling carry a
        // `rope_freqs.weight` tensor of per-frequency-pair factors (ggml
        // semantics: effective freq = base_freq / factor); load and apply
        // it when present so scaled models match the HF reference exactly.
        let rope_factors = loader.take_optional_f32(&["rope_freqs.weight".to_string()])?;
        let rope_factors = match rope_factors {
            Some(t) => {
                anyhow::ensure!(
                    t.shape() == [config.head_dim / 2],
                    "rope_freqs.weight must have shape [{}], got {:?}",
                    config.head_dim / 2,
                    t.shape()
                );
                anyhow::ensure!(
                    t.data()
                        .iter()
                        .all(|factor| factor.is_finite() && *factor > 0.0),
                    "rope_freqs.weight must contain only finite positive factors"
                );
                log::info!("llama: applying rope_freqs.weight factors (llama3-style rope scaling)");
                Some(t.data().to_vec())
            }
            None => None,
        };
        // Check the mandatory inventory before computing RoPE tables. A
        // truncated/hostile GGUF must not pay the metadata-sized allocation
        // cost merely to fail on its first missing tensor.
        let mut required = Vec::with_capacity(2 + n_layers * 9);
        required.push("token_embd.weight".to_string());
        required.push("output_norm.weight".to_string());
        for layer in 0..n_layers {
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
            ] {
                required.push(format!("blk.{layer}.{suffix}"));
            }
        }
        crate::loader::require_tensors(&loader, &required)?;
        // Validate the cheap tensor geometry before allocating tables sized by
        // hostile context metadata. `tensor_meta` retains GGUF-native [in,
        // out] dimensions; direct unit-test loaders fall back to their
        // in-memory tensor shapes.
        let tensor_dims = |name: &str| -> anyhow::Result<Vec<usize>> {
            if let Some(meta) = loader.tensor_meta.get(name) {
                return Ok(meta.dims.clone());
            }
            match loader.tensors.get(name) {
                Some(crate::loader::LoadedTensor::F32(tensor)) => Ok(tensor.shape().to_vec()),
                Some(crate::loader::LoadedTensor::Half(weight)) => Ok(weight.dims().to_vec()),
                Some(crate::loader::LoadedTensor::Q8_0(weight)) => {
                    Ok(vec![weight.in_features(), weight.out_features()])
                }
                Some(crate::loader::LoadedTensor::KQuant(weight)) => {
                    Ok(vec![weight.in_features(), weight.out_features()])
                }
                None => anyhow::bail!("missing tensor geometry for '{name}'"),
            }
        };
        let check_tensor_dims = |name: &str, expected: &[usize]| -> anyhow::Result<()> {
            let actual = tensor_dims(name)?;
            anyhow::ensure!(
                actual == expected,
                "tensor '{name}' has dimensions {:?}, expected {:?}",
                actual,
                expected
            );
            Ok(())
        };
        check_tensor_dims("token_embd.weight", &[config.embed_dim, config.vocab_size])?;
        check_tensor_dims("output_norm.weight", &[config.embed_dim])?;
        if loader.tensors.contains_key("output.weight") {
            check_tensor_dims("output.weight", &[config.embed_dim, config.vocab_size])?;
        }
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}.");
            check_tensor_dims(
                &format!("{prefix}attn_q.weight"),
                &[config.embed_dim, config.n_heads * config.head_dim],
            )?;
            for suffix in ["attn_k.weight", "attn_v.weight"] {
                check_tensor_dims(
                    &format!("{prefix}{suffix}"),
                    &[config.embed_dim, config.n_kv_heads * config.head_dim],
                )?;
            }
            check_tensor_dims(
                &format!("{prefix}attn_output.weight"),
                &[config.n_heads * config.head_dim, config.embed_dim],
            )?;
            for (suffix, width) in [
                ("attn_q.bias", config.n_heads * config.head_dim),
                ("attn_k.bias", config.n_kv_heads * config.head_dim),
                ("attn_v.bias", config.n_kv_heads * config.head_dim),
            ] {
                let name = format!("{prefix}{suffix}");
                if loader.tensors.contains_key(&name) {
                    check_tensor_dims(&name, &[width])?;
                }
            }
            for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                let name = format!("{prefix}{suffix}");
                if loader.tensors.contains_key(&name) {
                    check_tensor_dims(&name, &[config.head_dim])?;
                }
            }
            let gate_dims = tensor_dims(&format!("{prefix}ffn_gate.weight"))?;
            anyhow::ensure!(
                gate_dims.len() == 2
                    && gate_dims[0] == config.embed_dim
                    && gate_dims[1] <= crate::loader::limits::MAX_INTERMEDIATE_DIM,
                "tensor '{prefix}ffn_gate.weight' has dimensions {:?}, expected [embed_dim, intermediate_dim]",
                gate_dims
            );
            let intermediate_dim = gate_dims[1];
            check_tensor_dims(
                &format!("{prefix}ffn_up.weight"),
                &[config.embed_dim, intermediate_dim],
            )?;
            check_tensor_dims(
                &format!("{prefix}ffn_down.weight"),
                &[intermediate_dim, config.embed_dim],
            )?;
            for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
                check_tensor_dims(&format!("{prefix}{suffix}"), &[config.embed_dim])?;
            }
        }
        let (rope_cos, rope_sin) = compute_rope_freqs(
            config.max_seq_len,
            config.head_dim,
            config.rope_theta,
            rope_factors.as_deref(),
        );
        let rope_frequency_factors = rope_factors;
        log::debug!(
            "rope_cos shape: {:?}, rope_sin shape: {:?}",
            rope_cos.shape(),
            rope_sin.shape()
        );

        let rope_cos = Arc::new(rope_cos);
        let rope_sin = Arc::new(rope_sin);

        let embed_tokens: LlamaEmbedding<CpuBackend> =
            match loader.take_tensor("token_embd.weight")? {
                LoadedTensor::F32(tensor) => {
                    // GGUF stores the embedding as [embed, vocab] with vocab
                    // rows contiguous, i.e. already row-major [vocab, embed];
                    // only the dims need swapping for the row lookup.
                    let shape = tensor.shape();
                    anyhow::ensure!(shape.len() == 2, "token_embd.weight must be 2D");
                    let embedding_shape = [shape[1], shape[0]];
                    LlamaEmbedding::F32(tensor.into_reshape(&embedding_shape))
                }
                LoadedTensor::Half(weight) => {
                    anyhow::ensure!(weight.dims().len() == 2, "token_embd.weight must be 2D");
                    LlamaEmbedding::Half(weight)
                }
                LoadedTensor::Q8_0(weight) => LlamaEmbedding::Q8_0(weight),
                LoadedTensor::KQuant(weight) => LlamaEmbedding::KQuant(weight),
            };

        let mut packing_ns = 0u64;
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            // optionally load qk norm weights (qwen3, etc.)
            let qk_q_norm =
                take_optional_llama_norm(&mut loader, &format!("blk.{}.attn_q_norm.weight", i));
            let qk_k_norm =
                take_optional_llama_norm(&mut loader, &format!("blk.{}.attn_k_norm.weight", i));

            let attn = LlamaAttention::new_shared(
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.attn_q.weight", i),
                    Some(&format!("blk.{}.attn_q.bias", i)),
                    packed_decode_enabled,
                    &mut packing_ns,
                    packed_cache,
                )?,
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.attn_k.weight", i),
                    Some(&format!("blk.{}.attn_k.bias", i)),
                    packed_decode_enabled,
                    &mut packing_ns,
                    packed_cache,
                )?,
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.attn_v.weight", i),
                    Some(&format!("blk.{}.attn_v.bias", i)),
                    packed_decode_enabled,
                    &mut packing_ns,
                    packed_cache,
                )?,
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.attn_output.weight", i),
                    None,
                    packed_decode_enabled,
                    &mut packing_ns,
                    packed_cache,
                )?,
                Arc::clone(&rope_cos),
                Arc::clone(&rope_sin),
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                config.rope_layout,
                config.qk_norm_order,
                qk_q_norm,
                qk_k_norm,
            )
            .with_qk_norm_eps(config.norm_eps);

            let mlp = LlamaMlp::new(
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.ffn_gate.weight", i),
                    None,
                    packed_decode_enabled,
                    &mut packing_ns,
                    packed_cache,
                )?,
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.ffn_up.weight", i),
                    None,
                    packed_decode_enabled,
                    &mut packing_ns,
                    packed_cache,
                )?,
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.ffn_down.weight", i),
                    None,
                    packed_decode_enabled,
                    &mut packing_ns,
                    packed_cache,
                )?,
            );

            blocks.push(LlamaBlock::new(
                loader.take_f32(&format!("blk.{}.attn_norm.weight", i))?,
                attn,
                loader.take_f32(&format!("blk.{}.ffn_norm.weight", i))?,
                mlp,
                config.norm_eps,
            ));
        }

        // lm_head: use output.weight if present, otherwise tie with embed_tokens
        let has_output_weight = loader.tensors.contains_key("output.weight");
        let mut head = match loader.tensors.remove("output.weight") {
            Some(LoadedTensor::F32(tensor)) => {
                anyhow::ensure!(
                    tensor.ndim() == 2,
                    "output.weight must be a 2D linear weight, got {:?}",
                    tensor.shape()
                );
                Linear::new(crate::loader::gguf_to_row_major_f32(tensor), None)
            }
            Some(LoadedTensor::Half(weight)) => {
                anyhow::ensure!(
                    weight.dims().len() == 2,
                    "output.weight must be a 2D linear weight, got {:?}",
                    weight.dims()
                );
                Linear::new_half(weight, None)
            }
            Some(LoadedTensor::Q8_0(weight)) => Linear::new_q8_0(weight, None),
            Some(LoadedTensor::KQuant(weight)) => Linear::new_k(weight, None),
            None => match &embed_tokens {
                // Tied embeddings are already laid out as [vocab, embed] in
                // QuantizedWeight, exactly the [out, in] layout the Q8 matmul
                // expects. Reusing the mapping avoids a second ~1 GiB F32
                // embedding copy and keeps decode on the packed integer path.
                LlamaEmbedding::Q8_0(weight) => {
                    Linear::<CpuBackend>::new_q8_0(weight.clone(), None)
                }
                LlamaEmbedding::KQuant(weight) => Linear::<CpuBackend>::new_k(weight.clone(), None),
                // The table's GGUF dims [embed, vocab] are exactly a linear
                // weight's [in, out]: share the mapping, no transpose.
                LlamaEmbedding::Half(weight) => {
                    Linear::<CpuBackend>::new_half(weight.clone(), None)
                }
                LlamaEmbedding::F32(tensor) => {
                    // The embedding is already reinterpreted as [vocab, embed]
                    // row-major; the linear needs [embed, vocab], so a real
                    // transpose (data reorder) is required — not the raw-GGUF
                    // helper, which would double-transpose.
                    Linear::<CpuBackend>::new(tensor.transpose(), None)
                }
            },
        };
        let interleaved_start = std::time::Instant::now();
        // Stable cache entry name: the source tensor backing the head (tied
        // heads reuse the embedding tensor).
        let head_entry_name = if has_output_weight {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        head.prepare_interleaved_cached(
            INTERLEAVED_MIN_OUT_FEATURES,
            packed_cache,
            head_entry_name,
        );
        let packing_interleaved_ns =
            interleaved_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;

        let mut model = Self {
            embed_tokens,
            blocks,
            norm: loader.take_f32("output_norm.weight")?,
            head,
            config,
            fast_decode_inter_dim: None,
            plan_cache: OnceLock::new(),
            plan_provenance: RefCell::new((None, None, None)),
            k_decisions,
            head_tied: !has_output_weight,
            execution_mode: RefCell::new(ExecutionMode::Reference),
            decode_state: RefCell::new(None),
            packing_ns,
            packing_interleaved_ns,
            rope_frequency_factors,
        };
        model.validate_loaded_shapes()?;
        model.fast_decode_inter_dim = model.eligible_fast_decode_inter_dim();
        log::debug!(
            "llama q8 decode workspace: {}",
            if model.fast_decode_inter_dim.is_some() {
                "enabled"
            } else {
                "unavailable (mixed or unsupported weight shapes)"
            }
        );
        if let Some(cache) = packed_cache {
            log::debug!(
                "packed cache: {} hits, {} misses",
                cache.hits(),
                cache.misses()
            );
            cache.finish_write();
        }
        Ok(model)
    }

    pub(super) fn validate_loaded_shapes(&self) -> anyhow::Result<()> {
        let backend = CpuBackend;
        let (vocab_size, embed_dim) = match &self.embed_tokens {
            LlamaEmbedding::F32(tensor) => {
                anyhow::ensure!(tensor.shape().len() == 2, "token embedding must be 2D");
                (tensor.shape()[0], tensor.shape()[1])
            }
            LlamaEmbedding::Half(weight) => {
                anyhow::ensure!(weight.dims().len() == 2, "token embedding must be 2D");
                (weight.rows(), weight.row_len())
            }
            LlamaEmbedding::Q8_0(weight) => (weight.out_features(), weight.in_features()),
            LlamaEmbedding::KQuant(weight) => (weight.out_features(), weight.in_features()),
        };
        anyhow::ensure!(
            (vocab_size, embed_dim) == (self.config.vocab_size, self.config.embed_dim),
            "token embedding must have shape [{}, {}], got [{vocab_size}, {embed_dim}]",
            self.config.vocab_size,
            self.config.embed_dim
        );
        anyhow::ensure!(
            self.norm.shape() == [self.config.embed_dim],
            "output_norm.weight must have shape [{}], got {:?}",
            self.config.embed_dim,
            self.norm.shape()
        );
        anyhow::ensure!(
            self.head.in_features(&backend) == self.config.embed_dim
                && self.head.out_features(&backend) == self.config.vocab_size,
            "output head must be {} -> {}, got {} -> {}",
            self.config.embed_dim,
            self.config.vocab_size,
            self.head.in_features(&backend),
            self.head.out_features(&backend)
        );

        let q_width = self
            .config
            .n_heads
            .checked_mul(self.config.head_dim)
            .context("query projection width overflow")?;
        let kv_width = self
            .config
            .n_kv_heads
            .checked_mul(self.config.head_dim)
            .context("KV projection width overflow")?;
        for (index, block) in self.blocks.iter().enumerate() {
            anyhow::ensure!(
                block.input_layernorm.shape() == [self.config.embed_dim]
                    && block.post_attention_layernorm.shape() == [self.config.embed_dim],
                "block {index} RMS norm shapes must be [{}]",
                self.config.embed_dim
            );
            let attention = &block.self_attn;
            for (name, projection, output) in [
                ("query", &attention.q_proj, q_width),
                ("key", &attention.k_proj, kv_width),
                ("value", &attention.v_proj, kv_width),
            ] {
                anyhow::ensure!(
                    projection.in_features(&backend) == self.config.embed_dim
                        && projection.out_features(&backend) == output,
                    "block {index} {name} projection must be {} -> {output}",
                    self.config.embed_dim
                );
            }
            anyhow::ensure!(
                attention.o_proj.in_features(&backend) == q_width
                    && attention.o_proj.out_features(&backend) == self.config.embed_dim,
                "block {index} attention output projection must be {q_width} -> {}",
                self.config.embed_dim
            );
            for (name, norm) in [("query", &attention.q_norm), ("key", &attention.k_norm)] {
                if let Some(norm) = norm {
                    anyhow::ensure!(
                        norm.shape() == [self.config.head_dim],
                        "block {index} {name} norm must have shape [{}]",
                        self.config.head_dim
                    );
                }
            }
            let mlp = &block.mlp;
            let intermediate = mlp.gate_proj.out_features(&backend);
            anyhow::ensure!(
                intermediate > 0
                    && mlp.gate_proj.in_features(&backend) == self.config.embed_dim
                    && mlp.up_proj.in_features(&backend) == self.config.embed_dim
                    && mlp.up_proj.out_features(&backend) == intermediate
                    && mlp.down_proj.in_features(&backend) == intermediate
                    && mlp.down_proj.out_features(&backend) == self.config.embed_dim,
                "block {index} MLP projection shapes are inconsistent"
            );
        }
        Ok(())
    }

    /// Construct packed representations for the selected existing projection
    /// group. Packing and eviction are timed separately; when eviction is
    /// requested it happens immediately after each weight is packed, matching
    /// the production path's bounded temporary residency.
    pub fn prepare_packed_decode_selected(
        &mut self,
        selection: LlamaPackedSelection,
        evict_source_pages: bool,
    ) -> anyhow::Result<LlamaPackingStats> {
        if self.config.rope_layout != RopeLayout::AdjacentPair {
            anyhow::bail!(
                "packed lifecycle experiments require adjacent-pair Llama RoPE; \
                 this architecture remains on the generic path"
            );
        }
        if !crate::simd::packed_q8_0_vnni_supported() {
            anyhow::bail!("packed Q8_0 AVX-512 VNNI kernel is unavailable on this CPU");
        }

        let mut stats = LlamaPackingStats::default();
        for block in &mut self.blocks {
            if selection.includes_attention() {
                prepare_experimental_linear(
                    &mut block.self_attn.q_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
                prepare_experimental_linear(
                    &mut block.self_attn.k_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
                prepare_experimental_linear(
                    &mut block.self_attn.v_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
                prepare_experimental_linear(
                    &mut block.self_attn.o_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
            }
            if selection.includes_gate_up() {
                prepare_experimental_linear(
                    &mut block.mlp.gate_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
                prepare_experimental_linear(
                    &mut block.mlp.up_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
            }
            if selection.includes_down() {
                prepare_experimental_linear(
                    &mut block.mlp.down_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
            }
        }
        Ok(stats)
    }

    /// Re-issue `MADV_DONTNEED` for selected packed projection sources after a
    /// generic prefill may have faulted their row-contiguous pages back in.
    pub fn reevict_packed_decode_sources(
        &self,
        selection: LlamaPackedSelection,
    ) -> anyhow::Result<LlamaEvictionStats> {
        let mut stats = LlamaEvictionStats::default();
        for block in &self.blocks {
            if selection.includes_attention() {
                reevict_experimental_linear(&block.self_attn.q_proj, &mut stats)?;
                reevict_experimental_linear(&block.self_attn.k_proj, &mut stats)?;
                reevict_experimental_linear(&block.self_attn.v_proj, &mut stats)?;
                reevict_experimental_linear(&block.self_attn.o_proj, &mut stats)?;
            }
            if selection.includes_gate_up() {
                reevict_experimental_linear(&block.mlp.gate_proj, &mut stats)?;
                reevict_experimental_linear(&block.mlp.up_proj, &mut stats)?;
            }
            if selection.includes_down() {
                reevict_experimental_linear(&block.mlp.down_proj, &mut stats)?;
            }
        }
        Ok(stats)
    }

    /// Return the number and byte size of selected packed projections now
    /// owned by the model.
    pub fn packed_decode_summary(&self, selection: LlamaPackedSelection) -> (usize, usize) {
        let mut weights = 0;
        let mut bytes = 0;
        let mut account = |linear: &Linear<CpuBackend>| {
            if linear.has_packed_decode() {
                weights += 1;
                bytes += linear.packed_decode_bytes();
            }
        };
        for block in &self.blocks {
            if selection.includes_attention() {
                account(&block.self_attn.q_proj);
                account(&block.self_attn.k_proj);
                account(&block.self_attn.v_proj);
                account(&block.self_attn.o_proj);
            }
            if selection.includes_gate_up() {
                account(&block.mlp.gate_proj);
                account(&block.mlp.up_proj);
            }
            if selection.includes_down() {
                account(&block.mlp.down_proj);
            }
        }
        (weights, bytes)
    }
}

/// Take a linear projection tensor, optionally with a fused f32 bias.
///
/// qwen2/qwen2.5 attention projections carry `blk.{i}.attn_q.bias` /
/// `attn_k.bias` / `attn_v.bias`; llama and qwen3 GGUFs do not, and pass
/// through with no bias.
#[allow(
    clippy::too_many_arguments,
    reason = "loader state and load options are separate inputs"
)]
pub(super) fn take_llama_linear(
    loader: &mut crate::loader::GgufLoader,
    name: &str,
    bias_name: Option<&str>,
    prepare_packed: bool,
    packing_ns: &mut u64,
    packed_cache: Option<&crate::packed_cache::PackedCache>,
) -> anyhow::Result<Linear<CpuBackend>> {
    use crate::loader::LoadedTensor;

    let bias = match bias_name {
        Some(bias_name) => loader.take_optional_f32(&[bias_name.to_string()])?,
        None => None,
    };
    let mut linear = match loader.take_tensor(name)? {
        LoadedTensor::F32(tensor) => {
            anyhow::ensure!(
                tensor.ndim() == 2,
                "{name} must be a 2D linear weight, got {:?}",
                tensor.shape()
            );
            Linear::new(crate::loader::gguf_to_row_major_f32(tensor), bias)
        }
        LoadedTensor::Half(weight) => {
            anyhow::ensure!(
                weight.dims().len() == 2,
                "{name} must be a 2D linear weight, got {:?}",
                weight.dims()
            );
            Linear::new_half(weight, bias)
        }
        LoadedTensor::Q8_0(weight) => Linear::new_q8_0(weight, bias),
        LoadedTensor::KQuant(weight) => Linear::new_k(weight, bias),
    };
    if prepare_packed {
        let pack_start = std::time::Instant::now();
        // A validated on-disk entry replaces the repack; otherwise pack and
        // record the result for the next run.
        let cached = packed_cache.and_then(|cache| {
            let weight = linear.q8_weight_without_bias()?;
            cache.get_vnni(name, weight.out_features(), weight.in_features())
        });
        match cached {
            Some(packed) => linear.packed_decode = Some(packed),
            None => {
                linear.prepare_packed_decode();
                if let (Some(cache), Some(packed)) = (packed_cache, linear.packed_decode.as_ref()) {
                    cache.record_vnni(name, packed);
                }
            }
        }
        *packing_ns =
            packing_ns.saturating_add(pack_start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
    }
    Ok(linear)
}

fn take_optional_llama_norm(
    loader: &mut crate::loader::GgufLoader,
    name: &str,
) -> Option<CpuTensor> {
    match loader.tensors.remove(name) {
        Some(crate::loader::LoadedTensor::F32(tensor)) => Some(tensor),
        Some(crate::loader::LoadedTensor::Half(weight)) => Some(weight.to_f32_tensor()),
        _ => None,
    }
}

fn prepare_experimental_linear(
    linear: &mut Linear<CpuBackend>,
    evict_source_pages: bool,
    stats: &mut LlamaPackingStats,
) -> anyhow::Result<()> {
    let packing_start = std::time::Instant::now();
    let packed_bytes = linear.prepare_packed_decode_without_eviction();
    let packing_elapsed = packing_start.elapsed();
    let Some(packed_bytes) = packed_bytes else {
        return Ok(());
    };

    stats.weights_packed += 1;
    stats.packed_bytes += packed_bytes;
    stats.packing_ns = stats
        .packing_ns
        .saturating_add(packing_elapsed.as_nanos() as u64);

    if evict_source_pages && linear.has_mapped_q8_source() {
        stats.eviction_attempts += 1;
        let eviction_start = std::time::Instant::now();
        let evicted = linear.evict_packed_source_pages()?;
        stats.eviction_ns = stats
            .eviction_ns
            .saturating_add(eviction_start.elapsed().as_nanos() as u64);
        stats.eviction_successes += usize::from(evicted);
    }
    Ok(())
}

fn reevict_experimental_linear(
    linear: &Linear<CpuBackend>,
    stats: &mut LlamaEvictionStats,
) -> anyhow::Result<()> {
    if !linear.has_packed_decode() || !linear.has_mapped_q8_source() {
        return Ok(());
    }
    stats.eviction_attempts += 1;
    let eviction_start = std::time::Instant::now();
    let evicted = linear.evict_packed_source_pages()?;
    stats.eviction_ns = stats
        .eviction_ns
        .saturating_add(eviction_start.elapsed().as_nanos() as u64);
    stats.eviction_successes += usize::from(evicted);
    Ok(())
}

#[inline]
pub(super) fn record_profiled_q8(
    layer: usize,
    operator: &'static str,
    weight: &crate::quant::QuantizedWeight,
    elapsed: std::time::Duration,
) {
    let execution_mode =
        if crate::simd::q8_decode_uses_row_parallel(weight.out_features(), weight.in_features()) {
            crate::decode_profile::DecodeExecutionMode::RowParallelRayon
        } else {
            crate::decode_profile::DecodeExecutionMode::Serial
        };
    crate::decode_profile::record_quantized(
        layer,
        operator,
        weight.in_features(),
        weight.out_features(),
        execution_mode,
        "Q8_0",
        elapsed,
    );
}

#[inline]
pub(super) fn record_profiled_packed(
    layer: usize,
    operator: &'static str,
    weight: &crate::quant::QuantizedWeightVnni,
    elapsed: std::time::Duration,
) {
    crate::decode_profile::record_quantized(
        layer,
        operator,
        weight.in_features(),
        weight.out_features(),
        crate::decode_profile::DecodeExecutionMode::PackedRowParallelRayon,
        "Q8_0",
        elapsed,
    );
}

#[doc(hidden)]
pub fn llama_embed_tokens<B: Backend>(
    backend: &B,
    table: &LlamaEmbedding<B>,
    token_ids: &[u32],
    embed_dim: usize,
) -> Result<B::Tensor, B::Error> {
    let mut output = backend.zeroes(&[token_ids.len(), embed_dim])?;
    for (row, &token) in token_ids.iter().enumerate() {
        match table {
            LlamaEmbedding::F32(table) => {
                backend.assign_row_from_table(&mut output, row, table, token as usize)?;
            }
            LlamaEmbedding::Half(table) => {
                backend.assign_row_from_half(&mut output, row, table, token as usize)?;
            }
            LlamaEmbedding::Q8_0(table) => {
                backend.assign_row_from_q8_0(&mut output, row, table, token as usize)?;
            }
            LlamaEmbedding::KQuant(table) => {
                backend.assign_row_from_k(&mut output, row, table, token as usize)?;
            }
        }
    }
    Ok(output)
}
