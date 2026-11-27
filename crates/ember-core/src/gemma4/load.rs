use super::*;

impl Gemma4<CpuBackend> {
    /// Build a Gemma-family model from a GGUF loader.
    pub fn from_loader(loader: GgufLoader) -> anyhow::Result<Self> {
        Self::from_loader_impl(loader, None)
    }

    /// As [`Self::from_loader`], consulting an optional on-disk cache for the
    /// packed Q8_0 gate/up decode layouts (see [`crate::packed_cache`]).
    /// Cache problems degrade to in-memory packing.
    pub fn from_loader_cached(
        loader: GgufLoader,
        packed_cache: Option<&crate::packed_cache::PackedCache>,
    ) -> anyhow::Result<Self> {
        Self::from_loader_impl(loader, packed_cache)
    }

    fn from_loader_impl(
        mut loader: GgufLoader,
        packed_cache: Option<&crate::packed_cache::PackedCache>,
    ) -> anyhow::Result<Self> {
        // Gemma consumes f32 tensors only; widen f16/bf16 ones exactly as
        // the loader used to.
        loader.materialize_half_tensors();
        if let Some(architecture) = loader.metadata.get("general.architecture") {
            match architecture {
                GgufValue::Str(architecture) => anyhow::ensure!(
                    matches!(architecture.as_str(), "gemma3" | "gemma4"),
                    "unsupported Gemma-family GGUF architecture '{architecture}'"
                ),
                _ => anyhow::bail!("GGUF general.architecture must be a string"),
            }
        }
        let mut config = Gemma4Config::from_gguf_metadata(&loader)?;
        log::debug!("gemma4 config: {:?}", config);

        let embed_tokens: Gemma4Embedding<CpuBackend> =
            match loader.take_tensor("token_embd.weight")? {
                LoadedTensor::F32(tensor) => {
                    // GGUF stores the embedding with the vocab dim contiguous,
                    // i.e. already row-major [vocab, embed]; swap dims only so
                    // assign_row_from_table / tied_embedding_logits see the
                    // orientation they expect.
                    let shape = tensor.shape();
                    anyhow::ensure!(shape.len() == 2, "token_embd.weight must be 2D");
                    Gemma4Embedding::F32(Arc::new(crate::tensor::CpuTensor::from_data(
                        vec![shape[1], shape[0]],
                        tensor.data().to_vec(),
                    )))
                }
                LoadedTensor::Q8_0(weight) => Gemma4Embedding::Q8_0(Arc::new(weight)),
                LoadedTensor::KQuant(_) => {
                    anyhow::bail!(NO_K_QUANT)
                }
                LoadedTensor::Half(_) => {
                    anyhow::bail!(HALF_NOT_WIDENED)
                }
            };

        // The GGUF does not reliably carry a vocab_size metadata key; derive it
        // from the actual token embedding so 262144-vocab variants (e.g. E2B)
        // load instead of failing the 256000 fallback.
        config.vocab_size = match &embed_tokens {
            Gemma4Embedding::F32(table) => table.shape()[0],
            Gemma4Embedding::Q8_0(table) => table.out_features(),
        };
        // The embedding tensor is authoritative for Gemma vocab size. Re-run
        // validation after deriving it so a hostile tensor cannot bypass the
        // metadata-time vocabulary cap.
        config.validate()?;

        // Load rope_freqs for partial RoPE application (global layers).
        // Validate this optional tensor before compute_rope_freqs: malformed
        // shapes or factors would otherwise trip its assertions.
        let rope_freqs: Option<Vec<f32>> = match loader.tensors.remove("rope_freqs.weight") {
            Some(LoadedTensor::F32(tensor)) => {
                anyhow::ensure!(
                    tensor.shape() == [config.global_head_dim / 2],
                    "rope_freqs.weight must have shape [{}], got {:?}",
                    config.global_head_dim / 2,
                    tensor.shape()
                );
                anyhow::ensure!(
                    tensor
                        .data()
                        .iter()
                        .all(|factor| factor.is_finite() && *factor > 0.0),
                    "rope_freqs.weight must contain only finite positive factors"
                );
                Some(tensor.data)
            }
            _ => None,
        };

        // Validate the mandatory inventory before allocating the local/global
        // RoPE tables. Shared-KV layers intentionally omit their own K/V
        // tensors; every other layer must carry both projections.
        let first_shared_layer = if config.num_kv_shared_layers > 0 {
            config.n_layers - config.num_kv_shared_layers
        } else {
            usize::MAX
        };
        let mut required = Vec::with_capacity(1 + config.n_layers * 10);
        required.push("output_norm.weight".to_string());
        let mut last_local_source = false;
        let mut last_global_source = false;
        for layer in 0..config.n_layers {
            for suffix in [
                "attn_q.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                required.push(format!("blk.{layer}.{suffix}"));
            }
            for suffix in ["attn_post_norm.weight", "ffn_post_norm.weight"] {
                let primary = format!("blk.{layer}.{suffix}");
                let alias_suffix = if suffix == "attn_post_norm.weight" {
                    "post_attention_norm.weight"
                } else {
                    "post_ffw_norm.weight"
                };
                let alias = format!("blk.{layer}.{alias_suffix}");
                if !loader.tensors.contains_key(&primary) && !loader.tensors.contains_key(&alias) {
                    required.push(primary);
                }
            }
            let layer_type = config.layer_type(layer);
            let has_k = loader
                .tensors
                .contains_key(&format!("blk.{layer}.attn_k.weight"));
            let has_v = loader
                .tensors
                .contains_key(&format!("blk.{layer}.attn_v.weight"));
            anyhow::ensure!(
                has_k == has_v,
                "Gemma 4 layer {layer} must provide both K and V projections or neither"
            );
            let has_own_kv = has_k;
            let is_shared = layer >= first_shared_layer;
            let has_source = match layer_type {
                Gemma4AttentionType::Local => last_local_source,
                Gemma4AttentionType::Global => last_global_source,
            };
            if is_shared {
                if !has_source {
                    required.push(format!("blk.{layer}.attn_k.weight"));
                    required.push(format!("blk.{layer}.attn_v.weight"));
                }
            } else {
                anyhow::ensure!(
                    has_own_kv,
                    "Gemma 4 non-shared layer {layer} must provide both K and V projections"
                );
                match layer_type {
                    Gemma4AttentionType::Local => last_local_source = true,
                    Gemma4AttentionType::Global => last_global_source = true,
                }
            }
        }
        crate::loader::require_tensors(&loader, &required)?;
        loader.check_model_dequantization_budget()?;
        let per_layer_dim = config
            .hidden_size_per_layer_input
            .unwrap_or(config.embed_dim);
        // Automatic Q8 packed decode repacks eligible projections into
        // same-size anonymous buffers; skip it for oversized Q8_0 payloads so
        // a large valid model does not double its resident footprint.
        let pack_gate_up =
            loader.q8_0_encoded_bytes()? <= crate::loader::limits::MAX_PACKED_DECODE_BYTES;
        // Validate tensor geometry before allocating tables sized by hostile
        // context metadata. The token embedding was consumed above, so its
        // in-memory orientation is checked separately.
        match &embed_tokens {
            Gemma4Embedding::F32(table) => anyhow::ensure!(
                table.shape() == [config.vocab_size, config.embed_dim],
                "token_embd.weight has dimensions {:?}, expected [{}, {}]",
                table.shape(),
                config.vocab_size,
                config.embed_dim
            ),
            Gemma4Embedding::Q8_0(table) => anyhow::ensure!(
                (table.out_features(), table.in_features())
                    == (config.vocab_size, config.embed_dim),
                "token_embd.weight has dimensions [{}, {}], expected [{}, {}]",
                table.out_features(),
                table.in_features(),
                config.vocab_size,
                config.embed_dim
            ),
        };
        let tensor_dims = |name: &str| -> anyhow::Result<Vec<usize>> {
            if let Some(meta) = loader.tensor_meta.get(name) {
                return Ok(meta.dims.clone());
            }
            match loader.tensors.get(name) {
                Some(LoadedTensor::F32(tensor)) => Ok(tensor.shape().to_vec()),
                Some(LoadedTensor::Half(weight)) => Ok(weight.dims().to_vec()),
                Some(LoadedTensor::Q8_0(weight)) => {
                    Ok(vec![weight.in_features(), weight.out_features()])
                }
                Some(LoadedTensor::KQuant(weight)) => {
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
        check_tensor_dims("output_norm.weight", &[config.embed_dim])?;
        for layer in 0..config.n_layers {
            let layer_type = config.layer_type(layer);
            let head_dim = config.head_dim_for(layer_type);
            let kv_heads = config.kv_heads_for(layer_type);
            let prefix = format!("blk.{layer}.");
            check_tensor_dims(
                &format!("{prefix}attn_q.weight"),
                &[config.embed_dim, config.n_heads * head_dim],
            )?;
            check_tensor_dims(
                &format!("{prefix}attn_output.weight"),
                &[config.n_heads * head_dim, config.embed_dim],
            )?;
            let has_k = loader
                .tensors
                .contains_key(&format!("{prefix}attn_k.weight"));
            let has_v = loader
                .tensors
                .contains_key(&format!("{prefix}attn_v.weight"));
            anyhow::ensure!(
                has_k == has_v,
                "Gemma 4 layer {layer} must provide both K and V projections or neither"
            );
            let is_shared = layer >= first_shared_layer;
            if has_k && !is_shared {
                for suffix in ["attn_k.weight", "attn_v.weight"] {
                    check_tensor_dims(
                        &format!("{prefix}{suffix}"),
                        &[config.embed_dim, kv_heads * head_dim],
                    )?;
                }
            }
            for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                check_tensor_dims(&format!("{prefix}{suffix}"), &[head_dim])?;
            }
            for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
                check_tensor_dims(&format!("{prefix}{suffix}"), &[config.embed_dim])?;
            }
            for suffix in ["attn_post_norm.weight", "ffn_post_norm.weight"] {
                let alias_suffix = if suffix == "attn_post_norm.weight" {
                    "post_attention_norm.weight"
                } else {
                    "post_ffw_norm.weight"
                };
                let name = if loader.tensors.contains_key(&format!("{prefix}{suffix}")) {
                    format!("{prefix}{suffix}")
                } else {
                    format!("{prefix}{alias_suffix}")
                };
                check_tensor_dims(&name, &[config.embed_dim])?;
            }
            // KV-shared layers may carry the double-wide MLP variant
            // (`use_double_wide_mlp`); the builder reads the real tensor
            // widths, so accept either architectural width there and require
            // gate/up/down to agree.
            let ffn_width = if is_shared {
                let doubled = config
                    .intermediate_dim
                    .checked_mul(2)
                    .ok_or_else(|| anyhow::anyhow!("Gemma 4 layer {layer} MLP width overflow"))?;
                let gate = tensor_dims(&format!("{prefix}ffn_gate.weight"))?;
                if gate == [config.embed_dim, doubled] {
                    doubled
                } else {
                    config.intermediate_dim
                }
            } else {
                config.intermediate_dim
            };
            for suffix in ["ffn_gate.weight", "ffn_up.weight"] {
                check_tensor_dims(&format!("{prefix}{suffix}"), &[config.embed_dim, ffn_width])?;
            }
            check_tensor_dims(
                &format!("{prefix}ffn_down.weight"),
                &[ffn_width, config.embed_dim],
            )?;
            let has_ple = loader.tensors.contains_key(&format!("{prefix}proj.weight"));
            if has_ple {
                check_tensor_dims(
                    &format!("{prefix}inp_gate.weight"),
                    &[config.embed_dim, per_layer_dim],
                )?;
                check_tensor_dims(
                    &format!("{prefix}proj.weight"),
                    &[per_layer_dim, config.embed_dim],
                )?;
                check_tensor_dims(&format!("{prefix}post_norm.weight"), &[config.embed_dim])?;
                check_tensor_dims(&format!("{prefix}layer_output_scale.weight"), &[1])?;
            }
        }

        // Optional head and global-PLE geometry: malformed optional tensors
        // must fail before any materialization, not after all blocks/RoPE
        // tables are built.
        if loader.tensors.contains_key("output.weight") {
            check_tensor_dims("output.weight", &[config.embed_dim, config.vocab_size])?;
        }
        if loader.tensors.contains_key("per_layer_model_proj.weight") {
            let global_ple_width = config
                .n_layers
                .checked_mul(per_layer_dim)
                .ok_or_else(|| anyhow::anyhow!("global PLE projection width overflow"))?;
            check_tensor_dims(
                "per_layer_model_proj.weight",
                &[config.embed_dim, global_ple_width],
            )?;
            check_tensor_dims("per_layer_proj_norm.weight", &[per_layer_dim])?;
        }
        // File loads carry GGUF-native dims in `tensor_meta`; the PLE
        // embedding is a 3D tensor, so the fallback path used by direct
        // unit-test loaders is not shape-consistent for it and stays covered
        // by `validate_loaded_shapes` after materialization.
        for name in [
            "per_layer_token_embd.weight",
            "token_embd_per_layer.weight",
            "per_layer_embd.weight",
        ] {
            if let Some(meta) = loader.tensor_meta.get(name) {
                // A packed Q8_0 PLE tensor is 2D in GGUF
                // ([n_layers * per_layer_dim, vocab]) and is validated after
                // materialization; only the 3D hidden form is checked here
                // before allocation.
                let packed_q8 = crate::loader::ggml_dtype_name(meta.dtype) == Some("q8_0");
                anyhow::ensure!(
                    packed_q8
                        || (meta.dims.len() == 3
                            && meta.dims[0] == config.n_layers
                            && meta.dims[1] >= config.vocab_size
                            && meta.dims[2] == per_layer_dim),
                    "tensor '{name}' must have shape [layers={}, vocab>={}, dim={}], got {:?}",
                    config.n_layers,
                    config.vocab_size,
                    per_layer_dim,
                    meta.dims
                );
            }
        }

        let local_rope = {
            // SWA (local) layers rotate with the plain SWA base frequency;
            // llama.cpp passes no freq_factors for them.
            let (cos, sin) = compute_rope_freqs(
                config.max_seq_len,
                config.local_head_dim,
                config.local_rope_theta,
                None,
            );
            (Arc::new(cos), Arc::new(sin))
        };
        let global_rope = {
            let (cos, sin) = compute_rope_freqs(
                config.max_seq_len,
                config.global_head_dim,
                config.global_rope_theta,
                rope_freqs.as_deref(),
            );
            (Arc::new(cos), Arc::new(sin))
        };

        let mut blocks = Vec::with_capacity(config.n_layers);
        let mut last_local_source = None;
        let mut last_global_source = None;
        // KV sharing starts at n_layers - num_kv_shared_layers regardless of
        // whether the checkpoint still carries per-layer k/v weights: HF
        // ignores those tensors for shared layers.
        for i in 0..config.n_layers {
            let layer_type = config.layer_type(i);
            let k_name = format!("blk.{}.attn_k.weight", i);
            let v_name = format!("blk.{}.attn_v.weight", i);
            let has_k = loader.tensors.contains_key(&k_name);
            let has_v = loader.tensors.contains_key(&v_name);
            anyhow::ensure!(
                has_k == has_v,
                "Gemma 4 layer {i} must provide both K and V projections or neither"
            );
            let has_own_kv = has_k;
            let is_shared = i >= first_shared_layer;
            anyhow::ensure!(
                is_shared || has_own_kv,
                "Gemma 4 non-shared layer {i} must provide both K and V projections"
            );
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
                q_proj: take_gemma4_linear(&mut loader, &format!("blk.{}.attn_q.weight", i))?,
                k_proj: if shared_source_layer.is_some() {
                    None
                } else {
                    Some(take_gemma4_linear(
                        &mut loader,
                        &format!("blk.{}.attn_k.weight", i),
                    )?)
                },
                v_proj: if shared_source_layer.is_some() {
                    None
                } else {
                    Some(take_gemma4_linear(
                        &mut loader,
                        &format!("blk.{}.attn_v.weight", i),
                    )?)
                },
                o_proj: take_gemma4_linear(&mut loader, &format!("blk.{}.attn_output.weight", i))?,
                q_norm: loader.take_f32(&format!("blk.{}.attn_q_norm.weight", i))?,
                k_norm: loader.take_f32(&format!("blk.{}.attn_k_norm.weight", i))?,
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

            let post_attn_norm = loader
                .take_optional_f32(&[
                    format!("blk.{}.attn_post_norm.weight", i),
                    format!("blk.{}.post_attention_norm.weight", i),
                ])?
                .ok_or_else(|| {
                    anyhow::anyhow!("Missing tensor: blk.{}.attn_post_norm.weight", i)
                })?;
            let post_ffn_norm = loader
                .take_optional_f32(&[
                    format!("blk.{}.ffn_post_norm.weight", i),
                    format!("blk.{}.post_ffw_norm.weight", i),
                ])?
                .ok_or_else(|| anyhow::anyhow!("Missing tensor: blk.{}.ffn_post_norm.weight", i))?;

            let proj_name = format!("blk.{}.proj.weight", i);
            let has_ple = loader.tensors.contains_key(&proj_name);
            let inp_gate = if has_ple {
                let linear =
                    take_gemma4_linear(&mut loader, &format!("blk.{}.inp_gate.weight", i))?;
                if linear.in_features(&CpuBackend) != config.embed_dim
                    || linear.out_features(&CpuBackend) != per_layer_dim
                {
                    anyhow::bail!(
                        "Gemma 4 layer {} input gate must be {} -> {}, got {} -> {}",
                        i,
                        config.embed_dim,
                        per_layer_dim,
                        linear.in_features(&CpuBackend),
                        linear.out_features(&CpuBackend)
                    );
                }
                Some(linear)
            } else {
                None
            };
            let ple_proj = if has_ple {
                let linear = take_gemma4_linear(&mut loader, &proj_name)?;
                if linear.in_features(&CpuBackend) != per_layer_dim
                    || linear.out_features(&CpuBackend) != config.embed_dim
                {
                    anyhow::bail!(
                        "Gemma 4 layer {} PLE projection must be {} -> {}, got {} -> {}",
                        i,
                        per_layer_dim,
                        config.embed_dim,
                        linear.in_features(&CpuBackend),
                        linear.out_features(&CpuBackend)
                    );
                }
                Some(linear)
            } else {
                None
            };
            let post_ple_norm = if has_ple {
                Some(loader.take_f32(&format!("blk.{}.post_norm.weight", i))?)
            } else {
                None
            };
            let layer_output_scale = if has_ple {
                Some(loader.take_f32(&format!("blk.{}.layer_output_scale.weight", i))?)
            } else {
                None
            };

            let gate_name = format!("blk.{i}.ffn_gate.weight");
            let up_name = format!("blk.{i}.ffn_up.weight");
            let mut gate_proj = take_gemma4_linear(&mut loader, &gate_name)?;
            let mut up_proj = take_gemma4_linear(&mut loader, &up_name)?;
            if pack_gate_up {
                gate_proj.prepare_packed_decode_cached(packed_cache, &gate_name);
                up_proj.prepare_packed_decode_cached(packed_cache, &up_name);
            }

            blocks.push(Gemma4Block {
                input_norm: loader.take_f32(&format!("blk.{}.attn_norm.weight", i))?,
                attn,
                inp_gate,
                ple_proj,
                post_ple_norm,
                layer_output_scale,
                post_attn_norm,
                pre_ffn_norm: loader.take_f32(&format!("blk.{}.ffn_norm.weight", i))?,
                mlp: Gemma4Mlp {
                    gate_proj,
                    up_proj,
                    down_proj: take_gemma4_linear(
                        &mut loader,
                        &format!("blk.{}.ffn_down.weight", i),
                    )?,
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
        let ple = match take_optional_f32_only(&mut loader, &ple_names) {
            Some(t) => {
                if t.shape().len() != 3
                    || t.shape()[0] != config.n_layers
                    || t.shape()[1] < config.vocab_size
                    || t.shape()[2] != per_layer_dim
                {
                    anyhow::bail!(
                        "Gemma 4 PLE tensor must have shape [layers={}, vocab>={}, dim={}], got {:?}",
                        config.n_layers,
                        config.vocab_size,
                        per_layer_dim,
                        t.shape()
                    );
                }
                Some(Gemma4Ple::Hidden(t))
            }
            None => {
                take_optional_q8(&mut loader, &ple_names).map(|embeddings| Gemma4Ple::PackedQ8 {
                    embeddings,
                    per_layer_dim,
                })
            }
        };
        if ple.is_none()
            && (config.vocab_size_per_layer_input.is_some()
                || config.hidden_size_per_layer_input.is_some())
        {
            log::warn!(
                "Gemma 4 PLE metadata is present but no supported packed PLE tensor was found"
            );
        }

        let head = match loader.tensors.remove("output.weight") {
            Some(LoadedTensor::F32(tensor)) => {
                anyhow::ensure!(
                    tensor.ndim() == 2,
                    "output.weight must be a 2D linear weight, got {:?}",
                    tensor.shape()
                );
                Gemma4Head::Linear(Linear::new(
                    crate::loader::gguf_to_row_major_f32(tensor),
                    None,
                ))
            }
            Some(LoadedTensor::Q8_0(weight)) => Gemma4Head::Linear(Linear::new_q8_0(weight, None)),
            Some(LoadedTensor::KQuant(_)) => {
                anyhow::bail!(NO_K_QUANT)
            }
            Some(LoadedTensor::Half(_)) => {
                anyhow::bail!(HALF_NOT_WIDENED)
            }
            None => Gemma4Head::TiedEmbedding(embed_tokens.clone()),
        };

        // Global PLE projection (llama.cpp pathway): combines hidden state
        // with per-layer token embeddings before the per-layer gate.
        let (per_layer_model_proj, per_layer_proj_norm) =
            match loader.tensors.remove("per_layer_model_proj.weight") {
                Some(t) => {
                    let proj = match t {
                        // BF16 arrives as loader F32 with GGUF dims; reorient
                        // like every other linear or the PLE context
                        // projection reads transposed bytes
                        LoadedTensor::F32(tensor) => {
                            anyhow::ensure!(
                                tensor.ndim() == 2,
                                "per_layer_model_proj.weight must be a 2D linear weight, got {:?}",
                                tensor.shape()
                            );
                            Linear::new(crate::loader::gguf_to_row_major_f32(tensor), None)
                        }
                        LoadedTensor::Q8_0(weight) => Linear::new_q8_0(weight, None),
                        LoadedTensor::KQuant(_) => {
                            anyhow::bail!(NO_K_QUANT)
                        }
                        LoadedTensor::Half(_) => {
                            anyhow::bail!(HALF_NOT_WIDENED)
                        }
                    };
                    let norm = loader.take_f32("per_layer_proj_norm.weight")?;
                    (Some(proj), Some(norm))
                }
                _ => (None, None),
            };

        let model = Self {
            embed_tokens,
            blocks,
            norm: loader.take_f32("output_norm.weight")?,
            head,
            ple,
            per_layer_model_proj,
            per_layer_proj_norm,
            config,
        };
        model.validate_loaded_shapes()?;
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

    fn validate_loaded_shapes(&self) -> anyhow::Result<()> {
        let backend = CpuBackend;
        let (vocab_size, embed_dim) = match &self.embed_tokens {
            Gemma4Embedding::F32(table) => {
                anyhow::ensure!(table.shape().len() == 2, "token embedding must be 2D");
                (table.shape()[0], table.shape()[1])
            }
            Gemma4Embedding::Q8_0(table) => (table.out_features(), table.in_features()),
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
        let head_shape = match &self.head {
            Gemma4Head::Linear(head) => (head.in_features(&backend), head.out_features(&backend)),
            Gemma4Head::TiedEmbedding(Gemma4Embedding::F32(table)) => {
                (table.shape()[1], table.shape()[0])
            }
            Gemma4Head::TiedEmbedding(Gemma4Embedding::Q8_0(table)) => {
                (table.in_features(), table.out_features())
            }
        };
        anyhow::ensure!(
            head_shape == (self.config.embed_dim, self.config.vocab_size),
            "output head must be {} -> {}, got {} -> {}",
            self.config.embed_dim,
            self.config.vocab_size,
            head_shape.0,
            head_shape.1
        );
        let per_layer_dim = if let Some(ple) = &self.ple {
            match ple {
                Gemma4Ple::Hidden(tensor) => {
                    let shape = tensor.shape();
                    anyhow::ensure!(
                        shape.len() == 3
                            && shape[0] == self.config.n_layers
                            && shape[1] >= self.config.vocab_size
                            && shape[2] > 0,
                        "PLE tensor must have shape [layers={}, vocab>={}, dim>0], got {:?}",
                        self.config.n_layers,
                        self.config.vocab_size,
                        shape
                    );
                    shape[2]
                }
                Gemma4Ple::PackedQ8 {
                    embeddings,
                    per_layer_dim,
                } => {
                    let expected = self
                        .config
                        .n_layers
                        .checked_mul(*per_layer_dim)
                        .ok_or_else(|| anyhow::anyhow!("PLE packed width overflow"))?;
                    anyhow::ensure!(
                        embeddings.out_features() >= self.config.vocab_size
                            && embeddings.in_features() == expected,
                        "packed PLE must have at least {} vocab rows and width {}, got {} rows x {}",
                        self.config.vocab_size,
                        expected,
                        embeddings.out_features(),
                        embeddings.in_features()
                    );
                    *per_layer_dim
                }
            }
        } else {
            0
        };
        match (
            &self.per_layer_model_proj,
            &self.per_layer_proj_norm,
            self.ple.is_some(),
        ) {
            (Some(proj), Some(norm), true) => {
                let expected = self
                    .config
                    .n_layers
                    .checked_mul(per_layer_dim)
                    .ok_or_else(|| anyhow::anyhow!("global PLE projection width overflow"))?;
                anyhow::ensure!(
                    proj.in_features(&backend) == self.config.embed_dim
                        && proj.out_features(&backend) == expected,
                    "global PLE projection must be {} -> {}, got {} -> {}",
                    self.config.embed_dim,
                    expected,
                    proj.in_features(&backend),
                    proj.out_features(&backend)
                );
                anyhow::ensure!(
                    norm.shape() == [per_layer_dim],
                    "global PLE projection norm must have shape [{}], got {:?}",
                    per_layer_dim,
                    norm.shape()
                );
            }
            (None, None, false) => {}
            (Some(_), _, false) | (_, Some(_), false) => {
                anyhow::bail!("global PLE projection requires a PLE tensor")
            }
            _ => anyhow::bail!("global PLE projection and norm must be provided together"),
        }

        for (index, block) in self.blocks.iter().enumerate() {
            for (name, norm) in [
                ("attention", &block.input_norm),
                ("post-attention", &block.post_attn_norm),
                ("FFN", &block.pre_ffn_norm),
                ("post-FFN", &block.post_ffn_norm),
            ] {
                anyhow::ensure!(
                    norm.shape() == [self.config.embed_dim],
                    "block {index} {name} norm must have shape [{}], got {:?}",
                    self.config.embed_dim,
                    norm.shape()
                );
            }
            let attention = &block.attn;
            let query_width = attention
                .n_heads
                .checked_mul(attention.head_dim)
                .ok_or_else(|| anyhow::anyhow!("block {index} query width overflow"))?;
            let kv_width = attention
                .n_kv_heads
                .checked_mul(attention.head_dim)
                .ok_or_else(|| anyhow::anyhow!("block {index} KV width overflow"))?;
            anyhow::ensure!(
                attention.q_proj.in_features(&backend) == self.config.embed_dim
                    && attention.q_proj.out_features(&backend) == query_width,
                "block {index} query projection must be {} -> {query_width}",
                self.config.embed_dim
            );
            for (name, projection) in [
                ("key", attention.k_proj.as_ref()),
                ("value", attention.v_proj.as_ref()),
            ] {
                if let Some(projection) = projection {
                    anyhow::ensure!(
                        projection.in_features(&backend) == self.config.embed_dim
                            && projection.out_features(&backend) == kv_width,
                        "block {index} {name} projection must be {} -> {kv_width}",
                        self.config.embed_dim
                    );
                }
            }
            anyhow::ensure!(
                attention.o_proj.in_features(&backend) == query_width
                    && attention.o_proj.out_features(&backend) == self.config.embed_dim,
                "block {index} attention output projection must be {query_width} -> {}",
                self.config.embed_dim
            );
            anyhow::ensure!(
                attention.q_norm.shape() == [attention.head_dim]
                    && attention.k_norm.shape() == [attention.head_dim],
                "block {index} Q/K norms must have shape [{}]",
                attention.head_dim
            );
            let intermediate = block.mlp.gate_proj.out_features(&backend);
            anyhow::ensure!(
                intermediate > 0
                    && block.mlp.gate_proj.in_features(&backend) == self.config.embed_dim
                    && block.mlp.up_proj.in_features(&backend) == self.config.embed_dim
                    && block.mlp.up_proj.out_features(&backend) == intermediate
                    && block.mlp.down_proj.in_features(&backend) == intermediate
                    && block.mlp.down_proj.out_features(&backend) == self.config.embed_dim,
                "block {index} MLP projection shapes are inconsistent"
            );
            if let Some(norm) = &block.post_ple_norm {
                anyhow::ensure!(
                    norm.shape() == [self.config.embed_dim],
                    "block {index} post-PLE norm must have shape [{}]",
                    self.config.embed_dim
                );
            }
            if self.ple.is_some() {
                let inp_gate = block.inp_gate.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("block {index} is missing its PLE input gate")
                })?;
                let ple_proj = block.ple_proj.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("block {index} is missing its PLE projection")
                })?;
                anyhow::ensure!(
                    block.post_ple_norm.is_some(),
                    "block {index} is missing its post-PLE norm"
                );
                anyhow::ensure!(
                    inp_gate.in_features(&backend) == self.config.embed_dim
                        && inp_gate.out_features(&backend) == per_layer_dim,
                    "block {index} PLE input gate must be {} -> {}",
                    self.config.embed_dim,
                    per_layer_dim
                );
                anyhow::ensure!(
                    ple_proj.in_features(&backend) == per_layer_dim
                        && ple_proj.out_features(&backend) == self.config.embed_dim,
                    "block {index} PLE projection must be {} -> {}",
                    per_layer_dim,
                    self.config.embed_dim
                );
            } else {
                anyhow::ensure!(
                    block.inp_gate.is_none()
                        && block.ple_proj.is_none()
                        && block.post_ple_norm.is_none(),
                    "block {index} has PLE block tensors without a PLE embedding"
                );
            }
            if let Some(scale) = &block.layer_output_scale {
                anyhow::ensure!(
                    scale.shape() == [1],
                    "block {index} layer output scale must have shape [1], got {:?}",
                    scale.shape()
                );
            }
        }
        Ok(())
    }
}

pub(super) fn take_gemma4_linear(
    loader: &mut GgufLoader,
    name: &str,
) -> anyhow::Result<Linear<CpuBackend>> {
    match loader.take_tensor(name)? {
        LoadedTensor::F32(tensor) => {
            anyhow::ensure!(
                tensor.ndim() == 2,
                "{name} must be a 2D linear weight, got {:?}",
                tensor.shape()
            );
            Ok(Linear::new(
                crate::loader::gguf_to_row_major_f32(tensor),
                None,
            ))
        }
        LoadedTensor::Q8_0(weight) => Ok(Linear::new_q8_0(weight, None)),
        LoadedTensor::KQuant(_) => {
            anyhow::bail!(NO_K_QUANT)
        }
        LoadedTensor::Half(_) => {
            anyhow::bail!(HALF_NOT_WIDENED)
        }
    }
}

fn take_optional_f32_only(loader: &mut GgufLoader, names: &[String]) -> Option<CpuTensor> {
    let name = names.iter().find(|name| {
        matches!(
            loader.tensors.get(name.as_str()),
            Some(LoadedTensor::F32(_))
        )
    })?;
    match loader.tensors.remove(name.as_str())? {
        LoadedTensor::F32(tensor) => Some(tensor),
        LoadedTensor::Q8_0(_) => unreachable!("tensor kind checked before removal"),
        LoadedTensor::KQuant(_) => unreachable!("tensor kind checked before removal"),
        LoadedTensor::Half(_) => unreachable!("tensor kind checked before removal"),
    }
}

fn take_optional_q8(
    loader: &mut GgufLoader,
    names: &[String],
) -> Option<crate::quant::QuantizedWeight> {
    let name = names.iter().find(|name| {
        matches!(
            loader.tensors.get(name.as_str()),
            Some(LoadedTensor::Q8_0(_))
        )
    })?;
    match loader.tensors.remove(name.as_str())? {
        LoadedTensor::Q8_0(weight) => Some(weight),
        LoadedTensor::F32(_) => unreachable!("tensor kind checked before removal"),
        LoadedTensor::KQuant(_) => unreachable!("tensor kind checked before removal"),
        LoadedTensor::Half(_) => unreachable!("tensor kind checked before removal"),
    }
}
