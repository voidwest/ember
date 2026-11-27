use super::*;

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------

impl MmsVits {
    pub fn from_gguf(path: &std::path::Path) -> Result<Self> {
        let mut loader = load_gguf(path).with_context(|| format!("loading {}", path.display()))?;
        // clone the small metadata values out first (take_f32 needs &mut)
        let get_u32 = |l: &crate::loader::GgufLoader, k: &str| -> Result<usize> {
            match l.metadata.get(k) {
                Some(GgufValue::U32(v)) => Ok(*v as usize),
                Some(o) => anyhow::bail!("{k} must be U32, got {o:?}"),
                None => anyhow::bail!("vits gguf missing metadata {k}"),
            }
        };
        let get_f32 = |l: &crate::loader::GgufLoader, k: &str| -> Result<f32> {
            match l.metadata.get(k) {
                Some(GgufValue::F32(v)) => Ok(*v),
                Some(o) => anyhow::bail!("{k} must be F32, got {o:?}"),
                None => anyhow::bail!("vits gguf missing metadata {k}"),
            }
        };
        let config = VitsConfig {
            vocab_size: get_u32(&loader, "vits.vocab_size")?,
            hidden_size: get_u32(&loader, "vits.hidden_size")?,
            num_layers: get_u32(&loader, "vits.num_layers")?,
            num_heads: get_u32(&loader, "vits.num_heads")?,
            window_size: get_u32(&loader, "vits.window_size")?,
            ffn_dim: get_u32(&loader, "vits.ffn_dim")?,
            ffn_kernel_size: get_u32(&loader, "vits.ffn_kernel_size")?,
            flow_size: get_u32(&loader, "vits.flow_size")?,
            wavenet_layers: get_u32(&loader, "vits.wavenet_layers")?,
            prior_flows: get_u32(&loader, "vits.prior_flows")?,
            dp_dds_layers: get_u32(&loader, "vits.dp_dds_layers")?,
            dp_flows: get_u32(&loader, "vits.dp_flows")?,
            dp_bins: get_u32(&loader, "vits.dp_bins")?,
            dp_tail_bound: get_f32(&loader, "vits.dp_tail_bound")?,
            sample_rate: get_u32(&loader, "vits.sample_rate")? as u32,
            hop_length: get_u32(&loader, "vits.hop_length")?,
            leaky_relu_slope: get_f32(&loader, "vits.leaky_relu_slope")?,
            ln_eps: get_f32(&loader, "vits.ln_eps")?,
        };
        config.validate()?;

        fn take_vec(l: &mut crate::loader::GgufLoader, name: &str) -> Result<CpuTensor> {
            l.take_f32(name)
                .with_context(|| format!("vits gguf missing tensor {name}"))
        }
        fn take_flat(l: &mut crate::loader::GgufLoader, name: &str) -> Result<Vec<f32>> {
            Ok(take_vec(l, name)?.data().to_vec())
        }
        fn take_lin(l: &mut crate::loader::GgufLoader, name: &str) -> Result<Lin> {
            // HF layout [out, in] (or conv [out,in,1]); materialize the REAL
            // transposed buffer [in, out] (label-only relabeling would be
            // wrong even for square matrices — this bug cost an hour).
            let w_hf = gguf_to_hf(&take_vec(l, &format!("{name}.w"))?);
            let (o, i) = match w_hf.shape().len() {
                3 if w_hf.shape()[2] == 1 => (w_hf.shape()[0], w_hf.shape()[1]),
                2 => (w_hf.shape()[0], w_hf.shape()[1]),
                s => anyhow::bail!("{name}: expected [out,in(,1)], got {s:?}"),
            };
            let b = take_flat(l, &format!("{name}.b"))?;
            ensure!(b.len() == o, "{name}: bias {} != out {o}", b.len());
            let mut tw = vec![0.0f32; i * o];
            for oo in 0..o {
                for ii in 0..i {
                    tw[ii * o + oo] = w_hf.data()[oo * i + ii];
                }
            }
            Ok(Lin {
                w: CpuTensor::from_data(vec![i, o], tw),
                b,
            })
        }
        fn take_dense(l: &mut crate::loader::GgufLoader, name: &str) -> Result<DenseConv1d> {
            let w = gguf_to_hf(&take_vec(l, &format!("{name}.w"))?);
            ensure!(
                w.shape().len() == 3,
                "{name}: expected conv [out, in, k], got {:?}",
                w.shape()
            );
            let b = take_flat(l, &format!("{name}.b"))?;
            ensure!(
                b.len() == w.shape()[0],
                "{name}: bias {} != out {}",
                b.len(),
                w.shape()[0]
            );
            Ok(DenseConv1d::from_hf_weight(&w, b))
        }

        // text encoder ------------------------------------------------------
        let embed = take_flat(&mut loader, "v.embed")?;
        ensure!(
            Some(embed.len()) == config.vocab_size.checked_mul(config.hidden_size),
            "vits embedding has {} values, expected {} x {}",
            embed.len(),
            config.vocab_size,
            config.hidden_size
        );
        let hd = config.head_dim();
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let g = format!("v.layer.{i}");
            layers.push(EncoderLayer {
                q: take_lin(&mut loader, &format!("{g}.attn.q"))?,
                k: take_lin(&mut loader, &format!("{g}.attn.k"))?,
                v: take_lin(&mut loader, &format!("{g}.attn.v"))?,
                o: take_lin(&mut loader, &format!("{g}.attn.o"))?,
                rel_k: take_flat(&mut loader, &format!("{g}.rel_k"))?,
                rel_v: take_flat(&mut loader, &format!("{g}.rel_v"))?,
                ffn1: take_dense(&mut loader, &format!("{g}.ffn1"))?,
                ffn2: take_dense(&mut loader, &format!("{g}.ffn2"))?,
                ln1_w: take_flat(&mut loader, &format!("{g}.ln1.w"))?,
                ln1_b: take_flat(&mut loader, &format!("{g}.ln1.b"))?,
                ln2_w: take_flat(&mut loader, &format!("{g}.ln2.w"))?,
                ln2_b: take_flat(&mut loader, &format!("{g}.ln2.b"))?,
            });
        }
        debug_assert_eq!(layers[0].rel_k.len(), (2 * config.window_size + 1) * hd);
        let project = take_dense(&mut loader, "v.project")?;

        // SDP ---------------------------------------------------------------
        let dds = Self::take_dds(
            &mut loader,
            "v.sdp.dds",
            config.dp_dds_layers,
            config.hidden_size,
        )?;
        let mut affine = None;
        let mut conv_flows = Vec::new();
        for j in 0..=config.dp_flows {
            if j == 0 {
                affine = Some((
                    take_flat(&mut loader, "v.sdp.flow0.translate")?,
                    take_flat(&mut loader, "v.sdp.flow0.log_scale")?,
                ));
            } else {
                let dds_j = Self::take_dds(
                    &mut loader,
                    &format!("v.sdp.flow{j}.dds"),
                    config.dp_dds_layers,
                    config.hidden_size,
                )?;
                conv_flows.push(ConvFlow {
                    conv_pre: take_dense(&mut loader, &format!("v.sdp.flow{j}.conv_pre"))?,
                    conv_proj: take_dense(&mut loader, &format!("v.sdp.flow{j}.conv_proj"))?,
                    dds: dds_j,
                });
            }
        }
        let sdp = Sdp {
            conv_pre: take_dense(&mut loader, "v.sdp.conv_pre")?,
            conv_proj: take_dense(&mut loader, "v.sdp.conv_proj")?,
            dds,
            affine,
            conv_flows,
        };

        // prior flow ---------------------------------------------------------
        let mut flows = Vec::with_capacity(config.prior_flows);
        for j in 0..config.prior_flows {
            let g = format!("v.flow{j}");
            let mut wn_in = Vec::with_capacity(config.wavenet_layers);
            let mut wn_rs = Vec::with_capacity(config.wavenet_layers);
            for k in 0..config.wavenet_layers {
                wn_in.push(take_dense(&mut loader, &format!("{g}.wn.in{k}"))?);
                wn_rs.push(take_dense(&mut loader, &format!("{g}.wn.rs{k}"))?);
            }
            flows.push(FlowLayer {
                conv_pre: take_dense(&mut loader, &format!("{g}.conv_pre"))?,
                conv_post: take_dense(&mut loader, &format!("{g}.conv_post"))?,
                wn_in,
                wn_rs,
            });
        }

        // HiFi-GAN -------------------------------------------------------------
        let conv_pre = take_dense(&mut loader, "v.hifigan.conv_pre")?;
        const UPS_RATES: [usize; 4] = [8, 8, 2, 2];
        const UPS_KERNELS: [usize; 4] = [16, 16, 4, 4];
        let mut ups = Vec::with_capacity(UPS_RATES.len());
        for i in 0..UPS_RATES.len() {
            let w = gguf_to_hf(&take_vec(&mut loader, &format!("v.up{i}.w"))?);
            ensure!(w.shape().len() == 3, "up weight must be 3D");
            ups.push(ConvTranspose1d {
                stride: UPS_RATES[i],
                padding: (UPS_KERNELS[i] - UPS_RATES[i]) / 2,
                b: take_flat(&mut loader, &format!("v.up{i}.b"))?,
                w,
            });
        }
        let nblk = 3;
        let ndil = 3;
        let mut resblocks = Vec::with_capacity(ups.len() * nblk);
        for stage in 0..ups.len() {
            for blk in 0..nblk {
                let g = format!("v.rb{stage}{blk}");
                let mut c1 = Vec::with_capacity(ndil);
                let mut c2 = Vec::with_capacity(ndil);
                for k in 0..ndil {
                    c1.push(take_dense(&mut loader, &format!("{g}.c1{k}"))?);
                    c2.push(take_dense(&mut loader, &format!("{g}.c2{k}"))?);
                }
                resblocks.push(ResBlock { c1, c2 });
            }
        }
        let hifigan = HifiGan {
            conv_pre,
            ups,
            resblocks,
            conv_post_w: take_vec(&mut loader, "v.hifigan.conv_post.w")?,
        };

        // vocab -----------------------------------------------------------------
        let vocab_str = match loader.metadata.get("vits.vocab") {
            Some(GgufValue::Str(s)) => s.clone(),
            // gguf-py add_array(String) -> ARRAY of STR
            Some(GgufValue::Array(items)) => {
                let mut joined = String::new();
                for it in items {
                    match it {
                        GgufValue::Str(s) => {
                            joined.push_str(s);
                            joined.push('\n');
                        }
                        other => anyhow::bail!("vits.vocab array holds {other:?}"),
                    }
                }
                joined
            }
            other => anyhow::bail!(
                "vits.vocab missing or unsupported ({})",
                if other.is_some() { "type" } else { "absent" }
            ),
        };
        let mut char_to_id = std::collections::HashMap::new();
        for (i, line) in vocab_str
            .split('\n')
            .filter(|l: &&str| !l.is_empty())
            .enumerate()
        {
            if let Some(ch) = line.chars().next() {
                char_to_id.entry(ch).or_insert(i as u32);
            }
        }
        ensure!(
            char_to_id.len() >= 20,
            "suspiciously small vocab parsed ({})",
            char_to_id.len()
        );
        // token ids index the embedding rows directly
        ensure!(
            char_to_id
                .values()
                .all(|&id| (id as usize) < config.vocab_size),
            "vits.vocab lists more entries than the {} embedding rows",
            config.vocab_size
        );
        // Declared pad_token (VitsTokenizer added token). Older GGUFs lack
        // the metadata; fall back to the first vocab entry, which is what
        // the converter now records and what mms checkpoints ship as pad.
        let pad_token = match loader.metadata.get("vits.pad_token") {
            Some(GgufValue::Str(s)) => s.clone(),
            _ => vocab_str.split('\n').next().unwrap_or_default().to_string(),
        };

        Ok(Self {
            config,
            embed,
            layers,
            project,
            sdp,
            flows,
            hifigan,
            char_to_id,
            pad_token,
        })
    }

    pub(super) fn take_dds(
        l: &mut crate::loader::GgufLoader,
        prefix: &str,
        n: usize,
        channels: usize,
    ) -> Result<DdsConv> {
        let mut dilated_w = Vec::with_capacity(n);
        let mut dilated_b = Vec::with_capacity(n);
        let mut pointwise = Vec::with_capacity(n);
        let mut ln1 = Vec::with_capacity(n);
        let mut ln2 = Vec::with_capacity(n);
        for j in 0..n {
            let dw = gguf_to_hf(&l.take_f32(&format!("{prefix}.d{j}.w"))?);
            ensure!(
                dw.shape()[0] == channels && dw.shape()[1] == 1,
                "dds depthwise shape {:?}",
                dw.shape()
            );
            dilated_w.push(dw);
            dilated_b.push(l.take_f32(&format!("{prefix}.d{j}.b"))?.data().to_vec());
            let pw_hf = gguf_to_hf(&l.take_f32(&format!("{prefix}.p{j}.w"))?);
            let (o, i) = (pw_hf.shape()[0], pw_hf.shape()[1]);
            let b = l.take_f32(&format!("{prefix}.p{j}.b"))?.data().to_vec();
            ensure!(b.len() == o);
            let mut tw = vec![0.0f32; i * o];
            for oo in 0..o {
                for ii in 0..i {
                    tw[ii * o + oo] = pw_hf.data()[oo * i + ii];
                }
            }
            pointwise.push(Lin {
                w: CpuTensor::from_data(vec![i, o], tw),
                b,
            });
            ln1.push((
                l.take_f32(&format!("{prefix}.ln1_{j}.w"))?.data().to_vec(),
                l.take_f32(&format!("{prefix}.ln1_{j}.b"))?.data().to_vec(),
            ));
            ln2.push((
                l.take_f32(&format!("{prefix}.ln2_{j}.w"))?.data().to_vec(),
                l.take_f32(&format!("{prefix}.ln2_{j}.b"))?.data().to_vec(),
            ));
        }
        Ok(DdsConv {
            dilated_w,
            dilated_b,
            pointwise,
            ln1,
            ln2,
        })
    }
}

// ---------------------------------------------------------------------------
// frontend: reference VitsTokenizer behavior for script vocabs
// ---------------------------------------------------------------------------

impl MmsVits {
    /// Transformers-faithful VitsTokenizer pipeline (phonemize=false,
    /// normalize=true, add_blank=true):
    ///
    /// 1. `normalize_text`: keep vocab characters verbatim, lowercase the rest.
    /// 2. drop characters outside the vocab, then `.strip()` both ends
    ///    (reference does this in one filter+strip step).
    /// 3. split on the declared pad_token — it is an ADDED token, so every
    ///    occurrence is emitted as a BARE id (its own vocab id, from
    ///    `added_tokens_encoder`) with NO surrounding blanks; remaining
    ///    segments are blank-interleaved `[0, c1, 0, ..., cn, 0]`.
    ///
    /// For mms-tts-ara pad_token='ا' (id 0), so alefs never get blank frames:
    /// e.g. "السلام" -> `0(ا bare)` + `0 ل 0 س 0 ل` + `0(ا bare)` + ...
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        // stage 1+2: normalize then filter to vocab chars, strip ends.
        // (A pad_token whose characters are not all in-vocab cannot survive
        // this filter, mirroring the reference order: filter runs BEFORE the
        // added-token split.)
        let filtered: String = text
            .chars()
            .map(|c| {
                if self.char_to_id.contains_key(&c) {
                    c
                } else {
                    c.to_lowercase().next().unwrap_or(c)
                }
            })
            .filter(|c| self.char_to_id.contains_key(c))
            .collect();
        let stripped = filtered.trim_matches(|c: char| c.is_whitespace());
        let mut ids = Vec::with_capacity(stripped.len() * 2 + 1);
        if stripped.is_empty() {
            return ids;
        }

        // the bare id an occurrence of the pad token resolves to
        let pad_id = self
            .pad_token
            .chars()
            .next()
            .and_then(|c| self.char_to_id.get(&c).copied())
            .unwrap_or(0);

        let push_segment = |ids: &mut Vec<u32>, seg: &str| {
            if seg.is_empty() {
                return;
            }
            ids.push(0u32);
            for c in seg.chars() {
                ids.push(*self.char_to_id.get(&c).expect("filtered char"));
                ids.push(0u32);
            }
        };

        let mut rest = stripped;
        loop {
            match rest.find(self.pad_token.as_str()) {
                Some(pos) => {
                    push_segment(&mut ids, &rest[..pos]);
                    ids.push(pad_id); // bare added-token emission, no blanks
                    rest = &rest[pos + self.pad_token.len()..];
                }
                None => {
                    push_segment(&mut ids, rest);
                    break;
                }
            }
        }
        ids
    }
}
