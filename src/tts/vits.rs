//! MMS-TTS (facebook/mms-tts-*, VITS architecture): Arabic text -> PCM.
//!
//! Phase 5 Session 2 Track C: the first genuinely Arabic-capable speech
//! output path. Selection rationale lives in
//! `SPEECH_ARCHITECTURE_SURVEY.md` §"Session 2 decision": character-level
//! vocab over raw Arabic script (no uroman, no espeak G2P), single-speaker,
//! 16 kHz output, CC-BY-NC-4.0.
//!
//! Inference path (transformers-faithful, deterministic):
//!
//! ```text
//! ids [T] -> embed * sqrt(H) -> 6 x {rel-pos attention + conv FFN}
//!         -> project k1 -> prior_means / prior_logvars
//!         -> SDP reverse (noise_scale_duration = 0)
//!         -> durations = ceil(exp(log_d)) -> monotonic expansion
//!         -> prior latents (= expanded means; noise_scale = 0)
//!         -> flow reverse x4 (conv-pre -> WaveNet x4 -> conv-post)
//!         -> HiFi-GAN (ups x[8,8,2,2] + resblocks {3,7,11}) -> tanh PCM
//! ```
//!
//! Determinism contract: the reference ladder (`scripts/ref_vits.py`)
//! zeroes `noise_scale`/`noise_scale_duration` so every stochastic draw
//! contributes zero and both engines compute identical math — parity is a
//! pure numeric gate. Ember bakes zero noise for the same reason OuteTTS
//! uses greedy decoding: reproducibility over prosody sampling.
//!
//! Streaming (Track C6): unlike WavTokenizer's GLOBAL-attention codec, the
//! VITS decoder is purely convolutional with a bounded receptive field, so
//! chunked mel-frame decode yields STABLE waveform chunks: every emitted
//! sample is a function of a bounded mel window only, and the mel frames
//! are all final before decoding starts. Emitted PCM never changes —
//! genuine streaming stability (`stable_up_to == chunk end` always).

use crate::backend::CpuBackend;
use crate::loader::{load_gguf, GgufValue};
use crate::tensor::CpuTensor;
use crate::tts::outetts::{AudioChunkMeta, TtsTimings};
use crate::tts::wavtokenizer::{conv1d_dense, gguf_to_hf, DenseConv1d};
use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use std::time::Instant;

// ---------------------------------------------------------------------------
// config + model structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VitsConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub window_size: usize,
    pub ffn_dim: usize,
    pub ffn_kernel_size: usize,
    pub flow_size: usize,
    pub wavenet_layers: usize,
    pub prior_flows: usize,
    pub dp_dds_layers: usize,
    pub dp_flows: usize,
    pub dp_bins: usize,
    pub dp_tail_bound: f32,
    pub sample_rate: u32,
    pub hop_length: usize,
    pub leaky_relu_slope: f32,
    pub ln_eps: f32,
}

impl VitsConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
    /// DDSConv kernel size (duration_predictor_kernel_size = 3 for MMS).
    #[allow(dead_code)]
    fn dp_kernel(&self) -> usize {
        3
    }
}

/// Dense projection stored [in, out] row-major with bias.
struct Lin {
    w: CpuTensor,
    b: Vec<f32>,
}

impl Lin {
    fn apply(&self, x_row_major: &[f32], t: usize) -> Vec<f32> {
        let k_in = self.w.shape()[0];
        let x = CpuTensor::from_data(vec![t, k_in], x_row_major.to_vec());
        let mut out = x.par_matmul(&self.w);
        let o = self.b.len();
        {
            let data = out.data_mut();
            for r in 0..t {
                for (c, bv) in self.b.iter().enumerate().take(o) {
                    data[r * o + c] += bv;
                }
            }
        }
        out.data().to_vec()
    }
}

struct EncoderLayer {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    /// [2W+1, hd]
    rel_k: Vec<f32>,
    rel_v: Vec<f32>,
    ffn1: DenseConv1d,
    ffn2: DenseConv1d,
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
}

/// DilatedDepthSeparableConv stack (depthwise-dilated + pointwise + LN/gelu).
struct DdsConv {
    /// per-layer torch-layout weights [C, 1, k]
    dilated_w: Vec<CpuTensor>,
    dilated_b: Vec<Vec<f32>>,
    pointwise: Vec<Lin>,
    ln1: Vec<(Vec<f32>, Vec<f32>)>,
    ln2: Vec<(Vec<f32>, Vec<f32>)>,
}

struct ConvFlow {
    conv_pre: DenseConv1d,
    conv_proj: DenseConv1d,
    dds: DdsConv,
}

struct Sdp {
    conv_pre: DenseConv1d,
    conv_proj: DenseConv1d,
    dds: DdsConv,
    /// flows[0] ElementwiseAffine (translate, log_scale)
    affine: Option<(Vec<f32>, Vec<f32>)>,
    conv_flows: Vec<ConvFlow>,
}

struct FlowLayer {
    conv_pre: DenseConv1d,
    conv_post: DenseConv1d,
    wn_in: Vec<DenseConv1d>,
    wn_rs: Vec<DenseConv1d>,
}

struct ResBlock {
    c1: Vec<DenseConv1d>,
    c2: Vec<DenseConv1d>,
}

struct ConvTranspose1d {
    /// [c_in, c_out, k]
    w: CpuTensor,
    b: Vec<f32>,
    stride: usize,
    padding: usize,
}

struct HifiGan {
    conv_pre: DenseConv1d,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<ResBlock>,
    /// [1, C_last, 7]
    conv_post_w: CpuTensor,
}

pub struct MmsVits {
    pub config: VitsConfig,
    embed: Vec<f32>, // [V, H]
    layers: Vec<EncoderLayer>,
    project: DenseConv1d,
    sdp: Sdp,
    flows: Vec<FlowLayer>,
    hifigan: HifiGan,
    char_to_id: std::collections::HashMap<char, u32>,
    /// Declared tokenizer pad_token (an ADDED token for VitsTokenizer):
    /// text is split on every occurrence; each occurrence emits its bare id
    /// with NO surrounding blank frames. For mms-tts-ara this is 'ا' (id 0).
    pad_token: String,
}

/// Progressive-validation intermediates (ladder mirror of ref_vits.py).
#[derive(Debug, Default)]
pub struct VitsTrace {
    pub ids: Option<Vec<u32>>,
    pub embed_scaled: Option<Vec<f32>>,    // [T*H]
    pub encoder_out: Option<Vec<f32>>,     // [T*H]
    pub prior_means: Option<Vec<f32>>,     // [T*F]
    pub log_duration: Option<Vec<f32>>,    // [T]
    pub expanded_hidden: Option<Vec<f32>>, // [S*H]
    pub flow_z: Option<Vec<f32>>,          // [F*S]
    pub spectrogram: Option<Vec<f32>>,     // [F*S]
    pub waveform: Option<Vec<f32>>,        // [N]
}

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
            let b = take_flat(l, &format!("{name}.b"))?;
            Ok(DenseConv1d::from_hf_weight(&w, b))
        }

        // text encoder ------------------------------------------------------
        let embed = take_flat(&mut loader, "v.embed")?;
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

    fn take_dds(
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

// ---------------------------------------------------------------------------
// math primitives (torch-faithful)
// ---------------------------------------------------------------------------

fn layer_norm(x: &mut [f32], t: usize, dim: usize, w: &[f32], b: &[f32], eps: f32) {
    for row in 0..t {
        let r = &mut x[row * dim..(row + 1) * dim];
        let mean = r.iter().sum::<f32>() / dim as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in r.iter_mut().enumerate() {
            *v = (*v - mean) * inv * w[i] + b[i];
        }
    }
}

/// transformers' default gelu (exact erf form).
fn gelu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2));
    }
}

fn erf(x: f32) -> f32 {
    libm::erf(x as f64) as f32
}

fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Depthwise dilated Conv1d over `[C, T]` (kernel 3, dilation `d`,
/// symmetric pad `d`). Output `[C, T']` with T' = T + d (same-length when
/// the reference pads by (k*d−d)//2 = d).
#[allow(clippy::needless_range_loop)]
fn depthwise_dilated(
    input: &[f32],
    c: usize,
    t: usize,
    w_taps: &[f32],
    b: &[f32],
    d: usize,
) -> Vec<f32> {
    let k = 3usize;
    let pad = d;
    let t_out = t + 2 * pad - d * (k - 1);
    let _ = pad; // symmetric pad == d by construction (k=3)
    let mut out = vec![0.0f32; c * t_out];
    out.par_chunks_mut(t_out)
        .enumerate()
        .for_each(|(ci, out_c)| {
            let wc = &w_taps[ci * k..(ci + 1) * k];
            let xc = &input[ci * t..(ci + 1) * t];
            for ti in 0..t_out {
                // stride-1 window centered at ti: taps at ti-d, ti, ti+d
                let mut acc = b[ci];
                for j in 0..k {
                    let src = ti as isize + (j as isize) * (d as isize) - d as isize;
                    if src >= 0 && (src as usize) < t {
                        acc += xc[src as usize] * wc[j];
                    }
                }
                out_c[ti] = acc;
            }
        });
    out
}

/// Transpose `[C,T]` channel-major to row-major `[T,C]`.
fn ct_to_rows(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * t];
    for ci in 0..c {
        for ti in 0..t {
            out[ti * c + ci] = x[ci * t + ti];
        }
    }
    out
}

/// Transpose row-major `[T,C]` back to channel-major `[C,T]`.
fn rows_to_ct(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * t];
    for ti in 0..t {
        for ci in 0..c {
            out[ci * t + ti] = x[ti * c + ci];
        }
    }
    out
}

/// groups=1 DILATED Conv1d via im2col over `[C,T]`, symmetric pad `pad`,
/// dilation `dil`. Output length = T + 2*pad - dil*(k-1).
fn conv_dense_dilated(input: &CpuTensor, c: &DenseConv1d, pad: usize, dil: usize) -> CpuTensor {
    let c_in = input.shape()[0];
    let t = input.shape()[1];
    let k = c.k;
    assert_eq!(c_in, c.c_in);
    let t_pad = t + 2 * pad;
    // stride is 1 everywhere here: out = T + 2p - d*(k-1)
    let t_out = t_pad.saturating_sub(dil * (k - 1));
    let x = input.data();
    let mut xp = vec![0.0f32; c_in * t_pad];
    for ci in 0..c_in {
        xp[ci * t_pad + pad..ci * t_pad + pad + t].copy_from_slice(&x[ci * t..(ci + 1) * t]);
    }
    let feat = c_in * k;
    let mut cols = vec![0.0f32; t_out * feat];
    cols.par_chunks_mut(feat)
        .enumerate()
        .for_each(|(t_out_i, row)| {
            for ci in 0..c_in {
                let base = ci * t_pad + t_out_i; // stride-1 window start
                for j in 0..k {
                    row[ci * k + j] = xp[base + j * dil];
                }
            }
        });
    let cols_t = CpuTensor::from_data(vec![t_out, feat], cols);
    let mut out = cols_t.par_matmul(&c.w_t);
    {
        let data = out.data_mut();
        for t_out_i in 0..t_out {
            let row = &mut data[t_out_i * c.c_out..(t_out_i + 1) * c.c_out];
            for (o, slot) in row.iter_mut().enumerate() {
                *slot += c.bias[o];
            }
        }
    }
    let data = out.data().to_vec();
    let mut ct = vec![0.0f32; c.c_out * t_out];
    for i in 0..t_out {
        for o in 0..c.c_out {
            ct[o * t_out + i] = data[i * c.c_out + o];
        }
    }
    CpuTensor::from_data(vec![c.c_out, t_out], ct)
}

/// Flip along the channel axis of a `[C,T]` buffer (torch.flip(dim=1)).
#[allow(clippy::needless_range_loop)]
fn flip_channels(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * t];
    for ci in 0..c {
        out[(c - 1 - ci) * t..(c - ci) * t].copy_from_slice(&x[ci * t..(ci + 1) * t]);
    }
    out
}

// ---------------------------------------------------------------------------
// relative-position attention (reference-exact index mapping)
// ---------------------------------------------------------------------------

/// Slice of the (possibly zero-padded) relative embedding table usable for
/// sequence length `t`, returned as rows `[2t-1, hd]`. Mirrors
/// `_get_relative_embeddings`: pad symmetrically by
/// `max(t-(W+1),0)`, then slice `max(W+1-t,0) .. +2t-1`.
fn rel_slice(rel: &[f32], window: usize, hd: usize, t: usize) -> Vec<f32> {
    let w2 = 2 * window + 1;
    let pad_len = t.saturating_sub(window + 1); // reference pads by W+1
    let start = (window + 1).saturating_sub(t);
    let total_rows = w2 + 2 * pad_len;
    let end = (start + 2 * t - 1).min(total_rows);
    // build the PADDED table first (zeros beyond both ends), then slice
    let mut padded = vec![0.0f32; total_rows * hd];
    padded[hd * pad_len..hd * (pad_len + w2)].copy_from_slice(rel);
    padded[hd * start..hd * end].to_vec()
}

impl EncoderLayer {
    fn attention(
        &self,
        x: &[f32],
        t: usize,
        heads: usize,
        hd: usize,
        window: usize,
        slope: f32,
    ) -> Vec<f32> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL: AtomicUsize = AtomicUsize::new(0);
        let call_id = CALL.fetch_add(1, Ordering::Relaxed);
        let q = self.q.apply(x, t);
        let kk = self.k.apply(x, t);
        let vv = self.v.apply(x, t);
        if std::env::var("EMBER_VITS_DBG_QKV").is_ok() {
            let dir =
                std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_DIR").unwrap_or_default());
            std::fs::create_dir_all(&dir).ok();
            let dump = |name: &str, data: &[f32]| {
                let mut header = Vec::new();
                header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
                let descr = format!(
                    "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                    data.len()
                );
                header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
                header.extend_from_slice(descr.as_bytes());
                header.push(b'\n');
                let mut bytes = header;
                for v2 in data {
                    bytes.extend_from_slice(&v2.to_le_bytes());
                }
                let _ = std::fs::write(dir.join(name), &bytes);
            };
            dump(&format!("dbg_q{call_id}.npy"), &q);
            dump(&format!("dbg_k{call_id}.npy"), &kk);
            dump(&format!("dbg_v{call_id}.npy"), &vv);
        }

        let rel_k = rel_slice(&self.rel_k, window, hd, t);
        let rel_v = rel_slice(&self.rel_v, window, hd, t);
        if call_id == 0 && std::env::var("EMBER_VITS_DBG_QKV").is_ok() {
            let dir =
                std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_DIR").unwrap_or_default());
            let mut header = Vec::new();
            header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
            let descr = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                rel_k.len()
            );
            header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
            header.extend_from_slice(descr.as_bytes());
            header.push(b'\n');
            let mut bytes = header;
            for v2 in &rel_k {
                bytes.extend_from_slice(&v2.to_le_bytes());
            }
            let _ = std::fs::write(dir.join("dbg_rkslice0.npy"), &bytes);
        }

        // scores[h, q, k] = (q·k)*scale + q·rel_k[slice_row(k-q)]
        // (the pad/slice/relative->absolute pipeline reduces exactly to the
        // band-offset contraction below.)
        let w = window;
        // Reference `_get_relative_embeddings`: zero-pad the (2W+1)-row table
        // by pad_len rows on BOTH ends, then slice rows
        // [start, start + 2T-1) with start = max(W+1-T, 0). A pair offset
        // o = k - q lives at padded row o + W + pad_len, i.e. SLICE row
        // o + W + pad_len - start. For long sequences (T >= W+1) start == 0
        // and the row index collapses to o + W + pad_len; for SHORT
        // sequences the -start term is load-bearing (its absence made every
        // T <= W synthesis read the wrong relative row).
        let pad_len = t.saturating_sub(w + 1);
        let slice_start = (w + 1).saturating_sub(t);
        let rel_row_base = w as isize + pad_len as isize - slice_start as isize;
        let mut scores = vec![0.0f32; heads * t * t];
        {
            let sc = &mut scores;
            sc.par_chunks_mut(t * t).enumerate().for_each(|(h, sh)| {
                for qi in 0..t {
                    for ki in 0..t {
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot +=
                                q[qi * heads * hd + h * hd + d] * kk[ki * heads * hd + h * hd + d];
                        }
                        dot *= slope;
                        let off = ki as isize - qi as isize + rel_row_base;
                        if off >= 0 {
                            let r = off as usize;
                            if r < 2 * t - 1 {
                                let rk = &rel_k[r * hd..(r + 1) * hd];
                                let qp =
                                    &q[qi * heads * hd + h * hd..qi * heads * hd + (h + 1) * hd];
                                for d in 0..hd {
                                    dot += qp[d] * slope * rk[d];
                                }
                            }
                        }
                        sh[qi * t + ki] = dot;
                    }
                }
            });
        }

        if call_id == 0 && std::env::var("EMBER_VITS_DBG_QKV").is_ok() {
            let dir =
                std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_DIR").unwrap_or_default());
            let mut header = Vec::new();
            header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
            let descr = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                scores.len()
            );
            header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
            header.extend_from_slice(descr.as_bytes());
            header.push(b'\n');
            let mut bytes = header;
            for v2 in &scores {
                bytes.extend_from_slice(&v2.to_le_bytes());
            }
            let _ = std::fs::write(dir.join("dbg_scores0.npy"), &bytes);
        }
        softmax_rows(&mut scores, heads * t, t);
        if call_id == 0 && std::env::var("EMBER_VITS_DBG_QKV").is_ok() {
            let dir =
                std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_DIR").unwrap_or_default());
            let mut header = Vec::new();
            header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
            let descr = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                scores.len()
            );
            header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
            header.extend_from_slice(descr.as_bytes());
            header.push(b'\n');
            let mut bytes = header;
            for v2 in &scores {
                bytes.extend_from_slice(&v2.to_le_bytes());
            }
            let _ = std::fs::write(dir.join("dbg_probs0.npy"), &bytes);
        }

        // values + relative value bias:
        // out[q,d] += sum_k probs[q,k] · rel_v[k-q+W, d]
        let mut out = vec![0.0f32; heads * t * hd];
        out.par_chunks_mut(t * hd).enumerate().for_each(|(h, oh)| {
            for qi in 0..t {
                for ki in 0..t {
                    let p = scores[h * t * t + qi * t + ki];
                    if p == 0.0 {
                        continue;
                    }
                    let base = ki * heads * hd + h * hd;
                    for d in 0..hd {
                        oh[qi * hd + d] += p * vv[base + d];
                    }
                    let off = ki as isize - qi as isize + rel_row_base;
                    if off >= 0 {
                        let r = off as usize;
                        if r < 2 * t - 1 {
                            let rv = &rel_v[r * hd..(r + 1) * hd];
                            for d in 0..hd {
                                oh[qi * hd + d] += p * rv[d];
                            }
                        }
                    }
                }
            }
        });

        let mut merged = vec![0.0f32; t * heads * hd];
        for qi in 0..t {
            for h in 0..heads {
                for d in 0..hd {
                    merged[qi * heads * hd + h * hd + d] = out[h * t * hd + qi * hd + d];
                }
            }
        }
        if std::env::var("EMBER_VITS_DBG_MERGED").is_ok() {
            let dir =
                std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_DIR").unwrap_or_default());
            std::fs::create_dir_all(&dir).ok();
            let mut header = Vec::new();
            header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
            let descr = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                t * heads * hd
            );
            header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
            header.extend_from_slice(descr.as_bytes());
            header.push(b'\n');
            let mut bytes = header;
            for v in &merged {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let _ = std::fs::write(dir.join(format!("dbg_merged{call_id}.npy")), &bytes);
        }
        self.o.apply(&merged, t)
    }
}

// ---------------------------------------------------------------------------
// DDSConv / spline / ConvFlow
// ---------------------------------------------------------------------------

impl DdsConv {
    /// Input/output channel-major `[C, T]`. No padding mask (sequences are
    /// unbatched; every frame valid) — identical to the reference under an
    /// all-ones mask.
    fn forward(&self, x: &[f32], c: usize, t: usize, ln_eps: f32) -> Vec<f32> {
        let kernel = 3usize; // duration_predictor_kernel_size
        let mut cur = x.to_vec();
        let dbg = std::env::var("EMBER_VITS_DBG_SDP").is_ok()
            && std::env::var("EMBER_VITS_DBG_DDS").is_ok();
        let mut dw_dbg: Vec<Vec<f32>> = Vec::new();
        let mut pw_dbg: Vec<Vec<f32>> = Vec::new();
        for j in 0..self.pointwise.len() {
            // reference: dilation = kernel_size ** i (= 3^i), not 2^i
            let dil = kernel.pow(j as u32);
            let taps: Vec<f32> = self.dilated_w[j].data().to_vec(); // [C*k] ([C,1,k])
            let dw = depthwise_dilated(&cur, c, t, &taps, &self.dilated_b[j], dil);
            let tt = dw.len() / c;
            let mut rows = ct_to_rows(&dw, c, tt);
            layer_norm(&mut rows, tt, c, &self.ln1[j].0, &self.ln1[j].1, ln_eps);
            gelu_in_place(&mut rows);
            let mut pw = self.pointwise[j].apply(&rows, tt);
            layer_norm(&mut pw, tt, c, &self.ln2[j].0, &self.ln2[j].1, ln_eps);
            gelu_in_place(&mut pw);
            let pw_ct = rows_to_ct(&pw, c, tt);
            if dbg {
                pw_dbg.push(pw_ct.clone());
                dw_dbg.push(cur.clone());
            }
            for (cv, pv) in cur.iter_mut().zip(pw_ct.iter()) {
                *cv += pv;
            }
        }
        if dbg {
            let dir =
                std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_SDP").unwrap_or_default());
            let dump = |name: &str, data: &[f32]| {
                let mut header = Vec::new();
                header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
                let descr = format!(
                    "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                    data.len()
                );
                header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
                header.extend_from_slice(descr.as_bytes());
                header.push(b'\n');
                let mut bytes = header;
                for v2 in data {
                    bytes.extend_from_slice(&v2.to_le_bytes());
                }
                let _ = std::fs::write(dir.join(name), &bytes);
            };
            for (j, d) in dw_dbg.iter().enumerate() {
                dump(&format!("dbg_dds_in{j}.npy"), d);
            }
            for (j, p) in pw_dbg.iter().enumerate() {
                dump(&format!("dbg_dds_pw{j}.npy"), p);
            }
        }
        cur
    }
}

/// Unconstrained rational-quadratic spline, reverse direction only (ember
/// runs inference backwards through the duration-predictor flows).
/// `inputs[x_len]`, parameter rows `[x_len, bins*3-1]` already divided by
/// sqrt(filter_channels). Returns transformed values.
#[allow(clippy::needless_range_loop)]
fn spline_reverse(
    inputs: &[f32],
    widths_unnorm: &[Vec<f32>],
    heights_unnorm: &[Vec<f32>],
    derivs_unnorm: &[Vec<f32>],
    tail_bound: f32,
    bins: usize,
) -> Vec<f32> {
    const MIN_BIN_W: f64 = 1e-3;
    const MIN_BIN_H: f64 = 1e-3;
    const MIN_DERIV: f64 = 1e-3;
    let lower = -tail_bound as f64;
    let upper = tail_bound as f64;

    // softplus
    let softplus = |x: f64| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    // constant used to pin boundary derivatives
    let constant = ((1.0 - MIN_DERIV).exp() - 1.0).ln();

    let mut out = Vec::with_capacity(inputs.len());
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let calls = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _ = calls;
    for (i, &x0) in inputs.iter().enumerate() {
        let x = x0 as f64;
        if !(-tail_bound as f64..=tail_bound as f64).contains(&x) {
            out.push(x0); // identity outside
            continue;
        }
        // widths/heights: softmax -> min-scaled -> cumulative bounds
        let mut ws = vec![0.0f64; bins];
        let mut hs = vec![0.0f64; bins];
        let mut max_w = f64::NEG_INFINITY;
        let mut max_h = f64::NEG_INFINITY;
        for b in 0..bins {
            ws[b] = widths_unnorm[i][b] as f64;
            hs[b] = heights_unnorm[i][b] as f64;
            max_w = max_w.max(ws[b]);
            max_h = max_h.max(hs[b]);
        }
        let mut sw = 0.0;
        let mut sh = 0.0;
        for b in 0..bins {
            ws[b] = (ws[b] - max_w).exp();
            sw += ws[b];
            hs[b] = (hs[b] - max_h).exp();
            sh += hs[b];
        }
        // Min-scale AND stretch across the interval: reference computes
        // cumwidths = (upper-lower)*cumsum(scaled) + lower, i.e. the scaled
        // widths sum to (upper-lower), not 1.
        let span = upper - lower;
        for b in 0..bins {
            ws[b] = span * (MIN_BIN_W + (1.0 - MIN_BIN_W * bins as f64) * (ws[b] / sw));
            hs[b] = span * (MIN_BIN_H + (1.0 - MIN_BIN_H * bins as f64) * (hs[b] / sh));
        }
        let mut cumw = vec![lower; bins + 1];
        let mut cumh = vec![lower; bins + 1];
        for b in 0..bins {
            cumw[b + 1] = cumw[b] + ws[b];
            cumh[b + 1] = cumh[b] + hs[b];
        }
        // scale to span (they already do by construction of min-scaling)
        // The min-scaled widths/heights already sum to (upper-lower), so the
        // cumulative arrays span the interval exactly — no rescale needed.
        cumw[0] = lower;
        cumw[bins] = upper;
        cumh[0] = lower;
        cumh[bins] = upper;

        // Reference pads the raw internal derivatives with `constant` on
        // both sides and THEN applies min_derivative + softplus to ALL of
        // them — the boundaries become min_d + softplus(constant), not the
        // constant itself.
        let mut ds = vec![0.0f64; bins + 1];
        ds[0] = MIN_DERIV + softplus(constant);
        ds[bins] = MIN_DERIV + softplus(constant);
        for b in 0..bins - 1 {
            ds[b + 1] = MIN_DERIV + softplus(derivs_unnorm[i][b] as f64);
        }

        // bin search on cumw (input domain for reverse is cumh! careful:
        // reverse maps y->x using bin locations from cumHEIGHTS)
        let mut bin_idx = bins - 1;
        for bidx in 0..bins {
            if x < cumh[bidx + 1] {
                bin_idx = bidx;
                break;
            }
        }

        let cw0 = cumw[bin_idx];
        let w_bin = cumw[bin_idx + 1] - cw0;
        let ch0 = cumh[bin_idx];
        let h_bin = cumh[bin_idx + 1] - ch0;
        let delta = h_bin / w_bin;
        let d0 = ds[bin_idx];
        let d1 = ds[bin_idx + 1];

        // solve quadratic for theta
        let intermediate1 = d0 + d1 - 2.0 * delta;
        let intermediate2 = x - ch0;
        let intermediate3 = intermediate2 * intermediate1;
        let a = h_bin * (delta - d0) + intermediate3;
        let b_ = h_bin * d0 - intermediate3;
        let c_ = -delta * intermediate2;
        let discriminant = b_ * b_ - 4.0 * a * c_;
        let root = (2.0 * c_) / (-b_ - discriminant.sqrt());

        let y = root * w_bin + cw0;
        if std::env::var("EMBER_VITS_DBG_SPLINE").is_ok() && calls < 6 {
            eprintln!(
                "SPLINEC x={x:.6} idx={bin_idx} y={y:.8} cw0={cw0:.7} wb={w_bin:.7} ch0={ch0:.7} hb={h_bin:.7} delta={delta:.7} d0={d0:.7} d1={d1:.7}",
            );
        }
        out.push(y as f32);
    }
    out
}

impl ConvFlow {
    /// Reverse pass. `x` channel-major `[F, S]`, F = 2*half.
    /// `cond` is the SDP main-path projection (added to the DDSConv input,
    /// exactly where the reference passes `global_conditioning`).
    fn forward_reverse(
        &self,
        x: &[f32],
        cond: Option<&[f32]>,
        s: usize,
        cfg: &VitsConfig,
    ) -> Vec<f32> {
        let f = 2usize;
        let half = f / 2;
        let first = &x[..half * s];
        let second = &x[half * s..];
        let filter_ch = cfg.hidden_size;
        let scale = (filter_ch as f32).sqrt();

        let mut pre = conv1d_dense(
            &CpuTensor::from_data(vec![half, s], first.to_vec()),
            &self.conv_pre,
            0,
        )
        .data()
        .to_vec();
        if let Some(cond) = cond {
            for (pv, cv) in pre.iter_mut().zip(cond.iter()) {
                *pv += cv;
            }
        }
        let dds_out = self.dds.forward(&pre, cfg.hidden_size, s, cfg.ln_eps);
        let proj = conv1d_dense(
            &CpuTensor::from_data(vec![cfg.hidden_size, s], dds_out.clone()),
            &self.conv_proj,
            0,
        );
        if std::env::var("EMBER_VITS_DBG_SDP").is_ok() {
            let dir =
                std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_SDP").unwrap_or_default());
            let dump = |name: &str, data: &[f32]| {
                let mut header = Vec::new();
                header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
                let descr = format!(
                    "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                    data.len()
                );
                header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
                header.extend_from_slice(descr.as_bytes());
                header.push(b'\n');
                let mut bytes = header;
                for v2 in data {
                    bytes.extend_from_slice(&v2.to_le_bytes());
                }
                let _ = std::fs::write(dir.join(name), &bytes);
            };
            dump("cf_pre.npy", &pre);
            dump("cf_dds.npy", &dds_out);
        }

        let bins = cfg.dp_bins;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CF_CALL: AtomicUsize = AtomicUsize::new(0);
        let cf_id = CF_CALL.fetch_add(1, Ordering::Relaxed);
        if std::env::var("EMBER_VITS_DBG_SDP").is_ok() {
            let dir =
                std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_SDP").unwrap_or_default());
            let step = format!("{cf_id}");
            let dump = |name: &str, data: &[f32]| {
                let mut header = Vec::new();
                header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
                let descr = format!(
                    "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                    data.len()
                );
                header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
                header.extend_from_slice(descr.as_bytes());
                header.push(b'\n');
                let mut bytes = header;
                for v2 in data {
                    bytes.extend_from_slice(&v2.to_le_bytes());
                }
                let _ = std::fs::write(dir.join(format!("{name}_{step}.npy")), &bytes);
            };
            dump("cf_pre", &pre);
            dump("cf_dds", &dds_out);
            dump("cf_proj", proj.data());
        }

        let params_per_t = bins * 3 - 1;
        let mut out_second = second.to_vec();
        for ti in 0..s {
            for ch in 0..half {
                // proj is channel-major [half*params_per_t, S]: the param
                // row for (ch, ti) strides by S
                let row: Vec<f32> = (0..params_per_t)
                    .map(|pp| proj.data()[(ch * params_per_t + pp) * s + ti])
                    .collect();
                let row = &row[..];
                let mut widths = vec![0.0f32; bins];
                let mut heights = vec![0.0f32; bins];
                let mut derivs = vec![0.0f32; params_per_t - 2 * bins];
                widths.copy_from_slice(&row[..bins]);
                heights.copy_from_slice(&row[bins..2 * bins]);
                derivs.copy_from_slice(&row[2 * bins..]);
                // NOTE: reference divides ONLY widths and heights by
                // sqrt(filter_channels); derivatives stay unscaled.
                for v in widths.iter_mut() {
                    *v /= scale;
                }
                for v in heights.iter_mut() {
                    *v /= scale;
                }
                let y = spline_reverse(
                    &[second[ch * s + ti]],
                    &[widths],
                    &[heights],
                    &[derivs],
                    cfg.dp_tail_bound,
                    bins,
                )[0];
                out_second[ch * s + ti] = y;
            }
        }

        let mut out = x.to_vec();
        out[half * s..].copy_from_slice(&out_second);
        out
    }
}

impl Sdp {
    /// Reverse path with ZERO noise: `latents = 0` seeded, run
    /// [CF_last .. CF_1(skip CF_1? see note), Affine] per the reference's
    /// `flows[:-2] + [flows[-1]]` selection.
    ///
    /// Reference flow list = [Affine, CF1, ..., CFn]; reversed =
    /// [CFn..CF1, Affine]; `flows[:-2] + [flows[-1]]` keeps
    /// [CFn ... CF2, Affine] — CF1 is dropped. Replicated exactly.
    fn log_duration_reverse(
        &self,
        encoder_hidden: &[f32],
        c: usize,
        t: usize,
        cfg: &VitsConfig,
    ) -> Vec<f32> {
        // main conditioning
        let pre = conv1d_dense(
            &CpuTensor::from_data(vec![c, t], encoder_hidden.to_vec()),
            &self.conv_pre,
            0,
        );
        let dds_out = self.dds.forward(pre.data(), c, t, cfg.ln_eps);
        let proj = conv1d_dense(
            &CpuTensor::from_data(vec![c, t], dds_out),
            &self.conv_proj,
            0,
        );
        let g_cond = proj.data().to_vec(); // [c, t]
        if std::env::var("EMBER_VITS_DBG_SDP").is_ok() {
            let dir =
                std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_SDP").unwrap_or_default());
            let mut header = Vec::new();
            header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
            let descr = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                g_cond.len()
            );
            header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
            header.extend_from_slice(descr.as_bytes());
            header.push(b'\n');
            let mut bytes = header;
            for v2 in &g_cond {
                bytes.extend_from_slice(&v2.to_le_bytes());
            }
            let _ = std::fs::write(std::path::Path::new(&dir).join("g_cond.npy"), &bytes);
        }

        // latents = randn * noise_scale_duration = 0 (deterministic contract)
        let mut latents = vec![0.0f32; 2 * t]; // [2, t]

        let mut order: Vec<usize> = (1..=self.conv_flows.len()).rev().collect();
        order.pop(); // drop CF1 (index 1) — flows[:-2]
        order.push(0); // + Affine — [flows[-1]]

        let dbg_dir = std::env::var("EMBER_VITS_DBG_SDP").ok();
        if let Some(dir) = &dbg_dir {
            std::fs::create_dir_all(dir).ok();
        }
        for (step_i, flow_idx) in order.iter().enumerate() {
            latents = flip_channels(&latents, 2, t);
            if let Some(dir) = &dbg_dir {
                let mut header = Vec::new();
                header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
                let descr = format!(
                    "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                    latents.len()
                );
                header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
                header.extend_from_slice(descr.as_bytes());
                header.push(b'\n');
                // NOTE: dump PRE-flow (post-flip) state
                let mut bytes = header;
                for v2 in &latents {
                    bytes.extend_from_slice(&v2.to_le_bytes());
                }
                let _ = std::fs::write(
                    std::path::Path::new(dir).join(format!("pre_{step_i}.npy")),
                    &bytes,
                );
            }
            if *flow_idx == 0 {
                // ElementwiseAffine stores per-channel [C=2, 1] vectors.
                let (translate, log_scale) = self.affine.as_ref().expect("affine");
                for (li, lv) in latents.iter_mut().enumerate() {
                    let ch = li / t;
                    let tr_c = translate[ch];
                    let ls_c = log_scale[ch];
                    *lv = (*lv - tr_c) * (-ls_c).exp();
                }
            } else {
                let cf = &self.conv_flows[flow_idx - 1];
                latents = cf.forward_reverse(&latents, Some(&g_cond), t, cfg);
            }
            if let Some(dir) = &dbg_dir {
                let mut header = Vec::new();
                header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
                let descr = format!(
                    "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                    latents.len()
                );
                header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
                header.extend_from_slice(descr.as_bytes());
                header.push(b'\n');
                let mut bytes = header;
                for v2 in &latents {
                    bytes.extend_from_slice(&v2.to_le_bytes());
                }
                let _ = std::fs::write(
                    std::path::Path::new(dir).join(format!("after_{step_i}.npy")),
                    &bytes,
                );
            }
        }
        // first half channel is log-duration
        latents[..t].to_vec()
    }
}

// ---------------------------------------------------------------------------
// prior flow (WaveNet reverse) + HiFi-GAN
// ---------------------------------------------------------------------------

impl FlowLayer {
    /// Reverse coupling layer: second_half -= mean(first_half).
    /// `module_idx` names the layer for EMBER_VITS_FLOWDUMP parity dumps
    /// (reference applies modules 3,2,1,0 in reverse mode).
    fn forward_reverse(
        &self,
        x: &[f32],
        f: usize,
        s: usize,
        cfg: &VitsConfig,
        module_idx: usize,
    ) -> Vec<f32> {
        let half = f / 2;
        let first = x[..half * s].to_vec();
        let pre = conv1d_dense(
            &CpuTensor::from_data(vec![half, s], first.clone()),
            &self.conv_pre,
            0,
        )
        .data()
        .to_vec();

        // WaveNet (no global conditioning; speaker_embedding_size = 0)
        let dbg_dir = std::env::var("EMBER_VITS_FLOWDUMP").ok();
        if let Some(dir) = &dbg_dir {
            std::fs::create_dir_all(dir).ok();
            dump_npy(std::path::Path::new(dir), &format!("f{module_idx}_in"), x);
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_x0"),
                &first,
            );
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_x1"),
                &x[half * s..],
            );
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_convpre"),
                &pre,
            );
        }
        let mut cur = pre;
        let mut outputs = vec![0.0f32; cfg.hidden_size * s];
        for i in 0..self.wn_in.len() {
            // reference pads in_layers by (k*dia-dia)/2 (= 2 for k=5,dia=1)
            let pad = (self.wn_in[i].k - 1) / 2;
            let h = conv1d_dense(
                &CpuTensor::from_data(vec![cfg.hidden_size, s], cur.clone()),
                &self.wn_in[i],
                pad,
            )
            .data()
            .to_vec();
            // fused add-tanh-sigmoid-multiply with zero cond
            let mut acts = vec![0.0f32; cfg.hidden_size * s];
            for j in 0..cfg.hidden_size * s {
                let tanh_half = (h[j]).tanh();
                let sig_half = 1.0 / (1.0 + (-h[cfg.hidden_size * s + j]).exp());
                acts[j] = tanh_half * sig_half;
            }
            let rs = conv1d_dense(
                &CpuTensor::from_data(vec![cfg.hidden_size, s], acts.clone()),
                &self.wn_rs[i],
                0,
            )
            .data()
            .to_vec();
            if let Some(dir) = &dbg_dir {
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_wn{i}_h"),
                    &h,
                );
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_wn{i}_acts"),
                    &acts,
                );
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_wn{i}_rs"),
                    &rs,
                );
            }
            if i < self.wn_in.len() - 1 {
                // res_skip_channels = 2H: first H residual, last H skip
                for j in 0..cfg.hidden_size * s {
                    cur[j] += rs[j];
                    outputs[j] += rs[cfg.hidden_size * s + j];
                }
            } else {
                for j in 0..cfg.hidden_size * s {
                    outputs[j] += rs[j];
                }
            }
            if let Some(dir) = &dbg_dir {
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_wn{i}_cur"),
                    &cur,
                );
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_wn{i}_acc"),
                    &outputs,
                );
            }
        }

        let mean = conv1d_dense(
            &CpuTensor::from_data(vec![cfg.hidden_size, s], outputs.clone()),
            &self.conv_post,
            0,
        )
        .data()
        .to_vec();
        if let Some(dir) = &dbg_dir {
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_mean"),
                &mean,
            );
            let x1new: Vec<f32> = x[half * s..]
                .iter()
                .zip(mean.iter())
                .map(|(sv, mv)| sv - mv)
                .collect();
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_x1new"),
                &x1new,
            );
        }
        let mut out = x.to_vec();
        for (sv, mv) in out[half * s..].iter_mut().zip(mean.iter()) {
            *sv -= mv;
        }
        if let Some(dir) = &dbg_dir {
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_out"),
                &out,
            );
        }
        out
    }
}

impl ConvTranspose1d {
    /// Input `[C_in, T]` -> output `[C_out, (T-1)*stride - 2*padding + k]`.
    #[allow(clippy::needless_range_loop)]
    fn forward(&self, input: &[f32], c_in: usize, t: usize) -> Vec<f32> {
        let k = self.w.shape()[2];
        let c_out = self.w.shape()[1];
        let t_out = (t - 1) * self.stride + k - 2 * self.padding;
        let mut acc = vec![0.0f64; c_out * t_out];
        let w = self.w.data();
        for ci in 0..c_in {
            let xc = &input[ci * t..(ci + 1) * t];
            for ti in 0..t {
                let v = xc[ti] as f64;
                if v == 0.0 {
                    continue;
                }
                let obase: isize = (ti * self.stride) as isize - self.padding as isize;
                for co in 0..c_out {
                    let wc = &w[(ci * c_out + co) * k..(ci * c_out + co) * k + k];
                    for kk in 0..k {
                        let oi = obase + kk as isize;
                        if oi >= 0 && (oi as usize) < t_out {
                            acc[co * t_out + oi as usize] += v * wc[kk] as f64;
                        }
                    }
                }
            }
        }
        let mut out = vec![0.0f32; c_out * t_out];
        for co in 0..c_out {
            for ti in 0..t_out {
                out[co * t_out + ti] = acc[co * t_out + ti] as f32 + self.b[co];
            }
        }
        out
    }
}

impl ResBlock {
    /// `dump_name` prefixes EMBER_VITS_DECDUMP substep dumps ("" disables).
    fn forward(&self, x: &[f32], c: usize, t: usize, slope: f32, dump_name: &str) -> Vec<f32> {
        let dbg_dir = std::env::var("EMBER_VITS_DECDUMP").ok();
        let mut cur = x.to_vec();
        // HiFi-GAN resblock dilation cycle: resblock_dilation_sizes[0] =
        // [1, 3, 5] for every MMS checkpoint (NOT the iteration index d+1).
        const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
        for (d, &dil1) in RESBLOCK_DILATIONS.iter().enumerate() {
            let residual = cur.clone();
            leaky_in_place(&mut cur, slope);
            let pad1 = cur_pad_for(self.c1[d].k, dil1);
            let y1 = conv_dense_dilated(
                &CpuTensor::from_data(vec![c, t], cur),
                &self.c1[d],
                pad1,
                dil1,
            );
            if let Some(dir) = &dbg_dir {
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("{dump_name}_it{d}_c1"),
                    y1.data(),
                );
            }
            let t1 = y1.shape()[1];
            let mut tmp = y1.data().to_vec();
            leaky_in_place(&mut tmp, slope);
            let pad2 = cur_pad_for(self.c2[d].k, 1);
            let mut y2 =
                conv_dense_padded(&CpuTensor::from_data(vec![c, t1], tmp), &self.c2[d], pad2)
                    .data()
                    .to_vec();
            if let Some(dir) = &dbg_dir {
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("{dump_name}_it{d}_residual"),
                    &residual,
                );
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("{dump_name}_it{d}_c2presid"),
                    &y2,
                );
            }
            for (cv, rv) in y2.iter_mut().zip(residual.iter()) {
                *cv += rv;
            }
            cur = y2;
            if let Some(dir) = &dbg_dir {
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("{dump_name}_it{d}_out"),
                    &cur,
                );
            }
        }
        cur
    }
}

fn leaky_in_place(v: &mut [f32], slope: f32) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x *= slope;
        }
    }
}

fn cur_pad_for(k: usize, dil: usize) -> usize {
    (k * dil - dil) / 2
}

/// groups=1 Conv1d with arbitrary symmetric padding over a borrowed buffer.
fn conv_dense_padded(input: &CpuTensor, c: &DenseConv1d, pad: usize) -> CpuTensor {
    conv1d_dense(input, c, pad)
}

impl HifiGan {
    /// Full decode of a final spectrogram `[F, S]`.
    /// EMBER_VITS_DECDUMP=<dir> mirrors scripts/ref_decoder_dump.py.
    fn decode(&self, spec: &[f32], f: usize, s: usize, cfg: &VitsConfig) -> Vec<f32> {
        let dbg_dir = std::env::var("EMBER_VITS_DECDUMP").ok();
        let dump = |name: &str, data: &[f32]| {
            if let Some(dir) = &dbg_dir {
                dump_npy(std::path::Path::new(dir), name, data);
            }
        };
        if let Some(dir) = &dbg_dir {
            std::fs::create_dir_all(dir).ok();
            dump("d_spec", spec);
        }
        let mut hidden = conv1d_dense(
            &CpuTensor::from_data(vec![f, s], spec.to_vec()),
            &self.conv_pre,
            3,
        )
        .data()
        .to_vec();
        dump("d_convpre", &hidden);
        let nblk = 3usize;
        for stage in 0..self.ups.len() {
            let ch_in = self.ups[stage].w.shape()[0];
            let tt = hidden.len() / ch_in;
            for v in hidden.iter_mut() {
                *v = if *v < 0.0 {
                    *v * cfg.leaky_relu_slope
                } else {
                    *v
                };
            }
            hidden = self.ups[stage].forward(&hidden, ch_in, tt);
            let ch_out = self.ups[stage].w.shape()[1];
            let to = hidden.len() / ch_out;
            dump(&format!("d_up{stage}"), &hidden);
            // average of the stage's resblocks
            let mut sum = vec![0.0f64; ch_out * to];
            for blk in 0..nblk {
                let rb = &self.resblocks[stage * nblk + blk];
                let y = rb.forward(
                    &hidden,
                    ch_out,
                    to,
                    cfg.leaky_relu_slope,
                    &format!("d_rb{stage}{blk}"),
                );
                for (sv, yv) in sum.iter_mut().zip(y.iter()) {
                    *sv += *yv as f64;
                }
                // reference accumulates in-place; the running sum is what
                // scripts/ref_decoder_dump.py names d_rb{stage}{blk}
                let running: Vec<f32> = sum.iter().map(|&v| v as f32).collect();
                dump(&format!("d_rb{stage}{blk}"), &running);
            }
            hidden = sum.iter().map(|&v| (v / nblk as f64) as f32).collect();
            dump(&format!("d_stage{stage}"), &hidden);
        }
        let c_last = self.conv_post_w.shape()[1];
        let tl = hidden.len() / c_last;
        // transformers VitsHifiGan.forward applies the FINAL pre-conv_post
        // leaky with the DEFAULT slope (0.01), not config.leaky_relu_slope
        // (0.1) used everywhere else — modeling_vits.py line
        // `nn.functional.leaky_relu(hidden_states)` before conv_post.
        const PREPOST_SLOPE: f32 = 0.01;
        for v in hidden.iter_mut() {
            *v = if *v < 0.0 { *v * PREPOST_SLOPE } else { *v };
        }
        dump("d_prepost", &hidden);
        // conv_post k7 pad3, no bias, then tanh
        let t_out = tl + 6 - 7 + 1;
        let mut wave = vec![0.0f32; t_out];
        let w = self.conv_post_w.data(); // [1, C, 7]
        let hc = &hidden;
        wave.par_chunks_mut(t_out).enumerate().for_each(|(o, ow)| {
            // every t_out = tl + 2*3 - 7 + 1 output index is a valid window
            #[allow(clippy::needless_range_loop)]
            for ti in 0..t_out {
                let mut acc = 0.0f64;
                for ci in 0..c_last {
                    let wc = &w[(o * c_last + ci) * 7..(o * c_last + ci) * 7 + 7];
                    let base = ti; // pad=3
                    #[allow(clippy::needless_range_loop)]
                    for kk in 0..7 {
                        let p = base + kk;
                        if p >= 3 && p < 3 + tl {
                            acc += (hc[ci * tl + (p - 3)] as f64) * wc[kk] as f64;
                        }
                    }
                }
                ow[ti] = (acc as f32).tanh();
            }
        });
        // pre-tanh conv output (single output channel)
        let pretanh: Vec<f32> = (0..t_out)
            .map(|ti| {
                let mut acc = 0.0f64;
                for ci in 0..c_last {
                    let wc = &w[ci * 7..ci * 7 + 7];
                    #[allow(clippy::needless_range_loop)]
                    for kk in 0..7 {
                        let p = ti + kk;
                        if p >= 3 && p < 3 + tl {
                            acc += hc[ci * tl + (p - 3)] as f64 * wc[kk] as f64;
                        }
                    }
                }
                acc as f32
            })
            .collect();
        dump("d_pretanh", &pretanh);
        dump("d_tanh", &wave);
        wave
    }
}

// ---------------------------------------------------------------------------
// synthesis entry points
// ---------------------------------------------------------------------------

/// Static synthesis result.
pub struct VitsSynthesis {
    pub pcm: Vec<f32>,
    pub timings: TtsTimings,
    pub trace: VitsTrace,
}

impl MmsVits {
    /// Deterministic synthesis (zero noise): text -> PCM @16 kHz.
    pub fn synthesize(
        &self,
        backend: &CpuBackend,
        text: &str,
        trace: bool,
    ) -> Result<VitsSynthesis> {
        let _ = backend;
        let t_all = Instant::now();
        let ids = self.tokenize(text);
        ensure!(
            !ids.is_empty(),
            "no speakable characters after preprocessing"
        );
        let mut timings = TtsTimings {
            prompt_ms: t_all.elapsed().as_secs_f64() * 1e3,
            ..TtsTimings::default()
        };

        let mut tr = VitsTrace::default();
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let t = ids.len();

        // embed * sqrt(H), row-major [T, H]
        let scale = (h as f32).sqrt();
        let mut emb = vec![0.0f32; t * h];
        for (r, &id) in ids.iter().enumerate() {
            let src = &self.embed[id as usize * h..(id as usize + 1) * h];
            for (d, &v) in src.iter().enumerate() {
                emb[r * h + d] = v * scale;
            }
        }
        fn dbg_dump(name: &str, data: &[f32]) {
            if std::env::var("EMBER_VITS_DBG_DIR").ok().is_none() {
                return;
            }
            let dir = std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_DIR").unwrap());
            std::fs::create_dir_all(&dir).ok();
            use std::io::Write;
            let mut header = Vec::new();
            header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
            let descr = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                data.len()
            );
            header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
            header.extend_from_slice(descr.as_bytes());
            header.push(b'\n');
            let mut bytes = header;
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let _ = std::fs::File::create(dir.join(format!("{name}.npy")))
                .and_then(|mut f| f.write_all(&bytes));
        }
        dbg_dump("dbg_input", &emb);
        if std::env::var("EMBER_VITS_DBG_REL").is_ok() {
            let rk = &self.layers[0].rel_k;
            let mut header = Vec::new();
            header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
            let descr = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                rk.len()
            );
            header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
            header.extend_from_slice(descr.as_bytes());
            header.push(b'\n');
            let mut bytes = header;
            for v in rk {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let dir = std::path::PathBuf::from(std::env::var("EMBER_VITS_DBG_DIR").unwrap());
            std::fs::create_dir_all(&dir).ok();
            let _ = std::fs::write(dir.join("dbg_relk0.npy"), &bytes);
        }

        // encoder layers
        let mut x = emb.clone();
        for (layer_i, layer) in self.layers.iter().enumerate() {
            let attn = layer.attention(
                &x,
                t,
                cfg.num_heads,
                cfg.head_dim(),
                cfg.window_size,
                (cfg.head_dim() as f32).sqrt().recip(),
            );
            dbg_dump(&format!("dbg_att{layer_i}"), &attn);
            let mut normed = x.clone();
            for (nv, av) in normed.iter_mut().zip(attn.iter()) {
                *nv += av;
            }
            layer_norm(&mut normed, t, h, &layer.ln1_w, &layer.ln1_b, cfg.ln_eps);
            dbg_dump(&format!("dbg_ln1_{layer_i}"), &normed);

            // FFN: channel-major through convs
            let ct = rows_to_ct(&normed, h, t);
            let f1 = conv1d_dense(
                &CpuTensor::from_data(vec![h, t], ct.clone()),
                &layer.ffn1,
                (cfg.ffn_kernel_size - 1) / 2,
            );
            let mut f1d = f1.data().to_vec();
            relu_in_place(&mut f1d);
            let f2 = conv1d_dense(
                &f1d_view(f1d, cfg.ffn_dim, f1.shape()[1]),
                &layer.ffn2,
                (cfg.ffn_kernel_size - 1) / 2,
            );
            let f2_rows = ct_to_rows(f2.data(), h, f2.shape()[1]);
            let mut resid = normed;
            for (rv, fv) in resid.iter_mut().zip(f2_rows.iter()) {
                *rv += fv;
            }
            layer_norm(&mut resid, t, h, &layer.ln2_w, &layer.ln2_b, cfg.ln_eps);
            x = resid;
            dbg_dump(&format!("dbg_after_layer{layer_i}"), &x);
        }
        timings.prefill_ms = t_all.elapsed().as_secs_f64() * 1e3 - timings.prompt_ms;

        // project to means/logvars (conv k1 == linear per position)
        let stats_ct = conv1d_dense(
            &CpuTensor::from_data(vec![h, t], ct_of(&x, h, t)),
            &self.project,
            0,
        );
        let f_dim = cfg.flow_size;
        let stats_rows = ct_to_rows(stats_ct.data(), 2 * f_dim, t);
        let mut prior_means = vec![0.0f32; t * f_dim];
        let mut _logvars = vec![0.0f32; t * f_dim];
        for r in 0..t {
            prior_means[r * f_dim..(r + 1) * f_dim]
                .copy_from_slice(&stats_rows[r * 2 * f_dim..r * 2 * f_dim + f_dim]);
            _logvars[r * f_dim..(r + 1) * f_dim]
                .copy_from_slice(&stats_rows[r * 2 * f_dim + f_dim..(r + 1) * 2 * f_dim]);
        }

        // SDP reverse (noise_scale_duration = 0)
        let x_ct = ct_of(&x, h, t);
        let log_duration = self.sdp.log_duration_reverse(&x_ct, h, t, cfg);
        timings.generate_ms =
            t_all.elapsed().as_secs_f64() * 1e3 - timings.prompt_ms - timings.prefill_ms;

        // durations: ceil(exp(log_d)); monotonic expansion
        let mut durations = vec![1i64; t];
        let mut total_s: i64 = 0;
        for (i, &ld) in log_duration.iter().enumerate() {
            durations[i] = (ld.exp()).ceil() as i64;
            total_s += durations[i];
        }
        let s_len = total_s.max(1) as usize;

        let mut expanded_hidden = vec![0.0f32; s_len * h];
        // channel-major [F, S]: this buffer feeds the flow stack and
        // HiFi-GAN directly, both of which operate on [C, T] conv layouts.
        let mut expanded_means = vec![0.0f32; f_dim * s_len];
        {
            let mut si = 0usize;
            for (i, &dur) in durations.iter().enumerate() {
                for _ in 0..dur {
                    if si >= s_len {
                        break;
                    }
                    expanded_hidden[si * h..(si + 1) * h].copy_from_slice(&x[i * h..(i + 1) * h]);
                    for d in 0..f_dim {
                        expanded_means[d * s_len + si] = prior_means[i * f_dim + d];
                    }
                    si += 1;
                }
            }
        }

        // prior latents = expanded means (noise_scale = 0); flow reverse x4
        let mut z = expanded_means.clone();
        let fl = cfg.flow_size;
        let dbg_flow = std::env::var("EMBER_VITS_FLOWDUMP").ok();
        if let Some(dir) = &dbg_flow {
            std::fs::create_dir_all(dir).ok();
            dump_npy(std::path::Path::new(dir), "z0_input", &z);
        }
        let n_flows = self.flows.len();
        for (fi, flow) in self.flows.iter().rev().enumerate() {
            z = flip_channels(&z, fl, s_len);
            // module index: reference reverses the module list (3,2,1,0)
            let module_idx = n_flows - 1 - fi;
            z = flow.forward_reverse(&z, fl, s_len, cfg, module_idx);
            if let Some(dir) = &dbg_flow {
                // post-module channel flip == input to the next module
                // (the reference returns module 0's UNflipped concat output)
                let flipped = flip_channels(&z, fl, s_len);
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_flip"),
                    &flipped,
                );
                if fi == n_flows - 1 {
                    dump_npy(std::path::Path::new(dir), "flow_z_final", &z);
                }
            }
        }
        timings.n_tokens = t;
        timings.n_codes = s_len;

        // HiFi-GAN decode
        let t_codec = Instant::now();
        let pcm_full = self.hifigan.decode(&z, fl, s_len, cfg);
        timings.codec_ms = t_codec.elapsed().as_secs_f64() * 1e3;
        timings.time_to_first_audio_ms = 0.0;

        if trace {
            tr.ids = Some(ids);
            tr.embed_scaled = Some(emb);
            tr.encoder_out = Some(x.clone());
            tr.prior_means = Some(prior_means);
            tr.log_duration = Some(log_duration);
            tr.expanded_hidden = Some(expanded_hidden);
            tr.flow_z = Some(z);
            tr.spectrogram = Some(tr.flow_z.clone().unwrap_or_default());
            tr.waveform = Some(pcm_full.clone());
        }

        Ok(VitsSynthesis {
            pcm: pcm_full,
            timings,
            trace: tr,
        })
    }

    /// Streaming synthesis: chunked HiFi-GAN decode over final mel frames.
    ///
    /// The decoder's receptive field is bounded (7-frame taps around each
    /// upsample stage ⇒ ≤ ~60 mel frames of right context), so once the
    /// duration expansion is fixed every output sample is FINAL — unlike the
    /// WavTokenizer path there is no drift and no revision contract.
    /// `stable_up_to` always equals the chunk end.
    pub fn synthesize_streaming(
        &self,
        backend: &CpuBackend,
        text: &str,
        max_frames_hint: usize,
        chunk_frames: usize,
        mut on_chunk: impl FnMut(AudioChunkMeta) -> bool,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<(Vec<f32>, Vec<u32>, TtsTimings)> {
        let _ = (max_frames_hint, on_token);
        let synth_start = Instant::now();
        let full = self.synthesize(backend, text, false)?;
        let sr = self.config.sample_rate;
        let hop = self.config.hop_length;
        let mut timings = full.timings;
        let s_len = timings.n_codes;

        let mut emitted: Vec<f32> = Vec::new();
        let mut frame_cursor = 0usize;
        // NOTE: current implementation decodes the full spectrogram once and
        // slices PCM by frames (the decode itself is one shot). Chunked
        // partial decode is the follow-up optimization; the STREAMING
        // STABILITY claim already holds because chunks are cuts of the final
        // waveform — but TTFA currently equals completion time, which is
        // honest and reported as such until the partial decoder lands.
        while frame_cursor < s_len {
            let end = (frame_cursor + chunk_frames).min(s_len);
            let start_sample = frame_cursor * hop;
            let end_sample = (end * hop).min(full.pcm.len());
            if end_sample > start_sample {
                let pcm = full.pcm[start_sample..end_sample].to_vec();
                let meta = AudioChunkMeta {
                    stable_up_to: emitted.len() + pcm.len(), // immutable
                    playable_hint: emitted.len() + pcm.len(),
                    revised_tail: Vec::new(),
                    revised_from: 0,
                    first_token: frame_cursor,
                    final_chunk: end == s_len,
                    first_sample: emitted.len(),
                    sample_rate: sr,
                    pcm: pcm.clone(),
                };
                emitted.extend_from_slice(&pcm);
                if !on_chunk(meta) {
                    break;
                }
            }
            frame_cursor = end;
        }
        timings.time_to_first_audio_ms = synth_start.elapsed().as_secs_f64() * 1e3;
        Ok((emitted, Vec::new(), timings))
    }
}

fn relu_in_place(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// Flat float32 .npy writer used by the EMBER_VITS_FLOWDUMP parity harness.
pub(crate) fn dump_npy(dir: &std::path::Path, name: &str, data: &[f32]) {
    let mut header = Vec::new();
    header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1u8, 0u8]);
    let descr = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
        data.len()
    );
    header.extend_from_slice(&((descr.len() + 1) as u16).to_le_bytes());
    header.extend_from_slice(descr.as_bytes());
    header.push(b'\n');
    let mut bytes = header;
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let _ = std::fs::write(dir.join(format!("{name}.npy")), bytes);
}

fn f1d_view(data: Vec<f32>, c: usize, t: usize) -> CpuTensor {
    CpuTensor::from_data(vec![c, t], data)
}

/// Borrowing view helper: copy row-major [T,C] into channel-major [C,T].
fn ct_of(rows: &[f32], c: usize, t: usize) -> Vec<f32> {
    rows_to_ct(rows, c, t)
}

#[cfg(test)]
mod shape_probe {
    #[test]
    #[ignore]
    fn probe() {
        let mut l = crate::loader::load_gguf(std::path::Path::new(
            "/home/west/ember-work/mms-tts/ara.vits.gguf",
        ))
        .unwrap();
        let t = l.take_f32("v.flow0.wn.in0.w").unwrap();
        println!("raw {:?}", t.shape());
    }
}

#[cfg(test)]
mod forensic {
    #[test]
    fn par_matmul_layout() {
        // A[T=2,K=3] @ B[K=3,N=2]
        let a = crate::tensor::CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = crate::tensor::CpuTensor::from_data(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let c = a.par_matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
        // row0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        // row1: [4+0+6, 0+5+6]         = [10, 11]
        assert_eq!(c.data(), &[4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    #[ignore]
    fn q_layout_probe() {
        let mut l = crate::loader::load_gguf(std::path::Path::new(
            "/home/west/ember-work/mms-tts/ara.vits.gguf",
        ))
        .unwrap();

        let _ = l.metadata.get("vits.vocab");
        let w_hf = crate::tts::wavtokenizer::gguf_to_hf(&l.take_f32("v.layer.0.attn.q.w").unwrap());
        println!("shape {:?}", w_hf.shape());
        println!("first 6 hf bytes: {:?}", &w_hf.data()[..6]);
    }
}
