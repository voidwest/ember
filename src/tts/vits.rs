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

/// Most mel frames one synthesis may expand to. Durations come from the
/// model's weights, and HiFi-GAN peaks at ~8192 f32 per frame, so this
/// keeps one request near 1 GiB (~8.7 min at 16 kHz with hop 256).
pub const MAX_SYNTH_FRAMES: usize = 1 << 15;

impl VitsConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Reject metadata synthesis cannot run with (zero heads or kernel,
    /// absurd sizes, non-finite constants) before anything is sized by it.
    fn validate(&self) -> Result<()> {
        use crate::loader::limits::{
            MAX_EMBED_DIM, MAX_HEADS, MAX_HEAD_DIM, MAX_INTERMEDIATE_DIM, MAX_LAYERS,
            MAX_VOCAB_SIZE,
        };
        for (name, value, min, max) in [
            ("vocab_size", self.vocab_size, 1, MAX_VOCAB_SIZE),
            ("hidden_size", self.hidden_size, 1, MAX_EMBED_DIM),
            ("num_layers", self.num_layers, 1, MAX_LAYERS),
            ("num_heads", self.num_heads, 1, MAX_HEADS),
            ("window_size", self.window_size, 0, MAX_HEAD_DIM),
            ("ffn_dim", self.ffn_dim, 1, MAX_INTERMEDIATE_DIM),
            ("ffn_kernel_size", self.ffn_kernel_size, 1, MAX_HEAD_DIM),
            ("flow_size", self.flow_size, 1, MAX_EMBED_DIM),
            ("wavenet_layers", self.wavenet_layers, 0, MAX_LAYERS),
            ("prior_flows", self.prior_flows, 0, MAX_LAYERS),
            ("dp_dds_layers", self.dp_dds_layers, 0, MAX_LAYERS),
            ("dp_flows", self.dp_flows, 0, MAX_LAYERS),
            ("dp_bins", self.dp_bins, 1, MAX_HEAD_DIM),
            ("hop_length", self.hop_length, 1, MAX_INTERMEDIATE_DIM),
        ] {
            ensure!(
                (min..=max).contains(&value),
                "vits {name} {value} must be in {min}..={max}"
            );
        }
        ensure!(
            self.hidden_size.is_multiple_of(self.num_heads) && self.head_dim() <= MAX_HEAD_DIM,
            "vits hidden size {} does not split into {} heads of at most {MAX_HEAD_DIM}",
            self.hidden_size,
            self.num_heads
        );
        ensure!(
            self.ffn_kernel_size % 2 == 1,
            "vits ffn kernel size {} must be odd to keep the frame count",
            self.ffn_kernel_size
        );
        ensure!(self.sample_rate > 0, "vits sample rate must be non-zero");
        ensure!(
            self.dp_tail_bound.is_finite() && self.dp_tail_bound > 0.0,
            "vits dp_tail_bound must be finite and positive, got {}",
            self.dp_tail_bound
        );
        ensure!(
            self.leaky_relu_slope.is_finite(),
            "vits leaky_relu_slope must be finite"
        );
        ensure!(
            self.ln_eps.is_finite() && self.ln_eps >= 0.0,
            "vits ln_eps must be finite and non-negative, got {}",
            self.ln_eps
        );
        Ok(())
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

mod flow;
mod load;
mod math;
use math::*;

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

        // encoder layers
        let mut x = emb.clone();
        for layer in self.layers.iter() {
            let attn = layer.attention(
                &x,
                t,
                cfg.num_heads,
                cfg.head_dim(),
                cfg.window_size,
                (cfg.head_dim() as f32).sqrt().recip(),
            );
            let mut normed = x.clone();
            for (nv, av) in normed.iter_mut().zip(attn.iter()) {
                *nv += av;
            }
            layer_norm(&mut normed, t, h, &layer.ln1_w, &layer.ln1_b, cfg.ln_eps);

            // FFN: channel-major through convs
            let ct = rows_to_ct(&normed, h, t);
            let f1 = conv1d_dense(
                &CpuTensor::from_data(vec![h, t], ct),
                &layer.ffn1,
                (cfg.ffn_kernel_size - 1) / 2,
            );
            let mut f1d = f1.data().to_vec();
            relu_in_place(&mut f1d);
            let f2 = conv1d_dense(
                &CpuTensor::from_data(vec![cfg.ffn_dim, f1.shape()[1]], f1d),
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
        }
        timings.prefill_ms = t_all.elapsed().as_secs_f64() * 1e3 - timings.prompt_ms;

        // project to means/logvars (conv k1 == linear per position)
        let stats_ct = conv1d_dense(
            &CpuTensor::from_data(vec![h, t], rows_to_ct(&x, h, t)),
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
        let x_ct = rows_to_ct(&x, h, t);
        let log_duration = self.sdp.log_duration_reverse(&x_ct, h, t, cfg);
        timings.generate_ms =
            t_all.elapsed().as_secs_f64() * 1e3 - timings.prompt_ms - timings.prefill_ms;

        // durations: ceil(exp(log_d)); monotonic expansion. Durations come
        // from the weights, so their sum is bounded before it sizes
        // anything (an infinite log-duration saturates to i64::MAX).
        let mut durations = vec![1i64; t];
        let mut total_s: i64 = 0;
        for (i, &ld) in log_duration.iter().enumerate() {
            durations[i] = (ld.exp()).ceil() as i64;
            total_s = total_s.saturating_add(durations[i].max(0));
        }
        let s_len = total_s.max(1) as usize;
        ensure!(
            s_len <= MAX_SYNTH_FRAMES,
            "synthesis would expand to {s_len} frames, over the {MAX_SYNTH_FRAMES}-frame limit; split the text"
        );

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
            tr.encoder_out = Some(x);
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
        ensure!(chunk_frames > 0, "chunk_frames must be positive");
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

#[cfg(test)]
mod tests {
    use super::*;

    /// facebook/mms-tts-* geometry.
    fn mms() -> VitsConfig {
        VitsConfig {
            vocab_size: 40,
            hidden_size: 192,
            num_layers: 6,
            num_heads: 2,
            window_size: 4,
            ffn_dim: 768,
            ffn_kernel_size: 3,
            flow_size: 192,
            wavenet_layers: 4,
            prior_flows: 4,
            dp_dds_layers: 3,
            dp_flows: 4,
            dp_bins: 10,
            dp_tail_bound: 5.0,
            sample_rate: 16_000,
            hop_length: 256,
            leaky_relu_slope: 0.1,
            ln_eps: 1e-5,
        }
    }

    #[test]
    fn mms_config_is_accepted() {
        mms().validate().unwrap();
    }

    type Mutation = fn(&mut VitsConfig);

    #[test]
    fn hostile_metadata_is_an_error_not_a_panic() {
        let cases: [(&str, Mutation); 7] = [
            ("num_heads", |c| c.num_heads = 0),
            ("heads", |c| c.num_heads = 5),
            ("num_layers", |c| c.num_layers = 0),
            ("ffn_kernel_size", |c| c.ffn_kernel_size = 0),
            ("odd", |c| c.ffn_kernel_size = 4),
            ("hidden_size", |c| c.hidden_size = usize::MAX),
            ("dp_tail_bound", |c| c.dp_tail_bound = f32::INFINITY),
        ];
        for (needle, mutate) in cases {
            let mut config = mms();
            mutate(&mut config);
            let err = config.validate().expect_err(needle).to_string();
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }
}
