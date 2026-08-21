//! Ultravox audio encoder (a Whisper-style transformer encoder) plus its
//! SwiGLU projector.
//!
//! This is the audio counterpart of [`crate::multimodal::vision`]: it
//! composes existing Ember primitives (Linear, LayerNorm, sgemm/par_matmul,
//! erf-GELU, RMS norm, full attention via per-head matmul + row softmax)
//! and never touches the language-model path. Weights come from an audio
//! mmproj GGUF produced by `tools/convert_ultravox_audio.py`
//! (metadata `ultravox.audio.*`, tensors prefixed `a.`).
//!
//! Structure (HuggingFace `WhisperEncoder` as used by Ultravox v0.5):
//!
//! ```text
//! mel [128, T]  (T <= 3000)
//!   -> conv1 (128->1280, k3 p1) -> gelu(erf)          [C, T]
//!   -> conv2 (1280->1280, k3 p1 s2) -> gelu(erf)      [1280, T2], T2 = ceil(T/2)
//!   -> transpose + position_embedding[:T2]            [T2, d_model]
//!   -> 32 x (LN -> full attention -> residual; LN -> fc1 -> gelu -> fc2 -> residual)
//!   -> final LayerNorm                                [T2, d_model]
//!   -> projector: pad+stack frames by 8 -> RMSNorm -> linear(10240->4096)
//!      -> swiglu (silu(second half) * first half) -> RMSNorm -> linear(2048->2048)
//! ```

use crate::backend::{Backend, CpuBackend, CpuError, Module};
use crate::loader::{gguf_to_row_major_f32, GgufLoader};
use crate::model::{LayerNorm, Linear};
use crate::multimodal::vision::{attention_head, slice_rows_cols};
use crate::tensor::CpuTensor;
use anyhow::{Context, Result};
use rayon::prelude::*;

/// Whisper-encoder hyperparameters (mirrors `ultravox.audio.*` metadata).
#[derive(Debug, Clone)]
pub struct AudioEncoderConfig {
    pub num_mel_bins: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub max_source_positions: usize,
    pub layer_norm_eps: f32,
}

impl AudioEncoderConfig {
    /// Encoder output frames for `mel_frames` input frames.
    pub fn out_frames(&self, mel_frames: usize) -> usize {
        mel_frames.div_ceil(2)
    }
}

/// One pre-norm Whisper encoder layer.
pub struct AudioEncoderLayer {
    self_attn_layer_norm: LayerNorm<CpuBackend>,
    q_proj: Linear<CpuBackend>,
    k_proj: Linear<CpuBackend>,
    v_proj: Linear<CpuBackend>,
    out_proj: Linear<CpuBackend>,
    final_layer_norm: LayerNorm<CpuBackend>,
    fc1: Linear<CpuBackend>,
    fc2: Linear<CpuBackend>,
}

/// The Whisper encoder stack.
pub struct AudioEncoder {
    pub config: AudioEncoderConfig,
    /// Conv1d weights `[out_channels, in_channels, kernel]` (HF row-major),
    /// biases `[out_channels]`.
    conv1_weight: CpuTensor,
    conv1_bias: CpuTensor,
    conv2_weight: CpuTensor,
    conv2_bias: CpuTensor,
    /// Learned positions `[max_source_positions, d_model]`.
    pos_embed: CpuTensor,
    layers: Vec<AudioEncoderLayer>,
    final_norm: LayerNorm<CpuBackend>,
}

/// Progressive-validation intermediates of one audio encode + project.
#[derive(Debug, Default)]
pub struct AudioTrace {
    /// After conv1+gelu, `[d_model, T]`.
    pub conv1_output: Option<CpuTensor>,
    /// Every encoder layer output, `[T2, d_model]` each.
    pub layer_outputs: Vec<CpuTensor>,
    /// Final-layer-norm output, `[T2, d_model]`.
    pub encoder_output: Option<CpuTensor>,
}

impl AudioEncoder {
    /// Encode log-mel features `[n_mels, T]` into hidden states
    /// `[T2, d_model]`, `T2 = ceil(T / 2)`.
    pub fn encode(&self, backend: &CpuBackend, mel: &CpuTensor) -> Result<CpuTensor, CpuError> {
        self.encode_impl(backend, mel, None)
    }

    /// Like [`Self::encode`] but records progressive-validation
    /// intermediates.
    pub fn encode_traced(
        &self,
        backend: &CpuBackend,
        mel: &CpuTensor,
    ) -> Result<(CpuTensor, AudioTrace), CpuError> {
        let mut trace = AudioTrace::default();
        let out = self.encode_impl(backend, mel, Some(&mut trace))?;
        Ok((out, trace))
    }

    fn encode_impl(
        &self,
        backend: &CpuBackend,
        mel: &CpuTensor,
        mut trace: Option<&mut AudioTrace>,
    ) -> Result<CpuTensor, CpuError> {
        let cfg = &self.config;
        assert_eq!(mel.shape()[0], cfg.num_mel_bins, "mel bins mismatch");
        let t_len = mel.shape()[1];
        if !(4..=cfg.max_source_positions * 2).contains(&t_len) {
            return Err(CpuError::ShapeMismatch(format!(
                "audio encode: mel length {t_len} outside [4, {}]",
                cfg.max_source_positions * 2
            )));
        }

        // -- conv frontend --
        let x = backend.gelu(&self.conv1d(backend, mel, 1)?)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.conv1_output = Some(x.clone());
        }
        let x = self.conv1d(backend, &x, 2)?;
        let x_ct = backend.gelu(&x)?; // [d_model, T2]

        // -- transpose to [T2, d_model] and add learned positions --
        let t2 = x_ct.shape()[1];
        if t2 > self.pos_embed.shape()[0] {
            return Err(CpuError::ShapeMismatch(format!(
                "audio encode: {t2} frames exceed position table {}",
                self.pos_embed.shape()[0]
            )));
        }
        let mut rows = vec![0.0f32; t2 * cfg.d_model];
        rows.par_chunks_mut(cfg.d_model)
            .enumerate()
            .for_each(|(t, row)| {
                let src = &x_ct.data()[t..];
                for (d, slot) in row.iter_mut().enumerate() {
                    *slot = src[d * t2] + self.pos_embed.data()[t * cfg.d_model + d];
                }
            });
        let mut hidden = CpuTensor::from_data(vec![t2, cfg.d_model], rows);

        // -- transformer layers --
        let head_dim = cfg.d_model / cfg.n_heads;
        if !cfg.d_model.is_multiple_of(cfg.n_heads) {
            return Err(CpuError::ShapeMismatch(format!(
                "audio encoder: d_model {} not divisible by {} heads",
                cfg.d_model, cfg.n_heads
            )));
        }
        let scale = (head_dim as f32).sqrt().recip();
        for layer in &self.layers {
            let normed = layer.self_attn_layer_norm.forward(backend, &hidden)?;

            // full attention parallel over heads; each head works on its own
            // column block of the shared [T2, d_model] tensors
            let q = layer.q_proj.forward(backend, &normed)?;
            let k = layer.k_proj.forward(backend, &normed)?;
            let v = layer.v_proj.forward(backend, &normed)?;
            let mut attn_rows = vec![0.0f32; q.len()];
            attn_rows
                .par_chunks_mut(t2 * head_dim)
                .enumerate()
                .for_each(|(h, out_block)| {
                    let cols = h * head_dim..(h + 1) * head_dim;
                    let qh = slice_rows_cols(&q, 0, t2, cols.clone());
                    let kh = slice_rows_cols(&k, 0, t2, cols.clone());
                    let vh = slice_rows_cols(&v, 0, t2, cols);
                    let oh = attention_head(&qh, &kh, &vh, scale, None);
                    for row in 0..t2 {
                        let dst = &mut out_block[row * head_dim..(row + 1) * head_dim];
                        dst.copy_from_slice(&oh.data()[row * head_dim..(row + 1) * head_dim]);
                    }
                });
            // interleave head blocks back into [T2, d_model]
            let mut interleaved = vec![0.0f32; q.len()];
            for h in 0..cfg.n_heads {
                for r in 0..t2 {
                    let src_off = h * t2 * head_dim + r * head_dim;
                    let dst_off = r * cfg.d_model + h * head_dim;
                    interleaved[dst_off..dst_off + head_dim]
                        .copy_from_slice(&attn_rows[src_off..src_off + head_dim]);
                }
            }
            let attn = CpuTensor::from_data(vec![t2, cfg.d_model], interleaved);
            let attn = layer.out_proj.forward(backend, &attn)?;
            let added = backend.add(&hidden, &attn)?;
            hidden = added;

            let normed = layer.final_layer_norm.forward(backend, &hidden)?;
            let h1 = layer.fc1.forward(backend, &normed)?;
            let act = backend.gelu(&h1)?;
            let h2 = layer.fc2.forward(backend, &act)?;
            hidden = backend.add(&hidden, &h2)?;

            if let Some(trace) = trace.as_deref_mut() {
                trace.layer_outputs.push(hidden.clone());
            }
        }

        // -- final layer norm --
        let out = self.final_norm.forward(backend, &hidden)?;
        if let Some(trace) = trace {
            trace.encoder_output = Some(out.clone());
        }
        Ok(out)
    }

    /// Conv1d with kernel 3, padding 1, given stride, via im2col +
    /// parallel matmul. Input `[in_channels, T]` -> output
    /// `[out_channels, T']` (channel-major, matching HF layout).
    fn conv1d(
        &self,
        backend: &CpuBackend,
        input: &CpuTensor,
        stride: usize,
    ) -> Result<CpuTensor, CpuError> {
        let weight = if stride == 1 {
            &self.conv1_weight
        } else {
            &self.conv2_weight
        };
        let bias = if stride == 1 {
            &self.conv1_bias
        } else {
            &self.conv2_bias
        };
        let (out_ch, in_ch, kernel) = (weight.shape()[0], weight.shape()[1], weight.shape()[2]);
        assert_eq!(kernel, 3);
        let in_len = input.shape()[1];
        let out_len = (in_len + 2 - 3) / stride + 1;

        // im2col: row per output position, features [channel][kernel tap]
        let feat_dim = in_ch * kernel;
        let mut cols = vec![0.0f32; out_len * feat_dim];
        cols.par_chunks_mut(feat_dim)
            .enumerate()
            .for_each(|(t, row)| {
                for c in 0..in_ch {
                    for kk in 0..kernel {
                        // pad=1: window starts one sample before t*stride
                        let src_t = (t * stride + kk).checked_sub(1);
                        let val = match src_t {
                            Some(st) if st < in_len => input.data()[c * in_len + st],
                            _ => 0.0,
                        };
                        row[c * kernel + kk] = val;
                    }
                }
            });

        // weight [out, in, k] -> matmul weight [feat_dim, out]: w'[f][o] =
        // weight[o][f] where f indexes [c][k]
        let mut wm = vec![0.0f32; feat_dim * out_ch];
        for o in 0..out_ch {
            for f in 0..feat_dim {
                wm[f * out_ch + o] = weight.data()[o * feat_dim + f];
            }
        }
        let cols_t = CpuTensor::from_data(vec![out_len, feat_dim], cols);
        let w_t = CpuTensor::from_data(vec![feat_dim, out_ch], wm);
        let mut out = cols_t.par_matmul(&w_t);

        // bias broadcast
        let data = out.data_mut();
        for t in 0..out_len {
            let row = &mut data[t * out_ch..(t + 1) * out_ch];
            for (o, slot) in row.iter_mut().enumerate() {
                *slot += bias.data()[o];
            }
        }

        // back to channel-major [out_ch, out_len]
        let mut ct = vec![0.0f32; out_len * out_ch];
        for t in 0..out_len {
            for o in 0..out_ch {
                ct[o * out_len + t] = data[t * out_ch + o];
            }
        }
        let _ = backend;
        Ok(CpuTensor::from_data(vec![out_ch, out_len], ct))
    }
}

/// Ultravox SwiGLU projector: stack-8 -> RMSNorm -> linear -> swiglu ->
/// RMSNorm -> linear.
pub struct UltravoxProjector {
    pub stack_factor: usize,
    ln_pre_weight: CpuTensor,
    linear_1: Linear<CpuBackend>,
    ln_mid_weight: CpuTensor,
    linear_2: Linear<CpuBackend>,
    rms_eps: f32,
}

impl UltravoxProjector {
    /// Project encoder frames `[frames, d_model]` to LLM-width tokens.
    ///
    /// Returns `ceil(frames / stack_factor)` rows; zero-padding at the end
    /// when `frames` is not a multiple of the stack factor (exactly the
    /// reference `StackAudioFrames` behavior).
    pub fn forward(&self, backend: &CpuBackend, frames: &CpuTensor) -> Result<CpuTensor, CpuError> {
        let (n_frames, dim) = (frames.shape()[0], frames.shape()[1]);
        let s = self.stack_factor;
        let n_tokens = n_frames.div_ceil(s);
        let stacked_dim = dim * s;

        // stack consecutive frames per token: token j takes frames
        // [j*s .. j*s+s) concatenated feature-wise, zero-padded at the end
        let mut stacked = vec![0.0f32; n_tokens * stacked_dim];
        for j in 0..n_tokens {
            for f in 0..s {
                let frame = j * s + f;
                if frame >= n_frames {
                    break; // remaining taps stay zero (reference pads with zeros)
                }
                let dst = j * stacked_dim + f * dim;
                stacked[dst..dst + dim]
                    .copy_from_slice(&frames.data()[frame * dim..(frame + 1) * dim]);
            }
        }
        let stacked = CpuTensor::from_data(vec![n_tokens, stacked_dim], stacked);

        // RMSNorm(pre) -> linear_1 -> swiglu -> RMSNorm(mid) -> linear_2
        let normed = stacked.rms_norm(&self.ln_pre_weight, self.rms_eps);
        let h = self.linear_1.forward(backend, &normed)?;
        let half = h.shape()[1] / 2;
        let (rows, cols) = (h.shape()[0], h.shape()[1]);
        // swiglu (reference convention): value = first half, gate = second
        // half, output = silu(gate) * value, width halved
        let mut activated = vec![0.0f32; rows * half];
        for r in 0..rows {
            let row = &h.data()[r * cols..(r + 1) * cols];
            for c in 0..half {
                let value = row[c];
                let gate = row[c + half];
                activated[r * half + c] = gate / (1.0 + (-gate).exp()) * value;
            }
        }
        let activated = CpuTensor::from_data(vec![rows, half], activated);

        let normed_mid = activated.rms_norm(&self.ln_mid_weight, self.rms_eps);
        self.linear_2.forward(backend, &normed_mid)
    }
}

/// The full audio front-end: encoder + projector.
pub struct AudioModel {
    pub encoder: AudioEncoder,
    pub projector: UltravoxProjector,
}

/// Reverse a GGUF-loaded tensor's dimension order, keeping the flat
/// payload (which is already HF row-major). GGUF reports dims
/// fastest-first, so a Conv1d `[out, in, k]` arrives as `[k, in, out]`;
/// this restores the HF shape without touching the data.
fn gguf_to_hf_layout(t: &CpuTensor) -> CpuTensor {
    let mut shape = t.shape().to_vec();
    shape.reverse();
    CpuTensor::from_data(shape, t.data().to_vec())
}

impl AudioModel {
    /// Load from an audio mmproj GGUF (see `tools/convert_ultravox_audio.py`).
    pub fn from_mmproj_loader(loader: &mut GgufLoader) -> Result<Self> {
        use crate::loader::GgufValue;

        let get_u32 = |key: &str| -> Result<usize> {
            match loader.metadata.get(key) {
                Some(GgufValue::U32(v)) => Ok(*v as usize),
                Some(other) => anyhow::bail!("{key} must be U32, got {other:?}"),
                None => anyhow::bail!("audio mmproj missing required metadata {key}"),
            }
        };
        let get_f32 = |key: &str| -> Result<f32> {
            match loader.metadata.get(key) {
                Some(GgufValue::F32(v)) => Ok(*v),
                Some(other) => anyhow::bail!("{key} must be F32, got {other:?}"),
                None => anyhow::bail!("audio mmproj missing required metadata {key}"),
            }
        };

        let num_mel_bins = get_u32("ultravox.audio.num_mel_bins")?;
        let d_model = get_u32("ultravox.audio.d_model")?;
        let n_layers = get_u32("ultravox.audio.encoder_layers")?;
        let ffn_dim = get_u32("ultravox.audio.encoder_ffn_dim")?;
        let max_pos = get_u32("ultravox.audio.max_source_positions")?;
        let eps = get_f32("ultravox.audio.layer_norm_eps")?;
        let stack_factor = get_u32("ultravox.stack_factor")?;
        // whisper-large-v3-turbo uses 20 heads at d_model 1280
        let n_heads = match d_model {
            1280 => 20,
            other => anyhow::bail!("unsupported audio tower d_model {other}: add its head count"),
        };
        let config = AudioEncoderConfig {
            num_mel_bins,
            d_model,
            n_layers,
            n_heads,
            ffn_dim,
            max_source_positions: max_pos,
            layer_norm_eps: eps,
        };

        let take_linear = |loader: &mut GgufLoader, name: &str| -> Result<Linear<CpuBackend>> {
            let weight_name = format!("{name}.weight");
            let weight = loader
                .take_f32(&weight_name)
                .with_context(|| format!("audio mmproj missing tensor {weight_name}"))?;
            let bias = loader.take_optional_f32(&[format!("{name}.bias")]);
            let weight = gguf_to_row_major_f32(weight);
            Ok(Linear::new(weight, bias))
        };
        let take_linear_nb = |loader: &mut GgufLoader, name: &str| -> Result<Linear<CpuBackend>> {
            let weight_name = format!("{name}.weight");
            let weight = loader
                .take_f32(&weight_name)
                .with_context(|| format!("audio mmproj missing tensor {weight_name}"))?;
            let weight = gguf_to_row_major_f32(weight);
            Ok(Linear::new(weight, None))
        };
        let take_norm = |loader: &mut GgufLoader, name: &str| -> Result<LayerNorm<CpuBackend>> {
            let weight = loader
                .take_f32(&format!("{name}.weight"))
                .with_context(|| format!("audio mmproj missing {name}.weight"))?;
            let bias = loader
                .take_f32(&format!("{name}.bias"))
                .with_context(|| format!("audio mmproj missing {name}.bias"))?;
            Ok(LayerNorm::new(weight, bias, eps))
        };
        let take_vec = |loader: &mut GgufLoader, name: &str| -> Result<CpuTensor> {
            loader
                .take_f32(name)
                .with_context(|| format!("audio mmproj missing tensor {name}"))
        };

        let conv1_weight = gguf_to_hf_layout(&take_vec(loader, "a.audio_tower.conv1.weight")?);
        let conv1_bias = take_vec(loader, "a.audio_tower.conv1.bias")?;
        let conv2_weight = gguf_to_hf_layout(&take_vec(loader, "a.audio_tower.conv2.weight")?);
        let conv2_bias = take_vec(loader, "a.audio_tower.conv2.bias")?;
        let pos_embed = gguf_to_hf_layout(&take_vec(
            loader,
            "a.audio_tower.position_embedding.weight",
        )?);

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("a.audio_tower.layers.{i}");
            layers.push(AudioEncoderLayer {
                self_attn_layer_norm: take_norm(loader, &format!("{p}.self_attn_layer_norm"))?,
                q_proj: take_linear(loader, &format!("{p}.self_attn.q_proj"))?,
                k_proj: take_linear(loader, &format!("{p}.self_attn.k_proj"))?,
                v_proj: take_linear(loader, &format!("{p}.self_attn.v_proj"))?,
                out_proj: take_linear(loader, &format!("{p}.self_attn.out_proj"))?,
                final_layer_norm: take_norm(loader, &format!("{p}.final_layer_norm"))?,
                fc1: take_linear(loader, &format!("{p}.fc1"))?,
                fc2: take_linear(loader, &format!("{p}.fc2"))?,
            });
        }
        let final_norm = take_norm(loader, "a.audio_tower.layer_norm")?;

        let ln_pre_weight = take_vec(loader, "a.projector.ln_pre.weight")?;
        let linear_1 = take_linear_nb(loader, "a.projector.linear_1")?;
        let ln_mid_weight = take_vec(loader, "a.projector.ln_mid.weight")?;
        let linear_2 = take_linear_nb(loader, "a.projector.linear_2")?;

        Ok(Self {
            encoder: AudioEncoder {
                config,
                conv1_weight,
                conv1_bias,
                conv2_weight,
                conv2_bias,
                pos_embed,
                layers,
                final_norm,
            },
            projector: UltravoxProjector {
                stack_factor,
                ln_pre_weight,
                linear_1,
                ln_mid_weight,
                linear_2,
                rms_eps: 1e-6,
            },
        })
    }

    /// Encode + project: mel `[n_mels, T]` -> `[ceil(ceil(T/2)/8), llm_width]`.
    pub fn encode_and_project(
        &self,
        backend: &CpuBackend,
        mel: &CpuTensor,
    ) -> Result<(CpuTensor, AudioTrace), CpuError> {
        let (encoder_out, trace) = self.encoder.encode_traced(backend, mel)?;
        let projected = self.projector.forward(backend, &encoder_out)?;
        Ok((projected, trace))
    }
}
