//! Generic ViT/SigLIP-style vision tower plus projector.
//!
//! A straightforward vision transformer: conv patch embedding (stride =
//! patch size), learned position embeddings, pre-norm layers with
//! bidirectional (full) attention, post-layer-norm, then an optional
//! pixel-shuffle + linear connector into LLM width. This is the
//! SigLIP/CLIP/LLaVA-family structure — deliberately no dynamic-resolution
//! machinery, no class token, no rope (learned positions only).
//!
//! Everything here composes existing Ember primitives (Linear, LayerNorm,
//! sgemm matmul, full attention via per-head matmul + row softmax). The
//! only additions were `gelu_tanh` (vision towers use the tanh GELU
//! approximation) and the patch-extraction itself, which lives here rather
//! than in the tensor runtime. Weights come from a mmproj-style GGUF (see
//! [`VisionModel::from_mmproj_loader`]).

use crate::backend::{Backend, CpuBackend, CpuError, Module};
use crate::loader::GgufLoader;
use crate::model::{LayerNorm, Linear};
use crate::tensor::CpuTensor;
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::time::Instant;

/// Vision-tower hyperparameters (mirrors the HF `vision_config`).
#[derive(Debug, Clone)]
pub struct VisionTransformerConfig {
    pub patch_size: usize,
    /// Input image size (square).
    pub image_size: usize,
    pub embed_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,
}

impl VisionTransformerConfig {
    /// Number of patches for a full-size input.
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }
}

/// One pre-norm vision transformer layer: LN -> attention -> residual ->
/// LN -> MLP (gelu_tanh) -> residual.
pub struct VisionLayer {
    pub(crate) ln1: LayerNorm<CpuBackend>,
    pub(crate) q_proj: Linear<CpuBackend>,
    pub(crate) k_proj: Linear<CpuBackend>,
    pub(crate) v_proj: Linear<CpuBackend>,
    pub(crate) out_proj: Linear<CpuBackend>,
    pub(crate) ln2: LayerNorm<CpuBackend>,
    pub(crate) fc1: Linear<CpuBackend>,
    pub(crate) fc2: Linear<CpuBackend>,
}

/// The full vision transformer (SigLIP-style).
pub struct VisionTransformer {
    pub config: VisionTransformerConfig,
    pub(crate) patch_embed_weight: CpuTensor, // [out, in, kh, kw] row-major
    pub(crate) patch_embed_bias: CpuTensor,   // [out]
    pub(crate) pos_embed: CpuTensor,          // [num_patches, embed]
    pub(crate) layers: Vec<VisionLayer>,
    pub(crate) post_ln: LayerNorm<CpuBackend>,
}

impl VisionTransformer {
    /// Encode `pixels` (`[n_images, 3, image_size, image_size]` normalized
    /// pixels) into patch-sequence hidden states `[n_images, num_patches,
    /// embed_dim]` after the post layer-norm.
    pub fn encode(&self, backend: &CpuBackend, pixels: &CpuTensor) -> Result<CpuTensor, CpuError> {
        self.encode_impl(backend, pixels, None, None)
    }

    /// Encode with per-image patch-validity masks (`[n_images,
    /// patches_per_side, patches_per_side]`, 1 = valid pixel region).
    ///
    /// Used by padded video frames: the reference computes *variable*
    /// position ids over the valid rectangle (bucketized fractional coords)
    /// and excludes invalid patches from every layer's attention via an
    /// additive mask. With all-valid masks this path is bit-identical to
    /// [`Self::encode`] (the bucketized grid collapses to row-major ids).
    pub fn encode_with_patch_masks(
        &self,
        backend: &CpuBackend,
        pixels: &CpuTensor,
        masks: &CpuTensor,
    ) -> Result<CpuTensor, CpuError> {
        self.encode_impl(backend, pixels, Some(masks), None)
    }

    /// Like [`Self::encode`] but records the progressive-validation
    /// intermediates: patch embeddings (after position embeddings), every
    /// layer output, and the post-norm encoder output.
    pub fn encode_traced(
        &self,
        backend: &CpuBackend,
        pixels: &CpuTensor,
    ) -> Result<(CpuTensor, VisionTrace), CpuError> {
        let mut trace = VisionTrace::default();
        let out = self.encode_impl(backend, pixels, None, Some(&mut trace))?;
        Ok((out, trace))
    }

    fn encode_impl(
        &self,
        backend: &CpuBackend,
        pixels: &CpuTensor,
        masks: Option<&CpuTensor>,
        mut trace: Option<&mut VisionTrace>,
    ) -> Result<CpuTensor, CpuError> {
        // Timings are recorded whenever a trace is requested (the ~40
        // Instant::now() calls per layer cost microseconds); the plain
        // `encode` path stays uninstrumented.
        let profile = trace.is_some();
        let mut timings = VisionOpTimings::default();
        let (n_images, channels, height, width) = (
            pixels.shape()[0],
            pixels.shape()[1],
            pixels.shape()[2],
            pixels.shape()[3],
        );
        let cfg = &self.config;
        debug_assert_eq!(channels, 3);
        debug_assert_eq!((height, width), (cfg.image_size, cfg.image_size));

        // -- patch embedding (conv, stride = patch size) --
        let t_op = Instant::now();
        let patches_per_side = height / cfg.patch_size;
        let num_patches = patches_per_side * patches_per_side;
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
        let mut patch_rows = vec![0.0f32; n_images * num_patches * patch_dim];
        for n in 0..n_images {
            for py in 0..patches_per_side {
                for px in 0..patches_per_side {
                    let row = n * num_patches + py * patches_per_side + px;
                    let out = &mut patch_rows[row * patch_dim..(row + 1) * patch_dim];
                    for c in 0..3 {
                        for y in 0..cfg.patch_size {
                            for x in 0..cfg.patch_size {
                                let src = pixels.data()[n * 3 * height * width
                                    + c * height * width
                                    + (py * cfg.patch_size + y) * width
                                    + px * cfg.patch_size
                                    + x];
                                out[c * cfg.patch_size * cfg.patch_size + y * cfg.patch_size + x] =
                                    src;
                            }
                        }
                    }
                }
            }
        }
        let patches = CpuTensor::from_data(vec![n_images * num_patches, patch_dim], patch_rows);
        if profile {
            timings.patch_embed_ms += t_op.elapsed().as_secs_f64() * 1e3;
        }

        // conv weight [out, in, kh, kw] -> matmul weight [in, out] row-major
        let t_op = Instant::now();
        let mut w = vec![0.0f32; cfg.embed_dim * patch_dim];
        for o in 0..cfg.embed_dim {
            for i in 0..patch_dim {
                w[i * cfg.embed_dim + o] = self.patch_embed_weight.data()[o * patch_dim + i];
            }
        }
        let w = CpuTensor::from_data(vec![patch_dim, cfg.embed_dim], w);
        let mut x = patches.par_matmul(&w);
        // add bias (broadcast over rows)
        for r in 0..n_images * num_patches {
            for o in 0..cfg.embed_dim {
                x.data_mut()[r * cfg.embed_dim + o] += self.patch_embed_bias.data()[o];
            }
        }
        if profile {
            timings.patch_embed_ms += t_op.elapsed().as_secs_f64() * 1e3;
        }

        // -- learned position embeddings --
        let t_op = Instant::now();
        if let Some(m) = masks {
            // Variable positions over the valid rectangle (reference
            // Idefics3VisionEmbeddings): fractional coords of valid patches
            // are bucketized into the patch grid; invalid patches keep
            // position id 0. All arithmetic is f32, matching torch.
            ensure_mask_shape(m, n_images, patches_per_side)?;
            let n_side = patches_per_side;
            for n in 0..n_images {
                let mask = &m.data()[n * n_side * n_side..(n + 1) * n_side * n_side];
                let nb_h = (0..n_side).filter(|&r| mask[r * n_side] > 0.0).count();
                let nb_w = (0..n_side).filter(|&c| mask[c] > 0.0).count();
                let step_h = 1.0f32 / nb_h as f32;
                let step_w = 1.0f32 / nb_w as f32;
                let clamp_max = 1.0f32 - 1e-6f32;
                for py in 0..n_side {
                    for px in 0..n_side {
                        let row = n * num_patches + py * n_side + px;
                        let pos = if mask[py * n_side + px] > 0.0 {
                            let fh = (py as f32 * step_h).min(clamp_max);
                            let fw = (px as f32 * step_w).min(clamp_max);
                            bucket_right_true(fh, n_side) * n_side + bucket_right_true(fw, n_side)
                        } else {
                            0
                        };
                        let dst = &mut x.data_mut()[row * cfg.embed_dim..(row + 1) * cfg.embed_dim];
                        let src =
                            &self.pos_embed.data()[pos * cfg.embed_dim..(pos + 1) * cfg.embed_dim];
                        for (d, s) in dst.iter_mut().zip(src.iter()) {
                            *d += s;
                        }
                    }
                }
            }
        } else {
            for n in 0..n_images {
                for py in 0..patches_per_side {
                    for px in 0..patches_per_side {
                        let row = n * num_patches + py * patches_per_side + px;
                        let pos = py * patches_per_side + px;
                        let dst = &mut x.data_mut()[row * cfg.embed_dim..(row + 1) * cfg.embed_dim];
                        let src =
                            &self.pos_embed.data()[pos * cfg.embed_dim..(pos + 1) * cfg.embed_dim];
                        for (d, s) in dst.iter_mut().zip(src.iter()) {
                            *d += s;
                        }
                    }
                }
            }
        }
        if let Some(trace) = trace.as_mut() {
            trace.patch_embeddings = Some(x.clone());
        }
        if profile {
            timings.pos_embed_ms += t_op.elapsed().as_secs_f64() * 1e3;
        }

        // -- transformer layers: linears batched over all images, attention
        //    per image (bidirectional, no cross-image mixing) --
        for layer in &self.layers {
            let t_op = Instant::now();
            let normed = layer.ln1.forward(backend, &x)?;
            if profile {
                timings.ln_ms += t_op.elapsed().as_secs_f64() * 1e3;
            }
            let t_op = Instant::now();
            let q = layer.q_proj.forward(backend, &normed)?;
            let k = layer.k_proj.forward(backend, &normed)?;
            let v = layer.v_proj.forward(backend, &normed)?;
            if profile {
                timings.qkv_proj_ms += t_op.elapsed().as_secs_f64() * 1e3;
            }

            let t_attn = Instant::now();
            let head_dim = cfg.embed_dim / cfg.n_heads;
            if !cfg.embed_dim.is_multiple_of(cfg.n_heads) {
                return Err(CpuError::ShapeMismatch(format!(
                    "vision embed dim {} not divisible by {} heads",
                    cfg.embed_dim, cfg.n_heads
                )));
            }
            let scale = (head_dim as f32).sqrt().recip();
            // per-image additive key bias when masks are present (0 for
            // valid patches, f32::MIN for padding — the reference's extended
            // attention mask)
            let key_biases: Vec<Vec<f32>> = match masks {
                Some(m) => (0..n_images)
                    .map(|n| {
                        let mask = &m.data()[n * num_patches..(n + 1) * num_patches];
                        mask.iter()
                            .map(|&v| if v > 0.0 { 0.0 } else { f32::MIN })
                            .collect()
                    })
                    .collect(),
                None => Vec::new(),
            };
            // parallelize over images: each worker owns one contiguous
            // [num_patches, embed_dim] block of attn_rows and loops heads
            // inside, scattering each head's [seq, head_dim] result into its
            // interleaved columns (row*embed + h*head_dim).
            let mut attn_rows = vec![0.0f32; x.len()];
            let mut splits = vec![AttentionSplit::default(); n_images];
            attn_rows
                .par_chunks_mut(num_patches * cfg.embed_dim)
                .zip(splits.par_chunks_mut(1))
                .enumerate()
                .for_each(|(n, (out_block, slot))| {
                    let row_base = n * num_patches;
                    let bias = if key_biases.is_empty() {
                        None
                    } else {
                        Some(&key_biases[n][..])
                    };
                    for h in 0..cfg.n_heads {
                        let cols = h * head_dim..(h + 1) * head_dim;
                        let qh = slice_rows_cols(&q, row_base, num_patches, cols.clone());
                        let kh = slice_rows_cols(&k, row_base, num_patches, cols.clone());
                        let vh = slice_rows_cols(&v, row_base, num_patches, cols);
                        let split = if profile { Some(&mut slot[0]) } else { None };
                        let oh = attention_head(&qh, &kh, &vh, scale, split, bias);
                        for row in 0..num_patches {
                            let dst = &mut out_block[row * cfg.embed_dim + h * head_dim
                                ..row * cfg.embed_dim + (h + 1) * head_dim];
                            dst.copy_from_slice(&oh.data()[row * head_dim..(row + 1) * head_dim]);
                        }
                    }
                });
            if profile {
                for s in &splits {
                    timings.attn_scores_ms += s.scores_ms;
                    timings.softmax_ms += s.softmax_ms;
                    timings.attn_values_ms += s.values_ms;
                }
                let total = t_attn.elapsed().as_secs_f64() * 1e3;
                let accounted =
                    timings.attn_scores_ms + timings.softmax_ms + timings.attn_values_ms;
                // slicing/copy overhead outside the three matmuls:
                timings.residual_add_ms += (total - accounted).max(0.0);
            }
            let attn = CpuTensor::from_data(vec![n_images * num_patches, cfg.embed_dim], attn_rows);
            let t_op = Instant::now();
            let attn = layer.out_proj.forward(backend, &attn)?;
            if profile {
                timings.out_proj_ms += t_op.elapsed().as_secs_f64() * 1e3;
            }
            let t_op = Instant::now();
            x = backend.add(&x, &attn)?;
            if profile {
                timings.residual_add_ms += t_op.elapsed().as_secs_f64() * 1e3;
            }

            let t_op = Instant::now();
            let normed = layer.ln2.forward(backend, &x)?;
            if profile {
                timings.ln_ms += t_op.elapsed().as_secs_f64() * 1e3;
            }
            let t_op = Instant::now();
            let hidden = layer.fc1.forward(backend, &normed)?;
            if profile {
                timings.fc1_ms += t_op.elapsed().as_secs_f64() * 1e3;
            }
            let t_op = Instant::now();
            let hidden = backend.gelu_tanh(&hidden)?;
            if profile {
                timings.gelu_ms += t_op.elapsed().as_secs_f64() * 1e3;
            }
            let t_op = Instant::now();
            let mlp = layer.fc2.forward(backend, &hidden)?;
            if profile {
                timings.fc2_ms += t_op.elapsed().as_secs_f64() * 1e3;
            }
            let t_op = Instant::now();
            x = backend.add(&x, &mlp)?;
            if profile {
                timings.residual_add_ms += t_op.elapsed().as_secs_f64() * 1e3;
            }
            if let Some(trace) = trace.as_mut() {
                trace.layer_outputs.push(x.clone());
            }
        }

        // -- post layer-norm --
        let t_op = Instant::now();
        let out = self.post_ln.forward(backend, &x)?;
        if profile {
            timings.post_ln_ms += t_op.elapsed().as_secs_f64() * 1e3;
        }
        if let Some(trace) = trace.as_mut() {
            trace.encoder_output = Some(out.clone());
            trace.op_timings = timings;
        }
        Ok(out)
    }
}

/// Progressive-validation intermediates of a vision encode.
#[derive(Debug, Default)]
pub struct VisionTrace {
    /// Patch embeddings after position embeddings, `[n * num_patches, embed]`.
    pub patch_embeddings: Option<CpuTensor>,
    /// Every transformer layer output, `[n * num_patches, embed]` each.
    pub layer_outputs: Vec<CpuTensor>,
    /// Post-norm encoder output, `[n * num_patches, embed]`.
    pub encoder_output: Option<CpuTensor>,
    /// Per-operator accumulated timings (ms). Recorded on traced encodes.
    pub op_timings: VisionOpTimings,
}

/// Accumulated per-operator milliseconds for one vision encode. Attention is
/// split into its three matmuls plus softmax; everything else is the obvious
/// stage. Recorded on traced encodes.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct VisionOpTimings {
    pub patch_embed_ms: f64,
    pub pos_embed_ms: f64,
    pub ln_ms: f64,
    pub qkv_proj_ms: f64,
    pub attn_scores_ms: f64,
    pub softmax_ms: f64,
    pub attn_values_ms: f64,
    pub out_proj_ms: f64,
    pub residual_add_ms: f64,
    pub fc1_ms: f64,
    pub gelu_ms: f64,
    pub fc2_ms: f64,
    pub post_ln_ms: f64,
}

impl VisionOpTimings {
    pub fn total_ms(&self) -> f64 {
        self.patch_embed_ms
            + self.pos_embed_ms
            + self.ln_ms
            + self.qkv_proj_ms
            + self.attn_scores_ms
            + self.softmax_ms
            + self.attn_values_ms
            + self.out_proj_ms
            + self.residual_add_ms
            + self.fc1_ms
            + self.gelu_ms
            + self.fc2_ms
            + self.post_ln_ms
    }
}

/// Internal attention-op split for profiling.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct AttentionSplit {
    pub scores_ms: f64,
    pub softmax_ms: f64,
    pub values_ms: f64,
}
/// Full (bidirectional) multi-head attention over one sequence.
///
/// `q`, `k`, `v` are `[seq, embed_dim]` with heads contiguous per row
/// (`[head][head_dim]`). Scores are scaled by `1/sqrt(head_dim)` and
/// softmaxed over the whole row — no causal mask, no block boundaries.
/// Implemented with existing primitives (matmul, softmax) because ember's
/// generic attention is causal; vision towers are the one place a
/// bidirectional encoder stack is needed, and it must not leak into the
/// language-model attention path.
pub fn bidirectional_attention(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    n_heads: usize,
) -> Result<CpuTensor, CpuError> {
    attention_impl(q, k, v, n_heads, None)
}

fn attention_impl(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    n_heads: usize,
    mut split: Option<&mut AttentionSplit>,
) -> Result<CpuTensor, CpuError> {
    let seq = q.shape()[0];
    let embed = q.shape()[1];
    if !embed.is_multiple_of(n_heads) {
        return Err(CpuError::ShapeMismatch(format!(
            "bidirectional attention: embed {embed} not divisible by {n_heads} heads"
        )));
    }
    let head_dim = embed / n_heads;
    let scale = (head_dim as f32).sqrt().recip();

    let mut out = vec![0.0f32; seq * embed];
    for h in 0..n_heads {
        let cols = h * head_dim..(h + 1) * head_dim;
        let qh = slice_rows_cols(q, 0, seq, cols.clone());
        let kh = slice_rows_cols(k, 0, seq, cols.clone());
        let vh = slice_rows_cols(v, 0, seq, cols);
        let oh = attention_head(&qh, &kh, &vh, scale, split.as_deref_mut(), None);
        for row in 0..seq {
            let dst = &mut out[row * embed + h * head_dim..row * embed + (h + 1) * head_dim];
            dst.copy_from_slice(&oh.data()[row * head_dim..(row + 1) * head_dim]);
        }
    }
    Ok(CpuTensor::from_data(vec![seq, embed], out))
}

/// Whether the vision softmax uses the fast-exp kernel. Default ON after
/// the Phase-4 error ladder + benchmarks; set `EMBER_VISION_FAST_EXP=0`
/// for the exact libm-expf reference path (bit-identical to the
/// historical behavior).
fn fast_exp_softmax_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("EMBER_VISION_FAST_EXP").is_none_or(|v| v != "0"))
}

/// Row-wise in-place softmax using [`crate::simd::fast_exp_into`]. Same
/// max-subtract / normalize structure as `CpuTensor::softmax`; the exp
/// itself is the AVX2 polynomial approximation (see simd.rs for the
/// accuracy contract). Masked rows keep working: the additive f32::MIN
/// bias clamps to exp(-88) ≈ 6e-39, indistinguishable from zero at f32.
pub fn softmax_in_place_fast(t: &mut CpuTensor) {
    assert!(t.shape().len() >= 2, "softmax needs 2 dims min");
    let cols = t.shape()[t.shape().len() - 1];
    let rows = t.len() / cols;
    let data = t.data_mut();
    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];
        let max = row.iter().fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));
        if max == f32::NEG_INFINITY {
            let uniform = 1.0 / cols as f32;
            row.fill(uniform);
            continue;
        }
        for v in row.iter_mut() {
            *v -= max;
        }
        crate::simd::fast_exp_in_place(row);
        let inv_sum = row.iter().sum::<f32>().recip();
        for v in row.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// One head of full attention: `qh`, `kh`, `vh` are `[seq, head_dim]`.
/// Returns `[seq, head_dim]`. Optional split records sub-op wall time.
///
/// `key_bias`, when given, is an additive per-key bias `[seq_k]` applied to
/// the scaled scores before softmax (encoder-style padding masks: 0 for
/// valid keys, a large negative value for padding). The unmasked path is
/// bit-identical to the historical kernel.
pub(crate) fn attention_head(
    qh: &CpuTensor,
    kh: &CpuTensor,
    vh: &CpuTensor,
    scale: f32,
    mut split: Option<&mut AttentionSplit>,
    key_bias: Option<&[f32]>,
) -> CpuTensor {
    let t_scores = Instant::now();
    // scores [seq, seq] = qh * kh^T * scale
    let mut scores = qh.matmul(&kh.transpose());
    for s in scores.data_mut() {
        *s *= scale;
    }
    if let Some(bias) = key_bias {
        let klen = bias.len();
        let seq = scores.shape()[0];
        for r in 0..seq {
            let row = &mut scores.data_mut()[r * klen..(r + 1) * klen];
            for (j, s) in row.iter_mut().enumerate() {
                *s += bias[j];
            }
        }
    }
    if let Some(sp) = split.as_deref_mut() {
        sp.scores_ms += t_scores.elapsed().as_secs_f64() * 1e3;
    }
    let t_soft = Instant::now();
    let probs = if fast_exp_softmax_enabled() {
        softmax_in_place_fast(&mut scores);
        scores
    } else {
        scores.softmax()
    };
    if let Some(sp) = split.as_deref_mut() {
        sp.softmax_ms += t_soft.elapsed().as_secs_f64() * 1e3;
    }
    let t_val = Instant::now();
    let oh = probs.matmul(vh); // [seq, head_dim]
    if let Some(sp) = split {
        sp.values_ms += t_val.elapsed().as_secs_f64() * 1e3;
    }
    oh
}

/// Copy a `[rows, cols.len()]` block starting at data row `row_start`,
/// columns `cols`, from a row-major `[total_rows, width]` tensor.
pub(crate) fn slice_rows_cols(
    t: &CpuTensor,
    row_start: usize,
    rows: usize,
    cols: std::ops::Range<usize>,
) -> CpuTensor {
    let width = t.shape()[1];
    let mut out = vec![0.0f32; rows * cols.len()];
    for r in 0..rows {
        let src = (row_start + r) * width;
        out[r * cols.len()..(r + 1) * cols.len()]
            .copy_from_slice(&t.data()[src + cols.start..src + cols.end]);
    }
    CpuTensor::from_data(vec![rows, cols.len()], out)
}

/// Pixel-shuffle + linear connector (SmolVLM/Idefics3 style).
///
/// Rearranges `[seq, embed]` patch tokens (seq a perfect square) into
/// `[seq / scale^2, embed * scale^2]` tokens, then projects to LLM width.
pub struct PixelShuffleConnector {
    pub scale_factor: usize,
    pub proj: Linear<CpuBackend>,
}

impl PixelShuffleConnector {
    /// `x` is `[n_images * num_patches, embed]`; returns
    /// `[n_images * tokens_out, llm_width]`.
    pub fn forward(
        &self,
        backend: &CpuBackend,
        x: &CpuTensor,
        num_patches: usize,
    ) -> Result<CpuTensor, CpuError> {
        let rows = x.shape()[0];
        let embed = x.shape()[1];
        let n_images = rows / num_patches;
        let scale = self.scale_factor;
        if !rows.is_multiple_of(num_patches) || !num_patches.is_multiple_of(scale * scale) {
            return Err(CpuError::ShapeMismatch(format!(
                "pixel shuffle: rows {rows} / patches {num_patches} / scale {scale} geometry"
            )));
        }
        let side = (num_patches as f64).sqrt() as usize;
        if side * side != num_patches {
            return Err(CpuError::ShapeMismatch(format!(
                "pixel shuffle needs a square patch grid, got {num_patches}"
            )));
        }
        let tokens_per_image = num_patches / (scale * scale);
        let mut shuffled = vec![0.0f32; n_images * tokens_per_image * embed * scale * scale];

        // HF pixel_shuffle: view(bsz, h, w, e) -> view(bsz, h, w/s, e*s)
        // -> permute(0,2,1,3) -> reshape(bsz, w/s, h/s, e*s*s) ->
        // permute(0,2,1,3) -> reshape(bsz, seq/s^2, e*s^2)
        let s = scale;
        for n in 0..n_images {
            for py in 0..side {
                for px in 0..side {
                    let src_row = n * num_patches + py * side + px;
                    // destination in the permuted layout:
                    // new_w = px / s, new_h = py / s (after both permutes)
                    let ny = py / s;
                    let nx = px / s;
                    // channel group: within the 4D view (h, w/s, e*s):
                    // after first view, channel index c1 = (px % s) * embed + e
                    // after reshape/permute, output feature index:
                    //   f = (py % s) * (s * embed) + (px % s) * embed + e
                    let dst_row = n * tokens_per_image + ny * (side / s) + nx;
                    for e in 0..embed {
                        let src_v = x.data()[src_row * embed + e];
                        let dst_feature = (py % s) * (s * embed) + (px % s) * embed + e;
                        shuffled[dst_row * (embed * s * s) + dst_feature] = src_v;
                    }
                }
            }
        }
        let shuffled =
            CpuTensor::from_data(vec![n_images * tokens_per_image, embed * s * s], shuffled);
        self.proj.forward(backend, &shuffled)
    }
}

/// Vision tower + connector as one unit.
pub struct VisionModel {
    pub transformer: VisionTransformer,
    pub connector: PixelShuffleConnector,
}

impl VisionModel {
    /// Load a vision model from a mmproj-style GGUF.
    ///
    /// Expected metadata (`smolvlm.*` keys) and tensor layout are documented
    /// in `docs/multimodal-foundation-plan.md`; tensors mirror the HF state
    /// dict under a `v.` prefix with GGUF `[in, out]` linear dims (the same
    /// convention the llama loader uses).
    pub fn from_mmproj_loader(loader: &mut GgufLoader) -> Result<Self> {
        use crate::loader::{gguf_to_row_major_f32, GgufValue};

        let get_u32 = |key: &str| -> Result<usize> {
            match loader.metadata.get(key) {
                Some(GgufValue::U32(v)) => Ok(*v as usize),
                Some(other) => anyhow::bail!("{key} must be U32, got {other:?}"),
                None => anyhow::bail!("mmproj missing required metadata {key}"),
            }
        };
        let get_f32 = |key: &str| -> Result<f32> {
            match loader.metadata.get(key) {
                Some(GgufValue::F32(v)) => Ok(*v),
                Some(other) => anyhow::bail!("{key} must be F32, got {other:?}"),
                None => anyhow::bail!("mmproj missing required metadata {key}"),
            }
        };

        let patch_size = get_u32("smolvlm.vision.patch_size")?;
        let image_size = get_u32("smolvlm.vision.image_size")?;
        let embed_dim = get_u32("smolvlm.vision.hidden_size")?;
        let n_layers = get_u32("smolvlm.vision.num_hidden_layers")?;
        let n_heads = get_u32("smolvlm.vision.num_attention_heads")?;
        let intermediate_size = get_u32("smolvlm.vision.intermediate_size")?;
        let norm_eps = get_f32("smolvlm.vision.layer_norm_eps")?;
        let scale_factor = get_u32("smolvlm.scale_factor")?;
        let config = VisionTransformerConfig {
            patch_size,
            image_size,
            embed_dim,
            n_layers,
            n_heads,
            intermediate_size,
            norm_eps,
        };

        let take_linear = |loader: &mut GgufLoader, name: &str| -> Result<Linear<CpuBackend>> {
            let weight_name = format!("{name}.weight");
            let weight = loader
                .take_f32(&weight_name)
                .with_context(|| format!("mmproj missing tensor {weight_name}"))?;
            let bias = loader.take_optional_f32(&[format!("{name}.bias")]);
            let weight = gguf_to_row_major_f32(weight);
            Ok(Linear::new(weight, bias))
        };
        let take_norm =
            |loader: &mut GgufLoader, name: &str, eps: f32| -> Result<LayerNorm<CpuBackend>> {
                let weight = loader
                    .take_f32(&format!("{name}.weight"))
                    .with_context(|| format!("mmproj missing tensor {name}.weight"))?;
                let bias = loader
                    .take_f32(&format!("{name}.bias"))
                    .with_context(|| format!("mmproj missing tensor {name}.bias"))?;
                Ok(LayerNorm::new(weight, bias, eps))
            };

        // patch embedding: 4-D conv weight. GGUF dims are the reversed HF
        // shape [kw, kh, in, out]; the payload is HF row-major
        // [out][in][kh][kw] (kw fastest), which is what the patch-extraction
        // loop indexes.
        let patch_embed_weight = loader
            .take_f32("v.vision.embeddings.patch_embedding.weight")
            .context("mmproj missing patch_embedding.weight")?;
        anyhow::ensure!(
            patch_embed_weight.shape() == [patch_size, patch_size, 3, embed_dim],
            "patch_embedding.weight gguf dims {:?} != [{patch_size}, {patch_size}, 3, {embed_dim}]",
            patch_embed_weight.shape()
        );
        let patch_embed_bias = loader
            .take_f32("v.vision.embeddings.patch_embedding.bias")
            .context("mmproj missing patch_embedding.bias")?;
        // GGUF dims are reversed HF shape: [embed, num_patches]; payload is
        // HF row-major [num_patches][embed].
        let pos_embed = loader
            .take_f32("v.vision.embeddings.position_embedding.weight")
            .context("mmproj missing position_embedding.weight")?;
        anyhow::ensure!(
            pos_embed.shape() == [embed_dim, config.num_patches()],
            "position_embedding.weight gguf dims {:?} != [{embed_dim}, {}]",
            pos_embed.shape(),
            config.num_patches()
        );

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("v.vision.encoder.layers.{i}.");
            layers.push(VisionLayer {
                ln1: take_norm(loader, &format!("{p}layer_norm1"), norm_eps)?,
                q_proj: take_linear(loader, &format!("{p}self_attn.q_proj"))?,
                k_proj: take_linear(loader, &format!("{p}self_attn.k_proj"))?,
                v_proj: take_linear(loader, &format!("{p}self_attn.v_proj"))?,
                out_proj: take_linear(loader, &format!("{p}self_attn.out_proj"))?,
                ln2: take_norm(loader, &format!("{p}layer_norm2"), norm_eps)?,
                fc1: take_linear(loader, &format!("{p}mlp.fc1"))?,
                fc2: take_linear(loader, &format!("{p}mlp.fc2"))?,
            });
        }
        let post_ln = take_norm(loader, "v.vision.post_layernorm", norm_eps)?;

        let proj_weight = loader
            .take_f32("v.connector.modality_projection.proj.weight")
            .context("mmproj missing connector proj.weight")?;
        let llm_width = proj_weight.shape()[1];
        let proj = Linear::new(gguf_to_row_major_f32(proj_weight), None);

        let transformer = VisionTransformer {
            config,
            patch_embed_weight,
            patch_embed_bias,
            pos_embed,
            layers,
            post_ln,
        };
        let connector = PixelShuffleConnector { scale_factor, proj };
        let _ = llm_width;
        Ok(Self {
            transformer,
            connector,
        })
    }

    /// LLM embedding width this connector projects to.
    pub fn llm_width(&self, backend: &CpuBackend) -> usize {
        self.connector.proj.out_features(backend)
    }

    /// Full encode: `pixels [n, 3, size, size]` ->
    /// `[n * tokens_per_image, llm_width]` visual embeddings.
    pub fn encode(&self, backend: &CpuBackend, pixels: &CpuTensor) -> Result<CpuTensor, CpuError> {
        let hidden = self.transformer.encode(backend, pixels)?;
        let num_patches = self.transformer.config.num_patches();
        self.connector.forward(backend, &hidden, num_patches)
    }

    /// [`Self::encode`] with per-image patch masks (padded video frames).
    pub fn encode_masked(
        &self,
        backend: &CpuBackend,
        pixels: &CpuTensor,
        masks: &CpuTensor,
    ) -> Result<CpuTensor> {
        let hidden = self
            .transformer
            .encode_with_patch_masks(backend, pixels, masks)?;
        let num_patches = self.transformer.config.num_patches();
        Ok(self.connector.forward(backend, &hidden, num_patches)?)
    }
}

/// torch.bucketize(v, boundaries, right=True) over boundaries
/// `k/n_side` for k in 1..n_side: the count of boundaries <= v. For the
/// SmolVLM grid (n_side = 32) every boundary is exactly representable in
/// f32, so this matches torch bit-for-bit.
fn bucket_right_true(v: f32, n_side: usize) -> usize {
    let mut idx = 0usize;
    for k in 1..n_side {
        if (k as f32 / n_side as f32) <= v {
            idx += 1;
        } else {
            break;
        }
    }
    idx
}

fn ensure_mask_shape(
    m: &CpuTensor,
    n_images: usize,
    patches_per_side: usize,
) -> Result<(), CpuError> {
    if m.shape() != [n_images, patches_per_side, patches_per_side] {
        return Err(CpuError::ShapeMismatch(format!(
            "patch mask shape {:?} != [{n_images}, {patches_per_side}, {patches_per_side}]",
            m.shape()
        )));
    }
    Ok(())
}
