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
        self.encode_impl(backend, pixels, None)
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
        let out = self.encode_impl(backend, pixels, Some(&mut trace))?;
        Ok((out, trace))
    }

    fn encode_impl(
        &self,
        backend: &CpuBackend,
        pixels: &CpuTensor,
        mut trace: Option<&mut VisionTrace>,
    ) -> Result<CpuTensor, CpuError> {
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

        // conv weight [out, in, kh, kw] -> matmul weight [in, out] row-major
        let mut w = vec![0.0f32; cfg.embed_dim * patch_dim];
        for o in 0..cfg.embed_dim {
            for i in 0..patch_dim {
                w[i * cfg.embed_dim + o] = self.patch_embed_weight.data()[o * patch_dim + i];
            }
        }
        let w = CpuTensor::from_data(vec![patch_dim, cfg.embed_dim], w);
        let mut x = patches.matmul(&w);
        // add bias (broadcast over rows)
        for r in 0..n_images * num_patches {
            for o in 0..cfg.embed_dim {
                x.data_mut()[r * cfg.embed_dim + o] += self.patch_embed_bias.data()[o];
            }
        }

        // -- learned position embeddings (identity grid for full images) --
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
        if let Some(trace) = trace.as_mut() {
            trace.patch_embeddings = Some(x.clone());
        }

        // -- transformer layers: linears batched over all images, attention
        //    per image (bidirectional, no cross-image mixing) --
        for layer in &self.layers {
            let normed = layer.ln1.forward(backend, &x)?;
            let q = layer.q_proj.forward(backend, &normed)?;
            let k = layer.k_proj.forward(backend, &normed)?;
            let v = layer.v_proj.forward(backend, &normed)?;

            let mut attn_rows = vec![0.0f32; x.len()];
            for n in 0..n_images {
                let slice = |t: &CpuTensor| -> CpuTensor {
                    let start = n * num_patches * cfg.embed_dim;
                    CpuTensor::from_data(
                        vec![num_patches, cfg.embed_dim],
                        t.data()[start..start + num_patches * cfg.embed_dim].to_vec(),
                    )
                };
                let attn =
                    bidirectional_attention(&slice(&q), &slice(&k), &slice(&v), cfg.n_heads)?;
                let start = n * num_patches * cfg.embed_dim;
                attn_rows[start..start + num_patches * cfg.embed_dim].copy_from_slice(attn.data());
            }
            let attn = CpuTensor::from_data(vec![n_images * num_patches, cfg.embed_dim], attn_rows);
            let attn = layer.out_proj.forward(backend, &attn)?;
            x = backend.add(&x, &attn)?;

            let normed = layer.ln2.forward(backend, &x)?;
            let hidden = layer.fc1.forward(backend, &normed)?;
            let hidden = backend.gelu_tanh(&hidden)?;
            let mlp = layer.fc2.forward(backend, &hidden)?;
            x = backend.add(&x, &mlp)?;
            if let Some(trace) = trace.as_mut() {
                trace.layer_outputs.push(x.clone());
            }
        }

        // -- post layer-norm --
        let out = self.post_ln.forward(backend, &x)?;
        if let Some(trace) = trace.as_mut() {
            trace.encoder_output = Some(out.clone());
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
        let qh = slice_cols(q, cols.clone());
        let kh = slice_cols(k, cols.clone());
        let vh = slice_cols(v, cols);
        // scores [seq, seq] = qh * kh^T * scale
        let mut scores = qh.matmul(&kh.transpose());
        for v in scores.data_mut() {
            *v *= scale;
        }
        let probs = scores.softmax();
        let oh = probs.matmul(&vh); // [seq, head_dim]
        for row in 0..seq {
            let dst = &mut out[row * embed + h * head_dim..row * embed + (h + 1) * head_dim];
            dst.copy_from_slice(&oh.data()[row * head_dim..(row + 1) * head_dim]);
        }
    }
    Ok(CpuTensor::from_data(vec![seq, embed], out))
}

fn slice_cols(t: &CpuTensor, cols: std::ops::Range<usize>) -> CpuTensor {
    let (rows, width) = (t.shape()[0], t.shape()[1]);
    let mut out = vec![0.0f32; rows * cols.len()];
    for r in 0..rows {
        out[r * cols.len()..(r + 1) * cols.len()]
            .copy_from_slice(&t.data()[r * width + cols.start..r * width + cols.end]);
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
}
