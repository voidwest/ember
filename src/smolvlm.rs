//! SmolVLM-256M-Instruct: the first image-capable model on the multimodal
//! foundation.
//!
//! Architecture (reference: HuggingFace `Idefics3ForConditionalGeneration`,
//! model `HuggingFaceTB/SmolVLM-256M-Instruct`):
//!
//! ```text
//! image  -> preprocess (LANCZOS resize 2048 -> 512 tiles + global tile,
//!           rescale 1/255, normalize mean/std 0.5)
//!        -> SigLIP-style ViT (12 layers, 768 hidden, patch 16)
//!        -> pixel-shuffle connector (scale 4) + linear (12288 -> 576)
//! text   -> tokenizer -> token embedding lookup
//! text + visual embeddings -> SmolVLM assembler (chat template, tile
//!           expansion, masked scatter over <image> tokens)
//!        -> EmbeddingSequence -> normal Llama prefill -> KV cache -> decode
//! ```
//!
//! The LLM is SmolLM2-135M (llama arch, full RoPE, head_dim 64, theta 1e5)
//! loaded through the standard `Llama::from_loader`; the vision tower +
//! connector load from a separate mmproj GGUF (see
//! `tools/convert_smolvlm_mmproj.py`). Nothing in the Llama transformer,
//! KV cache, attention, or MLP knows about images.

use crate::backend::{Backend, CpuBackend};
use crate::embedding::EmbeddingSequence;
use crate::llama::Llama;
use crate::loader::load_gguf;
use crate::model::ForwardModel;
use crate::multimodal::assembler::{EmbeddingAssembler, ImageFeatures, SmolVlmAssembler};
use crate::multimodal::image::{decode_rgb, preprocess, ImagePreprocessConfig, PreprocessedImage};
use crate::multimodal::vision::{VisionModel, VisionTrace};
use crate::multimodal::InputPart;
use crate::tensor::CpuTensor;
use crate::tokenizer::EmberTokenizer;
use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;

/// Per-stage timing for one multimodal run. Stages are reported separately
/// on purpose: heterogeneous inference costs must be visible rather than
/// folded into an aggregate latency.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct MultimodalTimings {
    /// Image preprocessing (decode, resize, tile, normalize), ms.
    pub preprocess_ms: f64,
    /// Vision transformer only, ms.
    pub vision_ms: f64,
    /// Connector (pixel shuffle + projection), ms.
    pub projector_ms: f64,
    /// LLM prefill (assembled embeddings through the transformer), ms.
    pub llm_prefill_ms: f64,
    /// Wall time from start to the first generated token, ms.
    pub ttft_ms: f64,
    /// Decode tokens per second (generated tokens only).
    pub decode_tok_s: f64,
    /// Number of generated tokens.
    pub n_decode_tokens: usize,
}

/// SmolVLM-256M: LLM + vision tower + connector + assembler + recipe.
pub struct SmolVlm {
    pub llm: Llama<CpuBackend>,
    pub vision: VisionModel,
    pub assembler: SmolVlmAssembler,
    pub preprocess_config: ImagePreprocessConfig,
}

/// All progressive-validation intermediates of one multimodal prefill.
pub struct MultimodalTrace {
    /// One processed image per `InputPart::Image`, in request order.
    pub images: Vec<PreprocessedImage>,
    /// Concatenated vision trace (all tiles of all images in one batch).
    pub vision: VisionTrace,
    /// Connector output `[total_image_tokens, llm_width]`.
    pub projector_output: CpuTensor,
    /// Merged LLM input embeddings `[seq, llm_width]`.
    pub assembled_embeddings: CpuTensor,
    /// Full token sequence (chat template + tile expansion).
    pub input_ids: Vec<u32>,
}

impl SmolVlm {
    /// Load the text GGUF (llama arch) and the mmproj GGUF (vision).
    pub fn from_ggufs(text_path: &Path, mmproj_path: &Path) -> Result<Self> {
        let loader = load_gguf(text_path)
            .with_context(|| format!("failed to load text model {}", text_path.display()))?;
        let llm = Llama::from_loader(loader)
            .with_context(|| format!("failed to build LLM from {}", text_path.display()))?;
        let mut mmproj = load_gguf(mmproj_path)
            .with_context(|| format!("failed to load mmproj {}", mmproj_path.display()))?;
        let vision = VisionModel::from_mmproj_loader(&mut mmproj).with_context(|| {
            format!(
                "failed to build vision model from {}",
                mmproj_path.display()
            )
        })?;
        let assembler = SmolVlmAssembler::default();
        let preprocess_config = ImagePreprocessConfig {
            resize_longest_edge: Some(2048),
            tile_size: Some(512),
            resample: crate::multimodal::image::Resample::Lanczos,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        Ok(Self {
            llm,
            vision,
            assembler,
            preprocess_config,
        })
    }

    /// Preprocess + encode + assemble with an explicit tokenizer
    /// (single-image convenience wrapper).
    pub fn build_inputs_with_tokenizer(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        image: &Path,
        prompt: &str,
        start_pos: usize,
    ) -> Result<(MultimodalTrace, EmbeddingSequence<CpuBackend>)> {
        let parts = vec![
            InputPart::Image(image.to_path_buf()),
            InputPart::Text(prompt.to_string()),
        ];
        let (trace, sequence) = self.build_inputs_parts(backend, tokenizer, &parts, start_pos)?;
        Ok((trace, sequence))
    }

    /// Preprocess, encode and assemble a full multi-part request.
    ///
    /// Every [`InputPart::Image`] must be bound to one `<image>` placeholder
    /// in the concatenated text (placeholders bind in order of appearance).
    /// All tiles from all images are encoded in one batched pass — the
    /// vision tower already treats dim 0 as independent images (attention
    /// never mixes them) — and the connector output is split back per image
    /// before assembly.
    pub fn build_inputs_parts(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        parts: &[InputPart],
        start_pos: usize,
    ) -> Result<(MultimodalTrace, EmbeddingSequence<CpuBackend>)> {
        // 1. preprocess every image part
        let t0 = Instant::now();
        let mut processed_images = Vec::new();
        for part in parts {
            if let InputPart::Image(path) = part {
                let decoded = decode_rgb(path)?;
                processed_images.push(preprocess(&decoded, &self.preprocess_config)?);
            }
        }
        let _preprocess_ms = t0.elapsed().as_secs_f64() * 1e3;

        // 2. one batched encode over all tiles (empty when text-only)
        let total_tiles: usize = processed_images.iter().map(|p| p.tiles.shape()[0]).sum();
        let t1 = Instant::now();
        let (encoder_out, vtrace) = if total_tiles > 0 {
            let first = &processed_images[0].tiles;
            anyhow::ensure!(
                processed_images
                    .iter()
                    .all(|p| p.tiles.shape()[1..] == first.shape()[1..]),
                "all images must share tile geometry"
            );
            let mut all_pixels =
                vec![0.0f32; total_tiles * first.shape()[1] * first.shape()[2] * first.shape()[3]];
            let mut offset = 0usize;
            for p in &processed_images {
                all_pixels[offset..offset + p.tiles.len()].copy_from_slice(p.tiles.data());
                offset += p.tiles.len();
            }
            let pixels = CpuTensor::from_data(
                vec![
                    total_tiles,
                    first.shape()[1],
                    first.shape()[2],
                    first.shape()[3],
                ],
                all_pixels,
            );
            self.vision.transformer.encode_traced(backend, &pixels)?
        } else {
            return Err(anyhow::anyhow!(
                "multimodal request contains no image parts"
            ));
        };
        let _vision_ms = t1.elapsed().as_secs_f64() * 1e3;

        // 3. connector over the whole batch
        let t2 = Instant::now();
        let num_patches = self.vision.transformer.config.num_patches();
        let features_all = self
            .vision
            .connector
            .forward(backend, &encoder_out, num_patches)?;
        let _projector_ms = t2.elapsed().as_secs_f64() * 1e3;

        // 4. split features per image and assemble with ordered binding
        let tokens_per_tile =
            num_patches / (self.vision.connector.scale_factor * self.vision.connector.scale_factor);
        let embed_dim = features_all.shape()[1];
        let mut images_features = Vec::with_capacity(processed_images.len());
        let mut row = 0usize;
        for p in &processed_images {
            let rows = p.tiles.shape()[0] * tokens_per_tile;
            images_features.push(ImageFeatures {
                features: CpuTensor::from_data(
                    vec![rows, embed_dim],
                    features_all.data()[row * embed_dim..(row + rows) * embed_dim].to_vec(),
                ),
                tile_grid: p.tile_grid,
            });
            row += rows;
        }
        let text: String = parts
            .iter()
            .map(|p| match p {
                InputPart::Text(t) => t.clone(),
                InputPart::Image(_) => String::new(),
            })
            .collect();
        let assembled = self.assembler.assemble(
            backend,
            tokenizer,
            &text,
            &images_features,
            &self.llm.embed_tokens,
        )?;
        let trace = MultimodalTrace {
            images: processed_images,
            vision: vtrace,
            projector_output: features_all,
            assembled_embeddings: assembled.embeddings.clone(),
            input_ids: assembled.input_ids.clone(),
        };
        let sequence = EmbeddingSequence::causal(assembled.embeddings, start_pos);
        Ok((trace, sequence))
    }

    /// Full image-conditioned greedy generation with separate stage timings
    /// (single-image convenience wrapper).
    pub fn generate_with_image(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        image: &Path,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, String, MultimodalTimings)> {
        let parts = vec![
            InputPart::Image(image.to_path_buf()),
            InputPart::Text(prompt.to_string()),
        ];
        self.generate_with_parts(backend, tokenizer, &parts, max_tokens)
    }

    /// Greedy generation over a full multi-part request (text + images),
    /// with separate stage timings.
    pub fn generate_with_parts(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        parts: &[InputPart],
        max_tokens: usize,
    ) -> Result<(Vec<u32>, String, MultimodalTimings)> {
        let wall_start = Instant::now();
        let mut timings = MultimodalTimings::default();

        let t0 = Instant::now();
        let mut processed_images = Vec::new();
        for part in parts {
            if let InputPart::Image(path) = part {
                let decoded = decode_rgb(path)?;
                processed_images.push(preprocess(&decoded, &self.preprocess_config)?);
            }
        }
        anyhow::ensure!(
            !processed_images.is_empty(),
            "generate_with_parts requires at least one image part"
        );
        timings.preprocess_ms = t0.elapsed().as_secs_f64() * 1e3;

        // batch all tiles of all images into one encode pass
        let total_tiles: usize = processed_images.iter().map(|p| p.tiles.shape()[0]).sum();
        let first = &processed_images[0].tiles;
        anyhow::ensure!(
            processed_images
                .iter()
                .all(|p| p.tiles.shape()[1..] == first.shape()[1..]),
            "all images must share tile geometry"
        );
        let (_, channels, height, width) = (
            first.shape()[0],
            first.shape()[1],
            first.shape()[2],
            first.shape()[3],
        );
        let tile_len = channels * height * width;
        let mut all_pixels = vec![0.0f32; total_tiles * tile_len];
        let mut offset = 0usize;
        for p in &processed_images {
            all_pixels[offset..offset + p.tiles.len()].copy_from_slice(p.tiles.data());
            offset += p.tiles.len();
        }
        let pixels = CpuTensor::from_data(vec![total_tiles, channels, height, width], all_pixels);

        let t1 = Instant::now();
        let encoder_out = self.vision.transformer.encode(backend, &pixels)?;
        timings.vision_ms = t1.elapsed().as_secs_f64() * 1e3;

        let t2 = Instant::now();
        let num_patches = self.vision.transformer.config.num_patches();
        let features_all = self
            .vision
            .connector
            .forward(backend, &encoder_out, num_patches)?;
        timings.projector_ms = t2.elapsed().as_secs_f64() * 1e3;

        // split features per image and assemble with ordered placeholder binding
        let tokens_per_tile =
            num_patches / (self.vision.connector.scale_factor * self.vision.connector.scale_factor);
        let embed_dim = features_all.shape()[1];
        let mut images_features = Vec::with_capacity(processed_images.len());
        let mut row = 0usize;
        for p in &processed_images {
            let rows = p.tiles.shape()[0] * tokens_per_tile;
            images_features.push(ImageFeatures {
                features: CpuTensor::from_data(
                    vec![rows, embed_dim],
                    features_all.data()[row * embed_dim..(row + rows) * embed_dim].to_vec(),
                ),
                tile_grid: p.tile_grid,
            });
            row += rows;
        }
        let text: String = parts
            .iter()
            .map(|p| match p {
                InputPart::Text(t) => t.clone(),
                InputPart::Image(_) => String::new(),
            })
            .collect();
        let assembled = self.assembler.assemble(
            backend,
            tokenizer,
            &text,
            &images_features,
            &self.llm.embed_tokens,
        )?;

        let mut cache = self
            .llm
            .create_cache(backend, self.llm.max_seq_len(backend));
        let t3 = Instant::now();
        // prefill the assembled embeddings; the first generated token comes
        // from the prefill's last-position logits (the reference generate()
        // semantics — the last prompt token is never re-fed as a decode step)
        let prefill_logits = self.llm.forward_last_logits_embeddings_with_cache(
            backend,
            &assembled.embeddings,
            &mut cache,
            0,
        )?;
        timings.llm_prefill_ms = t3.elapsed().as_secs_f64() * 1e3;

        let eos_ids = tokenizer.eos_token_ids();
        let mut generated: Vec<u32> = Vec::new();
        let start_pos = assembled.input_ids.len();
        let mut logits = prefill_logits;
        for step in 0..max_tokens {
            let data = backend.data(&logits);
            let best = crate::sampler::argmax_token(data);
            let best = u32::try_from(best)
                .map_err(|_| anyhow::anyhow!("model vocabulary exceeds u32 token-ID space"))?;
            generated.push(best);
            if generated.len() == 1 {
                // time to first token: everything from request start through
                // the prefill logits that selected this token
                let ttft = wall_start.elapsed();
                timings.ttft_ms = ttft.as_secs_f64() * 1e3;
            }
            if eos_ids.contains(&best) {
                break;
            }
            if step + 1 < max_tokens {
                logits = self.llm.forward_last_logits_with_cache(
                    backend,
                    &[best],
                    &mut cache,
                    start_pos + step,
                )?;
            }
        }
        timings.n_decode_tokens = generated.len();
        // decode rate covers only the token loop after the first token
        let total_ms = wall_start.elapsed().as_secs_f64() * 1e3;
        let decode_ms = total_ms - timings.ttft_ms;
        if generated.len() > 1 && decode_ms > 0.0 {
            timings.decode_tok_s = (generated.len() - 1) as f64 / (decode_ms / 1e3);
        } else if generated.len() == 1 && decode_ms > 0.0 {
            timings.decode_tok_s = 1.0 / (decode_ms / 1e3);
        }
        let text = tokenizer.decode(&generated)?;
        Ok((generated, text, timings))
    }
}
