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
use crate::multimodal::assembler::{EmbeddingAssembler, SmolVlmAssembler};
use crate::multimodal::image::{decode_rgb, preprocess, ImagePreprocessConfig, PreprocessedImage};
use crate::multimodal::vision::{VisionModel, VisionTrace};
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
    pub processed: PreprocessedImage,
    pub vision: VisionTrace,
    /// Connector output `[n_image_tokens, llm_width]`.
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

    /// Preprocess + encode + assemble with an explicit tokenizer.
    pub fn build_inputs_with_tokenizer(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        image: &Path,
        prompt: &str,
        start_pos: usize,
    ) -> Result<(MultimodalTrace, EmbeddingSequence<CpuBackend>)> {
        let t0 = Instant::now();
        let decoded = decode_rgb(image)?;
        let processed = preprocess(&decoded, &self.preprocess_config)?;
        let preprocess_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let (encoder_out, vtrace) = self
            .vision
            .transformer
            .encode_traced(backend, &processed.tiles)?;
        let vision_ms = t1.elapsed().as_secs_f64() * 1e3;

        let t2 = Instant::now();
        let num_patches = self.vision.transformer.config.num_patches();
        let features = self
            .vision
            .connector
            .forward(backend, &encoder_out, num_patches)?;
        let projector_ms = t2.elapsed().as_secs_f64() * 1e3;

        let assembled = self.assembler.assemble(
            backend,
            tokenizer,
            prompt,
            &features,
            processed.tile_grid,
            &self.llm.embed_tokens,
        )?;
        let trace = MultimodalTrace {
            processed,
            vision: vtrace,
            projector_output: features,
            assembled_embeddings: assembled.embeddings.clone(),
            input_ids: assembled.input_ids.clone(),
        };
        let sequence = EmbeddingSequence::causal(assembled.embeddings, start_pos);
        let _ = (preprocess_ms, vision_ms, projector_ms);
        Ok((trace, sequence))
    }

    /// Full image-conditioned greedy generation with separate stage timings.
    pub fn generate_with_image(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        image: &Path,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, String, MultimodalTimings)> {
        let wall_start = Instant::now();
        let mut timings = MultimodalTimings::default();

        let t0 = Instant::now();
        let decoded = decode_rgb(image)?;
        let processed = preprocess(&decoded, &self.preprocess_config)?;
        timings.preprocess_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let encoder_out = self.vision.transformer.encode(backend, &processed.tiles)?;
        timings.vision_ms = t1.elapsed().as_secs_f64() * 1e3;

        let t2 = Instant::now();
        let num_patches = self.vision.transformer.config.num_patches();
        let features = self
            .vision
            .connector
            .forward(backend, &encoder_out, num_patches)?;
        timings.projector_ms = t2.elapsed().as_secs_f64() * 1e3;

        let assembled = self.assembler.assemble(
            backend,
            tokenizer,
            prompt,
            &features,
            processed.tile_grid,
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
