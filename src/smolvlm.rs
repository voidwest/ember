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
use crate::loader::load_gguf; // mmproj (f16/f32 vision tensors) only
use crate::multimodal::assembler::{EmbeddingAssembler, ImageFeatures, SmolVlmAssembler};
use crate::multimodal::batch::BatchedImageInput;
use crate::multimodal::image::{preprocess, ImagePreprocessConfig, PreprocessedImage};
use crate::multimodal::request::{ContentPart, ImageInput, SegmentId};
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

/// Media pipeline result: per-input features, per-group traces, the
/// concatenated projector output, freshly processed images, and
/// (preprocess ms, encode ms).
pub(crate) type MediaPipelineResult = (
    Vec<ImageFeatures>,
    Vec<VisionTrace>,
    CpuTensor,
    Vec<PreprocessedImage>,
    (f64, f64),
);

/// SmolVLM-256M: LLM + vision tower + connector + assembler + recipe.
pub struct SmolVlm {
    pub llm: Llama<CpuBackend>,
    pub vision: VisionModel,
    pub assembler: SmolVlmAssembler,
    pub preprocess_config: ImagePreprocessConfig,
    /// Identity of the vision weights (sha256 of the mmproj file); folds
    /// into feature-cache keys so features never cross model boundaries.
    pub vision_identity: u64,
    /// Optional encoded-media cache ([`Self::with_feature_cache`]); `None`
    /// disables reuse entirely (the historical behavior).
    pub feature_cache: Option<std::sync::Mutex<crate::multimodal::cache::MediaFeatureCache>>,
}

/// All progressive-validation intermediates of one multimodal prefill.
pub struct MultimodalTrace {
    /// One processed image per `ContentPart::Image`, in request order.
    pub images: Vec<PreprocessedImage>,
    /// One vision trace per geometry group (heterogeneous requests encode
    /// one group per distinct tile shape; homogeneous requests see a single
    /// trace, the historical shape).
    pub vision: Vec<VisionTrace>,
    /// Connector output `[total_image_tokens, llm_width]` (all groups).
    pub projector_output: CpuTensor,
    /// Merged LLM input embeddings `[seq, llm_width]`.
    pub assembled_embeddings: CpuTensor,
    /// Full token sequence (chat template + tile expansion).
    pub input_ids: Vec<u32>,
}

impl SmolVlm {
    /// Load the text GGUF (llama arch) and the mmproj GGUF with the default
    /// production K strategy (compressed-resident `auto`).
    pub fn from_ggufs(text_path: &Path, mmproj_path: &Path) -> Result<Self> {
        Self::from_ggufs_with_k_strategy(text_path, mmproj_path, crate::quant_k::KStrategy::Auto)
    }

    /// Load with an explicit K-family execution policy for the text model.
    /// `EagerF32` is the exact-f32 oracle path; `Auto` keeps Q4_K/Q6_K
    /// compressed-resident on the integer kernels.
    pub fn from_ggufs_with_k_strategy(
        text_path: &Path,
        mmproj_path: &Path,
        k_strategy: crate::quant_k::KStrategy,
    ) -> Result<Self> {
        let loader = crate::loader::load_gguf_with_k_strategy(text_path, k_strategy, false)
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
        anyhow::ensure!(
            vision.llm_width(&CpuBackend) == llm.config.embed_dim,
            "vision connector output width {} does not match text embedding width {}",
            vision.llm_width(&CpuBackend),
            llm.config.embed_dim
        );
        let assembler = SmolVlmAssembler::default();
        let preprocess_config = ImagePreprocessConfig {
            resize_longest_edge: Some(2048),
            tile_size: Some(512),
            resample: crate::multimodal::image::Resample::Lanczos,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        // content hash of the mmproj file: cache keys are invalid across
        // different encoder weights. Streamed with a hard cap and path
        // identity checks so a large/sparse mmproj cannot be materialized
        // just to be hashed.
        let vision_identity = crate::loader::gguf_content_identity(mmproj_path)
            .context("failed to hash mmproj for feature-cache identity")?;
        Ok(Self {
            llm,
            vision,
            assembler,
            preprocess_config,
            vision_identity,
            feature_cache: None,
        })
    }

    /// Enable the encoded-media feature cache with a byte budget.
    pub fn with_feature_cache(mut self, max_bytes: usize) -> Self {
        self.feature_cache = Some(std::sync::Mutex::new(
            crate::multimodal::cache::MediaFeatureCache::new(max_bytes),
        ));
        self
    }

    /// Cache key for one decoded image under this wrapper's configuration.
    fn cache_key(
        &self,
        media_id: crate::multimodal::request::MediaId,
    ) -> crate::multimodal::cache::FeatureCacheKey {
        use crate::multimodal::cache::{FeatureCacheKey, PreprocessFingerprint};
        let cfg = &self.preprocess_config;
        let mut fp = PreprocessFingerprint::new("smolvlm-image-v1");
        match cfg.resize_longest_edge {
            Some(v) => fp.mix_u64(v as u64 + 1),
            None => fp.mix_u64(0),
        }
        match cfg.tile_size {
            Some(v) => fp.mix_u64(v as u64 + 1),
            None => fp.mix_u64(0),
        }
        fp.mix_u64(match cfg.resample {
            crate::multimodal::image::Resample::Lanczos => 1,
            crate::multimodal::image::Resample::Bicubic => 2,
        });
        fp.mix_f64(cfg.rescale_factor as f64);
        for v in cfg.mean {
            fp.mix_f64(v as f64);
        }
        for v in cfg.std {
            fp.mix_f64(v as f64);
        }
        FeatureCacheKey {
            media_id,
            kind: crate::multimodal::request::MediaKind::Image,
            preprocess: fp.value(),
            tower_identity: self.vision_identity,
        }
    }

    // -----------------------------------------------------------------
    // shared pipeline: content parts -> prepared request
    // -----------------------------------------------------------------

    /// Split ordered content parts into the concatenated prompt text and
    /// the image inputs in part order. SmolVLM supports text + image parts;
    /// audio/video parts fail closed here — the adapter, not the substrate,
    /// decides what a model accepts.
    pub fn split_parts(parts: &[ContentPart]) -> Result<(String, Vec<&ImageInput>)> {
        let mut text = String::new();
        let mut images = Vec::new();
        for part in parts {
            match part {
                ContentPart::Text(t) => text.push_str(t),
                ContentPart::Image(img) => images.push(img),
                ContentPart::Audio(_) | ContentPart::Video(_) => anyhow::bail!(
                    "SmolVLM accepts only text and image parts; got {:?}",
                    part.media_kind().unwrap()
                ),
            }
        }
        Ok((text, images))
    }

    /// Full media pipeline for a request's image inputs, in order:
    /// decode -> (cache hit ? reuse : preprocess + grouped encode + insert).
    /// Cached features are bit-exact replays of the encode that produced
    /// them (same key => same decoded pixels + same recipe + same weights).
    /// Returns per-input features plus only the *freshly processed* images
    /// (cache hits skip preprocessing; validation dumps should run with the
    /// cache disabled).
    fn media_features(
        &self,
        backend: &CpuBackend,
        image_inputs: &[&ImageInput],
    ) -> Result<MediaPipelineResult> {
        // 1. decode everything once; content ids from the decoded pixels
        let t_media = Instant::now();
        let mut decoded = Vec::with_capacity(image_inputs.len());
        let mut keys = Vec::with_capacity(image_inputs.len());
        for img in image_inputs {
            let d = img.decode()?;
            keys.push(self.cache_key(crate::multimodal::request::MediaId::from_tensor(&d)));
            decoded.push(d);
        }

        // 2. consult the cache
        let mut cached: Vec<Option<CpuTensor>> = vec![None; decoded.len()];
        if let Some(cache) = &self.feature_cache {
            let mut cache = cache.lock().expect("feature cache poisoned");
            for (i, key) in keys.iter().enumerate() {
                if let Some(f) = cache.lookup(key) {
                    cached[i] = Some(f.clone());
                }
            }
        }

        // 3. preprocess + encode only the misses (grouped by geometry)
        let mut miss_idx: Vec<usize> = Vec::new();
        let mut processed: Vec<PreprocessedImage> = Vec::new();
        for (i, d) in decoded.iter().enumerate() {
            if cached[i].is_none() {
                miss_idx.push(i);
                processed.push(preprocess(d, &self.preprocess_config)?);
            }
        }
        let t_encode = Instant::now();
        let (miss_features, traces, projector_output) = self.encode_images(backend, &processed)?;
        debug_assert_eq!(miss_features.len(), miss_idx.len());
        // (preprocess+decode ms, encode ms) — cache hits shrink both
        let media_timings = (
            t_media.elapsed().as_secs_f64() * 1e3,
            t_encode.elapsed().as_secs_f64() * 1e3,
        );

        // 4. interleave hits and fresh features in request order, inserting
        //    fresh entries into the cache
        let mut ordered = Vec::with_capacity(decoded.len());
        let mut miss_pos = 0usize;
        for i in 0..decoded.len() {
            if let Some(f) = cached[i].take() {
                let grid = crate::multimodal::image::tile_grid_for(
                    (decoded[i].shape()[1], decoded[i].shape()[2]),
                    &self.preprocess_config,
                );
                ordered.push(ImageFeatures {
                    features: f,
                    tile_grid: grid,
                });
            } else {
                if let Some(cache) = &self.feature_cache {
                    cache
                        .lock()
                        .expect("feature cache poisoned")
                        .insert(keys[i].clone(), miss_features[miss_pos].features.clone());
                }
                let f = &miss_features[miss_pos];
                let tile_grid = processed[miss_pos].tile_grid;
                miss_pos += 1;
                ordered.push(ImageFeatures {
                    features: f.features.clone(),
                    tile_grid,
                });
            }
        }
        debug_assert_eq!(miss_pos, miss_features.len());
        Ok((ordered, traces, projector_output, processed, media_timings))
    }

    /// Encode processed images through the shared cross-request batching
    /// core (single request here: owners are synthetic per-part ids), with
    /// traced groups for the validation dumps.
    fn encode_images(
        &self,
        backend: &CpuBackend,
        processed: &[PreprocessedImage],
    ) -> Result<(Vec<ImageFeatures>, Vec<VisionTrace>, CpuTensor)> {
        if processed.is_empty() {
            return Ok((
                Vec::new(),
                Vec::new(),
                CpuTensor::from_data(vec![0, 0], vec![]),
            ));
        }
        let inputs: Vec<BatchedImageInput> = processed
            .iter()
            .enumerate()
            .map(|(i, p)| BatchedImageInput {
                owner: SegmentId::new(0, i),
                tiles: p.tiles.clone(),
            })
            .collect();
        let patch_size = self.vision.transformer.config.patch_size;
        let scale = self.vision.connector.scale_factor;
        let vision = &self.vision;
        let (outputs, traces, projected_all) = crate::multimodal::batch::batch_encode_images(
            backend,
            &inputs,
            patch_size,
            scale,
            |be, batch| {
                let (enc, trace) = vision.transformer.encode_traced(be, batch)?;
                let projected =
                    vision
                        .connector
                        .forward(be, &enc, vision.transformer.config.num_patches())?;
                Ok((projected, trace))
            },
        )?;
        debug_assert_eq!(outputs.len(), processed.len());
        let features: Vec<ImageFeatures> = outputs
            .into_iter()
            .zip(processed.iter())
            .map(|(o, p)| ImageFeatures {
                features: o.features,
                tile_grid: p.tile_grid,
            })
            .collect();
        Ok((features, traces, projected_all))
    }

    /// Preprocess, encode and assemble a full multi-part request.
    ///
    /// Every [`ContentPart::Image`] must be bound to one `<image>` placeholder
    /// in the concatenated text (placeholders bind in order of appearance).
    /// Text-only requests assemble as pure token embeddings through the same
    /// chat template.
    pub fn build_inputs_parts(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        parts: &[ContentPart],
        start_pos: usize,
    ) -> Result<(MultimodalTrace, EmbeddingSequence<CpuBackend>)> {
        let (text, image_inputs) = Self::split_parts(parts)?;
        let (images_features, vtraces, projector_output, fresh_images, (_pre_ms, _vis_ms)) =
            self.media_features(backend, &image_inputs)?;

        let assembled = self.assembler.assemble(
            backend,
            tokenizer,
            &text,
            &images_features,
            &self.llm.embed_tokens,
        )?;
        let trace = MultimodalTrace {
            images: fresh_images,
            vision: vtraces,
            projector_output,
            assembled_embeddings: assembled.embeddings.clone(),
            input_ids: assembled.input_ids.clone(),
        };
        let sequence = EmbeddingSequence::causal(assembled.embeddings, start_pos);
        Ok((trace, sequence))
    }

    /// Full greedy generation over an ordered multi-part request with
    /// separate stage timings. This is the single frontend the CLI (and any
    /// other embedding host) uses; file-backed and in-memory media take the
    /// identical path.
    pub fn generate_with_parts(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        parts: &[ContentPart],
        max_tokens: usize,
    ) -> Result<(Vec<u32>, String, MultimodalTimings)> {
        let wall_start = Instant::now();
        let mut timings = MultimodalTimings::default();

        let (text, image_inputs) = Self::split_parts(parts)?;
        let (images_features, _vtraces, _projector_output, _fresh, (pre_ms, vis_ms)) =
            self.media_features(backend, &image_inputs)?;
        timings.preprocess_ms = pre_ms;
        timings.vision_ms = vis_ms;

        let assembled = self.assembler.assemble(
            backend,
            tokenizer,
            &text,
            &images_features,
            &self.llm.embed_tokens,
        )?;

        let mut cache =
            self.llm
                .create_request_cache(backend, assembled.input_ids.len(), max_tokens);
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
        let out_text = tokenizer.decode(&generated)?;
        Ok((generated, out_text, timings))
    }
}
