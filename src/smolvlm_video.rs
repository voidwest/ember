//! SmolVLM2-256M-Video-Instruct: the first video-capable model on the
//! multimodal foundation.
//!
//! Architecture (reference: HuggingFace `SmolVLM2-256M-Video-Instruct`,
//! `Idefics3ForConditionalGeneration` + `SmolVLMProcessor`):
//!
//! ```text
//! video  -> frame sampling (explicit policy, default uniform ≤ 64)
//!        -> per selected frame: stretch-resize to 512×512, rescale,
//!           normalize (the reference's non-splitting square resize; the
//!           pixel mask is trivially all-valid for this model)
//!        -> ONE batched vision encode [n_frames, 3, 512, 512]
//!        -> pixel-shuffle connector -> 64 tokens/frame
//! text   -> tokenizer; each <video> placeholder expands to the reference
//!           prompt: intro line ("You are provided the following series of
//!           N frames from a H:MM:SS [H:MM:SS] video."), one
//!           "Frame from MM:SS:" single-image block per frame, outro
//! text + visual embeddings -> ordered scatter over <image> tokens
//!        -> EmbeddingSequence -> normal Llama prefill -> KV cache -> decode
//! ```
//!
//! Frames are processed independently through the vision tower because that
//! is exactly what the reference does (frame-independent encoding); no
//! temporal mixing is pretended beyond what the timestamp text labels
//! provide. The LLM is the SmolLM2-135M backbone loaded through the
//! standard `Llama::from_loader`; nothing in the transformer knows about
//! video.

use crate::backend::{Backend, CpuBackend};
use crate::embedding::EmbeddingSequence;
use crate::llama::Llama;
use crate::loader::load_gguf;
use crate::multimodal::assembler::embed_and_scatter;
use crate::multimodal::image::{preprocess, ImagePreprocessConfig, Resample};
use crate::multimodal::request::{ContentPart, VideoInput};
use crate::multimodal::video::{FrameSampling, SampledVideo};
use crate::multimodal::vision::{VisionModel, VisionTrace};
use crate::tensor::CpuTensor;
use crate::tokenizer::EmberTokenizer;
use anyhow::{ensure, Result};
use std::time::Instant;

/// The `<video>` placeholder understood by `render_prompt`.
pub const VIDEO_PLACEHOLDER: &str = "<video>";

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct VideoTimings {
    pub preprocess_ms: f64,
    pub vision_ms: f64,
    pub projector_ms: f64,
    pub llm_prefill_ms: f64,
    pub ttft_ms: f64,
    pub decode_tok_s: f64,
    pub n_decode_tokens: usize,
}

/// Progressive-validation intermediates of one video prefill.
pub struct VideoTrace {
    pub sampled: SampledVideo,
    /// Normalized frames actually encoded, `[n_frames, 3, size, size]`.
    pub pixels: CpuTensor,
    pub vision: VisionTrace,
    /// Connector output `[n_frames * tokens_per_frame, llm_width]`.
    pub projector_output: CpuTensor,
    pub assembled_embeddings: CpuTensor,
    pub input_ids: Vec<u32>,
}

/// SmolVLM2-Video: LLM + vision tower + sampler + recipe.
pub struct SmolVlmVideo {
    pub llm: Llama<CpuBackend>,
    pub vision: VisionModel,
    pub sampling: FrameSampling,
    pub preprocess_config: ImagePreprocessConfig,
}

impl SmolVlmVideo {
    /// Load with the default production K strategy (compressed-resident
    /// `auto`) for the text model.
    pub fn from_ggufs(text_path: &std::path::Path, mmproj_path: &std::path::Path) -> Result<Self> {
        Self::from_ggufs_with_k_strategy(text_path, mmproj_path, crate::quant_k::KStrategy::Auto)
    }

    /// Load with an explicit K-family execution policy for the text model.
    /// `EagerF32` is the exact-f32 oracle path; `Auto` keeps Q4_K/Q6_K
    /// compressed-resident on the integer kernels.
    pub fn from_ggufs_with_k_strategy(
        text_path: &std::path::Path,
        mmproj_path: &std::path::Path,
        k_strategy: crate::quant_k::KStrategy,
    ) -> Result<Self> {
        let loader = crate::loader::load_gguf_with_k_strategy(text_path, k_strategy, false)?;
        let llm = Llama::from_loader(loader)?;
        let mut mmproj = load_gguf(mmproj_path)?;
        let vision = VisionModel::from_mmproj_loader(&mut mmproj)?;
        anyhow::ensure!(
            vision.llm_width(&CpuBackend) == llm.config.embed_dim,
            "vision connector output width {} does not match text embedding width {}",
            vision.llm_width(&CpuBackend),
            llm.config.embed_dim
        );
        let preprocess_config = ImagePreprocessConfig {
            // reference video path: stretch-resize straight onto the tower's
            // square input (no longest-edge stage, no tiling)
            resize_longest_edge: None,
            tile_size: None,
            resample: Resample::Lanczos,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        Ok(Self {
            llm,
            vision,
            sampling: FrameSampling::Uniform { max_frames: 64 },
            preprocess_config,
        })
    }

    /// Tokens per frame after the connector.
    pub fn tokens_per_frame(&self) -> usize {
        let cfg = &self.vision.transformer.config;
        let num_patches = cfg.num_patches();
        num_patches / (self.vision.connector.scale_factor * self.vision.connector.scale_factor)
    }

    /// Preprocess sampled frames into one normalized batch tensor.
    pub fn prepare_frames(&self, sampled: &SampledVideo) -> Result<CpuTensor> {
        let t0 = Instant::now();
        ensure!(!sampled.frames.is_empty(), "no frames survived sampling");
        let (h0, w0) = (sampled.frames[0].shape()[1], sampled.frames[0].shape()[2]);
        let size = self.vision.transformer.config.image_size;
        let mut pixels = vec![0.0f32; sampled.frames.len() * 3 * size * size];
        for (i, f) in sampled.frames.iter().enumerate() {
            // every frame must share geometry (a decoded stream guarantees it)
            ensure!(
                (f.shape()[1], f.shape()[2]) == (h0, w0),
                "frame {i} geometry {:?} differs from frame 0",
                f.shape()
            );
            // Phase 5 Track H: the STOCK reference video chain upsamples
            // each frame to `longest_edge` (2048) and back down to the
            // tower size with PIL bicubic (uint8 domain, fixed point).
            // Reproduce that chain exactly instead of a single LANCZOS
            // stretch; for square sources both legs are square stretches.
            let longest = self
                .preprocess_config
                .resize_longest_edge
                .map(|v| v as usize)
                .unwrap_or(size * 4);
            let up = if (h0, w0) == (longest, longest) {
                f.clone()
            } else {
                crate::multimodal::image::resize(f, longest, longest, Resample::Bicubic)?
            };
            let resized = crate::multimodal::image::resize(&up, size, size, Resample::Bicubic)?;
            let pp = preprocess(&resized, &self.preprocess_config)?;
            ensure!(
                pp.tiles.shape() == [1, 3, size, size],
                "video frame preprocessing produced {:?}",
                pp.tiles.shape()
            );
            let len = 3 * size * size;
            pixels[i * len..(i + 1) * len].copy_from_slice(pp.tiles.data());
        }
        let _ = t0.elapsed();
        Ok(CpuTensor::from_data(
            vec![sampled.frames.len(), 3, size, size],
            pixels,
        ))
    }
}

// ---------------------------------------------------------------------------
// reference prompt rendering
// ---------------------------------------------------------------------------

/// English number words for small counts (num2words-compatible for the
/// range frame counts can reach).
fn number_words(mut n: usize) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    if n < 20 {
        return ONES[n].to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    if n >= 100 {
        let hundreds = n / 100;
        parts.push(format!("{} hundred", ONES[hundreds]));
        n %= 100;
        if n > 0 {
            parts.last_mut().unwrap().push_str(" and");
        }
    }
    if n >= 20 {
        let t = n / 10;
        let o = n % 10;
        parts.push(if o == 0 {
            TENS[t].to_string()
        } else {
            format!("{}-{}", TENS[t], ONES[o])
        });
    } else if n > 0 {
        parts.push(ONES[n].to_string());
    }
    parts.join(" ")
}

fn hmmss(total_seconds: u64) -> String {
    let h = total_seconds / 3600;
    let m = (total_seconds % 3600) / 60;
    let s = total_seconds % 60;
    format!("{h}:{m:02}:{s:02}")
}

fn mmss(seconds: f64) -> String {
    let total = seconds.floor().max(0.0) as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Expand `<video>` placeholders into the reference prompt structure.
///
/// `videos[i]` binds to the i-th placeholder in order. Timestamps come from
/// each video's sampled metadata; fps falls back to 24 exactly like the
/// reference does for pre-sampled frames without metadata.
pub fn expand_video_placeholder(sampled: &SampledVideo, tokens_per_frame: usize) -> String {
    // timestamps are already absolute; fps only documents provenance
    let n = sampled.n_frames();
    let duration_s = sampled.source_duration_s.unwrap_or_else(|| {
        sampled
            .timestamps_ms
            .last()
            .map(|ms| ms / 1000.0)
            .unwrap_or(0.0)
    });
    let mut out = String::new();
    out.push_str(&format!(
        "You are provided the following series of {} frames from a {} [H:MM:SS] video.\n",
        number_words(n),
        hmmss(duration_s.floor() as u64),
    ));
    for i in 0..n {
        let ts = mmss(sampled.timestamps_ms[i] / 1000.0);
        out.push_str(&format!("\nFrame from {ts}:"));
        out.push_str("<fake_token_around_image><global-img>");
        out.push_str(&"<image>".repeat(tokens_per_frame));
        out.push_str("<fake_token_around_image>");
    }
    out.push_str("\n\n");
    out
}

impl SmolVlmVideo {
    /// Render the full chat-template prompt for one user message whose
    /// content interleaves text and `<video>` placeholders. Each
    /// placeholder is *replaced* by its video's expansion, in order of
    /// appearance.
    pub fn render_prompt(&self, text: &str, videos: &[SampledVideo]) -> Result<String> {
        let placeholders = text.matches(VIDEO_PLACEHOLDER).count();
        ensure!(
            placeholders == videos.len(),
            "prompt has {placeholders} <video> placeholders but {} videos were provided",
            videos.len()
        );
        let tokens_per_frame = self.tokens_per_frame();
        let mut content = String::with_capacity(text.len());
        let mut rest = text;
        for v in videos {
            let pos = rest
                .find(VIDEO_PLACEHOLDER)
                .expect("placeholder count verified above");
            content.push_str(&rest[..pos]);
            content.push_str(&expand_video_placeholder(v, tokens_per_frame));
            rest = &rest[pos + VIDEO_PLACEHOLDER.len()..];
        }
        content.push_str(rest);
        // reference template: ':' separator only when the FIRST content
        // element is an image; video-first messages keep ': '
        Ok(format!(
            "<|im_start|>User: {content}<end_of_utterance>\nAssistant:"
        ))
    }

    /// Build prefill inputs for an ordered request containing text and video
    /// parts (video frames sampled with this wrapper's policy).
    pub fn build_inputs_parts(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        parts: &[ContentPart],
        start_pos: usize,
    ) -> Result<(VideoTrace, EmbeddingSequence<CpuBackend>, VideoTimings)> {
        let mut timings = VideoTimings::default();
        // split ordered parts: concatenated text + video inputs in order
        let mut text = String::new();
        let mut raw_videos: Vec<&VideoInput> = Vec::new();
        for part in parts {
            match part {
                ContentPart::Text(t) => text.push_str(t),
                ContentPart::Video(v @ VideoInput::Frames(_)) => raw_videos.push(v),
                ContentPart::Image(_) => anyhow::bail!(
                    "SmolVlmVideo: image parts not supported on the video path (use SmolVlm)"
                ),
                ContentPart::Audio(_) => {
                    anyhow::bail!("SmolVlmVideo accepts only text and video parts; got audio")
                }
            }
        }
        let t0 = Instant::now();
        let mut sampled_videos = Vec::with_capacity(raw_videos.len());
        for v in &raw_videos {
            let VideoInput::Frames(frames) = v;
            sampled_videos.push(self.sampling.sample(frames)?);
        }
        let mut pixels_batches = Vec::with_capacity(sampled_videos.len());
        for s in &sampled_videos {
            pixels_batches.push(self.prepare_frames(s)?);
        }
        timings.preprocess_ms = t0.elapsed().as_secs_f64() * 1e3;

        // one encode per video, features concatenated in placeholder order
        let t1 = Instant::now();
        let tokens_per_frame = self.tokens_per_frame();
        let _ = tokens_per_frame;
        let width = self.vision.llm_width(backend);
        let mut all_features: Vec<f32> = Vec::new();
        let mut total_rows = 0usize;
        let mut last_trace = VisionTrace::default();
        for px in &pixels_batches {
            let (enc, trace) = self.vision.transformer.encode_traced(backend, px)?;
            last_trace = trace;
            let projected = self.vision.connector.forward(
                backend,
                &enc,
                self.vision.transformer.config.num_patches(),
            )?;
            total_rows += projected.shape()[0];
            all_features.extend_from_slice(projected.data());
        }
        let features =
            CpuTensor::from_data(vec![total_rows, width], std::mem::take(&mut all_features));
        timings.vision_ms = t1.elapsed().as_secs_f64() * 1e3;
        let projector_output = features.clone();

        let rendered = self.render_prompt(&text, &sampled_videos)?;
        let (input_ids, embeddings) = embed_and_scatter(
            backend,
            tokenizer,
            &rendered,
            "<image>",
            &features,
            &self.llm.embed_tokens,
        )?;
        let pixels = if pixels_batches.len() == 1 {
            pixels_batches.pop().expect("len checked")
        } else {
            // concatenate [n,3,h,w] batches along dim 0
            let per_frame = pixels_batches[0].shape()[1..].iter().product::<usize>();
            let rows: usize = pixels_batches.iter().map(|t| t.shape()[0]).sum();
            let mut data = Vec::with_capacity(rows * per_frame);
            let mut shape_tail = pixels_batches[0].shape()[1..].to_vec();
            for t in &pixels_batches {
                data.extend_from_slice(t.data());
            }
            shape_tail.insert(0, rows);
            CpuTensor::from_data(shape_tail, data)
        };
        let trace = VideoTrace {
            sampled: sampled_videos
                .into_iter()
                .next()
                .expect("at least one video"),
            pixels,
            vision: last_trace,
            projector_output,
            assembled_embeddings: embeddings.clone(),
            input_ids: input_ids.clone(),
        };
        Ok((
            trace,
            EmbeddingSequence::causal(embeddings, start_pos),
            timings,
        ))
    }

    /// Greedy generation over a mixed text/video request with stage timings.
    pub fn generate_with_parts(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        parts: &[ContentPart],
        max_tokens: usize,
    ) -> Result<(Vec<u32>, String, VideoTimings)> {
        let wall_start = Instant::now();
        let (_trace, sequence, mut timings) =
            self.build_inputs_parts(backend, tokenizer, parts, 0)?;
        let input_ids_len = _trace.input_ids.len();
        let mut cache = self
            .llm
            .create_request_cache(backend, input_ids_len, max_tokens);
        let t3 = Instant::now();
        let mut logits = self.llm.forward_last_logits_embeddings_with_cache(
            backend,
            &sequence.embeddings,
            &mut cache,
            0,
        )?;
        timings.llm_prefill_ms = t3.elapsed().as_secs_f64() * 1e3;
        let eos_ids = tokenizer.eos_token_ids();
        let mut generated: Vec<u32> = Vec::new();
        for step in 0..max_tokens {
            let data = backend.data(&logits);
            let best = crate::sampler::argmax_token(data);
            let best = u32::try_from(best)
                .map_err(|_| anyhow::anyhow!("vocabulary exceeds u32 token ids"))?;
            generated.push(best);
            if generated.len() == 1 {
                timings.ttft_ms = wall_start.elapsed().as_secs_f64() * 1e3;
            }
            if eos_ids.contains(&best) {
                break;
            }
            if step + 1 < max_tokens {
                logits = self.llm.forward_last_logits_with_cache(
                    backend,
                    &[best],
                    &mut cache,
                    input_ids_len + step,
                )?;
            }
        }
        timings.n_decode_tokens = generated.len();
        let total_ms = wall_start.elapsed().as_secs_f64() * 1e3;
        let decode_ms = total_ms - timings.ttft_ms;
        if generated.len() > 1 && decode_ms > 0.0 {
            timings.decode_tok_s = (generated.len() - 1) as f64 / (decode_ms / 1e3);
        }
        let text = tokenizer.decode(&generated)?;
        Ok((generated, text, timings))
    }
}
