//! Ultravox v0.5 (llama-3.2-1b): the first audio-capable model on the
//! multimodal foundation.
//!
//! Architecture (reference: fixie-ai/ultravox-v0_5-llama-3_2-1b):
//!
//! ```text
//! audio  -> decode WAV / normalize channels -> resample to 16 kHz
//!        -> log-mel spectrogram (Whisper feature extractor, 128 mels)
//!        -> Whisper encoder (32 layers, d_model 1280, 20 heads)
//!        -> SwiGLU projector (stack 8: 10240 -> 4096 -> silu-glu -> 2048)
//! text   -> tokenizer (Llama-3.2; <|audio|> placeholders become
//!           token_len x <|eot_id|> runs, exactly like the reference
//!           processor)
//! text + audio embeddings -> assembler scatter over the eot runs
//!        -> EmbeddingSequence -> normal Llama prefill -> KV cache -> decode
//! ```
//!
//! The LLM is Llama-3.2-1B-Instruct (llama arch) loaded through the
//! standard `Llama::from_loader`; the audio tower + projector load from a
//! separate audio GGUF (`tools/convert_ultravox_audio.py`). Nothing in the
//! Llama transformer, KV cache, attention, or MLP knows about audio.

use crate::backend::{Backend, CpuBackend};
use crate::embedding::EmbeddingSequence;
use crate::llama::{Llama, LlamaEmbedding};
use crate::loader::{load_gguf, load_gguf_with_k_strategy};
use crate::multimodal::audio::{self, to_mono_16k, AudioInput, MAX_FRAMES};
use crate::multimodal::audio_encoder::{AudioModel, AudioTrace};
use crate::multimodal::stream::{AudioStream, StreamProgress, StreamedAudio};
use crate::tensor::CpuTensor;
use crate::tokenizer::EmberTokenizer;
use anyhow::{ensure, Context, Result};
use std::time::Instant;

/// The audio placeholder understood by [`UltravoxAssembler`]. Not part of
/// the vocabulary: it splits the text and becomes a run of `<|eot_id|>`
/// tokens whose embeddings are overwritten by the projector output.
pub const AUDIO_PLACEHOLDER: &str = "<|audio|>";

/// Per-stage timing for one audio run.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct AudioTimings {
    /// Decode + channel/rate normalization, ms.
    pub decode_ms: f64,
    /// Log-mel spectrogram extraction, ms.
    pub features_ms: f64,
    /// Whisper encoder only, ms.
    pub encoder_ms: f64,
    /// Projector only, ms.
    pub projector_ms: f64,
    /// LLM prefill on assembled embeddings, ms.
    pub llm_prefill_ms: f64,
    /// Wall time request start -> first generated token, ms.
    pub ttft_ms: f64,
    /// Generated tokens per second (token loop after the first token).
    pub decode_tok_s: f64,
    pub n_decode_tokens: usize,
    /// Input waveform duration in seconds (after normalization).
    pub audio_seconds: f64,
    /// Number of encoder windows (long-form chunking; 1 = single window).
    pub n_chunks: usize,
}

impl AudioTimings {
    /// Encoder wall time divided by audio duration (>1 means slower than
    /// real time).
    pub fn encoder_real_time_factor(&self) -> f64 {
        if self.audio_seconds > 0.0 {
            self.encoder_ms / 1e3 / self.audio_seconds
        } else {
            0.0
        }
    }
}

/// One audio segment's projector output plus its placeholder binding.
#[derive(Debug)]
pub struct AudioFeatures {
    /// `[n_tokens, llm_width]` projected embedding rows for this segment.
    pub features: CpuTensor,
}

/// Assembled result: token ids plus merged embeddings.
#[derive(Debug)]
pub struct AssembledAudioSequence {
    pub input_ids: Vec<u32>,
    pub embeddings: CpuTensor,
}

/// Ultravox assembler: chat template, `<|audio|>` expansion into eot runs,
/// token lookup, sequential scatter.
///
/// Text is split around each [`AUDIO_PLACEHOLDER`] and each part is
/// tokenized separately (the reference processor does exactly this because
/// the placeholder is not in the vocabulary); between parts, `n_tokens`
/// copies of `<|eot_id|>` are inserted. Feature rows overwrite those eot
/// positions in order — any mismatch fails closed.
pub struct UltravoxAssembler {
    /// `<|eot_id|>`
    pub eot_token: String,
    /// `<|begin_of_text|>`
    pub bos_token: String,
}

impl Default for UltravoxAssembler {
    fn default() -> Self {
        Self {
            eot_token: "<|eot_id|>".into(),
            bos_token: "<|begin_of_text|>".into(),
        }
    }
}

impl UltravoxAssembler {
    /// Render the Llama-3.2-Instruct chat template for one user turn with a
    /// fixed date string (both ember and the reference script use this
    /// exact constant so tokenization matches bit-for-bit).
    pub fn render_chat_template(&self, user_content: &str) -> String {
        const DATE: &str = "01 Jan 2026";
        format!(
            "{bos}<|start_header_id|>system<|end_header_id|>\n\n\
             Cutting Knowledge Date: December 2023\nToday Date: {DATE}\n\n\
             <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
             {user_content}\
             <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            bos = self.bos_token
        )
    }

    fn resolve_ids(&self, tokenizer: &EmberTokenizer) -> Result<(u32, u32)> {
        let eos = tokenizer
            .token_to_id(&self.eot_token)
            .ok_or_else(|| anyhow::anyhow!("tokenizer missing {}", self.eot_token))?;
        // BOS must exist structurally but the template embeds it literally;
        // validate so a mismatched tokenizer fails closed before running.
        tokenizer
            .token_to_id(&self.bos_token)
            .ok_or_else(|| anyhow::anyhow!("tokenizer missing {}", self.bos_token))?;
        Ok((eos, 0))
    }

    /// Assemble one multimodal request (text with N placeholders + N audio
    /// segments) into embeddings ready for prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        text: &str,
        audios: &[AudioFeatures],
        embed_table: &LlamaEmbedding<CpuBackend>,
    ) -> Result<AssembledAudioSequence> {
        let (eos_id, _) = self.resolve_ids(tokenizer)?;

        let parts: Vec<&str> = text.split(AUDIO_PLACEHOLDER).collect();
        ensure!(
            !parts.is_empty() && parts.len() == audios.len() + 1,
            "prompt has {} <|audio|> placeholders but {} audio segments were provided",
            parts.len().saturating_sub(1),
            audios.len()
        );

        // tokenize segments separately; interleave eot runs per placeholder
        let mut input_ids: Vec<u32> = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            input_ids.extend(tokenizer.encode_no_special(part)?.iter().copied());
            if i < audios.len() {
                let n = audios[i].features.shape()[0];
                ensure!(n > 0, "audio segment {i} produced zero tokens");
                ranges.push((input_ids.len(), n));
                input_ids.extend(std::iter::repeat_n(eos_id, n));
            }
        }

        // token embeddings through the same row-copy ops the token path uses
        let embed_dim = match embed_table {
            LlamaEmbedding::F32(t) => t.shape()[1],
            LlamaEmbedding::Q8_0(w) => w.in_features(),
            LlamaEmbedding::KQuant(w) => w.in_features(),
        };
        let mut embeddings = backend.zeroes(&[input_ids.len(), embed_dim])?;
        for (row, &token) in input_ids.iter().enumerate() {
            match embed_table {
                LlamaEmbedding::F32(table) => {
                    backend.assign_row_from_table(&mut embeddings, row, table, token as usize)?;
                }
                LlamaEmbedding::Q8_0(table) => {
                    backend.assign_row_from_q8_0(&mut embeddings, row, table, token as usize)?;
                }
                LlamaEmbedding::KQuant(table) => {
                    backend.assign_row_from_k(&mut embeddings, row, table, token as usize)?;
                }
            }
        }

        // scatter projected audio rows over their eot runs
        for ((start, n), audio) in ranges.iter().zip(audios.iter()) {
            ensure!(
                *n == audio.features.shape()[0],
                "placeholder run length {n} != feature rows {}",
                audio.features.shape()[0]
            );
            for k in 0..*n {
                let dst = &mut embeddings.data_mut()
                    [(start + k) * embed_dim..(start + k + 1) * embed_dim];
                let src = &audio.features.data()[k * embed_dim..(k + 1) * embed_dim];
                dst.copy_from_slice(src);
            }
        }

        Ok(AssembledAudioSequence {
            input_ids,
            embeddings,
        })
    }
}

/// Progressive-validation intermediates of one audio prefill.
pub struct UltravoxTrace {
    pub mel: CpuTensor,
    pub encoder: AudioTrace,
    /// Projector output `[n_audio_tokens, llm_width]`.
    pub projector_output: CpuTensor,
    pub assembled_embeddings: CpuTensor,
    pub input_ids: Vec<u32>,
}

/// Ultravox v0.5: LLM + audio tower + projector + assembler.
pub struct Ultravox {
    pub llm: Llama<CpuBackend>,
    pub audio: AudioModel,
    pub assembler: UltravoxAssembler,
    /// Content hash of the audio mmproj GGUF: session feature-cache keys
    /// are invalid across different encoder weights (same contract as
    /// SmolVLM's `vision_identity`).
    pub audio_identity: u64,
}

impl Ultravox {
    /// Load the text GGUF (llama arch) and the audio GGUF with the default
    /// production K strategy (compressed-resident `auto`).
    pub fn from_ggufs(text_path: &std::path::Path, audio_path: &std::path::Path) -> Result<Self> {
        Self::from_ggufs_with_k_strategy(text_path, audio_path, crate::quant_k::KStrategy::Auto)
    }

    /// Load with an explicit K-family execution policy for the text model.
    /// `EagerF32` is the exact-f32 oracle path; `Auto` keeps Q4_K/Q6_K
    /// compressed-resident on the integer kernels. Every per-tensor decision
    /// (including fallbacks when allowed) is recorded in the loader.
    pub fn from_ggufs_with_k_strategy(
        text_path: &std::path::Path,
        audio_path: &std::path::Path,
        k_strategy: crate::quant_k::KStrategy,
    ) -> Result<Self> {
        let loader = load_gguf_with_k_strategy(text_path, k_strategy, false)
            .with_context(|| format!("failed to load text model {}", text_path.display()))?;
        let llm = Llama::from_loader(loader)
            .with_context(|| format!("failed to build LLM from {}", text_path.display()))?;
        // audio tower tensors are f32/f16 — K policy does not apply
        let mut mmproj = load_gguf(audio_path)
            .with_context(|| format!("failed to load audio model {}", audio_path.display()))?;
        let audio = AudioModel::from_mmproj_loader(&mut mmproj).with_context(|| {
            format!("failed to build audio model from {}", audio_path.display())
        })?;
        anyhow::ensure!(
            audio.projector.output_width() == llm.config.embed_dim,
            "audio projector output width {} does not match text embedding width {}",
            audio.projector.output_width(),
            llm.config.embed_dim
        );
        // content hash of the audio tower file (feature-cache identity).
        // Streamed with a hard cap and path identity checks so a
        // large/sparse mmproj cannot be materialized just to be hashed.
        let audio_identity = crate::loader::gguf_content_identity(audio_path)
            .context("failed to hash audio model for feature-cache identity")?;
        Ok(Self {
            llm,
            audio,
            assembler: UltravoxAssembler::default(),
            audio_identity,
        })
    }

    /// Preprocess one audio input into mel features (any length; long-form
    /// inputs are chunked at encode time).
    pub fn build_mel(&self, input: &AudioInput) -> Result<(CpuTensor, f64, f64, f64)> {
        let t0 = Instant::now();
        let decoded = to_mono_16k(input)?;
        let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
        let seconds = decoded.samples.len() as f64 / audio::TARGET_SAMPLE_RATE as f64;

        let t1 = Instant::now();
        let mel = audio::log_mel_spectrogram_full(&decoded.samples)?;
        let features_ms = t1.elapsed().as_secs_f64() * 1e3;
        Ok((mel, decode_ms, features_ms, seconds))
    }

    /// Encode + project mel into LLM-width tokens.
    pub fn encode_mel(
        &self,
        backend: &CpuBackend,
        mel: &CpuTensor,
    ) -> Result<(CpuTensor, AudioTrace, f64, f64)> {
        let t2 = Instant::now();
        let (encoder_out, trace) = self.audio.encoder.encode_traced(backend, mel)?;
        let encoder_ms = t2.elapsed().as_secs_f64() * 1e3;
        let t3 = Instant::now();
        let projected = self.audio.projector.forward(backend, &encoder_out)?;
        let projector_ms = t3.elapsed().as_secs_f64() * 1e3;
        Ok((projected, trace, encoder_ms, projector_ms))
    }

    /// Long-form entry point: chunk any mel longer than [`MAX_FRAMES`] into
    /// 30 s windows following the reference processor's protocol exactly
    /// (`_chunk_and_pad_audio`):
    ///
    /// - windows are `[0..3000), [3000..6000), …`;
    /// - a *continuation* window (offset > 0) shorter than the context is
    ///   zero-padded in the mel domain to exactly 3000 frames;
    /// - every window is encoded with an attention padding mask over its
    ///   valid output frames (`ceil(valid_mel/2)` after the conv frontend);
    /// - projected rows are truncated to `ceil(valid_frames/16)` tokens per
    ///   window and concatenated, so one `<|audio|>` placeholder still binds
    ///   all rows of one source segment.
    ///
    /// Single-window mel takes the historical unmasked path bit-identically.
    pub fn encode_mel_chunked(
        &self,
        backend: &CpuBackend,
        mel: &CpuTensor,
    ) -> Result<(CpuTensor, AudioTrace, f64, f64, usize)> {
        let total = mel.shape()[1];
        if total <= MAX_FRAMES {
            let (projected, trace, encoder_ms, projector_ms) = self.encode_mel(backend, mel)?;
            return Ok((projected, trace, encoder_ms, projector_ms, 1));
        }

        let n_mels = mel.shape()[0];
        let windows = audio::long_form_windows(total, MAX_FRAMES);
        let n_chunks = windows.len();
        let mut encoder_ms = 0.0f64;
        let mut projector_ms = 0.0f64;
        let mut chunk_features: Vec<CpuTensor> = Vec::with_capacity(n_chunks);
        let mut last_trace: Option<AudioTrace> = None;

        for (c, &(start, valid)) in windows.iter().enumerate() {
            let end = start + valid;

            // slice [n_mels, valid]
            let mut data = vec![0.0f32; n_mels * MAX_FRAMES];
            for j in 0..n_mels {
                let src = &mel.data()[j * total + start..j * total + end];
                data[j * MAX_FRAMES..j * MAX_FRAMES + valid].copy_from_slice(src);
            }
            // continuation chunks are zero-padded to the full window (the
            // zeros live in the normalized log-mel domain, like F.pad(...,0))
            let window = CpuTensor::from_data(vec![n_mels, MAX_FRAMES], data);

            let t_enc = Instant::now();
            // the final window is traced (progressive-validation dumps);
            // earlier windows run plain
            let (enc_out, trace) = if c + 1 == n_chunks {
                let (o, tr) = self
                    .audio
                    .encoder
                    .encode_with_padding_mask_traced(backend, &window, valid)?;
                (o, Some(tr))
            } else {
                let o = self
                    .audio
                    .encoder
                    .encode_with_padding_mask(backend, &window, valid)?;
                (o, None)
            };
            encoder_ms += t_enc.elapsed().as_secs_f64() * 1e3;
            let t_proj = Instant::now();
            let projected = self.audio.projector.forward(backend, &enc_out)?;
            projector_ms += t_proj.elapsed().as_secs_f64() * 1e3;

            // truncate to this window's real token count: ceil(valid/16)
            // (= encoder_ds_factor 2 × stack_factor 8, reference formula)
            let token_len = valid.div_ceil(16);
            anyhow::ensure!(
                token_len <= projected.shape()[0],
                "chunk {c}: token_len {token_len} exceeds projected rows {}",
                projected.shape()[0]
            );
            let width = projected.shape()[1];
            chunk_features.push(CpuTensor::from_data(
                vec![token_len, width],
                projected.data()[..token_len * width].to_vec(),
            ));
            if let Some(tr) = trace {
                last_trace = Some(tr);
            }
        }

        let rows: usize = chunk_features.iter().map(|t| t.shape()[0]).sum();
        let width = chunk_features[0].shape()[1];
        let mut all = Vec::with_capacity(rows * width);
        for t in &chunk_features {
            all.extend_from_slice(t.data());
        }
        // trace of the final (masked) window for validation dumps
        let trace = last_trace.expect("final chunk always traced");
        Ok((
            CpuTensor::from_data(vec![rows, width], all),
            trace,
            encoder_ms,
            projector_ms,
            n_chunks,
        ))
    }

    /// Build the full prefill inputs for `text` + one or more audio inputs.
    ///
    /// Every [`AUDIO_PLACEHOLDER`] in `text` binds to one entry of
    /// `audios`, in order of appearance. The returned trace carries the
    /// *last* segment's boundaries (validation uses single-segment inputs).
    #[allow(clippy::type_complexity)]
    pub fn build_inputs(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        text: &str,
        audios: &[AudioInput],
        start_pos: usize,
    ) -> Result<(UltravoxTrace, EmbeddingSequence<CpuBackend>, AudioTimings)> {
        let mut timings = AudioTimings::default();

        let mut features = Vec::new();
        let mut last_trace: Option<AudioTrace> = None;
        let mut last_mel: Option<CpuTensor> = None;
        let mut last_projected: Option<CpuTensor> = None;
        timings.n_chunks = 0;
        for input in audios {
            let (mel, decode_ms, features_ms, seconds) = self.build_mel(input)?;
            timings.decode_ms += decode_ms;
            timings.features_ms += features_ms;
            timings.audio_seconds += seconds;
            let (projected, trace, encoder_ms, projector_ms, n_chunks) =
                self.encode_mel_chunked(backend, &mel)?;
            timings.encoder_ms += encoder_ms;
            timings.projector_ms += projector_ms;
            timings.n_chunks += n_chunks;
            last_trace = Some(trace);
            last_mel = Some(mel);
            last_projected = Some(projected.clone());
            features.push(AudioFeatures {
                features: projected,
            });
        }

        let rendered = self.assembler.render_chat_template(text);
        let assembled = self.assembler.assemble(
            backend,
            tokenizer,
            &rendered,
            &features,
            &self.llm.embed_tokens,
        )?;

        let trace = UltravoxTrace {
            mel: last_mel.expect("at least one audio segment"),
            encoder: last_trace.expect("at least one audio segment"),
            projector_output: last_projected.expect("at least one audio segment"),
            assembled_embeddings: assembled.embeddings.clone(),
            input_ids: assembled.input_ids.clone(),
        };
        let sequence = EmbeddingSequence::causal(assembled.embeddings, start_pos);
        Ok((trace, sequence, timings))
    }

    /// Greedy generation over text + audio with separate stage timings.
    pub fn generate_with_audio(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        prompt: &str,
        audios: &[AudioInput],
        max_tokens: usize,
    ) -> Result<(Vec<u32>, String, AudioTimings)> {
        let wall_start = Instant::now();
        ensure!(
            !audios.is_empty(),
            "generate_with_audio requires at least one audio input"
        );

        // prefill on assembled embeddings; first token comes from the
        // prefill's last-position logits (reference generate() semantics)
        let (trace, sequence, timings) =
            self.build_inputs(backend, tokenizer, prompt, audios, 0)?;
        let start_pos = trace.input_ids.len();

        let mut cache = self
            .llm
            .create_request_cache(backend, trace.input_ids.len(), max_tokens);
        let t3 = Instant::now();
        let mut logits = self.llm.forward_last_logits_embeddings_with_cache(
            backend,
            &sequence.embeddings,
            &mut cache,
            0,
        )?;
        // prefill timing includes assembly + encode stages already recorded
        // in `timings`; measure the transformer pass alone here
        let llm_ms = t3.elapsed().as_secs_f64() * 1e3;
        let mut timings = timings;
        timings.llm_prefill_ms = llm_ms;

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
                    start_pos + step,
                )?;
            }
        }
        timings.n_decode_tokens = generated.len();
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

// ---------------------------------------------------------------------------
// Streaming audio encoder scheduling (Phase 4 session 2, Track C4)
//
// The Whisper encoder has no recurrent state; honest streaming therefore
// means an explicit recompute policy on top of the validated static
// long-form path:
//
// ```text
// completed 30 s window  -> encode once -> cache projected rows
// active partial window  -> encode only when provisional inference is
//                           requested; re-encoded from scratch each time
// finish                 -> final window encoded from the definitive mel;
//                           cached windows validated against the final
//                           global log-mel floor and re-encoded when stale
// ```
//
// The subtlety the policy must respect is the **global** Whisper
// normalization: `max - 8` spans the whole utterance, so a window encoded
// mid-stream under the then-current running max is bit-valid at finish
// only if that floor equals the final one (`floor_used ==
// StreamedAudio::floor_log`). Raw log-mel columns are immutable once
// finalized, so stale windows are rebuilt exactly (no frontend
// recomputation) -- only encoder/projector work repeats, and it is counted
// in `reencoded_seconds`, never hidden.
// ---------------------------------------------------------------------------

/// Separate encoder/projector wall timings for one window pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct WindowTimings {
    pub encoder_ms: f64,
    pub projector_ms: f64,
}

/// Encode+project ONE zero-padded long-form window exactly as the static
/// chunked path does (padding mask over positions >= ceil(valid/2); for
/// full windows the mask is all-zero, which leaves results unchanged).
/// Implemented by [`AudioModel`]; a seam so scheduling policies can be
/// unit-tested without tower weights. Every variant returns separate
/// encoder/projector wall timings so recompute accounting stays honest.
pub trait AudioWindowEncoder {
    /// `window` is `[n_mels, MAX_FRAMES]` in the normalized mel domain
    /// (zero-padded); `valid_mel_frames` counts the unpadded frames.
    fn encode_project_window(
        &self,
        backend: &CpuBackend,
        window: &CpuTensor,
        valid_mel_frames: usize,
    ) -> Result<(CpuTensor, WindowTimings)>;

    /// [`Self::encode_project_window`] with optional progressive-validation
    /// intermediates of the encoder pass (real towers only).
    fn encode_project_window_traced(
        &self,
        backend: &CpuBackend,
        window: &CpuTensor,
        valid_mel_frames: usize,
    ) -> Result<(CpuTensor, Option<AudioTrace>, WindowTimings)> {
        let (out, t) = self.encode_project_window(backend, window, valid_mel_frames)?;
        Ok((out, None, t))
    }

    /// Static SINGLE-window semantics: an UNMASKED encode over exactly the
    /// given mel (no zero-padding, `window` length == mel length).
    fn encode_project_window_unmasked(
        &self,
        backend: &CpuBackend,
        window: &CpuTensor,
    ) -> Result<(CpuTensor, Option<AudioTrace>, WindowTimings)>;
}

impl AudioWindowEncoder for AudioModel {
    fn encode_project_window(
        &self,
        backend: &CpuBackend,
        window: &CpuTensor,
        valid_mel_frames: usize,
    ) -> Result<(CpuTensor, WindowTimings)> {
        let t0 = Instant::now();
        let enc_out = self
            .encoder
            .encode_with_padding_mask(backend, window, valid_mel_frames)
            .map_err(|e| anyhow::anyhow!("audio window encode failed: {e}"))?;
        let encoder_ms = t0.elapsed().as_secs_f64() * 1e3;
        let projected = self
            .projector
            .forward(backend, &enc_out)
            .map_err(|e| anyhow::anyhow!("audio projector failed: {e}"))?;
        Ok((
            projected,
            WindowTimings {
                encoder_ms,
                projector_ms: t0.elapsed().as_secs_f64() * 1e3 - encoder_ms,
            },
        ))
    }

    fn encode_project_window_traced(
        &self,
        backend: &CpuBackend,
        window: &CpuTensor,
        valid_mel_frames: usize,
    ) -> Result<(CpuTensor, Option<AudioTrace>, WindowTimings)> {
        let t0 = Instant::now();
        let (enc_out, trace) = self
            .encoder
            .encode_with_padding_mask_traced(backend, window, valid_mel_frames)
            .map_err(|e| anyhow::anyhow!("audio window encode failed: {e}"))?;
        let encoder_ms = t0.elapsed().as_secs_f64() * 1e3;
        let projected = self
            .projector
            .forward(backend, &enc_out)
            .map_err(|e| anyhow::anyhow!("audio projector failed: {e}"))?;
        Ok((
            projected,
            Some(trace),
            WindowTimings {
                encoder_ms,
                projector_ms: t0.elapsed().as_secs_f64() * 1e3 - encoder_ms,
            },
        ))
    }

    fn encode_project_window_unmasked(
        &self,
        backend: &CpuBackend,
        window: &CpuTensor,
    ) -> Result<(CpuTensor, Option<AudioTrace>, WindowTimings)> {
        let t0 = Instant::now();
        let (enc_out, trace) = self
            .encoder
            .encode_traced(backend, window)
            .map_err(|e| anyhow::anyhow!("audio encode failed: {e}"))?;
        let encoder_ms = t0.elapsed().as_secs_f64() * 1e3;
        let projected = self
            .projector
            .forward(backend, &enc_out)
            .map_err(|e| anyhow::anyhow!("audio projector failed: {e}"))?;
        Ok((
            projected,
            Some(trace),
            WindowTimings {
                encoder_ms,
                projector_ms: t0.elapsed().as_secs_f64() * 1e3 - encoder_ms,
            },
        ))
    }
}

/// Projected features of one finalized 30 s window plus the floor they
/// were computed under.
#[derive(Debug)]
struct CachedWindow {
    /// `[ceil(MAX_FRAMES/16), llm_width]` truncated projected rows.
    features: CpuTensor,
    /// The `running_max - 8` floor used to normalize this window's mel.
    floor_used: f64,
}

/// Cumulative scheduler counters (never decrease within one stream).
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamEncodeStats {
    /// Cumulative encoder wall time across all encodes, ms.
    pub encoder_wall_ms: f64,
    /// Cumulative projector wall time across all encodes, ms.
    pub projector_wall_ms: f64,
    /// Seconds of audio whose encode ran for the first time (finalized
    /// windows only; first active-window encodes count as re-encodes).
    pub freshly_encoded_seconds: f64,
    /// Seconds of audio whose encode ran more than once: every
    /// active-window inference plus finish-time stale-window rebuilds.
    pub reencoded_seconds: f64,
}

/// One incremental update from [`Ultravox::stream_update`].
#[derive(Debug)]
pub struct StreamUpdate {
    // ---- cumulative schedule state ----
    /// Finalized windows encoded and cached so far.
    pub finalized_windows: usize,
    /// Immutable frames beyond the last fixed window boundary.
    pub active_window_frames: usize,
    /// Cumulative encode counters (see [`StreamEncodeStats`]).
    pub totals: StreamEncodeStats,

    // ---- this call's deltas ----
    pub new_audio_samples: u64,
    pub new_audio_seconds: f64,
    pub freshly_encoded_seconds: f64,
    pub reencoded_seconds: f64,
    pub encoder_wall_ms: f64,
    pub projector_wall_ms: f64,
    /// Provisional projection of the active partial window when requested
    /// (and enough frames exist). Explicitly unstable: built under the
    /// running floor and replaced wholesale as audio arrives.
    pub active_window_features: Option<CpuTensor>,
}

/// Per-stream schedule state: cached finalized-window projections plus the
/// honest accounting of what was recomputed. Created per open stream via
/// [`Ultravox::stream_schedule_new`].
#[derive(Debug, Default)]
pub struct StreamingSchedule {
    cached: Vec<CachedWindow>,
    last_progress: Option<StreamProgress>,
    stats: StreamEncodeStats,
    /// Provisional active-window features from the last update.
    active_window_features: Option<CpuTensor>,
}

impl StreamingSchedule {
    /// Concatenated projected rows of all finalized windows, in window
    /// order -- the `<|audio|>` prefix available before finish. PROVISIONAL
    /// until [`Ultravox::stream_finish`] validates floors: a window whose
    /// encode-time floor went stale is replaced by finish.
    pub fn finalized_prefix_features(&self) -> Option<CpuTensor> {
        let parts: Vec<&CpuTensor> = self.cached.iter().map(|c| &c.features).collect();
        concat_rows(&parts)
    }

    /// Provisional active-window features from the last update, if any.
    pub fn active_window_features(&self) -> Option<&CpuTensor> {
        self.active_window_features.as_ref()
    }
}

fn concat_rows(parts: &[&CpuTensor]) -> Option<CpuTensor> {
    if parts.is_empty() {
        return None;
    }
    let rows: usize = parts.iter().map(|t| t.shape()[0]).sum();
    let width = parts[0].shape()[1];
    let mut all = Vec::with_capacity(rows * width);
    for t in parts {
        all.extend_from_slice(t.data());
    }
    Some(CpuTensor::from_data(vec![rows, width], all))
}

/// Minimum immutable frames for a provisional active-window encode (the
/// conv frontend needs >= 4 mel frames to produce output).
const MIN_ACTIVE_ENCODE_FRAMES: usize = 8;

impl Ultravox {
    /// Fresh schedule state for one streaming input.
    pub fn stream_schedule_new() -> StreamingSchedule {
        StreamingSchedule::default()
    }

    /// Incremental update after one or more [`AudioStream::push_pcm`] calls.
    ///
    /// Work performed (only):
    /// - windows that became fully determined since the last update are
    ///   encoded ONCE and cached (`freshly_encoded_seconds`);
    /// - when `infer_active_window` is set, the trailing partial window is
    ///   re-encoded from scratch under the running floor
    ///   (`reencoded_seconds`) -- Whisper has no encoder state to reuse.
    pub fn stream_update(
        &self,
        backend: &CpuBackend,
        sched: &mut StreamingSchedule,
        stream: &AudioStream,
        infer_active_window: bool,
    ) -> Result<StreamUpdate> {
        schedule_stream_update(&self.audio, backend, sched, stream, infer_active_window)
    }

    /// Thin typed wrapper over the crate-internal `schedule_stream_finish`.
    pub fn stream_finish(
        &self,
        backend: &CpuBackend,
        sched: StreamingSchedule,
        streamed: StreamedAudio,
    ) -> Result<(CpuTensor, StreamEncodeStats, Option<AudioTrace>, usize)> {
        schedule_stream_finish(
            &self.audio,
            self.audio.encoder.config.max_source_positions * 2,
            backend,
            sched,
            streamed,
        )
    }
}

/// The scheduling policy proper (free function over any
/// [`AudioWindowEncoder`] so unit tests can drive it with fakes).
pub(crate) fn schedule_stream_update<E: AudioWindowEncoder + ?Sized>(
    encoder: &E,
    backend: &CpuBackend,
    sched: &mut StreamingSchedule,
    stream: &AudioStream,
    infer_active_window: bool,
) -> Result<StreamUpdate> {
    let progress = stream.progress();
    let (new_samples, new_seconds) = match sched.last_progress {
        Some(last) => (
            (progress.input_samples - last.input_samples) as u64,
            progress.seconds_received - last.seconds_received,
        ),
        None => (progress.input_samples as u64, progress.seconds_received),
    };
    sched.last_progress = Some(progress);

    let mut delta = StreamUpdate {
        finalized_windows: sched.cached.len(),
        active_window_frames: 0,
        totals: sched.stats,
        new_audio_samples: new_samples,
        new_audio_seconds: new_seconds,
        freshly_encoded_seconds: 0.0,
        reencoded_seconds: 0.0,
        encoder_wall_ms: 0.0,
        projector_wall_ms: 0.0,
        active_window_features: None,
    };

    // -- newly finalized full windows: encode once --
    let fixed = stream.fixed_windows_finalized();
    while sched.cached.len() < fixed {
        let w = sched.cached.len();
        let start = w * MAX_FRAMES;
        let floor = stream
            .running_floor()
            .ok_or_else(|| anyhow::anyhow!("window {w} finalized but no running floor"))?;
        let mel_win = stream.window_mel_with_floor(start, MAX_FRAMES, floor)?;
        let (projected, timing) = encoder.encode_project_window(backend, &mel_win, MAX_FRAMES)?;
        // truncate to this window's real token count (reference formula)
        let projected = truncate_rows(&projected, MAX_FRAMES.div_ceil(16))?;
        let (encoder_ms, projector_ms) = (timing.encoder_ms, timing.projector_ms);
        let fresh_secs = MAX_FRAMES as f64 / 100.0; // hop 160 @16 kHz
        sched.cached.push(CachedWindow {
            features: projected,
            floor_used: floor,
        });
        sched.stats.freshly_encoded_seconds += fresh_secs;
        sched.stats.encoder_wall_ms += encoder_ms;
        sched.stats.projector_wall_ms += projector_ms.max(0.0);
        delta.freshly_encoded_seconds += fresh_secs;
        delta.encoder_wall_ms += encoder_ms;
        delta.projector_wall_ms += projector_ms.max(0.0);
    }

    // -- optional provisional inference over the active window --
    let finalized = stream.finalized_frames();
    let active_start = fixed * MAX_FRAMES;
    let active_frames = finalized.saturating_sub(active_start);
    if infer_active_window && active_frames >= MIN_ACTIVE_ENCODE_FRAMES {
        let floor = stream.running_floor().ok_or_else(|| {
            anyhow::anyhow!("active window has {active_frames} frames but no running floor")
        })?;
        let partial = stream.window_mel_with_floor(active_start, active_frames, floor)?;
        let padded = zero_pad_window(&partial, MAX_FRAMES);
        let (projected, timing) = encoder.encode_project_window(backend, &padded, active_frames)?;
        let projected = truncate_rows(&projected, active_frames.div_ceil(16))?;
        let (encoder_ms, projector_ms) = (timing.encoder_ms, timing.projector_ms);
        let re_secs = active_frames as f64 / 100.0;
        sched.stats.reencoded_seconds += re_secs;
        sched.stats.encoder_wall_ms += encoder_ms;
        sched.stats.projector_wall_ms += projector_ms.max(0.0);
        delta.reencoded_seconds += re_secs;
        delta.encoder_wall_ms += encoder_ms;
        delta.projector_wall_ms += projector_ms.max(0.0);
        sched.active_window_features = Some(projected);
    } else if !infer_active_window {
        sched.active_window_features = None;
    }
    delta
        .active_window_features
        .clone_from(&sched.active_window_features);

    delta.finalized_windows = sched.cached.len();
    delta.active_window_frames = active_frames;
    delta.totals = sched.stats;
    Ok(delta)
}

/// Finish-time validation + final-window encode (see
/// [`Ultravox::stream_finish`]); `max_window_input` is the encoder's
/// maximum mel input length (3000 for Whisper towers).
pub(crate) fn schedule_stream_finish<E: AudioWindowEncoder + ?Sized>(
    encoder: &E,
    max_window_input: usize,
    backend: &CpuBackend,
    mut sched: StreamingSchedule,
    streamed: StreamedAudio,
) -> Result<(CpuTensor, StreamEncodeStats, Option<AudioTrace>, usize)> {
    debug_assert_eq!(max_window_input, MAX_FRAMES);
    let total = streamed.mel.shape()[1];
    ensure!(total > 0, "streamed audio produced zero mel frames");
    let n_mels = streamed.mel.shape()[0];

    // -- validate/rebuild stale cached windows --
    for w in 0..sched.cached.len() {
        if sched.cached[w].floor_used == streamed.floor_log {
            continue; // bit-valid: same raw columns, same explicit floor
        }
        let start = w * MAX_FRAMES;
        let mut data = vec![0.0f32; n_mels * MAX_FRAMES];
        for j in 0..n_mels {
            let src = &streamed.mel.data()[j * total + start..j * total + start + MAX_FRAMES];
            data[j * MAX_FRAMES..(j + 1) * MAX_FRAMES].copy_from_slice(src);
        }
        let window = CpuTensor::from_data(vec![n_mels, MAX_FRAMES], data);
        let (projected, timing) = encoder.encode_project_window(backend, &window, MAX_FRAMES)?;
        let projected = truncate_rows(&projected, MAX_FRAMES.div_ceil(16))?;
        let (encoder_ms, projector_ms) = (timing.encoder_ms, timing.projector_ms);
        let re_secs = MAX_FRAMES as f64 / 100.0;
        sched.stats.reencoded_seconds += re_secs;
        sched.stats.encoder_wall_ms += encoder_ms;
        sched.stats.projector_wall_ms += projector_ms.max(0.0);
        sched.cached[w] = CachedWindow {
            features: projected,
            floor_used: streamed.floor_log,
        };
    }

    // -- final window (always encoded here, matching static semantics) --
    let start = sched.cached.len() * MAX_FRAMES;
    ensure!(
        start < total || sched.cached.is_empty(),
        "window layout drift: {start} >= {total}"
    );
    let valid = total - start;
    let single_window = sched.cached.is_empty() && total <= MAX_FRAMES;
    let win_len = if single_window { total } else { MAX_FRAMES };
    let mut data = vec![0.0f32; n_mels * win_len];
    for j in 0..n_mels {
        let src = &streamed.mel.data()[j * total + start..j * total + start + valid];
        data[j * win_len..j * win_len + valid].copy_from_slice(src);
    }
    let window = CpuTensor::from_data(vec![n_mels, win_len], data);

    // single-window case mirrors the static path's unmasked traced encode;
    // multi-window mirrors its final masked (optionally traced) window.
    let (projected_full, trace, timing) = if single_window {
        encoder.encode_project_window_unmasked(backend, &window)?
    } else {
        encoder.encode_project_window_traced(backend, &window, valid)?
    };
    let (encoder_ms, projector_ms) = (timing.encoder_ms, timing.projector_ms);
    let final_secs = valid as f64 / 100.0;
    sched.stats.freshly_encoded_seconds += final_secs;
    sched.stats.encoder_wall_ms += encoder_ms;
    sched.stats.projector_wall_ms += projector_ms;

    let token_len = valid.div_ceil(16);
    let projected_final = truncate_rows(&projected_full, token_len)?;

    // -- assemble all rows in window order (== static layout) --
    let mut parts: Vec<&CpuTensor> = sched.cached.iter().map(|c| &c.features).collect();
    parts.push(&projected_final);
    let all = concat_rows(&parts).expect("at least the final window");
    let n_windows = sched.cached.len() + 1;
    Ok((all, sched.stats, trace, n_windows))
}

/// Truncate projected rows to the reference per-window token count.
fn truncate_rows(projected: &CpuTensor, token_len: usize) -> Result<CpuTensor> {
    let rows = projected.shape()[0];
    let width = projected.shape()[1];
    ensure!(
        token_len <= rows,
        "token_len {token_len} exceeds projected rows {rows}"
    );
    Ok(CpuTensor::from_data(
        vec![token_len, width],
        projected.data()[..token_len * width].to_vec(),
    ))
}

/// Zero-pad a `[n_mels, valid]` mel slice to `[n_mels, context]` in the
/// normalized domain (reference `F.pad(..., 0)` semantics).
fn zero_pad_window(partial: &CpuTensor, context: usize) -> CpuTensor {
    let n_mels = partial.shape()[0];
    let valid = partial.shape()[1];
    debug_assert!(valid <= context);
    let mut data = vec![0.0f32; n_mels * context];
    for j in 0..n_mels {
        let src = &partial.data()[j * valid..(j + 1) * valid];
        data[j * context..j * context + valid].copy_from_slice(src);
    }
    CpuTensor::from_data(vec![n_mels, context], data)
}

#[cfg(test)]
mod stream_schedule_tests {
    use super::*;
    use crate::multimodal::stream::AudioStreamConfig;
    use std::cell::RefCell;

    /// Records every window request; returns deterministic features with
    /// the exact reference row count (`ceil(valid/16)`).
    struct FakeEncoder {
        width: usize,
        calls: RefCell<Vec<(usize, usize)>>, // (valid_mel_frames, window_len)
    }

    impl FakeEncoder {
        fn new(width: usize) -> Self {
            Self {
                width,
                calls: RefCell::new(Vec::new()),
            }
        }

        fn record(&self, valid: usize, window_len: usize) -> (CpuTensor, WindowTimings) {
            self.calls.borrow_mut().push((valid, window_len));
            let rows = valid.div_ceil(16);
            let data = vec![valid as f32 + window_len as f32 * 1e-3; rows * self.width];
            (
                CpuTensor::from_data(vec![rows, self.width], data),
                WindowTimings::default(),
            )
        }
    }

    impl AudioWindowEncoder for FakeEncoder {
        fn encode_project_window(
            &self,
            _backend: &CpuBackend,
            window: &CpuTensor,
            valid_mel_frames: usize,
        ) -> Result<(CpuTensor, WindowTimings)> {
            Ok(self.record(valid_mel_frames, window.shape()[1]))
        }

        fn encode_project_window_unmasked(
            &self,
            _backend: &CpuBackend,
            window: &CpuTensor,
        ) -> Result<(CpuTensor, Option<AudioTrace>, WindowTimings)> {
            let valid = window.shape()[1];
            let (out, t) = self.record(valid, valid);
            Ok((out, None, t))
        }
    }

    #[test]
    fn finalized_window_encoded_once_active_reencoded_on_request() {
        let backend = CpuBackend;
        let encoder = FakeEncoder::new(64);
        let mut stream = AudioStream::open(AudioStreamConfig::default()).expect("open stream");
        let mut sched = Ultravox::stream_schedule_new();

        // 29 s quiet audio: everything stays in the active window
        let quiet = vec![0.01f32; 29 * 16_000];
        for part in quiet.chunks(16_000) {
            stream.push_pcm(part).unwrap();
            let u = schedule_stream_update(&encoder, &backend, &mut sched, &stream, true).unwrap();
            assert_eq!(u.finalized_windows, 0);
            assert_eq!(u.freshly_encoded_seconds, 0.0);
            assert!(u.reencoded_seconds > 0.0, "active window must re-encode");
            assert!(u.active_window_features.is_some());
        }
        let calls_before_boundary = encoder.calls.borrow().len();

        // cross the 30 s boundary -> window 0 becomes finalized exactly once
        // (needs > 30 s: the frontend keeps a 600-sample finalize margin)
        stream.push_pcm(&[0.9f32; 16_000]).unwrap();
        stream.push_pcm(&[0.9f32; 16_000]).unwrap();
        let u = schedule_stream_update(&encoder, &backend, &mut sched, &stream, false).unwrap();
        assert_eq!(u.finalized_windows, 1);
        assert_eq!(u.freshly_encoded_seconds, 30.0);
        // no active inference requested -> no re-encode this call
        assert_eq!(u.reencoded_seconds, 0.0);
        assert!(u.active_window_features.is_none());

        // more audio + updates without active inference: zero encode work
        stream.push_pcm(&[0.9f32; 16_000]).unwrap();
        let _ = schedule_stream_update(&encoder, &backend, &mut sched, &stream, false).unwrap();
        let total_calls = encoder.calls.borrow().len();
        assert_eq!(
            total_calls,
            calls_before_boundary + 1,
            "exactly one finalized-window encode expected"
        );

        // the one finalized call was a full unmasked-length window
        let calls = encoder.calls.borrow();
        assert_eq!(calls[calls_before_boundary], (MAX_FRAMES, MAX_FRAMES));

        // prefix features available before finish
        let prefix = sched.finalized_prefix_features().unwrap();
        assert_eq!(prefix.shape(), &[MAX_FRAMES.div_ceil(16), 64]);
    }

    /// Fabricate a finished result so finish-policy tests stay fast.
    fn fabricated_streamed(total_frames: usize, floor_log: f64) -> StreamedAudio {
        let n_mels = crate::multimodal::audio::N_MELS;
        let mut data = vec![0.25f32; n_mels * total_frames];
        // deterministic variation so rebuilds consume real values
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i % 97) as f32) * 0.01;
        }
        StreamedAudio {
            mel: CpuTensor::from_data(vec![n_mels, total_frames], data),
            encoder_windows: crate::multimodal::audio::long_form_windows(total_frames, MAX_FRAMES),
            floor_log,
            input_samples: total_frames * 160,
            input_sample_rate: 16_000,
            samples_16k: total_frames * 160,
            input_seconds: total_frames as f64 / 100.0,
        }
    }

    #[test]
    fn finish_rebuilds_stale_windows_and_counts_them() {
        let backend = CpuBackend;
        // two cached windows whose floors went stale against final floor
        let mut sched = Ultravox::stream_schedule_new();
        for floor in [5.0f64, 6.0] {
            let rows = MAX_FRAMES.div_ceil(16);
            sched.cached.push(CachedWindow {
                features: CpuTensor::from_data(vec![rows, 64], vec![0.0; rows * 64]),
                floor_used: floor,
            });
        }
        let encoder = FakeEncoder::new(64);
        // 2 full windows + 123-frame tail
        let streamed = fabricated_streamed(2 * MAX_FRAMES + 123, 42.0);

        let (all, stats, _, n_windows) =
            schedule_stream_finish(&encoder, MAX_FRAMES, &backend, sched, streamed).unwrap();

        // both stale windows rebuilt (valid=3000) + final tail window encoded
        let calls = encoder.calls.borrow();
        assert_eq!(calls.len(), 3, "two rebuilds + one final-window encode");
        assert_eq!(calls[0], (MAX_FRAMES, MAX_FRAMES));
        assert_eq!(calls[1], (MAX_FRAMES, MAX_FRAMES));
        assert_eq!(
            calls[2],
            (123, MAX_FRAMES),
            "tail is zero-padded to context"
        );

        // layout matches static long-form token accounting
        assert_eq!(n_windows, 3);
        assert_eq!(
            all.shape()[0],
            188 + 188 + 123usize.div_ceil(16),
            "token rows must equal sum of ceil(valid/16)"
        );
        // honest cost accounting: 2 x 30 s re-encodes + fresh final seconds
        assert!((stats.reencoded_seconds - 60.0).abs() < 1e-9);
        assert!((stats.freshly_encoded_seconds - 1.23).abs() < 1e-9);
    }

    #[test]
    fn finish_keeps_fresh_windows_without_reencode() {
        let backend = CpuBackend;
        let mut sched = Ultravox::stream_schedule_new();
        let rows = MAX_FRAMES.div_ceil(16);
        sched.cached.push(CachedWindow {
            features: CpuTensor::from_data(vec![rows, 64], vec![7.0; rows * 64]),
            floor_used: 42.0, // == streamed.floor_log below
        });
        let encoder = FakeEncoder::new(64);
        let streamed = fabricated_streamed(MAX_FRAMES + 500, 42.0);

        let (all, stats, _, _) =
            schedule_stream_finish(&encoder, MAX_FRAMES, &backend, sched, streamed).unwrap();

        // only the final window was encoded now
        let calls = encoder.calls.borrow();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], (500, MAX_FRAMES));
        // cached rows reused verbatim (content marker 7.0 survives)
        assert_eq!(all.data()[0], 7.0);
        assert_eq!(stats.reencoded_seconds, 0.0);
    }

    #[test]
    fn single_window_finish_takes_unmasked_path_with_exact_length() {
        let backend = CpuBackend;
        let sched = Ultravox::stream_schedule_new();
        let encoder = FakeEncoder::new(64);
        let streamed = fabricated_streamed(1234, 42.0);

        let (all, _, _, n_windows) =
            schedule_stream_finish(&encoder, MAX_FRAMES, &backend, sched, streamed).unwrap();
        assert_eq!(n_windows, 1);
        assert_eq!(
            encoder.calls.borrow()[0],
            (1234, 1234),
            "single window must be encoded unpadded/unmasked"
        );
        assert_eq!(all.shape()[0], 1234usize.div_ceil(16));
    }
}
