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
use crate::loader::load_gguf;
use crate::model::ForwardModel;
use crate::multimodal::audio::{self, log_mel_spectrogram, to_mono_16k, AudioInput};
use crate::multimodal::audio_encoder::{AudioModel, AudioTrace};
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
}

impl Ultravox {
    /// Load the text GGUF (llama arch) and the audio GGUF.
    pub fn from_ggufs(text_path: &std::path::Path, audio_path: &std::path::Path) -> Result<Self> {
        let loader = load_gguf(text_path)
            .with_context(|| format!("failed to load text model {}", text_path.display()))?;
        let llm = Llama::from_loader(loader)
            .with_context(|| format!("failed to build LLM from {}", text_path.display()))?;
        let mut mmproj = load_gguf(audio_path)
            .with_context(|| format!("failed to load audio model {}", audio_path.display()))?;
        let audio = AudioModel::from_mmproj_loader(&mut mmproj).with_context(|| {
            format!("failed to build audio model from {}", audio_path.display())
        })?;
        Ok(Self {
            llm,
            audio,
            assembler: UltravoxAssembler::default(),
        })
    }

    /// Preprocess one audio input into mel features.
    pub fn build_mel(&self, input: &AudioInput) -> Result<(CpuTensor, f64, f64, f64)> {
        let t0 = Instant::now();
        let decoded = to_mono_16k(input)?;
        let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
        let seconds = decoded.samples.len() as f64 / audio::TARGET_SAMPLE_RATE as f64;

        let t1 = Instant::now();
        let mel = log_mel_spectrogram(&decoded.samples)?;
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
        for input in audios {
            let (mel, decode_ms, features_ms, seconds) = self.build_mel(input)?;
            timings.decode_ms += decode_ms;
            timings.features_ms += features_ms;
            timings.audio_seconds += seconds;
            let (projected, trace, encoder_ms, projector_ms) = self.encode_mel(backend, &mel)?;
            timings.encoder_ms += encoder_ms;
            timings.projector_ms += projector_ms;
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
            .create_cache(backend, self.llm.max_seq_len(backend));
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
