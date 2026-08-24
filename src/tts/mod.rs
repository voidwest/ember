//! Speech-output support: text-to-speech model adapters and their codecs.
//!
//! Direction kept clean (no speech-decoder logic leaks into the LLM):
//!
//! ```text
//! Session / LLM (text tokens)
//!     -> speech adapter (OuteTTS-style codec-token generator)
//!     -> acoustic representation (codec token ids)
//!     -> codec decoder (wavtokenizer)  <- this module tree
//!     -> PCM
//! ```
pub mod outetts;
pub mod vits;
pub mod wavtokenizer;

use crate::backend::CpuBackend;
use anyhow::Result;

/// Engine-agnostic speech-output seam (Phase 5 Session 2 Track D): lets the
/// conversation driver compose with any validated synthesizer.
pub trait SpeechOut {
    fn stream_speech(
        &self,
        backend: &CpuBackend,
        text: &str,
        max_tokens: usize,
        chunk_tokens: usize,
        on_chunk: &mut dyn FnMut(outetts::AudioChunkMeta) -> bool,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<(Vec<f32>, Vec<u32>, outetts::TtsTimings)>;

    fn sample_rate(&self) -> u32;
}

impl SpeechOut for outetts::OuteTts {
    fn stream_speech(
        &self,
        backend: &CpuBackend,
        text: &str,
        max_tokens: usize,
        chunk_tokens: usize,
        on_chunk: &mut dyn FnMut(outetts::AudioChunkMeta) -> bool,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<(Vec<f32>, Vec<u32>, outetts::TtsTimings)> {
        self.synthesize_streaming(backend, text, max_tokens, chunk_tokens, on_chunk, on_token)
    }

    fn sample_rate(&self) -> u32 {
        self.codec.config.sample_rate
    }
}

impl SpeechOut for vits::MmsVits {
    fn stream_speech(
        &self,
        backend: &CpuBackend,
        text: &str,
        max_tokens: usize,
        chunk_tokens: usize,
        on_chunk: &mut dyn FnMut(outetts::AudioChunkMeta) -> bool,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<(Vec<f32>, Vec<u32>, outetts::TtsTimings)> {
        self.synthesize_streaming(backend, text, max_tokens, chunk_tokens, on_chunk, on_token)
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}
