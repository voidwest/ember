//! Output-side modality boundary + model capability description.
//!
//! [`OutputEvent`] is the general generation-output surface: today Ember's
//! models produce only text deltas, and nothing in this module pretends
//! otherwise — there is no audio-producing decoder implemented yet. The
//! enum exists so that adding one (codec-token decoder, vocoder, …) is an
//! adapter behind this boundary rather than a redesign of every generate()
//! call site.
//!
//! [`ModelCapabilities`] describes what a loaded model actually accepts and
//! produces, derived from the wrapper implementation (not marketing
//! metadata): applications should reject unsupported requests before
//! inference instead of failing deep inside a tower.

/// One streamed generation event.
#[derive(Debug, Clone)]
pub enum OutputEvent {
    /// A generated text token: id plus its decoded piece.
    TextDelta { token_id: u32, piece: String },
    /// A chunk of PCM samples (future speech output). Not produced by any
    /// current model; see module docs.
    #[allow(dead_code)]
    AudioChunk { pcm: Vec<f32>, sample_rate: u32 },
    /// End of stream.
    Done,
}

/// What a model can do. Derived from the wrapper's actual implementation,
/// never marketing metadata: applications gate requests here before paying
/// for preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub input_text: bool,
    pub input_image: bool,
    pub input_audio: bool,
    pub input_video: bool,
    pub output_text: bool,
    /// Real speech synthesis exists behind this flag; no current model
    /// sets it (the boundary is [`OutputEvent::AudioChunk`]).
    pub output_audio: bool,
    /// More than one image per request is accepted.
    pub multi_image: bool,
    /// Incremental/streaming audio input is supported end-to-end
    /// (frontend + encoder scheduling). The Phase-4 frontend
    /// ([`crate::multimodal::stream::AudioStream`]) exists, but no wrapper
    /// consumes it incrementally yet, so every model reports false:
    /// capabilities reflect executable behavior only.
    pub streaming_audio_input: bool,
    /// Audio longer than one encoder context works via chunked windows.
    pub long_form_audio: bool,
    /// Chunked speech-output streaming. Not implemented anywhere yet.
    pub streaming_audio_output: bool,
    /// Simultaneous input + output with barge-in. Not implemented.
    pub full_duplex: bool,
}

impl ModelCapabilities {
    pub const TEXT_ONLY: Self = Self {
        input_text: true,
        input_image: false,
        input_audio: false,
        input_video: false,
        output_text: true,
        output_audio: false,
        multi_image: false,
        streaming_audio_input: false,
        long_form_audio: false,
        streaming_audio_output: false,
        full_duplex: false,
    };
}

impl crate::smolvlm::SmolVlm {
    pub fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            input_text: true,
            input_image: true,
            input_audio: false,
            input_video: false,
            output_text: true,
            output_audio: false,
            multi_image: true,
            streaming_audio_input: false,
            long_form_audio: false,
            streaming_audio_output: false,
            full_duplex: false,
        }
    }
}

impl crate::ultravox::Ultravox {
    pub fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            input_text: true,
            input_image: false,
            input_audio: true,
            input_video: false,
            output_text: true,
            output_audio: false,
            multi_image: false,
            // incremental streaming works end-to-end (Phase 4 session 2):
            // PCM -> AudioStream -> finalized-window scheduler -> VoiceSession
            streaming_audio_input: true,
            long_form_audio: true,
            streaming_audio_output: false,
            // Truthful only when the concurrent audio-I/O path is compiled
            // in AND has an executable demonstration (`ember voice
            // --duplex-smoke`, Phase 5 Track A). Without the feature the
            // runtime has no live I/O and the flag stays false.
            full_duplex: cfg!(feature = "audio"),
        }
    }
}

impl crate::tts::outetts::OuteTts {
    pub fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            input_text: true,
            input_image: false,
            input_audio: false,
            input_video: false,
            output_text: true, // codec-token stream is text-side decoding
            // validated PCM output (wavtokenizer ladder + synthesis battery)
            output_audio: true,
            multi_image: false,
            streaming_audio_input: false,
            long_form_audio: false,
            // PCM chunks are produced before generation completes
            // (synthesize_streaming; TTFA measured in the report)
            streaming_audio_output: true,
            full_duplex: false,
        }
    }
}

impl crate::smolvlm_video::SmolVlmVideo {
    pub fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            input_text: true,
            input_image: false,
            input_audio: false,
            input_video: true,
            output_text: true,
            output_audio: false,
            multi_image: false,
            streaming_audio_input: false,
            long_form_audio: false,
            streaming_audio_output: false,
            full_duplex: false,
        }
    }
}
