//! Multimodal input foundation.
//!
//! The language-model core consumes [`crate::embedding::EmbeddingSequence`]s
//! and never knows whether the rows came from token IDs, a vision encoder,
//! an audio encoder, or any future modality encoder. This module owns
//! everything that turns *raw modality input* into those embedding rows:
//!
//! ```text
//! raw input -> modality processor -> modality encoder -> projector
//!           -> LLM-width embeddings
//! text      -> tokenizer -> token embedding lookup
//! text embeddings + media embeddings -> model-specific assembler
//!           -> EmbeddingSequence -> normal Ember LLM prefill
//! ```
//!
//! Stage contract (by convention, enforced at the wrapper boundary rather
//! than by a shared trait — two modalities did not force shared trait
//! ergonomics, so none was invented):
//!
//! - [`request`] represents ordered mixed-modality requests
//!   ([`ContentPart`]) with in-memory or file-backed media;
//! - one processor per modality (`image`, `audio`, video sampling) turns raw
//!   input into normalized tensors + metadata;
//! - one encoder (+ projector) per modality (`vision`, `audio_encoder`)
//!   produces LLM-width embedding rows;
//! - a model-specific assembler binds those rows to placeholders in text.
//!
//! Interfaces here are deliberately minimal: only what real architectures
//! require today.

pub mod assembler;
pub mod audio;
pub mod audio_encoder;
pub mod batch;
pub mod cache;
pub mod image;
pub mod output;
pub mod request;
pub mod session;
pub mod stream;
pub mod video;
pub mod vision;

pub use request::{
    ContentPart, ImageInput, MediaId, MediaKind, SegmentId, VideoFrames, VideoInput,
};
pub use session::{GenerationControl, Role, SessionStats, TurnRecord, TurnState, VoiceSession};
pub use video::{FrameSampling, SampledVideo};
