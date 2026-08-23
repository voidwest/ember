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
pub mod wavtokenizer;
