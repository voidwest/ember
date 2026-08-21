//! Multimodal input foundation.
//!
//! The language-model core consumes [`crate::embedding::EmbeddingSequence`]s
//! and never knows whether the rows came from token IDs, a vision encoder,
//! or a future audio/video encoder. This module owns everything that turns
//! *raw modality input* (text, image files) into those embedding rows:
//!
//! ```text
//! raw input -> modality processor -> modality encoder -> projector
//!           -> LLM-width visual embeddings
//! text      -> tokenizer -> token embedding lookup
//! text embeddings + visual embeddings -> embedding assembler
//!           -> EmbeddingSequence -> normal Ember LLM prefill
//! ```
//!
//! [`image`] is the image preprocessing module (decoding, RGB, LANCZOS
//! resizing, normalization, tiling — nothing model-specific is hardcoded
//! into the tensor runtime). [`vision`] is the generic ViT/SigLIP-style
//! vision tower plus projector. [`assembler`] inserts image features into
//! text sequences behind an architecture-specific adapter.
//!
//! Interfaces here are deliberately minimal: only what image support
//! requires today. Audio/video/agents/graph execution are out of scope.

pub mod assembler;
pub mod audio;
pub mod audio_encoder;
pub mod image;
pub mod vision;

use crate::tensor::CpuTensor;
use anyhow::Result;
use std::path::PathBuf;

/// A raw input part for a multimodal request.
///
/// This is the *request* surface; the model-specific assembler turns parts
/// into an [`crate::embedding::EmbeddingSequence`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputPart {
    /// Free-form text. Image placeholders inside the text (e.g. `<image>`)
    /// are interpreted by the model-specific assembler.
    Text(String),
    /// An image file to decode and preprocess.
    Image(PathBuf),
}

/// Processed modality payload produced by a [`ModalityProcessor`] and
/// consumed by a [`ModalityEncoder`].
///
/// For images this is the normalized pixel tensor plus the geometry needed
/// by the model-specific assembler (tile grid, original dimensions).
#[derive(Debug, Clone)]
pub struct ProcessedModality {
    /// Normalized pixel tensor, shape `[n_images, channels, height, width]`.
    pub pixels: CpuTensor,
    /// Original image dimensions `(height, width)` before preprocessing.
    pub original_dims: (usize, usize),
    /// Dimensions of the resized image before tiling `(height, width)`.
    pub processed_dims: (usize, usize),
    /// Tile grid `(rows, cols)` of the split image (0,0 when no splitting).
    pub tile_grid: (usize, usize),
    /// Per-image pixel validity mask `[n_images, height, width]` (1 = valid).
    pub mask: CpuTensor,
}

/// Turns raw modality input into a [`ProcessedModality`].
///
/// Implementations are model-agnostic: an image processor config chooses
/// the recipe (resize target, tiling, normalization) without baking it into
/// the tensor runtime.
pub trait ModalityProcessor {
    type Input;
    type Output;

    fn process(&self, input: Self::Input) -> Result<Self::Output>;
}

/// Turns a [`ProcessedModality`] into model-width embedding rows.
///
/// Implementations are modality-specific but model-agnostic: a vision
/// encoder knows nothing about the language model that will consume its
/// output.
pub trait ModalityEncoder {
    /// The encoder output (for images: `[n_images, tokens, embed_dim]`).
    type Output;

    fn encode(&self, input: &ProcessedModality) -> Result<Self::Output>;
}
