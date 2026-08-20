//! Precomputed-embedding prefill: the boundary between *input assembly*
//! (token lookup, image/audio/video encoders, model-specific formatting)
//! and the generic transformer runtime.
//!
//! The transformer core consumes an [`EmbeddingSequence`] and never needs to
//! know whether the rows came from token IDs, a vision encoder, or any future
//! modality encoder. Token-based prefill and embedding-based prefill share
//! one internal path (`forward_embeddings_with_cache`); the token path is
//! simply `embed lookup -> EmbeddingSequence -> transformer`.
//!
//! Today the runtime's positional and attention contract is deliberately
//! narrow: positions are contiguous from `start_pos`, and attention is causal
//! and anchored at the KV-cache cursor. [`PositionInfo`] and
//! [`AttentionLayout`] make that contract explicit at the API surface so a
//! future modality (non-contiguous positions, encoder-style full attention,
//! block-diagonal layouts) can extend it without changing the transformer's
//! signature.

use crate::backend::Backend;

/// Absolute position of the first embedding in a sequence.
///
/// Positions are implicit and contiguous: embedding `i` sits at absolute
/// position `start_pos + i`. This matches how RoPE tables and the KV cache
/// are addressed today (`start_pos` flows into `apply_rotary_emb` and the
/// cache cursor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionInfo {
    pub start_pos: usize,
}

/// How a sequence attends to itself and to already-cached tokens.
///
/// The only layout the runtime implements today is causal self-attention
/// anchored at the KV-cache cursor (`causal == true`). The enum exists so
/// future layouts (bidirectional encoder blocks, block-diagonal probing)
/// can be expressed without widening the transformer entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionLayout {
    /// Causal attention over `cache.cursor() .. cache.cursor() + seq_len`.
    Causal,
}

/// Model-ready input embeddings plus the positional/attention metadata the
/// transformer needs to consume them.
///
/// `embeddings` is a `[seq_len, embed_dim]` tensor of *fully formed* model
/// input embeddings (for GPT-2 this includes the learned position embedding;
/// for RoPE models it is the token-embedding rows / encoder output rows).
pub struct EmbeddingSequence<B: Backend> {
    /// Row-major `[seq_len, embed_dim]` embedding rows.
    pub embeddings: B::Tensor,
    /// Absolute position of `embeddings[0]`.
    pub positions: PositionInfo,
    /// Attention layout for this sequence.
    pub attention: AttentionLayout,
}

impl<B: Backend> EmbeddingSequence<B> {
    /// Build a sequence at the given absolute start position with causal
    /// attention (the standard prefill contract).
    pub fn causal(embeddings: B::Tensor, start_pos: usize) -> Self {
        Self {
            embeddings,
            positions: PositionInfo { start_pos },
            attention: AttentionLayout::Causal,
        }
    }

    /// Number of embedding rows.
    pub fn seq_len(&self, backend: &B) -> usize {
        backend.shape(&self.embeddings)[0]
    }
}
