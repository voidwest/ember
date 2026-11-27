use crate::backend::{
    AttentionSpec, Backend, CachedAttentionSpec, CpuBackend, Module, TensorTriple,
};
use crate::quant::{QuantizedWeight, QuantizedWeightInterleaved, QuantizedWeightVnni};
use crate::tensor::CpuTensor;
use alloc::vec::Vec;
use anyhow::Context;

pub use crate::llama::{Llama, LlamaConfig};

/// Execute a complete CPU inference session on a Rayon worker.
///
/// Enter once around model preparation and the generation loop, not once per
/// token. Nested calls reuse the caller's current pool. External callers submit
/// one job at session entry, keeping per-projection joins off the global injector.
/// Raw forward calls outside a session remain supported but may allocate scheduler
/// queue blocks. This helper is an experimental Rust API.
pub fn with_cpu_session<R: Send>(run: impl FnOnce() -> R + Send) -> R {
    // The session thread also drives single-token decode regions on the
    // spin-waiting decode team (see `decode_pool::session`).
    if rayon::current_thread_index().is_some() {
        crate::decode_pool::session(run)
    } else {
        rayon::join(|| crate::decode_pool::session(run), || ()).0
    }
}

/// a model that can run inference with a kv cache.
/// gpt-2 and llama-family models implement this trait so the
/// `generate` / `demo_mode` / `interactive_mode` functions
/// in `main.rs` are generic over architecture.
pub trait ForwardModel<B: Backend> {
    fn create_cache(&self, backend: &B, max_seq_len: usize) -> crate::kv_cache::KVCache;
    /// maximum sequence length supported by this loaded model.
    fn max_seq_len(&self, backend: &B) -> usize;
    fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error>;
    fn forward_last_logits_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error>;

    /// Compute last-token logits, reusing `output` when the model supports it.
    /// The default retains the allocating implementation for other backends.
    fn forward_last_logits_with_cache_reusing(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        output: &mut B::Tensor,
    ) -> Result<(), B::Error> {
        *output = self.forward_last_logits_with_cache(backend, token_ids, cache, start_pos)?;
        Ok(())
    }

    /// Run the transformer body directly on precomputed input embeddings.
    ///
    /// `embeddings` is a `[seq_len, embed_dim]` tensor of *fully formed*
    /// model input embeddings: for RoPE models (Llama) these are the
    /// token-embedding rows (or encoder/projector output rows); for GPT-2
    /// they must already include the learned position-embedding rows. The
    /// transformer consumes the same internal path as token-based prefill
    /// and never needs to know where the embeddings came from.
    ///
    /// The default implementation panics because a generic model cannot
    /// construct its own backend error type; models that support
    /// precomputed-embedding prefill override it (Llama, Gpt2).
    fn forward_embeddings_with_cache(
        &self,
        _backend: &B,
        _embeddings: &B::Tensor,
        _cache: &mut crate::kv_cache::KVCache,
        _start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        panic!(
            "this model does not support precomputed-embedding prefill              (forward_embeddings_with_cache is not implemented)"
        )
    }

    /// Greedy (argmax) next-token inference: returns the top token id and
    /// its logit. Defaults to the full-logits path plus an in-place argmax;
    /// models with a fused fast path (e.g. LLaMA's Q8_0 workspace decode)
    /// may override this to avoid materializing the full vocabulary.
    fn greedy_next_token_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<(u32, f32), B::Error> {
        let logits = self.forward_last_logits_with_cache(backend, token_ids, cache, start_pos)?;
        let data = backend.data(&logits);
        let vocab_size = self.vocab_size(backend);
        assert_eq!(
            backend.shape(&logits),
            [1, vocab_size],
            "greedy inference requires last logits shaped [1, vocab]"
        );
        assert_eq!(
            data.len(),
            vocab_size,
            "greedy logits payload length mismatch"
        );
        let best = crate::sampler::argmax_token(data);
        let best = u32::try_from(best).expect("model vocabulary exceeds u32 token-ID space");
        Ok((best, data[best as usize]))
    }
    /// number of transformer layers (for display/debug).
    fn n_layers(&self) -> usize;
    /// hidden dimension (for display/debug).
    fn embed_dim(&self) -> usize;
    /// number of token IDs accepted by the embedding table and emitted by the
    /// language-model head.
    fn vocab_size(&self, backend: &B) -> usize;

    /// run forward pass and collect hidden states after each transformer block.
    ///
    /// returns `(per_layer_activations, final_logits)` where
    /// `per_layer_activations[layer]` is the flattened hidden state after
    /// block `layer`, with shape `[seq_len * embed_dim]`.
    ///
    /// this is the probing entry point - the same model code, but collecting
    /// intermediate representations instead of discarding them.
    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    fn forward_with_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
    ) -> Result<(alloc::vec::Vec<alloc::vec::Vec<f32>>, B::Tensor), B::Error>;

    /// run forward pass and collect pooled hidden states after each block.
    ///
    /// `token_index_groups` contains one or more token-index sets to pool from
    /// each layer. The returned vector has one flat `[n_layers * embed_dim]`
    /// buffer per group. Implementations can override this to avoid storing
    /// full sequence activations.
    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    fn forward_pooled_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[alloc::vec::Vec<usize>],
    ) -> Result<(alloc::vec::Vec<alloc::vec::Vec<f32>>, B::Tensor), B::Error> {
        let (layer_states, logits) = self.forward_with_activations(backend, token_ids)?;
        let embed_dim = self.embed_dim();
        let n_layers = self.n_layers();
        let mut pooled = token_index_groups
            .iter()
            .map(|_| alloc::vec![0.0f32; n_layers * embed_dim])
            .collect::<alloc::vec::Vec<_>>();
        for (li, state) in layer_states.iter().enumerate() {
            for (gi, token_indices) in token_index_groups.iter().enumerate() {
                let offset = li * embed_dim;
                pool_layer_activation(
                    state,
                    token_indices,
                    embed_dim,
                    &mut pooled[gi][offset..offset + embed_dim],
                );
            }
        }
        Ok((pooled, logits))
    }

    /// Collect pooled block states without evaluating the final normalization
    /// and language-model head. Implementations override this when hidden-only
    /// extraction can stop before logits computation.
    fn forward_pooled_hidden_states(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>, B::Error> {
        self.forward_pooled_activations(backend, token_ids, token_index_groups)
            .map(|(pooled, _)| pooled)
    }

    /// batched forward pass with block-diagonal attention for independent sequences.
    ///
    /// The returned logits contain one row per block, evaluated at that block's
    /// final token. The default implementation evaluates blocks independently;
    /// models with native block-masking support can override it for throughput.
    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    fn forward_pooled_with_blocks(
        &self,
        backend: &B,
        token_ids: &[u32],
        block_boundaries: &[usize],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        if block_boundaries.is_empty() {
            return self.forward_pooled_activations(backend, token_ids, token_index_groups);
        }
        validate_block_partition(token_ids.len(), block_boundaries, token_index_groups);

        let mut pooled = vec![None; token_index_groups.len()];
        let mut block_logits = Vec::new();
        let mut vocab_size = 0;

        for (block_index, &start) in block_boundaries.iter().enumerate() {
            let end = block_boundaries
                .get(block_index + 1)
                .copied()
                .unwrap_or(token_ids.len());
            assert!(
                start < end && end <= token_ids.len(),
                "invalid block boundary {start}..{end} for {} tokens",
                token_ids.len()
            );

            let mut group_indices = Vec::new();
            let mut local_groups = Vec::new();
            for (group_index, group) in token_index_groups.iter().enumerate() {
                if group.iter().all(|&index| index >= start && index < end) {
                    group_indices.push(group_index);
                    local_groups.push(group.iter().map(|&index| index - start).collect());
                }
            }

            let (local_pooled, logits) =
                self.forward_pooled_activations(backend, &token_ids[start..end], &local_groups)?;
            for (group_index, values) in group_indices.into_iter().zip(local_pooled) {
                pooled[group_index] = Some(values);
            }

            let shape = backend.shape(&logits);
            vocab_size = *shape
                .last()
                .expect("logits tensor must have a vocabulary axis");
            let data = backend.data(&logits);
            block_logits.extend_from_slice(&data[data.len() - vocab_size..]);
        }

        assert!(
            pooled.iter().all(Option::is_some),
            "every probe group must belong to exactly one attention block"
        );
        let pooled = pooled
            .into_iter()
            .map(|values| values.expect("probe group must belong to one block"))
            .collect();
        let logits = backend.load_from_cpu(block_logits, &[block_boundaries.len(), vocab_size])?;
        Ok((pooled, logits))
    }
}

/// Validate a concatenated independent-sequence layout used by block-diagonal
/// probing. This remains an assertion-level contract because `ForwardModel`
/// cannot construct an arbitrary backend's error type.
pub(crate) fn validate_block_partition(
    token_count: usize,
    block_boundaries: &[usize],
    token_index_groups: &[Vec<usize>],
) {
    assert!(token_count > 0, "block probing requires at least one token");
    assert!(
        !block_boundaries.is_empty() && block_boundaries[0] == 0,
        "block boundaries must start at zero"
    );
    assert!(
        block_boundaries.windows(2).all(|pair| pair[0] < pair[1]),
        "block boundaries must be strictly increasing"
    );
    assert!(
        block_boundaries.iter().all(|&start| start < token_count),
        "every block boundary must be inside the token sequence"
    );

    for (group_index, group) in token_index_groups.iter().enumerate() {
        assert!(
            !group.is_empty(),
            "probe token-index group {group_index} is empty"
        );
        assert!(
            group.iter().all(|&index| index < token_count),
            "probe token-index group {group_index} contains an out-of-range token"
        );
        let containing_blocks = block_boundaries
            .iter()
            .enumerate()
            .filter(|(block_index, start)| {
                let end = block_boundaries
                    .get(*block_index + 1)
                    .copied()
                    .unwrap_or(token_count);
                group.iter().all(|index| *index >= **start && *index < end)
            })
            .count();
        assert_eq!(
            containing_blocks, 1,
            "probe token-index group {group_index} must lie wholly within exactly one block"
        );
    }
}

pub fn pool_layer_activation(
    layer_state: &[f32],
    token_indices: &[usize],
    embed_dim: usize,
    out: &mut [f32],
) {
    assert!(embed_dim > 0, "pooling requires a non-zero embedding width");
    assert_eq!(out.len(), embed_dim, "pooled output width mismatch");
    assert!(
        !token_indices.is_empty(),
        "pooling requires at least one token index"
    );
    assert_eq!(
        layer_state.len() % embed_dim,
        0,
        "layer-state length is not divisible by embedding width"
    );
    let token_count = layer_state.len() / embed_dim;
    out.fill(0.0);
    for &token_index in token_indices {
        assert!(
            token_index < token_count,
            "pool token index {token_index} out of bounds for {token_count} tokens"
        );
        let row_start = token_index * embed_dim;
        for (j, value) in out.iter_mut().enumerate() {
            *value += layer_state[row_start + j];
        }
    }
    let scale = 1.0 / token_indices.len() as f32;
    for value in out {
        *value *= scale;
    }
}

mod gpt2;
mod linear;
#[cfg(test)]
use gpt2::take_gpt2_linear;
pub use gpt2::{Attention, Block, Gpt2, LayerNorm, Mlp};
pub use linear::{Linear, WeightKind, WeightKindView};

#[cfg(test)]
mod tests;
