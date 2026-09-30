//! Shared-prefix reuse for paired experiment runs.
//!
//! A baseline and an intervention run over the same prompt compute the same
//! prefill up to the first site the intervention changes. The experiment
//! driver records, during the baseline prefill, the residual stream entering
//! block `k` and the KV cache after prefill; the intervention run then starts
//! its prefill at block `k` from that state instead of recomputing blocks
//! `0..k`.
//!
//! Exactness argument. In the generic hooked prefill (the only route a
//! multi-token prefill takes), block `j` reads only the residual stream
//! entering it and writes only cache layer `j`; attention for block `j`
//! reads cache layer `j` positions `[0, seq_len)`, which it has just written.
//! When no hook mutates anything before block `k`, blocks `0..k` compute the
//! same bits in both runs, so the recorded hidden state and cache layers
//! `< k` are exactly what the intervention run would have computed. Blocks
//! `k..` then run the same code on the same inputs through
//! [`crate::llama::Llama::forward_last_logits_from_layer_hooked`], the same
//! function the full forward uses.

use super::{ActiveHooks, ExecutionContext, ExecutionPhase, ExperimentRunner};
use crate::artifact::DispatchPath;
use crate::backend::{CpuBackend, CpuError};
use crate::kv_cache::KVCache;
use crate::llama::Llama;
use crate::tensor::CpuTensor;

/// Run a prefill from block `first_layer` with the experiment attached.
///
/// `hidden` is the `[seq, embed]` residual stream entering `first_layer`
/// (after every hook of block `first_layer - 1`), and `cache` holds this
/// sequence's K/V for layers `< first_layer` with its cursor at
/// `start_pos`. Only a multi-token prefill at position 0 is accepted: a
/// single-token evaluation takes the fused decode routes in a full forward,
/// so resuming it through the generic route could change its numerics.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn forward_last_logits_resumed_with_experiment(
    model: &Llama<CpuBackend>,
    backend: &CpuBackend,
    hidden: CpuTensor,
    first_layer: usize,
    cache: &mut KVCache,
    start_pos: usize,
    execution: ExecutionContext<'_>,
    runner: &mut ExperimentRunner,
) -> Result<CpuTensor, CpuError> {
    if execution.phase != ExecutionPhase::Prefill || start_pos != 0 {
        return Err(CpuError::Kernel(
            "prefix resume applies only to a prefill at position 0".into(),
        ));
    }
    let shape = hidden.shape();
    if shape.len() != 2 || shape[0] < 2 || shape[0] != execution.input_token_count {
        return Err(CpuError::ShapeMismatch(format!(
            "prefix resume needs a multi-token [seq, embed] hidden state matching the \
             {}-token prompt; got {shape:?}",
            execution.input_token_count
        )));
    }
    if shape[1] != model.config.embed_dim || first_layer > model.blocks.len() {
        return Err(CpuError::ShapeMismatch(format!(
            "prefix resume at layer {first_layer} with width {} does not fit a \
             {}-layer model of width {}",
            shape[1],
            model.blocks.len(),
            model.config.embed_dim
        )));
    }
    let mut hooks = ActiveHooks::new(runner, execution);
    // A full multi-token prefill always records the generic route.
    hooks.note_dispatch_path(DispatchPath::Generic);
    model.forward_last_logits_from_layer_hooked(
        backend,
        hidden,
        first_layer,
        cache,
        start_pos,
        &mut hooks,
    )
}
