//! Downstream diagnostics for two same-target KV snapshots.
//!
//! Forced-token hook comparisons answer where hidden execution first differs;
//! an independent greedy rollout answers whether those differences change
//! observable continuation tokens. No mapper is fitted or executed here.

use crate::backend::{Backend, CpuBackend};
use crate::experiments::{
    ExecutionContext, ExecutionPhase, Experiment, ExperimentError, ExperimentRunner,
    ExperimentalForwardModel, LayerContext, ModelContext, ModelFamily, TensorAccess, TracingState,
};
use crate::kv_compare::{
    ensure_snapshots_aligned, import_diagnostic_cache, KvDiagnosticAlteration,
    KvDiagnosticPerturbationReceipt,
};
use crate::kv_snapshot::{KvCompatibilityTarget, KvSnapshot, KvSnapshotOrigin};
use crate::llama::Llama;
use crate::model::ForwardModel;
use crate::sampler::argmax_token;
use crate::tensor::CpuTensor;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub const KV_CONTINUATION_DIAGNOSTICS_SCHEMA: &str = "ember.kv-continuation-diagnostics.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvFloatVectorMetrics {
    pub element_count: usize,
    pub cosine_similarity: Option<f64>,
    pub reference_l2_norm: f64,
    pub candidate_l2_norm: f64,
    pub relative_l2_error: Option<f64>,
    pub mse: f64,
    pub max_abs_error: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvAttentionLayerDiagnostics {
    pub layer: usize,
    pub metrics: KvFloatVectorMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvForcedStepDiagnostics {
    pub evaluation_index: usize,
    pub input_token_id: u32,
    pub absolute_input_position: usize,
    pub predicted_continuation_index: usize,
    pub attention_by_layer: Vec<KvAttentionLayerDiagnostics>,
    pub final_logits: KvFloatVectorMetrics,
    pub reference_top1_token_id: u32,
    pub candidate_top1_token_id: u32,
    pub top1_agreement: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvContinuationDiagnostics {
    pub schema: String,
    pub reference_snapshot_hash: String,
    pub candidate_identity: String,
    pub candidate_snapshot_hash: Option<String>,
    pub diagnostic_perturbation: Option<KvDiagnosticPerturbationReceipt>,
    pub model_sha256: String,
    pub prefix_length: usize,
    pub max_tokens: usize,
    pub initial_token_id: u32,
    pub forced_token_policy: String,
    pub forced_input_token_ids: Vec<u32>,
    pub forced_steps: Vec<KvForcedStepDiagnostics>,
    pub attention_by_layer: Vec<KvAttentionLayerDiagnostics>,
    pub final_logit_cosine: Option<f64>,
    pub final_top1_agreement: Option<bool>,
    pub forced_top1_all_agree: bool,
    pub first_forced_top1_divergence: Option<usize>,
    pub reference_greedy_token_ids: Vec<u32>,
    pub candidate_greedy_token_ids: Vec<u32>,
    pub greedy_sequence_agreement_count: usize,
    pub greedy_sequence_agreement_fraction: f64,
    pub first_generated_token_divergence: Option<usize>,
    pub first_generated_token_divergence_absolute_position: Option<usize>,
    pub greedy_common_prefix_length: usize,
    pub no_greedy_divergence_through_horizon: bool,
    pub greedy_sequences_match: bool,
    pub hook_semantics: String,
    pub diagnostic_execution_caveat: String,
}

#[derive(Clone, Copy)]
pub enum KvContinuationCandidate<'a> {
    Snapshot(&'a KvSnapshot),
    Diagnostic {
        source: &'a KvSnapshot,
        alteration: &'a KvDiagnosticAlteration,
    },
}

impl KvContinuationCandidate<'_> {
    fn sequence_length(self) -> usize {
        match self {
            Self::Snapshot(snapshot)
            | Self::Diagnostic {
                source: snapshot, ..
            } => snapshot.manifest().sequence_length,
        }
    }

    fn identity(self) -> String {
        match self {
            Self::Snapshot(snapshot) => snapshot.manifest().snapshot_hash.clone(),
            Self::Diagnostic { alteration, .. } => alteration.receipt().diagnostic_id.clone(),
        }
    }

    fn snapshot_hash(self) -> Option<String> {
        match self {
            Self::Snapshot(snapshot) => Some(snapshot.manifest().snapshot_hash.clone()),
            Self::Diagnostic { .. } => None,
        }
    }

    fn receipt(self) -> Option<KvDiagnosticPerturbationReceipt> {
        match self {
            Self::Snapshot(_) => None,
            Self::Diagnostic { alteration, .. } => Some(alteration.receipt().clone()),
        }
    }

    fn import_cache(
        self,
        target: &KvCompatibilityTarget,
    ) -> anyhow::Result<crate::kv_cache::KVCache> {
        match self {
            Self::Snapshot(snapshot) => snapshot.import_cache(target),
            Self::Diagnostic { source, alteration } => {
                import_diagnostic_cache(source, alteration, target)
            }
        }
    }
}

#[derive(Clone)]
struct AttentionCapture {
    layers: Arc<Mutex<Vec<Option<Vec<f32>>>>>,
    expected_layers: usize,
    expected_hidden_size: usize,
    expected_start_position: usize,
    expected_input_token: u32,
}

impl Experiment for AttentionCapture {
    fn name(&self) -> &'static str {
        "kv-continuation-attention-capture"
    }

    fn after_attention(
        &mut self,
        context: &LayerContext<'_>,
        attention_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        if context.execution.phase != ExecutionPhase::Decode
            || context.execution.start_position != self.expected_start_position
            || context.execution.input_token_count != 1
            || context.execution.input_token_ids != Some(&[self.expected_input_token])
        {
            return Err(ExperimentError::new(format!(
                "attention capture execution context does not match decode token {} at position {}",
                self.expected_input_token, self.expected_start_position
            )));
        }
        if context.layer_index >= self.expected_layers {
            return Err(ExperimentError::new(format!(
                "attention capture layer {} exceeds expected {}",
                context.layer_index, self.expected_layers
            )));
        }
        if *attention_output.shape() != [1, self.expected_hidden_size] {
            return Err(ExperimentError::new(format!(
                "attention output at layer {} has shape {:?}, expected [1, {}]",
                context.layer_index,
                attention_output.shape(),
                self.expected_hidden_size
            )));
        }
        if let Some((index, value)) = attention_output
            .values()
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(ExperimentError::new(format!(
                "attention output at layer {} has non-finite value {} at index {}",
                context.layer_index, value, index
            )));
        }
        let mut layers = self
            .layers
            .lock()
            .map_err(|_| ExperimentError::new("attention capture lock poisoned"))?;
        if context.layer_index > 0 && layers[context.layer_index - 1].is_none() {
            return Err(ExperimentError::new(format!(
                "attention layer {} arrived before layer {}",
                context.layer_index,
                context.layer_index - 1
            )));
        }
        if layers[context.layer_index].is_some() {
            return Err(ExperimentError::new(format!(
                "attention layer {} captured twice in one evaluation",
                context.layer_index
            )));
        }
        let mut copied = Vec::new();
        copied
            .try_reserve_exact(attention_output.values().len())
            .map_err(|error| {
                ExperimentError::new(format!(
                    "cannot allocate attention capture at layer {}: {error}",
                    context.layer_index
                ))
            })?;
        copied.extend_from_slice(attention_output.values());
        layers[context.layer_index] = Some(copied);
        Ok(())
    }
}

/// Compare forced-token semantic-hook outputs and an independent greedy
/// rollout from two already verified, same-target snapshots.
///
/// `max_tokens` includes the initial resume/override token. If
/// `forced_input_tokens` is supplied it must contain exactly
/// `max_tokens - 1` evaluated tokens and begin with `initial_token_id`.
/// Otherwise the reference snapshot's top-1 prediction supplies each next
/// forced token.
#[allow(clippy::too_many_arguments)]
pub fn diagnose_continuation(
    model: &Llama<CpuBackend>,
    backend: &CpuBackend,
    reference: &KvSnapshot,
    candidate: KvContinuationCandidate<'_>,
    target: &KvCompatibilityTarget,
    family: ModelFamily,
    initial_token_id: u32,
    max_tokens: usize,
    forced_input_tokens: Option<&[u32]>,
) -> anyhow::Result<KvContinuationDiagnostics> {
    anyhow::ensure!(
        (2..=64).contains(&max_tokens),
        "continuation diagnostics require max_tokens within 2..=64"
    );
    anyhow::ensure!(
        (initial_token_id as usize) < model.config.vocab_size,
        "initial token {initial_token_id} is outside model vocabulary"
    );
    let evaluation_count = max_tokens - 1;
    if let Some(tokens) = forced_input_tokens {
        anyhow::ensure!(
            tokens.len() == evaluation_count,
            "forced token list has {} entries; expected {evaluation_count}",
            tokens.len()
        );
        if let Some(first) = tokens.first() {
            anyhow::ensure!(
                *first == initial_token_id,
                "forced token list begins with {first}, expected initial token {initial_token_id}"
            );
        }
        for (index, token) in tokens.iter().enumerate() {
            anyhow::ensure!(
                (*token as usize) < model.config.vocab_size,
                "forced token {token} at index {index} is outside model vocabulary"
            );
        }
    }
    reference.verify()?;
    anyhow::ensure!(
        reference.manifest().provenance.origin == KvSnapshotOrigin::Native,
        "continuation diagnostics require the reference snapshot to be native"
    );
    match candidate {
        KvContinuationCandidate::Snapshot(snapshot) => {
            snapshot.verify()?;
            ensure_snapshots_aligned(reference, snapshot)?;
        }
        KvContinuationCandidate::Diagnostic { source, alteration } => {
            source.verify()?;
            anyhow::ensure!(
                source.manifest().snapshot_hash == reference.manifest().snapshot_hash,
                "diagnostic alteration must be derived from the reference snapshot"
            );
            anyhow::ensure!(
                alteration.receipt().source_snapshot_hash == reference.manifest().snapshot_hash,
                "diagnostic alteration receipt names a different reference snapshot"
            );
        }
    }
    let prefix_length = reference.manifest().sequence_length;
    anyhow::ensure!(
        candidate.sequence_length() == prefix_length,
        "diagnostic candidates have different prefix lengths"
    );
    let required_capacity = prefix_length
        .checked_add(evaluation_count)
        .context("diagnostic context length overflow")?;
    anyhow::ensure!(
        target.max_seq >= required_capacity,
        "diagnostics require cache capacity {required_capacity}, target provides {}",
        target.max_seq
    );
    let model_context = ModelContext::new(
        family,
        None,
        &target.architecture,
        target.layer_count,
        model.config.embed_dim,
    )
    .with_provenance(
        Some(&target.model_sha256),
        target.tokenizer_sha256.as_deref(),
    );

    let mut reference_cache = reference.import_cache(target)?;
    let mut candidate_cache = candidate.import_cache(target)?;
    let mut forced_steps = Vec::new();
    forced_steps
        .try_reserve_exact(evaluation_count)
        .context("cannot allocate forced-step diagnostics")?;
    let mut forced_input_token_ids = Vec::new();
    forced_input_token_ids
        .try_reserve_exact(evaluation_count)
        .context("cannot allocate forced input token IDs")?;
    let aggregate_elements = evaluation_count
        .checked_mul(model.config.embed_dim)
        .context("aggregate attention element count overflow")?;
    let mut aggregate_reference = Vec::new();
    let mut aggregate_candidate = Vec::new();
    aggregate_reference
        .try_reserve_exact(target.layer_count)
        .context("cannot allocate reference attention layer aggregates")?;
    aggregate_candidate
        .try_reserve_exact(target.layer_count)
        .context("cannot allocate candidate attention layer aggregates")?;
    for _ in 0..target.layer_count {
        let mut reference_values = Vec::new();
        reference_values
            .try_reserve_exact(aggregate_elements)
            .context("cannot allocate reference attention aggregate")?;
        aggregate_reference.push(reference_values);
        let mut candidate_values = Vec::new();
        candidate_values
            .try_reserve_exact(aggregate_elements)
            .context("cannot allocate candidate attention aggregate")?;
        aggregate_candidate.push(candidate_values);
    }
    let mut current = initial_token_id;
    for evaluation_index in 0..evaluation_count {
        if let Some(tokens) = forced_input_tokens {
            current = tokens[evaluation_index];
        }
        forced_input_token_ids.push(current);
        let start_position = prefix_length
            .checked_add(evaluation_index)
            .context("forced diagnostic position overflow")?;
        let (reference_attention, reference_logits) = captured_step(
            model,
            backend,
            &mut reference_cache,
            current,
            start_position,
            model_context,
            target.layer_count,
            model.config.embed_dim,
            model.config.vocab_size,
        )?;
        let (candidate_attention, candidate_logits) = captured_step(
            model,
            backend,
            &mut candidate_cache,
            current,
            start_position,
            model_context,
            target.layer_count,
            model.config.embed_dim,
            model.config.vocab_size,
        )?;
        let mut attention_by_layer = Vec::new();
        attention_by_layer
            .try_reserve_exact(target.layer_count)
            .context("cannot allocate per-step attention metrics")?;
        for layer in 0..target.layer_count {
            let metrics = float_metrics(&reference_attention[layer], &candidate_attention[layer])?;
            aggregate_reference[layer].extend_from_slice(&reference_attention[layer]);
            aggregate_candidate[layer].extend_from_slice(&candidate_attention[layer]);
            attention_by_layer.push(KvAttentionLayerDiagnostics { layer, metrics });
        }
        let final_logits = float_metrics(&reference_logits, &candidate_logits)?;
        let reference_top1_token_id = argmax_token(&reference_logits) as u32;
        let candidate_top1_token_id = argmax_token(&candidate_logits) as u32;
        forced_steps.push(KvForcedStepDiagnostics {
            evaluation_index,
            input_token_id: current,
            absolute_input_position: start_position,
            predicted_continuation_index: evaluation_index + 1,
            attention_by_layer,
            final_logits,
            reference_top1_token_id,
            candidate_top1_token_id,
            top1_agreement: reference_top1_token_id == candidate_top1_token_id,
        });
        if forced_input_tokens.is_none() {
            current = reference_top1_token_id;
        }
    }

    let mut attention_by_layer = Vec::new();
    attention_by_layer
        .try_reserve_exact(target.layer_count)
        .context("cannot allocate aggregate attention metrics")?;
    for layer in 0..target.layer_count {
        attention_by_layer.push(KvAttentionLayerDiagnostics {
            layer,
            metrics: float_metrics(&aggregate_reference[layer], &aggregate_candidate[layer])?,
        });
    }
    let final_logit_cosine = forced_steps
        .last()
        .and_then(|step| step.final_logits.cosine_similarity);
    let final_top1_agreement = forced_steps.last().map(|step| step.top1_agreement);
    let first_forced_top1_divergence = forced_steps
        .iter()
        .find(|step| !step.top1_agreement)
        .map(|step| step.predicted_continuation_index);

    let reference_greedy_token_ids = greedy_rollout(
        model,
        backend,
        reference,
        target,
        initial_token_id,
        max_tokens,
    )?;
    let candidate_greedy_token_ids = greedy_rollout_candidate(
        model,
        backend,
        candidate,
        target,
        initial_token_id,
        max_tokens,
    )?;
    let greedy_sequence_agreement_count = reference_greedy_token_ids
        .iter()
        .zip(&candidate_greedy_token_ids)
        .filter(|(left, right)| left == right)
        .count();
    let first_generated_token_divergence = reference_greedy_token_ids
        .iter()
        .zip(&candidate_greedy_token_ids)
        .position(|(left, right)| left != right);
    let greedy_sequences_match = first_generated_token_divergence.is_none();

    Ok(KvContinuationDiagnostics {
        schema: KV_CONTINUATION_DIAGNOSTICS_SCHEMA.into(),
        reference_snapshot_hash: reference.manifest().snapshot_hash.clone(),
        candidate_identity: candidate.identity(),
        candidate_snapshot_hash: candidate.snapshot_hash(),
        diagnostic_perturbation: candidate.receipt(),
        model_sha256: target.model_sha256.clone(),
        prefix_length,
        max_tokens,
        initial_token_id,
        forced_token_policy: if forced_input_tokens.is_some() {
            "explicit-same-token-inputs"
        } else {
            "reference-greedy-teacher-forced-v1"
        }
        .into(),
        forced_input_token_ids,
        forced_steps,
        attention_by_layer,
        final_logit_cosine,
        final_top1_agreement,
        forced_top1_all_agree: first_forced_top1_divergence.is_none(),
        first_forced_top1_divergence,
        reference_greedy_token_ids,
        candidate_greedy_token_ids,
        greedy_sequence_agreement_count,
        greedy_sequence_agreement_fraction: greedy_sequence_agreement_count as f64
            / max_tokens as f64,
        first_generated_token_divergence,
        first_generated_token_divergence_absolute_position: first_generated_token_divergence
            .and_then(|index| prefix_length.checked_add(index)),
        greedy_common_prefix_length: first_generated_token_divergence.unwrap_or(max_tokens),
        no_greedy_divergence_through_horizon: greedy_sequences_match,
        greedy_sequences_match,
        hook_semantics: "attention-output is the zero-based per-layer semantic O-projection result after O bias and before the attention residual add; forced diagnostics use observer hooks at all current semantic sites, while greedy rollouts use ordinary decode"
            .into(),
        diagnostic_execution_caveat: "observer hooks can select a different internal dispatch than ordinary decode (notably planned Q8); paired forced comparisons remain same-mode/same-route, and behavioral greedy results are reported separately"
            .into(),
    })
}

#[allow(clippy::too_many_arguments)]
fn captured_step(
    model: &Llama<CpuBackend>,
    backend: &CpuBackend,
    cache: &mut crate::kv_cache::KVCache,
    input_token: u32,
    start_position: usize,
    model_context: ModelContext<'_>,
    layer_count: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<f32>)> {
    anyhow::ensure!(
        cache.cursor() == start_position,
        "diagnostic cache cursor {} does not equal start position {start_position}",
        cache.cursor()
    );
    let mut layer_slots = Vec::new();
    layer_slots
        .try_reserve_exact(layer_count)
        .context("cannot allocate attention capture slots")?;
    layer_slots.resize_with(layer_count, || None);
    let layers = Arc::new(Mutex::new(layer_slots));
    let capture = AttentionCapture {
        layers: Arc::clone(&layers),
        expected_layers: layer_count,
        expected_hidden_size: hidden_size,
        expected_start_position: start_position,
        expected_input_token: input_token,
    };
    let mut runner = ExperimentRunner::new(capture);
    runner.on_model_loaded(&model_context)?;
    let input_tokens = [input_token];
    let execution = ExecutionContext::new_with_token_ids(
        model_context,
        ExecutionPhase::Decode,
        start_position,
        &input_tokens,
        TracingState::Disabled,
    );
    let logits = ExperimentalForwardModel::forward_last_logits_with_experiment(
        model,
        backend,
        &input_tokens,
        cache,
        start_position,
        execution,
        &mut runner,
    )?;
    validate_logits(backend, &logits, vocab_size)?;
    drop(runner);
    let mut captured = layers
        .lock()
        .map_err(|_| anyhow::anyhow!("attention capture lock poisoned"))?;
    let attention = captured
        .iter_mut()
        .enumerate()
        .map(|(layer, values)| {
            values
                .take()
                .ok_or_else(|| anyhow::anyhow!("attention layer {layer} was not captured"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut logits_values = Vec::new();
    logits_values
        .try_reserve_exact(logits.data().len())
        .context("cannot allocate captured logits")?;
    logits_values.extend_from_slice(logits.data());
    Ok((attention, logits_values))
}

fn greedy_rollout(
    model: &Llama<CpuBackend>,
    backend: &CpuBackend,
    snapshot: &KvSnapshot,
    target: &KvCompatibilityTarget,
    initial_token: u32,
    max_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    let cache = snapshot.import_cache(target)?;
    greedy_rollout_with_cache(model, backend, cache, initial_token, max_tokens)
}

fn greedy_rollout_candidate(
    model: &Llama<CpuBackend>,
    backend: &CpuBackend,
    candidate: KvContinuationCandidate<'_>,
    target: &KvCompatibilityTarget,
    initial_token: u32,
    max_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    let cache = candidate.import_cache(target)?;
    greedy_rollout_with_cache(model, backend, cache, initial_token, max_tokens)
}

fn greedy_rollout_with_cache(
    model: &Llama<CpuBackend>,
    backend: &CpuBackend,
    mut cache: crate::kv_cache::KVCache,
    initial_token: u32,
    max_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(max_tokens)
        .context("cannot allocate greedy diagnostic tokens")?;
    let mut current = initial_token;
    tokens.push(current);
    for _ in 1..max_tokens {
        let start_position = cache.cursor();
        let logits = ForwardModel::forward_last_logits_with_cache(
            model,
            backend,
            &[current],
            &mut cache,
            start_position,
        )?;
        validate_logits(backend, &logits, model.config.vocab_size)?;
        current = argmax_token(logits.data()) as u32;
        tokens.push(current);
    }
    Ok(tokens)
}

fn validate_logits(
    backend: &CpuBackend,
    logits: &CpuTensor,
    expected_vocab_size: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        backend.shape(logits) == [1, expected_vocab_size],
        "diagnostic logits shape {:?}, expected [1, {expected_vocab_size}]",
        backend.shape(logits)
    );
    if let Some((index, value)) = logits
        .data()
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("diagnostic logits contain non-finite value {value} at index {index}");
    }
    Ok(())
}

fn float_metrics(reference: &[f32], candidate: &[f32]) -> anyhow::Result<KvFloatVectorMetrics> {
    anyhow::ensure!(
        reference.len() == candidate.len(),
        "diagnostic vector lengths differ ({} versus {})",
        reference.len(),
        candidate.len()
    );
    if reference.is_empty() {
        return Ok(KvFloatVectorMetrics {
            element_count: 0,
            cosine_similarity: None,
            reference_l2_norm: 0.0,
            candidate_l2_norm: 0.0,
            relative_l2_error: None,
            mse: 0.0,
            max_abs_error: 0.0,
        });
    }
    let mut dot = 0.0f64;
    let mut left_norm_squared = 0.0f64;
    let mut right_norm_squared = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut max_abs_error = 0.0f64;
    for (index, (left, right)) in reference.iter().zip(candidate).enumerate() {
        anyhow::ensure!(
            left.is_finite() && right.is_finite(),
            "diagnostic vector has non-finite value at index {index}"
        );
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        left_norm_squared += left * left;
        right_norm_squared += right * right;
        let difference = left - right;
        squared_error += difference * difference;
        max_abs_error = max_abs_error.max(difference.abs());
    }
    let reference_l2_norm = left_norm_squared.sqrt();
    let candidate_l2_norm = right_norm_squared.sqrt();
    let cosine_similarity = if reference_l2_norm == 0.0 || candidate_l2_norm == 0.0 {
        None
    } else if squared_error == 0.0 {
        Some(1.0)
    } else {
        Some((dot / (reference_l2_norm * candidate_l2_norm)).clamp(-1.0, 1.0))
    };
    Ok(KvFloatVectorMetrics {
        element_count: reference.len(),
        cosine_similarity,
        reference_l2_norm,
        candidate_l2_norm,
        relative_l2_error: (reference_l2_norm > 0.0)
            .then_some(squared_error.sqrt() / reference_l2_norm),
        mse: squared_error / reference.len() as f64,
        max_abs_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_metric_edges_are_defined() {
        let exact = float_metrics(&[0.0, 0.0], &[0.0, 0.0]).unwrap();
        assert_eq!(exact.cosine_similarity, None);
        let orthogonal = float_metrics(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert_eq!(orthogonal.cosine_similarity, Some(0.0));
        assert_eq!(orthogonal.mse, 1.0);
        assert_eq!(orthogonal.max_abs_error, 1.0);
        assert!(float_metrics(&[f32::NAN], &[0.0]).is_err());
    }
}
