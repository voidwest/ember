//! Logit lens over the residual-stream captures of a verified bundle.
//!
//! For every captured residual-stream row, the lens applies the model's own
//! final RMS norm and LM head ([`Llama::final_norm`], [`Llama::lm_head`]) to
//! obtain the next-token distribution that the stream would produce if the
//! network stopped at that depth. The last layer's lens is therefore the
//! model's final-logits computation itself, not a re-implementation of it.
//!
//! The lens reads a bundle and never writes into it (bundles are immutable);
//! the CLI (`ember experiment lens`) verifies the bundle and pins the model
//! and tokenizer to the recorded SHA-256 values before calling
//! [`compute_lens`].
//!
//! Depth convention: `residual-pre-attention` at layer `L` is the stream
//! after `L` blocks and `residual-post-mlp` at layer `L` is the stream after
//! `L + 1` blocks. A row whose depth equals the model's layer count is the
//! exact input of the final norm, so its lens distribution is the model's
//! output distribution at that position. KL divergences are reported as
//! `KL(final || layer)` in nats against that row.

use crate::backend::CpuBackend;
use crate::llama::Llama;
use crate::tensor::CpuTensor;
use crate::v05::hook::SemanticHookSite;
use crate::v05::verify::{CaptureIndexEntry, LoadedBundle};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Lens report schema identifier.
pub const LENS_SCHEMA_V1: &str = "ember.lens.v1";

/// Upper bound on `top_k`, to keep reports readable and bounded.
pub const MAX_TOP_K: usize = 100;

/// A final norm + unembedding that maps one residual-stream row to logits.
pub trait LensHead {
    fn n_layers(&self) -> usize;
    fn embed_dim(&self) -> usize;
    fn vocab_size(&self) -> usize;
    /// Project one `[embed_dim]` residual-stream row to `[vocab_size]`
    /// logits.
    fn project(&self, hidden_row: &[f32]) -> Result<Vec<f32>, String>;
}

/// The lens head of a loaded Llama-family model: its final RMS norm and LM
/// head (untied `output.weight` or tied embeddings), applied to one row at a
/// time exactly as the generic forward path applies them to the last
/// prompt row before sampling the first generated token.
pub struct ModelLens<'a> {
    model: &'a Llama<CpuBackend>,
}

impl<'a> ModelLens<'a> {
    pub fn new(model: &'a Llama<CpuBackend>) -> ModelLens<'a> {
        ModelLens { model }
    }
}

impl LensHead for ModelLens<'_> {
    fn n_layers(&self) -> usize {
        self.model.config.n_layers
    }

    fn embed_dim(&self) -> usize {
        self.model.config.embed_dim
    }

    fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }

    fn project(&self, hidden_row: &[f32]) -> Result<Vec<f32>, String> {
        let embed_dim = self.embed_dim();
        if hidden_row.len() != embed_dim {
            return Err(format!(
                "lens row has {} values, the model's embed_dim is {embed_dim}",
                hidden_row.len()
            ));
        }
        let backend = CpuBackend;
        let hidden = CpuTensor::from_data(vec![1, embed_dim], hidden_row.to_vec());
        let normed = self
            .model
            .final_norm(&backend, &hidden)
            .map_err(|error| format!("final norm failed: {error}"))?;
        let logits = self
            .model
            .lm_head(&backend, &normed)
            .map_err(|error| format!("lm head failed: {error}"))?;
        Ok(logits.into_data())
    }
}

/// Whether the lens is meaningful at a hook site: only residual-stream
/// sites carry a state the final norm and LM head are defined on.
pub fn site_eligibility(site: SemanticHookSite) -> Result<(), &'static str> {
    match site {
        SemanticHookSite::ResidualPreAttention | SemanticHookSite::ResidualPostMlp => Ok(()),
        SemanticHookSite::AttentionOutput | SemanticHookSite::MlpOutput => Err(
            "a projection output before its residual add, not a residual-stream state; \
             the final norm and LM head are not defined on it",
        ),
        SemanticHookSite::FinalNormOutput => {
            Err("already normalized; the model's LM head output is the logits site")
        }
        SemanticHookSite::Logits => Err("already logits; nothing to project"),
    }
}

/// Number of blocks applied to the stream at `(site, layer)`.
fn depth(site: SemanticHookSite, layer: usize) -> usize {
    match site {
        SemanticHookSite::ResidualPreAttention => layer,
        _ => layer + 1,
    }
}

/// Lens options.
#[derive(Debug, Clone, Copy)]
pub struct LensOptions {
    pub top_k: usize,
}

impl Default for LensOptions {
    fn default() -> Self {
        LensOptions { top_k: 5 }
    }
}

/// One ranked token of a lens distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LensToken {
    pub token_id: u32,
    pub text: String,
    pub probability: f64,
    pub logit: f32,
}

/// A token of the input or generated sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenRef {
    pub token_id: u32,
    pub text: String,
}

/// The token that actually followed a captured position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NextToken {
    pub token_id: u32,
    pub text: String,
    /// `prompt` (the next input token) or `generated` (the model's output).
    pub source: String,
}

/// Lens statistics for one captured row at one layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LensLayer {
    pub layer: usize,
    /// Blocks applied to the stream at this row (see module docs).
    pub depth: usize,
    pub top: Vec<LensToken>,
    /// 1-based rank of the actual next token (ties broken by lower id).
    pub next_token_rank: Option<usize>,
    pub next_token_probability: Option<f64>,
    pub entropy_nats: f64,
    /// `KL(final || layer)` in nats; `None` when the final-depth row for
    /// this position was not captured.
    pub kl_to_final_nats: Option<f64>,
}

/// All lens layers of one captured `(capture, input, position)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LensRow {
    pub capture_id: String,
    pub input_id: String,
    pub site: SemanticHookSite,
    /// Absolute model-input position (prompt tokens, then generated tokens).
    pub position: usize,
    /// `prompt` or `generated`.
    pub phase: String,
    /// Token at this position, when the position is in the sequence.
    pub token: Option<TokenRef>,
    pub next_token: Option<NextToken>,
    /// Hook route recorded by the capture (for example `unfused`).
    pub hook_route: String,
    /// Stored capture dtype (an `F16` capture is a rounded stream).
    pub dtype: String,
    /// For a final-depth row whose next token was generated: whether the
    /// lens top-1 equals it (expected under greedy decoding).
    pub final_top1_is_next: Option<bool>,
    pub layers: Vec<LensLayer>,
}

/// A capture the lens did not project, with the reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkippedCapture {
    pub capture_id: String,
    pub site: SemanticHookSite,
    pub reason: String,
}

/// Final-layer consistency: final-depth rows whose next token the model
/// generated, and how many of those the lens top-1 reproduces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinalLayerCheck {
    pub checked: usize,
    pub top1_matches: usize,
    /// Whether the bundle generated greedily (temperature 0), in which case
    /// every checked row is expected to match; `None` if unknown.
    pub greedy: Option<bool>,
}

/// The lens report. Written to stdout or a path outside the bundle, never
/// into the bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LensReport {
    pub schema: String,
    pub bundle: String,
    pub semantic_hash: String,
    pub payload_hash: String,
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub top_k: usize,
    pub kl_direction: String,
    pub final_layer_check: FinalLayerCheck,
    pub rows: Vec<LensRow>,
    pub skipped: Vec<SkippedCapture>,
    pub notes: Vec<String>,
}

/// Log-softmax summary of one logits vector (f64 for stability).
struct Distribution {
    log_probs: Vec<f64>,
    logits: Vec<f32>,
}

impl Distribution {
    fn new(logits: Vec<f32>) -> Result<Distribution, String> {
        if logits.is_empty() {
            return Err("lens produced empty logits".into());
        }
        if let Some(index) = logits.iter().position(|value| !value.is_finite()) {
            return Err(format!("lens produced a non-finite logit at token {index}"));
        }
        let max = logits
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &value| acc.max(value as f64));
        let sum: f64 = logits.iter().map(|&value| (value as f64 - max).exp()).sum();
        let log_sum = max + sum.ln();
        let log_probs = logits.iter().map(|&value| value as f64 - log_sum).collect();
        Ok(Distribution { log_probs, logits })
    }

    fn probability(&self, token: usize) -> f64 {
        self.log_probs[token].exp()
    }

    fn entropy(&self) -> f64 {
        -self
            .log_probs
            .iter()
            .map(|&lp| {
                let p = lp.exp();
                if p > 0.0 {
                    p * lp
                } else {
                    0.0
                }
            })
            .sum::<f64>()
    }

    /// `KL(self || other)`.
    fn kl_from(&self, other: &Distribution) -> f64 {
        self.log_probs
            .iter()
            .zip(&other.log_probs)
            .map(|(&lp, &lq)| {
                let p = lp.exp();
                if p > 0.0 {
                    p * (lp - lq)
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            .max(0.0)
    }

    /// 1-based rank; equal logits rank the lower token id first (the argmax
    /// tie rule used by greedy decoding).
    fn rank(&self, token: usize) -> usize {
        let target = self.logits[token];
        1 + self
            .logits
            .iter()
            .enumerate()
            .filter(|&(index, &value)| value > target || (value == target && index < token))
            .count()
    }

    fn top(&self, k: usize) -> Vec<usize> {
        let order =
            |a: &usize, b: &usize| self.logits[*b].total_cmp(&self.logits[*a]).then(a.cmp(b));
        let mut indices: Vec<usize> = (0..self.logits.len()).collect();
        let k = k.min(indices.len());
        if k == 0 {
            return Vec::new();
        }
        if k < indices.len() {
            indices.select_nth_unstable_by(k - 1, order);
            indices.truncate(k);
        }
        indices.sort_by(order);
        indices
    }
}

/// Per-input token sequence: prompt tokens then generated tokens.
struct InputSequence {
    prompt_len: usize,
    tokens: Vec<u32>,
}

fn input_sequences(bundle: &LoadedBundle) -> Result<BTreeMap<String, InputSequence>, String> {
    #[derive(Deserialize)]
    struct TokenizationLine {
        input_id: String,
        token_ids: Vec<u32>,
    }
    let text = std::str::from_utf8(bundle.required_file("tokenization.jsonl")?)
        .map_err(|error| format!("tokenization.jsonl is not UTF-8: {error}"))?;
    let mut prompts: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let parsed: TokenizationLine = serde_json::from_str(line)
            .map_err(|error| format!("tokenization.jsonl is malformed: {error}"))?;
        prompts.insert(parsed.input_id, parsed.token_ids);
    }
    let manifest = &bundle.semantic_manifest;
    let mut sequences = BTreeMap::new();
    for (index, input) in manifest.inputs.iter().enumerate() {
        let prompt = prompts
            .remove(&input.id)
            .ok_or_else(|| format!("tokenization.jsonl has no entry for input '{}'", input.id))?;
        let generated = manifest
            .generated
            .token_ids
            .get(index)
            .cloned()
            .unwrap_or_default();
        let prompt_len = prompt.len();
        let mut tokens = prompt;
        tokens.extend(generated);
        sequences.insert(input.id.clone(), InputSequence { prompt_len, tokens });
    }
    Ok(sequences)
}

/// One captured residual row awaiting projection.
struct PendingRow<'a> {
    entry: &'a CaptureIndexEntry,
    row: usize,
}

/// Compute the logit lens for every residual-stream capture row of a
/// verified bundle.
///
/// The caller is responsible for verifying `bundle` and for proving that
/// `head` is the model (and `decode` the tokenizer) the bundle recorded.
pub fn compute_lens(
    bundle: &LoadedBundle,
    head: &dyn LensHead,
    decode: &dyn Fn(u32) -> String,
    options: LensOptions,
) -> Result<LensReport, String> {
    if options.top_k == 0 || options.top_k > MAX_TOP_K {
        return Err(format!("top-k must be between 1 and {MAX_TOP_K}"));
    }
    let manifest = &bundle.semantic_manifest;
    let n_layers = head.n_layers();
    let embed_dim = head.embed_dim();
    let vocab_size = head.vocab_size();
    if manifest.model.layer_count != n_layers
        || manifest.model.embed_dim != embed_dim
        || manifest.model.vocab_size != vocab_size
    {
        return Err(format!(
            "model shape (layers {n_layers}, embed_dim {embed_dim}, vocab {vocab_size}) does not \
             match the bundle's recorded model (layers {}, embed_dim {}, vocab {})",
            manifest.model.layer_count, manifest.model.embed_dim, manifest.model.vocab_size
        ));
    }
    let sequences = input_sequences(bundle)?;

    // Group rows by (capture, input, position); remember which rows reach
    // the final depth so every position gets its KL reference.
    let mut skipped: Vec<SkippedCapture> = Vec::new();
    let mut groups: BTreeMap<(String, String, usize), Vec<PendingRow<'_>>> = BTreeMap::new();
    let mut final_rows: BTreeMap<(String, usize), PendingRow<'_>> = BTreeMap::new();
    for entry in &bundle.capture_index {
        if let Err(reason) = site_eligibility(entry.site) {
            if !skipped
                .iter()
                .any(|s| s.capture_id == entry.capture_id && s.site == entry.site)
            {
                skipped.push(SkippedCapture {
                    capture_id: entry.capture_id.clone(),
                    site: entry.site,
                    reason: reason.to_string(),
                });
            }
            continue;
        }
        if entry.tensor_name.is_empty() {
            if !skipped.iter().any(|s| s.capture_id == entry.capture_id) {
                skipped.push(SkippedCapture {
                    capture_id: entry.capture_id.clone(),
                    site: entry.site,
                    reason: "summary-only capture: no stored rows to project".to_string(),
                });
            }
            continue;
        }
        if entry.layer >= n_layers {
            return Err(format!(
                "capture '{}' names layer {} of a {n_layers}-layer model",
                entry.capture_id, entry.layer
            ));
        }
        if entry.shape.len() != 2
            || entry.shape[0] != entry.positions.len()
            || entry.shape[1] != embed_dim
        {
            return Err(format!(
                "capture '{}' layer {} has shape {:?}; expected [{}, {embed_dim}]",
                entry.capture_id,
                entry.layer,
                entry.shape,
                entry.positions.len()
            ));
        }
        for (row, &position) in entry.positions.iter().enumerate() {
            if depth(entry.site, entry.layer) == n_layers {
                final_rows
                    .entry((entry.input_id.clone(), position))
                    .or_insert(PendingRow { entry, row });
            }
            groups
                .entry((entry.capture_id.clone(), entry.input_id.clone(), position))
                .or_default()
                .push(PendingRow { entry, row });
        }
    }
    if groups.is_empty() {
        return Err(if skipped.is_empty() {
            "the bundle has no captures; the lens needs residual-stream captures \
             (residual-post-mlp or residual-pre-attention)"
                .to_string()
        } else {
            "the bundle has no residual-stream captures with stored rows; the lens needs \
             residual-post-mlp or residual-pre-attention captures"
                .to_string()
        });
    }

    // Decode each payload tensor once.
    let mut tensors: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut hidden_row = |entry: &CaptureIndexEntry, row: usize| -> Result<Vec<f32>, String> {
        if !tensors.contains_key(&entry.tensor_name) {
            let values = bundle.tensor_f32_by_name(&entry.tensor_name)?;
            tensors.insert(entry.tensor_name.clone(), values);
        }
        let values = &tensors[&entry.tensor_name];
        Ok(values[row * embed_dim..(row + 1) * embed_dim].to_vec())
    };

    let mut final_distributions: BTreeMap<(String, usize), Distribution> = BTreeMap::new();
    for (key, pending) in &final_rows {
        let hidden = hidden_row(pending.entry, pending.row)?;
        final_distributions.insert(key.clone(), Distribution::new(head.project(&hidden)?)?);
    }

    let mut rows = Vec::new();
    let mut check = FinalLayerCheck {
        checked: 0,
        top1_matches: 0,
        greedy: crate::v05::verify::bound_resolved_experiment(bundle)
            .ok()
            .map(|spec| spec.generation.temperature == 0.0),
    };
    for ((capture_id, input_id, position), mut pending) in groups {
        pending.sort_by_key(|p| p.entry.layer);
        let sequence = sequences.get(&input_id);
        let token_at = |index: usize| sequence.and_then(|s| s.tokens.get(index).copied());
        let prompt_len = sequence.map(|s| s.prompt_len).unwrap_or(0);
        let next_token = token_at(position + 1).map(|token_id| NextToken {
            token_id,
            text: decode(token_id),
            source: if position + 1 < prompt_len {
                "prompt".to_string()
            } else {
                "generated".to_string()
            },
        });
        let next_index = next_token
            .as_ref()
            .map(|next| next.token_id as usize)
            .filter(|&id| id < vocab_size);
        let key = (input_id.clone(), position);
        let reference = final_distributions.get(&key);
        let reference_source = final_rows.get(&key);
        let first = pending[0].entry;
        let mut layers = Vec::with_capacity(pending.len());
        let mut final_top1_is_next = None;
        for item in &pending {
            let layer_depth = depth(item.entry.site, item.entry.layer);
            // The recorded final row of this position is the KL reference
            // itself; every other row is projected here.
            let is_reference = reference_source.is_some_and(|source| {
                std::ptr::eq(source.entry, item.entry) && source.row == item.row
            });
            let owned;
            let dist: &Distribution = match reference {
                Some(reference) if is_reference => reference,
                _ => {
                    let hidden = hidden_row(item.entry, item.row)?;
                    owned = Distribution::new(head.project(&hidden)?)?;
                    &owned
                }
            };
            let top: Vec<LensToken> = dist
                .top(options.top_k)
                .into_iter()
                .map(|index| LensToken {
                    token_id: index as u32,
                    text: decode(index as u32),
                    probability: dist.probability(index),
                    logit: dist.logits[index],
                })
                .collect();
            if layer_depth == n_layers
                && let Some(next) = &next_token
                && next.source == "generated"
            {
                let matches = top.first().map(|t| t.token_id) == Some(next.token_id);
                final_top1_is_next = Some(matches);
                check.checked += 1;
                if matches {
                    check.top1_matches += 1;
                }
            }
            layers.push(LensLayer {
                layer: item.entry.layer,
                depth: layer_depth,
                top,
                next_token_rank: next_index.map(|id| dist.rank(id)),
                next_token_probability: next_index.map(|id| dist.probability(id)),
                entropy_nats: dist.entropy(),
                kl_to_final_nats: reference.map(|reference| reference.kl_from(dist)),
            });
        }
        rows.push(LensRow {
            capture_id,
            input_id,
            site: first.site,
            position,
            phase: if position < prompt_len {
                "prompt".to_string()
            } else {
                "generated".to_string()
            },
            token: token_at(position).map(|token_id| TokenRef {
                token_id,
                text: decode(token_id),
            }),
            next_token,
            hook_route: first.hook_route.clone(),
            dtype: first.dtype.clone(),
            final_top1_is_next,
            layers,
        });
    }

    let mut notes = vec![
        "rows at prompt positions reproduce the prefill route exactly; rows at generated \
         positions were produced by the decode route, whose fused kernels agree with the \
         lens projection within kernel tolerance"
            .to_string(),
    ];
    if !manifest.interventions.is_empty() {
        notes.push(
            "the bundle declares interventions: the lens projects the captured (as-run) \
             stream"
                .to_string(),
        );
    }
    if rows
        .iter()
        .any(|row| !row.dtype.eq_ignore_ascii_case("f32"))
    {
        notes.push("some captures are stored as F16; their lens is of a rounded stream".into());
    }
    if final_distributions.is_empty() {
        notes.push(format!(
            "no final-depth row (residual-post-mlp at layer {}) was captured, so KL to the \
             final distribution is unavailable",
            n_layers.saturating_sub(1)
        ));
    }

    Ok(LensReport {
        schema: LENS_SCHEMA_V1.to_string(),
        bundle: bundle.root.display().to_string(),
        semantic_hash: bundle.semantic_hash.clone(),
        payload_hash: bundle.payload_hash.clone(),
        model_sha256: manifest.model.sha256.clone(),
        tokenizer_sha256: manifest.tokenizer.sha256.clone(),
        n_layers,
        vocab_size,
        top_k: options.top_k,
        kl_direction: "KL(final || layer), nats".to_string(),
        final_layer_check: check,
        rows,
        skipped,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::{
        ExecutionContext, ExecutionPhase, Experiment, ExperimentError, ExperimentRunner,
        ExperimentalForwardModel, LayerContext, ModelContext, ModelFamily, TensorAccess,
        TracingState,
    };
    use crate::v05::testutil::{temp_root, tiny_llama_gguf, write_test_bundle};
    use crate::v05::verify::{load_verified_bundle, VerifyOptions};
    use std::sync::{Arc, Mutex};

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn load_tiny_llama(layers: usize, tied: bool) -> Llama<CpuBackend> {
        let root = temp_root("lens-model");
        std::fs::create_dir_all(&root).unwrap();
        // One file per test: never truncate a file another loader mapped.
        let path = root.join("tiny.gguf");
        std::fs::write(&path, tiny_llama_gguf(layers, tied)).unwrap();
        let loader = crate::loader::load_gguf(&path).expect("tiny GGUF loads");
        Llama::from_loader(loader).expect("tiny llama builds")
    }

    #[derive(Default)]
    struct Observed {
        /// Every layer's `after_layer` stream, per evaluation.
        hidden: Vec<Vec<Vec<f32>>>,
        logits: Vec<Vec<f32>>,
    }

    /// Records the residual stream the v0.5 capture site
    /// `residual-post-mlp` sees (the `after_layer` hook) and the logits the
    /// model actually produced.
    struct Recorder(Arc<Mutex<Observed>>);

    impl Experiment for Recorder {
        fn name(&self) -> &'static str {
            "lens-recorder"
        }

        fn after_layer(
            &mut self,
            ctx: &LayerContext<'_>,
            hidden: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError> {
            let mut observed = self.0.lock().unwrap();
            if ctx.layer_index == 0 {
                observed.hidden.push(Vec::new());
            }
            observed
                .hidden
                .last_mut()
                .unwrap()
                .push(hidden.values().to_vec());
            Ok(())
        }

        fn after_logits(
            &mut self,
            _ctx: &ExecutionContext<'_>,
            logits: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError> {
            self.0.lock().unwrap().logits.push(logits.values().to_vec());
            Ok(())
        }
    }

    /// The final layer's lens is the model's own final logits: on the
    /// experiment execution path (the one `ember experiment run` uses), the
    /// captured last-layer residual row projected through `ModelLens` is
    /// bit-identical to the logits the model produced, for prefill and for
    /// a decode step, with an untied head and with tied embeddings.
    #[test]
    fn final_layer_lens_equals_model_logits() {
        for tied in [false, true] {
            let n_layers = 3;
            let model = load_tiny_llama(n_layers, tied);
            assert_eq!(model.head_tied, tied);
            let embed_dim = model.config.embed_dim;
            let backend = CpuBackend;
            let observed = Arc::new(Mutex::new(Observed::default()));
            let mut runner = ExperimentRunner::new(Recorder(Arc::clone(&observed)));
            let mut cache = model.create_cache(&backend, model.config.max_seq_len);
            let context = ModelContext::new(ModelFamily::Llama, None, "llama", n_layers, embed_dim);
            let lens = ModelLens::new(&model);

            let prompt = [3u32, 17, 42, 5, 9];
            let prefill = model
                .forward_last_logits_with_experiment(
                    &backend,
                    &prompt,
                    &mut cache,
                    0,
                    ExecutionContext::new_with_token_ids(
                        context,
                        ExecutionPhase::Prefill,
                        0,
                        &prompt,
                        TracingState::Disabled,
                    ),
                    &mut runner,
                )
                .unwrap();
            let next = prefill
                .data()
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                    if v > best.1 {
                        (i, v)
                    } else {
                        best
                    }
                })
                .0 as u32;
            let step = [next];
            let decode = model
                .forward_last_logits_with_experiment(
                    &backend,
                    &step,
                    &mut cache,
                    prompt.len(),
                    ExecutionContext::new_with_token_ids(
                        context,
                        ExecutionPhase::Decode,
                        prompt.len(),
                        &step,
                        TracingState::Disabled,
                    ),
                    &mut runner,
                )
                .unwrap();

            let observed = observed.lock().unwrap();
            assert_eq!(observed.hidden.len(), 2);
            for (evaluation, produced) in [prefill.data(), decode.data()].into_iter().enumerate() {
                let layers = &observed.hidden[evaluation];
                assert_eq!(layers.len(), n_layers);
                let last_layer = &layers[n_layers - 1];
                let row = &last_layer[last_layer.len() - embed_dim..];
                let projected = lens.project(row).unwrap();
                assert_eq!(
                    bits(&projected),
                    bits(produced),
                    "tied={tied} evaluation={evaluation}: final-layer lens != model logits"
                );
                assert_eq!(bits(&projected), bits(&observed.logits[evaluation]));
                // Earlier layers are genuinely different distributions.
                let early = &layers[0][layers[0].len() - embed_dim..];
                assert_ne!(bits(&lens.project(early).unwrap()), bits(produced));
            }
        }
    }

    /// A deterministic stand-in head for the fixture bundle (1 layer,
    /// embed 4, vocab 16).
    struct FixtureHead;

    impl LensHead for FixtureHead {
        fn n_layers(&self) -> usize {
            1
        }
        fn embed_dim(&self) -> usize {
            4
        }
        fn vocab_size(&self) -> usize {
            16
        }
        fn project(&self, hidden_row: &[f32]) -> Result<Vec<f32>, String> {
            Ok((0..16)
                .map(|i| {
                    hidden_row
                        .iter()
                        .enumerate()
                        .map(|(j, &x)| x * (((i * 3 + j * 5) % 7) as f32 - 3.0) * 0.25)
                        .sum()
                })
                .collect())
        }
    }

    #[test]
    fn lens_report_over_a_verified_bundle() {
        let root = temp_root("lens-bundle");
        write_test_bundle(&root, &[1.0, 2.0, 3.0, 4.0], &[0]);
        let bundle = load_verified_bundle(&root, &VerifyOptions::default()).unwrap();
        let before: Vec<_> = std::fs::read_dir(&root).unwrap().collect();
        let decode = |token: u32| format!("t{token}");
        let report = compute_lens(&bundle, &FixtureHead, &decode, LensOptions { top_k: 3 })
            .expect("lens computes");
        assert_eq!(report.schema, LENS_SCHEMA_V1);
        assert_eq!(report.rows.len(), 1);
        let row = &report.rows[0];
        assert_eq!(row.position, 0);
        assert_eq!(row.phase, "prompt");
        let next = row.next_token.as_ref().unwrap();
        assert_eq!((next.token_id, next.source.as_str()), (1, "generated"));
        assert_eq!(row.layers.len(), 1);
        let layer = &row.layers[0];
        assert_eq!(layer.depth, 1);
        assert_eq!(layer.top.len(), 3);
        assert!(layer
            .top
            .windows(2)
            .all(|pair| pair[0].logit >= pair[1].logit));
        // The final-depth row is its own reference.
        assert_eq!(layer.kl_to_final_nats, Some(0.0));
        let logits = FixtureHead.project(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected = Distribution::new(logits).unwrap();
        assert_eq!(layer.next_token_rank, Some(expected.rank(1)));
        assert!((layer.next_token_probability.unwrap() - expected.probability(1)).abs() < 1e-12);
        assert!((layer.entropy_nats - expected.entropy()).abs() < 1e-12);
        assert_eq!(
            row.final_top1_is_next,
            Some(layer.top[0].token_id == 1),
            "the final-layer check compares top-1 with the generated token"
        );
        assert_eq!(report.final_layer_check.checked, 1);
        // Nothing was written into the bundle.
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), before.len());
        // A head that does not match the bundle's recorded model is refused.
        struct WrongHead;
        impl LensHead for WrongHead {
            fn n_layers(&self) -> usize {
                2
            }
            fn embed_dim(&self) -> usize {
                4
            }
            fn vocab_size(&self) -> usize {
                16
            }
            fn project(&self, _: &[f32]) -> Result<Vec<f32>, String> {
                unreachable!()
            }
        }
        assert!(compute_lens(&bundle, &WrongHead, &decode, LensOptions::default()).is_err());
        assert!(compute_lens(&bundle, &FixtureHead, &decode, LensOptions { top_k: 0 }).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn only_residual_stream_sites_are_eligible() {
        assert!(site_eligibility(SemanticHookSite::ResidualPostMlp).is_ok());
        assert!(site_eligibility(SemanticHookSite::ResidualPreAttention).is_ok());
        for site in [
            SemanticHookSite::AttentionOutput,
            SemanticHookSite::MlpOutput,
            SemanticHookSite::FinalNormOutput,
            SemanticHookSite::Logits,
        ] {
            assert!(site_eligibility(site).is_err(), "{site}");
        }
        assert_eq!(depth(SemanticHookSite::ResidualPreAttention, 3), 3);
        assert_eq!(depth(SemanticHookSite::ResidualPostMlp, 3), 4);
    }

    #[test]
    fn distribution_statistics() {
        let dist = Distribution::new(vec![2.0, 5.0, 5.0, -1.0]).unwrap();
        // Ties rank the lower id first, as greedy argmax does.
        assert_eq!(dist.rank(1), 1);
        assert_eq!(dist.rank(2), 2);
        assert_eq!(dist.rank(0), 3);
        assert_eq!(dist.top(2), vec![1, 2]);
        let total: f64 = (0..4).map(|i| dist.probability(i)).sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert_eq!(dist.kl_from(&dist), 0.0);
        let uniform = Distribution::new(vec![0.0; 4]).unwrap();
        assert!((uniform.entropy() - 4f64.ln()).abs() < 1e-12);
        assert!(dist.kl_from(&uniform) > 0.0);
        assert!(Distribution::new(vec![f32::NAN]).is_err());
    }
}
