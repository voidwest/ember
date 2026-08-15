//! v0.5 experiment runner: captures and interventions driven through the
//! existing six-hook execution machinery (contract sections 2, 3, 8).
//!
//! The runner implements the v0.4 `Experiment` trait so it rides the same
//! reference and planned execution paths without duplicating inference.
//! At each hook fire: (1) the pre-intervention snapshot is taken when an
//! intervention targets this site, (2) captures copy their selected rows,
//! (3) interventions apply in declaration order.
//!
//! Row access rule: prefill tensors are `[seq, embed]` indexed by absolute
//! position; decode tensors are `[1, embed]` whose single row is the
//! evaluated absolute position `start_position`.

use crate::artifact::ActivationStage;
use crate::experiments::{
    ExecutionContext, ExecutionPhase, Experiment, ExperimentError, GenerationContext, LayerContext,
    ModelContext, TensorAccess,
};
use crate::v05::capture::{CaptureStorage, InputSelector, LayerSelector};
use crate::v05::hook::SemanticHookSite;
use crate::v05::intervention::{InterventionOperation, InterventionSource, InterventionSpec};
use crate::v05::spec::{ExperimentSpecV1, InputSpec};
use crate::v05::token_select::{
    resolve_generated_step, resolve_static_selector, AmbiguityStatus, CoverageKind,
    RoundTripStatus, TextNormalization, TokenSelectionRecord, TokenSelector, TokenizationInfo,
};
use std::collections::HashMap;

/// Static model facts the runner needs.
#[derive(Debug, Clone, Copy)]
pub struct ModelFacts {
    pub n_layers: usize,
    pub embed_dim: usize,
    pub vocab_size: usize,
}

/// One capture target resolved for the current input.
#[derive(Debug, Clone)]
pub struct CaptureTarget {
    pub capture_id: String,
    pub site: SemanticHookSite,
    pub layers: Vec<usize>,
    pub storage: CaptureStorage,
    pub dtype: crate::v05::capture::CaptureDType,
    pub selector: TokenSelector,
    /// Resolved static selection (`None` for generated-step selectors).
    pub static_record: Option<TokenSelectionRecord>,
    /// 1-based decode steps to capture.
    pub generated_steps: Vec<usize>,
}

/// One intervention target resolved for the current input.
#[derive(Debug, Clone)]
pub struct InterventionTarget {
    pub intervention_id: String,
    pub site: SemanticHookSite,
    pub layers: Vec<usize>,
    pub operation: InterventionOperation,
    pub source: Option<InterventionSource>,
    pub selector: TokenSelector,
    pub static_record: Option<TokenSelectionRecord>,
    /// 1-based decode steps to intervene at.
    pub generated_steps: Vec<usize>,
}

/// A captured tensor payload (owned; never borrows scratch).
#[derive(Debug, Clone)]
pub struct CapturedTensor {
    pub capture_id: String,
    pub input_id: String,
    pub site: SemanticHookSite,
    pub layer: usize,
    /// Absolute token positions of the stored rows (ascending).
    pub positions: Vec<usize>,
    /// Row-major values, positions-major.
    pub rows: Vec<f32>,
    /// Embedding width (columns).
    pub columns: usize,
    pub full_tensor: bool,
    pub bytes: usize,
    /// Requested output dtype.
    pub dtype: crate::v05::capture::CaptureDType,
}

/// Deterministic summary statistics for a `summary-only` capture
/// (never usable as an intervention source).
#[derive(Debug, Clone)]
pub struct CaptureSummary {
    pub capture_id: String,
    pub input_id: String,
    pub site: SemanticHookSite,
    pub layer: usize,
    pub positions: Vec<usize>,
    /// Shape of the summarized rows tensor: [positions, columns].
    pub shape: [usize; 2],
    pub finite_count: usize,
    pub minimum: f32,
    pub maximum: f32,
    pub mean: f64,
    pub l2_norm: f64,
}

/// An intervention application event (contract section 5 provenance).
#[derive(Debug, Clone)]
pub struct InterventionEvent {
    pub intervention_id: String,
    pub input_id: String,
    pub site: SemanticHookSite,
    pub layer: usize,
    pub positions: Vec<usize>,
    pub operation: InterventionOperation,
    pub source_kind: Option<String>,
    pub snapshot_checksum: Option<String>,
    pub applied: bool,
}

/// Per-input run result collected for the bundle.
#[derive(Debug, Clone)]
pub struct InputResult {
    pub input: InputSpec,
    pub tokenization: TokenizationInfo,
    pub selection_records: Vec<TokenSelectionRecord>,
    pub captures: Vec<CapturedTensor>,
    pub summaries: Vec<CaptureSummary>,
    pub events: Vec<InterventionEvent>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
    pub final_top1: Option<(u32, f32)>,
}

/// Snapshot key: (site, layer, absolute position).
type SnapshotKey = (SemanticHookSite, usize, usize);

/// The v0.5 experiment driving one input through generation.
pub struct V05Experiment {
    pub spec: ExperimentSpecV1,
    pub input_index: usize,
    pub facts: ModelFacts,
    pub model_sha256: Option<String>,
    pub tokenizer_sha256: Option<String>,
    captures: Vec<CaptureTarget>,
    interventions: Vec<InterventionTarget>,
    bundle_sources: Vec<BundleSource>,
    // runtime state
    prompt_len: usize,
    tokenizations: HashMap<TextNormalization, TokenizationInfo>,
    result: Option<InputResult>,
    snapshots: HashMap<SnapshotKey, Vec<f32>>,
    snapshot_checksums: HashMap<SnapshotKey, String>,
    final_top1: Option<(u32, f32)>,
}

impl V05Experiment {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spec: ExperimentSpecV1,
        input_index: usize,
        facts: ModelFacts,
        model_sha256: Option<String>,
        tokenizer_sha256: Option<String>,
    ) -> V05Experiment {
        V05Experiment {
            spec,
            input_index,
            facts,
            model_sha256,
            tokenizer_sha256,
            captures: Vec::new(),
            interventions: Vec::new(),
            bundle_sources: Vec::new(),
            prompt_len: 0,
            tokenizations: HashMap::new(),
            result: None,
            snapshots: HashMap::new(),
            snapshot_checksums: HashMap::new(),
            final_top1: None,
        }
    }

    /// Inject a cross-bundle source resolved by the driver before
    /// execution.
    pub fn inject_bundle_source(&mut self, source: BundleSource) {
        self.bundle_sources.push(source);
    }

    /// Inject the tokenization computed by the CLI driver.
    pub fn inject_tokenization(&mut self, info: TokenizationInfo) {
        self.tokenizations.insert(TextNormalization::None, info);
    }

    /// Set the generated text after generation (from the driver).
    pub fn set_generated_text(&mut self, text: String) {
        if let Some(result) = self.result.as_mut() {
            result.generated_text = text;
        }
    }

    /// Resolve the current input's selectors; called from `before_prefill`.
    fn prepare_input(&mut self, ctx: &ExecutionContext<'_>) -> Result<(), ExperimentError> {
        let input = self
            .spec
            .inputs
            .get(self.input_index)
            .cloned()
            .ok_or_else(|| ExperimentError::new("input index out of range"))?;
        let all_input_ids = self
            .spec
            .inputs
            .iter()
            .map(|i| i.id.clone())
            .collect::<Vec<_>>();

        let mut captures: Vec<CaptureTarget> = Vec::new();
        for capture in &self.spec.captures {
            let Some((layers, static_record, generated_steps)) = resolve_target_addressing(
                &capture.inputs,
                &capture.layers,
                &capture.tokens,
                &all_input_ids,
                &input.id,
                self.facts.n_layers,
                &self.tokenizations,
            )?
            else {
                continue;
            };
            captures.push(CaptureTarget {
                capture_id: capture.id.clone(),
                site: capture.site,
                layers,
                storage: capture.storage,
                dtype: capture.dtype,
                selector: capture.tokens.clone(),
                static_record,
                generated_steps,
            });
        }

        let mut interventions: Vec<InterventionTarget> = Vec::new();
        for intervention in &self.spec.interventions {
            let Some((layers, static_record, generated_steps)) = resolve_target_addressing(
                &intervention.inputs,
                &intervention.layers,
                &intervention.tokens,
                &all_input_ids,
                &input.id,
                self.facts.n_layers,
                &self.tokenizations,
            )?
            else {
                continue;
            };
            interventions.push(InterventionTarget {
                intervention_id: intervention.id.clone(),
                site: intervention.site,
                layers,
                operation: intervention.operation,
                source: intervention.source.clone(),
                selector: intervention.tokens.clone(),
                static_record,
                generated_steps,
            });
        }

        self.captures = captures;
        self.interventions = interventions;
        self.prompt_len = ctx.input_token_count;
        Ok(())
    }

    /// Whether any capture of this input targets (site, layer).
    fn site_has_captures(&self, site: SemanticHookSite, layer: usize) -> bool {
        self.captures
            .iter()
            .any(|c| c.site == site && c.layers.contains(&layer))
    }

    /// Whether any intervention of this input targets (site, layer).
    fn site_has_interventions(&self, site: SemanticHookSite, layer: usize) -> bool {
        self.interventions
            .iter()
            .any(|i| i.site == site && i.layers.contains(&layer))
    }

    /// Whether an intervention applies at (site, layer, absolute position).
    fn intervenes_at(&self, site: SemanticHookSite, layer: usize, absolute: usize) -> bool {
        self.interventions.iter().any(|target| {
            if target.site != site || !target.layers.contains(&layer) {
                return false;
            }
            if let Some(record) = &target.static_record {
                if record.selected_indices.contains(&absolute) {
                    return true;
                }
            }
            if !target.generated_steps.is_empty() {
                let step = absolute.saturating_sub(self.prompt_len) + 1;
                if target.generated_steps.contains(&step) {
                    return true;
                }
            }
            false
        })
    }

    /// The 1-based decode step of a fire at `start_position`.
    fn decode_step(&self, start_position: usize) -> usize {
        start_position.saturating_sub(self.prompt_len) + 1
    }

    /// Fire the site: snapshot, capture, intervene.
    fn fire_site(
        &mut self,
        ctx: &ExecutionContext<'_>,
        layer: usize,
        site: SemanticHookSite,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        let [rows, columns] = *tensor.shape();
        let is_decode = ctx.phase == ExecutionPhase::Decode;
        let start_position = ctx.start_position;
        let input_id = self
            .spec
            .inputs
            .get(self.input_index)
            .map(|i| i.id.clone())
            .ok_or_else(|| ExperimentError::new("input index out of range"))?;

        // 1. snapshots for interventions at this site.
        if self.site_has_interventions(site, layer) {
            for local_row in 0..rows {
                let absolute = if is_decode { start_position } else { local_row };
                if !self.intervenes_at(site, layer, absolute) {
                    continue;
                }
                let key = (site, layer, absolute);
                if !self.snapshots.contains_key(&key) {
                    let row = &tensor.values()[local_row * columns..(local_row + 1) * columns];
                    let checksum = checksum_f32(row);
                    self.snapshot_checksums.insert(key, checksum);
                    self.snapshots.insert(key, row.to_vec());
                }
            }
        }

        // 2. captures.
        if self.site_has_captures(site, layer) {
            let mut pending: Vec<(usize, CapturedTensor)> = Vec::new();
            let mut summaries: Vec<CaptureSummary> = Vec::new();
            for target in self.captures.iter() {
                if target.site != site || !target.layers.contains(&layer) {
                    continue;
                }
                if target.storage == CaptureStorage::SummaryOnly {
                    // Deterministic statistics over the selected rows.
                    let wanted: Vec<usize> = if let Some(record) = &target.static_record {
                        record
                            .selected_indices
                            .iter()
                            .copied()
                            .filter(|&position| {
                                if is_decode {
                                    position == start_position
                                } else {
                                    position < rows
                                }
                            })
                            .collect()
                    } else {
                        let step = self.decode_step(start_position);
                        if target.generated_steps.contains(&step) {
                            vec![start_position]
                        } else {
                            Vec::new()
                        }
                    };
                    if wanted.is_empty() {
                        continue;
                    }
                    let mut values: Vec<f32> = Vec::new();
                    for &position in &wanted {
                        let local = if is_decode { 0 } else { position };
                        values.extend_from_slice(
                            &tensor.values()[local * columns..(local + 1) * columns],
                        );
                    }
                    summaries.push(summary_stats(
                        &target.capture_id,
                        &input_id,
                        site,
                        layer,
                        wanted,
                        columns,
                        &values,
                    ));
                    continue;
                }
                if let Some(record) = &target.static_record {
                    let wanted: Vec<usize> = record
                        .selected_indices
                        .iter()
                        .copied()
                        .filter(|&position| {
                            if is_decode {
                                position == start_position
                            } else {
                                position < rows
                            }
                        })
                        .collect();
                    if wanted.is_empty() {
                        continue;
                    }
                    let full = target.storage == CaptureStorage::FullTensor;
                    let mut rows_out = Vec::new();
                    let positions: Vec<usize> = if full && !is_decode {
                        rows_out.extend_from_slice(tensor.values());
                        (0..rows).collect()
                    } else {
                        for &position in &wanted {
                            let local = if is_decode { 0 } else { position };
                            let row = &tensor.values()[local * columns..(local + 1) * columns];
                            rows_out.extend_from_slice(row);
                        }
                        wanted
                    };
                    let bytes = rows_out.len() * 4;
                    pending.push((
                        site_order(site) * 1_000_000 + layer,
                        CapturedTensor {
                            capture_id: target.capture_id.clone(),
                            input_id: input_id.clone(),
                            site,
                            layer,
                            positions,
                            columns,
                            rows: rows_out,
                            full_tensor: full,
                            bytes,
                            dtype: target.dtype,
                        },
                    ));
                } else {
                    // generated-step capture: buffer only the requested steps.
                    let step = self.decode_step(start_position);
                    if !target.generated_steps.contains(&step) {
                        continue;
                    }
                    let row = tensor.values()[..columns].to_vec();
                    pending.push((
                        site_order(site) * 1_000_000 + layer,
                        CapturedTensor {
                            capture_id: target.capture_id.clone(),
                            input_id: input_id.clone(),
                            site,
                            layer,
                            positions: vec![start_position],
                            columns,
                            rows: row,
                            full_tensor: false,
                            bytes: columns * 4,
                            dtype: target.dtype,
                        },
                    ));
                }
            }
            // deterministic order: site/layer key, then declaration index
            pending.sort_by_key(|(key, _)| *key);
            let result = self
                .result
                .as_mut()
                .ok_or_else(|| ExperimentError::new("result not initialized before prefill"))?;
            for (_, captured) in pending {
                result.captures.push(captured);
            }
            result.summaries.extend(summaries);
        }

        // 3. interventions in declaration order.
        if self.site_has_interventions(site, layer) {
            let mut current_sources: HashMap<String, Vec<f32>> = HashMap::new();
            let mut events: Vec<InterventionEvent> = Vec::new();
            for target in self.interventions.iter() {
                if target.site != site || !target.layers.contains(&layer) {
                    continue;
                }
                let positions: Vec<usize> = if let Some(record) = &target.static_record {
                    record
                        .selected_indices
                        .iter()
                        .copied()
                        .filter(|&position| {
                            if is_decode {
                                position == start_position
                            } else {
                                position < rows
                            }
                        })
                        .collect()
                } else {
                    let step = self.decode_step(start_position);
                    if target.generated_steps.contains(&step) {
                        vec![start_position]
                    } else {
                        Vec::new()
                    }
                };
                if positions.is_empty() {
                    continue;
                }
                let source_rows: Option<Vec<f32>> = match &target.source {
                    None => None,
                    Some(InterventionSource::Zero) => Some(vec![0.0; columns]),
                    Some(InterventionSource::InlineVector { values }) => {
                        if values.len() != columns {
                            return Err(ExperimentError::new(format!(
                                "intervention '{}': inline source has {} values but the site \
                                 tensor has {columns} columns",
                                target.intervention_id,
                                values.len()
                            )));
                        }
                        Some(values.clone())
                    }
                    Some(InterventionSource::CaptureFromCurrentRun { capture_id }) => {
                        if !current_sources.contains_key(capture_id) {
                            let captured_rows = {
                                let result = self.result.as_ref().ok_or_else(|| {
                                    ExperimentError::new("result not initialized before prefill")
                                })?;
                                let captured = result
                                    .captures
                                    .iter()
                                    .find(|c| c.capture_id == *capture_id && c.input_id == input_id)
                                    .ok_or_else(|| {
                                        ExperimentError::new(format!(
                                            "intervention '{}': source capture '{capture_id}' has \
                                             not recorded a value at this point (execution order \
                                             violation)",
                                            target.intervention_id
                                        ))
                                    })?;
                                if captured.columns != columns {
                                    return Err(ExperimentError::new(format!(
                                        "intervention '{}': source capture '{capture_id}' has {} \
                                         columns; the target site has {columns}",
                                        target.intervention_id, captured.columns
                                    )));
                                }
                                captured.rows.clone()
                            };
                            current_sources.insert(capture_id.clone(), captured_rows);
                        }
                        current_sources.get(capture_id).cloned()
                    }
                    Some(InterventionSource::CaptureFromBundle { .. }) => {
                        let bundle = self
                            .bundle_sources
                            .iter()
                            .find(|source| source.intervention_id == target.intervention_id)
                            .ok_or_else(|| {
                                ExperimentError::new(format!(
                                    "intervention '{}': cross-bundle source was not resolved \
                                     before execution",
                                    target.intervention_id
                                ))
                            })?;
                        if bundle.columns != columns {
                            return Err(ExperimentError::new(format!(
                                "intervention '{}': cross-bundle source has {} columns; the \
                                 target site has {columns}",
                                target.intervention_id, bundle.columns
                            )));
                        }
                        Some(bundle.rows.clone())
                    }
                };
                let mut applied = false;
                for &position in &positions {
                    let local = if is_decode { 0 } else { position };
                    let row_start = local * columns;
                    let row = &mut tensor.values_mut()[row_start..row_start + columns];
                    let snapshot = self.snapshots.get(&(site, layer, position)).cloned();
                    match target.operation {
                        InterventionOperation::Replace => {
                            let source = source_rows.as_deref().ok_or_else(|| {
                                ExperimentError::new(format!(
                                    "intervention '{}': replace requires a source",
                                    target.intervention_id
                                ))
                            })?;
                            row.copy_from_slice(source);
                            applied = true;
                        }
                        InterventionOperation::Zero => {
                            row.fill(0.0);
                            applied = true;
                        }
                        InterventionOperation::Scale { factor } => {
                            for value in row.iter_mut() {
                                *value *= factor;
                            }
                            applied = true;
                        }
                        InterventionOperation::Interpolate { alpha } => {
                            let source = source_rows.as_deref().ok_or_else(|| {
                                ExperimentError::new(format!(
                                    "intervention '{}': interpolate requires a source",
                                    target.intervention_id
                                ))
                            })?;
                            for (value, &src) in row.iter_mut().zip(source.iter()) {
                                *value = (1.0 - alpha) * *value + alpha * src;
                            }
                            applied = true;
                        }
                        InterventionOperation::AddDelta => {
                            let source = source_rows.as_deref().ok_or_else(|| {
                                ExperimentError::new(format!(
                                    "intervention '{}': add-delta requires a source",
                                    target.intervention_id
                                ))
                            })?;
                            for (value, &src) in row.iter_mut().zip(source.iter()) {
                                *value += src;
                            }
                            applied = true;
                        }
                        InterventionOperation::RestoreOriginal => {
                            let original = snapshot.as_deref().ok_or_else(|| {
                                ExperimentError::new(format!(
                                    "intervention '{}': no original snapshot exists at \
                                     {} layer {layer} position {position}",
                                    target.intervention_id, site
                                ))
                            })?;
                            row.copy_from_slice(original);
                            applied = true;
                        }
                    }
                }
                let snapshot_checksum = positions
                    .first()
                    .and_then(|position| self.snapshot_checksums.get(&(site, layer, *position)))
                    .cloned();
                events.push(InterventionEvent {
                    intervention_id: target.intervention_id.clone(),
                    input_id: input_id.clone(),
                    site,
                    layer,
                    positions,
                    operation: target.operation,
                    source_kind: target.source.as_ref().map(source_kind),
                    snapshot_checksum,
                    applied,
                });
            }
            let result = self
                .result
                .as_mut()
                .ok_or_else(|| ExperimentError::new("result not initialized before prefill"))?;
            result.events.extend(events);
        }
        Ok(())
    }

    /// Resolve generated-step selection records after generation.
    fn finalize_generated(&mut self, generated: &[u32]) -> Result<(), ExperimentError> {
        let result = self
            .result
            .as_mut()
            .ok_or_else(|| ExperimentError::new("result not initialized"))?;
        let generated_positions: Vec<usize> = (0..generated.len())
            .map(|step| self.prompt_len + step)
            .collect();
        for target in &self.captures {
            for &step in &target.generated_steps {
                let position = resolve_generated_step(step, &generated_positions)
                    .map_err(ExperimentError::new)?;
                let first_position = position[0];
                result.selection_records.push(TokenSelectionRecord {
                    selector: target.selector.clone(),
                    rule: target.selector.rule_id().to_string(),
                    input_text: result.input.text.clone(),
                    normalized_text: result.tokenization.normalized_text.clone(),
                    token_ids: result.tokenization.token_ids.clone(),
                    pieces: result.tokenization.pieces.clone(),
                    byte_offsets: result.tokenization.byte_offsets.clone(),
                    matched_byte_span: None,
                    selected_indices: position,
                    coverage: CoverageKind::Exact,
                    boundary_expansion: None,
                    ambiguity: AmbiguityStatus::Resolved,
                    round_trip: RoundTripStatus::NotApplicable,
                    note: Some(format!(
                        "generated step {step} at absolute position {first_position}"
                    )),
                });
            }
        }
        Ok(())
    }
}

/// Resolved addressing for one capture/intervention target: layer list,
/// optional static token-selection record, and generated steps.
type ResolvedAddressing = (Vec<usize>, Option<TokenSelectionRecord>, Vec<usize>);

/// Resolve the shared capture/intervention addressing pipeline: input
/// membership, layer list, and per-step token selection. Returns `None`
/// when the input is not addressed by the selector.
fn resolve_target_addressing(
    inputs_selector: &InputSelector,
    layers_selector: &LayerSelector,
    tokens_selector: &TokenSelector,
    all_input_ids: &[String],
    input_id: &str,
    n_layers: usize,
    tokenizations: &HashMap<TextNormalization, TokenizationInfo>,
) -> Result<Option<ResolvedAddressing>, ExperimentError> {
    let inputs = inputs_selector
        .resolve(all_input_ids)
        .map_err(ExperimentError::new)?;
    if !inputs.iter().any(|id| id == input_id) {
        return Ok(None);
    }
    let layers = layers_selector
        .resolve(n_layers)
        .map_err(ExperimentError::new)?;
    let (static_record, generated_steps) = if tokens_selector.is_generated() {
        (None, vec![generated_step_of(tokens_selector)?])
    } else {
        let info = tokenizations
            .get(&TextNormalization::None)
            .cloned()
            .ok_or_else(|| ExperimentError::new("tokenization not injected before prefill"))?;
        let record =
            resolve_static_selector(tokens_selector, &info).map_err(ExperimentError::new)?;
        (Some(record), Vec::new())
    };
    Ok(Some((layers, static_record, generated_steps)))
}

fn generated_step_of(selector: &TokenSelector) -> Result<usize, ExperimentError> {
    match selector {
        TokenSelector::GeneratedStep { step } => Ok(*step),
        other => Err(ExperimentError::new(format!(
            "selector {other:?} is not a generated-step selector"
        ))),
    }
}

fn source_kind(source: &InterventionSource) -> String {
    match source {
        InterventionSource::InlineVector { .. } => "inline-vector".into(),
        InterventionSource::CaptureFromCurrentRun { .. } => "capture-from-current-run".into(),
        InterventionSource::CaptureFromBundle { .. } => "capture-from-bundle".into(),
        InterventionSource::Zero => "zero".into(),
    }
}

fn checksum_f32(values: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Deterministic summary statistics over a row slice.
fn summary_stats(
    capture_id: &str,
    input_id: &str,
    site: SemanticHookSite,
    layer: usize,
    positions: Vec<usize>,
    columns: usize,
    values: &[f32],
) -> CaptureSummary {
    let mut finite_count = 0usize;
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &value in values {
        if value.is_finite() {
            finite_count += 1;
            minimum = minimum.min(value);
            maximum = maximum.max(value);
            sum += value as f64;
            sum_sq += (value as f64) * (value as f64);
        }
    }
    if finite_count == 0 {
        minimum = 0.0;
        maximum = 0.0;
    }
    let shape = [positions.len(), columns];
    CaptureSummary {
        capture_id: capture_id.to_string(),
        input_id: input_id.to_string(),
        site,
        layer,
        positions,
        shape,
        finite_count,
        minimum,
        maximum,
        mean: if finite_count > 0 {
            sum / finite_count as f64
        } else {
            0.0
        },
        l2_norm: sum_sq.sqrt(),
    }
}

fn argmax_row(values: &[f32]) -> Option<(usize, f32)> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, &value)| (index, value))
}

impl Experiment for V05Experiment {
    fn name(&self) -> &'static str {
        "v05-experiment"
    }

    fn intervenes(&self) -> bool {
        !self.spec.interventions.is_empty()
    }

    fn uses_activation_site(
        &self,
        stage: ActivationStage,
        layer: Option<usize>,
        phase: ExecutionPhase,
    ) -> bool {
        let site = match stage {
            ActivationStage::BeforeLayer => SemanticHookSite::ResidualPreAttention,
            ActivationStage::AfterAttention => SemanticHookSite::AttentionOutput,
            ActivationStage::AfterMlp => SemanticHookSite::MlpOutput,
            ActivationStage::AfterLayer => SemanticHookSite::ResidualPostMlp,
            ActivationStage::BeforeLogits => SemanticHookSite::FinalNormOutput,
            ActivationStage::AfterLogits => SemanticHookSite::Logits,
        };
        let phase_matches = |generated: bool| match phase {
            ExecutionPhase::Prefill => !generated,
            ExecutionPhase::Decode => generated,
        };
        let layer_matches = |layers: &crate::v05::capture::LayerSelector| {
            if !site.is_per_layer() {
                return layer.is_none();
            }
            layer.is_some_and(|layer| {
                layers
                    .resolve(self.facts.n_layers)
                    .is_ok_and(|layers| layers.contains(&layer))
            })
        };
        self.spec.captures.iter().any(|target| {
            target.site == site
                && phase_matches(target.tokens.is_generated())
                && layer_matches(&target.layers)
        }) || self.spec.interventions.iter().any(|target| {
            target.site == site
                && phase_matches(target.tokens.is_generated())
                && layer_matches(&target.layers)
        })
    }

    fn arguments(&self) -> serde_json::Value {
        serde_json::json!({
            "spec": self.spec.experiment.name,
            "input_index": self.input_index,
        })
    }

    fn on_model_loaded(&mut self, ctx: &ModelContext<'_>) -> Result<(), ExperimentError> {
        self.model_sha256 = ctx.model_sha256.map(ToString::to_string);
        self.tokenizer_sha256 = ctx.tokenizer_sha256.map(ToString::to_string);
        Ok(())
    }

    fn before_prefill(&mut self, ctx: &ExecutionContext<'_>) -> Result<(), ExperimentError> {
        self.prepare_input(ctx)?;
        self.result = Some(InputResult {
            input: self
                .spec
                .inputs
                .get(self.input_index)
                .cloned()
                .ok_or_else(|| ExperimentError::new("input index out of range"))?,
            tokenization: self
                .tokenizations
                .get(&TextNormalization::None)
                .cloned()
                .ok_or_else(|| ExperimentError::new("tokenization not injected"))?,
            selection_records: Vec::new(),
            captures: Vec::new(),
            summaries: Vec::new(),
            events: Vec::new(),
            generated_token_ids: Vec::new(),
            generated_text: String::new(),
            final_top1: None,
        });
        // Static selection records move into the result in declaration
        // order.
        let result = self.result.as_mut().expect("result initialized");
        for target in &self.captures {
            if let Some(record) = &target.static_record {
                result.selection_records.push(record.clone());
            }
        }
        Ok(())
    }

    fn before_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_site(
            &ctx.execution,
            ctx.layer_index,
            SemanticHookSite::ResidualPreAttention,
            hidden,
        )
    }

    fn after_attention(
        &mut self,
        ctx: &LayerContext<'_>,
        attention_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_site(
            &ctx.execution,
            ctx.layer_index,
            SemanticHookSite::AttentionOutput,
            attention_output,
        )
    }

    fn after_mlp(
        &mut self,
        ctx: &LayerContext<'_>,
        mlp_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_site(
            &ctx.execution,
            ctx.layer_index,
            SemanticHookSite::MlpOutput,
            mlp_output,
        )
    }

    fn after_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_site(
            &ctx.execution,
            ctx.layer_index,
            SemanticHookSite::ResidualPostMlp,
            hidden,
        )
    }

    fn before_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_site(ctx, 0, SemanticHookSite::FinalNormOutput, hidden)
    }

    fn after_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        logits: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_site(ctx, 0, SemanticHookSite::Logits, logits)?;
        // Track the final top-1: the last evaluation to fire wins.
        if let Some((index, value)) = argmax_row(logits.values()) {
            self.final_top1 = Some((index as u32, value));
        }
        Ok(())
    }

    fn on_generation_complete(
        &mut self,
        ctx: &GenerationContext<'_>,
    ) -> Result<(), ExperimentError> {
        self.finalize_generated(ctx.generated_token_ids)?;
        let result = self.result.as_mut().expect("result initialized");
        result.generated_token_ids = ctx.generated_token_ids.to_vec();
        result.final_top1 = self.final_top1;
        Ok(())
    }
}

impl V05Experiment {
    /// Consume the per-input result for the bundle, with generated-step
    /// rows merged into single per-(capture, site, layer) tensors in
    /// position order.
    pub fn into_result(&mut self) -> Result<InputResult, ExperimentError> {
        let mut result = self
            .result
            .take()
            .ok_or_else(|| ExperimentError::new("run did not complete"))?;
        // Merge generated-step row buffers: group by (capture, site, layer,
        // full), concatenate rows in ascending position order.
        let mut grouped: Vec<CapturedTensor> = Vec::new();
        for capture in result.captures.drain(..) {
            let mut merged = false;
            for existing in grouped.iter_mut() {
                if existing.capture_id == capture.capture_id
                    && existing.site == capture.site
                    && existing.layer == capture.layer
                    && existing.full_tensor == capture.full_tensor
                    && existing.columns == capture.columns
                {
                    existing.positions.extend(capture.positions.iter().copied());
                    existing.rows.extend(capture.rows.iter().copied());
                    existing.bytes = existing.rows.len() * 4;
                    merged = true;
                    break;
                }
            }
            if !merged {
                grouped.push(capture);
            }
        }
        for capture in &mut grouped {
            // stable sort rows by position (positions and rows are
            // interleaved per fire; merge keeps fire order, so reorder by
            // position deterministically).
            let mut order: Vec<usize> = (0..capture.positions.len()).collect();
            order.sort_by_key(|&i| capture.positions[i]);
            if order
                .windows(2)
                .any(|pair| capture.positions[pair[0]] > capture.positions[pair[1]])
            {
                let columns = capture.columns;
                let mut rows = Vec::with_capacity(capture.rows.len());
                let mut positions = Vec::with_capacity(capture.positions.len());
                for &i in &order {
                    rows.extend_from_slice(&capture.rows[i * columns..(i + 1) * columns]);
                    positions.push(capture.positions[i]);
                }
                capture.rows = rows;
                capture.positions = positions;
            }
        }
        grouped.sort_by(|a, b| {
            a.capture_id
                .cmp(&b.capture_id)
                .then(site_order(a.site).cmp(&site_order(b.site)))
                .then(a.layer.cmp(&b.layer))
        });
        result.captures = grouped;
        result.summaries.sort_by(|a, b| {
            a.capture_id
                .cmp(&b.capture_id)
                .then(site_order(a.site).cmp(&site_order(b.site)))
                .then(a.layer.cmp(&b.layer))
        });
        result.selection_records.sort_by(|a, b| {
            a.rule
                .cmp(&b.rule)
                .then(a.selector.to_string().cmp(&b.selector.to_string()))
        });
        Ok(result)
    }
}

fn site_order(site: SemanticHookSite) -> usize {
    match site {
        SemanticHookSite::ResidualPreAttention => 0,
        SemanticHookSite::AttentionOutput => 1,
        SemanticHookSite::MlpOutput => 2,
        SemanticHookSite::ResidualPostMlp => 3,
        SemanticHookSite::FinalNormOutput => 4,
        SemanticHookSite::Logits => 5,
    }
}

/// A cross-bundle source preloaded by the driver (contract section 5).
#[derive(Debug, Clone)]
pub struct BundleSource {
    pub intervention_id: String,
    pub source: InterventionSource,
    pub rows: Vec<f32>,
    pub columns: usize,
}

/// Load and validate a cross-bundle source against the target experiment.
///
/// Fails closed on model/tokenizer SHA mismatch, hook-site mismatch, layer
/// mismatch, shape mismatch, missing capture, or unverified source bundle.
pub fn load_bundle_source(
    intervention: &InterventionSpec,
    source: &InterventionSource,
    target_model_sha: &str,
    target_tokenizer_sha: &str,
    n_layers: usize,
) -> Result<BundleSource, String> {
    let InterventionSource::CaptureFromBundle {
        bundle_path,
        capture_id,
        input_id,
        layer,
    } = source
    else {
        return Err("source is not a bundle source".into());
    };
    if *layer >= n_layers {
        return Err(format!(
            "intervention '{}': source bundle layer {layer} is out of range for a \
             {n_layers}-layer model",
            intervention.id
        ));
    }
    let bundle = crate::v05::verify::load_bundle_for_source(bundle_path)?;
    if bundle.semantic_manifest.model.sha256 != target_model_sha
        && !intervention.compatibility.allow_model_mismatch
    {
        return Err(format!(
            "intervention '{}': source bundle model SHA {} does not match the target model \
             {} (expert override available but discouraged)",
            intervention.id, bundle.semantic_manifest.model.sha256, target_model_sha
        ));
    }
    if bundle.semantic_manifest.tokenizer.sha256 != target_tokenizer_sha
        && !intervention.compatibility.allow_tokenizer_mismatch
    {
        return Err(format!(
            "intervention '{}': source bundle tokenizer SHA does not match the target \
             tokenizer",
            intervention.id
        ));
    }
    let index_entry = bundle
        .capture_index
        .iter()
        .find(|entry| {
            entry.capture_id == *capture_id && entry.input_id == *input_id && entry.layer == *layer
        })
        .ok_or_else(|| {
            format!(
                "intervention '{}': source bundle has no capture '{capture_id}' for input \
                 '{input_id}' at layer {layer}",
                intervention.id
            )
        })?;
    if index_entry.site != intervention.site {
        return Err(format!(
            "intervention '{}': source capture site {} does not match the intervention site {}",
            intervention.id, index_entry.site, intervention.site
        ));
    }
    let rows = bundle
        .tensor_f32_by_name(&index_entry.tensor_name)
        .map_err(|error| {
            format!(
                "intervention '{}': failed to load source tensor: {error}",
                intervention.id
            )
        })?;
    Ok(BundleSource {
        intervention_id: intervention.id.clone(),
        source: source.clone(),
        rows,
        columns: index_entry.shape.last().copied().unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::{LayerContext, ModelFamily, TracingState};
    use crate::v05::spec::RawExperimentSpec;

    /// A resolvable v0.5 spec exercising both captures (selected-rows and
    /// summary-only) and three interventions (zero, scale, replace with an
    /// inline source) across two layers, two inputs.
    fn test_spec() -> ExperimentSpecV1 {
        let text = r#"
schema = "ember.experiment.v1"

[experiment]
name = "runner-test"
description = "unit test of the v0.5 runner"
seed = 42

[model]
path = "/models/tiny.gguf"
expected_sha256 = "aa"

[execution]
mode = "planned"
threads = 1
deterministic = true

[generation]
max_new_tokens = 2
temperature = 0.0

[[inputs]]
id = "i1"
text = "hello world"

[[inputs]]
id = "i2"
text = "second prompt"

[[captures]]
id = "cap-attn"
site = "attention-output"
layers = [0]

[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "cap-mlp"
site = "mlp-output"
layers = [0, 1]
storage = "summary-only"

[captures.tokens]
kind = "prompt-final"

[[interventions]]
id = "iv-zero"
site = "mlp-output"
layers = [1]
operation = { kind = "zero" }

[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "iv-scale"
site = "mlp-output"
layers = [1]
operation = { kind = "scale", factor = 0.5 }

[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "iv-replace"
site = "attention-output"
layers = [0]
operation = { kind = "replace" }
source = { kind = "inline-vector", values = [7.0, 8.0, 9.0, 10.0] }

[interventions.tokens]
kind = "prompt-final"

[output]
directory = "runs/runner-test"
"#;
        RawExperimentSpec::from_toml_str(text)
            .expect("spec parses")
            .resolve()
            .expect("spec resolves")
    }

    fn model_ctx() -> ModelContext<'static> {
        ModelContext::new(ModelFamily::Llama, Some("tiny.gguf"), "llama", 2, 4)
    }

    fn new_experiment(spec: &ExperimentSpecV1, input_index: usize) -> V05Experiment {
        let mut experiment = V05Experiment::new(
            spec.clone(),
            input_index,
            ModelFacts {
                n_layers: 2,
                embed_dim: 4,
                vocab_size: 100,
            },
            Some("model-sha".into()),
            Some("tokenizer-sha".into()),
        );
        // Prompt-final selectors resolve against the tokenization, so inject
        // a synthetic one ("hello world" -> 3 tokens).
        experiment.inject_tokenization(TokenizationInfo {
            text: "hello world".into(),
            normalized_text: "hello world".into(),
            token_ids: vec![1, 2, 3],
            pieces: vec!["<s>".into(), "hello".into(), "world".into()],
            byte_offsets: vec![(0, 0), (0, 5), (6, 11)],
        });
        experiment
    }

    fn exec_ctx(
        model: ModelContext<'static>,
        phase: ExecutionPhase,
        position: usize,
        token_count: usize,
    ) -> ExecutionContext<'static> {
        ExecutionContext::new(model, phase, position, token_count, TracingState::Disabled)
    }

    #[test]
    fn plan_site_routing_is_phase_and_layer_exact() {
        let experiment = new_experiment(&test_spec(), 0);
        assert!(experiment.uses_activation_site(
            ActivationStage::AfterAttention,
            Some(0),
            ExecutionPhase::Prefill,
        ));
        assert!(!experiment.uses_activation_site(
            ActivationStage::AfterAttention,
            Some(1),
            ExecutionPhase::Prefill,
        ));
        assert!(!experiment.uses_activation_site(
            ActivationStage::AfterAttention,
            Some(0),
            ExecutionPhase::Decode,
        ));
        assert!(experiment.uses_activation_site(
            ActivationStage::AfterMlp,
            Some(1),
            ExecutionPhase::Prefill,
        ));
        assert!(!experiment.uses_activation_site(
            ActivationStage::BeforeLogits,
            None,
            ExecutionPhase::Prefill,
        ));
    }

    #[test]
    fn before_prefill_resolves_captures_and_interventions_per_input() {
        let spec = test_spec();
        let mut e1 = new_experiment(&spec, 0);
        e1.before_prefill(&exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3))
            .expect("prepare i1");
        assert_eq!(e1.captures.len(), 2, "i1 targets both captures");
        assert_eq!(
            e1.interventions.len(),
            3,
            "i1 targets all three interventions"
        );
        assert_eq!(e1.prompt_len, 3);
        // capture layer resolution
        assert_eq!(e1.captures[0].layers, vec![0]);
        assert_eq!(e1.captures[1].layers, vec![0, 1]);
        // selection predicates
        assert!(e1.site_has_captures(SemanticHookSite::AttentionOutput, 0));
        assert!(!e1.site_has_captures(SemanticHookSite::AttentionOutput, 1));
        assert!(e1.site_has_captures(SemanticHookSite::MlpOutput, 1));
        assert!(e1.site_has_interventions(SemanticHookSite::MlpOutput, 1));
        assert!(!e1.site_has_interventions(SemanticHookSite::MlpOutput, 0));
        // result initialized with selection records for static selectors
        let result = e1.result.as_ref().expect("result initialized");
        assert_eq!(result.selection_records.len(), 2);
        assert_eq!(result.input.id, "i1");

        // i2 does not restrict anything: same resolutions
        let mut e2 = new_experiment(&spec, 1);
        e2.before_prefill(&exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3))
            .expect("prepare i2");
        assert_eq!(e2.captures.len(), 2);
        assert_eq!(e2.result.as_ref().unwrap().input.id, "i2");
    }

    #[test]
    fn fire_site_captures_prompt_final_row() {
        let spec = test_spec();
        let mut e = new_experiment(&spec, 0);
        let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        e.before_prefill(&ctx).expect("prepare");

        // 3 rows x 4 cols; prompt-final selects the LAST row.
        let mut values = vec![1.0f32; 12];
        values[8..12].copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
        let mut tensor = TensorAccess::new(3, 4, &mut values);
        e.after_attention(&LayerContext::new(ctx, 0), &mut tensor)
            .expect("fire capture");

        let result = e.result.as_ref().unwrap();
        assert_eq!(result.captures.len(), 1, "attention-output capture");
        let captured = &result.captures[0];
        assert_eq!(captured.capture_id, "cap-attn");
        assert_eq!(captured.site, SemanticHookSite::AttentionOutput);
        assert_eq!(
            captured.positions,
            vec![2],
            "prompt-final selects last position"
        );
        assert_eq!(captured.columns, 4);
        assert_eq!(captured.rows, vec![10.0, 20.0, 30.0, 40.0]);
        assert!(!captured.full_tensor);
    }

    #[test]
    fn fire_site_captures_summary_only_stats() {
        let spec = test_spec();
        let mut e = new_experiment(&spec, 0);
        let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        e.before_prefill(&ctx).expect("prepare");

        let mut values = vec![0.0f32; 12];
        values[8..12].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let mut tensor = TensorAccess::new(3, 4, &mut values);
        // cap-mlp targets layers [0, 1] at residual-post-mlp
        e.after_mlp(&LayerContext::new(ctx, 1), &mut tensor)
            .expect("fire summary capture");

        let result = e.result.as_ref().unwrap();
        assert_eq!(
            result.captures.len(),
            0,
            "summary-only produces no tensor capture"
        );
        assert_eq!(result.summaries.len(), 1);
        let summary = &result.summaries[0];
        assert_eq!(summary.capture_id, "cap-mlp");
        assert_eq!(summary.layer, 1);
        assert_eq!(summary.shape, [1, 4]);
        assert_eq!(summary.minimum, 1.0);
        assert_eq!(summary.maximum, 4.0);
        assert!((summary.mean - 2.5).abs() < 1e-6);
        assert_eq!(summary.finite_count, 4);
    }

    #[test]
    fn fire_site_zero_intervention_zeros_target_row() {
        let spec = test_spec();
        let mut e = new_experiment(&spec, 0);
        let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        e.before_prefill(&ctx).expect("prepare");

        let mut values = vec![5.0f32; 12];
        let mut tensor = TensorAccess::new(3, 4, &mut values);
        e.after_mlp(&LayerContext::new(ctx, 1), &mut tensor)
            .expect("fire zero intervention");

        // iv-zero targets (residual-post-mlp, layer 1), prompt-final -> row 2.
        assert_eq!(tensor.values()[8..12], vec![0.0; 4]);
        // other rows untouched
        assert_eq!(tensor.values()[0..8], vec![5.0; 8]);

        let result = e.result.as_ref().unwrap();
        let event = result
            .events
            .iter()
            .find(|ev| ev.intervention_id == "iv-zero")
            .expect("zero event recorded");
        assert!(event.applied);
        assert_eq!(event.positions, vec![2]);
        assert!(
            event.snapshot_checksum.is_some(),
            "snapshot taken before zeroing"
        );
    }

    #[test]
    fn fire_site_scale_intervention_scales_row() {
        let mut spec = test_spec();
        // Isolate: without iv-zero firing first (declaration order), the row
        // is scaled but not zeroed.
        spec.interventions.retain(|i| i.id == "iv-scale");
        let mut e = new_experiment(&spec, 0);
        let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        e.before_prefill(&ctx).expect("prepare");

        let mut values = vec![4.0f32; 12];
        let mut tensor = TensorAccess::new(3, 4, &mut values);
        e.after_mlp(&LayerContext::new(ctx, 1), &mut tensor)
            .expect("fire scale intervention");

        // iv-scale: factor 0.5 on the prompt-final row (index 2).
        assert_eq!(tensor.values()[8..12], vec![2.0; 4]);
        assert_eq!(tensor.values()[0..8], vec![4.0; 8]);
    }

    #[test]
    fn fire_site_replace_uses_inline_source_and_checks_columns() {
        let spec = test_spec();
        let mut e = new_experiment(&spec, 0);
        let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        e.before_prefill(&ctx).expect("prepare");

        let mut values = vec![1.0f32; 12];
        let mut tensor = TensorAccess::new(3, 4, &mut values);
        e.after_attention(&LayerContext::new(ctx, 0), &mut tensor)
            .expect("fire replace");
        assert_eq!(
            tensor.values()[8..12],
            vec![7.0, 8.0, 9.0, 10.0],
            "inline source replaces the prompt-final row"
        );
        assert_eq!(tensor.values()[0..8], vec![1.0; 8]);

        // Column mismatch must be a clean error, not a panic.
        let mut e2 = new_experiment(&spec, 0);
        e2.before_prefill(&ctx).expect("prepare");
        let mut bad_values = vec![1.0f32; 6]; // 3 rows x 2 cols
        let mut bad_tensor = TensorAccess::new(3, 2, &mut bad_values);
        let err = e2
            .after_attention(&LayerContext::new(ctx, 0), &mut bad_tensor)
            .expect_err("inline source width mismatch is an error");
        assert!(err.message().contains("inline source"), "{err:?}");
    }

    #[test]
    fn restore_original_restores_pre_intervention_snapshot() {
        // Two interventions at the same site in declaration order: replace
        // (mutates), then restore-original (writes the snapshot back).
        let mut spec = test_spec();
        // build a derived spec adding restore-original after the replace
        let mut interventions = spec.interventions.clone();
        let mut restore = interventions[2].clone(); // iv-replace
        restore.id = "iv-restore".into();
        restore.operation = InterventionOperation::RestoreOriginal;
        restore.source = None;
        interventions.push(restore);
        spec.interventions = interventions;

        let mut e = new_experiment(&spec, 0);
        let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        e.before_prefill(&ctx).expect("prepare");

        let mut values = vec![3.0f32; 12];
        values[8..12].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let mut tensor = TensorAccess::new(3, 4, &mut values);
        e.after_attention(&LayerContext::new(ctx, 0), &mut tensor)
            .expect("fire replace+restore");

        // Replace writes 7..10, then restore-original writes the snapshot
        // (the pre-intervention row) back -> original 1.0s.
        assert_eq!(tensor.values()[8..12], vec![1.0, 1.0, 1.0, 1.0]);
        let result = e.result.as_ref().unwrap();
        let events: Vec<_> = result
            .events
            .iter()
            .filter(|ev| ev.intervention_id == "iv-restore")
            .collect();
        assert_eq!(events.len(), 1);
        assert!(events[0].applied);
        assert!(events[0].snapshot_checksum.is_some());
    }

    #[test]
    fn decode_phase_generated_step_intervention_applies_to_decode_row() {
        // Build a spec whose mlp-output intervention targets generated step 1
        // (the first decode token, absolute position = prompt_len + 0).
        let text = r#"
schema = "ember.experiment.v1"

[experiment]
name = "decode-intervention-test"
description = "generated-step intervention"
seed = 1

[model]
path = "/models/tiny.gguf"
expected_sha256 = "aa"

[execution]
mode = "planned"
threads = 1
deterministic = true

[generation]
max_new_tokens = 2
temperature = 0.0

[[inputs]]
id = "i1"
text = "hello world"

[[interventions]]
id = "iv-gen"
site = "mlp-output"
layers = [1]
operation = { kind = "zero" }

[interventions.tokens]
kind = "generated-step"
step = 1

[output]
directory = "runs/decode-intervention-test"
"#;
        let spec = RawExperimentSpec::from_toml_str(text)
            .unwrap()
            .resolve()
            .expect("resolves");
        let mut e = new_experiment(&spec, 0);
        let prefill = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        e.before_prefill(&prefill).expect("prepare");

        // Decode step 1 = absolute position 3 (prompt_len 3 + step 1 - 1).
        let decode = exec_ctx(model_ctx(), ExecutionPhase::Decode, 3, 1);
        let mut values = vec![2.0f32; 4];
        let mut tensor = TensorAccess::new(1, 4, &mut values);
        e.after_mlp(&LayerContext::new(decode, 1), &mut tensor)
            .expect("fire decode intervention");

        assert_eq!(
            tensor.values(),
            vec![0.0; 4],
            "zero applies to the decode row"
        );
        let event = e
            .result
            .as_ref()
            .unwrap()
            .events
            .iter()
            .find(|ev| ev.intervention_id == "iv-gen")
            .expect("event");
        assert_eq!(event.positions, vec![3], "absolute position recorded");
    }

    #[test]
    fn generated_step_capture_fires_only_on_requested_step() {
        let text = r#"
schema = "ember.experiment.v1"

[experiment]
name = "gen-capture-test"
description = "generated-step capture"
seed = 1

[model]
path = "/models/tiny.gguf"
expected_sha256 = "aa"

[execution]
mode = "planned"
threads = 1
deterministic = true

[generation]
max_new_tokens = 2
temperature = 0.0

[[inputs]]
id = "i1"
text = "hello world"

[[captures]]
id = "cap-gen"
site = "mlp-output"
layers = [0]

[captures.tokens]
kind = "generated-step"
step = 1

[output]
directory = "runs/gen-capture-test"
"#;
        let spec = RawExperimentSpec::from_toml_str(text)
            .unwrap()
            .resolve()
            .expect("resolves");
        let mut e = new_experiment(&spec, 0);
        let prefill = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        e.before_prefill(&prefill).expect("prepare");

        let mut values = vec![9.0f32; 4];
        let mut tensor = TensorAccess::new(1, 4, &mut values);
        // decode step 1 = position 3 (1-based: prompt_len 3 + step 1 - 1):
        // the requested step, so the capture fires here.
        e.after_mlp(
            &LayerContext::new(exec_ctx(model_ctx(), ExecutionPhase::Decode, 3, 1), 0),
            &mut tensor,
        )
        .expect("fire");
        let captures = &e.result.as_ref().unwrap().captures;
        assert_eq!(captures.len(), 1, "generated step 1 fires at position 3");
        assert_eq!(captures[0].capture_id, "cap-gen");
        assert_eq!(captures[0].positions, vec![3]);
        assert_eq!(captures[0].rows, vec![9.0; 4]);
        // decode step 2 (position 4): not requested -> no additional capture
        e.after_mlp(
            &LayerContext::new(exec_ctx(model_ctx(), ExecutionPhase::Decode, 4, 1), 0),
            &mut tensor,
        )
        .expect("fire");
        assert_eq!(e.result.as_ref().unwrap().captures.len(), 1);

        // finalize_generated appends the generated-step selection record.
        e.finalize_generated(&[7, 8]).expect("finalize");
        let result = e.result.as_ref().unwrap();
        assert!(result.generated_token_ids.is_empty()); // not set by finalize
        let record = result
            .selection_records
            .iter()
            .find(|r| r.selector.rule_id() == "generated-step")
            .expect("generated-step selection record");
        assert_eq!(
            record.selected_indices,
            vec![3],
            "prompt_len 3 + (step 1 - 1)"
        );
    }

    #[test]
    fn fire_site_before_prefill_is_a_noop() {
        // Without before_prefill there are no resolved captures/interventions,
        // so hook fires are harmless no-ops (the "result not initialized"
        // error is only reachable if prepare_input succeeded but before_prefill
        // failed midway — a defensive path).
        let spec = test_spec();
        let mut e = new_experiment(&spec, 0);
        let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        let mut values = vec![1.0f32; 4];
        let mut tensor = TensorAccess::new(1, 4, &mut values);
        e.after_attention(&LayerContext::new(ctx, 0), &mut tensor)
            .expect("noop fire succeeds");
        assert_eq!(tensor.values(), vec![1.0; 4]);
        assert!(e.result.is_none());
    }

    #[test]
    fn into_result_assembles_complete_input_result() {
        let spec = test_spec();
        let mut e = new_experiment(&spec, 0);
        let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        e.before_prefill(&ctx).expect("prepare");
        let mut values = vec![1.0f32; 12];
        let mut tensor = TensorAccess::new(3, 4, &mut values);
        e.after_attention(&LayerContext::new(ctx, 0), &mut tensor)
            .expect("fire");
        e.finalize_generated(&[5]).expect("finalize");
        e.set_generated_text("hello".into());

        let result = e.into_result().expect("into_result");
        assert_eq!(result.input.id, "i1");
        assert_eq!(result.generated_text, "hello");
        assert_eq!(result.captures.len(), 1);
        assert_eq!(result.selection_records.len(), 2);
        assert_eq!(
            result.events.len(),
            1,
            "iv-replace applied at attention-output"
        );
        assert!(result.final_top1.is_none());
    }
}
