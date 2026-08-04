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

use crate::experiments::{
    ExecutionContext, ExecutionPhase, Experiment, ExperimentError, GenerationContext, LayerContext,
    ModelContext, TensorAccess,
};
use crate::v05::capture::CaptureStorage;
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
            let inputs = capture
                .inputs
                .resolve(&all_input_ids)
                .map_err(ExperimentError::new)?;
            if !inputs.contains(&input.id) {
                continue;
            }
            let layers = capture
                .layers
                .resolve(self.facts.n_layers)
                .map_err(ExperimentError::new)?;
            let (static_record, generated_steps) = if capture.tokens.is_generated() {
                (None, vec![generated_step_of(&capture.tokens)?])
            } else {
                let info = self
                    .tokenizations
                    .get(&TextNormalization::None)
                    .cloned()
                    .ok_or_else(|| {
                        ExperimentError::new("tokenization not injected before prefill")
                    })?;
                let record = resolve_static_selector(&capture.tokens, &info)
                    .map_err(ExperimentError::new)?;
                (Some(record), Vec::new())
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
            let inputs = intervention
                .inputs
                .resolve(&all_input_ids)
                .map_err(ExperimentError::new)?;
            if !inputs.contains(&input.id) {
                continue;
            }
            let layers = intervention
                .layers
                .resolve(self.facts.n_layers)
                .map_err(ExperimentError::new)?;
            let (static_record, generated_steps) = if intervention.tokens.is_generated() {
                (None, vec![generated_step_of(&intervention.tokens)?])
            } else {
                let info = self
                    .tokenizations
                    .get(&TextNormalization::None)
                    .cloned()
                    .ok_or_else(|| {
                        ExperimentError::new("tokenization not injected before prefill")
                    })?;
                let record = resolve_static_selector(&intervention.tokens, &info)
                    .map_err(ExperimentError::new)?;
                (Some(record), Vec::new())
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
