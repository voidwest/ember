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
    /// Query heads and their width: the `attention-heads` site is
    /// `n_heads * head_dim` wide.
    pub n_heads: usize,
    pub head_dim: usize,
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
    /// Selected heads at `attention-heads` (`None`: every column).
    pub heads: Option<Vec<usize>>,
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
    /// Selected heads at `attention-heads` (`None`: every column).
    pub heads: Option<Vec<usize>>,
    /// The dictionary features the operation acts on (ADR-0003).
    pub features: Option<crate::v05::dictionary::FeatureSelection>,
}

/// The values of `row` in the columns of `heads` (each `head_dim` wide), in
/// head order; every value when `heads` is `None`.
fn gather_heads(row: &[f32], heads: Option<&[usize]>, head_dim: usize) -> Vec<f32> {
    match heads {
        None => row.to_vec(),
        Some(heads) => heads
            .iter()
            .flat_map(|&head| row[head * head_dim..(head + 1) * head_dim].iter().copied())
            .collect(),
    }
}

/// The width of a target: the site's `columns`, or the selected heads'
/// columns (the site must then be exactly `heads * head_dim` wide).
fn target_width(
    heads: Option<&[usize]>,
    columns: usize,
    head_dim: usize,
) -> Result<usize, ExperimentError> {
    let Some(heads) = heads else {
        return Ok(columns);
    };
    if head_dim == 0
        || !columns.is_multiple_of(head_dim)
        || heads.iter().any(|&head| (head + 1) * head_dim > columns)
    {
        return Err(ExperimentError::new(format!(
            "a heads selector needs a site of whole {head_dim}-wide heads; this one has \
             {columns} columns"
        )));
    }
    Ok(heads.len() * head_dim)
}

/// Rows of a capture source (`source_columns` wide) for a target of `width`
/// columns at a site of `columns`: as they are when the widths match, or,
/// with a heads selector, full site rows reduced to the selected heads, so a
/// head is patched from the same head of the source. `None` otherwise.
fn source_for_width(
    rows: &[f32],
    source_columns: usize,
    columns: usize,
    width: usize,
    heads: Option<&[usize]>,
    head_dim: usize,
) -> Option<Vec<f32>> {
    if source_columns == width {
        return Some(rows.to_vec());
    }
    match heads {
        Some(heads) if source_columns == columns && columns > 0 => Some(
            rows.chunks_exact(columns)
                .flat_map(|row| gather_heads(row, Some(heads), head_dim))
                .collect(),
        ),
        _ => None,
    }
}

/// Write `values` (from [`gather_heads`]) back into the columns of `heads`.
fn scatter_heads(row: &mut [f32], values: &[f32], heads: &[usize], head_dim: usize) {
    for (index, &head) in heads.iter().enumerate() {
        row[head * head_dim..(head + 1) * head_dim]
            .copy_from_slice(&values[index * head_dim..(index + 1) * head_dim]);
    }
}

/// Check a resolved `heads` selector against the model's head count.
fn check_heads(
    kind: &str,
    id: &str,
    heads: Option<&[usize]>,
    n_heads: usize,
) -> Result<(), ExperimentError> {
    match heads.and_then(|heads| heads.iter().find(|&&head| head >= n_heads)) {
        None => Ok(()),
        Some(head) => Err(ExperimentError::new(format!(
            "{kind} '{id}': head {head} is out of range for a {n_heads}-head model"
        ))),
    }
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
    /// Resolved directions of `vector-file`/`contrastive` sources, keyed by
    /// (intervention id, layer); head sites use layer 0.
    directions: HashMap<(String, usize), Vec<f32>>,
    /// Dictionaries the driver loaded before execution, by id (ADR-0003).
    dictionaries: HashMap<String, std::sync::Arc<crate::v05::dictionary::Dictionary>>,
    // runtime state
    prompt_len: usize,
    tokenizations: HashMap<TextNormalization, TokenizationInfo>,
    result: Option<InputResult>,
    snapshots: HashMap<SnapshotKey, Vec<f32>>,
    snapshot_checksums: HashMap<SnapshotKey, String>,
    final_top1: Option<(u32, f32)>,
    /// Observations recorded before the shared-prefix boundary by an
    /// observer instance during the base run; merged at `before_prefill`.
    pending_prefix: Option<PrefixObservations>,
}

/// Captures and summaries an experiment recorded during a prefill, before
/// the shared-prefix boundary (see `crate::v05::prefix`).
#[derive(Debug, Clone, Default)]
pub struct PrefixObservations {
    pub captures: Vec<CapturedTensor>,
    pub summaries: Vec<CaptureSummary>,
}

impl V05Experiment {
    #[allow(
        clippy::too_many_arguments,
        reason = "application call path: each argument is a separate input"
    )]
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
            directions: HashMap::new(),
            dictionaries: HashMap::new(),
            prompt_len: 0,
            tokenizations: HashMap::new(),
            result: None,
            snapshots: HashMap::new(),
            snapshot_checksums: HashMap::new(),
            final_top1: None,
            pending_prefix: None,
        }
    }

    /// Hand this experiment the observations its observer instance recorded
    /// before the shared-prefix boundary. They enter the result at
    /// `before_prefill`, exactly where the skipped hooks would have put them
    /// (every one of them precedes every hook that still fires).
    pub fn inject_prefix_observations(&mut self, observations: PrefixObservations) {
        self.pending_prefix = Some(observations);
    }

    /// Take what an observer instance recorded (its prefill captures and
    /// summaries). The observer is consumed for this input.
    pub fn take_prefix_observations(&mut self) -> Result<PrefixObservations, ExperimentError> {
        let result = self
            .result
            .take()
            .ok_or_else(|| ExperimentError::new("observer did not see a prefill"))?;
        Ok(PrefixObservations {
            captures: result.captures,
            summaries: result.summaries,
        })
    }

    /// Whether any intervention of this input acts at a per-layer site of
    /// `layer` during prefill (resolved at `before_prefill`; generated-step
    /// interventions act only in decode). Observers use it to refuse a hook
    /// that could mutate the shared tensor.
    pub fn intervenes_in_prefill_layer(&self, layer: usize) -> bool {
        self.interventions.iter().any(|target| {
            target.static_record.is_some()
                && target.site.is_per_layer()
                && target.layers.contains(&layer)
        })
    }

    /// Inject a cross-bundle source resolved by the driver before
    /// execution.
    pub fn inject_bundle_source(&mut self, source: BundleSource) {
        self.bundle_sources.push(source);
    }

    /// Inject a direction the driver resolved before execution for
    /// `intervention_id` at `layer` (layer 0 for head sites).
    pub fn inject_direction(&mut self, intervention_id: &str, layer: usize, values: Vec<f32>) {
        self.directions
            .insert((intervention_id.to_string(), layer), values);
    }

    /// Inject a dictionary the driver loaded before execution.
    pub fn inject_dictionary(
        &mut self,
        dictionary: std::sync::Arc<crate::v05::dictionary::Dictionary>,
    ) {
        self.dictionaries
            .insert(dictionary.record.id.clone(), dictionary);
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
            check_heads(
                "capture",
                &capture.id,
                capture.heads.as_deref(),
                self.facts.n_heads,
            )?;
            captures.push(CaptureTarget {
                capture_id: capture.id.clone(),
                site: capture.site,
                layers,
                storage: capture.storage,
                dtype: capture.dtype,
                selector: capture.tokens.clone(),
                static_record,
                generated_steps,
                heads: capture.heads.clone(),
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
            check_heads(
                "intervention",
                &intervention.id,
                intervention.heads.as_deref(),
                self.facts.n_heads,
            )?;
            interventions.push(InterventionTarget {
                intervention_id: intervention.id.clone(),
                site: intervention.site,
                layers,
                operation: intervention.operation,
                source: intervention.source.clone(),
                selector: intervention.tokens.clone(),
                static_record,
                generated_steps,
                heads: intervention.heads.clone(),
                features: intervention.features.clone(),
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
            if let Some(record) = &target.static_record
                && record.selected_indices.contains(&absolute)
            {
                return true;
            }
            if absolute >= self.prompt_len && !target.generated_steps.is_empty() {
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
        let head_dim = self.facts.head_dim;
        let is_decode = ctx.phase == ExecutionPhase::Decode;
        let start_position = ctx.start_position;
        // Prefill head hooks expose only the prompt-final row; layer hooks
        // expose the entire prompt. Decode exposes one absolute-position row.
        let row_base = if is_decode {
            start_position
        } else if !site.is_per_layer() {
            self.prompt_len.saturating_sub(1)
        } else {
            0
        };
        let input_id = self
            .spec
            .inputs
            .get(self.input_index)
            .map(|i| i.id.clone())
            .ok_or_else(|| ExperimentError::new("input index out of range"))?;

        // 1. snapshots for interventions at this site.
        if self.site_has_interventions(site, layer) {
            for local_row in 0..rows {
                let absolute = row_base + local_row;
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
                // A stored row is the site row, or its selected heads.
                let heads = target.heads.as_deref();
                let width = target_width(heads, columns, head_dim)?;
                let selected = |local: usize| {
                    gather_heads(
                        &tensor.values()[local * columns..(local + 1) * columns],
                        heads,
                        head_dim,
                    )
                };
                if target.storage == CaptureStorage::SummaryOnly {
                    // Deterministic statistics over the selected rows.
                    let wanted: Vec<usize> = if let Some(record) = &target.static_record {
                        record
                            .selected_indices
                            .iter()
                            .copied()
                            .filter(|&position| position >= row_base && position - row_base < rows)
                            .collect()
                    } else {
                        let step = self.decode_step(start_position);
                        if is_decode && target.generated_steps.contains(&step) {
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
                        values.extend(selected(position - row_base));
                    }
                    summaries.push(summary_stats(
                        &target.capture_id,
                        &input_id,
                        site,
                        layer,
                        wanted,
                        width,
                        &values,
                    ));
                    continue;
                }
                if let Some(record) = &target.static_record {
                    let wanted: Vec<usize> = record
                        .selected_indices
                        .iter()
                        .copied()
                        .filter(|&position| position >= row_base && position - row_base < rows)
                        .collect();
                    if wanted.is_empty() {
                        continue;
                    }
                    let full = target.storage == CaptureStorage::FullTensor;
                    let mut rows_out = Vec::new();
                    let positions: Vec<usize> = if full && !is_decode {
                        for local in 0..rows {
                            rows_out.extend(selected(local));
                        }
                        (row_base..row_base + rows).collect()
                    } else {
                        for &position in &wanted {
                            rows_out.extend(selected(position - row_base));
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
                            columns: width,
                            rows: rows_out,
                            full_tensor: full,
                            bytes,
                            dtype: target.dtype,
                        },
                    ));
                } else {
                    // generated-step capture: buffer only the requested steps.
                    let step = self.decode_step(start_position);
                    if !is_decode || !target.generated_steps.contains(&step) {
                        continue;
                    }
                    let row = selected(0);
                    pending.push((
                        site_order(site) * 1_000_000 + layer,
                        CapturedTensor {
                            capture_id: target.capture_id.clone(),
                            input_id: input_id.clone(),
                            site,
                            layer,
                            positions: vec![start_position],
                            columns: width,
                            rows: row,
                            full_tensor: false,
                            bytes: width * 4,
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
            let mut current_sources: HashMap<String, (Vec<f32>, usize)> = HashMap::new();
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
                        .filter(|&position| position >= row_base && position - row_base < rows)
                        .collect()
                } else {
                    let step = self.decode_step(start_position);
                    if is_decode && target.generated_steps.contains(&step) {
                        vec![start_position]
                    } else {
                        Vec::new()
                    }
                };
                if positions.is_empty() {
                    continue;
                }
                // A feature target: the operation sees the selected features
                // of a dictionary read at this site, not the row.
                let feature_view = match &target.features {
                    None => None,
                    Some(selection) => {
                        let dictionary =
                            self.dictionaries
                                .get(&selection.dictionary)
                                .ok_or_else(|| {
                                    ExperimentError::new(format!(
                                        "intervention '{}': dictionary '{}' was not loaded before \
                                     execution",
                                        target.intervention_id, selection.dictionary
                                    ))
                                })?;
                        if dictionary.d_in() != columns {
                            return Err(ExperimentError::new(format!(
                                "intervention '{}': dictionary '{}' reads {} columns; site \
                                 {site} has {columns}",
                                target.intervention_id,
                                selection.dictionary,
                                dictionary.d_in()
                            )));
                        }
                        dictionary.check_ids(&selection.ids).map_err(|error| {
                            ExperimentError::new(format!(
                                "intervention '{}': {error}",
                                target.intervention_id
                            ))
                        })?;
                        Some((std::sync::Arc::clone(dictionary), selection.ids.as_slice()))
                    }
                };
                // A capture source holds raw site rows; a feature target
                // encodes them with its dictionary.
                let fit_source =
                    |rows: &[f32], source_columns: usize, width: usize| match &feature_view {
                        Some((dictionary, ids)) => {
                            (source_columns == columns && columns > 0).then(|| {
                                rows.chunks_exact(columns)
                                    .flat_map(|row| dictionary.encode_selected(row, ids))
                                    .collect()
                            })
                        }
                        None => source_for_width(
                            rows,
                            source_columns,
                            columns,
                            width,
                            target.heads.as_deref(),
                            head_dim,
                        ),
                    };
                // The width the operation sees: the row, its selected heads,
                // or its selected features.
                let width = match &feature_view {
                    Some((_, ids)) => ids.len(),
                    None => target_width(target.heads.as_deref(), columns, head_dim)?,
                };
                let source_rows: Option<Vec<f32>> = match &target.source {
                    None => None,
                    Some(InterventionSource::Zero) => Some(vec![0.0; width]),
                    Some(InterventionSource::InlineVector { values }) => {
                        if values.len() != width {
                            return Err(ExperimentError::new(format!(
                                "intervention '{}': inline source has {} values but the \
                                 target has {width} columns",
                                target.intervention_id,
                                values.len()
                            )));
                        }
                        Some(values.clone())
                    }
                    Some(InterventionSource::CaptureFromCurrentRun { capture_id }) => {
                        // Cache the capture's own rows: interventions with
                        // different heads reduce them differently.
                        if !current_sources.contains_key(capture_id) {
                            let result = self.result.as_ref().ok_or_else(|| {
                                ExperimentError::new("result not initialized before prefill")
                            })?;
                            let captured = result
                                .captures
                                .iter()
                                .find(|c| {
                                    c.capture_id == *capture_id && c.input_id == input_id
                                        && c.site == site && c.layer == layer
                                })
                                .ok_or_else(|| {
                                    ExperimentError::new(format!(
                                        "intervention '{}': source capture '{capture_id}' has \
                                         not recorded a value at site {site}, layer {layer} at this point \
                                         (site/layer mismatch or execution order violation)",
                                        target.intervention_id
                                    ))
                                })?;
                            current_sources.insert(
                                capture_id.clone(),
                                (captured.rows.clone(), captured.columns),
                            );
                        }
                        let (rows, source_columns) = &current_sources[capture_id];
                        Some(fit_source(rows, *source_columns, width).ok_or_else(|| {
                            ExperimentError::new(format!(
                                "intervention '{}': source capture '{capture_id}' has \
                                     {source_columns} columns; the target has {width}",
                                target.intervention_id
                            ))
                        })?)
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
                        Some(
                            fit_source(&bundle.rows, bundle.columns, width).ok_or_else(|| {
                                ExperimentError::new(format!(
                                    "intervention '{}': cross-bundle source has {} columns; the \
                                     target has {width}",
                                    target.intervention_id, bundle.columns
                                ))
                            })?,
                        )
                    }
                    Some(
                        source @ (InterventionSource::VectorFile { .. }
                        | InterventionSource::Contrastive { .. }),
                    ) => {
                        let direction = self
                            .directions
                            .get(&(target.intervention_id.clone(), layer))
                            .ok_or_else(|| {
                                ExperimentError::new(format!(
                                    "intervention '{}': the {} direction for layer {layer} was \
                                     not resolved before execution",
                                    target.intervention_id,
                                    source.kind_name()
                                ))
                            })?;
                        if direction.len() != width {
                            return Err(ExperimentError::new(format!(
                                "intervention '{}': direction has {} values; the target at \
                                 site {site} has {width} columns",
                                target.intervention_id,
                                direction.len()
                            )));
                        }
                        Some(direction.clone())
                    }
                };
                // A single source row broadcasts; otherwise rows correspond
                // in selector order. Reject incompatible shapes before changing
                // any target row (copy_from_slice would panic, zip would truncate).
                if let Some(source) = &source_rows {
                    let matched = positions.len().checked_mul(width);
                    if width == 0 || (source.len() != width && Some(source.len()) != matched) {
                        return Err(ExperimentError::new(format!(
                            "intervention '{}': source has {} values; expected one row of {width} \
                             values or {} selected rows of {width} values",
                            target.intervention_id,
                            source.len(),
                            positions.len()
                        )));
                    }
                }
                let mut applied = false;
                for (row_index, &position) in positions.iter().enumerate() {
                    let source_row = source_rows.as_deref().map(|source| {
                        let offset = if source.len() == width {
                            0
                        } else {
                            row_index * width
                        };
                        &source[offset..offset + width]
                    });
                    let local = position - row_base;
                    let row_start = local * columns;
                    let row = &mut tensor.values_mut()[row_start..row_start + columns];
                    let original = self
                        .snapshots
                        .get(&(site, layer, position))
                        .map(|snapshot| gather_heads(snapshot, target.heads.as_deref(), head_dim));
                    match (&feature_view, target.heads.as_deref()) {
                        (Some((dictionary, ids)), _) => {
                            let before = dictionary.encode_selected(row, ids);
                            let mut view = before.clone();
                            apply_operation(
                                target, &mut view, source_row, None, site, layer, position,
                            )?;
                            dictionary.add_decoded(row, ids, &before, &view);
                        }
                        (None, None) => apply_operation(
                            target,
                            row,
                            source_row,
                            original.as_deref(),
                            site,
                            layer,
                            position,
                        )?,
                        (None, Some(heads)) => {
                            let mut view = gather_heads(row, Some(heads), head_dim);
                            apply_operation(
                                target,
                                &mut view,
                                source_row,
                                original.as_deref(),
                                site,
                                layer,
                                position,
                            )?;
                            scatter_heads(row, &view, heads, head_dim);
                        }
                    }
                    applied = true;
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
    source.kind_name().to_string()
}

/// SHA-256 over the little-endian bytes of `values`, as lowercase hex.
///
/// Hashes in fixed-size chunks through a stack buffer (one `update` per
/// chunk rather than per float) and hex-encodes through a lookup table; the
/// digest is identical to hashing each float's `to_le_bytes` in turn.
fn checksum_f32(values: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    const CHUNK: usize = 1024;
    let mut hasher = Sha256::new();
    let mut bytes = [0u8; CHUNK * 4];
    for chunk in values.chunks(CHUNK) {
        for (dst, value) in bytes.chunks_exact_mut(4).zip(chunk) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
        hasher.update(&bytes[..chunk.len() * 4]);
    }
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    out
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
        let site = SemanticHookSite::from_stage(stage);
        let phase_matches = |generated: bool| (phase == ExecutionPhase::Prefill) != generated;
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
        if let Some(prefix) = self.pending_prefix.take() {
            result.captures.extend(prefix.captures);
            result.summaries.extend(prefix.summaries);
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

    fn before_o_proj(
        &mut self,
        ctx: &LayerContext<'_>,
        heads: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_site(
            &ctx.execution,
            ctx.layer_index,
            SemanticHookSite::AttentionHeads,
            heads,
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
            if order.iter().enumerate().any(|(i, &original)| i != original) {
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

/// Apply `target`'s operation to one row (the whole row, or the gathered
/// columns of its heads). `source_row` and `original` have the row's width.
fn apply_operation(
    target: &InterventionTarget,
    row: &mut [f32],
    source_row: Option<&[f32]>,
    original: Option<&[f32]>,
    site: SemanticHookSite,
    layer: usize,
    position: usize,
) -> Result<(), ExperimentError> {
    match target.operation {
        InterventionOperation::Replace => {
            let source = source_row.ok_or_else(|| {
                ExperimentError::new(format!(
                    "intervention '{}': replace requires a source",
                    target.intervention_id
                ))
            })?;
            row.copy_from_slice(source);
        }
        InterventionOperation::Zero => {
            row.fill(0.0);
        }
        InterventionOperation::Scale { factor } => {
            for value in row.iter_mut() {
                *value *= factor;
            }
        }
        InterventionOperation::Interpolate { alpha } => {
            let source = source_row.ok_or_else(|| {
                ExperimentError::new(format!(
                    "intervention '{}': interpolate requires a source",
                    target.intervention_id
                ))
            })?;
            for (value, &src) in row.iter_mut().zip(source.iter()) {
                *value = (1.0 - alpha) * *value + alpha * src;
            }
        }
        InterventionOperation::AddDelta => {
            let source = source_row.ok_or_else(|| {
                ExperimentError::new(format!(
                    "intervention '{}': add-delta requires a source",
                    target.intervention_id
                ))
            })?;
            for (value, &src) in row.iter_mut().zip(source.iter()) {
                *value += src;
            }
        }
        InterventionOperation::Steer { alpha, normalize } => {
            let direction = source_row.ok_or_else(|| {
                ExperimentError::new(format!(
                    "intervention '{}': steer requires a direction",
                    target.intervention_id
                ))
            })?;
            crate::v05::steering::steer_row(row, direction, alpha, normalize).map_err(|error| {
                ExperimentError::new(format!(
                    "intervention '{}': {error}",
                    target.intervention_id
                ))
            })?;
        }
        InterventionOperation::AblateProjection => {
            let direction = source_row.ok_or_else(|| {
                ExperimentError::new(format!(
                    "intervention '{}': ablate-projection requires a direction",
                    target.intervention_id
                ))
            })?;
            crate::v05::steering::ablate_projection_row(row, direction).map_err(|error| {
                ExperimentError::new(format!(
                    "intervention '{}': {error}",
                    target.intervention_id
                ))
            })?;
        }
        InterventionOperation::RestoreOriginal => {
            let original = original.ok_or_else(|| {
                ExperimentError::new(format!(
                    "intervention '{}': no original snapshot exists at \
                     {} layer {layer} position {position}",
                    target.intervention_id, site
                ))
            })?;
            row.copy_from_slice(original);
        }
    }
    Ok(())
}

fn site_order(site: SemanticHookSite) -> usize {
    match site {
        SemanticHookSite::ResidualPreAttention => 0,
        SemanticHookSite::AttentionOutput => 1,
        SemanticHookSite::MlpOutput => 2,
        SemanticHookSite::ResidualPostMlp => 3,
        SemanticHookSite::FinalNormOutput => 4,
        SemanticHookSite::Logits => 5,
        SemanticHookSite::AttentionHeads => 6,
    }
}

/// A cross-bundle source preloaded by the driver (contract section 5).
#[derive(Debug, Clone)]
pub struct BundleSource {
    pub intervention_id: String,
    pub source: InterventionSource,
    pub rows: Vec<f32>,
    pub columns: usize,
    /// Deterministic provenance warnings for identity overrides that this
    /// source actually needed (recorded in the semantic manifest).
    pub warnings: Vec<String>,
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
        semantic_hash,
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
    let target_layers = if intervention.site.is_per_layer() {
        intervention.layers.resolve(n_layers)?
    } else {
        vec![0]
    };
    if target_layers != [*layer] {
        return Err(format!(
            "intervention '{}': source bundle layer {layer} does not match target layers {:?}",
            intervention.id, target_layers
        ));
    }
    let bundle = crate::v05::verify::load_bundle_for_source(bundle_path)?;
    if let Some(expected) = semantic_hash
        && bundle.semantic_hash != *expected
    {
        return Err(format!(
            "intervention '{}': source bundle '{}' has semantic hash {} but {expected} is pinned",
            intervention.id,
            bundle_path.display(),
            bundle.semantic_hash
        ));
    }
    let mut warnings = Vec::new();
    if semantic_hash.is_none() {
        // Without a pin the source is whatever bundle sits at the path, so
        // record which one it was: a later reproduce against a changed
        // source then differs here instead of only in its outputs.
        warnings.push(format!(
            "intervention '{}': source bundle not pinned; it resolved to semantic hash {} \
             (set semantic_hash to require it)",
            intervention.id, bundle.semantic_hash
        ));
    }
    if bundle.semantic_manifest.model.sha256 != target_model_sha {
        warnings.push(format!(
            "intervention '{}': allow_model_mismatch used; source bundle {} was run on model \
             {}, the target is {target_model_sha}",
            intervention.id, bundle.semantic_hash, bundle.semantic_manifest.model.sha256
        ));
    }
    if bundle.semantic_manifest.tokenizer.sha256 != target_tokenizer_sha {
        warnings.push(format!(
            "intervention '{}': allow_tokenizer_mismatch used; source bundle {} was run with \
             tokenizer {}, the target uses {target_tokenizer_sha}",
            intervention.id, bundle.semantic_hash, bundle.semantic_manifest.tokenizer.sha256
        ));
    }
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
        warnings,
    })
}

#[cfg(test)]
mod tests;
