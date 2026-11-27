use super::*;

// ---------------------------------------------------------------------------
// spec building (same raw TOML form a user would write, resolved + validated)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RunKind {
    Baseline,
    Intervention,
    Restore,
}

/// Build the raw (user-form) spec for one run kind and resolve it through
/// the standard `RawExperimentSpec::resolve()` gate.
pub(super) fn build_and_resolve_spec(
    cfg: &RunConfig,
    kind: RunKind,
    output_dir: &str,
) -> Result<(String, ember::v05::spec::ExperimentSpecV1), String> {
    let tokens = match cfg.token {
        TokenKind::PromptFinal => TokenSelector::PromptFinal,
        TokenKind::MatchedSpan => TokenSelector::MatchedTextSpan {
            text: cfg.span_text.clone(),
            occurrence: 0,
            subtoken_selection: SubtokenSelection::All,
            normalization: TextNormalization::None,
        },
    };
    let layers_for = |layer: Option<usize>| match layer {
        Some(layer) => LayerSelector::List(vec![layer]),
        None => LayerSelector::All("all".to_string()),
    };

    // The source capture layer: the source layer for capture sources, the
    // intervention layer otherwise (so baseline and intervention bundles
    // carry the same capture and compare cleanly).
    let capture_layer = if cfg.source == SourceKind::Capture && cfg.operation.requires_source() {
        cfg.source_layer.or(cfg.layer)
    } else {
        cfg.layer
    };

    let capture = CaptureSpec {
        id: "cap-src".to_string(),
        site: cfg.site,
        layers: layers_for(capture_layer),
        tokens: tokens.clone(),
        inputs: InputSelector::All("all".to_string()),
        storage: CaptureStorage::SelectedRows,
        dtype: CaptureDType::F32,
        heads: None,
        features: None,
    };
    // One selected activation row per layer gives the native workbench a
    // scientifically meaningful residual-stream trace without retaining a
    // second in-memory copy. The verified bundle owns the tensors; after the
    // pair completes, `compare_bundles` reduces them to compact metrics.
    let layer_trace = CaptureSpec {
        id: GUI_LAYER_TRACE_CAPTURE_ID.to_string(),
        site: SemanticHookSite::ResidualPostMlp,
        layers: LayerSelector::All("all".to_string()),
        tokens: tokens.clone(),
        inputs: InputSelector::All("all".to_string()),
        storage: CaptureStorage::SelectedRows,
        dtype: CaptureDType::F32,
        heads: None,
        features: None,
    };

    let operation = match cfg.operation {
        GuiOperation::Replace => InterventionOperation::Replace,
        GuiOperation::Zero => InterventionOperation::Zero,
        GuiOperation::Scale => InterventionOperation::Scale { factor: cfg.factor },
        GuiOperation::Interpolate => InterventionOperation::Interpolate { alpha: cfg.alpha },
        GuiOperation::AddDelta => InterventionOperation::AddDelta,
        GuiOperation::RestoreOriginal => InterventionOperation::RestoreOriginal,
    };
    let source = match (&cfg.operation, &cfg.source) {
        (operation, _) if operation.is_restore() || !operation.requires_source() => None,
        (_, SourceKind::Zero) => Some(InterventionSource::Zero),
        (_, SourceKind::Capture) => Some(InterventionSource::CaptureFromCurrentRun {
            capture_id: "cap-src".to_string(),
        }),
    };
    let intervention = InterventionSpec {
        id: "iv-1".to_string(),
        site: cfg.site,
        layers: layers_for(cfg.layer),
        tokens: tokens.clone(),
        inputs: InputSelector::All("all".to_string()),
        operation,
        source,
        shape_policy: ShapePolicy::Strict,
        compatibility: CompatibilityPolicy::default(),
        heads: None,
        features: None,
    };
    let restore = InterventionSpec {
        id: "restore-1".to_string(),
        site: cfg.site,
        layers: layers_for(cfg.layer),
        tokens,
        inputs: InputSelector::All("all".to_string()),
        operation: InterventionOperation::RestoreOriginal,
        source: None,
        shape_policy: ShapePolicy::Strict,
        compatibility: CompatibilityPolicy::default(),
        heads: None,
        features: None,
    };

    let (name, captures, interventions) = match kind {
        RunKind::Baseline => ("gui-baseline", vec![capture, layer_trace], Vec::new()),
        RunKind::Intervention => (
            "gui-intervention",
            vec![capture, layer_trace],
            vec![intervention],
        ),
        RunKind::Restore => ("gui-restore", Vec::new(), vec![restore]),
    };

    let raw = raw_spec(
        name,
        &cfg.model_path,
        cfg.execution.name(),
        cfg.max_new_tokens,
        &cfg.prompt,
        captures,
        interventions,
        PathBuf::from(output_dir),
        false,
    );
    let spec_text = toml::to_string_pretty(&raw)
        .map_err(|error| format!("cannot serialize the experiment specification: {error}"))?;
    let resolved = raw.resolve().map_err(|error| error.to_string())?;
    Ok((spec_text, resolved))
}

// ---------------------------------------------------------------------------
// the spec view: form -> TOML, and TOML -> form where the form can hold it
// ---------------------------------------------------------------------------

/// The output directory the spec view shows. A run writes to its own
/// timestamped directory instead.
pub(crate) const SPEC_VIEW_OUTPUT_DIR: &str = "runs/gui/intervention";

/// The intervention run's spec for `cfg`, as TOML: what the console runs,
/// with [`SPEC_VIEW_OUTPUT_DIR`] as the output directory.
#[cfg_attr(
    not(feature = "gui"),
    allow(dead_code, reason = "only the `gui` feature uses it")
)]
pub(crate) fn intervention_spec_text(cfg: &RunConfig) -> Result<String, String> {
    build_and_resolve_spec(cfg, RunKind::Intervention, SPEC_VIEW_OUTPUT_DIR).map(|(text, _)| text)
}

/// The fields of the console's form that an intervention spec sets. `None`
/// means the spec does not set that field, so the form keeps its value.
#[cfg_attr(
    not(feature = "gui"),
    allow(dead_code, reason = "only the `gui` feature uses it")
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpecFields {
    pub model_path: String,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub execution: String,
    /// The v0.4 hook stage id, as the form names sites.
    pub site: String,
    pub layer: Option<usize>,
    pub operation: String,
    /// The scale factor or the interpolate alpha.
    pub value: Option<f32>,
    /// `capture` or `zero`, for the operations that take a source.
    pub source: Option<String>,
    pub source_layer: Option<usize>,
    /// `prompt-final` or `matched-span`.
    pub token: String,
    pub span: Option<String>,
}

/// Read the form's fields from an edited spec. This fails, with the reason,
/// when the spec has a shape that the form cannot hold: other than one
/// input and one intervention, or another token selector or source. Whether
/// every other field matches is checked by [`check_same_experiment`].
#[cfg_attr(
    not(feature = "gui"),
    allow(dead_code, reason = "only the `gui` feature uses it")
)]
pub(crate) fn spec_fields(text: &str) -> Result<SpecFields, String> {
    let spec = RawExperimentSpec::from_toml_str(text)
        .and_then(|raw| raw.resolve())
        .map_err(|error| error.to_string())?;
    let [input] = spec.inputs.as_slice() else {
        return Err(format!(
            "The console runs one prompt. This spec has {} inputs.",
            spec.inputs.len()
        ));
    };
    let [intervention] = spec.interventions.as_slice() else {
        return Err(format!(
            "The console runs one intervention. This spec has {}.",
            spec.interventions.len()
        ));
    };
    let one_layer = |selector: &LayerSelector, what: &str| match selector {
        LayerSelector::List(layers) if layers.len() == 1 => Ok(layers[0]),
        _ => Err(format!(
            "The console sets one {what} layer. Write layers = [L]."
        )),
    };
    let site = intervention.site;
    let layer = if site.is_per_layer() {
        Some(one_layer(&intervention.layers, "intervention")?)
    } else {
        None
    };
    let operation = intervention.operation.kind_name().to_string();
    if matches!(
        GuiOperation::parse(&operation)?,
        GuiOperation::RestoreOriginal
    ) {
        return Err(
            "The console's operations are replace, zero, scale, interpolate and \
                    add-delta. This spec uses restore-original."
                .to_string(),
        );
    }
    let value = match intervention.operation {
        InterventionOperation::Scale { factor } => Some(factor),
        InterventionOperation::Interpolate { alpha } => Some(alpha),
        _ => None,
    };
    let (source, source_layer) = match &intervention.source {
        None => (None, None),
        Some(InterventionSource::Zero) => (Some("zero".to_string()), None),
        Some(InterventionSource::CaptureFromCurrentRun { capture_id })
            if capture_id == "cap-src" =>
        {
            let capture = spec
                .captures
                .iter()
                .find(|capture| capture.id == "cap-src")
                .ok_or("The spec has no source capture cap-src.")?;
            let source_layer = if site.is_per_layer() {
                Some(one_layer(&capture.layers, "source capture")?)
            } else {
                None
            };
            (Some("capture".to_string()), source_layer)
        }
        Some(other) => {
            return Err(format!(
                "The console's source is the zero tensor or its own capture cap-src. This spec \
                 uses {}.",
                other.kind_name()
            ))
        }
    };
    let (token, span) = match &intervention.tokens {
        TokenSelector::PromptFinal => ("prompt-final".to_string(), None),
        TokenSelector::MatchedTextSpan { text, .. } => {
            ("matched-span".to_string(), Some(text.clone()))
        }
        other => {
            let kind = serde_json::to_value(other)
                .ok()
                .and_then(|value| value["kind"].as_str().map(str::to_string))
                .unwrap_or_default();
            return Err(format!(
                "The console selects the final prompt token or a matched text span. This spec \
                 uses {kind}."
            ));
        }
    };
    Ok(SpecFields {
        model_path: spec.model.path.to_string_lossy().into_owned(),
        prompt: input.text.clone(),
        max_new_tokens: spec.generation.max_new_tokens,
        execution: spec.execution.mode.name().to_string(),
        site: site.stage_id().to_string(),
        layer,
        operation,
        value,
        source,
        source_layer,
        token,
        span,
    })
}

/// Check that an edited spec runs the same experiment as the one the form
/// generates: equal once resolved, apart from the name, the description and
/// the recorded defaults, which the console sets itself. The error names the
/// first field that differs.
#[cfg_attr(
    not(feature = "gui"),
    allow(dead_code, reason = "only the `gui` feature uses it")
)]
pub(crate) fn check_same_experiment(edited: &str, generated: &str) -> Result<(), String> {
    let normalize = |text: &str| -> Result<serde_json::Value, String> {
        let mut spec = RawExperimentSpec::from_toml_str(text)
            .and_then(|raw| raw.resolve())
            .map_err(|error| error.to_string())?;
        spec.experiment.name.clear();
        spec.experiment.description.clear();
        spec.defaults.clear();
        serde_json::to_value(&spec).map_err(|error| error.to_string())
    };
    let (edited, generated) = (normalize(edited)?, normalize(generated)?);
    match first_difference(&edited, &generated, "") {
        None => Ok(()),
        Some(path) => Err(format!(
            "The form cannot hold `{path}` as written. Save the spec to a file and run it \
             with `ember experiment run`."
        )),
    }
}

/// The dotted path of the first place where `a` and `b` differ.
fn first_difference(a: &serde_json::Value, b: &serde_json::Value, path: &str) -> Option<String> {
    use serde_json::Value;
    let join = |key: &str| {
        if path.is_empty() {
            key.to_string()
        } else {
            format!("{path}.{key}")
        }
    };
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
            keys.into_iter()
                .find_map(|key| match (a.get(key), b.get(key)) {
                    (Some(x), Some(y)) => first_difference(x, y, &join(key)),
                    _ => Some(join(key)),
                })
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => a
            .iter()
            .zip(b)
            .enumerate()
            .find_map(|(index, (x, y))| first_difference(x, y, &format!("{path}[{index}]"))),
        _ if a == b => None,
        _ => Some(if path.is_empty() {
            "spec".into()
        } else {
            path.to_string()
        }),
    }
}
