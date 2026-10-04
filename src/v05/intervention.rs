//! v0.5 intervention specifications (contract sections 3, 15).
//!
//! Interventions use the same semantic addressing model as captures:
//! site, layer selector, token selector, input selector. Operations are
//! narrow and explicit; sources are validated before execution and fail
//! closed on any incompatibility.

use crate::v05::capture::{InputSelector, LayerSelector};
use crate::v05::hook::SemanticHookSite;
use crate::v05::token_select::TokenSelector;
use serde::{Deserialize, Serialize};

/// The supported v0.5 intervention operations (contract section 3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "kebab-case",
    from = "StrictInterventionOperation"
)]
pub enum InterventionOperation {
    /// Replace the target rows with the source rows.
    Replace,
    /// Zero the target rows in place.
    Zero,
    /// Multiply the target rows by `factor`.
    Scale { factor: f32 },
    /// `target := (1 - alpha) * target + alpha * source`.
    Interpolate { alpha: f32 },
    /// `target := target + source`.
    AddDelta,
    /// Write the run's own pre-intervention snapshot back at the same site.
    RestoreOriginal,
    /// Steering: `target := target + alpha * c * direction`, where the
    /// direction is the source row and `c` comes from `normalize`.
    Steer {
        alpha: f32,
        #[serde(default)]
        normalize: SteerNormalization,
    },
    /// Remove the component along the source direction:
    /// `target := target - (target . u) u` with `u = d / |d|`.
    AblateProjection,
}

/// How a steering direction is scaled before `alpha` multiplies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SteerNormalization {
    /// Use the direction as given: `target += alpha * d`.
    #[default]
    None,
    /// Unit length: `target += alpha * d / |d|`.
    Unit,
    /// Unit length times the norm of the target row before the change:
    /// `target += alpha * |target| * d / |d|`, so `alpha` is a fraction of
    /// the row's own norm.
    MatchResidualNorm,
}
// Empty struct variants reject stray fields that Serde ignores for unit variants.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum StrictInterventionOperation {
    /// Replace the target rows with the source rows.
    Replace {},
    /// Zero the target rows in place.
    Zero {},
    /// Multiply the target rows by `factor`.
    Scale { factor: f32 },
    /// `target := (1 - alpha) * target + alpha * source`.
    Interpolate { alpha: f32 },
    /// `target := target + source`.
    AddDelta {},
    /// Write the run's own pre-intervention snapshot back at the same site.
    RestoreOriginal {},
    /// Steering along the source direction.
    Steer {
        alpha: f32,
        #[serde(default)]
        normalize: SteerNormalization,
    },
    /// Remove the component along the source direction.
    AblateProjection {},
}

impl From<StrictInterventionOperation> for InterventionOperation {
    fn from(value: StrictInterventionOperation) -> Self {
        match value {
            StrictInterventionOperation::Replace {} => Self::Replace,
            StrictInterventionOperation::Zero {} => Self::Zero,
            StrictInterventionOperation::Scale { factor } => Self::Scale { factor },
            StrictInterventionOperation::Interpolate { alpha } => Self::Interpolate { alpha },
            StrictInterventionOperation::AddDelta {} => Self::AddDelta,
            StrictInterventionOperation::RestoreOriginal {} => Self::RestoreOriginal,
            StrictInterventionOperation::Steer { alpha, normalize } => {
                Self::Steer { alpha, normalize }
            }
            StrictInterventionOperation::AblateProjection {} => Self::AblateProjection,
        }
    }
}

impl InterventionOperation {
    /// Whether this operation consumes a source tensor.
    pub const fn requires_source(self) -> bool {
        matches!(
            self,
            InterventionOperation::Replace
                | InterventionOperation::Interpolate { .. }
                | InterventionOperation::AddDelta
                | InterventionOperation::Steer { .. }
                | InterventionOperation::AblateProjection
        )
    }

    /// Whether this operation acts along a direction (its source is a
    /// direction, not a replacement row).
    pub const fn is_direction_op(self) -> bool {
        matches!(
            self,
            InterventionOperation::Steer { .. } | InterventionOperation::AblateProjection
        )
    }

    /// The operation's `alpha`, for operations that have one (the value a
    /// sweep's `alphas` sets).
    pub const fn alpha(self) -> Option<f32> {
        match self {
            InterventionOperation::Interpolate { alpha }
            | InterventionOperation::Steer { alpha, .. } => Some(alpha),
            _ => None,
        }
    }

    /// The kebab-case operation kind (matches the TOML/JSON `kind` tag).
    pub fn kind_name(self) -> &'static str {
        match self {
            InterventionOperation::Replace => "replace",
            InterventionOperation::Zero => "zero",
            InterventionOperation::Scale { .. } => "scale",
            InterventionOperation::Interpolate { .. } => "interpolate",
            InterventionOperation::AddDelta => "add-delta",
            InterventionOperation::RestoreOriginal => "restore-original",
            InterventionOperation::Steer { .. } => "steer",
            InterventionOperation::AblateProjection => "ablate-projection",
        }
    }
}

/// Intervention sources (contract section 5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "kebab-case",
    from = "StrictInterventionSource"
)]
pub enum InterventionSource {
    /// An inline row vector (`values`).
    InlineVector { values: Vec<f32> },
    /// A capture from the current run (same input).
    CaptureFromCurrentRun { capture_id: String },
    /// A capture from an existing verified bundle.
    CaptureFromBundle {
        bundle_path: std::path::PathBuf,
        capture_id: String,
        input_id: String,
        layer: usize,
    },
    /// The zero tensor (for `replace`).
    Zero,
    /// A direction read from a `.npy` or `.safetensors` file whose SHA-256
    /// is pinned (direction operations only).
    VectorFile {
        path: std::path::PathBuf,
        sha256: String,
        /// Tensor name inside a safetensors file (required when it holds
        /// more than one tensor; not allowed for `.npy`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tensor: Option<String>,
    },
    /// A contrastive direction computed in the run: the mean capture over
    /// `positive` prompts minus the mean over `negative` prompts, at the
    /// intervention's site and layer (direction operations only).
    Contrastive {
        positive: Vec<String>,
        negative: Vec<String>,
        /// Rows averaged per prompt (default `prompt-final`).
        #[serde(default = "default_contrastive_tokens")]
        tokens: TokenSelector,
    },
}
// Empty struct variants reject stray fields that Serde ignores for unit variants.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum StrictInterventionSource {
    /// An inline row vector (`values`).
    InlineVector { values: Vec<f32> },
    /// A capture from the current run (same input).
    CaptureFromCurrentRun { capture_id: String },
    /// A capture from an existing verified bundle.
    CaptureFromBundle {
        bundle_path: std::path::PathBuf,
        capture_id: String,
        input_id: String,
        layer: usize,
    },
    /// The zero tensor (for `replace`).
    Zero {},
    /// A direction read from a pinned file.
    VectorFile {
        path: std::path::PathBuf,
        sha256: String,
        #[serde(default)]
        tensor: Option<String>,
    },
    /// A contrastive direction computed in the run.
    Contrastive {
        positive: Vec<String>,
        negative: Vec<String>,
        #[serde(default = "default_contrastive_tokens")]
        tokens: TokenSelector,
    },
}

impl From<StrictInterventionSource> for InterventionSource {
    fn from(value: StrictInterventionSource) -> Self {
        match value {
            StrictInterventionSource::InlineVector { values } => Self::InlineVector { values },
            StrictInterventionSource::CaptureFromCurrentRun { capture_id } => {
                Self::CaptureFromCurrentRun { capture_id }
            }
            StrictInterventionSource::CaptureFromBundle {
                bundle_path,
                capture_id,
                input_id,
                layer,
            } => Self::CaptureFromBundle {
                bundle_path,
                capture_id,
                input_id,
                layer,
            },
            StrictInterventionSource::Zero {} => Self::Zero,
            StrictInterventionSource::VectorFile {
                path,
                sha256,
                tensor,
            } => Self::VectorFile {
                path,
                sha256,
                tensor,
            },
            StrictInterventionSource::Contrastive {
                positive,
                negative,
                tokens,
            } => Self::Contrastive {
                positive,
                negative,
                tokens,
            },
        }
    }
}

fn default_contrastive_tokens() -> TokenSelector {
    TokenSelector::PromptFinal
}

impl InterventionSource {
    /// Whether this source yields a direction the driver resolves before
    /// execution (a pinned file or a contrastive mean difference).
    pub const fn is_resolved_direction(&self) -> bool {
        matches!(
            self,
            InterventionSource::VectorFile { .. } | InterventionSource::Contrastive { .. }
        )
    }

    /// The kebab-case source kind (matches the TOML/JSON `kind` tag).
    pub fn kind_name(&self) -> &'static str {
        match self {
            InterventionSource::InlineVector { .. } => "inline-vector",
            InterventionSource::CaptureFromCurrentRun { .. } => "capture-from-current-run",
            InterventionSource::CaptureFromBundle { .. } => "capture-from-bundle",
            InterventionSource::Zero => "zero",
            InterventionSource::VectorFile { .. } => "vector-file",
            InterventionSource::Contrastive { .. } => "contrastive",
        }
    }
}

/// Shape/dtype compatibility policy for an intervention source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShapePolicy {
    /// Shape must match exactly; dtype conversion allowed only between f32
    /// and f16 (default).
    #[default]
    Strict,
    /// Explicit dtype-cast permission (still never allows rank/shape
    /// mismatch; recorded in provenance).
    AllowDtypeCast,
}

/// Expert override policy for cross-bundle sources.
///
/// The default is fully strict. An expert override is allowed only where
/// semantically defensible and is recorded prominently in provenance;
/// tensor shape incompatibility is never overridable.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityPolicy {
    /// Permit a model SHA mismatch between a source bundle and the target
    /// model (recorded in provenance).
    #[serde(default)]
    pub allow_model_mismatch: bool,
    /// Permit a tokenizer SHA mismatch (recorded in provenance).
    #[serde(default)]
    pub allow_tokenizer_mismatch: bool,
}

/// One declared intervention (contract section 5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterventionSpec {
    /// Unique intervention id within the experiment.
    pub id: String,
    /// Public semantic hook site.
    pub site: SemanticHookSite,
    /// Layers to intervene at.
    #[serde(default = "default_layers")]
    pub layers: LayerSelector,
    /// Token selector.
    pub tokens: TokenSelector,
    /// Which inputs this intervention applies to.
    #[serde(default = "default_inputs")]
    pub inputs: InputSelector,
    /// The operation.
    pub operation: InterventionOperation,
    /// The source; required by `replace`, `interpolate`, and `add-delta`.
    /// `zero`, `scale`, and `restore-original` do not consume a source.
    pub source: Option<InterventionSource>,
    /// Shape/dtype policy (default strict).
    #[serde(default)]
    pub shape_policy: ShapePolicy,
    /// Expert override policy for cross-bundle sources (default strict).
    #[serde(default)]
    pub compatibility: CompatibilityPolicy,
}

fn default_layers() -> LayerSelector {
    LayerSelector::All("all".to_string())
}

fn default_inputs() -> InputSelector {
    InputSelector::All("all".to_string())
}

impl InterventionSpec {
    /// Validate the operation/source combination and finite parameters.
    pub fn validate_self(&self) -> Result<(), String> {
        if let InterventionOperation::Scale { factor } = self.operation
            && !factor.is_finite()
        {
            return Err(format!(
                "intervention '{}': scale factor must be finite",
                self.id
            ));
        }
        if let InterventionOperation::Interpolate { alpha } = self.operation
            && !alpha.is_finite()
        {
            return Err(format!(
                "intervention '{}': interpolate alpha must be finite",
                self.id
            ));
        }
        if self.operation.requires_source() {
            let Some(source) = &self.source else {
                return Err(format!(
                    "intervention '{}': operation {:?} requires a source",
                    self.id, self.operation
                ));
            };
            if matches!(source, InterventionSource::CaptureFromCurrentRun { capture_id } if *capture_id == self.id)
            {
                return Err(format!(
                    "intervention '{}': source capture id must not equal the intervention id",
                    self.id
                ));
            }
        }
        if let Some(InterventionSource::InlineVector { values }) = &self.source
            && values.is_empty()
        {
            return Err(format!(
                "intervention '{}': inline vector must not be empty",
                self.id
            ));
        }
        if let InterventionOperation::Steer { alpha, .. } = self.operation
            && !alpha.is_finite()
        {
            return Err(format!(
                "intervention '{}': steer alpha must be finite",
                self.id
            ));
        }
        if let Some(source) = &self.source {
            let direction_source = source.is_resolved_direction();
            if self.operation.is_direction_op()
                && !direction_source
                && !matches!(source, InterventionSource::InlineVector { .. })
            {
                return Err(format!(
                    "intervention '{}': {} takes a direction: use an inline-vector, \
                     vector-file or contrastive source, not {}",
                    self.id,
                    self.operation.kind_name(),
                    source.kind_name()
                ));
            }
            if direction_source && !self.operation.is_direction_op() {
                return Err(format!(
                    "intervention '{}': a {} source is a direction and needs a direction \
                     operation (steer or ablate-projection), not {}",
                    self.id,
                    source.kind_name(),
                    self.operation.kind_name()
                ));
            }
            if let Some(values) = match source {
                InterventionSource::InlineVector { values } => Some(values),
                _ => None,
            } && values.iter().any(|value| !value.is_finite())
            {
                return Err(format!(
                    "intervention '{}': inline vector values must be finite",
                    self.id
                ));
            }
        }
        match &self.source {
            Some(InterventionSource::VectorFile {
                path,
                sha256,
                tensor,
            }) => {
                if path.as_os_str().is_empty() {
                    return Err(format!(
                        "intervention '{}': vector-file path must not be empty",
                        self.id
                    ));
                }
                if sha256.len() != 64
                    || !sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                {
                    return Err(format!(
                        "intervention '{}': vector-file sha256 must be 64 lowercase hex \
                         characters (the file's SHA-256 is part of the experiment's identity)",
                        self.id
                    ));
                }
                if tensor.as_deref().is_some_and(str::is_empty) {
                    return Err(format!(
                        "intervention '{}': vector-file tensor name must not be empty",
                        self.id
                    ));
                }
            }
            Some(InterventionSource::Contrastive {
                positive,
                negative,
                tokens,
            }) => {
                for (name, prompts) in [("positive", positive), ("negative", negative)] {
                    if prompts.is_empty() {
                        return Err(format!(
                            "intervention '{}': contrastive {name} prompts must not be empty",
                            self.id
                        ));
                    }
                    if prompts.iter().any(|prompt| prompt.is_empty()) {
                        return Err(format!(
                            "intervention '{}': contrastive {name} prompts must be non-empty \
                             text",
                            self.id
                        ));
                    }
                }
                if tokens.is_generated() {
                    return Err(format!(
                        "intervention '{}': contrastive prompts are only prefilled; their \
                         token selector cannot be generated-step",
                        self.id
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replace_spec() -> InterventionSpec {
        InterventionSpec {
            id: "iv-1".into(),
            site: SemanticHookSite::AttentionOutput,
            layers: LayerSelector::All("all".into()),
            tokens: TokenSelector::PromptFinal,
            inputs: InputSelector::All("all".into()),
            operation: InterventionOperation::Replace,
            source: Some(InterventionSource::CaptureFromCurrentRun {
                capture_id: "cap-1".into(),
            }),
            shape_policy: ShapePolicy::Strict,
            compatibility: CompatibilityPolicy::default(),
        }
    }

    #[test]
    fn operation_source_matrix() {
        assert!(replace_spec().validate_self().is_ok());
        let mut no_source = replace_spec();
        no_source.source = None;
        assert!(no_source.validate_self().is_err());
        let mut zero = replace_spec();
        zero.operation = InterventionOperation::Zero;
        zero.source = None;
        assert!(zero.validate_self().is_ok());
        let mut restore = replace_spec();
        restore.operation = InterventionOperation::RestoreOriginal;
        restore.source = None;
        assert!(restore.validate_self().is_ok());
        let mut self_source = replace_spec();
        self_source.id = "cap-1".into();
        assert!(self_source.validate_self().is_err());
    }

    #[test]
    fn finite_parameter_validation() {
        let mut scale = replace_spec();
        scale.operation = InterventionOperation::Scale { factor: f32::NAN };
        assert!(scale.validate_self().is_err());
        let mut interp = replace_spec();
        interp.operation = InterventionOperation::Interpolate {
            alpha: f32::INFINITY,
        };
        assert!(interp.validate_self().is_err());
        interp.operation = InterventionOperation::Interpolate { alpha: 0.5 };
        interp.source = None;
        assert!(interp.validate_self().is_err()); // interpolate requires source
        interp.source = Some(InterventionSource::Zero);
        assert!(interp.validate_self().is_ok());
    }
}
