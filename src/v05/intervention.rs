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
#[serde(tag = "kind", rename_all = "kebab-case")]
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
}

impl InterventionOperation {
    /// Whether this operation consumes a source tensor.
    pub const fn requires_source(self) -> bool {
        matches!(
            self,
            InterventionOperation::Replace
                | InterventionOperation::Interpolate { .. }
                | InterventionOperation::AddDelta
        )
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
        }
    }

    /// Whether this operation needs the pre-intervention snapshot.
    pub const fn uses_snapshot(self) -> bool {
        matches!(
            self,
            InterventionOperation::RestoreOriginal
                | InterventionOperation::Scale { .. }
                | InterventionOperation::Interpolate { .. }
                | InterventionOperation::AddDelta
        )
    }
}

/// Intervention sources (contract section 5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
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
    /// The source; required unless the operation is `zero` (which may omit
    /// it or use `source = { kind = "zero" }`) or `restore-original`.
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
        if let InterventionOperation::Scale { factor } = self.operation {
            if !factor.is_finite() {
                return Err(format!(
                    "intervention '{}': scale factor must be finite",
                    self.id
                ));
            }
        }
        if let InterventionOperation::Interpolate { alpha } = self.operation {
            if !alpha.is_finite() {
                return Err(format!(
                    "intervention '{}': interpolate alpha must be finite",
                    self.id
                ));
            }
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
        if let Some(InterventionSource::InlineVector { values }) = &self.source {
            if values.is_empty() {
                return Err(format!(
                    "intervention '{}': inline vector must not be empty",
                    self.id
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v05::token_select::SubtokenSelection;

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

    #[test]
    fn subtoken_selection_parses() {
        let value: SubtokenSelection = serde_json::from_str("\"first\"").unwrap();
        assert_eq!(value, SubtokenSelection::First);
    }
}
