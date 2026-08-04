//! v0.5 public semantic hook-site schema (`ember.hook.v1`).
//!
//! The six public sites map one-to-one onto the existing v0.4 execution
//! hook stages (docs/v05-research-contract.md section 1). The public
//! identifiers describe model semantics, not Rust implementation details.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Semantic hook schema version (`"v05-hook/1"`).
pub const HOOK_SCHEMA_VERSION: u32 = 1;

/// The six public semantic hook sites.
///
/// Serialized as kebab-case identifiers:
/// `residual-pre-attention`, `attention-output`, `mlp-output`,
/// `residual-post-mlp`, `final-norm-output`, `logits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SemanticHookSite {
    /// Residual stream entering the block, before the input RMS norm
    /// (v0.4 stage `before-layer`).
    ResidualPreAttention,
    /// Attention output projection result, before the attention residual
    /// add (v0.4 stage `after-attention`).
    AttentionOutput,
    /// MLP down-projection result, before the MLP residual add
    /// (v0.4 stage `after-mlp`).
    MlpOutput,
    /// Residual stream leaving the block, after both residual adds
    /// (v0.4 stage `after-layer`).
    ResidualPostMlp,
    /// Final RMS-norm output feeding the LM head (v0.4 stage
    /// `before-logits`). No layer component.
    FinalNormOutput,
    /// Raw LM-head logits (v0.4 stage `after-logits`). No layer component.
    Logits,
}

impl SemanticHookSite {
    /// All six sites in canonical order.
    pub const ALL: [SemanticHookSite; 6] = [
        SemanticHookSite::ResidualPreAttention,
        SemanticHookSite::AttentionOutput,
        SemanticHookSite::MlpOutput,
        SemanticHookSite::ResidualPostMlp,
        SemanticHookSite::FinalNormOutput,
        SemanticHookSite::Logits,
    ];

    /// The v0.4 execution stage id this site maps onto.
    pub const fn stage_id(self) -> &'static str {
        match self {
            SemanticHookSite::ResidualPreAttention => "before-layer",
            SemanticHookSite::AttentionOutput => "after-attention",
            SemanticHookSite::MlpOutput => "after-mlp",
            SemanticHookSite::ResidualPostMlp => "after-layer",
            SemanticHookSite::FinalNormOutput => "before-logits",
            SemanticHookSite::Logits => "after-logits",
        }
    }

    /// Whether this site carries a per-layer tensor.
    pub const fn is_per_layer(self) -> bool {
        !matches!(
            self,
            SemanticHookSite::FinalNormOutput | SemanticHookSite::Logits
        )
    }

    /// Parse a kebab-case public identifier.
    pub fn parse_id(id: &str) -> Result<SemanticHookSite, String> {
        match id {
            "residual-pre-attention" => Ok(SemanticHookSite::ResidualPreAttention),
            "attention-output" => Ok(SemanticHookSite::AttentionOutput),
            "mlp-output" => Ok(SemanticHookSite::MlpOutput),
            "residual-post-mlp" => Ok(SemanticHookSite::ResidualPostMlp),
            "final-norm-output" => Ok(SemanticHookSite::FinalNormOutput),
            "logits" => Ok(SemanticHookSite::Logits),
            other => Err(format!(
                "unknown semantic hook site '{other}' (expected one of: \
                 residual-pre-attention, attention-output, mlp-output, \
                 residual-post-mlp, final-norm-output, logits)"
            )),
        }
    }
}

impl fmt::Display for SemanticHookSite {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            SemanticHookSite::ResidualPreAttention => "residual-pre-attention",
            SemanticHookSite::AttentionOutput => "attention-output",
            SemanticHookSite::MlpOutput => "mlp-output",
            SemanticHookSite::ResidualPostMlp => "residual-post-mlp",
            SemanticHookSite::FinalNormOutput => "final-norm-output",
            SemanticHookSite::Logits => "logits",
        })
    }
}

/// Whether an intervention at a site writes pre-residual or post-residual
/// semantics (contract section 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InterventionSemantics {
    /// The site exposes a projection output before its residual add; an
    /// intervention lands before the add.
    PreResidual,
    /// The site exposes the residual stream itself; an intervention lands
    /// on the stream.
    ResidualStream,
    /// The site is the final norm output or raw logits; an intervention
    /// lands on the head input / logits.
    HeadBoundary,
}

/// Machine-readable descriptor for one stable hook site (contract section 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookSiteDescriptor {
    pub id: SemanticHookSite,
    pub schema_version: u32,
    pub description: &'static str,
    /// Tensor rank at the site: 2 for all v0.5 sites.
    pub rank: usize,
    /// Index of the token axis: 0 (rows) for all v0.5 sites.
    pub feature_axis: usize,
    pub intervention_semantics: InterventionSemantics,
}

impl HookSiteDescriptor {
    /// The frozen descriptor table, in canonical site order.
    pub const fn for_site(site: SemanticHookSite) -> HookSiteDescriptor {
        let (description, semantics) = match site {
            SemanticHookSite::ResidualPreAttention => (
                "residual stream entering the transformer block, before the input RMS norm",
                InterventionSemantics::ResidualStream,
            ),
            SemanticHookSite::AttentionOutput => (
                "attention output projection result, before the attention residual add",
                InterventionSemantics::PreResidual,
            ),
            SemanticHookSite::MlpOutput => (
                "MLP down-projection result, before the MLP residual add",
                InterventionSemantics::PreResidual,
            ),
            SemanticHookSite::ResidualPostMlp => (
                "residual stream leaving the block, after both residual adds",
                InterventionSemantics::ResidualStream,
            ),
            SemanticHookSite::FinalNormOutput => (
                "final RMS-norm output feeding the LM head",
                InterventionSemantics::HeadBoundary,
            ),
            SemanticHookSite::Logits => (
                "raw LM-head logits before sampling",
                InterventionSemantics::HeadBoundary,
            ),
        };
        HookSiteDescriptor {
            id: site,
            schema_version: HOOK_SCHEMA_VERSION,
            description,
            rank: 2,
            feature_axis: 0,
            intervention_semantics: semantics,
        }
    }

    /// The frozen table of all six descriptors.
    pub const fn all() -> [HookSiteDescriptor; 6] {
        [
            HookSiteDescriptor::for_site(SemanticHookSite::ResidualPreAttention),
            HookSiteDescriptor::for_site(SemanticHookSite::AttentionOutput),
            HookSiteDescriptor::for_site(SemanticHookSite::MlpOutput),
            HookSiteDescriptor::for_site(SemanticHookSite::ResidualPostMlp),
            HookSiteDescriptor::for_site(SemanticHookSite::FinalNormOutput),
            HookSiteDescriptor::for_site(SemanticHookSite::Logits),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_mapping_is_frozen() {
        assert_eq!(
            SemanticHookSite::ResidualPreAttention.stage_id(),
            "before-layer"
        );
        assert_eq!(
            SemanticHookSite::AttentionOutput.stage_id(),
            "after-attention"
        );
        assert_eq!(SemanticHookSite::MlpOutput.stage_id(), "after-mlp");
        assert_eq!(SemanticHookSite::ResidualPostMlp.stage_id(), "after-layer");
        assert_eq!(
            SemanticHookSite::FinalNormOutput.stage_id(),
            "before-logits"
        );
        assert_eq!(SemanticHookSite::Logits.stage_id(), "after-logits");
    }

    #[test]
    fn ids_round_trip() {
        for site in SemanticHookSite::ALL {
            let id = site.to_string();
            assert_eq!(SemanticHookSite::parse_id(&id).unwrap(), site);
        }
        assert!(SemanticHookSite::parse_id("after-layer").is_err());
        assert!(SemanticHookSite::parse_id("residual-post-attention").is_err());
    }

    #[test]
    fn descriptors_are_stable() {
        let table = HookSiteDescriptor::all();
        assert_eq!(table.len(), 6);
        let expected_ids = [
            "residual-pre-attention",
            "attention-output",
            "mlp-output",
            "residual-post-mlp",
            "final-norm-output",
            "logits",
        ];
        for (descriptor, expected) in table.iter().zip(expected_ids) {
            assert_eq!(descriptor.schema_version, HOOK_SCHEMA_VERSION);
            assert_eq!(descriptor.rank, 2);
            assert_eq!(descriptor.feature_axis, 0);
            assert_eq!(descriptor.id.to_string(), expected);
            assert_eq!(SemanticHookSite::parse_id(expected).unwrap(), descriptor.id);
        }
    }

    #[test]
    fn per_layer_classification() {
        assert!(SemanticHookSite::ResidualPreAttention.is_per_layer());
        assert!(SemanticHookSite::AttentionOutput.is_per_layer());
        assert!(SemanticHookSite::MlpOutput.is_per_layer());
        assert!(SemanticHookSite::ResidualPostMlp.is_per_layer());
        assert!(!SemanticHookSite::FinalNormOutput.is_per_layer());
        assert!(!SemanticHookSite::Logits.is_per_layer());
    }
}
