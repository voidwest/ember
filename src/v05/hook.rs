//! v0.5 public semantic hook-site schema (`ember.hook.v1`).
//!
//! The seven public sites map one-to-one onto v0.4 execution hook stages
//! (docs/v05-research-contract.md section 1). The six original sites are
//! unchanged; `attention-heads` was added within `ember.hook.v1` as an
//! addition that leaves every plan and bundle without it unchanged
//! (docs/adr/0002-attention-head-sites.md). The public identifiers describe
//! model semantics, not Rust implementation details.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Semantic hook schema version (`"v05-hook/1"`).
pub const HOOK_SCHEMA_VERSION: u32 = 1;

/// The public semantic hook sites.
///
/// Serialized as kebab-case identifiers:
/// `residual-pre-attention`, `attention-heads`, `attention-output`,
/// `mlp-output`, `residual-post-mlp`, `final-norm-output`, `logits`.
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
    /// Per-head attention outputs, concatenated, before the attention
    /// output projection (v0.4 stage `before-o-proj`): `[seq, n_heads *
    /// head_dim]`, head `h` in columns `h * head_dim..(h + 1) * head_dim`.
    AttentionHeads,
}

impl SemanticHookSite {
    /// All sites in canonical (execution) order.
    pub const ALL: [SemanticHookSite; 7] = [
        SemanticHookSite::ResidualPreAttention,
        SemanticHookSite::AttentionHeads,
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
            SemanticHookSite::AttentionHeads => "before-o-proj",
        }
    }

    /// The v0.4 execution stage this site maps onto.
    pub const fn stage(self) -> crate::artifact::ActivationStage {
        use crate::artifact::ActivationStage;
        match self {
            SemanticHookSite::ResidualPreAttention => ActivationStage::BeforeLayer,
            SemanticHookSite::AttentionHeads => ActivationStage::BeforeOProj,
            SemanticHookSite::AttentionOutput => ActivationStage::AfterAttention,
            SemanticHookSite::MlpOutput => ActivationStage::AfterMlp,
            SemanticHookSite::ResidualPostMlp => ActivationStage::AfterLayer,
            SemanticHookSite::FinalNormOutput => ActivationStage::BeforeLogits,
            SemanticHookSite::Logits => ActivationStage::AfterLogits,
        }
    }

    /// The site a v0.4 execution stage maps onto.
    pub const fn from_stage(stage: crate::artifact::ActivationStage) -> SemanticHookSite {
        use crate::artifact::ActivationStage;
        match stage {
            ActivationStage::BeforeLayer => SemanticHookSite::ResidualPreAttention,
            ActivationStage::BeforeOProj => SemanticHookSite::AttentionHeads,
            ActivationStage::AfterAttention => SemanticHookSite::AttentionOutput,
            ActivationStage::AfterMlp => SemanticHookSite::MlpOutput,
            ActivationStage::AfterLayer => SemanticHookSite::ResidualPostMlp,
            ActivationStage::BeforeLogits => SemanticHookSite::FinalNormOutput,
            ActivationStage::AfterLogits => SemanticHookSite::Logits,
        }
    }

    /// Whether this site carries a per-layer tensor.
    pub const fn is_per_layer(self) -> bool {
        !matches!(
            self,
            SemanticHookSite::FinalNormOutput | SemanticHookSite::Logits
        )
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
            SemanticHookSite::AttentionHeads => "attention-heads",
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
    /// The site is the input of a projection; the projection reads the
    /// changed value.
    PreProjection,
}

/// Machine-readable descriptor for one stable hook site (contract section 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookSiteDescriptor {
    pub id: SemanticHookSite,
    pub schema_version: u32,
    pub description: &'static str,
    /// Tensor rank at the site: 2 for every site.
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
            SemanticHookSite::AttentionHeads => (
                "per-head attention outputs, concatenated, before the attention output projection",
                InterventionSemantics::PreProjection,
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

    /// The table of all seven descriptors, in canonical site order.
    pub const fn all() -> [HookSiteDescriptor; 7] {
        [
            HookSiteDescriptor::for_site(SemanticHookSite::ResidualPreAttention),
            HookSiteDescriptor::for_site(SemanticHookSite::AttentionHeads),
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
        assert_eq!(SemanticHookSite::AttentionHeads.stage_id(), "before-o-proj");
        for site in SemanticHookSite::ALL {
            assert_eq!(SemanticHookSite::from_stage(site.stage()), site);
            assert_eq!(site.stage().to_string(), site.stage_id());
        }
    }

    #[test]
    fn descriptors_are_stable() {
        let table = HookSiteDescriptor::all();
        assert_eq!(table.len(), 7);
        let expected_ids = [
            "residual-pre-attention",
            "attention-heads",
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
            assert_eq!(
                serde_json::from_value::<SemanticHookSite>(expected.into()).unwrap(),
                descriptor.id
            );
        }
        // Stage ids and near-miss names are not semantic site ids.
        assert!(serde_json::from_value::<SemanticHookSite>("after-layer".into()).is_err());
        assert!(
            serde_json::from_value::<SemanticHookSite>("residual-post-attention".into()).is_err()
        );
    }

    #[test]
    fn per_layer_classification() {
        assert!(SemanticHookSite::ResidualPreAttention.is_per_layer());
        assert!(SemanticHookSite::AttentionOutput.is_per_layer());
        assert!(SemanticHookSite::MlpOutput.is_per_layer());
        assert!(SemanticHookSite::ResidualPostMlp.is_per_layer());
        assert!(SemanticHookSite::AttentionHeads.is_per_layer());
        assert!(!SemanticHookSite::FinalNormOutput.is_per_layer());
        assert!(!SemanticHookSite::Logits.is_per_layer());
    }
}
