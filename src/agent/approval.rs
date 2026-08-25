//! Approval gating (Track H, Phase 2): risk-aware execution policy.
//!
//! Phase 1 recorded each tool's [`ToolEffect`](crate::agent::ToolEffect)
//! but never gated on it. This module closes that gap with a small,
//! embeddable policy seam: the session consults
//! [`ApprovalPolicy::approve`] between validation and execution, and a
//! denial becomes the same structured, traced, model-visible rejection
//! as any other (kind `denied_by_policy`), so the model can adapt and
//! researchers can audit exactly what was blocked and why.
//!
//! Deliberately NOT an interactive/security framework: there is no user
//! prompt loop, no sandboxing, no capability tokens. Hosts embed their
//! own gate via [`ApprovalPolicy::custom`].

use std::sync::Arc;

use super::schema::{ToolEffect, ToolSchema, ValidatedArguments};

/// The decision for one validated call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    Approve,
    Deny { reason: String },
}

/// Host-supplied gate. Called only AFTER schema validation succeeds, so
/// implementations can rely on well-typed arguments.
pub trait ApprovalGate: Send + Sync {
    fn approve(&self, schema: &ToolSchema, args: &ValidatedArguments) -> ApprovalDecision;
}

/// Built-in policies.
#[derive(Clone, Default)]
pub enum ApprovalPolicy {
    /// Execute everything (Phase 1 behavior; safe because Phase 1/2
    /// built-ins are deterministic and read/local-write only).
    Auto,
    /// Auto-approve `ReadOnly` and `LocalWrite`; deny
    /// `ExternalSideEffect`. The default going forward: unknown tools
    /// that declare external effects fail closed.
    #[default]
    DenyExternalSideEffect,
    /// Delegate to a host-supplied gate.
    Custom(Arc<dyn ApprovalGate>),
}

impl std::fmt::Debug for ApprovalPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApprovalPolicy::Auto => write!(f, "Auto"),
            ApprovalPolicy::DenyExternalSideEffect => write!(f, "DenyExternalSideEffect"),
            ApprovalPolicy::Custom(_) => write!(f, "Custom(<gate>)"),
        }
    }
}

impl ApprovalGate for ApprovalPolicy {
    fn approve(&self, schema: &ToolSchema, args: &ValidatedArguments) -> ApprovalDecision {
        match self {
            ApprovalPolicy::Auto => ApprovalDecision::Approve,
            ApprovalPolicy::DenyExternalSideEffect => {
                if schema.effect == ToolEffect::ExternalSideEffect {
                    ApprovalDecision::Deny {
                        reason: format!(
                            "tool `{}` declares {} effects",
                            schema.name,
                            ToolEffect::ExternalSideEffect.as_str()
                        ),
                    }
                } else {
                    ApprovalDecision::Approve
                }
            }
            ApprovalPolicy::Custom(gate) => gate.approve(schema, args),
        }
    }
}

impl ApprovalPolicy {
    pub fn custom(gate: impl ApprovalGate + 'static) -> Self {
        ApprovalPolicy::Custom(Arc::new(gate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::schema::{JsonType, ParamSchema};

    fn schema_with(effect: ToolEffect) -> ToolSchema {
        ToolSchema::new("probe", "d")
            .effect(effect)
            .param(ParamSchema::new("x", JsonType::String).required())
    }

    fn args(schema: &ToolSchema) -> ValidatedArguments {
        ValidatedArguments::parse(schema, r#"{"x":"v"}"#).unwrap()
    }

    #[test]
    fn auto_approves_everything() {
        let policy = ApprovalPolicy::Auto;
        assert_eq!(
            policy.approve(
                &schema_with(ToolEffect::ExternalSideEffect),
                &args(&schema_with(ToolEffect::ExternalSideEffect))
            ),
            ApprovalDecision::Approve
        );
    }

    #[test]
    fn deny_external_fails_closed_on_declared_effects_only() {
        let policy = ApprovalPolicy::DenyExternalSideEffect;
        let ext = schema_with(ToolEffect::ExternalSideEffect);
        assert!(matches!(
            policy.approve(&ext, &args(&ext)),
            ApprovalDecision::Deny { .. }
        ));
        let ro = schema_with(ToolEffect::ReadOnly);
        assert_eq!(policy.approve(&ro, &args(&ro)), ApprovalDecision::Approve);
        let lw = schema_with(ToolEffect::LocalWrite);
        assert_eq!(policy.approve(&lw, &args(&lw)), ApprovalDecision::Approve);
    }

    struct DenyNamed(&'static str);

    impl ApprovalGate for DenyNamed {
        fn approve(&self, schema: &ToolSchema, _args: &ValidatedArguments) -> ApprovalDecision {
            if schema.name == self.0 {
                ApprovalDecision::Deny {
                    reason: "blocked by test policy".to_string(),
                }
            } else {
                ApprovalDecision::Approve
            }
        }
    }

    #[test]
    fn custom_gate_controls_individual_tools() {
        let policy = ApprovalPolicy::custom(DenyNamed("probe"));
        let s = schema_with(ToolEffect::ReadOnly);
        assert!(matches!(
            policy.approve(&s, &args(&s)),
            ApprovalDecision::Deny { .. }
        ));
    }
}
