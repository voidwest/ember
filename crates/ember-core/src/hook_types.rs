//! Records shared by the hook framework ([`crate::experiments`]) and the
//! artifact writers built on it: which stage fired, which dispatch path ran,
//! and the active experiment's provenance. `ember::artifact` re-exports them.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// The six semantic hook stages a capture record can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActivationStage {
    BeforeLayer,
    AfterAttention,
    AfterMlp,
    AfterLayer,
    BeforeLogits,
    AfterLogits,
}

impl fmt::Display for ActivationStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::BeforeLayer => "before-layer",
            Self::AfterAttention => "after-attention",
            Self::AfterMlp => "after-mlp",
            Self::AfterLayer => "after-layer",
            Self::BeforeLogits => "before-logits",
            Self::AfterLogits => "after-logits",
        };
        f.write_str(name)
    }
}

impl FromStr for ActivationStage {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "before-layer" => Ok(Self::BeforeLayer),
            "after-attention" => Ok(Self::AfterAttention),
            "after-mlp" => Ok(Self::AfterMlp),
            "after-layer" => Ok(Self::AfterLayer),
            "before-logits" => Ok(Self::BeforeLogits),
            "after-logits" => Ok(Self::AfterLogits),
            _ => Err(format!(
                "unknown stage '{value}'; expected one of: before-layer, after-attention, \
                 after-mlp, after-layer, before-logits, after-logits"
            )),
        }
    }
}

/// Kernel/dispatch path used for an evaluation. A single run can mix paths
/// (generic prefill, fast/workspace decode), so dispatch is recorded per
/// evaluation and per captured record, plus as run-level observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DispatchPath {
    /// Allocation-free/workspace-backed single-token decode.
    Fast,
    /// Plan-driven single-token decode (v0.4 execution plan interpreter).
    Planned,
    /// Generic tensor path (prefill and ineligible decode).
    Generic,
    /// Unknown or not recorded.
    Unknown,
}

impl fmt::Display for DispatchPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fast => f.write_str("fast"),
            Self::Planned => f.write_str("planned"),
            Self::Generic => f.write_str("generic"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

/// One (phase, dispatch path) observation; a run can mix paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchObservation {
    pub phase: String,
    pub dispatch: DispatchPath,
}

/// Active experiment provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestExperiment {
    pub name: String,
    pub arguments: serde_json::Value,
}
