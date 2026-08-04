//! v0.5 capture specifications (contract sections 8, 10).
//!
//! Captures are declared in the experiment specification, resolved into a
//! capture plan before inference, and fired at the six public semantic
//! hook sites during execution.

use crate::v05::hook::SemanticHookSite;
use crate::v05::token_select::TokenSelector;
use serde::{Deserialize, Serialize};

/// Layer selector (contract section 4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LayerSelector {
    /// The string `"all"`.
    All(String),
    /// An explicit list of layer indices.
    List(Vec<usize>),
    /// A `{ start, end, step }` range (end exclusive).
    Range(LayerRange),
}

/// Range form of a layer selector (end exclusive).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerRange {
    pub start: usize,
    pub end: usize,
    #[serde(default = "default_step")]
    pub step: usize,
}

fn default_step() -> usize {
    1
}

impl LayerSelector {
    /// Resolve against the model's layer count into a sorted, deduplicated
    /// list of in-range layer indices.
    pub fn resolve(&self, n_layers: usize) -> Result<Vec<usize>, String> {
        if n_layers == 0 {
            return Err("layer selector: model has no layers".into());
        }
        let mut layers: Vec<usize> = match self {
            LayerSelector::All(value) => {
                if value != "all" {
                    return Err(format!(
                        "layer selector: expected the string \"all\", found {value:?}"
                    ));
                }
                (0..n_layers).collect()
            }
            LayerSelector::List(list) => list.clone(),
            LayerSelector::Range(range) => {
                if range.step == 0 {
                    return Err("layer selector: range step must be >= 1".into());
                }
                if range.start >= range.end {
                    return Err(format!(
                        "layer selector: range start {} >= end {}",
                        range.start, range.end
                    ));
                }
                (range.start..range.end).step_by(range.step).collect()
            }
        };
        for &layer in &layers {
            if layer >= n_layers {
                return Err(format!(
                    "layer selector: layer {layer} is out of range for a {n_layers}-layer model"
                ));
            }
        }
        layers.sort_unstable();
        layers.dedup();
        Ok(layers)
    }
}

/// Input selector: which experiment inputs a capture or intervention
/// applies to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputSelector {
    /// The string `"all"`.
    All(String),
    /// An explicit list of input ids.
    List(Vec<String>),
}

impl InputSelector {
    /// Resolve against the experiment's input ids.
    pub fn resolve(&self, input_ids: &[String]) -> Result<Vec<String>, String> {
        match self {
            InputSelector::All(value) => {
                if value != "all" {
                    return Err(format!(
                        "input selector: expected the string \"all\", found {value:?}"
                    ));
                }
                Ok(input_ids.to_vec())
            }
            InputSelector::List(list) => {
                for id in list {
                    if !input_ids.iter().any(|known| known == id) {
                        return Err(format!(
                            "input selector: input id {id:?} does not exist in the experiment"
                        ));
                    }
                }
                let mut out = list.clone();
                out.sort_unstable();
                out.dedup();
                Ok(out)
            }
        }
    }
}

/// Storage policy for a capture (contract section 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CaptureStorage {
    /// Store only the selected token rows (default).
    SelectedRows,
    /// Store the complete sequence tensor; explicit and reported as a cost.
    FullTensor,
    /// Record deterministic summary statistics only; never usable as an
    /// intervention source.
    SummaryOnly,
}

impl Default for CaptureStorage {
    fn default() -> Self {
        CaptureStorage::SelectedRows
    }
}

/// Output dtype for captured payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CaptureDType {
    F32,
    F16,
}

impl Default for CaptureDType {
    fn default() -> Self {
        CaptureDType::F32
    }
}

/// One declared capture (contract section 8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSpec {
    /// Unique capture id within the experiment.
    pub id: String,
    /// Public semantic hook site.
    pub site: SemanticHookSite,
    /// Layers to capture at; must be `all` or omitted for the
    /// non-per-layer sites (`final-norm-output`, `logits`).
    #[serde(default = "default_all_layers")]
    pub layers: LayerSelector,
    /// Token selector.
    pub tokens: TokenSelector,
    /// Which inputs this capture applies to.
    #[serde(default = "default_all_inputs")]
    pub inputs: InputSelector,
    /// Storage policy (default `selected-rows`).
    #[serde(default)]
    pub storage: CaptureStorage,
    /// Output dtype (default f32).
    #[serde(default)]
    pub dtype: CaptureDType,
}

fn default_all_layers() -> LayerSelector {
    LayerSelector::All("all".to_string())
}

fn default_all_inputs() -> InputSelector {
    InputSelector::All("all".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_selector_resolution() {
        let all = LayerSelector::All("all".into());
        assert_eq!(all.resolve(16).unwrap(), (0..16).collect::<Vec<_>>());
        assert!(all.resolve(0).is_err());

        let list = LayerSelector::List(vec![3, 1, 3]);
        assert_eq!(list.resolve(16).unwrap(), vec![1, 3]);
        assert!(list.resolve(2).is_err());

        let range = LayerSelector::Range(LayerRange {
            start: 1,
            end: 8,
            step: 2,
        });
        assert_eq!(range.resolve(16).unwrap(), vec![1, 3, 5, 7]);
        let bad = LayerSelector::Range(LayerRange {
            start: 5,
            end: 5,
            step: 1,
        });
        assert!(bad.resolve(16).is_err());
        let zero_step = LayerSelector::Range(LayerRange {
            start: 0,
            end: 3,
            step: 0,
        });
        assert!(zero_step.resolve(16).is_err());
    }

    #[test]
    fn input_selector_resolution() {
        let known = ["a".to_string(), "b".to_string()];
        let all = InputSelector::All("all".into());
        assert_eq!(all.resolve(&known).unwrap(), known);
        let list = InputSelector::List(vec!["b".to_string(), "a".to_string()]);
        assert_eq!(list.resolve(&known).unwrap(), vec!["a", "b"]);
        let missing = InputSelector::List(vec!["c".to_string()]);
        assert!(missing.resolve(&known).is_err());
    }
}
