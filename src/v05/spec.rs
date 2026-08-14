//! v0.5 experiment specification v1 (`ember.experiment.v1`).
//!
//! The user-authored TOML form is parsed strictly (unknown fields and
//! unknown schema majors fail), defaults are applied explicitly and
//! recorded, and the fully resolved specification is serialized into the
//! bundle.

use crate::plan::ExecutionMode;
use crate::v05::capture::{CaptureSpec, LayerSelector};
use crate::v05::intervention::{InterventionSource, InterventionSpec};
use crate::v05::token_select::TokenSelector;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Experiment specification schema version identifier.
pub const EXPERIMENT_SCHEMA_V1: &str = "ember.experiment.v1";

/// Field-path error carrying the exact spec location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecError {
    /// TOML field path, e.g. `captures[0].tokens.text`.
    pub path: String,
    pub message: String,
}

impl SpecError {
    pub fn at(path: impl Into<String>, message: impl Into<String>) -> SpecError {
        SpecError {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.path, self.message)
    }
}

impl std::error::Error for SpecError {}

/// One recorded default applied during resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DefaultRecord {
    /// Field path the default applies to.
    pub field: String,
    /// The default value as serialized.
    pub value: String,
}

/// Experiment metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentMetadata {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Sampling seed; 0 means "no stochastic sampling requested".
    #[serde(default)]
    pub seed: u64,
}

/// Model specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    /// Path to the GGUF model file.
    pub path: PathBuf,
    /// Expected model SHA-256 (hex); verified at load when present.
    #[serde(default)]
    pub expected_sha256: String,
    /// Path to `tokenizer.json`; resolved from the architecture when
    /// omitted.
    pub tokenizer: Option<PathBuf>,
    /// Expected tokenizer SHA-256 (hex).
    #[serde(default)]
    pub tokenizer_expected_sha256: String,
    /// Architecture override (`auto`, `gpt2`, `llama`, `qwen3`, `gemma4`);
    /// defaults to `auto`.
    #[serde(default = "default_arch")]
    pub arch: String,
}

fn default_arch() -> String {
    "auto".to_string()
}

/// Execution specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionSpec {
    /// `reference` | `planned` | `planned-fused` (default `reference`).
    pub mode: ExecutionMode,
    /// Thread count; 0 resolves to the machine's available parallelism.
    #[serde(default)]
    pub threads: usize,
    /// Deterministic execution (default true): requires greedy sampling
    /// unless an explicit seed is given.
    #[serde(default = "default_true")]
    pub deterministic: bool,
}

fn default_true() -> bool {
    true
}

/// Generation specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationSpec {
    #[serde(default)]
    pub max_new_tokens: usize,
    #[serde(default)]
    pub temperature: f32,
}

/// One experiment input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSpec {
    pub id: String,
    pub text: String,
}

/// Output specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    /// Bundle output directory (relative to the working directory).
    pub directory: PathBuf,
    /// Tensor payload format; `safetensors` is the only v0.5 format.
    #[serde(default = "default_tensor_format")]
    pub tensor_format: String,
    /// Refuse to overwrite an existing bundle unless true.
    #[serde(default)]
    pub overwrite: bool,
}

fn default_tensor_format() -> String {
    "safetensors".to_string()
}

/// The fully resolved experiment specification (serialized into every
/// bundle as `resolved-experiment.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentSpecV1 {
    pub schema: String,
    pub experiment: ExperimentMetadata,
    pub model: ModelSpec,
    pub execution: ExecutionSpec,
    pub generation: GenerationSpec,
    pub inputs: Vec<InputSpec>,
    pub captures: Vec<CaptureSpec>,
    pub interventions: Vec<InterventionSpec>,
    pub output: OutputSpec,
    /// Every default applied during resolution, in field order.
    pub defaults: Vec<DefaultRecord>,
}

/// The strict user-authored TOML form: every defaultable field is
/// optional so omitted values are distinguishable from explicit ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawExperimentSpec {
    pub schema: String,
    pub experiment: RawExperimentMetadata,
    pub model: RawModelSpec,
    #[serde(default)]
    pub execution: Option<RawExecutionSpec>,
    #[serde(default)]
    pub generation: Option<RawGenerationSpec>,
    pub inputs: Vec<RawInputSpec>,
    #[serde(default)]
    pub captures: Vec<CaptureSpec>,
    #[serde(default)]
    pub interventions: Vec<InterventionSpec>,
    pub output: RawOutputSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawExperimentMetadata {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawModelSpec {
    pub path: PathBuf,
    #[serde(default)]
    pub expected_sha256: Option<String>,
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    #[serde(default)]
    pub tokenizer_expected_sha256: Option<String>,
    #[serde(default)]
    pub arch: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawExecutionSpec {
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub threads: Option<usize>,
    #[serde(default)]
    pub deterministic: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawGenerationSpec {
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawInputSpec {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawOutputSpec {
    pub directory: PathBuf,
    #[serde(default)]
    pub tensor_format: Option<String>,
    #[serde(default)]
    pub overwrite: Option<bool>,
}

fn check_schema_version(schema: &str) -> Result<(), SpecError> {
    if schema == EXPERIMENT_SCHEMA_V1 {
        return Ok(());
    }
    // Reject unknown majors; accept only exact v1 (no minor variants
    // exist yet).
    let major_ok = schema
        .strip_prefix("ember.experiment.")
        .map(|version| {
            version
                .strip_prefix("v1")
                .map(|rest| !rest.starts_with(|c: char| c.is_ascii_digit()))
                .unwrap_or(false)
        })
        .unwrap_or(false);
    if major_ok {
        return Err(SpecError::at(
            "schema",
            format!(
                "experiment schema minor version '{schema}' is not supported; \
                 this build supports exactly '{EXPERIMENT_SCHEMA_V1}'"
            ),
        ));
    }
    Err(SpecError::at(
        "schema",
        format!(
            "unsupported experiment schema '{schema}'; this build supports \
             exactly '{EXPERIMENT_SCHEMA_V1}'"
        ),
    ))
}

impl RawExperimentSpec {
    /// Parse a strict TOML document.
    pub fn from_toml_str(text: &str) -> Result<RawExperimentSpec, SpecError> {
        let spec: RawExperimentSpec = toml::from_str(text).map_err(|error| {
            SpecError::at(
                "<toml>",
                format!("malformed experiment specification: {error}"),
            )
        })?;
        Ok(spec)
    }

    /// Parse a strict TOML document from a file.
    pub fn from_toml_path(path: &std::path::Path) -> Result<RawExperimentSpec, SpecError> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| SpecError::at("<file>", format!("cannot read {path:?}: {error}")))?;
        Self::from_toml_str(&text)
    }

    /// Validate the schema identifier and resolve all defaults.
    pub fn resolve(self) -> Result<ExperimentSpecV1, SpecError> {
        check_schema_version(&self.schema)?;
        let mut defaults = Vec::new();

        if self.experiment.description.is_none() {
            defaults.push(DefaultRecord {
                field: "experiment.description".into(),
                value: String::new(),
            });
        }
        let description = self.experiment.description.unwrap_or_default();
        if self.experiment.seed.is_none() {
            defaults.push(DefaultRecord {
                field: "experiment.seed".into(),
                value: "0".into(),
            });
        }
        let seed = self.experiment.seed.unwrap_or(0);

        if self.model.expected_sha256.is_none() {
            defaults.push(DefaultRecord {
                field: "model.expected_sha256".into(),
                value: String::new(),
            });
        }
        let expected_sha256 = self.model.expected_sha256.unwrap_or_default();
        if self.model.tokenizer_expected_sha256.is_none() {
            defaults.push(DefaultRecord {
                field: "model.tokenizer_expected_sha256".into(),
                value: String::new(),
            });
        }
        let tokenizer_expected_sha256 = self.model.tokenizer_expected_sha256.unwrap_or_default();
        if self.model.arch.is_none() {
            defaults.push(DefaultRecord {
                field: "model.arch".into(),
                value: default_arch(),
            });
        }
        let arch = self.model.arch.unwrap_or_else(default_arch);

        let mode = match self.execution.as_ref().and_then(|e| e.mode.as_deref()) {
            Some(value) => ExecutionMode::from_cli(value)
                .map_err(|error| SpecError::at("execution.mode", error.to_string()))?,
            None => {
                defaults.push(DefaultRecord {
                    field: "execution.mode".into(),
                    value: "reference".into(),
                });
                ExecutionMode::Reference
            }
        };
        let threads = self.execution.as_ref().and_then(|e| e.threads).unwrap_or(0);
        if self.execution.as_ref().and_then(|e| e.threads).is_none() {
            defaults.push(DefaultRecord {
                field: "execution.threads".into(),
                value: "0 (auto)".into(),
            });
        }
        let deterministic = self
            .execution
            .as_ref()
            .and_then(|e| e.deterministic)
            .unwrap_or(true);
        if self
            .execution
            .as_ref()
            .and_then(|e| e.deterministic)
            .is_none()
        {
            defaults.push(DefaultRecord {
                field: "execution.deterministic".into(),
                value: "true".into(),
            });
        }

        let max_new_tokens = self
            .generation
            .as_ref()
            .and_then(|g| g.max_new_tokens)
            .unwrap_or(0);
        if self
            .generation
            .as_ref()
            .and_then(|g| g.max_new_tokens)
            .is_none()
        {
            defaults.push(DefaultRecord {
                field: "generation.max_new_tokens".into(),
                value: "0".into(),
            });
        }
        let temperature = self
            .generation
            .as_ref()
            .and_then(|g| g.temperature)
            .unwrap_or(0.0);
        if self
            .generation
            .as_ref()
            .and_then(|g| g.temperature)
            .is_none()
        {
            defaults.push(DefaultRecord {
                field: "generation.temperature".into(),
                value: "0.0".into(),
            });
        }
        if !temperature.is_finite() {
            return Err(SpecError::at(
                "generation.temperature",
                "temperature must be finite",
            ));
        }
        if deterministic && temperature != 0.0 && seed == 0 {
            return Err(SpecError::at(
                "execution.deterministic",
                "deterministic execution requires temperature = 0.0 or an explicit \
                 experiment.seed",
            ));
        }

        let tensor_format = self
            .output
            .tensor_format
            .clone()
            .unwrap_or_else(default_tensor_format);
        if self.output.tensor_format.is_none() {
            defaults.push(DefaultRecord {
                field: "output.tensor_format".into(),
                value: tensor_format.clone(),
            });
        }
        if tensor_format != "safetensors" {
            return Err(SpecError::at(
                "output.tensor_format",
                format!(
                    "unsupported tensor format '{tensor_format}'; v0.5 supports only \
                     'safetensors'"
                ),
            ));
        }
        let overwrite = self.output.overwrite.unwrap_or(false);
        if self.output.overwrite.is_none() {
            defaults.push(DefaultRecord {
                field: "output.overwrite".into(),
                value: "false".into(),
            });
        }

        if self.inputs.is_empty() {
            return Err(SpecError::at(
                "inputs",
                "the experiment must declare at least one input",
            ));
        }

        let resolved = ExperimentSpecV1 {
            schema: EXPERIMENT_SCHEMA_V1.to_string(),
            experiment: ExperimentMetadata {
                name: self.experiment.name.clone(),
                description,
                seed,
            },
            model: ModelSpec {
                path: self.model.path.clone(),
                expected_sha256,
                tokenizer: self.model.tokenizer.clone(),
                tokenizer_expected_sha256,
                arch,
            },
            execution: ExecutionSpec {
                mode,
                threads,
                deterministic,
            },
            generation: GenerationSpec {
                max_new_tokens,
                temperature,
            },
            inputs: self
                .inputs
                .iter()
                .map(|input| InputSpec {
                    id: input.id.clone(),
                    text: input.text.clone(),
                })
                .collect(),
            captures: self.captures.clone(),
            interventions: self.interventions.clone(),
            output: OutputSpec {
                directory: self.output.directory.clone(),
                tensor_format,
                overwrite,
            },
            defaults,
        };
        resolved.validate()?;
        Ok(resolved)
    }
}

impl ExperimentSpecV1 {
    /// Validate all cross-references and fail-closed rules without model
    /// metadata (contract Gate A).
    pub fn validate(&self) -> Result<(), SpecError> {
        check_schema_version(&self.schema)?;
        if self.experiment.name.trim().is_empty() {
            return Err(SpecError::at(
                "experiment.name",
                "experiment name must not be empty",
            ));
        }
        if !is_safe_id(&self.experiment.name) {
            return Err(SpecError::at(
                "experiment.name",
                format!(
                    "experiment name {:?} contains characters that are unsafe in paths; \
                     use [a-zA-Z0-9._-]",
                    self.experiment.name
                ),
            ));
        }

        let input_ids: Vec<&str> = self.inputs.iter().map(|input| input.id.as_str()).collect();
        for (index, id) in input_ids.iter().enumerate() {
            if !is_safe_id(id) {
                return Err(SpecError::at(
                    format!("inputs[{index}].id"),
                    format!("input id {id:?} is not a safe identifier"),
                ));
            }
            if input_ids[..index].iter().any(|prior| prior == id) {
                return Err(SpecError::at(
                    format!("inputs[{index}].id"),
                    format!("duplicate input id {id:?}"),
                ));
            }
        }

        let capture_ids: Vec<&str> = self.captures.iter().map(|c| c.id.as_str()).collect();
        for (index, id) in capture_ids.iter().enumerate() {
            if !is_safe_id(id) {
                return Err(SpecError::at(
                    format!("captures[{index}].id"),
                    format!("capture id {id:?} is not a safe identifier"),
                ));
            }
            if capture_ids[..index].iter().any(|prior| prior == id) {
                return Err(SpecError::at(
                    format!("captures[{index}].id"),
                    format!("duplicate capture id {id:?}"),
                ));
            }
        }

        let intervention_ids: Vec<&str> =
            self.interventions.iter().map(|i| i.id.as_str()).collect();
        for (index, id) in intervention_ids.iter().enumerate() {
            if !is_safe_id(id) {
                return Err(SpecError::at(
                    format!("interventions[{index}].id"),
                    format!("intervention id {id:?} is not a safe identifier"),
                ));
            }
            if intervention_ids[..index].iter().any(|prior| prior == id) {
                return Err(SpecError::at(
                    format!("interventions[{index}].id"),
                    format!("duplicate intervention id {id:?}"),
                ));
            }
            if capture_ids.iter().any(|capture| capture == id) {
                return Err(SpecError::at(
                    format!("interventions[{index}].id"),
                    format!("intervention id {id:?} collides with a capture id"),
                ));
            }
        }

        let input_ids: Vec<String> = input_ids.iter().map(|s| s.to_string()).collect();
        for (index, capture) in self.captures.iter().enumerate() {
            let path = format!("captures[{index}]");
            capture
                .inputs
                .resolve(&input_ids)
                .map_err(|message| SpecError::at(format!("{path}.inputs"), message))?;
            if !capture.site.is_per_layer() {
                if !matches!(capture.layers, LayerSelector::All(_)) {
                    return Err(SpecError::at(
                        format!("{path}.layers"),
                        format!(
                            "capture site {} does not carry layers; use layers = \"all\" \
                             (or omit it)",
                            capture.site
                        ),
                    ));
                }
                capture
                    .layers
                    .resolve(1)
                    .map_err(|message| SpecError::at(format!("{path}.layers"), message))?;
            }
            if capture.tokens.requires_text() {
                for input in &self.inputs {
                    if input.text.is_empty() {
                        return Err(SpecError::at(
                            format!("{path}.tokens"),
                            format!(
                                "token selector {:?} requires non-empty input text; input {} \
                                 is empty",
                                capture.tokens, input.id
                            ),
                        ));
                    }
                }
            }
            if let TokenSelector::GeneratedStep { .. } = &capture.tokens {
                if self.generation.max_new_tokens == 0 {
                    return Err(SpecError::at(
                        format!("{path}.tokens"),
                        "generated-step token selection requires generation.max_new_tokens > 0",
                    ));
                }
            }
        }

        for (index, intervention) in self.interventions.iter().enumerate() {
            let path = format!("interventions[{index}]");
            intervention
                .validate_self()
                .map_err(|message| SpecError::at(path.clone(), message))?;
            intervention
                .inputs
                .resolve(&input_ids)
                .map_err(|message| SpecError::at(format!("{path}.inputs"), message))?;
            if !intervention.site.is_per_layer() {
                if !matches!(intervention.layers, LayerSelector::All(_)) {
                    return Err(SpecError::at(
                        format!("{path}.layers"),
                        format!(
                            "intervention site {} does not carry layers; use layers = \"all\" \
                             (or omit it)",
                            intervention.site
                        ),
                    ));
                }
                intervention
                    .layers
                    .resolve(1)
                    .map_err(|message| SpecError::at(format!("{path}.layers"), message))?;
            }
            if let Some(InterventionSource::CaptureFromCurrentRun { capture_id }) =
                &intervention.source
            {
                if !capture_ids.iter().any(|known| known == capture_id) {
                    return Err(SpecError::at(
                        format!("{path}.source"),
                        format!(
                            "source capture id {capture_id:?} does not exist among captures \
                             (known: {capture_ids:?})"
                        ),
                    ));
                }
            }
            if let Some(InterventionSource::CaptureFromBundle { bundle_path, .. }) =
                &intervention.source
            {
                if bundle_path.as_os_str().is_empty() {
                    return Err(SpecError::at(
                        format!("{path}.source"),
                        "source bundle path must not be empty",
                    ));
                }
            }
            if let TokenSelector::GeneratedStep { .. } = &intervention.tokens {
                if self.generation.max_new_tokens == 0 {
                    return Err(SpecError::at(
                        format!("{path}.tokens"),
                        "generated-step token selection requires generation.max_new_tokens > 0",
                    ));
                }
            }
        }

        if self.output.directory.as_os_str().is_empty() {
            return Err(SpecError::at(
                "output.directory",
                "output directory must not be empty",
            ));
        }
        Ok(())
    }
}

/// Restrict experiment/input/capture/intervention ids to characters that
/// are safe in paths and bundle identifiers.
pub fn is_safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_SPEC: &str = r#"
schema = "ember.experiment.v1"

[experiment]
name = "layerwise-target-capture"
description = "capture prompt-final and target-final-subtoken representations."
seed = 42

[model]
path = "/models/model.gguf"
expected_sha256 = "aa"

[execution]
mode = "planned-fused"
threads = 8
deterministic = true

[generation]
max_new_tokens = 0
temperature = 0.0

[[inputs]]
id = "example-001"
text = "some prompt"

[[captures]]
id = "prompt-final"
site = "residual-post-mlp"
layers = "all"

[captures.tokens]
kind = "prompt-final"

[output]
directory = "runs/layerwise-target-capture"
tensor_format = "safetensors"
overwrite = false
"#;

    #[test]
    fn valid_spec_parses_and_resolves() {
        let raw = RawExperimentSpec::from_toml_str(VALID_SPEC).unwrap();
        let resolved = raw.resolve().unwrap();
        assert_eq!(resolved.schema, EXPERIMENT_SCHEMA_V1);
        assert_eq!(resolved.experiment.seed, 42);
        assert_eq!(resolved.execution.mode, ExecutionMode::PlannedFused);
        assert_eq!(resolved.execution.threads, 8);
        assert!(resolved.execution.deterministic);
        assert_eq!(resolved.generation.max_new_tokens, 0);
        assert_eq!(resolved.captures.len(), 1);
        // Only tokenizer_expected_sha256 and arch are unset in VALID_SPEC.
        assert_eq!(resolved.defaults.len(), 2);
    }

    #[test]
    fn unknown_schema_major_fails() {
        let text = VALID_SPEC.replace("ember.experiment.v1", "ember.experiment.v2");
        let raw = RawExperimentSpec::from_toml_str(&text).unwrap();
        let error = raw.resolve().unwrap_err();
        assert!(error.to_string().contains("unsupported experiment schema"));
        assert_eq!(error.path, "schema");
    }

    #[test]
    fn minor_schema_versions_fail_closed() {
        let text = VALID_SPEC.replace("ember.experiment.v1", "ember.experiment.v1.1");
        let raw = RawExperimentSpec::from_toml_str(&text).unwrap();
        assert!(raw.resolve().is_err());
    }

    #[test]
    fn unknown_fields_fail() {
        let text = VALID_SPEC.replace("max_new_tokens = 0", "max_new_tokens = 0\nunknown = 1");
        let error = RawExperimentSpec::from_toml_str(&text).unwrap_err();
        assert!(error.message.contains("unknown"), "{}", error.message);
    }

    #[test]
    fn omitted_defaults_are_recorded() {
        let text = r#"
schema = "ember.experiment.v1"

[experiment]
name = "minimal"

[model]
path = "m.gguf"

[[inputs]]
id = "i1"
text = "hello"

[output]
directory = "runs/minimal"
"#;
        let raw = RawExperimentSpec::from_toml_str(text).unwrap();
        let resolved = raw.resolve().unwrap();
        assert_eq!(resolved.execution.mode, ExecutionMode::Reference);
        assert_eq!(resolved.execution.threads, 0);
        assert_eq!(resolved.generation.temperature, 0.0);
        assert_eq!(resolved.output.tensor_format, "safetensors");
        assert!(!resolved.output.overwrite);
        assert!(!resolved.defaults.is_empty());
        // resolved serialization is deterministic JSON
        let a = serde_json::to_vec(&resolved).unwrap();
        let b = serde_json::to_vec(&resolved).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn duplicate_ids_fail() {
        let text = VALID_SPEC.replace(
            "[[captures]]",
            "[[captures]]\nid = \"dup\"\nsite = \"mlp-output\"\nlayers = \"all\"\n\
             [captures.tokens]\nkind = \"prompt-final\"\n\n[[captures]]",
        );
        let text = text.replace(
            "id = \"prompt-final\"\nsite = \"residual-post-mlp\"",
            "id = \"dup\"\nsite = \"residual-post-mlp\"",
        );
        let raw = RawExperimentSpec::from_toml_str(&text).unwrap();
        let error = raw.resolve().unwrap_err();
        assert!(error.message.contains("duplicate capture id"), "{}", error);
        assert!(error.path.starts_with("captures["));
    }

    #[test]
    fn unsupported_execution_mode_fails_before_inference() {
        let text = VALID_SPEC.replace("mode = \"planned-fused\"", "mode = \"quantum\"");
        let raw = RawExperimentSpec::from_toml_str(&text).unwrap();
        let error = raw.resolve().unwrap_err();
        assert!(error.message.contains("unknown --execution"), "{}", error);
        assert_eq!(error.path, "execution.mode");
    }

    #[test]
    fn deterministic_requires_greedy_or_seed() {
        let text = VALID_SPEC
            .replace("temperature = 0.0", "temperature = 0.7")
            .replace("seed = 42", "seed = 0");
        let raw = RawExperimentSpec::from_toml_str(&text).unwrap();
        let error = raw.resolve().unwrap_err();
        assert!(error.message.contains("deterministic"), "{}", error.message);
        // with an explicit seed it is allowed
        let seeded = text.replace("seed = 0", "seed = 7");
        let raw = RawExperimentSpec::from_toml_str(&seeded).unwrap();
        assert!(raw.resolve().is_ok());
    }

    #[test]
    fn unsupported_tensor_format_fails() {
        let text = VALID_SPEC.replace("safetensors", "npy");
        let raw = RawExperimentSpec::from_toml_str(&text).unwrap();
        let error = raw.resolve().unwrap_err();
        assert!(error.message.contains("tensor format"), "{}", error);
        assert_eq!(error.path, "output.tensor_format");
    }

    #[test]
    fn capture_source_references_resolve() {
        let text = r#"
schema = "ember.experiment.v1"

[experiment]
name = "intervention"

[model]
path = "m.gguf"

[[inputs]]
id = "i1"
text = "hello world"

[[captures]]
id = "cap-1"
site = "attention-output"
layers = [0]

[captures.tokens]
kind = "prompt-final"

[[interventions]]
id = "iv-1"
site = "attention-output"
layers = [0]
operation = { kind = "replace" }
source = { kind = "capture-from-current-run", capture_id = "cap-1" }

[interventions.tokens]
kind = "prompt-final"

[output]
directory = "runs/intervention"
"#;
        let raw = RawExperimentSpec::from_toml_str(text).unwrap();
        assert!(raw.resolve().is_ok());

        let broken = text.replace(
            "source = { kind = \"capture-from-current-run\", capture_id = \"cap-1\" }",
            "source = { kind = \"capture-from-current-run\", capture_id = \"cap-nope\" }",
        );
        let raw = RawExperimentSpec::from_toml_str(&broken).unwrap();
        let error = raw.resolve().unwrap_err();
        assert!(
            error.message.contains("does not exist among captures"),
            "{}",
            error
        );
    }

    #[test]
    fn unsafe_ids_fail() {
        let text = VALID_SPEC.replace("name = \"layerwise-target-capture\"", "name = \"../evil\"");
        let raw = RawExperimentSpec::from_toml_str(&text).unwrap();
        let error = raw.resolve().unwrap_err();
        assert!(error.message.contains("unsafe in paths"), "{}", error);
    }

    #[test]
    fn non_per_layer_sites_reject_explicit_layers() {
        let text = VALID_SPEC.replace(
            "site = \"residual-post-mlp\"\nlayers = \"all\"",
            "site = \"logits\"\nlayers = [3]",
        );
        let raw = RawExperimentSpec::from_toml_str(&text).unwrap();
        let error = raw.resolve().unwrap_err();
        assert!(error.message.contains("does not carry layers"), "{}", error);
    }
}
