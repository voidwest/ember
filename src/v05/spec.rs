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

/// Resolve one optional spec field: record its default (as the display
/// string given by `record`) when the field is absent and return the value.
/// Validate one id list: every id must be a safe path identifier, must be
/// unique within the list, and must not collide with `other` (intervention
/// ids vs capture ids).
fn validate_unique_ids(kind: &str, ids: &[&str], other: &[&str]) -> Result<(), SpecError> {
    for (index, id) in ids.iter().enumerate() {
        if !is_safe_id(id) {
            return Err(SpecError::at(
                format!("{kind}s[{index}].id"),
                format!("{kind} id {id:?} is not a safe identifier"),
            ));
        }
        if ids[..index].iter().any(|prior| prior == id) {
            return Err(SpecError::at(
                format!("{kind}s[{index}].id"),
                format!("duplicate {kind} id {id:?}"),
            ));
        }
        if other.iter().any(|other_id| other_id == id) {
            return Err(SpecError::at(
                format!("{kind}s[{index}].id"),
                format!("{kind} id {id:?} collides with a capture id"),
            ));
        }
    }
    Ok(())
}

fn take_default<T: Clone>(
    opt: Option<T>,
    field: &str,
    default: T,
    record: impl Into<String>,
    defaults: &mut Vec<DefaultRecord>,
) -> T {
    if opt.is_none() {
        defaults.push(DefaultRecord {
            field: field.into(),
            value: record.into(),
        });
    }
    opt.unwrap_or(default)
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
    pub captures: Option<Vec<RawDefinition<CaptureSpec>>>,
    #[serde(default)]
    pub interventions: Option<Vec<RawDefinition<InterventionSpec>>>,
    pub output: RawOutputSpec,
    /// A layer sweep (`crate::v05::sweep`): the spec is then a template
    /// from which one experiment per point is derived, and it does not
    /// resolve as a single experiment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sweep: Option<crate::v05::sweep::RawSweepSpec>,
}

/// A strictly validated definition retaining the fields actually supplied by
/// the author. Keeping the wire value prevents nested Serde defaults from
/// erasing omission provenance before resolution. Serialization retains that
/// omission information as well.
#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
pub struct RawDefinition<T> {
    value: serde_json::Value,
    #[serde(skip)]
    marker: std::marker::PhantomData<T>,
}

impl<'de, T: serde::de::DeserializeOwned> Deserialize<'de> for RawDefinition<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = serde_json::Value::deserialize(deserializer)?;
        serde_json::from_value::<T>(value.clone()).map_err(serde::de::Error::custom)?;
        omit_null_fields(&mut value);
        Ok(Self {
            value,
            marker: std::marker::PhantomData,
        })
    }
}

impl<T: Serialize + serde::de::DeserializeOwned> RawDefinition<T> {
    /// Construct a definition for programmatic callers such as the GUI.
    /// Concrete values are explicit; absent optional values remain omitted,
    /// since TOML has no null literal.
    pub fn explicit(value: T) -> Result<Self, serde_json::Error> {
        let mut value = serde_json::to_value(value)?;
        omit_null_fields(&mut value);
        Ok(Self {
            value,
            marker: std::marker::PhantomData,
        })
    }

    fn resolve(self, path: &str, defaults: &mut Vec<DefaultRecord>) -> Result<T, SpecError> {
        let resolved: T = serde_json::from_value(self.value.clone())
            .map_err(|error| SpecError::at(path, error.to_string()))?;
        let mut serialized = serde_json::to_value(&resolved)
            .map_err(|error| SpecError::at(path, error.to_string()))?;
        crate::plan::sort_value_keys(&mut serialized);
        record_nested_defaults(Some(&self.value), &serialized, path, defaults);
        Ok(resolved)
    }
}

fn omit_null_fields(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(fields) => {
            fields.retain(|_, value| !value.is_null());
            for value in fields.values_mut() {
                omit_null_fields(value);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                omit_null_fields(value);
            }
        }
        _ => {}
    }
}

fn record_nested_defaults(
    supplied: Option<&serde_json::Value>,
    resolved: &serde_json::Value,
    path: &str,
    defaults: &mut Vec<DefaultRecord>,
) {
    if let Some(object) = resolved.as_object() {
        for (key, value) in object {
            record_nested_defaults(
                supplied.and_then(|value| value.get(key)),
                value,
                &format!("{path}.{key}"),
                defaults,
            );
        }
    } else if supplied.is_none() {
        defaults.push(DefaultRecord {
            field: path.into(),
            value: resolved
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| resolved.to_string()),
        });
    }
}

fn resolve_definitions<T: Serialize + serde::de::DeserializeOwned>(
    definitions: Option<Vec<RawDefinition<T>>>,
    path: &str,
    defaults: &mut Vec<DefaultRecord>,
) -> Result<Vec<T>, SpecError> {
    let Some(definitions) = definitions else {
        defaults.push(DefaultRecord {
            field: path.into(),
            value: "[]".into(),
        });
        return Ok(Vec::new());
    };
    definitions
        .into_iter()
        .enumerate()
        .map(|(index, definition)| definition.resolve(&format!("{path}[{index}]"), defaults))
        .collect()
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
        if self.sweep.is_some() {
            return Err(SpecError::at(
                "sweep",
                "this specification declares a [sweep]: it describes one experiment per sweep \
                 point and is run as a sweep (`ember experiment run` handles it), not resolved \
                 as a single experiment",
            ));
        }
        let mut defaults = Vec::new();

        let description = take_default(
            self.experiment.description,
            "experiment.description",
            String::new(),
            "",
            &mut defaults,
        );
        let seed = take_default(
            self.experiment.seed,
            "experiment.seed",
            0,
            "0",
            &mut defaults,
        );
        let expected_sha256 = take_default(
            self.model.expected_sha256,
            "model.expected_sha256",
            String::new(),
            "",
            &mut defaults,
        );
        let tokenizer_expected_sha256 = take_default(
            self.model.tokenizer_expected_sha256,
            "model.tokenizer_expected_sha256",
            String::new(),
            "",
            &mut defaults,
        );
        let arch_default = default_arch();
        let arch = take_default(
            self.model.arch,
            "model.arch",
            arch_default.clone(),
            arch_default,
            &mut defaults,
        );
        if self.model.tokenizer.is_none() {
            defaults.push(DefaultRecord {
                field: "model.tokenizer".into(),
                value: "null (auto)".into(),
            });
        }

        let mode = match self.execution.as_ref().and_then(|e| e.mode.as_deref()) {
            Some(value) => ExecutionMode::from_cli(value)
                .map_err(|error| SpecError::at("execution.mode", error.to_string()))?,
            None => take_default(
                None,
                "execution.mode",
                ExecutionMode::Reference,
                "reference",
                &mut defaults,
            ),
        };
        let threads = take_default(
            self.execution.as_ref().and_then(|e| e.threads),
            "execution.threads",
            0,
            "0 (auto)",
            &mut defaults,
        );
        let deterministic = take_default(
            self.execution.as_ref().and_then(|e| e.deterministic),
            "execution.deterministic",
            true,
            "true",
            &mut defaults,
        );

        let max_new_tokens = take_default(
            self.generation.as_ref().and_then(|g| g.max_new_tokens),
            "generation.max_new_tokens",
            0,
            "0",
            &mut defaults,
        );
        let temperature = take_default(
            self.generation.as_ref().and_then(|g| g.temperature),
            "generation.temperature",
            0.0,
            "0.0",
            &mut defaults,
        );
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

        let format_default = default_tensor_format();
        let tensor_format = take_default(
            self.output.tensor_format.clone(),
            "output.tensor_format",
            format_default.clone(),
            format_default,
            &mut defaults,
        );
        if tensor_format != "safetensors" {
            return Err(SpecError::at(
                "output.tensor_format",
                format!(
                    "unsupported tensor format '{tensor_format}'; v0.5 supports only \
                     'safetensors'"
                ),
            ));
        }
        let overwrite = take_default(
            self.output.overwrite,
            "output.overwrite",
            false,
            "false",
            &mut defaults,
        );

        if self.inputs.is_empty() {
            return Err(SpecError::at(
                "inputs",
                "the experiment must declare at least one input",
            ));
        }

        let captures = resolve_definitions(self.captures, "captures", &mut defaults)?;
        let interventions =
            resolve_definitions(self.interventions, "interventions", &mut defaults)?;
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
            captures,
            interventions,
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
        let capture_ids: Vec<&str> = self.captures.iter().map(|c| c.id.as_str()).collect();
        let intervention_ids: Vec<&str> =
            self.interventions.iter().map(|i| i.id.as_str()).collect();
        validate_unique_ids("input", &input_ids, &[])?;
        validate_unique_ids("capture", &capture_ids, &[])?;
        validate_unique_ids("intervention", &intervention_ids, &capture_ids)?;

        let input_ids: Vec<String> = input_ids.iter().map(|s| s.to_string()).collect();
        for (index, capture) in self.captures.iter().enumerate() {
            let path = format!("captures[{index}]");
            let selected_inputs = capture
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
                    if selected_inputs.contains(&input.id) && input.text.is_empty() {
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
            if let TokenSelector::GeneratedStep { .. } = &capture.tokens
                && self.generation.max_new_tokens == 0
            {
                return Err(SpecError::at(
                    format!("{path}.tokens"),
                    "generated-step token selection requires generation.max_new_tokens > 0",
                ));
            }
        }

        for (index, intervention) in self.interventions.iter().enumerate() {
            let path = format!("interventions[{index}]");
            intervention
                .validate_self()
                .map_err(|message| SpecError::at(path.clone(), message))?;
            let selected_inputs = intervention
                .inputs
                .resolve(&input_ids)
                .map_err(|message| SpecError::at(format!("{path}.inputs"), message))?;
            if intervention.tokens.requires_text() {
                for input in &self.inputs {
                    if selected_inputs.contains(&input.id) && input.text.is_empty() {
                        return Err(SpecError::at(
                            format!("{path}.tokens"),
                            format!(
                                "token selector {:?} requires non-empty input text; input {} \
                                 is empty",
                                intervention.tokens, input.id
                            ),
                        ));
                    }
                }
            }
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
                && !capture_ids.iter().any(|known| known == capture_id)
            {
                return Err(SpecError::at(
                    format!("{path}.source"),
                    format!(
                        "source capture id {capture_id:?} does not exist among captures \
                             (known: {capture_ids:?})"
                    ),
                ));
            }
            if let Some(InterventionSource::CaptureFromBundle { bundle_path, .. }) =
                &intervention.source
                && bundle_path.as_os_str().is_empty()
            {
                return Err(SpecError::at(
                    format!("{path}.source"),
                    "source bundle path must not be empty",
                ));
            }
            if let TokenSelector::GeneratedStep { .. } = &intervention.tokens
                && self.generation.max_new_tokens == 0
            {
                return Err(SpecError::at(
                    format!("{path}.tokens"),
                    "generated-step token selection requires generation.max_new_tokens > 0",
                ));
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
        let fields: Vec<_> = resolved
            .defaults
            .iter()
            .map(|record| record.field.as_str())
            .collect();
        assert_eq!(
            fields,
            [
                "model.tokenizer_expected_sha256",
                "model.arch",
                "model.tokenizer",
                "captures[0].dtype",
                "captures[0].inputs",
                "captures[0].storage",
                "interventions"
            ]
        );
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
    fn nested_selector_operation_and_source_fields_are_strict() {
        for selector in [
            r#"{ kind = "prompt-final", unexpected = true }"#,
            r#"{ kind = "absolute-token", index = 0, unexpected = true }"#,
        ] {
            let mut value: toml::Value = toml::from_str(VALID_SPEC).unwrap();
            let fragment: toml::Value = toml::from_str(&format!("tokens = {selector}")).unwrap();
            value["captures"][0]["tokens"] = fragment["tokens"].clone();
            assert!(RawExperimentSpec::from_toml_str(&toml::to_string(&value).unwrap()).is_err());
        }
        for operation in [
            r#"{"kind":"zero","unexpected":true}"#,
            r#"{"kind":"scale","factor":0.5,"unexpected":true}"#,
        ] {
            assert!(
                serde_json::from_str::<crate::v05::intervention::InterventionOperation>(operation)
                    .is_err()
            );
        }
        for source in [
            r#"{"kind":"zero","unexpected":true}"#,
            r#"{"kind":"capture-from-current-run","capture_id":"c","unexpected":true}"#,
        ] {
            assert!(serde_json::from_str::<InterventionSource>(source).is_err());
        }
    }

    #[test]
    fn intervention_text_selectors_reject_only_selected_empty_inputs() {
        for selector in [
            serde_json::json!({"kind":"matched-span", "text":"word", "occurrence":0, "subtokens":"all"}),
            serde_json::json!({"kind":"byte-span", "start":0, "end":4, "subtokens":"all"}),
        ] {
            let mut spec = RawExperimentSpec::from_toml_str(VALID_SPEC)
                .unwrap()
                .resolve()
                .unwrap();
            spec.captures.clear();
            spec.inputs[0].text.clear();
            spec.inputs.push(InputSpec {
                id: "nonempty".into(),
                text: "word".into(),
            });
            spec.interventions.push(
                serde_json::from_value(serde_json::json!({
                    "id":"patch", "site":"mlp-output", "tokens":selector,
                    "operation":{"kind":"zero"},
                }))
                .unwrap(),
            );
            let error = spec.validate().unwrap_err();
            assert_eq!(error.path, "interventions[0].tokens");
            assert!(error.message.contains("requires non-empty input text"));
            assert!(error.message.contains("example-001"));
            spec.interventions[0].inputs =
                crate::v05::capture::InputSelector::List(vec!["nonempty".into()]);
            spec.validate().unwrap();
            // Position-only selectors do not require prompt text at this stage.
            spec.interventions[0].inputs = crate::v05::capture::InputSelector::All("all".into());
            spec.interventions[0].tokens = TokenSelector::PromptFinal;
            spec.validate().unwrap();
        }
    }

    #[test]
    fn required_fields_cannot_be_defaulted_away() {
        for path in [
            vec!["schema"],
            vec!["experiment"],
            vec!["model"],
            vec!["inputs"],
            vec!["output"],
            vec!["experiment", "name"],
            vec!["model", "path"],
            vec!["output", "directory"],
        ] {
            let mut value: toml::Value = toml::from_str(VALID_SPEC).unwrap();
            let mut table = &mut value;
            for key in &path[..path.len() - 1] {
                table = table.get_mut(*key).unwrap();
            }
            table.as_table_mut().unwrap().remove(*path.last().unwrap());
            assert!(
                RawExperimentSpec::from_toml_str(&toml::to_string(&value).unwrap()).is_err(),
                "{path:?}"
            );
        }
        for (array, fields) in [
            ("inputs", &["id", "text"][..]),
            ("captures", &["id", "site", "tokens"][..]),
        ] {
            for field in fields {
                let mut value: toml::Value = toml::from_str(VALID_SPEC).unwrap();
                value[array][0].as_table_mut().unwrap().remove(*field);
                assert!(
                    RawExperimentSpec::from_toml_str(&toml::to_string(&value).unwrap()).is_err(),
                    "{array}.{field}"
                );
            }
        }
    }

    #[test]
    fn every_tagged_variant_requires_its_declared_fields() {
        fn check<T: serde::de::DeserializeOwned>(cases: &[&str], optional: &[&str]) {
            for text in cases {
                let complete: serde_json::Value = serde_json::from_str(text).unwrap();
                assert!(
                    serde_json::from_value::<T>(complete.clone()).is_ok(),
                    "{text}"
                );
                for key in complete.as_object().unwrap().keys() {
                    if optional.contains(&key.as_str()) {
                        continue;
                    }
                    let mut missing = complete.clone();
                    missing.as_object_mut().unwrap().remove(key);
                    assert!(
                        serde_json::from_value::<T>(missing).is_err(),
                        "{text} accepted missing {key}"
                    );
                }
            }
        }
        check::<TokenSelector>(
            &[
                r#"{"kind":"prompt-final"}"#,
                r#"{"kind":"absolute-token","index":0}"#,
                r#"{"kind":"relative-token","offset_from_end":0}"#,
                r#"{"kind":"generated-step","step":1}"#,
                r#"{"kind":"matched-span","text":"word","occurrence":1,"subtokens":"final","normalization":"none"}"#,
                r#"{"kind":"byte-span","start":0,"end":4,"subtokens":"all"}"#,
            ],
            &["normalization"],
        );
        check::<crate::v05::intervention::InterventionOperation>(
            &[
                r#"{"kind":"replace"}"#,
                r#"{"kind":"zero"}"#,
                r#"{"kind":"scale","factor":0.5}"#,
                r#"{"kind":"interpolate","alpha":0.5}"#,
                r#"{"kind":"add-delta"}"#,
                r#"{"kind":"restore-original"}"#,
            ],
            &[],
        );
        check::<InterventionSource>(
            &[
                r#"{"kind":"inline-vector","values":[1.0]}"#,
                r#"{"kind":"capture-from-current-run","capture_id":"cap"}"#,
                r#"{"kind":"capture-from-bundle","bundle_path":"source","capture_id":"cap","input_id":"i","layer":0}"#,
                r#"{"kind":"zero"}"#,
            ],
            &[],
        );
        check::<InterventionSpec>(
            &[
                r#"{"id":"i","site":"mlp-output","tokens":{"kind":"prompt-final"},"operation":{"kind":"zero"}}"#,
            ],
            &[],
        );
        check::<crate::v05::capture::LayerRange>(&[r#"{"start":0,"end":4,"step":1}"#], &["step"]);
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
    fn nested_defaults_are_recorded_without_changing_semantics() {
        let text = VALID_SPEC.replace("layers = \"all\"", "layers = { start = 0, end = 2 }")
            .replace("kind = \"prompt-final\"", "kind = \"matched-span\"\ntext = \"some\"\noccurrence = 0\nsubtokens = \"all\"")
            + "\n[[interventions]]\nid = \"zero\"\nsite = \"mlp-output\"\noperation = { kind = \"zero\" }\ntokens = { kind = \"prompt-final\" }\n";
        let raw = RawExperimentSpec::from_toml_str(&text).unwrap();
        let mut omitted = raw.clone().resolve().unwrap();
        // A raw TOML round trip must not materialize defaults before resolve.
        let round_trip = RawExperimentSpec::from_toml_str(&toml::to_string(&raw).unwrap())
            .unwrap()
            .resolve()
            .unwrap();
        assert_eq!(omitted, round_trip);
        let defaults: std::collections::BTreeMap<_, _> = omitted
            .defaults
            .iter()
            .map(|record| (record.field.as_str(), record.value.as_str()))
            .collect();
        for (path, value) in [
            ("captures[0].layers.step", "1"),
            ("captures[0].tokens.normalization", "none"),
            ("captures[0].dtype", "f32"),
            ("captures[0].storage", "selected-rows"),
            ("interventions[0].layers", "all"),
            ("interventions[0].inputs", "all"),
            ("interventions[0].source", "null"),
            ("interventions[0].shape_policy", "strict"),
            (
                "interventions[0].compatibility.allow_model_mismatch",
                "false",
            ),
            (
                "interventions[0].compatibility.allow_tokenizer_mismatch",
                "false",
            ),
        ] {
            assert_eq!(defaults.get(path), Some(&value), "{path}");
        }
        let mut explicit = raw;
        explicit.captures = Some(
            omitted
                .captures
                .iter()
                .cloned()
                .map(RawDefinition::explicit)
                .collect::<Result<_, _>>()
                .unwrap(),
        );
        explicit.interventions = Some(
            omitted
                .interventions
                .iter()
                .cloned()
                .map(RawDefinition::explicit)
                .collect::<Result<_, _>>()
                .unwrap(),
        );
        let mut explicit = RawExperimentSpec::from_toml_str(&toml::to_string(&explicit).unwrap())
            .unwrap()
            .resolve()
            .unwrap();
        assert!(!explicit
            .defaults
            .iter()
            .any(|record| record.field.starts_with("captures[0]")));
        assert!(!explicit
            .defaults
            .iter()
            .any(|record| record.field.contains("compatibility")));
        omitted.defaults.clear();
        explicit.defaults.clear();
        assert_eq!(omitted, explicit);
    }

    #[test]
    fn explicit_empty_collections_do_not_count_as_defaults() {
        let text = VALID_SPEC.replace("[experiment]", "interventions = []\n\n[experiment]");
        let spec = RawExperimentSpec::from_toml_str(&text)
            .unwrap()
            .resolve()
            .unwrap();
        assert!(!spec
            .defaults
            .iter()
            .any(|record| record.field == "interventions"));
        assert!(spec.interventions.is_empty());
    }

    #[test]
    fn text_requirements_apply_only_to_selected_inputs() {
        let mut spec = RawExperimentSpec::from_toml_str(VALID_SPEC)
            .unwrap()
            .resolve()
            .unwrap();
        spec.captures[0].tokens = TokenSelector::ByteSpan {
            start: 0,
            end: 4,
            subtoken_selection: crate::v05::token_select::SubtokenSelection::All,
        };
        spec.inputs.push(InputSpec {
            id: "unused-empty".into(),
            text: String::new(),
        });
        spec.captures[0].inputs =
            crate::v05::capture::InputSelector::List(vec!["example-001".into()]);
        assert!(spec.validate().is_ok());
        spec.captures[0].inputs = crate::v05::capture::InputSelector::All("all".into());
        assert!(spec
            .validate()
            .unwrap_err()
            .message
            .contains("unused-empty"));
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
