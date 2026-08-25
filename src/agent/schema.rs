//! Generic tool argument schemas (Track A1).
//!
//! A deliberately small JSON-Schema-*compatible* subset: enough to render
//! tool definitions for a model prompt and to validate model-produced
//! arguments strictly, far short of a complete JSON Schema implementation.
//!
//! Supported: `string`, `number` (+ `integer`), `boolean`, `array`
//! (typed items), nested `object` (strict or permissive), `enum`
//! (string values), optional fields (`required`). Everything else is
//! expressed through these.
//!
//! Validation is strict by design: unknown fields are rejections, not
//! warnings ("no silent parser recovery").

use serde::{Deserialize, Serialize};
use std::fmt;

/// Maximum nesting depth accepted for argument objects/arrays. Keeps a
/// hostile schema/model pair from blowing the stack during validation.
pub const MAX_VALIDATION_DEPTH: usize = 32;

/// JSON value types supported by [`ParamSchema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JsonType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

impl JsonType {
    pub fn as_str(self) -> &'static str {
        match self {
            JsonType::String => "string",
            JsonType::Number => "number",
            JsonType::Integer => "integer",
            JsonType::Boolean => "boolean",
            JsonType::Array => "array",
            JsonType::Object => "object",
        }
    }

    /// Set-membership test used by validation diagnostics.
    pub fn matches_json_value(self, value: &serde_json::Value) -> bool {
        match self {
            JsonType::String => value.is_string(),
            JsonType::Number => value.is_number(),
            // integer accepts integral numbers rendered either way
            JsonType::Integer => match value {
                serde_json::Value::Number(n) => n.is_i64() || n.is_u64() || is_integral_f64(n),
                _ => false,
            },
            JsonType::Boolean => value.is_boolean(),
            JsonType::Array => value.is_array(),
            JsonType::Object => value.is_object(),
        }
    }
}

fn is_integral_f64(n: &serde_json::Number) -> bool {
    n.as_f64()
        .is_some_and(|f| f.fract() == 0.0 && f.is_finite())
}

/// One named parameter. Nested object `properties` reuse this type, so
/// objects nest arbitrarily (bounded by [`MAX_VALIDATION_DEPTH`] during
/// validation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSchema {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub ty: JsonType,
    #[serde(default)]
    pub required: bool,
    /// Item schema for `Array`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<ParamSchema>>,
    /// Property schemas for `Object`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<ParamSchema>,
    /// Strict objects reject unknown keys (default true).
    #[serde(default = "default_true")]
    pub additional_properties: bool,
    /// Allowed string values (string enums).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

fn default_true() -> bool {
    true
}

impl ParamSchema {
    pub fn new(name: impl Into<String>, ty: JsonType) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            ty,
            required: false,
            items: None,
            properties: Vec::new(),
            additional_properties: true,
            enum_values: None,
        }
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    pub fn describe(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn item(mut self, item: ParamSchema) -> Self {
        self.items = Some(Box::new(item));
        self
    }

    /// Adds a strict sub-property (unknown keys inside this object become
    /// validation errors unless further properties opt out).
    pub fn property(mut self, property: ParamSchema) -> Self {
        self.properties.push(property);
        self.additional_properties = false;
        self
    }

    pub fn one_of(mut self, values: &[&str]) -> Self {
        self.enum_values = Some(values.iter().map(|v| (*v).to_string()).collect());
        self
    }

    /// JSON-Schema-compatible rendering of this parameter.
    pub fn to_json_schema(&self) -> serde_json::Value {
        let mut obj = serde_json::json!({ "type": self.ty.as_str() });
        if !self.description.is_empty() {
            obj["description"] = serde_json::json!(self.description);
        }
        if let Some(items) = &self.items {
            obj["items"] = items.to_json_schema();
        }
        if !self.properties.is_empty() {
            let mut props = serde_json::Map::new();
            for p in &self.properties {
                props.insert(p.name.clone(), p.to_json_schema());
            }
            obj["properties"] = serde_json::Value::Object(props);
            obj["additionalProperties"] = serde_json::json!(self.additional_properties);
        }
        if let Some(values) = &self.enum_values {
            obj["enum"] = serde_json::json!(values);
        }
        obj
    }
}

/// Complete tool contract exposed to both the model prompt and validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// Top-level parameters of the tool's object argument.
    #[serde(default)]
    pub parameters: Vec<ParamSchema>,
    /// Risk classification preserved for future approval policies
    /// (Track H); Phase 1 records it, nothing gates on it.
    #[serde(default)]
    pub effect: ToolEffect,
}

/// What executing this tool can change. Recorded in traces and schema
/// snapshots; Phase 1 executes deterministic built-ins automatically and
/// never registers `ExternalSideEffect` built-ins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ToolEffect {
    /// No observable state change outside the trace.
    #[default]
    ReadOnly,
    /// Writes local files (artifact store).
    LocalWrite,
    /// Touches external systems (out of Phase 1 scope for built-ins).
    ExternalSideEffect,
}

impl ToolEffect {
    pub fn as_str(self) -> &'static str {
        match self {
            ToolEffect::ReadOnly => "read_only",
            ToolEffect::LocalWrite => "local_write",
            ToolEffect::ExternalSideEffect => "external_side_effect",
        }
    }
}

impl ToolSchema {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
            effect: ToolEffect::ReadOnly,
        }
    }

    pub fn param(mut self, param: ParamSchema) -> Self {
        self.parameters.push(param);
        self
    }

    pub fn effect(mut self, effect: ToolEffect) -> Self {
        self.effect = effect;
        self
    }

    /// JSON-Schema-compatible document for model prompting:
    /// `{ "name", "description", "parameters": {object schema} }`.
    pub fn to_json_schema(&self) -> serde_json::Value {
        let mut props = serde_json::Map::new();
        let mut required = Vec::new();
        for p in &self.parameters {
            if p.required {
                required.push(serde_json::json!(p.name));
            }
            props.insert(p.name.clone(), p.to_json_schema());
        }
        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": serde_json::Value::Object(props),
                "required": required,
            },
        })
    }
}

/// Structured validation rejection: one field-level problem each. A full
/// validation collects every problem rather than failing on the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    /// Dot path into the arguments object ("" = top level).
    pub path: String,
    pub kind: ValidationErrorKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorKind {
    MissingRequired,
    UnknownField,
    WrongType,
    EnumViolation,
    DepthExceeded,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            write!(f, "{}", self.message)
        } else {
            write!(f, "`{}`: {}", self.path, self.message)
        }
    }
}

impl std::error::Error for ValidationError {}

/// All problems found while validating one tool-call argument object.
#[derive(Debug, Clone)]
pub struct ArgumentErrors {
    pub tool_name: String,
    pub errors: Vec<ValidationError>,
}

impl fmt::Display for ArgumentErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid arguments for tool `{}`:", self.tool_name)?;
        for e in &self.errors {
            write!(f, "\n  - {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ArgumentErrors {}

/// Model output that failed JSON parsing before any schema work.
#[derive(Debug, Clone)]
pub struct MalformedJson {
    pub tool_name: String,
    pub source_text_excerpt: String,
    pub message: String,
}

impl fmt::Display for MalformedJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "malformed JSON arguments for tool `{}`: {}",
            self.tool_name, self.message
        )
    }
}

impl std::error::Error for MalformedJson {}

/// Why validating a raw call failed. Exactly one variant applies.
#[derive(Debug, Clone)]
pub enum ArgumentError {
    MalformedJson(MalformedJson),
    Schema(ArgumentErrors),
}

impl fmt::Display for ArgumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArgumentError::MalformedJson(e) => write!(f, "{e}"),
            ArgumentError::Schema(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ArgumentError {}

/// Validated, canonical arguments ready for execution. Construction is the
/// only way in — raw model text can never reach a tool unvalidated.
#[derive(Debug, Clone)]
pub struct ValidatedArguments {
    tool_name: String,
    value: serde_json::Value,
}

impl ValidatedArguments {
    /// Validate `raw` against `schema`. Collects all schema violations.
    pub fn parse(schema: &ToolSchema, raw: &str) -> Result<Self, ArgumentError> {
        let parsed: serde_json::Value = serde_json::from_str(raw).map_err(|e| {
            let excerpt: String = raw.chars().take(160).collect();
            ArgumentError::MalformedJson(MalformedJson {
                tool_name: schema.name.clone(),
                source_text_excerpt: excerpt,
                message: e.to_string(),
            })
        })?;
        Self::from_value(schema, parsed)
    }

    /// Validate an already-parsed JSON value against `schema`.
    pub fn from_value(
        schema: &ToolSchema,
        value: serde_json::Value,
    ) -> Result<Self, ArgumentError> {
        let mut errors = Vec::new();
        validate_object(schema, &value, "", MAX_VALIDATION_DEPTH, &mut errors);
        if !errors.is_empty() {
            return Err(ArgumentError::Schema(ArgumentErrors {
                tool_name: schema.name.clone(),
                errors,
            }));
        }
        Ok(Self {
            tool_name: schema.name.clone(),
            value,
        })
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Canonical JSON encoding used by traces and reinjection.
    pub fn to_json(&self) -> &serde_json::Value {
        &self.value
    }

    /// Compact canonical serialization (stable key order from serde_json's
    /// BTree-backed Map).
    pub fn serialize_compact(&self) -> String {
        serde_json::to_string(&self.value).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.value.get(key)
    }
}

fn validate_object(
    schema: &ToolSchema,
    value: &serde_json::Value,
    path: &str,
    depth: usize,
    errors: &mut Vec<ValidationError>,
) {
    // The TOP-LEVEL arguments object is always strict (unknown fields are
    // rejections); nested objects honor their param's
    // `additional_properties` via the synthetic-schema trick below.
    validate_object_inner(schema, value, path, depth, true, errors)
}

fn validate_object_inner(
    schema: &ToolSchema,
    value: &serde_json::Value,
    path: &str,
    depth: usize,
    reject_unknown: bool,
    errors: &mut Vec<ValidationError>,
) {
    let serde_json::Value::Object(map) = value else {
        errors.push(ValidationError {
            path: path.to_string(),
            kind: ValidationErrorKind::WrongType,
            message: format!("expected an object, got {}", json_kind_name(value)),
        });
        return;
    };
    if depth == 0 {
        errors.push(ValidationError {
            path: path.to_string(),
            kind: ValidationErrorKind::DepthExceeded,
            message: format!("nesting exceeds {MAX_VALIDATION_DEPTH} levels"),
        });
        return;
    }
    for param in &schema.parameters {
        let child_path = join_path(path, &param.name);
        match map.get(&param.name) {
            None => {
                if param.required {
                    errors.push(ValidationError {
                        path: child_path,
                        kind: ValidationErrorKind::MissingRequired,
                        message: format!("missing required argument `{}`", param.name),
                    });
                }
            }
            Some(v) => validate_param(param, v, &child_path, depth, errors),
        }
    }
    if !reject_unknown {
        return;
    }
    for key in map.keys() {
        if !schema.parameters.iter().any(|p| &p.name == key) {
            errors.push(ValidationError {
                path: join_path(path, key),
                kind: ValidationErrorKind::UnknownField,
                message: format!("unknown field `{key}`"),
            });
        }
    }
}

fn validate_param(
    param: &ParamSchema,
    value: &serde_json::Value,
    path: &str,
    depth: usize,
    errors: &mut Vec<ValidationError>,
) {
    if depth == 0 {
        errors.push(ValidationError {
            path: path.to_string(),
            kind: ValidationErrorKind::DepthExceeded,
            message: format!("nesting exceeds {MAX_VALIDATION_DEPTH} levels"),
        });
        return;
    }
    if !param.ty.matches_json_value(value) {
        errors.push(ValidationError {
            path: path.to_string(),
            kind: ValidationErrorKind::WrongType,
            message: format!(
                "expected {}, got {}",
                param.ty.as_str(),
                json_kind_name(value)
            ),
        });
        return;
    }
    if let Some(expected) = &param.enum_values {
        let ok = value
            .as_str()
            .is_some_and(|s| expected.contains(&s.to_string()));
        if !ok {
            errors.push(ValidationError {
                path: path.to_string(),
                kind: ValidationErrorKind::EnumViolation,
                message: format!("expected one of {expected:?}, got {value}"),
            });
            return;
        }
    }
    if let (Some(item_schema), serde_json::Value::Array(items)) = (&param.items, value) {
        for (i, item) in items.iter().enumerate() {
            let child_path = format!("{path}[{i}]");
            validate_param(item_schema, item, &child_path, depth - 1, errors);
        }
    }
    if !param.properties.is_empty() {
        let pseudo = pseudo_schema(param);
        validate_object_inner(
            &pseudo,
            value,
            path,
            depth - 1,
            !param.additional_properties,
            errors,
        );
    }
}

/// Reuse the object validator for nested params by lifting them into a
/// synthetic schema whose `parameters` are the nested `properties`.
fn pseudo_schema(param: &ParamSchema) -> ToolSchema {
    ToolSchema {
        name: param.name.clone(),
        description: String::new(),
        parameters: param.properties.clone(),
        effect: ToolEffect::ReadOnly,
    }
}

fn join_path(parent: &str, key: &str) -> String {
    if parent.is_empty() {
        key.to_string()
    } else {
        format!("{parent}.{key}")
    }
}

fn json_kind_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn calc_schema() -> ToolSchema {
        ToolSchema::new("calc", "arithmetic")
            .param(
                ParamSchema::new("op", JsonType::String)
                    .required()
                    .one_of(&["add", "mul"]),
            )
            .param(ParamSchema::new("a", JsonType::Integer).required())
            .param(ParamSchema::new("b", JsonType::Number))
    }

    #[test]
    fn accepts_valid_args() {
        let v = ValidatedArguments::parse(&calc_schema(), r#"{"op":"add","a":2,"b":3.5}"#).unwrap();
        assert_eq!(v.get("a"), Some(&json!(2)));
        assert_eq!(
            serde_json::to_value(v.to_json()).unwrap(),
            json!({"op":"add","a":2,"b":3.5})
        );
    }

    #[test]
    fn collects_all_schema_violations() {
        let err = ValidatedArguments::parse(&calc_schema(), r#"{"op":"div","zz":1}"#).unwrap_err();
        let ArgumentError::Schema(errors) = err else {
            panic!("expected schema errors");
        };
        let mut kinds: Vec<_> = errors.errors.iter().map(|e| e.kind).collect();
        kinds.sort_unstable_by_key(|k| format!("{k:?}"));
        let expected = vec![
            ValidationErrorKind::EnumViolation,   // op=div
            ValidationErrorKind::MissingRequired, // a absent
            ValidationErrorKind::UnknownField,    // zz
        ];
        let mut expected = expected;
        expected.sort_unstable_by_key(|k| format!("{k:?}"));
        assert_eq!(kinds, expected);
    }

    #[test]
    fn rejects_malformed_json_with_structured_error() {
        let err = ValidatedArguments::parse(&calc_schema(), "{\"op\": ").unwrap_err();
        let ArgumentError::MalformedJson(m) = err else {
            panic!("expected malformed json");
        };
        assert_eq!(m.tool_name, "calc");
        assert!(!m.source_text_excerpt.is_empty());
    }

    #[test]
    fn integer_rejects_fractional_number_and_bool() {
        for raw in [r#"{"op":"add","a":1.5}"#, r#"{"op":"add","a":true}"#] {
            let err = ValidatedArguments::parse(&calc_schema(), raw).unwrap_err();
            let ArgumentError::Schema(errors) = err else {
                panic!("expected schema errors");
            };
            assert_eq!(errors.errors.len(), 1);
            assert_eq!(errors.errors[0].path, "a");
            assert_eq!(errors.errors[0].kind, ValidationErrorKind::WrongType);
        }
    }

    #[test]
    fn integer_accepts_integral_float_rendering() {
        let v = ValidatedArguments::parse(&calc_schema(), r#"{"op":"add","a":2.0}"#).unwrap();
        assert_eq!(v.get("a"), Some(&serde_json::json!(2.0)));
    }

    #[test]
    fn nested_objects_and_arrays_validate_recursively() {
        let schema = ToolSchema::new("t", "nested").param(
            ParamSchema::new("rows", JsonType::Array).required().item(
                ParamSchema::new("", JsonType::Object)
                    .property(ParamSchema::new("id", JsonType::Integer).required()),
            ),
        );
        let ok = ValidatedArguments::parse(&schema, r#"{"rows":[{"id":1},{"id":2}]}"#).unwrap();
        assert_eq!(
            ok.get("rows").and_then(|r| r.as_array()).map(Vec::len),
            Some(2)
        );
        let bad = ValidatedArguments::parse(&schema, r#"{"rows":[{"id":"x"}]}"#).unwrap_err();
        let ArgumentError::Schema(errors) = bad else {
            panic!("expected schema errors");
        };
        assert_eq!(errors.errors[0].path, "rows[0].id");
    }

    #[test]
    fn depth_beyond_limit_is_rejected_not_a_stack_overflow() {
        // 40 nested array levels declared and provided; the walk must bail
        // with a structured DepthExceeded well before the stack is at risk.
        let mut item = ParamSchema::new("", JsonType::Number);
        for _ in 0..40 {
            item = ParamSchema::new("", JsonType::Array).item(item);
        }
        let schema = ToolSchema::new("deep", "")
            .param(ParamSchema::new("v", JsonType::Array).required().item(item));
        let mut raw = String::new();
        for _ in 0..40 {
            raw.push('[');
        }
        raw.push('0');
        for _ in 0..40 {
            raw.push(']');
        }
        let wrapped = format!(r#"{{"v":{raw}}}"#);
        let err = ValidatedArguments::parse(&schema, &wrapped).unwrap_err();
        let ArgumentError::Schema(errors) = err else {
            panic!("expected schema errors");
        };
        assert!(errors
            .errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::DepthExceeded));
    }

    #[test]
    fn json_schema_rendering_is_round_trip_serializable() {
        let rendered = calc_schema().to_json_schema();
        assert_eq!(rendered["parameters"]["type"], "object");
        assert_eq!(rendered["parameters"]["required"], json!(["op", "a"]));
        assert_eq!(
            rendered["parameters"]["properties"]["op"]["enum"],
            json!(["add", "mul"])
        );
    }
}

/// Deterministic compact JSON encoding with lexicographically sorted
/// object keys, independent of `serde_json` feature flags.
///
/// Why not `serde_json::to_string`: another dependency can enable
/// `preserve_order`, switching `Map` from BTree- to IndexMap-backed and
/// silently changing serialized key order PER PLATFORM. Tool definitions
/// reach the model as prompt bytes and are pinned by golden tests, so
/// their encoding must not depend on the feature graph. This walker
/// sorts every object it encounters and delegates scalar escaping to
/// serde_json.
pub fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

/// [`canonical_json`] with two-space indentation (same ordering rules).
pub fn canonical_json_pretty(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical_pretty(value, 0, &mut out);
    out
}

fn write_canonical(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).expect("string escapes"));
                out.push(':');
                write_canonical(&map[*key], out);
            }
            out.push('}');
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap_or_else(|_| "null".to_string())),
    }
}

fn write_canonical_pretty(value: &serde_json::Value, depth: usize, out: &mut String) {
    let pad = |n: usize, out: &mut String| {
        for _ in 0..n {
            out.push_str("  ");
        }
    };
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            if keys.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                pad(depth + 1, out);
                out.push_str(&serde_json::to_string(key).expect("string escapes"));
                out.push_str(": ");
                write_canonical_pretty(&map[*key], depth + 1, out);
            }
            out.push('\n');
            pad(depth, out);
            out.push('}');
        }
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                pad(depth + 1, out);
                write_canonical_pretty(item, depth + 1, out);
            }
            out.push('\n');
            pad(depth, out);
            out.push(']');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap_or_else(|_| "null".to_string())),
    }
}

#[cfg(test)]
mod canonical_tests {
    use super::*;

    /// Pins the platform-independent encoding: sorted keys, compact
    /// separators. Guards against serde_json `preserve_order` feature
    /// unification changing byte output per platform (caught on aarch64
    /// CI during the v0.6.7 release).
    #[test]
    fn canonical_json_sorts_keys_recursively_and_is_platform_stable() {
        let v = serde_json::json!({
            "zeta": 1,
            "alpha": {"y": [3, {"b": true, "a": null}], "x": "s\"q"},
            "mid": []
        });
        assert_eq!(
            canonical_json(&v),
            r#"{"alpha":{"x":"s\"q","y":[3,{"a":null,"b":true}]},"mid":[],"zeta":1}"#
        );
        let pretty = canonical_json_pretty(&v);
        assert!(pretty.starts_with("{\n  \"alpha\": {"));
        assert_eq!(
            pretty,
            concat!(
                "{\n",
                "  \"alpha\": {\n",
                "    \"x\": \"s\\\"q\",\n",
                "    \"y\": [\n",
                "      3,\n",
                "      {\n",
                "        \"a\": null,\n",
                "        \"b\": true\n",
                "      }\n",
                "    ]\n",
                "  },\n",
                "  \"mid\": [],\n",
                "  \"zeta\": 1\n",
                "}"
            )
        );
    }
}
