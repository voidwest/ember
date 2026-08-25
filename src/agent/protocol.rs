//! Model-specific tool-call protocols behind one boundary (Track C).
//!
//! The agent loop never learns how Llama/Qwen/Gemma serializes a tool
//! decision. It drives [`ToolCallProtocol`]: render messages, parse the
//! assistant turn into an explicit [`AssistantAction`], render results
//! back. Parsing is explicit — a present-but-broken tool call is a
//! structured [`AssistantAction::MalformedToolCall`], never silently
//! recovered into plain text ("no silent parser recovery").
//!
//! Phase 1 ships three codecs:
//!
//! - [`Qwen25ToolProtocol`] — Qwen2.5 ChatML + Hermes `<tool_call>` JSON
//!   (the official Qwen2.5 convention);
//! - [`LlamaToolProtocol`] — Llama 3.x header scaffold + `<|python_tag|>`
//!   JSON custom-function calling (the Meta zero-shot convention);
//! - [`EmberJsonToolProtocol`] — Ember's own generic JSON mode
//!   (`{"type":"tool_call",...}`), useful for tests and models prompted
//!   to follow it. Not presented as native support for every model.
//!
//! Phase 1 limit: ONE tool call per assistant step is parsed (the first).
//! Multi-call steps are recorded as such in traces; fan-out is future work.

use super::schema::ToolSchema;

/// What the model chose to do this turn, extracted explicitly.
#[derive(Debug, Clone, PartialEq)]
pub enum AssistantAction {
    /// Plain completion; ends the run.
    FinalText(String),
    /// One or more tool decisions: names plus RAW argument JSON. Schema
    /// validation happens later, against the registry - parsing never
    /// validates. Calls execute in order; the session enforces limits
    /// between them.
    ToolCalls(Vec<RawToolCall>),
    /// Tool-call syntax was present but could not be parsed. Explicitly
    /// distinct from `FinalText` so the loop can reject it loudly.
    MalformedToolCall { excerpt: String, reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawToolCall {
    pub name: String,
    /// Raw argument JSON exactly as extracted (validated downstream).
    pub arguments_json: String,
}

impl RawToolCall {
    pub fn new(name: impl Into<String>, arguments_json: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments_json: arguments_json.into(),
        }
    }
}

/// The protocol boundary. All rendered strings are COMPLETE messages or
/// scaffolding pieces the session commits verbatim through the tokenizer.
pub trait ToolCallProtocol: Send + Sync {
    /// Stable identifier recorded in trace provenance.
    fn id(&self) -> &'static str;

    /// Complete system message (role wrapper included) carrying the base
    /// instructions plus the tool definitions in this family's dialect.
    fn render_system_message(&self, base: Option<&str>, tools: &[ToolSchema]) -> String;

    /// Complete user message (role wrapper included).
    fn render_user_message(&self, content: &str) -> String;

    /// Scaffold committed before generation begins (speculatively; rolled
    /// back on cancellation).
    fn render_assistant_prefix(&self) -> String;

    /// Terminal scaffold appended when an assistant turn commits.
    fn render_assistant_suffix(&self) -> String;

    /// Parse a finished generation into an action. Total function: never
    /// panics; falls back to FinalText only when NO tool-call syntax is
    /// present.
    fn parse_assistant_output(&self, raw: &str) -> AssistantAction;

    /// Complete message committing one tool result (role wrapper
    /// included). `result.error` marks failed executions so the model can
    /// distinguish outcomes.
    fn render_tool_result_message(&self, result: &ToolResultMessage<'_>) -> String;

    /// Strings that end generation immediately when decoded (stripped
    /// from the committed text). Token-level eos still applies first.
    fn stop_strings(&self) -> Vec<String> {
        Vec::new()
    }

    /// Additional special-token literals treated as eos for agent turns
    /// (on top of the tokenizer's own set).
    fn extra_eos_tokens(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Canonical view of one executed tool call, protocol-rendered on commit.
#[derive(Debug, Clone)]
pub struct ToolResultMessage<'a> {
    pub call_name: &'a str,
    pub ok: bool,
    /// Compact JSON payload: text results arrive as a JSON string.
    pub content_json: serde_json::Value,
}

impl<'a> ToolResultMessage<'a> {
    pub fn from_text(call_name: &'a str, ok: bool, text: &str) -> Self {
        Self {
            call_name,
            ok,
            content_json: serde_json::Value::String(text.to_string()),
        }
    }

    /// Compact serialization used inside rendered messages. Canonical
    /// (sorted keys) so prompt bytes are platform-stable.
    pub fn content_compact(&self) -> String {
        super::schema::canonical_json(&self.content_json)
    }
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// Find the first balanced JSON object in `s` at/after `from`, respecting
/// string literals and escapes. Returns (start, end_exclusive).
pub(crate) fn find_balanced_json_object(s: &str, from: usize) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    let mut i = from;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        let start = i;
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;
        while i < bytes.len() {
            let b = bytes[i];
            if in_string {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_string = false;
                }
            } else {
                match b {
                    b'"' => in_string = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            return Some((start, i + 1));
                        }
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        return None; // unbalanced object: no recovery past it
    }
    None
}

/// Extract `{"name": ..., "parameters"|"arguments": {...}}` fields from a
/// parsed object. Returns Err(reason) on structural violations.
pub(crate) fn extract_call_fields(obj: &serde_json::Value) -> Result<(String, String), String> {
    let Some(map) = obj.as_object() else {
        return Err("call payload is not a JSON object".to_string());
    };
    let Some(name) = map.get("name").and_then(|v| v.as_str()) else {
        return Err("missing string field `name`".to_string());
    };
    if name.is_empty() {
        return Err("empty tool name".to_string());
    }
    let args_value = map
        .get("arguments")
        .or_else(|| map.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
    if !args_value.is_object() {
        return Err(format!("`arguments` must be an object, got {}", args_value));
    }
    let arguments_json = super::schema::canonical_json(&args_value);
    Ok((name.to_string(), arguments_json))
}

// ---------------------------------------------------------------------------
// Qwen2.5 (ChatML + Hermes <tool_call>)
// ---------------------------------------------------------------------------

/// Qwen2.5 tool-calling dialect.
///
/// ```text
/// <|im_start|>system\n{instructions + <tools> JSON}<|im_end|>\n
/// <|im_start|>user\n{content}<|im_end|>\n
/// <|im_start|>assistant\n<tool_call>\n{"name":...,"arguments":{...}}\n</tool_call><|im_end|>
/// <|im_start|>user\n<tool_response>\n{"name":...,"content":...}\n</tool_response><|im_end|>\n
/// ```
pub struct Qwen25ToolProtocol {
    /// Base instruction block placed before the tool definitions.
    pub base_instruction: String,
}

impl Default for Qwen25ToolProtocol {
    fn default() -> Self {
        Self {
            base_instruction: "You are a helpful assistant with access to the \
                               following tools. When you need information from a \
                               tool, call it instead of guessing."
                .to_string(),
        }
    }
}

const QWEN_TOOL_CALL_OPEN: &str = "<tool_call>";
const QWEN_TOOL_CALL_CLOSE: &str = "</tool_call>";

impl Qwen25ToolProtocol {
    fn open_role(&self, role: &str) -> String {
        format!("<|im_start|>{role}\n")
    }

    fn close_role(&self) -> String {
        "<|im_end|>\n".to_string()
    }
}

impl ToolCallProtocol for Qwen25ToolProtocol {
    fn id(&self) -> &'static str {
        "qwen2.5-tool-call-v1"
    }

    fn render_system_message(&self, base: Option<&str>, tools: &[ToolSchema]) -> String {
        let mut s = self.open_role("system");
        s.push_str(base.unwrap_or(&self.base_instruction));
        if !tools.is_empty() {
            s.push_str("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
            for tool in tools {
                s.push_str(&super::schema::canonical_json(&tool.to_json_schema()));
                s.push('\n');
            }
            s.push_str("</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name-in-string>, \"arguments\": <args-json-object>}\n</tool_call>");
        }
        s.push_str(&self.close_role());
        s
    }

    fn render_user_message(&self, content: &str) -> String {
        let mut s = self.open_role("user");
        s.push_str(content);
        s.push_str(&self.close_role());
        s
    }

    fn render_assistant_prefix(&self) -> String {
        self.open_role("assistant")
    }

    fn render_assistant_suffix(&self) -> String {
        // terminal <|im_end|> without trailing newline (nothing follows yet)
        "<|im_end|>".to_string()
    }

    fn parse_assistant_output(&self, raw: &str) -> AssistantAction {
        let Some(open_pos) = raw.find(QWEN_TOOL_CALL_OPEN) else {
            return AssistantAction::FinalText(raw.trim().to_string());
        };
        // Every <tool_call> block parses or the whole step is malformed:
        // partial acceptance would silently drop requested side effects.
        let mut cursor = 0usize;
        let mut calls: Vec<RawToolCall> = Vec::new();
        while let Some(open_rel) = raw[cursor..].find(QWEN_TOOL_CALL_OPEN) {
            let after_open = cursor + open_rel + QWEN_TOOL_CALL_OPEN.len();
            let tail = &raw[after_open..];
            let (body, next) = match tail.find(QWEN_TOOL_CALL_CLOSE) {
                Some(p) => (&tail[..p], after_open + p + QWEN_TOOL_CALL_CLOSE.len()),
                None => (tail, raw.len()), // unclosed tag: still explicit intent
            };
            match serde_json::from_str::<serde_json::Value>(body.trim())
                .map_err(|e| format!("invalid JSON inside <tool_call>: {e}"))
                .and_then(|v| extract_call_fields(&v))
            {
                Ok((name, arguments_json)) => {
                    calls.push(RawToolCall::new(name, arguments_json));
                }
                Err(reason) => {
                    return AssistantAction::MalformedToolCall {
                        excerpt: body.trim().chars().take(160).collect(),
                        reason,
                    };
                }
            }
            cursor = next;
        }
        if calls.is_empty() {
            // marker existed (checked above) but nothing parsable inside
            return AssistantAction::MalformedToolCall {
                excerpt: raw[open_pos..].chars().take(160).collect(),
                reason: "<tool_call> marker present but no call could be parsed".to_string(),
            };
        }
        AssistantAction::ToolCalls(calls)
    }

    fn render_tool_result_message(&self, result: &ToolResultMessage<'_>) -> String {
        let payload = serde_json::json!({
            "name": result.call_name,
            "content": result.content_json,
        });
        let mut s = self.open_role("user");
        s.push_str("<tool_response>\n");
        s.push_str(&serde_json::to_string(&payload).unwrap_or_else(|_| "\"\"".to_string()));
        s.push('\n');
        s.push_str("</tool_response>");
        s.push_str(&self.close_role());
        s
    }

    fn stop_strings(&self) -> Vec<String> {
        vec![QWEN_TOOL_CALL_CLOSE.to_string()]
    }
}

// ---------------------------------------------------------------------------
// Llama 3.x (header scaffold + <|python_tag|> JSON)
// ---------------------------------------------------------------------------

/// Llama 3.1/3.2 custom-function calling dialect.
///
/// System carries the Meta tool instructions + JSON schema array; a tool
/// decision is `<|python_tag|>{"name":...,"parameters":{...}}` closed by
/// `<|eom_id|>`; results enter as an `ipython` role message.
pub struct LlamaToolProtocol {
    /// Rendered into the system header (deterministic by contract).
    pub today_date: String,
    /// Base system instruction placed after the date lines.
    pub base_instruction: String,
}

impl Default for LlamaToolProtocol {
    fn default() -> Self {
        Self {
            today_date: "01 Jan 2026".to_string(),
            base_instruction: "You are a helpful assistant.".to_string(),
        }
    }
}

const LLAMA_BOS: &str = "<|begin_of_text|>";
const LLAMA_EOT: &str = "<|eot_id|>";
const LLAMA_EOL_ID: &str = "<|eom_id|>";

impl LlamaToolProtocol {
    fn open_header(&self, role: &str) -> String {
        format!("<|start_header_id|>{role}<|end_header_id|>\n\n")
    }
}

impl ToolCallProtocol for LlamaToolProtocol {
    fn id(&self) -> &'static str {
        "llama3-python-tag-v1"
    }

    fn render_system_message(&self, base: Option<&str>, tools: &[ToolSchema]) -> String {
        let mut s = String::from(LLAMA_BOS);
        s.push_str(&self.open_header("system"));
        s.push_str(&format!(
            "Environment: ipython\nCutting Knowledge Date: December 2023\nToday Date: {}\n\n",
            self.today_date
        ));
        s.push_str(base.unwrap_or(&self.base_instruction));
        if !tools.is_empty() {
            let schemas: Vec<serde_json::Value> =
                tools.iter().map(|t| t.to_json_schema()).collect();
            s.push_str("\n\nYou have access to the following functions. To call a function, respond with JSON for a function call. Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n\n");
            let canonical_array =
                super::schema::canonical_json_pretty(&serde_json::Value::Array(schemas));
            s.push_str(&canonical_array);
        }
        s.push_str(LLAMA_EOT);
        s
    }

    fn render_user_message(&self, content: &str) -> String {
        let mut s = self.open_header("user");
        s.push_str(content);
        s.push_str(LLAMA_EOT);
        s
    }

    fn render_assistant_prefix(&self) -> String {
        self.open_header("assistant")
    }

    fn render_assistant_suffix(&self) -> String {
        LLAMA_EOT.to_string()
    }

    fn parse_assistant_output(&self, raw: &str) -> AssistantAction {
        let had_tag = raw.contains("<|python_tag|>");
        let stripped = raw.replace("<|python_tag|>", "").trim().to_string();

        // Whole-message JSON is the official shape.
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&stripped)
            && let Ok((name, arguments_json)) = extract_call_fields(&value)
        {
            return AssistantAction::ToolCalls(vec![RawToolCall::new(name, arguments_json)]);
        }

        // Fallback: first balanced object (models sometimes add prose).
        if let Some((start, end)) = find_balanced_json_object(&stripped, 0)
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(&stripped[start..end])
            && let Ok((name, arguments_json)) = extract_call_fields(&value)
        {
            return AssistantAction::ToolCalls(vec![RawToolCall::new(name, arguments_json)]);
        }

        if had_tag {
            // Intent was declared; do NOT silently downgrade to prose.
            AssistantAction::MalformedToolCall {
                excerpt: stripped.chars().take(160).collect(),
                reason: "`<|python_tag|>` present but no parsable {\"name\",...} call followed"
                    .to_string(),
            }
        } else {
            AssistantAction::FinalText(stripped)
        }
    }

    fn render_tool_result_message(&self, result: &ToolResultMessage<'_>) -> String {
        // The reference inserts the function output verbatim under the
        // `ipython` role; structured payloads serialize compactly.
        let mut s = self.open_header("ipython");
        s.push_str(&result.content_compact());
        s.push_str(LLAMA_EOT);
        s
    }

    fn stop_strings(&self) -> Vec<String> {
        vec![LLAMA_EOL_ID.to_string(), LLAMA_EOT.to_string()]
    }

    fn extra_eos_tokens(&self) -> Vec<String> {
        vec![LLAMA_EOL_ID.to_string(), LLAMA_EOT.to_string()]
    }
}

// ---------------------------------------------------------------------------
// Ember generic JSON mode
// ---------------------------------------------------------------------------

/// Ember's own generic structured mode: ChatML scaffold, and a tool call
/// is any JSON object of the form
/// `{"type":"tool_call","name":"...","arguments":{...}}` appearing alone
/// or embedded in the reply. Honest scope: a testing/interop protocol,
/// NOT native tool support for arbitrary models.
pub struct EmberJsonToolProtocol {
    pub base_instruction: String,
}

impl Default for EmberJsonToolProtocol {
    fn default() -> Self {
        Self {
            base_instruction: "You are a helpful assistant with access to the \
                               following tools."
                .to_string(),
        }
    }
}

const GENERIC_TYPE_TAG: &str = "\"type\"";
const GENERIC_CALL_MARKER: &str = "tool_call";

impl EmberJsonToolProtocol {
    fn looks_like_call(value: &serde_json::Value) -> bool {
        value.get("type").and_then(|t| t.as_str()) == Some("tool_call")
    }
}

impl ToolCallProtocol for EmberJsonToolProtocol {
    fn id(&self) -> &'static str {
        "ember-generic-json-v1"
    }

    fn render_system_message(&self, base: Option<&str>, tools: &[ToolSchema]) -> String {
        let mut s = String::from("<|im_start|>system\n");
        s.push_str(base.unwrap_or(&self.base_instruction));
        if !tools.is_empty() {
            let schemas: Vec<serde_json::Value> =
                tools.iter().map(|t| t.to_json_schema()).collect();
            s.push_str("\n\nTo call a tool, emit exactly this JSON object (and nothing else required):\n{\"type\":\"tool_call\",\"name\":<tool-name>,\"arguments\":{...}}\n\nAvailable tools (JSON schemas):\n");
            let canonical_array =
                super::schema::canonical_json_pretty(&serde_json::Value::Array(schemas));
            s.push_str(&canonical_array);
        } else {
            s.push_str("\n\nNo tools are available; answer in plain text.");
        }
        s.push_str("\n<|im_end|>\n");
        s
    }

    fn render_user_message(&self, content: &str) -> String {
        format!("<|im_start|>user\n{content}<|im_end|>\n")
    }

    fn render_assistant_prefix(&self) -> String {
        "<|im_start|>assistant\n".to_string()
    }

    fn render_assistant_suffix(&self) -> String {
        "<|im_end|>".to_string()
    }

    fn parse_assistant_output(&self, raw: &str) -> AssistantAction {
        // whole-text first
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
            if Self::looks_like_call(&value) {
                return match extract_call_fields(&value) {
                    Ok((name, arguments_json)) => {
                        AssistantAction::ToolCalls(vec![RawToolCall::new(name, arguments_json)])
                    }
                    Err(reason) => AssistantAction::MalformedToolCall {
                        excerpt: raw.trim().chars().take(160).collect(),
                        reason,
                    },
                };
            }
            // a plain JSON value that is NOT a call: surface as text
            return AssistantAction::FinalText(raw.trim().to_string());
        }
        // Embedded calls COLLECT (multi-call steps are first-class);
        // any candidate that looks like a call but fails to parse is an
        // explicit MalformedToolCall — never silently dropped into prose.
        let mut search_from = 0usize;
        let mut calls: Vec<RawToolCall> = Vec::new();
        if find_balanced_json_object(raw, 0).is_none()
            && let Some(brace) = raw.find('{')
            && raw[brace..].contains(GENERIC_TYPE_TAG)
            && raw[brace..].contains(GENERIC_CALL_MARKER)
        {
            // truncated/unterminated call object (e.g. generation cut-off):
            // report explicitly instead of degrading to plain text
            return AssistantAction::MalformedToolCall {
                excerpt: raw[brace..].chars().take(160).collect(),
                reason: "unterminated tool_call object (truncated?)".to_string(),
            };
        }
        while let Some((start, end)) = find_balanced_json_object(raw, search_from) {
            let candidate = &raw[start..end];
            match serde_json::from_str::<serde_json::Value>(candidate) {
                Ok(value) if Self::looks_like_call(&value) => match extract_call_fields(&value) {
                    Ok((name, arguments_json)) => {
                        calls.push(RawToolCall::new(name, arguments_json));
                        search_from = end;
                    }
                    Err(reason) => {
                        return AssistantAction::MalformedToolCall {
                            excerpt: candidate.chars().take(160).collect(),
                            reason,
                        };
                    }
                },
                Ok(_) => {
                    search_from = end;
                }
                Err(e) => {
                    if candidate.contains(GENERIC_CALL_MARKER)
                        && candidate.contains(GENERIC_TYPE_TAG)
                    {
                        return AssistantAction::MalformedToolCall {
                            excerpt: candidate.chars().take(160).collect(),
                            reason: format!("invalid JSON in tool_call object: {e}"),
                        };
                    }
                    search_from = end;
                }
            }
        }
        if !calls.is_empty() {
            return AssistantAction::ToolCalls(calls);
        }
        AssistantAction::FinalText(raw.trim().to_string())
    }

    fn render_tool_result_message(&self, result: &ToolResultMessage<'_>) -> String {
        let payload = serde_json::json!({
            "type": "tool_result",
            "name": result.call_name,
            "ok": result.ok,
            "content": result.content_json,
        });
        format!(
            "<|im_start|>user\n{}<|im_end|>\n",
            super::schema::canonical_json(&payload)
        )
    }
}

#[cfg(test)]
mod tests;
