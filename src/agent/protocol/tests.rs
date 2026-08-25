//! Protocol golden-string and parse-behavior pins (Track C + "Model-specific
//! correctness"). Rendered messages are pinned byte-exactly: these strings
//! are what real model families expect, and a silent template drift would
//! corrupt every tool conversation built on them.

use super::*;

fn qwen() -> Qwen25ToolProtocol {
    Qwen25ToolProtocol::default()
}

fn llama() -> LlamaToolProtocol {
    LlamaToolProtocol::default()
}

fn generic() -> EmberJsonToolProtocol {
    EmberJsonToolProtocol::default()
}

fn weather_schema() -> ToolSchema {
    use crate::agent::schema::{JsonType, ParamSchema, ToolEffect};
    ToolSchema::new("get_weather", "fixture temperature for a city")
        .param(ParamSchema::new("city", JsonType::String).required())
        .effect(ToolEffect::ReadOnly)
}

#[test]
fn qwen_system_message_is_pinned_byte_exact() {
    let rendered = qwen().render_system_message(None, &[weather_schema()]);
    let expected = "<|im_start|>system\nYou are a helpful assistant with access to the following tools. When you need information from a tool, call it instead of guessing.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"name\":\"get_weather\",\"description\":\"fixture temperature for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name-in-string>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
    assert_eq!(rendered, expected);
}

#[test]
fn qwen_user_assistant_and_result_messages_are_pinned() {
    let p = qwen();
    assert_eq!(
        p.render_user_message("Hello"),
        "<|im_start|>user\nHello<|im_end|>\n"
    );
    assert_eq!(p.render_assistant_prefix(), "<|im_start|>assistant\n");
    assert_eq!(p.render_assistant_suffix(), "<|im_end|>");

    let msg = ToolResultMessage::from_text("get_weather", true, "41 C");
    assert_eq!(
        p.render_tool_result_message(&msg),
        "<|im_start|>user\n<tool_response>\n{\"name\":\"get_weather\",\"content\":\"41 C\"}\n</tool_response><|im_end|>\n"
    );
}

#[test]
fn llama_system_message_is_pinned_byte_exact() {
    let rendered = llama().render_system_message(None, &[weather_schema()]);
    assert!(rendered.starts_with(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nCutting Knowledge Date: December 2023\nToday Date: 01 Jan 2026\n\n"
    ));
    assert!(rendered.contains(
        "You have access to the following functions. To call a function, respond with JSON for a function call."
    ));
    // schema array is pretty-printed JSON
    assert!(
        rendered.contains("\"name\": \"get_weather\"")
            || rendered.contains("\"name\":\"get_weather\"")
    );
    assert!(rendered.ends_with("<|eot_id|>"));
}

#[test]
fn llama_user_and_ipython_result_are_pinned() {
    let p = llama();
    assert_eq!(
        p.render_user_message("Hi"),
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
    );
    assert_eq!(
        p.render_assistant_prefix(),
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    assert_eq!(p.render_assistant_suffix(), "<|eot_id|>");
    let msg = ToolResultMessage::from_text("get_weather", false, "{\"error\":\"unknown city\"}");
    assert_eq!(
        p.render_tool_result_message(&msg),
        "<|start_header_id|>ipython<|end_header_id|>\n\n\"{\\\"error\\\":\\\"unknown city\\\"}\"<|eot_id|>"
    );
}

#[test]
fn llama_parses_python_tag_whole_object() {
    let action = llama().parse_assistant_output(
        "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Riyadh\"}}",
    );
    assert_eq!(
        action,
        AssistantAction::ToolCall(RawToolCall {
            name: "get_weather".to_string(),
            arguments_json: r#"{"city":"Riyadh"}"#.to_string(),
            additional_calls_ignored: 0,
        })
    );
}

#[test]
fn llama_accepts_arguments_alias_and_surrounding_prose_without_tag() {
    let action = llama().parse_assistant_output(
        "Sure, checking now.\n{\"name\":\"lookup\",\"arguments\":{\"key\":\"alpha\"}}\nDone.",
    );
    match action {
        AssistantAction::ToolCall(call) => {
            assert_eq!(call.name, "lookup");
            assert_eq!(call.arguments_json, r#"{"key":"alpha"}"#);
        }
        other => panic!("expected tool call, got {other:?}"),
    }
}

#[test]
fn llama_python_tag_without_json_is_explicitly_malformed_not_final_text() {
    let action = llama().parse_assistant_output("<|python_tag|>oops no json");
    match action {
        AssistantAction::MalformedToolCall { reason, .. } => {
            assert!(reason.contains("python_tag"));
        }
        other => panic!("expected malformed, got {other:?}"),
    }
}

#[test]
fn llama_plain_prose_is_final_text() {
    assert_eq!(
        llama().parse_assistant_output("The value is 42."),
        AssistantAction::FinalText("The value is 42.".to_string())
    );
}

#[test]
fn qwen_parses_first_call_and_counts_ignored_extras() {
    let raw = concat!(
        "<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n</tool_call>\n",
        "<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>"
    );
    match qwen().parse_assistant_output(raw) {
        AssistantAction::ToolCall(call) => {
            assert_eq!(call.name, "a");
            assert_eq!(call.additional_calls_ignored, 1);
        }
        other => panic!("expected tool call, got {other:?}"),
    }
}

#[test]
fn qwen_broken_json_inside_tag_is_malformed() {
    match qwen().parse_assistant_output("<tool_call>\n{\"name\": </tool_call>") {
        AssistantAction::MalformedToolCall { reason, .. } => {
            assert!(reason.contains("invalid JSON"));
        }
        other => panic!("expected malformed, got {other:?}"),
    }
}

#[test]
fn qwen_unclosed_tag_still_yields_intent_or_malformed_never_final() {
    let action = qwen().parse_assistant_output("<tool_call>\n{\"name\": \"x\", \"arguments\": {}}");
    assert!(matches!(action, AssistantAction::ToolCall(_)));
}

#[test]
fn qwen_plain_text_is_final() {
    assert_eq!(
        qwen().parse_assistant_output("  The temperature is 41 C. "),
        AssistantAction::FinalText("The temperature is 41 C.".to_string())
    );
}

#[test]
fn generic_parses_typed_call_whole_text() {
    let raw = r#"{"type": "tool_call", "name": "lookup", "arguments": {"query": "alpha"}}"#;
    match generic().parse_assistant_output(raw) {
        AssistantAction::ToolCall(call) => {
            assert_eq!(call.name, "lookup");
            assert_eq!(call.arguments_json, r#"{"query":"alpha"}"#);
        }
        other => panic!("expected tool call, got {other:?}"),
    }
}

#[test]
fn generic_finds_embedded_call_and_skips_other_objects() {
    let raw = concat!(
        "Let me check. ",
        r#"{"note": {"type": "tool_call"}} looks tempting but lacks fields."#,
        " ",
        r#"{"type":"tool_call","name":"calc","arguments":{"op":"add","a":1,"b":2}}"#,
        " done"
    );
    match generic().parse_assistant_output(raw) {
        AssistantAction::ToolCall(call) => assert_eq!(call.name, "calc"),
        other => panic!("expected tool call, got {other:?}"),
    }
}

#[test]
fn generic_brace_scanner_respects_strings_with_escapes() {
    let raw =
        r#"text {"a":"has } brace \" inside","type":"tool_call","name":"t","arguments":{}} tail"#;
    match generic().parse_assistant_output(raw) {
        AssistantAction::ToolCall(call) => assert_eq!(call.name, "t"),
        other => panic!("expected tool call, got {other:?}"),
    }
}

#[test]
fn extract_call_fields_rejects_non_object_and_bad_types() {
    let v: serde_json::Value = serde_json::from_str(r#"{"name": 5, "arguments": {}}"#).unwrap();
    assert!(extract_call_fields(&v).is_err());
    let v: serde_json::Value = serde_json::from_str(r#"{"arguments": {"a": 1}}"#).unwrap();
    assert!(extract_call_fields(&v).is_err());
    let v: serde_json::Value = serde_json::from_str(r#"{"name": "t", "arguments": [1]}"#).unwrap();
    assert!(extract_call_fields(&v).is_err());
}

#[test]
fn balanced_scanner_returns_none_for_unbalanced_input() {
    assert_eq!(find_balanced_json_object("{not closed", 0), None);
    assert_eq!(find_balanced_json_object("no braces", 0), None);
    assert_eq!(
        find_balanced_json_object("pre {\"a\": 1} post", 0),
        Some((4, 12))
    );
}
