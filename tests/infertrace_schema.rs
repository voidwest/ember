//! Contract tests for the inference-trace document schema.
//!
//! The trace document previously carried only a bare `schema_version: 1`
//! integer and had no reader at all. These tests pin the named schema
//! (`ember.infertrace.v1`), the fail-closed reader, and the forward-
//! compatibility rule that a reader ignores unknown fields.
//!
//! Run model-free: `cargo test --test infertrace_schema`.

use ember::trace::{
    parse_infertrace_document, InferTraceDocument, InferTraceError, TraceReport, TRACE_SCHEMA,
    TRACE_SCHEMA_MAJOR,
};

fn report(phase: &str) -> TraceReport {
    TraceReport {
        phase: phase.to_string(),
        token_index: 0,
        events: Vec::new(),
        total_duration_ns: 1234,
        run_metadata: None,
    }
}

/// The writer stamps the named schema, not just an integer.
#[test]
fn written_document_carries_the_named_schema() {
    let doc = InferTraceDocument::new(Some(report("prefill")), Some(report("decode")));
    assert_eq!(doc.schema, TRACE_SCHEMA);
    assert_eq!(doc.schema_version, TRACE_SCHEMA_MAJOR);
    let value = serde_json::to_value(&doc).expect("serialize");
    assert_eq!(
        value["schema"].as_str(),
        Some(TRACE_SCHEMA),
        "the document must name its schema on the wire"
    );
    // The legacy integer is retained so pre-existing readers still work.
    assert_eq!(value["schema_version"].as_u64(), Some(1));
}

/// Round-trips through the validator.
#[test]
fn document_round_trips_through_the_reader() {
    let doc = InferTraceDocument::new(Some(report("prefill")), Some(report("decode")));
    let raw = serde_json::to_string(&doc).expect("serialize");
    let back = parse_infertrace_document(&raw).expect("must read back its own output");
    assert_eq!(back.schema, TRACE_SCHEMA);
    assert_eq!(back.decode.expect("decode present").total_duration_ns, 1234);
    assert_eq!(back.prefill.expect("prefill present").phase, "prefill");
}

/// THE POINT: a legacy document with only the integer is rejected, not
/// silently accepted. If this ever passes, "versioned" is not true.
#[test]
fn legacy_unnamed_document_is_rejected() {
    let legacy = r#"{"schema_version":1,"prefill":null,"decode":null}"#;
    match parse_infertrace_document(legacy) {
        Err(InferTraceError::MissingSchema { expected }) => assert_eq!(expected, TRACE_SCHEMA),
        other => panic!("expected MissingSchema, got {other:?}"),
    }
}

/// A future major is rejected rather than misread.
#[test]
fn unknown_major_is_rejected() {
    for found in [
        "ember.infertrace.v2",
        "ember.infertrace.v99",
        "ember.agent.trace.v1",
    ] {
        let raw = format!(r#"{{"schema":"{found}","schema_version":1}}"#);
        match parse_infertrace_document(&raw) {
            Err(InferTraceError::UnsupportedSchema {
                found: got,
                expected,
            }) => {
                assert_eq!(got, found);
                assert_eq!(expected, TRACE_SCHEMA);
            }
            other => panic!("expected UnsupportedSchema for {found}, got {other:?}"),
        }
    }
}

/// A non-string schema is a missing schema, not a panic.
#[test]
fn non_string_schema_is_rejected_cleanly() {
    let raw = r#"{"schema":1,"schema_version":1}"#;
    assert!(matches!(
        parse_infertrace_document(raw),
        Err(InferTraceError::MissingSchema { .. })
    ));
}

/// Additive fields are compatible: a future writer may add keys and this
/// reader must still work. This is the rule docs/trace-schema.md states.
#[test]
fn unknown_fields_are_ignored_for_compatibility() {
    let raw = format!(
        r#"{{"schema":"{TRACE_SCHEMA}","schema_version":1,
             "prefill":null,"decode":null,
             "a_future_field":{{"nested":[1,2,3]}},"another":42}}"#
    );
    let doc = parse_infertrace_document(&raw).expect("additive fields must stay compatible");
    assert_eq!(doc.schema, TRACE_SCHEMA);
    assert!(doc.prefill.is_none());
    assert!(doc.decode.is_none());
}

/// Malformed bytes are a clean error, never a panic.
#[test]
fn malformed_json_is_a_clean_error() {
    for bad in ["", "not json", "{", "[]", "null"] {
        assert!(
            parse_infertrace_document(bad).is_err(),
            "{bad:?} must be an error"
        );
    }
}

/// `null` and `[]` are valid JSON but not documents: they must fail the
/// schema check rather than deserialize into an empty struct.
#[test]
fn json_that_is_not_an_object_is_rejected() {
    assert!(parse_infertrace_document("[]").is_err());
    assert!(parse_infertrace_document("null").is_err());
}

/// The schema id follows the same dotted convention as the frozen
/// artifact schemas, and its major is the number after the final dot.
#[test]
fn schema_id_follows_the_project_convention() {
    assert!(TRACE_SCHEMA.starts_with("ember."));
    assert!(TRACE_SCHEMA.ends_with(".v1"));
    let major: u32 = TRACE_SCHEMA
        .rsplit(".v")
        .next()
        .expect("a .vN suffix")
        .parse()
        .expect("numeric major");
    assert_eq!(major, TRACE_SCHEMA_MAJOR);
}
