//! Contract tests for the env-gated suite helper.
//!
//! `tests/common/mod.rs` exists so the real-weight suites (agent, voice,
//! converse, Arabic S2S, tokenizer/chat-template parity) can be
//! **fail-closed** rather than silently skipping. The failure mode this
//! removes is subtle: with the fixture absent, a gated test returns
//! immediately and reports green while asserting nothing, so a passing CI
//! run is not evidence that the suite ran.
//!
//! These tests exercise the decision logic **without** the real weights and
//! **without** mutating the process environment, so they are cheap enough to
//! be a required part of fast CI. `tests/k_parity.rs` already demonstrates
//! the same idea in production form with `EMBER_PARITY_REQUIRED=1`.

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;

fn p(s: &str) -> Option<PathBuf> {
    Some(PathBuf::from(s))
}

/// Without enforcement and without the enable flag: skip, never fail.
#[test]
fn default_behavior_is_a_silent_skip() {
    let keys = ["EMBER_T_MODEL", "EMBER_T_TOKENIZER"];
    let out = common::decide(false, false, "ENABLE", &keys, &[None, None]).expect("no enforcement");
    assert!(out.is_none(), "an unenabled suite must skip, not fail");
}

/// Without enforcement, a missing path still skips. This is the legacy
/// behavior and must be preserved for ordinary laptop `cargo test` runs.
#[test]
fn enabled_but_missing_paths_skips_when_not_enforcing() {
    let keys = ["EMBER_T_MODEL", "EMBER_T_TOKENIZER"];
    let out =
        common::decide(false, true, "ENABLE", &keys, &[p("a.gguf"), None]).expect("no enforcement");
    assert!(
        out.is_none(),
        "a partially configured suite must skip when enforcement is off"
    );
}

/// Fully configured: the resolved paths come back in the requested order.
#[test]
fn fully_configured_suite_yields_ordered_paths() {
    let keys = ["EMBER_T_TEXT", "EMBER_T_AUDIO", "EMBER_T_TOKENIZER"];
    let resolved = [p("text.gguf"), p("audio.gguf"), p("tokenizer.json")];
    let out = common::decide(false, true, "ENABLE", &keys, &resolved)
        .expect("no enforcement")
        .expect("configured suite must run");
    assert_eq!(
        out,
        vec![
            PathBuf::from("text.gguf"),
            PathBuf::from("audio.gguf"),
            PathBuf::from("tokenizer.json"),
        ]
    );
}

/// THE POINT OF THIS FILE: enforcement on + not enabled is a failure, so a
/// release job that expects the suite to run cannot silently pass.
#[test]
fn enforcement_fails_when_suite_is_not_enabled() {
    let keys = ["EMBER_AGENT_E2E", "EMBER_T_MODEL"];
    let err = common::decide(true, false, "EMBER_AGENT_E2E", &keys, &[None, None])
        .expect_err("enforcement must reject an unenabled suite");
    assert!(
        err.contains("EMBER_AGENT_E2E"),
        "the message must name the variable to set, got: {err}"
    );
    // It must NOT name a *path* variable as though it were the switch: a
    // gate that tells the operator to set EMBER_AGENT_MODEL=1 wastes the
    // time of whoever is reading a red CI log.
    assert!(
        !err.contains("EMBER_T_MODEL=1"),
        "the enable hint must not name a path variable, got: {err}"
    );
}

/// Enforcement on + enabled but a path missing: also a failure, and the
/// message must name the *specific* missing variable rather than a generic
/// complaint, or it is useless when it fires in CI.
#[test]
fn enforcement_fails_and_names_the_missing_variable() {
    let keys = ["EMBER_T_TEXT", "EMBER_T_AUDIO", "EMBER_T_TOKENIZER"];
    let resolved = [p("text.gguf"), None, None];
    let err = common::decide(true, true, "ENABLE", &keys, &resolved)
        .expect_err("enforcement must reject a partially configured suite");
    assert!(
        err.contains("EMBER_T_AUDIO"),
        "must name the first missing variable, got: {err}"
    );
}

/// An empty-string variable counts as missing, exactly as the env-reading
/// wrapper treats it (an empty path is never a usable fixture).
#[test]
fn enforcement_rejects_a_completely_empty_value() {
    let keys = ["EMBER_T_MODEL"];
    let err = common::decide(true, true, "ENABLE", &keys, &[Some(PathBuf::new())])
        .expect_err("enforcement must reject an empty value");
    assert!(err.contains("EMBER_T_MODEL"), "got: {err}");
}

/// Suites with no separate enable variable (a reference dump that is simply
/// pointed at a file) still honor enforcement.
#[test]
fn dump_only_suites_are_gated_on_presence() {
    let keys = ["EMBER_T_PARITY_JSON"];
    // Present -> runs, enforcement irrelevant.
    let out = common::decide(false, true, "ENABLE", &keys, &[p("dump.json")])
        .expect("no enforcement")
        .expect("present dump must run");
    assert_eq!(out, vec![PathBuf::from("dump.json")]);
    // Absent + enforcement on -> failure. With no separate switch, the
    // label the caller passes IS the path variable, so that is what the
    // message must name.
    let err = common::decide(true, false, "EMBER_T_PARITY_JSON", &keys, &[None])
        .expect_err("enforcement must reject a missing dump");
    assert!(err.contains("EMBER_T_PARITY_JSON"), "got: {err}");
}

/// The flag itself is read exactly the way `k_parity.rs` reads
/// `EMBER_PARITY_REQUIRED`, so the two gates cannot drift apart.
#[test]
fn enforcement_flag_is_read_as_exactly_one() {
    // Uses a variable that is not set in the ambient environment; this
    // asserts the default, which is the important direction.
    assert!(!common::enforcement_on("EMBER_T_CONTRACT_SHOULD_BE_UNSET"));
}
