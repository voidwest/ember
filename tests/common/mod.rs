//! Shared gating for the real-weight / real-reference test suites.
//!
//! Most end-to-end suites in this repository are *env-gated* because they
//! load multi-hundred-megabyte GGUFs that CI does not download. The house
//! pattern is a silent skip:
//!
//! ```text
//! EMBER_AGENT_E2E=1 EMBER_AGENT_MODEL=… EMBER_AGENT_TOKENIZER=…
//! ```
//!
//! A silent skip is correct for a developer running `cargo test` on a laptop.
//! It is *not* correct evidence that a suite passed: with the variables
//! absent the test body returns immediately and reports green while
//! asserting nothing. `tests/k_parity.rs` already solves this for the K-quant
//! ladder with `EMBER_PARITY_REQUIRED=1`, which converts a missing variable
//! into a panic. This module generalises that pattern so every gated suite
//! can opt into the same fail-closed behaviour.
//!
//! Each suite names its own fail-closed flag, following the
//! `EMBER_PARITY_REQUIRED` convention already used by `tests/k_parity.rs`:
//!
//! | enable variable      | fail-closed flag             |
//! |----------------------|------------------------------|
//! | `EMBER_AGENT_E2E`    | `EMBER_AGENT_E2E_REQUIRED=1`    |
//! | `EMBER_VOICE_E2E`    | `EMBER_VOICE_E2E_REQUIRED=1`    |
//! | `EMBER_TTS_E2E`      | `EMBER_TTS_E2E_REQUIRED=1`      |
//! | `EMBER_CONVERSE_E2E` | `EMBER_CONVERSE_E2E_REQUIRED=1` |
//! | `EMBER_TOK_PARITY`   | `EMBER_TOK_PARITY_REQUIRED=1`   |
//! | (dump only)          | `EMBER_CHAT_PARITY_REQUIRED=1`  |
//!
//! `tests/k_parity.rs` keeps its own inline `EMBER_PARITY_REQUIRED` check
//! rather than importing this module; the naming convention is what must
//! stay consistent, not the call site.
//!
//! Nothing here changes a default run: with the required variable unset the
//! behaviour is exactly the previous silent skip.
//!
//! This file is a module, not an integration target — cargo only
//! auto-discovers top-level `tests/*.rs`, so `tests/common/mod.rs` is never
//! compiled as its own test binary.
#![allow(dead_code)]

use std::path::PathBuf;

/// Is fail-closed enforcement on for this suite?
///
/// `EMBER_*_REQUIRED=1` turns "the fixture is absent" from a silent skip
/// into a panic. Use it in the pre-tag and release CI jobs so a suite that
/// was supposed to run cannot report green without running.
pub fn enforcement_on(required_var: &str) -> bool {
    std::env::var(required_var).as_deref() == Ok("1")
}

/// The decision, as a pure function of already-read values.
///
/// Split out from [`gate`] so the fail-closed behaviour is testable without
/// touching the process environment (see `tests/gated_suite_contract.rs`)
/// and therefore without the real weights these suites normally need.
///
/// `keys` is carried purely so the failure message can name the offending
/// variable: a fail-closed gate that says only "something is missing" is
/// close to useless when it fires in CI.
pub fn decide(
    enforce: bool,
    enabled: bool,
    enable_label: &str,
    keys: &[&str],
    resolved: &[Option<PathBuf>],
) -> Result<Option<Vec<PathBuf>>, String> {
    if enforce && !enabled {
        return Err(format!(
            "fail-closed enforcement is on but the suite is not enabled (expected {enable_label}=1)"
        ));
    }
    if !enabled {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(resolved.len());
    for (key, value) in keys.iter().zip(resolved) {
        // An empty path is never a usable fixture, so it is treated as
        // missing here as well as in the env-reading wrapper below. Keeping
        // the rule in both places is deliberate: this function is the
        // contract, and `tests/gated_suite_contract.rs` tests it directly.
        match value.as_ref().filter(|path| !path.as_os_str().is_empty()) {
            Some(path) => out.push(path.clone()),
            None if enforce => {
                return Err(format!(
                    "fail-closed enforcement is on but {key} is missing or empty"
                ))
            }
            None => return Ok(None),
        }
    }
    Ok(Some(out))
}

/// Gate a real-weight suite behind an enable flag plus one or more paths.
///
/// `enable` is the suite's opt-in variable; pass `None` for suites that only
/// need a path (for example a reference dump that has no separate switch).
/// `keys` are the required path variables, in the order the caller wants
/// them. `required_var` is the fail-closed flag for this suite.
///
/// Returns `Ok(None)` for the ordinary skip. Returns `Err` — and the caller
/// should `panic!` — when enforcement is on but the suite cannot run.
pub fn gate(
    required_var: &str,
    enable: Option<&str>,
    keys: &[&str],
) -> Result<Option<Vec<PathBuf>>, String> {
    let enforce = enforcement_on(required_var);
    let enabled = match enable {
        // No separate switch: the presence of the path IS the opt-in.
        None => keys
            .iter()
            .any(|k| std::env::var(k).is_ok_and(|v| !v.is_empty())),
        Some(name) => std::env::var(name).as_deref() == Ok("1"),
    };
    let resolved: Vec<Option<PathBuf>> = keys
        .iter()
        .map(|k| {
            std::env::var(k)
                .ok()
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .collect();
    // For a dump-only suite there is no separate switch, so name the first
    // path variable: it is the thing the caller has to point at a file.
    let enable_label = enable.unwrap_or_else(|| keys.first().copied().unwrap_or("<ENABLE_VAR>"));
    decide(enforce, enabled, enable_label, keys, &resolved)
}
