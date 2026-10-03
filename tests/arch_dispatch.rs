//! Architecture dispatch must be single-sourced and fail closed.
//!
//! The `general.architecture` -> engine-family mapping was duplicated in four
//! places: the loader resolver, the extraction resolver, the differential
//! harness (`src/diff_outcome.rs`), and the EmberSEC harness
//! (`tests/_embersec_harness.rs`). Two of them defaulted an absent or
//! unrecognized architecture to `llama`, so a GGUF with no
//! `general.architecture` was accepted by those harnesses while the CLI
//! rejected it. In a corpus that records outcomes as evidence, a false
//! "accepted" is worse than a hard reject.
//!
//! These tests pin the shared mapping and the fail-closed contract. They are
//! model-free: no GGUF is loaded.

use ember::loader::{resolve_generation_architecture, GgufLoader, GgufValue};
use ember::support::{engine_family_for, resolve_engine_family, ArchitectureError, EngineFamily};
use std::collections::HashMap;

/// Build a loader that declares `general.architecture` (or omits it).
fn loader_with(arch: Option<&str>) -> GgufLoader {
    loader_with_value(arch.map(|arch| GgufValue::Str(arch.to_string())))
}

/// Build a loader whose `general.architecture` holds `value` (or is absent).
fn loader_with_value(value: Option<GgufValue>) -> GgufLoader {
    GgufLoader {
        metadata: value
            .map(|value| ("general.architecture".to_string(), value))
            .into_iter()
            .collect(),
        tensors: HashMap::new(),
        k_strategy: ember::quant_k::KStrategy::Auto,
        k_decisions: HashMap::new(),
        tensor_meta: HashMap::new(),
    }
}

const DECLARED: &[(&str, EngineFamily)] = &[
    ("gpt2", EngineFamily::Gpt2),
    ("llama", EngineFamily::Llama),
    ("qwen2", EngineFamily::Qwen3),
    ("qwen3", EngineFamily::Qwen3),
    ("gemma3", EngineFamily::Gemma4),
    ("gemma4", EngineFamily::Gemma4),
];

/// The documented mapping, pinned.
#[test]
fn declared_architectures_map_to_the_documented_families() {
    for (declared, expected) in DECLARED {
        assert_eq!(
            engine_family_for(declared).unwrap_or_else(|e| panic!("{declared}: {e}")),
            *expected,
            "{declared} must map to {expected:?}"
        );
    }
}

/// The CLI dispatches on the resolver's literal output (`qwen2` resolves to
/// the `qwen3` engine, `gemma3` to `gemma4`).
#[test]
fn auto_resolves_to_the_engine_strings_the_cli_dispatches_on() {
    for (declared, engine) in [
        ("gpt2", "gpt2"),
        ("llama", "llama"),
        ("qwen2", "qwen3"),
        ("qwen3", "qwen3"),
        ("gemma3", "gemma4"),
        ("gemma4", "gemma4"),
    ] {
        assert_eq!(
            resolve_generation_architecture("auto", &loader_with(Some(declared))).unwrap(),
            engine,
            "{declared}"
        );
    }
}

/// A missing architecture is a hard failure, not a guess. This is the exact
/// case the two harnesses used to accept.
#[test]
fn missing_architecture_fails_closed() {
    let loader = loader_with(None);
    assert_eq!(
        resolve_engine_family(&loader).unwrap_err(),
        ArchitectureError::Missing
    );
    // The generation resolver must agree.
    let err = resolve_generation_architecture("auto", &loader).unwrap_err();
    assert!(
        err.to_string()
            .contains("missing required general.architecture"),
        "got: {err}"
    );
}

/// A non-string architecture is a hard failure.
#[test]
fn non_string_architecture_fails_closed() {
    let loader = loader_with_value(Some(GgufValue::U32(7)));
    assert_eq!(
        resolve_engine_family(&loader).unwrap_err(),
        ArchitectureError::NotAString
    );
    assert!(resolve_generation_architecture("auto", &loader).is_err());
}

/// Unrecognized architectures are rejected, never coerced to llama. The old
/// harness default made "phi3" and a typo'd "llamma" load as Llama.
#[test]
fn unknown_architectures_are_rejected_not_coerced() {
    for bogus in ["phi3", "llamma", "llama2", "mixtral", "", "LLAMA", "qwen4"] {
        match engine_family_for(bogus) {
            Err(ArchitectureError::Unsupported(found)) => assert_eq!(found, bogus),
            other => panic!("{bogus:?} must be Unsupported, got {other:?}"),
        }
        assert!(
            resolve_generation_architecture("auto", &loader_with(Some(bogus))).is_err(),
            "{bogus:?} must not load"
        );
    }
}

/// The resolver returns the family AND the declaration, because callers need
/// the declaration for support warnings and conflict messages.
#[test]
fn resolver_returns_family_and_declared_name() {
    let loader = loader_with(Some("qwen2"));
    let (family, declared) = resolve_engine_family(&loader).expect("resolvable");
    assert_eq!(family, EngineFamily::Qwen3);
    assert_eq!(
        declared, "qwen2",
        "must report the declared name, not the family"
    );
}

/// `from_alias` and `engine_family_for` must not disagree — they are two
/// entry points onto the same table.
#[test]
fn alias_normalization_agrees_with_the_family_mapping() {
    for (declared, expected) in DECLARED {
        assert_eq!(EngineFamily::from_alias(declared), Some(*expected));
    }
    for bogus in ["phi3", "", "llama2"] {
        assert_eq!(EngineFamily::from_alias(bogus), None);
    }
    // Round-trip: a family label is itself a valid alias.
    for (declared, family) in DECLARED {
        assert_eq!(
            EngineFamily::from_alias(family.label()),
            Some(*family),
            "label of {declared} must re-resolve to the same family"
        );
    }
}

/// `--arch` conflict detection must still work for every family, and
/// `--arch auto` must always accept.
#[test]
fn arch_conflict_detection_survives_the_consolidation() {
    for (declared, family) in DECLARED {
        let loader = loader_with(Some(declared));
        assert_eq!(
            resolve_generation_architecture("auto", &loader).unwrap(),
            family.label()
        );
        // Its own family label is accepted.
        assert!(resolve_generation_architecture(family.label(), &loader).is_ok());
        // A different family is rejected.
        for (_, other) in DECLARED {
            if other != family {
                assert!(
                    resolve_generation_architecture(other.label(), &loader).is_err(),
                    "{declared} must reject --arch {}",
                    other.label()
                );
            }
        }
    }
}

/// Every error variant renders a message that names the remediation, so a
/// red CI log is actionable without reading the source.
#[test]
fn error_messages_are_actionable() {
    assert!(ArchitectureError::Missing
        .to_string()
        .contains("general.architecture"));
    assert!(ArchitectureError::NotAString
        .to_string()
        .contains("must be a string"));
    let unsupported = ArchitectureError::Unsupported("phi3".to_string()).to_string();
    assert!(
        unsupported.contains("phi3"),
        "must name the input: {unsupported}"
    );
    assert!(
        unsupported.contains("qwen2"),
        "must list the accepted set: {unsupported}"
    );
}

// ---------------------------------------------------------------------------
// call-site guards
// ---------------------------------------------------------------------------

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn source(relative: &str) -> String {
    std::fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|e| panic!("read {relative}: {e}"))
}

/// The unit tests above pin `support::resolve_engine_family`, but they do not
/// pin the *call sites* to use it. A previous attempt at this consolidation
/// passed every test while `src/diff_outcome.rs` still coerced an
/// unrecognized architecture to Llama, because reverting a call site is
/// invisible to a test of the shared helper.
///
/// These guards are source-level but deliberately *semantic*, not textual: an
/// earlier version grepped for the literal `_ => "llama"` and was defeated by
/// writing `_ => EngineFamily::Llama`. The invariant that actually matters is
/// that a dispatch site never touches the `Option`-returning accessor at all.
#[test]
fn dispatch_sites_never_use_the_permissive_accessor() {
    // `declared_architecture` returns Option and is the exact shape that let a
    // missing/unknown architecture be silently treated as llama. A dispatch
    // site must use `resolve_engine_family`, which fails closed.
    // research/embersec/comparative/harness/ is the CANONICAL, tracked copy.
    // tests/_embersec_harness.rs is a gitignored local copy of the same file,
    // so guarding only the latter would protect the version nobody ships.
    for file in [
        "src/diff_outcome.rs",
        "src/loader.rs",
        "src/cli_commands.rs",
        "src/main.rs",
        "research/embersec/comparative/harness/_embersec_harness.rs",
    ] {
        let text = source(file);
        for (number, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            assert!(
                !code.contains("declared_architecture"),
                "{file}:{number} uses the Option-returning declared_architecture; \
                 dispatch must go through ember::support::resolve_engine_family: {}",
                code.trim()
            );
        }
    }
}

/// A dispatch site must never read `general.architecture` out of metadata
/// itself.
///
/// This exists because guarding on the *name* `declared_architecture` was not
/// enough: an earlier version of this guard passed while the canonical
/// harness still did its own `match loader.metadata.get("general.architecture")
/// { ... _ => "llama" }`. Reading the key directly is the shape of the bug,
/// so the harness must not do it at all.
///
/// `src/loader.rs` is deliberately excluded: it legitimately inspects the key
/// for the GPT-2 dequantization budget, which is not a dispatch decision.
#[test]
fn harness_sites_never_read_the_architecture_key_directly() {
    for file in [
        "research/embersec/comparative/harness/_embersec_harness.rs",
        "src/diff_outcome.rs",
    ] {
        let text = source(file);
        for (number, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            assert!(
                !code.contains(r#"metadata.get("general.architecture")"#),
                "{file}:{number} reads general.architecture directly; \
                 use ember::support::resolve_engine_family so an absent or \
                 unknown architecture fails closed: {}",
                code.trim()
            );
        }
        assert!(
            text.contains("resolve_engine_family"),
            "{file} must route through the shared resolver"
        );
    }
}

/// The permissive `Option`-returning accessor is deleted, not merely hidden.
///
/// It used to return `Option<&str>`, which is precisely the shape that let a
/// dispatch site treat an absent architecture as llama. Narrowing its
/// visibility would have left the footgun in place for the next caller, so
/// [`resolve_engine_family`] is the only way to read the declaration.
#[test]
fn permissive_accessor_has_been_deleted() {
    let text = source("src/support.rs");
    assert!(
        !text.contains("fn declared_architecture"),
        "declared_architecture must not be reintroduced: it returns Option and \
         invites a permissive dispatch default. Use resolve_engine_family."
    );
}
