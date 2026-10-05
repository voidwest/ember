//! Advertised support levels for architectures and quantization paths.
//!
//! `docs/support.md` is the human-readable single source of truth; this module
//! mirrors it so the runtime can warn (or, with `--strict-support`, fail) when
//! a model is outside the validated matrix instead of silently presenting an
//! untested path as supported.
//!
//! Levels are deliberately coarse. An architecture is [`SupportLevel::Supported`]
//! only when a golden-logit or equivalent reference record exists for it; see
//! `docs/validation.md` for those records.

use crate::loader::{GgufLoader, GgufValue};
use std::collections::BTreeMap;

/// How much validation a surface has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportLevel {
    /// Validated against a reference and covered by tests; safe to depend on.
    Supported,
    /// Validated, but expect changes before 1.0.
    Evolving,
    /// An execution path exists; numerical validation is incomplete.
    Experimental,
    /// Legacy loader-baseline surface, not a user-facing generation path.
    Internal,
}

/// Support level for a GGUF `general.architecture` value.
///
/// The declared metadata string is used (not the CLI `--arch` alias), so
/// `qwen2` and `qwen3` are distinguished: qwen2.5 carries a completed golden
/// check, qwen3's golden target is still pending.
pub fn architecture_support(declared: &str) -> SupportLevel {
    match declared {
        "llama" | "qwen2" => SupportLevel::Supported,
        "qwen3" => SupportLevel::Experimental,
        "gemma3" | "gemma4" => SupportLevel::Experimental,
        "gpt2" => SupportLevel::Internal,
        _ => SupportLevel::Experimental,
    }
}

/// Warning text for a declared architecture, or `None` when it is in the
/// supported matrix.
pub fn architecture_note(declared: &str) -> Option<&'static str> {
    match architecture_support(declared) {
        SupportLevel::Supported | SupportLevel::Evolving => None,
        SupportLevel::Experimental => Some(
            "not in the validated support matrix (docs/support.md); numerical validation is \
             incomplete, so results are research-grade only — pass --strict-support to fail here",
        ),
        SupportLevel::Internal => Some(
            "loader-baseline architecture; generation is not a supported user path \
             (docs/support.md)",
        ),
    }
}

/// Log a warning when the declared architecture is outside the supported
/// matrix. Called from the shared architecture resolver so every Rust caller
/// (CLI, bindings, research tools) sees the same signal.
pub fn warn_if_unsupported(declared: &str) {
    if let Some(note) = architecture_note(declared) {
        log::warn!("architecture '{declared}': {note}");
    }
}

/// Fail when `strict` is set and the declared architecture is outside the
/// supported matrix.
pub fn strict_gate(declared: &str, strict: bool) -> anyhow::Result<()> {
    if strict && architecture_note(declared).is_some() {
        anyhow::bail!(
            "architecture '{declared}' is outside the supported matrix (docs/support.md); \
             rerun with --strict-support removed to proceed under the documented caveats"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// architecture dispatch
// ---------------------------------------------------------------------------

/// The engine family Ember dispatches a declared GGUF architecture to.
///
/// This is the single mapping from `general.architecture` to the Rust type
/// that implements the forward pass. It lived inline in four places (the
/// loader resolver, the extraction resolver, the EmberSEC harness, and the
/// differential harness), each with its own default for unknown input. Those
/// copies could disagree, and two of them defaulted an unrecognized
/// architecture to [`EngineFamily::Llama`] instead of failing closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineFamily {
    Gpt2,
    Llama,
    Qwen3,
    Gemma4,
}

impl EngineFamily {
    /// The canonical family label, matching the resolver's output string.
    ///
    /// Note that `qwen2` and `qwen3` both map to [`EngineFamily::Qwen3`]: they
    /// share one engine and differ only in numerics, which
    /// [`architecture_support`] reports separately.
    pub fn label(self) -> &'static str {
        match self {
            EngineFamily::Gpt2 => "gpt2",
            EngineFamily::Llama => "llama",
            EngineFamily::Qwen3 => "qwen3",
            EngineFamily::Gemma4 => "gemma4",
        }
    }

    /// Normalize a family alias as a user may pass it to `--arch`.
    ///
    /// Accepts both the declared architecture spellings and the canonical
    /// family labels, so `--arch qwen2` and `--arch qwen3` are equivalent.
    pub fn from_alias(alias: &str) -> Option<Self> {
        match alias {
            "gpt2" => Some(EngineFamily::Gpt2),
            "llama" => Some(EngineFamily::Llama),
            "qwen2" | "qwen3" => Some(EngineFamily::Qwen3),
            "gemma3" | "gemma4" => Some(EngineFamily::Gemma4),
            _ => None,
        }
    }
}

/// Why a declared architecture could not be dispatched.
///
/// Every variant is a hard failure. There is deliberately no catch-all that
/// guesses a family: a GGUF whose architecture is missing, mistyped, or
/// unrecognized must be rejected rather than silently loaded as some other
/// model, because a differential harness that reports "accepted" for a file
/// the CLI rejects is worse than no harness at all.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArchitectureError {
    /// `general.architecture` is absent.
    #[error("GGUF is missing required general.architecture metadata")]
    Missing,
    /// `general.architecture` is present but is not a string.
    #[error("GGUF general.architecture must be a string")]
    NotAString,
    /// `general.architecture` names an architecture Ember cannot run.
    #[error(
        "GGUF architecture '{0}' is not supported by generation; \
         expected gpt2, llama, qwen2/qwen3, or gemma3/gemma4"
    )]
    Unsupported(String),
}

/// Map a declared `general.architecture` string to its engine family.
///
/// Fails closed on anything unrecognized.
pub fn engine_family_for(declared: &str) -> Result<EngineFamily, ArchitectureError> {
    EngineFamily::from_alias(declared)
        .ok_or_else(|| ArchitectureError::Unsupported(declared.to_string()))
}

/// Read `general.architecture` from a loader and resolve it, failing closed.
///
/// Returns the family together with the *declared* string, because callers
/// need the declaration for support warnings and `--arch` conflict messages.
///
/// This is the shared entry point for every runtime that needs to decide which
/// model to build. It replaces per-call-site `match` blocks whose defaults
/// disagreed.
pub fn resolve_engine_family(
    loader: &GgufLoader,
) -> Result<(EngineFamily, &str), ArchitectureError> {
    let declared = match loader.metadata.get("general.architecture") {
        Some(GgufValue::Str(value)) => value.as_str(),
        Some(_) => return Err(ArchitectureError::NotAString),
        None => return Err(ArchitectureError::Missing),
    };
    Ok((engine_family_for(declared)?, declared))
}

/// Human notes for tensors that fell back to eager-f32 because no resident
/// kernel exists for their dtype. The fallback is legal and recorded per
/// tensor, but it changes the memory/speed envelope, so it is worth saying
/// out loud.
pub fn eager_fallback_notes(loader: &GgufLoader) -> Vec<String> {
    let mut per_dtype: BTreeMap<u32, usize> = BTreeMap::new();
    for decision in loader.k_decisions.values() {
        if decision.fallback_reason.is_some() {
            *per_dtype.entry(decision.gguf_dtype).or_default() += 1;
        }
    }
    per_dtype
        .into_iter()
        .map(|(dtype, count)| {
            format!(
                "{count} tensor(s) with GGML dtype {dtype} fell back to eager-f32 \
                 (no resident kernel); expect 2.6-4.5x more RAM and much slower decode"
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_and_qwen2_are_supported_qwen3_is_not_yet() {
        assert_eq!(architecture_support("llama"), SupportLevel::Supported);
        assert_eq!(architecture_support("qwen2"), SupportLevel::Supported);
        assert_eq!(architecture_support("qwen3"), SupportLevel::Experimental);
        assert!(architecture_note("llama").is_none());
        assert!(architecture_note("qwen3").is_some());
    }

    #[test]
    fn gemma_and_gpt2_are_flagged() {
        assert_eq!(architecture_support("gemma4"), SupportLevel::Experimental);
        assert_eq!(architecture_support("gemma3"), SupportLevel::Experimental);
        assert_eq!(architecture_support("gpt2"), SupportLevel::Internal);
        assert!(architecture_note("gemma4").is_some());
        assert!(architecture_note("gpt2").is_some());
    }

    #[test]
    fn strict_gate_only_fails_for_flagged_architectures() {
        assert!(strict_gate("llama", true).is_ok());
        assert!(strict_gate("qwen3", false).is_ok());
        let error = strict_gate("gemma4", true).unwrap_err();
        assert!(error.to_string().contains("supported matrix"));
    }
}
