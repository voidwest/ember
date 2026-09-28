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

/// The declared `general.architecture` string from GGUF metadata, if present.
pub fn declared_architecture(loader: &GgufLoader) -> Option<&str> {
    match loader.metadata.get("general.architecture") {
        Some(GgufValue::Str(value)) => Some(value.as_str()),
        _ => None,
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
