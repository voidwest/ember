//! `ember manifest`: run-manifest verification (EmberSEC Phase III —
//! execution identity / reproducible replay).
//!
//! A run manifest written by `--write-run-manifest` carries an `identity`
//! section: a canonical, sorted JSON object of every output-affecting input
//! plus its SHA-256 digest. `ember manifest verify` recomputes that digest
//! from the recorded canonical object and fails if anything was edited, so a
//! recorded result can be meaningfully attributed to one execution.

use anyhow::{Context, Result};
use clap::{Args as ClapArgs, Subcommand};
use ember::extraction::sha256_bytes;

#[derive(ClapArgs)]
pub(crate) struct ManifestCommand {
    #[command(subcommand)]
    pub(crate) command: ManifestSubcommand,
}

#[derive(Subcommand)]
pub(crate) enum ManifestSubcommand {
    /// Recompute the execution identity of a recorded run manifest and
    /// verify it against the recorded digest.
    Verify(VerifyCommand),
}

#[derive(ClapArgs)]
pub(crate) struct VerifyCommand {
    /// path to a run manifest JSON written by `--write-run-manifest`
    path: String,
}

pub(crate) fn run_manifest_command(command: &ManifestCommand) -> Result<()> {
    match &command.command {
        ManifestSubcommand::Verify(cmd) => verify_manifest(cmd),
    }
}

fn verify_manifest(cmd: &VerifyCommand) -> Result<()> {
    let raw = std::fs::read_to_string(&cmd.path)
        .with_context(|| format!("failed to read manifest {}", cmd.path))?;
    let manifest: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("manifest {} is not valid JSON", cmd.path))?;
    let identity = manifest
        .get("identity")
        .context("manifest has no identity section (was it written with schema_version >= 2?)")?;
    let recorded = identity
        .get("sha256")
        .and_then(|v| v.as_str())
        .context("identity.sha256 is missing or not a string")?;
    let canonical = identity
        .get("canonical")
        .context("identity.canonical is missing")?;
    let recomputed = recompute_identity_sha256(canonical)?;
    if recomputed != recorded {
        anyhow::bail!(
            "identity mismatch: recorded {recorded} != recomputed {recomputed}; \
             the manifest was edited after the run, or the canonical section is inconsistent"
        );
    }
    println!(
        "OK  execution identity {recomputed} (schema {})",
        identity
            .get("schema")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
    );
    print_summary(&manifest);
    Ok(())
}

/// SHA-256 over the compact canonical JSON. The serde_json Map is
/// BTreeMap-backed, so keys serialize in sorted order and re-parsing the
/// recorded canonical object reproduces the exact same bytes.
pub(crate) fn recompute_identity_sha256(canonical: &serde_json::Value) -> Result<String> {
    let bytes = serde_json::to_vec(canonical)?;
    Ok(sha256_bytes(&bytes))
}

fn print_summary(manifest: &serde_json::Value) {
    let get = |section: &str, field: &str| -> String {
        manifest
            .get(section)
            .and_then(|v| v.get(field))
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string()
    };
    println!(
        "  model      : {} (sha256 {})",
        get("model", "path"),
        get("model", "sha256")
    );
    println!(
        "  tokenizer  : {} (sha256 {})",
        get("tokenizer", "path"),
        get("tokenizer", "sha256")
    );
    println!("  arch       : {}", get("model", "architecture"));
    if let Some(seed) = manifest
        .get("execution")
        .and_then(|v| v.get("seed"))
        .and_then(|v| v.as_u64())
    {
        println!("  seed       : {seed}");
    } else {
        println!("  seed       : (unseeded / greedy)");
    }
    if let Some(prompt) = manifest.get("execution").and_then(|v| v.get("prompt")) {
        let prompt = prompt.as_str().unwrap_or("?");
        let shown: String = prompt.chars().take(80).collect();
        println!(
            "  prompt     : {shown}{}",
            if prompt.len() > 80 { "…" } else { "" }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_with_identity() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 2,
            "identity": {
                "schema": "execution-identity-v1",
                "sha256": "placeholder",
                "canonical": {
                    "model": {"sha256": "abc", "architecture": "llama"},
                    "prompt": "The capital of France is",
                    "sampler": {"temperature": 0.0, "top_k": null, "top_p": null, "seed": null},
                },
            },
        })
    }

    #[test]
    fn recomputed_identity_matches_recorded_digest() {
        let mut manifest = manifest_with_identity();
        let canonical = manifest["identity"]["canonical"].clone();
        let digest = recompute_identity_sha256(&canonical).unwrap();
        manifest["identity"]["sha256"] = serde_json::json!(digest);
        let raw = serde_json::to_vec(&manifest).unwrap();
        let reparsed: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let again = recompute_identity_sha256(&reparsed["identity"]["canonical"]).unwrap();
        assert_eq!(again, manifest["identity"]["sha256"].as_str().unwrap());
    }

    #[test]
    fn tampered_canonical_changes_the_digest() {
        let mut manifest = manifest_with_identity();
        let canonical = manifest["identity"]["canonical"].clone();
        let digest = recompute_identity_sha256(&canonical).unwrap();
        manifest["identity"]["sha256"] = serde_json::json!(digest);
        // an edit to any output-affecting field must invalidate the identity
        manifest["identity"]["canonical"]["sampler"]["temperature"] = serde_json::json!(0.8);
        let tampered = recompute_identity_sha256(&manifest["identity"]["canonical"]).unwrap();
        assert_ne!(tampered, digest);
    }
}
