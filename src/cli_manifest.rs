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
    let recomputed = verify_manifest_identity(&manifest)?;
    println!(
        "OK  execution identity {recomputed} (schema {})",
        manifest["identity"]["schema"].as_str().unwrap()
    );
    print_summary(&manifest);
    Ok(())
}

pub(crate) const EXECUTION_IDENTITY_SCHEMA: &str = "execution-identity-v2";

/// Check the declared versions as well as the recorded digest. Historical v1
/// identities retain their original insertion-order encoding; v2 sorts recursively.
pub(crate) fn verify_manifest_identity(manifest: &serde_json::Value) -> Result<String> {
    anyhow::ensure!(
        manifest.get("schema_version").and_then(|v| v.as_u64()) == Some(2),
        "unsupported run manifest schema_version; expected 2"
    );
    let identity = manifest
        .get("identity")
        .context("manifest has no identity section")?;
    let schema = identity
        .get("schema")
        .and_then(|v| v.as_str())
        .context("identity.schema is missing or not a string")?;
    anyhow::ensure!(
        matches!(schema, "execution-identity-v1" | EXECUTION_IDENTITY_SCHEMA),
        "unsupported execution identity schema '{schema}'"
    );
    let canonical = identity
        .get("canonical")
        .context("identity.canonical is missing")?;
    anyhow::ensure!(
        canonical.get("schema").and_then(|v| v.as_str()) == Some(schema),
        "identity and canonical schema declarations disagree"
    );
    let recorded = identity
        .get("sha256")
        .and_then(|v| v.as_str())
        .context("identity.sha256 is missing or not a string")?;
    let recomputed = recompute_identity_sha256(canonical)?;
    anyhow::ensure!(
        recorded == recomputed,
        "identity mismatch: recorded {recorded} != recomputed {recomputed}"
    );
    Ok(recomputed)
}

/// Recompute an explicitly versioned identity without rewriting its stored hash.
pub(crate) fn recompute_identity_sha256(canonical: &serde_json::Value) -> Result<String> {
    let bytes = match canonical.get("schema").and_then(|v| v.as_str()) {
        Some("execution-identity-v1") => serde_json::to_vec(canonical)?,
        Some(EXECUTION_IDENTITY_SCHEMA) => crate::cli_evidence::canonical_bytes(canonical)?,
        other => anyhow::bail!("unsupported canonical execution identity schema {other:?}"),
    };
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
                    "schema": "execution-identity-v1",
                    "model": {"sha256": "abc", "architecture": "llama"},
                    "prompt": "The capital of France is",
                    "sampler": {"temperature": 0.0, "top_k": null, "top_p": null, "seed": null},
                },
            },
        })
    }

    #[test]
    fn retained_v1_identity_keeps_its_original_digest() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/run-manifest-development/identity.json"
        ))
        .unwrap();
        assert_eq!(
            verify_manifest_identity(&manifest).unwrap(),
            "9ecd501e9ac8ba978e7981cc3bf38d2d1c898d2f60ed6032ddfb54bffa225926"
        );
    }

    #[test]
    fn v2_identity_ignores_recursive_object_order_but_preserves_array_order() {
        let mut manifest = manifest_with_identity();
        manifest["identity"]["schema"] = EXECUTION_IDENTITY_SCHEMA.into();
        manifest["identity"]["canonical"]["schema"] = EXECUTION_IDENTITY_SCHEMA.into();
        manifest["identity"]["canonical"]["extra"] = serde_json::json!([{"z": 1, "a": 2}, 3]);
        let digest = recompute_identity_sha256(&manifest["identity"]["canonical"]).unwrap();
        manifest["identity"]["sha256"] = digest.clone().into();
        let reordered: serde_json::Value =
            serde_json::from_slice(&crate::cli_evidence::canonical_bytes(&manifest).unwrap())
                .unwrap();
        assert_eq!(verify_manifest_identity(&reordered).unwrap(), digest);
        manifest["identity"]["canonical"]["extra"]
            .as_array_mut()
            .unwrap()
            .reverse();
        assert!(verify_manifest_identity(&manifest).is_err());
    }

    #[test]
    fn manifest_rejects_unknown_versions_and_missing_identity_fields() {
        let mut valid = manifest_with_identity();
        valid["identity"]["sha256"] = recompute_identity_sha256(&valid["identity"]["canonical"])
            .unwrap()
            .into();
        assert!(verify_manifest_identity(&valid).is_ok());
        for field in ["schema", "sha256", "canonical"] {
            let mut missing = valid.clone();
            missing["identity"].as_object_mut().unwrap().remove(field);
            assert!(verify_manifest_identity(&missing).is_err(), "{field}");
        }
        for path in ["outer", "identity", "canonical"] {
            let mut unknown = valid.clone();
            match path {
                "outer" => unknown["schema_version"] = 999.into(),
                "identity" => unknown["identity"]["schema"] = "execution-identity-v999".into(),
                _ => unknown["identity"]["canonical"]["schema"] = "execution-identity-v999".into(),
            }
            assert!(verify_manifest_identity(&unknown).is_err(), "{path}");
        }
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
