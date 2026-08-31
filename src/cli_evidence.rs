//! `ember evidence`: signed execution evidence (EmberSEC Phase IV —
//! attested execution, pre-TEE).
//!
//! A signed evidence envelope binds a recorded record (typically a run
//! manifest with its execution identity) to a signing key:
//!
//! ```text
//! record JSON ──canonicalize (sorted keys)──► canonical bytes
//!         ──sha256──► digest_sha256
//!         ──ed25519 sign──► signature_hex
//! envelope = { schema, algorithm, signer_fingerprint, signed_at_unix,
//!              digest_sha256, input: <record>, signature_hex }
//! ```
//!
//! What this proves (and does not):
//! - PROVES: the record bytes are exactly what the signer signed; the
//!   signature is fresh for that key; and — when the record is a v2 run
//!   manifest — the manifest's internal execution identity is internally
//!   consistent (tamper-evident provenance for an inference result).
//! - DOES NOT prove: that the execution happened on trusted hardware, that
//!   the environment was honest, or that the key holder is who they claim.
//!   Those are the TDX/SNP attestation layer (Phase IVb); the envelope
//!   schema is designed so the local signing key can later be replaced by a
//!   key attested inside an enclave without changing the record format.

use anyhow::{anyhow, ensure, Context, Result};
use clap::{Args as ClapArgs, Subcommand};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use rand::RngCore;
use std::path::Path;

/// Schema tag for signed evidence envelopes.
pub const EVIDENCE_SCHEMA: &str = "signed-evidence-v1";

#[derive(ClapArgs)]
pub(crate) struct EvidenceCommand {
    #[command(subcommand)]
    pub(crate) command: EvidenceSubcommand,
}

#[derive(Subcommand)]
pub(crate) enum EvidenceSubcommand {
    /// Generate an Ed25519 signing key for run evidence
    Init(InitCommand),
    /// Sign a JSON record (e.g. a `--write-run-manifest` output) into a
    /// self-contained evidence envelope
    Sign(SignCommand),
    /// Verify a signed evidence envelope
    Verify(VerifyCommand),
}

#[derive(ClapArgs)]
pub(crate) struct InitCommand {
    /// path to write the private key (hex, 32 bytes, mode 0600); the public
    /// key is written to `<key>.pub`. Refuses to overwrite an existing key.
    #[arg(long)]
    key: String,
}

#[derive(ClapArgs)]
pub(crate) struct SignCommand {
    /// path to the JSON record to sign (run manifest or any JSON)
    #[arg(long)]
    manifest: String,
    /// path to the private key written by `evidence init`
    #[arg(long)]
    key: String,
    /// output envelope path (default: `<manifest>.signed.json`)
    #[arg(long)]
    out: Option<String>,
}

#[derive(ClapArgs)]
pub(crate) struct VerifyCommand {
    /// path to a signed evidence envelope
    path: String,
}

/// Outcome of a successful envelope verification.
#[derive(Debug)]
pub(crate) struct EvidenceVerified {
    pub signer_fingerprint: String,
    pub digest_sha256: String,
    pub identity_ok: bool,
    pub identity_digest: Option<String>,
}

pub(crate) fn run_evidence_command(command: &EvidenceCommand) -> Result<()> {
    match &command.command {
        EvidenceSubcommand::Init(cmd) => run_init(cmd),
        EvidenceSubcommand::Sign(cmd) => run_sign(cmd),
        EvidenceSubcommand::Verify(cmd) => run_verify(cmd),
    }
}

fn run_init(cmd: &InitCommand) -> Result<()> {
    let path = Path::new(&cmd.key);
    if path.exists() {
        anyhow::bail!("refusing to overwrite existing key {}", path.display());
    }
    let mut seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut seed);
    let key = SigningKey::from_bytes(&seed);
    let pub_hex = hex_encode(&key.verifying_key().to_bytes());
    write_key_file(path, &hex_encode(&seed))?;
    let pub_path = path.with_extension("pub");
    write_key_file(&pub_path, &pub_hex)?;
    println!(
        "wrote private key {} (0600) and public key {}",
        path.display(),
        pub_path.display()
    );
    println!("signer fingerprint: {pub_hex}");
    Ok(())
}

fn run_sign(cmd: &SignCommand) -> Result<()> {
    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&cmd.manifest)
            .with_context(|| format!("failed to read record {}", cmd.manifest))?,
    )
    .with_context(|| format!("record {} is not valid JSON", cmd.manifest))?;
    let seed = read_key_seed(&cmd.key)?;
    let envelope = build_envelope(&record, &seed, ember::extraction::unix_timestamp())?;
    let out = cmd
        .out
        .clone()
        .unwrap_or_else(|| format!("{}.signed.json", cmd.manifest));
    crate::cli_support::write_json_file(&out, &envelope)?;
    println!(
        "signed evidence written to {out} (signer {})",
        envelope["signer_fingerprint"].as_str().unwrap_or("?")
    );
    Ok(())
}

fn run_verify(cmd: &VerifyCommand) -> Result<()> {
    let envelope: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&cmd.path)
            .with_context(|| format!("failed to read envelope {}", cmd.path))?,
    )
    .with_context(|| format!("envelope {} is not valid JSON", cmd.path))?;
    let verified = verify_envelope(&envelope)?;
    if !verified.identity_ok {
        anyhow::bail!(
            "signed run manifest has an internally inconsistent execution identity              (identity.sha256 does not match identity.canonical)"
        );
    }
    println!(
        "OK  signature valid (ed25519, signer {})",
        verified.signer_fingerprint
    );
    println!(
        "    digest {} (canonical input sha256)",
        verified.digest_sha256
    );
    if let Some(digest) = verified.identity_digest {
        println!("    execution identity {digest} (verified)");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// envelope construction / verification (pure functions; unit-testable)
// ---------------------------------------------------------------------------

pub(crate) fn build_envelope(
    input: &serde_json::Value,
    key_seed: &[u8; 32],
    signed_at_unix: u64,
) -> Result<serde_json::Value> {
    let signing_key = SigningKey::from_bytes(key_seed);
    let canonical = canonical_bytes(input)?;
    let signature = signing_key.sign(&canonical);
    Ok(serde_json::json!({
        "schema": EVIDENCE_SCHEMA,
        "algorithm": "ed25519",
        "signed_at_unix": signed_at_unix,
        "signer_fingerprint": hex_encode(&signing_key.verifying_key().to_bytes()),
        "digest_sha256": ember::extraction::sha256_bytes(&canonical),
        "input": input,
        "signature_hex": hex_encode(&signature.to_bytes()),
    }))
}

pub(crate) fn verify_envelope(envelope: &serde_json::Value) -> Result<EvidenceVerified> {
    ensure!(
        envelope.get("schema").and_then(|v| v.as_str()) == Some(EVIDENCE_SCHEMA),
        "envelope schema is not {EVIDENCE_SCHEMA}"
    );
    ensure!(
        envelope.get("algorithm").and_then(|v| v.as_str()) == Some("ed25519"),
        "envelope algorithm is not ed25519"
    );
    let fingerprint = envelope
        .get("signer_fingerprint")
        .and_then(|v| v.as_str())
        .context("envelope has no signer_fingerprint")?
        .to_string();
    let recorded_digest = envelope
        .get("digest_sha256")
        .and_then(|v| v.as_str())
        .context("envelope has no digest_sha256")?;
    let signature_hex = envelope
        .get("signature_hex")
        .and_then(|v| v.as_str())
        .context("envelope has no signature_hex")?;
    let input = envelope
        .get("input")
        .context("envelope has no input section")?;

    let canonical = canonical_bytes(input)?;
    let recomputed = ember::extraction::sha256_bytes(&canonical);
    ensure!(
        recomputed == recorded_digest,
        "digest mismatch: recorded {recorded_digest} != recomputed {recomputed}"
    );

    let fp_bytes: [u8; 32] = hex_decode(&fingerprint)?
        .try_into()
        .map_err(|_| anyhow!("signer fingerprint must decode to 32 bytes"))?;
    let vk = VerifyingKey::from_bytes(&fp_bytes)
        .context("signer fingerprint is not a valid Ed25519 public key")?;
    let sig_bytes: [u8; 64] = hex_decode(signature_hex)?
        .try_into()
        .map_err(|_| anyhow!("signature_hex must decode to 64 bytes"))?;
    let signature = Signature::from_bytes(&sig_bytes);
    vk.verify_strict(&canonical, &signature)
        .context("signature verification failed")?;

    // When the signed record is a v2 run manifest, also assert its internal
    // execution identity is self-consistent (Phase III seam).
    let mut identity_ok = true;
    let mut identity_digest = None;
    if let Some(identity) = input.get("identity")
        && let Some(canon) = identity.get("canonical")
    {
        let digest = crate::cli_manifest::recompute_identity_sha256(canon)?;
        identity_digest = Some(digest.clone());
        if let Some(recorded) = identity.get("sha256").and_then(|v| v.as_str()) {
            identity_ok = digest == recorded;
        }
    }

    Ok(EvidenceVerified {
        signer_fingerprint: fingerprint,
        digest_sha256: recorded_digest.to_string(),
        identity_ok,
        identity_digest,
    })
}

/// Canonical JSON bytes: keys sorted recursively, compact serialization.
/// Field order in the original record is irrelevant to the signature.
pub(crate) fn canonical_bytes(value: &serde_json::Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&sort_value(value))?)
}

fn sort_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut sorted: Vec<(String, serde_json::Value)> = map
                .iter()
                .map(|(k, v)| (k.clone(), sort_value(v)))
                .collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(sort_value).collect())
        }
        other => other.clone(),
    }
}

fn read_key_seed(path: &str) -> Result<[u8; 32]> {
    let hex =
        std::fs::read_to_string(path).with_context(|| format!("failed to read key {}", path))?;
    let bytes = hex_decode(hex.trim())?;
    bytes
        .try_into()
        .map_err(|_| anyhow!("key file must contain exactly 32 bytes (64 hex chars)"))
}

fn write_key_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(path, perms)
            .with_context(|| format!("failed to chmod 0600 {}", path.display()))?;
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    let hex = hex.trim();
    ensure!(
        hex.len().is_multiple_of(2),
        "hex string has odd length {}",
        hex.len()
    );
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|_| anyhow!("invalid hex byte {:?}", &hex[i..i + 2]))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_manifest_value() -> serde_json::Value {
        let mut manifest = serde_json::json!({
            "schema_version": 2,
            "identity": {
                "schema": "execution-identity-v1",
                "sha256": "placeholder",
                "canonical": {
                    "model": {"sha256": "abc", "architecture": "llama"},
                    "prompt": "hello",
                    "sampler": {"temperature": 0.0, "top_k": null, "top_p": null, "seed": null},
                },
            },
            "execution": {"prompt": "hello"},
        });
        let digest =
            crate::cli_manifest::recompute_identity_sha256(&manifest["identity"]["canonical"])
                .unwrap();
        manifest["identity"]["sha256"] = serde_json::json!(digest);
        manifest
    }

    const SEED: [u8; 32] = [7u8; 32];

    #[test]
    fn sign_verify_round_trip_with_identity_check() {
        let manifest = run_manifest_value();
        let envelope = build_envelope(&manifest, &SEED, 1_700_000_000).unwrap();
        let verified = verify_envelope(&envelope).unwrap();
        assert!(verified.identity_ok);
        assert!(verified.identity_digest.is_some());
        assert_eq!(verified.signer_fingerprint.len(), 64);
    }

    #[test]
    fn tampered_input_fails_digest_check() {
        let manifest = run_manifest_value();
        let mut envelope = build_envelope(&manifest, &SEED, 1_700_000_000).unwrap();
        envelope["input"]["execution"]["prompt"] = serde_json::json!("tampered");
        let err = verify_envelope(&envelope).expect_err("tampered input must fail");
        assert!(err.to_string().contains("digest mismatch"), "{err}");
    }

    #[test]
    fn tampered_signature_fails_verification() {
        let manifest = run_manifest_value();
        let mut envelope = build_envelope(&manifest, &SEED, 1_700_000_000).unwrap();
        let sig = envelope["signature_hex"].as_str().unwrap().to_string();
        let flipped = format!("{:02x}", u8::from_str_radix(&sig[..2], 16).unwrap() ^ 1) + &sig[2..];
        envelope["signature_hex"] = serde_json::json!(flipped);
        let err = verify_envelope(&envelope).expect_err("tampered signature must fail");
        assert!(
            err.to_string().contains("signature verification failed"),
            "{err}"
        );
    }

    #[test]
    fn wrong_signer_fingerprint_fails_verification() {
        let manifest = run_manifest_value();
        let mut envelope = build_envelope(&manifest, &SEED, 1_700_000_000).unwrap();
        // keep the signature, swap the fingerprint to a different key
        let other = SigningKey::from_bytes(&[9u8; 32]);
        envelope["signer_fingerprint"] =
            serde_json::json!(hex_encode(&other.verifying_key().to_bytes()));
        let err = verify_envelope(&envelope).expect_err("wrong signer must fail");
        assert!(
            err.to_string().contains("signature verification failed"),
            "{err}"
        );
    }

    #[test]
    fn canonical_bytes_ignore_key_order() {
        let a = serde_json::json!({"b": 1, "a": {"d": 2, "c": [3, 4]}});
        let b = serde_json::json!({"a": {"c": [3, 4], "d": 2}, "b": 1});
        assert_eq!(canonical_bytes(&a).unwrap(), canonical_bytes(&b).unwrap());
    }
}
