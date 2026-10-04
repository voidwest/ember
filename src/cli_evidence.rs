//! `ember evidence`: signed execution evidence (EmberSEC Phase IV —
//! attested execution, pre-TEE).
//!
//! A signed evidence envelope binds a recorded record (typically a run
//! manifest with its execution identity) to a signing key. `sign` writes
//! `signed-evidence-v2`, which authenticates every envelope field:
//!
//! ```text
//! record JSON ──canonicalize (sorted keys)──► canonical input bytes
//!         ──sha256──► digest_sha256
//! signed = { schema, algorithm, signer_fingerprint, signed_at_unix,
//!            digest_sha256, input: <record> }
//!         ──canonicalize──► ed25519 sign ──► signature_hex
//! envelope = signed + { signature_hex }
//! ```
//!
//! `verify` also accepts the older `signed-evidence-v1`, whose signature
//! covers only the canonical `input`; its schema, algorithm, fingerprint and
//! timestamp are unauthenticated and the verifier says so.
//!
//! What this proves (and does not):
//! - PROVES: the record bytes are exactly what the holder of the signing key
//!   signed; and — when the record is a v2 run manifest — the manifest's
//!   internal execution identity is internally consistent (tamper-evident
//!   provenance for an inference result).
//! - PROVES WHO SIGNED only with `verify --trusted-key`. The envelope names
//!   its own signer, so without a pinned key anyone can re-sign an edited
//!   record with a fresh key and still verify.
//! - `signed_at_unix` is the signer's claim of when it signed, authenticated
//!   in v2 (not in v1) but not a trusted timestamp: the key holder chooses it.
//! - DOES NOT prove: that the execution happened on trusted hardware, that
//!   the environment was honest, or that the key holder is who they claim.
//!   Those are the TDX/SNP attestation layer (Phase IVb); the envelope
//!   schema is designed so the local signing key can later be replaced by a
//!   key attested inside an enclave without changing the record format.

use anyhow::{anyhow, ensure, Context, Result};
use clap::{Args as ClapArgs, Subcommand};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use ember::v05::manifest::hex as hex_encode;
use rand::RngCore;
use std::path::Path;

/// Schema tag written by `evidence sign`: every envelope field is signed.
pub const EVIDENCE_SCHEMA: &str = "signed-evidence-v2";

/// The original schema: only the canonical `input` is signed. Still verified.
pub const EVIDENCE_SCHEMA_V1: &str = "signed-evidence-v1";

/// The v2 envelope fields covered by the signature (everything except
/// `signature_hex` itself).
const V2_SIGNED_FIELDS: [&str; 6] = [
    "schema",
    "algorithm",
    "signer_fingerprint",
    "signed_at_unix",
    "digest_sha256",
    "input",
];

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
    /// the signer's public key: a `.pub` file written by `evidence init`, or
    /// its 64-char hex fingerprint. Without it, verification only shows the
    /// record is intact for the key the envelope itself names.
    #[arg(long)]
    trusted_key: Option<String>,
}

/// Outcome of a successful envelope verification.
#[derive(Debug)]
pub(crate) struct EvidenceVerified {
    pub schema: &'static str,
    pub signer_fingerprint: String,
    pub digest_sha256: String,
    /// The recorded signing time, and whether the signature covers it
    /// (`false` for v1 envelopes).
    pub signed_at_unix: Option<u64>,
    pub timestamp_signed: bool,
    /// The signed record.
    pub input: serde_json::Value,
    /// The recomputed execution identity, when the record is a run manifest.
    /// An inconsistent identity is a verification error, not a `None`.
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
    let pub_path = path.with_extension("pub");
    // `--key evidence.pub` would otherwise write the public key over the seed.
    ensure!(
        pub_path != path,
        "--key must not end in .pub; that name is used for the public key"
    );
    let mut seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut seed);
    let key = SigningKey::from_bytes(&seed);
    let pub_hex = hex_encode(&key.verifying_key().to_bytes());
    write_key_file(path, &hex_encode(&seed))?;
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
    let out = cmd
        .out
        .clone()
        .unwrap_or_else(|| format!("{}.signed.json", cmd.manifest));
    let envelope = sign_record_file(Path::new(&cmd.manifest), &cmd.key, Path::new(&out))?;
    println!(
        "signed evidence written to {out} (signer {})",
        envelope["signer_fingerprint"].as_str().unwrap_or("?")
    );
    Ok(())
}

/// Sign the JSON record at `record` with the private key at `key` and write
/// a `signed-evidence-v2` envelope to `out`. Returns the envelope.
pub(crate) fn sign_record_file(record: &Path, key: &str, out: &Path) -> Result<serde_json::Value> {
    let parsed: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(record)
            .with_context(|| format!("failed to read record {}", record.display()))?,
    )
    .with_context(|| format!("record {} is not valid JSON", record.display()))?;
    let seed = read_key_seed(key)?;
    let envelope = build_envelope(&parsed, &seed, ember::extraction::unix_timestamp())?;
    crate::cli_support::write_json_file(&out.to_string_lossy(), &envelope)?;
    Ok(envelope)
}

fn run_verify(cmd: &VerifyCommand) -> Result<()> {
    let envelope: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&cmd.path)
            .with_context(|| format!("failed to read envelope {}", cmd.path))?,
    )
    .with_context(|| format!("envelope {} is not valid JSON", cmd.path))?;
    let verified = verify_envelope(&envelope)?;
    match &cmd.trusted_key {
        Some(trusted) => {
            check_trusted_signer(&verified, trusted)?;
            println!(
                "OK  signature valid (ed25519, trusted signer {})",
                verified.signer_fingerprint
            );
        }
        None => {
            println!(
                "OK  signature valid (ed25519, signer {})",
                verified.signer_fingerprint
            );
            println!(
                "    signer is named by the envelope itself; pass --trusted-key to check who signed"
            );
        }
    }
    println!(
        "    digest {} (canonical input sha256)",
        verified.digest_sha256
    );
    println!("    schema {}", verified.schema);
    match (verified.signed_at_unix, verified.timestamp_signed) {
        (Some(at), true) => println!("    signed_at_unix {at} (covered by the signature)"),
        (Some(at), false) => println!(
            "    signed_at_unix {at} is NOT covered by the {} signature (only `input` is \
             signed); re-sign to get {EVIDENCE_SCHEMA}",
            verified.schema
        ),
        (None, _) => println!("    no signed_at_unix recorded"),
    }
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
    let mut envelope = serde_json::json!({
        "schema": EVIDENCE_SCHEMA,
        "algorithm": "ed25519",
        "signed_at_unix": signed_at_unix,
        "signer_fingerprint": hex_encode(&signing_key.verifying_key().to_bytes()),
        "digest_sha256": ember::extraction::sha256_bytes(&canonical_bytes(input)?),
        "input": input,
    });
    let signature = signing_key.sign(&canonical_bytes(&envelope)?);
    envelope["signature_hex"] = hex_encode(&signature.to_bytes()).into();
    Ok(envelope)
}

/// A `signed-evidence-v1` envelope (signature over the canonical input
/// only), as released Ember versions wrote it. Kept to test that old
/// envelopes still verify.
#[cfg(test)]
pub(crate) fn build_envelope_v1(
    input: &serde_json::Value,
    key_seed: &[u8; 32],
    signed_at_unix: u64,
) -> Result<serde_json::Value> {
    let signing_key = SigningKey::from_bytes(key_seed);
    let canonical = canonical_bytes(input)?;
    let signature = signing_key.sign(&canonical);
    Ok(serde_json::json!({
        "schema": EVIDENCE_SCHEMA_V1,
        "algorithm": "ed25519",
        "signed_at_unix": signed_at_unix,
        "signer_fingerprint": hex_encode(&signing_key.verifying_key().to_bytes()),
        "digest_sha256": ember::extraction::sha256_bytes(&canonical),
        "input": input,
        "signature_hex": hex_encode(&signature.to_bytes()),
    }))
}

pub(crate) fn verify_envelope(envelope: &serde_json::Value) -> Result<EvidenceVerified> {
    let object = envelope
        .as_object()
        .context("envelope must be a JSON object")?;
    let schema = match object.get("schema").and_then(|v| v.as_str()) {
        Some(EVIDENCE_SCHEMA) => EVIDENCE_SCHEMA,
        Some(EVIDENCE_SCHEMA_V1) => EVIDENCE_SCHEMA_V1,
        _ => anyhow::bail!("envelope schema is not {EVIDENCE_SCHEMA} or {EVIDENCE_SCHEMA_V1}"),
    };
    let v2 = schema == EVIDENCE_SCHEMA;
    ensure!(
        object.get("algorithm").and_then(|v| v.as_str()) == Some("ed25519"),
        "envelope algorithm is not ed25519"
    );
    let fingerprint = object
        .get("signer_fingerprint")
        .and_then(|v| v.as_str())
        .context("envelope has no signer_fingerprint")?
        .to_string();
    let recorded_digest = object
        .get("digest_sha256")
        .and_then(|v| v.as_str())
        .context("envelope has no digest_sha256")?;
    let signature_hex = object
        .get("signature_hex")
        .and_then(|v| v.as_str())
        .context("envelope has no signature_hex")?;
    let input = object
        .get("input")
        .context("envelope has no input section")?;
    let signed_at_unix = object.get("signed_at_unix").and_then(|v| v.as_u64());
    ensure!(
        !v2 || signed_at_unix.is_some(),
        "envelope signed_at_unix is missing or not an unsigned integer"
    );
    if v2 {
        // Every field is authenticated, so an unsigned extra field would be
        // a place to put claims the signature does not cover.
        for key in object.keys() {
            ensure!(
                key == "signature_hex" || V2_SIGNED_FIELDS.contains(&key.as_str()),
                "envelope field '{key}' is not covered by the {EVIDENCE_SCHEMA} signature"
            );
        }
    }

    let canonical_input = canonical_bytes(input)?;
    let recomputed = ember::extraction::sha256_bytes(&canonical_input);
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
    let signed_bytes = if v2 {
        let mut signed = object.clone();
        signed.remove("signature_hex");
        canonical_bytes(&serde_json::Value::Object(signed))?
    } else {
        canonical_input
    };
    vk.verify_strict(&signed_bytes, &signature)
        .context("signature verification failed")?;

    // When the signed record is a v2 run manifest, also assert its internal
    // execution identity is self-consistent (Phase III seam).
    let declares_execution_identity = input
        .get("identity")
        .and_then(|identity| identity.get("schema"))
        .and_then(|schema| schema.as_str())
        .is_some_and(|schema| schema.starts_with("execution-identity-"));
    let identity_digest = if declares_execution_identity
        || input.get("schema_version").and_then(|v| v.as_u64()) == Some(2)
    {
        Some(crate::cli_manifest::verify_manifest_identity(input)?)
    } else {
        None
    };

    Ok(EvidenceVerified {
        schema,
        signer_fingerprint: fingerprint.to_ascii_lowercase(),
        digest_sha256: recorded_digest.to_string(),
        signed_at_unix,
        timestamp_signed: v2,
        input: input.clone(),
        identity_digest,
    })
}

/// Require the envelope to be signed by `trusted` (a `.pub` file or a hex
/// fingerprint).
pub(crate) fn check_trusted_signer(verified: &EvidenceVerified, trusted: &str) -> Result<()> {
    let trusted = read_trusted_fingerprint(trusted)?;
    ensure!(
        trusted == verified.signer_fingerprint,
        "envelope was signed by {}, not the trusted key {trusted}",
        verified.signer_fingerprint
    );
    Ok(())
}

/// Read and verify an envelope file, requiring the trusted signer.
pub(crate) fn verify_envelope_file_with_trusted_key(
    path: &Path,
    trusted: &str,
) -> Result<EvidenceVerified> {
    let envelope: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path)
            .with_context(|| format!("failed to read envelope {}", path.display()))?,
    )
    .with_context(|| format!("envelope {} is not valid JSON", path.display()))?;
    let verified = verify_envelope(&envelope)?;
    check_trusted_signer(&verified, trusted)?;
    Ok(verified)
}

/// Canonical JSON bytes: keys sorted recursively, compact serialization.
/// Field order in the original record is irrelevant to the signature.
pub(crate) fn canonical_bytes(value: &serde_json::Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&sort_value(value))?)
}

pub(crate) fn sort_value(value: &serde_json::Value) -> serde_json::Value {
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

/// A `.pub` file from `evidence init`, or the fingerprint itself.
fn read_trusted_fingerprint(value: &str) -> Result<String> {
    let text = if Path::new(value).is_file() {
        std::fs::read_to_string(value)
            .with_context(|| format!("failed to read trusted key {value}"))?
    } else {
        value.to_string()
    };
    let bytes = hex_decode(text.trim())?;
    ensure!(
        bytes.len() == 32,
        "trusted key must be a 32-byte Ed25519 public key (64 hex chars)"
    );
    Ok(hex_encode(&bytes))
}

/// Create a key file that is never readable by others, not even briefly, and
/// never replaces an existing one.
fn write_key_file(path: &Path, content: &str) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            anyhow!("refusing to overwrite existing key {}", path.display())
        } else {
            anyhow!("failed to create {}: {error}", path.display())
        }
    })?;
    file.write_all(content.as_bytes())
        .and_then(|()| file.sync_all())
        .with_context(|| format!("failed to write {}", path.display()))
}

fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    let hex = hex.trim();
    // Byte-indexed slicing below would panic inside a multi-byte character.
    ensure!(hex.is_ascii(), "hex string contains non-ASCII characters");
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
                    "schema": "execution-identity-v1",
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
    fn signed_manifest_cannot_omit_identity_fields_or_claim_unknown_version() {
        for field in ["schema", "sha256", "canonical"] {
            let mut manifest = run_manifest_value();
            manifest["identity"].as_object_mut().unwrap().remove(field);
            let envelope = build_envelope(&manifest, &SEED, 0).unwrap();
            assert!(verify_envelope(&envelope).is_err(), "{field}");
        }
        let mut unknown = run_manifest_value();
        unknown["schema_version"] = 999.into();
        let envelope = build_envelope(&unknown, &SEED, 0).unwrap();
        assert!(verify_envelope(&envelope).is_err());
        let mut missing = run_manifest_value();
        missing.as_object_mut().unwrap().remove("identity");
        let envelope = build_envelope(&missing, &SEED, 0).unwrap();
        assert!(verify_envelope(&envelope).is_err());
    }

    #[test]
    fn signed_v2_identity_survives_object_reordering() {
        let mut manifest = run_manifest_value();
        let schema = crate::cli_manifest::EXECUTION_IDENTITY_SCHEMA;
        manifest["identity"]["schema"] = schema.into();
        manifest["identity"]["canonical"]["schema"] = schema.into();
        manifest["identity"]["sha256"] =
            crate::cli_manifest::recompute_identity_sha256(&manifest["identity"]["canonical"])
                .unwrap()
                .into();
        let envelope = build_envelope(&manifest, &SEED, 0).unwrap();
        let reordered = serde_json::from_slice(&canonical_bytes(&envelope).unwrap()).unwrap();
        assert!(verify_envelope(&reordered)
            .unwrap()
            .identity_digest
            .is_some());
    }

    #[test]
    fn arbitrary_record_identity_is_not_misclassified_as_a_run_manifest() {
        let record = serde_json::json!({"identity": {"name": "example"}});
        let envelope = build_envelope(&record, &SEED, 0).unwrap();
        assert!(verify_envelope(&envelope)
            .unwrap()
            .identity_digest
            .is_none());
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
    fn v1_envelopes_still_verify_but_their_timestamp_is_unsigned() {
        let manifest = run_manifest_value();
        let mut envelope = build_envelope_v1(&manifest, &SEED, 1_700_000_000).unwrap();
        let verified = verify_envelope(&envelope).unwrap();
        assert_eq!(verified.schema, EVIDENCE_SCHEMA_V1);
        assert!(!verified.timestamp_signed);
        assert!(verified.identity_digest.is_some());
        // The v1 signature does not cover the timestamp: editing it still
        // verifies, which is exactly what v2 fixes.
        envelope["signed_at_unix"] = 1.into();
        assert_eq!(verify_envelope(&envelope).unwrap().signed_at_unix, Some(1));
        // v1 still protects the input.
        envelope["input"]["execution"]["prompt"] = "tampered".into();
        assert!(verify_envelope(&envelope).is_err());
    }

    #[test]
    fn v2_signature_covers_every_envelope_field() {
        let manifest = run_manifest_value();
        let envelope = build_envelope(&manifest, &SEED, 1_700_000_000).unwrap();
        assert_eq!(envelope["schema"], EVIDENCE_SCHEMA);
        let verified = verify_envelope(&envelope).unwrap();
        assert!(verified.identity_digest.is_some());
        assert_eq!(verified.signer_fingerprint.len(), 64);
        assert!(verified.timestamp_signed);
        assert_eq!(verified.signed_at_unix, Some(1_700_000_000));
        let tamper: [(&str, serde_json::Value); 4] = [
            ("signed_at_unix", 1_700_000_001.into()),
            ("schema", EVIDENCE_SCHEMA_V1.into()),
            ("algorithm", "ed25519ph".into()),
            ("extra_claim", "trusted-hardware".into()),
        ];
        for (field, value) in tamper {
            let mut edited = envelope.clone();
            edited[field] = value;
            assert!(verify_envelope(&edited).is_err(), "{field}");
        }
        let mut missing = envelope.clone();
        missing.as_object_mut().unwrap().remove("signed_at_unix");
        assert!(verify_envelope(&missing).is_err());
        // Envelope field order is irrelevant to the signature. (A record
        // without a legacy insertion-order identity digest, which would
        // itself be order-sensitive.)
        let record = serde_json::json!({"z": 1, "a": {"y": [1, 2], "b": "x"}});
        let envelope = build_envelope(&record, &SEED, 5).unwrap();
        let reordered = serde_json::from_slice(&canonical_bytes(&envelope).unwrap()).unwrap();
        verify_envelope(&reordered).unwrap();
    }

    #[test]
    fn trusted_signer_is_enforced_for_both_schemas() {
        let fingerprint = hex_encode(&SigningKey::from_bytes(&SEED).verifying_key().to_bytes());
        let other = hex_encode(
            &SigningKey::from_bytes(&[9u8; 32])
                .verifying_key()
                .to_bytes(),
        );
        for envelope in [
            build_envelope(&run_manifest_value(), &SEED, 0).unwrap(),
            build_envelope_v1(&run_manifest_value(), &SEED, 0).unwrap(),
        ] {
            let verified = verify_envelope(&envelope).unwrap();
            assert!(check_trusted_signer(&verified, &fingerprint).is_ok());
            assert!(check_trusted_signer(&verified, &other).is_err());
        }
    }

    #[test]
    fn canonical_bytes_ignore_key_order() {
        let a = serde_json::json!({"b": 1, "a": {"d": 2, "c": [3, 4]}});
        let b = serde_json::json!({"a": {"c": [3, 4], "d": 2}, "b": 1});
        assert_eq!(canonical_bytes(&a).unwrap(), canonical_bytes(&b).unwrap());
    }

    #[test]
    fn non_ascii_hex_is_an_error_not_a_panic() {
        assert!(hex_decode("aé").is_err());
        let mut envelope = build_envelope(&run_manifest_value(), &SEED, 0).unwrap();
        envelope["signature_hex"] = serde_json::json!("é".repeat(64));
        assert!(verify_envelope(&envelope).is_err());
    }

    #[test]
    fn key_files_are_private_and_never_overwritten() {
        let dir = std::env::temp_dir().join(format!("ember-evidence-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let key = dir.join("evidence.key");
        write_key_file(&key, "seed").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&key).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let err = write_key_file(&key, "other").unwrap_err();
        assert!(err.to_string().contains("refusing to overwrite"), "{err}");
        assert_eq!(std::fs::read_to_string(&key).unwrap(), "seed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn trusted_key_accepts_a_pub_file_or_a_fingerprint() {
        let fingerprint = hex_encode(&SigningKey::from_bytes(&SEED).verifying_key().to_bytes());
        assert_eq!(read_trusted_fingerprint(&fingerprint).unwrap(), fingerprint);
        let upper = fingerprint.to_ascii_uppercase();
        assert_eq!(read_trusted_fingerprint(&upper).unwrap(), fingerprint);
        assert!(read_trusted_fingerprint("abcd").is_err());
    }
}
