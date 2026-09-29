# EmberSEC Phase IV: Attested Execution: Signed Evidence (pre-TEE)

**Status:** landed (main, freeze tag `embersec-freeze-2026-08-31`);
attestation (TDX/SNP) is the documented next step, but the useful work :
what a signature must cover and how verification works: is complete without
any TEE hardware.

---

## 1. What "attested execution" means here

Phase III gave every run a canonical **execution identity** (SHA-256 over all
output-affecting inputs) and a tamper-evident manifest record. Phase IV binds
that record to a key:

```
record JSON (run manifest, includes identity)
    │  canonicalize: recursively sorted keys, compact JSON
    ▼
canonical input bytes ──sha256──► digest_sha256
    ▼
signed object = { schema: "signed-evidence-v2", algorithm, signer_fingerprint,
                  signed_at_unix, digest_sha256, input: <record> }
    │  canonicalize
    └──ed25519 sign──► signature_hex
    ▼
signed-evidence-v2 envelope = signed object + { signature_hex }
```

Verification recomputes the canonical bytes from the **embedded** input
(so the envelope is self-contained and verifiable offline), checks the digest
and the Ed25519 signature over the canonical signed object, and: when the
record is a v2 run manifest: also re-checks the internal execution identity
(`identity.sha256` vs `identity.canonical`). A v2 envelope with any field
besides the signed ones and `signature_hex` is rejected, so nothing in it is
unauthenticated. Three independent integrity layers:

1. **Envelope digest**: catches any edit to the record after signing.
2. **Ed25519 signature**: binds the record, the schema, the algorithm, the
   signer fingerprint and the signing time to a specific signing key
   (`verify_strict`, so malleability attacks fail closed). Changing the
   schema to `signed-evidence-v1` does not downgrade the check: the v2
   signature is not a valid v1 signature over the input alone.
3. **Execution identity**: catches a record that was already internally
   inconsistent before signing (e.g. a manifest whose canonical section was
   edited without updating its digest).

## 2. Honest threat model (what this does NOT prove)

- A valid signature proves the record bytes are exactly what the key holder
  signed. `signed_at_unix` is the key holder's own claim of when it signed:
  authenticated in v2 (not in v1), but not a trusted timestamp. It does
  **not** prove the execution happened on
  trusted hardware, that the model/tokenizer files were the originals (only
  that their recorded hashes were signed), or that the key holder is who
  they claim.
- The local key is only as safe as the machine it lives on (mode 0600).
- Hardware attestation (TDX/SNP) is the layer that would bind the signing
  key to an enclave-measured environment; the envelope schema is designed so
  that replacement is a drop-in (the key material changes, the record format
  does not).

## 3. CLI

```bash
ember evidence init --key ~/.config/ember/evidence.key
#   writes the private key (0600) + <key>.pub; prints the fingerprint

ember evidence sign --manifest run.json --key ~/.config/ember/evidence.key
#   writes run.json.signed.json (or --out)

ember evidence verify run.json.signed.json --trusted-key ~/.config/ember/evidence.pub
#   OK  signature valid (ed25519, trusted signer <fingerprint>)
#       digest <sha256> (canonical input sha256)
#       schema signed-evidence-v2
#       signed_at_unix <t> (covered by the signature)
#       execution identity <sha256> (verified)
```

Without `--trusted-key`, `verify` checks the signature against the key the
envelope names itself. That shows the record is intact, not who signed it:
anyone can re-sign an edited record with a fresh key. Pin the signer's `.pub`
file (or its hex fingerprint) whenever the signer's identity matters.

`sign` writes `signed-evidence-v2`. `verify` still accepts
`signed-evidence-v1` envelopes from earlier releases; their signature covers
only the canonical `input`, so `verify` reports that their `signed_at_unix`
(like their other envelope fields) is not covered by the signature. Re-sign
the record to obtain a v2 envelope.

### Anchoring an experiment bundle

A bundle's `manifest.json` carries its semantic and payload hashes. Signing
it gives others an external anchor for `ember experiment verify`:

```bash
ember evidence sign --manifest runs/example/manifest.json --key evidence.key \
  --out example.evidence.json
ember experiment verify runs/example \
  --expect-evidence example.evidence.json --trusted-key evidence.pub
```

The check passes only when the envelope verifies against the trusted key and
its signed `semantic_hash` and `payload_hash` equal the values recomputed from
the bundle (see [reproducibility](../reproducibility.md#what-verified-means)).

Key format: hex-encoded 32-byte Ed25519 seed (private) / 32-byte public key.
`init` refuses to overwrite an existing key.

## 4. Implementation notes

- `src/cli_evidence.rs`: envelope build/verify are pure functions
  (`build_envelope`, `verify_envelope`) over `serde_json::Value`; the CLI is
  a thin file wrapper. Canonicalization (`canonical_bytes`) sorts keys
  recursively so field order never affects the signature.
- Dependency: `ed25519-dalek` 2.x (pure Rust, `verify_strict`).
- Tests: round trip with identity check, tampered input → digest mismatch,
  tampered signature → verification failure, wrong signer fingerprint →
  failure, key-order-invariant canonical bytes; v2: tampered timestamp,
  schema (downgrade), algorithm, or an added field → failure; v1 envelopes
  still verify with their timestamp reported as unsigned.
- Integration: `verify_envelope` reuses
  `cli_manifest::recompute_identity_sha256`, so Phase III and IV share one
  identity implementation.

## 5. Roadmap to hardware attestation (Phase IVb)

1. TEE-present build: inside the enclave, generate/attest a key whose
   fingerprint is bound to the measured environment (TDX quote / SNP
   attestation report).
2. `evidence sign` accepts an attested key source; the envelope gains an
   `attestation` section (quote, PCRs/measurements, nonce) under a new
   schema version, since every v2 field is signed and v2 admits no extra
   fields.
3. `evidence verify` checks the attestation when present and reports
   "locally signed" otherwise: a single verification path for both eras.
