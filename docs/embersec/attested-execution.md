# EmberSEC Phase IV — Attested Execution: Signed Evidence (pre-TEE)

**Status:** local-key signed evidence landed (main @ freeze `6b4723e1`);
hardware
attestation (TDX/SNP) is the documented next step, but the useful work —
what a signature must cover and how verification works — is complete without
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
canonical bytes ──sha256──► digest_sha256
    │
    └──ed25519 sign──► signature_hex
    ▼
signed-evidence-v1 envelope:
  { schema, algorithm, signer_fingerprint, signed_at_unix,
    digest_sha256, input: <record>, signature_hex }
```

Verification recomputes the canonical bytes from the **embedded** input
(so the envelope is self-contained and verifiable offline), checks the digest
and the Ed25519 signature, and — when the record is a v2 run manifest — also
re-checks the internal execution identity (`identity.sha256` vs
`identity.canonical`). Three independent integrity layers:

1. **Envelope digest** — catches any edit to the record after signing.
2. **Ed25519 signature** — binds the record to a specific signing key
   (`verify_strict`, so malleability attacks fail closed).
3. **Execution identity** — catches a record that was already internally
   inconsistent before signing (e.g. a manifest whose canonical section was
   edited without updating its digest).

## 2. Honest threat model (what this does NOT prove)

- A valid signature proves the record bytes are exactly what the key holder
  signed at `signed_at_unix`. It does **not** prove the execution happened on
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

ember evidence verify run.json.signed.json
#   OK  signature valid (ed25519, signer <fingerprint>)
#       digest <sha256> (canonical input sha256)
#       execution identity <sha256> (verified)
```

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
  failure, key-order-invariant canonical bytes.
- Integration: `verify_envelope` reuses
  `cli_manifest::recompute_identity_sha256`, so Phase III and IV share one
  identity implementation.

## 5. Roadmap to hardware attestation (Phase IVb)

1. TEE-present build: inside the enclave, generate/attest a key whose
   fingerprint is bound to the measured environment (TDX quote / SNP
   attestation report).
2. `evidence sign` accepts an attested key source; the envelope gains an
   optional `attestation` section (quote, PCRs/measurements, nonce) without
   changing `signed-evidence-v1`'s core fields.
3. `evidence verify` checks the attestation when present and reports
   "locally signed" otherwise — a single verification path for both eras.
