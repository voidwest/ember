# EmberSEC: Security Research Program Map (Phases I–V)

**Freeze record:** tag `embersec-freeze-2026-08-31` on `main`, 2026-08-31.
The tagged commit is the final docs commit; implementation commits run
through `6b4723e1`.
**CI at freeze:** `ci` ✅ (run 33353747668, 15m) · `parser-fuzz` ✅ (run 33353747698, 41m,
8 targets) · pages ✅: all green on the freeze commit and re-verified after the
documentation freeze push.

EmberSEC is the security research program over **ember**, a CPU-first Rust
inference research layer. It moves up one trust boundary per phase: hostile
artifacts → hostile media → execution evidence → attested execution →
quantized-inference faults. Each phase has a threat model, a bounded
implementation, and an honest "what this proves / does not prove" statement;
the claims ladder below keeps engineering hardening separate from
paper-sized claims.

---

## Phase map

| # | Phase | Scope / threat model | Code artifacts (main) | Findings (measured or code-verified) | Proves | Does NOT prove |
|---|---|---|---|---|---|---|
| **I** | Hostile model artifacts | Attacker-controlled GGUF/tokenizer bytes → loader, tensor inventory, dequant | loader hardening (`src/loader.rs`, `src/quant_k.rs`, `src/tokenizer.rs`), cargo-fuzz targets, loader limits; full comparative audit lives on `embersec/secure-gguf-loader` (see note) | Structured rejection of malformed/oversized metadata; tokenizer-only inputs are `NOT_COMPARABLE`; observed external `GGML_ASSERT` crash surfaces; 0 exploitable RustSec advisories at audit time | Malformed artifacts fail closed with structured errors; length/shape math is checked before allocation | Does not prove absence of all decoder bugs; does not prove exploitability of observed external crashes (classified as crash/DoS, not memory corruption) |
| **II** | Hostile multimodal inputs | Attacker-controlled image/audio bytes + shapes → decode, preprocess, encoders, fusion | `src/multimodal/image.rs` (limits), `audio.rs` (WAV hardening + resample caps), `vision.rs`/`audio_encoder.rs` (shape checks), `request.rs` (validated types), `batch.rs` guards, fuzz targets `wav_bytes`/`image_bytes`/`image_preprocess` | WAV unsupported-format panic → `Err`; zero/absurd sample rates → 16,000× resample amplification and capacity-overflow panic, both closed; long-form audio DoS bounded (1 h / 16 segments); PNG/JPEG decode bounded (8192 px/edge, 256 MiB); encoder shape panics → `CpuError` | Hostile media cannot crash or exhaust the process; parsed state is validated at a single seam (`ValidatedImageInput`/`ValidatedAudioInput`) | Does not prove decoder-internal memory safety (pure-Rust crates, out of scope); fuzz campaigns are bounded smoke runs, not years of coverage |
| **III** | Reproducible execution | What must be fixed for a result to be reproducible and attributable | `--seed`, run-manifest v2 (`src/cli_support.rs`), `ember manifest verify` (`src/cli_manifest.rs`) | Execution identity = SHA-256 over a canonical sorted JSON of all output-affecting inputs (binary build, model/tokenizer hashes, prompt, sampler+seed, limits, threads, CPU features, behavior env knobs); tamper detection verified | Two runs with identical identity are expected to produce identical output; an edited record is detected | Does not prove the environment was honest (no TEE); does not prove bit-exactness across *different* machines/builds (that is replay verification, deferred) |
| **IV** | Attested execution (pre-TEE) | Bind a record to a key; later, to an enclave | `ember evidence init/sign/verify` (`src/cli_evidence.rs`, Ed25519, `ed25519-dalek`) | Self-contained signed-evidence envelope over canonical JSON; signature verified with `verify_strict`; embedded execution identity re-checked on verify | The record bytes are exactly what the key holder signed; provenance is key-attributable | Does not prove trusted hardware, honest environment, or signer identity: TEE attestation (IVb) is the remaining layer; envelope schema is designed for drop-in key replacement |
| **V** | Quantized-inference security | Single-bit faults in packed Q8_0/Q4_K/Q6_K weights | `src/quant_fault.rs` harness, `validate_integrity()` on both weight types, `EMBER_VERIFY_QUANT=1` load hook | Payload faults stay finite in the deterministic sweep; FP16 `d` exponent bits 10–14 are structurally dangerous (the current test probes 40 flips/dtype across 8 synthetic blocks and early-stops per block; bit 14 yields non-finite logits) | The dangerous fault class is the scale header, not the payload; integrity validation detects corrupted headers at load | Does not prove immunity to multi-bit/rowhammer patterns; does not cover ECC/DRAM behavior; determinism claims remain tied to the existing serial≡parallel and scalar≡SIMD parity tests |

**Phase I note:** the comparative GGUF security audit (llama.cpp/Candle findings,
tokenizer-comparability matrix, loader check harness, frozen artifact hashes)
is committed on the **`embersec/secure-gguf-loader`** branch, not on `main`;
the loader hardening it informed is part of main. See that branch for
`docs/embersec/fuzzing-plan.md` and `docs/embersec/loader-threat-surface.md`.

---

## Claim ladder: engineering hardening vs paper-sized claims

Not every phase belongs in one manuscript. The program separates cleanly:

**Engineering hardening (no manuscript by itself): Phases I, II, and the
tooling half of IV:**
- Structured rejection, bounded allocations, validated seams, fuzz targets,
  signed-envelope plumbing. These are audit-and-harden results: citable as
  repo artifacts and (if ever needed) as an "auditing CPU-first LLM
  inference stacks" experience report, but not a research claim.

**Paper candidate A: "Execution identity for reproducible inference
evidence" (Phases III + IV together):**
- The research question is attribution: *what exactly must be fixed for an
  inference result to be reproducible and meaningfully attributable to a
  specific execution?* The canonical-identity definition, the
  stability/sensitivity properties (verified: prompt, temperature, seed,
  arch, model hash, env knobs each flip the digest), and the signed-evidence
  envelope are one coherent methods story. This is the manuscript-sized
  contribution of the program and pairs naturally with the existing
  reproducibility work (v0.5 experiment contracts, semantic identity).

**Paper candidate B: Phase V as its own focused note (DECIDED: separate,
not an extension of A):**
- Phase V answers a *different* question (fault tolerance of quantized
  weights, not attribution of a run). It has self-contained fault-class and
  format checks, an analytical FP16 exponent-field finding, a hermetic seeded
  methodology, and a direct mechanism tie-in to the published pilot finding
  (near-threshold flips: payload faults only flip top-1 at small margins). It
  therefore deserves its **own focused note** rather than being folded into
  the execution-evidence manuscript.
- The bridge to A is real but small: `EMBER_VERIFY_QUANT` integrity
  verification is an input to attribution (a result is attributable to
  weights whose integrity was verified). That belongs as a one-paragraph
  extension in A (or a manifest field), not as shared content.

**Deferred paper surface: Phase IVb (TEE attestation):** hardware-bound
key attestation would extend paper candidate A with an environment-integrity
claim; not started, no hardware dependency on this host.

---

## Hostile-review log (outdated claims found and corrected at freeze)

1. `execution-identity.md` §1 status line referenced `75183e67` and called
   the phase "core landed": updated to the freeze commit and full-landed
   status.
2. `execution-identity.md` §6 listed "signed evidence (Phase IV)" as
   *deferred* after Phase IV had landed: rewritten to record the landing
   and leave only the IVb (hardware attestation) step deferred.
3. `multimodal-input-threat-surface.md` §2 described pre-fix crash paths in
   present tense with no status marker: annotated "as of audit time; all
   P1/P2 fixed (see §4)" so the historical findings cannot be misread as
   open vulnerabilities.
4. Cross-checked: `attested-execution.md` (IV) and
   `quantized-inference-security.md` (V) contained no stale commit/status
   claims; the §4 remediation table in the Phase II doc already records the
   fix commits; the Phase V findings table carries its measured numbers
   with no later-phase corrections.

---

## Frozen state

- Freeze tag: `embersec-freeze-2026-08-31` (main, pushed; implementation
  through `6b4723e1`, phase docs committed with the tag).
- Phase docs (this README + one per phase) are committed under
  `docs/embersec/`; the comparative Phase I docs remain on the
  `embersec/secure-gguf-loader` branch.
- CI at freeze: `ci`, `parser-fuzz` (8 targets incl. the Phase II
  wav/image/preprocess campaigns), and pages deployment all green; the
  `parser-fuzz` timeout was raised from 30 → 90 min (cold libFuzzer build
  needs ~32–41 min) and the workflow now watches `src/multimodal/**`.
- Open items deliberately out of scope for the freeze: IVb (TEE
  attestation), cross-machine replay verification, sampler expansion, and
  kernel-dispatch recording: all documented in the phase docs.
