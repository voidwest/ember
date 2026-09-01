# EmberSEC Phase III: Reproducible Execution: Execution Identity, Replay, Provenance

**Status:** landed in full (main, freeze tag `embersec-freeze-2026-08-31`,
implementation through `6b4723e1`); read the "Implemented" section for what
exists and "Deferred" for the roadmap ahead.
**Research question:** what exactly must be fixed for an inference result to
be reproducible and meaningfully attributable to a specific execution?

---

## 1. The trust boundary (from artifact safety to execution evidence)

Phase I/II hardened *hostile artifacts* (GGUF, tokenizer, image/audio bytes):
given a fixed program, garbage in → structured error out. Phase III asks the
next question: given a *specific execution*, what identifies it, and how do we
prove a recorded result came from that execution and no other?

The execution-affecting input surface, traced end-to-end:

```
model file ──► GGUF loader (arch, K-strategy, presplit)
tokenizer ──► tokenization (template, bos/eos)
prompt ─────► assembled input sequence
sampler ────► temperature / top-k / top-p / seed / RNG
runtime ────► rayon threads, CPU features, kernel dispatch
build ──────► binary commit + dirty flag + rustc + target
env ────────► behavior knobs (EMBER_*) that change numerics/paths
```

## 2. What already existed (inventory)

- `--write-run-manifest` (manifest schema v1): model SHA-256 + GGUF metadata,
  tokenizer SHA-256, git commit, rustc version, rayon threads, CPU features,
  k-strategy, sampler temperature/top-k/top-p, probe/experiment switches.
- `build.rs` embeds build-time `EMBER_GIT_COMMIT` / `EMBER_GIT_DIRTY` /
  `EMBER_RUSTC_VERSION` / `EMBER_TARGET` into the binary (used by trace.rs,
  plan_build.rs, agent/session.rs).
- v0.5 experiment bundles: semantic identity (sanitized execution plan,
  payload hash, `resolved-experiment.json`), `experiment reproduce/verify`.
- `SeededRng::Std(StdRng::seed_from_u64)` already existed in the generation
  loop but was only reachable via the v0.5 experiment path (`--seed` did not
  exist on the CLI; plain generation always used the thread-local RNG).

## 3. Gaps found (this phase)

1. **No canonical identity digest**: the manifest was a JSON blob; timestamps
   and argv made it non-reproducible as-is, and nothing bound its contents.
2. **Prompt not recorded**: the single most output-affecting input was absent.
3. **No seed on the CLI**: temperature > 0 runs were unreproducible across
   processes by construction.
4. **Behavior env knobs not recorded**: `EMBER_VISION_FAST_EXP` (changes
   softmax numerics), `EMBER_FUSED_GREEDY`, `EMBER_PRESPLIT`, `EMBER_K_AVX512`,
   `EMBER_LLAMA_PACKED_Q8`, `EMBER_PARALLEL_REPACK`, `EMBER_GEMMA_DUMP`,
   `EMBER_CONVERSE_DBG` were all invisible to the manifest.
5. **No verification path**: nothing recomputed or checked the recorded
   identity after the fact.

## 4. Execution identity: definition

**Execution identity = SHA-256 over a canonical, sorted JSON object of every
output-affecting input**, specifically:

| Group | Fields |
|---|---|
| binary | build-time git commit, git-dirty flag, rustc version, target triple |
| model | file SHA-256, architecture, K-strategy, allow-fallback |
| tokenizer | file SHA-256 |
| prompt | the exact prompt text |
| sampler | temperature, top-k, top-p, seed |
| limits | max tokens, max seq len |
| runtime | rayon thread count, detected CPU features |
| env | the behavior-knob snapshot (value or null) |
| mode | probe flags, experiment (zero-layer-output / activation-stats) |

Canonicality rules:
- **Sorted keys**: the canonical object is built from `BTreeMap` and the
  JSON written compactly, so field order is irrelevant and re-parsing the
  recorded object reproduces the same bytes.
- **Volatile data excluded**: timestamps, argv, paths are *reported* in the
  manifest but are not part of the identity.
- **Sensitivity**: any change to any listed field changes the digest
  (verified by unit tests: prompt, temperature, seed, arch, model hash, and
  env knobs each flip the identity).

Attribution semantics: two runs with the same identity are expected to produce
the same output *for the same binary behavior*; a result recorded alongside a
matching identity is attributable to that execution. A mismatch means the
record was edited, the environment differed, or the binary was rebuilt :
each a legitimate reason to distrust the result.

## 5. Implemented

- `--seed <u64>` CLI arg; threaded through `run_single_prompt` /
  `run_single_prompt_with_experiment` and recorded in the manifest. Same seed
  + same inputs ⇒ same token sequence for temperature > 0.
- Manifest schema v2 (`--write-run-manifest`):
  - `identity.{schema, sha256, canonical}`: canonical object + digest,
  - `execution.prompt`, `execution.seed`,
  - `binary.{git_commit, git_dirty, rustc_version, target}` (build-time),
  - env knob snapshot lives inside `identity.canonical.env`.
- `ember manifest verify <path>`: recomputes the digest from the recorded
  canonical object, prints `OK` + a summary, or fails with a mismatch
  (tamper detection). Verified end-to-end: intact manifest verifies, an
  edited `temperature` field fails.
- Tests: identity stability/sensitivity (prompt, temp, seed, arch, model,
  env knobs), JSON round-trip stability, tamper detection.
- Files: `src/cli_support.rs` (canonical + digest + manifest v2),
  `src/cli_manifest.rs` (verify), `src/main.rs` (--seed, `manifest`),
  `src/cli_generation.rs` / `src/cli_score_batch.rs` (seed plumbing).

## 6. Deferred / roadmap

- **Replay verification across machines**: `manifest verify` proves the
  record is self-consistent; cross-machine replay (run again, compare
  identity + outputs bit-exactly) is the natural `--verify` extension.
- **Sampler expansion**: repeat-penalty / min-p would join the sampler group.
- **Signed evidence**: LANDED in Phase IV (`ember evidence init|sign|verify`,
  `src/cli_evidence.rs`): the identity digest is signed under a local
  Ed25519 key; only the hardware-attestation step (IVb: TDX/SNP) remains
  deferred.
- **Kernel-dispatch recording**: per-tensor K decisions and the resolved
  v0.4 execution plan hash are candidates for the `mode`/`runtime` groups if
  bit-exactness across dispatch tiers ever needs to be asserted.
