# Code changes: first-class KV snapshots

**Working-tree date:** 2026-08-08
**Snapshot schema:** `ember.kv-snapshot.v1`
**Scope:** same-model Llama/Qwen KV-prefix export, verification, strict import, greedy replay, deterministic comparison, and controlled continuation diagnostics. No cross-model mapper was implemented.

## Files changed

| File | Change |
|---|---|
| `.gitignore` | Whitelists the five requested tracked design/change documents. |
| `src/atomic_file.rs` | Adds atomic no-replace publication for trace sidecars so concurrent destinations cannot be silently overwritten. |
| `src/npy.rs` | Adds streaming NPY no-replace publication; native/replay validation rows no longer accumulate in an unbounded RAM vector. |
| `src/kv_cache.rs` | Adds fallible allocation, geometry/stride inspection, and crate-private compact f16 prefix copy/import. The live layout and decode API remain unchanged. |
| `src/kv_snapshot.rs` | Adds the independent manifest/schema, deterministic f16le serialization, verification, compatibility report, execution fingerprint, atomic staging writer, safe loader, and fresh-cache import. |
| `src/kv_compare.rs` | Adds strict same-coordinate K/V comparison, deterministic global/layer-head metrics and thresholds, plus typed in-memory zero/scale controls that cannot masquerade as snapshots. |
| `src/kv_diagnostics.rs` | Adds same-input semantic attention-output/full-logit diagnostics and independent greedy divergence from two freshly imported caches. |
| `src/kv_transfer/mod.rs` | Adds the experimental transfer/key-space module boundary. |
| `src/kv_transfer/rope.rs` | Adds allocation-bearing, tested adjacent-pair/split-half forward/inverse RoPE utilities and stored-key/content conversion for supported metadata. |
| `src/cli_kv.rs` | Adds `ember kv export|inspect|verify|compare|replay|trace-native`, optional full-logit trace output, typed measurement/trace metadata, and named phase timing. |
| `src/lib.rs` | Exports `kv_snapshot`, `kv_compare`, `kv_diagnostics`, and `kv_transfer`. |
| `src/main.rs` | Registers and dispatches the nested `kv` command. |
| `src/llama.rs` | Completes the execution-plan cache key with capacity/model/tokenizer inputs; adds plan-cache and exact replay tests. No existing cache APIs were removed. |
| `src/plan.rs` | Documents KV stride units as scalar elements and corrects the test helper's byte-like stride values. The `v04-plan/1` serialized field set is unchanged. |
| `KV_ARCHITECTURE_AUDIT.md` | Detailed ownership/layout/cursor/RoPE/execution/provenance audit. |
| `KV_FIRST_CLASS_PLAN.md` | Incremental design, gates, stop conditions, and deferred work. |
| `docs/kv-snapshots.md` | User/schema/security/reproducibility documentation. |
| `docs/kv-transfer-research.md` | Conservative design note for future arXiv:2608.03893-like research. |
| `CODE_CHANGES_KV_SNAPSHOT.md` | This change and validation record. |
| `scripts/validate_kv_replay_matrix.py` | Runs the independent-process real-GGUF matrix, exact f32-bit comparisons, input/inventory checks, phase/process timing, and safe evidence publication. |
| `artifacts/benchmark-kv-v1/2026-08-08/` | Compact manifest, exact commands, and mechanically derived 12-cell matrix summary. Large/sensitive raw data stays under ignored `runs/`. |

No v0.5 source, experiment schema, bundle schema, hook meaning, bundle payload, paper output, or pilot artifact was changed. `Cargo.toml` and `Cargo.lock` were not changed; raw deterministic f16le payloads use existing dependencies.

The pre-existing untracked `docs/road-to-1.0-plan.html` was not modified.

## Public APIs added

### Live cache

- `KVCache::try_new(...) -> Result<KVCache, String>`
- `KVCache::n_layers()`
- `KVCache::element_strides()`

The compact f16 copy/import helpers are `pub(crate)`, so external callers cannot bypass snapshot compatibility validation. Import allocates fresh owned storage and copies bits; it never aliases snapshot buffers.

### Snapshot artifact

The public snapshot module adds, among other typed metadata:

- `KvSnapshot`
- `KvSnapshotManifest`
- `KvCompatibilityTarget`
- `KvCompatibilityReport`
- `KvSnapshotCompatibilityMetadata`
- `KvRopeMetadata`
- `KvPayloadDescriptor`
- `KvSnapshotProvenance`
- `KvTransformProvenance` (reserved provenance only)
- `KvPrecision`, `KvLayout`, `KvRopeLayout`, `KvQkNormOrder`, `KvSnapshotOrigin`
- `validate_compatibility(...)`

Main operations:

- `KvSnapshot::export_native(...)`
- `KvSnapshot::verify()` / `verify_dir(...)`
- `KvSnapshot::save_dir(...)`
- `KvSnapshot::load_dir(...)` / `load_dir_with_limit(...)`
- `KvSnapshot::compatibility_report(...)`
- `KvSnapshot::import_cache(...)` / `import_cache_with_limit(...)`
- `KvCompatibilityTarget::live_cache_bytes()`
- deterministic summary/manifest/payload accessors

`KvCompatibilityTarget::from_execution_plan(...)` derives the supported Llama/Qwen target without modifying the frozen plan schema.

### Process-level validation trace CLI

- `ember kv trace-native` writes uninterrupted `[N, vocab]` full-f32 selection logits and typed metadata.
- `kv export --boundary-logits-output ... --metrics-output ...` writes the independent prefix-boundary row.
- `kv replay --logits-output ... --metrics-output ...` writes only the `N-1` logits rows recomputed after import.
- Trace rows stream to staged NPY files; no `N * vocab` full-logit vector is retained in RAM.
- Canonical path checks reject aliases inside the strict snapshot, model/tokenizer/executable collisions, and identical NPY/JSON outputs.
- Existing outputs require `--overwrite`; default publication is atomic no-replace rather than a racy preflight-only check.
- `ember.kv-replay-trace.v1` records row offsets, absolute positions, effective/stored resume IDs, cache capacity, model/tokenizer/plan/snapshot provenance, output hashes, and named timing phases.

Trace output is an opt-in validation facility, not a change to `ember.kv-snapshot.v1` or ordinary generation. Duplicated model/generation flags supplied before `kv` are rejected with a placement error rather than silently ignoring the top-level value; top-level `--k-strategy` and fallback controls remain valid.

### Pre-mapper comparison and continuation measurement

- `compare_snapshots(...)` verifies exact target/prefix coordinate alignment
  and reports deterministic global plus per-layer/head K/V cosine, MSE,
  optional directional R2, max-absolute error, and f16 bit mismatches.
- Optional max-absolute/MSE and minimum-cosine/R2 thresholds record every
  failing layer/head and the first exceedance without turning numerical
  difference into a process failure.
- `prepare_diagnostic_perturbation(...)` creates an owned in-memory zero/scale
  control for one typed layer/head and K/V selection. Its receipt covers the
  native source snapshot, exact f32 factor bits, operation, and affected counts.
  It is not serializable as a snapshot and never populates mapper provenance.
- `diagnose_continuation(...)` uses common reference-greedy teacher-forced
  inputs to compare zero-based semantic `attention-output` by layer, full
  logits, and top-1 agreement, then re-imports clean caches for independent
  greedy token sequence agreement and first divergence.
- `ember kv compare LEFT RIGHT --json` exposes the snapshot pair; omitting
  `RIGHT` requires a complete `--perturb-*` control. Supplying model,
  tokenizer, architecture, and a 2..=64 horizon adds continuation diagnostics.
- Comparison has 16-GiB aggregate payload and two-cache limits and a one-million
  layer/head report-row limit. Reports contain no timings, paths, host, or PID.
- Ordinary `kv replay` now explicitly requires native origin. Altered state is
  admitted only by the diagnostic candidate type and cannot leak into replay.

The snapshot boundary row is not reconstructed: the first forced measurement
feeds the common stored/overridden resume token at the restored cursor and
predicts the following token. Observer-route attention metrics and ordinary
unhooked greedy behavior remain separate because planned Q8 can dispatch them
differently.

### Transfer seam

- `KvKeySpace`
- `RopeDirection`
- `rotate_key_row_in_place(...)`
- `KvContentKeys`
- `stored_keys_to_content(...)`
- `content_keys_to_stored(...)`

These functions are offline/experimental. Production RoPE code was not rerouted through them, so ordinary decode numerics are not changed by the seam.

## Schema added

A snapshot directory contains exactly:

```text
manifest.json
keys.f16le
values.f16le
```

The schema is independently named `ember.kv-snapshot.v1`, kind `kv-prefix`, serialization `manifest-json+f16le-v1`.

K and V are compact raw IEEE-754 f16 bits, little endian, logical order:

```text
[layer][kv_head][position][head_dim]
```

Only `0..sequence_length` is serialized; unused live capacity and attention scratch are excluded. Each payload descriptor records fixed filename, element count, byte length, and SHA-256. `snapshot_hash` hashes deterministic typed manifest JSON with its own field cleared and therefore covers both payload hashes and all metadata/provenance.

The manifest includes compatible model/tokenizer hashes, architecture, sequence/source capacity, geometry, precision/layout, full RoPE and Q/K-norm semantics (including epsilon), V state, execution mode, full plan hash, capacity-independent execution fingerprint, native/transformed origin, source model, prefix-token digest/count, optional greedy resume token, and optional transform provenance.

## Compatibility rules

Import is fail-closed. There is no force/warning path. A native replay requires:

1. exact compatible-model SHA-256;
2. exact architecture;
3. exact f16 precision and compact logical layout;
4. exact layer count, KV-head count, and head dimension;
5. `sequence_length <= target.max_seq`;
6. exact RoPE layout, dimension count, theta bits, frequency layout, position origin, and post-RoPE key state;
7. exact Q/K-normalization order, presence flags, and epsilon;
8. exact V representation (`projection-output` in v1);
9. exact execution mode and capacity-independent execution fingerprint; and
10. exact tokenizer SHA-256 when a tokenized-prefix digest is present.

The full plan hash is retained as provenance but is not compared because a safe target capacity can differ. The execution fingerprint removes only plan hash, scratch sizing/layout, KV capacity/strides, capped context length, and plan-build time. It retains operation/dispatch/build semantics. The plan cache itself is now keyed by capacity and supplied model/tokenizer hashes, avoiding stale provenance or undersized score arenas.

`exact_same_model` is true only for a native snapshot whose source and compatible model hashes equal the target model hash. A future transformed snapshot can be target-compatible while correctly reporting `exact_same_model = false`, but no mapper or transformed-snapshot authoring path exists in this change.

## Tests added or extended

Snapshot tests cover:

- compact f16 bit-exact import/export and cursor restoration;
- deterministic manifest, snapshot identity, and payload checksums;
- directory serialization round trip;
- truncated payload;
- one-byte payload corruption;
- unexpected payload trailing bytes;
- unexpected directory files;
- malformed/overflowing dimensions;
- independent compact-payload and destination-live-cache allocation limits;
- overwrite refusal for arbitrary directories and dangerous working-directory ancestors;
- strict-snapshot-only verified replacement;
- manifest tampering;
- wrong model hash;
- wrong architecture;
- wrong head dimension;
- wrong KV-head count;
- wrong layer count;
- sequence too long;
- incompatible RoPE layout;
- incompatible Q/K-norm order;
- wrong tokenizer;
- wrong execution mode;
- empty prefix; and
- maximum valid prefix boundary.

RoPE tests cover hand vectors for adjacent-pair and split-half layouts, forward/inverse directions, multiple-head isolation, tight f32 stored→content→stored round trip, malformed shapes, and fail-closed after-RoPE K normalization.

Comparison/diagnostic tests cover exact duplicate payloads, localized one-head K/V perturbations, invalid factors, immutable source identity, altered-cache cursor/layout preservation, undefined finite JSON metrics, incompatible prefix rejection, clap forms, and a two-layer planned Llama control. The Llama control proves duplicate attention/logit/greedy identity, fixed teacher inputs, a measurable perturbed attention effect, and source re-verification.

NPY/atomic/CLI tests cover no-replace publication under a simulated concurrent destination, incomplete-stream cleanup, domain-separated prefix hashes, lexical path equality, and symlink-parent alias/snapshot-containment rejection. Manual clap checks confirm misplaced top-level KV options fail instead of being silently ignored.

Llama tests cover:

- execution-plan cache separation by capacity and supplied provenance; and
- same-model planned replay using the deterministic in-memory two-layer Q8 fixture: prefill, write disk artifact, drop source cache, reload/import, compare pre-import payload bits/cursor, then compare every continuation logit vector bit-for-bit and greedy token exactly.

The replay proof also counts allocations on a warmed ordinary planned-decode token after snapshot import and enforces Gate E's existing limit of three logits-tensor allocations. The pre-existing zero-steady-state-allocation test remains unchanged. No snapshot API is invoked in ordinary per-token decode.

## Commands run and results

### Required Rust checks

```text
cargo fmt -- --check
PASS

cargo clippy --all-targets --all-features -- -D warnings
PASS

cargo test
PASS
  library: 300 passed, 7 ignored
  binary: 38 passed
  integration: 46 passed
  env-gated parity harness invocation: 5 returned pass/skip without configured model
  property: 12 passed
  doctest: 1 ignored

cargo test --release
PASS
  same test groups; release profile built successfully

RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
PASS
```

### Focused snapshot/replay checks

```text
cargo test --lib kv_
PASS: 35 passed

cargo test --lib kv_snapshot_same_model_replay_is_bit_exact
PASS: exact synthetic continuation logits/tokens

cargo test --lib kv_continuation_diagnostics_duplicate_and_perturbed_control
PASS: duplicate hook/logit/greedy control and localized in-memory perturbation
```

### Real-model existing parity gate

```text
EMBER_PARITY_MODEL=models/v03-ladder/llama-3.2-1b-q4_k_m.gguf \
EMBER_PARITY_TOKENIZER=tokenizer.json \
EMBER_PARITY_ARCH=llama \
EMBER_PARITY_TOKENS=2 \
cargo test --release --test k_parity \
  v04_planned_matches_reference_real_model -- --nocapture
```

Result: **PASS**, one test, all six frozen English/Arabic prompts, 74.06 seconds. Frozen thresholds were not changed.

### Real-model CLI smoke/replay

Two real local Llama-3.2-1B runs were exercised:

- Q8_0, `reference`: export, inspect JSON, verify, replay three greedy tokens;
- Q4_K_M, `planned`: export and replay three greedy tokens from a two-token prefix.

For both, replay stdout was byte-identical to the ordinary native greedy-generation stdout for the same prompt (`", I'm"`). The planned Q4 snapshot independently verified and recorded model/tokenizer/payload/snapshot hashes.

### Real-model comparison/diagnostic smoke

A current-build Qwen2.5-1.5B Q6_K planned snapshot (model SHA-256
`c6bc806dd29f9dd3f32e320d90cd6f3facf94f2bdff0b13fc8311113a7f354d1`)
was exercised through `kv compare` with a two-token horizon:

- duplicate snapshot: K/V payload bit-exact, every paired attention/logit
  difference zero, final top-1 agreement true, and independent greedy sequence
  agreement true;
- in-memory layer-0/head-0 K+V zero control: raw comparison localized the
  selected head, layer-0 attention-output cosine was about 0.886, final-logit
  cosine about 0.9953, final top-1 still agreed, and no greedy flip occurred
  within the one predicted-token horizon.

This is a structural/differential smoke, not a transfer-quality or morphology
claim. The prompt-derived JSON was kept transient, the snapshot stayed under ignored
`runs/`, and neither was published as benchmark evidence.

### Real-model process-level full-logit matrix

```text
.venv/bin/python scripts/validate_kv_replay_matrix.py \
  --output runs/kv-replay-matrix/2026-08-08-full \
  --evidence-output artifacts/benchmark-kv-v1/2026-08-08 \
  --skip-build --tokens 4 --max-seq-len 256 \
  --observations 2 --threads 4 --timeout 900
```

Result: **PASS**, 12/12 cells and 24/24 chronological observations:

- Llama-3.2-1B and Qwen2.5-1.5B;
- Q8_0, Q6_K, and Q4_K_M;
- fixed English and Arabic prompts;
- 12 independent exports and verifies plus 24 native/replay process pairs;
- 72 successful subprocesses total; and
- 13,449,216 full-f32 values compared with zero bit mismatches.

For each observation, the gate proved
`native[N,V] == concatenate(export_boundary[1,V], replay[N-1,V])` and also checked token IDs/argmax alignment, model/tokenizer hashes, prefix hash, execution fingerprint, common capacity, snapshot identity, exact inventory, input stability, and output hashes. Regenerable raw snapshots/logits/process records occupy about 99 MiB under ignored `runs/`; only the compact, path-sanitized JSON evidence is tracked.

Two matched chronological timing observations and `/usr/bin/time` resource records were saved. They are explicitly marked observational: best-effort cache advice is defeated as a strict cold control by Ember's mandatory full-model hashing, OS residency was not measured, and two short samples cannot support a throughput or cold/warm speedup claim.

### Other repository checks

```text
.venv/bin/python -m pytest tests -q
PASS: 38 passed

.venv/bin/python scripts/check_docs.py
PASS
```

## Known limitations

- The CLI currently supports only uniform Llama/Qwen-family caches and only `reference|planned`. Export/replay use hooks disabled; opt-in continuation measurement uses observation hooks.
- Gemma has heterogeneous local/global geometry, shared source layers, different RoPE tables, and normalized V; it is deliberately rejected pending a richer per-layer schema and trusted reference validation.
- GPT-2 absolute-position caches are not represented by this RoPE-bearing v1 target.
- Replay is greedy. It stores the first boundary argmax token, not the full boundary logits. It has no RNG/sampler state and does not stop on EOS.
- `--token-id` starts an alternate branch; it is not reproduction of the saved greedy branch.
- Content-space conversion is f16-derived and approximate in f32. It rejects partial/nonuniform RoPE and after-RoPE K norm. Exact replay never uses it.
- There is no transformed-snapshot constructor, mapper fitting, ridge/MLP execution, layer selection, or cross-model CLI.
- Snapshot loading owns both payloads and import allocates a separate live cache; it is not mmap/streaming/zero-copy. Peak memory exceeds artifact size.
- The default compact-payload and destination live-cache limits are independently 16 GiB. Applications accepting untrusted artifacts should choose smaller explicit limits.
- Hashes provide integrity/identity, not authentication or encryption. KV payloads can leak information about prefixes.
- Overwrite is restricted to a re-verified strict snapshot and never the working directory/ancestors, but it still removes that snapshot before renaming staging, so replacement has a visibility gap; default behavior is atomic no-replace.
- Exact compatibility is intentionally strict across execution/build fingerprints and can reject a numerically equivalent environment rather than silently accept an unproved one.
- The broad real-model matrix covers requested `planned` mode with four tokens and one common capacity; a full `reference`/capacity-variation/eight-token matrix is not claimed.

## Deferred work

- Cross-model ridge/MLP mapping (critical constraint: **not implemented**).
- External transformed-snapshot construction and provenance tooling.
- `planned-fused` route/provenance cleanup and snapshot admission.
- Gemma per-layer geometry/shared-KV/value-state schema after Gemma numerical trust is restored.
- GPT-2/absolute-position snapshot profile if useful.
- Boundary logits plus versioned sampler/RNG state for stochastic checkpointing.
- EOS-aware replay parity.
- Streaming checksum/decode and lower-peak-memory import.
- Optional GUI snapshot metadata/prefix inspection.
- KV compare/diff diagnostics.
- Pre-registered, cache-controlled resident-model serialization/replay timing with enough repetitions for a performance claim (the recorded matrix timings are diagnostic only).

## Ordinary inference behavior and performance

Ordinary cached attention still consumes the same f16 arrays with the same indexing, RoPE timing, cursor advancement, and reference/fast/planned dispatch. No snapshot allocation, checksum, metadata validation, file I/O, or content-space conversion occurs unless a snapshot API/CLI is explicitly invoked. No semantic hook site was added or redefined.

The live-cache constructor now delegates to a fallible implementation but performs the same three preallocations outside the token loop. The plan-cache key correction only distinguishes inputs that should never have shared a plan. The production execution-plan schema and v0.5 bundle semantics are unchanged.

Existing zero-steady-state-allocation tests and the new post-import allocation regression pass in debug and release, and the real Q4 v0.4 planned/reference gate passes. The full-logit matrix records phase and process wall times plus peak RSS/faults, but it is not a controlled before/after ordinary-decode benchmark; therefore no percentage throughput claim is made. Based on code placement, bit-exact replay, allocation gates, and the unchanged parity gate, ordinary per-token behavior is unchanged; a baseline/candidate Gate-H timing comparison remains future work.
