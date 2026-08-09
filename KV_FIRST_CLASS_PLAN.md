# KV snapshots as a first-class Ember artifact

**Status:** incremental implementation and review plan
**Artifact schema:** `ember.kv-snapshot.v1`
**Working-tree baseline reviewed:** 2026-08-08, Ember `0.6.1`
**Scope:** deterministic same-model KV-prefix export, verification, strict import, and greedy replay; an experimental coordinate-space seam for later transfer research

## 1. Decision and non-claims

Ember should treat a KV prefix as an independently versioned research artifact, not as an incidental dump of the live cache and not as a new payload inside an `ember.bundle.v1` experiment bundle.

The first contract is intentionally narrow:

1. copy the initialized prefix of a Llama/Qwen-family f16 KV cache;
2. serialize its exact f16 bits in a deterministic, compact layout;
3. pin the model, tokenizer, execution, RoPE, and value semantics required for safe reuse;
4. reject every unproven compatibility case;
5. import into fresh owned storage, possibly with a larger capacity; and
6. reproduce same-model greedy continuation without changing ordinary inference.

Passing this contract demonstrates an Ember-internal checkpoint/replay property. It does **not** establish:

- numerical validity against llama.cpp or another reference;
- generation quality;
- compatibility between different model files;
- semantic equivalence between different tokenizers;
- validity of a learned cross-model mapper;
- support for Gemma 4; or
- a causal morphology result.

Those claims require their own gates below. There is no `--force` or “close enough” import path.

## 2. Review vocabulary

This plan uses four status labels:

- **Implemented** — present in the current working tree, but not necessarily released.
- **Observed** — exercised locally during this review; the command and result are stated.
- **Gate** — evidence required before the relevant phase is accepted.
- **Deferred** — deliberately excluded from the current schema/CLI rather than silently approximated.

No later phase should weaken an earlier gate merely to admit a new architecture or mapper.

## 3. Implemented substrate

### 3.1 Independent snapshot module

`src/kv_snapshot.rs` currently defines:

- schema `ember.kv-snapshot.v1`;
- kind `kv-prefix`;
- serialization `manifest-json+f16le-v1`;
- strict manifest types with `deny_unknown_fields`;
- native export from a completed `KVCache` prefix;
- deterministic manifest and payload hashing;
- exact directory loading and verification;
- allocation and metadata limits;
- a machine-readable compatibility report;
- verified import into a new cache; and
- concise human-readable inspection.

The on-disk artifact contains exactly three regular files:

```text
snapshot/
├── manifest.json
├── keys.f16le
└── values.f16le
```

There is no dependency on the v0.5 safetensors codec and no dependency change in `Cargo.toml`.

### 3.2 Live-cache boundary

`src/kv_cache.rs` now has a fallible `KVCache::try_new` for metadata-driven import. Ordinary inference retains `KVCache::new`.

Snapshot-only helpers:

- copy the active prefix from live strided storage to compact owned f16 vectors;
- copy compact f16 vectors into a newly allocated live cache without an f16→f32→f16 round trip;
- restore the cursor to the completed prefix length; and
- expose geometry/strides without exposing mutable raw buffers.

The live layout remains `[layer][head][position][dimension]`. The live head stride uses `max_seq * head_dim`; the serialized head stride uses `sequence_length * head_dim`. Uninitialized capacity and `qk_scratch` are never serialized.

### 3.3 Execution provenance

`KvCompatibilityTarget::from_execution_plan` derives current Llama/Qwen metadata from the frozen v0.4 execution plan.

Two execution identities have different jobs:

- `execution_plan_hash` is retained as full provenance;
- `execution_fingerprint` is the compatibility key.

The fingerprint removes plan hash, scratch layout, cache capacity/strides, GGUF context capacity, and plan build time. It retains the operation graph, dispatch, runtime build/provenance, model semantics, and execution mode that may affect continuation numerics. Therefore a prefix may be imported into a larger safe capacity without treating an otherwise identical target as a different execution.

The Llama execution-plan cache key now includes capacity and supplied model/tokenizer hashes. This prevents reuse of an undersized plan or stale provenance. It does not add a v0.4 schema field or change a correctly keyed plan's serialization.

### 3.4 CLI

`src/cli_kv.rs` and `src/main.rs` expose:

```text
ember kv export
ember kv inspect
ember kv verify
ember kv replay
```

Current CLI scope:

- explicit `--model`, `--tokenizer`, and `--arch`;
- architectures `llama` and `qwen3` (the latter accepts GGUF `qwen2`/`qwen3` family metadata);
- execution modes `reference` and `planned`;
- model and tokenizer SHA-256 computed before export/replay;
- export after full prompt prefill;
- stored greedy `resume_token_id` from the boundary logits;
- inspect as text or manifest JSON;
- offline structural/integrity verification; and
- strict same-model greedy replay.

`max_tokens` in replay includes the stored or overridden resume token. The resume token is output first; each subsequent step evaluates the preceding generated token at the restored cursor. The final emitted token need not be evaluated, matching the existing generation loop's no-unneeded-final-forward convention.

The stored resume token is convenience provenance, not part of the KV tensors. Full boundary logits, sampling RNG state, stochastic replay, EOS stopping, and `planned-fused` CLI replay are not part of this first contract.

### 3.5 Experimental transfer seam

`src/kv_transfer/rope.rs` implements an allocation-bearing, off-hot-path coordinate conversion for keys:

- stored post-RoPE f16 keys → owned f32 content-space keys;
- content-space f32 keys → post-RoPE f32 keys;
- adjacent-pair and split-half RoPE;
- forward and inverse rotation using Ember's table generator; and
- fail-closed rejection of unsupported normalization/partial-RoPE cases.

This is a coordinate-space utility, not a learned mapper. It does not construct a transformed snapshot, transform values, select layers/heads, align tokenizers, or claim cross-model replay.

### 3.6 Evidence observed during this review

Validation completed on 2026-08-08 includes:

```text
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo test --lib kv_
  26 passed, including the synthetic Llama disk replay test
```

The real Llama-3.2-1B Q4_K_M planned/reference parity test also passed on the six frozen prompt fixtures, and real Q8-reference and Q4-planned CLI snapshot smokes produced the same tested three-token greedy stdout as uninterrupted generation. A later independent-process matrix covered Llama/Qwen × Q8/Q6/Q4 × English/Arabic in requested `planned` mode: all 12 cells and 24 observations reconstructed 13,449,216 saved f32 logits with zero bit mismatches. Evidence is in `artifacts/benchmark-kv-v1/2026-08-08/`. This closes the broad planned-mode/four-token portion of Gate D, not its reference-mode, eight-token, capacity-variation, re-exported-KV, portability, or performance requirements.

## 4. Frozen boundaries

### 4.1 v0.4 remains frozen

This work must not change the meaning of `v04-plan/1`, the F1–F5 fusion set, kernel selection, hook defusion, scratch-arena semantics, or reference/planned arithmetic.

Allowed implementation-adjacent changes are limited to:

- correcting plan-cache key completeness;
- documenting that KV strides are scalar-element strides; and
- deriving a separate, capacity-independent snapshot fingerprint without adding fields to `ExecutionPlan`.

Snapshot export/import and transfer utilities are explicit, allocation-bearing calls. Ordinary prefill/decode must never be rerouted through serialization, compact copies, inverse RoPE, or mapper code.

Gate G below must re-run the frozen v0.4 correctness, allocation, and performance checks. Same-model replay is an additional check, not a replacement for golden logits.

### 4.2 v0.5 remains frozen

There are no changes to:

- `ember.experiment.v1`;
- `ember.bundle.v1`;
- semantic/payload identity rules;
- bundle file layout;
- v0.5 safetensors bytes;
- hook/capture/intervention semantics;
- bundle verification, comparison, or reproduction; or
- existing Paper 1/pilot artifacts.

A KV snapshot is not inserted into an existing v0.5 bundle. If a future experiment needs to cite one, it should do so through a separately reviewed schema/version rather than changing the bytes or identity of `ember.bundle.v1`.

No phase in this plan may regenerate or rewrite citable v0.5 outputs merely to demonstrate KV support.

### 4.3 Schema v1 remains uniform

`ember.kv-snapshot.v1` describes a uniform model-wide geometry and one uniform RoPE/value-state contract. Per-layer geometry or per-layer frequency metadata requires a new schema version. It must not be smuggled into v1 through optional, ambiguous fields.

## 5. Exact v1 artifact contract

### 5.1 Tensor bytes

Both payloads are raw f16 bit patterns in little-endian order:

```text
[layer][kv_head][position][head_dimension]
```

The logical shape is:

```text
[layer_count, n_kv_heads, sequence_length, head_dim]
```

For each payload:

```text
elements    = layer_count * n_kv_heads * sequence_length * head_dim
byte_length = elements * 2
```

Every product is checked. File size must equal `byte_length` exactly; truncation and trailing bytes both fail. Import preserves all f16 bits and copies into fresh owned storage.

`sequence_length` is the initialized prefix and restored cursor. `max_seq` records the source allocation capacity. A target capacity may differ, but it must be at least `sequence_length`.

An empty programmatic prefix is structurally supported. CLI export continues to reject a prompt that tokenizes to zero tokens.

### 5.2 Manifest identity

Each payload descriptor pins file name, element count, byte length, and SHA-256.

`snapshot_hash` is SHA-256 over the deterministic serialized manifest with `snapshot_hash` cleared. Payload checksums are inside that manifest and are therefore transitively covered.

Consequences:

- payload corruption changes verification;
- provenance tampering changes verification;
- field order is fixed by the Rust manifest types;
- output directory, timestamp, hostname, PID, and timing are absent;
- the source `max_seq` is part of snapshot identity even though it is not a target-compatibility equality key; and
- whitespace in the pretty on-disk JSON is not itself the scientific identity after parsing.

The directory must contain exactly the three canonical regular files. Symlinks, subdirectories, non-UTF-8 names, missing files, and extra files fail.

### 5.3 Provenance

A native export records:

- compatible model SHA-256;
- source model SHA-256, equal to the compatible model for native origin;
- tokenizer SHA-256 when token provenance is supplied;
- architecture;
- Ember version;
- execution mode, plan hash, and execution fingerprint;
- RoPE/QK-normalization semantics;
- prefix token count;
- a domain-separated SHA-256 over little-endian u32 token IDs; and
- optional greedy resume token.

Token IDs and prompt text are not stored in v1. The token hash proves equality only when a candidate token sequence is independently available; it is not reversible.

`origin=transformed` and transform provenance fields are reserved. Their presence does not mean that a mapper or transformed-snapshot constructor has been accepted.

### 5.4 Loader trust boundary

The current loader enforces:

- manifest size at most 1 MiB;
- bounded layer/head/dimension/sequence metadata;
- checked shape arithmetic;
- exact file metadata before payload allocation;
- independent default 16-GiB limits for compact payload reads and destination live-cache allocation, each with an explicit override API;
- fallible vector reservation; and
- full verification before cache import.

The current implementation loads owned payload vectors and may temporarily create additional byte buffers during save/verification. This is acceptable for the first correctness contract, but peak-memory behavior must be measured before advertising long-context operational use.

Atomic publication means a new artifact becomes visible by sibling-directory rename only after all three files are written and synced. Overwrite is restricted to a destination that re-verifies as a strict snapshot; arbitrary directories, the working directory/ancestors, staging collisions, and a no-overwrite race are rejected. It still removes that verified destination before rename, so crash-durable atomic replacement is **not** claimed and remains part of Gate F.

## 6. Exact compatibility contract

Import first calls full snapshot verification. Compatibility is then strict and produces all rejection reasons; one reason is sufficient to prevent allocation/import.

| Field/condition | v1 rule |
|---|---|
| Model SHA-256 | Must equal target. Native cross-model import always fails. |
| Architecture | Exact string equality. |
| Precision | Exact; currently f16 only. |
| Layout | Exact; currently compact layer/head/position/dimension only. |
| Layer count | Exact equality. |
| KV head count | Exact equality. |
| Head dimension | Exact equality. |
| Prefix length | Must be `<= target.max_seq`; source and target capacities need not equal. |
| RoPE layout | Exact equality. |
| RoPE dimension count | Exact equality. |
| RoPE theta | Exact f32 bit equality. |
| Frequency layout | Exact equality; currently `uniform-theta`. |
| Position origin | Exact equality; currently absolute, zero-based. |
| Stored-key state | Exact equality; currently post-RoPE. |
| QK norm order | Exact equality. |
| Q/K norm presence | Exact equality. |
| QK norm epsilon | Exact optional f32 bit equality. |
| Value state | Exact equality; currently `projection-output`. |
| Execution mode | Exact equality. |
| Execution fingerprint | Exact equality. |
| Full plan hash | Provenance only; not an equality key because safe capacity changes alter it. |
| Tokenizer | For a token-hashed prefix, both hashes must be known and equal. Otherwise, two known hashes must agree; unknown tokenization remains explicitly unknown. |

For a native snapshot, `source_model_sha256` must equal `model_sha256` and transform provenance must be absent.

A future transformed snapshot must name the **target** model in `model_sha256`, preserve the producer in `source_model_sha256`, set `origin=transformed`, and carry non-placeholder transform provenance. It still must satisfy every target geometry/execution/coordinate rule. The compatibility validator will not be loosened to make a mapper appear successful.

`exact_same_model` is a report classification: native origin plus matching source/compatible/target model hashes. It is not a second, weaker import mode.

## 7. Incremental phases and gates

### Phase 0 — substrate review and contract freeze

**Status:** implementation present; review in progress.

Review the current diff as a contract change, not merely as serialization code.

#### Gate A — scope and freeze boundary

Required:

1. Diff inventory identifies every touched file and why it is necessary.
2. No `src/v05/*` file or tracked v0.5 artifact changes.
3. `PLAN_SCHEMA_VERSION` remains `1`; serialized v0.4 plans for identical inputs remain identical.
4. No ordinary forward path calls `kv_snapshot` or `kv_transfer`.
5. Schema/kind/serialization names, layout order, hash algorithm, and replay semantics receive explicit maintainer review.
6. Known limitations in Sections 9–11 are accepted rather than hidden.

Exit: tag the schema text and test fixtures used for all later gates. Any incompatible format change after this point becomes `ember.kv-snapshot.v2`.

### Phase 1 — deterministic artifact and fail-closed import

**Status:** core implementation and unit tests present.

#### Gate B — serialization and integrity

Required automated cases:

- repeated export produces identical manifest structs, JSON bytes, f16 payload bytes, and snapshot hash;
- compact export/import/export preserves key/value f16 bits and cursor;
- zero-length and full-capacity boundaries;
- corrupted, truncated, and extended payloads;
- malformed JSON and unknown fields;
- manifest and provenance tampering;
- absolute/extra/missing names, subdirectories, and symlinks;
- duplicate publication target refusal;
- failed staging cleanup; and
- deterministic behavior under parallel test execution.

The existing targeted tests cover most structural cases. Add explicit symlink, staging-failure, and CLI-created byte-identity cases before closing the gate.

#### Gate C — compatibility matrix

Mutate and reject each compatibility dimension in Section 6 independently. Also prove the allowed case:

- identical source/target semantics;
- target capacity larger than source capacity;
- plan hashes differ only because of capacity; and
- capacity-independent execution fingerprints remain equal.

Add explicit mutations for precision/layout, every RoPE field, norm presence/epsilon, value state, execution fingerprint, missing tokenizer knowledge, and transformed-origin structural rules. A test that only changes a few geometry fields is insufficient.

### Phase 2 — same-model replay validation on real GGUFs

**Status:** synthetic replay passes; the 12-cell real-GGUF planned-mode/four-token process matrix passes, while the full Gate D expansion below remains partial.

#### Gate D — uninterrupted versus replayed continuation

Minimum model matrix:

| Family | Primary file | Quantization paths |
|---|---|---|
| Llama | Llama-3.2-1B-Instruct | Q8_0, Q6_K, Q4_K_M |
| Qwen | Qwen2.5-1.5B-Instruct through `--arch qwen3` | Q8_0, Q6_K, Q4_K_M |

Run `reference` and `planned` for every available combination. Use at least:

- a short ASCII prompt;
- a non-ASCII Arabic prompt;
- a prefix near a small configured capacity boundary; and
- import into both equal and larger target capacities.

For at least eight greedy continuation tokens, record at every boundary:

- uninterrupted full logits hash;
- replayed full logits hash;
- exact float-array equality;
- selected token equality;
- cursor equality; and
- re-exported active key/value hashes.

Pass criterion: bit-exact logits, token IDs, and active KV bytes at every compared boundary. If a mode is intentionally not bit-exact, it does not receive same-model replay support in v1.

Evidence bundle:

```text
artifacts/kv-snapshot-v1/<date>/
├── matrix.json
├── commands.txt
├── model-tokenizer-hashes.json
├── replay-results.jsonl
├── corruption-results.json
└── SUMMARY.md
```

This gate is differential Ember-versus-Ember validation. Existing llama.cpp golden-logit gates must still pass; replay equality cannot validate a shared inference error.

Completed partial evidence uses `scripts/validate_kv_replay_matrix.py` and stores compact, path-sanitized JSON under `artifacts/benchmark-kv-v1/2026-08-08/`; raw NPY/snapshot/process records stay under ignored `runs/`. The harness uses a common 256-position capacity to isolate replay bytes, two fixed prompts, four tokens, two chronological observations, exact inventory/provenance/token checks, and bitwise f32 comparisons. It deliberately labels timing non-claims because cache state is unverified and mandatory GGUF hashing warms data before inference.

#### Gate E — CLI contract

Required:

- clap parsing tests for all five subcommands and invalid mode/architecture combinations;
- `export → verify → inspect → replay` end to end on one Llama and one Qwen file;
- corrupt artifact and wrong model/tokenizer return nonzero without partial output;
- replay `--token-id` override behavior is explicit and tested;
- `max_tokens=0` and minimum-capacity arithmetic are tested;
- text and JSON output are stable enough for scripting, or missing `--json` modes are explicitly deferred; and
- help text states greedy-only replay, lack of EOS stopping, and that the first token is stored/overridden boundary output.

Do not describe replay as general generation-state restoration until full boundary logits and sampling state have a separately reviewed contract.

### Phase 3 — resource, security, and frozen-regression gates

#### Gate F — resource and publication behavior

Measure export, save, load, verify, and import peak RSS for representative prefix lengths. Record:

- live cache bytes;
- compact key/value bytes;
- temporary byte-buffer overhead;
- wall time;
- allocation-limit failures; and
- staging cleanup after injected I/O failure.

Before release, choose one overwrite policy:

1. refuse overwrite in the stable CLI; or
2. implement and test a crash-safe replacement protocol.

Do not call the current remove-then-rename overwrite path atomically replacing.

Directory/file symlink rejection, exact file inventory, manifest-size limit, payload-size limit, checked products, and fallible allocation are release blockers. TOCTOU hardening and streaming/mmap loading may be later work if the local trusted-artifact threat model is stated precisely.

#### Gate G — no v0.4/v0.5 regression

Run the project validation suite:

```bash
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
.venv/bin/python -m pytest tests -q
```

Also re-run:

- v0.4 golden logits for the supported primary models;
- v0.4 activation reference checks relevant to Llama/Qwen;
- Gate E zero-steady-state decode allocation;
- planned/reference decode parity; and
- v0.5 bundle verify/reproduce fixtures with unchanged semantic and payload identities.

Benchmark ordinary generation with no KV command. Export/import code must add zero steady-state decode work and no new hot-path allocation. Any performance movement beyond normal measurement noise blocks release until explained.

### Phase 4 — content-coordinate seam

**Status:** key-only RoPE utility and unit tests present; experimental.

#### Gate H — mathematical seam

Required before using the seam in mapper research:

1. hand-computed forward/inverse vectors for both pairing layouts;
2. multi-head and multi-position boundary tests;
3. comparison against actual stored keys captured before/after Ember's inference RoPE call;
4. Qwen before-RoPE K normalization retained in “content” semantics;
5. fail-closed after-RoPE K normalization;
6. fail-closed partial dimensions and non-uniform frequency tables;
7. reported f32 round-trip error envelope across realistic positions, including long positions; and
8. proof that a no-op path preserves original f16 bytes rather than inverse/forward re-quantizing them.

Passing Gate H validates only the coordinate conversion. It does not validate cross-model correspondence.

### Phase 4b — pre-mapper measurement harness

**Status:** implemented for strict same-coordinate snapshots and controlled
same-model in-memory perturbations.

`ember kv compare` verifies and aligns a reference/candidate pair, then reports
global and every layer/head K/V cosine, MSE, optional directional R2,
maximum-absolute error, f16 bit mismatches, optional threshold failures, and the
first exceedance in deterministic JSON. A model-backed mode measures semantic
attention-output and full logits on a reference-greedy teacher-forced path,
then re-imports clean caches for independent greedy sequence agreement and
first-divergence indexing.

The native-vs-altered causal control can zero or scale one K/V head across the
initialized prefix. It is deliberately ephemeral: the typed receipt pins the
source snapshot and exact operation, but no transformed snapshot is authored,
no reserved mapper field is populated, and ordinary replay remains native-only.
The first measurable continuation row is after feeding the common resume token;
the KV artifact alone does not contain the prefix-boundary activation/logit row.

This harness establishes measurement semantics and negative controls. It is
not evidence of cross-model correspondence and does not advance Gate I.

### Phase 5 — cross-model mapper research

**Status:** deferred. No mapper is implemented or implied by the reserved provenance fields.

Before writing a learned mapper, approve a separate preregistered design containing:

- mapper artifact schema and SHA-256 identity;
- source and target model/tokenizer hashes;
- exact source/target layer and head selection;
- key and value transformation definitions;
- token/position alignment policy;
- handling of unequal layer counts, KV heads, head dimensions, and tokenizer segmentations;
- train/dev/test split with prompt-family leakage controls;
- baselines (identity where defined, mean, random orthogonal, and untrained mapper);
- reconstruction metrics in content and target stored spaces;
- target-logit and greedy-continuation criteria; and
- negative/control prompts appropriate to the morphology question.

The first mapper should run outside ordinary inference and consume verified snapshots. Its output constructor must:

1. preserve the source model hash in provenance;
2. name the target model as the compatible model;
3. pin the mapper artifact hash;
4. state the layer mapping and transformation type;
5. generate target-coordinate K and V payloads;
6. pass ordinary strict target compatibility; and
7. remain distinguishable from `origin=native`.

#### Gate I — mapper validity

A mapper is not accepted because tensor shapes line up or reconstruction loss falls. Required held-out evidence, in order:

1. verified target-coordinate payload structure;
2. improvement over preregistered controls on held-out reconstruction;
3. target next-token logit comparison against a native target prefix;
4. greedy token/trajectory comparison;
5. robustness across prompt classes and prefix lengths; and
6. only then, causal or behavioral morphology experiments.

Report failure and null results. Never relax model SHA, execution, RoPE, tokenizer, or value-state checks to manufacture compatibility.

Cross-tokenizer mapping is a distinct research problem. A source `resume_token_id` is not meaningful in a different target vocabulary. Cross-model replay therefore needs a target-space boundary-logit/token policy, not a copied source token ID.

### Phase 6 — Gemma 4 admission

**Status:** deferred and blocked on trusted Gemma inference.

Gemma must not be represented by the uniform v1 contract. Current Gemma execution can vary by layer in:

- local/global attention type;
- KV head count;
- head dimension;
- local/global RoPE theta;
- custom/partial RoPE frequency data;
- cache padding inside a maximum geometry; and
- value semantics, including V RMS normalization.

Gemma also lacks the current Llama/Qwen execution-plan-derived compatibility target, and repository guidance treats Gemma outputs as untrusted pending numerical validation.

#### Gate J — Gemma prerequisite

Before snapshot design:

1. restore trusted Gemma golden-logit evidence against a reference;
2. validate internal attention/KV states for local and global layers;
3. specify per-layer active shapes and padding exclusion;
4. specify each layer's frequency table/dimension/theta and position/window semantics;
5. name stored K and V states exactly, including V normalization;
6. define a trusted compatibility-target builder; and
7. decide whether the result is `ember.kv-snapshot.v2` or a distinct Gemma serialization.

Then repeat Gates B–G with real Gemma files. Until all prerequisites pass, CLI architecture validation must continue to reject Gemma.

## 8. Release threshold

The independently versioned v1 artifact may be called first-class when Gates A–G pass and their evidence is checked in or otherwise preserved reproducibly.

Gate H may ship as an explicitly experimental utility if its limitations are documented. Gates I and J are not required for same-model v1 release because cross-model mapping and Gemma are separate research programs.

A release summary should say only:

> Ember can deterministically export, verify, strictly re-import, and greedily replay compatible Llama/Qwen KV prefixes under the tested same-model execution contracts.

It must not say “portable across models,” “model-agnostic,” or “validated for Gemma.”

## 9. Explicitly deferred v1 features

- full boundary-logit storage;
- RNG/sampler-state checkpoints;
- temperature/top-k/top-p replay;
- EOS-aware parity with general generation;
- planned-fused CLI replay;
- prompt text or recoverable token-ID storage;
- persisted partial-cache/layer editing or transformed-snapshot authoring;
- streaming or mmap payload loading;
- compressed or quantized KV precision other than f16;
- remote/untrusted multi-user artifact service hardening;
- v0.5 bundle embedding;
- cross-model learned mapping;
- cross-tokenizer alignment; and
- Gemma 4.

Each is additive only if it preserves v1 interpretation; otherwise it requires a new schema.

## 10. Review checklist

### Artifact reviewer

- [ ] Three-file layout and byte order are unambiguous.
- [ ] Snapshot identity covers all deterministic compatibility/provenance data.
- [ ] No volatile path/time/machine field enters identity accidentally.
- [ ] Unknown fields, extra files, symlinks, truncation, and trailing bytes fail.
- [ ] Allocation bounds and peak-memory behavior are acceptable.

### Inference reviewer

- [ ] Compact indexing matches live indexing for every layer/head/position.
- [ ] Cursor semantics match uninterrupted prefill/decode.
- [ ] Execution fingerprint excludes capacity only, not numerical semantics.
- [ ] Reference and planned real-model replays pass independently.
- [ ] Normal inference and v0.4 performance remain unchanged.

### Research reviewer

- [ ] Same-model replay claims are separated from reference validity.
- [ ] RoPE conversion claims are separated from mapper validity.
- [ ] Native cross-model import remains impossible.
- [ ] Mapper splits, controls, and behavioral criteria are preregistered.
- [ ] Gemma stays blocked until its inference and per-layer contract are trusted.

### v0.5 artifact reviewer

- [ ] No v0.5 schema or payload bytes changed.
- [ ] Existing bundles retain their semantic/payload hashes.
- [ ] Bundle verify/reproduce behavior is unchanged.
- [ ] Paper/pilot outputs were not rewritten.

## 11. Stop conditions

Stop and require redesign rather than patching around a failure if:

- a supposedly compatible same-model replay changes one logit bit under a claimed bit-exact mode;
- compatibility needs a model-SHA bypass;
- a new model needs per-layer geometry hidden inside uniform v1 fields;
- mapper evaluation requires target-test prompts during training or selection;
- ordinary inference must call allocation-bearing transfer code;
- v0.4 plan or v0.5 bundle identities must be changed to carry the artifact; or
- a smoke/round-trip result is being presented as external numerical or behavioral validation.

The first-class property comes from a narrow contract that fails closed, not from maximizing the number of models or transforms that the CLI will accept.
