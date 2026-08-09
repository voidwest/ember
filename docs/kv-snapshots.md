# KV snapshots

Status: implemented with independent schema `ember.kv-snapshot.v1`.

KV snapshots persist a completed attention-cache prefix so the same
Llama/Qwen-family model can continue deterministic greedy generation without
running that prefix through the model again. Snapshot work is explicit and
allocation-bearing: ordinary prefill and decode do not serialize, hash, or
validate snapshot metadata.

This document describes the implementation in `src/kv_snapshot.rs` and the
`ember kv` command family. A KV snapshot is not an activation artifact, an
`ember.bundle.v1` experiment bundle, or a replacement for the v0.4 execution
plan. Its schema is versioned independently of all three.

## Supported scope

The CLI supports the CPU Llama-family runtime for:

- `--arch llama` when GGUF metadata names the `llama` architecture;
- `--arch qwen3` when GGUF metadata names `qwen2` or `qwen3` (this includes
  the Qwen2.5 models handled by Ember's Llama-family loader);
- `reference` and `planned` execution with hooks disabled; and
- native, exact-same-model export and ordinary replay only; and
- explicit same-coordinate comparison/diagnostics, including an in-memory
  deterministic perturbation that is never published as a snapshot.

`planned-fused` is represented by the library metadata types, but is not
accepted by `kv export` and is explicitly rejected by `kv replay`. Gemma 4 and GPT-2 are not exposed by this CLI. No command authors a
transformed/cross-model snapshot, and ordinary replay rejects non-native
origins. `kv compare` may inspect an already valid aligned candidate and may
apply a typed same-model perturbation in memory for causal diagnostics; it does
not weaken replay admission or implement a mapper.

## Artifact layout

A snapshot is one real directory containing exactly three regular files:

```text
snapshot-dir/
├── manifest.json
├── keys.f16le
└── values.f16le
```

Extra files, subdirectories, symlinks, and non-UTF-8 entry names are rejected
when loading. The manifest cannot redirect either payload to another path;
payload descriptors must name the two fixed filenames above.

### Binary payloads

`keys.f16le` and `values.f16le` contain raw IEEE-754 binary16 bit patterns,
one little-endian `u16` per element. The logical order is:

```text
[layer][kv_head][position][dimension]
```

or, as a flat index:

```text
(((layer * n_kv_heads + kv_head) * sequence_length + position)
 * head_dim + dimension)
```

Each file therefore contains exactly:

```text
layer_count * n_kv_heads * sequence_length * head_dim
```

f16 elements and twice that number of bytes. The payload is compact: it
contains only positions `0..sequence_length`, not the unused tail of the live
cache's `max_seq` allocation. Import scatters the compact head rows into a new
owned live cache and copies f16 bits directly, without an f16-to-f32-to-f16
round trip. This permits a compatible replay target to use a different cache
capacity as long as the stored sequence fits.

The attention-score scratch buffer is not semantic KV state and is not
serialized.

### What K and V mean

For the supported Llama/Qwen runtime, cached K is the representation actually
consumed by cached attention: **after RoPE**, and after any Q/K normalization
at the architecture's configured side of RoPE. The manifest records:

- adjacent-pair or split-half RoPE layout;
- rotated dimension count and theta;
- `uniform-theta` frequency layout;
- `absolute-zero-based` position origin;
- `post-rope` stored-key state;
- Q/K normalization order, tensor presence, and epsilon.

Cached V is the f16 form of the unmodified value-projection output, recorded as
`value_state = "projection-output"`.

These meanings matter: stored keys must not be interpreted as raw K-projection
or position-independent content vectors. The experimental `kv_transfer::rope`
utility can allocate f32 content-space keys by removing and reapplying RoPE for
supported metadata, but it is approximate, is not used by ordinary replay,
and does not create cross-model snapshots. Exact same-model replay preserves
and imports the original f16 stored-key bytes without that conversion.

## Manifest schema

`manifest.json` is a typed JSON object with unknown fields rejected. Its main
fields are:

| Field | Meaning |
|---|---|
| `schema` | Exactly `ember.kv-snapshot.v1`. |
| `kind` | Exactly `kv-prefix`. |
| `serialization` | Exactly `manifest-json+f16le-v1`. |
| `model_sha256` | Lowercase SHA-256 of the compatible GGUF model. |
| `tokenizer_sha256` | Lowercase tokenizer-file SHA-256, or `null` for library-created snapshots without tokenizer provenance. |
| `architecture` | Architecture recorded by the execution plan. |
| `sequence_length` | Completed prefix positions stored in both payloads and restored as the cache cursor. |
| `max_seq` | Source cache/table capacity. It is provenance, not the compact payload shape and not an equality requirement for replay. |
| `layer_count`, `n_kv_heads`, `head_dim` | Uniform cache geometry. |
| `precision` | Currently only `f16`. |
| `layout` | Currently only `layer-head-position-dimension-compact`. |
| `rope` | Exact RoPE and Q/K-normalization semantics described above. |
| `value_state` | Currently only `projection-output`. |
| `provenance` | Runtime, plan, origin, prefix, resume-token, and transform provenance. |
| `keys`, `values` | Fixed filename, element count, byte length, and SHA-256 for each payload. |
| `snapshot_hash` | Deterministic snapshot identity described below. |

The `provenance` object records:

- Ember version;
- execution mode;
- full execution-plan hash when available;
- capacity-independent execution fingerprint;
- `native` or reserved `transformed` origin;
- source-model SHA-256;
- optional prefix token count and token-ID hash;
- optional greedy `resume_token_id`; and
- optional transform provenance.

The CLI always exports a native snapshot with model and tokenizer hashes, a
prefix token count/hash, and a greedy resume token. The prompt text and raw
prefix token IDs are **not** stored. The token-ID digest hashes domain-separated
little-endian `u32` IDs; it proves equality when the IDs are available
elsewhere, but cannot reconstruct the prompt.

A native manifest must name the same SHA-256 as source and compatible model and
must not contain transform provenance. The transformed form and its provenance
fields are reserved for future work; there is currently no CLI or learned
mapper that produces such an artifact.

### Checksums and identity

Each binary payload has its own SHA-256. `snapshot_hash` is SHA-256 over the
typed manifest's deterministic JSON serialization with `snapshot_hash` itself
set to the empty string. Because that manifest includes both payload
descriptors, the identity transitively covers the payload hashes, lengths,
shape, provenance, and resume token.

This is an integrity and deterministic-identity mechanism, not a digital
signature. Anyone able to replace an artifact can recompute all hashes.

## Strict compatibility

Import has no warning or "close enough" path. It first verifies the snapshot,
builds a target from the loaded model's execution plan, and rejects any
semantic mismatch. Compatibility requires:

1. identical model SHA-256 and architecture;
2. identical f16 precision and compact logical layout;
3. identical layer count, KV-head count, and head dimension;
4. `sequence_length <= target.max_seq`;
5. bit-identical RoPE theta and identical RoPE layout, dimension count,
   frequency layout, position origin, and stored-key state;
6. identical Q/K-normalization order, tensor-presence flags, and epsilon;
7. identical V representation (`projection-output`);
8. identical execution mode and execution fingerprint; and
9. matching tokenizer SHA-256 when tokenized-prefix provenance is present (as
   it always is for CLI exports).

The execution fingerprint is derived from the immutable execution plan. It
covers numerical operation/dispatch and runtime provenance, including the
selected execution and K-quant paths, CPU/thread plan, Ember build, compiler,
and model semantics. It deliberately normalizes fields that only resize a
compatible run: plan hash, scratch layout, KV capacity/strides, GGUF context
cap, and plan-build timestamp. Consequently:

- source and destination cache capacities may differ;
- the stored sequence still must fit the destination;
- the full plan hash remains recorded for provenance but is not a compatibility
  key; and
- changing execution mode, dispatch, runtime build, or another numerical plan
  input fails closed even when the GGUF hash matches.

The Llama execution-plan cache is keyed by capacity and model/tokenizer
provenance, so importing into a different permitted capacity cannot reuse an
undersized scratch plan or stale provenance.

## CLI

The examples assume the release binary is available as `ember`. With Cargo,
replace `ember` with `cargo run --release --`. Model, tokenizer, prompt,
capacity, token-count, architecture, and execution flags belong after the KV
leaf command as shown; supplying duplicated generation flags before `kv` is
rejected rather than silently ignored. Top-level `--k-strategy` and
`--k-allow-fallback` remain valid before `kv`.

### Export

```bash
ember kv export \
  --model Llama-3.2-1B-Instruct.Q6_K.gguf \
  --tokenizer tokenizer.json \
  --arch llama \
  --prompt 'Morphology is' \
  --execution planned \
  --max-seq-len 2048 \
  --output runs/kv/morphology-prefix
```

Export performs these steps:

1. hash the model and tokenizer files;
2. tokenize the prompt and reject an empty token sequence;
3. load and validate the requested Llama/Qwen-family model;
4. prefill the complete prompt into a new cache;
5. compute greedy argmax from the prefix's final logits;
6. build the compatibility target from a disabled-hook execution plan;
7. copy and verify the compact KV prefix; and
8. publish the three-file directory through a staging directory.

`--execution` is `reference` by default and accepts `reference|planned`.
`--max-seq-len` defaults to exactly the tokenized prefix length; it must be at
least that length. A value above the GGUF context is capped by the existing
model loader, while a resulting capacity below the prefix is rejected.
`--overwrite` is required to replace an existing output directory. Replacement
is allowed only when the destination re-verifies as a strict three-file KV
snapshot; arbitrary directories, the working directory, and its ancestors are
never recursively removed. Without the flag, export refuses to overwrite.

For Qwen2/Qwen3-family GGUFs, use `--arch qwen3`. Top-level weight strategy
options such as `--k-strategy` and `--k-allow-fallback` apply to export and
must be reproduced at replay closely enough to yield the same strict execution
fingerprint.

### Inspect

```bash
ember kv inspect runs/kv/morphology-prefix
ember kv inspect --json runs/kv/morphology-prefix
```

Inspect is not an unverified metadata peek: it loads and verifies the complete
snapshot before printing. Text output summarizes identity, model/tokenizer,
shape, RoPE/value semantics, execution provenance, and payload hashes. `--json`
prints the typed manifest.

### Verify

```bash
ember kv verify runs/kv/morphology-prefix
```

Verify checks directory structure, schema and metadata limits, exact payload
lengths, payload hashes, and manifest identity. On success it prints the
snapshot hash and path. It does not load a model or establish compatibility
with a particular replay target; `kv replay` performs that separate check.

### Compare and continuation diagnostics

`kv compare` is the opt-in measurement harness for two snapshots in the same
coordinate system:

```bash
ember kv compare REFERENCE_SNAPSHOT CANDIDATE_SNAPSHOT --json --r2
```

Both directories are fully verified before comparison. Elementwise comparison
requires exact equality of target model/tokenizer identity, architecture,
active shape `[layer, kv_head, sequence_length, head_dim]`, f16/layout and
K/V representation semantics, execution mode/fingerprint, and proven prefix
token-ID hash. It does not silently compare a common prefix. Source capacity,
full plan hash, origin/provenance, payload hashes, and resume token may differ
and are reported rather than used as tensor coordinates. Nonempty library
snapshots without a prefix token-ID hash are rejected because their alignment
cannot be proved.

The deterministic JSON envelope uses `ember.kv-measurement.v1` and embeds an
`ember.kv-compare.v1` report. It includes global and every zero-based
`(layer, head)` K/V metric:

- cosine similarity; numerically identical vectors are exactly 1 (including two
  zero vectors), while a non-identical zero-norm case is `null`;
- mean squared error accumulated in f64;
- optional directional R2 (`--r2`), where the first snapshot is the reference;
  a numerically exact constant reference is 1 and a non-identical constant
  reference is `null`;
- maximum absolute error; and
- f16 bit-mismatch count, which distinguishes numeric equality from payload
  equality (for example positive and negative zero).

Nonfinite f16 values are rejected rather than serialized as invalid JSON.
Thresholds are optional and apply to every per-head K and V leaf; an undefined
cosine/R2 fails a requested minimum. They are not implicit accuracy gates:

```bash
ember kv compare LEFT RIGHT --json \
  --max-abs 0.001 --max-mse 1e-6 --min-cosine 0.999 --min-r2 0.99
```

The report records all failing layer/head entries and the first exceedance in
stable layer/head order. A completed comparison exits successfully even when
values or thresholds differ; automation should read `thresholds_passed` or
`payload_bit_exact`. Malformed/unverified/unaligned inputs still fail.

A controlled altered-state candidate can be constructed **only in memory**:

```bash
ember kv compare NATIVE_SNAPSHOT --json \
  --perturb-layer 7 --perturb-head 0 --perturb-component keys --zero

ember kv compare NATIVE_SNAPSHOT --json \
  --perturb-layer 7 --perturb-head 0 --perturb-component both --scale 0.5
```

This path accepts one zero-based layer/head, `keys|values|both`, and exactly
one deterministic operation. Zero writes positive f16 zero. Scale performs
f16-to-f32 multiplication followed by one f16 round and rejects zero, one,
nonfinite factors, and f16 overflow. The receipt pins the source snapshot,
typed selection, exact f32 factor bits, affected counts, and a domain-separated
diagnostic identity. The source files and snapshot remain unchanged. The
altered bytes cannot be saved as `ember.kv-snapshot.v1`, never populate the
reserved mapper provenance, and never enter ordinary replay.

Optional model-backed continuation diagnostics add same-input causal
localization and an independent behavioral rollout:

```bash
ember kv compare LEFT RIGHT --json \
  --model MODEL.gguf --tokenizer TOKENIZER.json --arch llama \
  --continuation-tokens 8

ember kv compare NATIVE_SNAPSHOT --json \
  --perturb-layer 7 --perturb-head 0 --perturb-component values --zero \
  --model MODEL.gguf --tokenizer TOKENIZER.json --arch llama \
  --continuation-tokens 8
```

All four model options are required together; the horizon is `2..=64` and
includes the common initial token. By default both stored resume IDs must be
equal and present; `--token-id ID` provides an explicit common override. A
fresh strict target plan is built at the exact minimum capacity, and two live
caches are admitted under a separate aggregate limit.

The forced phase evaluates the same input token on both caches. Its later
inputs follow the reference's greedy predictions
(`reference-greedy-teacher-forced-v1`), even after candidate top-1 disagreement.
For every step it reports per-layer semantic `attention-output` cosine/MSE/
max error, full-logit cosine, both top-1 IDs, agreement, and the first forced
prediction disagreement. Aggregate attention metrics concatenate the fixed
same-input steps per layer. `attention-output` is the O-projection result after
O bias and before the attention residual add—not per-head weighted V.

The behavioral phase re-imports two clean caches and independently follows
each side's own greedy argmax. It reports both token-ID sequences, agreement
over the requested horizon, common-prefix length, and first generated-token
divergence. There are no decoded strings in JSON.

For prefix length `P`, the stored resume token belongs at absolute position
`P` but is not cached. The first measurable hook/logit row evaluates that
common token at `P`, appends its KV, and predicts continuation index 1 at
absolute position `P+1`. Prefix-boundary logits and attention cannot be
reconstructed from a KV-only artifact; the stored resume ID is provenance,
not a recomputed boundary diagnostic.

Observer hooks can choose a different internal dispatch from ordinary decode,
notably for planned Q8 where ordinary serving retains its frozen native fast
path. Forced comparisons remain paired on one same-mode observer route, while
the independent greedy phase uses ordinary decode. The JSON records this
caveat; do not mix those phases into a performance claim.

Comparison/diagnostic JSON includes snapshot-derived token IDs and may be
prompt-derived research data. Keep raw reports under ignored `runs/` unless a
sanitized evidence policy explicitly permits publication. Reports contain no
timings, paths, host, PID, or unordered maps. The harness enforces a 16-GiB
aggregate compact-payload limit for the pair, a 16-GiB aggregate two-cache
limit for model diagnostics, and a one-million layer/head report-row limit.

### Replay

```bash
ember kv replay \
  --snapshot runs/kv/morphology-prefix \
  --model Llama-3.2-1B-Instruct.Q6_K.gguf \
  --tokenizer tokenizer.json \
  --arch llama \
  --max-tokens 20
```

Replay first requires `provenance.origin = native`; altered/transformed artifacts
must use the explicit diagnostic path above and cannot enter ordinary replay.
Replay defaults to the execution mode recorded by export. Supplying
`--execution reference|planned` is permitted only when it equals the recorded
mode. It hashes and loads the supplied model/tokenizer, builds a fresh target
plan, prints every compatibility failure if strict validation fails, then
imports into newly allocated cache storage.

`--max-tokens` defaults to 20 and counts the stored or overridden resume token
as continuation token 1. Destination capacity defaults to:

```text
sequence_length + max(max_tokens - 1, 0)
```

because the first continuation token was selected from prefix logits but has
not yet been evaluated into the cache. A larger `--max-seq-len` is allowed; a
smaller one is rejected. `--max-tokens 0` emits an empty continuation and does
not require a resume token. The command emits continuation text to stdout and
a snapshot/status line to stderr.

`--token-id ID` deliberately overrides the recorded first greedy token. This
starts an alternative continuation conditioned on that token; it does not
change the snapshot and must not be described as reproduction of the original
greedy branch.

The current replay loop generates the requested count greedily and does not
stop early on an EOS token.

### Full-logit replay validation traces

The optional validation trace surface is separate from the snapshot schema:

```bash
ember kv trace-native \
  --model MODEL.gguf --tokenizer TOKENIZER.json --arch llama \
  --prompt 'The capital of France is' --execution planned \
  --max-seq-len 256 --max-tokens 4 \
  --logits-output native.npy --metrics-output native.json

ember kv export \
  --model MODEL.gguf --tokenizer TOKENIZER.json --arch llama \
  --prompt 'The capital of France is' --execution planned \
  --max-seq-len 256 --output snapshot \
  --boundary-logits-output boundary.npy --metrics-output export.json

ember kv replay \
  --snapshot snapshot --model MODEL.gguf --tokenizer TOKENIZER.json \
  --arch llama --execution planned --max-seq-len 256 --max-tokens 4 \
  --logits-output replay.npy --metrics-output replay.json
```

Trace NPY/JSON files must be outside `snapshot/`; adding them to the strict
three-file directory correctly makes snapshot verification fail. Existing
trace paths are refused unless `--overwrite` is supplied, and no-replace
publication closes the preflight race by failing if a destination appears.
Canonical parent resolution rejects symlink aliases, snapshot descendants,
and collisions with the model, tokenizer, or running executable. Trace mode
checks shape, finiteness, vocabulary width, and tokenizer membership before
choosing the deterministic full-f32 argmax. Native/replay rows stream through
a staged `NpyStreamWriter`; they are not accumulated as an unbounded
`N * vocab` RAM vector.

For prefix length `P`, requested tokens `N >= 1`, vocabulary size `V`, and
selection-logit rows `L0..L(N-1)`:

- `trace-native` writes `[N,V]` containing all rows;
- export writes `[1,V]` containing prefix-boundary `L0`; and
- replay writes `[N-1,V]` containing recomputed `L1..L(N-1)`.

The exact process-level gate is therefore:

```text
native == concatenate(export_boundary, replay_continuation)
```

Equality is f32 bit equality, not an `allclose` tolerance. JSON sidecars use
`ember.kv-replay-trace.v1` and record token IDs, row semantics, cache capacity,
model/tokenizer hashes, plan identity/fingerprint, snapshot identity, and
named phase timings. Timing fields are measurements, not semantic identity.
The first replay token is emitted from the manifest and has no replay logits
row or forward latency.

## Exact greedy-resume semantics

After export, the cache cursor equals `sequence_length` and covers the full
prompt. The stored `resume_token_id` is `argmax(prefix_final_logits)`, but that
token is not yet in the cache. Replay therefore:

1. emits the resume token as its first continuation token without evaluating
   the prompt or appending a cache row;
2. when another token is requested, evaluates that first token at absolute
   position `sequence_length`, appending its K/V row and producing logits for
   continuation token 2; and
3. repeats one token at a time from the restored cursor.

It must **not** feed the last prompt token into the restored full-prefix cache;
that would duplicate the token and change the sequence.

For the same strictly compatible model/runtime, the imported f16 prefix is
bit-identical to the exported prefix. The CI proof uses a deterministic tiny
Q8 Llama, performs independent prefill, writes and reloads a real three-file
artifact, drops the source cache, imports fresh owned storage, and compares
full logits and four greedy continuation tokens against uninterrupted planned
decode. It also asserts that the planned interpreter actually ran.

### Logits and RNG caveat

A v1 snapshot does **not** store prefix-final logits. It stores only their
greedy argmax token. Therefore it supports exact continuation of that greedy
choice, but it cannot:

- reconstruct or inspect the prefix-final logit vector;
- apply temperature, top-k, top-p, or another sampler to those logits;
- choose a different first token except through the explicit `--token-id`
  branch override; or
- prove how close the winning logit was to another token from the snapshot
  alone.

A v1 snapshot also stores no RNG or sampler state. The CLI uses deterministic
argmax only, so RNG is irrelevant to its advertised replay. Exact stochastic
resume would require at least prefix-final logits plus a versioned sampler and
RNG state; it is outside this schema.

## Verification, safety, and trust boundary

The loader treats snapshot geometry as untrusted metadata and validates it
before importing a live cache. Current limits are:

| Item | Limit |
|---|---:|
| Manifest | 1 MiB |
| Layers | 4,096 |
| KV heads | 4,096 |
| Head dimension | 65,536 |
| `max_seq` | 16,777,216 positions |
| Default total compact K+V payload | 16 GiB |
| Default destination live cache (K+V+scratch) | 16 GiB |

Shape and byte products use checked arithmetic. File lengths must equal the
shape-derived lengths exactly, so truncation and trailing bytes fail. Reads
use fallible reservations, and live-cache allocation uses a fallible
constructor. The library exposes `load_dir_with_limit` for callers that need a
smaller or explicitly larger compact-payload trust boundary and
`import_cache_with_limit` for destination allocation. The CLI uses 16 GiB for
each independent bound and rejects oversized live capacity before prefill or
replay decode.

Publication uses collision-safe uniquely named staging directories and
create-new writes, syncs each file, rejects an output symlink, and refuses
replacement unless `--overwrite` is supplied. Even with that flag, the
existing destination must verify as a strict KV snapshot immediately before
removal; the working directory and ancestors are forbidden. Failed
publication cleans its staging directory. Verified overwrite still has a
remove-then-rename visibility gap and is not claimed as crash-durable atomic
replacement; default no-overwrite publication has no clobber path.
Import validates first and returns a new owned cache; a failed import cannot
partially mutate an existing inference cache.

These checks reduce malformed-input, path, overflow, and accidental-corruption
risk. They do not authenticate the producer, encrypt prompt-derived state, or
make arbitrary artifacts safe to share. KV tensors can encode information
about their source prompt even though the manifest omits prompt text and raw
token IDs. Treat snapshots as sensitive model-derived research data.

Loading is not zero-copy: verification reads and owns both binary payloads,
and import allocates a separate live cache. Peak memory can therefore exceed
the on-disk payload size. Choose a lower library allocation limit when loading
untrusted or unexpectedly large artifacts.

## Reproducibility contract

For identical cache bits and manifest inputs, snapshot payload bytes,
payload hashes, manifest content, and `snapshot_hash` are deterministic. No
wall-clock creation timestamp participates in identity. Prefix import restores
the exact f16 bits and cursor; it does not numerically reconstruct K/V.

That is a narrow systems claim. It proves exact same-model/cache continuation
under strict execution compatibility. It does not prove model quality,
semantic correctness, cross-hardware equivalence outside the fingerprint, or
cross-model transfer. Preserve the snapshot directory, model/tokenizer hashes,
command line (including top-level K strategy), Ember commit/build, thread/CPU
environment, and emitted snapshot hash in any reproducibility record.

Ordinary inference remains unchanged: the snapshot module performs no work
unless export, load, verify, compatibility, or import is explicitly requested.
Existing decode-throughput/allocation gates remain in force, and the replay
proof applies the same allocation bound to warmed planned decode after import.

### Real-GGUF process matrix (2026-08-08)

`scripts/validate_kv_replay_matrix.py` ran independent export, verify, native,
and replay processes for:

- Llama-3.2-1B and Qwen2.5-1.5B;
- Q8_0, Q6_K, and Q4_K_M;
- fixed English and Arabic prompts; and
- two chronological observations per cell, four greedy tokens, execution mode
  `planned`, four Rayon threads, and a common 256-position cache capacity.

All 12 cells and 24 observations passed. The harness compared 13,449,216 f32
values and found zero bit mismatches; native token IDs, boundary resume IDs,
replay token IDs, hashes, execution fingerprints, capacities, snapshot
identities, and the exact three-file inventory also matched. Q8 retains its
frozen native fast-path behavior even though snapshot provenance names the
requested mode `planned`.

Compact evidence is in
`artifacts/benchmark-kv-v1/2026-08-08/{benchmark_manifest.json,benchmark_summary.json,commands.json}`.
The roughly 99 MiB of full logits, snapshots, stdout/stderr, and `/usr/bin/time`
records remain regenerable under ignored
`runs/kv-replay-matrix/2026-08-08-full/`.

The timing record is diagnostic only. Median named phases across the 12 cells
were:

| chronological observation | native prefill | snapshot load+verify+import | native 3-token decode | replay 3-token decode |
| --- | ---: | ---: | ---: | ---: |
| first observed after export | 1224.65 ms | 4.24 ms | 438.79 ms | 476.94 ms |
| subsequent observed | 1233.05 ms | 1.36 ms | 446.98 ms | 462.59 ms |

Observation 0 received best-effort `POSIX_FADV_DONTNEED`, but every CLI process
then hashes the full GGUF before inference. OS residency was not measured or
controlled, the sample count is two, and work boundaries differ between
prefill and import. These numbers are not a cold/warm, steady-throughput,
Gate-H, or end-to-end speedup claim. Trace rows also stream between timed
forwards and can perturb later cache state; import timing includes repeated
in-memory integrity verification. The raw phase samples, process walls, peak
RSS, and fault counts are retained so a controlled follow-up can be
pre-registered rather than retrofitting a claim.

## Gemma 4 is deferred

The generic live `KVCache` can allocate maximum strides, but the v1 manifest
requires uniform `n_kv_heads` and `head_dim` semantics across layers and fixes
V to `projection-output`. Gemma 4 needs more information:

- local and global attention layers can use different KV-head counts and head
  dimensions;
- some layers reuse another layer's K/V cache slab;
- its V path applies per-head RMS normalization before cache storage; and
- sliding-window/global attention semantics need per-layer provenance.

Treating that cache as uniform Llama-style V would be misleading even if the
bytes fit the same flat allocation. Gemma replay remains closed until a
versioned per-layer geometry/value-state schema and architecture-specific
compatibility checks are implemented and validated.

## Non-goals

Version 1 is deliberately not:

- cross-model KV transplantation, learned mapping, layer alignment, or an
  empirical claim that different models share a KV space;
- a transformed-snapshot authoring workflow (the transform provenance and
  RoPE seam are reserved research infrastructure only);
- a stochastic or general generation checkpoint containing logits, RNG,
  sampler state, prompt text, raw tokens, or generated-history text;
- a portable cache for Gemma 4, GPT-2, arbitrary backends, paged attention,
  quantized KV, or unknown RoPE variants;
- a way to bypass model, tokenizer, execution-mode, dispatch, or provenance
  checks;
- an activation-capture/intervention artifact or a v0.5 experiment bundle;
- a modification of `v04-plan/1` or `ember.bundle.v1` compatibility;
- an authenticated, encrypted, privacy-preserving, mmap, or zero-copy format;
  or
- a promise that unknown future snapshot schemas will be accepted. Unsupported
  schema, kind, serialization, metadata, and unknown fields fail closed.
