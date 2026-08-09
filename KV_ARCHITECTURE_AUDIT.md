# Ember KV Architecture Audit

**Audit date:** 2026-08-08
**Scope:** the existing live KV-cache implementation and the currently visible, uncommitted KV snapshot/replay implementation in `src/kv_cache.rs`, `src/kv_snapshot.rs`, `src/kv_transfer/`, `src/cli_kv.rs`, `src/llama.rs`, `src/plan.rs`, and their callers/tests.
**Nature of this document:** an implementation audit, not a compatibility promise or a model-quality claim.

## 1. Executive summary

Ember's live KV cache remains a caller-owned, preallocated CPU object. A loaded model can construct a cache, but it does not retain that cache. The generation caller owns one `KVCache` for one sequence and passes it mutably through the model, blocks, attention implementation, and backend. K and V occupy separate flat `Vec<f16>` allocations in logical `[layer][head][position][dimension]` order. A single cursor describes the common initialized prefix for all layers.

The new implementation adds a separate, independently versioned `ember.kv-snapshot.v1` artifact. It copies the initialized prefix out of a live cache into compact little-endian f16 payloads, records strict model/execution/RoPE metadata, verifies hashes and shape products before import, and copies the payload into a newly owned live cache. It does **not** put file I/O, hashing, or transformation work in ordinary inference. It also does not change the concrete `ForwardModel` cache type or introduce shared/paged/quantized KV storage.

The deterministic tiny Llama-family unit test performs prefill, writes a snapshot directory, drops the source cache, reloads into fresh owned storage, and obtains bit-identical cached f16 payloads, continuation logits, and greedy tokens in planned decode. A subsequent independent-process real-GGUF matrix covered Llama-3.2-1B and Qwen2.5-1.5B at Q8_0/Q6_K/Q4_K_M for fixed English/Arabic prompts: all 12 cells and 24 observations passed with zero mismatches across 13,449,216 compared f32 logits. This remains a same-host/same-binary planned-mode result, not a cross-version guarantee, a Gemma replay result, or evidence that cross-model KV transfer works.

The official CLI is deliberately narrower than the general live cache:

- `ember kv export|inspect|verify|replay|trace-native` supports the Llama/Qwen loader path;
- export/replay expose `reference` and `planned`, not `planned-fused`;
- replay is strict same-model greedy continuation;
- GPT-2 and Gemma are not represented by the current snapshot compatibility target;
- the transfer module only supplies a tested RoPE coordinate-space seam. It contains no learned mapper and creates no transformed snapshot.

A concrete v0.4 plan-cache defect found during the audit has been fixed in the visible implementation: plan cache keys now include KV capacity and the supplied model/tokenizer hashes. This prevents a small-capacity plan or empty provenance from being silently reused for a later same-mode request. The plan's full hash still changes with capacity, while the new execution fingerprint intentionally normalizes capacity-only plan fields so a compact prefix can be imported into a larger compatible destination cache.

## 2. Before and after: architectural boundary

### 2.1 Before the snapshot work

The committed baseline had one live type, `KVCache`, with private K/V/cursor fields and public construction/append/read operations. It had no artifact schema, no f16-native import, no content hash, no model identity, no serialization, no safe cursor restoration, and no compatibility report. The only practical way to inject a prefix through public methods was to append f32 K/V for every layer and position and call `advance_cursor()` once per position.

The v0.2 experiment contract explicitly listed KV-cache mutation as unsupported. Existing experiment hooks observe or intervene on hidden/residual/logit tensors, not on cached K/V. The v0.4 plan described the cache, but the plan interpreter did not use `plan.kv` as a runtime type check.

### 2.2 New design now visible

The new design preserves the live cache and adds an artifact layer around it:

1. `KVCache` remains the hot-path allocation.
2. `KvSnapshot::export_native` copies only the initialized prefix.
3. A snapshot owns compact f16 keys and values plus a manifest.
4. `save_dir` writes three files through a staging directory.
5. `load_dir` checks the directory, manifest, dimensions, file lengths, payload hashes, and manifest identity.
6. `compatibility_report` compares the snapshot with a target derived from an `ExecutionPlan`.
7. `import_cache` allocates a fresh live cache at the **target** capacity, copies exact f16 bits, and restores the cursor.
8. CLI replay resumes from a recorded or overridden greedy token without recomputing the saved prefix.

This is additive at the main public inference boundary. `ForwardModel::create_cache` still returns concrete `KVCache`, and cached forward methods still receive `&mut KVCache`. The new `KVCache::try_new` is fallible for metadata-driven allocation, while `KVCache::new` retains its existing signature and remains the ordinary model constructor.

## 3. Live KV ownership and allocation

### 3.1 Owner

The generation or command caller owns live KV. Typical flow is:

```text
CLI/generation caller
  -> model.create_cache(...)
  -> GenerationExecution / ForwardModel / ExperimentalForwardModel
  -> model cached forward
  -> block cached forward
  -> attention cached forward
  -> backend attention over borrowed f16 slices
```

The model owns weights, configuration, RoPE tables, and—on Llama CPU paths—persistent plan/decode scratch state. It does not own the live prefix cache. There is no cache `Arc`, cache pool, batch dimension, cache ID, or automatic association between a `KVCache` instance and the model that created it.

A fresh cache is allocated for each ordinary generation run, demo prompt, probe continuation, logit/layer dump, native-reference sample, lifecycle run, and decode benchmark run. Interactive GPT-2 mode calls the normal generation function for each turn and therefore does not retain conversational KV between turns. A resident GUI model also receives fresh per-run cache state through the common generation pipeline.

Gemma activation and pooled-hidden-state paths allocate their own prompt-length cache internally. The `model_backend.rs` cache implementation is test scaffolding rather than a separate production owner.

### 3.2 Allocation

For live geometry

```text
L = n_layers
H = n_kv_heads
P = max_seq_len
D = head_dim
N = L * H * P * D
```

`KVCache` owns:

- K: `Vec<f16>` of `N` elements;
- V: `Vec<f16>` of `N` elements;
- generic-attention scratch: `Vec<f32>` of `P` elements;
- scalar geometry and cursor fields.

The shape product is checked. The new `try_new` also uses `try_reserve_exact` and returns a string error for invalid dimensions, shape overflow, or allocator reservation failure. It then zero-initializes the allocations. `new` calls `try_new(...).expect(...)`; ordinary construction therefore remains panic-based on invalid/unallocatable geometry.

K and V reserve `4*N` bytes together because each has `N` two-byte elements. Cache scratch reserves another `4*P` bytes, excluding vector headers and allocator overhead. `storage_bytes()` intentionally reports only K+V capacities, not score scratch.

There is no custom destructor. Dropping the caller-owned cache drops its vectors. `reset()` does not free or zero storage; it only resets the cursor.

### 3.3 Planned-decode scratch is separate

The Llama plan-driven path owns a model-side `PlannedDecodeState` containing a `DecodeArena` and resolved operations. Its attention score region is f32 `[n_heads * max_seq]`. Planned attention reads live cache K/V through `cache.get(layer)` and does not use `KVCache::qk_scratch`.

Generic serial attention and the Q8 fast path use cache scratch. Generic parallel attention can instead use backend thread-local scratch. Consequently, a KV snapshot contains only K and V; it does not serialize generic scratch, TLS scratch, the decode arena, or resolved plan state.

## 4. Live layout, strides, and precision

K and V are separate arrays with logical order:

```text
[layer][kv_head][position][dimension]
```

The element index is:

```text
(((layer * H + head) * P + position) * D + dimension)
```

The element strides are:

```text
layer    H * P * D
head         P * D
position         D
dimension        1
```

These are scalar-element strides, not byte strides. `KVCache::element_strides()` now exposes them. `plan::KvLayout` now explicitly documents the same unit, and the plan test helper has been corrected to stop multiplying strides by two. The production plan builder and existing JSON fixture were already element-based.

`append`, `append_with_head_dim`, and `append_with_layout` accept one position of f32 K and V in contiguous `[active_head][active_dimension]` order. They convert into f16 while copying into the physical slab. `get(layer)` and `get_with_scratch(layer)` return the entire physical layer allocation, including unused future positions and any head/dimension padding.

No layout method returns only the initialized prefix. The new crate-private snapshot export performs the compaction explicitly.

## 5. Cursor lifecycle and failure semantics

A live cache has one global cursor shared by every layer.

1. It starts at zero.
2. The model-level cached entry point normally asserts `start_pos == cursor`.
3. Every layer writes its K/V for the same `cursor..cursor+seq_len` positions.
4. Attention includes the newly written positions, using total length `cursor+seq_len`.
5. Only after all layers complete does model-level code advance the cursor once per input token.

This separation is essential: advancing after one layer would make later layers write different positions. `append*` therefore does not advance the cursor itself and permits an explicit position argument.

Single-token fast and planned decode follow the same logical rule and advance once after all blocks. Planned KV store uses `start_pos`; its entry point first validates that this equals the live cursor. Reference/fast code often reads `cache.cursor()` directly for the destination after validation.

`advance_cursor()` asserts before exceeding capacity. `reset()` sets the cursor to zero without clearing old bits. Normal reuse is safe only because attention bounds its reads by total sequence length and the next run rewrites each active position before using it.

The live mutations are not transactional. An error in a later layer or hook can leave earlier layer slabs written while the cursor remains unchanged. An error after cursor advancement, such as a final output hook failure, can return an error with the cursor already advanced. Current callers normally discard a failed run's cache; snapshot export also requires its requested sequence length to equal the cursor.

Gemma's `forward_last_logits_with_layer_dump` is an exception to the otherwise consistent top-level checks: it does not itself validate `start_pos` or assert nonempty tokens before entering its layer loop.

## 6. What the cache contains: architecture matrix

The same physical f16 type does not mean every architecture stores the same representation.

| Family | Physical geometry | Stored K | Stored V | Position mechanism | Current snapshot support |
|---|---|---|---|---|---|
| GPT-2 | uniform heads/dim | projection output | projection output | learned absolute position embedding before projections; no RoPE | no |
| Llama | uniform GQA heads/dim | post-RoPE; optional post-RoPE K norm if configured | projection output | adjacent-pair RoPE | yes, official path |
| Qwen2/Qwen3 through `Llama` | uniform GQA heads/dim | post-headwise-norm and post-RoPE when norms exist | projection output | split-half RoPE | yes, official path |
| Gemma 4 | physical maxima with per-layer active geometry; shared-source slabs | learned headwise norm, then split-half RoPE, in source layers | plain per-head RMS-normalized V, in source layers | local/global RoPE tables/theta and local/global head geometry | no |

### 6.1 Llama and Qwen

Llama-family reference attention projects f32 Q/K/V, applies the configured Q/K normalization and RoPE to Q and K, converts K and V to f16, stores them, and then attends over the full prefix.

- Llama uses adjacent-pair coordinates. Its usual models have no Q/K norm tensors. If norms exist in this configuration, `QkNormOrder::AfterRope` applies them after rotation.
- Qwen uses split-half coordinates and `QkNormOrder::BeforeRope`. Qwen2.5 commonly has no q/k norm tensors; Qwen3 commonly has them. The Llama-family headwise q/k norm helper uses epsilon `1e-6`.
- V is not rotated. On this path its semantic state is the projection output.
- The stored K state is therefore position-dependent and described as `post-rope` in the new schema.

The generic path orders Q transform, K transform, store, attention. The planned unfused graph expresses the same order. The current planned F4 form leaves K transform as a separate pre-store op and moves Q transform into attention. K cannot be rotated after store because cached K must already be in the coordinate system consumed by attention.

### 6.2 RoPE tables

Llama/Qwen tables are precomputed at model load as f32 `[model_max_seq, head_dim/2]` and shared by `Arc` across attention layers. Some field comments in `llama.rs` still describe `[max_seq, head_dim]`; the actual table returned by `compute_rope_freqs` has half-width.

The cached position is absolute zero-based `start_pos+s`. A manually constructed cache can have capacity beyond the model's loaded RoPE table, because public `create_cache` does not itself cap its argument. Normal CLI paths cap at load/context validation; a caller bypassing those checks can reach an out-of-range RoPE table access even if the cache allocation itself is valid.

### 6.3 Gemma heterogeneity and shared KV

Gemma cannot be treated as merely another uniform Llama cache:

- local and global layers may have different KV-head counts;
- local and global layers may have different head dimensions and theta values;
- `Gemma4::create_cache` allocates the maximum KV-head count and maximum head dimension across the model;
- each producing layer uses `append_with_layout` to write only its active rectangle into that larger slab;
- attention carries both the active head dimension and the physical cache head dimension so it can index padding correctly;
- a layer without its own K/V tensors reads a previous source layer of the same local/global type; its own physical slab remains unused;
- Gemma applies a learned RMS norm to K before split-half RoPE and applies a separate plain per-head RMS norm to V before caching it.

A physical dump of Gemma's max-sized slabs would not be enough for strict portable compatibility. It would also need per-layer active heads/dim, local/global RoPE/frequency metadata, value-state metadata, and the shared-source mapping. The new target has one global `n_kv_heads`, `head_dim`, theta, and value state `projection-output`; it therefore intentionally does not derive Gemma targets or expose Gemma through `ember kv`.

## 7. Cached execution paths

### 7.1 GPT-2

`src/model.rs` implements cached GPT-2 attention. It stores projected K/V for each input row, attends against the full cache, and advances at model level. Position embeddings use the caller's `start_pos` before the attention projections.

### 7.2 Llama/Qwen reference path

`LlamaAttention::forward_with_cache` performs projection, Q/K transform, cache append, cached attention, and O projection. `LlamaBlock` wraps attention/MLP residual operations and experiment hooks. The model loops layers and advances the cursor after all layers.

### 7.3 Llama Q8 fast path

Eligible adjacent-pair Q8 models use a thread-local reusable workspace. The fast path applies decode RoPE/qk norm, appends one f16 K/V position per layer, runs allocation-free cached attention into workspace storage, and advances once after all layers. In the ordinary `ForwardModel` path it has precedence over planned execution, preserving the special Q8 route.

The experiment-aware dispatch is ordered differently: for one token with planned mode and tracing off, it enters the planned interpreter before trying the Q8 hooked fast path. This remains an execution-path distinction worth preserving in provenance and tests.

### 7.4 Planned path

Plan-driven execution supports one token and tracing off. Multi-token prefill stays generic; a one-token prompt can satisfy the planned entry condition. Planned K/V operations borrow the same live cache and use the model-owned arena only for activations/scores.

`forward_last_logits_planned` currently requests `ExecutionMode::Planned` when resolving its plan. Thus the broader runtime's `PlannedFused` selection does not prove that a fused plan was executed. The new KV CLI avoids this ambiguity by accepting only `reference` and `planned`. Existing tests named for planned-fused compare results/hooks but do not directly assert that fused ops backed the session.

### 7.5 Gemma

Gemma uses the common live `KVCache` allocation but custom attention indexing for heterogeneous physical/active dimensions, optional sliding windows, and shared source layers. Its experiment and layer-dump paths also mutate live KV.

### 7.6 Callers

Caches are allocated or passed in:

- `src/cli_generation.rs`: standard/experiment generation, demos, logits and Gemma layer dumps;
- `src/cli_commands.rs`: native references, lifecycle work, and decode benchmarks;
- `src/cli_probe.rs`: probe continuations;
- `src/experiments/mod.rs`: experiment-aware cached forward trait;
- `src/llama.rs`, `src/model.rs`, `src/gemma4.rs`: model/block/attention implementations;
- `src/backend.rs`: f16 cached-attention consumers;
- tests, including direct cache geometry/attention tests.

## 8. Snapshot artifact and ownership

### 8.1 Independent schema

The snapshot declares:

```text
schema         ember.kv-snapshot.v1
kind           kv-prefix
serialization  manifest-json+f16le-v1
```

It is independent of `v04-plan/1` and `ember.bundle.v1`. The artifact directory contains exactly:

```text
manifest.json
keys.f16le
values.f16le
```

`KvSnapshot` owns decoded `Vec<f16>` keys and values. A loaded snapshot never aliases the files or a live cache. Import creates another owned allocation and copies into it.

### 8.2 Compact payload

Live capacity is `[L,H,P,D]`; snapshot payload is compact `[L,H,S,D]`, where `S=cursor`. Unused live positions are omitted. The compact head stride is `S*D`, not `P*D`.

Each f16 is serialized by raw IEEE-754 binary16 bits in little-endian order. Import copies decoded f16 values directly, avoiding f16-to-f32-to-f16 conversion. This preserves all stored 16-bit patterns through a valid artifact round trip.

`export_native` requires live layer/head/dim/capacity geometry to match the supplied source target and requires the prefix token count, if supplied, to equal the cursor. CLI export always supplies token IDs and tokenizer hash.

### 8.3 Manifest contents

The manifest records:

- model and optional tokenizer SHA-256;
- architecture;
- sequence length and original maximum capacity;
- layer/head/dimension geometry;
- f16 precision and compact layout;
- RoPE layout, dimension count, exact f32 theta bits through serialization/comparison, frequency layout, position origin, stored-key state, q/k norm presence/order/epsilon;
- V representation state;
- Ember version, execution mode, full plan hash, capacity-independent execution fingerprint, native/transformed origin, and source model SHA;
- optional prefix-token count/hash and greedy resume token;
- key/value filenames, element counts, byte lengths, and SHA-256;
- snapshot hash.

`resume_token_id` is convenience/provenance metadata, not part of the KV tensor state. A cache alone does not contain the logits needed to choose the first continuation token; replay therefore needs either this recorded prediction or `--token-id`.

The manifest uses `deny_unknown_fields`. That makes v1 parsing fail closed on undeclared fields rather than silently accepting a structurally different artifact.

## 9. Hash and provenance interactions

Four identities have different roles.

### 9.1 Payload hashes

K and V descriptors hash the exact little-endian payload bytes independently. File length and element count are also checked.

### 9.2 Snapshot hash

`snapshot_hash` hashes canonical serde serialization of the complete manifest with `snapshot_hash` temporarily empty. Because the manifest contains the payload hashes, snapshot identity transitively commits to K and V bytes and to all recorded metadata, including original capacity and resume token.

This is deterministic for the current Rust types/serde serialization and equal inputs. It should not be described as a timeless canonical-JSON standard independent of implementation changes unless that representation is separately frozen and tested across versions.

### 9.3 Full execution-plan hash

The full v0.4 plan hash includes scratch/KV capacity and other complete plan metadata. Loading the same model with a different context cap legitimately creates a different plan hash. The snapshot retains this hash as provenance but does not require it to equal the destination plan hash.

### 9.4 Execution fingerprint

The compatibility target computes an execution fingerprint from the plan after:

- removing `plan_hash`;
- removing the scratch plan;
- zeroing KV layer/head/position strides and `max_seq`;
- zeroing GGUF context length;
- clearing plan build time.

The remaining operation graph, tensors, dispatch, build/runtime provenance, model semantics, hook/mode information, and other serialized plan fields remain covered. The intent is to permit destination capacity to differ while rejecting changes that can affect continuation numerics.

This is deliberately strict: execution mode must also match, even when two modes are expected to be numerically close. The fingerprint can reject a same-weight cache loaded under a different dispatch/build/thread environment. That is conservative and consistent with an exact-replay claim; it is not a statement that every rejected pairing would actually diverge.

### 9.5 Model and tokenizer hashes

Model SHA is mandatory and must be lowercase 64-digit hexadecimal. Official native snapshots require the producer model to equal the compatible model. A tokenized prefix records a token-ID hash and requires known, equal source/target tokenizer SHA values.

The token-ID payload itself is not stored. Verification can prove that the recorded token hash was not changed without changing snapshot identity; it cannot independently reconstruct or prove the original prompt/token sequence. CLI replay does not accept a prompt to cross-check against that hash.

### 9.6 `exact_same_model`

`KvCompatibilityReport::exact_same_model` means native origin plus equality of source/compatible/target model hashes. It does not replace `compatible`. A report can identify the same model yet remain incompatible because execution, tokenizer, geometry, or RoPE metadata differs.

## 10. Strict compatibility and capacity handling

Compatibility currently compares:

- target validity;
- model SHA;
- architecture;
- precision and layout;
- layer count, KV heads, and head dimension;
- `snapshot.sequence_length <= target.max_seq`;
- RoPE layout, dimension count, theta bit pattern, frequency layout, position origin, stored-key state;
- q/k norm presence/order/epsilon;
- V state;
- execution mode and execution fingerprint;
- tokenizer identity under the prefix-provenance rules.

Original source `max_seq` is **not** required to equal target `max_seq`. This is intentional because payloads are compact. Import allocates using target capacity and restores only `S` initialized positions, then sets cursor to `S`.

The full source plan hash is also not a compatibility key because it contains source capacity. The normalized fingerprint is the execution compatibility key.

`KvCompatibilityTarget::from_execution_plan` currently understands only plan precision `f16`, layout `layer-head-pos-dim`, adjacent/split RoPE, before/after qk norm, value state `projection-output`, and modes known to `ExecutionMode`. It derives q/k norm epsilon `1e-6` when Llama-family norm tensors are present.

## 11. Plan metadata, plan-cache fix, and remaining plan caveats

### 11.1 Metadata relevant to KV

`ExecutionPlan.kv` records precision, layout, element strides, head dimension, KV-head count, and maximum sequence length. `gguf` and `rope` summaries add layer/query/KV counts, context, RoPE dimension/theta/layout, and q/k norm state. Provenance supplies model/tokenizer hashes and execution information.

For Llama/Qwen this is sufficient to construct the current uniform compatibility target. It is not a general description of GPT-2 or heterogeneous/shared Gemma caches.

### 11.2 Fixed cache-key defect

Previously the per-model plan cache key contained only:

```text
(execution mode, hook mode, active stages)
```

Yet plan contents and arena sizing also depend on runtime KV capacity and supplied model/tokenizer hashes. Reusing the first plan could therefore:

- return a score arena sized for an earlier, smaller cache;
- report a stale `kv.max_seq` and plan hash;
- retain empty or incorrect model/tokenizer provenance from the first call.

The visible implementation now keys on:

```text
(execution mode,
 hook mode,
 active stages,
 max_seq_len,
 model_sha256-or-empty,
 tokenizer_sha256-or-empty)
```

A focused test confirms different capacity/provenance requests return distinct plan `Arc`s and preserve their own metadata. This fixes the identified capacity/hash reuse path.

CPU features and Rayon thread facts remain plan fields but are not explicit cache-key elements. They are normally stable for one process/global pool; if Ember later allows those to vary for the same model instance, they must also participate in cache invalidation or be recomputed.

### 11.3 Descriptive rather than enforced metadata

The planned interpreter still uses live cache geometry and actual model fields. It does not validate the runtime cache against `plan.kv`, and it ignores serialized per-op RoPE strings/norm refs when applying the model's actual RoPE/qk norm implementation. `ExecutionPlan::validate` checks plan references and structure, not a live cache.

This is why snapshot import performs its own strict target comparison and constructs cache geometry from that target. A plan hash by itself is not a content hash and should not be used as the only cache-import gate.

### 11.4 Architecture naming

Plan construction infers `architecture` from RoPE layout: adjacent becomes `llama`, split-half becomes `qwen2`. Consequently, a Qwen3 model loaded through the Qwen CLI family can currently be recorded as plan/snapshot architecture `qwen2`. Same-code-path export/replay remains internally consistent, but the label is not a precise original GGUF architecture identity. The mandatory model hash is the stronger model identity.

## 12. Disk loading, trust boundaries, and publication

`load_dir` enforces:

- a real directory containing exactly the three expected regular files;
- manifest size at most 1 MiB;
- declared dimension ceilings and checked shape products;
- exact expected file lengths, rejecting truncation and trailing bytes;
- a default combined payload limit of 16 GiB, with an explicit-limit alternative;
- key/value hashes and snapshot hash before import.

Dimension ceilings are generous and separate from the payload byte limit. Import allocates a **target-capacity** cache after compatibility succeeds; an intentionally enormous valid target can still request much more live memory than the compact snapshot. `try_new` makes arithmetic/reservation failures recoverable, but no Rust API can promise recovery from every OS overcommit or process-wide OOM condition.

`save_dir` writes to a collision-safe sibling staging directory, syncs individual files, and renames the completed directory into place. For a new destination, consumers do not observe partial files at the final path and a concurrent destination is never clobbered. `overwrite=true` is restricted to a destination that re-verifies as a strict snapshot and can never target the working directory/ancestors; the old verified snapshot is still removed before rename, so replacement has a gap and is not an atomic swap or complete crash-durability protocol. Parent-directory fsync is also not performed.

## 13. CLI behavior

### 13.1 Export

`ember kv export`:

- hashes the model and tokenizer files;
- validates requested Llama/Qwen-family architecture;
- tokenizes a nonempty prompt;
- loads the model with a capacity cap;
- runs prefill and captures the greedy next token from final-prefix logits;
- builds a plan with hashes and actual cache capacity;
- derives a compatibility target;
- exports and verifies a native snapshot;
- writes the snapshot directory.

Allowed execution values are `reference` and `planned`. A multi-token planned export still performs generic prefill under the existing dispatch rule; the recorded mode governs the intended continuation route and fingerprint.

### 13.2 Inspect and verify

Inspect loads and therefore verifies the artifact before printing text or JSON. Verify prints the recomputed deterministic snapshot identity after all load checks succeed.

### 13.3 Replay

Replay:

- loads/verifies the snapshot;
- requires requested/default execution mode to equal the recorded mode;
- reloads and hashes the target model/tokenizer;
- computes minimum capacity `prefix_length + max_tokens - 1`;
- builds a capacity-specific target plan;
- reports all strict incompatibilities before allocation/import;
- imports into fresh owned live cache storage;
- emits the stored/overridden first greedy token;
- evaluates each emitted token except the final one to generate the next token.

The `-1` is correct for this representation: the prefix cache already exists and the first continuation token was selected from prefix logits, so it is added to KV only when another prediction is needed.

Replay is greedy and does not currently stop on EOS. It prints decoded generated IDs. It does not recreate temperature/top-k/top-p sampler state, random-number state, hook state, trace state, or an interactive conversation object.

## 14. RoPE transfer seam

`src/kv_transfer/rope.rs` is allocation-bearing experimental utility code, not an inference route.

It can:

- rotate a selected f32 key row forward or inverse in adjacent-pair or split-half coordinates;
- convert a verified compact snapshot's f16 stored keys to owned f32 content-space keys by applying inverse RoPE per layer/head/position;
- apply forward RoPE to compatible content-space keys.

Here, “content” means post-K-normalization/pre-RoPE when K norm occurs before RoPE. The conversion rejects:

- non-`uniform-theta` frequency metadata;
- partial RoPE dimensions;
- non-post-RoPE stored keys;
- a K norm applied after RoPE, because that normalization is not invertible from the available metadata.

Forward/inverse work is f32 and is approximate. The exact native replay path bypasses it and preserves original f16 stored bytes. The module does not transform V, align layers/heads, learn a mapper, create a transformed snapshot, or establish cross-model compatibility. `KvTransformProvenance` and `Transformed` origin are reserved schema seams only.

## 15. API surfaces and future break risks

The current snapshot work avoids most existing source breaks because it is additive. The following concrete surfaces would be affected by a deeper cache redesign:

- `ForwardModel::create_cache` returns concrete `KVCache`;
- `forward_with_cache`, `forward_last_logits_with_cache`, and `greedy_next_token_with_cache` accept `&mut KVCache`;
- `ExperimentalForwardModel` accepts the same concrete cache;
- Gpt2/Llama/Gemma model, block, and attention methods accept it;
- private CLI `GenerationExecution` accepts it;
- backend cached-attention APIs hard-code `&[f16]` K/V and a uniform `CachedAttentionSpec`;
- Llama fast and planned paths use concrete cache accessors;
- Gemma custom attention assumes max-padded physical slabs;
- tests directly construct and inspect `KVCache`;
- `ExecutionPlan.kv`, plan hashes, fixtures, and snapshot compatibility metadata encode current precision/layout.

Likely breaking changes include replacing `Vec<f16>` with borrowed/mmap/shared/paged storage, adding a batch dimension, quantizing KV, making dtype generic, changing physical order, associating a cache with a backend device, or changing cursor semantics. Such changes need a new snapshot serialization identifier or schema rather than silently reinterpreting v1 payloads.

Adding KV intervention hooks is a separate research API change. Existing activation hooks do not represent paired f16 K/V, post-RoPE timing, per-layer variable dimensions, or Gemma source-layer sharing. A correct hook would need an explicit semantic site around transformed K/V storage and equivalent implementation in reference, fast, planned, and Gemma routes. It would also amend the prior experiment contract that excluded KV mutation.

## 16. Validation evidence present now

At audit time, these focused commands passed in the current working tree:

```text
cargo test --lib kv_ --quiet
# 26 passed; 0 failed

cargo test --lib execution_plan_cache_keys_capacity_and_provenance --quiet
# 1 passed; 0 failed
```

Additional validation completed after the focused audit commands:

- the complete debug and release Rust suites passed;
- the real Q4_K_M `v04_planned_matches_reference_real_model` parity gate passed on all six frozen English/Arabic prompts with unchanged thresholds; and
- real Llama-3.2-1B Q8 reference and Q4 planned CLI snapshot replays each produced the same three-token greedy stdout as ordinary native generation for the tested prompt.

The later process-level full-logit matrix also passed all 12 Llama/Qwen × Q8/Q6/Q4 × English/Arabic cells twice. Its 72 subprocesses saved boundary/native/replay NPY traces and proved exact reconstruction of 13,449,216 f32 values with no mismatches. Compact evidence is tracked in `artifacts/benchmark-kv-v1/2026-08-08/`; the roughly 99 MiB raw run is ignored under `runs/`.

The covered cases include:

- basic live-cache round trip, storage accounting, active-layout padding, and cursor mismatch;
- compact f16 export/import with cursor restoration;
- deterministic manifest/payload hashes;
- directory round trip and integrity verification;
- rejection of truncated, corrupted, extra, malformed, overflowing, and allocation-limit inputs;
- strict compatibility mismatches;
- empty and full-capacity prefix boundaries;
- manifest tampering;
- adjacent/split RoPE hand vectors, head isolation, approximate f32 round trip, and fail-closed post-RoPE K norm handling;
- capacity/provenance plan-cache keying;
- a tiny same-model disk replay comparing exact cached bits, logits, and greedy tokens.

The existing repository also has cached-attention parity/shape tests and planned/reference/fast-path tests. These support implementation confidence but do not broaden the claim beyond their fixtures and tolerances.

## 17. Important unvalidated or unsupported cases

The current implementation does **not** establish:

- cross-process replay under differing CPU/thread/build environments, or cross-host exact replay;
- a real-model full-logit matrix in `reference` or `planned-fused` mode (the completed broad matrix requested `planned`);
- compatibility across Ember versions or plan schema changes;
- `planned-fused` KV replay;
- GPT-2 snapshots;
- Gemma local/global/shared-KV snapshots;
- snapshot integration with `ember.bundle.v1` experiments;
- stochastic sampler continuation;
- EOS-aware replay behavior;
- cross-model KV transfer;
- a learned representation mapper;
- exact round trip through content-space inverse/forward RoPE;
- atomic crash-safe replacement of an existing snapshot directory;
- safety against every possible OOM/OS overcommit event.

Before turning the recorded planned-mode result into a broader performance or portability claim, the next high-value gate is a pre-registered resident-model benchmark with controlled cache state, enough repetitions, baseline/candidate binaries, and the existing Gate-H rule. A separate correctness extension can cover real-model `reference`, larger/minimum destination capacities, and a second supported host/build without weakening bit-exact acceptance. The completed matrix already preserves binary/build, threads, commands, model/tokenizer hashes, resource samples, and every full-logit comparison.

## 18. File map

- `src/kv_cache.rs` — live f16 allocation, strides, cursor, compact crate-private import/export.
- `src/kv_snapshot.rs` — v1 artifact types, hashing, validation, compatibility, disk I/O, fresh-cache import.
- `src/kv_transfer/mod.rs` — transfer-space terminology.
- `src/kv_transfer/rope.rs` — tested experimental stored/content RoPE conversion seam.
- `src/cli_kv.rs` — Llama/Qwen export/inspect/verify/replay and full-logit trace commands.
- `src/model.rs` — `ForwardModel`, GPT-2 cached path, backend-neutral ownership boundary.
- `src/llama.rs` — Llama/Qwen reference, fast, planned paths; plan cache and same-model replay test.
- `src/planned_decode.rs` — plan interpreter and model-owned decode arena; borrows live KV.
- `src/gemma4.rs` — heterogeneous geometry, V normalization, local/global/shared KV.
- `src/backend.rs` — f16 cached-attention consumers and indexing.
- `src/plan.rs` — v0.4 plan schema, KV element-stride metadata, plan hashing.
- `src/cli_generation.rs`, `src/cli_commands.rs`, `src/cli_probe.rs` — cache-owning callers.
- `src/experiments/mod.rs` — experiment cached-forward boundary; no KV semantic hook.

## 19. Bottom line

The new snapshot layer respects Ember's existing ownership model: ordinary inference still uses a simple caller-owned f16 cache, while snapshot work is explicit, allocation-bearing, and independently versioned. Same-model replay is designed to fail closed on model, execution, geometry, RoPE, q/k norm, V-state, and tokenizer mismatches, and it restores exact stored f16 bits into fresh storage.

The design is credible for its declared, tested same-binary Llama/Qwen planned-mode scope and now has both focused unit evidence and a broad real-GGUF process-level exact-logit matrix. Its boundaries remain clear: plan metadata is not a universal live-cache type system; Gemma needs a richer per-layer/shared schema; content-space RoPE conversion is approximate and not transfer; portability across builds/hosts and controlled performance remain unestablished.
