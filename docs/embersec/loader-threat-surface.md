# EmberSEC: GGUF loader threat surface (audit note)

> **Phase I provenance:** frozen audit documentation from branch snapshot
> `e1fe6269`; the measured hardened Ember target is `3ceb7039`. Current main
> retains the applicable hardening, but implementation names and dataflow may
> have evolved. Read this as the published Phase I evidence record.

Status: published Phase I audit record; original implementation history is on `embersec/secure-gguf-loader`.
Scope: the GGUF model-loading path in the frozen Phase I implementation
(`loader.rs` at commit `3ceb7039`). This is a factual audit of the hardened
current dataflow and its trust transitions; historical implicit
assumptions are retained where they explain the hardening. It does not
claim vulnerabilities; it records where the boundary between
attacker-controlled bytes and executable tensor state is explicit, where
it remains implicit, and what the loader assumes about GGUF input.

## 1. Current model-loading dataflow

```
model file (attacker-controlled bytes)
  -> File::open + memmap2::Mmap (loader.rs load_gguf*)
  -> header parse: magic, version, tensor_count, metadata_kv_count
  -> metadata parse: key/value table (GgufValue, incl. nested arrays)
  -> tensor-info parse: TensorInfo { name, dims, dtype, offset }
  -> alignment + data_start computation
  -> per-tensor byte-length / file-range computation (the only
     semantic gate today) + overlap rejection
  -> per-tensor view construction:
       f32/f16/bf16: read_exact into Vec<u8>, convert into CpuTensor
       q8_0:        read or mmap range into QuantizedWeight
       q2_k..q6_k:  read or mmap range into KQuantWeight, or dequant
                    to f32 (eager path)
  -> GgufLoader { metadata, tensors: LoadedTensor, tensor_meta, ... }
  -> model construction (llama.rs / gemma4.rs / model.rs gpt2):
       architecture string check, config from metadata, take_tensor/
       take_f32, per-block shape validation
  -> kernels consume LoadedTensor (CpuTensor / QuantizedWeight /
     KQuantWeight)
```

Entry points: `load_gguf` (mmap-backed, production), `load_gguf_from_reader`
(owned-buffer, tests), both funnel into `load_gguf_from_reader_impl`.

## 2. Trust transitions

An explicit trust transition exists when a value derived from the file
passes a check before it is allowed to influence memory layout or
execution. Current explicit transitions:

1. Header counts vs. file length: `tensor_count > file_len / 32` and
   `metadata_kv_count > file_len / 13` are rejected before any table
   reservation (heuristic magic literals — see section 6).
2. String lengths vs. remaining file bytes: every GGUF string
   (metadata keys/values, tensor names) is length-checked against the
   remaining bytes before `try_reserve_exact` + read.
3. Metadata array size: minimum serialized byte size of the declared
   element count is checked against remaining bytes before reserve;
   nesting depth is capped at 16; bool values are restricted to 0/1;
   duplicate metadata keys are rejected.
4. Tensor descriptors: names non-empty + unique; rank in 1..=4;
   dimensions non-zero; each dimension converted with
   `usize::try_from(u64)`.
5. Alignment: `general.alignment` must be a non-zero power of two;
   `data_start` is computed with `checked_add`.
6. Tensor byte extents: element-count product via `checked_mul`;
   per-dtype encoded byte length via checked arithmetic with
   block-alignment requirements (Q8_0 % 32, K-family % 256);
   `start = data_start + offset` and `end = start + byte_len` via
   `checked_add`; `end <= file_len` enforced; overlapping tensor
   ranges rejected after sorting.
7. View construction re-checks: `QuantizedWeight::try_from_mmap` and
   `KQuantWeight::try_from_mmap` verify the range against the mapping
   length; both `try_new_storage` paths verify shape rank, non-zero
   dims, block alignment of the contiguous dim, shape-product
   overflow, and byte-length consistency with the actual buffer.
8. Model construction: architecture strings are allow-listed
   (llama/qwen2/qwen3, gemma3/gemma4, gpt2); `LlamaConfig::validate`
   and per-block shape validators constrain metadata-derived counts
   and tensor shapes before inference structures are built.

Places with NO explicit transition (implicit trust):

- Parsed descriptor -> execution state is not type-separated. The raw
  `TensorInfo` (name/dims/dtype/offset) is consumed directly by the
  byte-length/range computation AND by the view-construction loop;
  the checks in item 6 are procedural checks interleaved with
  construction, not a distinct validated representation. Nothing
  stops a future edit from constructing a view from a raw descriptor
  before those checks run. This is the gap the first EmberSEC slice
  closes (see section 8).
- Metadata values (GgufValue) -> model configuration. Values are typed
  by the parser, but `LlamaConfig::from_gguf_metadata` performs `as`
  casts (u32 -> usize) and falls back to defaults for wrong types;
  config validation happens after construction, not at the metadata
  trust boundary. (Model construction remains the config gate; this
  is documented, not changed in this slice.)
- Numeric tensor payload -> kernels. Weight bytes are inherently
  attacker-controlled; kernels assume validated shapes/layouts (see
  unsafe-loader-boundary.md). No semantic checks exist on the payload
  VALUES themselves, by design (weights are opaque data).
- `tensor_meta` provenance map: raw dims/dtype/offset re-exposed after
  load. Consumed only for inventory/provenance and the qwen2 vocab
  fallback; never by numerical kernels.

## 3. Checked vs unchecked arithmetic

Checked (all use `checked_add`/`checked_mul`/`try_from`, or are
bounded before use):
- header count heuristics, string lengths, array sizes (u64->usize
  via `try_from` with context),
- rank (`n_dims` bounded 1..=4 before `with_capacity`),
- dimension u64->usize conversion,
- element-count products, per-dtype byte lengths,
- `data_start` alignment add, tensor start/end adds,
- mmap range usize conversions (`usize::try_from(tensor_offset)`).

Unchecked or implicit:
- `*a as u64` / `*n as usize` casts from metadata values in model
  config (`LlamaConfig::from_gguf_metadata`, gemma4 config): u32->usize
  cannot truncate on supported targets, but the cast is implicit; the
  resulting values are validated later by `config.validate()`.
- `alignment as u64` for GgufValue::U32 alignment (cannot truncate).
- `n_dims as usize` after the 1..=4 range check (bounded).
- `read_u32`/`read_u64`/etc. are raw little-endian reads (bounded by
  read_exact at EOF).

## 4. unsafe / FFI boundaries reachable from model-controlled values

Full inventory with classifications: docs/embersec/unsafe-loader-boundary.md.
Summary of the loader-reachable set:

- `loader.rs` `memmap2::Mmap::map` — maps the attacker-controlled file
  read-only. The mapping operation itself is safe; the risk is
  lifetime/truncation assumptions documented in the safety comment.
- `QuantizedWeight::evict_mapped_pages` (MADV_DONTNEED) — invoked from
  model.rs during repack; requires caller ordering guarantees.
- `tensor.rs` `matrixmultiply::sgemm` — operates on CpuTensor payloads
  with assert-checked shapes.
- `simd.rs` / `k_quant_matmul.rs` AVX2/AVX-512/NEON kernels — consume
  slices whose lengths/shapes are established by validated
  `QuantizedWeight`/`KQuantWeight`/`CpuTensor` objects.
- `plan.rs` scratch-arena `from_raw_parts_mut` — sized by the
  execution plan built after load; not directly model-file-derived.

No `unsafe` exists in the GGUF *parsing* itself (all scalar reads are
safe `read_exact` into fixed buffers).

## 5. Resource-exhaustion surfaces

- Metadata KV table: `try_reserve(metadata_kv_count)` — bounded by the
  `file_len / 13` heuristic (reserve failure is an error, not abort).
- Tensor table: `try_reserve_exact(tensor_count)` — bounded by
  `file_len / 32`.
- Strings: `try_reserve_exact(len)` with len <= remaining bytes
  (1:1 with file size; no amplification).
- Metadata arrays: `try_reserve_exact(count)` with count <= remaining
  bytes / min-element-size (no amplification).
- Eager dequantization: f32 materialization is at most ~4x the encoded
  tensor bytes (f16/bf16 2x), and encoded bytes are file-bounded, so
  per-tensor memory is bounded by file size. Model-wide resident
  memory is a function of legitimate model size.
- Tokenizer: loaded from a separate JSON file (`tokenizers` crate),
  not part of the GGUF path (out of scope for this slice).
- No absolute caps exist today: the heuristics scale with file size,
  so a sufficiently large hostile file can still force large
  reservations (bounded but potentially multi-GB for a multi-GB file).
  The first slice adds named absolute caps (section 8).

## 6. Assumptions currently made about valid GGUF structure

- GGUF v3 only (`version == 3`), magic `GGUF`.
- Tensor rank 1..=4, all dimensions non-zero.
- `general.alignment` is a power of two (spec: alignment used by
  writers; llama.cpp defaults 32).
- Tensor data regions do not overlap (spec-compliant writers never
  overlap; Ember rejects overlap outright — a deliberate strictness,
  since overlapping tensors cannot be two distinct weights).
- dtype codes are the post-2024 GGML numbering (Q2_K=10..Q6_K=14,
  Q8_0=8, bf16=30); Ember supports {0,1,8,10,11,12,13,14,30} with
  Q4_K/Q6_K getting compressed-resident paths and the rest eager-f32.
- Block-quantized tensors have their contiguous dim (dims[0], the
  in-features dim) aligned to the block size (Q8_0: 32; K-family:
  256). The current loader checks total element count alignment, and
  the compressed-resident constructors check the contiguous dim, but
  the eager K-quant dequant path relies on element-count alignment
  only — see section 8.
- Metadata keys are non-empty and unique; strings are UTF-8.
- Model architecture is declared in `general.architecture` and is one
  of the allow-listed families; per-family tensor names/shapes are
  validated during model construction, not at load.

## 7. Confirmed vs. suspected issues

None of the following is a demonstrated exploitable vulnerability; the
list records invariants that are enforced only procedurally (checked
inline in one function) rather than structurally:

- (structure) Parsed descriptors flow directly into view construction;
  the semantic gate is a set of inline checks, not a distinct
  validated type. A future code change could bypass them.
- (magic literals) `file_len / 32` and `file_len / 13` count
  heuristics are undocumented magic numbers; no absolute caps exist.
- (layout gap) For K-family dtypes on the eager-f32 path, only the
  total element count is checked for 256-alignment; the contiguous
  dim itself is not required to be block-aligned (the compressed path
  catches this via `KQuantWeight::try_new_storage`, the eager path
  does not). A malformed file could therefore dequantize with a
  different value ordering than a compliant runtime would produce.
- (implicit casts) Metadata u32 -> usize casts in model configs are
  validated later (`config.validate()`), not at the metadata boundary.

## 8. What this branch changes (first slice)

- Adds a distinct validated descriptor type (`ValidatedTensorInfo`)
  produced only by a `validate(...)` gate over the raw parsed
  `TensorInfo`. View construction consumes only validated
  descriptors.
- Moves the semantic checks (rank/dims, element-count product, dtype
  support, block layout, encoded byte length, file bounds) into that
  single gate, and removes the duplicated inline arithmetic.
- Adds the contiguous-dim block-alignment rule for block-quantized
  dtypes (Q8_0 % 32, K-family % 256) at the gate, closing the eager
  K-quant layout gap above.
- Replaces the count heuristics with named, documented limits
  (`loader::limits`) while keeping the file-relative bounds.
- Documents the unsafe surface and the fuzzing plan
  (docs/embersec/unsafe-loader-boundary.md, docs/embersec/fuzzing-plan.md).

## 8b. Model-construction boundary (second slice)

The loader gate validates tensor *descriptors*; the model-construction
boundary validates the *configuration* and *inventory* that architecture
builders derive from metadata before they allocate anything.

- Config caps: `loader::limits` now also bounds metadata-derived
  magnitudes (`MAX_CONTEXT_LEN`, `MAX_LAYERS`, `MAX_EMBED_DIM`,
  `MAX_VOCAB_SIZE`, `MAX_HEADS`, `MAX_HEAD_DIM`,
  `MAX_INTERMEDIATE_DIM`, `MAX_SLIDING_WINDOW`). `LlamaConfig::validate`,
  `Gemma4Config::validate`, and the GPT-2 builder enforce them before
  rope-table/KV-cache/block-vector allocation. Before this slice, a
  hostile `context_length = u32::MAX` (about 2.2 TB of RoPE tables on a
  typical head_dim) would reach `vec!` allocation and abort the process;
  it now fails with a structured error naming the violated cap. The
  llama `--max-seq-len` clamp runs after validation, so hostile metadata
  cannot hide behind a caller cap (fail-closed).
- Inventory gate: `loader::require_tensors` verifies every tensor an
  architecture builder will consume is present *before* any
  metadata-sized allocation, and reports all missing names in one
  error. Applied to the llama and GPT-2 builders (uniform per-layer
  tensor sets). Gemma4 is deliberately excluded: its shared-KV/PLE
  per-layer tensor sets are conditional on layer type and source-layer
  state, so a strict gate would duplicate the builder's own logic;
  gemma4's pre-tensor allocations are bounded by the config caps alone.
- Known pre-existing incompatibility (unchanged by this work): the
  local `gemma-4-E2B-it-Q8_0.gguf` fails `validate_loaded_shapes`
  (embedding rows 262144 vs metadata vocab 256000) — present on `main`
  before this branch; gemma4 output remains untrusted per AGENTS.md.

## 9. Loader overhead (measured)

One-off release-mode measurement on this host (2026-08, synthetic
fixtures): the validation gate costs ~69 ns per tensor descriptor, and a
full synthetic load of 500 f32 tensors (128 MiB payload) takes ~92 ms —
of which validation is ~35 µs (~0.04%). Real model loads are dominated
by I/O and dequantization (seconds), so the gate is not a measurable
factor. No permanent loader benchmark was added; re-measure if the gate
ever grows.

## 10. Out of scope (later EmberSEC phases)

TDX/SEV-SNP, attestation, network services, cryptographic inference
receipts, KV-cache security, GPU work, side-channel defenses, and the
tokenizer JSON path.
