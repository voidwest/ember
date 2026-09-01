# EmberSEC: hostile-model bug taxonomy

> **Phase I provenance:** frozen audit documentation from branch snapshot
> `e1fe6269`; the measured hardened Ember target is `3ceb7039`. Current main
> retains the applicable hardening, but implementation names and dataflow may
> have evolved. Read this as the published Phase I evidence record.

Frozen before the comparative evaluation (research/embersec/comparative/)
so that corpus classification and outcome tables use one stable schema.
The taxonomy separates *failure classes* from *evidence strength*.

## 1. Failure classes (corpus `bug_class`)

| code | class | description |
|---|---|---|
| A | parser rejection failure | the parser accepts bytes it should reject, or rejects with an unstructured/panic path instead of an error |
| B | arithmetic / extent validation failure | overflow, narrowing, or wrapping in counts/offsets/extents leading to wrong allocation or range |
| C | semantic model-configuration validation failure | metadata-derived config (layers, heads, context, vocab) not validated before use |
| D | tensor-layout validation failure | block alignment, rank/shape, dtype layout accepted when malformed |
| E | model-construction panic | panic/assert during model building from hostile metadata or tensors |
| F | tokenizer/deserializer panic | panic in tokenizer JSON parsing or its dependency stack |
| G | resource-exhaustion / allocation amplification | unbounded or absurd allocation driven by hostile counts/strings/metadata |
| H | behavioral misinterpretation / correctness failure | input accepted but interpreted differently than a compliant runtime would |
| I | unsafe-boundary reachability | raw attacker-influenced values can reach `unsafe` code without a validated gate |
| J | unsupported-but-cleanly-rejected input | valid-format input Ember does not support, rejected with a structured error (expected) |

Control cases use `bug_class = "control"`.

## 2. Evidence-strength classification (for known findings)

- **confirmed panic** — a panic was demonstrated on a concrete input (fuzz artifact or regression fixture), before the fix.
- **confirmed behavioral bug** — demonstrably different numerical/structural interpretation vs a compliant runtime (e.g. llama.cpp) on a concrete input.
- **confirmed resource-exhaustion gap** — a concrete input drives an allocation that is unbounded/absurd relative to input size (demonstrated by arithmetic; OOM abort observed where feasible).
- **third-party panic exposure** — panic originates in a dependency (not Ember code); Ember boundary converts it to a structured error.
- **structural unsafe-boundary weakness** — no demonstrated memory error; the weakness is that a procedural gate (or raw public kernel signature) is not type-enforced.
- **theoretical concern only** — no concrete triggering input demonstrated; reasoning only.

## 3. Known Ember findings mapped to the taxonomy

| # | finding | bug class | evidence | status |
|---|---|---|---|---|
| 1 | K-quant eager path accepted `dims[0] % 256 != 0` (contiguous dim misaligned); dequant order differs from llama.cpp | D | confirmed behavioral bug (fixture `q4_k_dim_misaligned`) | fixed (validation gate) |
| 2 | Q8_0 contiguous-dim misalignment accepted by loader (caught only in compressed constructor) | D | confirmed behavioral bug (fixture `q8_0_dim_misaligned`) | fixed (validation gate) |
| 3 | `attention.key_length = 1` (odd) passed config validation, panicked in `compute_rope_freqs` | C+E | confirmed panic (fuzz-minimized artifact) | fixed (evenness check) |
| 4 | gemma4 `rope_freqs.weight` hostile length/values would panic `compute_rope_freqs` asserts | D+E | confirmed panic (assert reachable; reviewed, not fuzz-demonstrated) | fixed (validated before use) |
| 5 | 1-D F32 linear weights panic `gguf_to_row_major_f32`/`transpose` 2-D asserts | D+E | confirmed panic (assert reachable; reviewed) | fixed (rank check at take helpers) |
| 6 | `context_length = u32::MAX` drives multi-TB RoPE allocation before any tensor consumption | C+G | confirmed resource-exhaustion gap | fixed (config caps) |
| 7 | `MAX_CONTEXT_LEN x MAX_HEAD_DIM` product gap (16M x 4096 = ~256 GiB) | G | confirmed resource-exhaustion gap | fixed (`MAX_ROPE_TABLE_ELEMENTS`) |
| 8 | `block_count = 1M` sizes a million-element block vector | C+G | confirmed resource-exhaustion gap | fixed (config caps) |
| 9 | tokenizers-0.20.4 `decoders/mod.rs:90 .expect("Helper")` panic on malformed JSON | F | third-party panic exposure (fuzz-demonstrated, 26-byte repro) | fixed at boundary (UTF-8 + JSON gates + catch_unwind) |
| 10 | no tokenizer file-size bound; whole file read before parse | G | confirmed resource-exhaustion gap | fixed (`MAX_TOKENIZER_BYTES`) |
| 11 | raw parsed `TensorInfo` flowed into view construction without a distinct validated type | I | structural unsafe-boundary weakness (no demonstrated memory error) | fixed (`ValidatedTensorInfo` boundary) |
| 12 | `simd::dequantize_q8_0_row` pub raw-slice kernel entry | I | structural unsafe-boundary weakness | fixed (`Q8WeightView` contract) |
| 13 | header count heuristics (`file_len/32`, `file_len/13`) as undocumented magic; no absolute caps | G | theoretical concern only (counts were file-bounded 1:1) | fixed (named limits) |
| 14 | metadata u32 -> usize implicit casts in model configs | C | theoretical concern only (values re-validated in `config.validate`) | fixed (explicit caps + docs) |
| 15 | Oniguruma regex ReDoS at tokenizer encode time | G | theoretical concern only (no demonstrated input) | documented, not fixed |

Pre-existing (present in the baseline `1157277`, NOT introduced or fixed by EmberSEC): duplicate-key/name rejection, string-vs-remaining bounds, rank 1..=4 and non-zero dims, checked element-count products and byte lengths, file-range bounds, overlap rejection, power-of-two alignment, unsupported-dtype rejection. These are baseline `STRUCTURED_REJECT` behavior and are control data for the delta.

## 4. Outcome schema (harness)

| outcome | meaning |
|---|---|
| ACCEPT | artifact loaded/parsed successfully (control or comparator-supported feature) |
| STRUCTURED_REJECT | clean error path; process exit code indicates a reported error; no panic/crash |
| PANIC | Rust panic (unwound or aborted) |
| PROCESS_CRASH | signal death (SIGSEGV/SIGABRT/etc.), including OOM abort |
| TIMEOUT | exceeded per-case wall-clock budget |
| RESOURCE_LIMIT | killed by the OS for memory (OOM killer) or measured RSS pathological |
| SEMANTIC_MISINTERPRETATION | accepted but interpreted differently than a compliant runtime |
| UNSUPPORTED | valid-format input outside the runtime's supported feature set |
| HARNESS_ERROR | runner/setup failure, not attributable to the artifact |
