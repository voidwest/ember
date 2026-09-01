# Suspected external bugs found during the comparative evaluation

PRIVATE research note. Do not publish a weaponized reproducer; the
fixtures themselves are the minimal, inert reproducers (37 and 96 bytes)
already committed to the corpus. Upstream disclosure status and links are
recorded below; unmerged fixes are not treated as measured behavior.

## S1. llama.cpp: silent SIGFPE on zero-dimension GGUF tensor

- fixture: fixtures/gguf/zero_dim (96 bytes; header + one tensor info
  with n_dims=1, dim=0, dtype=f32)
- runtime: llama.cpp b7999 (`0c1f39a9`), loader harness, Release, gcc 16.1.1
  (the same crash was also reproduced through llama-cli b5999)
- behavior: process dies with SIGFPE (exit -8) with NO message during
  `gguf_init_from_file_impl` (ggml/src/gguf.cpp), before model
  construction. GDB backtrace: signal in gguf_init_from_file_impl called
  from llama_model_loader.
- classification: crash on malformed input (integer division by zero).
  Ember: structured reject; Candle parser: accepts (no dim check at
  parse). No memory-safety evidence gathered; stopped at crash
  attribution.

## S2. llama.cpp: GGML_ASSERT abort on empty metadata key

- fixture: fixtures/gguf/empty_key (37 bytes; one metadata KV with an
  empty key)
- runtime: llama.cpp b7999 loader harness (also reproduced on b5999 CLI)
- behavior: `GGML_ASSERT(!key.empty()) failed` at ggml/src/gguf.cpp:132,
  SIGABRT (exit -6). GGML_ASSERT is compiled into Release builds.
- classification: assert/abort on malformed input. Ember: structured
  reject ("GGUF metadata keys must not be empty"); Candle parser:
  accepts.

## Comparator context (same fixtures)

- Ember baseline: STRUCTURED_REJECT for both (pre-EmberSEC checks).
- Ember current: STRUCTURED_REJECT for both.

## S3. llama.cpp: unbounded string allocation on hostile GGUF string length

- input: results/diff_fuzz/crashes/llama-cpp/be627a5a65f6d91d.bin (25.8 KB,
  a mutated corpus seed whose tokenizer-tokens array string length was
  patched to ~25.8 GB)
- runtime: llama.cpp b7999 (`0c1f39a9`) loader harness, Release
  (also reproduced through llama-cli b5999)
- behavior: process HANGS (verified >15 s, killed by harness timeout)
  inside `gguf_read_emplace_helper<std::string>` ->
  `std::string::_M_replace_aux` (gdb backtrace): the GGUF string reader
  resizes the destination to the declared length (~26 GB) and zero-fills
  before any check against remaining file bytes. CPU-bound thrash until
  OOM/timeout — a resource-exhaustion DoS on a 25 KB file.
- Ember (baseline AND current): structured reject (string length vs
  remaining-bytes check), no large allocation.
- classification: resource-exhaustion hang on malformed input. No
  memory-safety evidence gathered.

## S4. Candle 0.11.0: divide-by-zero panic on general.alignment = 0

- input: results/diff_fuzz/crashes/candle/ac4c8ef93b38e3a0.bin (57 bytes:
  GGUF v3, zero tensors, one KV `general.alignment = 0`)
- runtime: candle-core/candle-transformers 0.11.0 (parser harness)
- behavior: Rust panic "attempt to divide by zero" at
  `candle-core-0.11.0/src/quantized/gguf_file.rs:560`
  (`position.div_ceil(alignment)` with alignment = 0) — exit 101.
- same input: Ember (baseline + current) STRUCTURED_REJECT ("invalid
  GGUF alignment 0"); llama.cpp b7999 STRUCTURED_REJECT.
- classification: panic on malformed input (no memory-safety evidence).
- found by the structured differential fuzz run (10,000 mutations).

## S5. llama.cpp: GGML_ABORT on block_count = 0 (construction layer)

- input: a mutated tiny-llama valid model with `llama.block_count = 0`
  (construction-layer differential fuzz; saved under
  results/diff_fuzz/crashes/llama-cpp/)
- runtime: llama.cpp b7999 (`0c1f39a9`) loader harness, Release
  (also reproduced through llama-cli b5999)
- behavior: SIGABRT via `GGML_ABORT("fatal error")` in
  `llama_hparams::n_head(il)` (src/llama-hparams.cpp:26) when a layer
  index `il >= n_layer(0)` is queried during model construction.
- Ember: structured reject ("model must contain at least one layer").
- classification: assert-style abort on malformed config (no
  memory-safety evidence).

## Upstream status update (2026-08-12, after PR updates)

- S1 (zero-dim SIGFPE): llama.cpp PR #26946 now contains ONLY this fix
  (the other three were already addressed upstream by the time master
  moved 998 commits past b7999). Rebased onto master 89e0aa6, single
  commit, MERGEABLE; verified on master: "dimension 1 must be positive,
  got 0" instead of SIGFPE.
- S2 (empty-key assert): FIXED upstream on master (empty keys rejected
  with a structured error at gguf.cpp KV read); verified empirically.
- S3 (string-length hang): FIXED upstream on master
  (GGUF_MAX_STRING_LENGTH + remaining-bytes checks in read(std::string));
  verified empirically (clean reject).
- S5 (block_count=0): MITIGATED upstream on master — a clear
  GGML_ASSERT(n_layer_all > 0 && <= LLAMA_MAX_LAYERS) now fires instead
  of the opaque "fatal error" abort; a process abort on hostile input
  remains (assert, not a structured error).
- S4 (candle alignment-zero): candle PR #3876 MERGEABLE; CI runs show
  action_required (fork-workflow approval gate) — awaiting maintainer
  approval, no failure of the change.

## What was NOT done

- No exploit development, no payload beyond the tiny inert fixtures.
- No additional disclosure beyond the tracked llama.cpp and Candle PRs
  listed above.
- No claim of memory unsafety for either finding (S1 is a divide
  crash; S2 is an assert abort).
