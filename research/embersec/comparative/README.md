# EmberSEC comparative evaluation — results summary

Area: `research/embersec/comparative/`
Question: do recurring hostile-model trust-boundary failures appear across
local GGUF runtimes, and does Ember's explicit validation architecture
convert those failure classes into bounded, structured rejection with low
overhead?

## Method (see environments.json, corpus.json, run_eval.py)

- Corpus: 62 fixtures (48 hostile, 14 control; 53 GGUF + 9 tokenizer JSON;
  all < 4 KiB except two valid-model fixtures, 91,928 bytes total), with
  stable IDs, SHA-256, origin,
  bug class (taxonomy A-J), and semantic-comparability labels.
- Every case runs in its own subprocess: fixed 30 s timeout, stdout/stderr
  capture, exit code, peak RSS (wait4). Child crashes cannot kill the
  runner.
- Targets: Ember current (`3ceb7039`), Ember baseline (`1157277`, the
  parent of the first EmberSEC commit — already contains the pre-EmberSEC
  fail-closed loader checks), llama.cpp `b7999` (`0c1f39a9`, loader harness),
  Candle `0.11.0` (parser-level harness).

## Headline numbers

| target | ACCEPT | STRUCTURED_REJECT | PANIC | PROCESS_CRASH | TIMEOUT | NOT_COMPARABLE |
|---|---|---|---|---|---|---|
| Ember baseline | 15 | 40 | 5 | 1 | 1 | 0 |
| Ember current | 14 | 48 | 0 | 0 | 0 | 0 |
| llama.cpp b7999 (53 GGUF, loader) | 2 | 49 | 0 | 2 | 0 | 9 |
| Candle 0.11 (53 GGUF, parser-level) | 39 | 13 | 0 | 0 | 0 | 9 |

## Cases whose outcome changed after EmberSEC hardening (8 of 62)

| case | bug class | baseline | current |
|---|---|---|---|
| gguf-025 q4_k-dim-misaligned | D | ACCEPT (semantic misinterpretation; eager dequant with non-compliant block order) | STRUCTURED_REJECT |
| gguf-042 llama-context-u32-max | G | TIMEOUT (multi-TB rope zero-fill thrashes; killed at 30 s) | STRUCTURED_REJECT (10.4 ms) |
| gguf-045 llama-odd-key-length | E | PANIC (compute_rope_freqs assert) | STRUCTURED_REJECT |
| gguf-046 llama-1d-attn-q | E | PANIC (transpose 2-D assert) | STRUCTURED_REJECT |
| gguf-047 llama-rope-product-cap | G | PROCESS_CRASH (SIGABRT, ~256 GiB alloc) | STRUCTURED_REJECT |
| tok-002 decoder-invalid-utf8-26b | F | PANIC (upstream tokenizers .expect) | STRUCTURED_REJECT |
| tok-003 decoder-bad-value-15b | F | PANIC | STRUCTURED_REJECT |
| tok-004 decoder-invalid-utf8-nested | F | PANIC | STRUCTURED_REJECT |

The other 54 cases behave identically in baseline and current: valid
controls still load, and the pre-EmberSEC fail-closed checks (bounds,
counts, strings, rank, overlap, dtype) still reject cleanly — the
hardening added bounded rejection without changing accepted-input
behavior.

## Comparator notes

- **llama.cpp b7999** (loader harness) rejects every malformed GGUF case we
  threw at it *except two, where it crashes*: `zero_dim` (96 B) → silent
  SIGFPE (integer division
  by zero) inside `gguf_init_from_file_impl`; `empty_key` (37 B) →
  `GGML_ASSERT(!key.empty())` abort at ggml/src/gguf.cpp:132. Both are
  classified as SUSPECTED EXTERNAL BUGS (see
  suspected-external-bugs.md). llama.cpp also enforces the exact
  contiguous-dim block-alignment rule EmberSEC added ("tensor 't.weight'
  of type 12 (q4_K) has 128 elements per row, not a multiple of block
  size (256)") — i.e. baseline Ember *misinterpreted* what a compliant
  runtime rejects.
- **Candle** (parser-level `gguf_file::Content::read`) accepts 39/53 GGUF
  cases including zero dims, empty keys, misaligned quant blocks, and
  hostile metadata — expected for a structure-only parser with no model
  construction; the harness intentionally does not load tensor data or
  build models, so these are "parser accepted; downstream unverified",
  not defects. Candle rejects 13 (magic, version, absurd counts/strings,
  truncated structures) with its own caps.
- Tokenizer cases are TOKENIZER_ONLY (llama.cpp/candle do not consume
  tokenizer.json) — marked NOT_COMPARABLE, not run.

## Rejection cost (median of 3, release builds, this host)

| case | ember-current | ember-baseline |
|---|---|---|
| valid tiny llama (3.9 KB) | 10.5 ms / 20.7 MB RSS | 10.5 ms / 20.5 MB |
| bad magic (early reject) | 10.3 ms | 10.3 ms |
| hostile context (late semantic reject) | 10.4 ms | **30.9 s** (thrash) |
| Llama-3.2-1B-Instruct Q8_0 (1.3 GB) | 1.77 s / 3.94 GB | 2.03 s / 3.94 GB |

The validation gate adds no measurable load cost (valid-tiny and real
model are identical within noise; the gate itself measured ~69 ns per
tensor descriptor in an earlier microbenchmark). The hostile-context
case shows the strongest cost contrast: 10.4 ms structured rejection vs
30+ seconds of pathological allocation in the baseline.

## Coverage (semantic matrix)

header/count 6, metadata 7, strings/arrays 5, tensor descriptors 16,
extent arithmetic 9, quantization layout 7, overlap/range 2, architecture
metadata 7, model construction 7, tokenizer JSON 9 (sum > 62 because
cases can cover several areas). Baseline failures per area: see
tables/D_coverage.md.

## Experiment 1: structured differential fuzzing (diff_fuzz.py)

Two campaigns, sequential execution (per-case subprocess isolation;
multi-worker pools proved unreliable under the baseline's thrashing
children — OOM-killed workers corrupted results, so the final numbers
are from the sequential, per-case-logged executor):

### Raw mutation campaign (10,000 mutations, seed 7; results/diff_fuzz/summary_raw-10000-7.json)

| target | ACCEPT | STRUCTURED_REJECT | PANIC | PROCESS_CRASH | TIMEOUT |
|---|---|---|---|---|---|
| ember-current | 881 | 9119 | 0 | 0 | 0 |
| ember-baseline | 917 | 9083 | 0 | 0 | 0 |
| llama.cpp b7999 | 196 | 9603 | 0 | 197 (2.0%) | 4 |
| candle 0.11 (parser) | 3010 | 6984 | 6 | 0 | 0 |

llama.cpp failures: the known classes (empty-key GGML_ASSERT, zero-dim
SIGFPE) plus S3 (unbounded string allocation hang on mutated length
fields). Candle's panics are the S4 class (`general.alignment = 0` ->
`div_ceil(0)`). Ember (both) shows zero failures; the raw mutation
space does not reach the semantic/config layer.

### Construction-layer campaign (2,000 config-only mutations of the valid models, seed 11; results/diff_fuzz/summary_construction-2000-11.json)

| target | ACCEPT | STRUCTURED_REJECT | PANIC | PROCESS_CRASH | TIMEOUT |
|---|---|---|---|---|---|
| ember-baseline | 627 | 1179 | 89 (4.5%) | 44 (2.2%) | 61 (3.1%) |
| ember-current | 613 | 1387 | 0 | 0 | 0 |
| llama.cpp b7999 | 444 | 1521 | 0 | 32 (1.6%) | 3 |
| candle 0.11 (parser) | 1866 | 134 | 0 | 0 | 0 |

At the layer where config semantics live, baseline Ember fails ~9.7%
of mutations (odd head dims -> panic, huge context -> thrash/timeout,
huge counts -> alloc crash); hardened Ember converts all of them to
structured rejection (0.0%). llama.cpp's construction layer is now
reachable and adds a new crash class: GGML_ABORT on `block_count = 0`
(S5). The campaign also found and fixed one residual Ember issue: the
gemma4 config builder allocated O(n_layers) BEFORE the layer cap check
(~3.8 s rejection on 2^32-1 layers; now constant-time — the early cap
is regression-tested). Both known llama.cpp crash classes reproduce on
the neighbor commit b5998 (446595b9), spanning at least two releases.

## Limitations (kept deliberately conservative)

- The corpus is adversarial-fixture-based, not a random fuzz campaign;
  it demonstrates failure classes, not failure rates.
- llama.cpp b7999 was exercised through its loader API (no generation),
  so parse-level outcomes are direct return values rather than stderr
  markers. Release build: GGML_ASSERT is always compiled in (unlike
  NDEBUG-style asserts) and aborting on `empty_key` confirms it.
- Candle comparison is parser-level only.
- Host is thermally noisy; timings are medians of 3, not microbenchmarks.
- No claim is made about any runtime's overall security posture.
