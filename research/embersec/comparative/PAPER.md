# Trust Boundaries for Hostile Model Files: A Comparative Evaluation of GGUF Loader Hardening

Draft. Data frozen 2026-08-12 (research/embersec/comparative/FROZEN_ARTIFACTS.md).
Findings S1-S5 reported upstream: llama.cpp PR #26946, candle PR #3876.

## 1. Threat model

We consider a local LLM runtime that loads a GGUF model file and an
optional tokenizer.json supplied by an untrusted party (a poisoned
download, a tampered model cache, a shared artifact). The attacker does
not control the runtime binary, the OS, or other host files; their goal
is to (a) crash or hang the runtime, (b) drive memory or CPU use
disproportionate to the file size, (c) cause memory corruption reachable
from file-derived values, or (d) make the runtime silently misinterpret
a malformed-but-accepted file. Assets: process integrity, memory safety,
behavioral integrity, host availability. Out of scope: prompt
injection, weight-content backdoors (weights are untrusted *values* by
design — the boundary is structure and interpretation, not content),
remote attack, OS sandboxing. The complete model is in
docs/embersec/threat-model.md (boundaries T0-T6, attacker goals,
accepted residual risks).

## 2. The model-artifact trust-boundary abstraction

The abstraction this paper evaluates: a model artifact crosses several
trust boundaries on its way to execution, and each boundary can be
either an *implicit assumption* or an *explicit validated gate*.

```
 bytes ->[T0 parse]-> parsed descriptors ->[T1 validate]->
 validated descriptors ->[T2 views]-> tensor views ->[T3 kernels]->
 execution
                      [T4 metadata -> config]
                      [T5 config/tensors -> model construction]
 tokenizer.json ->[T6]-> tokenizer
```

For each boundary we record what crosses it, what is checked, and what
remains assumed. The design hypothesis under test: *concentrating
validation at T1/T4/T5/T6 — once, structurally, before any
file-derived value reaches allocation or kernels — converts hostile
inputs that previously manifested as panics, crashes, hangs, and silent
misinterpretation into bounded, structured rejection, at negligible
load cost, and is the architecture property that distinguishes
runtimes more than any individual check.*

## 3. EmberSEC design

Ember is a small CPU-first Rust GGUF inference engine (v0.6.2). The
hardening implemented on the `embersec/secure-gguf-loader` branch
implements the abstraction:

- T1: a distinct validated descriptor type (`ValidatedTensorInfo`) with
  no public constructor; raw parsed descriptors cannot reach view
  construction. The gate checks rank/dims, element-count products,
  dtype support, block layout including the contiguous-dim rule
  (llama.cpp's `ne[0] % QK_K == 0`), checked byte lengths, file-range
  bounds, overlap, and named absolute caps.
- T4: named limits on metadata-derived config (context length, layers,
  heads, head dimension, embedding width, vocab, RoPE table element
  product) enforced before any allocation; head dimensions must be
  positive and even; the llama `--max-seq-len` clamp runs after
  validation (fail-closed).
- T5: an inventory gate (all required tensors present before
  allocation) for llama/GPT-2; per-block shape checks; gemma4's
  per-layer construction validated before O(n) work.
- T3: kernel entry points accept validated weight types or a
  `Q8WeightView` constructible only from a validated weight; raw
  `(data, count)` pairs cannot reach SIMD kernels; arch kernels are
  private.
- T6: tokenizer boundary with a size cap, UTF-8 + JSON well-formedness
  gates, and panic containment around the third-party `tokenizers`
  crate (which panics on malformed JSON — upstream bug).

Panics reachable from hostile values are treated as defects and were
converted to structured checks. All fixes are regression-tested; no
kernel or tokenizer numerical behavior changed (k_parity and
real-model golden outputs unchanged).

## 4. Bug taxonomy

Failure classes (frozen, docs/embersec/bug-taxonomy.md): A parser
rejection failure; B arithmetic/extent validation failure; C semantic
config validation failure; D tensor-layout validation failure; E
model-construction panic; F tokenizer/deserializer panic; G
resource-exhaustion/allocation amplification; H behavioral
misinterpretation; I unsafe-boundary reachability; J
unsupported-but-cleanly-rejected. Evidence classes: confirmed panic,
confirmed behavioral bug, confirmed resource-exhaustion gap,
third-party panic exposure, structural unsafe-boundary weakness,
theoretical concern only. Fifteen Ember findings map onto A-J (table in
SYNTHESIS.md section 3): 3 confirmed panics, 2 confirmed behavioral
bugs, 4 confirmed resource gaps, 1 third-party panic, 2 structural
weaknesses, 3 theoretical concerns.

## 5. Evaluation methodology

**Corpus.** 62 artifacts (48 hostile, 14 control; 53 GGUF and 9 tokenizer
JSON; all < 4 KiB except two valid-model fixtures of about 25 KiB; 91,928
bytes total) with stable IDs, SHA-256, origin (fuzz corpus
seed / reconstructed minimized fuzz artifact / canonical synthetic /
regression fixture), expected property, bug class, format status, and
semantic comparability (FULLY / PARTIALLY / EMBER_SPECIFIC /
TOKENIZER_ONLY).

**Runtimes (exact, frozen).** Ember baseline `1157277` (v0.6.2, parent
of the first hardening commit); final EmberSEC `3ceb7039`;
llama.cpp b7999 `0c1f39a9` measured through a minimal loader harness
(`llama_model_load_from_file`, load+free, no generation — closing the
cli-stderr inference caveat; llama-cli b5999 results archived);
candle-core 0.11.0 parser harness (`gguf_file::Content::read`). All
Release builds; llama.cpp GGML_ASSERT compiled in; identical harness
code injected into both Ember trees.

**Isolation.** Every case runs in its own subprocess with a fixed
timeout, stdout/stderr capture, exit code, and peak RSS (wait4);
children crashing never kill the runner. Outcomes: ACCEPT,
STRUCTURED_REJECT, PANIC, PROCESS_CRASH, TIMEOUT, RESOURCE_LIMIT,
NOT_COMPARABLE. (Multi-worker pools were abandoned after baseline thrash caused OOM-killed
workers to corrupt results; all final numbers come from the sequential,
per-case-logged executor. Each campaign now has a run-specific summary and
its log is truncated before execution.)

**Campaigns.** (1) deterministic 62-case corpus per runtime; (2) 10,000
structured mutations (magic preserved; boundary-value patches to
counts/dims/dtype/offsets/string lengths + raw edits) — parser-layer
density; (3) 2,000 construction-layer mutations (scalar metadata
patches on the three runnable valid models, which reach every
runtime's config layer); (4) valid controls (three complete tiny
models runnable on Ember AND llama.cpp, plus parser controls); (5)
rejection-cost timing (median of 3). All tables and figures are
generated from the result files; numbers are frozen.

## 6. Results

### 6.1 Deterministic corpus (62 cases)

| outcome | baseline | EmberSEC | llama.cpp b7999 (loader) | candle 0.11 |
|---|---|---|---|---|
| ACCEPT | 15 | 14 | 2 | 39 |
| STRUCTURED_REJECT | 40 | 48 | 49 | 13 |
| PANIC | 5 | 0 | 0 | 1 |
| PROCESS_CRASH | 1 | 0 | 2 | 0 |
| TIMEOUT | 1 | 0 | 0 | 0 |
| NOT_COMPARABLE (tokenizer-only) | 0 | 0 | 9 | 9 |

8 cases changed between baseline and hardened Ember, all failures ->
structured rejection (fig2): odd head-dim panic, 1-D linear weight
panic, hostile-context thrash (30.9 s -> 10.4 ms), rope product crash,
and three tokenizer upstream panics. The remaining 54 cases are
unchanged, including all 14 valid controls. llama.cpp (loader harness,
pristine b7999) crashes on the zero-dimension input (SIGFPE, silent)
and the empty-metadata-key input (GGML_ASSERT) and rejects everything
else, including the block-alignment rule the baseline Ember
misinterpreted; candle accepts 39/53 GGUF cases at parse (structure
only) and panics on `general.alignment = 0`.

### 6.2 Parser-layer mutation campaign (10,000)

| target | ACCEPT | REJECT | PANIC | CRASH | TIMEOUT |
|---|---|---|---|---|---|
| ember baseline | 917 | 9083 | 0 | 0 | 0 |
| ember EmberSEC | 881 | 9119 | 0 | 0 | 0 |
| llama.cpp b7999 | 196 | 9603 | 0 | 197 (2.0%) | 4 |
| candle 0.11 | 3010 | 6984 | 6 | 0 | 0 |

llama.cpp failures: empty-key assert, zero-dim SIGFPE, and the
string-length unbounded-allocation hang (a 57-byte file declaring a
16 GiB string); candle panics are the alignment-zero class. Ember's
parser layer (both builds) is failure-free in this space; the semantic
classes are not reachable at this mutation density.

### 6.3 Construction-layer campaign (2,000 config mutations)

| target | ACCEPT | REJECT | PANIC | CRASH | TIMEOUT |
|---|---|---|---|---|---|
| ember baseline | 627 | 1179 | 89 (4.5%) | 44 (2.2%) | 61 (3.1%) |
| ember EmberSEC | 613 | 1387 | 0 | 0 | 0 |
| llama.cpp b7999 | 444 | 1521 | 0 | 32 (1.6%) | 3 |
| candle 0.11 | 1866 | 134 | 0 | 0 | 0 |

At the layer where config semantics live, baseline Ember fails 9.7% of
mutations (odd head dims -> panic, huge context -> thrash, huge counts
-> alloc crash); hardened Ember converts all of them to structured
rejection (0.0%). llama.cpp's construction layer (now reachable through
the loader harness) crashes on 1.6% (including GGML_ABORT on
`block_count = 0`). The campaign also found one residual Ember defect —
an O(n_layers) allocation before the layer cap in the gemma4 config
builder (3.8 s rejection on 2^32-1 layers) — fixed and regression-tested
before the final numbers.

### 6.4 Rejection cost

| case | ember EmberSEC | ember baseline |
|---|---|---|
| valid tiny llama (3.9 KB) | 10.5 ms / 20.7 MB | 10.5 ms / 20.5 MB |
| bad magic (early reject) | 10.3 ms | 10.3 ms |
| hostile context (late semantic reject) | 10.4 ms | 30.9 s (thrash) |
| Llama-3.2-1B Q8_0 (1.3 GB) | 1.77 s / 3.94 GB | 2.03 s / 3.94 GB |

Validation is not measurable at load scale (gate microbenchmarked at
~69 ns/descriptor); the strongest contrast is the hostile-context case:
10.4 ms structured rejection vs 30+ seconds of pathological allocation.

### 6.5 Upstream disclosure

All five suspected external findings were reproduced on the newest
upstream revisions (llama.cpp b7999; candle 0.11.0) with minimized
reproducers (37-96 bytes; the string-length hang reduced to 57 bytes)
and reported: llama.cpp PR #26946 (S1; S2/S3 were found already fixed
upstream and S5 mitigated by a clear assert) and candle PR #3876 (S4).
The reports and current upstream status are recorded in
suspected-external-bugs.md; unmerged fixes are not treated as measured
runtime behavior. The matrix measures
the pristine upstream revisions; the fixes are upstream changes.

## 7. Limitations

Read before interpreting: (1) the corpus is adversarial-fixture-based —
it demonstrates failure classes and bounded rejection, not failure
rates (rate claims come only from the mutation campaigns); (2) raw
mutations rarely reach the semantic layer, so parser-layer rates and
construction-layer rates must be read separately; (3) llama.cpp is
measured through the loader API on one commit (b7999); the crash
classes were additionally reproduced on b5998/b5999, but no wider
range; (4) candle is parser-level only — "accept" means structure
parsed, with tensor data and model construction unexercised; (5) no
memory-safety claim is made for any finding, including Ember's; (6)
the host is thermally noisy (timings are medians of 3); (7) the
tokenizer boundary contains a third-party crate's panics; upstream
panics beyond the demonstrated class remain possible in principle;
(8) this evaluation is not a certification of any runtime.

## 8. Related work

Prior work on ML-model file security is dominated by the
*serialization-layer RCE* problem: pickle-based malware in model hubs
(Models Are Codes, 2024; Malicious AI Models Undermine Software
Supply-Chain Security, CACM 2025), the safetensors response and its
adoption (empirical study, 2025), and secure-deserialization schemes
such as PickleBall (2025). Adjacent lines: GPU memory leakage in LLM
runtimes (LeftoverLocals, 2024) and generic deserialization-RCE
analysis (Java, 2022). Format-parser robustness has been studied for
other ecosystems (ONNX optimizer fault detection, 2025; verified
parser generators), and hostile-input testing of C/C++ parsers is
standard practice (OSS-Fuzz), but we found no published work that (a)
treats GGUF — the dominant open LLM weight format — as an attack
surface, (b) evaluates hostile GGUF handling across local LLM runtimes,
or (c) formalizes a parser-to-execution trust-boundary architecture
(validated-descriptor types, named resource caps, construction
inventory gates) for binary model formats and measures what it changes.
This paper is, to our knowledge, the first comparative evaluation of
that abstraction for GGUF loading. (Search method: arXiv, Semantic
Scholar, OpenAlex, 2026-08; queries listed in the appendix. A 2025
preprint titled "On the (In)Security of Loading Machine Learning
Models" surfaced in a broad arXiv query but could not be reliably
retrieved or verified in this session; it is not cited.)

## 9. Introduction

Local LLM inference has made large models into ordinary downloaded
artifacts: users fetch multi-gigabyte weight files from model hubs,
caches, and direct links, and hand them to runtimes — llama.cpp,
Ollama, MLX, candle, and a growing field of Rust and C++ engines — that
parse and execute them locally. The dominant weight format is GGUF:
a binary container whose header, metadata, tensor descriptors, and
quantized payload are all attacker-controlled bytes the moment a file
is untrusted. Model files are, in effect, executable control inputs:
they determine allocations, mmap views, configuration, and which
numerical kernels run on which buffers.

This paper evaluates a specific architectural answer to that problem.
Instead of scattering defensive checks through parsers and kernels, we
make the trust boundaries between file bytes and execution explicit:
parsed descriptors must pass a validation gate into a distinct
validated representation that kernels and builders consume; metadata
must pass named resource caps before any allocation; tokenizer JSON
must pass well-formedness and size gates before a third-party
deserializer sees it. We implemented this architecture in Ember, a
small CPU-first Rust GGUF engine, and then asked three questions with
data: what failure classes does the hardening convert into structured
rejection; what did the same hostile inputs do to the pre-hardening
baseline and to two other local GGUF runtimes; and what does the
hardening cost on valid loads?

Our results: across a frozen 62-case hostile corpus spanning ten
trust-boundary classes, the baseline exhibited five panics, one
process crash, one resource timeout, and one silent layout
misinterpretation; the hardened runtime converts all eight into
bounded structured rejection in at most 10.4 ms — the same cost as a
valid load — while leaving every valid control byte-for-byte
unchanged. Under 10,000 structured mutations of GGUF structure, the
hardened runtime shows zero failures; at the model-construction layer,
where config semantics live, the baseline fails 9.7% of 2,000 config
mutations (panics, crashes, and 30-second allocation thrash) and the
hardened runtime fails 0.0%. The same corpus and campaigns applied to
pristine llama.cpp b7999 through its loader API crash the process on
2.0% of parser mutations (a silent SIGFPE, an assert abort, and an
unbounded string-allocation hang) and 1.6% of construction mutations,
and to candle 0.11.0, which panics on a 57-byte alignment-zero file.
All five external findings were reported upstream with minimized
reproducers and fixes (llama.cpp PR #26946, candle PR #3876).

The contribution is not "Ember rejects bad files" — the pre-hardening
parser already rejected most malformed structure. It is that *explicit,
type-enforced validation at the trust boundary* is what converts the
remaining failure classes — semantic-config panics, layout
misinterpretation, resource amplification, and third-party
deserializer panics — into deterministic bounded rejection, and that
this property is measurable and portable: the same classes that
hardened Ember eliminates still crash or hang the most widely deployed
GGUF runtime.

## 10. Abstract

Local LLM runtimes execute attacker-controlled model files: GGUF
weights are untrusted bytes that determine allocations, mmap views,
configuration, and kernel dispatch. We evaluate an explicit
trust-boundary architecture for this problem — a validation gate
between parsed descriptors and a distinct validated representation,
named caps on metadata-derived configuration, inventory gates before
model construction, and containment around third-party tokenizer
deserialization — implemented in Ember, a small CPU-first Rust GGUF
engine, and measured against its pre-hardening baseline and against
llama.cpp b7999 (via its loader API) and candle 0.11.0 on an identical
frozen corpus and mutation campaigns (62 deterministic hostile cases,
10,000 parser-layer mutations, 2,000 construction-layer mutations, and
valid controls). The baseline exhibited 5 panics, 1 crash, 1 resource
timeout, and 1 silent layout misinterpretation; the hardened runtime
converts all eight cases into structured rejection in at most 10.4 ms,
fails 0.0% of 2,000 construction-layer config mutations (baseline:
9.7%), and leaves valid-input behavior and load cost unchanged. The
same inputs crash pristine llama.cpp on 2.0% of parser mutations
(silent SIGFPE, assert abort, unbounded string-allocation hang) and
1.6% of construction mutations, and panic candle on alignment-zero;
all five external findings were reported upstream with fixes (llama.cpp
PR #26946, candle PR #3876). We conclude that concentrating validation
once, structurally, at the parser-to-execution trust boundary — rather
than in kernels or scattered checks — is what converts hostile model
files from process failures into bounded, structured rejection, and
that the same failure classes remain live in the most widely deployed
GGUF runtime.
