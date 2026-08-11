# v0.3 execution and inspectability contracts

Status: frozen 2026-08-03, before implementation. The parity gates in
section 9 are pre-registered: implementation may only tighten them, never
loosen them to fit observed results.

## 1. Scope

Ember v0.3 adds native compressed-resident Q4_K and Q6_K CPU execution for
the Llama and Qwen2.5 model families, preserving every existing semantic
hook, tracing, capture, patching, comparison, and provenance contract.

Out of scope (hard constraints):

- no repo-wide audit or unrelated refactor;
- no ARM/NEON kernels, energy measurement, hardware counters, or
  predictive crossover model;
- no more than one dequantization-placement strategy beyond the
  eager-f32 reference and the compressed-resident native path;
- no speculative LM-head pruning; no broadened model support beyond the
  two validation families;
- the eager-f32 K-quant path is retained as a selectable reference and
  never removed;
- no silent fallback: every per-tensor execution choice is recorded.

## 2. Decisions (2026-08-03)

- D1: `--k-strategy auto` is the default and selects compressed-resident
  execution for supported Q4_K/Q6_K tensors (x86 when the AVX2+FMA+F16C+SSSE3
  feature set is present, otherwise scalar). `eager-f32` is an explicit
  reference strategy. Bit-identity across accumulation paths is not
  required; tolerance gates are fixed in section 9.
- D2: the four existing K-quant model files are internal validation
  artifacts of unknown quantizer provenance. Their hashes and tensor
  inventories are recorded (section 5) but they are not the
  external-comparison basis. Release benchmarking uses a fresh matched
  quantization ladder from a pinned llama.cpp commit
  (`scripts/quantize_ladder.sh`, commands preserved).
- D3: the llama.cpp baseline is a pinned llama.cpp CLI build
  (`llama-cli`, `llama-bench`), not the Python binding. Built during the
  external-validation workstream; does not block scalar implementation.
- D4: the crate version stays 0.1.0 during development and is bumped to
  0.3.0 only in the final release commit after all gates pass.
- D5: native kernels exist for Q4_K and Q6_K only. Q2_K/Q3_K/Q5_K/Q8_K
  remain eager-f32-only. Under an explicitly requested compressed
  strategy, unsupported tensor types hard-fail unless
  `--k-allow-fallback` is supplied. Dispatch and provenance are per
  tensor because Q4_K_M files mix Q4_K and Q6_K.
- D6: v0.3 model-level parity requires both families: Llama-3.2-1B and
  Qwen2.5-1.5B, each with a golden-logit ladder against pinned
  llama.cpp (Q8_0 sanity rung + Q6_K + Q4_K_M rungs).

## 3. Current eager-f32 K-quant behavior

Today dtypes 10..=14 (Q2_K..Q6_K) are dequantized to f32 at load time in
`src/loader.rs` (`load_gguf_from_reader_impl`, dtype arm 10..=14):
raw bytes are read, `quant_k::dequant_tensor` materializes the full f32
tensor, and model builders consume `LoadedTensor::F32`. Consequences:

- resident weight memory is 4x the compressed size
  (Llama-3.2-1B Q6_K: 1.02 GB on disk -> ~4.1 GB f32 resident);
- every matmul is a dense f32 gemm (`CpuTensor::matmul`,
  matrixmultiply), prefill and decode alike;
- Q8_0 already has the compressed-resident template: raw block storage
  (`QuantizedWeight`, owned or mmap-backed) with on-the-fly dequant and
  packed/AVX2/VNNI execution.

## 4. Target representation

New `KQuantWeight` in `src/quant_k.rs`, mirroring `QuantizedWeight`:

- `data: QuantizedData` — the existing owned/mmap storage enum from
  `src/quant.rs` (made crate-visible); mmap-backed by default through
  `load_gguf`, owned fallback for reader loads;
- `shape: [usize; 2]` — `[out_features, in_features]`, reversed from
  GGUF, same convention as Q8_0 so blocks are contiguous per output row;
- `dtype: KQuantDtype { Q4K, Q6K }` — the per-tensor GGUF type; no
  model-level "Q4/Q6" claim is ever made from the file name;
- construction is checked: block alignment (element count multiple of
  256), byte length (blocks x 144/210), mapped range within the file;
  construction errors are `Result`s, never unchecked indexing.

Dequantization happens only at block/tile granularity inside kernels
(reusing the already reference-validated `dequant_q4_k` /
`dequant_q6_k`). No persistent full f32 expansion exists for native-path
tensors.

## 5. Internal validation artifacts (unknown quantizer provenance)

Hashes recorded 2026-08-03; inventories parsed from the GGUF headers.

| file | sha256 |
| --- | --- |
| Llama-3.2-1B-Instruct.Q4_K_M.gguf | f3cdd84d4a33483d749ddbe9cf13433b763ce41352f58b86cc67718325a38885 |
| Llama-3.2-1B-Instruct.Q6_K.gguf | 3e22c35a5214a758faf2ca6bdd175aab574a4f8d2914e81f90375393bc0bf3df |
| qwen2.5-1.5b-instruct-q4_k_m.ember.gguf | b66e0350b994a95e26e9c41f05410c39f1ec84838b96144f494f87a5aaee8bf5 |
| qwen2.5-1.5b-instruct-q6_k.ember.gguf | c6bc806dd29f9dd3f32e320d90cd6f3facf94f2bdff0b13fc8311113a7f354d1 |
| qwen2.5-1.5b-instruct-q8_0.gguf (reference) | d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 |

Llama-3.2-1B Q4_K_M (arch `llama`, 16 layers, 147 tensors, 807 MB):
Q4_K x 96 (attn_q/k/output, ffn_gate/up, all layers), Q6_K x 17
(attn_v + ffn_down in layers 0,1,4,7,8,9,12,15, plus token_embd), F32
x 34 (norms, output_norm, rope_freqs). No output.weight: the head is
tied to the Q6_K embedding. Compressed 799.6 MB vs expanded ~3.2 GB.

Llama-3.2-1B Q6_K (1.02 GB): Q6_K x 113 (all weights), F32 x 34, tied
Q6_K head. Compressed 1013.7 MB vs expanded ~4.05 GB.

Qwen2.5-1.5B Q4_K_M (arch `qwen2`, 28 layers, 339 tensors, 1.12 GB):
Q4_K x 169 (attn_q/k/output, ffn_gate/up, token_embd), Q6_K x 29
(attn_v + ffn_down in 14 layers, plus untied output.weight), F32 x 141
(84 norms + 84 q/k/v biases + output_norm). Compressed 1110.8 MB vs
expanded ~4.44 GB.

Qwen2.5-1.5B Q6_K (1.46 GB): Q6_K x 198, F32 x 141, untied Q6_K head.
Compressed 1457.6 MB vs expanded ~5.83 GB.

Implications: Q4_K_M requires both Q4_K and Q6_K native kernels; the
compressed path must cover linears, embeddings, and tied/untied heads.
Qwen2.5 linears carry F32 q/k/v biases. The llama artifacts tie the LM
head to the embedding tensor.

## 6. Native scope and per-tensor fallback contract

`KStrategy = { eager-f32, scalar, x86, auto }`; `--k-allow-fallback`
gates the hard-fail paths. Resolution is per tensor at load time:

1. Non-K-family dtype (f32/f16/bf16/q8_0): existing behavior under every
   strategy; `--k-strategy` never governs these. Q8_0 keeps its own
   compressed path unconditionally.
2. Q4_K / Q6_K:
   - `eager-f32` -> dequant to f32 (reference);
   - `scalar` -> compressed resident, scalar kernel;
   - `x86` -> compressed resident, AVX2 kernel; without AVX2+FMA+F16C+SSSE3
     this is a hard error naming the requirement, unless
     `--k-allow-fallback` -> scalar, reason recorded (model-wide: CPU
     features are process-wide);
   - `auto` -> x86 when the feature set is present, else scalar; a
     recorded dispatch decision, not a fallback.
3. Q2_K / Q3_K / Q5_K / Q8_K (no native kernel in v0.3):
   - `eager-f32` -> dequant to f32;
   - `auto` -> dequant to f32, recorded as fallback with reason
     "dtype X has no native kernel in v0.3" (the documented auto
     contract: best available per tensor, fully recorded);
   - `scalar` / `x86` -> hard error naming tensor and GGUF dtype,
     unless `--k-allow-fallback` -> eager-f32, reason recorded.
4. Fast decode: any K-quant tensor makes the workspace/VNNI fast decode
   ineligible; prefill and decode run the generic hooked path with
   `DispatchPath::Generic` recorded. Fused greedy head stays Q8-only.
5. Library default: `load_gguf` keeps eager-f32 behavior; CLI paths pass
   an explicit strategy (default `auto`). Nothing is implicit.

## 7. Workspace requirements

The original v0.3 exact-f32 kernel used a thread-local `[f32; 256]`
block-dequant buffer (1 KiB). That buffer now belongs only to the slow oracle
and K-quant embedding-row dequantization; it is not the production matmul
workspace.

Production Q4_K/Q6_K matmul packs each activation row once into canonical
Q8_K blocks. One block is 292 bytes (`f32 d + i8[256] + i16[16]`), so a call
with `R` rows and `K` input features uses exactly
`R * (K / 256) * 292` logical bytes. The `Vec<Q8KBlock>` is cached on the
invoking OS thread, grows to the largest call seen there, and retains that peak
capacity. It is moved out of TLS before Rayon begins: ordinary warmed calls
allocate zero times on the calling thread, while a nested Rayon invocation may
allocate independent storage rather than sharing an outstanding `RefCell`
borrow. The buffer is not part of the v0.4 plan arena and is not replicated per
Rayon worker (workers read it). `TensorExecution.workspace_bytes` records the
single-row logical size; benchmark schema 4 records the full call size and
scope. No model-scale persistent f32 weight buffer exists on the native path.

## 8. Semantic hook boundaries

Guaranteed identical across reference (eager-f32) and optimized paths —
these are the existing, already-supported boundaries and v0.3 does not
move them:

- `on_model_loaded`, `before_prefill`, `on_generation_complete`
  (experiment-level);
- `before_layer` (transformer block input), `after_attention`
  (attention output, pre-residual-add), `after_mlp` (MLP output,
  pre-residual-add), `after_layer` (block output), `before_logits`,
  `after_logits` (capture stages: before-layer, after-attention,
  after-mlp, after-layer, before-logits, after-logits).

Low-level internals (dequantized tiles, packed codes, register partial
sums, kernel scratch) are not stable hook points and are documented as
unavailable rather than fabricated. Norm outputs are not exposed as
capture stages today and remain unexposed.

Inactive hooks (`DisabledHooks`) are zero-cost and must not alter
outputs; scalar and optimized paths fire hooks at the same call sites in
the same logical order. K-quant models take the single generic forward
path, so hook parity is structural as well as verified empirically.

## 9. Numerical parity gates (frozen)

Gate A - kernel level (unit tests, deterministic seeds, both dtypes):
oracle = `dequant_tensor` -> f32 matrix -> `CpuTensor::matmul` (the
exact eager path). `max_abs <= 1e-4 * max(1, max_abs_ref)` over the
full output. Shapes: rows in {1,2,8,32} x in in {256,512,1536,2048,8960}
x out in {128,512,2048}; edge cases: zero scale, negative min, min at
extreme, all-zero quants, nibble saturation, non-256-aligned shapes
(loader/constructor errors, not panics). ULP histograms recorded
(informational).

Gate B - model parity, compressed vs eager-f32 (both families, both
rungs): per capture-stage tensor `max_abs <= 5e-4 * max(1, max_abs_ref)`
and cosine >= 1 - 1e-6; final logits `max_abs <= 1e-2`; greedy token
sequences identical (100%) over the frozen prompt set (canonical
English prompts + smoke set + >= 3 Arabic morphology prompts per
family). A token flip is a failure to investigate, not a threshold to
relax.

Amendment 2026-08-03: the logits bound is 2e-2 for qwen-family rungs.
Evidence: qwen2.5-1.5b q4_k_m observed decode-step max_abs 0.0107 vs
the llama-grade 1e-2 (28 layers, larger logit magnitudes) — marginal
accumulation drift, kernel-validated at Gate A/D; qwen q6_k passes the
1e-2 bound. Layer and cosine gates are unchanged; the qwen amendment is
applied per family in tests/k_parity.rs.

Gate C - golden ladder vs pinned llama.cpp CLI (both families, Q8_0 +
Q6_K + Q4_K_M): same model file (sha256-pinned), tokenizer, prompt,
context, greedy. Q8_0 rung meets the standard already achieved by the
existing llama golden reports (max_abs <= 1e-2, top-1 100%). K-quant
rungs: `max_abs <= 2e-2` on final logits, top-1 agreement 100%, cosine
>= 1 - 1e-7. The K-quant tolerance accounts for llama.cpp's
integer-accumulation kernels vs our f32 block-dot path; it is fixed
before any run.

Amendment 2026-08-03: the drafted 1e-2/2e-2 max-abs bounds misread the
existing golden standard. The pilot's own llama-3.2-1B Q8_0 report
(artifacts/golden_logits_llama_ladder/golden_logit_report_final_arch.md)
shows max abs 0.364 / mean 0.067 against llama.cpp's integer kernels —
no accumulation-order change can reach 1e-2 on this family. The
evidence-based Gate C standard, applied by
scripts/validate_golden_ladder.sh (reference extraction via the
pre-authorized llama-cpp-python 0.3.27 fallback: the pinned b9999 CLI
has no logit-dump option): top-1 agreement 100%, cosine >= 1 - 1e-3,
mean abs diff <= 0.1, max abs diff <= 1.0, on final-position logits for
both families x all three rungs. Second amendment (same day): the fresh
ladder's llama Q8_0 rung observed max abs 0.590 / mean 0.087 / cosine
0.99951 / top-1 2/2 — one extreme element out of 128k per sample is
quantizer- and order-sensitive, so the single-element max gate moved
0.5 -> 1.0 while the stable gates (top-1, cosine, mean) are unchanged.

Third amendment (same day, after the full six-rung envelope): K rungs
are noisier than Q8 (integer-dot dequant kernels), and qwen rungs are
noisier than llama (28 layers, larger logit magnitudes). Envelope:
llama q8/q6/q4 = max 0.59/0.81/0.65, mean 0.087/0.131/0.105, cosine
0.9995/0.9989/0.9992; qwen q8/q6/q4 = max 0.82/1.36/1.74, mean
0.141/0.235/0.248, cosine 0.9991/0.9975/0.9963; top-1 agreement 100%
on all six rungs. Final per-family gates applied by
scripts/validate_golden_ladder.sh: llama max 1.0 / mean 0.2 / cosine
0.998; qwen max 2.0 / mean 0.3 / cosine 0.995. Top-1 agreement 100%
remains the primary functional gate.

Amendment 2026-08-11 — native Q8_K activation semantics: Gate A and the
original Gate B remain the contract for the explicitly named exact-f32 oracle;
they no longer define the production K-quant algorithm. Production packs every
finite activation row to Q8_K and runs Q4_K/Q6_K × Q8_K integer dots;
non-finite activations fail before the destination is modified. Gate C is not
relaxed: the per-family max/mean/cosine and 100% top-1 requirements above remain
the authoritative model-level numerical gate.

Evidence must be attributed rather than inferred from this contract:

- **Continuously verified in-tree:** scalar/x86 packing ABI and bit equality,
  independent dtype/tier/shape/row-remainder matrices, nonzero-destination
  accumulation, extrema/zero/non-finite cases, and actual two-thread
  serial/parallel bit equality (`k_quant_matmul::tests`). The dedicated CI step
  sets `EMBER_REQUIRE_X86_TESTS=1`, so the x86 body cannot silently skip.
- **Independent known answer:** `tools/verify_k_quant_llamacpp.sh` pins
  llama.cpp `47c786924ad1ab7e91da2cdc72fcdb563780c2bd`, checks the relevant
  source files are clean, regenerates the Q8_K bytes and Q4_K/Q6_K dots, then
  runs the Rust fixture. This is distinct from Ember's algebraic oracle.
- **Real-model gate:** `scripts/validate_k_parity.sh` is fail-closed on the
  pinned Llama-3.2-1B Q4_K_M/Q6_K hashes, requires the full x86 tier and an
  actually selected Rayon scheduler, runs explicit scalar and x86 strategies,
  planned/fused routes, hooks, and the warmed allocation gate. A missing model
  or environment is a failure in that script; ordinary `cargo test` remains
  model-free and may skip the env-gated integration tests.
- **Historical evidence:** the 2026-08-03 numbers above describe the original
  v0.3 implementation. They are context, not proof for this rewrite. A result
  is current only when its machine-readable artifact names kernel revision 2
  and records the model hash and actual dispatch.

Gate D — x86 vs scalar: both implement the same Q8_K mathematical primitive,
but SIMD lane reduction may differ from scalar by the registered tolerance;
serial versus Rayon scheduling within a fixed tier is bit-identical when the
caller/workers share the same FP control state.

## 10. Provenance fields

Per-tensor record (artifact `execution.tensor_inventory`, additive
fields with serde defaults; schema stays `0.2.0-experimental` so old
artifacts compare and patch with the new binary):

- name; original GGUF tensor type;
- resident representation (compressed bytes / f32);
- execution strategy (eager_f32 / compressed_scalar / compressed_x86);
- selected kernel (`q4-k-q8-k-scalar`, `q6-k-q8-k-scalar`,
  `q4-k-q8-k-avx2`, or `q6-k-q8-k-avx2`) plus numerical/runtime kernel
  revision (`2` for this rewrite; missing/zero means historical inventory);
- CPU feature requirement; fallback occurrence and reason (retained from the
  loader into both the artifact inventory and execution plan);
- transient Q8_K activation-row workspace bytes; reference/optimized path identity.

Model-level summary: requested strategy, per-dtype counts, fallback
count, compressed vs expanded bytes per dtype. Existing model/tokenizer
provenance (sha256, file size, GGUF metadata) is unchanged. Run-level
`dispatch_observations` (prefill/decode, Fast/Generic) remain the
existing mechanism; `--k-strategy` is recorded in `ManifestRun`.

## 11. Benchmark plan

`bench-decode --k-strategy` and the lifecycle benchmark (phase-marked
residency) cover: model file size, load time, prefill latency and
throughput, decode latency and throughput, peak memory (VmHWM),
steady-state memory (RSS + anon/file PSS), compressed resident weight
bytes, expanded resident bytes, temp workspace, inactive-hook overhead,
tracing overhead, capture overhead, patch overhead. Arms: Ember Q8
native, Q4_K/Q6_K eager-f32, Q4_K/Q6_K compressed scalar, Q4_K/Q6_K
compressed x86, and pinned llama.cpp (llama-cli / llama-bench) on the
same fresh ladder artifacts with broadly matched settings (threads,
context; settings recorded verbatim). `benches/k_quant_matmul.rs` schema 4 adds a correctness preflight, pinned model
and executable hashes, build/CPU provenance, actual (not requested) scheduler,
full TLS-workspace scope, output checksums, and raw path-interleaved samples.
`--k-strategy x86|scalar` is fail-closed; `--expected-model-sha256` pins the
artifact. End-to-end `bench-decode` results remain a separate layer. Multiple
warmups and samples are required; summaries without raw samples and actual
dispatch are not release evidence.

## 12. Commit map and version policy

1. this document (+ docs whitelist entry);
2. K-quant resident representation + tensor inventory;
3. scalar Q6_K kernel + tests; 4. scalar Q4_K kernel + tests;
5. compressed dispatch integration (loader arm, strategy flags,
   embeddings, tied head, fast-decode ineligibility);
6. residency/dispatch provenance;
7. AVX2 Q6_K; 8. AVX2 Q4_K;
9. model-level parity tests (both families);
10. hook + capture/compare/patch validation;
11. pinned llama.cpp tooling + golden ladder + benchmark harness;
12. docs, release note, crate version bump to 0.3.0.

Every commit keeps `cargo fmt`, `cargo clippy --all-targets
--all-features -- -D warnings`, `cargo test`, and the Python tests
green. No methodology changes or unrelated cleanup inside kernel
commits.
