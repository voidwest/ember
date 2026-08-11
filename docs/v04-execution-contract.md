# v0.4 execution and inspectability contract

Status: frozen 2026-08-04, before implementation. The parity gates in
section 13 are pre-registered: implementation may only tighten them, never
loosen them to fit observed results. No fusion is implemented before this
document exists in this form.

## 1. Scope

Ember v0.4 adds a model-specific execution plan constructed once after load,
decode through that plan with reusable scratch storage, resolved kernel
dispatch, and a small frozen set of bounded fused operations, for the four
primary K-quant combinations:

- Llama-3.2-1B Q4_K_M;
- Llama-3.2-1B Q6_K;
- Qwen2.5-1.5B Q4_K_M;
- Qwen2.5-1.5B Q6_K.

Q8_0 behavior is preserved unchanged: the v0.3 Q8_0 native path (packed
weights, workspace decode, fused greedy head) is not rerouted through the
v0.4 plan and must not regress.

Out of scope (hard constraints, from the release spec):

- no Q2_K/Q3_K/Q5_K native kernels;
- no speculative decoding, continuous batching, GPU, quantized KV cache,
  paged attention, or JIT;
- no broad new architecture support beyond the two v0.3 validation families;
- no thread-pool redesign unless profiling proves the current pool blocks
  the release;
- no second hidden runtime that bypasses hooks;
- no graph rewriting beyond the frozen fusion set (section 6);
- no weakening of any v0.3 gate to make v0.4 pass;
- eager-f32 K-quant execution remains selectable as the reference oracle and
  is never removed.

## 2. Decisions (2026-08-04)

- D1: `--execution` selects the execution concept: `reference` (the existing
  v0.3 generic hooked path with per-tensor K dispatch, the readable
  oracle), `planned` (plan-driven dispatch, identical operation sequence,
  scratch arena, no fusion), or `planned-fused` (the frozen fusion set with
  per-layer de-fusion driven by active hooks). During development the
  default is `reference`; release default becomes `planned-fused` only after
  every gate in section 13 passes.
- D2: the execution plan is an immutable value built once, immediately after
  model loading and validation, by the model backend itself (it owns the
  tensors). The plan stores stable indices and validated metadata, never raw
  pointers. The decode interpreter maps `(layer, op)` to concrete model
  fields through the same code paths that construct the model, so weight
  access is structural and cannot dangle.
- D3: tensor identity in the plan is a `TensorRef { id }` into a per-model
  `tensor_table` (name, GGUF shape, dtype, execution, kernel, residency,
  mmap range). The table is derived from the v0.3 tensor inventory and is
  itself deterministic.
- D4: the crate version stays 0.3.0 during development and is bumped to
  0.4.0 only in the final release commit after all gates pass.
- D5: HookMode is resolved at plan build: `Disabled`, `Observe`, or
  `Intervene`, derived from the active experiment set. The plan is built
  per-run (mode and active sites are known before decode starts); the model
  execution-plan build is cached per `(model, execution, hook-mode,
  active-sites, KV capacity, model/tokenizer hashes, Rayon thread count)` key.
- D6: kernels are not duplicated. Planning resolves a `KernelId` per tensor;
  the serialized plan is authoritative and the release interpreter hard-fails
  if it disagrees with the resident weight. Plan validation enforces a
  one-to-one tensor/dispatch table, kernel identity, and CPU requirement. The
  Q4_K/Q6_K × Q8_K rewrite is `PLAN_KERNEL_REVISION = 2`; historical revision
  1 plans remain deserializable for offline bundle verification but cannot be
  executed as current plans.
- D7: `execution-plan.json` (schema `v04-plan/1`, kernel revision 2) is written under
  `artifacts/benchmark-v04/<run>/` for every benchmarked run and is also
  available through `ember inspect-plan`.

## 3. Current v0.3 decode sequence (frozen reference)

Both families run the single generic hooked forward path in `src/llama.rs`
(`Llama::forward_with_cache_hooked`, block path
`LlamaBlock::forward_with_cache_hooked`, attention
`LlamaAttention::forward_with_cache`). Config differences (RoPE layout,
qk-norm order, head counts, inter dim, biases) are data, not code:
`llama` uses `RopeLayout::AdjacentPair` with `QkNormOrder::AfterRope` and no
qk-norm tensors; `qwen2`/`qwen3` use `RopeLayout::SplitHalf` with
`QkNormOrder::BeforeRope` (qwen2.5-1.5B has no q_norm/k_norm tensors; qwen3
has them). Qwen2.5-1.5B is in scope as `qwen2`.

Model-level sequence (prefill `seq=N`, decode `seq=1`; `start_pos` is the
absolute KV position):

1. `x0 = Embedding(token_embd.weight, tokens)` — row lookup + K-quant
   dequant of one row; `[seq, embed]` f32.
2. For layer `l` in `0..n_layers` (`before_layer` hook fires on block input):
   a. `n1 = RMSNorm(x, input_layernorm, eps)` — `[seq, embed]` f32;
      norm weight is F32.
   b. `q = q_proj(n1)` — `[seq, n_heads*head_dim]` f32 (K-quant matvec;
      qwen adds F32 bias).
   c. `k = k_proj(n1)` — `[seq, n_kv_heads*head_dim]` f32.
   d. `v = v_proj(n1)` — `[seq, n_kv_heads*head_dim]` f32.
   e. `q_r = RoPE(q)` with optional qk-norm per `QkNormOrder` — same shape.
   f. `k_r = RoPE(k)` with optional k-norm — same shape.
   g. KV store: `k_r`, `v_r` converted f32→f16 into the flat cache
      `[layer][head][pos][head_dim]` at `cursor..cursor+seq`.
   h. `attn = causal_attention(q_r, cache_k, cache_v)` — scores
      `[n_heads, total_seq]` f32 scratch, softmax, weighted sum →
      `[seq, n_heads*head_dim]` f32.
   i. `o = o_proj(attn)` — `[seq, embed]` f32 (`after_attention` hook fires
      on `o`, pre-residual).
   j. `x1 = x + o` — residual add.
   k. `n2 = RMSNorm(x1, post_attention_layernorm, eps)` — `[seq, embed]`.
   l. `g = gate_proj(n2)` — `[seq, inter]` f32; `gs = silu(g)`.
   m. `u = up_proj(n2)` — `[seq, inter]` f32.
   n. `gu = gs * u` — elementwise multiply.
   o. `m = down_proj(gu)` — `[seq, embed]` f32 (`after_mlp` hook fires on
      `m`, pre-residual).
   p. `x2 = x1 + m` — residual add (`after_layer` hook fires on `x2`).
3. `hf = RMSNorm(x2, output_norm, eps)` — `[seq, embed]` (`before_logits`
   hook fires on `hf`).
4. `logits = head(hf)` — `[seq, vocab]` f32; tied head (llama) is the
   embedding tensor, untied head (qwen) is `output.weight` (`after_logits`
   hook fires on `logits`).

v0.3 dispatch facts carried forward unchanged:

- K-quant tensors run the compressed-resident path (`KExecution`:
  `CompressedScalar` or `CompressedX86`), decided per tensor at load by
  `--k-strategy`; Q8_0 keeps its own packed path; F32/F16 run eager.
- Q2_K/Q3_K/Q5_K/Q8_K remain eager-f32 with recorded fallback reasons.
- The v0.3 workspace-based fast decode and fused greedy head are Q8_0-only
  and do not apply to the four primary K-quant combinations; the v0.4
  planned path does not depend on them.

## 4. Semantic observation and intervention sites

Experiment-level: `on_model_loaded`, `before_prefill`,
`on_generation_complete` (fired by the CLI harness, unchanged).

Layer/execution-level (capture stages in parentheses):

- `before_layer` (stage `before-layer`) — block input `x` of layer `l`,
  shape `[seq, embed]` f32.
- `after_attention` (stage `after-attention`) — attention output `o`
  pre-residual-add, shape `[seq, embed]` f32.
- `after_mlp` (stage `after-mlp`) — MLP output `m` pre-residual-add,
  shape `[seq, embed]` f32.
- `after_layer` (stage `after-layer`) — block output `x2`, shape
  `[seq, embed]` f32.
- `before_logits` (stage `before-logits`) — final-norm output `hf`, shape
  `[seq, embed]` f32.
- `after_logits` (stage `after-logits`) — `logits`, shape `[seq, vocab]`
  f32.

These six stages are the complete set of hidden-state observation and
intervention sites. Norm outputs, Q/K/V projections, RoPE results, scores,
and MLP internals are **not** capture stages today and remain unexposed;
trace events for them carry shape/op metadata only, never values (v0.3
contract section 8, carried forward). A hook (capture or patch) targeting a
stage means that stage's tensor must exist as a real, addressable value at
the documented call site, with the exact shape above.

## 5. Tensor identity, dtype, and lifetime

Model/tensor inventory is the v0.3 inventory (v03 contract section 5),
carried forward unchanged. Per-stage tensor facts for the planned path:

- Every activation in section 3 is f32, row-major `[seq, dim]`, and lives
  only from its producing op until its last consuming op. No activation
  outlives the decode step that produced it.
- Weights: F32 norms/rope tables (small, resident f32); K-quant linears
  (mmap-resident compressed); embedding/head K-quant (mmap-resident
  compressed). Weight lifetime: whole program; the plan must not extend or
  copy them.
- KV cache: f16, flat `[layer][head][pos][head_dim]`, allocated once at
  cache creation; `qk_scratch` `[max_seq]` f32 preallocated. KV precision
  and representation are unchanged in v0.4.
- Scratch: plan-owned arena regions (section 10) with deterministic
  offsets; regions with non-overlapping lifetimes may share storage only
  when proven by the planner (documented in the arena report).

## 6. Frozen fusion set

Only these five fusions exist in v0.4. For each: the unfused sequence, the
fused sequence, eliminated intermediates, materialization under
observation, de-fusion under intervention, numerical equivalence tolerance
(Gate A numbers apply), selected kernel, and a provenance record.

F1 — RMSNorm + quantized linear projection.
- Unfused: `n = RMSNorm(x, w_n)`; `y = W·n`.
- Current concrete route: `FusedQkv` executes the norm once into the shared
  `scaled` arena region, then runs Q/K/V projections from that region. It is an
  executed fused plan op, but it does **not** yet fold per-element norm scaling
  into the matvec kernel and does not eliminate the scaled buffer. The earlier
  design text claiming that elimination was aspirational, not measured fact.
- Observation: norm outputs are not capture stages. If one is added, the
  planner must de-fuse.
- Tolerance/kernel: the ordinary resolved projection kernels after the same
  `simd::rms_norm_into`; test-only execution counters prove the concrete F1/F3
  branch runs rather than accepting output parity alone.

F2 — residual add + RMSNorm.
- Unfused: `x1 = x + o`; `n2 = RMSNorm(x1, w_n)`.
- Fused: single pass computes `n2` from `x + o` with the norm scale,
  without materializing `x1`.
- Eliminated: `x1` between attention residual and mlp norm (`x1` is not a
  hook site; `after_attention` observes `o` pre-add and `after_layer`
  observes the block output, both outside this fusion).
- Observation/de-fusion: as F1.
- Tolerance: Gate A; kernel parity-tested against `add` then `RMSNorm`.

F3 — Q/K/V projection orchestration with shared normalized input.
- Unfused: three separate `q/k/v_proj` matvec dispatches from `n1`.
- Fused: one orchestration pass over `n1` dispatching the three kernels
  from shared scratch input, with one dispatch decision instead of three.
- Eliminated: nothing — `q`, `k`, `v` are still materialized (RoPE, KV
  store, and attention consume them). Eliminated work is repeated
  dispatch and per-projection scratch bookkeeping.
- Observation/de-fusion: no stage targets `q/k/v`; if one is added, the
  planner de-fuses this layer.
- Tolerance: identical to unfused (same kernels, same input); Gate A.
- Kernel: resolved per projection.

F4 — RoPE within the planned attention path.
- Unfused: separate Q and K RoPE ops before KV storage/attention.
- Current concrete route: K remains a separate RoPE op because the roped K
  must be stored; Q RoPE (+ optional qk-norm in architecture order) executes
  inside the planned attention branch. Q and K remain materialized. Test-only
  counters prove the in-attention Q branch executes.
- Tolerance: the same architecture-specific routines and ordering as the
  reference route; Gate A.

F5 — output projection + residual add.
- Unfused: `o = o_proj(attn)`; `x1 = x + o`.
- Fused: accumulate the output projection directly into a destination seeded
  with the residual (`x1 = x + W·attn`). Dense f32 and canonical
  Q4_K/Q6_K × Q8_K kernels have explicit accumulate semantics and are eligible.
- Q8_0's legacy `matmul_q8_0_into` has assignment semantics. F5 is therefore
  automatically de-fused for a Q8_0 output projection (F2 remains active), with
  the reason serialized. This restriction fixed a latent bug previously masked
  by a hard-coded `Planned` route in the fused entry point.
- `after_attention` also forces de-fusion because `o` is that hook's real
  pre-residual tensor.
- Tests assert nonzero-destination K accumulation, actual F5 execution for an
  eligible projection, automatic Q8_0 de-fusion, and zero F5 execution when the
  hook is active; output parity alone is not the execution proof.

Cross-cutting rules:

- The plan chooses per layer `Fused | PartiallyFused | Unfused` at build
  time from the active hook set. A hook must never silently observe a
  different semantic tensor because fusion changed the graph.
- Prefer graceful de-fusion over "unsupported hook" failures; if a hook
  cannot be supported under a fusion, the planner either selects the
  unfused route during planning or fails clearly before execution — never
  mid-decode.
- No blind QKV re-packing: the three projection weights stay in their
  native GGUF tensors; F3 is orchestration only.

## 7. Hook-vs-fusion interaction

1. At plan build, the active hook set is resolved to a bitset of the six
   stages (section 4).
2. Any layer with an active `after_attention` stage or an assignment-only
   Q8_0 output projection gets `F5 = Unfused`; any layer with active
   `before_layer`/`after_layer` stages keeps those
   block-input/block-output tensors materialized (they are never
   eliminated by the fusion set, but the planner asserts it).
3. If a fusion would eliminate an intermediate that a hook targets, the
   layer is de-fused to `PartiallyFused` (other fusions kept) or
   `Unfused`; the selection and reason are recorded in the plan and in
   provenance.
4. In `Observe`/`Intervene` mode the interpreter calls the existing hook
   machinery at the same call sites, in the same order, as the reference
   path; the tensors handed to hooks are the real materialized tensors of
   the selected route.
5. Frozen equivalence: running with the hook system initialized but zero
   active hooks must be bit-identical to running with hooks fully
   disabled (Gate C).

## 8. Fallback policy

- No silent fallback anywhere. Every deviation from the requested
  `--execution` is recorded (plan field + log line + provenance).
- Unsupported architecture at plan build → hard error naming the
  architecture; never a silent generic-plan fallback.
- Unsupported fusion under an active hook → per-layer de-fusion with
  recorded reason (section 7), or clear pre-execution failure.
- Q2_K/Q3_K/Q5_K/Q8_K tensors: eager-f32, reason recorded, exactly as in
  v0.3.
- `planned-fused` on a model whose plan contains an unsupported tensor →
  the plan records the fallback and the run proceeds only if the fallback
  is within the documented contract; otherwise hard error.
- Missing metadata (e.g. absent rope tables, absent qk-norm tensors) →
  resolved by the same rules as v0.3 model construction; plan build fails
  with a named cause, not a silent substitution.
- Any fallback is visible in `execution-plan.json`, `--execution` run
  logs, and artifact provenance.

## 9. Execution concepts

- **reference execution** (`--execution reference`): the v0.3 generic
  hooked path with per-tensor dynamic dispatch (section 3). This is the
  readable oracle for parity gates and the default during development.
  Kernels are the same scalar/AVX2 implementations; dispatch is dynamic.
- **planned execution** (`--execution planned`): the identical operation
  sequence driven by the immutable plan — resolved kernel per tensor,
  scratch-region destinations, no per-token shape/dispatch rediscovery,
  no fusion. Must match reference within Gate A/B numbers.
- **fused planned execution** (`--execution planned-fused`): the plan with
  the frozen fusion set applied per the active hook set (sections 6-7).
  Must match reference within Gate A/B numbers and satisfy Gate C.

The three concepts remain separable for validation; the CLI names them
explicitly. `--execution` composes with `--k-strategy` (K-quant residency
decisions are load-time and orthogonal to execution planning).

## 10. Execution plan schema

`ExecutionPlan` (serde-serializable, deterministic):

```text
schema_version: 1            # "v04-plan/1"
kernel_revision: 2           # numerical/runtime ABI; legacy missing field = 1
architecture: string         # "llama" | "qwen2" | "qwen3" (scope: llama, qwen2)
model_sha256, tokenizer_sha256
gguf_metadata: { arch, block_count, embedding_length, head_count,
                 head_count_kv, ffn_dim, vocab_size, rope_dimension_count,
                 context_length, file_meta }
rope: { layout: "adjacent-pair"|"split-half",
        qk_norm_order: "before-rope"|"after-rope", has_q_norm, has_k_norm }
layers: [ LayerPlan ]        # one per transformer block
tensor_table: [ TensorRecord ]
scratch: ScratchPlan
kv: { precision: "f16", layout: "layer-head-pos-dim",
      layer_stride, head_stride, pos_stride, head_dim, n_kv_heads, max_seq }
hook_sites: HookSitePlan     # stage -> { mode, tensor, layer, materialized }
dispatch: DispatchPlan
cpu: { features: [...], threads, required: [...] }
provenance: PlanProvenance
```

`LayerPlan` = `{ layer_index, ops: [PlannedOp], fusion: Fused|PartiallyFused|Unfused, fusion_reason? }`.

`PlannedOp` (one enum, interpreted by the llama-family decode loop):
`Embedding`, `RmsNorm`, `Matvec` (weight ref, in/out regions, kernel id,
optional fused rms-norm weight), `Rope` (layout, qk-norm ref + order),
`KvStore`, `Attention` (spec + score scratch region), `Silu`, `Elemul`,
`ResidualAdd`, `OutputNorm`, `Logits` (weight ref, tied flag). Fused ops
carry `FusedOp { kind: F1..F5, components: [op ids], eliminated: [tensor ids] }`.

`TensorRecord` = `{ id, name, shape, gguf_dtype, execution, kernel,
resident_bytes, mmap: bool }` — derived from the v0.3 inventory; stable
`id` is the plan's `TensorRef`.

`ScratchPlan` = `{ total_bytes, alignment, regions: [ { name, offset,
size, alignment, first_op, last_op, shared_with? } ], arena_report }`.
Regions share storage only when the planner proves disjoint lifetimes;
the arena report lists every overlap and its proof.

`HookSitePlan` = `{ mode: Disabled|Observe|Intervene, active: [stage ids],
sites: [ { stage, tensor id or "fused-eliminated", layer?,
materialized: bool, route: fused|unfused } ] }`.

`DispatchPlan` = `{ kernel_per_tensor: [ { tensor id, kernel, cpu feature
requirement, fallback? } ], thread_strategy }`.

`PlanProvenance` = `{ ember_version, git_commit (build env
EMBER_GIT_COMMIT when set), rustc_version (from build.rs), plan_build_time
(iso8601, the only nondeterministic field), execution_mode, hook_mode }`.

Serialization: `artifacts/benchmark-v04/<run>/execution-plan.json`;
`ember inspect-plan` prints the same structure as text. Plan hashing: a
SHA-256 over the serialized plan with the timestamp zeroed, recorded in
provenance and run artifacts.

## 11. Scratch arena and allocation contract

- One arena per decode session, allocated once before decode begins,
  sized by `ScratchPlan.total_bytes`, aligned to 64 bytes (AVX2-safe;
  regions needing 256-byte alignment get it via offset rounding).
- Plan arithmetic performs zero calling-thread heap allocations after warmup.
  The returned logits `CpuTensor` still performs three attributable allocations
  (shape, strides, data). K-quant execution additionally owns a TLS Q8_K pack
  buffer outside the arena: first/larger/nested calls may allocate it; a warmed
  same-thread same-or-smaller call allocates zero times. Its exact formula and
  lifecycle are in `docs/v03-execution-contracts.md` section 7.
- Debug builds validate region bounds and detect illegal aliasing where
  practical (offset/size checks per op; region overlap asserted against
  the planner's proof).
- Capture and patch operations copy into owned artifacts; they never
  retain references into mutable scratch after the step completes
  (existing artifact contract).
- A counting global allocator (one relaxed atomic per allocation) records
  per-run allocation events. Gate E separately asserts zero warmed K-matmul
  workspace allocations and the documented logits/all-scheduler envelope for
  a full decode; it does not label output materialization as arena scratch.
- Diagnostic mode reports: total scratch bytes, region names, offsets,
  alignments, maximum live interval, and whether any decode-time
  allocation occurred.

## 12. Hook modes

```text
HookMode::Disabled   — fast normal path; no capture metadata, no clones,
                       no string lookup, no registry inspection, no trace
                       serialization, no unpredictable inner-loop branches.
HookMode::Observe    — existing capture semantics unchanged.
HookMode::Intervene  — existing patch semantics unchanged.
```

Active hook sites resolve to compact IDs/bitsets at plan build. The
interpreter's hot loop reads the resolved bitset, not dynamic registries.
Frozen requirement: zero active hooks under an initialized hook system is
bit-identical to hooks disabled (Gate C, asserted in tests).

Overhead is benchmarked separately for: hooks disabled; framework active
with no registered hooks; one hidden-state capture; one intervention
(section 14).

## 13. Correctness gates (frozen)

Gate A — kernel/operation parity (unit + integration tests, deterministic
seeds): planned and fused outputs vs the independent reference path
(`dequant_tensor` → f32 → `CpuTensor::matmul` for kernels; the unfused op
sequence for fusions). `max_abs <= 1e-4 * max(1, max_abs_ref)` over the
full output; shapes rows in {1,2,8,32} x in {256,512,1536,2048,8960} x out
{128,512,2048}; edge blocks, zero scale, negative min, all-zero quants,
nibble saturation; both dtypes; the v0.3 matvec gates are preserved and not
weakened. Planned-dispatch vs legacy-dispatch kernel equivalence is
asserted per supported tensor.

Gate B — model parity, all four combinations, three paths
(reference/planned/planned-fused): per capture-stage tensor
`max_abs <= 5e-4 * max(1, max_abs_ref)` and cosine >= 1 - 1e-6; final
logits `max_abs <= 1e-2` (llama family) / `2e-2` (qwen family, v0.3
amendment carried forward); greedy token sequences identical (100%) over
the frozen prompt set: >= 4 prompts per model including short, long,
ASCII, and Arabic prompts, plus boundary tokenization cases (canonical
English prompts + smoke set + >= 3 Arabic morphology prompts per family).
Top-1 agreement is the primary functional gate; logit/cosine envelopes are
recorded. A token flip is a failure to investigate, not a threshold to
relax.

Amendment 2026-08-11: for compressed Q4_K/Q6_K weights, all three execution
modes share the canonical Q8_K activation primitive. The matrix above is the
pre-registered v0.4 target, not a claim that every listed dimension is in one
unit test. Current attributable coverage is: full API rows
`{1,2,3,4,5,7}`, inputs `{256,512}`, outputs `{1,3,17}`, both dtypes and
scalar/x86 tiers; a forced two-thread case above the 2M-MAC threshold; extrema,
zero scale, finite subnormal, non-finite rejection, accumulation, x4 and tail
rows; a pinned independent llama.cpp known answer; and the fail-closed 1B model
script. The tight route-parity bounds apply when reference/planned/fused execute
the same Q8_K primitive; they do not require production to reproduce the
separate exact-f32 dequantize-and-dot oracle. Historical benchmark/golden
numbers are not relabeled as current evidence unless their artifact records
kernel revision 2 and actual dispatch.

Gate C — hook semantics, every supported site: inactive hooks bit-identical
to disabled; captures match between reference and planned paths (same
shape/indexing, same values within Gate B envelope); interventions occur
at the same tensor and layer; exact-restoration tests pass; fused
execution de-fuses correctly when a hook requires it; provenance records
the actual selected route.

Gate D — memory: no material regression of v0.3 compressed residency.
Llama-3.2-1B Q4_K_M: no more than 10% peak-RSS regression relative to v0.3
under the same benchmark; scratch allocation reported separately; any
increase explained by the named reusable arena. Packed Q4_K/Q6_K weights
remain mmap-resident (asserted via residency checks).

Gate E — allocation: after warmup, the normal planned decode loop with
hooks disabled performs zero heap allocations per token (counting
allocator, section 11). If absolute zero is blocked by a documented
dependency, the remaining allocation is isolated and quantified, then
removed before release unless clearly impossible.

Gate F — performance: median planned-fused decode throughput >= 1.75x the
v0.3 baseline (same model, same quant, same benchmark protocol) on at
least three of the four primary combinations; no supported model regresses
by more than 5%; Q8_0 does not regress materially; scalar and AVX2 results
reported separately. Release target remains ~2x v0.3; Gate F is the floor.
This gate is not amended after seeing final results unless profiling
demonstrates the target rests on a false assumption; any amendment must be
documented with measurements and committed before final benchmarking.

Gate G — external parity: same pinned llama.cpp revision and golden-ladder
strategy as v0.3 (scripts/validate_golden_ladder.sh); the final optimized
path preserves 100% greedy top-1 agreement on the frozen ladder and the
v0.3 per-family logit/cosine envelopes (llama max 1.0 / mean 0.2 / cosine
0.998; qwen max 2.0 / mean 0.3 / cosine 0.995); exact model and tokenizer
provenance recorded.

## 14. Benchmark protocol and hardware metadata

Arms: v0.3 baseline commit (871a8eab), v0.4 reference, v0.4 planned, v0.4
planned-fused, pinned llama.cpp (same revision and CLI/bench build as the
v0.3 ladder). Per model/quant: median decode tokens/s; median prefill
tokens/s; peak RSS; model mapping size; scratch size; KV-cache size;
startup/load time; first-token latency; steady-state per-token latency;
binary size; output token sequence; active CPU features; thread count;
compiler and Rust versions.

Protocol: 1 warmup; 5 measured repetitions where runtime permits; 64-token
greedy decode; fixed prompts; fixed seeds where relevant; no sampling;
identical context and token limits across arms. Raw measurements
preserved; deterministic machine-readable summaries.

Profiling (Phase 8): time per operation category — quantized matvec,
RMSNorm, RoPE, attention, softmax, KV writes, allocation, hook checks,
dispatch, thread synchronization — via the existing `decode_profile`
instrumentation extended to the planned path, fully disabled in release
benchmarks except where a category is the measured subject.

Hardware metadata required in every benchmark summary: CPU model and
microarchitecture, core/thread count, AVX2/FMA/F16C/SSSE3 presence, RAM, OS and
kernel, rustc and cargo versions, llama.cpp pinned commit, Ember commit,
thread count used.

Hook overhead arms (section 12) report: disabled; framework active with
zero registered hooks; one hidden-state capture; one intervention.

## 15. Provenance fields

Extend the existing schema (additive fields with serde defaults), never a
parallel system. Record per run: execution mode; execution-plan schema
version; plan hash; operation-graph hash where practical; kernel chosen
per operation; fusion state per operation; scratch size; CPU features;
thread count; hook mode; active hook sites; fallback reasons; model SHA;
tokenizer SHA; Ember commit; Rust compiler version; benchmark command;
llama.cpp reference commit. Artifacts stay deterministic apart from
explicitly timestamped metadata.

## 16. CLI and diagnostics

```bash
ember run model.gguf --execution reference
ember run model.gguf --execution planned
ember run model.gguf --execution planned-fused
ember inspect-plan model.gguf            # or via an existing inspect path
```

`inspect-plan` output: layer count; operation count; selected kernels;
fused operations; de-fused operations (with reasons); scratch bytes; hook
mode; CPU feature requirements; unsupported/fallback operations; quant-
format inventory. No silent fallback anywhere.

## 17. Commit map and version policy

1. this document (+ docs whitelist entry);
2. execution plan types and deterministic diagnostics (`ember inspect-plan`);
3. architecture-specific plan construction;
4. planned tensor and kernel dispatch (`--execution reference|planned`);
5. aligned reusable scratch arena;
6. zero-allocation planned decode path;
7. compact hook-site planning and HookMode (Disabled/Observe/Intervene);
8. zero-cost inactive hook path;
9. planned single-token attention + parity tests;
10. frozen fusions (F1-F5) and forced de-fusion;
11. hook capture/intervention parity tests;
12. external golden ladder + benchmark harness;
13. benchmark and profiling artifacts;
14. docs and release notes;
15. release v0.4.0 (crate version bump, all gates green).

Every commit keeps `cargo fmt -- --check`, `cargo clippy --all-targets
--all-features -- -D warnings`, `cargo test --all-targets`, and the Python
tests green. No methodology changes or unrelated cleanup inside
implementation commits. Development evidence is not squashed unless a
commit is genuinely broken or contains generated noise. Benchmark-generated
junk and model files are never committed; evidence artifacts live under
`artifacts/benchmark-v04/`.

## 18. Stop conditions

Stop and report before proceeding if: a proposed fusion cannot preserve
semantic hook identity; plan construction requires unstable tensor
lifetimes; the optimized path changes greedy output; memory regresses
because packed weights are expanded; performance gains come primarily from
disabling research functionality; the reference oracle is no longer
independent enough to validate the optimized path; a gate would need to be
weakened after observing final results; an unsupported architecture would
silently fall back; provenance cannot distinguish fused from unfused
execution.

## 19. Final framing

Ember v0.4 precomputes and reuses an inspectable execution plan, removes
steady-state decode allocations, and introduces bounded fusion while
preserving exact research hook semantics. It materially improves Ember's
own decode path; llama.cpp remains the external performance reference. No
competitive parity with llama.cpp is claimed unless the evidence
unexpectedly demonstrates it.
