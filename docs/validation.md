# Ember validation

Validation ladder, evidence status, and the 2026-08 pilot wave.
Moved from the top-level README.

> **Support vs trust.** "Supported" in the model tables means an execution path
> exists. Several rows still have `pending` golden-logit or activation-reference
> status, so "supported" does not imply numerically trustworthy for every
> architecture. The causal-localization result below is validated on the rows
> with completed golden checks (qwen3/llama families); do not cite it as
> validated across all architectures.

## validation ladder

Use these levels when interpreting Ember runs:

1. **smoke**: structural execution only. The command ran, loaded artifacts, and
   produced output. This is not numerical validation or output-quality evidence.
2. **golden logits**: output-logit comparison against a trusted reference for
   the same model, tokenizer, prompt, and quantization path.
3. **activation reference checks**: internal hidden-state comparison against a
   trusted implementation. This is required before treating layer geometry as
   numerically validated.
4. **probes**: linear or MLP classifiers over cached hidden states. These show
   decodability or recoverability, not causal use.
5. **interventions**: causal tests that first verify a probe-score drop after
   removing or perturbing a direction, then measure logit or generation effects.


## current evidence status

| architecture | smoke | golden logits | activation reference | probe runs | status |
|--------------|-------|---------------|----------------------|------------|--------|
| gpt-2 | structural smoke works when local GGUF is present | none | none | not a standard Arabic morphology run yet | loader baseline; negative-control work pending |
| llama | local/cloud structural smokes and probe extraction | v0.3 ladder golden vs pinned llama.cpp (top-1 match 100%, max abs diff 0.81, cosine ≥ 0.9989; see section below) | pending | preliminary LLaMA 1B/3B/8B decoder probe runs | research findings are preliminary until references and reports are complete |
| qwen2.5 | selected warning-prone smokes through llama-family path | v0.3 ladder golden vs pinned llama.cpp (top-1 match 100%, max abs diff 1.74, cosine ≥ 0.9963; see section below) | pending | pending validation | experimental; do not treat as quality-compatible |
| qwen3 | Qwen3 0.6B smoke/probe paths run locally | pending target | pending | Qwen3 0.6B local probe run exists | promising engineering path, not yet numerically validated |
| gemma4 | local BOS smoke and llama.cpp reference comparison run | final-logit cosine ~0.87; not a golden pass | per-layer comparison pipeline operational; L0 attn_norm bit-identical | pending full runs | structural fixes applied, but the remaining numerical gap still prevents a parity claim; RMSNorm amplification is the current working explanation |
| hf encoders | external Hugging Face extraction path works for mBERT smoke | not applicable to Ember GGUF numerics | external stack not activation-checked here | mBERT PADT smoke; full encoder suite pending | useful benchmark path, not an Ember inference validation result |


## gemma 4 status note (2026-08)

The gemma4 loader's f32/f16 orientation bug was fixed in the 2026-08
optimization pass (commit `bd1591c`): gemma 4 Q8 models now **load**,
which they previously could not. A later debugging pass (same date) added
three reference-derived fixes — attention scale 1.0 (gemma4 uses no
pre-attn scaling), per-head RMS norm on V before caching, and ggml-exact
rope factor division semantics — which moved single-token logits from
uncorrelated to cosine ~0.86 vs llama.cpp, but the model still does not
match (multi-token cosine ~0.45 and worse). Layer-by-layer comparison
against llama.cpp localizes the remaining divergence to accumulation
across the network rather than any single confirmed defect: early
layers agree to within numerical noise, and cosine erodes toward zero
by mid-network as per-layer fp differences (FMA contraction, matmul
accumulation order, norm epsilon) compound through the residual stream
over 35 layers. An apparent embedding-level "permutation" did not
survive exact-tolerance comparison (loose 1e-3 matches collapsed to
noise at 1e-6), so no layout defect is asserted. The rope
layout that matches the bundled llama-cpp-python reference is split-half;
master's LLAMA_ROPE_TYPE_NONE (adjacent-pair) does **not** match it.
The older tiny-gemma golden report under `artifacts/golden_logits_gemma/`
**predates the fix** and is stale — it can no longer be regenerated (the
tiny reference model is not on disk). Treat gemma 4 outputs as
numerically untrusted until a fresh golden run exists.

## recent validation wave

A validation + bounded pilot wave (2026-08) added K-quant support and ran an
Arabic-quantization pilot (Qwen2.5-1.5B and Llama-3.2-1B across Q8/Q6/Q4;
~500 deterministic runs). The full record lives in PILOT_REPORT.md in the
pilot directory (local branch, not published); the summary below is the
public record:

- **No Arabic-selective quantization degradation replicates** — a null at
  every precision/family combination tested.
- The robust output is the **causal-localization toolchain**: single-layer
  activation patches restore quantized-boundary failures, with the causal
  locus one layer before the divergence ramp (qwen L7/28, llama L1/16),
  demonstrated end-to-end by the pilot's `causal_demo.sh`. The mechanism is
  near-threshold flips: quantization noise crosses the model's smallest
  decision margins.
- Deployment: see the K-quant note above — dequant-to-f32 is a research
  loader, not a deployment path.

The v0.2 capture/patch/compare facilities that powered this are documented
in the research-experiments section below.

## v0.3 compressed-resident K-quant validation (2026-08)

v0.3 adds native Q4_K/Q6_K execution: weights stay packed and resident
(mmap-backed), dequantizing at super-block granularity inside scalar or
AVX2 kernels. The full record is in `docs/v03-execution-contracts.md`
(frozen gates, per-tensor fallback semantics, provenance fields) and the
scripts it names; this section is the evidence summary.

- **Gate A (kernel)** — scalar and AVX2 Q4_K/Q6_K kernels vs the
  eager-f32 dequant-then-gemm oracle: max_abs ≤ 1e-4·scale across the
  standard shape battery, zero-scale/min edges, extreme scale and
  saturated-nibble blocks. AVX2 vs scalar within tolerance and
  deterministic.
- **Gate B (model parity)** — compressed vs eager-f32 on the fresh
  ladder, both families × Q6_K/Q4_K_M: per-layer max_abs ≤ 5e-4·scale,
  cosine ≥ 1−1e-6, logits ≤ 1e-2 (llama) / 2e-2 (qwen, amended),
  greedy token sequences identical across 6 frozen prompts per rung.
  Inactive hooks (no-op experiment through ActiveHooks) leave outputs
  bit-identical.
- **Gate C (golden)** — final-position logits vs the pinned llama.cpp
  b9999 build (tools/logits_dump.c harness; the pinned CLI has no
  logit dump and llama-cpp-python 0.3.27 is broken for logits on
  python 3.14). All six ladder rungs: top-1 agreement 100%. Envelope
  (max/mean/cosine): llama q8/q6/q4 = 0.59/0.087/0.9995,
  0.81/0.131/0.9989, 0.65/0.105/0.9992; qwen q8/q6/q4 =
  0.82/0.141/0.9991, 1.36/0.235/0.9975, 1.74/0.248/0.9963.
- **Causal workflow** — capture → compare → patch → frozen verdict on
  the compressed path (scripts/validate_k_causal.sh): intervention
  flips 8/8 records; the patch restores every captured tensor
  bit-exactly.
- **Benchmarks** (artifacts/benchmark-v03/) — AVX2 compressed decode
  beats eager-f32 2–6× on K rungs (llama q6 2.3 vs 0.8 tps; qwen q4
  1.7 vs 0.9 tps) at ~1 GB resident vs 4–7 GB eager; scalar kernels
  are ~0.3–0.4 tps (reference only). Pinned llama-bench peak RSS
  1.1–1.9 GB on the same files.
- **Known limitations** — the extraction tokenization path has a
  pre-existing byte-offset limitation with non-ASCII prompts (the
  golden ladder uses English prompts); qwen2.5 GGUFs from the local
  fp16 source omit the family vocab_size metadata key, now handled by
  a loader fallback to the embedding row count; the gemma4 numerical
  gap is unchanged and out of v0.3 scope.

## v0.4 execution-planning validation (2026-08-04)

Contract: `docs/v04-execution-contract.md` (frozen before implementation).
Execution concepts: `reference` (the v0.3 generic hooked path, the oracle),
`planned` (plan-driven interpreter), `planned-fused` (frozen fusion set
F1-F5 with hook-driven defusion).

- **Gate A (kernel)** — the column-parallel decode matvec is bit-identical
  to the serial kernels (same per-column accumulation order): verified
  across gate/up/down/head shapes, both dtypes (Q4_K/Q6_K), and both
  execution paths (scalar + AVX2). The planned dispatch kernel equals the
  reference dynamic dispatch (debug assert + tests).
- **Gate B (model parity)** — reference/planned/planned-fused greedy token
  sequences identical on the six frozen prompts (English + Arabic) with
  per-step logits within the frozen envelopes, on all four primary
  combinations: Llama-3.2-1B and Qwen2.5-1.5B × Q4_K_M/Q6_K
  (`tests/k_parity.rs`, env-gated).
- **Gate C (hooks)** — the six semantic sites fire at the same call sites
  with the same stages/layers/shapes on the planned path; inactive hooks
  are bit-identical to disabled (synthetic + real model); a
  zero-layer-output intervention lands identically; planned-fused defuses
  F5 when after_attention is active so the hook sees the materialized o
  tensor.
- **Gate D (memory)** — packed K-quant weights stay mmap-resident on the
  planned path (no eager expansion); the scratch arena is a named, reusable
  allocation reported by `inspect-plan` and the arena report.
- **Gate E (allocation)** — after warmup, the planned decode loop performs
  zero heap allocations per token other than the documented logits tensor
  materialization (3: shape + strides + data), verified on the real model
  (counting allocator); the column-parallel matvec allocates nothing on a
  warm rayon pool.
- **Gate F (performance)** — see the benchmark matrix below; the final
  numbers are collected under the documented protocol
  (`artifacts/benchmark-v04/`).
- **Gate G (external)** — the v0.3 golden ladder remains the external
  reference; the planned/fused paths reproduce reference greedy outputs
  within the frozen envelopes, so the golden-logit agreement carries over
  by transitivity (ladder re-run for the release artifacts).
