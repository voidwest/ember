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
| llama | local/cloud structural smokes and probe extraction | q8/q6/q4 golden vs llama.cpp (top-1 match, max abs diff ≤ 0.33) | pending | preliminary LLaMA 1B/3B/8B decoder probe runs | research findings are preliminary until references and reports are complete |
| qwen2.5 | selected warning-prone smokes through llama-family path | q8/fp16/q4 golden vs llama.cpp (top-1 match, max abs diff ≤ 0.72) | pending | pending validation | experimental; do not treat as quality-compatible |
| qwen3 | Qwen3 0.6B smoke/probe paths run locally | pending target | pending | Qwen3 0.6B local probe run exists | promising engineering path, not yet numerically validated |
| gemma4 | local BOS smoke and llama.cpp reference comparison run | final-logit cosine ~0.87; not a golden pass | per-layer comparison pipeline operational; L0 attn_norm bit-identical | pending full runs | structural fixes applied, but the remaining numerical gap still prevents a parity claim; RMSNorm amplification is the current working explanation |
| hf encoders | external Hugging Face extraction path works for mBERT smoke | not applicable to Ember GGUF numerics | external stack not activation-checked here | mBERT PADT smoke; full encoder suite pending | useful benchmark path, not an Ember inference validation result |


## recent validation wave

A validation + bounded pilot wave (2026-08) added K-quant support and ran an
Arabic-quantization pilot (Qwen2.5-1.5B and Llama-3.2-1B across Q8/Q6/Q4;
~500 deterministic runs). Items, results, and reports live on the local
`pilot-001` branch under `research/pilots/arabic_quantization_001/`
(PILOT_REPORT.md has the full record):

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
