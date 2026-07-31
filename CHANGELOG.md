# Changelog

All notable changes to Ember are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Ember's Rust experiment API is explicitly unstable during the 0.1 series.

## [0.1.0] - 2026-07-31

### Added

- CPU inference for GPT-2, LLaMA-family, Qwen3, and dense text-only Gemma 4
  GGUF models.
- GGUF v3 loading for F32, F16, BF16, and Q8_0 tensors.
- mmap-backed Q8_0 weights, SIMD decode kernels, tiled prefill kernels, and
  measured packed gate/up dispatch for short Gemma batches.
- KV-cached generation, deterministic greedy decoding, top-k/top-p sampling,
  structured operation tracing, hidden-state extraction, and benchmark JSON.
- An intentionally unstable static Rust experiment API with exactly two
  built-ins:
  - `activation-stats`, an observation-only JSON recorder for activation
    norms and fingerprints;
  - `zero-layer-output`, an example attention/MLP/layer intervention.
- Real-shape Q8 matmul and model-only single-token decode benchmark harnesses.
- Numerical parity, trace-fingerprint, hidden-state, allocation, and
  controlled A/B regression tests for supported execution paths.

### Changed

- Removed stale scheduler, SIMD, backend, and Gemma workspace experiments that
  did not protect supported behavior or demonstrate a measured benefit.
- Kept Qwen split-half RoPE decode on the generic correctness path after a
  real-model fast-path experiment diverged.
- Preserved generic Q8 paths as inspectable correctness oracles alongside
  packed dispatch.

### Known limitations

- The experiment API is not a stable semver commitment.
- Experiments are built in statically; dynamic plugins, registries, multiple
  concurrent experiments, Python hooks, and WASM are unsupported.
- Active experiments do not participate in hidden-state extraction, probes,
  logits/layer dumps, demos, interactive mode, or benchmark subcommands.
- Qwen2.5 support remains experimental.
- Gemma 4 support is dense text-only; multimodal, MoE, drafter, and
  K-quantized models are unsupported.
- llama.cpp remains the external performance reference. Ember does not claim
  comparable breadth or throughput.

[0.1.0]: https://github.com/voidwest/ember/releases/tag/v0.1.0
