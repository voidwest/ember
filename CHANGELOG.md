# Changelog

All notable changes to Ember are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Ember's Rust experiment API is explicitly unstable during the 0.1 series;
the v0.2 activation-artifact schema (`0.2.0-experimental`) is versioned but
carries no compatibility guarantee.

## [Unreleased]

### Added

- Repo-wide hardening and provenance pass (recovered from the interrupted
  audit branch; split into 16 reviewable commits):
  - Extraction artifact contracts: `checksums.json` verification, prompt-hash
    recomputation, duplicate-ID/order/offset checks, tokenizer SHA-256
    provenance, and staging-dir runs that never overwrite an existing
    artifact.
  - Prompt-leakage audit and enforcement: label-revealed prompts are rejected
    by default; surface-only probe prompts (`en_surface_probe`) are the new
    default, with label-revealed variants as explicitly named positive
    controls (`--allow-label-revealed-prompts`).
  - Atomic file publication for npy/json/jsonl/plot outputs.
  - `--arch auto` GGUF architecture detection (replaces the silent gpt2
    default fallback; conflicts with an explicit `--arch` are hard errors).
  - New stimulus fixture `stimuli/nonce_root_pattern_surface.json` with
    audited prompt contracts and a provenance sidecar.
  - `scripts/download_models.sh` for fetching the gitignored GGUF models used
    by docs and benchmark fixtures.

### Changed

- Fail-closed validation across the CLI, loader, extraction, probes, and
  dataset pipeline: strict JSON (`parse_constant` rejection), non-finite
  rejection, `allow_pickle=False`, tokenizer/model vocabulary checks, and
  run-directory no-overwrite semantics.
- Probe analysis methodology corrections: held-out CCA, 5-fold MDL-style
  data-efficiency curves, matched logistic shuffled controls, per-fold
  character-ngram baselines with fold-local vectorizer fitting, and a
  per-fold majority baseline. Reruns are not numerically comparable to
  pre-2026-08 artifacts.
- llama.cpp extraction adapters now require `prompt_final` and refuse
  unverifiable word-position extraction.
- CI pins Rust 1.92.0 with `--locked`; Python gates cover probes workflow
  tests and docs determinism (`scripts/check_docs.py --check`).
- Dependencies: `sha2` (Rust), `scipy` (Python, previously imported without
  being declared).
- Benchmark fixtures migrate to the surface-only probe template; Arabic-UD
  fixtures disable activation reuse.

### Fixed

- `stimuli/generate_stimuli.py apply_pattern`: chained `str.replace` could
  corrupt inserted radicals containing `f`/`l`/`3`. Latent — no tracked
  stimulus was affected (verified against all 200 tracked surfaces).
- `tools/dump_llamacpp_layers.cpp`: removed the silent final-hidden-state
  fallback for unpatched llama.cpp builds; an unpatched build is now a hard
  failure.
- Word-position token selection fails on ambiguous spans and converts byte
  to character offsets (multi-byte alignment fix).
- KV-cache layout for layers with fewer KV heads (Gemma 4).
- Documentation: stale crossover gate claim corrected, label-revealed
  caveats stamped on research-note pages, Gemma-4 parity reports marked
  pre-orientation-fix/untrusted.

### Removed

- Final-hidden-state fallback path in the layer-dump tool (replaced by the
  hard-failure contract above).

## [0.2.0] - 2026-08-01

### Added

- Selective activation capture (`--capture-activations capture.toml`): a
  run-level recorder that copies live tensors only for explicitly selected
  records at the six semantic hook stages (before-layer, after-attention,
  after-mlp, after-layer, before-logits, after-logits), filtered by layer,
  phase, and decode position, with an optional record cap. Captures run
  after the experiment, so records reflect post-intervention values.
- Experimental activation artifacts (`0.2.0-experimental` schema): manifest
  + little-endian f32 npy tensors with deterministic naming, tensor hashes,
  model/tokenizer/GGUF provenance, prompt hash with optional prompt
  omission, input and generated token IDs, per-evaluation dispatch path
  observations, and the exact capture-config hash.
- `activation-patch` experiment: replaces one live activation in place from
  a captured artifact; targets resolve to exactly one source record
  (position-qualified or unique match, hard error otherwise); validates
  family, layer, width, dtype, and byte order; fails clearly when a target
  is never applied; zero allocation inside the hook after initialization.
- `compare-artifacts` subcommand: strict record alignment on
  (phase, layer, stage, start position) with hard refusal on duplicates;
  per-record bit-exact equality, max/mean/RMS diff, cosine, L2 norms,
  relative L2 error, shape/dtype match; missing/extra record reporting;
  deterministic JSON output; only `created_at_unix` is ignored.
- Per-evaluation dispatch recording (fast/workspace vs generic) surfaced in
  capture manifests as phase-specific observations.
- `GenerationContext` now carries input and generated token IDs.
- Research example workflow: `scripts/research_example_capture_patch.sh`
  (capture -> intervene -> compare -> patch -> restore) with a frozen
  restoration criterion: patched-run logits must be bit-identical to the
  baseline's.

### Changed

- `ExperimentRunner` now holds an optional experiment and an optional
  capture sink; the single-experiment invariant is unchanged.
- CLI: `--capture-activations`, `--activation-patch`,
  `--patch-target` (repeatable), and the `compare-artifacts` subcommand;
  capture and patching participate only in the generation path.

### Unchanged

- No-experiment parity: outputs, tracing, layer dumps, hidden-state
  extraction, and kernel selection are untouched when no experiment or
  capture is active; disabled per-layer hook calls still compile away.

### Known limitations

- `token_positions` filters decode steps only; prefill records are
  whole-sequence.
- One target layer/stage per patch experiment (multiple phase/position
  targets allowed).
- Capture and patching do not participate in probe/extract/dump modes.
- The v0.2 artifact schema is experimental and may change without notice.

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
