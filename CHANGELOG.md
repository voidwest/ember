# Changelog

All notable changes to Ember are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Ember's Rust experiment API is explicitly unstable during the 0.1 series;
the v0.2 activation-artifact schema (`0.2.0-experimental`) is versioned but
carries no compatibility guarantee.

## [0.5.0] - 2026-08-04

### Added: reproducible experiment workflow

The v0.5 release packages exact token selection, semantic hidden-state
capture, activation intervention, execution provenance, and offline
verification into deterministic, self-verifying experiment bundles that
can be run without writing Rust:

- `ember.experiment.v1` specification language (strict TOML, recorded
  defaults, fail-closed validation with exact field paths);
- six public semantic hook sites with a frozen machine-readable
  descriptor table (`ember.hook.v1`), mapped onto the v0.4 hook stages;
- typed, byte-exact token selection (prompt-final, absolute/relative,
  generated-step, matched-span with occurrence and subtoken selection,
  byte-span), including Arabic alignment with combining marks and
  punctuation (the `tokenizers` crate offsets are byte offsets; the
  wrapper previously validated them against character counts, which
  broke all non-ASCII prompts — fixed);
- capture plans (selected rows / full tensor / summary-only, f32/f16)
  with owned payloads in a strict safetensors codec;
- intervention plans (replace, zero, scale, interpolate, add-delta,
  restore-original) with fail-closed source validation and automatic
  de-fusion when a fused plan would eliminate a requested tensor;
- deterministic `ember.bundle.v1` bundles: semantic manifest with
  run-invariant semantic/payload hashes, sanitized execution plans,
  atomic staging and publish, never-overwrite policy;
- fully offline `ember experiment verify` (15 basic checks; optional
  deep model/tokenizer checks);
- `ember experiment compare` with scientific-first reporting and
  deterministic `--json`;
- `ember experiment reproduce` with classification
  (exact-semantic / exact / output-equivalent / top1-equivalent /
  failed);
- `ember experiment tokenize` for tokenizer and span-matching
  diagnostics;
- deterministic seeded sampling plumbing (fixed seed + temperature
  sampling);
- reference morphology workflow under `examples/experiments/` with an
  Arabic prompt: layerwise capture, layer-8 intervention, exact
  restoration (bit-identical baseline), verified and reproducible
  end-to-end on Llama-3.2-1B-Instruct-Q8_0.

Performance isolation: ordinary inference is untouched — no experiment
spec is parsed, no bundle metadata allocated, and no hooks fire unless
the `experiment` subcommand runs.

### Compatibility

- experiment specification schema, bundle schema, semantic hook schema,
  and execution-plan schema are versioned independently;
- unknown schema majors fail; v1 is stable within Ember 0.5.x;
- v0.1/v0.2 experiment interfaces (`--activation-stats`,
  `--zero-layer-output`, `--capture-activations`, `--activation-patch`,
  `compare-artifacts`) keep their v0.2 semantics unchanged.

## [0.4.0] - 2026-08-04

### Added

- **Execution planning** (frozen contract in
  `docs/v04-execution-contract.md`): an immutable per-model `ExecutionPlan`
  built once after load — architecture-specific operation sequence, resolved
  per-tensor kernel dispatch, scratch arena layout, KV layout, hook-site
  resolution, and provenance (model/tokenizer hashes, git commit, rustc,
  deterministic plan hash). `ember inspect-plan` prints it and writes
  `execution-plan.json` with `--output`.
- **Plan-driven decode interpreter** (`--execution planned`): the same
  operation sequence as the reference path, walking resolved ops against a
  reusable aligned scratch arena — no per-token shape/dispatch rediscovery,
  no per-token heap allocation (Gate E: after warmup, only the logits tensor
  materialization allocates, 3 documented allocations per token).
- **Frozen fusion set** (`--execution planned-fused`, F1-F5): fused QKV
  orchestration with a single norm pass, Q rope inside attention, output
  projection accumulating into the residual destination, and
  residual+RMSNorm with a 3-operand final add; fusions that would eliminate
  a hooked tensor are defused per layer with recorded reasons. Hook modes
  (`Disabled`/`Observe`/`Intervene`) ride on the plan; the six semantic hook
  sites fire at the same call sites as the reference path, and inactive
  hooks are bit-identical to disabled.
- **Column-parallel K-quant matvec**: large single-row decode matvecs split
  their output dimension across the rayon pool; each column accumulates
  identically to the serial kernel (Gate A bit-identity test across
  gate/up/down/head shapes, both dtypes, both execution paths).
- **Planned-path operator profiling**: `bench-decode --profile-operators`
  records per-operator timing (operators, dimensions, execution mode) for
  the planned interpreter; `bench-decode --execution` selects the concept
  being benchmarked.
- **Validation gates** (frozen in `docs/v04-execution-contract.md`): real-
  model parity (Gate B: identical greedy tokens on the six frozen English +
  Arabic prompts, logits within the frozen envelopes), hook semantics
  (Gate C), memory (Gate D), allocation (Gate E), performance (Gate F), and
  the external golden ladder (Gate G). Parity verified on all four primary
  combinations (Llama-3.2-1B and Qwen2.5-1.5B x Q4_K_M/Q6_K).

### Changed

- The v0.3 Q8_0 native fast path keeps precedence and is never rerouted
  through the plan; the generic reference path remains the default
  (`--execution reference`) and the readable oracle.
- A process-wide counting global allocator (`src/alloc_counter.rs`)
  replaces the test-local allocator; Gate E and memory reporting use it.

### Fixed

- Planned decode diverged from the reference after the first token when
  arena regions were reused across tokens: the quantized matvec kernels
  accumulate into their destination, which must be zero-initialized (the
  reference allocates fresh zeroed vectors; the arena reuses regions). The
  interpreter now clears matvec destinations before each projection.
- `planned-fused` silently ran the reference path until
  `planned_decode_eligible` accepted the mode.

## [0.3.0] - 2026-08-03

### Added

- **Native compressed-resident Q4_K/Q6_K execution** (the v0.3 release
  thesis): K-family weights stay packed and mmap-backed, dequantizing at
  super-block granularity inside scalar or AVX2 kernels. `--k-strategy
  eager-f32|scalar|x86|auto` (default `auto`: AVX2 when available, else
  scalar) with `--k-allow-fallback` and per-tensor recorded decisions.
  Q2_K/Q3_K/Q5_K/Q8_K remain eager-f32-only; unsupported dtypes under an
  explicit compressed strategy hard-fail naming the tensor unless the
  fallback flag is given.
- **Execution/residency provenance**: capture manifests gain a per-tensor
  execution inventory (GGUF dtype, resident representation, strategy,
  kernel, CPU features, fallback reason, workspace) plus model-level
  residency totals and the requested `k_strategy`; additive serde-default
  fields keep `0.2.0-experimental` artifacts comparable.
- **Validation gates** (frozen in `docs/v03-execution-contracts.md`):
  kernel (Gate A), model parity vs eager-f32 with identical greedy tokens
  (Gate B), golden logits vs pinned llama.cpp b9999 with 100% top-1
  agreement on all six ladder rungs (Gate C), x86-vs-scalar (Gate D), and
  the full capture → compare → patch → frozen-verdict causal workflow on
  the compressed path.
- **External validation tooling**: pinned llama.cpp setup
  (`scripts/setup_llama_cpp.sh`), fresh matched quantization ladder from
  fp16 (`scripts/quantize_ladder.sh`, manifest with commands + sha256),
  golden-ladder runner (`scripts/validate_golden_ladder.sh` +
  `tools/logits_dump.c`), and the benchmark matrix
  (`scripts/bench_v03.sh` → `artifacts/benchmark-v03/`).
- **qwen2.5 loader fix**: GGUFs from the local fp16 source omit the
  family `vocab_size` metadata key; `LlamaConfig` now falls back to the
  embedding tensor's row count when the key is absent.
- `bench-decode` output records `k_strategy`, K-tensor counts, fallback
  count, and compressed/expanded resident bytes.

### Changed

- The crate version jumps 0.1.0 → 0.3.0 (the 0.1 series never shipped a
  release; v0.3 is the first tagged milestone).
- K-quant models are ineligible for the Q8 workspace fast decode; they
  run the generic hooked path with `DispatchPath::Generic` recorded.

### Fixed

- Capture: after-logits records used `execution.input_token_count`, but
  the llama family returns last-token logits `[1, vocab]` even during
  prefill, so multi-token prefill records failed self-validation. The
  record token count now comes from the tensor's actual row count.

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
