# Changelog

All notable changes to Ember are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Ember's Rust experiment API is explicitly unstable during the 0.1 series;
the v0.2 activation-artifact schema (`0.2.0-experimental`) is versioned but
carries no compatibility guarantee.

## [Unreleased] - agentic phase 2

### Added

Follow-ups to the v0.6.7 agentic layer, all additive:

- multi-call steps: one assistant turn may request several tools; the loop
  validates, approves, executes, and reinjects them in order with limits
  and cancellation checked between calls. Protocols now parse ALL well-
  formed calls per step (Qwen `<tool_call>` blocks collect all-or-nothing
  on malformed bodies; the generic mode collects embedded calls).
- approval gating (completes the Track H seam): `ApprovalPolicy` of `Auto`
  / `DenyExternalSideEffect` (new default) / custom host gate; denials are
  structured `denied_by_policy` rejections that are traced and fed back to
  the model. CLI: `--allow-unsafe-effects`.
- trace tooling: `ember trace diff` (status/totals/final-digest plus
  event-skeleton comparison, `--fail-on-diff` for scripting),
  `ember trace replay` (re-executes recorded deterministic tool calls
  offline against a fresh registry and verifies stable payload digests;
  volatile fields excluded, legacy traces skipped honestly),
  `ember trace report` (self-contained HTML: summary, timeline bars,
  artifacts, full event table; inline CSS only).
- every tool execution now records both `payload_sha256` and a
  `replay_sha256` (stable across reruns) under all privacy modes.
- `image_fixture` built-in: deterministic PNG test pattern through the
  artifact path (`image/png`), proving binary media flows end-to-end
  (Track W seam).
- docs: `docs/agent-runtime.md`; README gains an agentic section.

### Fixed

- platform-stable canonical serialization for tool-schema prompts and
  tool-result payloads: serde_json key order depends on whether another
  dependency enables `preserve_order` (observed flipping byte-pinned
  protocol renders on aarch64 CI). Prompt bytes, digests, and golden pins
  now use an explicit canonical writer with sorted keys.

## [0.6.7] - 2026-08-25

### Added

Agentic execution layer + research tracing (`ember::agent`): Ember can now
act, not merely generate. The loop sits entirely ABOVE inference (no
attention/KV/tokenizer changes) and drives the existing session machinery.

- Tool runtime: `ToolSchema` (JSON-Schema-compatible subset: string,
  number/integer, boolean, array, nested object, string enums, required),
  strict argument validation with structured multi-error reporting
  (`ValidatedArguments`; unknown fields/missing/wrong-type/enum all
  collected), frozen `ToolRegistry` with duplicate rejection, watchdog-
  enforced tool timeouts (detached-worker caveat documented), contained
  panics, cooperative cancellation via the existing `GenerationControl`.
- Protocol boundary: `ToolCallProtocol` with three codecs - Qwen2.5
  ChatML+`<tool_call>` (official convention), Llama 3.x
  `<|python_tag|>` JSON custom functions (Meta zero-shot convention,
  `ipython` result role), and an honest generic-JSON testing mode.
  Rendered messages are pinned byte-exactly by unit tests; present-but-
  broken calls classify as explicit `MalformedToolCall`, never silently
  recovered to prose. One call per step (documented; extras counted in
  traces).
- Agent state machine (`AgentSession::run`): explicit loop with hard
  limits (max steps / max tool calls / wall time / per-tool timeout /
  per-turn output tokens / tool-result reinjection cap), commit ledger
  (`system → user → assistant_tool_call → tool_result → … → final`),
  limit outcomes as first-class `RunStatus::LimitReached` +
  `run_terminated` terminal events. Cancellation mid-generation commits
  nothing (engine rolls the KV cursor back); a tool that executed before
  cancellation keeps its side effect visible via `tool_result_uncommitted`.
- Deterministic built-ins: `calculate`, `lookup` (fixture table),
  `echo`, `write_artifact` (hashed), `fail`, sandboxed read-only
  `read_text_file`/`search_text` (traversal fails closed). No
  shell/network/delete tools by design.
- Research tracing: crash-tolerant JSONL (`ember.agent.trace.v1`,
  monotonic `seq`, flush-per-event, torn-line tolerant parser), span-ish
  step ids, privacy knobs (prompts/generated text on/off → hashes;
  tool payloads full/summary(2048 default)/hash; token events off),
  provenance event (ember version, git commit/rustc/target from build.rs,
  model identity incl. SHA-256 + quant + tokenizer hash, protocol id,
  tool-schema snapshot, config).
- Artifacts: run-local `ArtifactStore` writing sanitized, atomically
  published files with sha256/size/media-type/producer provenance, traced
  as `artifact_written`.
- CLI: `ember agent run|demo` (llama/qwen-family GGUFs, `--protocol`,
  deterministic built-ins incl. sandboxed file tools, trace out) and
  `ember trace inspect` (compact timeline + aggregates).
- Tests: 40 lib unit tests + 19 hermetic scripted-model integration tests
  (`tests/agent_runtime.rs`: exact one-tool round trip, multi-step,
  failure/malformed/unknown-tool recovery, max-steps/max-tool-calls/
  wall-time limits, mid-generation and mid-execution cancellation,
  timeout, panic containment, artifact hashing, incremental+torn JSONL)
  plus a real-GGUF gate (`tests/agent_e2e.rs`, `EMBER_AGENT_E2E=1`)
  executed against Llama-3.2-1B-Instruct-Q8_0 and Qwen2.5-1.5B-Q8_0.
- Bench: `benches/agent_overhead.rs` - orchestration ≈0.5–1.9 ms/run
  (mock tools), trace overhead ≈0.2 ms, ~16 events / ~5 KB per one-tool
  run.

## [0.6.6] - 2026-08-25

### Added

Multimodal patch (Phase 5, session 2): live model-in-the-loop voice
conversation and Arabic speech input/output over the v0.6.5 multimodal
foundation.

- `ember voice --converse`: full conversation loop in one command — cpal
  capture ring → energy VAD → streaming audio → `VoiceSession` → LLM →
  speech output → playback ring, with barge-in (generation-phase cancel +
  KV rollback via `KVCache::truncate_to`, playback-phase interrupt),
  deferred recapture seeded from the utterance buffer, and a total
  transition graph pinned by `tests/converse_machine.rs`.
- `duplex` pump rework: `pump_events`/`pump_with_chunk_cb` so onset AND
  endpoint in one chunk both fire; utterance-head-preserving collection;
  device-rate mixdown with the single validated sinc resample downstream.
- MMS-TTS (facebook/mms-tts-\*, VITS architecture) inference from scratch in
  Rust (`src/tts/vits.rs`): character frontend over raw Arabic script,
  relative-position transformer encoder, deterministic SDP reverse,
  monotonic duration expansion, WaveNet prior flow reverse, HiFi-GAN
  decoder. Parity vs the HuggingFace reference is measured at every
  boundary (`scripts/ref_vits.py`, `ref_flow_dump.py`, `ref_decoder_dump.py`,
  `ref_rb_dump.py`, `compare_vits.py`): embeddings exact; encoder out
  2.5e-7 rms_rel; flow z 3.4e-7; all decoder substages ≤5e-6; waveform
  rms_rel 8.8e-6 / cosine 0.99999999996 with exact sample counts. The
  residual is proven to sit below torch's own f32-vs-f64 cross-precision
  floor (accumulation order, not semantics). Fixes that closed it:
  resblock dilation cycle [1,3,5] (was d+1), final pre-conv_post leaky at
  transformers' default slope 0.01 (not config's 0.1), conv_post tail
  computing every valid output (last 6 were zeroed), and rel-pos slice-row
  indexing for sequences shorter than the attention window.
- `SpeechOut` trait unifying OuteTTS/MMS-VITS behind one streaming seam so
  `--vits-model` swaps the Arabic-capable engine into the conversation
  loop (`tests/arabic_s2s_vits.rs` drives bank audio → transcript → reply →
  PCM end-to-end against real weights).
- Streaming audio input scheduling surfaces: finalized-window encode-once,
  explicit-floor window slices, running floor — stream-validate proves
  streamed == static bit-exact across push patterns on an Arabic speech
  bank builder (`scripts/build_arabic_speech_bank.py`).
- JSONL benchmark harness (`scripts/bench_phase5.py`): per-record git/CPU/
  thermal/load metadata + workload timings.

### Fixed

- CI: `benches/multimodal_batch.rs` exited 2 when executed without
  arguments by `cargo test --all-targets`, turning the v0.6.5 release run
  red on both the x86_64 and aarch64 tiers; it now skips silently like the
  absent-fixture path.

## [0.6.5] - 2026-08-23

### Added

Multimodal foundation (Phases 1–4, sessions 1–2): image/audio/video input,
persistent voice sessions, and the first speech output — all through the
existing embedding-seam LLM path (`EmbeddingSequence`), so the transformer,
KV cache, and K-quant kernels stay modality-blind.

- `multimodal` request substrate: ordered `ContentPart`s (text / image /
  audio / video) with in-memory or file-backed media; model adapters fail
  closed on unsupported combinations.
- SmolVLM-256M still-image VLM: Pillow-exact preprocessing with the
  reference tile-rounding stage, heterogeneous tile geometries grouped per
  encoder pass, multi-image requests, cross-request ownership-aware batch
  encoding (`benches/multimodal_batch.rs`).
- SmolVLM2-256M video: deterministic frame-sampling policies with full
  provenance; reference prompt expansion byte-matches HF.
- Ultravox v0.5 audio input: Whisper-style encoder + SwiGLU projector in
  Rust; long-form audio via reference chunking protocol (30.1/45/60 s
  validated against fresh HuggingFace references); incremental streaming
  frontend (`AudioStream`: stateful resampler + log-mel, bit-exact under
  arbitrary partitioning) with encoder scheduling that encodes finalized
  30 s windows exactly once and rebuilds stale ones when the global mel
  floor moved — streamed inference is bit-exact vs static across partition
  patterns (`ember audio --stream-validate`).
- Persistent `VoiceSession`: committed-prefix KV reuse across turns (turn
  embeddings retained for O(1) re-prefill), media feature cache keyed by
  content ⊕ recipe ⊕ tower identity, provisional partial transcripts on a
  cloned scratch KV (committed state provably untouched), cancellation via
  `KVCache::truncate_to` cursor rollback.
- First speech output: OuteTTS-0.2-500M (qwen2 base, official GGUF) +
  WavTokenizer decoder implemented in Rust (Vocos backbone, iSTFT head,
  Bluestein FFT for N=1280); every decoder boundary validated against the
  unmodified reference implementation (waveform rms_rel ~6e-6, cos 1.0);
  greedy generation agrees with llama.cpp to a near-tie codec flip;
  `ember tts [--stream]` synthesizes WAV files with time-to-first-audio
  metrics. VoiceLoop barge-in policy: interrupt during generation cancels
  and rolls back; interrupt during playback stops audio but keeps committed
  text.
- Output-event boundary (`OutputEvent::{TextDelta, AudioChunk}`) and
  executable-truth capability reporting per wrapper.

### Changed

- Multimodal wrappers load quantized text GGUFs compressed-resident
  (`KStrategy::Auto`) instead of silently dequantizing K-quants at load:
  Q6_K/Q4_K_M decode ~0.6 → ~15 tok/s, same-request RSS ~13.1 → ~4.3 GB
  (request-sized KV allocation everywhere).
- Vision softmax uses an AVX2/FMA fast-exp path (two-part ln2 range
  reduction, degree-7 Taylor; opt-out `EMBER_VISION_FAST_EXP=0`);
  single-image encode −13%, 48-frame video −10%; all numerical gates held.

## [0.6.3] - 2026-08-16

### Changed

- Native experiment console migrated from iced to gpui 0.2 (Zed's
  GPU-accelerated framework, blade/Vulkan on Linux); CI installs the X11
  system libraries and the README documents the Vulkan requirement. gpui is
  now an optional dependency behind the default `gui` cargo feature —
  headless research builds use `--no-default-features` and skip the
  blade/ash/wayland stack.
- `ember experiment reproduce` reuses the bundle's recorded
  `resolved-experiment.json` instead of re-resolving, and its verdict gained
  capture-alignment gating (`captures-misaligned` outcome).
- `--arch auto` (read from GGUF metadata) is accepted across `bench-decode`,
  `inspect-plan`, and the `kv` subcommands; conflicting explicit `--arch`
  values fail closed.
- `validate-backends` dropped its dead `--model`/`--prompts`/`--layers` flags;
  the web-GUI server is bounded and panic-safe.
- The `arabic_morph_dataset` Python package moved from `src/` to `python/`.

### Added

- Opt-in Q4_K/Q6_K quant pre-split (`EMBER_PRESPLIT`): prefill four-row tiles
  read the pre-split lanes directly (bit-identical; speed measured on a cool
  machine).
- Eager K-quant/Q8_0 dequant is rayon-parallel over blocks/rows and prefill
  is row-tile-major (measured ~8.5% prefill, ~6% eager-load, bit-identical).
- CI: aarch64 headless tier (build + test + clippy on `ubuntu-24.04-arm`),
  a no-default-features check on x86_64, and a prebuilt cargo-audit action.

### Fixed

- gemma4 vocab size is derived from the token embedding instead of the
  256000 default, so 262144-vocab models load.
- Extraction locates the target word by placeholder position rather than
  unique substring, and tokenizer byte offsets are validated against byte
  length instead of char count (broke non-ASCII prompts).
- Dirty-tree builds are flagged in run-metadata commit hashes.

### Removed

- Regenerable artifact payloads untracked on main: per-bundle `model.json`
  exports in `artifacts/benchmark-v05/` (byte-identical across bundles),
  golden-v03 raw logits (`*.npy`/`*.bin`, regenerable via
  `tools/logits_dump.c`), and a stray docs screenshot + cached paper PDF.
  Summaries, specs, plans, and capture tensors remain tracked.

## [0.6.2] - 2026-08-12

### Changed

- Q4_K/Q6_K production matmul now uses canonical transient Q8_K activation
  packing and integer dots for decode and prefill. The exact-f32 dequantize/dot
  implementation remains an explicit slow oracle. Non-finite activations fail
  before destination mutation; warmed workspace and accumulate semantics are
  explicit contracts.
- Execution plans retain loader fallback provenance and identify the numerical
  runtime with `kernel_revision = 2` while preserving offline verification and
  hashes of historical revision-1 plans. Plan/dispatch disagreement is a
  release-mode error, and the cache key now includes Rayon thread count.
- The superseded internal K-quant modules (`k_gemv`, `k_prefill`,
  `k_matmul_x86`) and their ad-hoc examples/benches were replaced by
  `k_quant_matmul`. Ember's Rust library is internal/unstable; this is an
  intentional source-level break rather than compatibility shims over dead hot
  paths.
- The v0.4 planned-decode interpreter moved out of `src/llama.rs` into
  `src/planned_decode.rs` (resolved ops, scratch-arena session, planned and
  fused kernels, `forward_last_logits_planned`). Zero behavioral change;
  `src/llama.rs` keeps the model, eager forward, dispatch, and plan
  construction. The v0.5 research contract's reference to the interpreter
  path was updated accordingly.
- Toolchain pinned via `rust-toolchain.toml` to 1.92.0 (the declared MSRV
  and CI toolchain); local builds now match CI.
- Dependencies: `crossbeam-epoch` 0.9.18 → 0.9.20 (RUSTSEC-2026-0204),
  `anyhow` → 1.0.104 (RUSTSEC-2026-0190), `memmap2` → 0.9.11
  (RUSTSEC-2026-0186). `cargo audit` is now part of CI.

### Added

- First-class KV-prefix snapshots for the Llama/Qwen CPU runtime through the
  `ember kv` command family: deterministic `ember.kv-snapshot.v1` artifacts,
  strict integrity and compatibility checks, bit-exact same-model replay,
  cache comparison and perturbation diagnostics, replay traces, and explicit
  RoPE remove/reapply research utilities. The validation matrix and benchmark
  evidence are recorded under `artifacts/benchmark-kv-v1/2026-08-08/`.
- The bilingual “Road to Ember 1.0” documentation series and a substantial
  native-console visual refresh, while preserving the same v0.5 experiment
  execution path and offline behavior.
- A pinned llama.cpp known-answer verifier for Q8_K bytes and Q4_K/Q6_K dots,
  fail-closed 1B real-model scalar/x86 validation, a dedicated x86 CI gate,
  adversarial kernel matrices, and schema-4 path-interleaved benchmark output
  with checksums and full dispatch/workspace provenance.
- `tests/property.rs`: proptest suite (tensor shape ops vs hand-rolled
  references, decode-arena disjointness/alignment/isolation, K-quant dequant
  contracts) plus fuzz-style robustness tests for the untrusted-input
  boundaries: the GGUF loader, the v0.5 spec parser, and the npy reader must
  never panic on arbitrary input.
- `benches/hooks_overhead.rs`: planned-decode cost with a noop experiment
  runner attached vs bare (Gate H evidence; measured ≈0% overhead).
- Rustdoc lints (`broken_intra_doc_links`, `private_intra_doc_links`) and a
  CI docs step (`RUSTDOCFLAGS=-D warnings cargo doc`).

### Fixed

- `forward_last_logits_planned` had hard-coded `ExecutionMode::Planned`, so
  `planned-fused` parity tests never executed F1-F5. It now routes the model's
  actual mode; tests use execution counters. This exposed Q8_0 F5 overwriting
  the residual (Q8_0 assigns rather than accumulates), so Q8_0 F5 now de-fuses
  with a serialized reason while f32 and canonical K kernels execute F5.
- Owned reader-backed K weights now receive the loader-resolved execution tier
  at construction, matching mmap-backed weights. Loader fallback reasons and
  original dtype survive model construction into execution plans.
- Build provenance now exports the `EMBER_GIT_COMMIT` name consumed by plans,
  traces, and benchmarks (plus dirty-tree status for benchmark records).
- `tests/k_parity.rs::v04_planned_inactive_hooks_real_model` failed on Q8_0
  models: the plain run uses the v0.3 native fast path (contract D1) while
  the hooked run uses the generic hooked path, so bit-exact logits were not
  the right contract there (tokens always matched). The test now skips
  models without K-quant tensors; all parity suites (Q8_0, Q6_K, Q4_K_M)
  pass 5/5.
- All 8 rustdoc warnings (math-notation brackets parsed as links,
  private-item links, unclosed HTML tag in a doc comment).
- Stale `#[allow(dead_code)]` on `KQuantWeight::try_from_mmap` (the loader
  calls it), on the generic `LlamaMlp/Attention/Block` structs, and dead
  gemma4 `forward_full` paths (test-only attention ported behind
  `#[cfg(test)]`).
- `// Safety:` comments on the previously undocumented `unsafe` sites: the
  `matrixmultiply::sgemm` call in `tensor.rs` and 29 SIMD kernels in
  `simd.rs`.

## [0.6.1] - 2026-08-09

### Added

- `ember gui` (native console): light/dark theme toggle in the header
  (defaults to dark). Every color role in the console switches with the
  palette, including iced widgets (inputs, combo boxes, editor, buttons)
  which follow the iced `Theme`.

## [0.6.0] - 2026-08-06

### Added

- `ember gui` — a native, single-window experiment console for live demos
  (v0.6): an iced app on the tiny-skia software renderer (no GPU or
  webview dependency) with embedded Noto fonts (`src/gui_fonts/`) for
  offline Latin + Arabic coverage, dark console theme.
- `ember web-gui` — an offline, single-page browser console for live demos
  (v0.6). A thin presentation layer over the existing v0.5 pipeline: the
  page translates every action into an `ember.experiment.v1` specification,
  validates it through the standard `RawExperimentSpec::resolve()` gate, and
  executes it with the same `prepare_run` / `execute_prepared` code as
  `ember experiment run`. One resident model session serves repeated
  baseline / intervention / restore runs, so the demo loop never reloads the
  model. Bundles are written and self-verified exactly as in v0.5; the
  restore-original leg reports a bit-exact match against the baseline.
  Light/dark theme toggle (defaults to the system preference, persisted in
  localStorage). See `docs/v06-gui.md`.
- `src/gui_native.rs`: the native console — same `GuiSession` core and
  `parse_run_request` gate as the browser console, runs executed in a
  worker thread so the UI never blocks.
- `src/gui.rs` + `src/gui_page.html`: tiny embedded HTTP server (tiny_http,
  localhost only) and a self-contained page (no web framework, no external
  assets) with Arabic/RTL rendering via the browser (`dir="auto"` per field;
  the UI itself stays LTR).
- `cli_experiment::prepare_run` / `execute_prepared`: the v0.5 run path was
  split into a reusable model-load step and an execute step so a loaded
  model can be kept resident. `ember experiment run` behavior is unchanged
  (it calls both in sequence).

## [0.5.1] - 2026-08-04

### Fixed

- CI: `src/v05/testutil.rs` carried a file-level `#![cfg(test)]` while
  `mod.rs` already gates the module; rustc 1.92 (the pinned CI
  toolchain) rejects the duplicated attribute under clippy. Removed,
  verified with `cargo +1.92.0 clippy -- -D warnings` and
  `cargo +1.92.0 test --all-targets`.
- Docs: `docs/usage.md` linked the v0.2 activation-artifact schema to
  `docs/experiments.md`, which now documents the v0.5 workflow; the
  reference moves to `activation-artifacts.md`/`activation-patching.md`.

### Added

- Recorded real-model capture-from-bundle workflow (Gate D evidence):
  baseline captures prompt-final `attention-output` across all layers; a
  second run replaces layer 8's row with the baseline bundle's layer-3
  row; a third run adds `restore-original`. Comparison: layers 0-8
  bit-exact, 9-15 diverge; restoration reproduces the baseline with all
  16 capture layers exact and outputs equal. Specs, hashes, and commands
  under `artifacts/benchmark-v05/capture-from-bundle/` and
  `docs/interventions.md`.
- Refreshed Gate H matrix with the release binary: ordinary runs
  2.58 s / 2,751,028 kB RSS vs 2.61 s / 2,751,024 kB with the experiment
  machinery unused; experiment workloads +2.2% RSS (gate: <=3%).

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
