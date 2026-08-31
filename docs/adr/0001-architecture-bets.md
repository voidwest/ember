# ADR-0001: Keep Ember inspectable, CPU-first, and independently validated

- **Status:** Accepted
- **Date:** 2026-08-28
- **Scope:** inference/runtime ownership, native console, and correctness
  validation

## Context

Ember is a research instrument as well as an inference binary. It must expose
stable semantic hook boundaries, packed GGUF/K-quant residency, deterministic
experiment artifacts, and an independent path for detecting numerical drift.
Throughput and broad model coverage matter, but they are not the only design
objective. The choices below are deliberate constraints, not claims that the
rejected technologies are generally inferior.

## Decisions and why-not alternatives

### 1. Rust over a C++ implementation

**Decision.** Keep the model, loader, experiment, and artifact layers in Rust.
Allow a narrow, explicitly reviewed `unsafe` seam for target-feature SIMD, with
safe Rust reference/oracle code around it. Keep llama.cpp as an external
correctness and performance reference where its behavior is useful.

**Why not C++ as the host runtime?** C++ and llama.cpp are credible choices for
high-throughput inference, but making C++ the host would not by itself provide
Ember’s ownership proof, checked shape/lifetime boundaries, or an easy-to-audit
safe reference path. It would also couple the research hooks and artifact
semantics to an external C++ ABI and implementation lifecycle. Rust’s borrow
checking, explicit ownership, and MSRV-pinned builds fit the project’s priority
of inspectability. This is not a Rust-versus-C++ speed claim: the hot kernels
still require architecture-specific measurement, and llama.cpp remains a
valuable external comparison.

**Consequences.** Most code can stay safe and explicit; target-feature kernels
need `# Safety` contracts, runtime feature gates, and cross-target CI. The
crate is not a drop-in llama.cpp replacement, and some model support requires
more hand-written architecture code. The `Backend` seam leaves room for a
future backend without pretending one exists today.

### 2. gpui for the native console, with a separate plain browser console

**Decision.** Use gpui for the optional native `ember gui` console and retain
the self-contained localhost `ember web-gui` as the portable browser surface.
The native GUI stays behind the default `gui` feature; headless users can build
with `--no-default-features`. Both consoles call the same experiment pipeline
and never implement a second inference path.

**Why not browser-only or a webview?** A browser-only UI would be portable, but
would not provide the project’s chosen native window/input/font integration and
offline embedded Arabic-font path. A webview would add another runtime and
packaging/security surface. The browser console is still kept because it is a
useful low-dependency fallback for users who do not need a native window.

**Why not iced/tiny-skia, egui, or another native stack?** Those are reasonable
UI technologies, but retaining multiple rendering/input stacks would increase
maintenance and make Arabic text, worker/session behavior, and the single
experiment pipeline harder to reason about. gpui was selected for the native
console’s GPU/Vulkan integration; this ADR does not claim a universal toolkit
benchmark or forbid a future replacement supported by evidence.

**Consequences.** The native build has a heavy graphics dependency and needs a
Vulkan-capable display plus platform libraries; CI therefore keeps a headless
feature tier. The browser console remains intentionally small and local-only.
UI changes must preserve the shared `prepare_run`/`execute_prepared` path and
bundle verification rather than duplicating model logic. See
[`docs/v06-gui.md`](../v06-gui.md).

### 3. A hand-rolled, explicit runtime over an existing inference framework

**Decision.** Own the GGUF loader, model-family paths, KV cache, execution plan,
semantic hooks, interventions, and deterministic bundle writer in this crate.
Keep abstractions narrow and inspectable; use the exact/reference path as an
oracle and make optimized dispatch explicit.

**Why not a llama.cpp binding?** llama.cpp is the preferred external reference
for quantized logits and performance, but a binding would make Ember’s hook
locations, activation ownership, K-quant residency, plan de-fusion, and
artifact identity depend on foreign graph and ABI decisions. It is useful at
the boundary as a comparator, not as the implementation whose internals Ember
is meant to inspect.

**Why not Candle, Burn, ONNX Runtime, tract, or another graph runtime?** An
existing runtime could accelerate onboarding and add model coverage. Its graph
rewrites, allocator/lifetime policy, and backend semantics would nevertheless
be another abstraction to audit against Ember’s exact intervention and
provenance contracts. The project deliberately accepts narrower model support
and more local maintenance in exchange for direct control. This is a scoped
choice, not a claim that those frameworks cannot host a different research
instrument.

**Why not a dynamic plugin/event-bus runtime?** Dynamic registration and a
second execution graph would obscure ordering, allocations, and semantic hook
identity. The static experiment surface and versioned bundles make a smaller
failure domain and deterministic review easier. Broader extensibility requires
an explicit design record and new contract gates.

**Consequences.** Ember carries architecture-specific code and must maintain
its own tests, documentation, and compatibility policy. In return, a reviewer
can trace a tensor from GGUF bytes through a named operation to a capture or
intervention, and can compare the optimized path with the reference path.

### 4. Keep an independent exact oracle; never hide fallback

**Decision.** Retain the eager/exact-f32 K-quant path (including
[`src/k_matmul.rs`](../../src/k_matmul.rs)) alongside the production
Q4_K/Q6_K × Q8_K kernels in
[`src/k_quant_matmul.rs`](../../src/k_quant_matmul.rs). Record per-tensor
execution decisions and fallback reasons; do not silently substitute a path.

**Why not optimize-only or silent fallback?** A production path that validates
only against itself can reproduce the same bug on every test. A silent fallback
can produce plausible output while changing memory, timing, numerics, or hook
provenance without telling a researcher. The independent oracle and explicit
decision records make failures diagnosable and keep benchmark/research claims
honest.

**Consequences.** The oracle costs code, test time, and a slow execution option,
but it is a deliberate validation dependency. New kernels must prove parity
against it at the appropriate tolerance and then, where possible, against an
external reference. See [`docs/validation.md`](../validation.md) and the
[`v0.3 execution contract`](../v03-execution-contracts.md).

## Revisit triggers

Open a new ADR (rather than silently changing these choices) if any of the
following becomes true:

- a supported deployment target requires a GPU or a different language/ABI;
- native UI distribution, accessibility, or platform support makes gpui’s
  dependency cost unacceptable;
- an existing runtime can satisfy the hook, residency, intervention, and
  deterministic-artifact contracts with measured evidence;
- the independent oracle can no longer be built or an external reference
  changes the frozen numerical contract;
- a model family, schema, or performance target requires a broader execution
  graph.

A replacement proposal must include migration/compatibility impact, benchmark
and validation evidence, failure-mode ownership, and a rollback or fallback
plan. Update [`docs/audits/README.md`](../audits/README.md) and the nearest
frozen contract with the decision.
