# Contributing to Ember

Ember is a small, inspectable CPU inference engine and research artifact.
Contributions should preserve numerical behavior, tracing, hidden-state
extraction, benchmark reporting, and explicit model-family differences.

## Before opening a change

- Open an issue for model support, new experiments, public API changes, or
  performance work.
- Keep model files, tokenizers, generated activations, benchmark outputs, and
  paper artifacts out of commits.
- Prefer deletion and focused simplification over new framework layers.
- Do not merge LLaMA, Qwen, or Gemma behavior unless their architectural and
  numerical differences remain explicit and tested.

The v0.1 experiment surface is intentionally frozen at one observation
experiment and one intervention experiment. New dynamic loading systems,
registries, event buses, configuration DSLs, or multi-experiment pipelines are
out of scope.

## Public API and compatibility

Read [docs/api-stability.md](docs/api-stability.md) before adding or changing
an exported Rust item, CLI flag, feature, or serialized field. `pub` is not by
itself a stability promise while Ember is below 1.0: prefer `pub(crate)` for
implementation helpers, document experimental APIs explicitly, and include a
migration note for intentional breaks. Keep the declared Rust 1.92 MSRV and
run both default and `--no-default-features` checks.

## Local validation

CI pins Rust 1.92.0 (`rustup override set 1.92.0` locally makes your
toolchain match CI — newer toolchains emit different clippy lints). The
exact CI gates are:

```bash
cargo fmt --all -- --check
cargo check --locked --all-targets
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets
.venv/bin/python -m pytest tests probes/test_probe_workflows.py -q
.venv/bin/python scripts/check_docs.py --check
bash -n bench_compare.sh probes/run_all_5k.sh scripts/research_example_capture_patch.sh tools/crossover_sweep.sh
```

## Repository layout and boundaries

- `main` is the public branch: code, docs, tests, CI, and tracked data
  exports. It is pushed to origin.
- `paper/` and the `paper-private` branch hold the TACL submission sources
  and are **never pushed**. The submitted paper's pipeline lives in a
  separate repository (`~/research-stack`, pinned by its own code manifest);
  it will be released as a tagged artifact on acceptance.
- `pilot-001` is a local-only branch with Arabic-quantization pilot data.
  Nothing under `research/pilots/` belongs on `main`.
- Model files (`*.gguf`) are gitignored — fetch them with
  `scripts/download_models.sh`.
- The root `.gitignore` whitelist keeps most `*.md` files local (design
  notes and logs). If a doc needs to ship, add it to the whitelist.

Changes to model execution also need focused synthetic tests and real-model
checks where fixtures are available:

- compare logits or generation bytes before and after;
- compare hidden-state artifacts and trace fingerprints;
- exercise both prefill and single-token decode;
- retain the generic path as the correctness oracle for packed kernels.

## Performance changes

Performance claims need controlled alternating A/B measurements. Pin one
worker to each physical core, set an explicit Rayon thread count, warm model
pages and kernels, alternate process order, retain every raw sample, and
restore the host power policy afterward.

```bash
RAYON_NUM_THREADS=4 taskset -c 0-3 target/release/ember bench-decode \
  --model MODEL.gguf \
  --arch qwen3 \
  --tokens 32 \
  --warmups 2 \
  --repetitions 3 \
  --max-seq-len 128
```

Do not retain a performance-sensitive path when controlled measurements show
a regression or no repeatable benefit.

## Pull requests

Keep commits independently reviewable. Describe:

- the supported behavior affected;
- correctness and trace evidence;
- allocation and timing effects;
- model fixtures and exact commands used;
- anything deliberately unsupported.

The [PR template](.github/PULL_REQUEST_TEMPLATE.md) expands this into a
repeatable evidence checklist. CI green is necessary, but it is not a
substitute for a model parity or research gate that was not run.

## Newcomer workflow

### 1. Orient before editing

Read the [README](README.md),
[`docs/architecture.md`](docs/architecture.md), and
[`docs/validation.md`](docs/validation.md) first. The frozen execution and
research contracts are the source of truth for behavior:

- [`docs/v03-execution-contracts.md`](docs/v03-execution-contracts.md) for
  compressed-resident Q4_K/Q6_K execution;
- [`docs/v04-execution-contract.md`](docs/v04-execution-contract.md) for
  plans, fusion, scratch, and hook boundaries;
- [`docs/v05-research-contract.md`](docs/v05-research-contract.md) for
  captures, interventions, token selection, and bundle identity;
- [`docs/audits/README.md`](docs/audits/README.md) for subsystem owners,
  backups, and recurring handoffs;
- [`docs/adr/0001-architecture-bets.md`](docs/adr/0001-architecture-bets.md)
  for durable “why this, not that” decisions.

Before touching a worktree, run `git status --short --branch`. Never reset,
clean, or overwrite uncommitted work you did not create; use a separate
worktree if ownership is unclear. Create a focused branch from the revision
you reviewed, for example `docs/...`, `fix/...`, `test/...`, or `perf/...`.

### 2. Build a baseline, then choose the smallest patch

Rust 1.92.0 is pinned by `rust-toolchain.toml`. No model is needed for the
unit-test path. The default feature includes the gpui/Vulkan native console;
use the headless check when display/X11/ALSA development libraries are not
available:

```bash
rustup toolchain install 1.92.0 --profile minimal --component clippy,rustfmt
cargo check --locked --no-default-features --all-targets
cargo test --locked --no-default-features --all-targets
```

For a normal checkout, establish the full baseline when the host has the CI
system libraries:

```bash
cargo fmt --all -- --check
cargo check --locked --all-targets
cargo test --locked --all-targets
```

If a baseline fails, record the exact command, revision, and failure. Do not
hide a pre-existing failure by weakening a test. Keep model weights, tokenizer
copies, generated activations, paper files, and large benchmark payloads out of
the diff; the root `.gitignore` intentionally leaves most Markdown notes
untracked as well.

### 3. Make the change explainable

Map the owning module, reference path, tests, and contract before coding. Add
a focused test or documentation example with the change. Keep architecture-
specific behavior explicit, prefer deletion over framework layers, and write
down unsupported inputs instead of silently falling back. A change to a public
Rust item, CLI flag, feature, or serialized field must also follow
[`docs/api-stability.md`](docs/api-stability.md).

### 4. Validate the claim and leave a handoff

Use the ladder below rather than treating a smoke run as parity. Record
commands, model/tokenizer hashes, architecture, quantization and execution
strategy, tolerances, and skipped gates. A second maintainer should be able to
repeat the strongest check without private context. Kernel, unsafe, schema,
architecture, and performance changes need a named subsystem reviewer plus a
backup/second reviewer.

## Validation ladder

These levels answer different questions:

1. **Smoke** — structural execution only. Loading and producing output is not
   numerical validation or evidence of output quality.
2. **Golden logits** — output comparison against a trusted reference for the
   same model, tokenizer, prompt, architecture, context, and quantization path.
   Record Ember/reference paths, model and tokenizer hashes, max/mean absolute
   error, cosine, top-1/top-k agreement, and pass/warn/fail status. Prefer the
   pinned llama.cpp path for quantized GGUFs; a full-precision Hugging Face
   comparison has different expected error.
3. **Activation reference checks** — internal hidden states agree with a
   trusted implementation at named layers/stages. Final logits alone cannot
   establish this.
4. **Probes** — linear probes show decodability and MLP probes show nonlinear
   recoverability; neither establishes causal use.
5. **Interventions** — first demonstrate that the intervention changes the
   intended score/representation under a valid control, then measure logit or
   generation effects. Use “causal effect” only for this level.

Generation/output scoring is a separate behavioral result. Every report or PR
should label the highest level actually run and state what it does not prove.

### Repository gates

These are the default local equivalents of CI (run the applicable set):

```bash
cargo fmt --all -- --check
cargo check --locked --all-targets
cargo test --locked --all-targets
cargo clippy --locked --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --release
cargo check --locked --no-default-features --all-targets
```

The dedicated x86 K-quant tier is intentionally fail-closed:

```bash
EMBER_REQUIRE_X86_TESTS=1 RAYON_NUM_THREADS=2 \
  cargo test --locked --release --lib k_quant_matmul::tests -- --nocapture
```

For Python/tooling changes, use the project environment or an equivalent
Python 3.11+ interpreter and run the applicable CI checks:

```bash
.venv/bin/python -m compileall -q python probes stimuli scripts tests
.venv/bin/python probes/run_probe_matrix.py \
  --model smoke:dummy.gguf --generate-tokens 1 --dry-run
.venv/bin/python scripts/check_docs.py
.venv/bin/python -m pytest tests probes/test_probe_workflows.py -q
bash -n bench_compare.sh probes/run_all_5k.sh \
  scripts/research_example_capture_patch.sh scripts/conference_demo.sh \
  scripts/validate_k_parity.sh tools/crossover_sweep.sh \
  tools/verify_k_quant_llamacpp.sh
```

Docs-only changes need `git diff --check`, the Markdown links checked from
repo root, and `scripts/check_docs.py`; do not rewrite generated HTML for a
Markdown-only patch. If a gate is unavailable (no model, x86 host, display, or
external reference), mark it **not run**, explain why, and provide the strongest
available substitute.

## Design and review checklist: `src/k_quant_matmul.rs`

`src/k_quant_matmul.rs` is the canonical Q4_K/Q6_K-weight × transient Q8_K-
activation matmul. Read it with [`src/k_matmul.rs`](src/k_matmul.rs), the
exact-f32 oracle; [`src/quant_k.rs`](src/quant_k.rs), the checked block layout;
[`src/loader.rs`](src/loader.rs), the recorded strategy; and the frozen
[`v0.3 contract`](docs/v03-execution-contracts.md). It is a hot,
unsafe-adjacent boundary, not a place to add an unvalidated second kernel.

### Design invariants

- **Numerical definition:** preserve `dst += src × dequant(w)` and transient
  Q8_K packing. Keep the eager/exact-f32 oracle independent; a local
  self-comparison is not a golden check.
- **Shape and mutation:** reject zero rows, overflow, wrong source/destination
  lengths, malformed execution state, and non-finite activations without
  partially modifying `dst`. Preserve row remainders and output-column order.
- **Block transcription:** verify Q4_K scale/min unpacking, Q6_K signed int8
  scales/high-bit assembly, Q8_K signed-max scale selection and ties-to-even
  rounding, and the 256-element block/record sizes against `quant_k`.
- **Dispatch safety:** scalar remains complete. AVX2 requires runtime-checked
  AVX2+FMA+F16C+SSSE3; experimental AVX-512 stays opt-in and feature-gated.
  Every `unsafe` helper needs a precise `# Safety` contract, and unsupported
  CPUs must never enter a target-feature kernel.
- **Parallel ownership:** output-column intervals are disjoint and joined
  before the caller observes `dst`. Preserve the `DstColumns` ownership proof,
  nested Rayon behavior, and moving thread-local Q8_K storage out while Rayon
  runs (restoring it on success and error).
- **Execution boundaries:** do not move semantic hook stages, change the
  per-tensor `KExecution` decision, or add an observable kernel-only stage.
  Changes to plan/kernel revision, provenance, fallback, or gate tolerances
  need a contract/design update.
- **Allocation/performance:** retain warmed-call allocation guarantees and
  document new workspace or environment switches. A benchmark is evidence
  only after controlled alternating A/B runs; one hot or throttled host run is
  not a speed claim.

### Required evidence for a kernel PR

- [ ] Scalar and x86/AVX2 paths (where available) match the exact-f32 oracle
      for Q4_K and Q6_K, ordinary shapes, row remainders, zero scales,
      negative minima, saturated quants, and extreme values.
- [ ] The pinned llama.cpp known-answer vector passes via
      `tools/verify_k_quant_llamacpp.sh`; if the external checkout/compiler is
      unavailable, say so rather than calling the local oracle “golden”.
- [ ] `cargo test --locked --release --lib k_quant_matmul::tests` passes,
      including packing, route, nested-Rayon, tile, allocation, and shape
      tests. Run the dedicated x86 command on x86.
- [ ] Real-model K parity uses the pinned ladder and
      [`scripts/validate_k_parity.sh`](scripts/validate_k_parity.sh), with
      model/tokenizer SHA-256, architecture, strategy, thread count, and raw
      output recorded. Do not check in GGUFs.
- [ ] Both prefill (multi-row) and decode (single-row) behavior are covered;
      include Q8_0 regression evidence when shared code changes.
- [ ] Unsupported targets still compile and use the scalar/headless path (the
      aarch64/no-default-features CI tier is relevant).
- [ ] Allocation and dispatch claims have focused tests; performance claims
      include raw samples, CPU/thread/power controls, warmups, repetition
      order, and a comparison revision.
- [ ] Any changed hook, plan, schema, fallback, or tolerance contract is
      updated in the relevant `docs/v0*` document and called out in the PR.

### Suggested reviewer questions

1. What invariant changed, and where is its independent oracle?
2. Can malformed blocks, shape overflow, non-finite input, unsupported ISA, or
   nested Rayon calls panic, race, partially write, or silently fall back?
3. Is serial execution still bit-identical to the intended parallel path under
   the documented floating-point control state?
4. Is the performance result repeatable under counterbalanced controls, or is
   it only a hypothesis for a future benchmark?
5. Can a second maintainer reproduce the check and explain the unsafe boundary
   from the source and this checklist?

## PR and review expectations

Every PR should make these facts easy to find:

- **Purpose/scope:** supported behavior, files touched, and explicit non-goals.
- **Correctness:** tests/commands, reference implementation, tolerances, and
  trace/hook evidence; every skipped gate has a reason.
- **Reproducibility:** model/tokenizer identifiers and hashes, architecture,
  quantization/execution strategy, compiler/toolchain, Rayon threads, and
  commit where relevant.
- **Performance:** workload, warmups/repetitions, CPU and power policy,
  process/thread pinning, every raw sample, and gain/neutral/regression/
  inconclusive classification. Do not report an uncontrolled benchmark.
- **Research interpretation:** use “decodable”, “causal effect”, and
  “behavioral correctness” only at the evidence levels that support them.
- **Handoff:** subsystem, primary reviewer, and backup/second reviewer for
  unsafe, kernel, schema, architecture, or performance work. Refresh the
  corresponding [`docs/audits/README.md`](docs/audits/README.md) entry when
  ownership, commands, or risks change.
- **Documentation:** update the nearest contract/usage page and an ADR for a
  durable architecture change. Keep generated artifacts out; use focused
  commits that can be reverted independently.

Reviewers may request a smaller patch, an explicit unsupported case, a new
fixture, or a new gate instead of a relaxed threshold. The goal is a repository
that a newcomer can build, audit, and extend without relying on one author’s
memory.

By contributing, you agree that your contribution is licensed under the
repository's MIT license.
