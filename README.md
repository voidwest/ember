<p align="center"><img src="docs/assets/gui/icon.png" width="112" alt="Ember"></p>

# ember

[![rust](https://img.shields.io/badge/rust-1.92_headless_%C2%B7_1.98.1_GUI-blue)](https://www.rust-lang.org)
[![ci](https://github.com/voidwest/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/voidwest/ember/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Ember runs GGUF on CPU in a way you can inspect and replay. Capture states,
patch them, and pack the run into a bundle you can verify later.

**Inspectable GGUF forward pass → artifacts with checksums → replayable probes and interventions.**

Ember is not a llama.cpp replacement. llama.cpp remains the external
performance and correctness reference; Ember focuses on making the states,
interventions, and evidence behind a model's output inspectable.

**Golden path: Llama-3.2-1B-Instruct Q8_0**, with the model and matching
`tokenizer.json` pinned by SHA-256 in the [example specs](examples/experiments/README.md).
Other execution paths have different [validation coverage](docs/validation.md).

## five-minute workflow

Fetch the pinned model with `scripts/download_models.sh research-example`
and put it in the repository root; Ember does not download models
automatically. The matching `tokenizer.json` is already checked in there (the
[example instructions](examples/experiments/README.md) say where it comes
from). Build once, then capture and verify a run (build and download time are
additional):

```bash
# Inside the checkout, rust-toolchain.toml selects Rust 1.98.1, which the
# default GUI feature needs. A headless build on Rust 1.92 needs
# --no-default-features.
cargo build --release

target/release/ember experiment validate \
  examples/experiments/morphology-layerwise-capture.toml

target/release/ember experiment run \
  examples/experiments/morphology-layerwise-capture.toml

target/release/ember experiment verify runs/morphology-baseline
```

The result is `runs/morphology-baseline`: a verifiable bundle containing
provenance and hidden-state captures across all 16 layers for two token
selections. A mismatched model or tokenizer fails the pinned identity check.

Patch a state and compare the result:

```bash
target/release/ember experiment run \
  examples/experiments/morphology-intervention.toml

target/release/ember experiment compare \
  runs/morphology-baseline runs/morphology-intervention
```

The [full workflow](examples/experiments/README.md) adds restoration and
reproduction. On the reference machine, restoration compares bit-exact and
baseline reproduction reports `exact-semantic`. This is a workflow example,
not a reproduction of a paper result or a promise of cross-machine bit identity.

## what it has found

Ember exists to answer questions about model internals with evidence someone
else can replay. Results so far, each with its scripts and limits:

- **[Patching a quantized model: the sites that matter survive](https://voidwest.dev/research-notes/patching-survives-quantization.html).**
  Every candidate patched for real on F16, Q8_0, Q6_K and Q4_K_M builds of
  Llama-3.2-1B and Qwen2.5-1.5B (72 runs). The sites that carry a fact agree
  with F16 on every rung, within 8.5 points of recovered fraction. A
  pre-registered rank rule failed everywhere, because rank order among
  negligible effects is noise. Design, script and hashes:
  [research/quant-attribution](research/quant-attribution/README.md).
- **Quantization-boundary localization.** Across Qwen2.5-1.5B and
  Llama-3.2-1B at Q8, Q6 and Q4, about 500 deterministic runs found no
  Arabic-selective quantization degradation. Where a quantized model's output
  did flip, a single-layer activation patch restored it, one layer before the
  divergence ramp (Qwen2.5 L7 of 28, Llama L1 of 16): a near-threshold flip,
  not representational collapse. The summary is in
  [docs/validation.md](docs/validation.md); the full pilot record is not
  published.
- **[The probe can read it. Can the model use it?](https://voidwest.dev/research-notes/the-probe-can-read-it.html)**
  A probe recovered 6 of 8 held-out IDs from compressed features; the frozen
  receiver model could use only one of them, and more training erased that
  one.
- **[Finite is not intact](https://voidwest.dev/finite-is-not-intact.html).**
  In seven real GGUF files, no single bit flip in a scale word could reach
  Inf or NaN. Yet finite faults moved synthetic kernel outputs by thousands
  of times while passing Ember's finiteness checks. Kernel fixtures only, not
  an end-to-end accuracy result.

All notes, in English and Arabic: [voidwest.dev/research-notes](https://voidwest.dev/research-notes/).

## performance

Ember is an instrument first; llama.cpp is the performance reference. Decode
throughput for Llama-3.2-1B-Instruct on an Apple M1 Pro, 4 threads, CPU only,
median of three interleaved runs of 128 tokens (2026-10-02, Ember `ed5c1bc`,
llama.cpp `47c7869` with `-ngl 0`):

| Model | Ember | llama.cpp |
|---|---|---|
| Q8_0 | 57.8 tok/s | 65.8 tok/s |
| Q4_K_M | 67.5 tok/s | 88.6 tok/s |

Planned decode, which the benchmark runs, produces the same tokens as the
reference path that experiments use. At 8
threads both runtimes lose speed and scatter on this laptop under ordinary
desktop load, so those numbers are left out. Run `ember bench-decode` and
`llama-bench -ngl 0` on your own machine before relying on a ratio.

## the experiment console

Ember ships a native desktop console (`ember gui`) for running the same kind of
intervention without writing a spec. Setup is on the left and results on the
right; change one thing, run again, and pin an earlier result to compare the next
run against it. A result says when the settings have moved on since it was made.

![Ember workspace: setup on the left, results on the right, with a pinned earlier run compared against the current one](docs/assets/gui/results-overview.png)

![Layer-by-layer divergence with the pinned reference drawn as a dashed line behind the current run](docs/assets/gui/results-layers.png)

Silencing an early layer's output at the last prompt token turned "Paris. The
Eiffel Tower is located in Paris…" into "covered in a thick layer of fog…" on
Llama-3.2-1B-Instruct Q8_0; the Layers tab shows the change beginning exactly
at the layer that was touched. Zeroing a middle MLP instead leaves the words
unchanged while the internals still diverge. (One deterministic run on a
16-layer model, not a general claim.)

```bash
cargo run --release --bin ember -- gui
# macOS: a double-clickable app with an icon and a menu bar
scripts/bundle-macos.sh && open target/bundle/Ember.app
```

A **Sweep** runs a change at every layer and plots the curve (about twenty seconds for a 16-layer model). It has a built-in sample result that needs no model, dark and light themes, a
command palette (Cmd+K) and a Presentation mode. Runs are saved and can be
reopened. See the [console notes](docs/v06-gui.md) and the
[demo outline](docs/demo-outline.md).

## why this instrument exists

A probe can read a feature that the model's answer path does not use;
more fitting can even make transfer worse. That distinction motivates capture,
controlled interventions, and replayable evidence. Read
[the probe can read it. can the model use it?](https://voidwest.dev/research-notes/the-probe-can-read-it.html)
for the completed eight-ID diagnostic and its limits.

## core documentation

- [CLI usage](docs/usage.md) and [experiment specs and bundles](docs/experiments.md)
- [Support matrix](docs/support.md), [cancellation contract](docs/cancellation.md), and [agent trace schema](docs/trace-schema.md)
- [Model coverage](docs/models.md) and [numerical validation status](docs/validation.md)
- [Architecture](docs/architecture.md), [decode contract](docs/v04-execution-contract.md), and [research contract](docs/v05-research-contract.md)
- [Probing workflows](docs/research.md) and [dataset schemas](docs/dataset_pipeline.md)
- [Compatibility policy](docs/api-stability.md) and [release history](CHANGELOG.md)

The 1.0 priority is capture, intervention, bundles, and verification on a short,
explicitly validated model list. The project remains pre-1.0; an implemented
execution path does not imply completed numerical validation.

## related tools and research

These are optional extensions and separate research tracks around the CLI instrument:

- [Experiment consoles](docs/v06-gui.md): native and browser interfaces to the
  same bundle workflow. The GPUI Kit native GUI uses Metal on macOS and
  a GPU-backed X11/Wayland window on Linux. GUI builds use the pinned Rust
  1.98.1 toolchain; headless builds retain Rust 1.92 support.
- [Agent runtime](docs/agent-runtime.md): tool protocols and auditable traces,
  with their own tool-validation and approval boundaries.
- [EmberSEC](docs/embersec/README.md): hostile-artifact and quantized-fault
  research. The [frozen Phase I evaluation](research/embersec/comparative/README.md)
  retains its corpus, harnesses, results, and hashes as a separate evidence
  record; it is not a security certification of the evolving engine.
  [Phase V lab note: Finite Is Not Intact](https://voidwest.dev/finite-is-not-intact.html):
  the Inf result was a fixture; the operational failure is silent finite drift.
  The measured drift is from synthetic kernels, not a model accuracy drop.
- [Python bindings](bindings/python/README.md), [maintenance audits](docs/audits/README.md),
  and [external benchmarks](docs/external-benchmark.md)
- [Sarf Atlas](https://github.com/voidwest/sarf-atlas): the separate Arabic
  morphology workflow package.

## citation and license

See [CITATION.cff](CITATION.cff) and the [MIT license](LICENSE).
The MIT license covers the code. The Arabic research data derived from UD
Arabic-PADT under `data/arabic_morph_real/` is CC BY-NC-SA 3.0; see its
[notice](data/arabic_morph_real/NOTICE.md).

The actionable [road to 1.0](docs/road-to-1.0.md) tracks release gates and the GPUI Kit console migration.
