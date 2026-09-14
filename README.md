# ember

[![rust](https://img.shields.io/badge/rust-1.92-blue)](https://www.rust-lang.org)
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

After obtaining the pinned model and tokenizer, put both in the repository
root. Ember does not download them automatically. Build once, then capture
and verify a run (build and download time are additional):

```bash
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

## why this instrument exists

A probe can read a feature that the model's answer path does not use;
more fitting can even make transfer worse. That distinction motivates capture,
controlled interventions, and replayable evidence. Read
[the probe can read it. can the model use it?](https://voidwest.dev/research-notes/the-probe-can-read-it.html)
for the completed eight-ID diagnostic and its limits.

## research result: quantization-boundary localization

A deterministic validation wave across Qwen2.5-1.5B and Llama-3.2-1B at Q8,
Q6, and Q4 found no evidence of Arabic-selective quantization degradation
in the tested matrix.

The surviving result is methodological: Ember can localize rare
quantization-boundary failures causally. In validated cases, a single-layer
activation patch restored the quantized output, with the causal layer
preceding the visible divergence ramp. The observed mechanism was a
near-threshold decision flip rather than broad representational collapse.

| Model        | Layers | Causal locus |
|--------------|--------|--------------|
| Qwen2.5-1.5B | 28     | L7           |
| Llama-3.2-1B | 16     | L1           |

Validated on the qwen3/llama rows with completed golden checks; see
[docs/validation.md](docs/validation.md) for the full record.

## core documentation

- [CLI usage](docs/usage.md) and [experiment specs and bundles](docs/experiments.md)
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
  same bundle workflow. The native GUI is experimental and requires Vulkan
  plus X11/Wayland; it has no software-rendering fallback.
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
