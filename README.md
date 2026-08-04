# ember

[![rust](https://img.shields.io/badge/rust-1.92-blue)](https://www.rust-lang.org)
[![ci](https://github.com/voidwest/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/voidwest/ember/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ember is a CPU-first Rust research layer for hidden-state extraction,
leakage-aware probing, and reproducible experiments over GGUF models: an
inspectable instrument with its own inference path for validation, not a
llama.cpp competitor. Research direction: Arabic morphology probing and
validation.

research write-up: https://voidwest.dev/ember

## headline finding: causal localization of quantized-boundary failures

A 2026-08 validation wave (Qwen2.5-1.5B and Llama-3.2-1B across Q8/Q6/Q4,
~500 deterministic runs) found **no Arabic-selective quantization
degradation**: a null at every precision/family combination tested. The
robust output is the **causal-localization toolchain**: single-layer
activation patches restore quantized-boundary failures, with the causal
locus one layer before the divergence ramp (qwen L7/28, llama L1/16), and
the mechanism is near-threshold flips, where quantization noise crosses
the model's smallest decision margins.

**Validation caveat.** "Supported" in the model tables means an execution
path exists, not numerical trustworthiness. Several architectures still
have `pending` golden-logit or activation-reference status in
[docs/validation.md](docs/validation.md); the causal-localization result
above is validated on the rows with completed golden checks (qwen3/llama
families) and should not be cited as validated across all architectures.
The full pilot record (items, results, reports) is on the local
`pilot-001` branch under `research/pilots/arabic_quantization_001/`.

## v0.4: plan-driven decode

v0.4 (tag v0.4.0) builds an immutable per-model `ExecutionPlan` once after
load and runs single-token decode through a plan interpreter
(`--execution reference|planned|planned-fused`), with an aligned scratch
arena, a frozen fusion set (F1-F5), and a column-parallel K-quant matvec.
Decode on the four primary combos (Llama-3.2-1B and Qwen2.5-1.5B x
Q4_K_M/Q6_K) runs roughly 2.0-2.7x the v0.3 reference. Gates A-G are
pre-registered in
[docs/v04-execution-contract.md](docs/v04-execution-contract.md);
`ember inspect-plan` prints the plan and `bench-decode --execution`
measures it. Artifacts: `artifacts/benchmark-v04/2026-08-04/`.

## v0.5: reproducible experiment bundles

v0.5 (tag v0.5.0) ships the reproducible experiment workflow:
`ember experiment validate|run|inspect|verify|compare|reproduce|tokenize`
over `ember.experiment.v1` specs (strict TOML) producing deterministic
`ember.bundle.v1` bundles with semantic/payload identity, offline
verification, and atomic staging. A reference morphology example lives
under `examples/experiments/` (three specs: layerwise baseline, a
layer-8 intervention, and restoration). Run it with a
Llama-3.2-1B-Instruct-Q8_0 model present in the repo root:

```bash
ember experiment validate examples/experiments/morphology-layerwise-capture.toml
ember experiment run examples/experiments/morphology-layerwise-capture.toml
ember experiment verify runs/morphology-baseline
ember experiment run examples/experiments/morphology-intervention.toml
ember experiment compare runs/morphology-baseline runs/morphology-intervention
```

The restoration leg reproduces the baseline bit-exact. Gates A-I are in
[docs/v05-research-contract.md](docs/v05-research-contract.md); the Gate H
matrix is in `artifacts/benchmark-v05/`.

## quick start

```bash
cargo build --release          # debug builds have runtime asserts that throw
target/release/ember --arch qwen3 --model Qwen3-0.6B-Q8_0.gguf \
  --tokenizer tokenizer-qwen3.json \
  --prompt "The capital of France is" --max-tokens 8 --temperature 0
```

The v0.2 research workflow (capture activations -> intervene -> compare ->
patch -> verify restoration) runs from `scripts/research_example_capture_patch.sh`;
see [docs/usage.md](docs/usage.md) and [docs/validation.md](docs/validation.md).

## docs

- [docs/usage.md](docs/usage.md) - CLI flags, subcommands, modes, benchmarks, testing
- [docs/validation.md](docs/validation.md) - validation ladder, evidence status, pilot wave
- [docs/models.md](docs/models.md) - supported models and quantization (incl. K-quants)
- [docs/architecture.md](docs/architecture.md) - internals, design notes, optimization
- [docs/experiments.md](docs/experiments.md) - v0.5 experiment spec and bundle workflow
- [docs/v04-execution-contract.md](docs/v04-execution-contract.md) - plan-driven decode, gates A-G
- [docs/v05-research-contract.md](docs/v05-research-contract.md) - experiment workflow, gates A-I
- [docs/research.md](docs/research.md) - Arabic morphology dataset pipeline and probing
- [docs/dataset_pipeline.md](docs/dataset_pipeline.md) - dataset input/output schemas

Agent context: `AGENT.md` (research rules, validation ladder) and
`AGENTS.md` (current state, branches, gotchas).

## Sarf Atlas

Sarf Atlas has moved to its own repository:
https://github.com/voidwest/sarf-atlas

Use the standalone package for backend-agnostic Arabic morphology workflow
scaffolding:

```bash
pip install sarf-atlas
```

## license

MIT, see [LICENSE](LICENSE).
