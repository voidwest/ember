# ember

[![rust](https://img.shields.io/badge/rust-1.92-blue)](https://www.rust-lang.org)
[![ci](https://github.com/voidwest/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/voidwest/ember/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ember is a CPU-first Rust research layer for hidden-state extraction,
leakage-aware probing, and reproducible experiments over GGUF models, an
inspectable instrument with its own inference path for validation, not a
llama.cpp competitor. Research direction: Arabic morphology probing and
validation.

research write-up: https://voidwest.dev/ember

## headline finding: causal localization of quantized-boundary failures

A 2026-08 validation wave (Qwen2.5-1.5B and Llama-3.2-1B across Q8/Q6/Q4,
~500 deterministic runs) found **no Arabic-selective quantization
degradation**, a null at every precision/family combination tested. The
robust output is the **causal-localization toolchain**: single-layer
activation patches restore quantized-boundary failures, with the causal
locus one layer before the divergence ramp (qwen L7/28, llama L1/16), and
the mechanism is near-threshold flips quantization noise crosses the
model's smallest decision margins.

**Validation caveat.** "Supported" in the model tables means an execution
path exists, not numerical trustworthiness. Several architectures still
have `pending` golden-logit or activation-reference status in
[docs/validation.md](docs/validation.md); the causal-localization result
above is validated on the rows with completed golden checks (qwen3/llama
families) and should not be cited as validated across all architectures.
The full pilot record (items, results, reports) is on the local
`pilot-001` branch under `research/pilots/arabic_quantization_001/`.

## quick start

```bash
cargo build --release          # debug builds have runtime asserts that throw
target/release/ember --arch qwen3 --model Qwen3-0.6B-Q8_0.gguf \
  --tokenizer tokenizer-qwen3.json \
  --prompt "The capital of France is" --max-tokens 8 --temperature 0
```

The v0.2 research workflow (capture activations → intervene → compare →
patch → verify restoration) runs from `scripts/research_example_capture_patch.sh`;
see [docs/usage.md](docs/usage.md) and [docs/validation.md](docs/validation.md).

## docs

- [docs/usage.md](docs/usage.md) — CLI flags, subcommands, modes, benchmarks, testing
- [docs/validation.md](docs/validation.md) — validation ladder, evidence status, pilot wave
- [docs/models.md](docs/models.md) — supported models and quantization (incl. K-quants)
- [docs/architecture.md](docs/architecture.md) — internals, design notes, optimization
- [docs/research.md](docs/research.md) — Arabic morphology dataset pipeline and probing
- [docs/dataset_pipeline.md](docs/dataset_pipeline.md) — dataset input/output schemas

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
