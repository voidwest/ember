# ember

[![rust](https://img.shields.io/badge/rust-1.92-blue)](https://www.rust-lang.org)
[![ci](https://github.com/voidwest/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/voidwest/ember/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ember is a CPU-first Rust research layer for hidden-state extraction,
leakage-aware probing, and reproducible experiments over GGUF models: an
inspectable instrument with its own inference path for validation. Current
research focus: Arabic morphology, probing validity, and quantized-inference
failure localization.

research write-up: https://voidwest.dev/ember

## capabilities

- inspectable CPU inference over GGUF
- hidden-state capture and semantic interventions
- compressed-resident Q4_K/Q6_K execution
- plan-driven decode
- deterministic, verifiable experiment bundles
- native and browser experiment consoles (`ember gui`, `ember web-gui`)
- agentic tool-calling runtime with auditable research traces
- Arabic morphology and quantization research workflows

Ember is not a llama.cpp throughput competitor. llama.cpp remains the
external performance and correctness reference; Ember prioritizes
inspectability, intervention semantics, and reproducible research artifacts.

## five-minute workflow

Build once, then drive the reproducible experiment pipeline:

```bash
cargo build --release          # debug builds enable expensive runtime
                               # assertions and are not intended for benchmarking

target/release/ember experiment validate \
  examples/experiments/morphology-layerwise-capture.toml

target/release/ember experiment run \
  examples/experiments/morphology-layerwise-capture.toml

target/release/ember experiment verify \
  runs/morphology-baseline
```

The example spec pins `Llama-3.2-1B-Instruct-Q8_0.gguf` (SHA-256 recorded,
so a different file fails closed instead of producing unreproducible
numbers); keep that model in the repo root. Continue with the intervention
leg to see the compare workflow:

```bash
target/release/ember experiment run \
  examples/experiments/morphology-intervention.toml

target/release/ember experiment compare \
  runs/morphology-baseline runs/morphology-intervention
```

The restoration leg reproduces the baseline bit-exact. Full walkthrough:
`examples/experiments/README.md`.

## GUI usage (v0.6)

Two offline consoles drive the same v0.5 experiment pipeline: every action
builds an `ember.experiment.v1` spec, resolves it through the standard
validation gate, and runs it with one resident model serving repeated
baseline / intervention / restore runs. Bundles are written and
self-verified exactly as `ember experiment run`.

```bash
target/release/ember gui          # native single-window console (gpui,
                                  # Vulkan-accelerated; dark theme by
                                  # default, light/dark toggle in the header)
target/release/ember web-gui      # browser console on http://127.0.0.1:8337/
                                  # (light/dark toggle, defaults to the
                                  # system preference)
```

The native console needs a Vulkan-capable GPU and an X11/Wayland session
(no software-rendering fallback); the browser console only needs a browser.
Arabic input/output renders correctly in both: the native console shapes
RTL text with its embedded Noto fonts (offline, identical on any machine);
the browser console uses `dir="auto"` per field. Details:
[docs/v06-gui.md](docs/v06-gui.md).

### ordinary inference

Plain generation works too. `--arch auto` reads `general.architecture` from
the GGUF (the default), and the tokenizer resolves automatically:

```bash
target/release/ember --arch auto --model Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" --max-tokens 8 --temperature 0
```

### agentic execution (v0.6.7)

The model can request a tool through its own structured protocol; Ember
validates the call, executes it deterministically, reinjects the result
into the same session, and continues until a final answer - with every
step in an auditable JSONL trace:

```bash
target/release/ember agent run \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf --tokenizer tokenizer.json \
  --protocol llama3 --tools lookup --fixture riyadh="41 C" \
  --prompt "Use the available tool to tell me the fixture temperature in Riyadh." \
  --trace-out run.jsonl

target/release/ember trace inspect run.jsonl
```

Deterministic built-in tools only (arithmetic, fixtures, artifacts,
sandboxed file reads); unknown tools fail closed, hard limits bound every
run, and cancellation never leaves session state half-committed.
`ember trace diff|replay|report` compare runs, verify recorded tool calls
offline against their digests, and render self-contained HTML reports.
Details: [docs/agent-runtime.md](docs/agent-runtime.md).

## architecture

```
GGUF --> loader --> packed K-quant / f32 tensors
                        |
                        v
              ExecutionPlan (built once per model)
                        |
                        v
              plan interpreter --> logits
                        |
              capture hooks, interventions, patches
```

Ember loads GGUF directly, keeps quantized tensors packed, builds an
immutable execution plan once per model, and runs decode through a plan
interpreter, with capture hooks and interventions layered on the same path,
so the research facilities measure the exact numerics that produced the
output. Details: [docs/architecture.md](docs/architecture.md).

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

## version milestones

- **v0.3**: compressed-resident Q4_K/Q6_K execution; packed tensors stay
  mmap-backed with scalar/AVX2 kernels; the Q8_0 native path is untouched.
- **v0.4**: immutable per-model execution plans; plan-driven single-token
  decode (`--execution reference|planned|planned-fused`), aligned scratch
  arena, frozen fusion set F1-F5, column-parallel K-quant matvec
  (~2.0-2.7x the v0.3 reference). Gates A-G:
  [docs/v04-execution-contract.md](docs/v04-execution-contract.md).
- **v0.5**: deterministic experiment bundles; `ember.experiment.v1` specs
  producing `ember.bundle.v1` bundles with semantic/payload identity and
  offline verification (introduced in v0.5.0; current patch release v0.5.1).
  Gates A-I: [docs/v05-research-contract.md](docs/v05-research-contract.md).
- **v0.6**: experiment consoles - `ember gui` (native gpui/Vulkan window,
  dark theme with light/dark toggle) and `ember web-gui` (single-page
  browser console with a light/dark toggle); the v0.5 run path was split
  into `prepare_run` / `execute_prepared` so one resident model serves
  repeated runs. [docs/v06-gui.md](docs/v06-gui.md).
- **v0.6.7**: agentic execution layer - structured tool calls behind a
  model-family protocol boundary (Qwen2.5, Llama 3.x), strict validation,
  approval gating, crash-tolerant research traces with provenance and
  hashed artifacts, `ember agent` / `ember trace` CLI.
  [docs/agent-runtime.md](docs/agent-runtime.md).

## validation status

"Supported" means an execution path exists; it does not imply completed
numerical validation. See [docs/validation.md](docs/validation.md) for
per-architecture golden-logit and activation-reference status.

## documentation

- [docs/usage.md](docs/usage.md) - CLI flags, subcommands, modes, benchmarks, testing
- [docs/validation.md](docs/validation.md) - validation ladder, evidence status, pilot wave
- [docs/models.md](docs/models.md) - supported models and quantization (incl. K-quants)
- [docs/architecture.md](docs/architecture.md) - internals, design notes, optimization
- [docs/experiments.md](docs/experiments.md) - v0.5 experiment spec and bundle workflow
- [docs/v04-execution-contract.md](docs/v04-execution-contract.md) - plan-driven decode, gates A-G
- [docs/v05-research-contract.md](docs/v05-research-contract.md) - experiment workflow, gates A-I
- [docs/v06-gui.md](docs/v06-gui.md) - native + browser experiment consoles (v0.6)
- [docs/agent-runtime.md](docs/agent-runtime.md) - agentic runtime, tool calls, research traces
- [docs/api-stability.md](docs/api-stability.md) - Rust/CLI compatibility policy, MSRV, and release rules
- [docs/audits/README.md](docs/audits/README.md) - subsystem ownership, recurring audits, and handoffs
- [docs/adr/0001-architecture-bets.md](docs/adr/0001-architecture-bets.md) - architecture decisions and revisit triggers
- [docs/fuzzing.md](docs/fuzzing.md) - model-free hostile-input parser fuzzing
- [docs/external-benchmark.md](docs/external-benchmark.md) - neutral third-party benchmark submission pathway
- [docs/research.md](docs/research.md) - Arabic morphology dataset pipeline and probing
- [docs/dataset_pipeline.md](docs/dataset_pipeline.md) - dataset input/output schemas

## Sarf Atlas

Sarf Atlas has moved to its own repository:
https://github.com/voidwest/sarf-atlas

Use the standalone package for backend-agnostic Arabic morphology workflow
scaffolding:

```bash
pip install sarf-atlas
```

## citation and license

Citing: see [CITATION.cff](CITATION.cff). License: MIT, see
[LICENSE](LICENSE).
