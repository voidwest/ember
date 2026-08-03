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

By contributing, you agree that your contribution is licensed under the
repository's MIT license.
