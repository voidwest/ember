# Ember usage

CLI and workflow reference, moved from the top-level README
(2026-08) so the README stays a short orientation.

## usage

```bash
cargo run --release -- --model gpt2.Q8_0.gguf --prompt "hello"
```

Backend-ready hidden-state extraction uses a declarative config. Save the
minimal example below as `extract.toml`, then run:

```bash
cargo run --release -- extract --backend native --config extract.toml
```

Minimal config shape:

```toml
run_id = "qwen3_word_probe_smoke"
model_path = "model.gguf"
architecture = "llama"
backend = "native"
prompt_template = "Analyze the word: {word}"
input_jsonl_path = "data/prompts.jsonl"
output_dir = "runs"
layers = [0, 8, 16]
token_position = "word_final_subtoken"
batch_size = 1
dtype = "f32"
output_format = "npy"
```

The run writes the frozen Ember artifact contract under
`runs/qwen3_word_probe_smoke/`: `manifest.json`, `samples.jsonl`,
`tokenization.jsonl`, `positions.jsonl`, per-layer `layers/layer_XXXX.npy`
files, `checksums.json`, and `report.json`. See
[docs/artifact_contract.md](docs/artifact_contract.md).

Validate a single artifact run with `cargo run -- validate-run <run-dir>`.
Backend-to-backend comparisons use `validate-backends`, and external parity
audits use `gguf-parity-tools`; see
[docs/backend_validation.md](docs/backend_validation.md).

`llama-cpp` config validation is wired, but hidden-state extraction still needs
the external patched/custom llama.cpp binary integration. That backend must
write the same artifact contract as `native`.

The external-process backend is available as backend plumbing:

```bash
cargo run --release -- extract \
  --backend llama-cpp-external \
  --llama-bin ./build/bin/llama-ember-extract \
  --model ./models/qwen3-0.6b-q8_0.gguf \
  --samples ./data/samples.jsonl \
  --out runs/test-qwen-llama-backend
```

For now `llama-cpp-external` supports tokenization-only smoke plumbing when
paired with an external helper. Hidden-state layer requests are rejected until
the patched extractor contract is implemented.

### flags

| flag | default | description |
|------|---------|-------------|
| `-m`, `--model` | `gpt2.Q8_0.gguf` | path to gguf model file |
| `--arch` | `gpt2` | model architecture: `gpt2`, `llama`, `qwen3`, or `gemma4` |
| `--tokenizer` | arch-dependent | path to tokenizer.json (`tokenizer-gpt2.json` for gpt-2, `tokenizer.json` for llama/qwen2.5, `tokenizer-qwen3.json` for qwen3, `tokenizer-gemma4.json` for gemma 4) |
| `-p`, `--prompt` | `The` | text prompt to complete |
| `-n`, `--max-tokens` | `20` | tokens to generate |
| `--max-seq-len` | model metadata | cap usable context length below the model metadata value |
| `-t`, `--temperature` | `0.8` | sampling temp (0 = greedy) |
| `--top-k` | (none) | top-k sampling |
| `--top-p` | (none) | nucleus sampling |
| `-i`, `--interactive` | (none) | repl mode after first prompt |
| `--demo` | (none) | fixed prompts with timing and deterministic output |
| `--delay-ms` | `0` | delay between tokens in demo mode (0 = instant) |
| `--benchmark` | (none) | print prefill/decode timing to stderr |
| `--zero-layer-output` | (none) | experimental `LAYER:attention|mlp|layer` intervention for normal LLaMA, Qwen3, or Gemma 4 generation |
| `--activation-stats` | (none) | observation-only experiment that writes activation norms and fingerprints to JSON |
| `--trace` | (none) | enable per-operation execution tracing (`ops`) |
| `--trace-out` | stderr | write trace JSON to a file |
| `--trace-values` | `none` | optionally record output norms and fingerprints (`summary`) |
| `--trace-run-metadata` | (none) | attach CPU, thread, governor, and commit metadata to traces |
| `--dump-logits` | (none) | write last-prompt logits for `--prompt` to `.npy` and exit |
| `--dump-layers` | (none) | write Gemma 4 last-token layer states as flat little-endian f32 |
| `--write-run-manifest` | (none) | write a reproducibility manifest with model/tokenizer hashes, git commit, compiler, Rayon, and CPU feature data |
| `--record-model-sha256` | (none) | compute and record model file sha256 in probe metadata |
| `--dump-gguf-metadata` | (none) | write parsed GGUF metadata to JSON |
| `--probe` | (none) | run probe mode: extract hidden states from each block |
| `--probe-stimuli` | `stimuli/nonce_root_pattern_surface.json` | path to stimuli json for probe mode |
| `--probe-output` | `data/activations.npy` | output path for probe activations (.npy) |
| `--probe-template` | `en_surface_probe` | stimulus prompt key to probe; surface-only prompts are the representation-probing default |
| `--probe-templates` | (none) | comma-separated prompt template keys for batch probe extraction |
| `--probe-position` | `last` | hidden-state position to pool: `last`, `root`, `pattern`, or `prompt_mean` |
| `--probe-positions` | (none) | comma-separated hidden-state positions for batch probe extraction |
| `--probe-output-dir` | `data/probe_matrix` in batch mode | output directory for batch probe extraction |
| `--probe-output-prefix` | `probe` | output filename prefix for batch probe extraction |
| `--probe-generate-tokens` | `16` | continuation length for probe behavioral scoring |
| `--probe-limit` | (none) | cap probe extraction to the first N stimuli for smoke tests |

The historical `en_zero`, `en_one`, `ar_zero`, and `ar_one` templates print
the target root and pattern in the prompt. They are suitable for composition
behavior checks or explicitly named positive controls, not for evidence that a
representation independently encodes those labels. Use `en_surface_probe` or
`ar_surface_probe` for label-free representation probes.

### research experiments

### v0.2 research experiments: capture, patch, compare

v0.2 makes experiment runs first-class research artifacts:

```bash
# capture selected activations into an artifact (manifest.json + tensors/)
ember --arch qwen3 --model Qwen3-0.6B-Q8_0.gguf --tokenizer tokenizer-qwen3.json \
  --prompt "The capital of France is" --max-tokens 8 --temperature 0 \
  --capture-activations capture.toml

# replace one live activation with a captured tensor (unambiguous source)
ember --arch qwen3 --model Qwen3-0.6B-Q8_0.gguf --tokenizer tokenizer-qwen3.json \
  --prompt "The capital of France is" --max-tokens 8 --temperature 0 \
  --activation-patch runs/baseline/manifest.json \
  --patch-target 4:after-mlp:prefill --patch-target 4:after-mlp:decode:5

# compare two artifacts record-by-record (deterministic, strict alignment)
ember compare-artifacts --left runs/a/manifest.json --right runs/b/manifest.json
```

Capture is a run-level facility that rides alongside the single experiment
(or runs alone); patching is one built-in experiment. Both are documented in
[docs/activation-artifacts.md](docs/activation-artifacts.md) and
[docs/activation-patching.md](docs/activation-patching.md). The complete
capture -> intervene -> compare -> patch -> restore workflow is
`scripts/research_example_capture_patch.sh`, which enforces the frozen
restoration criterion: a patched run's captured logits must be bit-identical
to the baseline's.

Ember v0.1 has an intentionally unstable, statically compiled experiment API.
It intentionally ships with two proof points: one observation experiment and
one intervention experiment.

To observe execution without changing it, record activation norms and
fingerprints:

```bash
target/release/ember \
  --arch qwen3 \
  --model Qwen3-0.6B-Q8_0.gguf \
  --tokenizer tokenizer-qwen3.json \
  --prompt "The capital of France is" \
  --max-tokens 4 \
  --temperature 0 \
  --activation-stats activation-stats.json

jq '.records[] | select(.stage == "after_layer") |
    {phase, layer_index, sequence_length, l2_norm, fingerprint}' \
  activation-stats.json
```

The artifact is observation-only: generated output remains numerically
identical, although scanning activations and writing JSON adds work. To perform
an intervention instead, zero one selected layer contribution:

```bash
target/release/ember \
  --arch qwen3 \
  --model Qwen3-0.6B-Q8_0.gguf \
  --tokenizer tokenizer-qwen3.json \
  --prompt "The capital of France is" \
  --temperature 0 \
  --zero-layer-output 4:mlp
```

Experiment notices and summaries go to stderr; generated text keeps its normal
stdout format. The two options conflict because Ember supports one active
experiment per run. Active experiments do not currently participate in
probes, hidden-state extraction, logits/layer dumps, demos, or benchmark
subcommands. Dynamic third-party plugin loading and multiple simultaneous
experiments are intentionally unsupported. See
[docs/experiments.md](docs/experiments.md) for the artifact schema, hook
lifecycle, mutation boundaries, and guarantees.

### decode benchmark

For a model-only decode comparison with `llama-bench`, build once and exclude
loading, prefill, tokenization, and sampling on both sides:

```bash
cargo build --release
RAYON_NUM_THREADS=4 target/release/ember bench-decode \
  --model models/gemma-4-E2B-it.Q8_0.gguf \
  --arch gemma4 \
  --tokens 128 \
  --warmups 2 \
  --repetitions 5
```

The command creates a fresh cache per repetition, performs one untimed seed
evaluation, then times deterministic single-token evaluations. It emits JSON
with every timing sample, median throughput, thread count, CPU metadata, model
size, commit, and explicit timing exclusions. Use `--token-id` to change the
fixed input token and `--max-seq-len` to cap the benchmark context.

`bench-lifecycle` measures the weight-layout side instead of decode throughput:
it times generic vs packed gate/up dispatch across lifecycle strategies on
short Llama-family batches and records peak process residency alongside the
timings. See `docs/packed-q8-lifecycle.md`.

### reproducible release benchmarks

The release measurements use a fixed power policy, one worker per physical
core, warmups, retained raw samples, and alternating revision order such as
`ABBA BAAB`. On Intel P-state systems, verify both the selected power profile
and `energy_performance_preference`; the exposed scaling-governor name may
remain `powersave`.

```bash
powerprofilesctl set performance

taskset -c 0-3 env RAYON_NUM_THREADS=4 target/release/ember bench-decode \
  --model Qwen3-0.6B-Q8_0.gguf \
  --arch qwen3 \
  --tokens 32 \
  --warmups 2 \
  --repetitions 3 \
  --max-seq-len 128 \
  > qwen-decode.json

taskset -c 0-3 env RAYON_NUM_THREADS=4 target/release/ember \
  --model Qwen3-0.6B-Q8_0.gguf \
  --tokenizer tokenizer-qwen3.json \
  --arch qwen3 \
  --prompt "Explain why CPU cache locality matters in transformer inference." \
  --max-seq-len 128 \
  --max-tokens 1 \
  --temperature 0 \
  --benchmark

powerprofilesctl set balanced
```

Repeat both commands with clean binaries from revisions A and B in
counterbalanced order. See
[the cleanup validation report](docs/cleanup-audit-validation.md) and
[the packed Q8 note](docs/q8-packed-gate-up.md) for complete protocols,
correctness gates, raw-shape operator commands, and external llama.cpp
reference numbers.

### execution tracing

Trace the native generation path to stderr, or add `--trace-out trace.json` for
a reusable report:

```bash
RAYON_NUM_THREADS=1 target/release/ember \
  --arch qwen3 \
  --model Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" \
  --max-tokens 1 \
  --temperature 0 \
  --trace ops \
  --trace-values summary \
  --trace-run-metadata
```

Tracing is thread-local; use one Rayon thread when a complete per-operation
decode trace is more important than throughput. See `TRACE.md` for the event
schema, recorded operation types, and caveats.

### subcommands

| command | purpose |
|---------|---------|
| `extract` | write the versioned artifact contract with the native backend, or exercise the current llama.cpp backend plumbing |
| `native-logits-reference` | write a native logits-only artifact run from an extraction config |
| `validate-run` | validate checksums, metadata, row counts, and optional layer shards for one run directory |
| `validate-backends` | compare existing native and external artifact runs |
| `bench-decode` | measure model-only single-token decode and emit JSON |
| `bench-lifecycle` | time the Llama packed vs generic Q8 weight lifecycle across strategies and report process residency |

### demo mode

```bash
cargo run --release -- --demo
```

runs through a fixed set of prompts using greedy sampling (temperature 0)
for deterministic, repeatable output. useful for screen recordings
(`asciinema`, `script`, terminal capture) and benchmarking.

each prompt reports its completion, token counts, and per-phase timing.
a summary table at the end shows aggregate throughput across all prompts.

### smoke runs

Use the smoke wrapper for local GGUF checks instead of hand-running
`/usr/bin/time -v`. It records the command, model/tokenizer paths, arch, prompt,
generated token count, commit hash, host, date, raw generation text, benchmark
timing if parsed, and peak RSS under `logs/`.

```bash
python3 scripts/run_smoke.py --model qwen3_06b --tokens 32
```

Run every configured model that is available locally:

```bash
python3 scripts/run_smoke.py --all --tokens 32 --continue-on-fail
```

Inspect commands without running inference:

```bash
python3 scripts/run_smoke.py --all --dry-run
```

Smoke output is structural validation only. `smoke_pass` means the Ember command
exited 0 and produced output; `smoke_pass_generation_warning` means it exited 0
but a simple repetition heuristic, or a known experimental config marker, flagged
the raw generated text. `smoke_fail` means the command returned nonzero or did
not produce output. Smoke tests validate model loading, tokenization, generation
execution, benchmark logging, and memory use. They are not quality benchmarks.

Quality validation requires golden-logit or reference checks against trusted
implementations for the exact model, tokenizer, prompt, and quantization path.
TPS comparisons against llama.cpp require matched hardware, model, quantization,
prompt length, decode length, thread settings, and repeated runs. Qwen2.5 is
currently experimental in Ember: it is routed through the `qwen3` path, has shown
degenerate smoke generation, and should not be treated as quality-compatible
until reference checks pass.

Build a Markdown benchmark table from existing smoke summaries:

```bash
python3 scripts/summarize_smokes.py --logs logs --output data/smoke_benchmark_table.md
```

Benchmark decode throughput across Rayon thread counts:

```bash
python3 scripts/benchmark_threads.py \
  --model qwen3_06b:Qwen3-0.6B-Q8_0.gguf \
  --arch qwen3 \
  --tokenizer tokenizer-qwen3.json \
  --max-seq-len 128 \
  --threads 1,2,4,8 \
  --tokens 16 \
  --output data/thread_benchmarks.json
```

The script sets `RAYON_NUM_THREADS` for each run and parses Ember's
`--benchmark` output. This is the preferred way to compare the parallel
attention and q8 decode paths because small prompts and large vocab-head
projections scale differently.

### golden-logit validation

Ember can dump the final-position logits for one prompt:

```bash
cargo run --release -- \
  --arch llama \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --dump-logits data/golden/llama32_1b_ember_logits.npy

cargo run --release -- \
  --arch qwen3 \
  --model Qwen3-0.6B-Q8_0.gguf \
  --tokenizer tokenizer-qwen3.json \
  --prompt "The capital of France is" \
  --dump-logits data/golden/qwen3_06b_ember_logits.npy
```

`--dump-logits` also writes `*_metadata.json` with Ember's token audit. The
trusted reference must provide matching token IDs, either as a reference
metadata sidecar or as a combined token audit JSON.

Compare Ember logits to a trusted `.npy` reference:

```bash
python3 probes/check_golden_logits.py \
  --ember data/golden/qwen3_06b_ember_logits.npy \
  --reference data/golden/qwen3_06b_reference_logits.npy \
  --metadata data/golden/qwen3_06b_ember_logits_metadata.json \
  --reference-metadata data/golden/qwen3_06b_reference_logits_metadata.json \
  --label qwen3_06b \
  --tokenizer tokenizer-qwen3.json \
  --top-k 10 \
  --topk-overlap-threshold 0.8 \
  --output data/golden/qwen3_06b_golden_report.json
```

Build compact JSON and Markdown summaries from all golden reports:

```bash
python3 probes/golden_summary.py

python3 probes/golden_summary.py \
  --glob 'data/golden/*golden_report.json' \
  --output-json data/golden/golden_summary.json \
  --output-md data/golden/golden_summary.md
```

The report classifies runs as `golden_pass`, `golden_warn`, or `golden_fail`
using shape checks, top-1 agreement, top-k overlap, and any configured numerical
thresholds (`--max-diff-threshold`, `--mean-diff-threshold`,
`--topk-overlap-threshold`). Do not claim quality parity until these reports pass
for the exact artifacts being compared. `golden_summary.py` copies
classification/status fields from source reports only; if a report omits them,
the summary records `missing` rather than inferring pass/fail from metrics.

Reference logits can come from Hugging Face Transformers by loading the matching
model/tokenizer, running the same prompt with no generation, taking
`outputs.logits[:, -1, :]`, converting to `float32`, and saving with
`numpy.save`. llama.cpp is also acceptable if a local, audited logit-dump command
or patch is available for the same model and prompt. An exact llama.cpp logit
dump command is pending in this repo; do not substitute normal generated text for
golden-logit validation.

### interactive mode

```bash
cargo run --release -i
```

commands inside the repl: `/quit`, `/help`, `/stats`.

### probe mode

```bash
cargo run --release -- --probe --model Llama-3.2-1B-Instruct-Q8_0.gguf --arch llama
```

feeds each stimulus from the stimuli json file through the model and collects
pooled per-layer hidden states at the selected prompt position. saves a 3d
`.npy` array `(n_stimuli, n_layers, embed_dim)` plus `_correctness.json` and
`_metadata.json` sidecars with next-token predictions, generated continuations,
match results, and the exact prompt template, position, model, shape, and token
selections used.
works with gpt-2, llama/qwen-family models, and dense text-only gemma 4
models through the `ForwardModel` trait.

batch extraction lets one model load produce a full prompt/position matrix:

```bash
cargo run --release -- \
  --arch llama \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --probe \
  --probe-stimuli stimuli/nonce_root_pattern_surface.json \
  --probe-output-dir data/matrix \
  --probe-output-prefix llama1b \
  --probe-templates en_surface_probe,ar_surface_probe \
  --probe-positions last,root,pattern,prompt_mean \
  --probe-generate-tokens 1
```

when several positions are requested for the same template, extraction groups
them together. the prompt is tokenized once, the model forward pass runs once,
and pooled outputs are written separately for each requested position. this
keeps the existing file layout (`*_last_activations.npy`,
`*_root_activations.npy`, etc.) while avoiding redundant forwards across
`last`, `root`, `pattern`, and `prompt_mean`. probe extraction also pools
hidden states during the forward pass, so it no longer stores full per-layer
sequence activations just to average a selected token span.

the matrix runner wraps that extraction and then runs probes, cca, rsa, and
divergence for each emitted activation file:

```bash
python probes/run_probe_matrix.py \
  --model 1b:Llama-3.2-1B-Instruct-Q8_0.gguf \
  --templates en_surface_probe ar_surface_probe \
  --positions last root \
  --jobs 2 \
  --generate-tokens 1 \
  --dry-run
```

`--jobs` controls parallel post-extraction analysis bundles. each
template/position bundle still runs its own probe -> CCA -> RSA -> divergence
steps in order, but independent bundles can run concurrently after extraction
finishes. extraction itself remains serial per model to avoid multiplying GGUF
memory use.

canonical smoke probe:

```bash
cargo run --release -- \
  --arch qwen3 \
  --model Qwen3-0.6B-Q8_0.gguf \
  --probe \
  --probe-limit 5 \
  --probe-output data/qwen3_smoke_activations.npy \
  --probe-generate-tokens 1
```

gemma 4 uses the same probe pipeline:

```bash
cargo run --release -- \
  --arch gemma4 \
  --model models/gemma-4-E2B-it.Q8_0.gguf \
  --tokenizer tokenizer-gemma4.json \
  --probe \
  --probe-stimuli stimuli/nonce_root_pattern_surface.json \
  --probe-output data/gemma4_activations.npy \
  --probe-generate-tokens 1
```

the `probes/` directory contains python scripts for downstream analysis:

| script | purpose |
|--------|---------|
| `train_linear_probe.py` | logistic linear, SGD linear, and small-MLP probes with task-specific CV splits, sparse label filtering, control tasks, and selectivity |
| `cca_analysis.py` | canonical correlation analysis, layer similarity matrices |
| `rsa_analysis.py` | representational similarity analysis, distance metrics |
| `divergence_analysis.py` | correct-vs-incorrect hidden state divergence |
| `tokenizer_fertility.py` | subword tokenization comparison across tokenizers |
| `plot_results.py` | visualization: generic probe accuracy/selectivity, CCA/RSA heatmaps, cross-model comparison, fertility |
| `plot_root_scale_comparison.py` | compact root-accuracy comparison across Llama model scales |
| `run_probe_matrix.py` | repeatable model/template/position probe matrix runner |
| `build_conllu_benchmark.py` | convert CoNLL-U morphology annotations into token-level benchmark JSON |
| `extract_hf_encoder.py` | optional Hugging Face encoder hidden-state extractor |
| `mdl_probe.py` | data-efficiency / MDL-style probing curves |
| `run_benchmark.py` | manifest-driven extraction + probe + MDL + RSA benchmark runner |
| `render_benchmark_report.py` | render `benchmark_summary.json` into a conservative Markdown report |
| `check_golden_logits.py` | compare Ember logits with trusted reference logits |
| `golden_summary.py` | summarize golden-logit reports into compact JSON and Markdown |

stimuli are defined in `stimuli/` and generated by `stimuli/generate_stimuli.py`.
the current stimulus set targets **arabic nonce root-pattern morphology** (200
stimuli: 20 roots x 10 patterns, from Alakeel et al. 2026).
pass `--include-ablations` to add masked-root, masked-pattern, both-masked, and
fake-pattern control prompts without changing the default stimulus output.

generated probe outputs (`*_activations.npy`, `*_activations_correctness.json`,
`*_activations_metadata.json`, `.npz` bundles, benchmark outputs, golden-logit
artifacts, UD downloads, ad hoc plots, logs, and Python bytecode caches) are
ignored.
checked-in fixtures and published figures are kept small and explicit.

for smoke runs, `train_linear_probe.py --probe-kind sgd` gives a fast linear
classifier for pipeline validation. for headline results, use the full
logistic `linear` probe and report random-label selectivity/MDL. for hardening
runs, `--probe-kind mlp` tests whether features that drop under linear probing
remain recoverable non-linearly. `run_probe_matrix.py --dry-run` prints the
full extraction/analysis command matrix for model, prompt-template, and
probe-position ablations. the matrix runner uses batch probe extraction so each
model is loaded once per matrix run, and grouped extraction avoids rerunning the
same template forward pass for multiple pooling positions. for local cpu runs,
`--probe-generate-tokens 1` is the practical default for matrix sweeps; longer
behavioral continuations should run on a larger machine.

### probe split policies

`train_linear_probe.py` supports explicit split policies. Missing split fields
or impossible grouped splits fail with an error; they do not fall back to random
splits.

| policy | grouping | prevents |
|--------|----------|----------|
| `random` / `random-stratified` | stratified random folds by label | class imbalance across folds where possible |
| `root-heldout` / `root` | `root` | the same root appearing in train and test |
| `pattern-heldout` / `pattern` | `pattern` | the same pattern appearing in train and test |
| `combination-heldout` / `root-pattern` | `root` + `pattern` pair | the same root-pattern pair appearing in train and test |
| `template-heldout` / `template` | prompt template metadata | the same prompt template appearing in train and test |
| `--group-field FIELD` | any dotted JSON field | the same custom group appearing in train and test |

Defaults for nonce morphology preserve the established cross-generalization
setup: root probes use `pattern-heldout`, and pattern probes use
`root-heldout`. A direct `root-heldout` root probe is usually invalid because
test roots are unseen classes; Ember reports that as a split error instead of
training a misleading probe.

```bash
python probes/train_linear_probe.py \
  --activations data/activations.npy \
  --stimuli stimuli/nonce_root_pattern_surface.json \
  --tasks root pattern \
  --root-split pattern-heldout \
  --pattern-split root-heldout \
  --output data/probes.npz
```

Probe outputs include split metadata in the `.npz` under `split_policy_json`
and in a sidecar named like `*_split_policy.json`.

### benchmark manifests

`probes/run_benchmark.py` is the higher-level benchmark entry point. It runs a
JSON manifest that can mix Ember GGUF decoder extraction and optional Hugging
Face encoder extraction, then trains generic label-field probes, MDL-style
data-efficiency curves, CCA/RSA, plots, optional divergence, optional fertility,
and a canonical `benchmark_summary.json`.

```bash
python probes/run_benchmark.py \
  --config probes/benchmarks/qwen3_smoke.json \
  --dry-run
```

Render a human-readable Markdown report from a benchmark summary:

```bash
python probes/render_benchmark_report.py \
  --summary data/benchmarks/qwen3-smoke/benchmark_summary.json \
  --output data/benchmarks/qwen3-smoke/report.md
```

Manifest split policy examples:

```json
{
  "split_policy": {
    "root": "pattern-heldout",
    "pattern": "root-heldout"
  }
}
```

```json
{
  "split_policy": {
    "default": "template-heldout"
  }
}
```

For UD or other structured benchmarks, use a grouped field such as
`"group_field": "sentence_id"` to avoid leakage across rows from the same
sentence.

Encoder-side benchmarks use CoNLL-U-derived JSON rows:

```bash
python probes/build_conllu_benchmark.py \
  --input path/to/ar.conllu \
  --output data/benchmarks/ar_ud.json

python probes/extract_hf_encoder.py \
  --model bert-base-multilingual-cased \
  --benchmark data/benchmarks/ar_ud.json \
  --output data/benchmarks/bert_ar_ud_activations.npy
```

The encoder extractor requires the optional encoder stack:

```bash
.venv/bin/python -m pip install torch transformers datasets conllu
```

The generic probe runner can target fields such as `labels.upos`,
`labels.Gender`, `root`, or `pattern`. Sparse fields are filtered per task so
UD features such as `Gender` and `Aspect` do not need to exist on every token.

Current encoder benchmark manifests:

| manifest | purpose |
|----------|---------|
| `probes/benchmarks/ar_ud_mbert_smoke.json` | 1000-row PADT mBERT smoke using fast SGD linear probes |
| `probes/benchmarks/ar_ud_mbert_full.json` | full PADT mBERT run |
| `probes/benchmarks/ar_ud_encoder_suite.json` | mBERT, XLM-R, and AraBERTv2 encoder suite |

The first local mBERT smoke completed on Arabic UD PADT with activation shape
`(1000, 13, 768)`. Its `benchmark_summary.json` reported best probe accuracies
of `0.915` for `labels.upos`, `0.862` for `labels.Gender`, `0.900` for
`labels.Number`, and `0.895` for `labels.Aspect`. Treat this as a pipeline
smoke result; publishable claims need the full encoder suite and trusted
golden/reference checks.

## testing

```bash
cargo fmt -- --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
python3 -m compileall -q probes stimuli scripts
python3 probes/test_probe_workflows.py
```

the integration suite covers tensor operations, sampling, tokenizer loading,
in-memory and mmap-backed gguf fixtures, grouped q8_0 projections, and f16
cache attention. the model smoke test also runs a gpt-2 forward pass when
`gpt2.Q8_0.gguf` is present locally; otherwise it skips so ci does not need to
download large model weights.

### docs site

The static site lives in `docs/`. Shared HTML fragments such as the top
navigation and syntax-highlighting scripts are regenerated in-place:

```bash
python3 scripts/build_docs.py
python3 scripts/check_docs.py
```

Run this after changing docs navigation, language-pair links, or code-block
pages. The generated regions are marked with `docs:*` comments in each HTML
file, while the visual system lives in `docs/style.css`. Optional visual
snapshots can be captured with `python3 scripts/screenshot_docs.py` when
Playwright is installed. Open Graph preview images can be regenerated with:

```bash
python3 scripts/generate_og_images.py
```

Generated charts and social cards read their palette from `docs/style.css`
through `scripts/voidwest_theme.py`. Change the site tokens first, then rerun
the relevant generator; categorical colors, heatmaps, typography, borders,
and dark/light figure backgrounds are derived from that shared theme.

### llama models

ember supports llama-compatible architectures via `--arch llama`. qwen-family
ggufs run through the same llama-family model path; use `--arch qwen3` for
qwen3-specific metadata handling. the following models have been tested:

- **llama 3.2 1b instruct** (`Llama-3.2-1B-Instruct-Q8_0.gguf`) - 1.2b params, q8_0 (~1.3 gb)
- **llama 3.2 3b instruct** (`Llama-3.2-3B-Instruct-Q8_0.gguf`) - 3.2b params, q8_0 (~3.4 gb)
- **llama 3.1 8b instruct** (`meta-llama-3.1-8b-instruct.Q8_0.gguf`) - 8b params, q8_0 (~8.5 gb)
- **qwen2.5 1.5b instruct** (`qwen2.5-1.5b-instruct-q8_0.gguf`) - 1.5b params, q8_0 (~1.8 gb)

Both `--arch llama` and `--arch qwen3` dispatch to the shared Llama-family
implementation; the GGUF `general.architecture` metadata selects the `llama`,
`qwen2`, or `qwen3` configuration keys. The smoke wrapper labels Qwen2.5 as
`qwen3` and passes `tokenizer-qwen2.5.json` explicitly. Qwen2/Qwen2.5
attention projections carry q/k/v biases, which the loader now picks up;
a golden-logit check on Qwen2.5-1.5B matches llama.cpp (top-1 agreement,
max abs diff 0.29).

### support status

| architecture | loads | generates | probe smoke | full 200-stimulus probe | golden checked |
|--------------|-------|-----------|-------------|--------------------------|----------------|
| gpt-2 | yes | yes | yes | not standard | no |
| llama | yes | yes | yes | yes, local/cloud depending on size | no |
| qwen2.5 | yes, via `--arch qwen3` (attention projection biases loaded) | yes, coherent after bias fix | selected smoke runs | pending | yes, 1.5B vs llama.cpp (top-1 match, max diff 0.29) |
| qwen3 | yes, via `--arch qwen3` | yes | yes, 5-stimulus local smoke | yes, Qwen3 0.6B local run | no |
| gemma4 | yes | yes, coherent English | one-stimulus local smoke | pending | no (cosine ~0.87; L0 bit-identical; remaining gap unresolved) |

hidden-state probe results should be treated as research-grade only after a
trusted-reference logits or activation check exists for the exact architecture,
model file, tokenizer, and quantization path. gemma4 golden-logit checks now cover block layout, PLE, global projection,
embedding scaling, layer scales, GELU tanh, RoPE freq_factors, and BF16
loading. RMSNorm amplification of small upstream differences is the current
working explanation for the remaining cosine gap, not a completed root-cause
proof. See `docs/gemma4-parity-investigation.md` and
`docs/layer-dump-tooling.md` for details.

Ember can emit last-prompt logits for external golden checks:

```bash
cargo run --release -- \
  --arch qwen3 \
  --model Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" \
  --dump-logits data/qwen3_france_logits.npy
```

Compare against trusted reference logits with token metadata from both sides:

```bash
python probes/check_golden_logits.py \
  --ember data/qwen3_france_logits.npy \
  --reference reference/qwen3_france_logits.npy \
  --metadata data/qwen3_france_logits_metadata.json \
  --reference-metadata reference/qwen3_france_logits_metadata.json \
  --output data/qwen3_france_golden_report.json
```

Probe classifiers scale activations by default and use a higher logistic
regression iteration limit to avoid premature convergence failures:

```bash
python3 probes/train_linear_probe.py \
  --activations data/activations.npy \
  --stimuli stimuli/nonce_root_pattern_surface.json \
  --max-iter 2000 \
  --scale
```

Use `--no-scale` only when intentionally comparing against an unscaled probe
baseline.

### gemma 4 text models

ember supports dense text-only gemma 4 models via `--arch gemma4`. the path
targets e2b/e4b/31b-style ggufs with f32, f16, or q8_0 weights. it rejects
moe gemma 4 models, multimodal inputs, speculative drafter models, and
k-quantized ggufs in this first pass.

the gemma 4 loader handles long-context rope without cloning per-layer tables,
uses packed q8 per-layer embeddings without full dequantization, projects
per-layer embedding chunks through `blk.N.proj.weight`, and supports probe mode
for hidden-state extraction. a one-stimulus smoke probe on
`gemma-4-E2B-it.Q8_0.gguf` produced activations with shape `(1, 35, 1536)`.

```bash
cargo run --release -- \
  --arch gemma4 \
  --model models/gemma-4-E2B-it.Q8_0.gguf \
  --tokenizer tokenizer-gemma4.json \
  --prompt "The capital of France is" \
  -n 8 --temperature 0 --benchmark
```

download a quantized gguf from huggingface (e.g.
[unsloth/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF)),
then run:

```bash
cargo run --release -- \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --arch llama \
  --prompt "The capital of France is" \
  -n 30 \
  --temperature 0
```

> **note**: if `--tokenizer` is omitted, ember picks `tokenizer-gpt2.json`
> for `--arch gpt2`, `tokenizer.json` for llama/qwen, and
> `tokenizer-gemma4.json` for `--arch gemma4`.

> **note**: demo (`--demo`), single-prompt generation, and probe (`--probe`)
> mode work across the supported model families. Interactive mode (`-i`)
> remains GPT-2-only.
