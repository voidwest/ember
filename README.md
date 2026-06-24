# ember

[![rust](https://img.shields.io/badge/rust-1.92-blue)](https://www.rust-lang.org)
[![ci](https://github.com/voidwest/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/voidwest/ember/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Paper 1 artifact status.** This repository is preserved as the research
artifact for “Leakage-Aware Probing of Arabic Morphology in Small Language
Models.” Treat paper outputs, figures, tables, datasets, and probe reports as
citable artifacts; do not change them casually. Future Ember backend/refactor
work, including further llama.cpp integration, should happen in a separate
repository. See [ARTIFACT_STATUS.md](ARTIFACT_STATUS.md) and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

a lightweight research layer for hidden-state extraction, leakage-aware probing,
and reproducible morphology experiments over GGUF models. Ember keeps an
inspectable Rust inference path for validation. This Paper 1 artifact includes
some backend-ready scaffolding, but ongoing external-backend work should happen
outside this repository.

research write-up: https://voidwest.dev/ember

## what ember is / is not

Ember is a research layer for hidden-state extraction, leakage-aware probing,
and reproducible morphology experiments over GGUF models. The native Rust path
remains an inspectable reference backend for small-to-medium models and
validation work.

Ember is not trying to beat llama.cpp on throughput, model coverage, or
production readiness. llama.cpp is the better default if the goal is broad,
high-performance local inference. Future Ember backend work should use
llama.cpp where that is the right tool, while keeping dataset handling, prompt
construction, token-position selection, hidden-state artifact schemas, probes,
baselines, metrics, reports, and validation in Ember.

The current research direction is Arabic morphology probing and validation.

## Arabic morphology dataset pipeline

This repo includes a local Python pipeline for preparing CAMELMORPH/CAMeL-style
Arabic morphology exports for root/pattern probing and later SFT experiments.
It produces canonical morphology JSONL, SFT chat JSONL, probing JSONL,
deterministic held-out splits, stats, and leakage validation reports without
requiring CAMeL Tools at runtime.

Optional local install:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest -q
```

Run the tiny bundled sample:

```bash
python3 scripts/arabic_morph_dataset.py run-config --config configs/arabic_morph_sample.toml
```

Run the larger imbalanced fixture:

```bash
python3 scripts/generate_arabic_morph_fixture.py \
  --output data/arabic_morph_sample/camelmorph_imbalanced_sample.jsonl \
  --seed 17
python3 scripts/arabic_morph_dataset.py run-config --config configs/arabic_morph_imbalanced_sample.toml
```

To use real data, export CAMELMORPH/CAMeL/CALIMAStar analyses to JSONL, CSV, or
TSV with fields such as `word`, `diac`, `lex`, `root`, `pattern`,
`pattern_concrete`, `pos`, and feature columns like `gen`, `num`, `per`, `asp`,
`vox`, `mod`, `cas`, and `stt`. Then copy
`configs/arabic_morph_sample.toml`, point `input_path`, `output_dir`, and
`source_name` at the export, and choose a split strategy such as
`root_heldout`, `abstract_pattern_heldout`, `concrete_pattern_heldout`,
`root_pattern_heldout`, or `lemma_heldout`.

See [docs/dataset_pipeline.md](docs/dataset_pipeline.md) for the full input
format, output schemas, split guarantees, CLI commands, and validation reports.

## validation ladder

Use these levels when interpreting Ember runs:

1. **smoke**: structural execution only. The command ran, loaded artifacts, and
   produced output. This is not numerical validation or output-quality evidence.
2. **golden logits**: output-logit comparison against a trusted reference for
   the same model, tokenizer, prompt, and quantization path.
3. **activation reference checks**: internal hidden-state comparison against a
   trusted implementation. This is required before treating layer geometry as
   numerically validated.
4. **probes**: linear or MLP classifiers over cached hidden states. These show
   decodability or recoverability, not causal use.
5. **interventions**: causal tests that first verify a probe-score drop after
   removing or perturbing a direction, then measure logit or generation effects.

## current evidence status

| architecture | smoke | golden logits | activation reference | probe runs | status |
|--------------|-------|---------------|----------------------|------------|--------|
| gpt-2 | structural smoke works when local GGUF is present | none | none | not a standard Arabic morphology run yet | loader baseline; negative-control work pending |
| llama | local/cloud structural smokes and probe extraction | pending | pending | preliminary LLaMA 1B/3B/8B decoder probe runs | research findings are preliminary until references and reports are complete |
| qwen2.5 | selected warning-prone smokes through llama-family path | none | none | pending validation | experimental; do not treat as quality-compatible |
| qwen3 | Qwen3 0.6B smoke/probe paths run locally | pending target | pending | Qwen3 0.6B local probe run exists | promising engineering path, not yet numerically validated |
| gemma4 | local BOS smoke + golden-logit comparison passes | cosine ~0.87 against llama.cpp reference; coherent English output | per-layer hidden-state comparison pipeline operational; L0 attn_norm bit-identical | pending full runs | structural fixes applied (PLE, block layout, RoPE, global projection, BF16, embedding scale, layer scales); remaining gap ~0.13 attributed to RMSNorm weight amplification of sub-ULP differences, not a structural bug |
| hf encoders | external Hugging Face extraction path works for mBERT smoke | not applicable to Ember GGUF numerics | external stack not activation-checked here | mBERT PADT smoke; full encoder suite pending | useful benchmark path, not an Ember inference validation result |

## features

- **gguf v3 loader**: reads gguf model files, supports f32, f16, and q8_0 dtypes.
- **on-the-fly dequantization**: q8_0 weights stay in block-compressed form in
  memory (~4x smaller than f32) and are dequantized during matmul.  the 3B
  model uses ~3.4 GB of ram instead of ~12.8 GB.
- **block-wise sgemm**: quantized matmuls dequantize in blocks of 256 columns
  and multiply with `matrixmultiply::sgemm` - 5x faster prefill than the
  scalar path.
- **backend trait**: model code is generic over a `Backend` trait for linear ops,
  embeddings, and element-wise math - swap cpu for gpu later without rewriting
  those paths. (attention is cpu-scalar for now; see design notes.)
- **execution backend interface**: extraction can now be routed through a model
  execution backend. `native` wraps Ember's Rust inference path; `llama-cpp` is
  reserved for a patched/custom external extraction binary and currently errors
  clearly as not implemented.
- **explicit memory**: pre-allocated kv caches and explicit tensor ownership make
  inference memory use visible and easy to profile.
- **alloc-first design**: core tensor types and model code avoid `std` where practical, using `alloc` for vec-backed storage.
- **hidden-state probing**: extract per-layer activations at any token position.
  probe mode (`--probe`) feeds stimuli through the model and saves full
  hidden-state tensors as `.npy` for downstream analysis.
- **probing pipeline**: python scripts for linear probes (with task-specific
  splits, control tasks, and selectivity), CCA, RSA, divergence analysis,
  cross-model comparison plots, and
  tokenizer fertility analysis.

## what this demonstrates

- **systems programming in rust**: manual memory layout for the kv cache
  (`[layer][head][pos][head_dim]`), explicit stride math for tensor indexing,
  and scoped allocations that can be profiled and optimized directly.
- **generic backend architecture**: the transformer is written against a
  `Backend` trait - the same model code works on cpu today and could run
  on gpu tomorrow without modification.
- **ml fundamentals**: causal multi-head attention with kv caching,
  numerically stable softmax (handles all-masked rows), layer norm,
  gelu activation, top-k/top-p sampling.
- **file format parsing**: gguf v3 loader with f32, f16, and q8_0
  quantization support.
- **memory-conscious inference**: q8_0 weights stay quantized in memory
  and are dequantized in blocks during matmul - the 3.2B model runs in
  ~3.4 GB of ram on consumer hardware.
- **edge case handling**: uniform fallback when every logit is -inf,
  categorical sampling with inverse cdf, nucleus cutoff logic.

## usage

```bash
cargo run --release -- --model gpt2.Q8_0.gguf --prompt "hello"
```

Backend-ready hidden-state extraction uses a declarative config:

```bash
cargo run --release -- extract --backend native --config configs/extract.example.toml
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
| `--dump-logits` | (none) | write last-prompt logits for `--prompt` to `.npy` and exit |
| `--write-run-manifest` | (none) | write a reproducibility manifest with model/tokenizer hashes, git commit, compiler, Rayon, and CPU feature data |
| `--record-model-sha256` | (none) | compute and record model file sha256 in probe metadata |
| `--dump-gguf-metadata` | (none) | write parsed GGUF metadata to JSON |
| `--probe` | (none) | run probe mode: extract hidden states from each block |
| `--probe-stimuli` | `stimuli/nonce_root_pattern.json` | path to stimuli json for probe mode |
| `--probe-output` | `data/activations.npy` | output path for probe activations (.npy) |
| `--probe-template` | `en_zero` | stimulus prompt key to probe (`en_zero`, `en_one`, `ar_zero`, `ar_one`, or generated controls) |
| `--probe-templates` | (none) | comma-separated prompt template keys for batch probe extraction |
| `--probe-position` | `last` | hidden-state position to pool: `last`, `root`, `pattern`, or `prompt_mean` |
| `--probe-positions` | (none) | comma-separated hidden-state positions for batch probe extraction |
| `--probe-output-dir` | `data/probe_matrix` in batch mode | output directory for batch probe extraction |
| `--probe-output-prefix` | `probe` | output filename prefix for batch probe extraction |
| `--probe-generate-tokens` | `16` | continuation length for probe behavioral scoring |
| `--probe-limit` | (none) | cap probe extraction to the first N stimuli for smoke tests |

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
  --probe-stimuli stimuli/nonce_root_pattern.json \
  --probe-output-dir data/matrix \
  --probe-output-prefix llama1b \
  --probe-templates en_zero,en_one,ar_zero,ar_one \
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
  --templates en_zero en_one \
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
  --probe-stimuli stimuli/nonce_root_pattern.json \
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
  --stimuli stimuli/nonce_root_pattern.json \
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
and an in-memory gguf parser fixture. the model smoke test also runs a gpt-2
forward pass when `gpt2.Q8_0.gguf` is present locally; otherwise it skips so ci
does not need to download large model weights.

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

### llama models

ember supports llama-compatible architectures via `--arch llama`. qwen-family
ggufs run through the same llama-family model path; use `--arch qwen3` for
qwen3-specific metadata handling. the following models have been tested:

- **llama 3.2 1b instruct** (`Llama-3.2-1B-Instruct-Q8_0.gguf`) - 1.2b params, q8_0 (~1.3 gb)
- **llama 3.2 3b instruct** (`Llama-3.2-3B-Instruct-Q8_0.gguf`) - 3.2b params, q8_0 (~3.4 gb)
- **llama 3.1 8b instruct** (`meta-llama-3.1-8b-instruct.Q8_0.gguf`) - 8b params, q8_0 (~8.5 gb)
- **qwen2.5 1.5b instruct** (`qwen2.5-1.5b-instruct-q8_0.gguf`) - 1.5b params, q8_0 (~1.8 gb)

qwen2.5 models use `--arch llama`; ember auto-detects the qwen2 gguf metadata
inside the shared llama-family path. qwen3 models use `--arch qwen3`, which
dispatches through that same path while selecting qwen3 metadata keys.

### support status

| architecture | loads | generates | probe smoke | full 200-stimulus probe | golden checked |
|--------------|-------|-----------|-------------|--------------------------|----------------|
| gpt-2 | yes | yes | yes | not standard | no |
| llama | yes | yes | yes | yes, local/cloud depending on size | no |
| qwen2.5 | experimental, currently via `--arch qwen3` | warning-prone | selected smoke runs | pending architecture/tokenizer validation | no |
| qwen3 | yes, via `--arch qwen3` | yes | yes, 5-stimulus local smoke | yes, Qwen3 0.6B local run | no |
| gemma4 | yes | yes, coherent English | one-stimulus local smoke | pending | no (cosine ~0.87, L0 bit-identical, remaining gap ~0.13 from RMSNorm amplification) |

hidden-state probe results should be treated as research-grade only after a
trusted-reference logits or activation check exists for the exact architecture,
model file, tokenizer, and quantization path. gemma4 golden-logit checks now cover block layout, PLE, global projection,
embedding scaling, layer scales, GELU tanh, RoPE freq_factors, and BF16
loading. The remaining cosine gap (~0.13) is attributed to RMSNorm weight
amplification of sub-ULP differences across the 35-layer pipeline, not to a
known structural mismatch. See `docs/gemma4-parity-investigation.md` and
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
  --stimuli stimuli/nonce_root_pattern.json \
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
  --model Gemma-4-E2B-Q8_0.gguf \
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

> **note**: interactive (`-i`) and demo (`--demo`) modes are not yet wired
> for llama/qwen or gemma 4. the single-prompt generation path and probe
> (`--probe`) mode work with these architectures.

## research: arabic morphology probing

ember has been used for preliminary probes of how llama 3.2 models (1b, 3b,
8b) expose arabic nonce root-pattern morphology in hidden states. Treat these
as probe observations until golden-logit reports, activation references,
stronger controls, and generated benchmark reports are complete.

- **root identity is less linearly decodable in some larger-model mid-layers**:
  the current probe runs report root accuracy dropping from 100% (1b, all
  layers) to 78% (3b mid-layers) and 70% (8b mid-layers), forming a u-shaped
  curve in this setup.
- **pattern identity appears more surface-accessible in these runs**: pattern
  probe accuracy at layer 0 is reported as 20% (1b), 100% (3b), and 68.5%
  (8b), with early-layer recovery depending on scale.
- **behavioral generation did not solve the task in this setup**: these runs
  generated "the" for every prompt. This does not by itself prove why behavior
  failed, or that decoded features are causally used or unused.
- **tokenizer fertility is a control variable, not an explanation by itself**:
  the measured ar/en token ratio is 1.2x for the llama 3 tokenizer versus 2.4x
  for gpt-2 on the same prompts, but tokenizer effects need controls before
  explanatory claims.

full research write-up: https://voidwest.dev/ember

## architecture

the entry point is `main.rs` -> `generate()`, a generic `ForwardModel` path
used by gpt-2, llama/qwen, and gemma 4. generation runs a two-phase loop:

1. **prefill** - forward pass on the full prompt, populating the kv cache.
2. **decode** - one token at a time, reading from the cache.

shared model primitives live in `src/model.rs` (`ForwardModel`, `Linear`, and
the gpt-2 blocks). llama/qwen lives in `src/llama.rs`, gemma 4 lives in
`src/gemma4.rs`, tensors are `CpuTensor` in `src/tensor.rs`, and the gguf
parser is `src/loader.rs`.

```text
main.rs              entry point, cli args, dispatch, probe mode
|- loader.rs         gguf v3 parser, tensor loading
|- model.rs          shared model primitives + gpt-2 transformer
|- llama.rs          llama/qwen transformer
|- gemma4.rs         dense text-only gemma 4 transformer
|  |- backend.rs     backend trait + cpu backend impl
|  |- tensor.rs      row-major f32 tensor, rope, silu, elemul
|  `- kv_cache.rs    flat k/v cache, gqa-aware (n_kv_heads)
|- sampler.rs        temperature, top-k, top-p sampling
|- tokenizer.rs      huggingface tokenizer wrapper
|- quant.rs          q8_0 block dequantization + QuantizedWeight
`- probes/           python probe scripts (linear, cca, rsa, divergence)
```

## design notes

- **backend trait**: the transformer is generic - `CpuBackend` is the default,
  but any type implementing `Backend` works. the trait abstracts linear ops,
  element-wise math, layer norm, attention, and tensor lifecycle. the current
  attention backend is still scalar cpu code, but the model no longer owns
  those kernels directly.
- **q8_0 quantization**: 8-bit block quantization (fp16 scale + 32 int8
  values per block). weights stay in this quantized form in memory and
  are dequantized on the fly during matmul - ~4x smaller than f32 at
  rest with minimal perplexity loss.
- **kv cache**: flat `[layer][head][seq_position][head_dim]` layout. prefill
  stores k/v for all prompt tokens; decode reads from cache and appends one
  token at a time. uses `n_kv_heads` (not `n_heads`) for the head dimension,
  supporting grouped-query attention with zero overhead for mha models
  (`n_heads == n_kv_heads`).

## design justifications

these are the non-obvious trade-offs made in this codebase.

**transposed embeddings on load.** gguf stores token/position embeddings as
`[vocab, embed]`. the loader transposes them so `index_select` picks a row
directly - one contiguous slice per token - instead of gathering strided
elements at inference time. the cost is one transpose at load; the benefit
is simpler and faster lookups in the hot loop.

**`load_from_cpu` on the backend trait.** the method loads host-side f32
data into a backend tensor. for `CpuBackend` this is a thin wrapper around
`CpuTensor::from_data`; a future gpu backend would copy the data to device
memory here. the name was chosen over `from_cpu` to avoid tripping
`clippy::wrong_self_convention` (which expects `from_*` to be a constructor
without `&self`).

**`n_layers` is stored but never read.** the kv cache allocates per-layer
storage using `n_layers` in `new()`, then never reads the field again. it
exists only to size the flat buffer. removing it would require threading
the layer count through every cache method or hardcoding it. storing it is
the more explicit path.

**`matrixmultiply` for cpu matmul.** both f32 and q8_0 matmuls go through
`matrixmultiply::sgemm` - pure rust, no blas linking, decent simd.  the
`Backend` trait means faster kernels can be swapped in under a new backend
type without touching model code.  this is a pragmatic default, not a
final answer.

**softmax returns uniform for all-masked input.** when every logit is -inf
(fully masked row), softmax normally produces NaN. this code detects that
case and returns `1/n` per position. it costs one extra branch per row and
prevents the generation loop from producing NaNs on degenerate input.

## prerequisites

- rust stable toolchain
- a gguf model file (e.g. gpt2 in q8_0)
- a tokenizer file for the model (`tokenizer.json` for llama, `tokenizer-gpt2.json` for gpt-2; both are included in the repo)

## current limitations

- attention math is abstracted behind the `Backend` trait, but the only
  implementation today is the cpu backend. it uses SIMD helpers for inner
  dot/accumulate work and Rayon for larger per-head workloads; there is no gpu
  backend yet.
- the lm head (large vocab projection) is still the throughput bottleneck during
  decode. a fused/deferred or top-k-aware lm-head path is the next obvious
  optimization target.
- model loader supports gpt-2, llama/qwen, and dense text-only gemma 4 ggufs
  through architecture-specific tensor names. demo and interactive modes are
  not yet wired for llama/qwen or gemma 4; single-prompt generation and probe
  mode work with those architectures.
- not fully no_std - file i/o and mmap require std.

## optimization notes

the probe pipeline and CPU backend now have six CPU-friendly optimizations:

- grouped extraction avoids redundant forwards across positions for the same
  template.
- pooled activation extraction writes only selected hidden-state spans instead
  of storing full per-layer sequence activations.
- `run_probe_matrix.py --jobs` parallelizes independent downstream analysis
  bundles after extraction.
- full and cached attention paths use the shared SIMD dot-product and
  weighted-accumulate helpers where their head dimensions are contiguous.
- large q8_0 single-row decode matmuls can split output rows across Rayon
  workers, primarily targeting vocab-head-sized projections.
- shared CPU attention can split independent heads across Rayon workers for
  larger prefill/cached-attention workloads.

the next useful optimization targets are:

1. **lm-head specialization**: decode throughput is dominated by projecting the
   final hidden state to a large vocabulary. a fused, top-k-aware, or tiled
   lm-head path is likely higher impact than parallelizing small element-wise
   ops.
2. **richer thread-count benchmarks**: run `scripts/benchmark_threads.py` across
   Qwen3 0.6B, LLaMA 1B, Gemma 4, and selected 3B slices, then use the results
   to tune the parallelism thresholds.
3. **persistent scratch for parallel attention**: the parallel attention path
   currently allocates per-head scratch/output buffers. a small worker-local
   scratch pool could reduce allocation overhead on repeated decode steps.

## license

mit
