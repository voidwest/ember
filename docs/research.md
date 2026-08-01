# Ember research

The Arabic morphology dataset pipeline and probing overview.
Moved from the top-level README.

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
