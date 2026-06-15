# Arabic Morphology Dataset Pipeline

This pipeline prepares local, reproducible datasets for Arabic morphology probing
and targeted SFT experiments. It consumes exported CAMELMORPH/CAMeL
Tools/CALIMAStar-style analysis records and produces canonical morphology
JSONL, chat-style SFT JSONL, probing JSONL, split reports, validation reports,
and stats.

It does not install CAMeL Tools, run model training, or add LoRA code. The
expected workflow is to export analyses separately, then run this pipeline on the
exported records.

## Installation

The script wrapper works without installing the package:

```bash
python3 scripts/arabic_morph_dataset.py --help
```

For development, install the package in a local environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest -q
```

After installation, both entrypoints work:

```bash
.venv/bin/python -m arabic_morph_dataset --help
.venv/bin/arabic-morph-dataset --help
```

TOML configs use the Python standard library. YAML configs are optional and
require the `yaml` extra or `dev` extra, both of which install PyYAML.

## Expected Input

JSONL is the primary format. CSV and TSV with equivalent columns are also
accepted. Each row can be a single analysis:

```json
{"word":"المكتبات","diac":"ٱلْمَكْتَبَاتُ","lex":"مَكْتَبَة_1","root":"كتب","pattern":"مَفْعَلَة","pattern_concrete":"مكتبة","pos":"noun","gen":"f","num":"p","stt":"d"}
```

Rows with an `analyses` array are expanded into one record per analysis. Common
CAMeL-style aliases are normalized: `word` to `surface`, `diac` to
`diacritized`, `lex` to `lemma`, `pattern` to `abstract_pattern`, and feature
short names such as `gen`, `num`, `per`, `asp`, `vox`, `mod`, `cas`, and `stt`
to canonical feature names.

## Outputs

Canonical JSONL contains one morphology record per analysis with:

`id`, `surface`, `surface_dediac`, `diacritized`, `lemma`, `root`,
`abstract_pattern`, `concrete_pattern`, `pos`, `features`, `source`,
`analysis_id`, `is_ambiguous`, `metadata`, and, after splitting, `split`.

SFT JSONL uses a chat-compatible `messages` format. The implemented tasks are:

- `analyze_form`: surface to lemma, root, patterns, POS, and features.
- `root_pattern`: surface to root and abstract/concrete pattern.
- `feature_bundle`: surface to POS and morphological features.
- `reinflect`: lemma and target features to surface, when enabled.

Probing JSONL has no instruction wrapper and is intended for layer-wise
representation extraction and label decoding.

## Split Strategies

Splits are deterministic with a seed. Held-out strategies keep every inflected
form of the same lemma in the same split.

- `random`: true per-record random split for debugging. This can leak lemmas
  across splits and should not be used for the main experiments.
- `lemma_random`: random over lemma-connected groups for debugging.
- `root_heldout`: roots assigned wholly to train/dev/test.
- `abstract_pattern_heldout`: abstract patterns assigned wholly to splits.
- `concrete_pattern_heldout`: concrete patterns assigned wholly to splits.
- `root_pattern_heldout`: specific root-pattern pairs are held out while roots
  and patterns may appear individually in train.
- `lemma_heldout`: lemmas assigned wholly to splits.

Validation reports include leakage intersections such as `train_dev` and
`train_test`. Split assignment is size-aware: larger lemma/root/pattern
components are placed first, then assigned to the split with the largest
remaining target deficit. A component larger than a target split can still force
overshoot; the report records the resulting counts.

## Filtering

Filters run in this order:

1. label, ambiguity, and POS filters
2. minimum examples per root/pattern on the filtered set
3. maximum examples per root/pattern caps on the filtered set

That means `min_examples_per_root = 5` is interpreted as at least five examples
remaining after earlier filters such as `pos_allowlist` and `drop_ambiguous`.
Dropped records are counted by reason in `filter_report.json`.

## Tiny Sample

Run the complete sample pipeline:

```bash
python3 scripts/arabic_morph_dataset.py run-config --config configs/arabic_morph_sample.toml
```

This writes:

- `data/arabic_morph_sample/out/canonical.jsonl`
- `data/arabic_morph_sample/out/sft.jsonl`
- `data/arabic_morph_sample/out/probes.jsonl`
- `data/arabic_morph_sample/out/stats.json`
- `data/arabic_morph_sample/out/summary_report.json`
- `data/arabic_morph_sample/out/validation.json`

Equivalent step-by-step commands:

```bash
python3 scripts/arabic_morph_dataset.py ingest \
  --input data/arabic_morph_sample/camelmorph_sample.jsonl \
  --output data/arabic_morph_sample/out/ingested.jsonl \
  --source-name synthetic_camelmorph_msa

python3 scripts/arabic_morph_dataset.py normalize \
  --input data/arabic_morph_sample/out/ingested.jsonl \
  --output data/arabic_morph_sample/out/filtered.jsonl \
  --config configs/arabic_morph_sample.toml

python3 scripts/arabic_morph_dataset.py split \
  --input data/arabic_morph_sample/out/filtered.jsonl \
  --output data/arabic_morph_sample/out/canonical.jsonl \
  --strategy root_heldout --seed 7 --train-ratio 0.6 --dev-ratio 0.2 --test-ratio 0.2

python3 scripts/arabic_morph_dataset.py make-sft \
  --input data/arabic_morph_sample/out/canonical.jsonl \
  --output data/arabic_morph_sample/out/sft.jsonl

python3 scripts/arabic_morph_dataset.py make-probes \
  --input data/arabic_morph_sample/out/canonical.jsonl \
  --output data/arabic_morph_sample/out/probes.jsonl \
  --split-type root_heldout

python3 scripts/arabic_morph_dataset.py stats \
  --input data/arabic_morph_sample/out/canonical.jsonl \
  --output data/arabic_morph_sample/out/stats.json

python3 scripts/arabic_morph_dataset.py report \
  --input data/arabic_morph_sample/out/canonical.jsonl \
  --filter-report data/arabic_morph_sample/out/filter_report.json \
  --output data/arabic_morph_sample/out/summary_report.json

python3 scripts/arabic_morph_dataset.py validate \
  --input data/arabic_morph_sample/out/canonical.jsonl \
  --sft data/arabic_morph_sample/out/sft.jsonl \
  --probes data/arabic_morph_sample/out/probes.jsonl \
  --split-strategy root_heldout \
  --output data/arabic_morph_sample/out/validation.json
```

## Imbalanced Sample

The repo also includes a deterministic generator for a larger, real-ish
CAMeL-style fixture with root imbalance, POS skew, ambiguous analyses, missing
roots, missing patterns, and missing lemmas:

```bash
python3 scripts/generate_arabic_morph_fixture.py \
  --output data/arabic_morph_sample/camelmorph_imbalanced_sample.jsonl \
  --seed 17

python3 scripts/arabic_morph_dataset.py run-config \
  --config configs/arabic_morph_imbalanced_sample.toml
```

The generated fixture currently has 393 raw records and the sample filters keep
343. The run writes `data/arabic_morph_sample/out_imbalanced/summary_report.json`
with:

- records kept and dropped by reason
- unique root, abstract-pattern, and concrete-pattern counts
- root-heldout, abstract-pattern-heldout, and concrete-pattern-heldout leakage
  pass/fail checks
- top 20 roots by count
- top 20 abstract and concrete patterns by count

The `report --filter-report` option expects either a single JSON object or a
single-row JSONL file containing the filter report.

## Replacing The Sample

Export CAMELMORPH/CAMeL Tools analyses to JSONL, CSV, or TSV with columns for
surface form, lemma, root, POS, pattern labels, and features. Then update
`input_path`, `output_dir`, and `source_name` in a TOML config. The pipeline will
normalize aliases and report records removed by filters.

## Connection To Later Experiments

The probing JSONL is the stable label source for layer-by-layer decodability
experiments. The SFT JSONL is a compact instruction dataset for later LoRA
fine-tuning. Because train/dev/test splits are held out by root, pattern, lemma,
or root-pattern combination, the same split metadata can be reused before and
after fine-tuning to compare representation movement across layers.
