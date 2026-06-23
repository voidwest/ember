# Reproducibility Notes

This repository is preserved as the Paper 1 artifact for:

**Leakage-Aware Probing of Arabic Morphology in Small Language Models**

The goal of this document is to help readers audit or reproduce the current
artifact without accidentally changing the paper outputs.

## Artifact Scope

The paper draft is in `paper/main.tex`. The main citable result artifacts live
under:

- `data/arabic_morph_real/` for PADT-derived morphology data, activation dumps,
  probe summaries, heldout reports, diagnostics, and ablation artifacts.
- `paper/tables/` for LaTeX tables used by the paper.
- `paper/figures/` for figures used by the paper.
- `probes/` for probe training, controls, heldout evaluation, diagnostics, and
  plotting scripts.
- `src/arabic_morph_dataset/` plus `scripts/arabic_morph_dataset.py` for the
  dataset preparation pipeline.

Do not regenerate outputs in-place unless the goal is an intentional new
artifact version.

## Environment

Known local expectations:

- Rust toolchain: `1.92` as shown in the README badge.
- Python: `>=3.11` for the dataset package in `pyproject.toml`.
- Optional Python dev install:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Probe scripts may require additional scientific Python packages listed in
`probes/requirements.txt`.

## Sanity Checks

These checks exercise code paths without regenerating paper numbers:

```bash
cargo test
python3 -m pytest tests/test_arabic_morph_dataset.py
python3 -m pytest probes/test_probe_workflows.py
```

The probe workflow tests use synthetic fixtures. They are safer than rerunning
the full experiment in place.

## Dataset Pipeline Reference

The strict 5k PADT-derived dataset config is:

```bash
python3 scripts/arabic_morph_dataset.py run-config \
  --config configs/arabic_morph_disambig_padt_5000_strict.toml
```

This writes to `data/arabic_morph_real/out_disambig_padt_5000_strict/`.
Run it only in a clean copy or with a deliberate output directory override,
because it can overwrite reproducibility artifacts.

The tiny bundled sample can be used for smoke validation:

```bash
python3 scripts/arabic_morph_dataset.py run-config \
  --config configs/arabic_morph_sample.toml
```

## Probe Pipeline Reference

`probes/run_all_5k.sh` records the full 5k probe-analysis sequence after
activation extraction:

- `probes/run_baseline_probes.py`
- `probes/run_control_analysis.py`
- `probes/run_heldout_probes.py`
- `probes/run_group_variance.py`
- `probes/token_diagnostics.py`
- `probes/audit_probe_leakage.py`

The script is hard-coded for the local artifact layout and writes result files
under `data/arabic_morph_real/probe_baseline_*`. Treat it as a reproducibility
reference, not as a casual command to run on the preserved artifact tree.

## Models And Extraction

The paper uses GGUF model files and tokenizer JSON files that may be local and
large. Re-extraction can change timestamps, metadata, activation files, and
downstream reports. For the Paper 1 artifact, prefer auditing existing
activation metadata and probe outputs before rerunning extraction.

## Suggested Artifact Tag

Use `paper1-artifact-v0.1` for the first preserved artifact tag.
