# Paper 1 Artifact Status

This repository should be treated as the citable artifact for:

**Leakage-Aware Probing of Arabic Morphology in Small Language Models**

It is not the preferred place for future Ember backend architecture work. New
backend experiments, especially further llama.cpp integration, should move to a
separate repository so this tree remains stable for Paper 1.

## What This Repository Contains

Core Paper 1 research code:

- `src/arabic_morph_dataset/`
- `scripts/arabic_morph_dataset.py`
- `configs/arabic_morph_*`
- `probes/run_baseline_probes.py`
- `probes/run_control_analysis.py`
- `probes/run_heldout_probes.py`
- `probes/run_group_variance.py`
- `probes/token_diagnostics.py`
- `probes/audit_probe_leakage.py`
- `probes/visualization/`
- `paper/main.tex`

Experiment outputs and generated artifacts:

- `data/arabic_morph_real/`
- `data/arabic_morph_sample/`
- `artifacts/morphology_runs/`
- `paper/tables/`
- `paper/figures/`
- probe `.json`, `.npz`, `.npy`, and `.png` files under `data/`

Reusable Ember engine/backend code:

- `src/backend.rs`
- `src/model.rs`
- `src/llama.rs`
- `src/gemma4.rs`
- `src/loader.rs`
- `src/tokenizer.rs`
- `src/tensor.rs`
- `src/quant.rs`
- `src/simd.rs`
- `src/extraction.rs`
- `src/model_backend.rs`

Unfinished or future-facing scaffolding:

- backend abstraction and extraction contract code
- `llama-cpp` / `llama-cpp-external` backend placeholders
- backend validation scaffolding
- engineering notes about future activation-reference and backend work

These future-facing pieces are retained because they are already part of the
current codebase, but new backend milestones should not be developed here.

## Risk Classification

Safe to edit:

- documentation that labels the artifact status
- citation metadata
- reproducibility notes
- narrow `.gitignore` exceptions for artifact documentation

Edit only with care:

- `README.md`
- `configs/`
- `src/arabic_morph_dataset/`
- `probes/`
- `scripts/`
- `docs/`
- Rust engine code under `src/`

Do not touch casually:

- `paper/main.tex`
- `paper/tables/`
- `paper/figures/`
- `data/arabic_morph_real/`
- `data/arabic_morph_sample/`
- activation dumps, probe summaries, heldout reports, ablation outputs, and
  generated plots

Changes to the last group can affect paper claims or reproducibility.

## Preservation Policy

- Do not delete datasets.
- Do not delete experiment outputs.
- Do not modify numerical results.
- Do not regenerate plots in place.
- Do not change paper claims without creating a new artifact version.
- Do not continue llama.cpp backend/refactor milestones in this repository.

Suggested tag for this preserved state:

```bash
paper1-artifact-v0.1
```
