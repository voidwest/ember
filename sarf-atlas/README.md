# Sarf Atlas

Sarf is a backend-agnostic research framework for Arabic morphology probing
experiments. It organizes datasets, prompts, splits, extraction metadata, and
evaluation artifacts, and is designed to let hidden states or logits come from
Ember, llama.cpp, Transformers, or precomputed files.

Architecture slogan:

```text
Sarf organizes.
Backends extract.
Auditors compare.
```

This directory is not replacing Ember. Ember is currently the strongest local
backend path, but Sarf should not depend on one inference engine. Sarf owns
Arabic morphology schemas, task definitions, prompt templates, split
strategies, experiment configs, expected artifact schemas, report scaffolding,
backend-agnostic validation, and adapters for importing backend outputs.

Sarf does not own model execution internals, hidden-state extraction
implementation, llama.cpp compilation as a required behavior, or Ember
internals.

The current split is:

- `ember`: one validated backend/tooling path for extraction artifacts.
- `sarf-atlas`: Arabic morphology research workspace, paper configs, notes,
  experiment manifests, package skeleton, and backend-agnostic workflow
  planning.
- `gguf-parity-tools`: validation and parity harness.

This directory is experimental and local to the Ember repository for now. If the
shape proves useful, Sarf Atlas may later become a separate repository.

## Sarf v0.1 Scaffold

The initial `sarf` package skeleton lives under `sarf-atlas/src/sarf`. It can
generate a toy morphology dataset, prompts, split metadata, Ember config
placeholders, a Sarf artifact manifest, and an example workflow manifest
without installing dependencies:

```bash
PYTHONPATH=sarf-atlas/src python3 -m sarf example-workflow \
  --out-dir /tmp/sarf-atlas-v0.1-example
```

This is framework scaffolding only. It is not hidden-state extraction, Paper 1
reproduction, or research output.

Current adapter namespace:

- `sarf.backends.ember`
- `sarf.backends.files`

Future optional adapters may cover llama.cpp, Transformers/HF, vLLM, and
precomputed hidden-state formats.
