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

Sarf Atlas is not replacing Ember. Ember is currently the strongest local
backend path, but Sarf should not depend on one inference engine. Sarf owns
Arabic morphology schemas, task definitions, prompt templates, split
strategies, experiment configs, expected artifact schemas, report scaffolding,
backend-agnostic validation, and adapters for importing backend outputs.

Sarf does not own model execution internals, hidden-state extraction
implementation, llama.cpp compilation as a required behavior, or Ember
internals. Base Sarf works without local extraction backends installed.

The current split is:

- `ember`: one validated backend/tooling path for extraction artifacts.
- `sarf-atlas`: Arabic morphology research workspace, paper configs, notes,
  experiment manifests, package skeleton, and backend-agnostic workflow
  planning.
- `gguf-parity-tools`: validation and parity harness.

## Quickstart

Install Sarf Atlas:

```bash
pip install sarf-atlas
```

Generate the v0.1 toy workflow scaffold:

```bash
sarf example-workflow --out-dir /tmp/sarf-atlas-v0.1-example
```

This writes a toy morphology dataset, prompts, split metadata, Ember config
placeholders, a Sarf artifact manifest, and an example workflow manifest. It is
framework scaffolding only, not hidden-state extraction, Paper 1 reproduction,
or research output.

Current adapter namespace:

- `sarf.backends.ember`
- `sarf.backends.files`
- `sarf.backends.llama_cpp`

Future optional adapters may cover Transformers/HF, vLLM, and additional
precomputed hidden-state formats.

## Optional Backends

Sarf v0.1 can inspect local backend availability, but detection is optional and
does not make llama.cpp, Ember, Transformers/HF, or hidden-state extraction part
of the base package:

```bash
sarf backends list
sarf backend llama-cpp doctor
sarf backend ember doctor
```

llama.cpp detection checks `LLAMA_TOKENIZE_BIN`, `LLAMA_CLI_BIN`,
`LLAMA_SIMPLE_BIN`, and common PATH names. Default llama.cpp is useful for some
local tokenization/logits workflows, but it should not be treated as emitting
Sarf-compatible hidden-state artifacts.

Ember detection checks `EMBER_BIN`, PATH `ember`, and whether
`ember validate-run --help` is callable when an Ember binary is found. Ember is
optional; Sarf can still organize workflows and import artifacts from files.

Users may bring artifacts from llama.cpp, Ember, HF/Transformers, or
precomputed files. Hidden-state extraction is not built into base Sarf and
requires an emitting backend such as Ember, patched llama.cpp, HF/Transformers,
or precomputed Sarf-compatible artifacts.
