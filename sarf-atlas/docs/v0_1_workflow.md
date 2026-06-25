# Sarf v0.1 Workflow

Sarf v0.1 is backend-agnostic workflow scaffolding for Arabic morphology
probing. It does not extract hidden states and does not reproduce Paper 1.

Boundary:

- Sarf organizes toy data, prompts, split metadata, experiment configs,
  expected artifact schemas, adapter imports, and workflow manifests.
- Backends extract logits or hidden states.
- Auditors such as `gguf-parity-tools` compare backend agreement.
- Base Sarf works without llama.cpp, Ember, HF/Transformers, or other local
  extraction backends installed.

Run the toy workflow scaffold:

```bash
bash sarf-atlas/examples/run_v0_1_example.sh /tmp/sarf-atlas-v0.1-example
```

This writes:

- `toy_morphology.jsonl`
- `prompts.jsonl`
- `split_metadata.json`
- `ember_native_logits.placeholder.toml`
- `sarf_artifact_manifest.placeholder.json`
- `workflow.json`

The generated Ember config uses placeholder model paths. Replace them before
running extraction. The Sarf artifact manifest shows the backend-agnostic shape
that adapters should produce or import. The output is a tiny workflow scaffold,
not research output.

## Backend Detection

Use these commands to inspect optional local backend availability:

```bash
PYTHONPATH=sarf-atlas/src python3 -m sarf backends list
PYTHONPATH=sarf-atlas/src python3 -m sarf backend llama-cpp doctor
PYTHONPATH=sarf-atlas/src python3 -m sarf backend ember doctor
```

llama.cpp detection checks `LLAMA_TOKENIZE_BIN`, `LLAMA_CLI_BIN`,
`LLAMA_SIMPLE_BIN`, and PATH. Detection is optional and does not download,
build, install, or require llama.cpp. Default llama.cpp should not be described
as supporting Sarf-compatible hidden-state extraction.

Ember detection checks `EMBER_BIN`, PATH `ember`, and optionally probes whether
`ember validate-run --help` is callable. Ember is optional and is not required
to import or run base Sarf.

Users can bring outputs from llama.cpp, Ember, HF/Transformers, or precomputed
files. Hidden-state extraction is not built into base Sarf; hidden-state probing
requires an emitting backend such as Ember, patched llama.cpp, HF/Transformers,
or precomputed Sarf-compatible hidden-state artifacts.
