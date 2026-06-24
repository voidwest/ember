# Backend Validation

Ember uses three validation layers. They answer different questions and should
not be collapsed into one claim.

## 1. validate-run

`validate-run` checks one Ember artifact run for structural honesty:

```bash
cargo run -- validate-run /tmp/sarf-atlas-ember-smoke/runs/tokenization-parity-smoke
```

It validates:

- artifact contract version and layout
- required files from `manifest.json`
- `samples.jsonl`, `tokenization.jsonl`, and `positions.jsonl` row counts
- row ordering, `sample_index`, `sample_id`, and `prompt_hash` consistency
- token counts and selected token-position bounds
- layer shard presence when declared
- optional logits file presence when declared
- checksum references when present
- `report.json` status, schema, and layout
- backend identity fields
- mock/non-research/provenance markers when present

`validate-run` does not compare two backends and does not prove numerical model
parity. A tokenization-only run with `layers = []`, `no_logits = true`, and
`no_hidden_states = true` can pass if it is honest about those limits.

Use `--require-layers` when validating a run that is supposed to contain
hidden-state layer shards:

```bash
cargo run -- validate-run --require-layers runs/native-hidden-state-run
```

## 2. validate-backends

`validate-backends` compares two Ember artifact runs:

```bash
cargo run -- validate-backends \
  --native-run runs/native-smoke \
  --external-run runs/llama-cpp-external-smoke
```

It first validates each run with the artifact contract, then compares compatible
backend artifacts such as token IDs, selected positions, sample counts, and
logits availability. This command is for native-vs-external or
backend-vs-backend comparisons after both sides have already written Ember
artifact directories.

`validate-backends` is not a standalone external audit. It compares Ember run
directories to each other.

## 3. gguf-parity-tools

`gguf-parity-tools` is the external parity harness. It is used outside Ember's
own artifact-contract validation to audit:

- token IDs with `token-audit`
- logits with logits comparison tooling
- layer tensors in the future, once the external hidden-state path exists

For the Sarf Atlas tokenization smoke, the external audit shape is:

```bash
PYTHONPATH=/path/to/gguf-parity-tools \
python3 -m parity_tools token-audit \
  --candidate-metadata /tmp/sarf-atlas-ember-smoke/runs/tokenizer-json/metadata.tokenizer-json.json \
  --reference-metadata /tmp/sarf-atlas-ember-smoke/runs/tokenization-parity-smoke/metadata.llamacpp.json \
  --out /tmp/sarf-atlas-ember-smoke/runs/tokenization-parity-smoke/token-audit.json
```

A passing token audit means the audited prompt token IDs match between the two
metadata sources. It does not mean generation, logits, hidden states, or
research claims have been validated.

## Order Of Use

For a local smoke run:

1. Run extraction.
2. Run `validate-run` on the produced Ember artifact directory.
3. Run `validate-backends` only if there is a second Ember run to compare.
4. Run `gguf-parity-tools` for external token/logit/layer parity audits.

This keeps structural honesty, backend comparison, and external numerical
parity as separate checks.
