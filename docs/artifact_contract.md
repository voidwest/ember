# Ember Hidden-State Artifact Contract

This contract is the backend boundary. Native Ember and any llama.cpp extractor
must write the same files with the same schemas. Downstream probing code should
read this contract only; it must not branch on backend-specific outputs.

Contract version: `2`

Layout name: `ember.layer_sharded_npy.v1`

## Run Directory

`output_dir` is the run directory unless `run_id` is set. When `run_id` is set,
the run directory is `output_dir/run_id`.

```text
runs/<run_id>/
  config.toml
  manifest.json
  samples.jsonl
  tokenization.jsonl
  positions.jsonl
  layers/
    layer_0000.npy
    layer_0004.npy
    layer_0008.npy
  logits.npy          # optional, only when write_logits = true
  checksums.json
  report.json
```

## Tensor Storage

Hidden states are sharded by layer. Each layer file is an NPY tensor:

```text
layers/layer_XXXX.npy
shape: [n_samples, hidden_dim]
dtype: little-endian f32
axis 0: sample_index
axis 1: hidden dimension
```

Layer names are zero-padded decimal layer indices: `layer_0000`,
`layer_0001`, and so on. The file name is `<layer_name>.npy`.

The requested layer list and every layer shard path are recorded in
`manifest.json`. A backend must write rows in the same `sample_index` order as
`samples.jsonl`, `tokenization.jsonl`, and `positions.jsonl`.

For Milestone 3, `llama-cpp-external` may write no layer shards. In that case
`tensor_contract.layers` is an empty array and downstream code may use only
`samples.jsonl`, `tokenization.jsonl`, `positions.jsonl`, optional logits, and
metadata. A backend must not claim hidden-state support unless it writes valid
layer shard entries and files.

`logits.npy` is optional. When present:

```text
shape: [n_samples, vocab_size]
dtype: little-endian f32
axis 0: sample_index
axis 1: token id / vocab index
```

## JSONL Files

Every JSONL row has `schema_version = 2`, `sample_index`, and `sample_id`.
`sample_index` is the canonical ordering key across all files.

`samples.jsonl` stores sample identity and prompt provenance:

```json
{
  "schema_version": 2,
  "sample_index": 0,
  "sample_id": "abc",
  "input_index": 0,
  "prompt": "Analyze the word: kataba",
  "prompt_hash": "fnv1a64:..."
}
```

When `prompt_hashes_only = true`, `prompt` is `null` and `prompt_hash` remains
required.

`tokenization.jsonl` stores token IDs and tokenizer offsets:

```json
{
  "schema_version": 2,
  "sample_index": 0,
  "sample_id": "abc",
  "token_ids": [1, 42, 99],
  "token_count": 3,
  "prompt_hash": "fnv1a64:...",
  "offsets": [[0, 0], [0, 7], [8, 14]]
}
```

`positions.jsonl` stores the selected positions and pooling rule:

```json
{
  "schema_version": 2,
  "sample_index": 0,
  "sample_id": "abc",
  "position_mode": "word_mean",
  "pooling": "mean",
  "selected_token_positions": [4, 5],
  "source_field": "word",
  "source_value": "kataba",
  "source_byte_span": [18, 24]
}
```

Position modes:

- `prompt_final`: `pooling = "single"`; selected position is the final
  non-special prompt token.
- `word_final_subtoken`: `pooling = "single"`; selected position is the final
  subtoken overlapping `source_value`.
- `word_mean`: `pooling = "mean"`; selected positions are all subtokens
  overlapping `source_value`.
- `full_prompt_mean`: `pooling = "mean"`; selected positions are all
  non-special prompt tokens.

Backends store the pooled hidden state in each layer shard. They do not store
per-token hidden states in this contract.

## Manifest

`manifest.json` is the run-level index. It records:

- `schema_version`
- `layout`
- run paths
- model metadata
- backend metadata
- tokenizer/model metadata available to Ember
- tensor shapes and layer shard names
- sample count
- `sample_order_hash`
- `config_hash`
- extraction config

Tokenizer metadata belongs in `manifest.json` under backend/model/config
metadata and token IDs belong in `tokenization.jsonl`.

## External Backend Request

Ember calls an external llama.cpp-compatible extractor with:

```text
llama-ember-extract --request <request.json>
```

The request JSON is written by Ember in the run directory as
`llama_cpp_request.json`. It contains:

- `model_path`
- `input_jsonl_path`
- `output_dir`
- `config_path`
- `manifest_path`
- `samples_path`
- `tokenization_path`
- `positions_path`
- `checksums_path`
- `report_path`
- optional `logits_path`
- `prompt_template`
- `sample_id_field`
- `word_field`
- `token_position`
- requested `layers`
- `write_logits`
- `prompt_hashes_only`
- `max_seq_len`
- `run_metadata`

The external extractor is responsible for writing Ember-compatible artifacts to
those paths and exiting nonzero with useful stderr when it cannot. Ember
validates the produced contract after the process exits.

## Ordering

Sample ordering is verified by `sample_index` and `sample_order_hash`.

`sample_order_hash` is computed from the ordered sequence of:

```text
sample_id<TAB>prompt_hash<NEWLINE>
```

A consumer must reject a run when JSONL files have missing, duplicated, or
out-of-order `sample_index` values, or when the computed sample-order hash does
not match `manifest.json`.

## Resume Rules

The contract is resumable by prefix only:

1. `config_hash` must match the existing run.
2. Existing JSONL files must have the same line count.
3. Every existing layer shard must contain the same number of rows.
4. The computed `sample_order_hash` for completed rows must match.
5. A resumed writer may append only the next `sample_index`.

The native runner currently uses a fresh-run policy. External backends should
write temporary files and only publish final paths after row counts and hashes
agree.

## Corruption And Staleness

`checksums.json` maps contract-relative paths to SHA-256 checksums when
`sha256sum` is available. Consumers should verify checksums before reading
tensors.

`report.json` records `status = "complete"` only after all files are flushed,
checksummed, and the manifest has been written. Missing report, non-complete
status, checksum mismatch, config hash mismatch, or sample-order mismatch means
the artifact is stale or corrupted.
