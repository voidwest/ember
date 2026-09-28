# Bundle schema v1

`ember.bundle.v1` is the deterministic, self-verifying experiment bundle
produced by `ember experiment run`.

## Layout

```text
runs/example/
├── manifest.json            identity: schema, status, semantic/payload hashes, file list
├── semantic-manifest.json   deterministic semantics (the hashed document)
├── runtime.json             machine-dependent metadata (never hashed)
├── experiment.toml          verbatim user specification
├── resolved-experiment.json resolved spec with recorded defaults
├── model.json               model identity + GGUF metadata + quantization inventory
├── tokenizer.json           tokenizer identity
├── execution-plan.json      the v0.4 ExecutionPlan (build time sanitized)
├── inputs.jsonl             input ids, texts, prompt hashes
├── outputs.jsonl            generated token ids, text, final top-1 per input
├── tokenization.jsonl       token ids, pieces, byte offsets per input
├── captures/
│   ├── tensors.safetensors  tensor payloads (name-ordered, 8-byte aligned)
│   └── index.jsonl          per-tensor index entries
├── interventions/events.jsonl  every intervention application
├── traces/events.jsonl      capture route records + per-layer fusion state
├── checksums.sha256         SHA-256 of every file at publish time
└── verification.json        written by `verify` (runtime state)
```

## Identity

```json
{
  "semantic_hash": "...",
  "payload_hash": "..."
}
```

- The **semantic hash** identifies the scientific execution semantics:
  SHA-256 over the canonical JSON of `semantic-manifest.json`.
- The **payload hash** identifies the complete deterministic artifact
  contents: SHA-256 over the sorted inventory of `semantic-manifest.json`
  payload checksums plus the semantic manifest's own file (it cannot list
  itself).

Canonical JSON: sorted object keys, stable array order, shortest
round-trip float formatting, UTF-8. `runtime.json`, `manifest.json`,
`checksums.sha256`, and `verification.json` are excluded from both
hashes.

## What is deterministic vs runtime

Deterministic (in the semantic hash): schema versions, experiment name
and semantic configuration, model/tokenizer SHA-256, architecture, layer
count, execution mode, plan hash, hook sites, capture and intervention
definitions, resolved token selectors and selected token IDs, generated
token IDs, deterministic output text, payload checksums, deterministic
warnings.

Runtime (in `runtime.json` only): timestamp, hostname, OS, CPU features,
thread count, wall-clock timing, throughput, peak RSS, compiler version,
process ID, model/tokenizer local paths, scratch bytes.

`resolved-experiment.json` carries the output directory (a placement
decision) and is therefore excluded from the semantic payload inventory;
it remains checksum-verified.

## Compatibility

- unknown bundle schema major: verification fails;
- `semantic-manifest.json` records `experiment_schema`, `hook_schema`,
  and `plan_schema`; verification accepts exactly the implemented versions
  (`ember.experiment.v1`, hook 1, plan 1), even when unknown versions carry
  self-consistent hashes;
- the stored execution plan must use the declared schema and match the plan
  hash in the semantic manifest;
- renamed hook sites or changed hook meanings require a semantic hook
  schema major bump; verification never reinterprets an old hook id with
  new semantics.

## Publishing

Bundles are written into a sibling staging directory
(`runs/.example.tmp-<pid>-<seq>`) and atomically renamed into place only
after every payload is flushed, checksums are computed, the manifest is
complete, and internal verification passes. Publication uses a no-clobber
rename for new destinations. Explicit replacement uses an atomic directory
exchange, keeping the old bundle visible until the verified replacement is
ready. Linux/macOS filesystems must support the required rename operation;
unsupported platforms/filesystems fail without deleting the destination.
On failure the staging
directory is removed unless `--retain-incomplete` keeps it; a retained
staging directory starts with `.` and contains `.tmp-`, so `verify` can
never mistake it for a bundle. The writer alone can run internal verification
on a staging directory. Existing bundles are never overwritten
unless `output.overwrite = true`.

## File inventory and JSON ordering

Verification rejects symlinks and special files within a bundle before opening
its documents. Every regular file must appear in the manifest, apart from
`checksums.sha256` and the optional runtime `verification.json`. Every listed
file must have a checksum; duplicate checksum paths fail. The inventory walk
is bounded to 100,000 entries and 64 directory levels. Verify an immutable
bundle copy; this is not an atomic snapshot of a concurrently modified tree.

JSON object keys in newly written deterministic payloads are recursively
sorted, independently of dependency features and map insertion order. Arrays
retain their semantic ordering. Older bundle bytes are verified in place; the
verifier does not rewrite them into a newer encoding.

The verifier also recognizes the historical insertion-order semantic hash
encoding emitted by released Ember 0.5.0/0.5.1 and 0.6.0–0.6.8 builds. It
reports that encoding in the semantic-hash check and preserves the original
hash. This exception does not apply to newer producers. New writers use sorted
keys; no in-place bundle migration is performed. See the frozen fixture and
provenance in [the contract audit](audits/1.0-contract-audit.md).
