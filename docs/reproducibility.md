# Reproducibility (v0.5)

## Verification

`ember experiment verify <bundle>` is fully offline. Basic verification
checks: required files exist, bundle schema and kind, completion status,
every checksum in `checksums.sha256`, capture-index consistency
(unique ids, shape/byte-length arithmetic), tensor payload
shape/dtype/checksum agreement with the index, no unindexed or missing
payload tensors, token-selection record consistency, intervention
reference resolution, execution-plan hash recomputation, semantic hash
recomputation, payload hash recomputation, and semantic payload
checksums.

Deep verification (`--model <model.gguf> [--tokenizer <tokenizer.json>]`)
additionally checks the model SHA-256, architecture, and layer count
against the manifest.

Failures return a nonzero exit code; the machine-readable report
(`--json`) lists every check.

## Comparison

`ember experiment compare <a> <b>` separates scientific differences from
machine noise:

- identity: schema compatibility, semantic/model/tokenizer hashes,
  execution mode, plan hash, input ids, prompt and tokenization
  equality;
- outputs: generated token/text equality, final top-1 equality, first
  divergence step;
- captures: shape/dtype equality, exact equality, maximum and mean
  absolute difference, relative L2, cosine similarity, finite-value
  mismatches (payloads are loaded tensor-by-tensor, not all at once);
- interventions: operation, source, layer/site, selected-token, and
  de-fusion-route equality plus event counts;
- runtime (reported separately, never merged into semantic verdicts):
  decode/prefill throughput, first-token latency, peak RSS, scratch
  bytes, hook overhead.

Text output leads with scientific differences; `--json` is
deterministic.

## Reproduction

`ember experiment reproduce <bundle> --model <model.gguf>`:

1. reads the bundle's resolved experiment;
2. validates the supplied model SHA-256 against the bundle record;
3. re-runs the experiment to a new bundle (never overwriting the
   original);
4. compares against the original and classifies:

- `exact-semantic` — identical semantic hashes (same output directory
  placement, bit-identical execution);
- `exact` — identical tokens and exact captures;
- `output-equivalent` — identical tokens, captures within the
  float envelope;
- `top1-equivalent` — only the final top-1 agrees;
- `failed` — divergence or incompatibility.

A run is never called reproduced merely because generated text matches
while requested captures differ.

## Deterministic vs runtime metadata

The semantic hash covers only deterministic content. Timestamps,
hostnames, timing, local paths, RSS, and process IDs live in
`runtime.json`, which is excluded from both hashes. Two equivalent runs
produce identical semantic manifests and identical semantic hashes
(Gate E); the reference example reproduces `exact-semantic` on this
machine.

## Security assumptions

Experiment files and bundles are treated as untrusted input: path
traversal and absolute paths in bundle indexes are rejected, payload
sizes are validated before slicing, tensor dimensions are checked before
multiplication, outputs are written atomically, existing bundles are not
overwritten without permission, and no embedded commands or dynamic
libraries are ever executed.
