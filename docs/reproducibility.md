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
(`--json`) lists every check. Verification never writes into the bundle it
checks, so shared or read-only bundles are not modified; `--write-report
<path>` saves the JSON report to a path outside the bundle. A
`verification.json` left inside a bundle by an earlier Ember release is
ignored (and reported as ignored): it cannot influence the verdict.

### What "verified" means

Every value verification compares against, including `checksums.sha256`,
the hashes in `manifest.json`, and the payload checksums in
`semantic-manifest.json`, is written by whoever produced the bundle. A
verified bundle is therefore **self-consistent**: nothing was corrupted or
edited without also updating the hashes. It is not proof that the bundle is
the one a paper, a colleague, or your own earlier run produced: an edited
bundle can be fully resealed and will verify.

To bind a bundle to an identity obtained elsewhere, anchor it:

- `--expect-semantic-hash <hex>`: the recomputed semantic hash must equal a
  value you recorded somewhere you trust (a paper, a lab notebook, a signed
  commit). `verify` prints the semantic hash on success for this purpose.
- `--expect-evidence <envelope> --trusted-key <key.pub>`: a signed evidence
  envelope over the bundle's `manifest.json` (`ember evidence sign --manifest
  <bundle>/manifest.json`) must verify against the trusted key, and its
  signed semantic and payload hashes must equal the recomputed ones. See
  [attested execution](embersec/attested-execution.md). With only
  `--trusted-key <key.pub>`, the envelope is looked up next to the bundle as
  `<bundle>.evidence.json`; a missing envelope fails the anchor.

### Signed bundles

`ember experiment run` signs the bundle it writes when given a key:

```bash
ember evidence init --key ~/.ember/sign.key          # once; writes sign.key + sign.pub
ember experiment run spec.toml --sign-key ~/.ember/sign.key
export EMBER_SIGN_KEY=~/.ember/sign.key              # or sign every run by default
ember experiment run spec.toml                       # --no-sign opts out
```

After the bundle is written and self-verified, its `manifest.json` (which
carries the semantic and payload hashes) is signed into a
`signed-evidence-v2` envelope written **next to** the bundle as
`<bundle>.evidence.json`, never inside it: a file inside would change the
bundle's inventory and fail verification of the bundle it signs. Anyone
holding the public key checks it with

```bash
ember experiment verify runs/probe --trusted-key sign.pub
```

which finds `runs/probe.evidence.json` on its own (pass `--expect-evidence
<path>` if the envelope was moved). `experiment reproduce` takes the same
options for the original bundle. Publish the `.pub` file (or its
fingerprint) somewhere independent of the bundle, such as a paper or a
repository README: an envelope checked against a key shipped alongside it
proves nothing about who signed. `experiment reproduce` does not sign the
reproduction bundle; sign it with `ember evidence sign --manifest
<reproduction>/manifest.json --key <key> --out <reproduction>.evidence.json`.

`experiment reproduce` accepts the same options for the original bundle,
and `experiment compare` accepts `--expect-a-semantic-hash` and
`--expect-b-semantic-hash`. Anchors are checked before anything else runs.
Without an anchor the verdict still says `verified`, and the text output
states that this means self-consistent only.

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

1. verifies the bundle (checking any anchors) and reads its resolved
   experiment from the verified bytes;
2. binds `resolved-experiment.json`, which is outside the semantic hash, to
   the hashed identity: it must equal what the hashed `experiment.toml`
   resolves to (apart from the execution mode, thread count, and output
   directory a run may override, and the informational list of applied
   defaults), and its experiment metadata, execution mode, inputs, captures
   and interventions must match the semantic manifest; otherwise
   reproduction is refused;
3. validates the supplied model SHA-256 against the bundle record, and
   requires the tokenizer (the one the spec names, or `--tokenizer`) to
   match the recorded tokenizer SHA-256;
4. re-runs the experiment to a new bundle (never overwriting an existing
   directory, whatever the bundle's spec says);
5. compares against the original and classifies:

- `exact-semantic`: identical semantic hashes (same output directory
  placement, bit-identical execution);
- `inputs-differ`: the two bundles did not run the same input IDs and
  prompts, so no output comparison is meaningful;
- `exact`: identical tokens and exact captures;
- `output-equivalent`: identical tokens, captures within the
  float envelope;
- `captures-misaligned`: a requested capture is missing from one side;
- `top1-equivalent`: only the final top-1 agrees;
- `failed`: divergence or incompatibility.

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
