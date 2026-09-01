# v0.2 Validation Report (2026-08-01)

Validation of the v0.2 research-experiment release (capture, patch, compare)
on commit `77b20a9c`, release binary 26.2 MB (was 25.6 MB pre-v0.2; +2.3%).
All artifacts in this report were produced by the repository's own tooling.

## Repository gates

- `cargo fmt --all -- --check`: clean
- `cargo check --all-targets`: clean
- `cargo clippy --all-targets --all-features -- -D warnings`: clean
- `cargo test --all-targets`: **161 passed** (23 new v0.2 tests), 7 ignored
  (docs), 0 failed
- pytest: **27 passed**

New unit tests: npy reader round-trip + rejection, capture config parse /
validation / filters / record cap / artifact determinism, patch target parse,
unambiguous resolution (unique, position-qualified, ambiguous, missing),
width/layer validation, apply isolation (stage/layer/phase/position),
shape-mismatch-at-hook, never-applied failure, compare exact / perturbation /
shape / dtype / missing / duplicate-refusal / JSON determinism.

## No-experiment parity (pre-v0.2 binary vs v0.2 binary)

`--dump-logits` on "The capital of France is", compared bit-for-bit:

- Qwen3-0.6B: **bit-identical** (max diff 0.0)
- Llama-3.2-1B: **bit-identical**
- Llama-3.2-3B: **bit-identical**

No-experiment generation path (`run_single_prompt`, `DisabledHooks`) is
structurally untouched; the bench-decode sanity run on the same binary:
median 36 tok/s on 4 threads, commit `77b20a9c`.

## Frozen restoration criterion (research example, both families)

`scripts/research_example_capture_patch.sh` on Qwen3-0.6B (layer 4) and
Llama-3.2-1B (layer 6), prompt "The capital of France is", 4 tokens, greedy:

- **Run A** (capture): generated ids `12095 13 576 6722` (qwen3) /
  `12366 13 578 469` (llama)
- **Run B** (zero-layer-output L:mlp + capture): ids `279 6722 3283 315` /
  `12366 11 323 279`: fully diverged from A
- **Run C** (patch A's activations back + capture): ids identical to A on
  both families
- **A vs B**: `status=differs`, 0/8 identical, 8/8 differing
- **A vs C**: `status=identical`, 8/8 identical: **logits bit-identical**
  (sha256-equal `[1, 151936]` after-logits records)

PASS on both model families. Generated-text agreement alone was never used
as evidence; the artifact comparison is the criterion.

## Capture validation

- Real run captured 8 records: prefill `[5, 1024]` after-mlp (generic),
  3 decode `[1, 1024]` after-mlp (generic for qwen3), 4 `[1, 151936]`
  after-logits records. Manifest carries schema `0.2.0-experimental`,
  model sha256, GGUF metadata (file_type 7, quantization_version 2,
  size_label 0.6B), tokenizer hash, prompt hash, input/generated token IDs,
  thread count, tracing state, and the capture-config hash
  (`fnv1a64:cb177c97542f1639`).
- **Determinism**: rerun of run A produced identical record keys, tensor
  hashes, and prompt hash.
- **Post-intervention capture**: run B's captured after-mlp record at the
  zeroed layer has abs_max exactly 0.0 (capture runs after the experiment).
- **Dispatch provenance**: llama run records `prefill=generic`,
  `decode=fast` per evaluation and per record: a mixed-path run is
  represented explicitly, not collapsed to one value.

## Patch validation

- Source resolution: position-qualified and unique-triple targets resolve;
  ambiguous (two decode records, no position) and missing targets are hard
  errors listing candidates.
- Validation: dtype (`f32`), byte order (little-endian), record-vs-manifest
  shape, layer range, hidden width; live-vs-source length at the hook
  (different prompt length fails clearly, no partial write).
- Isolation: only the selected (layer, stage, phase, position) is replaced;
  other stages/layers/positions untouched; application counts correct.
- Never-applied target → generation fails, distinguishing "hook never
  reached" from "position never occurred".
- No mutation without `--activation-patch` (runs A/B in the workflow are
  the same binary with the flag absent).
- Provenance: run C manifest records the experiment
  `activation-patch` with source manifest path, per-target source record
  index, and source tensor sha256.

## Compare validation

- identical (8/8, status identical); one-element perturbation (max abs diff
  exactly 0.5 on the perturbed record only); shape mismatch (metrics None,
  shape_match false); dtype mismatch (status differs); missing records on
  either side; duplicate keys → hard refusal ("ambiguous record
  alignment"); JSON output byte-identical across repeated runs; only
  `created_at_unix` is ignored (listed as the sole ignored field).

## Scope notes

- Captured logits records (`after-logits`) are `[1, 151936]`: ~600 KB per
  record; capture cost is bounded by `max_records` when set.
- v0.2 deliberately unsupported: plugins, bindings, WASM, concurrent
  experiments, weight/KV mutation, backend/tokenizer replacement, async
  hooks, remote storage, cross-model projection, fuzzy alignment,
  benchmark claims from patched runs. Capture/patch participate only in the
  generation path.
- Artifacts may contain prompt text and activations; `omit_prompt_text`
  retains prompt hash + token IDs without the text (documented in
  docs/activation-artifacts.md).
