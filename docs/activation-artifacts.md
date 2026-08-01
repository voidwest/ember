# Activation Artifacts (v0.2, experimental)

**Schema `0.2.0-experimental`. No compatibility guarantee.** The manifest
shape, record naming, and file layout below are versioned for the v0.2 series
but may change in any future release. Treat every field as unstable unless a
later release explicitly stabilizes it.

## What an artifact is

A capture artifact is produced by `--capture-activations capture.toml` during
a generation run. It records selected live activations at the six semantic
hook stages:

- `before-layer` — residual stream before the layer's input RMSNorm
- `after-attention` — attention output projection, **before** the residual add
- `after-mlp` — MLP down projection, **before** the residual add
- `after-layer` — residual stream after the residual add
- `before-logits` — final-RMSNorm output (last token), before the LM head
- `after-logits` — logits `[1, vocab]`; for gemma4 this is **after** the final
  logit softcap

Layout:

```
output_dir/
  manifest.json
  tensors/
    prefill_layer004_after-mlp_pos000000.npy
    decode_layer004_after-mlp_pos000005.npy
    ...
```

Tensor files are little-endian f32 npy, one record per
(phase, layer, stage, start position). Names are deterministic and unique by
construction. Prefill records are whole-sequence tensors `[seq, hidden]`;
decode records are `[1, hidden]`. `before-logits` / `after-logits` records are
not per-layer and always carry layer `0` (the layers filter does not apply to
them).

## Capture config (TOML, typed)

```toml
schema_version = 1          # experimental; must be 1
output_dir = "runs/capture-demo"
layers = [4]                # required, non-empty
stages = ["after-mlp", "after-layer"]   # see the six stages above
phase = "both"              # prefill | decode | both
token_positions = []        # decode-step filter; empty = all decode steps
max_records = 64            # 0 = unlimited; truncation is flagged in the manifest
omit_prompt_text = false    # true: manifest prompt = null, hash + token IDs retained
```

This is not a general configuration language: the keys above are the entire
surface, and every value is validated. The exact config file bytes are hashed
(`config_hash`, fnv1a64) into the manifest so a capture can be replayed from
identical selection semantics.

## Manifest contents

- **schema**: `schema_version`, `artifact_kind`, `ember_version`, `git_commit`
- **model**: family, identifier, architecture, layer_count, hidden_size, model
  sha256, file size, tokenizer sha256, GGUF architecture + quantization
  metadata
- **run**: prompt text **or `null`** (with `omit_prompt_text`), `prompt_hash`,
  input token IDs, generated token IDs, thread count, tracing state, CPU
  metadata (from the run manifest), and per-evaluation dispatch observations
  (`prefill -> generic`, `decode -> fast`, ...)
- **experiment**: the active experiment's name and arguments, or `none`
- **capture_selection**: the effective selection (including `config_hash`)
- **records**: per record — index, phase, layer, stage, start position, token
  count, shape, dtype (`f32`), byte order (`little-endian`), tensor path,
  tensor sha256, l2 norm, abs max, dispatch path
- **truncated**: true when `max_records` stopped capture early
- **created_at_unix**: provenance only; the compare tooling explicitly
  ignores it

## Determinism

Given identical inputs (model, prompt, binary, thread count, tracing state),
repeated runs produce identical record names, tensor bytes, hashes, and
manifest content except `created_at_unix`. The compare tool ignores exactly
that one field.

## Sensitive data

Artifacts may contain the prompt text and model activations. Treat them as
research data:

- Set `omit_prompt_text = true` to keep prompt text out of the manifest while
  retaining the prompt hash and token IDs.
- Activations can be reconstructed into meaningful hidden states; do not
  publish them casually.
- Patched runs are not comparable to ordinary benchmark runs — never report
  throughput or quality numbers from intervention runs.

## Comparing artifacts

```bash
ember compare-artifacts --left runs/a/manifest.json --right runs/b/manifest.json
ember compare-artifacts --left ... --right ... --json --output report.json
```

Records align on (phase, layer, stage, start position). Duplicate keys on
either side are a hard error — alignment never guesses. Missing or extra
records are reported. Per-record output: bit-exact equality, max/mean/RMS
difference, cosine, L2 norms, relative L2 error, shape and dtype match.
Run-level provenance fields (model hash, tokenizer hash, prompt hash, token
IDs, versions) compare exactly or are reported; only `created_at_unix` is
excluded. JSON output is deterministic.

## Known limits (v0.2)

- `token_positions` filters decode steps only; prefill records are
  whole-sequence.
- `max_records` caps the record count; the run itself is unaffected.
- Capture participates only in the generation path (with or without an
  experiment), never in probe/extract/dump modes.
