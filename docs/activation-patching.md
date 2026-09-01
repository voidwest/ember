# Activation Patching (v0.2, experimental)

`activation-patch` replaces one live activation during generation with a
tensor captured from a previous run. It exists for controlled research
interventions: restoring a zeroed contribution, transplanting a hidden
state, ablating in reverse: not for inference optimization.

**Patched runs are not comparable to ordinary benchmark runs.** Output from
patched runs is research evidence with the patch provenance attached; never
report throughput or quality numbers from intervention runs.

## Usage

```bash
# capture the baseline (see docs/activation-artifacts.md)
ember --arch qwen3 --model MODEL.gguf --tokenizer tokenizer-qwen3.json \
  --prompt "P" --max-tokens 8 --temperature 0 \
  --capture-activations capture.toml

# intervene (here: zero layer 4's MLP contribution)
ember ... --zero-layer-output 4:mlp ...

# patch the intervention away using the baseline capture
ember ... --activation-patch runs/baseline/manifest.json \
  --patch-target 4:after-mlp:prefill \
  --patch-target 4:after-mlp:decode:5 \
  --patch-target 4:after-mlp:decode:6 \
  --capture-activations capture.toml
```

`--patch-target LAYER:STAGE:PHASE[:POSITION]`:

- `LAYER`: layer index
- `STAGE`: `before-layer`, `after-attention`, `after-mlp`, `after-layer`,
  `before-logits`, `after-logits`
- `PHASE`: `prefill` or `decode`
- `POSITION`: absolute decode position (optional)

`--patch-target` is repeatable. `--activation-patch` conflicts with the other
experiments; it can ride alongside `--capture-activations` (the captured
tensors then show the post-patch values, and the manifest records the patch
as the active experiment).

## Source resolution is unambiguous

Each target must resolve to **exactly one** record in the source artifact:

- with `POSITION`: exactly one record at that position;
- without `POSITION`: exactly one record matching (layer, stage, phase).

Zero or multiple matches are hard errors that list the candidates. The first
match is never chosen implicitly.

## Validation

At load time: artifact schema version, record dtype (`f32`) and byte order
(little-endian), record shape vs. the manifest, and (at model load) layer
range and hidden width. At hook time: the live tensor length must match the
source exactly: patching a different prompt length or hidden size fails
clearly instead of corrupting.

## The frozen restoration criterion

The reference workflow (`scripts/research_example_capture_patch.sh`) is:

1. **A**: normal run with capture (baseline activations + logits)
2. **B**: controlled intervention (`zero-layer-output`) with capture
3. compare A vs B: must differ
4. **C**: patch A's activation back into the same model/prompt
5. compare A vs C: the frozen criterion: **all captured logits must be
   bit-identical (sha256-equal)** to A's, because the patch restores the
   exact pre-intervention tensor and the downstream computation is
   deterministic.

Generated text equality alone is **not** sufficient evidence; the artifact
comparison is the criterion. If bit-identical restoration is ever not
achievable in an environment (nondeterministic kernels, changed binaries),
the degraded fallback is max abs diff ≤ 1e-5 and cosine ≥ 0.99999 on the
final-position logits, documented as such: never asserted silently.

## Failure modes

- target never applied → generation fails with a clear error naming the
  target (distinguishes "hook never reached" from "position never occurred");
- shape mismatch at the hook → clear error, no partial write;
- missing/ambiguous source record → error at startup, before any inference.

## Model-family stage semantics

The six stages are defined in docs/activation-artifacts.md. They are the same
semantic points across LLaMA, Qwen, and Gemma execution paths (fast,
workspace, and generic); gemma4's `after-logits` fires after the final logit
softcap.
