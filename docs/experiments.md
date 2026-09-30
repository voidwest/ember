# Ember experiments (v0.5)

The v0.5 experiment workflow turns exact token selection, semantic
hidden-state capture, activation intervention, execution provenance, and
offline verification into reproducible experiment bundles that can be
run without writing Rust.

## Five-minute quick start

Prerequisites: a release build (`cargo build --release`) and a supported
GGUF model with its `tokenizer.json`. The reference example is pinned to
Llama-3.2-1B-Instruct-Q8_0 (see `examples/experiments/README.md` for the
expected file and checksum).

```bash
# 1. validate the specification (no inference)
ember experiment validate examples/experiments/morphology-layerwise-capture.toml

# 2. run it: captures prompt-final and target-final-subtoken across all
#    16 layers, writes runs/morphology-baseline
ember experiment run examples/experiments/morphology-layerwise-capture.toml

# 3. inspect the bundle (captures, identity, hashes)
ember experiment inspect runs/morphology-baseline

# 4. verify it fully offline
ember experiment verify runs/morphology-baseline

# 5. run the intervention (zeroes layer 8) and compare
ember experiment run examples/experiments/morphology-intervention.toml
ember experiment compare runs/morphology-baseline runs/morphology-intervention

# 6. run the exact-restoration leg and confirm the baseline is reproduced
ember experiment run examples/experiments/morphology-restoration.toml
ember experiment compare runs/morphology-baseline runs/morphology-restoration

# 7. reproduce the baseline bundle from the model file
ember experiment reproduce runs/morphology-baseline --model Llama-3.2-1B-Instruct-Q8_0.gguf
```

## CLI surface

```text
ember experiment validate <spec.toml> [--json]           # also a sweep spec
ember experiment run <spec.toml> [--execution reference|planned|planned-fused]
                                [--threads <n>] [--output <dir>] [--retain-incomplete]
                                [--variant <spec.toml> ...] [--json]
ember experiment inspect <bundle> [--json]
ember experiment verify <bundle> [--model <model.gguf>] [--tokenizer <tokenizer.json>]
                                 [--json]
ember experiment compare <bundle-a> <bundle-b> [--json]
ember experiment reproduce <bundle> --model <model.gguf> [--output <dir>] [--json]
ember experiment tokenize --model <model.gguf> --arch <arch> --tokenizer <tokenizer.json>
                          --text "<text>" [--match-span "<span>"] [--json]
```

CLI output states the experiment schema, model and tokenizer identity,
execution mode, plan hash, capture/intervention counts, output directory,
semantic hash, and the verification result. Per-tensor detail lives in
the bundle and is exposed through `inspect`.

## Workflow semantics

1. **Resolve**: the TOML spec is parsed strictly (unknown fields and
   unknown schema majors fail); defaults are applied and recorded.
2. **Load and validate**: model and tokenizer SHA-256 are verified
   against the spec when provided; mismatches fail closed.
3. **Tokenize and align**: token selection is exact, byte-based, and
   fail-closed (see `docs/token-selection.md`).
4. **Execute**: every input is generated through the existing v0.4
   execution machinery; captures and interventions fire at the six
   public semantic hook sites (see `docs/v05-research-contract.md`).
5. **Bundle**: a deterministic `ember.bundle.v1` is staged and
   atomically renamed into place only after all payloads, checksums, and
   the manifest are complete (see `docs/bundle-schema-v1.md`).
6. **Self-verify**: the run command verifies the bundle it just wrote.

## Running a baseline and its variants together

```text
ember experiment run <baseline.toml> --variant <intervention.toml> [--variant <other.toml> ...]
```

loads the model once, runs the baseline, and then runs each variant,
writing one ordinary bundle per spec (each to its own `output.directory`).
A baseline and an intervention run over the same prompt compute the same
prefill up to the first block the intervention can change, so the
baseline computes that shared prefix once and each variant starts its
prefill there instead of recomputing it:

- **Boundary.** For each input, the boundary is the earliest block `k` in
  which a prefill-phase intervention of either run acts: a per-layer site
  at layer `L` gives `k = L` (the residual stream entering block `L`
  precedes every site of that block); a `final-norm-output` or `logits`
  site gives `k = n_layers`. Interventions addressed to generated steps act
  only in decode and do not move the boundary.
- **What is reused.** During the baseline prefill Ember records the
  residual stream entering block `k` and, after prefill, the KV cache. The
  variant copies that cache (layers `< k` hold exactly the K/V it would
  compute) and runs blocks `k..` through the same generic prefill code a
  full forward uses. Captures the variant requests before the boundary are
  recorded during the baseline by an observer instance of the variant's own
  experiment and handed over; captures, interventions and snapshots at or
  after the boundary fire in the variant's run as usual. Decode is never
  shared.
- **Identity.** A variant bundle is bit-identical to running its spec
  alone: same semantic hash, same payload hash. Only `runtime.json`
  differs; its `prefix_reuse` object records the role (`base`, `variant`,
  `co-baseline`) and, per input, `path = "resumed"` with `resume_layer`, or
  `path = "full-recompute"` with the reason.
- **Fallbacks.** An input recomputes in full when the runs differ in
  model, tokenizer, architecture, execution mode, threads, generation
  settings, seed or input; when an intervention acts in block 0; when the
  prompt is a single token (a one-token prefill takes the fused decode
  route, which has no block boundary); or when any runtime check (prompt
  tokens, cache capacity) fails. Specs naming a different model file,
  tokenizer or architecture are rejected outright.

The GUI runs every baseline/intervention pair this way.

## Layer sweeps

"Where does it matter?" is one spec, not sixteen. Add a `[sweep]` table to
an ordinary spec (see `examples/experiments/morphology-layer-sweep.toml`):

```toml
[sweep]
layers = "all"                # or [2, 4, 8], or { start = 0, end = 16, step = 2 }
positions = [3, 7]            # optional: absolute token positions
interventions = ["replace"]   # optional: which interventions move (default: all)
```

Each *point* is the spec with the swept interventions at `layers = [L]`
(and, with `positions`, `tokens = { kind = "absolute-token", index = P }`);
everything else is unchanged. The *baseline* is the spec without
interventions. Swept interventions must use per-layer sites; a
`capture-from-bundle` source (fixed layer) cannot be swept, and a
`capture-from-current-run` source must be at the swept site and capture
every swept layer. A sweep spec never resolves as a single experiment.

```text
ember experiment validate sweep.toml     # the sweep and its template
ember experiment run sweep.toml          # baseline + one bundle per point + summary
ember experiment verify <sweep-dir>      # every bundle, derivations, metrics, sweep hash
ember experiment compare <sweep-a> <sweep-b>
ember experiment inspect <sweep-dir>
```

**Layout.** One ordinary `ember.bundle.v1` per point was chosen over one
bundle with per-point outputs: the bundle schema, `verify`, `compare`,
`inspect` and `reproduce` stay exactly as they are, and a point bundle is
bit-identical to running its derived spec alone.

```text
<sweep>/sweep.toml          the sweep spec, byte for byte
<sweep>/sweep.json          ember.sweep.v1: identities, per-point metrics, sweep_hash
<sweep>/sweep.csv           the same metrics as a table (one row per point and input)
<sweep>/sweep-runtime.json  timings and each bundle's prefix-reuse path (not hashed)
<sweep>/baseline/           bundle; experiment.toml is the derived baseline spec
<sweep>/points/layer-07/    bundle per point (layer-07-pos-3 with positions)
```

A derived spec is the sweep spec with `[sweep]` removed, the swept
interventions moved, `experiment.name` suffixed with the point id and
`output.directory` set to `<spec output.directory>/points/<id>`, preceded by
a comment naming the sweep spec's SHA-256. `--output` moves the sweep
directory without changing any derived spec, so identities do not depend on
where a sweep was written.

**Metrics** (per point and input, against the baseline, from `compare`):
`first_divergent_step` (first generated step whose token differs),
`generated_text_equal`, `peak_relative_l2` (largest relative L2 difference
over every capture both bundles hold, rounded to 9 significant digits so it
survives JSON exactly) with the capture, site and layer where it occurs, and
how many captures were exact. `sweep_hash` covers the sweep spec hash, the
model and tokenizer, the layers, every bundle's semantic and payload hash,
and the metrics.

**Verification** recomputes `sweep_hash`, checks the stored sweep spec
against its hash, fully verifies every bundle (with `--model`/`--tokenizer`
for deep checks), checks each bundle's `experiment.toml` is exactly the spec
the sweep derives for that point, recomputes every metric from the bundles,
and checks `sweep.csv`. `--expect-semantic-hash` anchors the sweep hash.
`compare` verifies both sweeps and compares them point by point (semantic
and payload hash, metrics); the verdict is `exact` when every point and the
baseline have equal semantic hashes.

**Execution.** A sweep is one shared pass (previous section): the baseline
runs once with observers for every point, and each point resumes at its
layer, so a 16-layer sweep computes the prompt prefix once rather than
sixteen times and loads the model once. Points run one after another: each
forward already uses the whole thread pool, and the forward is not
reentrant on one thread (the fused decode workspace and the greedy logits
buffer are thread-local `RefCell`s that a rayon worker stealing another
point's task mid-forward would re-borrow), so parallel points would need
per-point pools and would compete for memory bandwidth. On the pinned
Llama-3.2-1B (Apple M1 Pro, 8 threads) the example sweep takes about 13 s
(4.9 s model load, 7.5 s for 17 bundles); running the baseline and the 16
derived specs as separate `experiment run`s takes about 86 s, with
hash-identical bundles.

## Determinism and identity

Two equivalent runs on the same environment produce identical semantic
manifests and identical semantic hashes. The semantic hash covers every
deterministic file (specs, token selection records, generated tokens,
payload checksums, plan). Timestamps, hostnames, paths, and timing live
in `runtime.json`, which is verifiable but excluded from the identity
(see `docs/reproducibility.md`).

## Performance isolation

The experiment machinery is inert unless the `experiment` subcommand
runs: no spec is parsed, no bundle metadata allocated, and no hooks fire
during ordinary `ember run` inference. The reference example runs in a
few seconds on the pinned Q8_0 model.

## References

- `docs/experiment-schema-v1.md`: the specification language.
- `docs/bundle-schema-v1.md`: the bundle layout and identity rules.
- `docs/token-selection.md`: token selection and Arabic alignment.
- `docs/interventions.md`: interventions and restoration.
- `docs/reproducibility.md`: verification, comparison, reproduction.
- `docs/v05-research-contract.md`: the frozen research contract and
  gates.

## Related v0.1/v0.2 interfaces

The earlier experiment interfaces remain available: `--activation-stats`,
`--zero-layer-output`, `--capture-activations`, `--activation-patch`, and
`compare-artifacts` (see `docs/activation-artifacts.md` and
`docs/activation-patching.md`). The v0.5 workflow supersedes them for new
research; the old interfaces keep their v0.2 semantics unchanged.
