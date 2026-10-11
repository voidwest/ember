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
ember experiment lens <bundle> --model <model.gguf> [--tokenizer <tokenizer.json>]
                            [--top-k <n>] [--json] [--out <lens.json>]
```

CLI output states the experiment schema, model and tokenizer identity,
execution mode, plan hash, capture/intervention counts, output directory,
semantic hash, and the verification result. Per-tensor detail lives in
the bundle and is exposed through `inspect`.

## Logit lens

`ember experiment lens` reads the residual-stream captures of a bundle and
asks, for every captured row, which next token the model would predict if
the network stopped at that depth. Each row goes through the model's own
final RMS norm and LM head (untied `output.weight` or tied embeddings); the
lens calls the same `Llama::final_norm` / `Llama::lm_head` functions the
forward pass uses, so it is not a re-implementation.

```bash
ember experiment run examples/experiments/morphology-layerwise-capture.toml
ember experiment lens runs/morphology-baseline \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf --top-k 5
ember experiment lens runs/morphology-baseline \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf --json --out /tmp/lens.json
```

Fail-closed rules (the same as `reproduce`):

- The bundle is fully verified first; the anchors `--expect-semantic-hash`
  and `--trusted-key` (with `--expect-evidence`, or the sibling
  `<bundle>.evidence.json`) are accepted.
- The model must hash to the bundle's recorded model SHA-256, and its
  layer count, width and vocabulary must match the manifest.
- The tokenizer (`--tokenizer`, else the path the bundle's bound spec
  names) must hash to the recorded tokenizer SHA-256; it is hashed and
  parsed from one read.
- Nothing is written into the bundle. `--out` must point outside it.

Per captured `(capture, input, position)` the report gives the token at the
position and the token that actually followed it (the next prompt token, or
the model's generated token), then for every captured layer: the top-k
tokens with probabilities, the rank (1-based, ties to the lower id as in
greedy argmax) and probability of the actual next token, the entropy (nats),
and `KL(final || layer)` (nats) against the final-depth row of the same
position.

Sites: only residual-stream sites are projected. `residual-post-mlp` at
layer `L` is the stream after `L + 1` blocks (reported as `depth`);
`residual-pre-attention` at layer `L` is the stream after `L` blocks.
`attention-output` and `mlp-output` (projections before their residual
add), `final-norm-output` and `logits` are listed as skipped with the
reason. KL needs the final-depth row, i.e. a `residual-post-mlp` capture of
the last layer.

Final-layer equality: the last layer's lens *is* the model's final-logits
computation. A unit test (`v05::lens::tests::final_layer_lens_equals_model_logits`)
runs a synthetic Llama (untied and tied heads) through the experiment
execution path and checks that the captured last-layer row, projected by
the lens, is bit-identical to the logits the model produced, for prefill
and for a decode step. The report's `final_layer_check` counts final-depth
rows whose next token was generated and how many have that token as lens
top-1 (all of them for a greedy run).

Limits: rows at prompt positions reproduce the prefill route exactly. Rows
at generated positions were produced by the decode route, which for Q8_0
models uses fused single-token kernels (and, in planned modes, the plan
interpreter); these agree with the lens projection within kernel
tolerance, not bit-for-bit. F16 captures are a rounded stream. The lens is
a read-out of the as-run stream: with interventions in the bundle it
projects the intervened state.

## Workflow semantics

1. **Resolve**: the TOML spec is parsed strictly (unknown fields and
   unknown schema majors fail); defaults are applied and recorded.
2. **Load and validate**: model and tokenizer SHA-256 are verified
   against the spec when provided; mismatches fail closed. The tokenizer
   is read once, and those bytes are hashed and checked against the pin
   before they are parsed. The model is hashed while it loads; a model
   load error, then a model mismatch, still takes precedence over any
   tokenizer error.
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
  after the boundary fire in the variant's run as usual. Decode steps are
  never shared between runs, but they can be batched (below).
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

The GUI runs every baseline/intervention pair this way, and its layer sweep
runs one shared pass for all layers: every layer's baseline observes a single
generation (co-baselines: non-intervening specs with the same inputs,
settings and generated-step sites) and each layer's intervention resumes at
its own layer.

**Batched decode.** On models with the Q8_0 fast decode path (the example
Llama-3.2-1B Q8_0 model), the runs of a shared pass -- `--variant` runs, a
sweep's baseline and points, the GUI pair and GUI sweep -- also decode
together: for each input, after the base and every variant have prefilled,
each decode step is one batched forward over every generation still running,
reading each weight once for the whole batch. Each sequence keeps its own KV
cache, position, hooks, captures, interventions, sampler/seed and stopping
condition, and leaves the batch when it stops. Batched decode is
bit-identical per sequence to single decode, so every bundle is unchanged.
`runtime.json` (not part of the identity) records `decode_batch`:
`{"path": "batched", "batch_size": N, "generations": M}`, or
`{"path": "sequential", "reason": ...}` when the pass cannot batch (a model
off the Q8_0 fast path, tracing, `EMBER_FUSED_GREEDY`, runs with different
thread counts or execution modes, or a single generation).
`EMBER_BATCHED_DECODE=0` forces the sequential pass.

## Layer sweeps

"Where does it matter?" is one spec, not sixteen. Add a `[sweep]` table to
an ordinary spec (see `examples/experiments/morphology-layer-sweep.toml`):

```toml
[sweep]
layers = "all"                # or [2, 4, 8], or { start = 0, end = 16, step = 2 }
positions = [3, 7]            # optional: absolute token positions
alphas = [0.0, 2.0, 4.0]      # optional: sets `alpha` of the swept interventions
interventions = ["replace"]   # optional: which interventions move (default: all)
```

`alphas` sweeps the `alpha` of `steer` (or `interpolate`) interventions,
crossed with `layers` (and `positions`) when both are given; with `alphas`
alone the swept interventions keep their declared layers and point ids are
`alpha-0`, `alpha-2`, `alpha-neg-1.5`; crossed, `layer-07-alpha-2`. Every
swept intervention must have an `alpha`. `sweep.json` then records `alphas`
and each point's `alpha`, and `sweep.csv` gains an `alpha` column (layer
sweeps keep their exact earlier format). See
`examples/experiments/steering-sentiment-sweep.toml`.

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
<sweep>/sweep-effect.csv    with [sweep.effect]: the effect summary, one row per point
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

### Effect statistics across inputs

A sweep point that changes one prompt tells you little. An effect table
measures one number per input and summarizes the change over all inputs, for
each point. Add `[sweep.effect]` to a sweep spec (see
`examples/experiments/capital-effect-sweep.toml`):

```toml
[[captures]]
id = "answer"
site = "logits"                 # the effect reads this row
[captures.tokens]
kind = "prompt-final"

[sweep.effect]
capture = "answer"
confidence = 0.95               # optional (default 0.95)
resamples = 10000               # optional bootstrap resamples (default 10000)
seed = 0                        # optional bootstrap seed (default 0)

[[sweep.effect.targets]]        # one entry for each input
input = "france"
target = " Paris"               # token text (exactly one token) or a token id
foil = " Berlin"
```

**Metric.** `m = logit(target) - logit(foil)` in the capture row of the
input. The effect of a point on an input is `m(point) - m(baseline)`. A
sweep with one point (`layers = [8]`) measures one intervention over the
prompt set.

**Rules.** The capture must be at site `logits`, store its rows (not
`summary-only`), and apply to every input. Each input must have exactly one
target. A token text that is not exactly one token, or a target that is the
same token as its foil, fails before the sweep runs. The capture must give
one row for each input; a `generated-step` capture fails when generation
stops before that step.

**Summary, per point.**

- `n`, `mean`: the number of inputs and the mean effect.
- `sd`, `standard_error`: the sample standard deviation (`n - 1`) and
  `sd / sqrt(n)`.
- `ci_low`, `ci_high`: a bootstrap percentile interval of the mean.
  Ember draws `resamples` samples of `n` inputs with replacement (SplitMix64,
  `seed`), takes each sample mean, and interpolates linearly between order
  statistics (Hyndman-Fan type 7). Every point uses the same seed, so all
  points use the same resampled input indices.
- `positive`, `negative`, `zero`: the sign counts of the effects.
- `sign_test_p`: a two-sided exact sign test over the non-zero effects
  (`min(1, 2 P(X <= min(positive, negative)))`, `X ~ Binomial(m, 1/2)`).

With one input, `sd`, `standard_error` and the interval are `null`. The
interval and the test describe variation across the prompts that you
supplied. They do not describe a population of prompts that you did not
sample, and a sweep over many points makes many comparisons.

**Outputs.** `sweep.json` gains an `effect` record (the metric, the
interval settings, and the resolved target and foil tokens for each input).
Each point gains an `effect` summary, and each input gains
`baseline_metric`, `point_metric` and `effect`. `sweep.csv` gains these
three columns, and `sweep-effect.csv` has one row for each point.
`experiment run` and `experiment inspect` print the summary table. A sweep
without `[sweep.effect]` writes the same files and the same `sweep_hash` as
before.

**Compatibility.** The new fields are optional additions. The sweep readers
are strict, so an Ember binary without these fields rejects a sweep spec with
`[sweep.effect]`, and a `sweep.json` with `effect` records. It does not reinterpret them. Point bundles do not change: each one
is an ordinary `ember.bundle.v1` bundle.

**Verification.** `verify` reads the effect from the bundles again,
calculates the summary again from the recorded per-input effects, and
compares both exactly. All calculations use IEEE addition, multiplication,
division and square root in a fixed order, so the result is the same on
every machine. `verify` compares token ids that the spec gives. It encodes
token text again only with `--tokenizer`; without it, the `sweep effect`
check says how many token texts it did not check.

## Attribution patching

"Which (layer, site, position) carries the difference between two prompts
into the answer?" without autograd and without patching everything. Add an
`[attribution]` table to a spec with a clean and a corrupted input of equal
token length (see `examples/experiments/attribution-capital.toml`):

```toml
[attribution]
clean = "clean"                 # input ids
corrupted = "corrupted"
target = " Paris"               # token text (exactly one token) or a token id
foil = " Rome"
sites = ["residual-pre-attention", "attention-output", "mlp-output"]  # default
layers = "all"                  # default
positions = "all"               # default; or "final", or [3, 7]
verify_top_k = 10               # default: real patches for the top 10
```

**Metric.** `m = logit(target) - logit(foil)` at the final prompt position.

**Approximation (direct-path attribution patching).** One capture pass
records every candidate site in full for both prompts, the corrupted run's
residual `x` entering the final norm, and both final logit rows. With
`u = W_U[target] - W_U[foil]` (dequantized LM-head rows) and the final RMS
norm `y = g * x / s`, `s = sqrt(mean(x^2) + eps)`, the exact gradient of `m`
with respect to `x` is

```text
r = (g*u)/s - x * sum(g*u*x) / (d * s^3)
```

and every candidate is scored

```text
estimate = (a_clean - a_corrupted) . r     at the final position
estimate = 0                                at every other position
```

A patched difference at `attention-output`/`mlp-output` is added to the
residual stream and at `residual-pre-attention` it replaces it; either way
its *direct* contribution to `x` is the difference itself, so this is
attribution patching (gradient x activation difference) with every path
through later blocks removed. It needs two forward passes for all
candidates and no backward pass.

**Limits, stated plainly.** Indirect effects are ignored: a difference that
later blocks transform or amplify is scored only by its direct projection
(early residual candidates are underestimated: in the example, the layer-10
residual scores 1.8 against a measured 8.4). Positions other than the last
reach the logits only through attention, so they score exactly zero; the
corrupted token's own position, often where the largest real effect is, is
not ranked by the estimate. Within tied (zero) estimates, candidates are
ordered by `|a_clean - a_corrupted|`. The estimate is first-order exact only
for the last block's projection outputs (up to the final norm's curvature).
This is why the workflow always verifies.

**Verification.** The `verify_top_k` best-ranked candidates are patched for
real: the corrupted prompt runs with the clean row written at that site,
layer and position (`replace`), and `actual = m(patched) - m(corrupted)` and
`recovered_fraction = actual / (m(clean) - m(corrupted))` are recorded.
The report gives the Spearman and Pearson correlations and the sign
agreement between `estimate` and `actual` over the verified candidates.

**Outputs.** The bundle is the ordinary run of the spec's inputs (captures
allowed; interventions are refused, the workflow runs its own patches) plus

```text
artifacts/attribution/attribution.json   ember.attribution.v1: tokens, metrics, every candidate in rank order
artifacts/attribution/candidates.csv     the same as a table
```

and `experiment run` prints the ranked table. Both files are hashed payloads;
`verify` adds an `attribution report` check (candidates in rank order, the
correlation summary equals what the recorded values give, verified
candidates are exactly the top ranks, the CSV is the report's table, the
inputs are the spec's). Re-running the spec reproduces the report bit for
bit. Attribution specs run standalone: not in a sweep or a `--variant` pass.

On the pinned Llama-3.2-1B (France vs Italy, 9 tokens, 432 candidates, 48
verified, about 8 s), the top five candidates are the final-position
residual stream at layers 11-15 (recovering 53-79% of the 16.3-logit gap),
with Spearman 0.72 and Pearson 0.92 between estimate and measured effect.

## Probe bridge: can the model use it?

A probe that reads a feature says nothing about whether the model's answer
path uses it (README: "the probe can read it. can the model use it?"). A
`[probe]` table measures both in one run (see
`examples/experiments/probe-bridge-sentiment.toml`):

```toml
[probe]
site = "residual-post-mlp"      # per-layer site the probe reads and the interventions act at
layer = 8
tokens = { kind = "prompt-final" }            # rows per labelled prompt (mean if several)
train = [{ text = "...", label = 1 }, ...]    # trained in the run when no `file`
test = [{ text = "...", label = 0 }, ...]     # held-out labelled prompts
ridge_lambda = 1.0              # default
# file = { path = "probe.npy", sha256 = "<64 hex>" }   # a pinned direction instead
ablate = true                   # default: remove the projection on the direction
steer_alphas = [-8.0, 8.0]      # steer along it (default none)
steer_normalize = "unit"        # default
intervene_tokens = { kind = "prompt-final" }  # rows of the behavioural inputs
target = " great"               # token whose logit/probability is measured
```

**Probe.** Trained probes are closed-form ridge regressions on +1/-1
targets with an unpenalized intercept (centred data, dual form
`w = Xc^T (Xc Xc^T + lambda I)^-1 yc`, Cholesky in f64): no seed, no
iterations, the same weights every run. A pinned `file` (`.npy` or
`.safetensors`, `[d]` or `[n_layers, d]`) replaces training; labelled
`train` examples then only fit the threshold (midpoint of the class-mean
projections). Accuracy is reported on `train` and `test`.

**Causal effect.** The spec's `[[inputs]]` are the behavioural prompts; they
must not appear among the probe's examples. The baseline and every variant
(`ablate`, then `steer<alpha>` per alpha) run them with the spec's
generation settings, intervening at the probe's site and layer on
`intervene_tokens` along the probe direction (the `ablate-projection` and
`steer` operations). For every variant and input the report records the
target token's logit and probability at the final prompt position, their
change from the baseline, the generated text, whether it changed, and the
first divergent step; a summary gives the mean changes and how many texts
changed per variant.

**Outputs.** The bundle is the ordinary (unintervened) run of the inputs plus

```text
artifacts/probe/probe.json              ember.probe-bridge.v1: probe record, effects, summary
artifacts/probe/effects.csv             effects as a table
artifacts/probe/direction.safetensors   the direction used (F32 [d])
```

and `experiment run` prints the accuracy/effect summary. `verify` adds a
`probe bridge report` check: the record describes the spec's probe (site,
layer, source, file hash, example counts), the direction tensor matches its
checksum, the effects cover the baseline and each variant for every input,
the summary is what the effects give, and the CSV is the table. Probe specs
declare no interventions and run standalone (not in a sweep or a
`--variant` pass).

On the pinned Llama-3.2-1B, a 12-example sentiment probe at layer 8 reads
sentiment perfectly (train and held-out accuracy 100%), yet removing its
projection barely moves the model (mean change in logit(" great") -0.16,
one of three continuations changed: "really good" became "pretty good");
steering along it by -8 turns all three continuations negative ("a
disaster", "terrible"). The direction is readable and, pushed hard enough,
usable; the model does not depend on its component along it.

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
