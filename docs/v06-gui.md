# Ember v0.6 experiment consoles (`ember gui`, `ember web-gui`)

The v0.6 GUI is a thin, offline presentation layer over the existing v0.5
experiment pipeline, built as a native research workbench. It is **not** a
chat client, analytics dashboard, web application, or parallel research API.
It adds no new experiment semantics, no inference logic, and no weaker
validation. Two consoles share one core: the same
`GuiSession` (resident model + baseline/intervention/restore state), the
same `parse_run_request` gate, and the same `prepare_run` /
`execute_prepared` run path.

- `ember gui`: native single-window workbench (GPUI Kit, GPU accelerated,
  Metal on macOS and X11/Wayland on Linux). See "Native console".
- `ember web-gui`: browser console: a tiny localhost HTTP server serving
  one self-contained page. Documented below.

## What it is (browser console)

`ember web-gui` starts a tiny HTTP server bound to `127.0.0.1` and serves
one self-contained page (`src/gui_page.html`: inline CSS/JS, no framework,
no external assets, no network access). The page renders in any modern
browser, which is what makes Arabic input/output render correctly: each
text field uses `dir="auto"`, so Arabic text is shaped and laid out RTL by
the browser while the application chrome stays LTR. A theme toggle in the
header switches the console between light and dark; the default follows
the system preference and the choice persists in localStorage.

Every action in the page is translated into an `ember.experiment.v1`
specification in the exact raw form a user would write by hand, resolved
through the standard `RawExperimentSpec::resolve()` gate (the same
validation `ember experiment validate` runs), and executed by the same
`prepare_run` / `execute_prepared` code path as `ember experiment run`.
Bundles are written with the v0.5 `write_bundle` machinery and self-verified
with the v0.5 `verify_bundle` machinery: schemas, determinism, and
verification semantics are untouched.

## Architecture

```
browser (one HTML page, dir="auto" Arabic/RTL)
   │  JSON over HTTP (localhost)
   ▼
src/gui.rs: request handling, session state, spec building
   │  reuses, never duplicates
   ▼
src/cli_experiment.rs: prepare_run() / execute_prepared()   (v0.5 run path)
src/cli_generation.rs: generate_with_experiment()            (generation loop)
src/cli_support.rs   : architecture/tokenizer resolution      (shared helpers)
ember::v05::{spec, run, verify, hook, intervention, capture, token_select, runner}
ember::llama / loader / plan / quant_k                          (model + kernels)
```

### The v0.5 split (`prepare_run` / `execute_prepared`)

`execute_resolved_inner` was split into two `pub(crate)` functions in
`src/cli_experiment.rs`:

- `prepare_run(resolved, k_strategy, k_allow_fallback) -> PreparedRun` :
  loads the GGUF model, resolves the architecture, loads and validates the
  tokenizer, and records provenance hashes. No inference happens here.
- `execute_prepared(prepared, resolved, spec_text, output_dir, retain) ->
  (path, BundleIdentity, VerificationReport, Vec<InputResult>)`: builds the
  execution plan, runs every input through `generate_with_experiment` with a
  v0.5 experiment attached, assembles + writes the bundle, and self-verifies
  it. The CLI's `execute_resolved` calls both in sequence, so
  `ember experiment run` behavior is unchanged; the GUI keeps the
  `PreparedRun` resident and calls `execute_prepared` per run.

The GUI therefore reuses: model loading, tokenization, generation, hooks,
captures, interventions, bundle assembly, bundle writing, and verification
all of it unchanged.

## Build and run

```bash
cargo build --release
./target/release/ember gui                # native console (GPUI Kit window)
./target/release/ember web-gui            # browser console; prints http://127.0.0.1:8337/ and opens a browser
./target/release/ember web-gui --port 9000    # custom port
./target/release/ember web-gui --no-open      # just print the URL
```

The browser console binds `127.0.0.1` by default (offline, local-only);
the native console is a local window and exposes no network surface at
all. K-quant and Q8_0 GGUF models are supported through the same
`--k-strategy` plumbing as the CLI (default `auto`). For a fast demo loop
prefer Q8_0 models: K-quant decode is intentionally much slower.

## Native console (`ember gui`)

`ember gui` is a native, single-window console over the exact same v0.5
pipeline. The UI is built with GPUI Kit 0.6.6. Kit owns the application root,
buttons, searchable selectors, single-line inputs, multiline prompt editor,
keyboard editing, clipboard, undo/redo, and platform input-method integration.
The shared experiment engine remains independent of the control library.
Embedded Noto Sans, Noto Sans Mono, and Noto Naskh Arabic fonts provide offline
glyph coverage (SIL OFL 1.1, see `src/gui_fonts/LICENSE.txt`). Platform shaping
can differ; identical font files do not promise identical rendering everywhere.

GUI development uses the pinned Rust 1.98.1 toolchain because GPUI Kit's
transitive dependencies do not compile on 1.92. The headless CLI/library still
supports `cargo +1.92.0 build --release --no-default-features`. This GUI
requirement is an unreleased next-minor change, not a 0.6.x patch guarantee.

The window is a guided experiment workbench: a narrow workflow rail (Prompt →
Intervention → Review), a dominant scrollable central workspace, a contextual
experiment inspector, and a persistent action dock. A compact two-track
pipeline makes the experiment the persistent visual object: one input forks
into an untouched baseline and an accented internal intervention, then rejoins
at comparison. Research presets configure valid starting points, exact
hook/execution controls remain available under Advanced, and paired results,
layer metrics, token differences, verification evidence, and session-local run
history stay attached to the configuration that produced them. Every
experiment runs in a worker thread through the shared `GuiSession` core :
the same code the browser console uses: so model residency, bundle
writing, and verification semantics are identical. Runs are serialized;
the model stays resident across baseline / intervention / restore.

Uses Metal on macOS and a GPU-backed display on Linux (X11 or Wayland); it is a local window, not
a server. It opens maximized with a 1180×720 restore size and a 980×620
minimum; the workflow rail, workspace, and inspector scroll independently.

## Page workflow

The native workbench uses progressive disclosure rather than presenting the
entire raw specification at once.

1. **Prompt**: pick a discovered `*.gguf` (or enter a raw path under
   Advanced), write the prompt, and choose a friendly Short/Medium/Long
   response length. Arabic and mixed-direction input are supported. Loading
   is optional: **Run experiment** automatically prepares an unloaded model,
   which then stays resident for the session.
2. **Intervention**: choose a plain-language research operation card, its
   location and layer, and the target tokens. Presets provide useful causal
   starting points, including an Arabic matched-span experiment.
3. **Review & results**: run the controlled baseline/intervention pair, then
   switch among Overview, Layers, Tokens, and Raw trace. Overview keeps paired
   outputs and the highest-value layer plot together. A compact landmark band
   reports the first internally divergent layer, peak relative-L2 location,
   and any exact stable token-ID tail; Tokens reports the exact one-based decode
   step at which token ids diverge. Recent run outcomes stay visible in the
   workflow rail.
4. **Advanced controls**: exposes the exact execution engine, numeric token
   limit, raw model path, stage id, operation id, and token-selection id.
5. **Hook stage**: one of the six v0.5 semantic sites, labelled with their
   v0.4 stage ids: `before-layer`, `after-attention`, `after-mlp`,
   `after-layer`, `before-logits`, `after-logits`. The list comes from
   `SemanticHookSite::ALL` (`stage_id()`), not a duplicated table.
6. **Layer**: 0..n-1 for per-layer sites; hidden for the two head-boundary
   sites.
7. **Intervention semantics**: `replace`, `zero`, `scale`, `interpolate`, `add-delta`
   with the same semantics as v0.5: `scale` takes a factor, `interpolate`
   takes an alpha, `replace`/`interpolate`/`add-delta` take a source.
   Sources are either `capture (previous layer)`: a v0.5
   `capture-from-current-run` at a configurable source layer (the capture
   fires before the intervention in the same pass, so the source layer must
   not be deeper than the intervention layer): or `zero`.
8. **Run experiment**: executes two real experiments: a capture-only
   baseline run and a capture+intervention run. Both write v0.5 bundles;
   both self-verify. Baseline and intervened text appear side by side. The
   primary action is progressive: Continue to Intervention, Continue to
   Review, then Run Experiment; two equally strong actions are never shown.
9. **Verify restore**: runs the `restore-original` leg at the same
   site/layer/selection and reports **restore: BIT-EXACT** when the output
   equals the stored baseline (it always does: that is the point of the
   check), or a mismatch otherwise. The comparison is only made when the
   shared configuration (model, prompt, site, layer, token selection) is
   unchanged since the last run.

Errors are surfaced in a red panel and never hidden: invalid specs, missing
spans, out-of-range layers, load failures, and bundle self-verification
failures all appear as readable messages.

## Native result data and charts

Each GUI run adds a `gui-layer-trace` capture at the residual stream leaving
every transformer block (`residual-post-mlp`). It records only the selected
prompt row(s), not full sequence tensors. The baseline and intervention
bundles therefore retain the research source of truth while the worker thread
reduces matching captures through the existing v0.5 `compare_bundles` path.
The UI receives only compact `LayerMetric` values (relative L2 difference,
cosine distance, maximum absolute difference, and exactness), paired generated
token ids/text, and the exact first divergent decode step. No activation tensor
is cloned into GPUI state.

The Phase 1 native plot is **Representation divergence by layer**. It uses the
already-defined relative L2 metric with the baseline capture as reference;
missing/non-finite sparse captures remain absent rather than becoming zero.
The plot includes subtle numeric Y-axis ticks and a hover readout for layer,
relative L2, and cosine distance. Hover is synchronized with the experiment
inspector, clicking pins the point there, the target layer is a dashed marker,
and Copy CSV exports the plotted numeric data. Output cards size to short
continuations while retaining a bounded scroll region for longer traces and
annotate the shared prefix / first changed token directly beneath the text.
GPUI's own `canvas` / `PathBuilder` / `paint_path` primitives render the chart;
there is no plotting dependency, webview, JavaScript runtime, or Python path.

`ExperimentComparison` also carries an optional renderer-independent
`LayerTokenGrid` (row-major values plus explicit layer and token axes). Phase 1
leaves it absent: no heatmap is rendered until a future experiment captures a
defensible layer × token metric efficiently. Activation magnitude and sweep
plots are likewise deferred until their data are retained directly and can be
labelled without implicit normalization.

## HTTP API (local only)

| Endpoint | Body | Returns |
| --- | --- | --- |
| `GET /` | n/a | the page |
| `GET /api/state` | n/a | version, commit, discovered models, hook stages, session info |
| `POST /api/prepare` | `{model_path}` | session info (arch, layers, embed dim, vocab, load ms) |
| `POST /api/run` | run configuration | baseline + intervention outputs, bundle ids, verification, timing |
| `POST /api/restore` | shared configuration | restore output + bit-exact verdict vs stored baseline |

Every response is `{ok: true, ...}` or `{ok: false, error: "..."}`; malformed
requests return a readable error envelope, never a silent failure.

## Ember APIs reused (not duplicated)

- `ember::v05::spec`: raw + resolved spec types, `resolve()` validation
  (Gate A)
- `ember::v05::hook::SemanticHookSite`: the six hook sites and their stage
  ids (source of truth for the hook selector)
- `ember::v05::intervention` / `capture` / `token_select`: operation,
  source, layer, and token semantics (prompt-final / matched-span)
- `cli_experiment::prepare_run` / `execute_prepared`: the run path
- `ember::v05::run::write_bundle` + `ember::v05::verify::verify_bundle` :
  bundle writing and self-verification (unchanged schemas)
- `ember::v05::runner::InputResult`: generated text, token counts, events
- `ember::v05::compare::compare_bundles`: established tensor metrics for the
  baseline/intervention layer trace
- `ember::loader` / `ember::llama` / `ember::plan` / `ember::quant_k` :
  model loading and execution (via the shared path)
- bundle `runtime.json`: honest wall-clock / throughput for the status bar

The only v0.5 change: the raw spec structs in `src/v05/spec.rs` gained
`Serialize` derives so the GUI can emit the canonical raw TOML form (they
were Deserialize-only). No behavior or schema change.

## Known limitations

- The GUI is a demo instrument, not a full experiment authoring tool:
  token selection is limited to `prompt-final` and `matched-span` (all
  subtokens, occurrence 0); cross-bundle sources, inline vectors, full-tensor
  captures, and generated-step selection are not exposed.
- Sampling is fixed at greedy (temperature 0.0) to keep runs deterministic;
  there is no temperature control.
- One run is serialized at a time; the page disables controls while running.
- The Phase 1 native result workspace implements layer divergence and exact
  token-id divergence. Activation-magnitude, parameter-sweep, logits/top-k,
  and heatmap views are intentionally not synthesized from unavailable data.
- The restore comparison is text-level: the output of the restore leg must
  equal the baseline text. Activation-level bit-exactness is guaranteed by
  the v0.5 snapshot mechanism and visible via the snapshot checksum in the
  intervention event; the bundle's own verification covers artifact
  integrity.
- The model picker scans the working directory tree (depth-limited); models
  elsewhere must be typed in by path.
- `before-logits` scaling usually does not change greedy output (logits are
  scaled near-uniformly): a real property of the model, not a bug.
- The GUI is not a server product: no auth, no telemetry, no cloud, no
  network access of any kind; it binds localhost and serves one client.

## Demo script (60–90 s)

1. `./target/release/ember web-gui`: the browser opens on the console.
2. Pick a Q8_0 model (e.g. Qwen2.5-1.5B or Llama-3.2-1B), click **Load**
   (~1–2 s; note the arch/layers/dim readout).
3. The prompt is prefilled: اكتب جملة قصيرة عن المدينة المنورة.
4. Hook `after-mlp`, Layer defaulted near the top, Type `scale` × 0.50.
   Click **Run experiment** (~2–4 s).
5. Baseline and intervention outputs are side by side and differ; the
   status bar shows the intervention, bundle id, elapsed time, and tok/s;
   the verification panel shows **VERIFIED**.
6. Click **Verify restore**: **restore: BIT-EXACT** appears: the restore
   leg reproduces the baseline output exactly.
7. Change Layer or Type (e.g. `zero`, or a lower layer), click **Run** again
   and watch the intervened output change while the baseline stays fixed.
8. Close with the status bar: model · layer/hook · intervention · bundle ·
   elapsed · throughput: everything a reproducibility-minded audience asks
   for. Bundles live under `runs/gui/`; inspect one with
   `ember experiment inspect runs/gui/intervention-*`.


## GPUI Kit migration validation

The console uses GPUI Kit controls rather than its former handwritten text
editor and dropdown implementation. Numeric fields retain the domain parser's
validation instead of silently removing invalid pasted characters. Selectors
search both the displayed description and the exact hook/model value. Native
Quit menus and Cmd+Q (macOS) / Ctrl+Q shortcuts close the application.

Run the automated interaction checks with:

```sh
cargo test --features gui-tests --bin ember gui_native::kit_tests
```

They exercise a rendered headless Kit window: workflow navigation, operation
selection, a preset, mixed Arabic/Latin multiline editing, undo, and searchable
hook selection. They do not inspect rendered pixels or establish native macOS
accessibility behavior.

The optional real-model worker check writes baseline, intervention, and restore
bundles under `runs/gui`, verifies them, and compares restored token IDs:

```sh
EMBER_GUI_TEST_MODEL=/absolute/path/to/model.gguf RAYON_NUM_THREADS=4 \
  cargo test --bin ember native_worker_runs_and_restores_real_model -- --ignored --nocapture
```

This fixture expects a compatible model with more than eight layers and an
available tokenizer, and uses four generated tokens. It is intentionally
explicit rather than downloading a model in ordinary tests. The local Q8 Llama
3.2 1B check passed; it does not establish every model family's equivalence.

On macOS, the optional test renderer captures the production console using real
CoreText fonts and Metal without opening desktop windows:

```sh
cargo run --features gui-tests --bin ember -- gui --render-test-dir /tmp/ember-gui-renders
```

Set `EMBER_GUI_TEST_MODEL` as above to include populated result tabs from a
verified four-token baseline/intervention run. Images cover light/dark themes
at normal and minimum window sizes. The command is only available in macOS
builds with `gui-tests`; it is not part of the release CLI contract.

Desktop accessibility inspection remains unverified: the macOS window starts,
but the inspection tool times out when querying Ember. Offscreen render checks
and virtual-window interaction tests do not establish VoiceOver compatibility.

## First-run and presentation aids

- **Home** carries a three-step "How Ember works", a **Try an example** button
  (offered only when no draft is waiting to be resumed) and **See a sample
  result**.
- The sample result opens Review with a finished comparison and needs no model.
  Its numbers are illustrative -- shaped like a real Llama-3.2-1B run of
  "scale x0.5 at layer 8" -- and the page says so. It is never written to run
  history or saved as a draft, and the first real run replaces it.
- **Presentation mode** (command palette) scales text by 1.18, hides the sidebar
  and inspector, and restores both on exit. Pane toggles made while presenting
  are not persisted.
- Narrow windows fold the inspector away below 1200pt and the sidebar below
  960pt without changing the saved preferences.
- On macOS the Arabic face is the system Geeza Pro: the bundled Noto Naskh
  renders its dots detached under this text stack.
- Keyboard focus rings come from the kit's Button (`focus_ring_enabled`
  defaults on and paints when focused). The offscreen renderer cannot move
  keyboard focus, so the ring has been confirmed in the kit source, not in a
  screenshot.
- Runs keeps each run's comparison (both outputs, the per-layer numbers, the
  token pairs and the landmarks) beside its configuration, so **Open** reopens
  it on Review without a model. Runs recorded before results were kept load
  fine and simply have no Open. Per-side timings are not kept, so a reopened run
  shows none rather than an invented split.

## Packaging on macOS

`scripts/bundle-macos.sh` builds the release binary and wraps it as
`target/bundle/Ember.app` with an icon, an `Info.plist` and a small launcher, so
the app gets a Dock icon and the menu bar reads "Ember" instead of "ember".

```sh
scripts/bundle-macos.sh
open target/bundle/Ember.app
```

Ember finds `.gguf` files by scanning the folder it starts in, and a Finder
launch starts in `/`, so the launcher starts in `$EMBER_MODELS_DIR` if set and
otherwise in the checkout the bundle was built from. The icon is drawn by
`scripts/make_icon.py` (standard library only) and stored as
`assets/macos/Ember.icns`. The bundle is unsigned; Gatekeeper will ask on first
open on another machine.

The menu bar (Ember, Edit, Experiment, View, Help) routes to the same console
methods as the command palette and the key handler. Shortcut hints are not
shown in the menus: the shortcuts are handled by the console's key handler, not
by action key bindings, and binding both would fire each twice.

## The workspace (repeated experiments)

Experiments is one page, not a wizard. Setup (model, prompt, generation length,
the change, where, target) is a pane on the left and is always editable; the
results are on the right. **Run** lives at the bottom of the setup pane and on
Cmd+Enter. This suits the real loop -- change one thing, run, compare, tweak,
run again -- and a second run on an already-loaded model takes about a second.

- A result whose settings have since been edited says so (**Settings changed
  since this result**), and the breadcrumb reads "Settings changed".
- **Pin as reference** keeps a result. The next run is then shown beside it: a
  small table (change, text output, first divergence, peak divergence, and the
  percentage change in peak) on Overview and Layers, and the pinned series drawn
  as a dashed line behind the new one on the chart. A run reopened from Runs can
  be pinned too.
- Before the first run the results side is an empty state with the examples.
- Narrow windows fold the sidebar away below 1240pt so the two panes keep room;
  the saved preference is untouched.

### Sweeps

For a change that acts on one layer, **Sweep all N layers** (under Run) runs the
same experiment at every layer in turn and plots the peak divergence each run
reached, by the layer that was changed. A second run on a loaded model takes
about a second, so a 16-layer model sweeps in roughly twenty. The **Sweep** tab
has the curve, a sentence saying at how many layers the words changed and where
the effect was largest, and one row per layer whose **Open** shows that run in
full. **Copy CSV** exports the curve. Layers the form rejects (a source layer
that is not earlier than the target, say) are skipped, and **Stop** ends a sweep
after the run in flight. Sweep runs are kept on the sweep, not written to
history: sixteen near-identical rows would bury the runs you chose to make.

### Cancelling a run

**Cancel run** (under Run while a run is in flight), **Esc**, the palette's
**Cancel run** and Experiment > Cancel Run all stop the run in flight. The
worker checks the run's cancel token before prefill, at every decode step of
both legs, between the legs and before each bundle is written, so it stops
within a token (a prefill in progress finishes first). The console waits in
*Cancelling* until the worker confirms, then returns to idle with a **Run
cancelled** notice. Nothing is recorded in history and no bundle is kept -- a
baseline bundle already written is removed, as half a pair is not an
experiment -- and the model stays loaded. A run waiting on a model load is
dropped; the load itself finishes. See `docs/cancellation.md`.

### Comparing two runs

On Runs, **Select** two rows and press **Compare runs** in the bar above the
table. The comparison replaces the table: a metrics table (change, model,
prompt, text output, tokens, where the words and the layers first diverged,
peak divergence, duration, verification) with the delta of each, both runs'
outputs side by side, a token diff of the two interventions, and both
layer-divergence series on one chart (the first selected dashed). **Swap**
flips the sides; **Back to runs** returns to the table. Runs recorded before
results were kept can be selected, but the bar says they can't be compared.
(**Pin** on a row is the older, different action: it makes that run the
reference for your next run in the workspace.)

### Exporting runs

**Export** on any Runs row opens an export strip for it:

- **Copy as Markdown** -- the same document as the result page's **Copy
  summary** (settings, outcome, both outputs, the per-layer table), headed
  "Run #N, from history". Older records without a saved result export what
  they recorded.
- **The bundle.** Each run records where its two verified bundles were
  written (absolute paths; history schema minor 3). If they are still on
  disk, **Reveal bundle** shows them in the file manager and **Copy verify
  command** copies `ember experiment verify <baseline> && ember experiment
  verify <intervention>`. If they are gone and the record kept its
  configuration, **Re-run to bundle** runs the stored configuration again
  through the same path as `ember experiment run` (`execute_prepared` ->
  `write_bundle`, self-verified) and points the record at the new bundles.
  No history row is added, and the strip says if the re-run's output differs
  from the stored one. It can be cancelled like any run. A record with neither
  a bundle nor a configuration says it cannot be re-run.

The result page offers **Reveal bundle** for a live result, or for a reopened
run whose bundle is still on disk, and its Markdown names the bundles with the
verify command.

### What changed from the wizard

The contextual inspector is gone (it restated the setup). Its advanced controls
-- execution engine, exact token limit, raw model path, hook ids -- are under
**Advanced** in the setup pane, and the selected layer's values are in the
chart's readout. The Model section collapses to one line once a model is loaded.
On Runs, **Star** (was Pin) keeps a run at the top, **Pin** (was Compare) pins a
saved run as the reference and returns to the workspace, **Open** shows its comparison and
**Reuse** loads its settings. A returning user (with history or an unfinished
experiment) launches straight into the workspace with their setup restored;
Home is for a first launch.
