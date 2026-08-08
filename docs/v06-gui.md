# Ember v0.6 experiment consoles (`ember gui`, `ember web-gui`)

The v0.6 GUI is a thin, offline presentation layer over the existing v0.5
experiment pipeline, built for one use case: a live conference demo on a
single laptop. It is **not** a desktop product, a web application, or a
parallel research API. It adds no new experiment semantics, no inference
logic, and no weaker validation. Two consoles share one core: the same
`GuiSession` (resident model + baseline/intervention/restore state), the
same `parse_run_request` gate, and the same `prepare_run` /
`execute_prepared` run path.

- `ember gui` — native single-window console (iced, tiny-skia software
  rendering; no GPU or webview dependency). See "Native console".
- `ember web-gui` — browser console: a tiny localhost HTTP server serving
  one self-contained page. Documented below.

## What it is (browser console)

`ember web-gui` starts a tiny HTTP server bound to `127.0.0.1` and serves
one self-contained page (`src/gui_page.html` — inline CSS/JS, no framework,
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
with the v0.5 `verify_bundle` machinery — schemas, determinism, and
verification semantics are untouched.

## Architecture

```
browser (one HTML page, dir="auto" Arabic/RTL)
   │  JSON over HTTP (localhost)
   ▼
src/gui.rs — request handling, session state, spec building
   │  reuses, never duplicates
   ▼
src/cli_experiment.rs — prepare_run() / execute_prepared()   (v0.5 run path)
src/cli_generation.rs — generate_with_experiment()            (generation loop)
src/cli_support.rs    — architecture/tokenizer resolution      (shared helpers)
ember::v05::{spec, run, verify, hook, intervention, capture, token_select, runner}
ember::llama / loader / plan / quant_k                          (model + kernels)
```

### The v0.5 split (`prepare_run` / `execute_prepared`)

`execute_resolved_inner` was split into two `pub(crate)` functions in
`src/cli_experiment.rs`:

- `prepare_run(resolved, k_strategy, k_allow_fallback) -> PreparedRun` —
  loads the GGUF model, resolves the architecture, loads and validates the
  tokenizer, and records provenance hashes. No inference happens here.
- `execute_prepared(prepared, resolved, spec_text, output_dir, retain) ->
  (path, BundleIdentity, VerificationReport, Vec<InputResult>)` — builds the
  execution plan, runs every input through `generate_with_experiment` with a
  v0.5 experiment attached, assembles + writes the bundle, and self-verifies
  it. The CLI's `execute_resolved` calls both in sequence, so
  `ember experiment run` behavior is unchanged; the GUI keeps the
  `PreparedRun` resident and calls `execute_prepared` per run.

The GUI therefore reuses: model loading, tokenization, generation, hooks,
captures, interventions, bundle assembly, bundle writing, and verification
— all of it unchanged.

## Build and run

```bash
cargo build --release
./target/release/ember gui                # native console (iced window)
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
pipeline. The UI is built with iced on the tiny-skia software renderer (no
GPU, no system-webview dependency), and the embedded Noto fonts in
`src/gui_fonts/` (Noto Sans, Noto Sans Mono, Noto Naskh Arabic — SIL OFL
1.1, see `src/gui_fonts/LICENSE.txt`) provide Latin + Arabic coverage
offline, so rendering is identical on any machine. Arabic input/output is
shaped and laid out RTL by cosmic-text (iced's text engine).

The window has a left sidebar (model picker + path, hook stage, layer,
intervention, source, target tokens), a main panel (prompt, baseline and
intervention outputs side by side, verification panel), and a status bar
(model · layer/hook · intervention · bundle · elapsed · throughput). Every
experiment runs in a worker thread through the shared `GuiSession` core —
the same code the browser console uses — so model residency, bundle
writing, and verification semantics are identical. Runs are serialized;
the model stays resident across baseline / intervention / restore.

Requires a display (X11 or Wayland); it is a local window, not a server.

## Page workflow

One screen: left sidebar (experiment configuration), main area (prompt,
baseline/intervention outputs, verification panel), bottom status bar.

1. **Model** — pick a discovered `*.gguf` or type a path, click **Load**.
   The model stays resident for the whole session.
2. **Prompt** — free text, Arabic works (`dir="auto"`).
3. **Hook stage** — one of the six v0.5 semantic sites, labelled with their
   v0.4 stage ids: `before-layer`, `after-attention`, `after-mlp`,
   `after-layer`, `before-logits`, `after-logits`. The list comes from
   `SemanticHookSite::ALL` (`stage_id()`), not a duplicated table.
4. **Layer** — 0..n-1 for per-layer sites; hidden for the two head-boundary
   sites.
5. **Intervention** — `replace`, `zero`, `scale`, `interpolate`, `add-delta`
   with the same semantics as v0.5: `scale` takes a factor, `interpolate`
   takes an alpha, `replace`/`interpolate`/`add-delta` take a source.
   Sources are either `capture (previous layer)` — a v0.5
   `capture-from-current-run` at a configurable source layer (the capture
   fires before the intervention in the same pass, so the source layer must
   not be deeper than the intervention layer) — or `zero`.
6. **Run experiment** — executes two real experiments: a capture-only
   baseline run and a capture+intervention run. Both write v0.5 bundles;
   both self-verify. Baseline and intervened text appear side by side.
7. **Verify restore** — runs the `restore-original` leg at the same
   site/layer/selection and reports **restore: BIT-EXACT** when the output
   equals the stored baseline (it always does — that is the point of the
   check), or a mismatch otherwise. The comparison is only made when the
   shared configuration (model, prompt, site, layer, token selection) is
   unchanged since the last run.

Errors are surfaced in a red panel and never hidden: invalid specs, missing
spans, out-of-range layers, load failures, and bundle self-verification
failures all appear as readable messages.

## HTTP API (local only)

| Endpoint | Body | Returns |
| --- | --- | --- |
| `GET /` | — | the page |
| `GET /api/state` | — | version, commit, discovered models, hook stages, session info |
| `POST /api/prepare` | `{model_path}` | session info (arch, layers, embed dim, vocab, load ms) |
| `POST /api/run` | run configuration | baseline + intervention outputs, bundle ids, verification, timing |
| `POST /api/restore` | shared configuration | restore output + bit-exact verdict vs stored baseline |

Every response is `{ok: true, ...}` or `{ok: false, error: "..."}`; malformed
requests return a readable error envelope, never a silent failure.

## Ember APIs reused (not duplicated)

- `ember::v05::spec` — raw + resolved spec types, `resolve()` validation
  (Gate A)
- `ember::v05::hook::SemanticHookSite` — the six hook sites and their stage
  ids (source of truth for the hook selector)
- `ember::v05::intervention` / `capture` / `token_select` — operation,
  source, layer, and token semantics (prompt-final / matched-span)
- `cli_experiment::prepare_run` / `execute_prepared` — the run path
- `ember::v05::run::write_bundle` + `ember::v05::verify::verify_bundle` —
  bundle writing and self-verification (unchanged schemas)
- `ember::v05::runner::InputResult` — generated text, token counts, events
- `ember::loader` / `ember::llama` / `ember::plan` / `ember::quant_k` —
  model loading and execution (via the shared path)
- bundle `runtime.json` — honest wall-clock / throughput for the status bar

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
- The restore comparison is text-level: the output of the restore leg must
  equal the baseline text. Activation-level bit-exactness is guaranteed by
  the v0.5 snapshot mechanism and visible via the snapshot checksum in the
  intervention event; the bundle's own verification covers artifact
  integrity.
- The model picker scans the working directory tree (depth-limited); models
  elsewhere must be typed in by path.
- `before-logits` scaling usually does not change greedy output (logits are
  scaled near-uniformly) — a real property of the model, not a bug.
- The GUI is not a server product: no auth, no telemetry, no cloud, no
  network access of any kind; it binds localhost and serves one client.

## Demo script (60–90 s)

1. `./target/release/ember web-gui` — the browser opens on the console.
2. Pick a Q8_0 model (e.g. Qwen2.5-1.5B or Llama-3.2-1B), click **Load**
   (~1–2 s; note the arch/layers/dim readout).
3. The prompt is prefilled: اكتب جملة قصيرة عن المدينة المنورة.
4. Hook `after-mlp`, Layer defaulted near the top, Type `scale` × 0.50.
   Click **Run experiment** (~2–4 s).
5. Baseline and intervention outputs are side by side and differ; the
   status bar shows the intervention, bundle id, elapsed time, and tok/s;
   the verification panel shows **VERIFIED**.
6. Click **Verify restore** — **restore: BIT-EXACT** appears: the restore
   leg reproduces the baseline output exactly.
7. Change Layer or Type (e.g. `zero`, or a lower layer), click **Run** again
   and watch the intervened output change while the baseline stays fixed.
8. Close with the status bar: model · layer/hook · intervention · bundle ·
   elapsed · throughput — everything a reproducibility-minded audience asks
   for. Bundles live under `runs/gui/`; inspect one with
   `ember experiment inspect runs/gui/intervention-*`.
