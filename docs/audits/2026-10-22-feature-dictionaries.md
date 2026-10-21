# Research layer audit (feature dictionaries): 2026-10-22

- **Revision/branch:** `feat/feature-dictionaries` on `main` at `c19878f`.
- **Audit date:** 2026-10-22
- **Scope and trigger:** feature dictionaries (ADR-0003):
  `src/v05/dictionary.rs`, the `features` paths in `src/v05/runner.rs`,
  the spec checks in `src/v05/spec.rs`, the verify check in
  `src/v05/verify.rs`, and the driver (`src/cli_experiment_dictionary.rs`,
  `activate_spec` and `finish_bundle`). Trigger: a public schema change.
  `ember.experiment.v1` gains `[[dictionaries]]` and a `features` field on
  captures and interventions, and bundles gain two artifact schemas,
  `ember.dictionary.v1` and `ember.features.v1`.
- **Primary:** research maintainer (voidwest).
- **Backup:** not assigned (open since the 2026-10-11 record).
- **Second reviewer:** an automated review agent read the diff without a
  build. No human second reviewer.
- **Last audit / next audit:** last: `2026-10-21-head-sweeps.md`. Next:
  before the next release tag, or at the next change to either artifact
  schema.

## 1. Map and boundaries

- User-facing commands/APIs: `ember experiment validate|run|verify|reproduce`
  with `[[dictionaries]]` and `features` in the spec. No new command.
- Source entry points: `Dictionary::load` and `from_parts` (SAELens
  folder), `encode`, `encode_selected`, `add_decoded`, `feature_row`,
  `artifact_files`, `verify_dictionary_records`, `verify_feature_artifact`;
  `ExperimentSpecV1::validate_dictionaries`; the feature view in
  `fire_site`; `load_dictionaries` and `PreparedRun::dictionary_cache`.
- Adjacent subsystems and handoffs: capture storage and the payload index;
  cross-bundle sources (encoded for a feature target); sweeps and the shared
  pass (they call `activate_spec`, so they load and inject the same way).
- Persistent artifacts or schemas: `artifacts/dictionaries/<id>.json`
  (`ember.dictionary.v1`) and `artifacts/features/<capture>.json`
  (`ember.features.v1`). Bundles do not copy dictionary weights.
- Environment switches and feature gates: none.
- External references or trusted implementations: SAELens `cfg.json` and
  `sae_weights.safetensors` conventions; an independent NumPy encode for the
  real-model check.
- Supported targets and deliberate non-goals: SAELens `standard` and
  `jumprelu` with a ReLU activation; no transcoders, no TopK, no input
  normalization, no feature sweeps.

## 2. Contracts and invariants

- Ownership/lifetime and resource limits: `cfg.json` is read with a 1 MiB
  cap and the weights with a 2 GiB cap through `bounded_read`; both files
  are hash-checked before they are parsed. Weights are shared through an
  `Arc` and loaded once per session for each folder and hash.
- Shape, encoding, safety, or error behavior: every tensor shape must match
  `cfg.json`, every value must be finite, and every feature id must be below
  `d_sae` (checked in the driver and again in the runner). `features` with
  `heads`, at another site or layer, or with `steer`, `ablate-projection` or
  `restore-original` fails at resolve time.
- Numerical, determinism, trace, or hook boundaries: encode and decode
  accumulate in f64 in index order and round once. The selected and the full
  encode give the same bits, and an edit that changes nothing is
  bit-identical. No hook timing changes.
- Compatibility/provenance requirements: every new field is skipped when it
  is absent, so earlier specs, plans and manifests serialize as before, and
  `HOOK_SCHEMA_VERSION` is unchanged.
- Reference/oracle path versus optimized path: not applicable. The feature
  view runs in the same runner code on every execution mode.

## 3. Checks and evidence

Host: Apple M1 Pro, macOS, Rust 1.98.1 (GUI) and 1.92.0 (headless MSRV),
`CARGO_BUILD_JOBS=2`.

| Check | Command / fixture | Result | Evidence or artifact |
| --- | --- | --- | --- |
| Static/build | `scripts/ci_local.sh quick` (part of `full`) | pass | fmt, clippy (all configurations), MSRV, docs |
| Focused unit/integration | `v05::dictionary` tests; `cli_experiment_dictionary` tests on a 4-layer synthetic model | pass | ReLU and JumpReLU encode; both encode paths equal; no-op edit bit-identical; edit equals `x + (f' - f) · W_dec` bit for bit; refusals; tamper checks |
| Full suites | `scripts/ci_local.sh full` | pass: 1,746 Rust tests and 143 Python tests (1 skipped); `cargo audit` not run (not installed locally; CI runs it) | all suites; run before the last fix (the cache key), which `ci_local.sh quick` and the dictionary tests cover |
| Compatibility | `tests/bundle_compatibility.rs`, `tests/sweep_compatibility.rs` | pass (in `full`) | frozen bundles and sweeps verify unchanged |
| External reference | NumPy float64 encode of the stored rows, 72 rows of 8 prompts, two SAELens checkpoints | pass | top-32 indices and `l0` agree on every row; values within 7.9e-6; relative errors within 1.6e-15 |
| Smoke (real model) | Llama-3.2-1B-Instruct Q8_0, layer 9 `residual-post-mlp`, eight corrupted capital prompts | pass | feature patch +9.76 logits against residual patch +9.80; `scale` 1.0 on all features: effect 0 on 8/8 |
| Golden logits | not run | n/a | no change to the forward pass without `features` |
| Performance (controlled A/B) | not run | n/a | no hot-path change for runs without dictionaries |
| Backup reproduction | not run | n/a | no backup assigned |

## 4. Findings

| Severity | Finding / evidence | Owner or issue | Due / disposition |
| --- | --- | --- | --- |
| required | Verify compared the recorded row norm for exact equality. `serde_json` without `float_roundtrip` can read a written f64 back one unit in the last place off, so a valid bundle could fail. | research maintainer | fixed before merge: compared to within 1e-12 relative, with a test. CI on x86_64 then showed the same round trip in an end-to-end test (`relative_error` read back one unit in the last place off); the test now compares the f64 fields to within 1e-12 too |
| required | `cfg.json` fields that change the encode were ignored: `activation_fn_str` (a TopK SAE would have run as ReLU), `activation_fn_kwargs`, `finetuning_scaling_factor` and `hook_head_index`. A missing `apply_b_dec_to_input` became false. | research maintainer | fixed before merge: each is refused, and `apply_b_dec_to_input` must be stated |
| required | Only the weights were hash-pinned, but `cfg.json` selects the architecture. | research maintainer | fixed before merge: `config_sha256` is required and recorded, and verify compares it |
| required | The session cache was keyed by folder and weights hash only, so a second entry with another `config_sha256` came from the cache unchecked. | research maintainer | fixed before merge: the key holds both hashes, with a test |
| observation | The public SAELens checkpoint used for the smoke check is not sparse on these activations: about 4,800 of 16,384 features are active per token (JumpReLU), and the ReLU checkpoint of the same release has almost none. Ember agrees with NumPy, so this is the checkpoint, not the encode. | research maintainer | documented in `docs/experiments.md`; single-feature results from this checkpoint are not claimed |
| observation | A feature patch of every feature is about 2.5 times slower than a raw patch (2.3 s against 0.9 s for 8 prompts), because it encodes both rows. | research maintainer | acceptable for version 1 |
| follow-up | No named backup for the research layer. | project owner | assign at the next audit |

## 5. Newcomer handoff

A dictionary is a SAELens folder pinned by two hashes. A capture with
`features` also writes the strongest features of each stored row; an
intervention with `features` applies its operation to those feature
activations and adds only the decoded change. Start with
`cargo test --no-default-features --lib -- v05::dictionary` and
`cargo test --no-default-features --bin ember -- cli_experiment_dictionary`.
For a real-model check, run a capture with `features` and compare the
artifact against a NumPy encode of the stored rows.

## 6. Close-out

- [x] Index row updated with this record.
- [ ] Primary and backup reviewed the evidence. (No backup is assigned.)
- [x] Governing docs/contracts updated, or an issue links the needed update.
- [x] Follow-ups are visible and have owners.
- [x] No model weights, private data, or generated tensor payloads were added.
  The SAE used for the smoke check states no license and stays outside the
  repository.

**Reviewer notes:** the automated review found the four required problems above (the last one in a second pass over the fixes), all fixed before merge. It confirmed that the new fields keep earlier specs and bundles unchanged, that the encode matches SAELens, and that forged feature artifacts fail checks without a panic.
