# Sweeps audit (head sweeps): 2026-10-21

- **Revision/branch:** `feat/head-sweep` on `main` at `09a1e45`.
- **Audit date:** 2026-10-21
- **Scope and trigger:** layer sweeps (`src/v05/sweep.rs`,
  `src/cli_experiment_sweep.rs`) and capture sources at `attention-heads`
  (`src/v05/runner.rs`). Trigger: a public schema change. The `[sweep]`
  table gains `heads`, and `ember.sweep.v1` gains optional `heads`,
  `head_count` and per-point `head` fields.
- **Primary:** research maintainer (voidwest).
- **Backup:** not assigned (open since the 2026-10-11 record).
- **Second reviewer:** an automated review agent read the diff without a
  build. No human second reviewer.
- **Last audit / next audit:** last: `2026-10-11-sweeps.md`. Next: before
  the next release tag, or at the next `ember.sweep.v1` change.

## 1. Map and boundaries

- User-facing commands/APIs: `ember experiment validate|run|verify|compare|inspect`
  with `heads` in `[sweep]`.
- Source entry points: `SweepDefinition::parse`, `resolve_heads` and
  `points(n_layers, n_heads)`; `verify_and_load_sweep`; `source_for_width`
  in the runner.
- Adjacent subsystems and handoffs: the shared pass and prefix reuse;
  bundle-source resolution; the `attention-heads` site (ADR-0002).
- Persistent artifacts or schemas: `sweep.json` (`heads`, `head_count`,
  per-point `head`), `sweep.csv` and `sweep-effect.csv` (`head` column only
  for head sweeps).
- Environment switches and feature gates: none new.
- External references or trusted implementations: none.
- Supported targets and deliberate non-goals: heads move only
  `attention-heads` interventions. Inline vectors keep the selected width:
  only capture sources record their column count, so only they may be
  full-width.

## 2. Contracts and invariants

- Ownership/lifetime and resource limits: `head_count` from `sweep.json` is
  checked against the loader limit (8,192) before it sizes the point list.
- Shape, encoding, safety, or error behavior: a head sweep needs
  `attention-heads` interventions without their own `heads`; an empty,
  malformed or out-of-range selector fails before inference.
- Numerical, determinism, trace, or hook boundaries: no change to hooks;
  each point bundle is the bundle that its derived spec gives alone.
- Compatibility/provenance requirements: sweeps without `heads` write the
  same `sweep.json`, CSV files and `sweep_hash`; the frozen fixtures
  (`tests/sweep_compatibility.rs`) still verify.
- Reference/oracle path versus optimized path: not applicable.

## 3. Checks and evidence

Host: Apple M1 Pro, macOS, Rust 1.98.1 (GUI) and 1.92.0 (headless MSRV),
`CARGO_BUILD_JOBS=2`.

| Check | Command / fixture | Result | Evidence or artifact |
| --- | --- | --- | --- |
| Static/build | `scripts/ci_local.sh quick` (part of `full`) | pass | fmt, clippy (all configurations), MSRV, docs |
| Focused unit/integration | `v05::sweep` and `v05::runner` tests; `head_sweeps_run_every_head_and_verify`, `head_patching_reads_each_head_from_the_same_head_of_the_source` | pass | grid order, tamper checks, identity head patching gives zero effect |
| Full suites | `scripts/ci_local.sh full` | pass: 1,725 Rust tests and 143 Python tests (1 skipped); `cargo audit` not run (not installed locally; CI runs it) | all suites |
| Compatibility | `tests/sweep_compatibility.rs` | pass | frozen sweeps verify with their recorded hashes |
| Smoke | `capital-head-patching-sweep.toml` on Llama-3.2-1B-Instruct Q8_0 | pass | 65 bundles verify with `--tokenizer`; 21 s; 275 MB |
| Golden logits / external reference | not run | n/a | no inference change |
| Performance (controlled A/B) | not run | n/a | no hot-path change |
| Backup reproduction | not run | n/a | no backup assigned |

## 4. Findings

| Severity | Finding / evidence | Owner or issue | Due / disposition |
| --- | --- | --- | --- |
| required | The research contract still said that source columns must match target columns, but the runner now takes the selected heads from a full-width capture source. | research maintainer | fixed before merge: dated amendment in contract sections 1 and 3, the schema doc and ADR-0002 |
| required | `verify` accepted any `head_count` up to the limit, so a sweep cut down to fewer heads than `"all"` gave on the model verified. | research maintainer | fixed before merge: `head_count` is checked against `<arch>.attention.head_count` in the baseline bundle's `model.json` |
| required | A swept intervention whose source capture held one head would patch every head from that head, with no error. | research maintainer | fixed before merge: refused at parse time (current-run captures) and before the run (bundle captures) |
| observation | Forged axes could make `verify` derive millions of point specs before any bundle check. | research maintainer | fixed before merge: the expected point count is compared with the recorded points first |
| observation | A layers x heads grid writes one bundle per point; with a full logits capture, the 64-point example is 275 MB. | research maintainer | documented in `docs/experiments.md`; a narrower metric capture is a future option |
| follow-up | No named backup for the sweeps surface. | project owner | assign at the next audit |

## 5. Newcomer handoff

A head sweep is a layer sweep whose points also set one attention head of
the swept `attention-heads` interventions. With `move_bundle_sources`, each
point patches head H of layer L from the same head and layer of a clean
bundle that captured every head. Start with
`cargo test --no-default-features --bin ember -- head_`. The real-model
example is `examples/experiments/capital-head-patching-sweep.toml` (run
`capital-clean.toml` first).

## 6. Close-out

- [x] Index row updated with this record.
- [ ] Primary and backup reviewed the evidence. (No backup is assigned.)
- [x] Governing docs/contracts updated, or an issue links the needed update.
- [x] Follow-ups are visible and have owners.
- [x] No model weights, private data, or generated tensor payloads were added.

**Reviewer notes:** the automated review found the three required problems and the observation above, all fixed before merge. It confirmed that sweeps without `heads` keep their files and hash, and that forged fields fail checks without a panic.
