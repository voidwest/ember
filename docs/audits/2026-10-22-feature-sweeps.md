# Sweeps audit (feature sweeps): 2026-10-22

- **Revision/branch:** `feat/feature-sweep` on `main` at `a5f42ae`.
- **Audit date:** 2026-10-22
- **Scope and trigger:** layer sweeps (`src/v05/sweep.rs`,
  `src/cli_experiment_sweep.rs`) and feature interventions
  (`src/v05/dictionary.rs`, `src/cli_experiment_dictionary.rs`). Trigger: a
  public schema change. The `[sweep]` table gains `features`, and
  `ember.sweep.v1` gains optional `features` and per-point `feature` fields.
- **Primary:** research maintainer (voidwest).
- **Backup:** not assigned (open since the 2026-10-11 record).
- **Second reviewer:** an automated review agent read the diff without a
  build. No human second reviewer.
- **Last audit / next audit:** last: `2026-10-21-head-sweeps.md` (sweeps)
  and `2026-10-22-feature-dictionaries.md` (dictionaries). Next: before the
  next release tag, or at the next `ember.sweep.v1` change.

## 1. Map and boundaries

- User-facing commands/APIs: `ember experiment validate|run|verify|compare|inspect`
  with `features = { dictionary, ids }` in `[sweep]`.
- Source entry points: `SweepDefinition::parse` and `parse_features`;
  `point_count` and `points`; the feature check in `run_sweep`;
  `verify_and_load_sweep`.
- Adjacent subsystems and handoffs: feature interventions (ADR-0003); the
  dictionary cache in `load_dictionaries`; the shared pass and prefix reuse.
- Persistent artifacts or schemas: `sweep.json` (`features`, per-point
  `feature`), `sweep.csv` and `sweep-effect.csv` (`feature` column only for
  feature sweeps). Point bundles are ordinary `ember.bundle.v1` bundles with
  `features = { dictionary, ids = [F] }` on the swept interventions.
- Environment switches and feature gates: none new.
- External references or trusted implementations: none.
- Supported targets and deliberate non-goals: no `ids = "all"`. A feature
  sweep does not also move layers, heads or positions, because a dictionary
  is trained at one site and layer. It can cross with `alphas`.

## 2. Contracts and invariants

- Ownership/lifetime and resource limits: a sweep sets at most 65,536
  features (`MAX_SWEPT_FEATURES`, the width of the largest dictionary that
  the weights cap admits). A range is counted before it is collected. The
  run checks every id against the dictionary's `d_sae` before the first
  point. The points reuse the cached weights.
- Shape, encoding, safety, or error behavior: the swept interventions must
  be at the dictionary's site and layer, must not declare `features`, and
  must use `zero`, `scale`, `replace`, `add-delta` or `interpolate`. An
  unknown dictionary, an empty list or a malformed range fails at parse
  time, with the `sweep.features` path.
- Numerical, determinism, trace, or hook boundaries: no change to hooks or
  to the feature-space arithmetic. Each point bundle is the bundle that its
  derived spec gives alone.
- Compatibility/provenance requirements: sweeps without `features` write
  the same `sweep.json`, CSV files and `sweep_hash`; the frozen fixtures
  (`tests/sweep_compatibility.rs`) still verify. `verify` compares the
  recorded `features` with the definition and each point's `feature` with
  its coordinates.
- Reference/oracle path versus optimized path: not applicable.

## 3. Checks and evidence

Host: Apple M1 Pro, macOS, Rust 1.98.1 (GUI) and 1.92.0 (headless MSRV),
`CARGO_BUILD_JOBS=2`.

| Check | Command / fixture | Result | Evidence or artifact |
| --- | --- | --- | --- |
| Static/build | `scripts/ci_local.sh quick` (part of `full`) | pass | fmt, clippy (all configurations), MSRV, docs |
| Focused unit/integration | `feature_sweeps_set_one_feature_per_point`, `feature_sweeps_refuse_what_a_dictionary_cannot_do`, `feature_sweeps_run_one_feature_per_point_and_verify` | pass | grid order and point ids; every refusal; a forged feature or id list fails `verify`; an id past `d_sae` fails before any bundle |
| Full suites | `scripts/ci_local.sh full` | pass: 1,760 Rust tests and 146 Python tests (1 skipped); `cargo audit` not run (not installed locally; CI runs it) | all suites |
| Compatibility | `tests/sweep_compatibility.rs` | pass | frozen sweeps verify with their recorded hashes |
| Smoke | `capital-feature-sweep.toml` on Llama-3.2-1B-Instruct Q8_0 | pass | 9 bundles verify; 2.5 s of compute; 38 MB; a second run gives the same `sweep-effect.csv` |
| Golden logits / external reference | not run | n/a | no inference change |
| Performance (controlled A/B) | not run | n/a | no hot-path change |
| Backup reproduction | not run | n/a | no backup assigned |

Model: `Llama-3.2-1B-Instruct-Q8_0.gguf`, SHA-256
`432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3`.
Dictionary: the qresearch layer-9 SAE converted to SAELens, weights SHA-256
`8ade5dc5488d13676a26cad16b4d9bf88c947b9f6df44441996fe87ffff6977f`. The
dictionary is local only and is not in the repository.

## 4. Findings

| Severity | Finding / evidence | Owner or issue | Due / disposition |
| --- | --- | --- | --- |
| required | A range such as `{ start = 0, end = 9223372036854775807, step = 1 }` was collected before any limit, so `validate` could exhaust memory. | research maintainer | fixed before merge: the range is counted first, and a sweep sets at most 65,536 features |
| required | A swept `steer`, `ablate-projection` or `restore-original` intervention passed the sweep parse and failed only when the points were derived, after the model loaded. | research maintainer | fixed before merge: refused at parse time with the `sweep.features` path |
| required | An id past the dictionary's width failed only when its point ran, after the earlier points wrote bundles. | research maintainer | fixed before merge: `run_sweep` checks every id against `d_sae` before the first point |
| observation | `validate` did not show the feature axis. | research maintainer | fixed before merge: the JSON output has `features`, and the text output names the count and the dictionary |
| observation | The first docs said that each feature moves only its own country's prompt. Six off-diagonal entries are not 0. | research maintainer | fixed before merge: the example header explains each of the six from a full feature capture of the rows |
| follow-up | No named backup for the sweeps surface. | project owner | assign at the next audit |

## 5. Newcomer handoff

A feature sweep keeps the swept interventions at the dictionary's site and
layer and gives each point one feature: `features = { dictionary, ids =
[F] }`. Start with
`cargo test --no-default-features -- feature_sweeps`. The real-model
example is `examples/experiments/capital-feature-sweep.toml` (run
`capital-clean.toml` first, and convert the SAE as its header shows).

## 6. Close-out

- [x] Index row updated with this record.
- [ ] Primary and backup reviewed the evidence. (No backup is assigned.)
- [x] Governing docs/contracts updated, or an issue links the needed update.
- [x] Follow-ups are visible and have owners.
- [x] No model weights, private data, or generated tensor payloads were added.

**Reviewer notes:** the automated review found the three required problems
and the two observations above, all fixed before merge. It confirmed that
sweeps without `features` keep their files and hash.
