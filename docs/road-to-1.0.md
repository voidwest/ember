# Steps to Ember 1.0

This is a release plan, not a declaration that 1.0 has shipped. The current
package remains 0.6.8. It turns the nine criteria in
[What Ember 1.0 has to earn](ember/road-to-ember-1.0/07-what-ember-1-0-has-to-earn.html)
into ordered work and reviewable evidence. A faster kernel or a new GUI alone
cannot satisfy these gates.

## 1. Finish the public workflow

- [x] Rewrite the native console with GPUI Kit controls, input handling, and
  application root. Preserve the shared `GuiSession` experiment path.
- [ ] Exercise model preparation, Arabic and Latin editing, baseline and
  intervention runs, capture inspection, bundle verification, comparison, and
  exact restoration through the new GUI.
- [ ] Check keyboard navigation, disabled controls while work is running,
  system/light/dark appearance, error recovery, and offline embedded fonts.
- [x] Keep headless CLI builds independent of GUI dependencies.

Evidence: GUI screenshots and workflow results, unit/integration checks, and a
headless build. The native UI must expose the same research semantics as the
CLI; it must not introduce a second inference implementation.

Local migration evidence: Kit interaction tests cover navigation, inputs,
selectors, loading guards, and error recovery; the real-model worker check
verifies baseline/intervention bundles and exact restored token IDs. macOS
CoreText/Metal renders cover both explicit themes at normal/minimum sizes.
Full desktop accessibility and end-to-end manual acceptance remain open.

## 2. Freeze the research contracts (criteria 1–3)

Implementation progress and evidence: [1.0 contract audit](audits/1.0-contract-audit.md).
The audit records fixes and historical fixtures; unchecked gates remain open.

- [x] Review every required field of `ember.experiment.v1` against the parser,
  resolver, examples, and negative tests. Done via the parser-to-resolver
  checklist and the deletion matrix in the [contract audit](audits/1.0-contract-audit.md);
  the intervention span-input validator is verified by 19 passing focused spec
  tests on both toolchains.
- [x] Review `ember.bundle.v1` against the writer and offline verifier; preserve
  fixtures written by earlier releases and test them with the release candidate.
  The `tests/fixtures/bundle-v050` historical fixture and the bundle
  compatibility/negative suites run in the candidate test matrix.
- [x] Freeze the six `ember.hook.v1` sites, layer/token conventions, capture
  timing, intervention ordering, and de-fusion behavior. Verified on the candidate:
  six-site captures 36/36 exact, zero/restore interventions 72, scale 72, the
  independent observer, source-shape replace/interpolate/add-delta, and the
  planned-fused de-fusion tests. See [final-candidate validation](audits/1.0-current-validation.md).
- [x] Replace the 0.5.x-only compatibility wording in the research contract with
  an explicitly reviewed 1.x commitment when the above evidence exists. Done in
  [research contract section 19](v05-research-contract.md).

Evidence: [research contract](v05-research-contract.md),
[experiment schema](experiment-schema-v1.md), [bundle schema](bundle-schema-v1.md),
versioned fixtures, and gates A–G. A schema name ending in `v1` does not itself
prove a package-level compatibility promise.

## 3. Publish compatibility and migration rules (criteria 4 and 9)

- [x] Inventory the supported CLI, Python, Rust, and artifact surfaces using
  [API stability policy](api-stability.md); separate diagnostics from promises.
  Done in the [source-derived public surface inventory](audits/1.0-public-surfaces.md)
  and research-contract section 19.
- [x] Specify unknown-major rejection, required-field behavior, semantic changes,
  deprecations, and explicit artifact migration without silent reinterpretation.
  Done in research-contract sections 17/19 and [API stability policy](api-stability.md).
- [x] Confirm the next planned features can extend the interface without another
  major redesign. Record any remaining incompatible changes before tagging 1.0.
  The extensibility review is recorded in the
  [public surface inventory](audits/1.0-public-surfaces.md#extensibility-review);
  the only known pre-1.0 output-changing corrections are already listed in the
  [candidate migration notes](migration-to-1.0.md).

Evidence: reviewed policy plus compatibility and malformed-input tests.
The [source-derived public surface inventory](audits/1.0-public-surfaces.md)
records current boundaries and remaining review gaps.

## 4. Reproduce installation and examples (criteria 5 and 6)

- [ ] On a clean supported machine, follow only the installation documentation:
  build, obtain an authorized model, validate a spec, run, inspect, verify,
  intervene, compare, and restore. **Blocked:** requires a clean machine. The
  tokenizer is not a blocker: it is tracked in the repository and
  `scripts/download_models.sh tokenizer` re-fetches the byte-identical public,
  ungated copy pinned to `unsloth/Llama-3.2-1B-Instruct` revision `5a8abab4…`.
- [x] Pin model/tokenizer hashes, example specs, execution mode, and reproducibility
  envelopes. Record exact commands and artifact identities. The
  [morphology workflow examples](../examples/experiments/README.md) embed the
  pinned model/tokenizer hashes and revision; the ladder manifest, golden ladder,
  and downloader record exact commands and identities; the verified run is
  `morphology-v1-m1-verified`.
- [ ] Check both a GUI install and the documented headless route. Record supported
  OS/toolchain versions and optional dependency requirements. Headless and GUI
  builds are both validated locally; a clean-machine install remains blocked as above.

Evidence: clean-machine transcript and independently verifiable example bundles.
Authentication, model licensing, and downloads must not depend on a developer's
private USB files or local credentials being copied into the repository.

## 5. Reconcile validation claims (criterion 7)

- [x] Update [validation](validation.md) so loadable, supported, parity-tested,
  and golden-validated models are distinguishable.
- [x] Run the required contract, artifact-corruption, model, Python, and CLI gates
  on the release candidate. Record skipped and failing checks explicitly. All ran
  green on candidate `099270c4…`; the x86 K-parity tier is skipped on this ARM
  host (recorded, not failed). See [final-candidate validation](audits/1.0-current-validation.md).
- [x] Resolve or explicitly scope the known K-quant scalar-versus-eager-f32
  greedy-continuation discrepancy. ARM/scalar equality does not close that gate.
  Disposition: explicitly scoped by the evidence-based contract decision in
  [1.0-q6-numerics.md](audits/1.0-q6-numerics.md) and the 2026-08-11 amendment in
  [v03-execution-contracts.md](v03-execution-contracts.md). Exact greedy-token
  equality holds within a compressed tier; the cross-tier eager-f32 comparison
  keeps the cosine envelope on shared-prefix steps and records near-tie flips;
  the llama.cpp golden ladder (Gate C) is the authoritative model-level gate.
  `scripts/validate_k_parity.sh` now passes Q4 7/7 and Q6 7/7.
- [x] Re-run the documented performance and memory protocol without weakening
  thresholds after seeing results. Preserve the CPU optimizations and headless
  behavior while changing the GUI. The 24-cell K-quant matrix is complete:
  all cells within limits, candidate +28×–96× throughput, RSS −0.8%…−1.4%. See
  [performance investigation](audits/1.0-performance-m1.md). Selected-row and
  full-tensor capture overhead is now measured: within noise, ~0.2% RSS, bundle
  payload +1.1–1.3 MB.

Evidence: candidate commit, toolchain/hardware, commands, raw results, and a
validation matrix. Earlier local test counts are background, not candidate proof.

Current broad regression snapshot: [development-tree validation](audits/1.0-current-validation.md).
It is not a frozen-candidate release verdict.

Initial same-host performance evidence: [M1 baseline investigation](audits/1.0-performance-m1.md).
This covers Q8 reference at four threads against an explicitly adapted v0.4
baseline; the complete performance/memory gate remains open.

## 6. Obtain external validation (criterion 8)

- [ ] A person other than the author installs Ember from the documentation,
  completes a supplied experiment, verifies the result, and reports the outcome.
- [ ] Address failures and record the tester's environment and reproducible
  evidence with their consent.

This gate requires a real external user. Automated local checks cannot substitute
for it, and it remains open until such evidence is recorded.
Use the [external acceptance checklist](1.0-external-acceptance.md) to prepare
an independently fetchable candidate and collect the tester's environment,
workflow evidence, assistance needed, and publication consent.

## 7. Cut the release

- [x] Review evidence for every criterion above and all required contract gates.
  Done on candidate `099270c4…`; the external gates (clean install, external
  tester, desktop GUI acceptance) are recorded as the only open items in the
  [release checklist](release-1.0.md).
- [x] Publish the final support/compatibility matrix, known limitations, migration
  notes, reproducible examples, and changelog. Staged: [support](support.md),
  [known limitations](release-1.0.md), [migration notes](migration-to-1.0.md),
  pinned [examples](../examples/experiments/README.md), and the staged
  `[1.0.0]` [changelog](../CHANGELOG.md) section.
- [ ] Only then bump the package/bindings versions, regenerate lockfiles as needed,
  run release checks, and create the 1.0.0 release through the project's release
  process. **Staged, not applied:** the exact steps are in the
  [release checklist](release-1.0.md); the version bump and tag wait on the three
  external gates. The `-D warnings` clippy, `cargo doc`, fmt, and both toolchain
  suites already pass on the candidate.

The GUI rewrite can land before these release gates close. It does not authorize
claiming external validation, universal numerical equivalence, or a 1.0 release.
