# Hooks audit (attention-heads site): 2026-10-21

- **Revision/branch:** `feat/attention-heads` on `main` at `8226c84`
  (the feature commit and the cleanup commits of the same pull request).
- **Audit date:** 2026-10-21
- **Scope and trigger:** the hook framework and every Llama execution path
  that fires hooks; the v0.5 capture and intervention layer. Trigger: a
  public schema change. `ember.hook.v1` gains the `attention-heads` site
  (ADR-0002), and captures and interventions gain an optional `heads`
  selector.
- **Primary:** runtime maintainer (voidwest).
- **Backup:** not assigned. See the findings.
- **Second reviewer:** an automated review agent read the diff without a
  build. No human second reviewer.
- **Last audit / next audit:** no earlier dated record for this row. Next:
  before the next release tag, or at the next hook-site change.

## 1. Map and boundaries

- User-facing commands/APIs: `ember experiment run|verify|reproduce|compare`
  with specs that name `site = "attention-heads"` or `heads = [..]`; the v0.2
  `--capture-activations` and patch paths with stage `before-o-proj`; the
  consoles' hook selectors (from `SemanticHookSite::ALL`).
- Source entry points: `ActivationStage::BeforeOProj`
  (`crates/ember-core/src/hook_types.rs`); `Experiment::before_o_proj`,
  `LayerHooks::before_o_proj` and `ExperimentRunner::refuse_stage`
  (`crates/ember-core/src/experiments/mod.rs`); the hook calls in
  `llama.rs` (reference attention and Q8 fast decode), `planned_decode.rs`
  (after the attention op) and `llama_batch.rs` (per sequence); the site
  record in `plan_build.rs`; `SemanticHookSite::AttentionHeads`
  (`src/v05/hook.rs`); head selection in `src/v05/runner.rs`.
- Adjacent subsystems and handoffs: prefix reuse and batched decode
  (`src/cli_experiment_shared.rs`); directions (`src/cli_experiment_steering.rs`);
  bundle verification (`src/v05/verify.rs`); the consoles.
- Persistent artifacts or schemas: `ember.hook.v1` (seven sites,
  `hook_schema` 1); `ember.experiment.v1` (optional `heads`); execution
  plans (a `before-o-proj` site record only where active); `ember.bundle.v1`
  (unchanged layout; per-head capture shapes `[rows, len(heads) * head_dim]`).
- Environment switches and feature gates: none new.
- External references or trusted implementations: none. The checks compare
  the paths with each other and with `attention-output`.
- Supported targets and deliberate non-goals: Llama and Qwen. Gemma-family
  models refuse the stage. No attention-pattern site. The logit lens,
  attribution patching and the probe bridge refuse the site.

## 2. Contracts and invariants

- Ownership/lifetime and resource limits: the hook views the existing
  attention buffer; no new allocation on the disabled-hook path. A `heads`
  selector is checked against the model's head count before inference.
- Shape, encoding, safety, or error behavior: the site is
  `[seq, n_heads * head_dim]` in prefill and `[1, n_heads * head_dim]` in
  decode. A source, direction or capture with a selector has width
  `len(heads) * head_dim`; a mismatch fails before any row changes.
- Numerical, determinism, trace, or hook boundaries: the hook fires after
  the attention op and before `o_proj` on every path. No fusion eliminates
  the tensor, so no de-fusion rule exists.
- Compatibility/provenance requirements: a spec without the site gives the
  same resolved spec, defaults, plan structure and hook calls as before; the
  plan records the site only where it is active; new fields are omitted
  when absent; `hook_schema` stays 1.
- Reference/oracle path versus optimized path: generic, planned and
  planned-fused per-head values are bit-identical on the synthetic model;
  the Q8 fast path agrees within 1% of the row's largest value there, and
  exactly on Llama-3.2-1B.

## 3. Checks and evidence

Host: Apple M1 Pro, macOS, Rust 1.98.1 (GUI) and 1.92.0 (headless MSRV),
`CARGO_BUILD_JOBS=2`.

| Check | Command / fixture | Result | Evidence or artifact |
| --- | --- | --- | --- |
| Static/build | `scripts/ci_local.sh quick` (part of `full`) | pass | fmt, clippy (all features, headless, x86_64 sibling, Python, MSRV), docs |
| Focused unit/integration | core: `per_head_site_*`, `zeroing_every_head_*`, `gemma_refuses_the_per_head_stage`; root: runner `heads_*`, `attention_heads_variants_resume_bit_identically`, `attention_head_sweeps_run_and_verify` | pass | hook order and shapes on every decode path; plans unchanged without the site; prefix reuse and batched decode equal full recompute |
| Full suites | `scripts/ci_local.sh full` | pass: 1,716 Rust tests and 143 Python tests (1 skipped); `cargo audit` not run (not installed locally; CI runs it) | default-feature, doctest, audio, GUI kit and K-quant tests |
| Smoke | Llama-3.2-1B-Instruct Q8_0, captures at layers 9 and 15, three modes | pass | bundles verify; per-head captures exact across reference, planned and planned-fused |
| Golden logits / external reference | not run | n/a | inference arithmetic unchanged; existing parity tests pass |
| Activation reference / probes / intervention | Llama-3.2-1B: zero every head at layer 9 vs zero `attention-output` | pass | tokens, logits and later captures exact in reference and planned mode |
| Performance (controlled A/B) | not run | n/a | the disabled path adds one inlined no-op call per layer; no measurement |
| Backup reproduction | not run | n/a | no backup assigned |

Model: `Llama-3.2-1B-Instruct-Q8_0.gguf`, SHA-256
`432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3`;
tokenizer SHA-256
`6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`;
4 threads.

## 4. Findings

| Severity | Finding / evidence | Owner or issue | Due / disposition |
| --- | --- | --- | --- |
| required | The v0.2 manifest check compared `before-o-proj` records with the hidden size. On a model where `n_heads * head_dim` differs (Qwen3-0.6B: 2048 vs 1024), a capture failed at finalize and a patch could not load. | runtime maintainer | fixed before merge: `record_width_fits` exempts the stage; unit test |
| required | The browser console hid the layer control for `before-o-proj`, so a run could never start. The native console labelled the site "Custom location". | product maintainer | fixed before merge: `/api/state` sends the per-layer stages; native labels added; GUI test |
| observation | A prompt-only capture at the site recorded the hook route `unknown`, because the decode plan has no record for it. | research maintainer | fixed before merge: the route is `unfused` |
| observation | The real-model six-site observer (`tests/six_site_observer.rs`) and its frozen capture matrix cover the six original sites only. | runtime maintainer | accepted: extending it needs new frozen evidence |
| follow-up | No head sweep: a layer sweep moves layers, positions and alphas, not heads. A study across heads needs one intervention per head. | research maintainer | next feature candidate |
| follow-up | No named backup for this row. | project owner | assign at the next audit |

## 5. Newcomer handoff

The hook framework lets an experiment read or change seven tensors per
block or head. `attention-heads` is the attention output for each head
before the output projection mixes the heads. It must fire on every path at
the same point, and a spec that does not name it must leave every plan and
bundle unchanged. Start with `cargo test -p ember-core --lib -- per_head`
and `cargo test --no-default-features --bin ember -- attention_heads`. The
design and its compatibility rules are in
`docs/adr/0002-attention-head-sites.md`.

## 6. Close-out

- [x] Index row updated with this record.
- [ ] Primary and backup reviewed the evidence. (No backup is assigned.)
- [x] Governing docs/contracts updated, or an issue links the needed update.
- [x] Follow-ups are visible and have owners.
- [x] No model weights, private data, or generated tensor payloads were added.

**Reviewer notes:** the automated review found the two required problems and the observation above. All three are fixed in the feature commit. It found no change for specs that do not use the site, and it confirmed the hook position on every path and in every forwarder.
