# Cutting Ember 1.0.0

Status: **staged, not released.** The current candidate is
[`v1.0.0-rc.2`](https://github.com/voidwest/ember/releases/tag/v1.0.0-rc.2) on
`d41a618` (`main` after the provenance fix in #9), frozen release binary
`b2f15f2e715c7d45213bebddf2a44c9b8ac1dfa8b3acd942eeb7c1f30fda3d89`
(`cargo build --locked --release`, Rust 1.98.1, Apple M1 Pro). The full
candidate battery passed on it on 2026-10-02; see the
[record](audits/1.0-current-validation.md). The `v1.0.0` tag is deliberately
held until the external gates below close, per [the roadmap](road-to-1.0.md)
gate 7.

## Candidates

- `v1.0.0-rc.1` points at `0ea407d`, which is not an ancestor of `main`
  (history was rewritten after the tag was pushed; `665d60c` on `main` has
  the identical tree). Its evidence (binary `099270c4…`, 2026-09-28) is kept
  below for history.
- `v1.0.0-rc.2` is on `main` at `d41a618`. It carries the work after rc.1:
  the ARM prefill and decode kernels, batched decode, f16/bf16 resident
  weights, bundle I/O changes, steering/attribution/probe/lens features, and
  build-time commits in bundle provenance. Outputs stayed bit-identical where
  the contract requires it: the golden-ladder metrics equal rc.1's to every
  printed digit.

Commits after `d41a618` must be covered by a new candidate before `v1.0.0` is
tagged on them; the cut can also tag `v1.0.0` on `d41a618` itself if nothing
release-relevant lands in between.

## Gate status

| Roadmap gate | Status | Evidence |
|---|---|---|
| 1. Public workflow (GUI + headless) | Implementation done; desktop accessibility/manual acceptance **open** | 86 `gui-tests` bin tests, 28 offscreen render scenes; headless builds independent |
| 2. Freeze research contracts | Done | [contract audit](audits/1.0-contract-audit.md); six-site captures/interventions/scale/observer on the candidate |
| 3. Compatibility & migration rules | Done | [research contract §19](v05-research-contract.md); [API stability](api-stability.md); [public-surface inventory](audits/1.0-public-surfaces.md) |
| 4. Installation & examples | Examples pinned; clean machine **open** | [morphology examples](../examples/experiments/README.md); tokenizer bundled in-repo with a public pinned fallback, so only a clean machine remains |
| 5. Reconcile validation | Done | [validation](validation.md); [Q6 disposition](audits/1.0-q6-numerics.md); [Gate C](audits/1.0-current-validation.md); [Gate H matrix + capture overhead](audits/1.0-performance-m1.md) |
| 6. External validation | **Open** | [external acceptance checklist](1.0-external-acceptance.md); needs a real second person |
| 7. Cut the release | **Blocked** | this document |

Candidate validation for `v1.0.0-rc.2` (see
[1.0 current validation](audits/1.0-current-validation.md)): local CI mirror
(`scripts/ci_local.sh full`) and Rust 1.92 headless all-target tests green;
k_parity Q4 7/7 + Q6 7/7; golden ladder 6/6 with metrics unchanged from rc.1;
golden path exact-semantic, also when reproduced from another directory;
six-site captures, interventions, scale and observer passed; Python binding
rebuilt, 133 tests passed; Gate H matrix 24/24 within limits.

## External blockers (need the project owner)

1. **Clean-machine install** — needs a machine without the developer's local
   models/credentials to follow the install docs end to end. The tokenizer is no
   longer a blocker: `tokenizer.json` is tracked in the repository (a fresh clone
   has it), and `scripts/download_models.sh tokenizer` re-fetches the
   byte-identical public copy pinned to `unsloth/Llama-3.2-1B-Instruct` revision
   `5a8abab4a5d6f164389b1079fb721cfab8d7126c` (ungated; verified same size and
   SHA-256), so no gated `meta-llama` access is required. Only the clean machine
   itself and the public model download remain.
2. **Independent external user** — a person other than the author installs from
   the docs, completes a supplied experiment, verifies the result, and consents
   to publishing the record.
3. **Desktop GUI acceptance** — the offscreen CoreText/Metal path is verified,
   but a human visual/accessibility pass of the native console is still required.

## Cut steps (run when the blockers above close)

1. Confirm the external gates are recorded (gate 4 transcript, gate 6 tester
   report + consent, gate 1 manual/GUI acceptance).
2. Re-freeze the candidate: rebuild `--locked --release`, then rerun the battery
   in [1.0 current validation](audits/1.0-current-validation.md) so every
   artifact identity matches the tagged commit.
3. Version bump to `1.0.0` is already applied in `Cargo.toml`, `Cargo.lock`, and
   `bindings/python/Cargo.toml`; the tree reports `1.0.0`.
4. Run the release checklist in [api-stability](api-stability.md): `cargo fmt
   --all -- --check`, `cargo test --locked --all-targets`, headless
   `--no-default-features --all-targets`, default + headless clippy `-D warnings`,
   `cargo doc --locked --no-deps`.
5. Move the staged `[1.0.0]` [CHANGELOG](../CHANGELOG.md) section out of
   "release candidate" and date it.
6. Create the `v1.0.0` tag and publish through the project's release process.

## Known limitations to publish with 1.0.0

- Cross-tier eager-f32 vs compressed K-quant agreement is a cosine envelope,
  not exact greedy-token equality; the pinned llama.cpp golden ladder is the
  authoritative model-level numerical gate ([Q6 audit](audits/1.0-q6-numerics.md)).
- Capture overhead is measured at the prompt-final and decode-step-1 positions
  only (within noise); per-step capture rates are not characterized.
- `qwen3` and `gemma4` remain experimental; see [support](support.md).
- The Rust library API has no stable subset; only the serialized anchors and
  documented CLI/Python surfaces are the 1.x promise.
- Windows has no CI tier. macOS has a headless build/test/lint tier and a GUI
  compile check on both architectures, but no model-level CI gate; local macOS
  ARM validation is documented in [ARM kernels](arm-kernels.md).
