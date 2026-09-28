# Cutting Ember 1.0.0

Status: **staged, not released.** The local release gates pass on candidate
`099270c44ccb02dde40a3797595bb180633ff286621894c4d861306c6d67db00`; the tree now
reports `version = "1.0.0"` (rebuilt binary
`f443082e83ddeb20916598b512dee0604c81c5f340ec87852bf51a478dd18ed8`, a
version-string-only delta). The `v1.0.0` tag is deliberately held until the
external gates below close, per [the roadmap](road-to-1.0.md) gate 7.

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

Candidate validation (see [1.0 current validation](audits/1.0-current-validation.md)):
k_parity Q4 7/7 + Q6 7/7; golden ladder 6/6; six-site captures 36/36,
interventions 72/72, scale 72/72; source-shapes 6/6; Python 20/20; tooling 25/25;
default 1.98.1 and headless 1.92 all-targets green; Gate H matrix 24/24 within
limits.

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
3. Bump versions in `Cargo.toml`, `Cargo.lock`, `bindings/python/Cargo.toml`
   (and the Python package metadata) from `0.6.8` to `1.0.0`.
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
- macOS/Windows have no CI tier; local macOS ARM validation is documented in
  [ARM kernels](arm-kernels.md).
