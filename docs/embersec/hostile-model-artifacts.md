# EmberSEC Phase I: Hostile Model Artifacts

**Status:** complete and published. The experiment snapshot is frozen at Ember
commit `3ceb7039`; the final artifact reconciliation is preserved by commit
`e1fe6269` on the historical `embersec/secure-gguf-loader` branch.

## Scope

Phase I treats GGUF and `tokenizer.json` bytes as attacker-controlled input.
It follows those bytes through parsing, tensor inventory and layout validation,
metadata-derived model construction, tokenizer construction, and the validated
views consumed by dequantization and SIMD kernels.

The threat model covers process integrity, memory safety, bounded resource use,
and faithful interpretation of accepted model artifacts. It does not cover
prompt injection, malicious weight semantics, remote serving, GPU runtimes, OS
sandboxing, or side channels.

## Published record

Design and boundary documentation:

- `threat-model.md`: attacker model, assets, trust boundaries, and residual risk.
- `loader-threat-surface.md`: audited GGUF dataflow and validation seams.
- `tokenizer-boundary.md`: hostile tokenizer input boundary.
- `unsafe-loader-boundary.md`: invariants between validated state and kernels.
- `bug-taxonomy.md`: failure classes used by the corpus and analysis.
- `fuzzing-plan.md`: targets, seed corpus, oracles, and campaign bounds.

The complete comparative experiment is under
`research/embersec/comparative/`:

- `README.md` and `SYNTHESIS.md`: methods, results, limitations, and conclusions.
- `corpus.json` plus `fixtures/`: 62 hashed hostile/control cases.
- `run_eval.py` and `diff_fuzz.py`: isolated evaluation and mutation campaigns.
- `results/`, `tables/`, and `figures/`: frozen outputs and derived presentation.
- `FROZEN_ARTIFACTS.md`: versions, SHA-256 inventory, and rebuild commands.
- `reference/`: pinned Candle and llama.cpp comparison harnesses.
- `suspected-external-bugs.md` and `disclosure/`: conservative classifications,
  disclosure status, and minimized reproducers.

## Main findings

- Hardened Ember converted all observed baseline panics, crashes, timeouts, and
  layout misinterpretation cases into structured rejection without changing
  accepted controls.
- In 10,000 raw mutations, both Ember snapshots had zero process failures; the
  campaign primarily exercised parser structure.
- In 2,000 construction-layer mutations, baseline Ember failed in about 9.7%
  of cases while hardened Ember had zero panics, crashes, or timeouts.
- External-runtime results are boundary-specific: llama.cpp was tested through
  a loader harness, while Candle was parser-only. Tokenizer-only inputs are
  marked `NOT_COMPARABLE` rather than forced into a misleading comparison.
- Validation overhead was below measurement noise in the recorded load tests.

Exact counts, environments, and limitations live in the frozen comparative
record and should be cited from there rather than recomputed from this summary.

## What this proves

For the bounded corpus and campaigns, malformed or oversized artifacts reached
structured validation failures before dangerous allocation or kernel use, and
the published artifact hashes allow the experiment record to be checked.

It does not prove the absence of every decoder bug, exploitability of external
crash surfaces, universal safety across arbitrary future inputs, or the overall
security posture of another runtime.

## Reproduction policy

Treat `research/embersec/comparative/` as a frozen record. Verify the hashes in
`FROZEN_ARTIFACTS.md` before use. Re-runs must record new runtime commits,
dependency versions, environments, outputs, and hashes instead of overwriting
the frozen results.
