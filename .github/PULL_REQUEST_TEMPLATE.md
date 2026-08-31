## Summary

<!-- What changed, and why? Link the issue or design note. -->

## Scope and non-goals

- Affected subsystem(s):
- Supported behavior changed:
- Deliberately unsupported / deferred:
- Risk and rollback plan:

## Evidence and validation

<!-- Mark each item pass, not run (with a reason), or not applicable. Do not
     call a smoke run a parity or quality result. -->

- [ ] `git diff --check`
- [ ] `cargo fmt --all -- --check`
- [ ] `cargo check --locked --all-targets`
- [ ] `cargo test --locked --all-targets`
- [ ] `cargo clippy --locked --all-targets --all-features -- -D warnings`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --release`
- [ ] `cargo check --locked --no-default-features --all-targets`
- [ ] Relevant Python/tests/docs checks (list exact commands below)
- [ ] Model, golden-logit, activation-reference, probe, intervention, or
      behavioral checks required by the claim (list exact level below)

**Commands and results:**

```text
<!-- Include pass/skip, revision, fixtures, and useful output/tolerances. -->
```

**Reproducibility metadata (when applicable):**

- Model/tokenizer identifiers and SHA-256:
- Architecture, quantization, and execution strategy:
- Rust/compiler revision and Rayon thread count:
- Reference implementation and output metrics (max/mean error, cosine,
  top-1/top-k, trace or artifact identity):

## K-quant or kernel change

<!-- Complete for changes touching src/k_quant_matmul.rs, src/k_matmul.rs,
     src/quant_k.rs, loader dispatch, or shared SIMD code. Otherwise write N/A. -->

- [ ] Scalar oracle and Q4_K/Q6_K edge cases are covered.
- [ ] x86 target-feature gates and `# Safety` contracts were reviewed; scalar
      or headless fallback remains valid on unsupported targets.
- [ ] Prefill (multi-row), decode (single-row), and row-remainder behavior are
      covered.
- [ ] Serial/parallel ownership, nested Rayon, dispatch, and allocation claims
      have focused tests.
- [ ] Pinned llama.cpp known-answer and real-model K-parity checks were run, or
      the reason they were not run is recorded above.
- [ ] Any changed hook, plan, provenance, fallback, or tolerance contract is
      documented.

## Performance change

<!-- Complete for performance-sensitive changes; retain raw samples. -->

- Workload/model:
- Baseline and candidate revisions:
- CPU, power policy, affinity, and Rayon threads:
- Warmups, repetitions, process order, and every raw sample:
- Result classification: gain / neutral / regression / inconclusive:

## Research interpretation and handoff

- Highest evidence level: smoke / golden logits / activation reference /
  probe / intervention / behavioral:
- What this result does **not** establish:
- Primary reviewer:
- Backup / second reviewer:
- Audit index or record updated: [`docs/audits/README.md`](../docs/audits/README.md)

## Hygiene checklist

- [ ] No model weights, private data, generated activations, or large
      regenerable payloads were added.
- [ ] No unrelated formatting/refactoring was included.
- [ ] Public API, CLI, feature, schema, and MSRV impact was reviewed against
      [`docs/api-stability.md`](../docs/api-stability.md).
- [ ] Nearest usage/contract documentation and changelog entry were updated
      when needed.
- [ ] A reviewer who was not the author can reproduce the strongest claim.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full newcomer workflow,
validation ladder, and K-quant review checklist.
