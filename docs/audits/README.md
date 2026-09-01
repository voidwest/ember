# Subsystem audit index and practice

This directory is the collaboration map for Ember’s runtime, research, and
release surfaces. It exists to reduce bus factor: a contributor who did not
write a subsystem should be able to find its entry points, invariants, checks,
known limits, and a person who can review or take over the work.

An audit is an engineering handoff, not a marketing review and not a claim
that every model or architecture is validated. Record what was actually run,
what was unavailable, and what remains unsupported. Historical design notes and
benchmark reports are useful evidence, but they do not replace a dated,
repeatable audit record.

## Index

The role labels below are deliberately roles rather than permanent names. Each
audit record must assign a **primary** and a **backup** contributor (or team
handle), and should rotate those assignments. If a row has no current named
backup, fixing that is an audit finding.

| Subsystem | Primary role / backup role | Start here | Minimum repeatable check | Audit record |
| --- | --- | --- | --- | --- |
| K-quant matmul | runtime maintainer / SIMD reviewer | [`src/k_quant_matmul.rs`](../../src/k_quant_matmul.rs), [`src/k_matmul.rs`](../../src/k_matmul.rs) | [`k_quant_matmul` tests](../../src/k_quant_matmul.rs), pinned known-answer gate | Add the next dated record; see [`CONTRIBUTING.md`](../../CONTRIBUTING.md) |
| Loader, quantization, and model families | runtime maintainer / architecture reviewer | [`src/loader.rs`](../../src/loader.rs), [`src/quant_k.rs`](../../src/quant_k.rs), [`docs/models.md`](../models.md) | loader/unit tests plus the relevant golden-logit or parity check | Add the next dated record |
| Execution plan, hooks, and experiments | runtime maintainer / research reviewer | [`src/plan.rs`](../../src/plan.rs), [`src/planned_decode.rs`](../../src/planned_decode.rs), [`src/experiments/`](../../src/experiments/) | plan/hook tests, reference-vs-planned parity, capture/intervention contract | Add the next dated record |
| Bundles, reproducibility, and agent traces | research maintainer / release reviewer | [`src/v05/`](../../src/v05/), [`src/agent/`](../../src/agent/), [`docs/reproducibility.md`](../reproducibility.md) | verify/reproduce a small fixture or contract test; inspect provenance | Add the next dated record |
| GUI, multimodal, audio, and TTS | product/runtime maintainer / headless reviewer | [`src/gui.rs`](../../src/gui.rs), [`src/multimodal/`](../../src/multimodal/), [`src/tts/`](../../src/tts/) | default-feature and `--no-default-features` builds; applicable focused tests | Add the next dated record |
| Python research and probe pipeline | research maintainer / statistics reviewer | [`probes/`](../../probes/), [`tests/`](../../tests/), [`docs/research.md`](../research.md) | compile/smoke plus the relevant pytest and artifact checks | Add the next dated record |
| CI, release, and documentation | release maintainer / docs reviewer | [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml), [`scripts/check_docs.py`](../../scripts/check_docs.py) | CI-equivalent gates and docs check | Add the next dated record |

**Audit log:** this index establishes the recurring practice. Dated records
should be named `YYYY-MM-DD-<subsystem>.md`, linked in the last column, and
reviewed alongside the index. Do not mark a row complete by linking only to a
historical report; the record must state its own revision, participants, and
commands.

## Cadence and triggers

- Audit every subsystem at least once per quarter, rotating the primary and
  backup. A release or a large change may split the work into smaller records,
  but must not make the audit less frequent.
- Start an audit after an incident, security finding, numerical/parity
  regression, benchmark surprise, unsafe-code change, public schema/CLI change,
  dependency or feature change, or a maintainer becoming unavailable.
- Audit before a release when a subsystem changed since its last record. The
  release reviewer checks that each changed row has a dated record or an
  explicitly accepted follow-up.
- Keep the scope small enough to finish in one sitting. A focused audit with
  an honest “not run” is more useful than an unbounded checklist with no
  evidence.

## Audit procedure

Use [`template.md`](template.md) for each record.

1. **Scope and handoff.** Write the revision/branch, subsystem boundary,
   reason for the audit, primary, backup, and a second reviewer. List adjacent
   subsystems so a finding is not silently assigned to the wrong owner.
2. **Map the path.** Identify user-facing commands or APIs, source entry
   points, data/control flow, persistent formats, environment switches, and
   external references. Note where the reference/oracle path differs from the
   optimized path.
3. **Read contracts first.** Link the governing design, validation, security,
   API, and research contracts. List invariants, error behavior, ownership and
   lifetime rules, numerical tolerances, determinism requirements, hook/schema
   boundaries, supported targets, and deliberate non-goals.
4. **Reproduce the minimum check.** Run the row’s minimum command and the
   narrowest relevant unit/integration check. For inference, distinguish smoke,
   golden logits, activation reference, probes, interventions, and behavioral
   scoring. For performance, retain raw samples and controls rather than a
   single timing.
5. **Inspect failure and maintenance paths.** Look for unchecked shape/size
   arithmetic, unsafe feature assumptions, fallback decisions, partial writes,
   stale provenance, ignored errors, unbounded resource use, missing tests,
   undocumented switches, and docs that point at deleted code. Confirm that a
   backup can run the check without private files or the original author’s
   machine.
6. **Record findings.** Classify each finding as `blocker`, `required`,
   `follow-up`, or `observation`; include evidence, an issue/owner, and a due
   date where applicable. Do not silently fix an unrelated problem during an
   audit.
7. **Close the handoff.** The backup/second reviewer signs off, the index links
   the record, contracts are updated if behavior changed, and follow-ups are
   visible in the issue tracker or the next audit date.

## Definition of done

An audit record is complete when it contains:

- a commit/revision, date, scope, primary, backup, and reviewer;
- the source entry points, supported and unsupported surfaces, dependencies,
  external references, and adjacent subsystem owners;
- a concise invariant/safety/numerical/serialization checklist;
- exact commands, environment and fixture identities, observed result, and
  every skipped command with a reason;
- findings with severity and ownership, plus links to fixes or accepted risks;
- a next-audit date or trigger and a newcomer handoff paragraph.

A green CI run is necessary evidence for its checks but does not prove model
parity, activation correctness, causal use, or a benchmark claim. Keep those
claims at the validation level the record actually established. Do not add
model weights, private research artifacts, or generated tensor payloads to an
audit record.

## Existing orientation reports

These documents are useful starting points, but they are not substitutes for
the recurring records above:

- [`docs/v03-execution-contracts.md`](../v03-execution-contracts.md): frozen
  Q4_K/Q6_K execution and parity contract.
- [`docs/cleanup-audit-validation.md`](../cleanup-audit-validation.md): a
  historical cleanup validation report and its controlled benchmark protocol.
- [`docs/validation.md`](../validation.md): current evidence status and the
  research validation ladder.
- [`docs/backend_validation.md`](../backend_validation.md) and
  [`docs/activation_reference_checks.md`](../activation_reference_checks.md) :
  numerical/reference-check orientation for runtime and research audits.

If an orientation link is stale, fix the link or remove it in the same PR; do
not leave a newcomer guessing which document is authoritative.
