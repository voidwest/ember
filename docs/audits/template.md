# `<subsystem>` audit: YYYY-MM-DD

> Copy this file to `YYYY-MM-DD-<subsystem>.md`, fill every field, and link it
> from [`README.md`](README.md). Keep claims at the evidence level actually
> established; `not run` is an acceptable result when it has a reason.

- **Revision/branch:**
- **Audit date:**
- **Scope and trigger:**
- **Primary:**
- **Backup:**
- **Second reviewer:**
- **Last audit / next audit:**

## 1. Map and boundaries

- User-facing commands/APIs:
- Source entry points:
- Adjacent subsystems and handoffs:
- Persistent artifacts or schemas:
- Environment switches and feature gates:
- External references or trusted implementations:
- Supported targets and deliberate non-goals:

## 2. Contracts and invariants

- Ownership/lifetime and resource limits:
- Shape, encoding, safety, or error behavior:
- Numerical, determinism, trace, or hook boundaries:
- Compatibility/provenance requirements:
- Reference/oracle path versus optimized path:

## 3. Checks and evidence

Record exact commands, environment, fixture/model identifiers, and observed
results. Include the strongest applicable level, not just a smoke result.

| Check | Command / fixture | Result | Evidence or artifact |
| --- | --- | --- | --- |
| Static/build |  |  |  |
| Focused unit/integration |  |  |  |
| Smoke |  |  |  |
| Golden logits / external reference |  |  |  |
| Activation reference / probes / intervention |  |  |  |
| Performance (controlled A/B) |  |  |  |
| Backup reproduction |  |  |  |

For skipped checks, write `not run` and the reason. For model work, record
model/tokenizer SHA-256, architecture, quantization/execution strategy,
compiler/toolchain, and thread count. For performance, retain raw samples,
warmups, process order, CPU/power controls, and comparison revision.

## 4. Findings

Use `blocker`, `required`, `follow-up`, or `observation`. Every non-obvious
finding needs evidence and an owner or issue.

| Severity | Finding / evidence | Owner or issue | Due / disposition |
| --- | --- | --- | --- |
|  |  |  |  |

## 5. Newcomer handoff

In a few sentences, explain what this subsystem does, what it must not do, the
first command a new contributor should run, and where a backup can find the
reference/oracle and fixtures. Mention any local-only prerequisite without
checking private paths into the repository.

## 6. Close-out

- [ ] Index row updated with this record.
- [ ] Primary and backup reviewed the evidence.
- [ ] Governing docs/contracts updated, or an issue links the needed update.
- [ ] Follow-ups are visible and have owners.
- [ ] No model weights, private data, or generated tensor payloads were added.

**Reviewer notes:**
