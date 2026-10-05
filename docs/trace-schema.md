# Trace schemas

Ember has two independent trace formats, each with its own named schema. They
are not interchangeable and neither supersedes the other.

| Format | Schema | Shape | Written by |
|---|---|---|---|
| Agent trace | `ember.agent.trace.v1` | JSONL, one object per line | `ember agent run` |
| Inference trace | `ember.infertrace.v1` | a single JSON document | `ember --trace ops --trace-out <file>` (generation) |

Both follow the same compatibility rules, stated once under
[Guarantees](#guarantees-and-non-goals) below.

## Agent trace (`ember.agent.trace.v1`)


One JSON object per line (JSONL). Ordering is guaranteed by `seq`, never by
wall-clock time. The writer is single-owner and crash-tolerant: it flushes per
event, and a torn final line is expected to be skipped by readers.

## Privacy defaults

Prompts and generated text are **opt-in**. With default configuration a trace
contains lengths and SHA-256 digests, never content:

- `--trace-content` (CLI) / `TraceConfig { trace_prompts: true }` records
  prompt text and generated text verbatim;
- tool payloads default to a 2048-byte excerpt (`ToolTraceMode::Summary`);
  `Full` and `Hash` modes are available programmatically;
- per-token events are off by default and are the one knob that can produce
  very large files on long runs.

## Common fields (every line)

| Field | Required | Meaning |
|---|---|---|
| `schema` | yes | `"ember.agent.trace.v1"`; a breaking change must mint a new identifier |
| `run_id` | yes | identity of this run; `seq` restarts per run |
| `seq` | yes | monotonic counter within the run; the only ordering guarantee |
| `event_type` | yes | frozen vocabulary; additions are compatible, removals are breaking |
| `t_ms` | yes | milliseconds since run start (monotonic clock) |
| `ts_epoch_ms` | yes | wall-clock ms; consumers must not order by this |
| `step` | yes | step id (`run`, `user`, `model-N`, `tool-N`, `finalize`) |
| `phase` | yes (may be `null`) | narrowed phase such as `prefill`, `decode`, `validate`, `execute` |
| `data` | yes (object or `null`) | event-specific payload |

## Event vocabulary

`run_started`, `provenance`, `model_call_started`, `model_call_finished`,
`assistant_action_parsed`, `tool_call_validated`, `tool_call_rejected`,
`tool_execution_started`, `tool_execution_finished`, `tool_result_committed`,
`tool_result_uncommitted`, `message_committed`, `session_state_changed`,
`artifact_written`, `run_completed`, `run_failed`, `run_cancelled`,
`run_terminated`, `generation_token` (only when `token_events` is on).

Terminal events carry the outcome reason (cancelled, timeout, limit kind, or
error category). `provenance` carries model identity: architecture, model
SHA-256 (when enabled), Ember version, and git commit.

## Inference trace (`ember.infertrace.v1`)

A single JSON document recording per-operation CPU timing for one generation
run. This is a **performance/provenance sidecar**, not a scientific bundle:
research conclusions rest on `ember.bundle.v1` and the golden logits, not on
this file.

Written by `--trace-out`:

```json
{
  "schema": "ember.infertrace.v1",
  "schema_version": 1,
  "prefill": { "phase": "prefill", "events": [ /* … */ ], "total_duration_ns": 0 },
  "decode":  { "phase": "decode",  "events": [ /* … */ ], "total_duration_ns": 0 }
}
```

| Field | Required | Meaning |
|---|---|---|
| `schema` | yes | `"ember.infertrace.v1"`; a breaking change must mint a new identifier |
| `schema_version` | yes | legacy integer, retained for readers predating the named id; new readers ignore it |
| `prefill` | no (may be absent) | prefill report, absent when tracing was not active during prefill |
| `decode` | no (may be absent) | decode report |

Each report holds `phase`, `token_index`, `events`, `total_duration_ns`, and an
optional `run_metadata` block. `run_metadata` is collected from `/proc` and is
therefore **populated on Linux and empty on macOS** — a known portability limit,
not a contract change.

Compatibility rules specific to this format:

- A reader must **reject** a document with no `schema` field. Documents written
  before the named id existed carried only `schema_version`; accepting them
  silently would mean the format was never versioned. `parse_infertrace_document`
  fails closed with a message naming the expected id.
- A reader must **reject** an unknown schema, including a future major
  (`ember.infertrace.v2`), rather than reinterpret it.
- A reader must **ignore unknown fields**, so additive changes stay compatible.

`ember::trace::parse_infertrace_document` implements this and is covered by
`tests/infertrace_schema.rs`.

## Guarantees and non-goals

- No tensors or logits are ever written to the trace.
- Content fields (`content`, `text`) appear only under the explicit opt-in and
  are absent (not empty) otherwise; consumers should tolerate both.
- Unknown fields must be ignored by readers; unknown `event_type` values must
  be preserved, not dropped.
- A breaking change to either schema must mint a new named identifier; a bare
  integer version is not a schema and will not be accepted as one.
- A trace is not a security boundary: it records what the run did, not a
  signed attestation. Use `ember evidence` for signed records.
