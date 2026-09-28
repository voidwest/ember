# Agent trace schema (`ember.agent.trace.v1`)

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

## Guarantees and non-goals

- No tensors or logits are ever written to the trace.
- Content fields (`content`, `text`) appear only under the explicit opt-in and
  are absent (not empty) otherwise; consumers should tolerate both.
- Unknown fields must be ignored by readers; unknown `event_type` values must
  be preserved, not dropped.
- A trace is not a security boundary: it records what the run did, not a
  signed attestation. Use `ember evidence` for signed records.
