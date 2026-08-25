# Ember agentic runtime (`ember agent`, `ember trace`)

The agentic layer (v0.6.7) lets Ember act, not merely generate: the model
can request a tool through its own structured protocol, Ember validates and
executes the call deterministically, reinjects the result into the same
session, and continues generation until the model produces a final answer.
Every step lands in an auditable research trace.

It is **not** an autonomous-agent framework: no shell/network tools, no
browser automation, no MCP servers, no multi-agent orchestration, no
memory systems. It sits entirely above inference - attention, KV kernels,
tokenization, encoders, and tensor math are untouched - and adds zero
unsafe code.

## capabilities

- generic tool schemas (JSON-Schema-compatible subset) with strict,
  structured argument validation; unknown fields are rejections, not
  warnings
- frozen tool registry with duplicate rejection; unknown tools fail closed
- model-family protocol boundary: Qwen2.5 `<tool_call>`, Llama 3.x
  `<|python_tag|>` custom functions, plus an honest generic-JSON testing
  mode; renders pinned byte-exactly by golden tests
- explicit state machine with hard limits: max steps, max tool calls,
  wall time, per-tool timeout, per-turn output tokens, result-size cap
- cooperative cancellation with exact commit semantics (see below)
- crash-tolerant JSONL research traces with provenance, hashed artifacts,
  privacy controls, and platform-stable canonical serialization
- multi-call steps: a single model turn may request several tools; they
  execute in order with limits and cancellation checked between calls
- approval gating: every tool declares a risk class; the default policy
  denies declared external effects (`--allow-unsafe-effects` opts out)
- trace tooling: `ember trace diff` compares two runs structurally,
  `ember trace replay` re-executes recorded deterministic calls offline
  and verifies payload digests, `ember trace report` renders a
  self-contained HTML report
- deterministic built-in tools: `calculate`, `lookup`, `echo`,
  `write_artifact`, `image_fixture`, `fail`, sandboxed
  `read_text_file` / `search_text`

## quick start

```bash
cargo build --release

target/release/ember agent run \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --tokenizer tokenizer.json \
  --protocol llama3 \
  --tools lookup \
  --fixture riyadh="41 C" \
  --prompt "Use the available tool to tell me the fixture temperature in Riyadh." \
  --trace-out run.jsonl

target/release/ember trace inspect run.jsonl
```

The model emits `<|python_tag|>{"name":"lookup","parameters":{"city":"riyadh"}}`,
the fixture returns `41 C`, the result enters the same session, and the
final answer quotes it. The trace reconstructs the whole timeline:

```text
run run-7390533c930aae78
steps: 5  tools: 4  rejected: 0  artifacts: 1

    0.000s  run start
   33.928s  model-0 start
   42.969s  tool-0 `read_text_file` ok (0.3ms)
   ...
  131.871s  final answer
```

## architecture

```
AgentSession::run(user task)
      |
      v
commit system+tools, user message        (engine.commit_message)
      |
      v
model turn                               (engine.generate_turn)
      |  speculative scaffold prefill -> decode
      |  cancel => KV rollback, nothing committed
      v
parse action                             (protocol.parse_assistant_output)
      |-- FinalText ------------------> ledger + RunCompleted
      |-- MalformedToolCall ----------> rejection event + feedback
      '-- ToolCalls([..])
            for each call, in order:
              validate -> approve -> execute -> reinject -> next turn
      |
      v
every arrow emits ordered TraceEvents    (TraceRecorder, JSONL)
```

Model-family syntax lives behind one boundary (`ToolCallProtocol`); the
loop never learns how Qwen or Llama serializes a decision. A present-but-
broken tool call classifies explicitly as `MalformedToolCall` - it never
silently degrades to plain text.

## commit semantics

The ledger records exactly what entered the conversation:
`system -> user -> assistant_tool_call -> tool_result -> ... ->
assistant_final`.

- a finished generation commits scaffold + content + terminal tokens as
  one transaction; cancellation mid-generation rolls the KV cursor back
  and commits nothing;
- validation failures and policy denials are data: traced and fed back to
  the model (`ok:false` results) so it can recover - they do not abort runs;
- if cancellation arrives after a tool executed, the side effect happened
  and stays visible via a `tool_result_uncommitted` event while the
  session stays clean; external effects are never pretended away;
- limits terminate cleanly (`RunStatus::LimitReached`) with the committed
  prefix valid.

## approval gating

Every tool declares a risk class (`ReadOnly`, `LocalWrite`,
`ExternalSideEffect`). The default policy executes read/local-write tools
and denies declared external effects (`denied_by_policy`, traced, fed back
to the model); pass `--allow-unsafe-effects` to approve them. Hosts embed
custom gates through `ApprovalPolicy::custom`. There is no interactive
prompt loop and no sandboxing: this seam exists so approval policies can
be added without touching the loop again.

## research traces

One JSON object per line (`ember.agent.trace.v1`), flushed per event, with
a monotonic `seq` for ordering (wall-clock fields are informational):

```json
{"schema":"ember.agent.trace.v1","run_id":"run-...","seq":29,
 "event_type":"tool_execution_finished","t_ms":102289.7,"step":"tool-3",
 "data":{"tool":"calculate","ok":true,"payload_sha256":"..."}}
```

Provenance captures ember version, git commit/rustc/target, model identity
(path, SHA-256, quantization, tokenizer hash), the full tool-schema
snapshot, limits, and sampling config. Artifacts are written atomically
under the run's directory with sanitized names and recorded SHA-256 +
producer provenance. A torn trailing line (crash mid-write) parses as a
prefix; the inspector reports it instead of failing.

## diff, replay, report

```bash
ember trace diff --a run-a.jsonl --b run-b.jsonl [--fail-on-diff]
ember trace replay --input run.jsonl \
  --tools read_text_file,calculate,write_artifact \
  --sandbox-root ./summaries
ember trace report --input run.jsonl --output report.html
```

`diff` compares status, totals, final-answer digest, and the event-type
skeleton (first divergence reported). `replay` re-executes every recorded
successful call against a freshly built registry - no model loaded - and
verifies each payload against its recorded stable digest (volatile fields
like artifact ids and paths are excluded; failed calls are skipped, not
counted as mismatches). `report` writes one self-contained HTML file:
summary card, timeline bars per step, artifact list, full event table;
inline CSS only, no JavaScript, no external assets.

Privacy defaults, documented: prompts ON, generated text ON, tool payloads
summarized at 2048 bytes, token events OFF. Disabling a mode records a
length + SHA-256 instead of content. Every payload also carries
`payload_sha256` regardless of mode, which is what deterministic replay
verifies against.

## limits

| limit | default |
|-------|---------|
| max steps (model turns) | 8 |
| max tool calls | 16 |
| wall time | 600 s |
| per-tool timeout | 60 s |
| output tokens per turn | 256 |
| reinjected result size | 32 KiB |

Timeout enforcement detaches the worker thread on expiry and discards its
eventual result; synchronous tools cannot be preempted safely, and that
trade-off is documented rather than hidden.

## testing and performance

40 lib unit tests plus 19 hermetic integration tests drive the whole loop
through a scripted engine (`ember::agent::testkit`): exact one-tool round
trips, multi-call steps, failure/malformed/unknown-tool recovery, every
limit, mid-generation and mid-execution cancellation, timeout, panic
containment, artifact hashing, and torn-trace parsing. No GGUF required.

A real-GGUF gate (`tests/agent_e2e.rs`, set `EMBER_AGENT_E2E=1` plus model
paths) validates the live path; it has been executed against
Llama-3.2-1B-Instruct-Q8_0 and Qwen2.5-1.5B-Q8_0. Orchestration overhead
measures ~0.5-1.9 ms per mock run (`benches/agent_overhead.rs`); tracing
adds ~0.2 ms, roughly 16 events / 5 KB per one-tool run.
