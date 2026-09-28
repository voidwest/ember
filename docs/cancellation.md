# Cancellation contract

Ember stops long-running work cooperatively. There is no preemption: a loop
checks a token at documented points and returns a typed error when it fires.

## The token

`ember::cancel::CancelToken` is a cloneable `Arc<AtomicBool>`. It carries no
scheduling or signal handling of its own — the CLI installs a SIGINT handler
that fires the token; tests fire it directly.

A cancelled operation returns `ember::cancel::Cancelled` (downcastable from
`anyhow::Error`). The CLI maps it to **exit code 4** and prints `cancelled`.
The token is the same one referenced by `docs/api-stability.md` policy; the
first stable surface that exposes it is the planned `ember::cancel` module.

## Check points (CLI generation)

| Phase | Interruptible? | Where |
|---|---|---|
| Before prefill | Yes | `generate_with_execution`, before the KV cache is created |
| Prefill forward pass | **No** (single pass) | A cancel arriving mid-prefill fires at the first decode-step check |
| Each decode step | Yes | Top of the decode loop, before sampling/forward |
| Interactive mode | No (yet) | Not wired; Ctrl-C still terminates the process |
| `demo` mode | No (yet) | Inline loop; not wired |
| Experiment runs (`ember experiment run`) | No (yet) | Not wired |
| `score-batch` generation lines | No (yet) | Not wired |

## Second Ctrl-C

The handler counts: the first SIGINT fires the token; a second SIGINT (any time
after the first) exits immediately with code 130, so a run stuck in a forward
pass can always be abandoned.

## State left behind

Cancellation never writes partial generation output. The KV cache belongs to
the caller: the CLI drops it; the agent and voice sessions roll back with
`KVCache::truncate_to` on their own cancellation paths (see
`docs/agent-runtime.md`). Trace files written up to the cancellation remain
valid: the JSONL writer flushes per event, so a cancelled run has a complete
prefix.
