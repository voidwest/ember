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
| Native console runs (`ember gui`) | Yes | Cancel button, Esc, or the palette: checked before prefill, at every decode step of both legs, between the legs, and before each bundle is written |
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

The native console keeps nothing from a cancelled run: no history row, and no
bundle. The token is checked before `write_bundle`, so the cancelled leg never
stages a directory; if the baseline leg had already published its bundle, the
console removes it, since half a pair is not an experiment. The resident model
is kept, so the next run starts without a reload.
