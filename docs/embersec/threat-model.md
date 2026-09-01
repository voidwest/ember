# EmberSEC: threat model and trust-boundary model

> **Phase I provenance:** frozen audit documentation from branch snapshot
> `e1fe6269`; the measured hardened Ember target is `3ceb7039`. Current main
> retains the applicable hardening, but implementation names and dataflow may
> have evolved. Read this as the published Phase I evidence record.

Frozen for the comparative evaluation. This is a defensive-systems
threat model for LOCAL model-artifact loading; it deliberately excludes
later-phase concerns (attestation, remote serving, GPU, side channels).

## 1. System under consideration

A local LLM runtime loads a GGUF model file and an optional
tokenizer.json and executes inference on untrusted weight bytes.
Components in scope: GGUF parser, tensor descriptor validation, model
construction, tokenizer load, and the kernels that consume validated
tensor state. Out of scope: prompt handling, KV-cache lifetime,
scheduler, OS sandboxing.

## 2. Assets

- A1 Process integrity: the runtime process must not be crashed,
  corrupted, or driven into unbounded resource use by a model file.
- A2 Memory safety: no out-of-bounds access, aliasing violation, or
  use-after-free reachable from model-file values.
- A3 Numerical/behavioral integrity: a well-formed model must be
  interpreted the same way a compliant runtime would interpret it
  (no silent layout misinterpretation).
- A4 Availability of the host: bounded memory/CPU/time during load.

## 3. Attacker model

- **Capability**: the attacker supplies the GGUF and/or tokenizer.json
  bytes (e.g., a poisoned model download, a malicious file in a model
  cache, a tampered artifact). The attacker does NOT control the
  runtime binary, the OS, or other files on the host.
- **Goal**: (a) crash or hang the runtime (DoS), (b) exhaust host
  memory/CPU disproportionately to file size (resource amplification),
  (c) cause memory corruption and possibly code execution via unsafe
  kernel paths (not demonstrated for any runtime in this evaluation;
  included for completeness), (d) make the runtime silently
  misinterpret a malformed-but-accepted file (integrity).
- **Non-goals** (out of scope): remote exploitation, OS-level
  sandbox escape, prompt injection, model-weight backdoors (weights are
  assumed untrusted *values* by design — the boundary is structure and
  interpretation, not content).

## 4. Trust boundaries

```
 T0            T1                 T2                  T3
bytes  -->  parsed descriptors --> validated views --> kernels
 (file)      (untrusted)          (trusted state)
              |                      |
              | T4 (metadata -> config)   T5 (config/tensors -> model)
              |                      |
 tokenizer.json --T6--> tokenizer   |                     |
                                    v                     v
                              model construction      execution
```

| boundary | crossing | checked | assumed |
|---|---|---|---|
| T0 bytes → parser | raw little-endian reads, string/array reads | EOF via read_exact; count sanity | file is a byte stream; no structure trusted yet |
| T1 parsed → validated descriptors | `TensorInfo::validate` (EmberSEC) | rank/dims non-zero; element product overflow; dtype support; block layout incl. contiguous dim; byte length; file range; overlap; absolute caps | validated descriptor arithmetic cannot overflow |
| T2 validated → tensor views | `ValidatedTensorInfo` → `CpuTensor`/`QuantizedWeight`/`KQuantWeight`; `Q8WeightView` | constructor re-checks byte length vs shape; mmap range vs mapping length | view invariants hold by construction |
| T3 views → kernels | SIMD/matmul entry points | entry points accept validated weight types or views; arch kernels private | kernel contracts (ISA, lengths) satisfied by validated objects |
| T4 metadata → config | `LlamaConfig`/`Gemma4Config`/gpt2 builders (EmberSEC) | named caps (context, layers, heads, head_dim, embed, vocab, rope product); head dim parity; finite floats | config values bounded before allocation |
| T5 config/tensors → model | `from_loader` | inventory gate (llama/gpt2); per-block shape checks | model structure internally consistent |
| T6 tokenizer.json → tokenizer | `EmberTokenizer` | size cap; UTF-8; JSON well-formedness; catch_unwind around crate | tokenizers crate does not panic (upstream bug worked around) |

Failure classes A-J (docs/embersec/bug-taxonomy.md) map onto these
boundaries: A=T0/T6, B=T1, C=T4, D=T1 (layout), E=T5, F=T6, G=T1/T4/T6,
H=T1 (misinterpretation), I=T2/T3, J=none (expected rejection).

## 5. Design principles encoded by the model

1. **Validation once at the boundary**, not scattered in kernels: T1/T4
   are the single gates; kernels consume only post-gate state (T3).
2. **Type-level enforcement where possible**: `ValidatedTensorInfo` has
   no public constructor; `Q8WeightView` is constructible only from a
   validated weight — raw `(data, count)` pairs cannot reach kernels.
3. **Bounded by construction**: every allocation reachable from file
   values is bounded by a named cap or by the file size itself.
4. **Fail closed**: any value that does not pass a gate is a structured
   error; there is no "continue with defaults" path for hostile values
   that affect memory layout.
5. **Panics are bugs**: asserts in Ember code that hostile values can
   reach are treated as defects (the fuzz campaign's odd-head-dim and
   1-D-weight findings were converted to structured checks); third-party
   panics (tokenizers crate) are contained at the boundary.

## 6. Residual risks (accepted)

- Weights remain untrusted *values*: kernels receive whatever numbers
  the file contains. This is inherent to the model-loading problem and
  out of scope for structure validation.
- The tokenizer encode path (Oniguruma regex) can still consume CPU on
  crafted patterns (theoretical ReDoS; no demonstrated input).
- The execution plan / scratch arena sizes are checked at plan build,
  not type-enforced against validated tensors (documented follow-up).
