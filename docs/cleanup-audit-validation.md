# Cleanup audit validation

This report records the cleanup series from `2aa15e4` through `9ef0ceb`.
The series preserves the supported CLI/API behavior, model outputs,
hidden-state extraction, tracing, and benchmark reporting.

## Removed or simplified paths

- Deleted the unused `src/decode_scheduler.rs` experiment.
- Removed the placeholder llama.cpp backend that could not execute requests.
- Removed unused SIMD activation/softmax experiments while retaining kernels
  used by production Q8 decode.
- Removed Gemma identity copies, temporary scale buffers, and an unused
  residual helper.
- Avoided the unused final decode evaluation and logits computation in
  hidden-only extraction.
- Moved GGUF tensor ownership directly into models and shared Llama RoPE
  tables across layers.
- Cached probe token selections and consolidated runtime/CLI helpers.
- Removed the direct `libc` dependency.

The net source change is 1,708 insertions and 2,221 deletions (513 lines
removed). Clean release binaries changed from 26,745,432 to 26,718,576 bytes
(-26,856 bytes, -0.10%).

## Numerical and behavioral parity

Gemma 4 used `models/gemma-4-E2B-it.Q8_0.gguf`. Its pre- and post-cleanup
logits artifacts were byte-identical:

```text
SHA-256  8fbb29dddd16444fd1e9487bb43e5280a3979e08116ada71b7d3f34a28f0ee9f
```

Real Qwen validation used:

```text
Qwen3-0.6B-Q8_0.gguf  9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
tokenizer-qwen3.json   aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
```

Enabling the allocation-free path for Qwen split-half RoPE passed a synthetic
single-layer test but failed real 28-layer greedy decoding. For the prompt
`The capital of France is`, the experimental path produced output hash
`d0b6aa5bf7f5eaea0775db2bd21ff2a1b1c47415a169d83437e0bc8eeccc42c4`;
the generic path produced
`34e28dc40d947fd74d4ae0a3720619c69cff98caafd921017930756006667594`.
The outputs diverged after the first generated token.

Commit `9ef0ceb` therefore restores the split-half eligibility guard. With the
guard, traced and untraced 16-token Qwen runs are byte-identical with output
hash `34e28dc40d947fd74d4ae0a3720619c69cff98caafd921017930756006667594`.
Their complete logits `.npy` artifacts are also byte-identical:

```text
SHA-256  55255b46694567b06f5fe1054eca04f1e60f7748ec58cec0530fbc803da8e8c8
```

Qwen remains on the generic decode path until a real-model test demonstrates
exact multi-position fast-path parity.

## Controlled benchmarks

Clean binaries were built from `2aa15e4` (A) and `9ef0ceb` (B). The host was
set to the `performance` power profile for the run and restored to `balanced`
afterward. This machine uses active Intel P-state: the kernel reports the
`powersave` scaling algorithm under both profiles, while
`energy_performance_preference` changed to `performance` on every pinned CPU.

Controls:

- Intel Core i5-1135G7, four physical cores and eight logical CPUs.
- `taskset -c 0-3` selected one logical CPU from each physical core.
- `RAYON_NUM_THREADS=4`.
- Process order: `ABBA BAAB`.
- Decode: two warmup repetitions and three measured repetitions per process.
- Four processes and 12 timed samples per revision and model.

| Workload | A median | B median | Median change | A mean | B mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma decode, 8 tokens | 1,016.03 ms | 995.78 ms | -1.99% | 999.35 ms | 1,001.18 ms |
| Qwen decode, 32 tokens | 1,007.54 ms | 1,004.33 ms | -0.32% | 1,000.95 ms | 1,003.87 ms |
| Gemma prefill, 40 tokens | 1,613.55 ms | 1,608.00 ms | -0.34% | 1,604.97 ms | 1,612.88 ms |

The median and mean move in opposite directions in two workloads and all mean
changes are within 0.5%. The cleanup is therefore classified as
performance-neutral on this host rather than as a claimed speedup.

Representative decode invocation:

```sh
taskset -c 0-3 env RAYON_NUM_THREADS=4 ./ember bench-decode \
  --model MODEL.gguf --arch ARCH --tokens TOKENS --warmups 2 --repetitions 3
```

## Rejected Gemma workspace

A guarded, single-token Gemma Q8 workspace was implemented and checked against
the generic path. A focused synthetic test matched every logit within `1e-5`,
and a real-model six-token greedy run was byte-identical.

The implementation added roughly 700 lines and alternating measurements did
not establish a repeatable gain under the original host settings. Some runs
were slower and others faster as CPU frequency drifted. Because the benefit
was not demonstrated, the entire workspace and its backend helper were
removed instead of retaining a second Gemma execution path. The controlled
results above confirm that the committed cleanup does not need that workspace
to remain performance-neutral.
