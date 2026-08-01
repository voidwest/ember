# Packed Q8_0 projections: Phase 0 checkpoint

Date: 2026-07-28

This note freezes the current packed-projection implementation and records a
local reproduction before any new ablation or optimization work. It separates
the previously measured Sapphire Rapids result from the local Tiger Lake
reproduction and records a page-residency limitation found during the
reproduction.

## Baseline state

- Git HEAD: `694f0331446b26c31e343f4807ed106b453d37e7`
- Initial unstaged tracked-diff SHA-256:
  `48814c7e6de6ad0b8f404b448569898cf6a068ec720af64f326144bb956a2ca9`
- Final unstaged tracked-diff SHA-256 after adding only the two Markdown
  visibility exceptions to `.gitignore`:
  `775f2c0680439924434d413230efcf6e884989871781626a58670253bcc15839`
- Staged diff: empty
- Branch state: `main`, one commit ahead of `origin/main`
- The working tree was already substantially dirty. No existing tracked or
  untracked file was reset, stashed, cleaned, deleted, or overwritten.

The initial tracked diff changed 11 files with 2,212 insertions and 131
deletions. The packed backend and scheduler experiments were part of that
pre-existing working tree.

## Previously verified Sapphire Rapids result

These are retained as prior evidence, not reproduced on the local host:

| System | Physical cores | Control tok/s | Packed tok/s | Improvement |
|---|---:|---:|---:|---:|
| c7i.2xlarge | 4 | 19.082 | 23.564 | 23.5% |
| c7i.4xlarge | 8 | 33.791 | 44.368 | 31.3% |

Prior packed Ember reached 68.8% and 69.2% of the matched llama.cpp
throughput. Reported packed RSS was about 62--72 MiB above Ember control and
about 641 MiB below the then-measured llama.cpp process. Those RSS comparisons
must be treated as workload- and llama.cpp-version-specific after the local
residency result below.

## Local environment

| Item | Value |
|---|---|
| CPU | Intel Core i5-1135G7, family 6 model 140 stepping 1 |
| Topology | 1 socket, 4 physical cores, 8 logical CPUs |
| Benchmark affinity | CPUs 0--3, one logical CPU per physical core |
| Caches | 48 KiB L1d/core, 32 KiB L1i/core, 1.25 MiB L2/core, 8 MiB shared L3 |
| ISA | AVX2, F16C, FMA, AVX-512F/BW/VL, AVX-512 VNNI |
| Governor | `powersave` on all logical CPUs |
| Memory | 15 GiB RAM, 19 GiB swap |
| Kernel | Linux 7.1.4-arch1-1 |
| Rust compiler | rustc 1.95.0, commit `59807616`, LLVM 22.1.2 |
| Cargo | 1.95.0 |
| C compiler | GCC 16.1.1 |

`speed.md` describes this laptop as 2 physical cores / 4 threads. Current
topology inspection shows 4 physical cores / 8 threads with sibling pairs
`0,4`, `1,5`, `2,6`, and `3,7`; the older note is stale on this point.

The benchmark JSON's `run_metadata.rust_version` reports `1.92`. That field is
currently populated from `CARGO_PKG_RUST_VERSION`, so it is the crate's minimum
Rust version, not the executing compiler version. The compiler version above
comes from `rustc -vV`.

## Model identity

| Artifact | Size | SHA-256 |
|---|---:|---|
| `Llama-3.2-1B-Instruct-Q8_0.gguf` | 1,321,083,008 B | `432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3` |
| `tokenizer.json` | 17 MiB | `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b` |
| `Qwen3-0.6B-Q8_0.gguf` | 639,446,688 B | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| `tokenizer-qwen3.json` | local file | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |

## Current architecture

### Row-contiguous source representation

The GGUF loader creates a read-only `mmap`. Every file-loaded Q8_0 tensor owns
an `Arc<Mmap>` plus its byte range. A Q8_0 block contains one FP16 scale and 32
signed bytes, for 34 bytes per 32 weights. No projection is materialized as an
F32 weight matrix.

The mapped source remains logically owned after packing. Packing therefore is
currently a second representation with source-page eviction, not a structural
replacement of the source object.

### Packed representation

Eligible Llama projections are repacked in 16-output tiles. For each Q8_0 input
block:

1. Each group of four input-coordinate bytes is copied for output rows 0--15
   into one contiguous 64-byte group.
2. Eight groups cover all 32 coordinates.
3. The 16 FP16 weight scales follow the quant bytes.

One tile/block record is therefore `8 * 64 + 16 * 2 = 544` bytes, the same
encoded size as 16 row-contiguous Q8_0 blocks. Llama-3.2-1B's eligible
projections are multiples of 16, so there is no tail-padding overhead. Their
packed storage is 1,033,895,936 bytes (986 MiB):

| Projection | Shape `[in, out]` | Packed bytes/layer |
|---|---:|---:|
| Q | 2,048 x 2,048 | 4,456,448 |
| K | 2,048 x 512 | 1,114,112 |
| V | 2,048 x 512 | 1,114,112 |
| O | 2,048 x 2,048 | 4,456,448 |
| gate | 2,048 x 8,192 | 17,825,792 |
| up | 2,048 x 8,192 | 17,825,792 |
| down | 8,192 x 2,048 | 17,825,792 |

The AVX-512 kernel broadcasts four activation bytes to all 16 output lanes,
loads the matching 64-byte weight group, forms signed products for VNNI, and
uses eight ZMM floating-point accumulators. Its final reduction deliberately
matches the existing row-contiguous kernel's reduction order.

Rayon partitions contiguous output rows. Chunk boundaries are rounded to the
16-row tile size. The packed pair and triple entry points reuse one
thread-local activation quantization but execute projections sequentially.

### Page eviction and ownership

After one projection finishes packing, model construction calls
`MADV_DONTNEED` on the source GGUF range. This removes resident file-backed
pages when the kernel and OS honor the advice. The mapping and tensor remain
valid. A later generic access faults the source pages back from the file/page
cache.

This behavior is sufficient for a decode-only benchmark that never uses the
generic projection representation. It is not sufficient for ordinary
generation today: multi-token prefill uses the row-contiguous batch kernel and
therefore reads the source projection pages after they were evicted.

### Tied embeddings and LM head

`token_embd.weight` remains mapped Q8_0 and is used directly for row lookup. A
missing `output.weight` causes the LM head to share that mapped Q8_0 weight.
The 16-row projection packer does not pack either object.

The existing LM-head path creates a separate four-row interleaved
representation when the output dimension is at least 65,536. For this model it
occupies 279,085,056 bytes (266.16 MiB). The packed on/off switch does not
change that LM-head representation.

### Fallback and dispatch

- `EMBER_LLAMA_PACKED_Q8=0` disables construction of the 16-row packed
  projections in the same binary.
- Packed construction also requires Llama's adjacent-pair RoPE and runtime
  support for AVX-512F, AVX-512BW, AVX-512 VNNI, AVX2, F16C, and FMA.
- Unsupported machines retain the mapped row-contiguous representation and
  use the existing AVX-512VL/VNNI, AVX2, or scalar dispatch.
- Qwen3 uses split-half RoPE. It remains on the generic model path and does not
  construct or invoke the Llama packed projection path.

### Unsafe boundaries

Packing and model-level ownership are safe Rust except for two narrow
boundaries:

1. The read-only GGUF `mmap`, whose lifetime is held by `Arc<Mmap>`.
2. The unchecked `madvise` range call, invoked only during exclusive model
   construction after source copying has finished.

The packed kernel is an unsafe target-feature function using pointer-based
intrinsics. A safe dispatcher checks all required ISA features, activation and
output lengths, tile-aligned chunk offsets, and packed-layout invariants before
entering it. No new general-purpose unsafe aliasing is used.

## Correctness reproduction

### Static validation

- `cargo fmt -- --check`: pass
- `cargo test --all-targets`: 104 active tests pass; 8 ignored benchmark or
  calibration tests
- `cargo clippy --all-targets --all-features -- -D warnings`: pass
- `RUSTFLAGS='-C target-cpu=native' cargo build --release`: pass
- `git diff --check`: pass

The active packed-kernel test ran on this AVX-512 VNNI host and compared the
packed and row-contiguous outputs bit-for-bit for 70x256, 32x2,048, and
32x8,192 shapes.

### Real-model parity

With four physical cores, a 16-token greedy Llama generation was byte-identical
between packed and `EMBER_LLAMA_PACKED_Q8=0`.

- Output SHA-256: `8df807d048b41abe2d937d07d28f2fbadfcea569c7e9759a25a89d0ad3f131c4`
- Output begins: ` Paris. The Eiffel Tower is located in Paris.`

An 8-token greedy Qwen3 generation was byte-identical with the packed switch
enabled or disabled, confirming that the switch has no effect on Qwen's
generic path.

- Output SHA-256: `12ec0a598c122250ecef9091ecf5e2611235f7f4acdef9310661de07f9d7bab4`
- Output: ` Paris. The capital of Italy is Rome`

## Local decode-only result

Command configuration: 4 physical cores, 128 timed single-token evaluations,
2 warmups, 5 measured repetitions, token id 1, context cap 129.

| Variant | Median tok/s | Samples tok/s | Peak RSS KiB | Minor faults |
|---|---:|---|---:|---:|
| Packed | 33.009 | 34.304, 33.538, 33.009, 32.539, 32.073 | 1,605,556 | 69,740 |
| Control | 22.085 | 22.825, 22.315, 22.085, 21.714, 21.492 | 1,585,364 | 19,484 |

Packed improved the local median by 49.5%. Peak RSS increased by 20,192 KiB
(19.7 MiB, 1.27%). Both variants slowed across samples, so this laptop result
is a functional reproduction and regime check, not a stable cross-system
performance claim.

## Existing operator-profiler result

The profiler was run for 32 tokens, 2 warmups, and 3 measured repetitions.
Times exclude activation quantization. `Speedup` is control time divided by
packed time.

| Operator | Shape `[in, out]` | Control us/call | Packed us/call | Speedup |
|---|---:|---:|---:|---:|
| Q | 2,048 x 2,048 | 172.395 | 99.611 | 1.73x |
| K | 2,048 x 512 | 48.834 | 29.409 | 1.66x |
| V | 2,048 x 512 | 47.147 | 28.568 | 1.65x |
| O | 2,048 x 2,048 | 168.928 | 97.395 | 1.73x |
| gate | 2,048 x 8,192 | 641.152 | 360.725 | 1.78x |
| up | 2,048 x 8,192 | 639.094 | 357.909 | 1.79x |
| down | 8,192 x 2,048 | 373.860 | 364.251 | 1.03x |
| LM head | 2,048 x 128,256 | 6,649.433 | 6,574.800 | 1.01x |

The profiler confirms that all seven per-layer projection classes select
`packed_row_parallel_rayon` only in the packed run. The LM head selects the
same `interleaved_row_parallel_rayon` mode in both runs. The local shape regime
agrees qualitatively with the prior server result: gate/up and attention
projections benefit strongly, while down changes little and the LM head is
outside the new mechanism. Hardware-counter evidence is still required before
assigning the difference to a cache or bandwidth cause.

## Startup, prefill, and residency result

A matched ordinary generation used a six-token prompt and generated 32 greedy
tokens.

| Variant | Prefill | Decode | Whole process | Peak RSS |
|---|---:|---:|---:|---:|
| Packed | 200.6 ms | 951.9 ms (34 tok/s) | 2.14 s | 2,717,456 KiB |
| Control | 194.7 ms | 1,291.5 ms (25 tok/s) | 2.05 s | 1,764,248 KiB |

This single end-to-end sample is not a statistically valid break-even
measurement. It does show that 32 tokens did not recover the observed startup
cost in that run. Five alternating, hot-page-cache, one-token processes were
stable at 532--534 ms packed and 171--173 ms control. This gives an
uninstrumented process-start delta of about 360 ms; it is not yet a direct
measurement of packing alone.

More importantly, a `smaps_rollup` snapshot during decode after ordinary
prefill showed:

| Residency | Packed KiB | Control KiB | Difference KiB |
|---|---:|---:|---:|
| RSS | 2,716,108 | 1,764,216 | 951,892 |
| PSS anonymous | 1,417,044 | 465,144 | 951,900 |
| PSS file-backed | 1,295,314 | 1,295,277 | 37 |

The file-backed residency is effectively identical after prefill, while the
packed run retains about 930 MiB more anonymous pages. This is direct evidence
that generic prefill re-resides the original projection pages while the
anonymous packed projection copy remains resident.

Therefore:

- The decode-only low-overhead result is reproducible.
- The current implementation does not yet provide replacement-like resident
  memory for the normal prefill-then-decode lifecycle.
- The prior low-RSS result is a property of a decode-only residency path, not
  yet a robust end-to-end memory result.

## Local llama.cpp context

A local `llama-bench` build at commit `0272ac9b1` (build 9815) ran the same
model with four threads, zero prompt tokens, 128 generation evaluations, and
five repetitions:

- Mean: 29.895 tok/s
- Median sample: 29.855 tok/s
- Samples: 31.302, 30.567, 29.855, 29.160, 28.594 tok/s
- Peak RSS: 1,373,172 KiB

Packed Ember reached 110.6% of this local median sample but used about 227 MiB
more peak RSS in the decode-only process. This does not contradict the prior
AWS comparison because the llama.cpp version/build and hardware differ. It
does show that a statement such as "Ember remains below llama.cpp RSS" cannot
be generalized without pinning the exact llama.cpp build and workload.

## Exact reproduction commands

Build and validation:

```bash
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
RUSTFLAGS='-C target-cpu=native' cargo build --release
git diff --check
```

Decode-only packed:

```bash
/usr/bin/time -v env RAYON_NUM_THREADS=4 taskset -c 0-3 \
  target/release/ember bench-decode \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf --arch llama \
  --tokens 128 --warmups 2 --repetitions 5 --token-id 1 \
  --max-seq-len 129
```

Decode-only control:

```bash
/usr/bin/time -v env EMBER_LLAMA_PACKED_Q8=0 RAYON_NUM_THREADS=4 \
  taskset -c 0-3 target/release/ember bench-decode \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf --arch llama \
  --tokens 128 --warmups 2 --repetitions 5 --token-id 1 \
  --max-seq-len 129
```

Ordinary generation:

```bash
env RAYON_NUM_THREADS=4 taskset -c 0-3 target/release/ember \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf --arch llama \
  --tokenizer tokenizer.json --prompt 'The capital of France is' \
  --max-tokens 32 --temperature 0 --max-seq-len 64 --benchmark

env EMBER_LLAMA_PACKED_Q8=0 RAYON_NUM_THREADS=4 \
  taskset -c 0-3 target/release/ember \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf --arch llama \
  --tokenizer tokenizer.json --prompt 'The capital of France is' \
  --max-tokens 32 --temperature 0 --max-seq-len 64 --benchmark
```

## Phase 0 verdict

Correctness, dispatch, rollback, Qwen fallback, and the decode throughput
effect are reproducible. The end-to-end replacement-memory hypothesis is not:
normal prefill currently restores the source projection residency. No new
optimization or ablation implementation should be layered on top until this
lifecycle is represented explicitly in the experimental design.
