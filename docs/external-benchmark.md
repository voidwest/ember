# External / adversarial benchmark pathway

This repository's performance scripts are useful for Ember development, but a
credible cross-runtime claim needs a run that another person can assemble and
inspect. `scripts/external_benchmark.py` is a small, model-file-free harness for
that purpose. It does not download, inspect, or ship a model. A benchmark
specification supplies each runtime's command, and the harness runs those
commands in fresh processes for the same declared cases.

The harness is deliberately **not** an evaluator with a hidden answer. It does
not choose Ember, llama.cpp, or another runtime as a reference implementation;
it reports measured wall/CPU/resource observations and pairwise stdout/stderr
hash comparisons. Different output is an observation, not proof that either
runtime is correct. A third party must choose an output-quality or numerical
oracle separately when the question requires one.

## What is recorded

For every warm-up and measured repetition, the output directory contains:

- the exact argv vector and working directory;
- captured binary `stdout.bin` and `stderr.bin` files (not lossy text
  snippets), their captured byte counts, and SHA-256 hashes; complete trials
  are exact, while output-limit failures retain bounded per-stream prefixes
  under the combined cap;
- UTC start/end times, monotonic process elapsed time (sampled at child exit),
  capture-inclusive elapsed wall time, exit status, timeout and output-limit
  status;
- child user/system CPU time, page faults, context switches, and peak RSS when
  the host exposes Linux `/proc/<pid>/status` (`VmHWM`), with a documented
  `getrusage` fallback;
- an independently auditable `trial.json` record.

The run starts by writing `manifest.json` exactly once and a
`manifest.sha256` file. The manifest includes the canonical input-spec hash,
runner-script hash, best-effort repository commit/dirty state, host/Python
facts, exact commands, working directories, environment override values,
inherited-environment names and hash, executable path/hash when resolvable,
and the complete execution matrix. Executable and working-directory identities
are revalidated immediately before each launch and recorded in each trial. It
is never rewritten after trials start. `results.json` is the per-trial aggregate;
`summary.json` contains medians and pairwise comparisons; `checksums.sha256`
covers all files except itself. The output directory must not already exist,
which avoids silently replacing a previous run. Warm-ups and measured repetitions
are interleaved across runtime IDs within each case to reduce a fixed ordering
advantage; the ordering is recorded in the manifest.

The harness executes argv vectors directly (`shell=False`); command strings are
not parsed or interpolated, and child stdin is `/dev/null`. Put prompts, input
paths, and seeds in the command or a reviewed input file rather than relying on
interactive input. A runtime may itself be a wrapper supplied by the benchmark
owner, but that wrapper is part of the command and should be made available for
review.

## Minimal specification

The input is UTF-8 JSON with schema `ember.external-benchmark.v1`. Commands are
lists, not shell strings. Paths and model files below are placeholders supplied
by the person running the benchmark; none are part of Ember or this repository.
The `inputs` object is metadata only: the harness records it but deliberately
does not open or hash model/tokenizer files.

```json
{
  "schema": "ember.external-benchmark.v1",
  "id": "decode-cross-runtime-2026-08",
  "description": "Greedy decode on an independently supplied GGUF",
  "inputs": {
    "model": {
      "label": "same GGUF for every runtime",
      "sha256": "<compute-and-record-before-running>"
    },
    "tokenizer": {
      "label": "same tokenizer where applicable",
      "sha256": "<compute-and-record-before-running>"
    }
  },
  "runtimes": [
    {
      "id": "ember",
      "command": [
        "/absolute/path/to/ember",
        "--arch", "auto",
        "--model", "/absolute/path/to/model.gguf",
        "--tokenizer", "/absolute/path/to/tokenizer.json",
        "--prompt", "The capital of France is",
        "--max-tokens", "32",
        "--temperature", "0"
      ],
      "cwd": "/absolute/path/to/working-directory",
      "inherit_env": false,
      "env": {"RAYON_NUM_THREADS": "8"},
      "metadata": {"source_commit": "<record-the-commit>"}
    },
    {
      "id": "external",
      "command": [
        "/absolute/path/to/independent-runtime",
        "--model", "/absolute/path/to/model.gguf",
        "--prompt", "The capital of France is",
        "--tokens", "32",
        "--temperature", "0"
      ],
      "cwd": "/absolute/path/to/working-directory",
      "inherit_env": false,
      "env": {"OMP_NUM_THREADS": "8"},
      "metadata": {"source_commit": "<runtime-version-or-commit>"}
    }
  ],
  "cases": [
    {
      "id": "greedy-short",
      "args": [],
      "warmups": 1,
      "repetitions": 5,
      "timeout_s": 600,
      "max_output_bytes": 67108864,
      "metadata": {"context_policy": "runtime command declares it"}
    }
  ]
}
```

`cases[].args` is appended to every runtime command. Use it only for arguments
that have the same meaning and spelling in every command. If runtimes expose
different flags, put equivalent fixed flags in each command (as above), or
provide reviewed wrapper executables with a common interface. The harness does
not infer that two differently named options mean the same thing.

Required fields are `schema`, `id`, a non-empty `runtimes` list, and a non-empty
`cases` list. Runtime and case IDs use `[A-Za-z0-9][A-Za-z0-9_.-]*`; duplicate
IDs are rejected. IDs are limited to 128 characters. Preflight limits are 64
runtimes, 256 cases, 1,000 warm-ups or repetitions per case, 10,000 total
trials, a six-hour timeout per case, and a 256 MiB combined stdout/stderr cap
per trial. The complete matrix is also limited to seven days of declared trial
time and 8 GiB of declared output allowance. The output tree is capped at
100,000 files, 200,000 entries, and 32 path components; output names containing
control characters or backslashes are rejected so checksum records remain
unambiguous. Defaults are one warm-up, three measured repetitions, a 600-second
timeout, a 64 MiB combined stdout/stderr cap per trial, and no inherited
environment (`inherit_env` defaults to `false`). Provide absolute executable
paths and explicit environment values for the most portable run.
Relative `cwd` values are resolved relative to the specification file. No shell
expansion (`$HOME`, `~`, pipes, redirects, or globbing) is performed.

## Designing adversarial cases

A single prompt and one short decode can hide mismatched defaults. A useful
pre-registered matrix should include cases that can falsify a favorable story,
for example:

- short and long contexts, including the context length at which a runtime
  changes cache or batching strategy;
- non-ASCII, newline, empty, and boundary-token inputs when the runtimes claim
  to support them;
- greedy decoding with a fixed token budget, plus a separately declared
  seeded-sampling case if sampling is part of the claim;
- cold-start and warm-start policies, stated explicitly rather than mixed in
  one median;
- an unsupported/invalid-input case if failure behavior or safety is being
  compared.

Keep each case's commands semantically equivalent and record all flags. A
failed or divergent case is evidence to investigate, not a value to discard.
The harness can preserve such failures, but it cannot decide whether a
behavior is desirable.

## Running a third-party comparison

1. **Define the question before measuring.** State the model/tokenizer bytes,
   prompt/input, context and generated-token policy, decoding parameters,
   thread/affinity policy, warm-ups, repetitions, timeout, and what resource
   metric is being compared. Do not change these after seeing output.
2. **Prepare inputs independently.** Obtain the model and tokenizer through
   their normal upstream channels. Record SHA-256 hashes, file sizes, and any
   conversion/quantization command. The harness does not bundle or validate
   them. For a quantized model, record the quantization format and metadata
   relevant to both runtimes.
3. **Pin runtimes.** Build or install Ember and the external runtime from
   separately identified revisions. Record compiler, CPU/OS, runtime flags,
   thread environment, and wrapper source. Prefer absolute executable paths.
4. **Write the JSON spec** and preserve the original bytes. Use equivalent
   commands, but do not silently normalize output or add a parser that favors
   Ember. A command can emit machine-readable metrics to stdout; the harness
   still treats them as opaque bytes and measures the process itself.
5. **Run from a cool, otherwise idle host** and keep the output path new:

   ```bash
   python3 scripts/external_benchmark.py \
     --spec ./external-benchmark.json \
     --output ./runs/external-benchmark-2026-08-28
   ```

   Exit status `0` means all trials completed with exit status 0; `1` means
   the matrix and artifacts were written but at least one trial failed,
   timed out, exceeded the output cap, or the spec changed after preflight;
   `2` means the harness/spec failed before a valid run. The harness continues after a trial failure so the
   failure itself remains comparable. Files created by a runtime inside the
   harness output tree count toward the aggregate output cap; exceeding it
   stops the matrix and marks the run failed. Symlinks and other non-regular
   output entries are rejected during final checksum collection.
6. **Verify before interpreting:**

   ```bash
   cd ./runs/external-benchmark-2026-08-28
   sha256sum -c manifest.sha256
   sha256sum -c checksums.sha256
   python3 -m json.tool manifest.json >/dev/null
   python3 -m json.tool summary.json >/dev/null
   ```

   Inspect `trial.json` files as well as `summary.json`; a median never hides
   a failed repetition. The process-wall/resource ratio fields in `summary.json` are
   descriptive comparisons for the two named runtime IDs, not a speedup claim
   or a correctness judgment. `stdout_hash_equal` means bytes matched when
   both sides have stable successful output (`null` means unavailable); it does
   not establish quality.

## What to submit

Submit a compressed output directory together with the exact JSON spec and a
short environment note. At minimum include:

- `manifest.json`, `manifest.sha256`, `results.json`, `summary.json`,
  `checksums.sha256`, and all `trials/**` files;
- model/tokenizer names, URLs or provenance, SHA-256 and sizes (the files
  themselves are not expected in the submission);
- runtime executable/source revisions, build flags, wrapper source, and the
  command used to invoke the harness;
- CPU model and topology, OS/kernel, compiler/Python versions, governor,
  thread/affinity settings, power/thermal state, and whether page caches were
  warm or cold;
- any separately defined correctness/numerical evaluation and its oracle,
  including its own inputs and version.

Review the archive for API keys and other secrets. The manifest intentionally
records names and a hash of inherited environment variables rather than their
values, while explicit `env` overrides are preserved; use a clean environment
or redact a submission copy only with a clear note that its checksum then
changes. Do not replace failed trials with hand-edited values. If the model or
prompt cannot be redistributed, submit the hashes and a private reproduction
procedure instead of claiming that another party reran it.

## Interpretation and limitations

This pathway makes comparisons inspectable; it cannot make unlike runtimes
semantically equivalent by itself.

- A command can point at a different model, tokenizer, chat template, backend,
  quantization, or context policy. The harness records argv and user metadata
  but cannot prove that external files match. Independent input hashes and a
  review of flags are required.
- Stdout/stderr equality is only byte equality. Token IDs, logits, generated
  text, and task scores need an explicit, versioned comparator and (where
  appropriate) a trusted or independently justified oracle. No runtime is
  treated as ground truth here.
- Process elapsed time, capture-inclusive wall time, and RSS are host observations, not portable constants. CPU
  frequency, thermal throttling, page cache, NUMA/affinity, background load,
  allocator state, filesystem, kernel, and process startup all matter. Use
  interleaved repetitions and report spread rather than one favorable run.
- Child CPU/resource accounting is supplied by the host. On Linux the harness
  samples `/proc` for the direct child; very short-lived processes or detached
  descendants may escape the RSS sample. The `getrusage` fallback can be
  cumulative across children and is labeled as such. POSIX trials run in a
  fresh process group and the group is terminated after every trial; a child
  that deliberately calls `setsid` can still escape. Windows cleanup reaches
  only the direct child because the harness has no job-object dependency.
  Hardware counters are intentionally out of scope.
- Warm-ups are recorded but excluded from measured summaries. A process is
  fresh for each repetition, so this is not a long-lived server benchmark and
  does not measure batching, continuous service, or request queuing.
- Timeout and output limits are safety bounds, not performance results. A capture-incomplete trial is marked `capture-failed`; capped hashes describe captured prefixes, not the unknown complete output. A
  killed process is a failure and remains in the report. The harness executes
  arbitrary commands supplied by the user; never run an untrusted
  specification, runtime, or wrapper.
- This script has no model files and no expected result fixtures. A successful
  run proves only that the declared commands ran and what they emitted under
  the recorded conditions. It does not prove Ember quality, external-runtime
  quality, numerical parity, causal validity, or superiority.
