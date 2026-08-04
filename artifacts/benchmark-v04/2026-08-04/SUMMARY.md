# v0.4 Gate F benchmark matrix (2026-08-04)

Protocol (docs/v04-execution-contract.md section 14): 1 warmup, 5 measured
repetitions, 64-token deterministic single-token decode, fixed token id,
full machine (8 threads, no taskset — the column-parallel matvec is the
performance lever), back-to-back reference/planned/planned-fused per model
so all arms share the machine state. Medians reported. Raw JSON per run in
this directory (32 tokens x 3 reps; the protocol numbers below are the
5-rep medians).

Machine: 8 cores, AVX2/FMA/F16C. Ember commit: see provenance in the JSON
artifacts (the release commit). Reference = v0.3 generic hooked path on the
same binary.

| model | reference tps | planned tps | planned-fused tps | planned ratio | fused ratio |
|---|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 1.48 | 3.42 | 3.41 | 2.32x | 2.31x |
| Llama-3.2-1B Q6_K | 1.43 | 3.31 | 3.29 | 2.32x | 2.31x |
| Qwen2.5-1.5B Q4_K_M | 1.52 | 4.04 | 4.03 | 2.66x | 2.66x |
| Qwen2.5-1.5B Q6_K | 1.97 | 4.06 | 3.89 | 2.06x | 1.97x |

Gate F (>= 1.75x on >= 3/4 primary combinations, no supported model
regressing > 5%, Q8_0 not regressing): PASS on 4/4 combinations for both
planned and planned-fused.

## Gates summary

- Gate A (kernel parity): column-parallel matvec bit-identical to serial
  (both dtypes, both execution paths, gate/up/down/head shapes).
- Gate B (model parity): greedy tokens identical on 6 frozen prompts
  (English + Arabic) for reference/planned/planned-fused on all four
  primary combinations; logits within the frozen envelopes.
- Gate C (hooks): six sites fire identically on the planned path; inactive
  hooks bit-identical; interventions land identically; planned-fused
  defuses F5 under an after_attention hook.
- Gate D (memory): peak RSS reference 844,840 KB vs planned 847,952 KB
  (+0.37%) on Llama-3.2-1B Q4_K_M; packed weights remain mmap-resident.
- Gate E (allocation): planned decode performs <= 3 allocations per token
  after warmup (the logits CpuTensor shape + strides + data), verified on
  the real model via the counting allocator; the column-parallel matvec
  allocates nothing on a warm rayon pool.
- Gate F (performance): matrix above.
- Gate G (external): reference greedy outputs reproduce on the planned and
  fused paths within the frozen envelopes; the v0.3 golden-ladder agreement
  carries over (llama/qwen families).
