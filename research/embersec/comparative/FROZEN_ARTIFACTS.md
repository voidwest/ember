# EmberSEC comparative evaluation — frozen artifacts

Freeze date: 2026-08-12. Everything needed to reproduce the evaluation
results is pinned below. The results JSON files, one complete log per
mutation campaign, and tables in this directory were generated from
exactly these versions; do not
regenerate or re-run without recording the new versions.

## Exact commits / versions

| component | version | commit / id | notes |
|---|---|---|---|
| Ember runtime under test (current) | 0.6.2 + EmberSEC | `3ceb7039f117` (branch embersec/secure-gguf-loader) | final branch; includes the gemma4 early-cap fix (3ceb703, constant-time rejection — no outcome changes; ember-current corpus/perf re-run with the final binary) |
| Ember baseline | 0.6.2 | `1157277ef170` (tag v0.6.2) | parent of the first EmberSEC commit; already contains pre-EmberSEC fail-closed checks (9e434df) |
| llama.cpp (FINAL matrix) | b7999 | `0c1f39a9ae68` (2026-07) | pristine upstream; loader harness (llama_model_load_from_file, no generation); S1 submitted as PR #26946, S2/S3 fixed upstream, S5 mitigated |
| llama.cpp (record) | b5999 | `1dc9614e0673` (2025-07-27) | llama-cli-based results archived in results/llama-cpp-cli-b5999.json |
| llama.cpp neighbor | b5998 | `446595b9b3a1` | used only for crash-class reproduction |
| candle (FINAL matrix) | 0.11.0 | crates.io (unpatched) | parser harness; fix for S4 submitted as PR #3876 |
| candle | 0.11.0 | crates.io candle-core/candle-transformers 0.11.0 | parser-level harness; reference/ dir carries main.rs, Cargo.toml, Cargo.lock |
| harness (injected test) | — | research/embersec/comparative/harness/_embersec_harness.rs | identical file injected into both worktrees |
| corpus manifest | schema 1 | research/embersec/comparative/corpus.json | 62 cases (48 hostile, 14 control) |

## Artifact checksums (SHA-256)

| artifact | sha256 |
|---|---|
| corpus.json | `26e45a5c760ca4e8299ab95b4ae650247072625788e661003225b01c0a168b64` |
| run_eval.py | `619cf0b14cef1e8d8cc7f8a503abb4e533c7a6587938d1da9f5bccfdb64af887` |
| diff_fuzz.py | `31fd50f915326e63382f9b54cfc760429d91bb789e1b3f5131ec3aacd15e3ab9` |
| make_tables.py | `edad4a6d4fbeb8e2fea1d0eb6bd09ce1b63324648f4bec5560db5277c36c4cea` |
| gen_corpus.py | `1724571549d87fede24bb29bfc9b0416a5d0f70edf93e7f62e78063b0392c2ac` |
| gen_valid_models.py | `1d3f94ab8f005a5abde35bc13a928f559d303ece8ae57eb1a126ff6cbc42a207` |
| harness/_embersec_harness.rs | `7f3f7c1a94c99b0959554c9eabdadb4d635e1d2b20d029ccb856fa5c70902380` |
| environments.json | `a1fcb2d98abd0cd39373d596a105583c9c363b77ef7323f7d1e1a39fadb07a4d` |
| results/ember-current.json | `9d120fe4415295c288be0622b1aeafc96a1f6598417f087dc83c30ccbbddf593` |
| results/ember-baseline.json | `fe06156304d51c6292e88cb2af882cfe94237eb5416cfdb0f3c8e0ca27e7725b` |
| results/llama-cpp.json | `60334a6d1357cbf5477facd387d00f2f129be9fc5a601ca1bf063041f8f1d4cf` |
| results/candle.json | `47b6e8aa97f07063e3b5439bb4e1db7ef85b5af1d1fb38310ed27a586d9d142e` |
| results/diff_fuzz/summary.json (latest construction run) | `dfd772b2b5b876c98c1565ff2a184a0c8827b1e718d24ff4c0d7fb0a60060448` |
| results/diff_fuzz/summary_raw-10000-7.json | `a8b56b4aff5ba0670c78e81c3a8f848f5a138b2225dd085ebcb4712d84bdad66` |
| results/diff_fuzz/summary_construction-2000-11.json | `dfd772b2b5b876c98c1565ff2a184a0c8827b1e718d24ff4c0d7fb0a60060448` |
| results/diff_fuzz/log_raw-10000-7.jsonl | `f91c72ff622a5dbebea065f1ed7195bc1ed6a7e2d1a54d30ab75352251ed5d4b` |
| results/diff_fuzz/log_construction-2000-11.jsonl | `00788f92cfdf0554601bb6718c3f0d217e3b3fe864e9d435d163fa1b980874bb` |
| results/diff_fuzz/confirmed/S3_llamacpp_string_length_hang.bin | `be627a5a65f6d91db3edbff74192d94cd3155b916f331aa91a83d32888f2af91` |
| results/diff_fuzz/confirmed/S4_candle_alignment0_panic.bin | `ac4c8ef93b38e3a0b79fb69a54c74d92a7ec97e86a41cd9e964a9a5d36524990` |
| disclosure/repro_S1 | `f200bdec8c12c0ac6cc26993d2901af54c3805aa0062a965259b9e88bba780ea` |
| disclosure/repro_S2 | `2cef2535ce942753bc61a01c22b2f50bad84fc8b5945f487a2542d42bbb81ced` |
| disclosure/repro_S3.bin | `a28b5645cfb845efe7f54a34a5d90fbdeece94e7099bd751f5ccf7f5b0e5a27a` |
| disclosure/repro_S4.bin | `ac4c8ef93b38e3a0b79fb69a54c74d92a7ec97e86a41cd9e964a9a5d36524990` |
| disclosure/repro_S5.bin | `cb60d380b51ceac58c01458ec44f7f869088cccc24fa28793081c7887b929bce` |
| results/real_model.sha256 | `c424db27bbca2c8a733c22b08dfc7169f45ebbd76df37b2be75b9b852cb0c189` |
| reference/candle/Cargo.lock | `905f12829406e63659e01988f386889bd5cfa7b7a02a8c50851c897e0337765e` |
| reference/candle/main.rs | `d1761fb88a48ce6c0c0b563a2c5ff586526f2e865783e9500009bb099ff1f3e4` |
| reference/llama_loader/CMakeLists.txt | `634a5d50e112253e1bc3cba67322c236a4adaeec9572de8347c5dc39dbad05bb` |
| reference/llama_loader/loader_check.cpp | `c03b4f09963ea65628c662068c31cbe1a569392ae2f6713e3bdf314be4fc2ff9` |
| figures/fig1_outcomes.png | `abd43fec34b4fe1a0c4ecbfa0400ab9d48822ce791573e8240c67e1e010bf984` |
| figures/fig2_delta.png | `c50280e198ce36a59d89bea86aa0d2d43e53ac6d87fa34d9af3aeafb033f706f` |
| figures/fig3_diff_fuzz.png | `25347b75e5f0ea2a9e5582425bc40814a91f04ee3dd9edc13fa22ea8f91e8275` |

## Rebuild commands (this host)

```bash
# ember current (use a clean worktree; do not switch a dirty checkout)
git worktree add /tmp/embersec-current embersec/secure-gguf-loader
cd /tmp/embersec-current   # final branch at 3ceb7039
cargo build --release
# inject harness + build the test binary:
cp research/embersec/comparative/harness/_embersec_harness.rs tests/
cargo test --release --test _embersec_harness --no-run

# ember baseline (clean worktree)
git worktree add /tmp/ember-baseline 1157277
cp research/embersec/comparative/harness/_embersec_harness.rs /tmp/ember-baseline/tests/
cd /tmp/ember-baseline && cargo build --release && \
  cargo test --release --test _embersec_harness --no-run

# llama.cpp b7999 loader harness
git clone --depth 1 --branch b7999 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cmake -S research/embersec/comparative/reference/llama_loader \
  -B /tmp/embersec-llama-loader-build \
  -DLLAMA_CPP_SOURCE=/tmp/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF \
  -DBUILD_SHARED_LIBS=OFF
cmake --build /tmp/embersec-llama-loader-build --target embersec_loader_check -j8

# candle harness (reference/candle is the exact source; Cargo.lock pins deps)
#   cargo new /tmp/embersec-candle && cp reference/candle/* /tmp/embersec-candle/ && cargo build --release

# corpus + results
python3 research/embersec/comparative/gen_corpus.py
python3 research/embersec/comparative/gen_valid_models.py
python3 research/embersec/comparative/run_eval.py --target ember-current --out results/ember-current.json
python3 research/embersec/comparative/run_eval.py --target ember-baseline --out results/ember-baseline.json
python3 research/embersec/comparative/run_eval.py --target llama-cpp --out results/llama-cpp.json
python3 research/embersec/comparative/run_eval.py --target candle --out results/candle.json
python3 research/embersec/comparative/diff_fuzz.py --n 10000 --seed 7
python3 research/embersec/comparative/diff_fuzz.py --mode construction --n 2000 --timeout 2 --seed 11 \
  --targets ember-baseline,ember-current,llama-cpp,candle
python3 research/embersec/comparative/make_tables.py
python3 research/embersec/comparative/make_figures.py
```

## Upstream disclosure (2026-08-12)

All five suspected external findings were reproduced on the newest
upstream revisions (llama.cpp b7999 `0c1f39a9`; candle 0.11.0) and
reported upstream:

- llama.cpp PR #26946 (ggml-org/llama.cpp): fix for S1 (zero-dim
  SIGFPE — the one finding still unfixed on master at PR-update time),
  rebased onto master 89e0aa6 (single commit, mergeable). S2 and S3
  were confirmed already fixed upstream on master; S5 is mitigated by a
  clear GGML_ASSERT. Minimal reproducers are hashed above; detailed status
  and non-weaponized reproduction notes are in suspected-external-bugs.md.
- candle PR #3876 (huggingface/candle): fix for S4 (alignment-zero
  div_ceil(0) panic), MERGEABLE; CI awaiting fork-workflow approval.
  Reproducer: disclosure/repro_S4.bin.

The final matrix measures the PRISTINE upstream revisions (b7999 loader
harness; candle 0.11.0 unpatched); the fixes are upstream changes, not
part of the measured runtimes.

## Host

Linux x86_64, 16 GB RAM, 8 cores (thermally noisy; timings are medians
of 3). Toolchains: rustc 1.92.0 (pinned) for both Ember trees; gcc
16.1.1 / cmake 4.4.2 for llama.cpp; rustc 1.95.0 for the candle harness.
