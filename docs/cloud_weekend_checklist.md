# Cloud Weekend Checklist

## 1. VM setup

```bash
apt update && apt install -y build-essential curl git tmux python3-venv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

Use `tmux` for long jobs:

```bash
tmux new -s ember
# detach: Ctrl+B then D
tmux attach -t ember
```

## 2. Repo setup

On the VM:

```bash
git clone https://github.com/voidwest/ember
cd ember
python3 -m venv .venv
source .venv/bin/activate
pip install -r probes/requirements.txt
pip install torch transformers datasets conllu
cargo build --release
huggingface-cli login
```

From local, only if Ember/GGUF checks need local model files:

```bash
rsync -avz \
  stimuli/nonce_root_pattern.json \
  tokenizer*.json \
  *.gguf \
  user@YOUR_VM_IP:~/ember/
```

## 3. Baseline checks

```bash
cargo fmt -- --check
cargo test
cargo test --release simd
cargo clippy --all-targets --all-features -- -D warnings
python -m compileall -q probes stimuli scripts
python probes/test_probe_workflows.py
python probes/run_benchmark.py --config probes/benchmarks/qwen3_smoke.json --dry-run
```

If GGUFs/tokenizers are present:

```bash
target/release/ember \
  --arch llama \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  -n 1 --temperature 0 --benchmark
```

## 4. Dry-run manifest

```bash
python probes/run_benchmark.py \
  --config probes/benchmarks/qwen3_smoke.json \
  --dry-run

python -m json.tool \
  data/benchmarks/qwen3-smoke/benchmark_summary.json | head -120
```

## 5. Golden logits

```bash
mkdir -p data/golden

sha256sum \
  Qwen3-0.6B-Q8_0.gguf \
  Llama-3.2-1B-Instruct-Q8_0.gguf \
  tokenizer-qwen3.json \
  tokenizer.json | tee data/golden/model_hashes.sha256
```

```bash
target/release/ember \
  --arch qwen3 \
  --model Qwen3-0.6B-Q8_0.gguf \
  --tokenizer tokenizer-qwen3.json \
  --prompt "The capital of France is" \
  --dump-logits data/golden/qwen3_06b_ember_logits.npy \
  --dump-gguf-metadata data/golden/qwen3_06b_gguf_metadata.json

target/release/ember \
  --arch llama \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --dump-logits data/golden/llama32_1b_ember_logits.npy \
  --dump-gguf-metadata data/golden/llama32_1b_gguf_metadata.json
```

Put trusted references here:

```bash
ls -lh \
  data/golden/qwen3_06b_reference_logits.npy \
  data/golden/llama32_1b_reference_logits.npy
```

Compare:

```bash
python probes/check_golden_logits.py \
  --ember data/golden/qwen3_06b_ember_logits.npy \
  --reference data/golden/qwen3_06b_reference_logits.npy \
  --label qwen3_06b_reference \
  --tokenizer tokenizer-qwen3.json \
  --top-k 10 \
  --topk-overlap-threshold 0.8 \
  --output data/golden/qwen3_06b_golden_report.json

python probes/check_golden_logits.py \
  --ember data/golden/llama32_1b_ember_logits.npy \
  --reference data/golden/llama32_1b_reference_logits.npy \
  --label llama32_1b_reference \
  --tokenizer tokenizer.json \
  --top-k 10 \
  --topk-overlap-threshold 0.8 \
  --output data/golden/llama32_1b_golden_report.json
```

## 6. PADT download/build

```bash
mkdir -p data/ud/UD_Arabic-PADT data/benchmarks
curl -L \
  -o data/ud/UD_Arabic-PADT/ar_padt-ud-train.conllu \
  https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu

python probes/build_conllu_benchmark.py \
  --input data/ud/UD_Arabic-PADT/ar_padt-ud-train.conllu \
  --output data/benchmarks/arabic_padt_train.json \
  --min-label-count 5
```

## 7. Full mBERT benchmark

```bash
python probes/run_benchmark.py \
  --config probes/benchmarks/ar_ud_mbert_full.json

python -m json.tool \
  data/benchmarks/ar-ud-mbert-full/benchmark_summary.json | head -120
```

## 8. Encoder suite

```bash
python probes/run_benchmark.py \
  --config probes/benchmarks/ar_ud_encoder_suite.json

python -m json.tool \
  data/benchmarks/ar-ud-encoder-suite/benchmark_summary.json | head -160
```

If mBERT is already complete, use a temporary no-mBERT copy on the VM and keep
`probes/benchmarks/ar_ud_encoder_suite.json` unchanged.

## 9. Pull compact artifacts

From local. Prefer compact artifacts. Do not pull `.npy` unless needed.

```bash
rsync -avz \
  --include='*/' \
  --include='benchmark_*.json' \
  --include='*_summary.json' \
  --include='*_report.json' \
  --include='*_plots/***' \
  --include='*.npz' \
  --exclude='*.npy' \
  --exclude='*' \
  user@YOUR_VM_IP:~/ember/data/benchmarks/ data/benchmarks/

rsync -avz \
  --include='*/' \
  --include='*_report.json' \
  --include='*_metadata.json' \
  --include='*.sha256' \
  --include='*.npz' \
  --exclude='*.npy' \
  --exclude='*' \
  user@YOUR_VM_IP:~/ember/data/golden/ data/golden/
```

Only when follow-up analysis requires raw arrays:

```bash
rsync -avz user@YOUR_VM_IP:~/ember/data/benchmarks/*.npy data/benchmarks/
rsync -avz user@YOUR_VM_IP:~/ember/data/golden/*.npy data/golden/
```

## 10. Cleanup

On the VM:

```bash
du -sh data models .cache ~/.cache/huggingface 2>/dev/null || true
git status --short
```

From local:

```bash
ls -lh data/benchmarks data/golden
find data/benchmarks data/golden \( -name 'benchmark_summary.json' -o -name '*_report.json' -o -name '*.npz' \)
```

Destroy the VM only after confirming pullback. Billing usually stops when the
instance is deleted, not when it is powered off.
