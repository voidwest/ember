#!/usr/bin/env bash
# v0.5 performance-isolation matrix (Gate H).
# Measures ordinary inference vs experiment workloads on the pinned
# Q8_0 model; writes JSON + a summary into artifacts/benchmark-v05/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EMBER=./target/release/ember
MODEL=Llama-3.2-1B-Instruct-Q8_0.gguf
TOKENIZER=tokenizer.json
ARCH=llama
OUT=artifacts/benchmark-v05
mkdir -p "$OUT"

PROMPT="في الجملة التالية، الكلمة المميزة هي: كِتَاب. اشرح معناها."
MAX_TOKENS=8

run_ordinary() {
  # ordinary inference (no experiment machinery)
  /usr/bin/time -v "$EMBER" --arch $ARCH --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" --max-tokens $MAX_TOKENS --temperature 0 2>"$1.time" >"$1.out"
}

run_experiment() {
  # experiment run; $2 = spec path, $3 = output dir
  rm -rf "$3"
  /usr/bin/time -v "$EMBER" experiment run "$2" --output "$3" 2>"$1.time" >"$1.out"
}

measure() {
  # $1 = tag; the runner function receives (prefix, ...) as its own args
  # with the prefix as its $1; the final argument is the bundle dir when
  # the case produces a bundle.
  local tag="$1"
  shift
  "$@"
  local wall total rss bundle
  wall=$(grep "wall clock" "$OUT/$tag.time" | awk '{print $8}' | awk -F: '{if (NF==2) print $1*60+$2; else print $1*3600+$2*60+$3}')
  rss=$(grep "Maximum resident" "$OUT/$tag.time" | awk '{print $6}')
  bundle="${@: -1}"
  if [ -d "$bundle" ]; then
    total=$(du -sb "$bundle" | cut -f1)
  else
    total=$(stat -c%s "$OUT/$tag.out")
  fi
  echo "{\"case\": \"$tag\", \"wall_s\": $wall, \"peak_rss_kb\": $rss, \"artifact_bytes\": $total}" > "$OUT/$tag.json"
}

# --- case 1: ordinary run (v0.4-equivalent) ---
measure "01-ordinary" run_ordinary "$OUT/01-ordinary"

# --- case 2: experiment machinery unused = same binary path ---
measure "02-ordinary-v05" run_ordinary "$OUT/02-ordinary-v05"

# --- case 3: experiment with no captures ---
cat > "$OUT/no-captures.toml" <<EOF
schema = "ember.experiment.v1"

[experiment]
name = "perf-no-captures"

[model]
path = "$MODEL"
expected_sha256 = "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3"
tokenizer = "$TOKENIZER"
tokenizer_expected_sha256 = "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b"
arch = "$ARCH"

[execution]
mode = "reference"
threads = 8

[generation]
max_new_tokens = $MAX_TOKENS
temperature = 0.0

[[inputs]]
id = "i1"
text = "$PROMPT"

[output]
directory = "$OUT/03-bundle"
overwrite = true
EOF
measure "03-no-captures" run_experiment "$OUT/03-no-captures" "$OUT/no-captures.toml" "$OUT/03-bundle"

# --- case 4: one selected-row capture at one layer ---
cat > "$OUT/one-layer.toml" <<EOF
schema = "ember.experiment.v1"

[experiment]
name = "perf-one-layer"

[model]
path = "$MODEL"
expected_sha256 = "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3"
tokenizer = "$TOKENIZER"
tokenizer_expected_sha256 = "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b"
arch = "$ARCH"

[execution]
mode = "reference"
threads = 8

[generation]
max_new_tokens = $MAX_TOKENS
temperature = 0.0

[[inputs]]
id = "i1"
text = "$PROMPT"

[[captures]]
id = "prompt-final"
site = "residual-post-mlp"
layers = [7]

[captures.tokens]
kind = "prompt-final"

[output]
directory = "$OUT/04-bundle"
overwrite = true
EOF
measure "04-one-capture" run_experiment "$OUT/04-one-capture" "$OUT/one-layer.toml" "$OUT/04-bundle"

# --- case 5: selected-row capture at every layer (reference example) ---
measure "05-capture-all-layers" run_experiment "$OUT/05-capture-all-layers" examples/experiments/morphology-layerwise-capture.toml "$OUT/05-bundle"

# --- case 6: one intervention (reference example) ---
measure "06-intervention" run_experiment "$OUT/06-intervention" examples/experiments/morphology-intervention.toml "$OUT/06-bundle"

# --- case 7: capture + intervention + restoration (reference example) ---
measure "07-capture-intervene-restore" run_experiment "$OUT/07-capture-intervene-restore" examples/experiments/morphology-restoration.toml "$OUT/07-bundle"

# --- case 8: full-sequence capture (stress) ---
cat > "$OUT/full-tensor.toml" <<EOF
schema = "ember.experiment.v1"

[experiment]
name = "perf-full-tensor"

[model]
path = "$MODEL"
expected_sha256 = "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3"
tokenizer = "$TOKENIZER"
tokenizer_expected_sha256 = "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b"
arch = "$ARCH"

[execution]
mode = "reference"
threads = 8

[generation]
max_new_tokens = $MAX_TOKENS
temperature = 0.0

[[inputs]]
id = "i1"
text = "$PROMPT"

[[captures]]
id = "full"
site = "residual-post-mlp"
layers = [7]
storage = "full-tensor"

[captures.tokens]
kind = "prompt-final"

[output]
directory = "$OUT/08-bundle"
overwrite = true
EOF
measure "08-full-capture" run_experiment "$OUT/08-full-capture" "$OUT/full-tensor.toml" "$OUT/08-bundle"

# --- summary ---
python3 - "$OUT" <<'PYEOF'
import json, sys, glob, os
out = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out, "[0-9][0-9]-*.json"))):
    with open(path) as f:
        rows.append(json.load(f))
summary = {"matrix": rows}
with open(os.path.join(out, "SUMMARY.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"{'case':<28} {'wall_s':>8} {'rss_kb':>9} {'artifact':>12}")
for row in rows:
    print(f"{row['case']:<28} {row['wall_s']:>8.2f} {row['peak_rss_kb']:>9} {row['artifact_bytes']:>12}")
PYEOF
