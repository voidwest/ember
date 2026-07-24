#!/usr/bin/env bash
# Synthetic crossover sweep: measure matmul_q8_0_decode at 6 sizes × 4 thread counts.
# Also runs scheduling-overhead benchmark.
set -euo pipefail

OUTDIR="${1:-artifacts/crossover_sweep}"
mkdir -p "$OUTDIR"

CSV="$OUTDIR/crossover.csv"
OVERHEAD_CSV="$OUTDIR/scheduling_overhead.csv"

echo "=== crossover sweep ==="
# Clear CSV and write header on first pass
> "$CSV"

for t in 1 2 4 8; do
    echo "  threads=$t ..."
    RAYON_NUM_THREADS=$t cargo test --release -- crossover_sweep --nocapture --ignored 2>/dev/null \
        | grep -E '^[0-9]' >> "$CSV"
done

echo "  → $CSV ($(wc -l < "$CSV") lines)"

echo ""
echo "=== scheduling overhead ==="
> "$OVERHEAD_CSV"
for t in 1 2 4 8; do
    echo "  threads=$t ..."
    RAYON_NUM_THREADS=$t cargo test --release -- scheduling_overhead --nocapture --ignored 2>/dev/null \
        | grep -E '^[0-9]' >> "$OVERHEAD_CSV"
done

echo "  → $OVERHEAD_CSV ($(wc -l < "$OVERHEAD_CSV") lines)"
echo ""
echo "Done.  Output in $OUTDIR/"
