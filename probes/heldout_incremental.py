"""Run heldout probes one strategy at a time, saving incrementally.

Solves the overwrite problem: each (task, strategy) pair writes to its own file.
"""

import argparse, json, sys, os, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_heldout_probes import (
    load_activations, load_stimuli, extract_labels,
    probe_one_strategy, compute_ngram_baseline
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=["pos"])
    parser.add_argument("--strategies", nargs="+", 
                        default=["random", "surface-heldout", "lemma-heldout", "root-heldout"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-examples", type=int, default=3)
    args = parser.parse_args()
    
    activations = load_activations(args.activations)
    stimuli = load_stimuli(args.stimuli)
    os.makedirs(args.output_dir, exist_ok=True)
    
    for task in args.tasks:
        labels, indices = extract_labels(stimuli, task, args.min_examples)
        if len(labels) < 2:
            print(f"SKIP {task}: insufficient labels")
            continue
        
        task_acts = activations[indices]
        task_stimuli = [stimuli[i] for i in indices]
        print(f"\n=== {task}: {len(labels)} examples, {len(set(labels))} classes ===")
        
        for strategy in args.strategies:
            out_path = os.path.join(args.output_dir, f"heldout_{task}_{strategy}.json")
            if os.path.exists(out_path):
                print(f"  SKIP {strategy}: already exists")
                continue
            
            print(f"  Running {strategy}...")
            result = probe_one_strategy(
                task_acts, labels, task_stimuli, task, strategy,
                args.folds, args.seed
            )
            
            with open(out_path, 'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            best = result.get("probe_best_accuracy", 0)
            best_l = result.get("probe_best_layer", "?")
            char = result.get("char_ngram_accuracy", 0)
            print(f"    L={best_l} probe={float(best)*100:.1f}% char={float(char)*100:.1f}%")

if __name__ == "__main__":
    main()
