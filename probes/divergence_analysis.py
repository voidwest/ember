"""correct-vs-incorrect divergence analysis.

answers Q4: at which layer do hidden states for correct and
incorrect predictions diverge? this tells us where the model
"figures out" the task.
"""

import argparse
import json
import numpy as np


def load_activations(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        data = np.load(path)
        return data[list(data.keys())[0]]
    return np.load(path)


def divergence_curves(
    activations: np.ndarray,
    correctness: list[dict],
) -> dict:
    """compute per-layer distance between correct and incorrect mean states.

    returns dict with keys: layer, cos_dist, eucl_dist
    each is (n_layers,) array.
    """
    n_stimuli, n_layers, hidden_dim = activations.shape

    correct_mask = np.array([
        bool(c["correct"])
        if "correct" in c
        else c["predicted"].strip() == c["expected"].strip()
        for c in correctness
    ])
    n_correct = correct_mask.sum()
    n_incorrect = n_stimuli - n_correct

    if n_correct == 0 or n_incorrect == 0:
        print(f"  WARNING: {n_correct} correct, {n_incorrect} incorrect — "
              "cannot compute divergence")
        return {
            "layer": np.arange(n_layers),
            "cos_dist": np.full(n_layers, np.nan),
            "eucl_dist": np.full(n_layers, np.nan),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
        }

    cos_dist = np.zeros(n_layers)
    eucl_dist = np.zeros(n_layers)

    for layer in range(n_layers):
        correct_states = activations[correct_mask, layer, :]
        incorrect_states = activations[~correct_mask, layer, :]

        mean_correct = correct_states.mean(axis=0)
        mean_incorrect = incorrect_states.mean(axis=0)

        # cosine distance: 1 - cosine similarity
        dot = np.dot(mean_correct, mean_incorrect)
        norm_c = np.linalg.norm(mean_correct)
        norm_i = np.linalg.norm(mean_incorrect)
        if norm_c > 0 and norm_i > 0:
            cos_dist[layer] = 1.0 - dot / (norm_c * norm_i)
        else:
            cos_dist[layer] = 1.0

        eucl_dist[layer] = np.linalg.norm(mean_correct - mean_incorrect)

    return {
        "layer": np.arange(n_layers),
        "cos_dist": cos_dist,
        "eucl_dist": eucl_dist,
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
    }


def main():
    parser = argparse.ArgumentParser(
        description="divergence analysis of correct vs incorrect predictions"
    )
    parser.add_argument(
        "--activations", required=True,
        help="path to activations .npy or .npz"
    )
    parser.add_argument(
        "--correctness", required=True,
        help="path to correctness.json (from ember --probe)"
    )
    parser.add_argument(
        "--output", default="data/divergence.npz",
        help="path to save results"
    )
    args = parser.parse_args()

    activations = load_activations(args.activations)
    with open(args.correctness) as f:
        correctness = json.load(f)

    n_stimuli, n_layers, _ = activations.shape
    print(f"activations: {activations.shape}")
    print(f"correctness records: {len(correctness)}")

    results = divergence_curves(activations, correctness)

    print(f"\ncorrect: {results['n_correct']}, incorrect: {results['n_incorrect']}")

    if not np.isnan(results["cos_dist"]).all():
        # find layer of maximum divergence
        max_div_layer = np.argmax(results["cos_dist"])
        print(f"\nmax cosine divergence at layer {max_div_layer}: "
              f"{results['cos_dist'][max_div_layer]:.4f}")
        print(f"max euclidean divergence at layer "
              f"{np.argmax(results['eucl_dist'])}: "
              f"{results['eucl_dist'][np.argmax(results['eucl_dist'])]:.2f}")

        print("\nper-layer divergence:")
        for i in range(n_layers):
            print(f"  layer {i:2d}: cos_dist={results['cos_dist'][i]:.4f}  "
                  f"eucl_dist={results['eucl_dist'][i]:.4f}")

    np.savez(args.output, **results)
    print(f"\nsaved results to {args.output}")


if __name__ == "__main__":
    main()
