"""Tokenizer diagnostics for probe analysis.

- token_count per stimulus
- single-token vs multi-token breakdown
- per-task probe accuracy by token_count bucket
- char n-gram baseline by token_count bucket
"""

import json
import numpy as np
from collections import Counter
from pathlib import Path


def load_tokenizer(tokenizer_id="Qwen/Qwen2.5-0.5B"):
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained(tokenizer_id)


def token_counts(rows, tokenizer=None):
    """Return list of token counts for each row's surface form."""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    counts = []
    for r in rows:
        surf = r.get("surface_dediac") or r.get("surface", "")
        counts.append(len(tokenizer.encode(surf).ids))
    return counts


def bucket_labels(counts):
    """Bucket token counts: '1', '2', '3', '4+', or 'single', 'multi'."""
    buckets = []
    for c in counts:
        if c == 1:
            buckets.append("single")
        elif c == 2:
            buckets.append("2")
        elif c == 3:
            buckets.append("3")
        else:
            buckets.append("4+")
    return buckets


def token_distribution_report(counts):
    """Return a dict with token count distribution."""
    dist = Counter(counts)
    return {
        "total": len(counts),
        "distribution": {str(k): v for k, v in sorted(dist.items())},
        "single_token": dist.get(1, 0),
        "multi_token": len(counts) - dist.get(1, 0),
        "single_token_pct": round(dist.get(1, 0) / len(counts) * 100, 1),
        "mean_tokens": round(np.mean(counts), 2),
        "median_tokens": int(np.median(counts)),
        "min_tokens": int(min(counts)),
        "max_tokens": int(max(counts)),
    }


def train_probe_on_subset(activations, labels, indices, n_folds=5, seed=42):
    """Train a RidgeClassifier probe on a subset of indices. Returns CV accuracy."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # filter to subset
    mask = np.array(indices)
    X = activations[mask]
    y_sub = y[mask]

    min_per_class = int(np.bincount(y_sub).min())
    effective_folds = min(n_folds, min_per_class)
    if effective_folds < 2:
        probe = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeClassifier(alpha=1.0))])
        probe.fit(X, y_sub)
        return float(probe.score(X, y_sub))

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in skf.split(np.zeros(len(y_sub)), y_sub):
        probe = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeClassifier(alpha=1.0))])
        probe.fit(X[train_idx], y_sub[train_idx])
        scores.append(probe.score(X[test_idx], y_sub[test_idx]))
    return float(np.mean(scores))


def token_bucket_probe_analysis(
    activations, rows, task, task_indices, labels, token_counts, best_layer,
    min_examples_per_label=3, n_folds=5, seed=42,
):
    """Analyze probe accuracy by token count bucket."""
    buckets = bucket_labels(token_counts)
    tc_arr = np.array(token_counts)

    # filter to task indices and select the best layer
    task_buckets = [buckets[i] for i in task_indices]
    task_tc = tc_arr[task_indices]
    task_acts = activations[task_indices, best_layer, :]  # (n, hidden_dim)

    results = {}
    for bucket_name in ["single", "2", "3", "4+"]:
        bucket_idx = [i for i, b in enumerate(task_buckets) if b == bucket_name]
        if len(bucket_idx) < 5:
            results[bucket_name] = {"n": len(bucket_idx), "accuracy": None, "note": "too few examples"}
            continue

        acc = train_probe_on_subset(task_acts, labels, bucket_idx, n_folds=n_folds, seed=seed)
        results[bucket_name] = {"n": len(bucket_idx), "accuracy": round(acc, 4)}

    # overall single vs multi
    single_idx = [i for i, b in enumerate(task_buckets) if b == "single"]
    multi_idx = [i for i, b in enumerate(task_buckets) if b != "single"]

    results["single"] = results.get("single", {"n": 0, "accuracy": None})
    results["multi"] = {"n": len(multi_idx), "accuracy": None}
    if len(multi_idx) >= 5:
        acc = train_probe_on_subset(task_acts, labels, multi_idx, n_folds=n_folds, seed=seed)
        results["multi"]["accuracy"] = round(acc, 4)

    return results


def char_baseline_by_bucket(rows, task, token_counts, min_examples_per_label=3, seed=42):
    """Char n-gram baseline accuracy by token bucket."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    import re
    ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670]")

    def dediac(s):
        return ARABIC_DIACRITICS.sub("", s)

    from run_baseline_probes import extract_labels

    indices, labels, info = extract_labels(rows, task, min_examples_per_label)
    buckets = bucket_labels(token_counts)
    task_buckets = [buckets[i] for i in indices]
    task_tc = [token_counts[i] for i in indices]

    surfaces = []
    for idx in indices:
        r = rows[idx]
        surf = r.get("surface") or r.get("expected_surface") or ""
        surfaces.append(dediac(surf))

    results = {}
    for bucket_name in ["single", "2", "3", "4+", "multi"]:
        if bucket_name == "multi":
            bucket_idx = [i for i, b in enumerate(task_buckets) if b != "single"]
        else:
            bucket_idx = [i for i, b in enumerate(task_buckets) if b == bucket_name]

        if len(bucket_idx) < 3:
            results[bucket_name] = {"n": len(bucket_idx), "accuracy": None}
            continue

        sub_surfaces = [surfaces[i] for i in bucket_idx]
        sub_labels = [labels[i] for i in bucket_idx]

        le = LabelEncoder()
        y = le.fit_transform(sub_labels)
        vec = CountVectorizer(analyzer="char", ngram_range=(1, 4), binary=True)
        X = vec.fit_transform(sub_surfaces)

        min_pc = int(np.bincount(y).min())
        ef = min(5, min_pc)
        if ef >= 2:
            skf = StratifiedKFold(n_splits=ef, shuffle=True, random_state=seed)
            scores = []
            for ti, vi in skf.split(np.zeros(len(y)), y):
                clf = LogisticRegression(max_iter=2000)
                clf.fit(X[ti], y[ti])
                scores.append(clf.score(X[vi], y[vi]))
            acc = float(np.mean(scores))
        else:
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X, y)
            acc = float(clf.score(X, y))

        results[bucket_name] = {"n": len(bucket_idx), "accuracy": round(acc, 4)}

    return results


def run_token_diagnostics(
    activations_path, stimuli_path, output_dir, tasks=None, best_layer_map=None,
    min_examples_per_label=3, seed=42, tokenizer_id="Qwen/Qwen2.5-0.5B",
):
    """Run full tokenizer diagnostics and save report."""
    from run_baseline_probes import (
        load_activations, load_stimuli, extract_labels,
        DEFAULT_TASKS, TASK_DISPLAY,
    )

    acts = load_activations(activations_path)
    rows = load_stimuli(stimuli_path)

    print(f"Loading tokenizer ({tokenizer_id})...")
    tok = load_tokenizer(tokenizer_id)
    tcs = token_counts(rows, tok)
    print(f"  tokenized {len(tcs)} words")

    tdist = token_distribution_report(tcs)
    print(f"  single-token: {tdist['single_token']} ({tdist['single_token_pct']}%)")
    print(f"  multi-token:  {tdist['multi_token']}")
    print(f"  mean tokens:  {tdist['mean_tokens']}")

    if tasks is None:
        tasks = DEFAULT_TASKS

    if best_layer_map is None:
        best_layer_map = {}

    report = {
        "token_distribution": tdist,
        "tasks": {},
    }

    for task in tasks:
        print(f"\n── {task} ──")
        try:
            indices, labels, info = extract_labels(rows, task, min_examples_per_label)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        best_layer = best_layer_map.get(task, int(acts.shape[1] // 2))

        bucket_probe = token_bucket_probe_analysis(
            acts, rows, task, indices, labels, tcs, best_layer,
            min_examples_per_label=min_examples_per_label, seed=seed,
        )
        print(f"  probe by bucket (layer {best_layer}):")
        for bk, bd in bucket_probe.items():
            acc_str = f"{bd['accuracy']:.4f}" if bd.get('accuracy') is not None else "N/A"
            print(f"    {bk:<8s}: n={bd['n']:>4d}  acc={acc_str}")

        char_bucket = char_baseline_by_bucket(
            rows, task, tcs, min_examples_per_label=min_examples_per_label, seed=seed,
        )
        print(f"  char n-gram by bucket:")
        for bk, bd in char_bucket.items():
            acc_str = f"{bd['accuracy']:.4f}" if bd.get('accuracy') is not None else "N/A"
            print(f"    {bk:<8s}: n={bd['n']:>4d}  acc={acc_str}")

        report["tasks"][task] = {
            "descriptive": info,
            "probe_by_bucket": bucket_probe,
            "char_ngram_by_bucket": char_bucket,
        }

    out_path = Path(output_dir) / "token_diagnostics.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved token diagnostics to {out_path}")

    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-examples-per-label", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B",
                    help="HuggingFace tokenizer ID (default: Qwen/Qwen2.5-0.5B)")
    ap.add_argument("--best-layer-root", type=int, default=1)
    ap.add_argument("--best-layer-lemma", type=int, default=1)
    ap.add_argument("--best-layer-pos", type=int, default=22)
    ap.add_argument("--best-layer-abs-pat", type=int, default=2)
    ap.add_argument("--best-layer-conc-pat", type=int, default=1)
    ap.add_argument("--best-layer-gender", type=int, default=7)
    ap.add_argument("--best-layer-number", type=int, default=12)
    args = ap.parse_args()

    best_map = {
        "root": args.best_layer_root, "lemma": args.best_layer_lemma, "pos": args.best_layer_pos,
        "abstract_pattern": args.best_layer_abs_pat, "concrete_pattern": args.best_layer_conc_pat,
        "features.gender": args.best_layer_gender, "features.number": args.best_layer_number,
    }
    run_token_diagnostics(
        args.activations, args.stimuli, args.output_dir,
        best_layer_map=best_map,
        min_examples_per_label=args.min_examples_per_label,
        seed=args.seed,
        tokenizer_id=args.tokenizer,
    )
