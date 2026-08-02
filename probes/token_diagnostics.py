"""Tokenizer diagnostics for probe analysis.

- token_count per stimulus
- single-token vs multi-token breakdown
- per-task probe accuracy by token_count bucket
- char n-gram baseline by token_count bucket
"""

import hashlib
import json
import numpy as np
from collections import Counter
from pathlib import Path

try:
    from .train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        make_splits,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )
except ImportError:  # direct script execution
    from train_linear_probe import (
        atomic_write_text,
        enforce_prompt_contract,
        load_activation_metadata,
        make_splits,
        sha256_file,
        validate_activation_provenance,
        validate_activation_tensor,
    )


def load_tokenizer(tokenizer_id="Qwen/Qwen2.5-0.5B"):
    from tokenizers import Tokenizer

    path = Path(tokenizer_id)
    return Tokenizer.from_file(str(path)) if path.is_file() else Tokenizer.from_pretrained(tokenizer_id)


def token_counts(rows, tokenizer=None):
    """Return list of token counts for each row's surface form."""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    counts = []
    for index, r in enumerate(rows):
        surf = (
            r.get("surface_dediac")
            or r.get("surface")
            or r.get("expected_surface")
        )
        if not isinstance(surf, str) or not surf:
            raise ValueError(f"stimulus {index} has no non-empty surface form")
        encoding = tokenizer.encode(surf)
        if len(encoding.ids) != len(encoding.special_tokens_mask):
            raise ValueError(
                f"tokenizer returned inconsistent ids/special-token mask for stimulus {index}"
            )
        count = sum(not special for special in encoding.special_tokens_mask)
        if count < 1:
            raise ValueError(f"tokenizer produced no tokens for stimulus {index}")
        counts.append(count)
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
    if not counts:
        raise ValueError("cannot summarize an empty token-count sequence")
    if any(isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1 for count in counts):
        raise ValueError("token counts must be positive integers")
    dist = Counter(counts)
    return {
        "total": len(counts),
        "distribution": {str(k): v for k, v in sorted(dist.items())},
        "single_token": dist.get(1, 0),
        "multi_token": len(counts) - dist.get(1, 0),
        "single_token_pct": float(dist.get(1, 0) / len(counts) * 100),
        "mean_tokens": float(np.mean(counts)),
        "median_tokens": float(np.median(counts)),
        "min_tokens": int(min(counts)),
        "max_tokens": int(max(counts)),
    }


def train_probe_on_subset(activations, labels, indices, n_folds=5, seed=42):
    """Train a standardized logistic probe on a subset. Returns CV accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # filter to subset
    mask = np.array(indices)
    X = activations[mask]
    y_sub = y[mask]

    if X.ndim != 2 or X.shape[0] != len(y_sub):
        raise ValueError("probe subset must be a sample-by-feature matrix aligned to labels")
    splits = make_splits(
        y_sub,
        n_folds=n_folds,
        split_name="token-bucket",
        random_state=seed,
    )
    predictions = np.full(y_sub.shape, -1, dtype=np.int64)
    for train_idx, test_idx in splits:
        probe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
        probe.fit(X[train_idx], y_sub[train_idx])
        predictions[test_idx] = probe.predict(X[test_idx])
    if np.any(predictions < 0):
        raise RuntimeError("token-bucket CV did not predict every sample")
    return float(np.mean(predictions == y_sub))


def token_bucket_probe_analysis(
    activations, rows, task, task_indices, labels, token_counts, best_layer,
    min_examples_per_label=3, n_folds=5, seed=42,
):
    """Analyze probe accuracy by token count bucket."""
    buckets = bucket_labels(token_counts)
    # filter to task indices and select the best layer
    task_buckets = [buckets[i] for i in task_indices]
    task_acts = activations[task_indices, best_layer, :]  # (n, hidden_dim)

    results = {}
    for bucket_name in ["single", "2", "3", "4+"]:
        bucket_idx = [i for i, b in enumerate(task_buckets) if b == bucket_name]
        if len(bucket_idx) < 5:
            results[bucket_name] = {"n": len(bucket_idx), "accuracy": None, "note": "too few examples"}
            continue

        try:
            acc = train_probe_on_subset(
                task_acts, labels, bucket_idx, n_folds=n_folds, seed=seed
            )
        except ValueError as error:
            results[bucket_name] = {
                "n": len(bucket_idx),
                "accuracy": None,
                "note": str(error),
            }
        else:
            results[bucket_name] = {"n": len(bucket_idx), "accuracy": float(acc)}

    # overall single vs multi
    single_idx = [i for i, b in enumerate(task_buckets) if b == "single"]
    multi_idx = [i for i, b in enumerate(task_buckets) if b != "single"]

    results["single"] = results.get("single", {"n": 0, "accuracy": None})
    results["multi"] = {"n": len(multi_idx), "accuracy": None}
    if len(multi_idx) >= 5:
        try:
            acc = train_probe_on_subset(
                task_acts, labels, multi_idx, n_folds=n_folds, seed=seed
            )
        except ValueError as error:
            results["multi"]["note"] = str(error)
        else:
            results["multi"]["accuracy"] = float(acc)

    return results


def char_baseline_by_bucket(
    rows,
    task,
    token_counts,
    min_examples_per_label=3,
    seed=42,
    n_folds=5,
):
    """Char n-gram baseline accuracy by token bucket."""
    if __package__:
        from .token_char_baseline import char_ngram_acc
    else:
        from token_char_baseline import char_ngram_acc
    import re
    ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670]")

    def dediac(s):
        return ARABIC_DIACRITICS.sub("", s)

    if __package__:
        from .run_baseline_probes import extract_labels
    else:
        from run_baseline_probes import extract_labels

    indices, labels, info = extract_labels(rows, task, min_examples_per_label)
    buckets = bucket_labels(token_counts)
    task_buckets = [buckets[i] for i in indices]
    surfaces = []
    for idx in indices:
        r = rows[idx]
        surf = r.get("surface") or r.get("expected_surface")
        if not isinstance(surf, str) or not surf:
            raise ValueError(f"stimulus row {idx} has no non-empty surface form")
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

        try:
            acc = char_ngram_acc(sub_surfaces, sub_labels, None, n_folds, seed)
        except ValueError as error:
            results[bucket_name] = {
                "n": len(bucket_idx),
                "accuracy": None,
                "note": str(error),
            }
        else:
            results[bucket_name] = {"n": len(bucket_idx), "accuracy": float(acc)}

    return results


def run_token_diagnostics(
    activations_path, stimuli_path, output_dir, tasks=None, best_layer_map=None,
    min_examples_per_label=3, seed=42, tokenizer_id=None, n_folds=5,
    require_activation_provenance=False, allow_label_revealed_prompts=False,
    allow_unverifiable_prompt_contract=False, allow_tokenizer_mismatch=False,
):
    """Run full tokenizer diagnostics and save report."""
    if __package__:
        from .run_baseline_probes import (
            DEFAULT_TASKS,
            TASK_DISPLAY,
            extract_labels,
            load_activations,
            load_stimuli,
        )
    else:
        from run_baseline_probes import (
            DEFAULT_TASKS,
            TASK_DISPLAY,
            extract_labels,
            load_activations,
            load_stimuli,
        )

    acts = load_activations(activations_path)
    rows = load_stimuli(stimuli_path)
    activation_metadata = load_activation_metadata(activations_path)
    validate_activation_provenance(
        activations_path,
        tuple(acts.shape),
        stimuli_path,
        activation_metadata,
        require=require_activation_provenance,
    )
    acts = validate_activation_tensor(acts, activations_path, expected_rows=len(rows))
    if min_examples_per_label < 2:
        raise ValueError("min_examples_per_label must be at least 2")

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if tasks is None:
        tasks = DEFAULT_TASKS
    if len(tasks) != len(set(tasks)):
        raise ValueError("tasks must not contain duplicates")
    leakage_report = enforce_prompt_contract(
        rows,
        tasks,
        activation_metadata,
        allow_label_revealed=allow_label_revealed_prompts,
        allow_unverifiable=allow_unverifiable_prompt_contract,
        context="token diagnostics",
    )

    if tokenizer_id is None:
        tokenizer_id = activation_metadata.get("tokenizer_path")
        if not isinstance(tokenizer_id, str) or not tokenizer_id:
            raise ValueError(
                "--tokenizer is required when activation metadata has no tokenizer_path"
            )
    if not Path(tokenizer_id).is_file() and not allow_tokenizer_mismatch:
        raise ValueError(
            "a local tokenizer file is required to verify token diagnostics against "
            "activation metadata; use --allow-tokenizer-mismatch for an explicit "
            "cross-tokenizer analysis"
        )
    print(f"Loading tokenizer ({tokenizer_id})...")
    tok = load_tokenizer(tokenizer_id)
    tokenizer_digest = hashlib.sha256(tok.to_str().encode("utf-8")).hexdigest()
    tokenizer_path = Path(tokenizer_id)
    tokenizer_file_sha = sha256_file(tokenizer_path) if tokenizer_path.is_file() else None
    recorded_tokenizer_sha = activation_metadata.get("tokenizer_sha256")
    tokenizer_match = (
        isinstance(recorded_tokenizer_sha, str)
        and tokenizer_file_sha is not None
        and tokenizer_file_sha == recorded_tokenizer_sha
    )
    if not tokenizer_match and not allow_tokenizer_mismatch:
        raise ValueError(
            "token diagnostics tokenizer cannot be verified against activation metadata; "
            "supply the extraction tokenizer file or pass --allow-tokenizer-mismatch for an "
            "explicit cross-tokenizer analysis"
        )
    tcs = token_counts(rows, tok)
    print(f"  tokenized {len(tcs)} words")

    tdist = token_distribution_report(tcs)
    print(f"  single-token: {tdist['single_token']} ({tdist['single_token_pct']}%)")
    print(f"  multi-token:  {tdist['multi_token']}")
    print(f"  mean tokens:  {tdist['mean_tokens']}")

    if best_layer_map is None:
        best_layer_map = {}

    report = {
        "schema_version": 2,
        "activations_sha256": sha256_file(activations_path),
        "stimuli_sha256": sha256_file(stimuli_path),
        "activation_shape": list(acts.shape),
        "tokenizer": tokenizer_id,
        "tokenizer_file_sha256": tokenizer_file_sha,
        "tokenizer_serialization_sha256": tokenizer_digest,
        "activation_tokenizer_sha256": recorded_tokenizer_sha,
        "tokenizer_matches_activation_metadata": tokenizer_match,
        "tokenizer_mismatch_allowed": allow_tokenizer_mismatch,
        "seed": seed,
        "folds_requested": n_folds,
        "prompt_leakage_audit": leakage_report,
        "label_revealed_prompt_allowed": allow_label_revealed_prompts,
        "unverifiable_prompt_contract_allowed": allow_unverifiable_prompt_contract,
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
        if isinstance(best_layer, bool) or not isinstance(best_layer, (int, np.integer)):
            raise ValueError(f"best layer for {task!r} must be an integer")
        if best_layer < 0 or best_layer >= acts.shape[1]:
            raise ValueError(
                f"best layer {best_layer} for {task!r} is outside 0..{acts.shape[1] - 1}"
            )

        bucket_probe = token_bucket_probe_analysis(
            acts, rows, task, indices, labels, tcs, best_layer,
            min_examples_per_label=min_examples_per_label, seed=seed,
            n_folds=n_folds,
        )
        print(f"  probe by bucket (layer {best_layer}):")
        for bk, bd in bucket_probe.items():
            acc_str = f"{bd['accuracy']:.4f}" if bd.get('accuracy') is not None else "N/A"
            print(f"    {bk:<8s}: n={bd['n']:>4d}  acc={acc_str}")

        char_bucket = char_baseline_by_bucket(
            rows,
            task,
            tcs,
            min_examples_per_label=min_examples_per_label,
            seed=seed,
            n_folds=n_folds,
        )
        print(f"  char n-gram by bucket:")
        for bk, bd in char_bucket.items():
            acc_str = f"{bd['accuracy']:.4f}" if bd.get('accuracy') is not None else "N/A"
            print(f"    {bk:<8s}: n={bd['n']:>4d}  acc={acc_str}")

        report["tasks"][task] = {
            "descriptive": info,
            "selected_layer": int(best_layer),
            "layer_selection": "externally_supplied_or_descriptive_default",
            "probe_by_bucket": bucket_probe,
            "char_ngram_by_bucket": char_bucket,
        }

    out_path = Path(output_dir) / "token_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        out_path,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    print(f"\nsaved token diagnostics to {out_path}")

    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--min-examples-per-label", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="local tokenizer file or HuggingFace ID (default: activation metadata tokenizer)",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--require-activation-provenance", action="store_true")
    ap.add_argument("--allow-label-revealed-prompts", action="store_true")
    ap.add_argument("--allow-unverifiable-prompt-contract", action="store_true")
    ap.add_argument("--allow-tokenizer-mismatch", action="store_true")
    ap.add_argument("--best-layer-root", type=int, default=None)
    ap.add_argument("--best-layer-lemma", type=int, default=None)
    ap.add_argument("--best-layer-pos", type=int, default=None)
    ap.add_argument("--best-layer-abs-pat", type=int, default=None)
    ap.add_argument("--best-layer-conc-pat", type=int, default=None)
    ap.add_argument("--best-layer-gender", type=int, default=None)
    ap.add_argument("--best-layer-number", type=int, default=None)
    args = ap.parse_args()

    if args.min_examples_per_label < 2:
        ap.error("--min-examples-per-label must be at least 2")
    if args.folds < 2:
        ap.error("--folds must be at least 2")

    candidate_best_map = {
        "root": args.best_layer_root, "lemma": args.best_layer_lemma, "pos": args.best_layer_pos,
        "abstract_pattern": args.best_layer_abs_pat, "concrete_pattern": args.best_layer_conc_pat,
        "features.gender": args.best_layer_gender, "features.number": args.best_layer_number,
    }
    best_map = {key: value for key, value in candidate_best_map.items() if value is not None}
    run_token_diagnostics(
        args.activations, args.stimuli, args.output_dir,
        tasks=args.tasks,
        best_layer_map=best_map,
        min_examples_per_label=args.min_examples_per_label,
        seed=args.seed,
        tokenizer_id=args.tokenizer,
        n_folds=args.folds,
        require_activation_provenance=args.require_activation_provenance,
        allow_label_revealed_prompts=args.allow_label_revealed_prompts,
        allow_unverifiable_prompt_contract=args.allow_unverifiable_prompt_contract,
        allow_tokenizer_mismatch=args.allow_tokenizer_mismatch,
    )
