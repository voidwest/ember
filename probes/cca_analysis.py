"""canonical correlation analysis on hidden states and probe weights
to compare morphological subspaces across layers and models.

answers Q3 (are root and pattern disentangled?) and provides a
geometry-based complement to the linear probe accuracy curves.
"""

import argparse
import json
import numpy as np

try:
    from .analysis_common import assert_row_alignment
    from .train_linear_probe import atomic_savez, load_activations, sha256_file
except ImportError:  # direct script execution
    from analysis_common import assert_row_alignment
    from train_linear_probe import atomic_savez, load_activations, sha256_file


def _validate_cca_inputs(X, Y, n_components, reg, *, min_samples=3):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CCA inputs must be aligned rank-2 matrices, got {X.shape} and {Y.shape}"
        )
    if X.shape[0] < min_samples or X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError(
            f"CCA requires at least {min_samples} samples and non-empty feature axes"
        )
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("CCA inputs contain non-finite values")
    if (
        isinstance(n_components, bool)
        or not isinstance(n_components, (int, np.integer))
        or n_components < 1
        or not np.isfinite(reg)
        or reg < 0.0
    ):
        raise ValueError("n_components must be positive and reg must be finite/non-negative")
    return X, Y


def _compact_pca_scores(train: np.ndarray, evaluate: np.ndarray):
    """Fit compact PCA on training rows and project training/evaluation rows."""
    mean = train.mean(axis=0)
    centered = train - mean
    U, singular, basis = np.linalg.svd(centered, full_matrices=False)
    tolerance = (
        np.finfo(np.float64).eps
        * max(centered.shape)
        * (singular[0] if len(singular) else 0.0)
    )
    rank = int(np.sum(singular > tolerance))
    if rank == 0:
        raise ValueError("CCA input is constant or numerically rank-zero")
    basis = basis[:rank]
    train_scores = U[:, :rank] * singular[:rank]
    evaluation_scores = (evaluate - mean) @ basis.T
    return train_scores, evaluation_scores


def _cca_from_compact_scores(
    X_train: np.ndarray,
    X_evaluate: np.ndarray,
    Y_train: np.ndarray,
    Y_evaluate: np.ndarray,
    n_components: int,
    reg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit regularized CCA in compact coordinates and evaluate canonical variates."""
    dof = X_train.shape[0] - 1
    variance_x = np.sum(X_train * X_train, axis=0) / dof
    variance_y = np.sum(Y_train * Y_train, axis=0) / dof
    # Treat reg as a dimensionless fraction of average component variance. This
    # preserves CCA's scale invariance while still stabilizing small singular
    # directions.
    ridge_x = reg * float(np.mean(variance_x))
    ridge_y = reg * float(np.mean(variance_y))
    inv_sqrt_x = 1.0 / np.sqrt(variance_x + ridge_x)
    inv_sqrt_y = 1.0 / np.sqrt(variance_y + ridge_y)
    cross_covariance = X_train.T @ Y_train / dof
    whitened = (
        inv_sqrt_x[:, None] * cross_covariance * inv_sqrt_y[None, :]
    )
    left, fitted_correlations, right_t = np.linalg.svd(
        whitened, full_matrices=False
    )
    count = min(n_components, len(fitted_correlations))
    axes_x = inv_sqrt_x[:, None] * left[:, :count]
    axes_y = inv_sqrt_y[:, None] * right_t.T[:, :count]
    return (
        fitted_correlations[:count],
        X_evaluate @ axes_x,
        Y_evaluate @ axes_y,
    )


def _heldout_component_correlations(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    n_components: int,
    reg: float,
) -> np.ndarray:
    _, X_canonical, Y_canonical = _cca_from_compact_scores(
        X_train,
        X_test,
        Y_train,
        Y_test,
        n_components,
        reg,
    )
    correlations = []
    for component in range(X_canonical.shape[1]):
        x = X_canonical[:, component] - X_canonical[:, component].mean()
        y = Y_canonical[:, component] - Y_canonical[:, component].mean()
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denominator <= np.finfo(np.float64).tiny:
            continue
        correlations.append(float(np.dot(x, y) / denominator))
    if not correlations:
        raise ValueError("held-out CCA canonical variates are constant")
    return np.clip(np.asarray(correlations), -1.0, 1.0)


def _cv_splits(sample_count: int, folds: int, random_state: int):
    if isinstance(folds, bool) or not isinstance(folds, (int, np.integer)) or folds < 2:
        raise ValueError("CCA cross-validation requires at least 2 folds")
    if sample_count < 6:
        raise ValueError("held-out CCA requires at least 6 aligned samples")
    effective_folds = min(int(folds), sample_count // 3)
    rng = np.random.default_rng(random_state)
    permutation = rng.permutation(sample_count)
    all_indices = np.arange(sample_count)
    splits = []
    for test in np.array_split(permutation, effective_folds):
        train_mask = np.ones(sample_count, dtype=bool)
        train_mask[test] = False
        train = all_indices[train_mask]
        splits.append((train, np.asarray(test)))
    return splits


def svd_cca(X, Y, n_components=10, reg=1e-4):
    """Fit regularized compact CCA and return in-sample correlations.

    X: (n_samples, d_x)
    Y: (n_samples, d_y)

    This fitting primitive is useful for reference tests. Reported layer and
    cross-model matrices use :func:`cross_validated_cca`, because in-sample
    CCA is optimistically biased when feature width approaches sample count.
    """
    X, Y = _validate_cca_inputs(X, Y, n_components, reg)
    X_scores, _ = _compact_pca_scores(X, X)
    Y_scores, _ = _compact_pca_scores(Y, Y)
    fitted, _, _ = _cca_from_compact_scores(
        X_scores, X_scores, Y_scores, Y_scores, n_components, reg
    )
    return np.clip(fitted, 0.0, 1.0)


def cross_validated_cca(
    X,
    Y,
    n_components=10,
    reg=1e-4,
    cv_folds=5,
    random_state=0,
):
    """Return held-out canonical correlations fitted independently per fold.

    Training-fold PCA and CCA directions are applied to unseen rows. Signed
    held-out correlations are averaged by fold size, then negative averages
    are clipped to zero so the result remains a conventional [0, 1]
    similarity. This prevents sample-space saturation from masquerading as
    cross-representation agreement.
    """
    X, Y = _validate_cca_inputs(X, Y, n_components, reg, min_samples=6)
    splits = _cv_splits(X.shape[0], cv_folds, random_state)
    X_views = []
    Y_views = []
    test_sizes = []
    for train, test in splits:
        X_views.append(_compact_pca_scores(X[train], X[test]))
        Y_views.append(_compact_pca_scores(Y[train], Y[test]))
        test_sizes.append(len(test))
    return _aggregate_heldout_cca(
        X_views,
        Y_views,
        test_sizes,
        n_components,
        reg,
    )


def _aggregate_heldout_cca(
    X_views: list[tuple[np.ndarray, np.ndarray]],
    Y_views: list[tuple[np.ndarray, np.ndarray]],
    test_sizes: list[int],
    n_components: int,
    reg: float,
) -> np.ndarray:
    if not (len(X_views) == len(Y_views) == len(test_sizes)) or not X_views:
        raise ValueError("held-out CCA views are empty or misaligned")
    sums = np.zeros(n_components, dtype=np.float64)
    weights = np.zeros(n_components, dtype=np.float64)
    for (X_train, X_test), (Y_train, Y_test), test_size in zip(
        X_views, Y_views, test_sizes, strict=True
    ):
        correlations = _heldout_component_correlations(
            X_train, X_test, Y_train, Y_test, n_components, reg
        )
        count = len(correlations)
        sums[:count] += correlations * test_size
        weights[:count] += test_size
    available = np.flatnonzero(weights > 0)
    if not len(available):
        raise ValueError("held-out CCA produced no evaluable components")
    count = int(available[-1]) + 1
    return np.clip(sums[:count] / weights[:count], 0.0, 1.0)


def cca_layer_matrix(
    activations: np.ndarray,
    n_components: int = 10,
    reg: float = 1e-4,
    cv_folds: int = 5,
) -> np.ndarray:
    """compute pairwise CCA similarity between every pair of layers.

    activations: (n_stimuli, n_layers, hidden_dim)
    returns: (n_layers, n_layers) matrix of mean canonical correlations.
    """
    activations = np.asarray(activations, dtype=np.float64)
    if (
        activations.ndim != 3
        or activations.shape[0] < 6
        or activations.shape[1] == 0
        or activations.shape[2] == 0
        or not np.isfinite(activations).all()
    ):
        raise ValueError(
            "CCA activations must be a finite [samples>=6, layers>0, hidden>0] tensor"
        )
    n_layers = activations.shape[1]
    sim = np.zeros((n_layers, n_layers))

    splits = _cv_splits(activations.shape[0], cv_folds, 0)
    test_sizes = [len(test) for _, test in splits]
    # PCA is layer/fold-specific, not layer-pair-specific. Caching these compact
    # scores avoids repeating the expensive hidden-width SVD for every pair.
    views = [
        [
            _compact_pca_scores(
                activations[train, layer, :],
                activations[test, layer, :],
            )
            for train, test in splits
        ]
        for layer in range(n_layers)
    ]

    for i in range(n_layers):
        for j in range(i, n_layers):
            c = _aggregate_heldout_cca(
                views[i],
                views[j],
                test_sizes,
                n_components,
                reg,
            )
            sim[i, j] = c.mean()
            sim[j, i] = sim[i, j]

    return sim


def cca_cross_model(mat_a: np.ndarray, mat_b: np.ndarray,
                    layers_a: list[int] | None = None,
                    layers_b: list[int] | None = None,
                    n_components: int = 10,
                    reg: float = 1e-4,
                    cv_folds: int = 5) -> np.ndarray:
    """compute CCA between layers of two different models.

    mat_a, mat_b: (n_stimuli, n_layers, hidden_dim)
    returns: (len(layers_a), len(layers_b)) CCA similarity matrix.
    """
    mat_a = np.asarray(mat_a, dtype=np.float64)
    mat_b = np.asarray(mat_b, dtype=np.float64)
    if mat_a.ndim != 3 or mat_b.ndim != 3:
        raise ValueError(
            f"cross-model CCA requires rank-3 tensors, got {mat_a.shape} and {mat_b.shape}"
        )
    if mat_a.shape[0] != mat_b.shape[0]:
        raise ValueError(
            f"cross-model CCA requires equal aligned sample counts, got {mat_a.shape[0]} and {mat_b.shape[0]}"
        )
    if layers_a is None:
        layers_a = list(range(mat_a.shape[1]))
    if layers_b is None:
        layers_b = list(range(mat_b.shape[1]))
    if not layers_a or not layers_b:
        raise ValueError("cross-model CCA layer selections must be non-empty")
    if any(layer < 0 or layer >= mat_a.shape[1] for layer in layers_a):
        raise ValueError("cross-model CCA model-A layer selection is out of range")
    if any(layer < 0 or layer >= mat_b.shape[1] for layer in layers_b):
        raise ValueError("cross-model CCA model-B layer selection is out of range")
    if not np.isfinite(mat_a).all() or not np.isfinite(mat_b).all():
        raise ValueError("cross-model CCA activations contain non-finite values")

    splits = _cv_splits(mat_a.shape[0], cv_folds, 0)
    test_sizes = [len(test) for _, test in splits]
    views_a = {
        layer: [
            _compact_pca_scores(mat_a[train, layer, :], mat_a[test, layer, :])
            for train, test in splits
        ]
        for layer in layers_a
    }
    views_b = {
        layer: [
            _compact_pca_scores(mat_b[train, layer, :], mat_b[test, layer, :])
            for train, test in splits
        ]
        for layer in layers_b
    }

    sim = np.zeros((len(layers_a), len(layers_b)))
    for ii, i in enumerate(layers_a):
        for jj, j in enumerate(layers_b):
            c = _aggregate_heldout_cca(
                views_a[i],
                views_b[j],
                test_sizes,
                n_components,
                reg,
            )
            sim[ii, jj] = c.mean()

    return sim


def probe_weight_similarity(
    probes_path: str,
    n_components: int = 5,
) -> np.ndarray:
    """compute subspace similarity between root and pattern probes per layer.

    loads .npz with keys: root_probe_weights, pattern_probe_weights
    each is a list of arrays shape (n_classes, hidden_dim).

    uses CCA on transposed weight matrices to measure the subspace
    angle between the root-discriminating and pattern-discriminating
    directions.

    returns per-layer mean canonical correlation.
    """
    with np.load(probes_path, allow_pickle=False) as data:
        if str(data.get("probe_weight_space", "")) != "raw_activation":
            raise ValueError("probe weights must declare raw_activation coordinate space")
        root_w = np.asarray(data["root_probe_weights"], dtype=np.float64)
        pat_w = np.asarray(data["pattern_probe_weights"], dtype=np.float64)
    if root_w.ndim != 3 or pat_w.ndim != 3 or root_w.shape[0] != pat_w.shape[0]:
        raise ValueError(f"invalid or mismatched probe weight tensors: {root_w.shape}, {pat_w.shape}")

    n_layers = len(root_w)
    cca_means = np.zeros(n_layers)

    for layer in range(n_layers):
        c = weight_subspace_correlations(
            root_w[layer],
            pat_w[layer],
            n_components=n_components,
        )
        cca_means[layer] = c.mean()

    return cca_means


def validate_probe_artifact_contract(
    probes_path: str,
    activations_path: str,
    activation_shape: tuple[int, int, int],
    *,
    allow_label_revealed: bool = False,
    allow_unverifiable: bool = False,
) -> dict:
    """Bind probe directions to their activation source and prompt contract."""
    with np.load(probes_path, allow_pickle=False) as data:
        def scalar(key: str):
            if key not in data:
                return None
            value = np.asarray(data[key])
            if value.size != 1:
                raise ValueError(f"probe artifact field {key!r} must be scalar")
            return value.reshape(-1)[0].item()

        recorded_sha = scalar("activations_sha256")
        recorded_shape_value = data.get("activation_shape")
        recorded_shape = (
            tuple(int(value) for value in np.asarray(recorded_shape_value).tolist())
            if recorded_shape_value is not None
            else None
        )
        leakage_json = scalar("prompt_leakage_audit_json")

    actual_sha = sha256_file(activations_path)
    if not isinstance(recorded_sha, str) or recorded_sha != actual_sha:
        raise ValueError("probe weights were not trained from the supplied activations")
    if (
        recorded_shape is None
        or recorded_shape != activation_shape
    ):
        raise ValueError(
            "probe artifact activation shape does not match supplied activations"
        )

    if not isinstance(leakage_json, str):
        leakage = {"status": "unverifiable_missing_probe_leakage_audit"}
    else:
        def reject_constant(value: str) -> None:
            raise ValueError(f"non-standard JSON constant {value!r} in probe leakage audit")

        leakage = json.loads(leakage_json, parse_constant=reject_constant)
        if not isinstance(leakage, dict) or not isinstance(leakage.get("status"), str):
            raise ValueError("probe leakage audit must be a JSON object with a status")

    status = leakage["status"]
    supported_statuses = {
        "passed",
        "not_applicable",
        "label_revealed",
        "not_checked_missing_probe_template_metadata",
        "unverifiable_missing_probe_leakage_audit",
    }
    if status not in supported_statuses:
        raise ValueError(f"probe leakage audit has unsupported status {status!r}")
    if status == "label_revealed" and not allow_label_revealed:
        raise ValueError(
            "probe directions come from label-revealed prompts; use surface-only probes or "
            "--allow-label-revealed-probes for an explicit positive-control analysis"
        )
    if status in {
        "not_checked_missing_probe_template_metadata",
        "unverifiable_missing_probe_leakage_audit",
    } and not allow_unverifiable:
        raise ValueError(
            "probe direction prompt contract is unverifiable; re-run the probe audit or pass "
            "--allow-unverifiable-prompt-contract after external verification"
        )
    return {
        "activations_sha256": actual_sha,
        "recorded_activation_shape": list(recorded_shape),
        "prompt_leakage_audit": leakage,
    }


def weight_subspace_correlations(
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    n_components: int = 5,
) -> np.ndarray:
    """Principal-angle cosines between discriminative classifier row spaces."""
    weights_a = np.asarray(weights_a, dtype=np.float64)
    weights_b = np.asarray(weights_b, dtype=np.float64)
    if (
        weights_a.ndim != 2
        or weights_b.ndim != 2
        or weights_a.shape[0] == 0
        or weights_b.shape[0] == 0
        or weights_a.shape[1] != weights_b.shape[1]
        or weights_a.shape[1] == 0
    ):
        raise ValueError(
            f"probe weights must be rank-2 with equal hidden width, got {weights_a.shape} and {weights_b.shape}"
        )
    if not np.isfinite(weights_a).all() or not np.isfinite(weights_b).all():
        raise ValueError("probe weights contain non-finite values")
    if weights_a.shape[0] > 1:
        weights_a = weights_a - weights_a.mean(axis=0, keepdims=True)
    if weights_b.shape[0] > 1:
        weights_b = weights_b - weights_b.mean(axis=0, keepdims=True)
    _, singular_a, basis_a = np.linalg.svd(weights_a, full_matrices=False)
    _, singular_b, basis_b = np.linalg.svd(weights_b, full_matrices=False)
    tolerance_a = np.finfo(np.float64).eps * max(weights_a.shape) * singular_a[0]
    tolerance_b = np.finfo(np.float64).eps * max(weights_b.shape) * singular_b[0]
    rank_a = int(np.sum(singular_a > tolerance_a))
    rank_b = int(np.sum(singular_b > tolerance_b))
    if rank_a == 0 or rank_b == 0:
        raise ValueError("probe discriminative weight subspace is rank-zero")
    overlap = basis_a[:rank_a] @ basis_b[:rank_b].T
    correlations = np.linalg.svd(overlap, compute_uv=False)
    return np.clip(correlations[: min(n_components, len(correlations))], 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="cca analysis of layer representations"
    )
    parser.add_argument(
        "--activations", required=True,
        help="path to activations .npy or .npz"
    )
    parser.add_argument(
        "--activations-b", default=None,
        help="second model's activations for cross-model comparison"
    )
    parser.add_argument(
        "--probes", default=None,
        help="path to probe weights .npz for root/pattern similarity"
    )
    parser.add_argument(
        "--output", default="data/cca_results.npz",
        help="path to save results"
    )
    parser.add_argument(
        "--n-components", type=int, default=10,
        help="number of CCA components"
    )
    parser.add_argument("--reg", type=float, default=1e-4, help="CCA ridge regularization")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="deterministic held-out folds used for reported CCA similarities",
    )
    parser.add_argument(
        "--assume-row-aligned",
        action="store_true",
        help="permit cross-model analysis without verifiable row-identity metadata",
    )
    parser.add_argument(
        "--allow-label-revealed-probes",
        action="store_true",
        help="analyze label-revealed probe directions as an explicit positive control",
    )
    parser.add_argument(
        "--allow-unverifiable-prompt-contract",
        action="store_true",
        help="analyze legacy probe directions whose source prompt cannot be audited",
    )
    args = parser.parse_args()

    # load activations
    if args.n_components < 1:
        parser.error("--n-components must be at least 1")
    if not np.isfinite(args.reg) or args.reg < 0.0:
        parser.error("--reg must be finite and non-negative")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    acts_a = load_activations(args.activations)

    n_stimuli, n_layers, hidden_dim = acts_a.shape
    print(f"activations: {acts_a.shape}")
    print(f"  {n_stimuli} stimuli, {n_layers} layers, {hidden_dim} dim")

    results = {}

    # ── within-model CCA ───────────────────────────────────────
    print("\n--- within-model CCA ---")
    cca_a = cca_layer_matrix(
        acts_a,
        args.n_components,
        args.reg,
        cv_folds=args.cv_folds,
    )
    results["cca_layer_matrix"] = cca_a

    for i in range(min(n_layers, 5)):
        print(f"  layer {i:2d} self:  {cca_a[i, i]:.4f}")

    # ── cross-model CCA ────────────────────────────────────────
    if args.activations_b:
        print("\n--- cross-model CCA ---")
        acts_b = load_activations(args.activations_b)
        print(f"  model B: {acts_b.shape}")
        alignment_evidence = assert_row_alignment(
            args.activations,
            args.activations_b,
            acts_a.shape[0],
            allow_assumed=args.assume_row_aligned,
        )

        cca_cross = cca_cross_model(
            acts_a,
            acts_b,
            n_components=args.n_components,
            reg=args.reg,
            cv_folds=args.cv_folds,
        )
        results["cca_cross_model"] = cca_cross
        results["cross_model_alignment_evidence"] = np.array(alignment_evidence)

        # report best-matching layer pairs
        best_per_a = np.argmax(cca_cross, axis=1)
        for i in range(min(cca_cross.shape[0], 8)):
            j = best_per_a[i]
            print(f"  A layer {i:2d} ↔ B layer {j:2d}: {cca_cross[i, j]:.4f}")

    # ── probe weight similarity ────────────────────────────────
    if args.probes:
        print("\n--- probe weight similarity (Q3: disentanglement) ---")
        probe_contract = validate_probe_artifact_contract(
            args.probes,
            args.activations,
            tuple(acts_a.shape),
            allow_label_revealed=args.allow_label_revealed_probes,
            allow_unverifiable=args.allow_unverifiable_prompt_contract,
        )
        subspace_sim = probe_weight_similarity(
            args.probes,
            n_components=min(args.n_components, 5),
        )
        results["root_pattern_cca"] = subspace_sim
        results["probe_prompt_contract_json"] = np.array(
            json.dumps(probe_contract, ensure_ascii=False, sort_keys=True)
        )

        for i in range(len(subspace_sim)):
            print(f"  layer {i:2d}: subspace CCA={subspace_sim[i]:.4f}")

        # disentanglement signal: low subspace CCA → root and pattern
        # probes use orthogonal subspaces. if CCA drops around mid
        # layers where probe accuracy is high, the model is
        # disentangling rather than encoding a fused vector.
        min_layer = np.argmin(subspace_sim)
        print(f"  min subspace CCA at layer {min_layer}: {subspace_sim[min_layer]:.4f}")

    # ── save ───────────────────────────────────────────────────
    results["schema_version"] = np.array(3, dtype=np.int64)
    results["n_components"] = np.array(args.n_components, dtype=np.int64)
    results["regularization"] = np.array(args.reg, dtype=np.float64)
    results["cv_folds"] = np.array(args.cv_folds, dtype=np.int64)
    results["evaluation"] = np.array("deterministic_cross_validated_regularized_cca")
    results["negative_heldout_correlation_policy"] = np.array("clip_to_zero")
    results["activations_a_sha256"] = np.array(sha256_file(args.activations))
    results["activations_a_shape"] = np.asarray(acts_a.shape, dtype=np.int64)
    if args.activations_b:
        results["activations_b_sha256"] = np.array(sha256_file(args.activations_b))
        results["activations_b_shape"] = np.asarray(acts_b.shape, dtype=np.int64)
    if args.probes:
        results["probes_sha256"] = np.array(sha256_file(args.probes))
    atomic_savez(args.output, **results)
    print(f"\nsaved results to {args.output}")


if __name__ == "__main__":
    main()
