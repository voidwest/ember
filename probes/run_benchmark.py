"""run a reproducible Ember benchmark manifest.

The manifest is intentionally simple JSON. Example:

{
  "name": "qwen3-root-pattern-smoke",
  "stimuli": "stimuli/nonce_root_pattern_surface.json",
  "out_dir": "data/benchmarks/qwen3_smoke",
  "tasks": ["root", "pattern"],
  "split_policy": {
    "root": "pattern-heldout",
    "pattern": "root-heldout"
  },
  "models": [
    {
      "label": "qwen3_0_6b",
      "kind": "ember",
      "arch": "qwen3",
      "model": "Qwen3-0.6B-Q8_0.gguf",
      "probe_limit": 5,
      "generate_tokens": 1
    }
  ]
}
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from .benchmark_summary import summarize_run
    from .train_linear_probe import atomic_write_text, metadata_path_for_activations
except ImportError:  # direct script execution
    from benchmark_summary import summarize_run
    from train_linear_probe import atomic_write_text, metadata_path_for_activations


PYTHON = sys.executable
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PROBE_KINDS = {"linear", "sgd", "mlp"}
PROBE_SOLVERS = {"lbfgs", "saga", "liblinear", "newton-cg", "newton-cholesky", "sag"}
SPLIT_POLICIES = {
    "combination",
    "combination-heldout",
    "pattern",
    "pattern-heldout",
    "random",
    "random-stratified",
    "root",
    "root-heldout",
    "root-pattern",
    "root-pattern-heldout",
    "stratified",
    "template",
    "template-heldout",
}
PROBE_POSITIONS = {"last", "root", "pattern", "prompt_mean"}
HF_POOLS = {"cls", "last", "mean", "target_mean", "target_first", "target_last"}
RSA_METRICS = {"correlation", "cosine", "euclidean"}
DEFAULT_TOKENIZERS = {
    "gpt2": "tokenizer-gpt2.json",
    "llama": "tokenizer.json",
    "qwen3": "tokenizer-qwen3.json",
    "gemma4": "tokenizer-gemma4.json",
}
HF_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")


def strict_json(path: str | Path):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_bool(value, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a JSON boolean")
    return value


def run(
    cmd: list[str],
    dry_run: bool,
    manifest: list[dict],
    checkpoint: Callable[[], None] | None = None,
) -> None:
    print(" ".join(cmd))
    record = {"cmd": cmd, "dry_run": dry_run, "status": "planned" if dry_run else "running"}
    manifest.append(record)
    if checkpoint is not None:
        checkpoint()
    if not dry_run:
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            record["status"] = "failed"
            if checkpoint is not None:
                checkpoint()
            raise
        record["status"] = "completed"
        if checkpoint is not None:
            checkpoint()


def reuse(
    cmd: list[str],
    manifest: list[dict],
    reason: str,
    checkpoint: Callable[[], None] | None = None,
) -> None:
    print(f"reusing existing artifact: {reason}")
    print(" ".join(cmd))
    manifest.append(
        {
            "cmd": cmd,
            "dry_run": False,
            "skipped": True,
            "status": "reused_verified",
            "reason": reason,
        }
    )
    if checkpoint is not None:
        checkpoint()


def sidecar_path(path: str, suffix: str) -> str:
    candidate = Path(path)
    if candidate.suffix != ".npy":
        raise ValueError(f"activation path must end in .npy: {path}")
    return str(candidate.with_name(f"{candidate.stem}{suffix}"))


def resolved_ember_tokenizer(model: dict) -> tuple[str, Path]:
    configured = model.get("tokenizer")
    path = Path(configured or DEFAULT_TOKENIZERS[model["arch"]])
    if path.is_file():
        return str(path), path
    if configured is None and model["arch"] == "llama":
        embedded_source = Path("tokenizer.json")
        if embedded_source.is_file():
            return "embedded:tokenizer.json", embedded_source
    raise FileNotFoundError(f"tokenizer does not exist: {path}")


def validate_finite_mmap(array: np.ndarray, *, rows_per_chunk: int = 256) -> None:
    for start in range(0, array.shape[0], rows_per_chunk):
        if not np.isfinite(array[start : start + rows_per_chunk]).all():
            raise ValueError("reused activation tensor contains non-finite values")


def ember_extract_cmd(
    model: dict, stimuli: str, out_dir: Path, config: dict
) -> tuple[list[str], str]:
    output = out_dir / f"{model['label']}_activations.npy"
    cmd = [
        "cargo",
        "run",
        "--release",
        "--",
        "--arch",
        model["arch"],
        "--model",
        model["model"],
        "--probe",
        "--probe-stimuli",
        stimuli,
        "--probe-output",
        str(output),
        "--probe-generate-tokens",
        str(model.get("generate_tokens", 1)),
    ]
    probe_template = model.get("probe_template", config.get("probe_template"))
    probe_position = model.get("probe_position", config.get("probe_position"))
    if probe_template:
        cmd.extend(["--probe-template", probe_template])
    if probe_position:
        cmd.extend(["--probe-position", probe_position])
    if model.get("tokenizer"):
        cmd.extend(["--tokenizer", model["tokenizer"]])
    if model.get("probe_limit") is not None:
        cmd.extend(["--probe-limit", str(model["probe_limit"])])
    # Benchmark artifacts are research inputs; pin the exact model bytes.
    cmd.append("--record-model-sha256")
    return cmd, str(output)


def hf_extract_cmd(model: dict, benchmark: str, out_dir: Path) -> tuple[list[str], str]:
    output = out_dir / f"{model['label']}_activations.npy"
    cmd = [
        PYTHON,
        "probes/extract_hf_encoder.py",
        "--model",
        model["model"],
        "--benchmark",
        benchmark,
        "--output",
        str(output),
        "--pool",
        model.get("pool", "target_mean"),
    ]
    if model.get("limit") is not None:
        cmd.extend(["--limit", str(model["limit"])])
    if model.get("device"):
        cmd.extend(["--device", model["device"]])
    if model.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if model.get("revision"):
        cmd.extend(["--revision", model["revision"]])
    return cmd, str(output)


def enabled(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, dict):
        return require_bool(value.get("enabled", default), f"{key}.enabled")
    return require_bool(value, key)


def split_policy_args(config: dict) -> list[str]:
    policy = config.get("split_policy") or {}
    if isinstance(policy, str):
        return ["--split-policy", policy]
    args: list[str] = []
    default_policy = policy.get("default") or policy.get("all")
    if default_policy:
        args.extend(["--split-policy", default_policy])
    if "root" in policy:
        args.extend(["--root-split", policy["root"]])
    if "pattern" in policy:
        args.extend(["--pattern-split", policy["pattern"]])
    if "template" in policy:
        args.extend(["--split-policy", policy["template"]])
    group_field = policy.get("group_field") or config.get("group_field")
    if group_field:
        args.extend(["--group-field", group_field])
    return args


def validate_split_policy(config: dict) -> None:
    policy = config.get("split_policy", {})
    if isinstance(policy, str):
        if policy not in SPLIT_POLICIES:
            raise ValueError(f"unsupported split_policy {policy!r}")
        return
    if not isinstance(policy, dict):
        raise ValueError("config.split_policy must be a string or object")
    unknown = set(policy) - {"default", "all", "root", "pattern", "template", "group_field"}
    if unknown:
        raise ValueError(f"unknown split_policy fields: {sorted(unknown)}")
    for field in ("default", "all", "root", "pattern", "template"):
        value = policy.get(field)
        if value is not None and (
            not isinstance(value, str) or value not in SPLIT_POLICIES
        ):
            raise ValueError(f"split_policy.{field} is unsupported: {value!r}")
    group_field = policy.get("group_field", config.get("group_field"))
    if group_field is not None and (
        not isinstance(group_field, str) or not SAFE_LABEL.fullmatch(group_field)
    ):
        raise ValueError("split policy group_field must be a safe dotted field name")


def fertility_config(config: dict, models: list[dict]) -> tuple[list[str], list[str], str | None]:
    fert = config.get("fertility")
    if not fert:
        return [], [], None
    fert_config = fert if isinstance(fert, dict) else {}
    output = fert_config.get("output")
    tokenizers = list(fert_config.get("tokenizers", []))
    labels = list(fert_config.get("labels", []))
    if not tokenizers:
        for model in models:
            tokenizer = model.get("tokenizer")
            if tokenizer:
                tokenizers.append(tokenizer)
                labels.append(model["label"])
    if tokenizers and not labels:
        labels = [Path(tokenizer).stem for tokenizer in tokenizers]
    if len(tokenizers) != len(labels):
        raise ValueError("fertility.tokenizers and fertility.labels must have the same length")
    return tokenizers, labels, output


def validate_config(config: dict, *, dry_run: bool) -> None:
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a JSON object")
    name = config.get("name")
    if not isinstance(name, str) or not SAFE_LABEL.fullmatch(name):
        raise ValueError("config.name must be a safe non-empty identifier")
    stimuli = config.get("stimuli")
    if not isinstance(stimuli, str) or not stimuli:
        raise ValueError("config.stimuli must be a non-empty path")
    if not Path(stimuli).is_file():
        raise FileNotFoundError(f"stimuli file does not exist: {stimuli}")
    for field in ("out_dir", "summary_output"):
        value = config.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"config.{field} must be a non-empty path string")
    tasks = config.get("tasks", ["root", "pattern"])
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
        or len(tasks) != len(set(tasks))
    ):
        raise ValueError("config.tasks must be a non-empty list of unique strings")
    if any(not SAFE_LABEL.fullmatch(task) for task in tasks):
        raise ValueError("config.tasks must contain safe dotted field names")
    task_keys = ["".join(c if c.isalnum() or c in "_-" else "_" for c in task) for task in tasks]
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("config.tasks produce colliding artifact keys")
    validate_split_policy(config)
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("config.models must be a non-empty array")
    labels = []
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"model {index} must be a JSON object")
        label = model.get("label")
        if not isinstance(label, str) or not SAFE_LABEL.fullmatch(label):
            raise ValueError(f"model {index} label must be a safe identifier")
        labels.append(label)
        kind = model.get("kind", "ember")
        if kind not in {"ember", "hf_encoder"}:
            raise ValueError(f"unknown model kind: {kind}")
        model_name = model.get("model")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(f"model {label!r} has no model path/name")
        if kind == "ember":
            if model.get("arch") not in {"gpt2", "llama", "qwen3", "gemma4"}:
                raise ValueError(f"model {label!r} requires a supported arch")
            if not dry_run and not Path(model_name).is_file():
                raise FileNotFoundError(f"model file does not exist: {model_name}")
            tokenizer = model.get("tokenizer")
            if tokenizer is not None and (
                not isinstance(tokenizer, str) or not tokenizer
            ):
                raise ValueError(f"model {label!r} tokenizer must be a non-empty path")
            if tokenizer and not Path(tokenizer).is_file():
                raise FileNotFoundError(f"model {label!r} tokenizer does not exist: {tokenizer}")
        else:
            pool = model.get("pool", "target_mean")
            if not isinstance(pool, str) or pool not in HF_POOLS:
                raise ValueError(f"model {label!r} has unsupported HF pool {pool!r}")
            for field in ("device", "revision"):
                value = model.get(field)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"models[{index}].{field} must be a non-empty string")
            if enabled(config, "reuse_activations", False):
                revision = model.get("revision")
                if revision is None or not HF_COMMIT.fullmatch(revision):
                    raise ValueError(
                        f"model {label!r} must use a 40-hex commit revision when "
                        "reuse_activations is enabled"
                    )
        for key in ("trust_remote_code",):
            if key in model:
                require_bool(model[key], f"models[{index}].{key}")
        for key in ("limit", "probe_limit", "generate_tokens"):
            if key in model and (
                isinstance(model[key], bool)
                or not isinstance(model[key], int)
                or model[key] < 1
            ):
                raise ValueError(f"models[{index}].{key} must be a positive integer")
        for field in ("probe_template", "probe_position"):
            value = model.get(field)
            if value is not None and (
                not isinstance(value, str) or not SAFE_LABEL.fullmatch(value)
            ):
                raise ValueError(f"models[{index}].{field} must be a safe identifier")
        model_probe_position = model.get("probe_position")
        if model_probe_position is not None and (
            not isinstance(model_probe_position, str)
            or model_probe_position not in PROBE_POSITIONS
        ):
            raise ValueError(f"models[{index}].probe_position is not supported")
    if len(labels) != len(set(labels)):
        raise ValueError("model labels must be unique")
    for key in (
        "control",
        "reuse_activations",
        "run_mdl",
        "run_cca",
        "run_rsa",
        "run_divergence",
        "run_plots",
        "dark_plots",
        "allow_label_revealed_prompts",
        "allow_unverifiable_prompt_contract",
        "allow_unlinked_correctness",
    ):
        if key in config:
            enabled(config, key, False)
    for key in ("control_repeats", "probe_max_iter", "max_rows"):
        if key in config and (
            isinstance(config[key], bool)
            or not isinstance(config[key], int)
            or config[key] < 1
        ):
            raise ValueError(f"config.{key} must be a positive integer")
    if "folds" in config and (
        isinstance(config["folds"], bool)
        or not isinstance(config["folds"], int)
        or config["folds"] < 2
    ):
        raise ValueError("config.folds must be an integer of at least 2")
    if "cca_folds" in config and (
        isinstance(config["cca_folds"], bool)
        or not isinstance(config["cca_folds"], int)
        or config["cca_folds"] < 2
    ):
        raise ValueError("config.cca_folds must be an integer of at least 2")
    if "cca_components" in config and (
        isinstance(config["cca_components"], bool)
        or not isinstance(config["cca_components"], int)
        or config["cca_components"] < 1
    ):
        raise ValueError("config.cca_components must be a positive integer")
    if "probe_n_jobs" in config and (
        isinstance(config["probe_n_jobs"], bool)
        or not isinstance(config["probe_n_jobs"], int)
        or config["probe_n_jobs"] == 0
    ):
        raise ValueError("config.probe_n_jobs must be a non-zero integer")
    probe_kind = config.get("probe_kind", "linear")
    if not isinstance(probe_kind, str) or probe_kind not in PROBE_KINDS:
        raise ValueError(f"config.probe_kind must be one of {sorted(PROBE_KINDS)}")
    solver = config.get("probe_solver", "lbfgs")
    if not isinstance(solver, str) or solver not in PROBE_SOLVERS:
        raise ValueError(f"config.probe_solver must be one of {sorted(PROBE_SOLVERS)}")
    fractions = config.get("mdl_fractions")
    if fractions is not None:
        if (
            not isinstance(fractions, list)
            or len(fractions) < 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
                or value > 1.0
                for value in fractions
            )
            or list(fractions) != sorted(set(fractions))
        ):
            raise ValueError("config.mdl_fractions must be unique, increasing values in (0, 1]")
    if "probe_tol" in config and (
        isinstance(config["probe_tol"], bool)
        or not isinstance(config["probe_tol"], (int, float))
        or not math.isfinite(config["probe_tol"])
        or config["probe_tol"] <= 0
    ):
        raise ValueError("config.probe_tol must be finite and positive")
    if "cca_reg" in config and (
        isinstance(config["cca_reg"], bool)
        or not isinstance(config["cca_reg"], (int, float))
        or not math.isfinite(config["cca_reg"])
        or config["cca_reg"] < 0
    ):
        raise ValueError("config.cca_reg must be finite and non-negative")
    rsa_metric = config.get("rsa_metric", "correlation")
    if not isinstance(rsa_metric, str) or rsa_metric not in RSA_METRICS:
        raise ValueError(f"config.rsa_metric must be one of {sorted(RSA_METRICS)}")
    for field in ("probe_template", "probe_position"):
        value = config.get(field)
        if value is not None and (
            not isinstance(value, str) or not SAFE_LABEL.fullmatch(value)
        ):
            raise ValueError(f"config.{field} must be a safe non-empty identifier")
    config_probe_position = config.get("probe_position")
    if config_probe_position is not None and (
        not isinstance(config_probe_position, str)
        or config_probe_position not in PROBE_POSITIONS
    ):
        raise ValueError("config.probe_position is not supported")

    fertility = config.get("fertility")
    if fertility is not None and not isinstance(fertility, (bool, dict)):
        raise ValueError("config.fertility must be a boolean or object")
    if isinstance(fertility, dict):
        unknown = set(fertility) - {"enabled", "output", "tokenizers", "labels"}
        if unknown:
            raise ValueError(f"unknown fertility fields: {sorted(unknown)}")
        if "enabled" in fertility:
            require_bool(fertility["enabled"], "fertility.enabled")
        for field in ("tokenizers", "labels"):
            value = fertility.get(field, [])
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item for item in value
            ):
                raise ValueError(f"fertility.{field} must be a list of non-empty strings")
        output = fertility.get("output")
        if output is not None and (not isinstance(output, str) or not output):
            raise ValueError("fertility.output must be a non-empty path")


def verify_reusable_activation(
    model: dict, stimuli: str, activations: str, config: dict
) -> str:
    activation_path = Path(activations)
    metadata_path = metadata_path_for_activations(activations)
    if not activation_path.is_file() or not metadata_path.is_file():
        raise ValueError("reuse requires both the activation file and its metadata sidecar")
    metadata = strict_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"activation metadata must be an object: {metadata_path}")
    recorded_activation_sha = metadata.get("activations_sha256")
    actual_activation_sha = sha256_file(activation_path)
    if recorded_activation_sha != actual_activation_sha:
        raise ValueError("reused activation file SHA-256 does not match its metadata")
    try:
        activation_array = np.load(activation_path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise ValueError(f"reused activation file is not a safe NPY tensor: {activation_path}") from error
    if (
        activation_array.ndim != 3
        or any(dimension <= 0 for dimension in activation_array.shape)
        or activation_array.dtype != np.dtype(np.float32)
    ):
        raise ValueError(
            f"reused activations must be a non-empty rank-3 float32 tensor, got "
            f"{activation_array.shape} {activation_array.dtype}"
        )
    validate_finite_mmap(activation_array)
    if metadata.get("activation_shape") != list(activation_array.shape):
        raise ValueError("reused activation shape does not match its metadata")
    expected_stimuli_sha = sha256_file(stimuli)
    recorded_stimuli_sha = metadata.get("stimuli_sha256", metadata.get("benchmark_sha256"))
    if recorded_stimuli_sha != expected_stimuli_sha:
        raise ValueError(f"reused activation stimuli hash mismatch: {activation_path}")

    kind = model.get("kind", "ember")
    if kind == "ember":
        tokenizer_identity, tokenizer_source = resolved_ember_tokenizer(model)
        expected = {
            "model_path": model["model"],
            "architecture": model["arch"],
            "tokenizer_path": tokenizer_identity,
            "probe_generate_tokens": model.get("generate_tokens", 1),
            "probe_limit": model.get("probe_limit"),
            "probe_template": model.get(
                "probe_template", config.get("probe_template", "en_surface_probe")
            ),
            "probe_position": model.get(
                "probe_position", config.get("probe_position", "last")
            ),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"reused activation metadata {key} mismatch: "
                    f"{metadata.get(key)!r} != {value!r}"
                )
        recorded_model_sha = metadata.get("model_sha256")
        if not isinstance(recorded_model_sha, str) or recorded_model_sha != sha256_file(model["model"]):
            raise ValueError("reused activation model SHA-256 mismatch")
        if metadata.get("tokenizer_sha256") != sha256_file(tokenizer_source):
            raise ValueError("reused activation tokenizer SHA-256 mismatch")
        source_rows = strict_json(stimuli)
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError("Ember reuse stimuli must be a non-empty JSON array")
        expected_rows = min(model.get("probe_limit", len(source_rows)), len(source_rows))
        if metadata.get("n_stimuli") != expected_rows:
            raise ValueError("reused Ember activation row limit mismatch")
        if metadata.get("row_indices") != list(range(expected_rows)):
            raise ValueError("reused Ember activation row order is not the expected source prefix")
    else:
        if metadata.get("model") != model["model"] or metadata.get("pool") != model.get(
            "pool", "target_mean"
        ):
            raise ValueError("reused HF activation model or pooling configuration mismatch")
        revision = model.get("revision")
        if not revision:
            raise ValueError(
                "safe HF activation reuse requires an explicit immutable model revision"
            )
        if metadata.get("requested_revision") != revision:
            raise ValueError("reused HF requested revision does not match metadata")
        if metadata.get("model_commit_hash") != revision:
            raise ValueError("reused HF resolved model commit does not match the pinned revision")
        tokenizer_commit = metadata.get("tokenizer_commit_hash")
        if tokenizer_commit is not None and tokenizer_commit != revision:
            raise ValueError(
                "reused HF resolved tokenizer commit does not match the pinned revision"
            )
        source_rows = strict_json(stimuli)
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError("HF reuse stimuli must be a non-empty JSON array")
        expected_rows = min(model.get("limit", len(source_rows)), len(source_rows))
        if metadata.get("n_rows") != expected_rows:
            raise ValueError("reused HF activation row limit mismatch")
        selections = metadata.get("token_selections")
        if not isinstance(selections, list) or len(selections) != expected_rows:
            raise ValueError("reused HF activation token selections are missing or incomplete")
        expected_ids = [str(row.get("id")) for row in source_rows[:expected_rows]]
        if any(
            not isinstance(selection, dict)
            or selection.get("index") != index
            or selection.get("row_id") != expected_ids[index]
            for index, selection in enumerate(selections)
        ):
            raise ValueError("reused HF activation row identities do not match the benchmark")
    return f"verified metadata and SHA-256 for {activation_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="run an Ember benchmark manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = strict_json(args.config)
    validate_config(config, dry_run=args.dry_run)
    out_dir = Path(config.get("out_dir", "data/benchmarks")) / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "benchmark_manifest.json"
    summary_path = Path(config.get("summary_output") or out_dir / "benchmark_summary.json")
    config_sha256 = sha256_file(args.config)
    created_at = datetime.now(timezone.utc).isoformat()
    stimuli = config["stimuli"]
    tasks = config.get("tasks", ["root", "pattern"])
    manifest: list[dict] = []
    model_artifacts: list[dict] = []
    plot_paths: list[str] = []
    fertility_path = None

    def checkpoint_manifest() -> None:
        atomic_write_text(
            manifest_path,
            json.dumps(
                {
                    "schema_version": 2,
                    "created_at": created_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "config_path": args.config,
                    "config_sha256": config_sha256,
                    "config": config,
                    "dry_run": args.dry_run,
                    "commands": manifest,
                    "model_artifacts": model_artifacts,
                    "fertility_path": fertility_path,
                    "plots": plot_paths,
                    "summary_path": str(summary_path),
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )

    checkpoint_manifest()

    for model in config["models"]:
        kind = model.get("kind", "ember")
        if kind == "ember":
            extract_cmd, activations = ember_extract_cmd(model, stimuli, out_dir, config)
        elif kind == "hf_encoder":
            extract_cmd, activations = hf_extract_cmd(model, stimuli, out_dir)
        else:
            raise ValueError(f"unknown model kind: {kind}")
        if (
            not args.dry_run
            and enabled(config, "reuse_activations", False)
            and Path(activations).exists()
        ):
            reuse(
                extract_cmd,
                manifest,
                verify_reusable_activation(model, stimuli, activations, config),
                checkpoint_manifest,
            )
        else:
            run(extract_cmd, args.dry_run, manifest, checkpoint_manifest)

        prefix = out_dir / model["label"]
        probes_path = f"{prefix}_probes.npz"
        mdl_path = f"{prefix}_mdl.npz"
        cca_path = f"{prefix}_cca.npz"
        rsa_path = f"{prefix}_rsa.npz"
        divergence_path = f"{prefix}_divergence.npz"
        probe_cmd = [
            PYTHON,
            "probes/train_linear_probe.py",
            "--activations",
            activations,
            "--stimuli",
            stimuli,
            "--tasks",
            *tasks,
            "--probe-kind",
            config.get("probe_kind", "linear"),
            "--output",
            probes_path,
            "--require-activation-provenance",
        ]
        if config.get("control", True):
            probe_cmd.append("--control")
        if config.get("folds") is not None:
            probe_cmd.extend(["--folds", str(config["folds"])])
        if config.get("control_repeats") is not None:
            probe_cmd.extend(["--control-repeats", str(config["control_repeats"])])
        if config.get("probe_max_iter") is not None:
            probe_cmd.extend(["--max-iter", str(config["probe_max_iter"])])
        if config.get("probe_solver") is not None:
            probe_cmd.extend(["--solver", str(config["probe_solver"])])
        if config.get("probe_tol") is not None:
            probe_cmd.extend(["--tol", str(config["probe_tol"])])
        if config.get("probe_n_jobs") is not None:
            probe_cmd.extend(["--n-jobs", str(config["probe_n_jobs"])])
        if config.get("allow_label_revealed_prompts", False):
            probe_cmd.append("--allow-label-revealed-prompts")
        if config.get("allow_unverifiable_prompt_contract", False):
            probe_cmd.append("--allow-unverifiable-prompt-contract")
        probe_cmd.extend(split_policy_args(config))
        if config.get("max_rows"):
            probe_cmd.extend(["--max-rows", str(config["max_rows"])])
        run(probe_cmd, args.dry_run, manifest, checkpoint_manifest)

        if enabled(config, "run_mdl", True):
            mdl_cmd = [
                PYTHON,
                "probes/mdl_probe.py",
                "--activations",
                activations,
                "--stimuli",
                stimuli,
                "--tasks",
                *tasks,
                "--probe-kind",
                config.get("probe_kind", "linear"),
                "--output",
                mdl_path,
                "--require-activation-provenance",
            ]
            if config.get("mdl_fractions"):
                mdl_cmd.extend(["--fractions", *[str(v) for v in config["mdl_fractions"]]])
            if config.get("max_rows"):
                mdl_cmd.extend(["--max-rows", str(config["max_rows"])])
            if config.get("folds") is not None:
                mdl_cmd.extend(["--folds", str(config["folds"])])
            if config.get("allow_label_revealed_prompts", False):
                mdl_cmd.append("--allow-label-revealed-prompts")
            if config.get("allow_unverifiable_prompt_contract", False):
                mdl_cmd.append("--allow-unverifiable-prompt-contract")
            mdl_cmd.extend(split_policy_args(config))
            run(mdl_cmd, args.dry_run, manifest, checkpoint_manifest)

        if enabled(config, "run_cca", True):
            cca_cmd = [
                PYTHON,
                "probes/cca_analysis.py",
                "--activations",
                activations,
                "--output",
                cca_path,
            ]
            if config.get("cca_components") is not None:
                cca_cmd.extend(["--n-components", str(config["cca_components"])])
            if config.get("cca_reg") is not None:
                cca_cmd.extend(["--reg", str(config["cca_reg"])])
            if config.get("cca_folds") is not None:
                cca_cmd.extend(["--cv-folds", str(config["cca_folds"])])
            if (
                config.get("probe_kind", "linear") == "linear"
                and {"root", "pattern"}.issubset(set(tasks))
            ):
                cca_cmd.extend(["--probes", probes_path])
                if config.get("allow_label_revealed_prompts", False):
                    cca_cmd.append("--allow-label-revealed-probes")
                if config.get("allow_unverifiable_prompt_contract", False):
                    cca_cmd.append("--allow-unverifiable-prompt-contract")
            run(cca_cmd, args.dry_run, manifest, checkpoint_manifest)

        if enabled(config, "run_rsa", True):
            rsa_cmd = [
                PYTHON,
                "probes/rsa_analysis.py",
                "--activations",
                activations,
                "--output",
                rsa_path,
            ]
            if config.get("rsa_metric") is not None:
                rsa_cmd.extend(["--metric", config["rsa_metric"]])
            run(rsa_cmd, args.dry_run, manifest, checkpoint_manifest)

        correctness_path = sidecar_path(activations, "_correctness.json")
        divergence_enabled = enabled(config, "run_divergence", True) and kind == "ember"
        if divergence_enabled:
            if not args.dry_run and not Path(correctness_path).is_file():
                raise FileNotFoundError(
                    f"divergence is enabled but correctness sidecar is missing: {correctness_path}"
                )
            divergence_cmd = [
                PYTHON,
                "probes/divergence_analysis.py",
                "--activations",
                activations,
                "--correctness",
                correctness_path,
                "--output",
                divergence_path,
            ]
            if config.get("allow_unlinked_correctness", False):
                divergence_cmd.append("--allow-unlinked-correctness")
            run(divergence_cmd, args.dry_run, manifest, checkpoint_manifest)

        if enabled(config, "run_plots", True):
            plot_dir = prefix.parent / f"{model['label']}_plots"
            plot_cmd = [
                PYTHON,
                "probes/plot_results.py",
                "--probes",
                probes_path,
                "--output",
                str(plot_dir),
                "--title",
                f"{config['name']} / {model['label']}",
            ]
            if enabled(config, "run_cca", True):
                plot_cmd.extend(["--cca", cca_path])
            if enabled(config, "run_rsa", True):
                plot_cmd.extend(["--rsa", rsa_path])
            if divergence_enabled:
                plot_cmd.extend(["--divergence", divergence_path])
            if config.get("dark_plots", True):
                plot_cmd.append("--dark")
            if config.get("allow_label_revealed_prompts", False):
                plot_cmd.append("--allow-label-revealed-inputs")
            if config.get("allow_unverifiable_prompt_contract", False):
                plot_cmd.append("--allow-unverifiable-prompt-contract")
            run(plot_cmd, args.dry_run, manifest, checkpoint_manifest)
            plot_paths.append(str(plot_dir / "probe_results.png"))

        model_artifacts.append(
            {
                "label": model["label"],
                "kind": kind,
                "activations": activations,
                "probes": probes_path,
                "mdl": mdl_path,
                "cca": cca_path,
                "rsa": rsa_path,
                "divergence": divergence_path,
                "enabled": {
                    "probe": True,
                    "mdl": enabled(config, "run_mdl", True),
                    "cca": enabled(config, "run_cca", True),
                    "rsa": enabled(config, "run_rsa", True),
                    "divergence": divergence_enabled,
                },
            }
        )
        checkpoint_manifest()

    tokenizers, labels, configured_fertility_output = fertility_config(config, config["models"])
    if enabled(config, "fertility", False) and not tokenizers:
        raise ValueError(
            "fertility is enabled but no tokenizer paths were configured or available from models"
        )
    if enabled(config, "fertility", False):
        fertility_path = configured_fertility_output or str(out_dir / "fertility.json")
        fertility_cmd = [
            PYTHON,
            "probes/tokenizer_fertility.py",
            "--stimuli",
            stimuli,
            "--tokenizers",
            *tokenizers,
            "--labels",
            *labels,
            "--output",
            fertility_path,
        ]
        run(fertility_cmd, args.dry_run, manifest, checkpoint_manifest)

    summary = summarize_run(
        config=config,
        dry_run=args.dry_run,
        commands=manifest,
        models=model_artifacts,
        fertility_path=fertility_path,
        plots=plot_paths,
        config_path=args.config,
        config_sha256=config_sha256,
    )
    checkpoint_manifest()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
