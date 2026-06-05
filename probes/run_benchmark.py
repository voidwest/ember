"""run a reproducible Ember benchmark manifest.

The manifest is intentionally simple JSON. Example:

{
  "name": "qwen3-root-pattern-smoke",
  "stimuli": "stimuli/nonce_root_pattern.json",
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
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark_summary import summarize_run


PYTHON = sys.executable


def run(cmd: list[str], dry_run: bool, manifest: list[dict]) -> None:
    print(" ".join(cmd))
    manifest.append({"cmd": cmd, "dry_run": dry_run})
    if not dry_run:
        subprocess.run(cmd, check=True)


def reuse(cmd: list[str], manifest: list[dict], reason: str) -> None:
    print(f"reusing existing artifact: {reason}")
    print(" ".join(cmd))
    manifest.append({"cmd": cmd, "dry_run": False, "skipped": True, "reason": reason})


def ember_extract_cmd(model: dict, stimuli: str, out_dir: Path) -> tuple[list[str], str]:
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
    if model.get("tokenizer"):
        cmd.extend(["--tokenizer", model["tokenizer"]])
    if model.get("probe_limit") is not None:
        cmd.extend(["--probe-limit", str(model["probe_limit"])])
    if model.get("record_model_sha256"):
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
    return cmd, str(output)


def enabled(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, dict):
        return bool(value.get("enabled", default))
    return bool(value)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="run an Ember benchmark manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(config.get("out_dir", "data/benchmarks")) / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stimuli = config["stimuli"]
    tasks = config.get("tasks", ["root", "pattern"])
    manifest: list[dict] = []
    model_artifacts: list[dict] = []
    plot_paths: list[str] = []

    for model in config["models"]:
        kind = model.get("kind", "ember")
        if kind == "ember":
            extract_cmd, activations = ember_extract_cmd(model, stimuli, out_dir)
        elif kind == "hf_encoder":
            extract_cmd, activations = hf_extract_cmd(model, stimuli, out_dir)
        else:
            raise ValueError(f"unknown model kind: {kind}")
        if (
            not args.dry_run
            and config.get("reuse_activations", False)
            and Path(activations).exists()
        ):
            reuse(extract_cmd, manifest, f"{activations} exists")
        else:
            run(extract_cmd, args.dry_run, manifest)

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
        probe_cmd.extend(split_policy_args(config))
        if config.get("max_rows"):
            probe_cmd.extend(["--max-rows", str(config["max_rows"])])
        run(probe_cmd, args.dry_run, manifest)

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
            ]
            if config.get("mdl_fractions"):
                mdl_cmd.extend(["--fractions", *[str(v) for v in config["mdl_fractions"]]])
            if config.get("max_rows"):
                mdl_cmd.extend(["--max-rows", str(config["max_rows"])])
            run(mdl_cmd, args.dry_run, manifest)

        if enabled(config, "run_cca", True):
            cca_cmd = [
                PYTHON,
                "probes/cca_analysis.py",
                "--activations",
                activations,
                "--output",
                cca_path,
            ]
            if (
                config.get("probe_kind", "linear") == "linear"
                and {"root", "pattern"}.issubset(set(tasks))
            ):
                cca_cmd.extend(["--probes", probes_path])
            run(cca_cmd, args.dry_run, manifest)

        if enabled(config, "run_rsa", True):
            run(
                [
                    PYTHON,
                    "probes/rsa_analysis.py",
                    "--activations",
                    activations,
                    "--output",
                    rsa_path,
                ],
                args.dry_run,
                manifest,
            )

        correctness_path = activations.replace(".npy", "_correctness.json")
        should_run_divergence = enabled(config, "run_divergence", True)
        if should_run_divergence and (args.dry_run or Path(correctness_path).exists()):
            run(
                [
                    PYTHON,
                    "probes/divergence_analysis.py",
                    "--activations",
                    activations,
                    "--correctness",
                    correctness_path,
                    "--output",
                    divergence_path,
                ],
                args.dry_run,
                manifest,
            )

        if enabled(config, "run_plots", True):
            plot_dir = prefix.parent / f"{model['label']}_plots"
            plot_cmd = [
                PYTHON,
                "probes/plot_results.py",
                "--probes",
                probes_path,
                "--cca",
                cca_path,
                "--rsa",
                rsa_path,
                "--output",
                str(plot_dir),
                "--title",
                f"{config['name']} / {model['label']}",
            ]
            if Path(divergence_path).exists() or args.dry_run:
                plot_cmd.extend(["--divergence", divergence_path])
            if config.get("dark_plots", True):
                plot_cmd.append("--dark")
            run(plot_cmd, args.dry_run, manifest)
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
            }
        )

    fertility_path = None
    tokenizers, labels, configured_fertility_output = fertility_config(config, config["models"])
    if enabled(config, "fertility", False) and tokenizers:
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
        run(fertility_cmd, args.dry_run, manifest)

    manifest_path = out_dir / "benchmark_manifest.json"
    summary_path = Path(config.get("summary_output") or out_dir / "benchmark_summary.json")
    summary = summarize_run(
        config=config,
        dry_run=args.dry_run,
        commands=manifest,
        models=model_artifacts,
        fertility_path=fertility_path,
        plots=plot_paths,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
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
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
