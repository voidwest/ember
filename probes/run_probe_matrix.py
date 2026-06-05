"""run the Ember probe hardening matrix.

This is intentionally a thin orchestrator around the existing CLI and analysis
scripts. It keeps prompt-template, probe-position, and model-scale reruns
reproducible without hiding the underlying commands.
"""

import argparse
import concurrent.futures
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TEMPLATES = ["en_zero", "en_one", "ar_zero", "ar_one"]
DEFAULT_POSITIONS = ["last", "root", "pattern", "prompt_mean"]
SPLIT_CHOICES = [
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
]


def parse_model(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("models must be LABEL:PATH")
    label, path = value.split(":", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("models must be LABEL:PATH")
    return label, path


def run(cmd: list[str], dry_run: bool, manifest: list[dict]):
    print(" ".join(cmd))
    manifest.append({"cmd": cmd, "dry_run": dry_run})
    if not dry_run:
        subprocess.run(cmd, check=True)


def record(cmd: list[str], dry_run: bool, manifest: list[dict]):
    print(" ".join(cmd))
    manifest.append({"cmd": cmd, "dry_run": dry_run})


def run_recorded_group(commands: list[list[str]]):
    for cmd in commands:
        subprocess.run(cmd, check=True)


def run_recorded_groups(groups: list[list[list[str]]], dry_run: bool, jobs: int):
    if dry_run:
        return
    if jobs <= 1 or len(groups) <= 1:
        for group in groups:
            run_recorded_group(group)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(run_recorded_group, group) for group in groups]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main():
    parser = argparse.ArgumentParser(description="run probe template/position matrix")
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        metavar="LABEL:PATH",
        help="model label and GGUF path; may be repeated",
    )
    parser.add_argument("--arch", default="llama", choices=["gpt2", "llama", "qwen3", "gemma4"])
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--stimuli",
        default="stimuli/nonce_root_pattern.json",
        help="stimulus JSON path",
    )
    parser.add_argument("--out-dir", default="data/matrix")
    parser.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES)
    parser.add_argument("--positions", nargs="+", default=DEFAULT_POSITIONS)
    parser.add_argument("--generate-tokens", type=int, default=16)
    parser.add_argument("--probe-kind", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--control", action="store_true")
    parser.add_argument(
        "--split-policy",
        choices=SPLIT_CHOICES,
        default="random",
        help="split policy for non-root/non-pattern tasks passed to train_linear_probe.py",
    )
    parser.add_argument(
        "--root-split",
        choices=SPLIT_CHOICES,
        default="pattern",
        help="split policy for root probes; default holds out patterns",
    )
    parser.add_argument(
        "--pattern-split",
        choices=SPLIT_CHOICES,
        default="root",
        help="split policy for pattern probes; default holds out roots",
    )
    parser.add_argument(
        "--group-field",
        default=None,
        help="optional dotted field for grouped CV; overrides task-specific grouping",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel analysis bundles after each model extraction (default: 1)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for label, model_path in args.model:
        extract_cmd = [
            "cargo", "run", "--release", "--",
            "--arch", args.arch,
            "--model", model_path,
            "--probe",
            "--probe-stimuli", args.stimuli,
            "--probe-output-dir", str(out_dir),
            "--probe-output-prefix", label,
            "--probe-templates", ",".join(args.templates),
            "--probe-positions", ",".join(args.positions),
            "--probe-generate-tokens", str(args.generate_tokens),
        ]
        if args.tokenizer:
            extract_cmd.extend(["--tokenizer", args.tokenizer])
        run(extract_cmd, args.dry_run, manifest)

        analysis_groups: list[list[list[str]]] = []
        for template in args.templates:
            for position in args.positions:
                prefix = out_dir / f"{label}_{template}_{position}"
                activations = f"{prefix}_activations.npy"
                probes = f"{prefix}_{args.probe_kind}_probes.npz"
                cca = f"{prefix}_{args.probe_kind}_cca.npz"
                rsa = f"{prefix}_rsa.npz"
                divergence = f"{prefix}_divergence.npz"

                probe_cmd = [
                    "python", "probes/train_linear_probe.py",
                    "--activations", activations,
                    "--stimuli", args.stimuli,
                    "--probe-kind", args.probe_kind,
                    "--output", probes,
                    "--split-policy", args.split_policy,
                    "--root-split", args.root_split,
                    "--pattern-split", args.pattern_split,
                ]
                if args.group_field:
                    probe_cmd.extend(["--group-field", args.group_field])
                if args.control:
                    probe_cmd.append("--control")
                group = [probe_cmd]
                record(probe_cmd, args.dry_run, manifest)

                cca_cmd = [
                    "python", "probes/cca_analysis.py",
                    "--activations", activations,
                    "--output", cca,
                ]
                if args.probe_kind == "linear":
                    cca_cmd.extend(["--probes", probes])
                group.append(cca_cmd)
                record(cca_cmd, args.dry_run, manifest)

                rsa_cmd = [
                    "python",
                    "probes/rsa_analysis.py",
                    "--activations",
                    activations,
                    "--output",
                    rsa,
                ]
                group.append(rsa_cmd)
                record(rsa_cmd, args.dry_run, manifest)

                divergence_cmd = [
                    "python",
                    "probes/divergence_analysis.py",
                    "--activations",
                    activations,
                    "--correctness",
                    activations.replace(".npy", "_correctness.json"),
                    "--output",
                    divergence,
                ]
                group.append(divergence_cmd)
                record(divergence_cmd, args.dry_run, manifest)
                analysis_groups.append(group)

        run_recorded_groups(analysis_groups, args.dry_run, args.jobs)

    manifest_path = out_dir / "run_probe_matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dry_run": args.dry_run,
                "arch": args.arch,
                "stimuli": args.stimuli,
                "templates": args.templates,
                "positions": args.positions,
                "generate_tokens": args.generate_tokens,
                "probe_kind": args.probe_kind,
                "split_policy": args.split_policy,
                "root_split": args.root_split,
                "pattern_split": args.pattern_split,
                "group_field": args.group_field,
                "jobs": args.jobs,
                "commands": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
