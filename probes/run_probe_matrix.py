"""run the Ember probe hardening matrix.

This is intentionally a thin orchestrator around the existing CLI and analysis
scripts. It keeps prompt-template, probe-position, and model-scale reruns
reproducible without hiding the underlying commands.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .train_linear_probe import (
        atomic_write_text,
        audit_label_revealing_prompt,
        load_rows,
    )
except ImportError:  # direct script execution
    from train_linear_probe import (
        atomic_write_text,
        audit_label_revealing_prompt,
        load_rows,
    )


DEFAULT_TEMPLATES = ["en_surface_probe", "ar_surface_probe"]
DEFAULT_POSITIONS = ["last", "prompt_mean"]
POSITION_CHOICES = {"last", "root", "pattern", "prompt_mean"}
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
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


def run(
    cmd: list[str],
    dry_run: bool,
    manifest: list[dict],
    checkpoint: Callable[[], None],
):
    print(" ".join(cmd))
    entry = {"cmd": cmd, "dry_run": dry_run, "status": "planned" if dry_run else "running"}
    manifest.append(entry)
    checkpoint()
    if not dry_run:
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            entry["status"] = "failed"
            checkpoint()
            raise
        entry["status"] = "completed"
        checkpoint()


def record(
    cmd: list[str],
    dry_run: bool,
    manifest: list[dict],
    checkpoint: Callable[[], None],
):
    print(" ".join(cmd))
    entry = {"cmd": cmd, "dry_run": dry_run, "status": "planned" if dry_run else "queued"}
    manifest.append(entry)
    checkpoint()
    return entry


def run_recorded_group(commands: list[dict], checkpoint: Callable[[], None]):
    for entry in commands:
        entry["status"] = "running"
        checkpoint()
        try:
            subprocess.run(entry["cmd"], check=True)
        except Exception:
            entry["status"] = "failed"
            checkpoint()
            raise
        entry["status"] = "completed"
        checkpoint()


def run_recorded_groups(
    groups: list[list[dict]],
    dry_run: bool,
    jobs: int,
    checkpoint: Callable[[], None],
):
    if dry_run:
        return
    if jobs <= 1 or len(groups) <= 1:
        for group in groups:
            run_recorded_group(group, checkpoint)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(run_recorded_group, group, checkpoint) for group in groups
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def sidecar_path(path: str, suffix: str) -> str:
    source = Path(path)
    if source.suffix != ".npy":
        raise ValueError(f"activation path must end in .npy: {path}")
    return str(source.with_name(f"{source.stem}{suffix}"))


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
        default="stimuli/nonce_root_pattern_surface.json",
        help="stimulus JSON path",
    )
    parser.add_argument("--out-dir", default="data/matrix")
    parser.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES)
    parser.add_argument("--positions", nargs="+", default=DEFAULT_POSITIONS)
    parser.add_argument("--generate-tokens", type=int, default=16)
    parser.add_argument("--probe-kind", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--control", action="store_true")
    parser.add_argument(
        "--allow-label-revealed-prompts",
        action="store_true",
        help="run an explicitly acknowledged positive-control matrix whose prompts expose targets",
    )
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
        parser.error("--jobs must be >= 1")
    if args.generate_tokens < 1:
        parser.error("--generate-tokens must be at least 1")
    if len(args.model) != len({label for label, _ in args.model}):
        parser.error("model labels must be unique")
    for label, model_path in args.model:
        if not SAFE_COMPONENT.fullmatch(label):
            parser.error(f"unsafe model label: {label!r}")
        if not args.dry_run and not Path(model_path).is_file():
            parser.error(f"model file does not exist: {model_path}")
    if not Path(args.stimuli).is_file():
        parser.error(f"stimuli file does not exist: {args.stimuli}")
    if args.tokenizer and not Path(args.tokenizer).is_file():
        parser.error(f"tokenizer file does not exist: {args.tokenizer}")
    if len(args.templates) != len(set(args.templates)) or any(
        not SAFE_COMPONENT.fullmatch(value) for value in args.templates
    ):
        parser.error("--templates must be unique safe identifiers")
    if len(args.positions) != len(set(args.positions)) or any(
        value not in POSITION_CHOICES for value in args.positions
    ):
        parser.error(f"--positions must be unique choices from {sorted(POSITION_CHOICES)}")
    prefixes = [
        f"{label}_{template}_{position}"
        for label, _ in args.model
        for template in args.templates
        for position in args.positions
    ]
    if len(prefixes) != len(set(prefixes)):
        parser.error("model/template/position names produce colliding output prefixes")

    stimulus_rows = load_rows(args.stimuli)
    leakage_audits = []
    for template in args.templates:
        for position in args.positions:
            try:
                audit = audit_label_revealing_prompt(
                    stimulus_rows,
                    ["root", "pattern"],
                    {"probe_template": template, "probe_position": position},
                )
            except ValueError as error:
                parser.error(str(error))
            leakage_audits.append(audit)
    leaking = [audit for audit in leakage_audits if audit["status"] == "label_revealed"]
    if leaking and not args.allow_label_revealed_prompts:
        parser.error(
            "selected matrix prompts/positions reveal root or pattern targets. "
            "Use label-free prompts and last/prompt_mean positions, or pass "
            "--allow-label-revealed-prompts only for a named positive-control run. "
            f"First leakage example: {leaking[0]['examples'][:1]}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    manifest_path = out_dir / "run_probe_matrix_manifest.json"
    digest = hashlib.sha256(Path(args.stimuli).read_bytes()).hexdigest()
    state = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
        "status": "running" if not args.dry_run else "planning",
        "dry_run": args.dry_run,
        "arch": args.arch,
        "models": [{"label": label, "path": path} for label, path in args.model],
        "tokenizer": args.tokenizer,
        "stimuli": args.stimuli,
        "stimuli_sha256": digest,
        "templates": args.templates,
        "positions": args.positions,
        "generate_tokens": args.generate_tokens,
        "probe_kind": args.probe_kind,
        "split_policy": args.split_policy,
        "root_split": args.root_split,
        "pattern_split": args.pattern_split,
        "group_field": args.group_field,
        "jobs": args.jobs,
        "label_revealed_prompts_allowed": args.allow_label_revealed_prompts,
        "prompt_leakage_audits": leakage_audits,
        "commands": manifest,
    }
    checkpoint_lock = threading.Lock()

    def checkpoint() -> None:
        with checkpoint_lock:
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_text(
                manifest_path,
                json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            )

    checkpoint()

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
            "--record-model-sha256",
        ]
        if args.tokenizer:
            extract_cmd.extend(["--tokenizer", args.tokenizer])
        run(extract_cmd, args.dry_run, manifest, checkpoint)

        analysis_groups: list[list[dict]] = []
        for template in args.templates:
            for position in args.positions:
                prefix = out_dir / f"{label}_{template}_{position}"
                activations = f"{prefix}_activations.npy"
                probes = f"{prefix}_{args.probe_kind}_probes.npz"
                cca = f"{prefix}_{args.probe_kind}_cca.npz"
                rsa = f"{prefix}_rsa.npz"
                divergence = f"{prefix}_divergence.npz"

                probe_cmd = [
                    sys.executable, "probes/train_linear_probe.py",
                    "--activations", activations,
                    "--stimuli", args.stimuli,
                    "--probe-kind", args.probe_kind,
                    "--output", probes,
                    "--split-policy", args.split_policy,
                    "--root-split", args.root_split,
                    "--pattern-split", args.pattern_split,
                    "--require-activation-provenance",
                ]
                if args.group_field:
                    probe_cmd.extend(["--group-field", args.group_field])
                if args.allow_label_revealed_prompts:
                    probe_cmd.append("--allow-label-revealed-prompts")
                if args.control:
                    probe_cmd.append("--control")
                group = [record(probe_cmd, args.dry_run, manifest, checkpoint)]

                cca_cmd = [
                    sys.executable, "probes/cca_analysis.py",
                    "--activations", activations,
                    "--output", cca,
                ]
                if args.probe_kind == "linear":
                    cca_cmd.extend(["--probes", probes])
                    if args.allow_label_revealed_prompts:
                        cca_cmd.append("--allow-label-revealed-probes")
                group.append(record(cca_cmd, args.dry_run, manifest, checkpoint))

                rsa_cmd = [
                    sys.executable,
                    "probes/rsa_analysis.py",
                    "--activations",
                    activations,
                    "--output",
                    rsa,
                ]
                group.append(record(rsa_cmd, args.dry_run, manifest, checkpoint))

                divergence_cmd = [
                    sys.executable,
                    "probes/divergence_analysis.py",
                    "--activations",
                    activations,
                    "--correctness",
                    sidecar_path(activations, "_correctness.json"),
                    "--output",
                    divergence,
                ]
                group.append(record(divergence_cmd, args.dry_run, manifest, checkpoint))
                analysis_groups.append(group)

        run_recorded_groups(
            analysis_groups, args.dry_run, args.jobs, checkpoint
        )

    state["status"] = "planned_dry_run" if args.dry_run else "completed"
    checkpoint()
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
