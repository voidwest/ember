"""run the Ember probe hardening matrix.

This is intentionally a thin orchestrator around the existing CLI and analysis
scripts. It keeps prompt-template, probe-position, and model-scale reruns
reproducible without hiding the underlying commands.
"""

import argparse
import subprocess
from pathlib import Path


DEFAULT_TEMPLATES = ["en_zero", "en_one", "ar_zero", "ar_one"]
DEFAULT_POSITIONS = ["last", "root", "pattern", "prompt_mean"]


def parse_model(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("models must be LABEL:PATH")
    label, path = value.split(":", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("models must be LABEL:PATH")
    return label, path


def run(cmd: list[str], dry_run: bool):
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


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
    parser.add_argument("--arch", default="llama", choices=["gpt2", "llama"])
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, model_path in args.model:
        for template in args.templates:
            for position in args.positions:
                prefix = out_dir / f"{label}_{template}_{position}"
                activations = f"{prefix}_activations.npy"
                probes = f"{prefix}_{args.probe_kind}_probes.npz"
                cca = f"{prefix}_{args.probe_kind}_cca.npz"
                rsa = f"{prefix}_rsa.npz"
                divergence = f"{prefix}_divergence.npz"

                extract_cmd = [
                    "cargo", "run", "--release", "--",
                    "--arch", args.arch,
                    "--model", model_path,
                    "--probe",
                    "--probe-stimuli", args.stimuli,
                    "--probe-output", activations,
                    "--probe-template", template,
                    "--probe-position", position,
                    "--probe-generate-tokens", str(args.generate_tokens),
                ]
                if args.tokenizer:
                    extract_cmd.extend(["--tokenizer", args.tokenizer])
                run(extract_cmd, args.dry_run)

                probe_cmd = [
                    "python", "probes/train_linear_probe.py",
                    "--activations", activations,
                    "--stimuli", args.stimuli,
                    "--probe-kind", args.probe_kind,
                    "--output", probes,
                ]
                if args.control:
                    probe_cmd.append("--control")
                run(probe_cmd, args.dry_run)

                cca_cmd = [
                    "python", "probes/cca_analysis.py",
                    "--activations", activations,
                    "--output", cca,
                ]
                if args.probe_kind == "linear":
                    cca_cmd.extend(["--probes", probes])
                run(cca_cmd, args.dry_run)

                run([
                    "python", "probes/rsa_analysis.py",
                    "--activations", activations,
                    "--output", rsa,
                ], args.dry_run)
                run([
                    "python", "probes/divergence_analysis.py",
                    "--activations", activations,
                    "--correctness", activations.replace(".npy", "_correctness.json"),
                    "--output", divergence,
                ], args.dry_run)


if __name__ == "__main__":
    main()
