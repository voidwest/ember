"""run a reproducible Ember benchmark manifest.

The manifest is intentionally simple JSON. Example:

{
  "name": "qwen3-root-pattern-smoke",
  "stimuli": "stimuli/nonce_root_pattern.json",
  "out_dir": "data/benchmarks/qwen3_smoke",
  "tasks": ["root", "pattern"],
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
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str], dry_run: bool, manifest: list[dict]) -> None:
    print(" ".join(cmd))
    manifest.append({"cmd": cmd, "dry_run": dry_run})
    if not dry_run:
        subprocess.run(cmd, check=True)


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
        "python",
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

    for model in config["models"]:
        kind = model.get("kind", "ember")
        if kind == "ember":
            extract_cmd, activations = ember_extract_cmd(model, stimuli, out_dir)
        elif kind == "hf_encoder":
            extract_cmd, activations = hf_extract_cmd(model, stimuli, out_dir)
        else:
            raise ValueError(f"unknown model kind: {kind}")
        run(extract_cmd, args.dry_run, manifest)

        prefix = out_dir / model["label"]
        probe_cmd = [
            "python",
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
            f"{prefix}_probes.npz",
        ]
        if config.get("control", True):
            probe_cmd.append("--control")
        if config.get("group_field"):
            probe_cmd.extend(["--group-field", config["group_field"]])
        if config.get("max_rows"):
            probe_cmd.extend(["--max-rows", str(config["max_rows"])])
        run(probe_cmd, args.dry_run, manifest)

        mdl_cmd = [
            "python",
            "probes/mdl_probe.py",
            "--activations",
            activations,
            "--stimuli",
            stimuli,
            "--tasks",
            *tasks,
            "--output",
            f"{prefix}_mdl.npz",
        ]
        if config.get("max_rows"):
            mdl_cmd.extend(["--max-rows", str(config["max_rows"])])
        run(mdl_cmd, args.dry_run, manifest)

        run(
            [
                "python",
                "probes/rsa_analysis.py",
                "--activations",
                activations,
                "--output",
                f"{prefix}_rsa.npz",
            ],
            args.dry_run,
            manifest,
        )

    manifest_path = out_dir / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "dry_run": args.dry_run,
                "commands": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
