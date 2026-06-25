from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from sarf.artifacts import BackendDescriptor, SarfArtifactManifest


def detect(*, check_validate_run: bool = True, timeout_seconds: float = 2.0) -> dict[str, Any]:
    env_path = os.environ.get("EMBER_BIN")
    if env_path:
        found = shutil.which(env_path)
        path = found or env_path if found or Path(env_path).is_file() else None
        source = "EMBER_BIN"
    else:
        path = shutil.which("ember")
        source = "PATH" if path else None
    validate_run = {
        "checked": False,
        "callable": False,
        "error": None,
    }

    if check_validate_run and path:
        validate_run["checked"] = True
        try:
            result = subprocess.run(
                [path, "validate-run", "--help"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
            )
            validate_run["callable"] = result.returncode == 0
            if result.returncode != 0:
                validate_run["error"] = f"exit status {result.returncode}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            validate_run["error"] = str(exc)

    available = bool(path)
    return {
        "backend": "Ember",
        "available": available,
        "binary": {
            "name": "ember",
            "env_var": "EMBER_BIN",
            "env_path": env_path,
            "path": path,
            "source": source,
            "missing": not available,
        },
        "missing_binaries": [] if available else ["ember"],
        "validate_run": validate_run,
        "capabilities": {
            "tokenization": available,
            "logits": available,
            "hidden_states": available,
            "validate_run": validate_run["callable"],
        },
        "caveats": [
            "Ember is optional; Sarf can import artifacts without Ember installed.",
            "Hidden-state extraction requires an emitting backend such as Ember, patched llama.cpp, HF/Transformers, or precomputed artifacts.",
        ],
    }


def import_ember_run(run_dir: str | Path, *, run_id: str | None = None) -> SarfArtifactManifest:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backend = manifest.get("backend", {})
    return SarfArtifactManifest(
        run_id=run_id or manifest.get("run_id") or run_dir.name,
        backend=BackendDescriptor(
            name=str(backend.get("name", "ember")),
            adapter="sarf.backends.ember",
            version=backend.get("version"),
            source_path=str(run_dir),
        ),
        prompts_path=str(run_dir / manifest["samples_path"]),
        tokenization_path=str(run_dir / manifest["tokenization_path"]),
        positions_path=str(run_dir / manifest["positions_path"]),
        logits_path=str(run_dir / manifest["logits_path"]) if manifest.get("logits_path") else None,
        report_path=str(run_dir / manifest["report_path"]),
        not_research_output=bool(
            manifest.get("not_research_output")
            or manifest.get("extraction_config", {}).get("run_metadata", {}).get("not_research_output", False)
        ),
    )
