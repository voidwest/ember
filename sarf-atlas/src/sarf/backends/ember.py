from __future__ import annotations

import json
from pathlib import Path

from sarf.artifacts import BackendDescriptor, SarfArtifactManifest


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
