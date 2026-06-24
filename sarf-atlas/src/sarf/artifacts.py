from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


SARF_ARTIFACT_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    adapter: str
    version: str | None = None
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SarfArtifactManifest:
    run_id: str
    backend: BackendDescriptor
    prompts_path: str
    tokenization_path: str | None = None
    positions_path: str | None = None
    logits_path: str | None = None
    hidden_states_path: str | None = None
    report_path: str | None = None
    not_research_output: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SARF_ARTIFACT_CONTRACT_VERSION
        return payload


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    required = ["schema_version", "run_id", "backend", "prompts_path"]
    missing = [key for key in required if key not in manifest]
    backend = manifest.get("backend")
    if not isinstance(backend, dict):
        missing.append("backend.name")
    else:
        for key in ["name", "adapter"]:
            if key not in backend:
                missing.append(f"backend.{key}")
    return {
        "passed": not missing,
        "missing": missing,
        "schema_version": manifest.get("schema_version"),
        "not_research_output": manifest.get("not_research_output"),
    }
