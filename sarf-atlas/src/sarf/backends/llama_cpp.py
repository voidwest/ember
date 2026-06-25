from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_BINARY_CHECKS = {
    "tokenize": ("LLAMA_TOKENIZE_BIN", ("llama-tokenize",)),
    "cli": ("LLAMA_CLI_BIN", ("llama-cli", "main")),
    "simple": ("LLAMA_SIMPLE_BIN", ("llama-simple", "simple")),
}


@dataclass(frozen=True)
class BinaryStatus:
    name: str
    env_var: str
    env_path: str | None
    path: str | None
    source: str | None
    missing: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_binary(name: str, env_var: str, path_names: tuple[str, ...]) -> BinaryStatus:
    env_path = os.environ.get(env_var)
    if env_path:
        found = shutil.which(env_path)
        if found or Path(env_path).is_file():
            return BinaryStatus(
                name=name,
                env_var=env_var,
                env_path=env_path,
                path=found or env_path,
                source=env_var,
                missing=False,
            )
        return BinaryStatus(name=name, env_var=env_var, env_path=env_path, path=None, source=env_var, missing=True)

    for path_name in path_names:
        found = shutil.which(path_name)
        if found:
            return BinaryStatus(name=name, env_var=env_var, env_path=None, path=found, source="PATH", missing=False)

    return BinaryStatus(name=name, env_var=env_var, env_path=None, path=None, source=None, missing=True)


def detect() -> dict[str, Any]:
    binaries = {
        name: _find_binary(name, env_var, path_names).to_dict()
        for name, (env_var, path_names) in _BINARY_CHECKS.items()
    }
    missing = [status["name"] for status in binaries.values() if status["missing"]]
    capabilities = {
        "tokenization": not binaries["tokenize"]["missing"],
        "logits": not (binaries["cli"]["missing"] and binaries["simple"]["missing"]),
        "hidden_states": False,
    }
    return {
        "backend": "llama.cpp",
        "available": any(not status["missing"] for status in binaries.values()),
        "binaries": binaries,
        "missing_binaries": missing,
        "capabilities": capabilities,
        "caveats": [
            "Default llama.cpp does not emit Sarf-compatible hidden-state artifacts.",
            "Hidden-state extraction requires an emitting backend such as Ember, patched llama.cpp, HF/Transformers, or precomputed artifacts.",
        ],
    }
