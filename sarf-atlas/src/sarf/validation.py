from __future__ import annotations

import subprocess
from pathlib import Path


def validate_run(run_dir: str, *, repo_root: str | Path = ".") -> int:
    command = ["cargo", "run", "--", "validate-run", run_dir]
    return subprocess.run(command, cwd=repo_root, check=False).returncode
