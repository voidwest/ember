#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from arabic_morph_dataset.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
