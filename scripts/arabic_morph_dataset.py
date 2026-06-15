#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_morph_dataset.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
