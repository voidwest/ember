#!/usr/bin/env python3
"""Static checks for the docs site."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def check_css(errors: list[str]) -> None:
    css = (DOCS / "style.css").read_text()
    stack: list[int] = []
    for index, char in enumerate(css):
        if char == "{":
            stack.append(index)
        elif char == "}":
            if not stack:
                errors.append(f"docs/style.css: unmatched }} at byte {index}")
                return
            stack.pop()
    if stack:
        errors.append(f"docs/style.css: unmatched {{ count {len(stack)}")


def local_target(value: str) -> Path | None:
    if value.startswith(("http://", "https://", "mailto:", "#")):
        return None
    if not value.startswith("/"):
        return None
    rel = value.split("#", 1)[0].split("?", 1)[0].lstrip("/")
    if not rel:
        return DOCS / "index.html"
    if rel.endswith("/"):
        return DOCS / rel / "index.html"
    return DOCS / rel


def check_html(errors: list[str]) -> None:
    for path in sorted(DOCS.rglob("*.html")):
        text = path.read_text()
        rel = path.relative_to(ROOT)
        for tag in ("html", "head", "body", "nav", "footer"):
            opens = len(re.findall(fr"<{tag}\b", text))
            closes = len(re.findall(fr"</{tag}>", text))
            if opens != closes:
                errors.append(f"{rel}: {tag} count {opens}/{closes}")
        if "<style>" in text:
            errors.append(f"{rel}: inline <style> block found")
        if re.search(r'\sstyle="', text):
            errors.append(f"{rel}: inline style attribute found")
        if "lang-toggle" in text:
            errors.append(f"{rel}: stale lang-toggle found")
        if "docs:nav start" not in text:
            errors.append(f"{rel}: missing generated nav marker")
        if "docs:footer start" not in text:
            errors.append(f"{rel}: missing generated footer marker")
        if ("<pre" in text) != ("highlight.min.js" in text):
            errors.append(f"{rel}: highlight/pre mismatch")

        for attr in ("href", "src"):
            for value in re.findall(fr'{attr}="([^"]+)"', text):
                target = local_target(value)
                if target is not None and not target.exists():
                    errors.append(f"{rel}: missing {attr} {value} -> {target.relative_to(ROOT)}")


def check_generator(errors: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_docs.py"), "--check"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        errors.append("scripts/build_docs.py --check failed:\n" + result.stdout.strip())


def main() -> int:
    errors: list[str] = []
    check_css(errors)
    check_html(errors)
    check_generator(errors)
    if errors:
        print("\n".join(errors))
        return 1
    print("docs checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
