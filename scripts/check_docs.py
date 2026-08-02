#!/usr/bin/env python3
"""Static checks for the docs site."""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class DocumentAudit(HTMLParser):
    """Collect structural fields without relying on quote/style-specific regexes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.opens: Counter[str] = Counter()
        self.closes: Counter[str] = Counter()
        self.anchors: list[str] = []
        self.references: list[tuple[str, str]] = []
        self.duplicate_attributes: list[tuple[str, str]] = []
        self.inline_style = False
        self.inline_style_blocks = 0
        self.has_pre = False
        self.has_highlight_script = False
        self.has_stale_lang_toggle = False

    def _start(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self.opens[tag] += 1
        names = [name.lower() for name, _ in attrs]
        self.duplicate_attributes.extend(
            (tag, name) for name, count in Counter(names).items() if count > 1
        )
        values = {name.lower(): value for name, value in attrs}
        identifier = values.get("id")
        if identifier:
            self.anchors.append(identifier)
        if tag == "a" and values.get("name"):
            self.anchors.append(values["name"] or "")
        for attribute in ("href", "src"):
            value = values.get(attribute)
            if value is not None:
                self.references.append((attribute, value))
        if "style" in values:
            self.inline_style = True
        if tag == "style":
            self.inline_style_blocks += 1
        if tag == "pre":
            self.has_pre = True
        if tag == "script" and "highlight.min.js" in (values.get("src") or ""):
            self.has_highlight_script = True
        classes = set((values.get("class") or "").split())
        if "lang-toggle" in classes:
            self.has_stale_lang_toggle = True

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs)
        self.closes[tag.lower()] += 1

    def handle_endtag(self, tag):
        self.closes[tag.lower()] += 1


@lru_cache(maxsize=None)
def parse_document(path: Path) -> DocumentAudit:
    parser = DocumentAudit()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def check_css(errors: list[str]) -> None:
    css = (DOCS / "style.css").read_text(encoding="utf-8")
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


def local_target(value: str, source: Path) -> tuple[Path, str] | None:
    parsed = urlsplit(value)
    if parsed.scheme.lower() in {"http", "https", "mailto", "tel", "data"} or parsed.netloc:
        return None
    if parsed.scheme:
        raise ValueError(f"unsafe or unsupported URI scheme in {value!r}")
    decoded_path = unquote(parsed.path)
    if decoded_path.startswith("/"):
        target = DOCS / decoded_path.lstrip("/")
    elif decoded_path:
        target = source.parent / decoded_path
    else:
        target = source
    if decoded_path.endswith("/") or not decoded_path:
        target = target / "index.html" if decoded_path else target
    resolved = target.resolve()
    try:
        resolved.relative_to(DOCS.resolve())
    except ValueError as error:
        raise ValueError(f"local target escapes docs root: {value!r}") from error
    if resolved.is_dir():
        resolved = resolved / "index.html"
    return resolved, unquote(parsed.fragment)


def document_ids(path: Path) -> tuple[set[str], list[str]]:
    anchors = parse_document(path).anchors
    duplicates = sorted(
        value for value, count in Counter(anchors).items() if count > 1
    )
    return set(anchors), duplicates


def check_html(errors: list[str]) -> None:
    for path in sorted(DOCS.rglob("*.html")):
        if path.is_symlink() or not path.is_file():
            errors.append(f"{path.relative_to(ROOT)}: HTML input is not a regular file")
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        try:
            audit = parse_document(path)
        except Exception as error:
            errors.append(f"{rel}: HTML parsing failed: {error}")
            continue
        managed_markers = (
            "docs:theme-script start",
            "docs:og-image start",
            "docs:nav start",
            "docs:footer start",
        )
        marker_presence = [marker in text for marker in managed_markers]
        managed = all(marker_presence)
        if any(marker_presence) and not managed:
            missing = [
                marker
                for marker, present in zip(managed_markers, marker_presence, strict=True)
                if not present
            ]
            errors.append(f"{rel}: partially managed page missing markers {missing}")
        _, duplicates = document_ids(path)
        if duplicates:
            errors.append(f"{rel}: duplicate id attributes: {duplicates}")
        for tag in ("html", "head", "body", "nav", "footer"):
            opens = audit.opens[tag]
            closes = audit.closes[tag]
            if opens != closes:
                errors.append(f"{rel}: {tag} count {opens}/{closes}")
        for tag in ("html", "head", "body"):
            if audit.opens[tag] != 1:
                errors.append(f"{rel}: expected exactly one {tag} element")
        if managed and (audit.opens["nav"] != 1 or audit.opens["footer"] != 1):
            errors.append(f"{rel}: managed page requires exactly one nav and footer")
        if audit.duplicate_attributes:
            errors.append(f"{rel}: duplicate attributes: {audit.duplicate_attributes}")
        if managed and audit.inline_style_blocks:
            errors.append(f"{rel}: inline <style> block found")
        if managed and audit.inline_style:
            errors.append(f"{rel}: inline style attribute found")
        if managed and audit.has_stale_lang_toggle:
            errors.append(f"{rel}: stale lang-toggle found")
        if managed and (audit.has_pre != audit.has_highlight_script):
            errors.append(f"{rel}: highlight/pre mismatch")

        for attr, value in audit.references:
            try:
                target_info = local_target(value, path)
            except ValueError as error:
                errors.append(f"{rel}: invalid {attr} {value!r}: {error}")
                continue
            if target_info is None:
                continue
            target, fragment = target_info
            if not target.is_file():
                try:
                    display = target.relative_to(ROOT)
                except ValueError:
                    display = target
                errors.append(f"{rel}: missing {attr} {value} -> {display}")
                continue
            if fragment and target.suffix.lower() in {".html", ".htm"}:
                target_ids, _ = document_ids(target)
                if fragment not in target_ids:
                    errors.append(
                        f"{rel}: missing fragment #{fragment} in {target.relative_to(ROOT)}"
                    )


def check_generator(errors: list[str]) -> None:
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_docs.py"), "--check"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        errors.append("scripts/build_docs.py --check timed out after 60 seconds")
        return
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
