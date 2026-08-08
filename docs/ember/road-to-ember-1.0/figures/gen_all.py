#!/usr/bin/env python3
"""Regenerate every Road to Ember 1.0 diagram from declarative specs.

The canonical drawing source is ``diagram_specs.json``.  It is intentionally
made of ordinary SVG primitives (rect, line, path, text, etc.) rather than
opaque editor output, so coordinates, labels, and grouping remain reviewable.
This program emits both theme variants and renders the 2x PNG fallbacks.

Usage:
    python3 gen_all.py                 # SVG + PNG
    python3 gen_all.py --svg-only      # SVG only (no rsvg-convert needed)
    python3 gen_all.py --audit         # regenerate, then run the SVG audit
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

D = pathlib.Path(__file__).resolve().parent
SPEC = D / "diagram_specs.json"
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

# All palette changes live here; geometry and content are shared by themes.
LIGHT = {
    "#090b0e": "#f3efe5",
    "#0d1014": "#fbf8f1",
    "#12151a": "#ece6da",
    "#f3efe5": "#18191b",
    "#f8f4ea": "#111216",
    "#a6a6ad": "#66676b",
    "#7e7f87": "#77746e",
    "rgba(206,199,186,0.32)": "rgba(92,84,72,0.38)",
    "rgba(206,199,186,0.18)": "rgba(92,84,72,0.2)",
    "#9b8fd1": "#8275bc",
    "#b3a7e2": "#6659a1",
    "#e0645a": "#ad3d30",
}


def recolor(value: str, light: bool) -> str:
    if not light:
        return value
    for dark, pale in LIGHT.items():
        value = value.replace(dark, pale)
    return value


def element(record: dict, light: bool) -> ET.Element:
    attrs = {key: recolor(value, light) for key, value in record.get("attrs", {}).items()}
    out = ET.Element(f"{{{SVG_NS}}}{record['tag']}", attrs)
    if "text" in record:
        out.text = recolor(record["text"], light)
    for child in record.get("children", []):
        out.append(element(child, light))
    return out


def write_svg(name: str, spec: dict, light: bool) -> pathlib.Path:
    root = ET.Element(f"{{{SVG_NS}}}svg", spec["attrs"])
    for child in spec["children"]:
        root.append(element(child, light))
    ET.indent(root, space="  ")
    path = D / f"{name}{'_light' if light else ''}.svg"
    payload = ET.tostring(root, encoding="unicode", xml_declaration=False)
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + payload + "\n", encoding="utf-8")
    return path


def render_png(svg: pathlib.Path) -> pathlib.Path:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        raise RuntimeError("rsvg-convert is required for PNG fallbacks (or pass --svg-only)")
    png = svg.with_suffix(".png")
    subprocess.run([renderer, "-w", "2240", str(svg), "-o", str(png)], check=True)
    return png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-only", action="store_true", help="do not render PNG fallbacks")
    parser.add_argument("--audit", action="store_true", help="run audit_diagrams.py on all generated SVGs")
    args = parser.parse_args()

    source = json.loads(SPEC.read_text(encoding="utf-8"))
    if source.get("schema") != "ember.diagram-spec.v1":
        raise RuntimeError(f"unsupported diagram spec schema in {SPEC}")

    generated: list[pathlib.Path] = []
    for name, spec in source["diagrams"].items():
        for light in (False, True):
            svg = write_svg(name, spec, light)
            generated.append(svg)
            if not args.svg_only:
                render_png(svg)
        print(f"ok  {name} (dark + light)")

    if args.audit:
        audit = D / "audit_diagrams.py"
        result = subprocess.run([sys.executable, str(audit), *map(str, generated)])
        if result.returncode:
            return result.returncode
    print(f"generated {len(generated)} SVGs" + ("" if args.svg_only else f" and {len(generated)} PNGs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
