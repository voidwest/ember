#!/usr/bin/env python3
"""Audit series diagrams for text overflow, box escapes, and collisions.

Estimates mono text width as 0.62 * font-size * char count (SFMono/Consolas
advance is ~0.55-0.62em). Flags:
  - text running beyond the viewBox
  - text whose bbox escapes the rect that contains its center
  - text bboxes that intersect other text bboxes (title collisions etc.)
  - rects running beyond the viewBox
"""
import sys
import xml.etree.ElementTree as ET

NS = {"svg": "http://www.w3.org/2000/svg"}
MONO = 0.62


def fs(el):
    st = el.get("style") or ""
    size = None
    if "font-size" in st:
        try:
            size = float(st.split("font-size:")[1].split(";")[0].rstrip("px"))
        except (ValueError, IndexError):
            pass
    return size


def text_bbox(t, cls_props):
    x = float(t.get("x", 0))
    y = float(t.get("y", 0))
    props = {}
    for cls in (t.get("class") or "").split():
        props.update(cls_props.get(cls, {}))
    s = float(str(props.get("font-size", "14.0")).rstrip("px"))
    if t.get("font-size"):
        s = float(t.get("font-size").rstrip("px"))
    txt = (t.text or "").strip()
    w = MONO * s * len(txt)
    h = s * 1.15
    anchor = t.get("text-anchor") or props.get("text-anchor", "start")
    if anchor == "middle":
        x0 = x - w / 2
    elif anchor == "end":
        x0 = x - w
    else:
        x0 = x
    return (x0, y - h, x0 + w, y)


def cls_props_from_style(root):
    """Parse .class { prop: val; ... } rules from the SVG <style> block."""
    out = {}
    for style in root.iter("{http://www.w3.org/2000/svg}style"):
        text = style.text or ""
        for rule in text.split("}"):
            if "." not in rule or "{" not in rule:
                continue
            sel, _, body = rule.partition("{")
            cls = sel.strip().lstrip(".").split()[0]
            props = {}
            for decl in body.split(";"):
                if ":" in decl:
                    k, _, v = decl.partition(":")
                    props[k.strip()] = v.strip()
            out[cls] = props
    return out


def rect_bbox(r):
    x = float(r.get("x", 0)); y = float(r.get("y", 0))
    w = float(r.get("width", 0)); h = float(r.get("height", 0))
    return (x, y, x + w, y + h)


def contains(big, small, pad=1.0):
    return big[0] - pad <= small[0] and big[1] - pad <= small[1] and \
           big[2] + pad >= small[2] and big[3] + pad >= small[3]


def intersect(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def audit(path):
    tree = ET.parse(path)
    root = tree.getroot()
    W = float(root.get("width", "1120").rstrip("px"))
    H = float(root.get("height", "0").rstrip("px"))
    props = cls_props_from_style(root)
    texts = [(t, text_bbox(t, props), t.get("class", "")) for t in root.iter("{http://www.w3.org/2000/svg}text")]
    rects = [(r, rect_bbox(r)) for r in root.iter("{http://www.w3.org/2000/svg}rect")]
    issues = []
    for t, tb, cls in texts:
        if tb[0] < -0.5 or tb[1] < -0.5 or tb[2] > W + 0.5 or tb[3] > H + 0.5:
            issues.append(f"CANVAS-OVERFLOW text='{t.text}' cls='{cls}' bbox=({tb[0]:.0f},{tb[1]:.0f},{tb[2]:.0f},{tb[3]:.0f}) canvas={W}x{H}")
        cx, cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
        container = None
        for r, rb in rects:
            if rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]:
                container = rb
                break
        if container and not contains(container, tb, pad=2.0):
            issues.append(f"BOX-ESCAPE text='{t.text}' cls='{cls}' bbox=({tb[0]:.0f}..{tb[2]:.0f}) box=({container[0]:.0f}..{container[2]:.0f})")
    # text-text intersections (same y band, overlapping x)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ti, tbi, ci = texts[i]; tj, tbj, cj = texts[j]
            if tbi[0] <= tbj[0] <= tbi[2] or tbj[0] <= tbi[0] <= tbj[2]:
                if abs(tbi[1] - tbj[1]) < 2 or abs(tbi[3] - tbj[3]) < 2 or (tbi[1] < tbj[3] and tbj[1] < tbi[3]):
                    if intersect(tbi, tbj):
                        issues.append(f"TEXT-COLLIDE '{ti.text}' ({tbi[0]:.0f},{tbi[1]:.0f}) vs '{tj.text}' ({tbj[0]:.0f},{tbj[1]:.0f})")
    for r, rb in rects:
        if rb[0] < -0.5 or rb[1] < -0.5 or rb[2] > W + 0.5 or rb[3] > H + 0.5:
            issues.append(f"RECT-OVERFLOW rect=({rb[0]:.0f},{rb[1]:.0f},{rb[2]:.0f},{rb[3]:.0f})")
    return W, H, issues


issue_count = 0
for path in sys.argv[1:]:
    W, H, issues = audit(path)
    print(f"== {path} ({W:.0f}x{H:.0f}) ==")
    if issues:
        issue_count += len(issues)
        for i in issues:
            print("  " + i)
    else:
        print("  clean")

raise SystemExit(1 if issue_count else 0)
