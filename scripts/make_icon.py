#!/usr/bin/env python3
"""Draw Ember's app icon with the standard library only, then build an .icns.

A warm-black rounded square with a glowing ember at its centre -- the same
palette as the console (dark canvas, one orange accent). Reproducible: run it
and commit the result.

    python3 scripts/make_icon.py            # writes assets/macos/Ember.icns
"""
import math, os, struct, subprocess, sys, tempfile, zlib

SIZE = 1024
MARGIN = 100            # macOS icon grid: the body sits inside a 1024 canvas
RADIUS = 185
BODY_TOP, BODY_BOTTOM = (0x22, 0x1f, 0x1c), (0x0d, 0x0c, 0x0b)
ORANGE = (0xEF, 0x8C, 0x48)
HOT = (0xFF, 0xDD, 0xB0)
SPARKS = [(560, 350, 9), (455, 300, 6), (620, 275, 5), (500, 215, 4), (585, 190, 3)]


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def rounded_rect_distance(x, y):
    """Signed distance to the icon body (negative inside)."""
    half = SIZE / 2 - MARGIN
    qx = abs(x - SIZE / 2) - (half - RADIUS)
    qy = abs(y - SIZE / 2) - (half - RADIUS)
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return outside + inside - RADIUS


def pixel(x, y):
    alpha = clamp(0.5 - rounded_rect_distance(x, y))
    if alpha <= 0.0:
        return (0, 0, 0, 0)
    t = clamp((y - MARGIN) / (SIZE - 2 * MARGIN))
    color = mix(BODY_TOP, BODY_BOTTOM, t)
    cx, cy = SIZE / 2, 585
    r = math.hypot(x - cx, y - cy)
    glow = math.exp(-((r / 250.0) ** 2)) * 0.85
    color = mix(color, ORANGE, glow)
    core = math.exp(-((r / 72.0) ** 2))
    color = mix(color, HOT, core * 0.95)
    for sx, sy, sr in SPARKS:
        d = math.hypot(x - sx, y - sy)
        if d < sr + 2:
            color = mix(color, ORANGE, clamp(sr + 1 - d))
    return (int(color[0]), int(color[1]), int(color[2]), int(alpha * 255))


def write_png(path, rows):
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def main():
    out = os.path.join(os.path.dirname(__file__), "..", "assets", "macos", "Ember.icns")
    out = os.path.abspath(out)
    with tempfile.TemporaryDirectory() as tmp:
        master = os.path.join(tmp, "master.png")
        rows = []
        for y in range(SIZE):
            row = bytearray()
            for x in range(SIZE):
                row.extend(pixel(x + 0.5, y + 0.5))
            rows.append(row)
        write_png(master, rows)
        iconset = os.path.join(tmp, "Ember.iconset")
        os.makedirs(iconset)
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = size * scale
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                subprocess.run(["sips", "-z", str(px), str(px), master, "--out",
                                os.path.join(iconset, name)], check=True, capture_output=True)
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out], check=True)
        # Keep a PNG beside it so the artwork can be looked at.
        subprocess.run(["sips", "-z", "512", "512", master, "--out",
                        out.replace(".icns", ".png")], check=True, capture_output=True)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
