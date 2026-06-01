#!/usr/bin/env python3
"""Generate Open Graph preview images for docs pages.

Requires the optional Python Playwright package and installed browser binaries:
    python3 -m pip install playwright
    python3 -m playwright install chromium
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "og"
SIZE = {"width": 1200, "height": 630}


@dataclass(frozen=True)
class OgCard:
    slug: str
    title: str
    subtitle: str
    eyebrow: str
    tags: tuple[str, ...]


CARDS = [
    OgCard(
        slug="voidwest",
        title="voidwest",
        subtitle="systems, ml, and the space where they overlap.",
        eyebrow="portfolio / research notes",
        tags=("rust", "inference", "arabic nlp"),
    ),
    OgCard(
        slug="ember",
        title="ember",
        subtitle="a cpu-first llm inference engine in rust.",
        eyebrow="systems / inference",
        tags=("gguf", "q8_0", "probing"),
    ),
    OgCard(
        slug="research-notes",
        title="research notes",
        subtitle="papers, experiments, and notes on Arabic NLP and mechanistic interpretability.",
        eyebrow="notes / experiments",
        tags=("arabic nlp", "tokenization", "probing"),
    ),
    OgCard(
        slug="llama-probing-results",
        title="what LLaMA knows about Arabic morphology",
        subtitle="probing internal representations across model scales.",
        eyebrow="research note",
        tags=("LLaMA", "morphology", "hidden states"),
    ),
    OgCard(
        slug="simd-qwen-gemma",
        title="simd kernels, qwen 3, and gemma 4",
        subtitle="narrow fast paths inside the same cpu-first inference loop.",
        eyebrow="ember update",
        tags=("SIMD", "Qwen", "Gemma"),
    ),
]


CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@500;600;700&display=swap');

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #dbe4ee;
    --text-dim: #8b949e;
    --accent: #f78166;
    --accent2: #d2a8ff;
    --blue: #79c0ff;
    --green: #7ee787;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    width: 1200px;
    height: 630px;
    background:
        linear-gradient(135deg, rgba(247, 129, 102, 0.12), transparent 38%),
        linear-gradient(315deg, rgba(121, 192, 255, 0.13), transparent 42%),
        var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.card {
    width: 100%;
    height: 100%;
    padding: 72px 82px 64px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.brand {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: var(--text-dim);
    font-size: 28px;
}
.brand strong {
    color: var(--accent);
    font-size: 32px;
}
.eyebrow {
    color: var(--blue);
    font-size: 30px;
    font-weight: 650;
    margin-bottom: 24px;
}
h1 {
    margin: 0;
    max-width: 980px;
    color: var(--text);
    font-size: 76px;
    line-height: 1.04;
    letter-spacing: 0;
}
.subtitle {
    max-width: 900px;
    margin-top: 26px;
    color: var(--text-dim);
    font-size: 34px;
    line-height: 1.34;
}
.tags {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}
.tag {
    border: 1px solid var(--border);
    background: rgba(22, 27, 34, 0.84);
    color: var(--accent2);
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 24px;
}
"""


def card_html(card: OgCard) -> str:
    tags = "\n".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in card.tags)
    return f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <style>{CSS}</style>
</head>
<body>
    <main class="card">
        <div class="brand">
            <strong>voidwest</strong>
            <span>voidwest.dev</span>
        </div>
        <section>
            <div class="eyebrow">{html.escape(card.eyebrow)}</div>
            <h1>{html.escape(card.title)}</h1>
            <div class="subtitle">{html.escape(card.subtitle)}</div>
        </section>
        <div class="tags">
            {tags}
        </div>
    </main>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Install it to generate OG images:")
        print("  python3 -m pip install playwright")
        print("  python3 -m playwright install chromium")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=SIZE, device_scale_factor=1)
        for card in CARDS:
            page.set_content(card_html(card), wait_until="networkidle")
            out = args.output / f"{card.slug}.png"
            page.screenshot(path=out, full_page=False)
            print(out.relative_to(ROOT))
        browser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
