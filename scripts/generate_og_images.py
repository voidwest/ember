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

from voidwest_theme import DARK


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
:root {
    --bg: __BG__;
    --surface: __SURFACE__;
    --border: __BORDER__;
    --border-soft: __BORDER_SOFT__;
    --text: __TEXT__;
    --heading: __HEADING__;
    --muted: __MUTED__;
    --subtle: __SUBTLE__;
    --accent: __ACCENT__;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    width: 1200px;
    height: 630px;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    overflow: hidden;
}
.card {
    width: 100%;
    height: 100%;
    padding: 0 72px 58px;
    display: grid;
    grid-template-rows: 78px 1fr auto;
}
.brand {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    color: var(--subtle);
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 17px;
}
.brand strong {
    color: var(--text);
    font-size: 18px;
    font-weight: 500;
}
.brand strong::before {
    content: "";
    display: inline-block;
    width: 24px;
    height: 17px;
    margin-right: 12px;
    border: 1px solid var(--text);
    border-left: 6px solid var(--accent);
    vertical-align: -2px;
}
.content {
    align-self: center;
    min-height: 0;
    padding-bottom: 8px;
}
.eyebrow {
    margin-bottom: 20px;
    color: var(--accent);
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 19px;
    line-height: 1.4;
}
h1 {
    margin: 0;
    max-width: 1010px;
    color: var(--heading);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 78px;
    font-weight: 400;
    line-height: 0.98;
    letter-spacing: -2.5px;
}
.long-title h1 {
    max-width: 1060px;
    font-size: 62px;
    line-height: 1;
    letter-spacing: -1.8px;
}
.long-title .subtitle {
    margin-top: 20px;
    font-size: 25px;
}
.subtitle {
    max-width: 930px;
    margin-top: 24px;
    color: var(--muted);
    font-size: 27px;
    line-height: 1.42;
}
.tags {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    padding-top: 18px;
    border-top: 1px solid var(--border-soft);
}
.tag {
    color: var(--muted);
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 16px;
}
.tag::before {
    content: "#";
    color: var(--accent);
}
"""

for token, value in {
    "__BG__": DARK.bg,
    "__SURFACE__": DARK.surface,
    "__BORDER__": DARK.border,
    "__BORDER_SOFT__": DARK.border_soft,
    "__TEXT__": DARK.text,
    "__HEADING__": DARK.heading,
    "__MUTED__": DARK.muted,
    "__SUBTLE__": DARK.subtle,
    "__ACCENT__": DARK.accent,
}.items():
    CSS = CSS.replace(token, value)


def card_html(card: OgCard) -> str:
    tags = "\n".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in card.tags)
    body_class = "long-title" if len(card.title) > 34 else ""
    return f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <style>{CSS}</style>
</head>
<body class="{body_class}">
    <main class="card">
        <div class="brand">
            <strong>voidwest</strong>
            <span>voidwest.dev</span>
        </div>
        <section class="content">
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
