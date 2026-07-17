#!/usr/bin/env python3
"""Generate 1200×630 Open Graph preview images for the docs site.

The renderer uses Pillow directly so cards are deterministic, work offline,
and do not depend on a browser or remote font service.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "og"
WIDTH = 1200
HEIGHT = 630

# Keep these synchronized with docs/style.css.
BG = "#0d0f12"
SURFACE = "#171a1f"
TEXT = "#f2f0e9"
TEXT_MUTED = "#92979f"
BORDER = "#2b3037"
ACCENT = "#6f93ff"


@dataclass(frozen=True)
class OgCard:
    slug: str
    title: str
    subtitle: str
    eyebrow: str
    tags: tuple[str, ...]


CARDS = (
    OgCard(
        slug="voidwest",
        title="voidwest",
        subtitle="systems, ml, and the space where they overlap.",
        eyebrow="independent ml systems research",
        tags=("ml systems", "arabic nlp", "technical writing"),
    ),
    OgCard(
        slug="ember",
        title="ember",
        subtitle="hidden-state extraction, leakage-aware probing, and reproducible morphology experiments.",
        eyebrow="research infrastructure / active · 2026",
        tags=("rust", "gguf", "probing"),
    ),
    OgCard(
        slug="research-notes",
        title="research notes",
        subtitle="papers, experiments, and field notes on Arabic NLP and model internals.",
        eyebrow="writing archive / ongoing · 2026",
        tags=("arabic nlp", "tokenization", "model internals"),
    ),
    OgCard(
        slug="llama-probing-results",
        title="what LLaMA knows about Arabic morphology",
        subtitle="probing internal representations across model scales.",
        eyebrow="research note / representation analysis",
        tags=("llama", "morphology", "hidden states"),
    ),
    OgCard(
        slug="simd-qwen-gemma",
        title="simd kernels, qwen 3, and gemma 4",
        subtitle="narrow fast paths inside an inspectable cpu-first inference loop.",
        eyebrow="engineering record / ember",
        tags=("simd", "qwen", "gemma"),
    ),
)


FONT_CANDIDATES = {
    "display": (
        Path("/usr/share/fonts/TTF/Georgia.TTF"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"),
        Path("/Library/Fonts/Georgia.ttf"),
        Path("C:/Windows/Fonts/georgia.ttf"),
    ),
    "sans": (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/TTF/Arial.TTF"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ),
    "mono": (
        Path("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"),
        Path("/Library/Fonts/Courier New.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
    ),
}


def load_font(image_font, role: str, size: int):
    for path in FONT_CANDIDATES[role]:
        if not path.exists():
            continue
        try:
            return image_font.truetype(str(path), size=size)
        except OSError:
            continue
    try:
        return image_font.load_default(size=size)
    except TypeError:
        return image_font.load_default()


def text_width(draw, value: str, font) -> float:
    left, _top, right, _bottom = draw.textbbox((0, 0), value, font=font)
    return right - left


def wrap_text(draw, value: str, font, max_width: int) -> list[str]:
    words = value.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def multiline_height(draw, lines: list[str], font, spacing: int) -> int:
    box = draw.multiline_textbbox(
        (0, 0), "\n".join(lines), font=font, spacing=spacing
    )
    return box[3] - box[1]


def draw_grid(draw) -> None:
    for x in range(32, WIDTH - 31, 72):
        draw.line((x, 32, x, HEIGHT - 33), fill=SURFACE, width=1)
    for y in range(32, HEIGHT - 31, 72):
        draw.line((32, y, WIDTH - 33, y), fill=SURFACE, width=1)


def draw_card(image_module, image_draw, image_font, card: OgCard, index: int):
    image = image_module.new("RGB", (WIDTH, HEIGHT), BG)
    draw = image_draw.Draw(image)

    draw_grid(draw)
    draw.rectangle((32, 32, WIDTH - 33, HEIGHT - 33), outline=BORDER, width=1)
    draw.line((64, 110, WIDTH - 65, 110), fill=BORDER, width=1)
    draw.line((64, 520, WIDTH - 65, 520), fill=BORDER, width=1)
    draw.line((64, 138, 64, 488), fill=ACCENT, width=2)
    draw.rectangle((64, 60, 75, 71), fill=ACCENT)

    mono_17 = load_font(image_font, "mono", 17)
    mono_19 = load_font(image_font, "mono", 19)
    sans_28 = load_font(image_font, "sans", 28)

    draw.text((88, 56), "VOIDWEST / RESEARCH ARCHIVE", font=mono_17, fill=TEXT)
    right_label = "VOIDWEST.DEV / STATIC RECORD"
    draw.text(
        (WIDTH - 64 - text_width(draw, right_label, mono_17), 56),
        right_label,
        font=mono_17,
        fill=TEXT_MUTED,
    )

    eyebrow = card.eyebrow.upper()
    draw.text((88, 143), eyebrow, font=mono_17, fill=ACCENT)
    record = f"{index:02d} / {len(CARDS):02d}"
    draw.text(
        (WIDTH - 64 - text_width(draw, record, mono_17), 143),
        record,
        font=mono_17,
        fill=TEXT_MUTED,
    )

    title_size = 84
    title_lines: list[str] = []
    while title_size >= 58:
        title_font = load_font(image_font, "display", title_size)
        title_lines = wrap_text(draw, card.title, title_font, 990)
        title_height = multiline_height(draw, title_lines, title_font, 2)
        if len(title_lines) <= 2 and title_height <= 176:
            break
        title_size -= 2

    title_y = 198
    draw.multiline_text(
        (88, title_y),
        "\n".join(title_lines),
        font=title_font,
        fill=TEXT,
        spacing=2,
    )

    subtitle_y = title_y + title_height + 24
    subtitle_lines = wrap_text(draw, card.subtitle, sans_28, 940)
    subtitle_height = multiline_height(draw, subtitle_lines, sans_28, 8)
    if subtitle_y + subtitle_height > 492:
        raise ValueError(f"social-card copy overflows: {card.slug}")
    draw.multiline_text(
        (88, subtitle_y),
        "\n".join(subtitle_lines),
        font=sans_28,
        fill=TEXT_MUTED,
        spacing=8,
    )

    tag_x = 64
    for tag_index, tag in enumerate(card.tags, start=1):
        label = f"[{tag_index:02d}]  {tag.upper()}"
        draw.text((tag_x, 552), label, font=mono_19, fill=TEXT)
        tag_x += int(text_width(draw, label, mono_19)) + 28
        if tag_index != len(card.tags):
            draw.line((tag_x - 14, 549, tag_x - 14, 577), fill=BORDER, width=1)

    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow is required to generate Open Graph images:")
        print("  python3 -m pip install Pillow")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    for index, card in enumerate(CARDS, start=1):
        image = draw_card(Image, ImageDraw, ImageFont, card, index)
        out = args.output / f"{card.slug}.png"
        image.save(out, format="PNG", optimize=True, dpi=(96, 96))
        try:
            display_path = out.relative_to(ROOT)
        except ValueError:
            display_path = out
        print(display_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
