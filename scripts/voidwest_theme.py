"""Shared visual tokens for generated voidwest assets.

The website stylesheet is the source of truth.  This module reads its custom
properties and adapts them for Matplotlib and the Open Graph card renderer so
generated assets cannot quietly drift back to a separate palette.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STYLESHEET = ROOT / "docs" / "style.css"


def _variables(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([\w-]+)\s*:\s*([^;]+);", block))


def _theme_blocks() -> tuple[dict[str, str], dict[str, str]]:
    css = STYLESHEET.read_text(encoding="utf-8")
    dark_match = re.search(r":root\s*\{(.*?)\}", css, re.DOTALL)
    light_match = re.search(r':root\[data-theme="light"\]\s*\{(.*?)\}', css, re.DOTALL)
    if not dark_match or not light_match:
        raise RuntimeError(f"could not read theme tokens from {STYLESHEET}")
    dark = _variables(dark_match.group(1))
    light = {**dark, **_variables(light_match.group(1))}
    return dark, light


@dataclass(frozen=True)
class Theme:
    bg: str
    surface: str
    surface_2: str
    text: str
    heading: str
    muted: str
    subtle: str
    border: str
    border_soft: str
    accent: str
    accent_strong: str
    code_bg: str


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _composite(css_color: str, background: str) -> str:
    """Convert the stylesheet's modern rgb(... / alpha) form to opaque hex."""
    match = re.fullmatch(r"rgb\((\d+)\s+(\d+)\s+(\d+)\s*/\s*([\d.]+)\)", css_color)
    if not match:
        return css_color
    foreground = tuple(int(match.group(i)) for i in range(1, 4))
    alpha = float(match.group(4))
    backdrop = _hex_rgb(background)
    mixed = tuple(round(alpha * fg + (1 - alpha) * bg) for fg, bg in zip(foreground, backdrop))
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def _make_theme(values: dict[str, str]) -> Theme:
    bg = values["bg"]
    return Theme(
        bg=bg,
        surface=values["surface"],
        surface_2=values["surface-2"],
        text=values["text"],
        heading=values["heading"],
        muted=values["muted"],
        subtle=values["subtle"],
        border=_composite(values["border"], bg),
        border_soft=_composite(values["border-soft"], bg),
        accent=values["accent"],
        accent_strong=values["accent-strong"],
        code_bg=values["code-bg"],
    )


_DARK_VALUES, _LIGHT_VALUES = _theme_blocks()
DARK = _make_theme(_DARK_VALUES)
LIGHT = _make_theme(_LIGHT_VALUES)

# A muted categorical extension of the violet site accent.  These are kept
# here—not in individual charts—so data series remain consistent everywhere.
BLUE = "#849fc4"
GREEN = "#83a995"
RED = "#c18484"
YELLOW = "#bca36f"
PURPLE = DARK.accent_strong
DARK_CYCLE = [PURPLE, BLUE, GREEN, YELLOW, RED, DARK.text, DARK.muted]
LIGHT_CYCLE = [LIGHT.accent_strong, "#526f96", "#527b68", "#8a6d35", "#925f5f", LIGHT.text, LIGHT.muted]


def matplotlib_style(*, dark: bool = True, dpi: int = 160) -> dict[str, object]:
    """Return Matplotlib rcParams matching the site's editorial system."""
    theme = DARK if dark else LIGHT
    cycle = DARK_CYCLE if dark else LIGHT_CYCLE
    return {
        "figure.facecolor": theme.bg,
        "axes.facecolor": theme.surface,
        "axes.edgecolor": theme.border,
        "axes.labelcolor": theme.muted,
        "axes.titlecolor": theme.heading,
        "axes.prop_cycle": __import__("matplotlib").cycler(color=cycle),
        "text.color": theme.text,
        "xtick.color": theme.muted if dark else theme.subtle,
        "ytick.color": theme.muted if dark else theme.subtle,
        "grid.color": theme.border_soft,
        "grid.alpha": 0.8,
        "grid.linewidth": 0.65,
        "legend.facecolor": theme.surface,
        "legend.edgecolor": theme.border_soft,
        "legend.labelcolor": theme.text,
        "legend.framealpha": 0.94,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.facecolor": theme.bg,
        "savefig.edgecolor": theme.bg,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    }


def apply_matplotlib_theme(*, dark: bool = True, dpi: int = 160) -> Theme:
    import matplotlib

    matplotlib.rcParams.update(matplotlib_style(dark=dark, dpi=dpi))
    return DARK if dark else LIGHT


def finish_axes(ax, *, dark: bool = True) -> None:
    """Apply the shared hairline frame and restrained grid to an axes."""
    theme = DARK if dark else LIGHT
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(theme.border_soft)
        spine.set_linewidth(0.7)


def sequential_cmap(*, dark: bool = True):
    from matplotlib.colors import LinearSegmentedColormap

    colors = (
        ["#111019", "#29233d", "#514575", "#8172b5", "#b8abe2", "#eee9fb"]
        if dark
        else ["#f5f0e7", "#ddd4ed", "#b9abd9", "#8b7ac0", "#62539a", "#302754"]
    )
    return LinearSegmentedColormap.from_list("voidwest_sequential", colors)


def diverging_cmap(*, dark: bool = True):
    from matplotlib.colors import LinearSegmentedColormap

    center = DARK.surface_2 if dark else LIGHT.surface
    return LinearSegmentedColormap.from_list(
        "voidwest_diverging",
        ["#6488b5", "#9bb3cf", center, "#c5bae3", "#7766b0"],
    )


def similarity_norm():
    """Keep a 0–1 scale while revealing differences among high similarities."""
    from matplotlib.colors import PowerNorm

    return PowerNorm(gamma=3.0, vmin=0.0, vmax=1.0)


def categorical_cmap(*, dark: bool = True):
    from matplotlib.colors import ListedColormap

    return ListedColormap(DARK_CYCLE if dark else LIGHT_CYCLE, name="voidwest_categorical")
