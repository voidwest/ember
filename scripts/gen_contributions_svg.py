#!/usr/bin/env python3
"""Regenerate the homepage GitHub contribution calendar SVG.

Fetches the last year of contributions through the GitHub GraphQL API (via the
`gh` CLI, or GITHUB_TOKEN when `gh` is unavailable) and writes a static SVG in
the format docs/homepage.js reads for its hover tooltips: one <rect> per day
with data-date, data-level and data-count attributes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "voidwest" / "github-contributions.svg"

QUERY = """
query($login: String!) {
  user(login: $login) {
    contributionsCollection {
      contributionCalendar {
        totalContributions
        weeks { contributionDays { date contributionCount contributionLevel } }
      }
    }
  }
}
"""

LEVELS = {
    "NONE": (0, "#ebedf0"),
    "FIRST_QUARTILE": (1, "#9be9a8"),
    "SECOND_QUARTILE": (2, "#40c463"),
    "THIRD_QUARTILE": (3, "#30a14e"),
    "FOURTH_QUARTILE": (4, "#216e39"),
}

LEFT, TOP, STEP, CELL = 28, 22, 12, 10
WIDTH, HEIGHT = 663, 112


def fetch_calendar(login: str) -> dict:
    if shutil.which("gh"):
        raw = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={QUERY}", "-f", f"login={login}"],
            check=True, capture_output=True, text=True,
        ).stdout
        payload = json.loads(raw)
    else:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise SystemExit("need the gh CLI or GITHUB_TOKEN to query GitHub")
        request = urllib.request.Request(
            "https://api.github.com/graphql",
            data=json.dumps({"query": QUERY, "variables": {"login": login}}).encode(),
            headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    if payload.get("errors"):
        raise SystemExit(f"GitHub GraphQL error: {payload['errors']}")
    return payload["data"]["user"]["contributionsCollection"]["contributionCalendar"]


def ordinal(day: int) -> str:
    suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def render(calendar: dict) -> str:
    weeks = calendar["weeks"]
    total = calendar["totalContributions"]

    labels = []
    last_month = None
    for index, week in enumerate(weeks):
        first = dt.date.fromisoformat(week["contributionDays"][0]["date"])
        x = LEFT + index * STEP
        # Label the first full week of each month, as GitHub does, if it fits.
        if first.month != last_month and x + 20 <= WIDTH:
            if index > 0 or first.day <= 7:
                labels.append(f'<text x="{x}" y="9">{first.strftime("%b")}</text>')
            last_month = first.month
    labels += [
        f'<text x="0" y="{TOP + 1 * STEP + 8}">Mon</text>',
        f'<text x="0" y="{TOP + 3 * STEP + 8}">Wed</text>',
        f'<text x="0" y="{TOP + 5 * STEP + 8}">Fri</text>',
    ]

    cells = []
    for index, week in enumerate(weeks):
        for day in week["contributionDays"]:
            date = dt.date.fromisoformat(day["date"])
            count = day["contributionCount"]
            level, fill = LEVELS[day["contributionLevel"]]
            weekday = (date.weekday() + 1) % 7  # Sunday-first rows
            when = f"{date.strftime('%B')} {ordinal(date.day)}"
            title = (
                f"No contributions on {when}." if count == 0
                else f"{count} contribution{'' if count == 1 else 's'} on {when}."
            )
            cells.append(
                f'<rect x="{LEFT + index * STEP}" y="{TOP + weekday * STEP}" width="{CELL}" '
                f'height="{CELL}" rx="2" fill="{fill}" data-date="{day["date"]}" '
                f'data-level="{level}" data-count="{count}"><title>{title}</title></rect>'
            )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">'
        '<title id="title">GitHub contribution calendar</title>'
        f'<desc id="desc">{total:,} contributions in the last year</desc>'
        '<g fill="#57606a" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
        f'font-size="9">{"".join(labels)}</g><g>{"".join(cells)}</g></svg>'
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login", default="voidwest")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    svg = render(fetch_calendar(args.login))
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
