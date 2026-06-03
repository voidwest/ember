#!/usr/bin/env python3
"""Build a Markdown benchmark table from Ember smoke summary JSON files."""

import argparse
import json
from pathlib import Path


def load_summary(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"unsupported summary JSON shape: {path}")


def iter_summaries(logs_dir):
    for path in sorted(logs_dir.glob("*summary.json")):
        for item in load_summary(path):
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["_summary_path"] = str(path)
            yield row


def dedupe(rows):
    seen = {}
    for row in rows:
        key = (
            row.get("label"),
            row.get("date"),
            row.get("command") or row.get("ember_command"),
        )
        seen[key] = row
    return sorted(seen.values(), key=lambda row: (row.get("date") or "", row.get("label") or ""))


def infer_quant(row):
    model = row.get("model") or ""
    upper = model.upper()
    for quant in ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "F16", "F32"]:
        if quant in upper:
            return quant
    return ""


def max_rss_gb(row):
    kb = row.get("max_rss_kb")
    if kb is None:
        return ""
    return f"{kb / 1024 / 1024:.2f}"


def fmt_number(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def notes(row):
    values = row.get("notes") or []
    if isinstance(values, str):
        return values
    return "; ".join(str(value) for value in values)


def markdown_table(rows):
    headers = [
        "label",
        "arch",
        "quant",
        "prompt tokens",
        "decode tokens",
        "prefill tok/s",
        "decode tok/s",
        "max RSS GB",
        "status",
        "notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [
            row.get("label") or "",
            row.get("arch") or "",
            row.get("quant") or infer_quant(row),
            fmt_number(row.get("prompt_token_count")),
            fmt_number(row.get("decode_token_count") or row.get("generated_token_count")),
            fmt_number(row.get("prefill_tps")),
            fmt_number(row.get("decode_tps")),
            max_rss_gb(row),
            row.get("status") or "",
            notes(row),
        ]
        escaped = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="summarize smoke JSON into a Markdown table")
    parser.add_argument("--logs", default="logs", help="directory containing smoke summary JSON files")
    parser.add_argument("--output", required=True, help="Markdown output path")
    args = parser.parse_args()

    logs_dir = Path(args.logs)
    rows = dedupe(iter_summaries(logs_dir)) if logs_dir.exists() else []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown_table(rows), encoding="utf-8")
    print(f"wrote {len(rows)} smoke rows to {output}")


if __name__ == "__main__":
    main()
