#!/usr/bin/env python3
"""Build a Markdown benchmark table from Ember smoke summary JSON files."""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


def load_summary(path):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f, parse_constant=reject_constant)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "summaries" in data:
            if data.get("schema_version") != 2 or not isinstance(
                data["summaries"], list
            ):
                raise ValueError(f"unsupported smoke aggregate envelope: {path}")
            return data["summaries"]
        return [data]
    raise ValueError(f"unsupported summary JSON shape: {path}")


def iter_summaries(logs_dir, excluded_path=None):
    for path in sorted(logs_dir.glob("*summary.json")):
        if excluded_path is not None and path.resolve() == excluded_path:
            continue
        for item in load_summary(path):
            if not isinstance(item, dict):
                raise ValueError(f"summary item in {path} is not an object")
            validate_summary(item, path)
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
        if key in seen:
            prior = {name: value for name, value in seen[key].items() if name != "_summary_path"}
            current = {name: value for name, value in row.items() if name != "_summary_path"}
            if prior != current:
                raise ValueError(
                    "conflicting duplicate smoke summaries for "
                    f"{key!r}: {seen[key]['_summary_path']} and {row['_summary_path']}"
                )
        else:
            seen[key] = row
    return sorted(seen.values(), key=lambda row: (row.get("date") or "", row.get("label") or ""))


def parse_status_list(value):
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def validate_summary(row, path):
    for field in ("label", "arch", "status"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"smoke summary in {path} requires non-empty {field}")
    if row.get("pass_fail") not in {"pass", "fail", "skip"}:
        raise ValueError(f"smoke summary in {path} has invalid pass_fail")
    expected = {
        "smoke_pass": "pass",
        "smoke_pass_generation_warning": "pass",
        "smoke_fail": "fail",
        "smoke_skipped": "skip",
        "dry_run": "skip",
    }
    if row["status"] in expected and row["pass_fail"] != expected[row["status"]]:
        raise ValueError(f"smoke status/pass_fail disagree in {path}")
    values = row.get("notes")
    if values is not None and not isinstance(values, (str, list)):
        raise ValueError(f"smoke notes in {path} must be a string or array")


def filter_statuses(rows, include_statuses, exclude_statuses):
    filtered = []
    for row in rows:
        status = row.get("status") or ""
        if include_statuses and status not in include_statuses:
            continue
        if exclude_statuses and status in exclude_statuses:
            continue
        filtered.append(row)
    return filtered


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
    if isinstance(kb, bool) or not isinstance(kb, (int, float)) or not math.isfinite(kb) or kb < 0:
        raise ValueError("max_rss_kb must be finite and non-negative")
    return f"{kb / 1024 / 1024:.2f}"


def fmt_number(value, *, non_negative=True):
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid numeric smoke metric: {value!r}")
    if non_negative and value < 0:
        raise ValueError(f"smoke metrics must be non-negative: {value!r}")
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def notes(row):
    values = row.get("notes") or []
    if isinstance(values, str):
        return values
    if any(not isinstance(value, str) for value in values):
        raise ValueError("smoke notes arrays must contain only strings")
    return "; ".join(values)


def markdown_table(rows):
    headers = [
        "label",
        "arch",
        "quant",
        "prompt tokens",
        "decode evaluations",
        "prefill tok/s",
        "decode eval/s",
        "max RSS GB",
        "status",
        "notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines = [
        "Smoke results are structural checks, not generation-quality benchmarks.",
        "",
        *lines,
    ]
    for row in rows:
        decode_count = row.get("decode_evaluation_count")
        if decode_count is None:
            decode_count = row.get("decode_token_count")
        decode_rate = row.get("decode_evaluations_per_second")
        if decode_rate is None:
            decode_rate = row.get("decode_tps")
        cells = [
            row.get("label") or "",
            row.get("arch") or "",
            row.get("quant") or infer_quant(row),
            fmt_number(row.get("prompt_token_count")),
            fmt_number(decode_count),
            fmt_number(row.get("prefill_tps")),
            fmt_number(decode_rate),
            max_rss_gb(row),
            row.get("status") or "",
            notes(row),
        ]
        escaped = [
            str(cell)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("|", "\\|")
            .replace("\n", " ")
            for cell in cells
        ]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="summarize smoke JSON into a Markdown table")
    parser.add_argument("--logs", default="logs", help="directory containing smoke summary JSON files")
    parser.add_argument("--output", required=True, help="Markdown output path")
    parser.add_argument(
        "--status",
        default=None,
        help="comma-separated status allowlist, e.g. smoke_pass,smoke_pass_generation_warning",
    )
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument(
        "--exclude-status",
        default=None,
        help="comma-separated status denylist, e.g. dry_run,smoke_skipped",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs)
    if logs_dir.exists() and not logs_dir.is_dir():
        raise ValueError(f"--logs is not a directory: {logs_dir}")
    output = Path(args.output)
    rows = (
        dedupe(iter_summaries(logs_dir, output.resolve())) if logs_dir.exists() else []
    )
    include_statuses = parse_status_list(args.status)
    exclude_statuses = parse_status_list(args.exclude_status)
    overlap = include_statuses & exclude_statuses
    if overlap:
        raise ValueError(f"statuses cannot be both included and excluded: {sorted(overlap)}")
    rows = filter_statuses(
        rows,
        include_statuses,
        exclude_statuses,
    )
    if not rows and not args.allow_empty:
        raise ValueError("no smoke summary rows matched")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.tmp-"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(markdown_table(rows))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {len(rows)} smoke rows to {output}")


if __name__ == "__main__":
    main()
