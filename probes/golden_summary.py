#!/usr/bin/env python3
"""Summarize golden-logit reports into compact JSON and Markdown outputs.

The normalizer is intentionally conservative: it copies explicit status fields
from source reports and never infers pass/fail from metrics.
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Any


DEFAULT_GLOB = "data/golden/*golden_report.json"
DEFAULT_OUTPUT_JSON = "data/golden/golden_summary.json"
DEFAULT_OUTPUT_MD = "data/golden/golden_summary.md"


FIELD_CANDIDATES = {
    "label": ["label", "model_label", "run_label", "name"],
    "model": ["model_name", "model_path", "model.path", "metadata.model", "metadata.model_name", "model"],
    "classification": ["classification", "golden_classification"],
    "status": ["status", "classification", "golden_classification", "pass_fail"],
    "max_abs_diff": ["max_abs_diff", "max_absolute_difference", "metrics.max_abs_diff"],
    "mean_abs_diff": ["mean_abs_diff", "mean_absolute_difference", "metrics.mean_abs_diff"],
    "top1_agreement": [
        "top_1_match",
        "top1_agreement",
        "top1_match",
        "top_token_matches",
        "top_token_match",
        "metrics.top_1_match",
    ],
    "top_k_overlap": [
        "top_k_overlap",
        "top_k_overlap_ratio",
        "topk_overlap",
        "topk_overlap_ratio",
        "metrics.top_k_overlap",
    ],
    "tokenizer": [
        "tokenizer_path",
        "tokenizer.path",
        "metadata.tokenizer_path",
        "metadata.tokenizer",
        "tokenizer",
    ],
    "model_sha256": [
        "model_sha256",
        "model_sha",
        "model_hash",
        "metadata.model_sha256",
        "metadata.model_hash",
    ],
    "reference": ["reference_path", "reference_logits", "reference.path", "reference"],
    "reference_source": [
        "reference_source",
        "reference_implementation",
        "reference.source",
        "metadata.reference_source",
        "metadata.reference_implementation",
    ],
    "ember": ["ember_path", "ember_logits", "ember.path", "ember"],
}


def _nested_get(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _first_present(data: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        value = _nested_get(data, key)
        if value is not None:
            return value
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
        return text if text else None
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "match", "matched"}:
            return True
        if lowered in {"false", "no", "n", "0", "mismatch", "different"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return None


def normalize_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Normalize a golden-logit report without inferring validation status."""

    row = {
        "report_path": str(path),
        "label": _string_or_none(_first_present(report, FIELD_CANDIDATES["label"])),
        "model": _string_or_none(_first_present(report, FIELD_CANDIDATES["model"])),
        "classification": _string_or_none(
            _first_present(report, FIELD_CANDIDATES["classification"])
        ),
        "status": _string_or_none(_first_present(report, FIELD_CANDIDATES["status"])),
        "max_abs_diff": _float_or_none(_first_present(report, FIELD_CANDIDATES["max_abs_diff"])),
        "mean_abs_diff": _float_or_none(_first_present(report, FIELD_CANDIDATES["mean_abs_diff"])),
        "top1_agreement": _bool_or_none(
            _first_present(report, FIELD_CANDIDATES["top1_agreement"])
        ),
        "top_k_overlap": _float_or_none(_first_present(report, FIELD_CANDIDATES["top_k_overlap"])),
        "tokenizer": _string_or_none(_first_present(report, FIELD_CANDIDATES["tokenizer"])),
        "model_sha256": _string_or_none(
            _first_present(report, FIELD_CANDIDATES["model_sha256"])
        ),
        "reference": _string_or_none(_first_present(report, FIELD_CANDIDATES["reference"])),
        "reference_source": _string_or_none(
            _first_present(report, FIELD_CANDIDATES["reference_source"])
        ),
        "ember": _string_or_none(_first_present(report, FIELD_CANDIDATES["ember"])),
    }
    row["missing_fields"] = [
        key
        for key, value in row.items()
        if key not in {"report_path", "missing_fields"} and value is None
    ]
    return row


def load_reports(pattern: str) -> list[dict[str, Any]]:
    rows = []
    for path_text in sorted(glob.glob(pattern)):
        path = Path(path_text)
        report = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(report, dict):
            rows.append(normalize_report(path, report))
            continue
        row = normalize_report(path, {})
        row["schema_warning"] = f"expected JSON object, got {type(report).__name__}"
        rows.append(row)
    return rows


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _md_escape(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _short_sha(value: str | None) -> str:
    if value is None:
        return "missing"
    return value[:12] if len(value) > 12 else value


def markdown_summary(summary: dict[str, Any]) -> str:
    rows = summary["reports"]
    lines = [
        "# Golden Logit Summary",
        "",
        f"- Glob: `{summary['glob']}`",
        f"- Reports: {summary['report_count']}",
        "- Classification/status values are copied from source reports only.",
        "",
    ]
    if not rows:
        lines.append("No golden-logit reports matched.")
        return "\n".join(lines) + "\n"

    headers = [
        "report",
        "label",
        "model",
        "status",
        "max abs diff",
        "mean abs diff",
        "top-1 match",
        "top-k overlap",
        "tokenizer",
        "model sha256",
        "ember",
        "reference",
        "reference source",
    ]
    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )
    for row in rows:
        cells = [
            row["report_path"],
            row["label"],
            row["model"],
            row["status"],
            row["max_abs_diff"],
            row["mean_abs_diff"],
            row["top1_agreement"],
            row["top_k_overlap"],
            row["tokenizer"],
            _short_sha(row["model_sha256"]),
            row["ember"],
            row["reference"],
            row["reference_source"],
        ]
        lines.append("| " + " | ".join(_md_escape(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_summary(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="summarize golden-logit reports")
    parser.add_argument("--glob", default=DEFAULT_GLOB, help="input report glob")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    reports = load_reports(args.glob)
    summary = {
        "generated_by": "probes/golden_summary.py",
        "glob": args.glob,
        "report_count": len(reports),
        "reports": reports,
    }
    write_json(Path(args.output_json), summary)
    write_markdown(Path(args.output_md), summary)
    print(f"wrote {len(reports)} golden report rows to {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
