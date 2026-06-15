from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .exporters import DEFAULT_SFT_TASKS, make_probe_records, make_sft_examples
from .filters import apply_filters
from .io import load_config, read_jsonl, read_morph_records, read_raw_records, write_json, write_jsonl, write_morph_records
from .normalize import normalize_records
from .report import make_summary_report
from .split import SPLIT_STRATEGIES, split_records
from .stats import dataset_stats
from .validate import validate_canonical, validate_canonical_rows, validate_probe_records, validate_sft_examples


class CliError(RuntimeError):
    pass


def entrypoint(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arabic-morph-dataset")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-name", default="camel_export")
    p.add_argument("--report")

    p = sub.add_parser("normalize")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--report")

    p = sub.add_parser("split")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--strategy", required=True, choices=sorted(SPLIT_STRATEGIES))
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--report")

    p = sub.add_parser("make-sft")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tasks", default=",".join(DEFAULT_SFT_TASKS))

    p = sub.add_parser("make-probes")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split-type", required=True)

    p = sub.add_parser("validate")
    p.add_argument("--input")
    p.add_argument("--sft")
    p.add_argument("--probes")
    p.add_argument("--split-strategy", choices=sorted(SPLIT_STRATEGIES))
    p.add_argument("--output")

    p = sub.add_parser("stats")
    p.add_argument("--input", required=True)
    p.add_argument("--output")

    p = sub.add_parser("report")
    p.add_argument("--input", required=True)
    p.add_argument("--filter-report")
    p.add_argument("--output")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)

    p = sub.add_parser("run-config")
    p.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        raw = read_raw_records(args.input)
        records, report = normalize_records(raw, args.source_name)
        write_morph_records(args.output, records)
        _write_optional_report(args.report, report)
    elif args.command == "normalize":
        cfg = load_config(args.config) if args.config else {}
        _ensure_mapping(cfg, "config")
        records = read_morph_records(args.input)
        records, report = apply_filters(records, cfg.get("filters", cfg))
        write_morph_records(args.output, records)
        _write_optional_report(args.report, report)
    elif args.command == "split":
        ratios = {"train": args.train_ratio, "dev": args.dev_ratio, "test": args.test_ratio}
        records, report = split_records(read_morph_records(args.input), args.strategy, args.seed, ratios)
        write_morph_records(args.output, records)
        _write_optional_report(args.report, report)
    elif args.command == "make-sft":
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        if not tasks:
            raise CliError("--tasks must include at least one task")
        write_jsonl(args.output, make_sft_examples(read_morph_records(args.input), tasks))
    elif args.command == "make-probes":
        write_jsonl(args.output, make_probe_records(read_morph_records(args.input), args.split_type))
    elif args.command == "validate":
        report = {}
        if args.input:
            report["canonical"] = validate_canonical_rows(read_jsonl(args.input), args.split_strategy)
        if args.sft:
            report["sft"] = validate_sft_examples(read_jsonl(args.sft))
        if args.probes:
            report["probes"] = validate_probe_records(read_jsonl(args.probes))
        report["passed"] = bool(report) and all(item.get("passed", False) for item in report.values() if isinstance(item, dict))
        if args.output:
            write_json(args.output, report)
        else:
            print_report(report)
        if not report["passed"]:
            return 1
    elif args.command == "stats":
        report = dataset_stats(read_morph_records(args.input))
        if args.output:
            write_json(args.output, report)
        else:
            print_report(report)
    elif args.command == "report":
        filter_report = _read_optional_json_report(args.filter_report)
        ratios = {"train": args.train_ratio, "dev": args.dev_ratio, "test": args.test_ratio}
        report = make_summary_report(read_morph_records(args.input), filter_report, args.seed, ratios)
        if args.output:
            write_json(args.output, report)
        else:
            print_report(report)
    elif args.command == "run-config":
        run_config(args.config)
    return 0


def run_config(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    _ensure_mapping(cfg, "config")
    _require_config_keys(cfg, ["input_path", "output_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_raw_records(cfg["input_path"])
    records, ingest_report = normalize_records(raw, cfg.get("source_name", "camel_export"))
    records, filter_report = apply_filters(records, cfg.get("filters", {}))
    split_records_out, split_report = split_records(
        records,
        cfg.get("split_strategy", "root_heldout"),
        int(cfg.get("seed", 13)),
        cfg.get("split_ratios", {"train": 0.8, "dev": 0.1, "test": 0.1}),
    )
    canonical_path = output_dir / "canonical.jsonl"
    sft_path = output_dir / "sft.jsonl"
    probes_path = output_dir / "probes.jsonl"
    sft_examples = make_sft_examples(split_records_out, cfg.get("sft_tasks", DEFAULT_SFT_TASKS))
    probe_records = make_probe_records(split_records_out, cfg.get("split_strategy", "root_heldout"))
    write_morph_records(canonical_path, split_records_out)
    write_jsonl(sft_path, sft_examples)
    write_jsonl(probes_path, probe_records)
    stats_report = dataset_stats(split_records_out)
    summary_report = make_summary_report(records, filter_report, int(cfg.get("seed", 13)), cfg.get("split_ratios", {"train": 0.8, "dev": 0.1, "test": 0.1}))
    validation_report = {
        "canonical": validate_canonical(split_records_out, cfg.get("split_strategy")),
        "sft": validate_sft_examples(sft_examples),
        "probes": validate_probe_records(probe_records),
    }
    validation_report["passed"] = all(item["passed"] for item in validation_report.values() if isinstance(item, dict))
    write_json(output_dir / "ingest_report.json", ingest_report)
    write_json(output_dir / "filter_report.json", filter_report)
    write_json(output_dir / "split_report.json", split_report)
    write_json(output_dir / "stats.json", stats_report)
    write_json(output_dir / "summary_report.json", summary_report)
    write_json(output_dir / "validation.json", validation_report)
    if not validation_report["passed"]:
        raise CliError(f"validation failed; see {output_dir / 'validation.json'}")


def _read_optional_json_report(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    if path.endswith(".jsonl"):
        rows = read_jsonl(path)
        if not rows:
            raise CliError(f"empty JSONL report: {path}")
        if len(rows) > 1:
            raise CliError(f"expected a single report object in {path}, found {len(rows)} JSONL rows")
        return rows[0]
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ensure_mapping(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise CliError(f"{name} must be a mapping/object")


def _require_config_keys(config: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise CliError(f"config missing required keys: {missing}")


def _write_optional_report(path: str | None, report: dict) -> None:
    if path:
        write_json(path, report)
    else:
        print_report(report)


def print_report(report: dict) -> None:
    import json

    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(entrypoint())
