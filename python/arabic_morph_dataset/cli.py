from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from .exporters import DEFAULT_SFT_TASKS, make_probe_records, make_sft_examples
from .filters import apply_filters
from .io import load_config, read_jsonl, read_morph_records, read_raw_records, write_json, write_jsonl, write_morph_records
from .normalize import normalize_records
from .report import make_summary_report
from .split import SPLIT_STRATEGIES, normalize_split_ratios, split_records
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
    p.add_argument("--split-type", required=True, choices=sorted(SPLIT_STRATEGIES))

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
        _validate_ratios(ratios)
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
        _validate_ratios(ratios)
        report = make_summary_report(read_morph_records(args.input), filter_report, args.seed, ratios)
        if args.output:
            write_json(args.output, report)
        else:
            print_report(report)
    elif args.command == "run-config":
        run_config(args.config)
    return 0


def run_config(config_path: str | Path) -> None:
    config_source = Path(config_path).resolve(strict=True)
    config_identity = _file_identity(config_source)
    cfg = load_config(config_source)
    config_sha256 = _sha256_file(config_source)
    if _file_identity(config_source) != config_identity:
        raise CliError(f"run config changed while it was being read: {config_source}")
    _ensure_mapping(cfg, "config")
    _validate_run_config(cfg)
    _require_config_keys(cfg, ["input_path", "output_dir"])
    output_dir = Path(cfg["output_dir"])
    if output_dir.is_symlink():
        raise CliError("output_dir must not be a symbolic link")
    output_dir = output_dir.resolve()
    input_resolved = Path(cfg["input_path"]).resolve(strict=True)
    if not input_resolved.is_file():
        raise CliError(f"input_path is not a regular file: {input_resolved}")
    input_identity = _file_identity(input_resolved)
    input_sha256 = _sha256_file(input_resolved)
    if input_resolved.is_relative_to(output_dir) or config_source.is_relative_to(output_dir):
        raise CliError("output_dir must not contain the input file or run config")
    ratios = cfg.get("split_ratios", {"train": 0.8, "dev": 0.1, "test": 0.1})
    _validate_ratios(ratios)
    seed = cfg.get("seed", 13)
    split_strategy = cfg.get("split_strategy", "root_heldout")
    formats = cfg.get(
        "output_formats",
        ["canonical", "sft", "probes", "stats", "validation", "summary_report"],
    )
    raw = read_raw_records(input_resolved)
    records, ingest_report = normalize_records(raw, cfg.get("source_name", "camel_export"))
    records, filter_report = apply_filters(records, cfg.get("filters", {}))
    split_records_out, split_report = split_records(
        records,
        split_strategy,
        seed,
        ratios,
    )
    sft_examples = make_sft_examples(split_records_out, cfg.get("sft_tasks", DEFAULT_SFT_TASKS))
    probe_records = make_probe_records(split_records_out, split_strategy)
    stats_report = dataset_stats(split_records_out)
    summary_report = make_summary_report(records, filter_report, seed, ratios)
    validation_report = {
        "canonical": validate_canonical(split_records_out, split_strategy),
        "sft": validate_sft_examples(sft_examples),
        "probes": validate_probe_records(probe_records),
    }
    validation_report["passed"] = all(item["passed"] for item in validation_report.values() if isinstance(item, dict))
    if not validation_report["passed"]:
        failure_path = output_dir.with_name(f"{output_dir.name}.validation_failed.json")
        write_json(failure_path, validation_report)
        raise CliError(f"validation failed; see {failure_path}")

    _verify_file_snapshot(config_source, config_identity, config_sha256, "run config")
    _verify_file_snapshot(input_resolved, input_identity, input_sha256, "input")

    if output_dir == Path(output_dir.anchor):
        raise CliError("refusing to use a filesystem root as output_dir")
    if output_dir.exists() and not output_dir.is_dir():
        raise CliError("output_dir exists and is not a directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    try:
        _copy_preserved_entries(output_dir, staging)
        writers = {
            "canonical": lambda: write_morph_records(staging / "canonical.jsonl", split_records_out),
            "sft": lambda: write_jsonl(staging / "sft.jsonl", sft_examples),
            "probes": lambda: write_jsonl(staging / "probes.jsonl", probe_records),
            "stats": lambda: write_json(staging / "stats.json", stats_report),
            "summary_report": lambda: write_json(staging / "summary_report.json", summary_report),
            "validation": lambda: write_json(staging / "validation.json", validation_report),
            "ingest_report": lambda: write_json(staging / "ingest_report.json", ingest_report),
            "filter_report": lambda: write_json(staging / "filter_report.json", filter_report),
            "split_report": lambda: write_json(staging / "split_report.json", split_report),
        }
        # Operational reports are always emitted; output_formats selects research exports.
        selected = list(dict.fromkeys([*formats, "ingest_report", "filter_report", "split_report"]))
        for name in selected:
            writers[name]()
        write_json(
            staging / "run_manifest.json",
            {
                "schema_version": 1,
                "config_path": str(config_source),
                "config_sha256": config_sha256,
                "input_path": str(input_resolved),
                "input_sha256": input_sha256,
                "seed": seed,
                "split_strategy": split_strategy,
                "split_ratios": normalize_split_ratios(ratios),
                "output_formats": formats,
                "record_count": len(split_records_out),
            },
        )
        _commit_output_directory(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


RUN_CONFIG_KEYS = {
    "input_path",
    "output_dir",
    "source_name",
    "seed",
    "split_strategy",
    "sft_tasks",
    "output_formats",
    "filters",
    "split_ratios",
}
OUTPUT_FORMATS = {
    "canonical",
    "sft",
    "probes",
    "stats",
    "summary_report",
    "validation",
    "ingest_report",
    "filter_report",
    "split_report",
}


def _validate_run_config(cfg: dict[str, Any]) -> None:
    unknown = sorted(set(cfg) - RUN_CONFIG_KEYS)
    if unknown:
        raise CliError(f"unknown run-config keys: {unknown}")
    _require_config_keys(cfg, ["input_path", "output_dir"])
    for key in ("input_path", "output_dir"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise CliError(f"{key} must be a non-empty string")
    source = cfg.get("source_name", "camel_export")
    if not isinstance(source, str) or not source.strip():
        raise CliError("source_name must be a non-empty string")
    seed = cfg.get("seed", 13)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CliError("seed must be an integer")
    strategy = cfg.get("split_strategy", "root_heldout")
    if strategy not in SPLIT_STRATEGIES:
        raise CliError(f"unknown split_strategy: {strategy!r}")
    tasks = cfg.get("sft_tasks", DEFAULT_SFT_TASKS)
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(task, str) or not task.strip() for task in tasks)
        or len(tasks) != len(set(tasks))
    ):
        raise CliError("sft_tasks must be a non-empty list of unique strings")
    formats = cfg.get(
        "output_formats",
        ["canonical", "sft", "probes", "stats", "validation", "summary_report"],
    )
    if (
        not isinstance(formats, list)
        or not formats
        or any(not isinstance(value, str) or value not in OUTPUT_FORMATS for value in formats)
        or len(formats) != len(set(formats))
    ):
        raise CliError(
            f"output_formats must be a non-empty unique subset of {sorted(OUTPUT_FORMATS)}"
        )
    _ensure_mapping(cfg.get("filters", {}), "filters")
    _ensure_mapping(cfg.get("split_ratios", {}), "split_ratios")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str | Path) -> tuple[int, int, int, int]:
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _verify_file_snapshot(
    path: Path,
    expected_identity: tuple[int, int, int, int],
    expected_sha256: str,
    label: str,
) -> None:
    if _file_identity(path) != expected_identity or _sha256_file(path) != expected_sha256:
        raise CliError(f"{label} changed during the dataset run: {path}")


def _commit_output_directory(staging: Path, destination: Path) -> None:
    backup = destination.with_name(
        f".{destination.name}.backup-{os.getpid()}-{uuid.uuid4().hex}"
    )
    moved_old = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_old = True
        os.replace(staging, destination)
    except Exception:
        if moved_old and not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _copy_preserved_entries(source: Path, staging: Path) -> None:
    """Preserve downstream/user files while replacing pipeline-owned outputs."""
    if not source.exists():
        return
    generated = {
        "canonical.jsonl",
        "sft.jsonl",
        "probes.jsonl",
        "stats.json",
        "summary_report.json",
        "validation.json",
        "ingest_report.json",
        "filter_report.json",
        "split_report.json",
        "run_manifest.json",
    }
    for child in source.iterdir():
        if child.name in generated:
            continue
        if child.is_symlink():
            raise CliError(f"refusing to preserve symbolic link in output_dir: {child}")
        destination = staging / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        elif child.is_file():
            shutil.copy2(child, destination)
        else:
            raise CliError(f"unsupported filesystem entry in output_dir: {child}")


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
    def reject_constant(value):
        raise CliError(f"non-standard JSON constant {value!r} in {path}")

    value = json.loads(
        Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    _ensure_mapping(value, "report")
    return value


def _ensure_mapping(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise CliError(f"{name} must be a mapping/object")


def _require_config_keys(config: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise CliError(f"config missing required keys: {missing}")


def _validate_ratios(ratios: dict[str, Any]) -> None:
    try:
        normalize_split_ratios(ratios)
    except ValueError as error:
        raise CliError(str(error)) from error


def _write_optional_report(path: str | None, report: dict) -> None:
    if path:
        write_json(path, report)
    else:
        print_report(report)


def print_report(report: dict) -> None:
    import json

    sys.stdout.write(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(entrypoint())
