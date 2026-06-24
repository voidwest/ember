from __future__ import annotations

import argparse
import sys

from .ember_config import write_ember_config
from .io import read_jsonl, write_json, write_jsonl
from .prompts import make_prompts
from .splits import lemma_heldout_split
from .toy import toy_records
from .validation import validate_run
from .workflow import DEFAULT_TEMPLATE, write_example_workflow
from .artifacts import validate_manifest
from .backends.ember import import_ember_run
from .backends.files import import_files


def entrypoint(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sarf",
        description="Backend-agnostic workflow scaffolding for Arabic morphology probing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("toy-dataset", help="write the bundled toy morphology dataset")
    p.add_argument("--out", required=True, help="output JSONL path")

    p = sub.add_parser("make-prompts", help="render prompt JSONL from morphology records")
    p.add_argument("--input", required=True, help="input morphology JSONL")
    p.add_argument("--out", required=True, help="output prompt JSONL")
    p.add_argument("--template", default=DEFAULT_TEMPLATE, help="Python format prompt template")

    p = sub.add_parser("split-metadata", help="assign toy lemma-heldout splits and metadata")
    p.add_argument("--input", required=True, help="input morphology JSONL")
    p.add_argument("--out-records", required=True, help="output split morphology JSONL")
    p.add_argument("--out-metadata", required=True, help="output split metadata JSON")

    p = sub.add_parser("ember-config", help="write an Ember extraction config placeholder")
    p.add_argument("--out", required=True, help="output TOML path")
    p.add_argument("--run-id", default="sarf-v01-toy-native-logits", help="Ember run_id")
    p.add_argument("--model-path", default="/path/to/model.gguf", help="local GGUF path placeholder")
    p.add_argument("--tokenizer-path", default="/path/to/tokenizer.json", help="local tokenizer JSON path placeholder")
    p.add_argument("--prompts-path", required=True, help="prompt JSONL path")
    p.add_argument("--output-dir", required=True, help="Ember output directory")
    p.add_argument("--backend", default="native", help="Ember backend name")
    p.add_argument("--architecture", default="qwen3", help="model architecture hint")
    p.add_argument("--no-logits", action="store_true", help="set write_logits = false")

    p = sub.add_parser("validate-artifact", help="run Ember validate-run for an artifact directory")
    p.add_argument("--run-dir", required=True, help="Ember artifact run directory")
    p.add_argument("--repo-root", default=".", help="repository root containing Cargo.toml")

    p = sub.add_parser("import-ember", help="import an Ember run as a Sarf artifact manifest")
    p.add_argument("--run-dir", required=True, help="Ember artifact run directory")
    p.add_argument("--out", required=True, help="output Sarf artifact manifest JSON")
    p.add_argument("--run-id", help="override Sarf run id")

    p = sub.add_parser("import-files", help="create a Sarf manifest from precomputed files")
    p.add_argument("--run-id", required=True, help="Sarf run id")
    p.add_argument("--prompts-path", required=True, help="prompt/sample artifact path")
    p.add_argument("--out", required=True, help="output Sarf artifact manifest JSON")
    p.add_argument("--tokenization-path", help="optional tokenization artifact path")
    p.add_argument("--positions-path", help="optional positions artifact path")
    p.add_argument("--logits-path", help="optional logits artifact path")
    p.add_argument("--hidden-states-path", help="optional hidden-state artifact path")
    p.add_argument("--report-path", help="optional report artifact path")

    p = sub.add_parser("validate-manifest", help="validate a Sarf artifact manifest JSON")
    p.add_argument("--manifest", required=True, help="Sarf artifact manifest JSON")
    p.add_argument("--out", help="optional validation report JSON")

    p = sub.add_parser("example-workflow", help="write the complete toy v0.1 workflow scaffold")
    p.add_argument("--out-dir", required=True, help="output workflow directory")

    args = parser.parse_args(argv)
    if args.command == "toy-dataset":
        write_jsonl(args.out, toy_records())
    elif args.command == "make-prompts":
        write_jsonl(args.out, make_prompts(read_jsonl(args.input), args.template))
    elif args.command == "split-metadata":
        records, metadata = lemma_heldout_split(read_jsonl(args.input))
        write_jsonl(args.out_records, records)
        write_json(args.out_metadata, metadata)
    elif args.command == "ember-config":
        write_ember_config(
            args.out,
            run_id=args.run_id,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            prompts_path=args.prompts_path,
            output_dir=args.output_dir,
            backend=args.backend,
            architecture=args.architecture,
            write_logits=not args.no_logits,
        )
    elif args.command == "validate-artifact":
        return validate_run(args.run_dir, repo_root=args.repo_root)
    elif args.command == "import-ember":
        write_json(args.out, import_ember_run(args.run_dir, run_id=args.run_id).to_dict())
    elif args.command == "import-files":
        write_json(
            args.out,
            import_files(
                run_id=args.run_id,
                prompts_path=args.prompts_path,
                tokenization_path=args.tokenization_path,
                positions_path=args.positions_path,
                logits_path=args.logits_path,
                hidden_states_path=args.hidden_states_path,
                report_path=args.report_path,
            ).to_dict(),
        )
    elif args.command == "validate-manifest":
        import json
        from pathlib import Path

        report = validate_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
        if args.out:
            write_json(args.out, report)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    elif args.command == "example-workflow":
        write_example_workflow(args.out_dir)
    return 0
