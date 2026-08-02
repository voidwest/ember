"""build token-level Arabic morphology benchmark rows from CoNLL-U.

Each output row is one target token inside its original sentence. The row keeps
the sentence text, target character span, UPOS, lemma, and all FEATS entries
under `labels.*`, so the generic probe runner can target fields such as:

    labels.upos
    labels.Gender
    labels.Number
    labels.Aspect

Rows are JSON-compatible with `extract_hf_encoder.py`.
"""

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .train_linear_probe import atomic_write_text, sha256_file
except ImportError:  # direct script execution
    from train_linear_probe import atomic_write_text, sha256_file


def parse_feats(value: str) -> dict[str, str]:
    if not value or value == "_":
        return {}
    feats = {}
    for item in value.split("|"):
        if "=" not in item:
            raise ValueError(f"malformed CoNLL-U feature {item!r}")
        key, val = item.split("=", 1)
        if not key or not val or key in feats:
            raise ValueError(f"invalid or duplicate CoNLL-U feature {item!r}")
        feats[key] = val
    return feats


def iter_conllu(path: str):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CoNLL-U input does not exist: {source}")
    sentence = []
    metadata = {}
    previous_token_id = 0
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            if sentence:
                yield metadata, sentence
            sentence = []
            metadata = {}
            previous_token_id = 0
            continue
        if line.startswith("#"):
            if "=" in line:
                key, value = line[1:].split("=", 1)
                key = key.strip()
                if key in metadata:
                    raise ValueError(f"{path}:{line_number}: duplicate metadata key {key!r}")
                metadata[key] = value.strip()
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            raise ValueError(f"{path}:{line_number}: expected 10 tab-separated CoNLL-U columns")
        if "-" in cols[0] or "." in cols[0]:
            continue
        if not cols[0].isdigit():
            raise ValueError(f"{path}:{line_number}: invalid token ID {cols[0]!r}")
        token_id = int(cols[0])
        if token_id <= previous_token_id:
            raise ValueError(f"{path}:{line_number}: token IDs must increase strictly")
        previous_token_id = token_id
        if not cols[1] or cols[1] == "_":
            raise ValueError(f"{path}:{line_number}: token FORM must be present")
        sentence.append(cols)
    if sentence:
        yield metadata, sentence


def token_spans(text: str, forms: list[str]) -> list[tuple[int, int] | None]:
    spans = []
    cursor = 0
    for form in forms:
        start = text.find(form, cursor)
        if start < 0:
            spans.append(None)
            continue
        end = start + len(form)
        spans.append((start, end))
        cursor = end
    return spans


def reconstruct_text(sentence: list[list[str]]) -> str:
    pieces = []
    for columns in sentence:
        pieces.append(columns[1])
        misc = columns[9].split("|") if columns[9] not in {"", "_"} else []
        if "SpaceAfter=No" not in misc:
            pieces.append(" ")
    return "".join(pieces).rstrip()


def _stable_limit(rows: list[dict], limit: int, seed: int) -> list[dict]:
    ranked = sorted(
        enumerate(rows),
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[1]['id']}".encode("utf-8")
        ).digest(),
    )[:limit]
    return [row for _, row in sorted(ranked)]


def build_rows(
    path: str,
    min_label_count: int = 2,
    limit: int | None = None,
    *,
    allow_unaligned: bool = False,
    limit_selection: str = "hash",
    seed: int = 42,
    audit: dict | None = None,
) -> list[dict]:
    if min_label_count < 1:
        raise ValueError("min_label_count must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if limit_selection not in {"hash", "head"}:
        raise ValueError("limit_selection must be 'hash' or 'head'")
    rows = []
    seen_ids = set()
    unaligned = []
    sentence_count = 0
    for sent_idx, (metadata, sentence) in enumerate(iter_conllu(path)):
        sentence_count += 1
        forms = [cols[1] for cols in sentence]
        text = metadata.get("text") or reconstruct_text(sentence)
        if not text:
            raise ValueError(f"sentence {sent_idx} has no usable text")
        spans = token_spans(text, forms)
        sent_id = metadata.get("sent_id", str(sent_idx)).strip()
        if not sent_id:
            raise ValueError(f"sentence {sent_idx} has an empty sent_id")

        for token_idx, (cols, span) in enumerate(zip(sentence, spans)):
            if span is None:
                detail = {"sentence_id": sent_id, "token_id": cols[0], "form": cols[1]}
                unaligned.append(detail)
                if allow_unaligned:
                    continue
                raise ValueError(
                    "could not align token to sentence text: "
                    f"sentence={sent_id!r} token={cols[0]!r} form={cols[1]!r}; "
                    "use --allow-unaligned only for an explicitly audited exclusion"
                )
            feats = parse_feats(cols[5])
            labels = {
                "form": cols[1],
                "lemma": None if cols[2] == "_" else cols[2],
                "upos": None if cols[3] == "_" else cols[3],
                **feats,
            }
            row_id = f"{sent_id}:{cols[0]}"
            if row_id in seen_ids:
                raise ValueError(f"duplicate benchmark row ID {row_id!r}")
            seen_ids.add(row_id)
            rows.append(
                {
                    "id": row_id,
                    "sentence_id": sent_id,
                    "token_index": token_idx,
                    "text": text,
                    "target": cols[1],
                    "target_span": [span[0], span[1]],
                    "labels": labels,
                }
            )

    pre_limit_count = len(rows)
    if limit is not None:
        rows = rows[:limit] if limit_selection == "head" else _stable_limit(rows, limit, seed)

    def record_audit() -> None:
        if audit is not None:
            audit.update(
                {
                    "sentence_count": sentence_count,
                    "aligned_rows_before_limit": pre_limit_count,
                    "unaligned_count": len(unaligned),
                    "unaligned_examples": unaligned[:20],
                    "allow_unaligned": allow_unaligned,
                    "limit_selection": limit_selection,
                    "seed": seed,
                }
            )

    if min_label_count <= 1:
        record_audit()
        return rows

    # Drop labels that are globally too rare by replacing them with None; the
    # row stays usable for other tasks.
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for key, value in row["labels"].items():
            if value is None:
                continue
            counts.setdefault(key, {})
            counts[key][value] = counts[key].get(value, 0) + 1
    for row in rows:
        for key, value in list(row["labels"].items()):
            if value is None:
                continue
            if counts[key].get(value, 0) < min_label_count:
                row["labels"][key] = None
    record_audit()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="build a JSON morphology benchmark from CoNLL-U")
    parser.add_argument("--input", required=True, help="CoNLL-U file")
    parser.add_argument("--output", required=True, help="output JSON rows")
    parser.add_argument("--limit", type=int, default=None, help="optional row limit")
    parser.add_argument("--min-label-count", type=int, default=2)
    parser.add_argument(
        "--limit-selection",
        choices=("hash", "head"),
        default="hash",
        help="deterministic row selection when --limit is set",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-unaligned",
        action="store_true",
        help="exclude and report tokens that cannot be aligned to sentence text",
    )
    args = parser.parse_args()

    if args.min_label_count < 1:
        parser.error("--min-label-count must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    source = Path(args.input)
    if not source.is_file():
        parser.error(f"CoNLL-U input does not exist: {source}")
    output = Path(args.output)
    metadata_path = output.with_name(f"{output.stem}_metadata.json")
    if len({output.resolve(), metadata_path.resolve()}) != 2:
        parser.error("output and metadata paths must differ")
    if source.resolve() in {output.resolve(), metadata_path.resolve()}:
        parser.error("output paths must not overwrite the CoNLL-U input")
    source_before = source.stat()
    source_sha256 = sha256_file(source)
    audit: dict = {}
    rows = build_rows(
        args.input,
        args.min_label_count,
        args.limit,
        allow_unaligned=args.allow_unaligned,
        limit_selection=args.limit_selection,
        seed=args.seed,
        audit=audit,
    )
    source_after = source.stat()
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(source_before, field) != getattr(source_after, field) for field in identity_fields):
        raise RuntimeError("CoNLL-U input changed while the benchmark was being built")
    if not rows:
        raise ValueError("CoNLL-U input produced no aligned benchmark rows")
    atomic_write_text(
        args.output,
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    output_sha256 = sha256_file(output)
    atomic_write_text(
        metadata_path,
        json.dumps(
            {
                "schema_version": 2,
                "source_path": str(source.resolve()),
                "source_sha256": source_sha256,
                "output_sha256": output_sha256,
                "row_count": len(rows),
                "min_label_count": args.min_label_count,
                "limit": args.limit,
                "alignment_audit": audit,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    label_keys = sorted({key for row in rows for key in row["labels"]})
    print(f"wrote {len(rows)} rows to {args.output}")
    print("label fields: " + ", ".join(f"labels.{key}" for key in label_keys))


if __name__ == "__main__":
    main()
