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
import json
from pathlib import Path


def parse_feats(value: str) -> dict[str, str]:
    if not value or value == "_":
        return {}
    feats = {}
    for item in value.split("|"):
        if "=" in item:
            key, val = item.split("=", 1)
            feats[key] = val
    return feats


def iter_conllu(path: str):
    sentence = []
    metadata = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            if sentence:
                yield metadata, sentence
            sentence = []
            metadata = {}
            continue
        if line.startswith("#"):
            if "=" in line:
                key, value = line[1:].split("=", 1)
                metadata[key.strip()] = value.strip()
            continue
        cols = line.split("\t")
        if len(cols) != 10 or "-" in cols[0] or "." in cols[0]:
            continue
        sentence.append(cols)
    if sentence:
        yield metadata, sentence


def token_spans(text: str, forms: list[str]) -> list[tuple[int, int] | None]:
    spans = []
    cursor = 0
    for form in forms:
        start = text.find(form, cursor)
        if start < 0:
            start = text.find(form)
        if start < 0:
            spans.append(None)
            continue
        end = start + len(form)
        spans.append((start, end))
        cursor = end
    return spans


def build_rows(path: str, min_label_count: int = 2) -> list[dict]:
    rows = []
    for sent_idx, (metadata, sentence) in enumerate(iter_conllu(path)):
        forms = [cols[1] for cols in sentence]
        text = metadata.get("text") or " ".join(forms)
        spans = token_spans(text, forms)
        sent_id = metadata.get("sent_id", str(sent_idx))

        for token_idx, (cols, span) in enumerate(zip(sentence, spans)):
            if span is None:
                continue
            feats = parse_feats(cols[5])
            labels = {
                "form": cols[1],
                "lemma": cols[2],
                "upos": cols[3],
                **feats,
            }
            rows.append(
                {
                    "id": f"{sent_id}:{cols[0]}",
                    "sentence_id": sent_id,
                    "token_index": token_idx,
                    "text": text,
                    "target": cols[1],
                    "target_span": [span[0], span[1]],
                    "labels": labels,
                }
            )

    if min_label_count <= 1:
        return rows

    # Drop labels that are globally too rare by replacing them with None; the
    # row stays usable for other tasks.
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for key, value in row["labels"].items():
            counts.setdefault(key, {})
            counts[key][value] = counts[key].get(value, 0) + 1
    for row in rows:
        for key, value in list(row["labels"].items()):
            if counts[key].get(value, 0) < min_label_count:
                row["labels"][key] = None
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="build a JSON morphology benchmark from CoNLL-U")
    parser.add_argument("--input", required=True, help="CoNLL-U file")
    parser.add_argument("--output", required=True, help="output JSON rows")
    parser.add_argument("--limit", type=int, default=None, help="optional row limit")
    parser.add_argument("--min-label-count", type=int, default=2)
    args = parser.parse_args()

    rows = build_rows(args.input, args.min_label_count)
    if args.limit is not None:
        rows = rows[: args.limit]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    label_keys = sorted({key for row in rows for key in row["labels"]})
    print(f"wrote {len(rows)} rows to {args.output}")
    print("label fields: " + ", ".join(f"labels.{key}" for key in label_keys))


if __name__ == "__main__":
    main()
