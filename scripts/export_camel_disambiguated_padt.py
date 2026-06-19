#!/usr/bin/env python3
"""Export one CAMeL disambiguated analysis per token for the dataset pipeline.

The script keeps the core arabic_morph_dataset pipeline unchanged. It reads
PADT-style CoNLL-U or simple sentence JSONL, runs a CAMeL Tools disambiguator
over each full sentence, and writes CAMeL-style JSONL records that the existing
normalizer already understands.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ELIGIBLE_UPOS = {"NOUN", "VERB", "ADJ"}
FEATURE_FIELDS = ["gen", "num", "per", "asp", "vox", "mod", "cas", "stt"]
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


@dataclass
class Token:
    form: str
    index: int
    upos: str = ""
    lemma: str = ""
    feats: dict[str, str] | None = None
    token_id: str = ""


@dataclass
class Sentence:
    sentence_id: str
    text: str
    tokens: list[Token]


def parse_feats(value: str) -> dict[str, str]:
    if not value or value == "_":
        return {}
    feats = {}
    for item in value.split("|"):
        if "=" in item:
            key, val = item.split("=", 1)
            feats[key] = val
    return feats


def iter_conllu(path: Path) -> Iterable[Sentence]:
    tokens: list[Token] = []
    metadata: dict[str, str] = {}
    sent_idx = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            if tokens:
                sent_idx += 1
                yield _make_sentence(metadata, tokens, sent_idx)
            tokens = []
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
        tokens.append(
            Token(
                form=cols[1],
                lemma=cols[2] if cols[2] != "_" else "",
                upos=cols[3] if cols[3] != "_" else "",
                feats=parse_feats(cols[5]),
                token_id=cols[0],
                index=len(tokens),
            )
        )
    if tokens:
        sent_idx += 1
        yield _make_sentence(metadata, tokens, sent_idx)


def _make_sentence(metadata: dict[str, str], tokens: list[Token], sent_idx: int) -> Sentence:
    text = metadata.get("text") or " ".join(token.form for token in tokens)
    sentence_id = metadata.get("sent_id") or metadata.get("newpar id") or str(sent_idx)
    return Sentence(sentence_id=sentence_id, text=text, tokens=tokens)


def iter_sentence_jsonl(path: Path) -> Iterable[Sentence]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_tokens = row.get("tokens")
            if not isinstance(raw_tokens, list):
                raise ValueError(f"{path}:{line_no}: expected tokens list")
            tokens = []
            for idx, item in enumerate(raw_tokens):
                if isinstance(item, str):
                    tokens.append(Token(form=item, index=idx))
                elif isinstance(item, dict):
                    form = str(item.get("form") or item.get("text") or item.get("token") or "")
                    tokens.append(
                        Token(
                            form=form,
                            lemma=str(item.get("lemma") or ""),
                            upos=str(item.get("upos") or item.get("pos") or ""),
                            feats=dict(item.get("feats") or item.get("features") or {}),
                            token_id=str(item.get("id") or idx + 1),
                            index=idx,
                        )
                    )
                else:
                    raise ValueError(f"{path}:{line_no}: token {idx} must be string or object")
            text = str(row.get("text") or " ".join(token.form for token in tokens))
            sentence_id = str(row.get("sentence_id") or row.get("id") or line_no)
            yield Sentence(sentence_id=sentence_id, text=text, tokens=tokens)


def load_disambiguator(model_name: str, top: int):
    try:
        from camel_tools.disambig.mle import MLEDisambiguator
    except ImportError as exc:
        raise RuntimeError("CAMeL Tools with MLE disambiguation support is required") from exc

    return MLEDisambiguator.pretrained(model_name, top=top)


def selected_analysis(disambig_word: Any) -> tuple[dict[str, Any] | None, float | None, int]:
    analyses = getattr(disambig_word, "analyses", [])
    if not analyses:
        return None, None, 0
    scored = analyses[0]
    return dict(scored.analysis), float(scored.score), len(analyses)


def clean_value(value: Any, *, missing_na: bool = False) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"NOAN", "N/A", "UNK"}:
        return ""
    if missing_na and text == "na":
        return ""
    return text


def camel_record(
    sentence: Sentence,
    token: Token,
    analysis: dict[str, Any],
    score: float | None,
    num_returned_analyses: int,
) -> dict[str, Any]:
    analysis_id = f"{sentence.sentence_id}:{token.token_id or token.index + 1}"
    record = {
        "analysis_id": analysis_id,
        "word": token.form,
        "diac": clean_value(analysis.get("diac")),
        "lex": clean_value(analysis.get("lex")),
        "root": clean_value(analysis.get("root"), missing_na=True),
        "pattern": clean_value(analysis.get("pattern"), missing_na=True),
        "pattern_concrete": clean_value(analysis.get("stem") or analysis.get("diac")),
        "pos": clean_value(analysis.get("pos")),
        "source": "camel_tools_disambig_mle_calima_msa_r13",
        "metadata": {
            "sentence_id": sentence.sentence_id,
            "token_index": token.index,
            "token_id": token.token_id,
            "sentence_text": sentence.text,
            "source_upos": token.upos,
            "source_lemma": token.lemma,
            "source_feats": token.feats or {},
            "selected_score": score,
            "num_returned_analyses": num_returned_analyses,
            "camel_bw": clean_value(analysis.get("bw")),
            "camel_gloss": clean_value(analysis.get("gloss")),
            "camel_source": clean_value(analysis.get("source")),
        },
    }
    for field in FEATURE_FIELDS:
        record[field] = clean_value(analysis.get(field))
    return record


def is_arabic_token(token: str) -> bool:
    return bool(token and ARABIC_RE.search(token))


def should_keep_token(token: Token, analysis: dict[str, Any], require_source_upos: bool) -> bool:
    if not is_arabic_token(token.form):
        return False
    if require_source_upos:
        return token.upos in ELIGIBLE_UPOS
    source_or_camel_pos = token.upos or str(analysis.get("pos") or "").upper()
    return source_or_camel_pos.upper() in ELIGIBLE_UPOS


def export_sentences(
    sentences: Iterable[Sentence],
    output: Path,
    report_path: Path,
    model_name: str,
    top: int,
    limit_sentences: int | None,
    limit_records: int | None,
    require_source_upos: bool,
) -> dict[str, Any]:
    disambiguator = load_disambiguator(model_name, top)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_sentences = 0
    total_tokens = 0
    eligible_tokens = 0
    disambiguated_tokens = 0
    skipped_no_analysis = 0
    missing = Counter()
    pos_counts = Counter()
    examples = []
    records_written = 0

    with output.open("w", encoding="utf-8", newline="\n") as out:
        for sentence in sentences:
            if limit_sentences is not None and total_sentences >= limit_sentences:
                break
            total_sentences += 1
            forms = [token.form for token in sentence.tokens]
            total_tokens += len(forms)
            disambig_words = disambiguator.disambiguate(forms)
            for token, disambig_word in zip(sentence.tokens, disambig_words):
                analysis, score, num_returned = selected_analysis(disambig_word)
                if analysis is None:
                    if token.upos in ELIGIBLE_UPOS:
                        skipped_no_analysis += 1
                    continue
                if not should_keep_token(token, analysis, require_source_upos):
                    continue
                eligible_tokens += 1
                record = camel_record(sentence, token, analysis, score, num_returned)
                disambiguated_tokens += 1
                pos_counts[record["pos"] or "<missing>"] += 1
                for field, reason in [
                    ("root", "missing_root"),
                    ("pattern", "missing_abstract_pattern"),
                    ("pattern_concrete", "missing_concrete_pattern"),
                    ("lex", "missing_lemma"),
                ]:
                    if not record.get(field):
                        missing[reason] += 1
                if len(examples) < 10:
                    examples.append(record)
                out.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                records_written += 1
                if limit_records is not None and records_written >= limit_records:
                    break
            if limit_records is not None and records_written >= limit_records:
                break

    report = {
        "input_sentences": total_sentences,
        "total_tokens": total_tokens,
        "eligible_tokens": eligible_tokens,
        "disambiguated_tokens": disambiguated_tokens,
        "skipped_no_analysis": skipped_no_analysis,
        "records_written": records_written,
        "missing": dict(sorted(missing.items())),
        "pos_distribution": dict(sorted(pos_counts.items())),
        "model_name": model_name,
        "top": top,
        "require_source_upos": require_source_upos,
        "examples": examples,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="export CAMeL-disambiguated morphology records")
    parser.add_argument("--input", required=True, help="PADT CoNLL-U or sentence JSONL input")
    parser.add_argument("--input-format", choices=["conllu", "jsonl"], default="conllu")
    parser.add_argument("--output", required=True, help="output CAMeL-style JSONL")
    parser.add_argument("--report", required=True, help="output JSON report")
    parser.add_argument("--model-name", default="calima-msa-r13", help="CAMeL MLE disambiguator model name")
    parser.add_argument("--top", type=int, default=1, help="number of analyses requested from CAMeL")
    parser.add_argument("--limit-sentences", type=int)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument(
        "--allow-camel-pos-eligibility",
        action="store_true",
        help="for JSONL or unlabeled input, keep tokens whose selected CAMeL POS is NOUN/VERB/ADJ",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    sentences = iter_conllu(input_path) if args.input_format == "conllu" else iter_sentence_jsonl(input_path)
    try:
        report = export_sentences(
            sentences=sentences,
            output=Path(args.output),
            report_path=Path(args.report),
            model_name=args.model_name,
            top=args.top,
            limit_sentences=args.limit_sentences,
            limit_records=args.limit_records,
            require_source_upos=not args.allow_camel_pos_eligibility,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: report[k] for k in ["records_written", "eligible_tokens", "total_tokens"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
