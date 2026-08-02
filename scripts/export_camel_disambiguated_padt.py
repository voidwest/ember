#!/usr/bin/env python3
"""Export one CAMeL disambiguated analysis per token for the dataset pipeline.

The script keeps the core arabic_morph_dataset pipeline unchanged. It reads
PADT-style CoNLL-U or simple sentence JSONL, runs a CAMeL Tools disambiguator
over each full sentence, and writes CAMeL-style JSONL records that the existing
normalizer already understands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ELIGIBLE_UPOS = {"NOUN", "VERB", "ADJ"}
FEATURE_FIELDS = ["gen", "num", "per", "asp", "vox", "mod", "cas", "stt"]
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
SCHEMA_VERSION = 2


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise RuntimeError(f"file changed while hashing it: {path}")
    return digest.hexdigest()


@dataclass
class Token:
    form: str
    index: int
    upos: str = ""
    lemma: str = ""
    feats: dict[str, str] | None = None
    token_id: str = ""
    space_after: bool = True


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
        if "=" not in item:
            raise ValueError(f"malformed CoNLL-U feature {item!r}")
        key, val = item.split("=", 1)
        if not key or not val:
            raise ValueError(f"malformed CoNLL-U feature {item!r}")
        if key in feats:
            raise ValueError(f"duplicate CoNLL-U feature {key!r}")
        feats[key] = val
    return feats


def iter_conllu(path: Path) -> Iterable[Sentence]:
    if not path.is_file():
        raise FileNotFoundError(f"CoNLL-U input does not exist: {path}")
    tokens: list[Token] = []
    metadata: dict[str, str] = {}
    sent_idx = 0
    seen_sentence_ids: set[str] = set()
    previous_token_number = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            if tokens:
                sent_idx += 1
                sentence = _make_sentence(metadata, tokens, sent_idx)
                if sentence.sentence_id in seen_sentence_ids:
                    raise ValueError(f"{path}:{line_number}: duplicate sentence ID {sentence.sentence_id!r}")
                seen_sentence_ids.add(sentence.sentence_id)
                yield sentence
            tokens = []
            metadata = {}
            previous_token_number = 0
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
        try:
            token_number = int(cols[0])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid token ID {cols[0]!r}") from error
        if token_number <= 0:
            raise ValueError(f"{path}:{line_number}: token ID must be positive")
        if token_number <= previous_token_number:
            raise ValueError(f"{path}:{line_number}: token IDs must increase strictly")
        previous_token_number = token_number
        if not cols[1] or cols[1] == "_":
            raise ValueError(f"{path}:{line_number}: token form must not be empty")
        tokens.append(
            Token(
                form=cols[1],
                lemma=cols[2] if cols[2] != "_" else "",
                upos=cols[3] if cols[3] != "_" else "",
                feats=parse_feats(cols[5]),
                token_id=cols[0],
                index=len(tokens),
                space_after="SpaceAfter=No" not in (
                    cols[9].split("|") if cols[9] not in {"", "_"} else []
                ),
            )
        )
    if tokens:
        sent_idx += 1
        sentence = _make_sentence(metadata, tokens, sent_idx)
        if sentence.sentence_id in seen_sentence_ids:
            raise ValueError(f"{path}: duplicate sentence ID {sentence.sentence_id!r}")
        yield sentence


def _make_sentence(metadata: dict[str, str], tokens: list[Token], sent_idx: int) -> Sentence:
    if metadata.get("text"):
        text = metadata["text"]
    else:
        pieces = []
        for token in tokens:
            pieces.append(token.form)
            if token.space_after:
                pieces.append(" ")
        text = "".join(pieces).rstrip()
    sentence_id = metadata.get("sent_id") or metadata.get("newpar id") or str(sent_idx)
    if not sentence_id.strip() or not text.strip() or not tokens:
        raise ValueError(f"sentence {sent_idx} has an empty ID, text, or token list")
    return Sentence(sentence_id=sentence_id, text=text, tokens=list(tokens))


def iter_sentence_jsonl(path: Path) -> Iterable[Sentence]:
    if not path.is_file():
        raise FileNotFoundError(f"sentence JSONL input does not exist: {path}")
    seen_sentence_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line, parse_constant=reject_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            raw_tokens = row.get("tokens")
            if not isinstance(raw_tokens, list) or not raw_tokens:
                raise ValueError(f"{path}:{line_no}: expected a non-empty tokens list")
            tokens = []
            seen_token_ids: set[str] = set()
            for idx, item in enumerate(raw_tokens):
                if isinstance(item, str):
                    form = item.strip()
                    if not form:
                        raise ValueError(f"{path}:{line_no}: token {idx} is empty")
                    tokens.append(Token(form=form, index=idx, token_id=str(idx + 1)))
                elif isinstance(item, dict):
                    form_value = item.get("form") or item.get("text") or item.get("token")
                    if not isinstance(form_value, str) or not form_value.strip():
                        raise ValueError(f"{path}:{line_no}: token {idx} has no non-empty string form")
                    form = form_value.strip()
                    lemma = item.get("lemma") or ""
                    upos = item.get("upos") or item.get("pos") or ""
                    token_id = item.get("id") or idx + 1
                    if not isinstance(lemma, str) or not isinstance(upos, str):
                        raise ValueError(f"{path}:{line_no}: token {idx} lemma/POS must be strings")
                    if isinstance(token_id, bool) or not isinstance(token_id, (str, int)):
                        raise ValueError(f"{path}:{line_no}: token {idx} id must be a string or integer")
                    raw_feats = item.get("feats") or item.get("features") or {}
                    if not isinstance(raw_feats, dict) or any(
                        not isinstance(key, str) or not isinstance(value, str)
                        for key, value in raw_feats.items()
                    ):
                        raise ValueError(f"{path}:{line_no}: token {idx} features must map strings to strings")
                    tokens.append(
                        Token(
                            form=form,
                            lemma=lemma,
                            upos=upos,
                            feats=dict(raw_feats),
                            token_id=str(token_id).strip(),
                            index=idx,
                        )
                    )
                else:
                    raise ValueError(f"{path}:{line_no}: token {idx} must be string or object")
                if tokens[-1].token_id in seen_token_ids:
                    raise ValueError(
                        f"{path}:{line_no}: duplicate token ID {tokens[-1].token_id!r}"
                    )
                seen_token_ids.add(tokens[-1].token_id)
                if not tokens[-1].token_id:
                    raise ValueError(f"{path}:{line_no}: token {idx} ID must not be empty")
            text_value = row.get("text") or " ".join(token.form for token in tokens)
            sentence_id_value = row.get("sentence_id") or row.get("id") or str(line_no)
            if not isinstance(text_value, str) or not text_value.strip():
                raise ValueError(f"{path}:{line_no}: sentence text must be a non-empty string")
            if isinstance(sentence_id_value, bool) or not isinstance(sentence_id_value, (str, int)):
                raise ValueError(f"{path}:{line_no}: sentence ID must be a string or integer")
            text = text_value.strip()
            sentence_id = str(sentence_id_value).strip()
            if not sentence_id:
                raise ValueError(f"{path}:{line_no}: sentence ID must not be empty")
            if sentence_id in seen_sentence_ids:
                raise ValueError(f"{path}:{line_no}: duplicate sentence ID {sentence_id!r}")
            seen_sentence_ids.add(sentence_id)
            yield Sentence(sentence_id=sentence_id, text=text, tokens=tokens)


def load_disambiguator(model_name: str, top: int):
    try:
        from camel_tools.disambig.mle import MLEDisambiguator
    except ImportError as exc:
        raise RuntimeError("CAMeL Tools with MLE disambiguation support is required") from exc

    return MLEDisambiguator.pretrained(model_name, top=top)


def camel_tools_version() -> str:
    try:
        return importlib.metadata.version("camel-tools")
    except importlib.metadata.PackageNotFoundError:
        return "unknown-unpackaged"


def selected_analysis(disambig_word: Any) -> tuple[dict[str, Any] | None, float | None, int]:
    analyses = getattr(disambig_word, "analyses", [])
    if not analyses:
        return None, None, 0
    scored = analyses[0]
    analysis = getattr(scored, "analysis", None)
    if not isinstance(analysis, dict):
        raise ValueError("CAMeL returned an analysis that is not an object")
    score = float(scored.score)
    if not math.isfinite(score):
        raise ValueError("CAMeL returned a non-finite analysis score")
    return dict(analysis), score, len(analyses)


def clean_value(value: Any, *, missing_na: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"CAMeL analysis value must be scalar, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("CAMeL analysis value must be finite")
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
    model_name: str,
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
        "source": f"camel_tools_disambig_mle_{model_name}",
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
    *,
    input_path: Path | None = None,
    input_format: str | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if top <= 0:
        raise ValueError("top must be greater than zero")
    if limit_sentences is not None and limit_sentences <= 0:
        raise ValueError("limit_sentences must be greater than zero")
    if limit_records is not None and limit_records <= 0:
        raise ValueError("limit_records must be greater than zero")
    if output.resolve() == report_path.resolve():
        raise ValueError("output and report paths must be different")
    input_resolved = input_path.resolve() if input_path is not None else None
    input_identity = input_path.stat() if input_path is not None else None
    input_sha256 = sha256_file(input_path) if input_path is not None else None
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
    seen_analysis_ids: set[str] = set()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary_output = Path(temporary_name)
    try:
        out = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        with out:
            for sentence in sentences:
                if limit_sentences is not None and total_sentences >= limit_sentences:
                    break
                if not sentence.tokens or any(not token.form for token in sentence.tokens):
                    raise ValueError(f"sentence {sentence.sentence_id!r} has empty token data")
                total_sentences += 1
                forms = [token.form for token in sentence.tokens]
                total_tokens += len(forms)
                disambig_words = list(disambiguator.disambiguate(forms))
                if len(disambig_words) != len(sentence.tokens):
                    raise ValueError(
                        f"CAMeL returned {len(disambig_words)} results for "
                        f"{len(sentence.tokens)} tokens in sentence {sentence.sentence_id!r}"
                    )
                for token, disambig_word in zip(sentence.tokens, disambig_words, strict=True):
                    analysis, score, num_returned = selected_analysis(disambig_word)
                    if analysis is None:
                        if token.upos in ELIGIBLE_UPOS:
                            skipped_no_analysis += 1
                        continue
                    if not should_keep_token(token, analysis, require_source_upos):
                        continue
                    eligible_tokens += 1
                    record = camel_record(
                        sentence, token, analysis, score, num_returned, model_name
                    )
                    analysis_id = record["analysis_id"]
                    if analysis_id in seen_analysis_ids:
                        raise ValueError(f"duplicate generated analysis ID {analysis_id!r}")
                    seen_analysis_ids.add(analysis_id)
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
                    out.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    records_written += 1
                    if limit_records is not None and records_written >= limit_records:
                        break
                if limit_records is not None and records_written >= limit_records:
                    break
            out.flush()
            os.fsync(out.fileno())
        if records_written == 0 and not allow_empty:
            raise ValueError("no eligible disambiguated records were produced")
        if input_path is not None and input_identity is not None:
            after = input_path.stat()
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if any(
                getattr(input_identity, field) != getattr(after, field)
                for field in fields
            ):
                raise RuntimeError("input changed while CAMeL export was running")
        os.replace(temporary_output, output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    if records_written != disambiguated_tokens or records_written != eligible_tokens:
        raise RuntimeError("internal export counters disagree")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
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
        "input_path": str(input_resolved) if input_resolved is not None else None,
        "input_sha256": input_sha256,
        "input_format": input_format,
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "camel_tools_version": camel_tools_version(),
        "examples": examples,
    }
    _atomic_write_json(report_path, report)
    return report


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument(
        "--allow-camel-pos-eligibility",
        action="store_true",
        help="for JSONL or unlabeled input, keep tokens whose selected CAMeL POS is NOUN/VERB/ADJ",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    if not input_path.is_file():
        parser.error(f"input does not exist: {input_path}")
    resolved_paths = [input_path.resolve(), output_path.resolve(), report_path.resolve()]
    if len(set(resolved_paths)) != len(resolved_paths):
        parser.error("input, output, and report paths must be distinct")
    if args.top <= 0:
        parser.error("--top must be greater than zero")
    if args.limit_sentences is not None and args.limit_sentences <= 0:
        parser.error("--limit-sentences must be greater than zero")
    if args.limit_records is not None and args.limit_records <= 0:
        parser.error("--limit-records must be greater than zero")
    sentences = iter_conllu(input_path) if args.input_format == "conllu" else iter_sentence_jsonl(input_path)
    try:
        report = export_sentences(
            sentences=sentences,
            output=output_path,
            report_path=report_path,
            model_name=args.model_name,
            top=args.top,
            limit_sentences=args.limit_sentences,
            limit_records=args.limit_records,
            require_source_upos=not args.allow_camel_pos_eligibility,
            input_path=input_path,
            input_format=args.input_format,
            allow_empty=args.allow_empty,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {k: report[k] for k in ["records_written", "eligible_tokens", "total_tokens"]},
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
