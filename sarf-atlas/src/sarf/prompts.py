from __future__ import annotations

from typing import Iterable

from .schemas import PromptRecord


def make_prompts(records: Iterable[dict[str, str]], template: str) -> list[dict[str, str]]:
    rows = []
    for record in records:
        prompt = template.format(**record)
        rows.append(
            PromptRecord(
                id=record["id"],
                word=record["word"],
                prompt=prompt,
                target_pos=record.get("pos", ""),
                target_root=record.get("root", ""),
                target_pattern=record.get("pattern", ""),
                source_lemma=record.get("lemma", ""),
            ).to_dict()
        )
    return rows
