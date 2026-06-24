#!/usr/bin/env python3
"""Export gguf-parity-tools token-audit metadata from a tokenizer JSON file.

This performs tokenizer JSON encoding only. It does not generate text, compute
logits, or extract hidden states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


PROVENANCE = {
    "real_llama_cpp": False,
    "real_tokenization": True,
    "no_generation": True,
    "no_logits": True,
    "no_hidden_states": True,
    "not_research_output": True,
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def render_prompt(template: str, row: dict) -> str:
    rendered = template
    for key, value in row.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-template", default="{prompt}")
    parser.add_argument("--sample-id-field", default="id")
    parser.add_argument("--engine", default="tokenizer-json")
    parser.add_argument("--arch", default="qwen3")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt_rows = []
    for index, row in enumerate(read_jsonl(Path(args.prompts))):
        prompt = render_prompt(args.prompt_template, row)
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_rows.append(
            {
                "index": index,
                "id": str(row.get(args.sample_id_field, index)),
                "prompt": prompt,
                "token_ids": encoded.ids,
            }
        )

    metadata = {
        "engine": args.engine,
        "tokenizer": args.tokenizer,
        "arch": args.arch,
        "prompts": prompt_rows,
        "purpose": "gguf-parity-tools token-audit metadata",
        **PROVENANCE,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
