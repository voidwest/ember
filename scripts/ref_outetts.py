#!/usr/bin/env python3
"""Reference capture for the OuteTTS LLM half (Track E5 ladder).

Runs the SAME Q8_0 GGUF ember uses through llama-cpp-python (the reference
inference path for outetts GGUF models) with GREEDY sampling, capturing:
  prompt_ids.npy    HF-tokenized prompt (add_special_tokens=False)
  gen_ids.npy       llama.cpp generated token ids (greedy, streamed)
  codes.npy         codec values extracted from audio-code tokens
  manifest.json     text/prompt/counts

Ember's side must reproduce prompt ids bit-exactly and generated ids
step-identically; the codec half has its own validated ladder.

Usage:
    python scripts/ref_outetts.py <model.gguf> <tokenizer_dir> <out_dir>         --text "..." [--max-tokens N]
"""
import argparse
import json
import os
import re

import numpy as np
from transformers import AutoTokenizer
from llama_cpp import Llama


def build_prompt(text: str) -> tuple[str, list[str]]:
    """Reference v1 interface processing (outetts 0.2.x, language=en)."""
    import inflect

    lec = inflect.engine()
    t = text.lower()
    t = re.sub(r"\d+(\.\d+)?", lambda x: lec.number_to_words(x.group()), t)
    t = re.sub(r"[-_/,\.\\]", " ", t)
    t = re.sub(r"[^a-z\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    words = [w.strip() for w in t.split()]
    prompt = (
        "<|im_start|>\n<|text_start|>"
        + "<|text_sep|>".join(words)
        + "<|text_end|>\n<|audio_start|>\n"
    )
    return prompt, words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tokenizer_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    prompt, words = build_prompt(args.text)
    print("words:", words[:12], "..." if len(words) > 12 else "")
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    print("prompt tokens:", len(prompt_ids))

    llm = Llama(model_path=args.model, n_ctx=4096, logits_all=False,
                verbose=False, seed=0)

    # Greedy loop through llama.cpp's own sampler (temp=0/top_k=1 == argmax).
    # eval_logits-based argmax proved unreliable on this build; sample() with
    # temp=0 is deterministic and is llama.cpp's own notion of greedy.
    toks = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
    assert toks == list(prompt_ids), (
        f"llama tokenizer disagrees with HF: {len(toks)} vs {len(prompt_ids)}"
    )
    ids_stream = []
    pieces = []
    llm.reset()
    llm.eval(toks)
    eos_id = int(llm.token_eos())
    for _ in range(args.max_tokens):
        nxt = int(llm.sample(temp=0.0, top_k=1))
        ids_stream.append(nxt)
        pieces.append(llm.detokenize([nxt]).decode("utf-8", errors="replace"))
        if nxt == eos_id:
            break
        llm.eval([nxt])

    gen_text = "".join(pieces)
    print("generated", len(ids_stream), "ids")

    code_map = {}
    for i in range(4096):
        enc = tok.encode(f"<|{i}|>", add_special_tokens=False)
        if len(enc) == 1:
            code_map[enc[0]] = i
    codes = [code_map[i] for i in ids_stream if i in code_map]
    print("codec codes:", len(codes))
    print("gen text preview:", gen_text[:160].replace("\n", "|"))

    np.save(os.path.join(args.out_dir, "prompt_ids.npy"), np.array(prompt_ids, dtype=np.int64))
    np.save(os.path.join(args.out_dir, "gen_ids.npy"), np.array(ids_stream, dtype=np.int64))
    np.save(os.path.join(args.out_dir, "codes.npy"), np.array(codes, dtype=np.int64))
    manifest = {
        "text": args.text,
        "prompt": prompt,
        "gen_text": gen_text,
        "n_prompt": len(prompt_ids),
        "n_gen": len(ids_stream),
        "n_codes": len(codes),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
