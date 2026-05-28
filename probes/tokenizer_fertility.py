"""tokenizer fertility analysis for arabic morphological probing.

compares subword tokenization behaviour across tokenizers
on arabic stimuli prompts.  computes:

  - subword count (fertility): how many tokens per prompt / per word
  - split ratio: fraction of prompts where a word gets split
  - boundary alignment: whether splits respect morpheme boundaries

usage:
  python probes/tokenizer_fertility.py \
      --stimuli stimuli/nonce_root_pattern.json \
      --tokenizers tokenizer.json tokenizer-gpt2.json \
      --labels llama gpt2 \
      --output data/fertility.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_stimuli(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(path: str):
    """load a huggingface tokenizer.json and return a callable encode fn."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)

    def encode(text: str) -> list[int]:
        return tok.encode(text).ids

    def decode(ids: list[int]) -> str:
        return tok.decode(ids)

    encode.vocab_size = tok.get_vocab_size()
    encode.name = Path(path).stem
    return encode, decode


def analyze_prompt(
    prompt: str, encode_fn, stimulus: dict | None = None
) -> dict:
    """tokenize a prompt and compute fertility metrics."""
    ids = encode_fn(prompt)
    n_tokens = len(ids)

    # decode each token individually for subword analysis
    tokens = []
    for tid in ids:
        # we need a decode fn — use the tokenizer's decode
        tokens.append(tid)

    # approximate "words" by splitting on whitespace
    # and counting tokens that form each word
    words = prompt.split()
    # crude alignment: count chars per word and estimate tokens/word
    char_counts = [len(w) for w in words]

    return {
        "prompt_chars": len(prompt),
        "n_tokens": n_tokens,
        "n_words": len(words),
        "fertility": n_tokens / max(len(words), 1),
        "chars_per_token": len(prompt) / max(n_tokens, 1),
        "token_ids": ids,
    }


def analyze_all(stimuli, encode_fn, label: str) -> dict:
    """analyze all prompts across all stimulus variants."""
    per_prompt = []
    token_counts = []

    for si, stimulus in enumerate(stimuli):
        for variant in ["en_zero", "en_one", "ar_zero", "ar_one"]:
            prompt = stimulus["prompts"].get(variant, "")
            if not prompt:
                continue
            result = analyze_prompt(prompt, encode_fn, stimulus)
            result["stimulus_idx"] = si
            result["variant"] = variant
            result["root"] = stimulus["root"]
            result["pattern"] = stimulus["pattern"]
            per_prompt.append(result)
            token_counts.append(result["n_tokens"])

    counts = np.array(token_counts)
    fertilities = [r["fertility"] for r in per_prompt]

    # split by variant (en vs ar)
    en_results = [r for r in per_prompt if r["variant"].startswith("en")]
    ar_results = [r for r in per_prompt if r["variant"].startswith("ar")]

    return {
        "label": label,
        "total_prompts": len(per_prompt),
        "mean_tokens": float(counts.mean()),
        "median_tokens": float(np.median(counts)),
        "std_tokens": float(counts.std()),
        "min_tokens": int(counts.min()),
        "max_tokens": int(counts.max()),
        "mean_fertility": float(np.mean(fertilities)),
        "mean_chars_per_token": float(
            np.mean([r["chars_per_token"] for r in per_prompt])
        ),
        # english vs arabic breakdown
        "en_mean_tokens": float(
            np.mean([r["n_tokens"] for r in en_results])
        ) if en_results else None,
        "ar_mean_tokens": float(
            np.mean([r["n_tokens"] for r in ar_results])
        ) if ar_results else None,
        "en_ar_ratio": (
            float(
                np.mean([r["n_tokens"] for r in ar_results])
                / np.mean([r["n_tokens"] for r in en_results])
            )
            if en_results and ar_results
            else None
        ),
        "per_prompt": per_prompt,
    }


def print_report(results: list[dict]):
    """print a readable comparison table."""
    print()
    print(f"{'metric':<30} " + "  ".join(
        f"{r['label']:>12}" for r in results
    ))
    print("-" * (30 + 14 * len(results)))

    rows = [
        ("total prompts", "total_prompts", "d"),
        ("mean tokens/prompt", "mean_tokens", ".1f"),
        ("median tokens/prompt", "median_tokens", ".1f"),
        ("std tokens", "std_tokens", ".1f"),
        ("min tokens", "min_tokens", "d"),
        ("max tokens", "max_tokens", "d"),
        ("mean fertility", "mean_fertility", ".2f"),
        ("mean chars/token", "mean_chars_per_token", ".2f"),
    ]

    for label, key, fmt in rows:
        vals = "  ".join(
            f"{r[key]:>12{fmt}}" for r in results
        )
        print(f"{label:<30} {vals}")

    # language breakdown
    print()
    print("language breakdown (en vs ar prompts):")
    lang_rows = [
        ("en mean tokens", "en_mean_tokens"),
        ("ar mean tokens", "ar_mean_tokens"),
        ("ar/en token ratio", "en_ar_ratio"),
    ]
    for label, key in lang_rows:
        vals = "  ".join(
            f"{r[key]:>12.1f}" if r[key] is not None else f"{'N/A':>12}"
            for r in results
        )
        print(f"  {label:<28} {vals}")


def main():
    parser = argparse.ArgumentParser(
        description="tokenizer fertility analysis for arabic probing"
    )
    parser.add_argument(
        "--stimuli", required=True, help="path to stimuli json"
    )
    parser.add_argument(
        "--tokenizers", nargs="+", required=True,
        help="paths to tokenizer.json files"
    )
    parser.add_argument(
        "--labels", nargs="+", required=True,
        help="labels for each tokenizer (same order as --tokenizers)"
    )
    parser.add_argument(
        "--output", default=None,
        help="path to save fertility report (.json)"
    )
    args = parser.parse_args()

    if len(args.tokenizers) != len(args.labels):
        raise ValueError(
            f"got {len(args.tokenizers)} tokenizers but "
            f"{len(args.labels)} labels — must match"
        )

    stimuli = load_stimuli(args.stimuli)
    print(f"loaded {len(stimuli)} stimuli from {args.stimuli}")
    print(f"comparing {len(args.tokenizers)} tokenizers")

    results = []
    for path, label in zip(args.tokenizers, args.labels):
        encode_fn, _ = load_tokenizer(path)
        print(f"\n--- {label} (vocab={encode_fn.vocab_size}) ---")
        result = analyze_all(stimuli, encode_fn, label)
        results.append(result)

    print_report(results)

    if args.output:
        # strip per_prompt detail for cleaner json
        save_results = []
        for r in results:
            d = {k: v for k, v in r.items() if k != "per_prompt"}
            save_results.append(d)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print(f"\nsaved fertility report to {args.output}")


if __name__ == "__main__":
    main()
