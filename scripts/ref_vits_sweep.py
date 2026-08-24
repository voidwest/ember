#!/usr/bin/env python3
"""Reference waveforms for a list of Arabic texts, deterministic (noise=0)."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_vits import load  # noqa: E402


def main():
    model_dir, texts_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    tok, model = load(model_dir)
    texts = [json.loads(l)["text"] for l in open(texts_path, encoding="utf-8")]
    results = []
    for i, text in enumerate(texts):
        inputs = tok(text=text, return_tensors="pt")
        if inputs["input_ids"].shape[1] == 0:
            print(f"{i}: empty after tokenize, skip")
            continue
        with torch.no_grad():
            audio = model(**inputs).waveform.squeeze(0)
        arr = audio.to(torch.float32).numpy()
        np.save(os.path.join(out_dir, f"t{i:02d}.npy"), arr)
        results.append({"i": i, "text": text, "samples": int(arr.shape[0])})
        print(f"{i}: {arr.shape[0]} samples  {text[:24]}")
    json.dump(results, open(os.path.join(out_dir, "index.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
