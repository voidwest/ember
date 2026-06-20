"""Token-level char n-gram baseline for heldout probes.

Tests whether probe accuracy on Llama's L0 embeddings can be explained
by the surface form of the last subword token (fragment identity leakage).

Usage:
  python3 probes/token_char_baseline.py \
    --stimuli data/arabic_morph_real/out_disambig_padt_5000_strict/stimuli_5000.json \
    --tokenizer tokenizer.json \
    --tasks pos features.gender features.number --folds 5
"""

import argparse, json, sys, numpy as np
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

SPECIAL_TOKENS = {'<s>', '</s>', '<pad>', '<unk>', '<|endoftext|>',
                  '<|im_start|>', '<|im_end|>', '<|begin_of_text|>',
                  '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>'}

def load_stimuli(path):
    if not Path(path).exists():
        path = str(Path(path).resolve())
    with open(path) as f:
        return json.load(f)

def extract_labels(stimuli, field, min_examples=3):
    labels = []
    for s in stimuli:
        v = s
        for part in field.split("."):
            if isinstance(v, dict):
                v = v.get(part, "")
            else:
                v = ""
        labels.append(str(v))
    counts = Counter(labels)
    keep = {l for l, c in counts.items() if c >= min_examples and l.strip()}
    indices = [i for i, l in enumerate(labels) if l in keep]
    return [labels[i] for i in indices], indices, dict(counts)

def get_last_tokens(prompts, tokenizer):
    """Return the last non-special subword token for each prompt."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer)
    last_tokens = []
    for prompt in prompts:
        enc = tok.encode(prompt)
        decoded = [tok.decode([i]) for i in enc.ids]
        non_special = [t for t in decoded if t.strip() and t not in SPECIAL_TOKENS]
        # filter trailing punctuation-only tokens
        while non_special and len(non_special[-1].strip()) <= 1 and non_special[-1].strip() in '.،:؛!?':
            non_special.pop()
        last_tokens.append(non_special[-1] if non_special else "")
    return last_tokens

def char_ngram_acc(tokens, labels, groups, n_folds, seed):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(set(y)) < 2:
        return 0.0
    
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 5), binary=True)
    X = vectorizer.fit_transform(tokens)
    
    if groups is not None:
        splitter = GroupKFold(n_splits=n_folds)
        splits = list(splitter.split(X, y, groups))
    else:
        effective = min(n_folds, int(np.bincount(y).min()))
        if effective < 2:
            # fallback: train accuracy
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X, y)
            return float(clf.score(X, y))
        splitter = StratifiedKFold(n_splits=effective, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))
    
    accs = []
    for train_idx, test_idx in splits:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(float(clf.score(X[test_idx], y[test_idx])))
    return float(np.mean(accs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tasks", nargs="+", default=["pos", "features.gender", "features.number"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-examples", type=int, default=3)
    args = parser.parse_args()
    
    stimuli = load_stimuli(args.stimuli)
    print(f"Loaded {len(stimuli)} stimuli")
    prompts = [s["prompts"]["morph_context"] for s in stimuli]
    
    # Extract last subword tokens
    last_tokens = get_last_tokens(prompts, args.tokenizer)
    print(f"Extracted {len(last_tokens)} last tokens")
    print(f"Sample: {last_tokens[:8]}")
    
    print(f"\n{'task':20s} {'split':20s} {'word-char':>8s} {'tok-char':>8s}")
    print("-" * 60)
    
    for task in args.tasks:
        field = task
        labels, indices, counts = extract_labels(stimuli, field, args.min_examples)
        if not labels or len(set(labels)) < 2:
            print(f"{task:20s} SKIPPED")
            continue
        
        task_tokens = [last_tokens[i] for i in indices]
        surfaces = [stimuli[i].get("surface_dediac", stimuli[i].get("surface", "")) for i in indices]
        lemmas = [stimuli[i].get("lemma", "") for i in indices]
        roots = [stimuli[i].get("root", "") for i in indices]
        
        # Word-level
        w_rand = char_ngram_acc(surfaces, labels, None, args.folds, args.seed)
        t_rand = char_ngram_acc(task_tokens, labels, None, args.folds, args.seed)
        print(f"{task:20s} {'random CV':20s} {w_rand*100:7.1f}% {t_rand*100:7.1f}%")
        
        # Lemma-heldout
        w_lemma = char_ngram_acc(surfaces, labels, lemmas, args.folds, args.seed)
        t_lemma = char_ngram_acc(task_tokens, labels, lemmas, args.folds, args.seed)
        print(f"{'':20s} {'lemma-heldout':20s} {w_lemma*100:7.1f}% {t_lemma*100:7.1f}%")
        
        # Root-heldout
        w_root = char_ngram_acc(surfaces, labels, roots, args.folds, args.seed)
        t_root = char_ngram_acc(task_tokens, labels, roots, args.folds, args.seed)
        print(f"{'':20s} {'root-heldout':20s} {w_root*100:7.1f}% {t_root*100:7.1f}%")
        
        print()

if __name__ == "__main__":
    main()
