"""
generate_stimuli.py — Build and validate the nonce root-pattern stimulus set.

Sources:
    - Alakeel et al. public dataset: https://github.com/YaraAlakeel/morphems_without_borders
    - Pattern templates from standard Arabic morphology references

Usage:
    python stimuli/generate_stimuli.py --source data/alakeel2026/ --output stimuli/nonce_root_pattern.json
"""

import argparse
import json
from pathlib import Path


# Common Arabic verb/noun patterns (fa3ala notation)
# f = first radical, 3 = second radical (ayn), l = third radical
# Trimmed to 5 patterns per 1-week PLAN.md scope
PATTERNS = [
    "fa3ala",     # فعل - basic past (he did)
    "yaf3alu",    # يفعل - basic present (he does)
    "fā3il",      # فاعل - active participle (doer)
    "maf3ūl",     # مفعول - passive participle (done)
    "fa33āl",     # فعّال - intensive/professional (doer intensively)
]


def generate_stimuli(nonce_roots, patterns):
    """Cross nonce roots with patterns to build stimulus set.
    
    Args:
        nonce_roots: list of root strings like ["q-l-z", "b-r-sh", ...]
        patterns: list of pattern strings like ["fa3ala", ...]
    
    Returns:
        list of dicts: {root, pattern, expected_surface_form}
    """
    stimuli = []
    for root in nonce_roots:
        f, ayn, l = root.split("-")
        for pattern in patterns:
            surface = apply_pattern(root, pattern, f, ayn, l)
            stimuli.append({
                "root": root,
                "pattern": pattern,
                "expected": surface,
                "root_consonants": [f, ayn, l],
            })
    return stimuli


def apply_pattern(root, pattern, f, ayn, l):
    """Apply a root to a pattern template.
    
    Simple substitution of f/3/l into pattern slots.
    TODO: handle pattern-specific phonological rules (e.g., hamza insertion,
    weak radical behavior, assimilation).
    """
    surface = pattern.replace("f", f).replace("3", ayn).replace("l", l)
    return surface


def validate_stimuli(stimuli, lexicon_path=None):
    """Validate that expected surface forms are not real Arabic words.
    
    If lexicon_path is provided, check each expected form against it.
    Otherwise, warn and proceed.
    """
    if lexicon_path is None:
        print("WARNING: No lexicon provided. Cannot validate against real words.")
        return stimuli
    
    with open(lexicon_path) as f:
        lexicon = set(line.strip() for line in f)
    
    valid = []
    collisions = []
    for s in stimuli:
        if s["expected"] in lexicon:
            collisions.append(s)
        else:
            valid.append(s)
    
    if collisions:
        print(f"WARNING: {len(collisions)} stimuli collide with real words:")
        for c in collisions[:10]:
            print(f"  {c['root']} + {c['pattern']} = {c['expected']}")
    
    return valid


def main():
    parser = argparse.ArgumentParser(description="Generate nonce root-pattern stimuli")
    parser.add_argument("--source", help="Path to Alakeel et al. dataset")
    parser.add_argument("--output", default="stimuli/nonce_root_pattern.json")
    parser.add_argument("--lexicon", default=None, help="Path to Arabic word list for validation")
    args = parser.parse_args()
    
    # TODO: load nonce roots from Alakeel et al. dataset
    # For now, use placeholder roots
    nonce_roots = [
        "q-l-z", "b-r-sh", "k-m-d", "s-t-f", "j-h-n",
        "z-r-q", "f-l-m", "d-r-s", "m-l-k", "n-b-t",
    ]
    
    stimuli = generate_stimuli(nonce_roots, PATTERNS)
    stimuli = validate_stimuli(stimuli, args.lexicon)
    
    print(f"Generated {len(stimuli)} stimuli ({len(nonce_roots)} roots × {len(PATTERNS)} patterns)")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stimuli, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
