"""build and validate the nonce root-pattern stimulus set."""

import argparse
import json
from pathlib import Path


# arabic verb/noun patterns (fa3ala notation).
# f = first radical, 3 = second radical (ayn), l = third radical
# trimmed to 5 patterns per 1-week scope
PATTERNS = [
    "fa3ala",     # فعل - basic past
    "yaf3alu",    # يفعل - basic present
    "fā3il",      # فاعل - active participle
    "maf3ūl",     # مفعول - passive participle
    "fa33āl",     # فعّال - intensive/professional
]


def generate_stimuli(nonce_roots, patterns):
    """cross nonce roots with patterns to build the stimulus set."""
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
    """apply a root to a pattern template via f/3/l substitution.
    TODO: handle pattern-specific phonological rules (hamza insertion,
    weak radical behavior, assimilation).
    """
    surface = pattern.replace("f", f).replace("3", ayn).replace("l", l)
    return surface


def validate_stimuli(stimuli, lexicon_path=None):
    """validate that expected surface forms are not real arabic words."""
    if lexicon_path is None:
        print("warning: no lexicon provided, skipping collision check")
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
        print(f"warning: {len(collisions)} stimuli collide with real words:")
        for c in collisions[:10]:
            print(f"  {c['root']} + {c['pattern']} = {c['expected']}")

    return valid


def main():
    parser = argparse.ArgumentParser(description="generate nonce root-pattern stimuli")
    parser.add_argument("--source", help="path to alakeel et al. dataset")
    parser.add_argument("--output", default="stimuli/nonce_root_pattern.json")
    parser.add_argument("--lexicon", default=None, help="path to arabic word list for validation")
    args = parser.parse_args()

    # TODO: load nonce roots from alakeel et al. dataset
    # for now, use placeholder roots
    nonce_roots = [
        "q-l-z", "b-r-sh", "k-m-d", "s-t-f", "j-h-n",
        "z-r-q", "f-l-m", "d-r-s", "m-l-k", "n-b-t",
    ]

    stimuli = generate_stimuli(nonce_roots, PATTERNS)
    stimuli = validate_stimuli(stimuli, args.lexicon)

    print(f"generated {len(stimuli)} stimuli ({len(nonce_roots)} roots x {len(PATTERNS)} patterns)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stimuli, f, ensure_ascii=False, indent=2)

    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
