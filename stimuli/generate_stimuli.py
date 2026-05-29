"""build and validate the nonce root-pattern stimulus set.

reads nonce roots from the Alakeel et al. (2026) productivity dataset,
crosses them with arabic morphological patterns, and generates
prompt-rendered stimuli for probing experiments.
"""

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Arabic → Latin transliteration (ASCII-safe for LLM probing)
# ---------------------------------------------------------------------------
AR_TO_LATIN = {
    "ب": "b",
    "ت": "t",
    "ث": "th",
    "ج": "j",
    "ح": "H",      # pharyngeal voiceless fricative
    "خ": "kh",
    "د": "d",
    "ذ": "dh",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "S",      # emphatic s
    "ط": "T",      # emphatic t
    "ظ": "Z",      # emphatic dh/z
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "ء": "'",      # hamza
    "ع": "3",      # ayn
    "و": "w",
    "ي": "y",
    "ل": "l",      # in case dataset uses ل
    "ر": "r",      # in case dataset uses ر
}

# ---------------------------------------------------------------------------
# Morphological patterns (fa3ala notation)
# ---------------------------------------------------------------------------
PATTERNS = [
    # Basic verb forms
    ("fa3ala",     "basic past (form I)"),
    ("yaf3alu",    "basic present (form I)"),
    # Participles
    ("fā3il",      "active participle (form I)"),
    ("maf3ūl",     "passive participle (form I)"),
    # Intensive / professional
    ("fa33āl",     "intensive/professional noun"),
    # Derived verb forms
    ("ifta3ala",   "form VIII past (reflexive)"),
    ("infa3ala",   "form VII past (passive-reflexive)"),
    ("istaf3ala",  "form X past (requestative)"),
    # Verbal nouns
    ("tafā3ul",    "verbal noun form VI"),
    ("mufā3ala",   "verbal noun form III"),
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "en_zero": 'Apply the Arabic pattern "{pattern}" to the root "{root}". Output only the resulting transliterated word.',
    "en_one":  'Apply the Arabic pattern "{pattern}" to the root "{root}". '
               'Example: applying "fa3ala" to "k-t-b" gives "kataba". '
               'Output only the resulting transliterated word.',
    "ar_zero": 'طبق النمط "{pattern}" على الجذر "{root}". أخرج الناتج بالحروف اللاتينية فقط.',
    "ar_one":  'طبق النمط "{pattern}" على الجذر "{root}". '
               'مثال: تطبيق "fa3ala" على الجذر "k-t-b" يعطي "kataba". '
               'أخرج الناتج بالحروف اللاتينية فقط.',
}

ABLATION_PROMPT_TEMPLATES = {
    "root_masked": {
        "en_zero": 'Apply the Arabic pattern "{pattern}" to the root "[ROOT]". Output only the resulting transliterated word.',
        "en_one":  'Apply the Arabic pattern "{pattern}" to the root "[ROOT]". '
                   'Example: applying "fa3ala" to "k-t-b" gives "kataba". '
                   'Output only the resulting transliterated word.',
        "ar_zero": 'طبق النمط "{pattern}" على الجذر "[ROOT]". أخرج الناتج بالحروف اللاتينية فقط.',
        "ar_one":  'طبق النمط "{pattern}" على الجذر "[ROOT]". '
                   'مثال: تطبيق "fa3ala" على الجذر "k-t-b" يعطي "kataba". '
                   'أخرج الناتج بالحروف اللاتينية فقط.',
    },
    "pattern_masked": {
        "en_zero": 'Apply the Arabic pattern "[PATTERN]" to the root "{root}". Output only the resulting transliterated word.',
        "en_one":  'Apply the Arabic pattern "[PATTERN]" to the root "{root}". '
                   'Example: applying "fa3ala" to "k-t-b" gives "kataba". '
                   'Output only the resulting transliterated word.',
        "ar_zero": 'طبق النمط "[PATTERN]" على الجذر "{root}". أخرج الناتج بالحروف اللاتينية فقط.',
        "ar_one":  'طبق النمط "[PATTERN]" على الجذر "{root}". '
                   'مثال: تطبيق "fa3ala" على الجذر "k-t-b" يعطي "kataba". '
                   'أخرج الناتج بالحروف اللاتينية فقط.',
    },
    "both_masked": {
        "en_zero": 'Apply the Arabic pattern "[PATTERN]" to the root "[ROOT]". Output only the resulting transliterated word.',
        "en_one":  'Apply the Arabic pattern "[PATTERN]" to the root "[ROOT]". '
                   'Example: applying "fa3ala" to "k-t-b" gives "kataba". '
                   'Output only the resulting transliterated word.',
        "ar_zero": 'طبق النمط "[PATTERN]" على الجذر "[ROOT]". أخرج الناتج بالحروف اللاتينية فقط.',
        "ar_one":  'طبق النمط "[PATTERN]" على الجذر "[ROOT]". '
                   'مثال: تطبيق "fa3ala" على الجذر "k-t-b" يعطي "kataba". '
                   'أخرج الناتج بالحروف اللاتينية فقط.',
    },
    "fake_pattern": {
        "en_zero": 'Apply the Arabic pattern "CVCCVC" to the root "{root}". Output only the resulting transliterated word.',
        "en_one":  'Apply the Arabic pattern "CVCCVC" to the root "{root}". '
                   'Example: applying "fa3ala" to "k-t-b" gives "kataba". '
                   'Output only the resulting transliterated word.',
        "ar_zero": 'طبق النمط "CVCCVC" على الجذر "{root}". أخرج الناتج بالحروف اللاتينية فقط.',
        "ar_one":  'طبق النمط "CVCCVC" على الجذر "{root}". '
                   'مثال: تطبيق "fa3ala" على الجذر "k-t-b" يعطي "kataba". '
                   'أخرج الناتج بالحروف اللاتينية فقط.',
    },
}


def load_nonce_roots(source_path: str | None) -> list[str]:
    """load unique nonce roots from a source file, or return defaults.

    supports:
    - alakeel productivity_dataset.json (key: "nonce_roots", field: "root")
    - plain text file, one root per line (dash-separated or dot-separated)

    returns list of roots in dash-separated latin format (e.g. "q-l-z").
    """
    if source_path is None:
        return [
            "q-l-z", "b-r-sh", "k-m-d", "s-t-f", "j-h-n",
            "z-r-q", "f-l-m", "d-r-s", "m-l-k", "n-b-t",
        ]

    path = Path(source_path)
    text = path.read_text(encoding="utf-8")

    # try JSON (alakeel format)
    try:
        data = json.loads(text)
        nonce_items = data.get("nonce_roots", [])
        if nonce_items:
            unique = sorted(set(item["root"] for item in nonce_items))
            return [dot_to_dash(r) for r in unique]
    except (json.JSONDecodeError, KeyError):
        pass

    # fallback: plain text, one root per line
    roots = [line.strip() for line in text.splitlines() if line.strip()]
    return [dot_to_dash(r) for r in roots if "-" in r or "." in r]


def dot_to_dash(root: str) -> str:
    """convert dot-separated arabic or latin root to dash-separated latin.

    e.g. "ط.د.غ" → "t-d-gh"  or  "t.d.gh" → "t-d-gh"
    """
    parts = root.split(".")
    latin_parts = [AR_TO_LATIN.get(p, p) for p in parts]
    return "-".join(latin_parts)


def apply_pattern(root: str, pattern: str) -> str:
    """apply a root to a fa3ala-notation pattern template.

    root is dash-separated latin (e.g. "k-t-b")
    pattern uses f/3/l placeholders (e.g. "fa3ala", "maf3ūl")
    """
    consonants = root.split("-")
    if len(consonants) < 3:
        raise ValueError(f"root '{root}' has fewer than 3 consonants")

    f, ayn, l = consonants[0], consonants[1], consonants[2]
    surface = pattern.replace("f", f).replace("3", ayn).replace("l", l)
    # handle double-char cases (e.g. "sh" → need to not split)
    # for now, the simple replace works for single-char latin mappings
    return surface


def generate_stimuli(
    nonce_roots: list[str],
    patterns: list[tuple[str, str]],
) -> list[dict]:
    """cross nonce roots with patterns to build the stimulus set."""
    stimuli = []
    for root in nonce_roots:
        consonants = root.split("-")
        if len(consonants) < 3:
            print(f"warning: skipping root '{root}' (< 3 consonants)")
            continue
        for pattern, description in patterns:
            surface = apply_pattern(root, pattern)
            stimuli.append({
                "root": root,
                "root_consonants": consonants,
                "pattern": pattern,
                "pattern_description": description,
                "expected_surface": surface,
            })
    return stimuli


def render_prompts(stimuli: list[dict], include_ablations: bool = False) -> list[dict]:
    """add prompt strings for each template to every stimulus."""
    for s in stimuli:
        s["prompts"] = {}
        for name, tmpl in PROMPT_TEMPLATES.items():
            s["prompts"][name] = tmpl.format(
                root=s["root"],
                pattern=s["pattern"],
            )
        if include_ablations:
            for ablation_name, templates in ABLATION_PROMPT_TEMPLATES.items():
                for name, tmpl in templates.items():
                    s["prompts"][f"{name}_{ablation_name}"] = tmpl.format(
                        root=s["root"],
                        pattern=s["pattern"],
                    )
    return stimuli


def validate_stimuli(
    stimuli: list[dict],
    lexicon_path: str | None = None,
) -> list[dict]:
    """validate that expected surface forms are not real arabic words."""
    if lexicon_path is None:
        print("warning: no lexicon provided, skipping collision check")
        return stimuli

    with open(lexicon_path, encoding="utf-8") as f:
        lexicon = set(line.strip() for line in f)

    collisions = [s for s in stimuli if s["expected_surface"] in lexicon]
    if collisions:
        print(f"warning: {len(collisions)} stimuli collide with real words:")
        for c in collisions[:10]:
            print(f"  {c['root']} + {c['pattern']} = {c['expected_surface']}")

    return [s for s in stimuli if s["expected_surface"] not in lexicon]


def compute_stats(stimuli: list[dict]):
    """print summary statistics."""
    roots = set(s["root"] for s in stimuli)
    patterns = set(s["pattern"] for s in stimuli)
    print(f"\n--- stimulus set summary ---")
    print(f"total stimuli:   {len(stimuli)}")
    print(f"unique roots:    {len(roots)}")
    print(f"unique patterns: {len(patterns)}")
    print(f"matrix:          {len(roots)} roots × {len(patterns)} patterns")
    # per-pattern breakdown
    for pat, desc in PATTERNS:
        count = sum(1 for s in stimuli if s["pattern"] == pat)
        print(f"  {pat:12s} ({desc}) → {count} stimuli")


def main():
    parser = argparse.ArgumentParser(
        description="generate nonce root-pattern stimuli"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="path to alakeel productivity_dataset.json or root list file",
    )
    parser.add_argument(
        "--output",
        default="stimuli/nonce_root_pattern.json",
        help="output path for stimulus json",
    )
    parser.add_argument(
        "--lexicon",
        default=None,
        help="path to arabic word list for collision validation",
    )
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="include masked and fake-pattern control prompt templates",
    )
    args = parser.parse_args()

    # load roots
    nonce_roots = load_nonce_roots(args.source)
    print(f"loaded {len(nonce_roots)} nonce roots")

    # generate and render
    stimuli = generate_stimuli(nonce_roots, PATTERNS)
    stimuli = render_prompts(stimuli, include_ablations=args.include_ablations)
    stimuli = validate_stimuli(stimuli, args.lexicon)

    compute_stats(stimuli)

    # write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stimuli, f, ensure_ascii=False, indent=2)

    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
