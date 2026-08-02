"""build and validate the nonce root-pattern stimulus set.

reads nonce roots from the Alakeel et al. (2026) productivity dataset,
crosses them with arabic morphological patterns, and generates
prompt-rendered stimuli for probing experiments.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)

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

# Probe-safe views expose only the generated surface. They support asking
# whether root/pattern labels are recoverable from the form, whereas the
# composition prompts above explicitly supply both labels and are suitable only
# for behavioral generation or label-revealed positive controls.
SURFACE_PROBE_TEMPLATES = {
    "en_surface_probe": (
        'Analyze the transliterated Arabic nonce form "{expected_surface}". '
        "Infer its morphology without being given the root or pattern."
    ),
    "ar_surface_probe": (
        'حلّل الصيغة العربية الافتراضية المنقحرة "{expected_surface}" صرفيًا من دون إعطاء الجذر أو الوزن.'
    ),
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


def normalize_root(root: str) -> str:
    """Normalize a dot/dash-separated triliteral root to Latin dash notation."""
    if not isinstance(root, str) or not root.strip():
        raise ValueError("root must be a non-empty string")
    parts = [part.strip() for part in re.split(r"[.\-]", root.strip())]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(f"root {root!r} must contain exactly three dot/dash-separated radicals")
    latin_parts = [AR_TO_LATIN.get(part, part) for part in parts]
    if any(not re.fullmatch(r"[A-Za-z0-9']+", part) for part in latin_parts):
        raise ValueError(f"root {root!r} contains an unsupported radical")
    return "-".join(latin_parts)


def _unique_roots(values: list[Any], *, source: str) -> list[str]:
    roots: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise ValueError(f"{source} root {index + 1} must be a string")
        normalized = normalize_root(value)
        if normalized not in seen:
            roots.append(normalized)
            seen.add(normalized)
    if not roots:
        raise ValueError(f"{source} contains no usable roots")
    return roots


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
    if not path.is_file():
        raise FileNotFoundError(f"nonce-root source does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"nonce-root source is empty: {path}")

    # JSON sources are parsed strictly; malformed JSON must not silently turn
    # into a one-line plain-text root list.
    if path.suffix.lower() == ".json" or text.lstrip().startswith(("{", "[")):
        try:
            data = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid JSON root source {path}: {error}") from error
        if not isinstance(data, dict):
            raise ValueError(f"JSON root source {path} must be an object")
        nonce_items = data.get("nonce_roots")
        if not isinstance(nonce_items, list):
            raise ValueError(f"JSON root source {path} must contain a nonce_roots list")
        raw_roots: list[Any] = []
        for index, item in enumerate(nonce_items):
            if not isinstance(item, dict) or "root" not in item:
                raise ValueError(f"nonce_roots item {index + 1} must be an object with a root")
            raw_roots.append(item["root"])
        return sorted(_unique_roots(raw_roots, source=str(path)))

    # fallback: plain text, one root per line
    roots = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return _unique_roots(roots, source=str(path))


def dot_to_dash(root: str) -> str:
    """convert dot-separated arabic or latin root to dash-separated latin.

    e.g. "ط.د.غ" → "T-d-gh"  or  "t.d.gh" → "t-d-gh"
    """
    return normalize_root(root)


def apply_pattern(root: str, pattern: str) -> str:
    """apply a root to a fa3ala-notation pattern template.

    root is dash-separated latin (e.g. "k-t-b")
    pattern uses f/3/l placeholders (e.g. "fa3ala", "maf3ūl")
    """
    normalized = normalize_root(root)
    consonants = normalized.split("-")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a non-empty string")
    missing = {placeholder for placeholder in "f3l" if placeholder not in pattern}
    if missing:
        raise ValueError(f"pattern {pattern!r} is missing placeholders: {sorted(missing)}")

    f, ayn, l = consonants[0], consonants[1], consonants[2]
    # Substitute in one pass. Chained str.replace calls corrupt inserted
    # radicals that themselves contain a placeholder letter (notably f/l).
    replacements = {"f": f, "3": ayn, "l": l}
    return "".join(replacements.get(character, character) for character in pattern)


def generate_stimuli(
    nonce_roots: list[str],
    patterns: list[tuple[str, str]],
) -> list[dict]:
    """cross nonce roots with patterns to build the stimulus set."""
    stimuli = []
    for root in nonce_roots:
        root = normalize_root(root)
        consonants = root.split("-")
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
        for name, tmpl in SURFACE_PROBE_TEMPLATES.items():
            s["prompts"][name] = tmpl.format(expected_surface=s["expected_surface"])
        s["prompt_contracts"] = {
            **{
                name: {
                    "target_labels_in_prompt": True,
                    "revealed_targets": ["root", "pattern"],
                    "intended_use": "composition_behavior_or_positive_control",
                }
                for name in PROMPT_TEMPLATES
            },
            **{
                name: {
                    "target_labels_in_prompt": False,
                    "revealed_targets": [],
                    "intended_use": "label_free_representation_probe",
                }
                for name in SURFACE_PROBE_TEMPLATES
            },
        }
        if include_ablations:
            for ablation_name, templates in ABLATION_PROMPT_TEMPLATES.items():
                for name, tmpl in templates.items():
                    s["prompts"][f"{name}_{ablation_name}"] = tmpl.format(
                        root=s["root"],
                        pattern=s["pattern"],
                    )
                    s["prompt_contracts"][f"{name}_{ablation_name}"] = {
                        "target_labels_in_prompt": ablation_name
                        not in {"both_masked"},
                        "revealed_targets": {
                            "root_masked": ["pattern"],
                            "pattern_masked": ["root"],
                            "both_masked": [],
                            "fake_pattern": ["root"],
                        }[ablation_name],
                        "intended_use": "ablation_control",
                    }
        # Contracts describe the actual row prompt, including one-shot
        # examples that can accidentally equal this row's target class.
        for name, prompt in s["prompts"].items():
            revealed = [
                target
                for target in ("root", "pattern")
                if s[target] in prompt
            ]
            contract = s["prompt_contracts"][name]
            contract["revealed_targets"] = revealed
            contract["target_labels_in_prompt"] = bool(revealed)
    return stimuli


def augment_existing_stimuli(path: str) -> list[dict]:
    """Add audited surface-only prompts without rewriting historical rows."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"existing stimulus file does not exist: {source}")
    try:
        rows = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid stimulus JSON {source}: {error}") from error
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"existing stimulus file must be a non-empty JSON array: {source}")

    result: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"stimulus row {index} must be an object")
        root = row.get("root")
        pattern = row.get("pattern")
        surface = row.get("expected_surface")
        if not all(isinstance(value, str) and value for value in (root, pattern, surface)):
            raise ValueError(
                f"stimulus row {index} requires non-empty root, pattern, and expected_surface"
            )
        if normalize_root(root) != root:
            raise ValueError(f"stimulus row {index} root is not canonical: {root!r}")
        pair = (root, pattern)
        if pair in seen_pairs:
            raise ValueError(f"duplicate root/pattern pair at stimulus row {index}: {pair!r}")
        seen_pairs.add(pair)
        prompts = row.get("prompts")
        if not isinstance(prompts, dict) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(prompt, str)
            or not prompt
            for name, prompt in prompts.items()
        ):
            raise ValueError(f"stimulus row {index} prompts must map names to non-empty strings")

        updated = dict(row)
        updated_prompts = dict(prompts)
        for name, template in SURFACE_PROBE_TEMPLATES.items():
            updated_prompts[name] = template.format(expected_surface=surface)
        updated["prompts"] = updated_prompts

        contracts = {}
        for name, prompt in updated_prompts.items():
            revealed_targets = [
                target
                for target, label in (("root", root), ("pattern", pattern))
                if label in prompt
            ]
            contracts[name] = {
                "target_labels_in_prompt": bool(revealed_targets),
                "revealed_targets": revealed_targets,
                "intended_use": (
                    "composition_behavior_or_positive_control"
                    if revealed_targets
                    else "label_free_representation_probe"
                ),
            }
        updated["prompt_contracts"] = contracts
        result.append(updated)
    return result


def validate_stimuli(
    stimuli: list[dict],
    lexicon_path: str | None = None,
    *,
    collision_policy: str = "error",
) -> list[dict]:
    """Validate that transliterated surface forms do not collide with a lexicon."""
    if collision_policy not in {"error", "drop"}:
        raise ValueError("collision_policy must be 'error' or 'drop'")
    if lexicon_path is None:
        print("warning: no transliterated lexicon provided; collision status is unknown")
        return stimuli

    lexicon_source = Path(lexicon_path)
    if not lexicon_source.is_file():
        raise FileNotFoundError(f"lexicon does not exist: {lexicon_source}")
    with lexicon_source.open(encoding="utf-8") as f:
        lexicon = {
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        }
    if not lexicon:
        raise ValueError(f"lexicon contains no entries: {lexicon_source}")

    collisions = [s for s in stimuli if s["expected_surface"] in lexicon]
    if collisions:
        examples = ", ".join(
            f"{row['root']}+{row['pattern']}={row['expected_surface']}"
            for row in collisions[:10]
        )
        if collision_policy == "error":
            raise ValueError(
                f"{len(collisions)} generated forms collide with the lexicon: {examples}"
            )
        print(f"warning: explicitly dropping {len(collisions)} collisions: {examples}")

    return [s for s in stimuli if s["expected_surface"] not in lexicon]


def validate_matrix(stimuli: list[dict]) -> None:
    if not stimuli:
        raise ValueError("stimulus generation produced no rows")
    pairs = [(row.get("root"), row.get("pattern")) for row in stimuli]
    if any(not all(isinstance(value, str) and value for value in pair) for pair in pairs):
        raise ValueError("each stimulus requires non-empty root and pattern labels")
    if len(pairs) != len(set(pairs)):
        raise ValueError("stimulus set contains duplicate root/pattern pairs")


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


def file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def compute_stats(stimuli: list[dict], patterns: list[tuple[str, str]] = PATTERNS) -> None:
    """print summary statistics."""
    roots = set(s["root"] for s in stimuli)
    observed_patterns = set(s["pattern"] for s in stimuli)
    print("\n--- stimulus set summary ---")
    print(f"total stimuli:   {len(stimuli)}")
    print(f"unique roots:    {len(roots)}")
    print(f"unique patterns: {len(observed_patterns)}")
    print(f"matrix:          {len(roots)} roots × {len(observed_patterns)} patterns")
    # per-pattern breakdown
    for pat, desc in patterns:
        count = sum(1 for s in stimuli if s["pattern"] == pat)
        print(f"  {pat:12s} ({desc}) → {count} stimuli")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="generate nonce root-pattern stimuli"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="path to alakeel productivity_dataset.json or root list file",
    )
    parser.add_argument(
        "--augment-existing",
        default=None,
        help="existing stimulus JSON to copy and augment with surface-only probe prompts",
    )
    parser.add_argument(
        "--output",
        default="artifacts/stimuli/nonce_root_pattern_generated.json",
        help="output path for stimulus json",
    )
    parser.add_argument(
        "--lexicon",
        default=None,
        help="path to a transliterated surface-form lexicon",
    )
    parser.add_argument(
        "--collision-policy",
        choices=("error", "drop"),
        default="error",
        help="behavior when a generated transliteration occurs in --lexicon",
    )
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="include masked and fake-pattern control prompt templates",
    )
    args = parser.parse_args()

    if args.source and args.augment_existing:
        parser.error("--source and --augment-existing are mutually exclusive")
    if args.augment_existing and args.include_ablations:
        parser.error("--include-ablations is not supported with --augment-existing")
    if args.augment_existing and Path(args.augment_existing).resolve() == Path(args.output).resolve():
        parser.error("--augment-existing output must be distinct from the historical input")
    output = Path(args.output)
    metadata_output = output.with_suffix(output.suffix + ".metadata.json")
    named_inputs = {
        name: Path(value).resolve()
        for name, value in (
            ("source", args.source),
            ("augmented_input", args.augment_existing),
            ("lexicon", args.lexicon),
        )
        if value
    }
    if output.resolve() in named_inputs.values() or metadata_output.resolve() in named_inputs.values():
        parser.error("output paths must not overwrite source, historical, or lexicon inputs")
    for name, path in named_inputs.items():
        if not path.is_file():
            parser.error(f"{name.replace('_', ' ')} is not a regular file: {path}")
    input_identities = {name: file_identity(path) for name, path in named_inputs.items()}
    input_hashes = {name: sha256_file(path) for name, path in named_inputs.items()}

    if args.augment_existing:
        stimuli = augment_existing_stimuli(str(named_inputs["augmented_input"]))
        print(f"loaded {len(stimuli)} existing stimuli")
    else:
        nonce_roots = load_nonce_roots(
            str(named_inputs["source"]) if "source" in named_inputs else None
        )
        print(f"loaded {len(nonce_roots)} nonce roots")
        stimuli = generate_stimuli(nonce_roots, PATTERNS)
        stimuli = render_prompts(stimuli, include_ablations=args.include_ablations)
    validate_matrix(stimuli)
    count_before_collision_filter = len(stimuli)
    stimuli = validate_stimuli(
        stimuli,
        str(named_inputs["lexicon"]) if "lexicon" in named_inputs else None,
        collision_policy=args.collision_policy,
    )
    validate_matrix(stimuli)

    for name, path in named_inputs.items():
        if file_identity(path) != input_identities[name] or sha256_file(path) != input_hashes[name]:
            raise RuntimeError(f"{name.replace('_', ' ')} changed during stimulus generation: {path}")

    compute_stats(stimuli)

    # write output
    _atomic_write_json(output, stimuli)
    _atomic_write_json(
        metadata_output,
        {
            "schema_version": 1,
            "output_path": portable_path(output),
            "output_sha256": sha256_file(output),
            "source_path": portable_path(named_inputs["source"]) if args.source else None,
            "source_sha256": input_hashes.get("source"),
            "augmented_input_path": (
                portable_path(named_inputs["augmented_input"])
                if args.augment_existing
                else None
            ),
            "augmented_input_sha256": input_hashes.get("augmented_input"),
            "lexicon_path": portable_path(named_inputs["lexicon"]) if args.lexicon else None,
            "lexicon_sha256": input_hashes.get("lexicon"),
            "collision_audit_status": (
                "not_checked_no_lexicon"
                if not args.lexicon
                else (
                    "collisions_dropped"
                    if count_before_collision_filter != len(stimuli)
                    else "passed_no_collisions"
                )
            ),
            "collision_policy": args.collision_policy,
            "collision_count": (
                count_before_collision_filter - len(stimuli) if args.lexicon else None
            ),
            "record_count": len(stimuli),
            "include_ablations": args.include_ablations,
            "contains_label_free_surface_prompts": all(
                all(name in row.get("prompts", {}) for name in SURFACE_PROBE_TEMPLATES)
                for row in stimuli
            ),
            "contains_label_revealed_behavioral_prompts": any(
                contract.get("target_labels_in_prompt") is True
                for row in stimuli
                for contract in row.get("prompt_contracts", {}).values()
                if isinstance(contract, dict)
            ),
        },
    )

    print(f"saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
