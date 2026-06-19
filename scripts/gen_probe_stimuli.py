"""Convert canonical.jsonl to probe stimuli.json with prompts."""
import json, hashlib, sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("usage: python gen_probe_stimuli.py <canonical.jsonl> [output.json]")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name("probe_stimuli_1416.json")

    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    stimuli = []

    for r in rows:
        sid = hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        prompt = (
            f"Arabic morphology token probe. "
            f"Surface: {r.get('surface_dediac', r.get('surface', ''))}\n"
            f"Token: {r.get('surface', '')}\n"
            f"Lemma: {r.get('lemma', '')}\n"
            f"Root: {r.get('root', '')}\n"
            f"Pattern: {r.get('abstract_pattern', '')}\n"
            f"Predict the token morphology."
        )
        stimuli.append({
            "id": sid,
            "surface": r.get("surface_dediac", r.get("surface", "")),
            "lemma": r.get("lemma", ""),
            "root": r.get("root", ""),
            "pattern": r.get("abstract_pattern", ""),
            "abstract_pattern": r.get("abstract_pattern", ""),
            "concrete_pattern": r.get("concrete_pattern", ""),
            "pos": r.get("pos", ""),
            "features": r.get("features", {}),
            "expected_surface": r.get("surface", ""),
            "prompts": {
                "morph_context": prompt,
            },
            "metadata": {
                "source": r.get("source", ""),
                "split": r.get("split", ""),
            },
        })

    dst.write_text(json.dumps(stimuli, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(stimuli)} stimuli to {dst}")

if __name__ == "__main__":
    main()
