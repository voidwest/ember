#!/usr/bin/env python3
"""Run and verify the pinned Arabic v1 example, including exact restoration."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--ember", type=Path, default=Path("target/release/ember"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    binary = args.ember.resolve()

    def command(label, *arguments):
        argv = [str(binary), "experiment", *map(str, arguments), "--json"]
        (work / f"{label}.command.json").write_text(json.dumps({"cwd": str(repo), "argv": argv}, indent=2) + "\n")
        with (work / f"{label}.json").open("w") as out, (work / f"{label}.stderr.log").open("w") as err:
            subprocess.run(argv, cwd=repo, stdout=out, stderr=err, check=True)
        return json.loads((work / f"{label}.json").read_text())

    with binary.open("rb") as handle:
        binary_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    for leg, filename in (("baseline", "morphology-layerwise-capture"), ("intervention", "morphology-intervention"), ("restoration", "morphology-restoration")):
        spec = repo / "examples/experiments" / f"{filename}.toml"
        command(f"{leg}-validate", "validate", spec)
        command(f"{leg}-run", "run", spec, "--output", work / leg)
        verification = command(f"{leg}-verify", "verify", work / leg, "--model", repo / "Llama-3.2-1B-Instruct-Q8_0.gguf", "--tokenizer", repo / "tokenizer.json")
        assert verification["ok"], f"{leg} failed deep verification"
        print(f"verified {leg}", flush=True)
    changed = command("intervention-compare", "compare", work / "baseline", work / "intervention")
    restored = command("restoration-compare", "compare", work / "baseline", work / "restoration")
    assert len(restored["captures"]) == 32, "expected two captures across all 16 layers"
    assert all(c["present_in_a"] and c["present_in_b"] and c["metrics"]["exact"] for c in restored["captures"]), "restored captures differ"
    assert len(restored["outputs"]) == 1
    assert all(o["generated_tokens_equal"] and o["generated_text_equal"] and o["final_top1_equal"] for o in restored["outputs"]), "restored outputs differ"
    assert any(c["metrics"] and not c["metrics"]["exact"] for c in changed["captures"]), "intervention had no captured effect"
    reproduced = command("reproduce", "reproduce", work / "baseline", "--model", repo / "Llama-3.2-1B-Instruct-Q8_0.gguf", "--output", work / "reproduced")
    assert reproduced["verdict"] == "exact-semantic" and reproduced["captures_exact"] and reproduced["tokens_equal"], "baseline reproduction differs"
    verification = command("reproduced-verify", "verify", work / "reproduced")
    assert verification["ok"]
    result = {"passed": True, "ember_sha256": binary_hash, "restored_capture_count": len(restored["captures"]), "reproduction_verdict": reproduced["verdict"]}
    (work / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
