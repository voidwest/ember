#!/usr/bin/env python3
"""Do activation-patching results survive quantization?

For each model family (Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct) and each
rung of its quantization ladder (F16 source, then Q8_0, Q6_K, Q4_K_M made from
that source by the pinned llama.cpp quantizer), run Ember's attribution
workflow on a fixed set of country/capital prompt pairs and patch *every*
candidate (site x layer x position) for real. The F16 run is the reference; each
quantized rung is compared with it on the measured patch effects, not on the
direct-path estimate.

The design was fixed before any run (see README.md): the prompt pairs, the
sites, the positions (the corrupted country token through the final token;
earlier positions are identical in both prompts, so a patch there is a no-op by
causality), and the metrics below.

    run_study.py run     --ember <bin> --out <workdir>   # bundles + reports
    run_study.py analyze --out <workdir> --results <dir> # tables + figures

`run` is resumable: a pair/rung whose bundle verified earlier is skipped.
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PAIRS = [
    # (id, clean country, corrupted country, target, foil)
    ("france-italy", "France", "Italy", "Paris", "Rome"),
    ("germany-spain", "Germany", "Spain", "Berlin", "Madrid"),
    ("japan-russia", "Japan", "Russia", "Tokyo", "Moscow"),
    ("egypt-greece", "Egypt", "Greece", "Cairo", "Athens"),
    ("poland-austria", "Poland", "Austria", "Warsaw", "Vienna"),
    ("portugal-sweden", "Portugal", "Sweden", "Lisbon", "Stockholm"),
]
TEMPLATE = "The capital of {} is the city of"
SITES = ["residual-pre-attention", "attention-output", "mlp-output"]
RUNGS = ["f16", "q8_0", "q6_k", "q4_k_m"]
REFERENCE = "f16"

FAMILIES = {
    "llama-3.2-1b": {
        "label": "Llama-3.2-1B-Instruct",
        "tokenizer": "tokenizer.json",
        "source": "Llama-3.2-1B-Instruct-f16.gguf",
        "layers": 16,
    },
    "qwen2.5-1.5b": {
        "label": "Qwen2.5-1.5B-Instruct",
        "tokenizer": "tokenizer-qwen2.5.json",
        "source": "qwen2.5-1.5b-instruct-fp16.gguf",
        "layers": 28,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_paths(ladder, sources):
    """family -> rung -> (path, sha256), with hashes from the pinned manifests."""
    manifest = json.loads((ladder / "ladder-manifest.json").read_text())
    by_target = {Path(entry["target"]["path"]).name: entry["target"]["sha256"] for entry in manifest}
    source_list = json.loads((sources / "sources.json").read_text())
    by_source = {entry["filename"]: entry["sha256"] for entry in source_list}
    out = {}
    for family, meta in FAMILIES.items():
        out[family] = {REFERENCE: (sources / meta["source"], by_source[meta["source"]])}
        for rung in RUNGS[1:]:
            name = f"{family}-{rung}.gguf"
            out[family][rung] = (ladder / name, by_target[name])
    return out


def tokenize(ember, model, tokenizer, text):
    result = subprocess.run(
        [str(ember), "experiment", "tokenize", "--model", str(model), "--tokenizer", str(tokenizer),
         "--text", text, "--json"],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)["pieces"]


def candidate_positions(ember, model, tokenizer, clean, corrupted):
    """First differing token through the final token, after checking equal length."""
    a = tokenize(ember, model, tokenizer, clean)
    b = tokenize(ember, model, tokenizer, corrupted)
    if len(a) != len(b):
        raise SystemExit(f"unequal token lengths: {clean!r} {len(a)} vs {corrupted!r} {len(b)}")
    differing = [index for index, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(differing) != 1:
        raise SystemExit(f"expected exactly one differing token: {a} vs {b}")
    return list(range(differing[0], len(a))), len(a)


def spec_text(family, rung, pair, model, model_sha, tokenizer, tokenizer_sha, positions, n_layers, threads, out_dir):
    pair_id, clean, corrupted, target, foil = pair
    k = len(SITES) * n_layers * len(positions)
    return f"""schema = "ember.experiment.v1"

[experiment]
name = "quant-attribution-{family}-{rung}-{pair_id}"
description = "Every candidate patched for real; reference for the quantization-transfer note."
seed = 42

[model]
path = "{model}"
expected_sha256 = "{model_sha}"
tokenizer = "{tokenizer}"
tokenizer_expected_sha256 = "{tokenizer_sha}"

[execution]
mode = "reference"
threads = {threads}
deterministic = true

[generation]
max_new_tokens = 1
temperature = 0.0

[[inputs]]
id = "clean"
text = "{TEMPLATE.format(clean)}"

[[inputs]]
id = "corrupted"
text = "{TEMPLATE.format(corrupted)}"

[attribution]
clean = "clean"
corrupted = "corrupted"
target = " {target}"
foil = " {foil}"
sites = {json.dumps(SITES)}
layers = "all"
positions = {json.dumps(positions)}
verify_top_k = {k}

[output]
directory = "{out_dir}"
overwrite = false
"""


def run(args):
    ember = args.ember.resolve()
    out = args.out.resolve()
    (out / "specs").mkdir(parents=True, exist_ok=True)
    (out / "bundles").mkdir(parents=True, exist_ok=True)
    models = model_paths(args.ladder.resolve(), args.sources.resolve())
    provenance = {"ember": str(ember), "ember_sha256": sha256(ember), "threads": args.threads, "runs": []}
    for family, meta in FAMILIES.items():
        tokenizer = REPO / meta["tokenizer"]
        tokenizer_sha = sha256(tokenizer)
        n_layers = meta["layers"]  # the report check below fails if this is wrong
        for pair in PAIRS:
            positions, length = candidate_positions(
                ember, models[family][REFERENCE][0], tokenizer,
                TEMPLATE.format(pair[1]), TEMPLATE.format(pair[2]))
            for rung in RUNGS:
                model, model_sha = models[family][rung]
                name = f"{family}-{rung}-{pair[0]}"
                bundle = out / "bundles" / name
                spec = out / "specs" / f"{name}.toml"
                spec.write_text(spec_text(family, rung, pair, model, model_sha, tokenizer, tokenizer_sha,
                                          positions, n_layers, args.threads, bundle))
                if (bundle / "artifacts/attribution/attribution.json").exists():
                    print(f"skip {name} (done)", flush=True)
                else:
                    print(f"run  {name}: {len(SITES) * n_layers * len(positions)} patches", flush=True)
                    with (out / f"{name}.log").open("w") as log:
                        subprocess.run([str(ember), "experiment", "run", str(spec), "--json"], cwd=REPO,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                verify = subprocess.run([str(ember), "experiment", "verify", str(bundle), "--json"], cwd=REPO,
                                        capture_output=True, text=True)
                verified = verify.returncode == 0 and json.loads(verify.stdout).get("ok") is True
                if not verified:
                    raise SystemExit(f"{name}: bundle failed verification\n{verify.stdout[-2000:]}{verify.stderr[-2000:]}")
                report = json.loads((bundle / "artifacts/attribution/attribution.json").read_text())
                expected = len(SITES) * n_layers * len(positions)
                if len(report["candidates"]) != expected or report["verified"] != expected:
                    raise SystemExit(f"{name}: {len(report['candidates'])} candidates, {report['verified']} verified; expected {expected}")
                manifest = json.loads((bundle / "manifest.json").read_text())
                provenance["runs"].append({
                    "family": family, "rung": rung, "pair": pair[0], "model_sha256": model_sha,
                    "sequence_length": length, "positions": positions, "layers": n_layers,
                    "bundle": name, "semantic_hash": manifest.get("semantic_hash"),
                    "payload_hash": manifest.get("payload_hash"),
                })
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"done; bundles in {out / 'bundles'}")


# ---------------------------------------------------------------- analysis


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def load_report(out, family, rung, pair_id):
    path = out / "bundles" / f"{family}-{rung}-{pair_id}" / "artifacts/attribution/attribution.json"
    report = json.loads(path.read_text())
    effects = {}
    for candidate in report["candidates"]:
        if candidate.get("actual") is None:
            raise SystemExit(f"{path}: candidate {candidate['rank']} was not patched; expected every candidate verified")
        key = (candidate["site"], candidate["layer"], candidate["position"])
        effects[key] = candidate
    return report, effects


def top(effects, k):
    """The k candidates with the largest measured effect (ties: key order)."""
    ranked = sorted(effects, key=lambda key: (-effects[key]["actual"], key))
    return ranked[:k]


def analyze(args):
    out = args.out.resolve()
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in FAMILIES:
        for pair in PAIRS:
            ref_report, ref = load_report(out, family, REFERENCE, pair[0])
            ref_gap = ref_report["clean_metric"] - ref_report["corrupted_metric"]
            keys = sorted(ref)
            ref_actual = [ref[key]["actual"] for key in keys]
            ref_top1 = top(ref, 1)[0]
            ref_top5 = set(top(ref, 5))
            # Sites that matter in the reference: recover at least 20% of the gap.
            material = {key for key in keys if ref[key]["recovered_fraction"] >= 0.2}
            for rung in RUNGS:
                report, effects = load_report(out, family, rung, pair[0])
                if sorted(effects) != keys:
                    raise SystemExit(f"{family}/{pair[0]}/{rung}: candidate set differs from the reference")
                actual = [effects[key]["actual"] for key in keys]
                fractions = [effects[key]["recovered_fraction"] for key in keys]
                ref_fractions = [ref[key]["recovered_fraction"] for key in keys]
                gap = report["clean_metric"] - report["corrupted_metric"]
                rung_material = {key for key in keys if effects[key]["recovered_fraction"] >= 0.2}
                union = material | rung_material
                rows.append({
                    "family": family,
                    "pair": pair[0],
                    "rung": rung,
                    "candidates": len(keys),
                    "clean_metric": report["clean_metric"],
                    "corrupted_metric": report["corrupted_metric"],
                    "gap": gap,
                    "gap_ratio_to_f16": gap / ref_gap if ref_gap else float("nan"),
                    "clean_prefers_target": report["clean_metric"] > 0,
                    "spearman_actual_vs_f16": spearman(actual, ref_actual),
                    "pearson_actual_vs_f16": pearson(actual, ref_actual),
                    "max_abs_fraction_diff": max(abs(a - b) for a, b in zip(fractions, ref_fractions)),
                    "mean_abs_fraction_diff": sum(abs(a - b) for a, b in zip(fractions, ref_fractions)) / len(keys),
                    "top1_matches_f16": top(effects, 1)[0] == ref_top1,
                    "top1": "/".join(map(str, top(effects, 1)[0])),
                    "top5_overlap_with_f16": len(set(top(effects, 5)) & ref_top5) / 5,
                    "material_sites_f16": len(material),
                    "material_sites": len(rung_material),
                    "material_jaccard_with_f16": len(material & rung_material) / len(union) if union else 1.0,
                    "estimate_spearman_vs_actual": spearman(
                        [effects[key]["estimate"] for key in keys if effects[key]["direct_path"]],
                        [effects[key]["actual"] for key in keys if effects[key]["direct_path"]]),
                })
    with (results / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.figures:
        figures(out, rows, results)


def summarize(rows):
    summary = {}
    for family in FAMILIES:
        summary[family] = {}
        for rung in RUNGS[1:]:
            subset = [row for row in rows if row["family"] == family and row["rung"] == rung]
            summary[family][rung] = {
                "pairs": len(subset),
                "min_spearman_actual_vs_f16": min(row["spearman_actual_vs_f16"] for row in subset),
                "median_spearman_actual_vs_f16": sorted(row["spearman_actual_vs_f16"] for row in subset)[len(subset) // 2],
                "max_abs_fraction_diff": max(row["max_abs_fraction_diff"] for row in subset),
                "top1_matches_f16": sum(row["top1_matches_f16"] for row in subset),
                "mean_top5_overlap": sum(row["top5_overlap_with_f16"] for row in subset) / len(subset),
                "min_material_jaccard": min(row["material_jaccard_with_f16"] for row in subset),
                "gap_ratio_range": [min(row["gap_ratio_to_f16"] for row in subset),
                                    max(row["gap_ratio_to_f16"] for row in subset)],
            }
    return summary


def figures(out, rows, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    themes = {
        "light": {"bg": "#f3f0e9", "fg": "#242424", "muted": "#85847e", "grid": "#d8d4ca",
                  "rungs": {"q8_0": "#315fd6", "q6_k": "#b07a1e", "q4_k_m": "#b23a3a"}},
        "dark": {"bg": "#16171a", "fg": "#e6e3dc", "muted": "#8d8b85", "grid": "#2c2d31",
                 "rungs": {"q8_0": "#7d9cf0", "q6_k": "#e0b25c", "q4_k_m": "#e07a7a"}},
    }
    for theme_name, theme in themes.items():
        # Figure 1: recovered fraction per layer at the final position, F16 vs
        # each rung, for one representative pair per family (the first pair).
        fig, axes = plt.subplots(len(FAMILIES), len(SITES), figsize=(10, 5.6), sharey="row")
        fig.patch.set_facecolor(theme["bg"])
        for row_index, family in enumerate(FAMILIES):
            pair = PAIRS[0][0]
            reports = {rung: load_report(out, family, rung, pair)[1] for rung in RUNGS}
            final = max(key[2] for key in reports[REFERENCE])
            for col, site in enumerate(SITES):
                ax = axes[row_index][col]
                ax.set_facecolor(theme["bg"])
                layers = sorted({key[1] for key in reports[REFERENCE] if key[0] == site})
                ref = [reports[REFERENCE][(site, layer, final)]["recovered_fraction"] for layer in layers]
                ax.plot(layers, ref, color=theme["fg"], linewidth=2.2, label="F16")
                for rung in RUNGS[1:]:
                    values = [reports[rung][(site, layer, final)]["recovered_fraction"] for layer in layers]
                    ax.plot(layers, values, color=theme["rungs"][rung], linewidth=1.2, linestyle="--",
                            label=rung.upper().replace("_K_M", "_K_M"))
                ax.grid(color=theme["grid"], linewidth=0.6)
                ax.tick_params(colors=theme["muted"], labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color(theme["grid"])
                if row_index == 0:
                    ax.set_title(site, color=theme["fg"], fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{FAMILIES[family]['label']}\nrecovered fraction", color=theme["fg"], fontsize=8)
                if row_index == len(FAMILIES) - 1:
                    ax.set_xlabel("layer (final position)", color=theme["muted"], fontsize=8)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, labelcolor=theme["fg"], fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(results / f"profile-{theme_name}.png", dpi=160, facecolor=theme["bg"])
        plt.close(fig)

        # Figure 2: Spearman (measured effects vs F16) per pair, by rung.
        fig, axes = plt.subplots(1, len(FAMILIES), figsize=(10, 3.4), sharey=True)
        fig.patch.set_facecolor(theme["bg"])
        for index, family in enumerate(FAMILIES):
            ax = axes[index]
            ax.set_facecolor(theme["bg"])
            for offset, rung in enumerate(RUNGS[1:]):
                values = [row["spearman_actual_vs_f16"] for row in rows if row["family"] == family and row["rung"] == rung]
                xs = [offset + 0.08 * (i - (len(values) - 1) / 2) for i in range(len(values))]
                ax.scatter(xs, values, color=theme["rungs"][rung], s=22)
            ax.set_xticks(range(len(RUNGS) - 1), [rung.upper() for rung in RUNGS[1:]])
            ax.set_title(FAMILIES[family]["label"], color=theme["fg"], fontsize=9)
            ax.grid(color=theme["grid"], linewidth=0.6, axis="y")
            ax.tick_params(colors=theme["muted"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(theme["grid"])
        axes[0].set_ylabel("Spearman of patch effects vs F16", color=theme["fg"], fontsize=8)
        fig.tight_layout()
        fig.savefig(results / f"agreement-{theme_name}.png", dpi=160, facecolor=theme["bg"])
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--ember", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--ladder", type=Path, default=REPO / "models/v03-ladder-reproduced")
    run_parser.add_argument("--sources", type=Path, default=REPO / "models/v03-sources")
    run_parser.add_argument("--threads", type=int, default=4)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--out", type=Path, required=True)
    analyze_parser.add_argument("--results", type=Path, required=True)
    analyze_parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    {"run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
