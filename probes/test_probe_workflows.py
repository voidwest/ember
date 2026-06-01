import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))

from benchmark_summary import summarize_probe
from build_conllu_benchmark import build_rows
from causal_intervention import (
    load_probe_direction,
    remove_direction,
    single_layer_probe_score,
)


class ProbeWorkflowTests(unittest.TestCase):
    def test_conllu_rows_keep_labels_and_group_fields(self):
        conllu = """# sent_id = s1
# text = كتب الولد
1\tكتب\tكتب\tVERB\t_\tGender=Masc|Number=Sing\t0\troot\t_\t_
2\tالولد\tولد\tNOUN\t_\tGender=Masc|Number=Sing\t1\tnsubj\t_\t_

# sent_id = s2
# text = كتبت البنت
1\tكتبت\tكتب\tVERB\t_\tGender=Fem|Number=Sing\t0\troot\t_\t_
2\tالبنت\tبنت\tNOUN\t_\tGender=Fem|Number=Sing\t1\tnsubj\t_\t_
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.conllu"
            path.write_text(conllu, encoding="utf-8")
            rows = build_rows(str(path), min_label_count=1)

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["sentence_id"], "s1")
        self.assertEqual(rows[0]["labels"]["lemma"], "كتب")
        self.assertEqual(rows[0]["labels"]["upos"], "VERB")
        self.assertEqual(rows[2]["labels"]["Gender"], "Fem")
        self.assertEqual(rows[0]["target_span"], [0, 3])

    def test_probe_summary_reports_best_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probes.npz"
            np.savez(
                path,
                tasks=np.array(["root", "labels.Gender"], dtype=object),
                probe_kind="linear",
                root_split="pattern",
                pattern_split="root",
                root_accuracy=np.array([0.1, 0.8, 0.4]),
                root_selectivity=np.array([0.0, 0.5, 0.2]),
                root_classes=np.array(["a", "b"], dtype=object),
                labels_Gender_accuracy=np.array([0.6, 0.7, 0.65]),
                labels_Gender_classes=np.array(["Fem", "Masc"], dtype=object),
            )
            summary = summarize_probe(str(path))

        self.assertTrue(summary["exists"])
        self.assertEqual(summary["task_metrics"]["root"]["best_layer"], 1)
        self.assertAlmostEqual(summary["task_metrics"]["root"]["best_accuracy"], 0.8)
        self.assertEqual(summary["task_metrics"]["labels.Gender"]["n_classes"], 2)

    def test_run_benchmark_dry_run_writes_summary_and_split_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "name": "dry",
                "stimuli": "stimuli/nonce_root_pattern.json",
                "out_dir": tmp,
                "tasks": ["root", "pattern"],
                "split_policy": {"root": "pattern", "pattern": "root"},
                "run_mdl": False,
                "run_cca": False,
                "run_rsa": False,
                "run_plots": False,
                "models": [
                    {
                        "label": "m",
                        "kind": "ember",
                        "arch": "qwen3",
                        "model": "missing.gguf",
                        "probe_limit": 2,
                    }
                ],
            }
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    "probes/run_benchmark.py",
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=ROOT,
                check=True,
            )
            summary = json.loads(
                (Path(tmp) / "dry" / "benchmark_summary.json").read_text(encoding="utf-8")
            )

        self.assertTrue(summary["dry_run"])
        probe_cmd = next(
            cmd["cmd"]
            for cmd in summary["commands"]
            if any(part.endswith("train_linear_probe.py") for part in cmd["cmd"])
        )
        self.assertIn("--root-split", probe_cmd)
        self.assertIn("--pattern-split", probe_cmd)

    def test_direction_removal_reduces_synthetic_probe_score(self):
        rng = np.random.RandomState(0)
        n = 80
        labels = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
        activations = rng.normal(scale=0.05, size=(n, 2, 4)).astype(np.float32)
        activations[: n // 2, 1, 0] -= 2.0
        activations[n // 2 :, 1, 0] += 2.0

        with tempfile.TemporaryDirectory() as tmp:
            probe_path = Path(tmp) / "probe.npz"
            np.savez(
                probe_path,
                labels_Gender_probe_weights=np.array(
                    [
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                        [[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                labels_Gender_classes=np.array(["a", "b"], dtype=object),
            )
            info = load_probe_direction(str(probe_path), "labels.Gender", 1, "b")
            intervened = remove_direction(activations, 1, info["direction"])

        before = single_layer_probe_score(activations, labels.tolist(), 1, "linear", 5)
        after = single_layer_probe_score(intervened, labels.tolist(), 1, "linear", 5)
        self.assertGreater(before, 0.95)
        self.assertLess(after, 0.7)


if __name__ == "__main__":
    unittest.main()
