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
    build_summary,
    load_probe_direction,
    remove_direction,
    render_markdown_summary,
    single_layer_probe_score,
    summarize_continuations,
    summarize_logits,
)
from train_linear_probe import groups_for_task, prepare_splits


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

    def test_nonce_grouped_split_policies_keep_groups_disjoint(self):
        rows = [
            {"root": root, "pattern": pattern, "prompt_template": template}
            for root in ["r1", "r2", "r3", "r4"]
            for pattern in ["p1", "p2", "p3", "p4"]
            for template in ["en_zero", "ar_zero"]
        ]

        cases = [
            ("pattern", "root-heldout", "pattern"),
            ("root", "pattern-heldout", "root"),
            ("root", "combination-heldout", "root"),
            ("root", "template-heldout", "root"),
        ]
        for task, split, label_field in cases:
            with self.subTest(split=split):
                groups, group_values, metadata = groups_for_task(task, split, rows)
                labels = [row[label_field] for row in rows]
                folds, _ = prepare_splits(
                    labels,
                    n_folds=4,
                    groups=groups,
                    group_values=group_values,
                    split_name=split,
                )
                self.assertIsNotNone(folds)
                self.assertEqual(metadata["effective_policy"], split)
                for train_idx, test_idx in folds:
                    train_groups = {group_values[i] for i in train_idx}
                    test_groups = {group_values[i] for i in test_idx}
                    self.assertFalse(train_groups & test_groups)

    def test_nonce_grouped_split_errors_when_target_label_is_held_out(self):
        rows = [
            {"root": root, "pattern": pattern}
            for root in ["r1", "r2", "r3"]
            for pattern in ["p1", "p2", "p3"]
        ]
        groups, group_values, _ = groups_for_task("root", "root-heldout", rows)
        labels = [row["root"] for row in rows]
        with self.assertRaisesRegex(ValueError, "absent from training"):
            prepare_splits(
                labels,
                n_folds=3,
                groups=groups,
                group_values=group_values,
                split_name="root-heldout",
            )

    def test_template_heldout_errors_without_template_metadata(self):
        rows = [
            {"root": root, "pattern": pattern}
            for root in ["r1", "r2"]
            for pattern in ["p1", "p2"]
        ]
        with self.assertRaisesRegex(ValueError, "prompt template metadata"):
            groups_for_task("root", "template-heldout", rows)

    def test_train_probe_writes_split_policy_metadata(self):
        rows = [
            {"root": root, "pattern": pattern}
            for root in ["r1", "r2", "r3"]
            for pattern in ["p1", "p2", "p3"]
        ]
        rng = np.random.RandomState(1)
        activations = rng.normal(size=(len(rows), 2, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stimuli = tmp_path / "stimuli.json"
            act_path = tmp_path / "activations.npy"
            output = tmp_path / "probes.npz"
            stimuli.write_text(json.dumps(rows), encoding="utf-8")
            np.save(act_path, activations)
            subprocess.run(
                [
                    sys.executable,
                    "probes/train_linear_probe.py",
                    "--activations",
                    str(act_path),
                    "--stimuli",
                    str(stimuli),
                    "--tasks",
                    "pattern",
                    "--pattern-split",
                    "root-heldout",
                    "--folds",
                    "3",
                    "--probe-kind",
                    "sgd",
                    "--max-iter",
                    "200",
                    "--tol",
                    "0.001",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            data = np.load(output, allow_pickle=True)
            metadata = json.loads(str(data["split_policy_json"]))
            summary = summarize_probe(str(output))
            sidecar = tmp_path / "probes_split_policy.json"
            sidecar_exists = sidecar.exists()

        self.assertEqual(metadata[0]["effective_policy"], "root-heldout")
        self.assertEqual(metadata[0]["group_field"], "root")
        self.assertEqual(
            summary["split_policy_metadata"][0]["effective_policy"],
            "root-heldout",
        )
        self.assertTrue(sidecar_exists)

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

    def test_causal_intervention_summary_reports_conservatively(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            logits_before = tmp_path / "before_logits.npy"
            logits_after = tmp_path / "after_logits.npy"
            cont_before = tmp_path / "before_continuations.json"
            cont_after = tmp_path / "after_continuations.json"
            np.save(logits_before, np.array([0.1, 0.9, 0.0], dtype=np.float32))
            np.save(logits_after, np.array([0.8, 0.2, 0.0], dtype=np.float32))
            cont_before.write_text(
                json.dumps([{"generated": "kataba"}, {"generated": "yaktubu"}]),
                encoding="utf-8",
            )
            cont_after.write_text(
                json.dumps([{"generated": "kataba"}, {"generated": "changed"}]),
                encoding="utf-8",
            )

            logit_shift = summarize_logits(str(logits_before), str(logits_after))
            continuation_changes = summarize_continuations(str(cont_before), str(cont_after))
            summary = build_summary(
                activations_path="acts.npy",
                output_path="intervened.npy",
                direction_output="direction.npz",
                task="labels.Gender",
                layer=1,
                class_label="Masc",
                direction_info={
                    "selected_class": "Masc",
                    "classes": ["Fem", "Masc"],
                    "norm_before_normalization": 2.0,
                },
                before_acc=0.9,
                after_acc=0.4,
                logit_shift=logit_shift,
                continuation_changes=continuation_changes,
            )
            markdown = render_markdown_summary(summary)

        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["probe_accuracy"]["drop"], 0.5)
        self.assertTrue(summary["probe_accuracy"]["target_probe_score_dropped"])
        self.assertTrue(summary["downstream"]["logit_shift"]["top_token_changed"])
        self.assertEqual(summary["downstream"]["continuation_changes"]["changed"], 1)
        self.assertFalse(summary["claims"]["behavioral_causality_claimed"])
        self.assertIn("probe-direction removal affected decodability", markdown)
        self.assertIn("not behavioral causality", markdown)


if __name__ == "__main__":
    unittest.main()
