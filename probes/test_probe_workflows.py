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
    nested_direction_probe_scores,
    remove_direction,
    render_markdown_summary,
    single_layer_probe_score,
    summarize_continuations,
    summarize_logits,
)
from cca_analysis import cross_validated_cca, svd_cca
from train_linear_probe import (
    audit_label_revealing_prompt,
    groups_for_task,
    prepare_splits,
)


class ProbeWorkflowTests(unittest.TestCase):
    def test_cross_validated_cca_rejects_high_dimensional_sample_space_saturation(self):
        rng = np.random.default_rng(7)
        samples, width = 90, 120
        independent_a = rng.normal(size=(samples, width))
        independent_b = rng.normal(size=(samples, width))
        latent = rng.normal(size=(samples, 5))
        shared_a = latent @ rng.normal(size=(5, width)) + 0.2 * rng.normal(
            size=(samples, width)
        )
        shared_b = latent @ rng.normal(size=(5, width)) + 0.2 * rng.normal(
            size=(samples, width)
        )

        self.assertGreater(float(np.mean(svd_cca(independent_a, independent_b, 3))), 0.95)
        independent_cv = cross_validated_cca(independent_a, independent_b, 3)
        shared_cv = cross_validated_cca(shared_a, shared_b, 3)
        self.assertLess(float(np.max(independent_cv)), 0.30)
        self.assertGreater(float(np.min(shared_cv)), 0.80)

    def test_prompt_leakage_audit_distinguishes_surface_and_revealed_prompts(self):
        rows = [
            {
                "root": "q-l-z",
                "pattern": "fa3ala",
                "prompts": {
                    "surface": "Analyze the word qalaza.",
                    "revealed": "Apply fa3ala to q-l-z.",
                },
            }
        ]
        safe = audit_label_revealing_prompt(
            rows,
            ["root", "pattern"],
            {"probe_template": "surface", "probe_position": "last"},
        )
        leaked = audit_label_revealing_prompt(
            rows,
            ["root", "pattern"],
            {"probe_template": "revealed", "probe_position": "last"},
        )
        unverifiable = audit_label_revealing_prompt(rows, ["root"], {})

        self.assertEqual(safe["status"], "passed")
        self.assertEqual(leaked["status"], "label_revealed")
        self.assertEqual(leaked["revealed_task_row_count"], 2)
        self.assertEqual(
            unverifiable["status"], "not_checked_missing_probe_template_metadata"
        )

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
                tasks=np.array(["root", "labels.Gender"], dtype=str),
                probe_kind="linear",
                root_split="pattern",
                pattern_split="root",
                root_accuracy=np.array([0.1, 0.8, 0.4]),
                root_selectivity=np.array([0.0, 0.5, 0.2]),
                root_classes=np.array(["a", "b"], dtype=str),
                labels_Gender_accuracy=np.array([0.6, 0.7, 0.65]),
                labels_Gender_classes=np.array(["Fem", "Masc"], dtype=str),
                labels_Gender_class_counts=np.array([3, 5]),
                labels_Gender_chance=np.array(0.5),
                labels_Gender_confusion_matrices=np.array(
                    [
                        [[2, 1], [1, 4]],
                        [[3, 0], [1, 4]],
                        [[2, 1], [2, 3]],
                    ],
                    dtype=np.int64,
                ),
            )
            summary = summarize_probe(str(path))

        self.assertTrue(summary["exists"])
        self.assertEqual(summary["task_metrics"]["root"]["best_layer"], 1)
        self.assertAlmostEqual(summary["task_metrics"]["root"]["best_accuracy"], 0.8)
        self.assertEqual(summary["task_metrics"]["labels.Gender"]["n_classes"], 2)
        self.assertEqual(
            summary["task_metrics"]["labels.Gender"]["class_counts"],
            {"Fem": 3, "Masc": 5},
        )
        self.assertEqual(summary["task_metrics"]["labels.Gender"]["chance"], 0.5)
        self.assertEqual(
            summary["task_metrics"]["labels.Gender"]["confusion_matrices"]["best_layer"],
            [[3, 0], [1, 4]],
        )

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
                    "--allow-unverifiable-prompt-contract",
                ],
                cwd=ROOT,
                check=True,
            )
            data = np.load(output, allow_pickle=False)
            metadata = json.loads(str(data["task_split_policy_json"]))
            split_policy = str(data["split_policy"])
            has_confusions = "pattern_confusion_matrices" in data
            has_class_counts = "pattern_class_counts" in data
            summary = summarize_probe(str(output))
            sidecar = tmp_path / "probes_split_policy.json"
            sidecar_exists = sidecar.exists()

        self.assertEqual(split_policy, "task-specific")
        self.assertTrue(has_confusions)
        self.assertTrue(has_class_counts)
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
                labels_Gender_classes=np.array(["a", "b"], dtype=str),
                probe_weight_space=np.array("raw_activation"),
            )
            info = load_probe_direction(str(probe_path), "labels.Gender", 1, "b")
            intervened = remove_direction(activations, 1, info["direction"])

        before = single_layer_probe_score(activations, labels.tolist(), 1, "linear", 5)
        after = single_layer_probe_score(intervened, labels.tolist(), 1, "linear", 5)
        self.assertGreater(before, 0.95)
        self.assertLess(after, 0.7)

        nested = nested_direction_probe_scores(
            activations[:, :, :1],
            labels.tolist(),
            1,
            "b",
            "linear",
            5,
        )
        self.assertGreater(nested["before"], 0.95)
        self.assertLess(nested["after"], 0.7)
        self.assertFalse(nested["direction_fit_uses_heldout_labels"])

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

        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["probe_accuracy"]["drop"], 0.5)
        self.assertTrue(summary["probe_accuracy"]["target_probe_score_dropped"])
        self.assertTrue(summary["downstream"]["logit_shift"]["top_token_changed"])
        self.assertEqual(summary["downstream"]["continuation_changes"]["changed"], 1)
        self.assertFalse(summary["claims"]["behavioral_causality_claimed"])
        self.assertIn("probe-direction removal affected decodability", markdown)
        self.assertIn("not behavioral causality", markdown)


if __name__ == "__main__":
    unittest.main()
