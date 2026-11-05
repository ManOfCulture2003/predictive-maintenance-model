import json
import subprocess
import sys
import unittest
from pathlib import Path

from predictive_maintenance_model import make_demo_data, run_experiment


ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_seed_is_reproducible(self):
        self.assertEqual(run_experiment(200, 42), run_experiment(200, 42))

    def test_different_seed_changes_data(self):
        first, _ = make_demo_data(200, 42)
        second, _ = make_demo_data(200, 43)
        self.assertFalse((first == second).all())

    def test_rejects_tiny_datasets(self):
        with self.assertRaisesRegex(ValueError, "at least 100"):
            run_experiment(99)

    def test_metrics_and_split(self):
        report = run_experiment(200)
        self.assertEqual(report["train_rows"] + report["test_rows"], 200)
        self.assertEqual(report["test_rows"], 60)
        for key in ("model", "majority_baseline"):
            metrics = report[key]
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                self.assertGreaterEqual(metrics[metric], 0)
                self.assertLessEqual(metrics[metric], 1)
            self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 60)

    def test_baseline_does_not_predict_positive(self):
        baseline = run_experiment(200)["majority_baseline"]
        self.assertEqual(baseline["recall"], 0)
        self.assertEqual(baseline["confusion_matrix"][1][1], 0)

    def test_cli_json(self):
        p = subprocess.run(
            [sys.executable, str(ROOT / "predictive_maintenance_model.py"), "--samples", "200"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(json.loads(p.stdout)["samples"], 200)

    def test_cli_rejects_invalid_sample_count(self):
        p = subprocess.run(
            [sys.executable, str(ROOT / "predictive_maintenance_model.py"), "--samples", "2"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("at least 100", p.stderr)


if __name__ == "__main__":
    unittest.main()
