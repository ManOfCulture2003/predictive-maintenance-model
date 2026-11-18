import json
import subprocess
import sys
import unittest
import tempfile
import contextlib
import io
from unittest.mock import patch
from pathlib import Path

from predictive_maintenance_model import evaluate, make_demo_data, run_experiment, main


ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_interrupted_write_preserves_previous_report(self):
        real_temporary_file = tempfile.NamedTemporaryFile
        def fail_after_partial_write(**kwargs):
            handle = real_temporary_file(**kwargs)
            def interrupted(payload):
                handle.file.write(payload[:3])
                raise OSError('simulated disk write failure')
            handle.write = interrupted
            return handle
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.json'
            target.write_text('previous report', encoding='utf-8')
            with patch('predictive_maintenance_model.run_experiment', return_value={'samples': 200}), \
                 patch('tempfile.NamedTemporaryFile', side_effect=fail_after_partial_write), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(['--output', str(target)])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(target.read_text(encoding='utf-8'), 'previous report')
            self.assertEqual(list(Path(folder).iterdir()), [target])

    def test_failed_report_replacement_preserves_previous_report(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.json'
            target.write_text('previous report', encoding='utf-8')
            error = io.StringIO()
            with patch('predictive_maintenance_model.run_experiment', return_value={'samples': 200}), \
                 patch('os.replace', side_effect=PermissionError('destination locked')), \
                 contextlib.redirect_stderr(error):
                with self.assertRaises(SystemExit) as raised:
                    main(['--output', str(target)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('destination locked', error.getvalue())
            self.assertEqual(target.read_text(encoding='utf-8'), 'previous report')
            self.assertEqual(list(Path(folder).iterdir()), [target])

    def test_report_replaces_existing_file_with_valid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'reports' / 'report.json'
            for result in ({'samples': 200}, {'samples': 300}):
                with patch('predictive_maintenance_model.run_experiment', return_value=result), contextlib.redirect_stdout(io.StringIO()):
                    main(['--output', str(target)])
                self.assertEqual(json.loads(target.read_text(encoding='utf-8')), result)
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_evaluation_requires_both_target_classes(self):
        for labels in ([], [0, 0], [1, 1]):
            with self.subTest(labels=labels):
                with self.assertRaisesRegex(ValueError, "both.*0.*1"):
                    evaluate(labels, labels, [0.5] * len(labels))

    def test_balanced_accuracy_weights_both_classes_equally(self):
        # Eight negatives and two positives: all-negative predictions look 80% accurate.
        labels = [0] * 8 + [1] * 2
        baseline = evaluate(labels, [0] * 10, [0.0] * 10)
        self.assertEqual(baseline["accuracy"], 0.8)
        self.assertEqual(baseline["balanced_accuracy"], 0.5)
        perfect = evaluate(labels, labels, labels)
        self.assertEqual(perfect["balanced_accuracy"], 1.0)

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
            for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"):
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
