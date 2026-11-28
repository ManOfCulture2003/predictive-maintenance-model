import unittest
from predictive_maintenance_model import evaluate, run_experiment


class ProbabilityLossTests(unittest.TestCase):
    def test_same_classification_different_probability_quality(self):
        strong = evaluate([0,1], [0,1], [0.1,0.9])
        weak = evaluate([0,1], [0,1], [0.4,0.6])
        self.assertEqual(strong['accuracy'], weak['accuracy'])
        self.assertEqual(strong['brier_loss'], 0.01)
        self.assertEqual(weak['brier_loss'], 0.16)

    def test_perfect_and_completely_wrong_probabilities(self):
        self.assertEqual(evaluate([0,1], [0,1], [0,1])['brier_loss'], 0)
        self.assertEqual(evaluate([0,1], [1,0], [1,0])['brier_loss'], 1)

    def test_baseline_loss_matches_observed_positive_share(self):
        report = run_experiment(200)
        expected = report['test_positive_count'] / report['test_rows']
        self.assertEqual(report['majority_baseline']['brier_loss'], round(expected,4))

    def test_non_probability_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([0,1], [0,1], [-0.1,1.1])
