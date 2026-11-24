import unittest
from predictive_maintenance_model import evaluate, run_experiment


class AveragePrecisionTests(unittest.TestCase):
    def test_constant_scores_match_positive_prevalence(self):
        result = evaluate([0] * 8 + [1] * 2, [0] * 10, [0.] * 10)
        self.assertEqual(result['average_precision'], 0.2)

    def test_ranking_uses_scores_not_thresholded_predictions(self):
        labels = [0, 1, 0, 1]
        # Positives at ranks one and three: AP = (1 + 2/3) / 2.
        result = evaluate(labels, [0] * 4, [0.3, 0.4, 0.1, 0.2])
        self.assertEqual(result['average_precision'], 0.8333)
        perfect = evaluate(labels, [0] * 4, [0.1, 0.4, 0.2, 0.3])
        self.assertEqual(perfect['average_precision'], 1.0)

    def test_experiment_reports_baseline_reference(self):
        result = run_experiment(200)
        prevalence = result['test_positive_count'] / result['test_rows']
        self.assertEqual(result['majority_baseline']['average_precision'], round(prevalence, 4))
        self.assertTrue(0 <= result['model']['average_precision'] <= 1)
