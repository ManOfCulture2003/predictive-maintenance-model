import unittest
from predictive_maintenance_model import run_experiment


class PriorBaselineTests(unittest.TestCase):
    def test_loss_uses_training_prevalence_not_holdout_labels(self):
        report = run_experiment(200, 42)
        probability = report['train_positive_count'] / report['train_rows']
        actual = report['test_positive_count'] / report['test_rows']
        self.assertNotEqual(probability, actual)
        expected = actual * (1 - probability)**2 + (1 - actual) * probability**2
        prior = report['class_prior_baseline']
        self.assertEqual(prior['brier_loss'], round(expected, 4))
        self.assertLess(prior['brier_loss'], report['majority_baseline']['brier_loss'])
        self.assertEqual(prior['confusion_matrix'], report['majority_baseline']['confusion_matrix'])
        self.assertEqual(prior['roc_auc'], 0.5)

    def test_reference_is_reproducible(self):
        self.assertEqual(run_experiment(200, 7)['class_prior_baseline'],
                         run_experiment(200, 7)['class_prior_baseline'])
