import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import predictive_maintenance_model as model


class TrainingIsolationTests(unittest.TestCase):
    def test_held_out_outliers_do_not_change_scaler_statistics(self):
        features, labels = model.make_demo_data(200, seed=42)
        train_ids, test_ids = train_test_split(
            np.arange(len(labels)), test_size=0.30, random_state=42, stratify=labels
        )
        changed = features.copy()
        changed[test_ids] += 10000
        original_scaler = StandardScaler
        fitted = []

        class RecordingScaler(original_scaler):
            def fit(self, X, y=None, **kwargs):
                result = super().fit(X, y, **kwargs)
                fitted.append(self)
                return result

        for dataset in (features, changed):
            with patch.object(model, 'make_demo_data', return_value=(dataset, labels)), patch.object(
                model, 'StandardScaler', RecordingScaler
            ):
                report = model.run_experiment(200, seed=42)
            self.assertEqual(report['test_rows'], len(test_ids))
        self.assertEqual(len(fitted), 2)
        for scaler in fitted:
            self.assertEqual(scaler.n_samples_seen_, len(train_ids))
            np.testing.assert_allclose(scaler.mean_, features[train_ids].mean(axis=0))
            np.testing.assert_allclose(scaler.var_, features[train_ids].var(axis=0))
        np.testing.assert_array_equal(fitted[0].mean_, fitted[1].mean_)


if __name__ == '__main__':
    unittest.main()
