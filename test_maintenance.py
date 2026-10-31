import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from predictive_maintenance_model import generate_data,evaluate

class MaintenanceTests(unittest.TestCase):
    def test_missing_columns_have_actionable_error(self):
        data = generate_data().drop(columns=["temperature_c", "maintenance_needed"])
        with self.assertRaisesRegex(ValueError, "Missing columns: maintenance_needed, temperature_c"):
            evaluate(data)

    def test_nonfinite_sensors_rejected_before_split(self):
        for value in (np.inf, -np.inf, "inf", "bad"):
            with self.subTest(value=value):
                data = generate_data()
                data["temperature_c"] = data["temperature_c"].astype(object)
                data.loc[0, "temperature_c"] = value
                with patch("predictive_maintenance_model.train_test_split") as split:
                    with self.assertRaisesRegex(ValueError, "temperature_c"):
                        evaluate(data)
                    split.assert_not_called()

    def test_numeric_sensor_strings_do_not_mutate_input(self):
        data = generate_data()
        data["temperature_c"] = data["temperature_c"].astype(str)
        original = data.copy(deep=True)
        metrics = evaluate(data)
        self.assertEqual(metrics["test_samples"], 250)
        pd.testing.assert_frame_equal(data, original)
    def test_repeatability(self):
        pd.testing.assert_frame_equal(generate_data(),generate_data())
    def test_baseline_and_holdout(self):
        m=evaluate(generate_data())
        self.assertEqual(m["test_samples"],250)
        self.assertIn("baseline_accuracy",m)
    def test_invalid_label(self):
        d=generate_data();d["maintenance_needed"]=1
        with self.assertRaises(ValueError):evaluate(d)

if __name__=="__main__": unittest.main()
