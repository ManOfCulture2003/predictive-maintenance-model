import unittest
import pandas as pd
from predictive_maintenance_model import generate_data,evaluate

class MaintenanceTests(unittest.TestCase):
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
