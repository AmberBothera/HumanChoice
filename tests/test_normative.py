import unittest

import pandas as pd

from humanchoice.normative import calibrate, predict, utility


class NormativeTests(unittest.TestCase):
    def test_hyperbolic_utility_at_zero_delay_equals_value(self):
        self.assertEqual(utility([10], [0], 0.1)[0], 10)

    def test_calibration_uses_supplied_rows_and_grid(self):
        frame = pd.DataFrame({
            "ss_value": [10, 10], "ss_time": [0, 0],
            "ll_value": [12, 12], "ll_time": [100, 100], "choice": [0, 0],
        })
        best, table = calibrate(frame, [0.0, 1.0])
        self.assertEqual(best, 1.0)
        self.assertTrue((predict(frame, best) == frame["choice"]).all())
        self.assertEqual(len(table), 2)
