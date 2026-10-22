import unittest

import numpy as np
import pandas as pd

from humanchoice.features import add_features, eligible_rows


def sample_frame():
    return pd.DataFrame({
        "paper": ["A", "A", "B"], "subj_ident": ["1", "2", "1"],
        "ss_value": [10, 10, 0], "ss_time": [0, 0, 0],
        "ll_value": [20, 20, 20], "ll_time": [30, 30, 30],
        "choice": [1, 0, 1], "rt": [1.0, 2.0, 3.0], "age": [18, 19, 20],
        "subj_excl": [False, True, False], "trial_excl": [False, False, False],
    })


class FeatureTests(unittest.TestCase):
    def test_eligibility_applies_exclusions_and_positive_rewards(self):
        result = eligible_rows(sample_frame())
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["participant_id"], "A::1")

    def test_features_are_computed_from_choice_attributes(self):
        result = add_features(eligible_rows(sample_frame()))
        self.assertTrue(np.isclose(result.iloc[0]["log_reward_ratio"], np.log(2)))
        self.assertTrue(np.isfinite(result.iloc[0]["log_delay_ratio"]))
