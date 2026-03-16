import unittest

import numpy as np
import pandas as pd

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks


class TestFitBlinksEmptyCandidates(unittest.TestCase):
    def test_dprocess_handles_empty_candidate_frame(self):
        signal = np.array([0.0, 1.0, 0.0], dtype=float)
        df = pd.DataFrame(columns=["start_blink", "end_blink"])

        fitblinks = FitBlinks(
            candidate_signal=signal,
            df=df,
            params=default_setting.DEFAULT_PARAMS,
        )

        fitblinks.dprocess()

        self.assertTrue(fitblinks.frame_blinks.empty)
        for column in (
            "max_value",
            "max_blink",
            "left_zero",
            "right_zero",
            "leftR2",
            "rightR2",
        ):
            self.assertIn(column, fitblinks.frame_blinks.columns)


if __name__ == "__main__":
    unittest.main()
