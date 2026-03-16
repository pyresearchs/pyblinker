import unittest

import numpy as np
import pandas as pd

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks


class TestFitBlinksTerminalEdgeCase(unittest.TestCase):
    def test_terminal_right_zero_skips_downstream_fit_metrics(self):
        """Match MATLAB when the trailing zero-crossing reaches the last sample."""

        signal = np.array([-2.0, -1.0, 0.5, 2.0, 5.0, 3.0, 2.0, 1.0], dtype=float)
        df = pd.DataFrame({"start_blink": [2], "end_blink": [len(signal) - 1]})

        params = default_setting.DEFAULT_PARAMS.copy()
        fitblinks = FitBlinks(candidate_signal=signal, df=df, params=params)
        fitblinks.dprocess(run_fit=True)

        row = fitblinks.frame_blinks.iloc[0]
        self.assertTrue(np.isnan(row["max_neg_vel_frame"]))
        self.assertTrue(np.isnan(row["right_base"]))
        self.assertTrue(np.isnan(row["leftR2"]))
        self.assertTrue(np.isnan(row["rightR2"]))
        self.assertTrue(np.isnan(row["x_intersect"]))
        self.assertEqual(row["nsize_x_left"], 0)
        self.assertEqual(row["nsize_x_right"], 0)


if __name__ == "__main__":
    unittest.main()
