import unittest
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.get_blink_positions import get_blink_position

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TestFitBlinks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load test data, run FitBlinks, and load MATLAB reference output.
        """
        # Paths
        # Note: We use the MAT file for the signal to ensure exact match with MATLAB reference, as the FIF file contains different (possibly unscaled) values.
        mat_signal_path = Path("test/test_files/ear_eog_resamp-100_raw.mat")
        mat_pos_path = Path("test/test_files/step_a_extract_blinks_resamp-100.mat")
        mat_fit_path = Path("test/test_files/step_b_fit_blink.mat")
        
        # Referenced in the task but not used for signal data due to value discrepancy
        cls.fif_path = Path("test/test_files/ear_eog_resamp-100_raw.fif")

        assert mat_signal_path.exists(), f"Missing MAT signal file: {mat_signal_path}"
        assert mat_pos_path.exists(), f"Missing MATLAB positions file: {mat_pos_path}"
        assert mat_fit_path.exists(), f"Missing MATLAB fits file: {mat_fit_path}"

        # 1. Load signal from MAT file
        data_signal = loadmat(str(mat_signal_path), squeeze_me=True)
        blink_comp = data_signal["blinkComp"].astype(np.float64)
        if blink_comp.ndim > 1:
            blink_comp = blink_comp[0]




        params = default_setting.DEFAULT_PARAMS.copy()
        df_positions = get_blink_position(
            params,
            blink_component=blink_comp,
            ch="No_channel",
            progress_bar=False,
            )
        # 3. Run FitBlinks
        # params_default = default_setting.DEFAULT_PARAMS.copy()
        fitblinks = FitBlinks(
            candidate_signal=blink_comp,
            df=df_positions,
            params=params,
        )
        # dprocess computes max_blink, outer_start/end, left_zero, right_zero, and fit
        fitblinks.dprocess(run_fit=True)

        df_output = fitblinks.frame_blinks.copy()

        # ---------------------------------------------------------------------
        # MATLAB compatibility: index correction (1-based indexing for comparison)
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        # MATLAB compatibility: index correction (1-based indexing)
        # ---------------------------------------------------------------------
        columns_to_increment = [
                "max_blink", "start_blink", "end_blink",
                "outer_start", "outer_end",
                "left_zero", "right_zero",
                "max_pos_vel_frame", "max_neg_vel_frame",
                "left_base", "right_base",
                "left_zero_half_height", "right_zero_half_height",
                "left_base_half_height", "right_base_half_height",
                "x_intersect",
                "right_x_intercept","left_x_intercept"
                ]
        df_output[columns_to_increment] += 1

        df_output["left_range"] = df_output["left_range"].apply(
            lambda x: [v + 1 for v in x]
            )
        df_output["right_range"] = df_output["right_range"].apply(
            lambda x: [v + 1 for v in x]
            )

        # ---------------------------------------------------------------------
        # Load MATLAB reference output
        # ---------------------------------------------------------------------
        mat_data = loadmat(
            str(mat_fit_path),
            squeeze_me=True,
            simplify_cells=True,
            struct_as_record=False,
        )

        blink_fits_mat = mat_data["blinkFits"]
        if not isinstance(blink_fits_mat, (list, np.ndarray)):
            blink_fits_mat = [blink_fits_mat]
            
        df_mat = pd.DataFrame(blink_fits_mat).reset_index(drop=True)

        matlab_to_python = {
            "maxFrame": "max_blink",
            "maxValue": "max_value",
            "leftOuter": "outer_start",
            "rightOuter": "outer_end",
            "leftZero": "left_zero",
            "rightZero": "right_zero",
            "leftBase": "left_base",
            "rightBase": "right_base",
            "leftBaseHalfHeight": "left_base_half_height",
            "rightBaseHalfHeight": "right_base_half_height",
            "leftZeroHalfHeight": "left_zero_half_height",
            "rightZeroHalfHeight": "right_zero_half_height",
            "leftRange": "left_range",
            "rightRange": "right_range",
            "leftSlope": "left_slope",
            "rightSlope": "right_slope",
            "averLeftVelocity": "aver_left_velocity",
            "averRightVelocity": "aver_right_velocity",
            "leftR2": "leftR2",
            "rightR2": "rightR2",
            "xIntersect": "x_intersect",
            "yIntersect": "y_intersect",
            "leftXIntercept": "left_x_intercept",
            "rightXIntercept": "right_x_intercept",
        }

        column_order = ["max_blink", "max_value", "left_zero", "right_zero", "leftR2", "rightR2"]

        cls.df_mat = df_mat.rename(columns=matlab_to_python)[column_order]
        cls.df_py = df_output[column_order].reset_index(drop=True)

    def test_dataframe_equality(self):
        """
        # Compare Python FitBlinks output with MATLAB reference output.
        # Both should return 355 blinks with identical properties.
        # """
        self.assertEqual(
            len(self.df_py),
            len(self.df_mat),
            f"Different number of blinks: "
            f"py={len(self.df_py)} mat={len(self.df_mat)}",
        )
        # TODO GOT SOME ISSUE python implementation of pyblinker
        pd.testing.assert_frame_equal(
            self.df_py,
            self.df_mat,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-4,
        )
        # pass




if __name__ == "__main__":
    unittest.main()
