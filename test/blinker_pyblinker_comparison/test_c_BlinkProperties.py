import unittest
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.get_blink_positions import get_blink_position
from pyblinker.blink_features.waveform_features.extract_blink_properties import (
    BlinkProperties,
)
from pyblinker.utils.statistics_utils import get_good_blink_mask, get_blink_statistic
from scipy.io import loadmat
from test.blinker_pyblinker_comparison.utils import test_file_path

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBlinkProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Load test data, run FitBlinks, and compute BlinkProperties output.
        """
        base_path = Path(__file__).resolve().parents[1]
        fif_path = test_file_path("ear_eog_raw.fif")

        mat_path = base_path / "test_files/step_c_blink_properties.mat"
        mat_data = loadmat(
            mat_path,
            squeeze_me=True,
            simplify_cells=True,
            struct_as_record=False,
            )

        cls.df_mat = pd.DataFrame(mat_data["blinkProperties"]).reset_index(drop=True)
        # Load raw FIF data
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")

        ch_name = "EEG-E8"
        if ch_name not in raw.ch_names:
            # Case-insensitive fallback
            ch_map = {c.lower(): c for c in raw.ch_names}
            ch_name = ch_map.get(ch_name.lower(), ch_name)

        raw = raw.copy().pick_channels([ch_name])

        if int(round(raw.info.get("sfreq", 100))) != 100:
            raw.resample(100)

        blink_comp = raw.get_data()[0].astype(np.float64)

        params = default_setting.DEFAULT_PARAMS.copy()

        df_positions = get_blink_position(
            params,
            blink_component=blink_comp,
            ch="No_channel",
            progress_bar=False,
        )

        # Run FitBlinks
        fitblinks = FitBlinks(
            candidate_signal=blink_comp,
            df=df_positions,
            params=params,
        )
        fitblinks.dprocess()
        df = fitblinks.frame_blinks.copy()

        # assert len(df.row) == 355
        # STEP 3: Extract blink statistics extractBlinkProperties.m
        # Calculate an amplitude criterion (frames in blink to those out) and Now calculate the cutoff ratios -- use default for the values
        blink_stats = get_blink_statistic(
            df,
            params["z_thresholds"],
            signal=blink_comp,
            )
        blink_stats["ch"] = "EEG"
        # There is a step for << Reduce the number of candidate signals based on the blink amp ratios >>, but we move it to channel selection step.

        # STEP 4: Get good blink mask extractBlinkProperties.m
        _, df = get_good_blink_mask(
            df,
            blink_stats["best_median"],
            blink_stats["best_robust_std"],
            params["z_thresholds"],
            )
        # # What happen if no good blinks are found or all blinks are bad?
        # if df.empty and verbose:
        #     logger.warning("No good blinks found in channel: %s", channel)
        #     return
        # STEP 5: Compute blink properties
        df_in = df.copy()
        df_out = BlinkProperties(
            blink_comp,
            df_in,
            params["sfreq"],
            params,
            ).df

        condition_1 = df_out["pos_amp_vel_ratio_zero"] < params["p_avr_threshold"]
        condition_2 = df_out["max_value"] < (
                blink_stats["best_median"] - blink_stats["best_robust_std"]
        )
        cls.df_py = df_out[~(condition_1 & condition_2)]


    def test_dataframe_equality(self):
        """
        Compare Python FitBlinks output with MATLAB reference output.
        Both should return 359 blinks with identical properties.
        """
        # pass
        self.assertEqual(
            len(self.df_py),
            len(self.df_mat),
            f"Different number of blinks: "
            f"py={len(self.df_py)} mat={len(self.df_mat)}",
        )

        column_mapping = {
            "duration_base": "durationBase",
            "duration_zero": "durationZero",
            "duration_tent": "durationTent",
            "duration_half_base": "durationHalfBase",
            "duration_half_zero": "durationHalfZero",
            "inter_blink_max_amp": "interBlinkMaxAmp",
            "inter_blink_max_vel_base": "interBlinkMaxVelBase",
            "inter_blink_max_vel_zero": "interBlinkMaxVelZero",
            "neg_amp_vel_ratio_base": "negAmpVelRatioBase",
            "pos_amp_vel_ratio_base": "posAmpVelRatioBase",
            "neg_amp_vel_ratio_zero": "negAmpVelRatioZero",
            "pos_amp_vel_ratio_zero": "posAmpVelRatioZero",
            "neg_amp_vel_ratio_tent": "negAmpVelRatioTent",
            "pos_amp_vel_ratio_tent": "posAmpVelRatioTent",
            "time_shut_base": "timeShutBase",
            "time_shut_zero": "timeShutZero",
            "time_shut_tent": "timeShutTent",
            "closing_time_zero": "closingTimeZero",
            "reopening_time_zero": "reopeningTimeZero",
            "closing_time_tent": "closingTimeTent",
            "reopening_time_tent": "reopeningTimeTent",
            "peak_time_blink": "peakTimeBlink",
            "peak_time_tent": "peakTimeTent",
            "peak_max_blink": "peakMaxBlink",
            "peak_max_tent": "peakMaxTent",
        }
        #
        required_py_columns = list(column_mapping.keys())
        for col in required_py_columns:
            self.assertIn(col, self.df_py.columns, f"Missing Python column: {col}")

        for col in column_mapping.values():
            self.assertIn(col, self.df_mat.columns, f"Missing MATLAB column: {col}")

        tolerance = 1e-3
        mismatches = []

        for row_idx in range(len(self.df_py)):
            for py_col, mat_col in column_mapping.items():
                py_val = self.df_py.at[row_idx, py_col]
                mat_val = self.df_mat.at[row_idx, mat_col]
                if pd.isna(py_val) and pd.isna(mat_val):
                    continue
                if pd.isna(py_val) != pd.isna(mat_val):
                    mismatches.append((row_idx, py_col, mat_col, mat_val, py_val))
                    continue
                if not np.isclose(py_val, mat_val, atol=tolerance, rtol=0):
                    mismatches.append((row_idx, py_col, mat_col, mat_val, py_val))

        if mismatches:
            details = [
                f"row={row} py_col={py_col} mat_col={mat_col} "
                f"mat={mat_val} py={py_val}"
                for row, py_col, mat_col, mat_val, py_val in mismatches
            ]
            raise AssertionError(
                f"{len(mismatches)} mismatches found.\n" + "\n".join(details)
            )
if __name__ == "__main__":
    unittest.main(verbosity=2)
