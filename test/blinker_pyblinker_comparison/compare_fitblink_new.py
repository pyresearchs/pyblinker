import unittest
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.get_blink_positions import get_blink_position


# -----------------------------------------------------------------------------
# Logger configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
from test.blinker_pyblinker_comparison.utils import load_matlab_blink_positions

# -----------------------------------------------------------------------------
# Test class
# -----------------------------------------------------------------------------
class TestFitBlinks(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		"""
		Load test data, run FitBlinks, and load MATLAB reference output.
		"""
		base_path = Path(__file__).resolve().parents[1] / "migration_files"
		fif_path = Path("test/test_files/ear_eog_raw.fif")
		mat_expected = Path("test/test_files/step_a_extract_blinks_resamp-100.mat")

		# Load MATLAB positions (2 x N), convert to DataFrame with 0-based indices
		# In MATLAB implementaion, we use the step_a_get_blink_position.m output as input into directly the blinkFits = fitBlinks(signalData(k).signal, signalData(k).blinkPositions); But for the purpose of Python validation, we will re-run the step of getting blink positions here.

		arr = load_matlab_blink_positions(mat_expected)
		df_mat = pd.DataFrame({
				"start_blink": arr[0, :].astype(np.int64),
				"end_blink": arr[1, :].astype(np.int64),
				})
		# ---------------------------------------------------------------------
		# Load raw FIF data
		# ---------------------------------------------------------------------
		raw = mne.io.read_raw_fif(
			fif_path, preload=True, verbose="ERROR"
			)

		ch_name = "EEG-E8"


		raw = raw.copy().pick_channels([ch_name])

		if int(round(raw.info.get("sfreq", 100))) != 100:
			raw.resample(100)

		blink_comp = raw.get_data()[0].astype(np.float64)

		# ---------------------------------------------------------------------
		# Detect blink positions
		# ---------------------------------------------------------------------
		params = dict(
			min_event_len=0.05,
			std_threshold=1.5,
			sfreq=100,
			)

		params_default = default_setting.DEFAULT_PARAMS.copy()

		df_positions = get_blink_position(
			params,
			blink_component=blink_comp,
			ch="No_channel",
			progress_bar=False,
			)

		# ---------------------------------------------------------------------
		# Run FitBlinks
		# ---------------------------------------------------------------------
		fitblinks = FitBlinks(
			candidate_signal=blink_comp,
			df=df_positions,
			params=params_default,
			)
		fitblinks.dprocess()

		df_output = fitblinks.frame_blinks.copy()

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
		# Reorder columns
		# ---------------------------------------------------------------------
		column_order = [
				"max_blink", "max_value",
				"outer_start", "outer_end",
				"left_zero", "right_zero",
				"left_base", "right_base",
				"left_base_half_height", "right_base_half_height",
				"left_zero_half_height", "right_zero_half_height",
				"left_range", "right_range",
				"left_slope", "right_slope",
				"aver_left_velocity", "aver_right_velocity",
				"leftR2", "rightR2",
				"x_intersect", "y_intersect",
				"left_x_intercept", "right_x_intercept",
				]

		cls.df_py = df_output[column_order].reset_index(drop=True)

		# ---------------------------------------------------------------------
		# Load MATLAB reference output
		# ---------------------------------------------------------------------
		mat_path = base_path / "step1bii_data_output_process_FitBlinks_rpb.mat"
		assert mat_path.exists(), f"Missing MATLAB file: {mat_path}"

		mat_data = loadmat(
			mat_path,
			squeeze_me=True,
			simplify_cells=True,
			struct_as_record=False,
			)

		df_mat = pd.DataFrame(mat_data["blinkFits"]).reset_index(drop=True)
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
		cls.df_mat = df_mat.rename(columns=matlab_to_python)[column_order]
	# -------------------------------------------------------------------------
	# Tests
	# -------------------------------------------------------------------------
	def test_dataframe_equality(self):
		"""
		Compare Python FitBlinks output with MATLAB reference output.
		Both should return 359 blinks with identical properties.
		"""
		self.assertEqual(
			len(self.df_py),
			len(self.df_mat),
			f"Different number of blinks: "
			f"py={len(self.df_py)} mat={len(self.df_mat)}",
			)

		pd.testing.assert_frame_equal(
			self.df_py,
			self.df_mat,
			check_dtype=False,
			)


# -----------------------------------------------------------------------------
# Run tests
# -----------------------------------------------------------------------------
if __name__ == "__main__":
	unittest.main()
