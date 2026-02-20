import logging
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.get_blink_positions import get_blink_position
from pyblinker.utils.statistics_utils import get_blink_statistic
from test.blinker_pyblinker_comparison.utils import get_test_file_path

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
		fif_path = get_test_file_path("ear_eog_raw.fif")

		mat_path = base_path / "test_files/step_a_extract_blinks_resamp-100.mat"
		mat_data = loadmat(
			mat_path,
			squeeze_me=True,
			simplify_cells=True,
			struct_as_record=False,
			)

		cls.df_mat = mat_data["blinks"]["signalData"]
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
		cls.blink_stats = get_blink_statistic(
			df,
			params["z_thresholds"],
			signal=blink_comp,
			)
		cls.blink_stats["ch"] = "EEG"



	def test_key_equality(self):
		# Map MATLAB keys -> Python keys
		key_map = {
			"numberBlinks": "number_blinks",
			"numberGoodBlinks": "number_good_blinks",
			"blinkAmpRatio": "blink_amp_ratio",
			"cutoff": "cutoff",
			"bestMedian": "best_median",
			"bestRobustStd": "best_robust_std",
			"goodRatio": "good_ratio",
		}

		# Create pandas DataFrames for comparison
		df_expected = pd.DataFrame(
			[{py_key: self.df_mat[mat_key] for mat_key, py_key in key_map.items()}]
		)
		df_actual = pd.DataFrame(
			[{py_key: self.blink_stats[py_key] for py_key in key_map.values()}]
		)

		# Ensure columns are in the same order
		df_expected = df_expected[sorted(df_expected.columns)]
		df_actual = df_actual[sorted(df_actual.columns)]

		pd.testing.assert_frame_equal(df_actual, df_expected, check_dtype=False, atol=1e-4)
if __name__ == "__main__":
	unittest.main(verbosity=0)
