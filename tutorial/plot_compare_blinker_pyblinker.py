import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.get_blink_positions import get_blink_position

from tutorial.make_report import make_report
# -----------------------------------------------------------------------------
# Logger configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tutorial helper functions
# -----------------------------------------------------------------------------
def load_and_prepare_blink_component(fif_path: Path, ch_name: str = "EEG-E8") -> np.ndarray:
	"""
	Load raw FIF data, pick the blink channel, ensure sfreq=100Hz, and return
	the blink component (1D numpy array).
	"""
	raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

	# Handle case-insensitive channel lookup
	if ch_name not in raw.ch_names:
		ch_map = {c.lower(): c for c in raw.ch_names}
		ch_name = ch_map.get(ch_name.lower(), ch_name)

	raw = raw.copy().pick_channels([ch_name])

	# Ensure sampling rate is 100 Hz
	if int(round(raw.info.get("sfreq", 100))) != 100:
		raw.resample(100)

	blink_comp = raw.get_data()[0].astype(np.float64)
	return blink_comp


def run_python_fitblinks(blink_comp: np.ndarray) -> pd.DataFrame:
	"""
	Detect blink positions, run FitBlinks, apply MATLAB compatibility fixes,
	and return the final Python dataframe output.
	"""
	# Blink detection parameters
	params = dict(
		min_event_len=0.05,
		std_threshold=1.5,
		sfreq=100,
		)

	params_default = default_setting.DEFAULT_PARAMS.copy()

	# Detect blink positions
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
		params=params_default,
		)
	fitblinks.dprocess()

	df_output = fitblinks.frame_blinks.copy()

	# -------------------------------------------------------------------------
	# MATLAB compatibility: index correction (MATLAB is 1-based)
	# -------------------------------------------------------------------------
	columns_to_increment = [
			"max_blink", "start_blink", "end_blink",
			"outer_start", "outer_end",
			"left_zero", "right_zero",
			"max_pos_vel_frame", "max_neg_vel_frame",
			"left_base", "right_base",
			"left_zero_half_height", "right_zero_half_height",
			"left_base_half_height", "right_base_half_height",
			"x_intersect",
			"right_x_intercept", "left_x_intercept",
			]
	df_output[columns_to_increment] += 1

	df_output["left_range"] = df_output["left_range"].apply(lambda x: [v + 1 for v in x])
	df_output["right_range"] = df_output["right_range"].apply(lambda x: [v + 1 for v in x])

	# -------------------------------------------------------------------------
	# Reorder columns (keep consistent ordering for reporting and comparison)
	# -------------------------------------------------------------------------
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

	df_py = df_output[column_order].reset_index(drop=True)
	return df_py


def load_matlab_reference(mat_path: Path) -> pd.DataFrame:
	"""
	Load MATLAB FitBlinks reference output and return it as a Python dataframe
	with matching column names and ordering.
	"""
	if not mat_path.exists():
		raise FileNotFoundError(f"Missing MATLAB file: {mat_path}")

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

	df_mat = df_mat.rename(columns=matlab_to_python)[column_order]
	return df_mat


def compare_outputs(df_py: pd.DataFrame, df_mat: pd.DataFrame) -> None:
	"""
	Tutorial-style validation: compare Python output vs MATLAB reference output.
	"""
	if len(df_py) != len(df_mat):
		raise ValueError(
			f"Different number of blinks: py={len(df_py)} mat={len(df_mat)}"
			)

	pd.testing.assert_frame_equal(
		df_py,
		df_mat,
		check_dtype=False,
		)


# -----------------------------------------------------------------------------
# Tutorial main flow
# -----------------------------------------------------------------------------
if __name__ == "__main__":
	# -----------------------------
	# Paths
	# -----------------------------
	base_path = Path(__file__).resolve().parents[0] / "migration_files"
	# fif_path = Path("test/test_files/ear_eog_raw.fif")
	# mat_path = base_path / "step1bii_data_output_process_FitBlinks_rpb.mat"
	mat_path=r'C:\Users\balan\IdeaProjects\pyblinker\test\migration_files\step1bii_data_output_process_FitBlinks_rpb.mat'
	fif_path=r"C:\Users\balan\IdeaProjects\pyblinker\test\test_files\ear_eog_raw.fif"
	# -----------------------------
	# Step 1) Load data + extract blink component
	# -----------------------------
	blink_comp = load_and_prepare_blink_component(
		fif_path=fif_path,
		ch_name="EEG-E8",
		)

	# -----------------------------
	# Step 2) Run Python FitBlinks pipeline
	# -----------------------------
	df_py = run_python_fitblinks(blink_comp)

	# -----------------------------
	# Step 3) Load MATLAB reference output
	# -----------------------------
	df_mat = load_matlab_reference(mat_path)

	# -----------------------------
	# Step 4) Compare outputs (optional tutorial validation)
	# -----------------------------
	compare_outputs(df_py, df_mat)
	print("✅ Python FitBlinks output matches MATLAB reference output.")

	# -------------------------------------------------------------------------
	# Step 5) Generate MNE reports via make_report()
	# -------------------------------------------------------------------------
	# NOTE: make_report() is assumed to already exist.
	# Example usage requested:

	make_report(df_py, "plot_python_base")
	make_report(df_mat, "plot_matlab_base")
