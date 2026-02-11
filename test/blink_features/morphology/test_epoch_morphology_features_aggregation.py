"""Integration of blink counts with morphology features.
In blinker, the morphology features includes the following
_LEGACY_MORPHOLOGY_METRICS = (
    "duration_zero",
    "duration_base",
    "duration_tent",
    "duration_half_base",
    "duration_half_zero",
    "closing_time_zero",
    "reopening_time_zero",
    "time_shut_zero",
    "time_shut_base",
    "closing_time_tent",
    "reopening_time_tent",
    "time_shut_tent",
    "inter_blink_max_amp",
	# "peak_time_blink",  			# TODO This metric is available in BLINKER but still not computed in pyblinker
	# "peak_time_tent",				# TODO This metric is available in BLINKER but still not computed in pyblinker
	# "peak_max_blink",				# TODO This metric is available in BLINKER but still not computed in pyblinker
	# "peak_max_tent",				# TODO This metric is available in BLINKER but still not computed in pyblinker
	# "inter_blink_max_vel_base",	# TODO This metric is available in BLINKER but still not computed in pyblinker
	# "inter_blink_max_vel_zero",	# TODO This metric is available in BLINKER but still not computed in pyblinker

	and is computed using
	from pyblinker.blink_features.morphology.core_metrics import (
    compute_blink_durations,
    compute_blink_peak_times,
    compute_time_base_shut,
    compute_time_zero_shut,
)
	"""
from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestMorphologyAggregation(unittest.TestCase):
    """Test aggregation of morphology features with blink counts."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        segmentation_config = build_segment_config(raw)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )

    def test_merge_blink_counts(self) -> None:
        """Joined DataFrame exposes why certain rows are NaN."""
        picks = ["EEG-E8"]
        df=compute_epoch_morphology_features(self.epochs, picks=picks)
        j=1
        # expected_cols = morphology_column_names(picks) + ["n_blinks"]
        # assert_df_has_columns(self, merged, expected_cols)
        # assert_numeric_or_nan(self, merged.iloc[0])

        # feature_cols = morphology_column_names(picks)
        # for idx, row in merged.iterrows():
        #     if row["n_blinks"] == 0:
        #         self.assertTrue(row[feature_cols].isna().all())
        #     else:
        #         self.assertTrue(np.isfinite(row[feature_cols]).any())


if __name__ == "__main__":
    unittest.main()
