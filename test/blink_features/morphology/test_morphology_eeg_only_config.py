"""Integration coverage for epoch morphology aggregation columns."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from pyblinker.blink_features.morphology.epoch_features import _available_styles, _style_windows
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EEG_CHANNEL = "EEG-E8"

_REQUIRED_LEGACY_MORPHOLOGY_METRICS = {
		"zero": {
				"duration_zero": [
						"eeg__zero__morphology__duration_zero_mean__EEG-E8",
						"eeg__zero__morphology__duration_zero_std__EEG-E8",
						"eeg__zero__morphology__duration_zero_cv__EEG-E8",
						],
				"closing_time_zero": [
						"eeg__zero__morphology__closing_time_zero_mean__EEG-E8",
						"eeg__zero__morphology__closing_time_zero_std__EEG-E8",
						"eeg__zero__morphology__closing_time_zero_cv__EEG-E8",
						],
				"reopening_time_zero": [
						"eeg__zero__morphology__reopening_time_zero_mean__EEG-E8",
						"eeg__zero__morphology__reopening_time_zero_std__EEG-E8",
						"eeg__zero__morphology__reopening_time_zero_cv__EEG-E8",
						],
				"time_shut_zero": [
						"eeg__zero__morphology__time_shut_zero_mean__EEG-E8",
						"eeg__zero__morphology__time_shut_zero_std__EEG-E8",
						"eeg__zero__morphology__time_shut_zero_cv__EEG-E8",
						],
				},
		"base": {
				"duration_base": [
						"eeg__base__morphology__duration_base_mean__EEG-E8",
						"eeg__base__morphology__duration_base_std__EEG-E8",
						"eeg__base__morphology__duration_base_cv__EEG-E8",
						],
				"time_shut_base": [
						"eeg__base__morphology__time_shut_base_mean__EEG-E8",
						"eeg__base__morphology__time_shut_base_std__EEG-E8",
						"eeg__base__morphology__time_shut_base_cv__EEG-E8",
						],
				},
		"tent": {
				"duration_tent": [
						"eeg__tent__morphology__duration_tent_mean__EEG-E8",
						"eeg__tent__morphology__duration_tent_std__EEG-E8",
						"eeg__tent__morphology__duration_tent_cv__EEG-E8",
						],
				"closing_time_tent": [
						"eeg__tent__morphology__closing_time_tent_mean__EEG-E8",
						"eeg__tent__morphology__closing_time_tent_std__EEG-E8",
						"eeg__tent__morphology__closing_time_tent_cv__EEG-E8",
						],
				"reopening_time_tent": [
						"eeg__tent__morphology__reopening_time_tent_mean__EEG-E8",
						"eeg__tent__morphology__reopening_time_tent_std__EEG-E8",
						"eeg__tent__morphology__reopening_time_tent_cv__EEG-E8",
						],
				"time_shut_tent": [
						"eeg__tent__morphology__time_shut_tent_mean__EEG-E8",
						"eeg__tent__morphology__time_shut_tent_std__EEG-E8",
						"eeg__tent__morphology__time_shut_tent_cv__EEG-E8",
						],
				},
		"half": {
				"duration_half_base": [
						"eeg__half__morphology__duration_half_base_mean__EEG-E8",
						"eeg__half__morphology__duration_half_base_std__EEG-E8",
						"eeg__half__morphology__duration_half_base_cv__EEG-E8",
						],
				"duration_half_zero": [
						"eeg__half__morphology__duration_half_zero_mean__EEG-E8",
						"eeg__half__morphology__duration_half_zero_std__EEG-E8",
						"eeg__half__morphology__duration_half_zero_cv__EEG-E8",
						],
				},
		"peak": {
				"peak_time_blink": [
						"eeg__peak__morphology__peak_time_blink_mean__EEG-E8",
						"eeg__peak__morphology__peak_time_blink_std__EEG-E8",
						"eeg__peak__morphology__peak_time_blink_cv__EEG-E8",
						],
				"peak_time_tent": [
						"eeg__peak__morphology__peak_time_tent_mean__EEG-E8",
						"eeg__peak__morphology__peak_time_tent_std__EEG-E8",
						"eeg__peak__morphology__peak_time_tent_cv__EEG-E8",
						],
				"peak_max_blink": [
						"eeg__peak__morphology__peak_max_blink_mean__EEG-E8",
						"eeg__peak__morphology__peak_max_blink_std__EEG-E8",
						"eeg__peak__morphology__peak_max_blink_cv__EEG-E8",
						],
				"peak_max_tent": [
						"eeg__peak__morphology__peak_max_tent_mean__EEG-E8",
						"eeg__peak__morphology__peak_max_tent_std__EEG-E8",
						"eeg__peak__morphology__peak_max_tent_cv__EEG-E8",
						],
				},
		"inter_blink": {
				"inter_blink_max_amp": [
						"eeg__inter_blink__morphology__inter_blink_max_amp_mean__EEG-E8",
						"eeg__inter_blink__morphology__inter_blink_max_amp_std__EEG-E8",
						"eeg__inter_blink__morphology__inter_blink_max_amp_cv__EEG-E8",
						],
				},
		}


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


    def test_style_windows_uses_landmark_frames_when_available(self) -> None:
        """Window extraction should use landmark frame boundaries before onset/duration."""
        metadata_row = {
            "start__left_zero__eeg": [10, 30],
            "end__right_zero__eeg": [16, 40],
            "onset__zero__eeg": [1.0, 2.0],
            "duration__zero__eeg": [0.0, 0.0],
        }

        windows = _style_windows(metadata_row, "eeg", "zero", sfreq=256.0, n_times=2560)

        self.assertEqual(windows, [(10, 16), (30, 40)])

    def test_epoch_output_contains_expected_morphology_features(self) -> None:
        """Epoch output includes expected style-aware and legacy morphology fields."""
        df = compute_epoch_morphology_features(self.epochs, picks=[EEG_CHANNEL])
        styles = _available_styles(tuple(self.epochs.metadata.columns), "eeg")
        self.assertTrue(styles)

        for style in styles:
            expected = f"eeg__{style}__morphology__duration_mean__{EEG_CHANNEL}"
            self.assertIn(expected, df.columns)

        for style in _REQUIRED_LEGACY_MORPHOLOGY_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)

        self.assertGreater(df.notna().sum().sum(), 0)

    def test_compare_with_excel_input(self) -> None:
        """Compare legacy-mean morphology outputs against Excel fixture values.
  		There is an issue with the excel, need to refine the header name, but basiclly, all the value is the same, and we succefully migrate to new headername convention
		the problematic Excel header is:

	eeg__peak__morphology__peak_time_tent_mean__EEG-E8 (the first one, without .1)

	It is mislabeled: its values match peak_max_tent, not peak_time_tent. That is why the test currently has this override mapping.

	What is correct in the Excel right now
	eeg__peak__morphology__peak_time_tent_mean__EEG-E8.1 matches the real output column eeg__peak__morphology__peak_time_tent_mean__EEG-E8.

	eeg__peak__morphology__peak_time_tent_mean__EEG-E8 (without .1) actually matches eeg__peak__morphology__peak_max_tent_mean__EEG-E8.

	So effectively:

	rename current ...peak_time_tent_mean... (first occurrence) → ...peak_max_tent_mean...

	keep the second one as ...peak_time_tent_mean... (or deduplicate cleanly in your fixture process)

	Why this appears in test code
	The test uses column_overrides specifically to correct this fixture header mismatch during comparison, not because feature extraction is wrong.


	"""
        df = compute_epoch_morphology_features(self.epochs, picks=[EEG_CHANNEL])

        fixture_path = (
            PROJECT_ROOT
            / "test"
            / "blink_features"
            / "morphology"
            / "expected_output_new_naming.xlsx"
        )
        expected_df = pd.read_excel(fixture_path)
        expected_df = expected_df.drop(columns=["Unnamed: 0"], errors="ignore")

        self.assertEqual(len(df), len(expected_df))

        column_overrides = {
            "eeg__peak__morphology__peak_time_tent_mean__EEG-E8": "eeg__peak__morphology__peak_max_tent_mean__EEG-E8",
            "eeg__peak__morphology__peak_time_tent_mean__EEG-E8.1": "eeg__peak__morphology__peak_time_tent_mean__EEG-E8",
        }

        for expected_col in expected_df.columns:
            output_col = column_overrides.get(expected_col, expected_col.split(".", 1)[0])
            self.assertIn(output_col, df.columns)

            expected_values = expected_df[expected_col].to_numpy(dtype=float)
            actual_values = df[output_col].to_numpy(dtype=float)
            self.assertTrue(
                np.allclose(actual_values, expected_values, atol=1e-8, equal_nan=True),
                msg=f"Column mismatch: fixture={expected_col} output={output_col}",
            )

if __name__ == "__main__":
    unittest.main()
