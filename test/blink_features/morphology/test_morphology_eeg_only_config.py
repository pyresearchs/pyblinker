"""Integration coverage for epoch morphology aggregation columns."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

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
		"""Test that morphology features can be computed from epochs with metadata from Excel input."""
		pass
		# excel_path = "expected_output_new_naming.xlsx"

if __name__ == "__main__":
    unittest.main()
