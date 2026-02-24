"""Tests for blink energy feature extraction."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config
from test.blink_features.utils.helpers import assert_df_has_columns

PROJECT_ROOT = Path(__file__).resolve().parents[3]


LEGACY_ENERGY_METRICS = {
		"zero": {
				"blink_signal_energy": [
						"eog__zero__energy__blink_signal_energy_mean__EOG-EEG-eog_vert_left",
						"eog__zero__energy__blink_signal_energy_std__EOG-EEG-eog_vert_left",
						"eog__zero__energy__blink_signal_energy_cv__EOG-EEG-eog_vert_left",
						],
				"teager_kaiser_energy": [
						"eog__zero__energy__teager_kaiser_energy_mean__EOG-EEG-eog_vert_left",
						"eog__zero__energy__teager_kaiser_energy_std__EOG-EEG-eog_vert_left",
						"eog__zero__energy__teager_kaiser_energy_cv__EOG-EEG-eog_vert_left",
						],
				"blink_line_length": [
						"eog__zero__energy__blink_line_length_mean__EOG-EEG-eog_vert_left",
						"eog__zero__energy__blink_line_length_std__EOG-EEG-eog_vert_left",
						"eog__zero__energy__blink_line_length_cv__EOG-EEG-eog_vert_left",
						],
				"blink_velocity_integral": [
						"eog__zero__energy__blink_velocity_integral_mean__EOG-EEG-eog_vert_left",
						"eog__zero__energy__blink_velocity_integral_std__EOG-EEG-eog_vert_left",
						"eog__zero__energy__blink_velocity_integral_cv__EOG-EEG-eog_vert_left",
						],
				},
		"base": {
				"blink_signal_energy": [
						"eog__base__energy__blink_signal_energy_mean__EOG-EEG-eog_vert_left",
						"eog__base__energy__blink_signal_energy_std__EOG-EEG-eog_vert_left",
						"eog__base__energy__blink_signal_energy_cv__EOG-EEG-eog_vert_left",
						],
				"teager_kaiser_energy": [
						"eog__base__energy__teager_kaiser_energy_mean__EOG-EEG-eog_vert_left",
						"eog__base__energy__teager_kaiser_energy_std__EOG-EEG-eog_vert_left",
						"eog__base__energy__teager_kaiser_energy_cv__EOG-EEG-eog_vert_left",
						],
				"blink_line_length": [
						"eog__base__energy__blink_line_length_mean__EOG-EEG-eog_vert_left",
						"eog__base__energy__blink_line_length_std__EOG-EEG-eog_vert_left",
						"eog__base__energy__blink_line_length_cv__EOG-EEG-eog_vert_left",
						],
				"blink_velocity_integral": [
						"eog__base__energy__blink_velocity_integral_mean__EOG-EEG-eog_vert_left",
						"eog__base__energy__blink_velocity_integral_std__EOG-EEG-eog_vert_left",
						"eog__base__energy__blink_velocity_integral_cv__EOG-EEG-eog_vert_left",
						],
				},
		"tent": {
				"blink_signal_energy": [
						"eog__tent__energy__blink_signal_energy_mean__EOG-EEG-eog_vert_left",
						"eog__tent__energy__blink_signal_energy_std__EOG-EEG-eog_vert_left",
						"eog__tent__energy__blink_signal_energy_cv__EOG-EEG-eog_vert_left",
						],
				"teager_kaiser_energy": [
						"eog__tent__energy__teager_kaiser_energy_mean__EOG-EEG-eog_vert_left",
						"eog__tent__energy__teager_kaiser_energy_std__EOG-EEG-eog_vert_left",
						"eog__tent__energy__teager_kaiser_energy_cv__EOG-EEG-eog_vert_left",
						],
				"blink_line_length": [
						"eog__tent__energy__blink_line_length_mean__EOG-EEG-eog_vert_left",
						"eog__tent__energy__blink_line_length_std__EOG-EEG-eog_vert_left",
						"eog__tent__energy__blink_line_length_cv__EOG-EEG-eog_vert_left",
						],
				"blink_velocity_integral": [
						"eog__tent__energy__blink_velocity_integral_mean__EOG-EEG-eog_vert_left",
						"eog__tent__energy__blink_velocity_integral_std__EOG-EEG-eog_vert_left",
						"eog__tent__energy__blink_velocity_integral_cv__EOG-EEG-eog_vert_left",
						],
				},
		"half": {
				"blink_signal_energy": [
						"eog__half__energy__blink_signal_energy_mean__EOG-EEG-eog_vert_left",
						"eog__half__energy__blink_signal_energy_std__EOG-EEG-eog_vert_left",
						"eog__half__energy__blink_signal_energy_cv__EOG-EEG-eog_vert_left",
						],
				"teager_kaiser_energy": [
						"eog__half__energy__teager_kaiser_energy_mean__EOG-EEG-eog_vert_left",
						"eog__half__energy__teager_kaiser_energy_std__EOG-EEG-eog_vert_left",
						"eog__half__energy__teager_kaiser_energy_cv__EOG-EEG-eog_vert_left",
						],
				"blink_line_length": [
						"eog__half__energy__blink_line_length_mean__EOG-EEG-eog_vert_left",
						"eog__half__energy__blink_line_length_std__EOG-EEG-eog_vert_left",
						"eog__half__energy__blink_line_length_cv__EOG-EEG-eog_vert_left",
						],
				"blink_velocity_integral": [
						"eog__half__energy__blink_velocity_integral_mean__EOG-EEG-eog_vert_left",
						"eog__half__energy__blink_velocity_integral_std__EOG-EEG-eog_vert_left",
						"eog__half__energy__blink_velocity_integral_cv__EOG-EEG-eog_vert_left",
						],
				},
		"peak": {
				"blink_signal_energy": [
						"eog__peak__energy__blink_signal_energy_mean__EOG-EEG-eog_vert_left",
						"eog__peak__energy__blink_signal_energy_std__EOG-EEG-eog_vert_left",
						"eog__peak__energy__blink_signal_energy_cv__EOG-EEG-eog_vert_left",
						],
				"teager_kaiser_energy": [
						"eog__peak__energy__teager_kaiser_energy_mean__EOG-EEG-eog_vert_left",
						"eog__peak__energy__teager_kaiser_energy_std__EOG-EEG-eog_vert_left",
						"eog__peak__energy__teager_kaiser_energy_cv__EOG-EEG-eog_vert_left",
						],
				"blink_line_length": [
						"eog__peak__energy__blink_line_length_mean__EOG-EEG-eog_vert_left",
						"eog__peak__energy__blink_line_length_std__EOG-EEG-eog_vert_left",
						"eog__peak__energy__blink_line_length_cv__EOG-EEG-eog_vert_left",
						],
				"blink_velocity_integral": [
						"eog__peak__energy__blink_velocity_integral_mean__EOG-EEG-eog_vert_left",
						"eog__peak__energy__blink_velocity_integral_std__EOG-EEG-eog_vert_left",
						"eog__peak__energy__blink_velocity_integral_cv__EOG-EEG-eog_vert_left",
						],
				},
		}


class TestEnergyFeatures(unittest.TestCase):
    """Verify energy metrics computed from :class:`mne.Epochs`."""

    def setUp(self) -> None:
        """Load test epochs with blink metadata."""
        raw_path = (
            PROJECT_ROOT
            / "test"
            / "test_files"
            / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        segmentation_config = build_segment_config(raw)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )


    def test_single_channel_columns(self) -> None:
        """Returned DataFrame has expected columns for one channel."""
        ch = "EOG-EEG-eog_vert_left"
        df = compute_energy_features(self.epochs, picks=ch)


        for style in LEGACY_ENERGY_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)



if __name__ == "__main__":
    unittest.main()
