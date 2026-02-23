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

REQUIRED_LEGACY_ENERGY_METRICS = {
        "zero": {
                "blink_signal_energy": [
                        "eeg__zero__energy__blink_signal_energy_mean__EEG-E8",
                        "eeg__zero__energy__blink_signal_energy_std__EEG-E8",
                        "eeg__zero__energy__blink_signal_energy_cv__EEG-E8",
                        ],
                "teager_kaiser_energy": [
                        "eeg__zero__energy__teager_kaiser_energy_mean__EEG-E8",
                        "eeg__zero__energy__teager_kaiser_energy_std__EEG-E8",
                        "eeg__zero__energy__teager_kaiser_energy_cv__EEG-E8",
                        ],
                "blink_line_length": [
                        "eeg__zero__energy__blink_line_length_mean__EEG-E8",
                        "eeg__zero__energy__blink_line_length_std__EEG-E8",
                        "eeg__zero__energy__blink_line_length_cv__EEG-E8",
                        ],
                "blink_velocity_integral": [
                        "eeg__zero__energy__blink_velocity_integral_mean__EEG-E8",
                        "eeg__zero__energy__blink_velocity_integral_std__EEG-E8",
                        "eeg__zero__energy__blink_velocity_integral_cv__EEG-E8",
                        ],
                },

        "base": {
                "blink_signal_energy": [
                        "eeg__base__energy__blink_signal_energy_mean__EEG-E8",
                        "eeg__base__energy__blink_signal_energy_std__EEG-E8",
                        "eeg__base__energy__blink_signal_energy_cv__EEG-E8",
                        ],
                "teager_kaiser_energy": [
                        "eeg__base__energy__teager_kaiser_energy_mean__EEG-E8",
                        "eeg__base__energy__teager_kaiser_energy_std__EEG-E8",
                        "eeg__base__energy__teager_kaiser_energy_cv__EEG-E8",
                        ],
                "blink_line_length": [
                        "eeg__base__energy__blink_line_length_mean__EEG-E8",
                        "eeg__base__energy__blink_line_length_std__EEG-E8",
                        "eeg__base__energy__blink_line_length_cv__EEG-E8",
                        ],
                "blink_velocity_integral": [
                        "eeg__base__energy__blink_velocity_integral_mean__EEG-E8",
                        "eeg__base__energy__blink_velocity_integral_std__EEG-E8",
                        "eeg__base__energy__blink_velocity_integral_cv__EEG-E8",
                        ],
                },

        "tent": {
                "blink_signal_energy": [
                        "eeg__tent__energy__blink_signal_energy_mean__EEG-E8",
                        "eeg__tent__energy__blink_signal_energy_std__EEG-E8",
                        "eeg__tent__energy__blink_signal_energy_cv__EEG-E8",
                        ],
                "teager_kaiser_energy": [
                        "eeg__tent__energy__teager_kaiser_energy_mean__EEG-E8",
                        "eeg__tent__energy__teager_kaiser_energy_std__EEG-E8",
                        "eeg__tent__energy__teager_kaiser_energy_cv__EEG-E8",
                        ],
                "blink_line_length": [
                        "eeg__tent__energy__blink_line_length_mean__EEG-E8",
                        "eeg__tent__energy__blink_line_length_std__EEG-E8",
                        "eeg__tent__energy__blink_line_length_cv__EEG-E8",
                        ],
                "blink_velocity_integral": [
                        "eeg__tent__energy__blink_velocity_integral_mean__EEG-E8",
                        "eeg__tent__energy__blink_velocity_integral_std__EEG-E8",
                        "eeg__tent__energy__blink_velocity_integral_cv__EEG-E8",
                        ],
                },

        "half": {
                "blink_signal_energy": [
                        "eeg__half__energy__blink_signal_energy_mean__EEG-E8",
                        "eeg__half__energy__blink_signal_energy_std__EEG-E8",
                        "eeg__half__energy__blink_signal_energy_cv__EEG-E8",
                        ],
                "teager_kaiser_energy": [
                        "eeg__half__energy__teager_kaiser_energy_mean__EEG-E8",
                        "eeg__half__energy__teager_kaiser_energy_std__EEG-E8",
                        "eeg__half__energy__teager_kaiser_energy_cv__EEG-E8",
                        ],
                "blink_line_length": [
                        "eeg__half__energy__blink_line_length_mean__EEG-E8",
                        "eeg__half__energy__blink_line_length_std__EEG-E8",
                        "eeg__half__energy__blink_line_length_cv__EEG-E8",
                        ],
                "blink_velocity_integral": [
                        "eeg__half__energy__blink_velocity_integral_mean__EEG-E8",
                        "eeg__half__energy__blink_velocity_integral_std__EEG-E8",
                        "eeg__half__energy__blink_velocity_integral_cv__EEG-E8",
                        ],
                },

        "peak": {
                "blink_signal_energy": [
                        "eeg__peak__energy__blink_signal_energy_mean__EEG-E8",
                        "eeg__peak__energy__blink_signal_energy_std__EEG-E8",
                        "eeg__peak__energy__blink_signal_energy_cv__EEG-E8",
                        ],
                "teager_kaiser_energy": [
                        "eeg__peak__energy__teager_kaiser_energy_mean__EEG-E8",
                        "eeg__peak__energy__teager_kaiser_energy_std__EEG-E8",
                        "eeg__peak__energy__teager_kaiser_energy_cv__EEG-E8",
                        ],
                "blink_line_length": [
                        "eeg__peak__energy__blink_line_length_mean__EEG-E8",
                        "eeg__peak__energy__blink_line_length_std__EEG-E8",
                        "eeg__peak__energy__blink_line_length_cv__EEG-E8",
                        ],
                "blink_velocity_integral": [
                        "eeg__peak__energy__blink_velocity_integral_mean__EEG-E8",
                        "eeg__peak__energy__blink_velocity_integral_std__EEG-E8",
                        "eeg__peak__energy__blink_velocity_integral_cv__EEG-E8",
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
        ch = "EEG-E8"
        df = compute_energy_features(self.epochs, picks=ch)


        for style in REQUIRED_LEGACY_ENERGY_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)



if __name__ == "__main__":
    unittest.main()
