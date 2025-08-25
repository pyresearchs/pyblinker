"""Tests for blink energy feature extraction."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.utils import slice_raw_into_mne_epochs
from unit_test.blink_features.utils.helpers import assert_df_has_columns

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEnergyFeatures(unittest.TestCase):
    """Verify energy metrics computed from :class:`mne.Epochs`."""

    def setUp(self) -> None:
        """Load test epochs with blink metadata."""
        raw_path = (
            PROJECT_ROOT
            / "unit_test"
            / "test_files"
            / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_single_channel_columns(self) -> None:
        """Returned DataFrame has expected columns for one channel."""
        ch = "EEG-E8"
        df = compute_energy_features(self.epochs, picks=ch)
        expected = [
            f"blink_signal_energy_mean_{ch}",
            f"blink_signal_energy_std_{ch}",
            f"blink_signal_energy_cv_{ch}",
            f"teager_kaiser_energy_mean_{ch}",
            f"teager_kaiser_energy_std_{ch}",
            f"teager_kaiser_energy_cv_{ch}",
            f"blink_line_length_mean_{ch}",
            f"blink_line_length_std_{ch}",
            f"blink_line_length_cv_{ch}",
            f"blink_velocity_integral_mean_{ch}",
            f"blink_velocity_integral_std_{ch}",
            f"blink_velocity_integral_cv_{ch}",
        ]
        assert_df_has_columns(self, df, expected)
        self.assertEqual(len(df), len(self.epochs))

    def test_epoch_without_blinks_is_nan(self) -> None:
        """Epochs lacking blinks yield NaNs for all metrics."""
        df = compute_energy_features(self.epochs, picks="EAR-avg_ear")
        no_blink_idx = self.epochs.metadata.index[
            self.epochs.metadata["blink_onset"].isna()
        ][0]
        self.assertTrue(df.loc[no_blink_idx].isna().all())

    def test_multiple_channels(self) -> None:
        """Processing multiple channels produces suffixed columns."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left"]
        df = compute_energy_features(self.epochs, picks=picks)
        for ch in picks:
            prefix = [
                f"blink_signal_energy_mean_{ch}",
                f"teager_kaiser_energy_mean_{ch}",
                f"blink_line_length_mean_{ch}",
                f"blink_velocity_integral_mean_{ch}",
            ]
            assert_df_has_columns(self, df, prefix)

    def test_missing_channel_raises(self) -> None:
        """Requesting an unknown channel results in ``ValueError``."""
        with self.assertRaises(ValueError):
            compute_energy_features(self.epochs, picks="bogus")


if __name__ == "__main__":
    unittest.main()
