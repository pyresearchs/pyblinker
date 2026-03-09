"""Tests for blink energy feature extraction."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config
from test.helper import build_expected_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EEG_CHANNEL = "EEG-E8"


stats = ["mean", "std", "cv"]
metrics = [
    "blink_signal_energy",
    "teager_kaiser_energy",
    "blink_line_length",
    "blink_velocity_integral",
]
landmarks = ["zero", "base", "tent", "half", "peak"]

REQUIRED_ENERGY_METRICS = build_expected_metrics(
    landmark=landmarks,
    metrics=metrics,
    stats=stats,
    modality="eeg",
    feature="energy",
    channel=EEG_CHANNEL,
)


class TestEnergyFeatures(unittest.TestCase):
    """Verify energy metrics computed from :class:`mne.Epochs`."""

    def setUp(self) -> None:
        """Load test epochs with blink metadata."""
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

    def test_single_channel_columns(self) -> None:
        """Returned DataFrame has expected columns for one channel."""
        df = compute_energy_features(self.epochs, picks=EEG_CHANNEL)

        for style in REQUIRED_ENERGY_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)


if __name__ == "__main__":
    unittest.main()
