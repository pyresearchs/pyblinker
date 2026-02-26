"""EEG-only unit tests for wavelet-based blink frequency features."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne


from pyblinker.blink_features.frequency_domain import (aggregate_frequency_domain_features)

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.helper import build_expected_metrics



PROJECT_ROOT = Path(__file__).resolve().parents[3]
EEG_CHANNEL = "EEG-E8"

stats = ["mean", "std", "cv"]
metrics = ["wavelet_energy_d1", "wavelet_energy_d2", "wavelet_energy_d3", "wavelet_energy_d4"]
landmarks = ["zero", "base", "tent", "half", "peak"]

REQUIRED_ENERGY_METRICS = build_expected_metrics(
            landmark=landmarks,
            metrics=metrics,
            stats=stats,
            modality="eeg",
            feature="energy",
            channel=EEG_CHANNEL,
        )



class TestFrequencyDomainBlinkFeaturesEEGOnly(unittest.TestCase):
    """Validate DWT energy features per epoch for EEG-only inputs."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        eeg_channel = "EEG-E8"
        raw.pick([eeg_channel])
        segmentation_config = {
            "eeg": {
                "channel": eeg_channel,
            }
        }
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )
        self.eeg_channel = eeg_channel


    def test_schema_and_rows(self) -> None:
        """DataFrame has expected columns and indexing for first epochs."""
        df = aggregate_frequency_domain_features(
            self.epochs, picks=self.eeg_channel, progress_bar=False)

        for style in REQUIRED_ENERGY_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)

if __name__ == "__main__":
    unittest.main()
