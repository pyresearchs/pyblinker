"""EEG-only unit tests for wavelet-based blink frequency features."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.frequency_domain import (
    aggregate_frequency_domain_features,
)

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot



PROJECT_ROOT = Path(__file__).resolve().parents[3]


REQUIRED_ENERGY_METRICS= {
        "zero": {
                "wavelet_energy_d1": [
                        "eeg__zero__energy__wavelet_energy_d1_mean__EEG-E8",
                        "eeg__zero__energy__wavelet_energy_d1_std__EEG-E8",
                        "eeg__zero__energy__wavelet_energy_d1_cv__EEG-E8",
                        ],
                "wavelet_energy_d2": [
                        "eeg__zero__energy__wavelet_energy_d2_mean__EEG-E8",
                        "eeg__zero__energy__wavelet_energy_d2_std__EEG-E8",
                        "eeg__zero__energy__wavelet_energy_d2_cv__EEG-E8",
                        ],
                "wavelet_energy_d3": [
                        "eeg__zero__energy__wavelet_energy_d3_mean__EEG-E8",
                        "eeg__zero__energy__wavelet_energy_d3_std__EEG-E8",
                        "eeg__zero__energy__wavelet_energy_d3_cv__EEG-E8",
                        ],
                "wavelet_energy_d4": [
                        "eeg__zero__energy__wavelet_energy_d4_mean__EEG-E8",
                        "eeg__zero__energy__wavelet_energy_d4_std__EEG-E8",
                        "eeg__zero__energy__wavelet_energy_d4_cv__EEG-E8",
                        ],
                },

        "base": {
                "wavelet_energy_d1": [
                        "eeg__base__energy__wavelet_energy_d1_mean__EEG-E8",
                        "eeg__base__energy__wavelet_energy_d1_std__EEG-E8",
                        "eeg__base__energy__wavelet_energy_d1_cv__EEG-E8",
                        ],
                "wavelet_energy_d2": [
                        "eeg__base__energy__wavelet_energy_d2_mean__EEG-E8",
                        "eeg__base__energy__wavelet_energy_d2_std__EEG-E8",
                        "eeg__base__energy__wavelet_energy_d2_cv__EEG-E8",
                        ],
                "wavelet_energy_d3": [
                        "eeg__base__energy__wavelet_energy_d3_mean__EEG-E8",
                        "eeg__base__energy__wavelet_energy_d3_std__EEG-E8",
                        "eeg__base__energy__wavelet_energy_d3_cv__EEG-E8",
                        ],
                "wavelet_energy_d4": [
                        "eeg__base__energy__wavelet_energy_d4_mean__EEG-E8",
                        "eeg__base__energy__wavelet_energy_d4_std__EEG-E8",
                        "eeg__base__energy__wavelet_energy_d4_cv__EEG-E8",
                        ],
                },

        "tent": {
                "wavelet_energy_d1": [
                        "eeg__tent__energy__wavelet_energy_d1_mean__EEG-E8",
                        "eeg__tent__energy__wavelet_energy_d1_std__EEG-E8",
                        "eeg__tent__energy__wavelet_energy_d1_cv__EEG-E8",
                        ],
                "wavelet_energy_d2": [
                        "eeg__tent__energy__wavelet_energy_d2_mean__EEG-E8",
                        "eeg__tent__energy__wavelet_energy_d2_std__EEG-E8",
                        "eeg__tent__energy__wavelet_energy_d2_cv__EEG-E8",
                        ],
                "wavelet_energy_d3": [
                        "eeg__tent__energy__wavelet_energy_d3_mean__EEG-E8",
                        "eeg__tent__energy__wavelet_energy_d3_std__EEG-E8",
                        "eeg__tent__energy__wavelet_energy_d3_cv__EEG-E8",
                        ],
                "wavelet_energy_d4": [
                        "eeg__tent__energy__wavelet_energy_d4_mean__EEG-E8",
                        "eeg__tent__energy__wavelet_energy_d4_std__EEG-E8",
                        "eeg__tent__energy__wavelet_energy_d4_cv__EEG-E8",
                        ],
                },

        "half": {
                "wavelet_energy_d1": [
                        "eeg__half__energy__wavelet_energy_d1_mean__EEG-E8",
                        "eeg__half__energy__wavelet_energy_d1_std__EEG-E8",
                        "eeg__half__energy__wavelet_energy_d1_cv__EEG-E8",
                        ],
                "wavelet_energy_d2": [
                        "eeg__half__energy__wavelet_energy_d2_mean__EEG-E8",
                        "eeg__half__energy__wavelet_energy_d2_std__EEG-E8",
                        "eeg__half__energy__wavelet_energy_d2_cv__EEG-E8",
                        ],
                "wavelet_energy_d3": [
                        "eeg__half__energy__wavelet_energy_d3_mean__EEG-E8",
                        "eeg__half__energy__wavelet_energy_d3_std__EEG-E8",
                        "eeg__half__energy__wavelet_energy_d3_cv__EEG-E8",
                        ],
                "wavelet_energy_d4": [
                        "eeg__half__energy__wavelet_energy_d4_mean__EEG-E8",
                        "eeg__half__energy__wavelet_energy_d4_std__EEG-E8",
                        "eeg__half__energy__wavelet_energy_d4_cv__EEG-E8",
                        ],
                },

        "peak": {
                "wavelet_energy_d1": [
                        "eeg__peak__energy__wavelet_energy_d1_mean__EEG-E8",
                        "eeg__peak__energy__wavelet_energy_d1_std__EEG-E8",
                        "eeg__peak__energy__wavelet_energy_d1_cv__EEG-E8",
                        ],
                "wavelet_energy_d2": [
                        "eeg__peak__energy__wavelet_energy_d2_mean__EEG-E8",
                        "eeg__peak__energy__wavelet_energy_d2_std__EEG-E8",
                        "eeg__peak__energy__wavelet_energy_d2_cv__EEG-E8",
                        ],
                "wavelet_energy_d3": [
                        "eeg__peak__energy__wavelet_energy_d3_mean__EEG-E8",
                        "eeg__peak__energy__wavelet_energy_d3_std__EEG-E8",
                        "eeg__peak__energy__wavelet_energy_d3_cv__EEG-E8",
                        ],
                "wavelet_energy_d4": [
                        "eeg__peak__energy__wavelet_energy_d4_mean__EEG-E8",
                        "eeg__peak__energy__wavelet_energy_d4_std__EEG-E8",
                        "eeg__peak__energy__wavelet_energy_d4_cv__EEG-E8",
                        ],
                },
        }


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
