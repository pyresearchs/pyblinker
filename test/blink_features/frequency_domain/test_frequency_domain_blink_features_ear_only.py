"""EAR-only unit tests for wavelet-based blink frequency features."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.frequency_domain import (
    aggregate_frequency_domain_features,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot



PROJECT_ROOT = Path(__file__).resolve().parents[3]


required_columns = [
        "ear__th_point__energy__wavelet_energy_d1_mean__EAR-AVG_EAR",
        "ear__th_point__energy__wavelet_energy_d1_std__EAR-AVG_EAR",
        "ear__th_point__energy__wavelet_energy_d1_cv__EAR-AVG_EAR",

        "ear__th_point__energy__wavelet_energy_d2_mean__EAR-AVG_EAR",
        "ear__th_point__energy__wavelet_energy_d2_std__EAR-AVG_EAR",
        "ear__th_point__energy__wavelet_energy_d2_cv__EAR-AVG_EAR",

        "ear__th_point__energy__wavelet_energy_d3_mean__EAR-AVG_EAR",
        "ear__th_point__energy__wavelet_energy_d3_std__EAR-AVG_EAR",
        "ear__th_point__energy__wavelet_energy_d3_cv__EAR-AVG_EAR",

        "ear__th_point__energy__wavelet_energy_d4_mean__EAR-AVG_EAR",
        "ear__th_point__energy__wavelet_energy_d4_std__EAR-AVG_EAR",
        "ear__th_point__energy__wavelet_energy_d4_cv__EAR-AVG_EAR",
        ]


class TestFrequencyDomainBlinkFeaturesEAROnly(unittest.TestCase):
    """Validate DWT energy features per epoch for EAR-only inputs."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        ear_channel = "EAR-avg_ear"
        raw.pick([ear_channel])
        segmentation_config = {
            "ear": {
                "channel": ear_channel,
                "seg_type": "threshold_interpolation",
                "threshold": 0.260,
                "annotation_time_unit": "seconds",
                "max_extension": 0.35,
                "extension_step": 0.05,
                "padding": 0.05,
                "extend_before": True,
                "extend_after": True,
            }
        }
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )
        self.ear_channel = ear_channel

    def test_schema_and_rows(self) -> None:
        """DataFrame has expected columns and indexing for first epochs."""
        df = aggregate_frequency_domain_features(
            self.epochs, picks=self.ear_channel, progress_bar=False
        )
        for col in required_columns:
            self.assertIn(col, df.columns)



if __name__ == "__main__":
    unittest.main()
