"""EAR-only unit tests for wavelet-based blink frequency features."""

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.frequency_domain import (
    aggregate_frequency_domain_features,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.helper import build_expected_metrics



PROJECT_ROOT = Path(__file__).resolve().parents[3]


stats = ["mean", "std", "cv"]
metrics = ["wavelet_energy_d1", "wavelet_energy_d2", "wavelet_energy_d3", "wavelet_energy_d4"]

modality = "ear"
landmark = ["th_point"]
feature = "energy"
channel = "EAR-AVG_EAR"

EAR_ENERGY_METRICS = build_expected_metrics(
    landmark=landmark,
    metrics=metrics,
    stats=stats,
    modality=modality,
    feature=feature,
    channel=channel,
)


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
        for landmark_metrics in EAR_ENERGY_METRICS.values():
            for metric in landmark_metrics.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)



if __name__ == "__main__":
    unittest.main()
