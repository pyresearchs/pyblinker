"""Scenario A: EAR-only kinematic pipeline coverage."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from test.helper import build_expected_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]


stats = ["mean", "std", "cv"]
metrics = [
    "amp_peak_abs_base",
    "amp_peak_signed_base",
    "amp_peak_to_trough_base",
    "amp_trough_signed_base",
    "area_abs_total_rect_base",
    "area_abs_total_trapz_base",
    "duration",
    "fall_time_10_90_base",
    "fall_time_peak_base",
    "half_width_base",
    "rise_time_10_90_base",
    "rise_time_peak_base",
    "symmetry_rect_base",
    "symmetry_trapz_base",
]

EAR_CHANNEL = "EAR-avg_ear"
modality = "ear"
landmark = ["th_point"]
feature = "morphology"


REQUIRED_MORPHOLOGY_METRICS = build_expected_metrics(
    landmark=landmark,
    metrics=metrics,
    stats=stats,
    modality=modality,
    feature=feature,
    channel=EAR_CHANNEL,
)


class TestEarOnlyKinematicPipeline(unittest.TestCase):
    """Tests for EAR-only kinematic pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"

    def test_ear_only_runs_with_single_modality_config(self) -> None:
        """EAR-only config produces EAR-only outputs without validating EEG."""

        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)

        segment_config = {
            "ear": {
                "channel": EAR_CHANNEL,
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

        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
        )

        df = compute_epoch_morphology_features(epochs, picks=[EAR_CHANNEL])
        for style in REQUIRED_MORPHOLOGY_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)


if __name__ == "__main__":
    unittest.main()
