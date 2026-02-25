"""Scenario A: EAR-only kinematic pipeline coverage."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.kinematics.kinematic_features import (
    KinematicBlinkFeatureExtractor,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.helper import build_expected_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EAR_CHANNEL = "EAR-avg_ear"


SHARED_KINEMATIC_METRICS = (
    "amp_vel_ratio_base",
    "amp_vel_ratio_tent",
    "amp_vel_ratio_zero_to_max",
    "aver_left_velocity",
    "aver_right_velocity",
    "blink_velocity",
    "inter_blink_max_vel",
    "inter_blink_max_vel_base",
    "inter_blink_max_vel_zero",
    "neg_amp_vel_ratio_base",
    "neg_amp_vel_ratio_tent",
    "neg_amp_vel_ratio_zero",
    "pos_amp_vel_ratio_base",
    "pos_amp_vel_ratio_tent",
    "pos_amp_vel_ratio_zero",
)
BASE_STYLE_SUFFIXED_METRICS = (
    "acc_mean_abs_base",
    "acc_peak_abs_base",
    "slope_fall_neg_base",
    "slope_rise_pos_base",
    "vel_mean_abs_base",
    "vel_peak_abs_base",
)

metrics_by_landmark = {
    landmark: list(SHARED_KINEMATIC_METRICS) + list(BASE_STYLE_SUFFIXED_METRICS)
    for landmark in ("th_interpolation", "th_point", " interpolated_threshold")
}
stats = ["mean", "std", "cv"]
REQUIRED_KINEMATICS_METRICS = build_expected_metrics(
    landmark=list(metrics_by_landmark.keys()),
    metrics=metrics_by_landmark,
    stats=stats,
    modality="ear",
    feature="kinematic",
    channel=EAR_CHANNEL,
)

# Backward-compatibility alias includes an extra separator before channel token.
REQUIRED_KINEMATICS_METRICS[" interpolated_threshold"] = {
    metric: [name.replace("__EAR-avg_ear", "____EAR-avg_ear") for name in names]
    for metric, names in REQUIRED_KINEMATICS_METRICS[" interpolated_threshold"].items()
}



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

        extractor = KinematicBlinkFeatureExtractor(epochs=epochs)
        df = extractor.compute(picks=EAR_CHANNEL)

        for style in REQUIRED_KINEMATICS_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)


if __name__ == "__main__":
    unittest.main()
