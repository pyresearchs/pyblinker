"""Full-modality kinematics coverage (EAR + EEG + EOG)."""

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
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"

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
STYLE_SUFFIXED_PREFIXES = (
    "acc_mean_abs",
    "acc_peak_abs",
    "slope_fall_neg",
    "slope_rise_pos",
    "vel_mean_abs",
    "vel_peak_abs",
)
BASE_STYLE_SUFFIXED_METRICS = (
    "acc_mean_abs_base",
    "acc_peak_abs_base",
    "slope_fall_neg_base",
    "slope_rise_pos_base",
    "vel_mean_abs_base",
    "vel_peak_abs_base",
)

stats = ["mean", "std", "cv"]

EAR_METRICS_BY_LANDMARK = {
    landmark: list(SHARED_KINEMATIC_METRICS) + list(BASE_STYLE_SUFFIXED_METRICS)
    for landmark in ("th_interpolation", "th_point")
}
EEG_EOG_METRICS_BY_LANDMARK = {
    style: list(SHARED_KINEMATIC_METRICS)
    + [f"{prefix}_{style}" for prefix in STYLE_SUFFIXED_PREFIXES]
    for style in ("base", "tent", "zero")
}

REQUIRED_EAR_METRICS = build_expected_metrics(
    landmark=list(EAR_METRICS_BY_LANDMARK.keys()),
    metrics=EAR_METRICS_BY_LANDMARK,
    stats=stats,
    modality="ear",
    feature="kinematic",
    channel=EAR_CHANNEL,
)
REQUIRED_EEG_METRICS = build_expected_metrics(
    landmark=list(EEG_EOG_METRICS_BY_LANDMARK.keys()),
    metrics=EEG_EOG_METRICS_BY_LANDMARK,
    stats=stats,
    modality="eeg",
    feature="kinematic",
    channel=EEG_CHANNEL,
)
REQUIRED_EOG_METRICS = build_expected_metrics(
    landmark=list(EEG_EOG_METRICS_BY_LANDMARK.keys()),
    metrics=EEG_EOG_METRICS_BY_LANDMARK,
    stats=stats,
    modality="eog",
    feature="kinematic",
    channel=EOG_CHANNEL,
)


class TestFullModalityKinematicPipeline(unittest.TestCase):
    """EAR+EEG+EOG kinematic pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"

    def test_full_modality_config_produces_all_channels(self) -> None:
        """EAR+EEG+EOG config yields kinematics for each configured channel."""
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
            },
            "eeg": {"channel": EEG_CHANNEL, "seg_type": "base"},
            "eog": {"channel": EOG_CHANNEL, "seg_type": "base"},
        }

        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
        )

        extractor = KinematicBlinkFeatureExtractor(epochs=epochs)
        df = extractor.compute(picks=[EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL])

        for style in REQUIRED_EAR_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)

        for style in REQUIRED_EEG_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)

        for style in REQUIRED_EOG_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)


if __name__ == "__main__":
    unittest.main()
