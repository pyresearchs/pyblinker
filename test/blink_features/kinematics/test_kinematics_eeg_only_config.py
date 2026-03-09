"""Scenario B: EEG-only kinematic pipeline coverage."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.kinematics.kinematic_features import (
    KinematicBlinkFeatureExtractor,
)
from test.helper import build_expected_metrics
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

metrics_by_landmark = {
    style: list(SHARED_KINEMATIC_METRICS)
    + [f"{prefix}_{style}" for prefix in STYLE_SUFFIXED_PREFIXES]
    for style in ("base", "tent", "zero")
}
EEG_CHANNEL = "EEG-E8"
stats = ["mean", "std", "cv"]
REQUIRED_KINEMATICS_METRICS = build_expected_metrics(
    landmark=list(metrics_by_landmark.keys()),
    metrics=metrics_by_landmark,
    stats=stats,
    modality="eeg",
    feature="kinematic",
    channel=EEG_CHANNEL,
)


class TestEegOnlyKinematicPipeline(unittest.TestCase):
    """EEG-only kinematic pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"

    def test_eeg_only_runs_without_ear_channel(self) -> None:
        """EEG-only config (with optional EOG) runs and yields EEG outputs."""

        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)

        segment_config = {
            "eeg": {"channel": EEG_CHANNEL, "seg_type": "base"},
        }

        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
        )

        extractor = KinematicBlinkFeatureExtractor(epochs=epochs)
        df = extractor.compute(picks=EEG_CHANNEL)
        for style in REQUIRED_KINEMATICS_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)

        # self.assertGreater(df.notna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
