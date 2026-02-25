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
EEG_CHANNEL = "EEG-E8"
# EOG_CHANNEL = "EOG-EEG-eog_vert_left"

# _REQUIRED_KINEMATIC_METRICS = (
#     "amp_vel_ratio_base",
#     "amp_vel_ratio_tent",
#     "amp_vel_ratio_zero_to_max",
#     "blink_velocity",
#     "inter_blink_max_vel",
#     "inter_blink_max_vel_base",
#     "inter_blink_max_vel_zero",
#     "aver_left_velocity",
#     "aver_right_velocity",
#     "neg_amp_vel_ratio_base",
#     "pos_amp_vel_ratio_base",
#     "neg_amp_vel_ratio_zero",
#     "pos_amp_vel_ratio_zero",
#     "neg_amp_vel_ratio_tent",
#     "pos_amp_vel_ratio_tent",
# )

metrics = [
        # shared across styles
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
        # style-specific (base)
        "acc_mean_abs_base",
        "acc_peak_abs_base",
        "slope_fall_neg_base",
        "slope_rise_pos_base",
        "vel_mean_abs_base",
        "vel_peak_abs_base",
        # style-specific (tent)
        "acc_mean_abs_tent",
        "acc_peak_abs_tent",
        "slope_fall_neg_tent",
        "slope_rise_pos_tent",
        "vel_mean_abs_tent",
        "vel_peak_abs_tent",
        # style-specific (zero)
        "acc_mean_abs_zero",
        "acc_peak_abs_zero",
        "slope_fall_neg_zero",
        "slope_rise_pos_zero",
        "vel_mean_abs_zero",
        "vel_peak_abs_zero",
        ]
stats = ["mean", "std", "cv"]
landmarks = ["zero", "base", "tent", "half", "peak"]
REQUIRED_KINEMATICS_METRICS = build_expected_metrics(
    landmark=landmarks,
    metrics=metrics,
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

        # self.assertNotIn("blink_onset_ear", epochs.metadata.columns)
        # self.assertIn("blink_onset_eeg", epochs.metadata.columns)
        # self.assertTrue(all(col.endswith(f"__{EEG_CHANNEL}") for col in df.columns))

        # styles = _available_styles(tuple(epochs.metadata.columns), "eeg")
        # self.assertTrue(styles)
        for style in REQUIRED_KINEMATICS_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)

        # self.assertGreater(df.notna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
