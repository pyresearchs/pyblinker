"""Full-modality morphology coverage (EAR + EEG + EOG)."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.blink_features.kinematics.kinematic_features import (
    KinematicBlinkFeatureExtractor,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.blink_features.morphology.epoch_features import _available_styles, _style_windows
from pyblinker.blink_features.morphology import compute_epoch_morphology_features
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EAR_CHANNEL = "EAR-avg_ear"
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"
required_columns=['ear__th_point__morphology__amp_peak_abs_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_abs_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_abs_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_signed_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_signed_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_signed_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_to_trough_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_to_trough_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_to_trough_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__amp_trough_signed_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__amp_trough_signed_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__amp_trough_signed_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_rect_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_rect_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_rect_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_trapz_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_trapz_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_trapz_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__duration_cv__EAR-AVG_EAR', 'ear__th_point__morphology__duration_mean__EAR-AVG_EAR', 'ear__th_point__morphology__duration_std__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_10_90_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_10_90_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_10_90_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_peak_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_peak_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_peak_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__half_width_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__half_width_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__half_width_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_10_90_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_10_90_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_10_90_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_peak_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_peak_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_peak_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_rect_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_rect_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_rect_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_trapz_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_trapz_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_trapz_base_std__EAR-AVG_EAR', ]

_REQUIRED_LEGACY_MORPHOLOGY_METRICS = {
        "zero": (
                "duration_zero",
                "closing_time_zero",
                "reopening_time_zero",
                "time_shut_zero",
                ),
        "base": (
                "duration_base",
                "time_shut_base",
                ),
        "tent": (
                "duration_tent",
                "closing_time_tent",
                "reopening_time_tent",
                "time_shut_tent",
                ),
        "half": (
                "duration_half_base",
                "duration_half_zero",
                ),
        "peak": (
                "peak_time_blink",
                "peak_time_tent",
                "peak_max_blink",
                "peak_max_tent",
                ),
        "inter_blink": (
                "inter_blink_max_amp",
                ),
        }
class TestFullModalityKinematicPipeline(unittest.TestCase):
    """EAR+EEG+EOG kinematic pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        # CSV containing the full list of expected kinematic columns
        cls.expected_columns_path = (
            PROJECT_ROOT
            / "test"
            / "blink_features"
            / "kinematics"
            / "column_full_modality_kinematics_ear_eeg_eog.csv"
        )

        # Read expected column names; each line is a single column name
        with cls.expected_columns_path.open("r", encoding="utf-8") as f:
            cls.expected_columns = [
                line.strip() for line in f.readlines() if line.strip()
            ]
            raw = mne.io.read_raw_fif(cls.raw_path, preload=True, verbose=False)

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

        cls.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
            )

        # extractor = KinematicBlinkFeatureExtractor(epochs=cls.epochs)
        cls.df = compute_epoch_morphology_features(epochs=cls.epochs,picks=[EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL])

    def test_eeg(self) -> None:

        styles = _available_styles(tuple(self.epochs.metadata.columns), "eeg")
        self.assertTrue(styles)

        for style in styles:
            expected = f"eeg__{style}__morphology__duration_mean__{EEG_CHANNEL}"
            self.assertIn(expected, self.df.columns)

        for metrics in _REQUIRED_LEGACY_MORPHOLOGY_METRICS.values():
            for metric in metrics:
                self.assertIn(metric, self.df.columns)

    def test_eog(self) -> None:
        styles = _available_styles(tuple(self.epochs.metadata.columns), "eog")
        self.assertTrue(styles)

        for style in styles:
            expected = f"eog__{style}__morphology__amp_vel_ratio_base_mean__{EOG_CHANNEL}"
            self.assertIn(expected, self.df.columns)

    def test_ear(self) -> None:
        for col in required_columns:
            self.assertIn(col, self.df.columns)

if __name__ == "__main__":
    unittest.main()
