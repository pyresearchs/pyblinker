"""Combined EAR/EEG/EOG blink feature coverage (energy, frequency, kinematics, morphology)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.blink_features.frequency_domain import aggregate_frequency_domain_features
from pyblinker.blink_features.kinematics.kinematic_features import (
    KinematicBlinkFeatureExtractor,
)
from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.helper import build_expected_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "test" / "major_structure_refactor"
BASELINE_PATH = OUTPUT_DIR / "blink_features_ear_eeg_eog.pkl"
UPDATE_ENV_VAR = "UPDATE_BLINK_FEATURES_BASELINE"

EAR_CHANNEL = "EAR-avg_ear"
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"

STATS = ["mean", "std", "cv"]

ENERGY_METRICS = [
    "blink_signal_energy",
    "teager_kaiser_energy",
    "blink_line_length",
    "blink_velocity_integral",
]
ENERGY_LANDMARKS = ["zero", "base", "tent", "half", "peak"]

FREQ_METRICS = ["wavelet_energy_d1", "wavelet_energy_d2", "wavelet_energy_d3", "wavelet_energy_d4"]
FREQ_LANDMARKS = ["zero", "base", "tent", "half", "peak"]

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

KIN_EAR_METRICS_BY_LANDMARK = {
    landmark: list(SHARED_KINEMATIC_METRICS) + list(BASE_STYLE_SUFFIXED_METRICS)
    for landmark in ("th_interpolation", "th_point")
}
KIN_EEG_EOG_METRICS_BY_LANDMARK = {
    style: list(SHARED_KINEMATIC_METRICS)
    + [f"{prefix}_{style}" for prefix in STYLE_SUFFIXED_PREFIXES]
    for style in ("base", "tent", "zero")
}

MORPHOLOGY_EAR_METRICS = [
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
MORPHOLOGY_METRICS_BY_LANDMARK = {
    "base": ["duration_base", "time_shut_base"],
    "half": ["duration_half_base", "duration_half_zero"],
    "inter_blink": ["inter_blink_max_amp"],
    "peak": ["peak_time_blink", "peak_time_tent", "peak_max_blink", "peak_max_tent"],
    "tent": ["duration_tent", "closing_time_tent", "reopening_time_tent", "time_shut_tent"],
    "zero": ["duration_zero", "closing_time_zero", "reopening_time_zero", "time_shut_zero"],
}

SEGMENT_CONFIG = {
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


class TestBlinkFeaturesAllModalities(unittest.TestCase):
    """Combined EAR/EEG/EOG blink feature coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

        cls.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=SEGMENT_CONFIG,
        )

        cls.energy_df = compute_energy_features(
            epochs=cls.epochs, picks=[EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL]
        )
        cls.freq_df = aggregate_frequency_domain_features(
            cls.epochs, picks=[EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL], progress_bar=False
        )
        extractor = KinematicBlinkFeatureExtractor(epochs=cls.epochs)
        cls.kin_df = extractor.compute(picks=[EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL])
        cls.morph_df = compute_epoch_morphology_features(
            epochs=cls.epochs, picks=[EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL]
        )
        cls.df = pd.concat(
            [cls.energy_df, cls.freq_df, cls.kin_df, cls.morph_df],
            axis=1,
        )

    def _load_baseline(self) -> pd.DataFrame:
        if not BASELINE_PATH.exists():
            raise AssertionError(
                "Missing baseline pickle. Set UPDATE_BLINK_FEATURES_BASELINE=1 and rerun the test to generate it."
            )
        return pd.read_pickle(BASELINE_PATH)

    def _maybe_write_baseline(self) -> None:
        if os.environ.get(UPDATE_ENV_VAR) == "1":
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            self.df.to_pickle(BASELINE_PATH)

    def test_energy_columns(self) -> None:
        ear_expected = build_expected_metrics(
            landmark=["th_point"],
            metrics=["blink_signal_energy"],
            stats=STATS,
            modality="ear",
            feature="energy",
            channel="EAR-AVG_EAR",
        )
        eeg_expected = build_expected_metrics(
            landmark=ENERGY_LANDMARKS,
            metrics=ENERGY_METRICS,
            stats=STATS,
            modality="eeg",
            feature="energy",
            channel=EEG_CHANNEL,
        )
        eog_expected = build_expected_metrics(
            landmark=ENERGY_LANDMARKS,
            metrics=ENERGY_METRICS,
            stats=STATS,
            modality="eog",
            feature="energy",
            channel=EOG_CHANNEL,
        )

        for style in ear_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

        for style in eeg_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

        for style in eog_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_frequency_columns(self) -> None:
        ear_expected = build_expected_metrics(
            landmark=["th_point"],
            metrics=FREQ_METRICS,
            stats=STATS,
            modality="ear",
            feature="energy",
            channel="EAR-AVG_EAR",
        )
        eeg_expected = build_expected_metrics(
            landmark=FREQ_LANDMARKS,
            metrics=FREQ_METRICS,
            stats=STATS,
            modality="eeg",
            feature="energy",
            channel=EEG_CHANNEL,
        )
        eog_expected = build_expected_metrics(
            landmark=FREQ_LANDMARKS,
            metrics=FREQ_METRICS,
            stats=STATS,
            modality="eog",
            feature="energy",
            channel=EOG_CHANNEL,
        )

        for style in ear_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

        for style in eeg_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

        for style in eog_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_kinematics_columns(self) -> None:
        ear_expected = build_expected_metrics(
            landmark=list(KIN_EAR_METRICS_BY_LANDMARK.keys()),
            metrics=KIN_EAR_METRICS_BY_LANDMARK,
            stats=STATS,
            modality="ear",
            feature="kinematic",
            channel=EAR_CHANNEL,
        )
        eeg_expected = build_expected_metrics(
            landmark=list(KIN_EEG_EOG_METRICS_BY_LANDMARK.keys()),
            metrics=KIN_EEG_EOG_METRICS_BY_LANDMARK,
            stats=STATS,
            modality="eeg",
            feature="kinematic",
            channel=EEG_CHANNEL,
        )
        eog_expected = build_expected_metrics(
            landmark=list(KIN_EEG_EOG_METRICS_BY_LANDMARK.keys()),
            metrics=KIN_EEG_EOG_METRICS_BY_LANDMARK,
            stats=STATS,
            modality="eog",
            feature="kinematic",
            channel=EOG_CHANNEL,
        )

        for style in ear_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

        for style in eeg_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

        for style in eog_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_morphology_columns(self) -> None:
        ear_expected = build_expected_metrics(
            landmark=["th_point"],
            metrics=MORPHOLOGY_EAR_METRICS,
            stats=STATS,
            modality="ear",
            feature="morphology",
            channel=EAR_CHANNEL,
        )
        eeg_expected = build_expected_metrics(
            landmark=list(MORPHOLOGY_METRICS_BY_LANDMARK.keys()),
            metrics=MORPHOLOGY_METRICS_BY_LANDMARK,
            stats=STATS,
            modality="eeg",
            feature="morphology",
            channel=EEG_CHANNEL,
        )
        eog_expected = build_expected_metrics(
            landmark=list(MORPHOLOGY_METRICS_BY_LANDMARK.keys()),
            metrics=MORPHOLOGY_METRICS_BY_LANDMARK,
            stats=STATS,
            modality="eog",
            feature="morphology",
            channel=EOG_CHANNEL,
        )

        for style in ear_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

        for style in eeg_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

        for style in eog_expected.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    # def test_compare_morphology_with_excel_fixture(self) -> None:
    #     try:
    #         import importlib.util
	#
    #         if importlib.util.find_spec("openpyxl") is None:
    #             self.skipTest("openpyxl is not installed")
    #     except Exception:
    #         self.skipTest("openpyxl is not installed")
	#
    #     fixture_path = (
    #         PROJECT_ROOT
    #         / "test"
    #         / "blink_features"
    #         / "morphology"
    #         / "expected_output_new_naming.xlsx"
    #     )
    #     expected_df = pd.read_excel(fixture_path)
    #     expected_df = expected_df.drop(columns=["Unnamed: 0"], errors="ignore")
	#
    #     self.assertEqual(len(self.morph_df), len(expected_df))
	#
    #     column_overrides = {
    #         "eeg__peak__morphology__peak_time_tent_mean__EEG-E8": "eeg__peak__morphology__peak_max_tent_mean__EEG-E8",
    #         "eeg__peak__morphology__peak_time_tent_mean__EEG-E8.1": "eeg__peak__morphology__peak_time_tent_mean__EEG-E8",
    #     }
	#
    #     for expected_col in expected_df.columns:
    #         output_col = column_overrides.get(expected_col, expected_col.split(".", 1)[0])
    #         self.assertIn(output_col, self.morph_df.columns)
	#
    #         expected_values = expected_df[expected_col].to_numpy(dtype=float)
    #         actual_values = self.morph_df[output_col].to_numpy(dtype=float)
    #         self.assertTrue(
    #             np.allclose(actual_values, expected_values, atol=1e-8, equal_nan=True),
    #             msg=f"Column mismatch: fixture={expected_col} output={output_col}",
    #         )

    def test_matches_baseline_pickle(self) -> None:
        # self._maybe_write_baseline()
        # self.df.to_pickle(BASELINE_PATH)
        baseline = self._load_baseline()

        pd.testing.assert_frame_equal(
            self.df.sort_index(axis=1),
            baseline.sort_index(axis=1),
            check_dtype=False,
            rtol=1e-6,
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()

