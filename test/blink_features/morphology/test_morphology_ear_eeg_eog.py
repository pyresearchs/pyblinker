"""Full-modality morphology coverage (EAR + EEG + EOG)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from test.segment_config import build_segment_config
from test.helper import build_expected_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "test" / "major_structure_refactor"
BASELINE_PATH = OUTPUT_DIR / "morphology_features_ear_eeg_eog.pkl"
UPDATE_ENV_VAR = "UPDATE_MORPHOLOGY_BASELINE"

EAR_CHANNEL = "EAR-avg_ear"
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"

stats = ["mean", "std", "cv"]

EAR_METRICS = [
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

REQUIRED_EAR_METRICS = build_expected_metrics(
    landmark=["th_point"],
    metrics=EAR_METRICS,
    stats=stats,
    modality="ear",
    feature="morphology",
    channel=EAR_CHANNEL,
)
REQUIRED_EEG_METRICS = build_expected_metrics(
    landmark=list(MORPHOLOGY_METRICS_BY_LANDMARK.keys()),
    metrics=MORPHOLOGY_METRICS_BY_LANDMARK,
    stats=stats,
    modality="eeg",
    feature="morphology",
    channel=EEG_CHANNEL,
)
REQUIRED_EOG_METRICS = build_expected_metrics(
    landmark=list(MORPHOLOGY_METRICS_BY_LANDMARK.keys()),
    metrics=MORPHOLOGY_METRICS_BY_LANDMARK,
    stats=stats,
    modality="eog",
    feature="morphology",
    channel=EOG_CHANNEL,
)

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
    "eeg": {"channel": EEG_CHANNEL},
    "eog": {"channel": EOG_CHANNEL},
}


class TestFullModalityMorphologyPipeline(unittest.TestCase):
    """EAR+EEG+EOG morphology pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(cls.raw_path, preload=True, verbose=False)
        segment_config = build_segment_config(raw)
        segment_config["ear"] = SEGMENT_CONFIG["ear"]

        cls.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
        )

        cls.df = compute_epoch_morphology_features(
            epochs=cls.epochs, picks=[EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL]
        )

    def _load_baseline(self) -> pd.DataFrame:
        if not BASELINE_PATH.exists():
            raise AssertionError(
                "Missing baseline pickle. Set UPDATE_MORPHOLOGY_BASELINE=1 and rerun the test to generate it."
            )
        return pd.read_pickle(BASELINE_PATH)

    def _maybe_write_baseline(self) -> None:
        if os.environ.get(UPDATE_ENV_VAR) == "1":
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            self.df.to_pickle(BASELINE_PATH)

    def test_ear_columns(self) -> None:
        for style in REQUIRED_EAR_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_eeg_columns(self) -> None:
        for style in REQUIRED_EEG_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_eog_columns(self) -> None:
        for style in REQUIRED_EOG_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_compare_with_excel_input(self) -> None:
        """Compare legacy-mean morphology outputs against Excel fixture values."""
        fixture_path = (
            PROJECT_ROOT
            / "test"
            / "blink_features"
            / "morphology"
            / "expected_output_new_naming.xlsx"
        )
        expected_df = pd.read_excel(fixture_path)
        expected_df = expected_df.drop(columns=["Unnamed: 0"], errors="ignore")

        self.assertEqual(len(self.df), len(expected_df))

        column_overrides = {
            "eeg__peak__morphology__peak_time_tent_mean__EEG-E8": "eeg__peak__morphology__peak_max_tent_mean__EEG-E8",
            "eeg__peak__morphology__peak_time_tent_mean__EEG-E8.1": "eeg__peak__morphology__peak_time_tent_mean__EEG-E8",
        }

        for expected_col in expected_df.columns:
            output_col = column_overrides.get(expected_col, expected_col.split(".", 1)[0])
            self.assertIn(output_col, self.df.columns)

            expected_values = expected_df[expected_col].to_numpy(dtype=float)
            actual_values = self.df[output_col].to_numpy(dtype=float)
            self.assertTrue(
                np.allclose(actual_values, expected_values, atol=1e-8, equal_nan=True),
                msg=f"Column mismatch: fixture={expected_col} output={output_col}",
            )

    def test_matches_baseline_pickle(self) -> None:
        self._maybe_write_baseline()
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
