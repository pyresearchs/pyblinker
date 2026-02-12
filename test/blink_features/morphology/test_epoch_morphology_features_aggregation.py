"""Integration coverage for epoch morphology aggregation columns."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from pyblinker.blink_features.morphology.epoch_features import _available_styles, _style_windows
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EEG_CHANNEL = "EEG-E8"

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


class TestMorphologyAggregation(unittest.TestCase):
    """Test aggregation of morphology features with blink counts."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        segmentation_config = build_segment_config(raw)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )


    def test_style_windows_uses_landmark_frames_when_available(self) -> None:
        """Window extraction should use landmark frame boundaries before onset/duration."""
        metadata_row = {
            "start__left_zero__eeg": [10, 30],
            "end__right_zero__eeg": [16, 40],
            "onset__zero__eeg": [1.0, 2.0],
            "duration__zero__eeg": [0.0, 0.0],
        }

        windows = _style_windows(metadata_row, "eeg", "zero")

        self.assertEqual(windows, [(10.0, 6.0), (30.0, 10.0)])

    def test_epoch_output_contains_expected_morphology_features(self) -> None:
        """Epoch output includes expected style-aware and legacy morphology fields."""
        df = compute_epoch_morphology_features(self.epochs, picks=[EEG_CHANNEL])
        styles = _available_styles(tuple(self.epochs.metadata.columns), "eeg")
        self.assertTrue(styles)

        for style in styles:
            expected = f"eeg__{style}__morphology__duration_mean__{EEG_CHANNEL}"
            self.assertIn(expected, df.columns)

        for metrics in _REQUIRED_LEGACY_MORPHOLOGY_METRICS.values():
            for metric in metrics:
                self.assertIn(metric, df.columns)

        self.assertGreater(df.notna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
