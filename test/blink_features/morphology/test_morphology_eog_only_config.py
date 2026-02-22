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
EOG_CHANNEL = "EOG-EEG-eog_vert_left"


_REQUIRED_LEGACY_MORPHOLOGY_METRICS = {
        "zero": {
                "duration_zero": [
                        "eog__zero__morphology__duration_zero_mean__EOG-EEG-eog_vert_left",
                        "eog__zero__morphology__duration_zero_std__EOG-EEG-eog_vert_left",
                        "eog__zero__morphology__duration_zero_cv__EOG-EEG-eog_vert_left",
                        ],
                "closing_time_zero": [
                        "eog__zero__morphology__closing_time_zero_mean__EOG-EEG-eog_vert_left",
                        "eog__zero__morphology__closing_time_zero_std__EOG-EEG-eog_vert_left",
                        "eog__zero__morphology__closing_time_zero_cv__EOG-EEG-eog_vert_left",
                        ],
                "reopening_time_zero": [
                        "eog__zero__morphology__reopening_time_zero_mean__EOG-EEG-eog_vert_left",
                        "eog__zero__morphology__reopening_time_zero_std__EOG-EEG-eog_vert_left",
                        "eog__zero__morphology__reopening_time_zero_cv__EOG-EEG-eog_vert_left",
                        ],
                "time_shut_zero": [
                        "eog__zero__morphology__time_shut_zero_mean__EOG-EEG-eog_vert_left",
                        "eog__zero__morphology__time_shut_zero_std__EOG-EEG-eog_vert_left",
                        "eog__zero__morphology__time_shut_zero_cv__EOG-EEG-eog_vert_left",
                        ],
                },
        "base": {
                "duration_base": [
                        "eog__base__morphology__duration_base_mean__EOG-EEG-eog_vert_left",
                        "eog__base__morphology__duration_base_std__EOG-EEG-eog_vert_left",
                        "eog__base__morphology__duration_base_cv__EOG-EEG-eog_vert_left",
                        ],
                "time_shut_base": [
                        "eog__base__morphology__time_shut_base_mean__EOG-EEG-eog_vert_left",
                        "eog__base__morphology__time_shut_base_std__EOG-EEG-eog_vert_left",
                        "eog__base__morphology__time_shut_base_cv__EOG-EEG-eog_vert_left",
                        ],
                },
        "tent": {
                "duration_tent": [
                        "eog__tent__morphology__duration_tent_mean__EOG-EEG-eog_vert_left",
                        "eog__tent__morphology__duration_tent_std__EOG-EEG-eog_vert_left",
                        "eog__tent__morphology__duration_tent_cv__EOG-EEG-eog_vert_left",
                        ],
                "closing_time_tent": [
                        "eog__tent__morphology__closing_time_tent_mean__EOG-EEG-eog_vert_left",
                        "eog__tent__morphology__closing_time_tent_std__EOG-EEG-eog_vert_left",
                        "eog__tent__morphology__closing_time_tent_cv__EOG-EEG-eog_vert_left",
                        ],
                "reopening_time_tent": [
                        "eog__tent__morphology__reopening_time_tent_mean__EOG-EEG-eog_vert_left",
                        "eog__tent__morphology__reopening_time_tent_std__EOG-EEG-eog_vert_left",
                        "eog__tent__morphology__reopening_time_tent_cv__EOG-EEG-eog_vert_left",
                        ],
                "time_shut_tent": [
                        "eog__tent__morphology__time_shut_tent_mean__EOG-EEG-eog_vert_left",
                        "eog__tent__morphology__time_shut_tent_std__EOG-EEG-eog_vert_left",
                        "eog__tent__morphology__time_shut_tent_cv__EOG-EEG-eog_vert_left",
                        ],
                },
        "half": {
                "duration_half_base": [
                        "eog__half__morphology__duration_half_base_mean__EOG-EEG-eog_vert_left",
                        "eog__half__morphology__duration_half_base_std__EOG-EEG-eog_vert_left",
                        "eog__half__morphology__duration_half_base_cv__EOG-EEG-eog_vert_left",
                        ],
                "duration_half_zero": [
                        "eog__half__morphology__duration_half_zero_mean__EOG-EEG-eog_vert_left",
                        "eog__half__morphology__duration_half_zero_std__EOG-EEG-eog_vert_left",
                        "eog__half__morphology__duration_half_zero_cv__EOG-EEG-eog_vert_left",
                        ],
                },
        "peak": {
                "peak_time_blink": [
                        "eog__peak__morphology__peak_time_blink_mean__EOG-EEG-eog_vert_left",
                        "eog__peak__morphology__peak_time_blink_std__EOG-EEG-eog_vert_left",
                        "eog__peak__morphology__peak_time_blink_cv__EOG-EEG-eog_vert_left",
                        ],
                "peak_time_tent": [
                        "eog__peak__morphology__peak_time_tent_mean__EOG-EEG-eog_vert_left",
                        "eog__peak__morphology__peak_time_tent_std__EOG-EEG-eog_vert_left",
                        "eog__peak__morphology__peak_time_tent_cv__EOG-EEG-eog_vert_left",
                        ],
                "peak_max_blink": [
                        "eog__peak__morphology__peak_max_blink_mean__EOG-EEG-eog_vert_left",
                        "eog__peak__morphology__peak_max_blink_std__EOG-EEG-eog_vert_left",
                        "eog__peak__morphology__peak_max_blink_cv__EOG-EEG-eog_vert_left",
                        ],
                "peak_max_tent": [
                        "eog__peak__morphology__peak_max_tent_mean__EOG-EEG-eog_vert_left",
                        "eog__peak__morphology__peak_max_tent_std__EOG-EEG-eog_vert_left",
                        "eog__peak__morphology__peak_max_tent_cv__EOG-EEG-eog_vert_left",
                        ],
                },
        "inter_blink": {
                "inter_blink_max_amp": [
                        "eog__inter_blink__morphology__inter_blink_max_amp_mean__EOG-EEG-eog_vert_left",
                        "eog__inter_blink__morphology__inter_blink_max_amp_std__EOG-EEG-eog_vert_left",
                        "eog__inter_blink__morphology__inter_blink_max_amp_cv__EOG-EEG-eog_vert_left",
                        ],
                },
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



    def test_epoch_output_contains_expected_morphology_features(self) -> None:
        """Epoch output includes expected style-aware and legacy morphology fields."""
        df = compute_epoch_morphology_features(self.epochs, picks=[EOG_CHANNEL])
        styles = _available_styles(tuple(self.epochs.metadata.columns), "eeg")
        self.assertTrue(styles)

        for style in styles:
            expected = f"eog__{style}__morphology__duration_mean__{EOG_CHANNEL}"
            self.assertIn(expected, df.columns)

        for style in _REQUIRED_LEGACY_MORPHOLOGY_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)

        self.assertGreater(df.notna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
