"""Integration coverage for epoch morphology aggregation columns."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config
from test.helper import build_expected_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EOG_CHANNEL = "EOG-EEG-eog_vert_left"

stats = ["mean", "std", "cv"]

metrics_by_landmark = {
    "base": ["duration_base", "time_shut_base"],
    "half": ["duration_half_base", "duration_half_zero"],
    "inter_blink": ["inter_blink_max_amp"],
    "peak": ["peak_time_blink", "peak_time_tent", "peak_max_blink", "peak_max_tent"],
    "tent": [
        "duration_tent",
        "closing_time_tent",
        "reopening_time_tent",
        "time_shut_tent",
    ],
    "zero": [
        "duration_zero",
        "closing_time_zero",
        "reopening_time_zero",
        "time_shut_zero",
    ],
}

REQUIRED_MORPHOLOGY_METRICS = build_expected_metrics(
    landmark=list(metrics_by_landmark.keys()),
    metrics=metrics_by_landmark,
    stats=stats,
    modality="eog",
    feature="morphology",
    channel=EOG_CHANNEL,
)


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

        for style in REQUIRED_MORPHOLOGY_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, df.columns)


if __name__ == "__main__":
    unittest.main()
