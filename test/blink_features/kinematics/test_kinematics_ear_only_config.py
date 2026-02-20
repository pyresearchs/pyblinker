"""Scenario A: EAR-only kinematic pipeline coverage."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.kinematics.kinematic_features import (
    KinematicBlinkFeatureExtractor,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EAR_CHANNEL = "EAR-avg_ear"


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

        required_columns = [
                "ear__th_point__kinematic__vel_peak_abs_base_mean__EAR-avg_ear",
                "ear__ interpolated_threshold__kinematic__vel_peak_abs_base_mean____EAR-avg_ear",
                ]

        for col in required_columns:
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
