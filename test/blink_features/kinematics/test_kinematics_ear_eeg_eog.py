"""Full-modality kinematics coverage (EAR + EEG + EOG)."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.blink_features.kinematics.kinematic_features import (
    KinematicBlinkFeatureExtractor,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EAR_CHANNEL = "EAR-avg_ear"
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"


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

        # Optional: save kinematic features to CSV in the test output directory
        # output_dir = PROJECT_ROOT / "test" / "test_outputs"
        # output_dir.mkdir(parents=True, exist_ok=True)
        # output_path = output_dir / "full_modality_kinematics_ear_eeg_eog.csv"
        # df.to_csv(output_path, index=False)

        # Assert that all expected columns (from CSV) are present in the dataframe
        df_columns = set(df.columns)
        missing = [col for col in self.expected_columns if col not in df_columns]

        # Provide a clear failure message listing any missing columns
        self.assertFalse(
            missing,
            msg=(
                f"Missing {len(missing)} expected kinematic columns. "
                f"Examples: {missing[:10]}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
