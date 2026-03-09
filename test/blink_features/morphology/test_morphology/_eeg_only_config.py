"""Scenario B: EEG-only morphology pipeline coverage."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.morphology.epoch_features import (
    MorphologyBlinkFeatureExtractor,
    _available_styles,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.evaluation import mat_data
from test.segment_config import build_segment_config


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EEG_CHANNEL = "EEG-E8"


data_dir = PROJECT_ROOT / "test" / "test_files"


class TestEegOnlyMorphologyPipeline(unittest.TestCase):
    """EEG-only morphology pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"

    def test_eeg_only_runs_(self) -> None:
        """EEG-only config runs and yields EEG outputs."""

        annotation_csv = data_dir / "ear_eog.csv"
        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)
        raw.set_annotations(mat_data.read_annotations_as_mne(annotation_csv))

        base_config = {
            "eeg": {
                "channel": EEG_CHANNEL,
                "seg_type": "base",
            }
        }
        segmentation_config = build_segment_config(raw, base_config=base_config)
        segmentation_config = {"eeg": segmentation_config["eeg"]}

        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )

        extractor = MorphologyBlinkFeatureExtractor(epochs=epochs)
        df = extractor.compute(picks=EEG_CHANNEL)

        self.assertNotIn("blink_onset_ear", epochs.metadata.columns)
        self.assertIn("blink_onset_eeg", epochs.metadata.columns)
        channel_columns = [col for col in df.columns if "__" in col]
        self.assertTrue(
            all(col.endswith(f"__{EEG_CHANNEL}") for col in channel_columns)
        )
        styles = _available_styles(tuple(epochs.metadata.columns), "eeg")
        for style in styles:
            for stat in ("mean", "std", "cv"):
                expected = f"eeg__{style}__morphology__duration_{stat}__{EEG_CHANNEL}"
                self.assertIn(expected, df.columns)
        for legacy in (
            "duration_zero",
            "duration_base",
            "duration_tent",
            "duration_half_base",
            "duration_half_zero",
            "closing_time_zero",
            "reopening_time_zero",
            "time_shut_zero",
            "time_shut_base",
            "closing_time_tent",
            "reopening_time_tent",
            "time_shut_tent",
            "inter_blink_max_amp",
        ):
            self.assertIn(legacy, df.columns)
        self.assertGreater(df.notna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
