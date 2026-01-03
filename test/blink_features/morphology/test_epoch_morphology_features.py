"""Unit tests for epoch-level blink morphology features."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np

from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from pyblinker.blink_features.morphology.per_blink import compute_blink_waveform_metrics
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot

from ..utils.helpers import (
    assert_df_has_columns,
    assert_numeric_or_nan,
    morphology_column_names,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEpochMorphologyFeatures(unittest.TestCase):
    """Validate morphology feature extraction from epochs."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_schema_and_alignment(self) -> None:
        """DataFrame has expected columns and indexing for first epochs."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        df = compute_epoch_morphology_features(self.epochs, picks=picks)
        assert_df_has_columns(self, df, morphology_column_names(picks))
        self.assertEqual(len(df), len(self.epochs))
        self.assertTrue(df.index.equals(self.epochs.metadata.index))
        for idx in range(4):
            self.assertIn(idx, df.index)
            assert_numeric_or_nan(self, df.iloc[idx])

        zero_idx = self.epochs.metadata.index[self.epochs.metadata["n_blinks"] == 0][0]
        self.assertTrue(df.loc[zero_idx].isna().all())
    #
    def test_channel_suffixes(self) -> None:
        """Verify per-channel suffixes are present in column names."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        df = compute_epoch_morphology_features(self.epochs, picks=picks)
        for ch in picks:
            cols = [c for c in df.columns if c.endswith(f"_{ch}")]
            self.assertTrue(cols, msg=f"Missing columns for {ch}")
    #
    def test_missing_channel(self) -> None:
        """Unknown channel names should raise a ValueError."""
        with self.assertRaises(ValueError):
            compute_epoch_morphology_features(self.epochs, picks="not-a-channel")
    #

    def test_empty_epochs(self) -> None:
        """Empty epoch objects yield empty feature DataFrames with schema."""
        picks = ["EAR-avg_ear"]
        df = compute_epoch_morphology_features(self.epochs[:0], picks=picks)
        assert_df_has_columns(self, df, morphology_column_names(picks))
        self.assertEqual(len(df), 0)

    def test_waveform_metric_suffixes(self) -> None:
        """Waveform metrics expose method suffixes and modality-specific NaNs."""
        segment = np.array([0.0, 0.5, 0.1, 0.0])
        metrics = compute_blink_waveform_metrics(
            {"base": segment, "tent": segment}, 100.0, modality="eeg"
        )
        self.assertIn("area_abs_total_rect_base", metrics)
        self.assertIn("area_abs_total_rect_tent", metrics)

        ear_metrics = compute_blink_waveform_metrics(
            segment, 100.0, methods=("zero",), modality="ear"
        )
        self.assertTrue(
            np.isnan(ear_metrics["area_abs_total_rect_zero"]),
            msg="EAR modality should yield NaN for zero-crossing metrics",
        )


if __name__ == "__main__":
    unittest.main()
