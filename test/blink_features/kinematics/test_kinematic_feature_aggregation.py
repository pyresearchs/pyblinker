"""Integration of blink counts with kinematic features."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.kinematics import compute_kinematic_features
from pyblinker.blink_features.kinematics.per_blink import compute_segment_kinematics
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config

from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestKinematicFeatureAggregation(unittest.TestCase):
    """Test aggregation with external blink counts."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = (
            PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        segmentation_config = build_segment_config(raw)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )

    def test_merge_blink_counts(self) -> None:
        """Joined DataFrame includes blink counts and kinematic stats."""
        ch = "EEG-E8"
        df = compute_kinematic_features(self.epochs, picks=ch)
        blink_counts_path = (
            PROJECT_ROOT
            / "test"
            / "test_files"
            / "ear_eog_blink_count_epoch.csv"
        )
        blink_counts = pd.read_csv(blink_counts_path, index_col="epoch_id")
        df = df.join(blink_counts)

        metric_keys = tuple(
            compute_segment_kinematics(np.zeros(3), 1.0, methods=("base",)).keys()
        )
        expected_cols = [
            f"{m}_{s}_{ch}" for m in metric_keys for s in ("mean", "std", "cv")
        ]
        assert_df_has_columns(self, df, expected_cols + ["blink_count"])
        assert_numeric_or_nan(self, df.iloc[0])
        zero_idx = self.epochs.metadata.index[
            self.epochs.metadata["n_blinks"] == 0
        ][0]
        self.assertTrue(df.drop(columns="blink_count").loc[zero_idx].isna().all())

if __name__ == "__main__":
    unittest.main()
