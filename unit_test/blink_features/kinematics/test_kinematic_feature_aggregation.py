"""Integration of blink counts with kinematic features."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.blink_features.kinematics import compute_kinematic_features
from pyblinker.utils import slice_raw_into_mne_epochs

from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestKinematicFeatureAggregation(unittest.TestCase):
    """Test aggregation with external blink counts."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = (
            PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_merge_blink_counts(self) -> None:
        """Joined DataFrame includes blink counts and kinematic stats."""
        ch = "EEG-E8"
        df = compute_kinematic_features(self.epochs, picks=ch)
        blink_counts_path = (
            PROJECT_ROOT
            / "unit_test"
            / "test_files"
            / "ear_eog_blink_count_epoch.csv"
        )
        blink_counts = pd.read_csv(blink_counts_path, index_col="epoch_id")
        df = df.join(blink_counts)

        metrics = [
            "peak_amp",
            "t2p",
            "vel_mean",
            "vel_peak",
            "acc_mean",
            "acc_peak",
            "rise_time",
            "fall_time",
            "auc",
            "symmetry",
        ]
        expected_cols = [
            f"{m}_{s}_{ch}" for m in metrics for s in ("mean", "std", "cv")
        ]
        assert_df_has_columns(self, df, expected_cols + ["blink_count"])

if __name__ == "__main__":
    unittest.main()
