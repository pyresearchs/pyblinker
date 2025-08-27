"""Tests for kinematic blink feature aggregation using epoch metadata."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.blink_features.kinematics import compute_kinematic_features
from pyblinker.blink_features.kinematics.per_blink import compute_segment_kinematics
from pyblinker.blink_features.energy.helpers import (
    _extract_blink_windows,
    _segment_to_samples,
    _safe_stats,
)
from pyblinker.utils import slice_raw_into_mne_epochs

from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestKinematicFeatures(unittest.TestCase):
    """Validate kinematic metrics computed from epoch metadata."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_dataframe_and_nan_epochs(self) -> None:
        """DataFrame has expected columns and NaNs for zero-blink epochs."""
        ch = "EEG-E8"
        df = compute_kinematic_features(self.epochs, picks=ch)
        blink_counts_path = (
            PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_blink_count_epoch.csv"
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
        expected_cols = [f"{m}_{s}_{ch}" for m in metrics for s in ("mean", "std", "cv")]
        assert_df_has_columns(self, df, expected_cols + ["blink_count"])
        assert_numeric_or_nan(self, df.iloc[0])

        zero_idx = blink_counts.index[blink_counts["blink_count"] == 0][0]
        self.assertTrue(df.drop(columns="blink_count").loc[zero_idx].isna().all())

    def test_manual_first_epoch(self) -> None:
        """Manual computation for the first epoch matches library output."""
        ch = "EEG-E8"
        df = compute_kinematic_features(self.epochs, picks=ch)
        sfreq = float(self.epochs.info["sfreq"])
        data = self.epochs.get_data(picks=[ch])
        meta = self.epochs.metadata.iloc[0]
        windows = _extract_blink_windows(meta, ch, 0)
        per_metric = {m: [] for m in (
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
        )}
        n_times = data.shape[-1]
        for onset, dur in windows:
            sl = _segment_to_samples(onset, dur, sfreq, n_times)
            seg = data[0, 0, sl]
            metrics = compute_segment_kinematics(seg, sfreq)
            for m in per_metric:
                per_metric[m].append(metrics[m])

        manual = {}
        for metric, values in per_metric.items():
            stats = _safe_stats(values)
            for stat_name, value in stats.items():
                manual[f"{metric}_{stat_name}_{ch}"] = value

        for key, val in manual.items():
            self.assertAlmostEqual(df.iloc[0][key], val, places=7)


if __name__ == "__main__":
    unittest.main()

