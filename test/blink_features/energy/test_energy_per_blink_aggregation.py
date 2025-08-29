"""Verify per-blink aggregation matches compute_energy_features.

The resulting :class:`pandas.DataFrame` also includes blink counts per epoch
from ``ear_eog_blink_count_epoch.csv`` to clarify rows with missing values.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.utils.refine_util import slice_raw_into_mne_epochs_refine_annot
from test.blink_features.utils.energy_manual import manual_epoch_energy_features
from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEnergyAggregation(unittest.TestCase):
    """Compare manual per-blink aggregation to library output."""

    def setUp(self) -> None:
        raw_path = (
            PROJECT_ROOT
            / "test"
            / "test_files"
            / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_manual_aggregation(self) -> None:
        """Library aggregation matches a manual per-epoch computation.

        Each epoch may contain a different number of blinks, so we iterate
        over epochs to build a ``manual_df`` using
        :func:`manual_epoch_energy_features`. The resulting DataFrame is
        compared with ``df`` from :func:`compute_energy_features` to validate
        the aggregation logic. Joining blink counts reveals that epochs with
        zero blinks naturally produce ``NaN`` values for energy metrics.
        """
        ch = "EEG-E8"
        df = compute_energy_features(self.epochs, picks=ch)
        blink_counts_path = (
            PROJECT_ROOT
            / "test"
            / "test_files"
            / "ear_eog_blink_count_epoch.csv"
        )
        blink_counts = pd.read_csv(blink_counts_path, index_col="epoch_id")
        df = df.join(blink_counts)
        metrics = (
            "blink_signal_energy",
            "teager_kaiser_energy",
            "blink_line_length",
            "blink_velocity_integral",
        )
        expected_cols = [
            f"{m}_{s}_{ch}" for m in metrics for s in ("mean", "std", "cv")
        ]
        assert_df_has_columns(self, df, expected_cols + ["blink_count"])
        assert_numeric_or_nan(self, df.iloc[0])
        sfreq = float(self.epochs.info["sfreq"])
        data = self.epochs.get_data(picks=[ch])
        records = [
            manual_epoch_energy_features(
                data[ei, 0], self.epochs.metadata.iloc[ei], sfreq, ch, ei
            )
            for ei in range(len(self.epochs))
        ]
        manual_df = pd.DataFrame.from_records(
            records, index=self.epochs.metadata.index
        ).join(blink_counts)
        pd.testing.assert_frame_equal(df, manual_df)
        zero_idx = blink_counts.index[blink_counts["blink_count"] == 0][0]
        self.assertTrue(
            df.drop(columns="blink_count").loc[zero_idx].isna().all()
        )


if __name__ == "__main__":
    unittest.main()
