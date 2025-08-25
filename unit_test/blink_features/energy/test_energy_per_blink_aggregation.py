"""Verify per-blink aggregation matches compute_energy_features.

The resulting :class:`pandas.DataFrame` also includes blink counts per epoch
from ``ear_eog_blink_count_epoch.csv`` to clarify rows with missing values.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.blink_features.energy.helpers import (
    _extract_blink_windows,
    _segment_to_samples,
    _safe_stats,
    _tkeo,
)
from pyblinker.utils import slice_raw_into_mne_epochs

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEnergyAggregation(unittest.TestCase):
    """Compare manual per-blink aggregation to library output."""

    def setUp(self) -> None:
        raw_path = (
            PROJECT_ROOT
            / "unit_test"
            / "test_files"
            / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_manual_aggregation(self) -> None:
        """Manual computation equals compute_energy_features."""
        ch = "EEG-E8"
        df = compute_energy_features(self.epochs, picks=ch)
        blink_counts_path = (
            PROJECT_ROOT
            / "unit_test"
            / "test_files"
            / "ear_eog_blink_count_epoch.csv"
        )
        blink_counts = pd.read_csv(blink_counts_path, index_col="epoch_id")
        df = df.join(blink_counts)
        sfreq = float(self.epochs.info["sfreq"])
        data = self.epochs.get_data(picks=[ch])
        n_epochs, _, n_times = data.shape
        records = []
        for ei in range(n_epochs):
            metadata_row = self.epochs.metadata.iloc[ei]
            windows = _extract_blink_windows(metadata_row)
            energies: list[float] = []
            tkeo_vals: list[float] = []
            lengths: list[float] = []
            vel_ints: list[float] = []
            for onset_s, duration_s in windows:
                sl = _segment_to_samples(onset_s, duration_s, sfreq, n_times)
                segment = data[ei, 0, sl]
                if segment.size == 0:
                    continue
                energies.append(float(np.sum(segment ** 2)))
                if segment.size >= 3:
                    psi = _tkeo(segment)
                    tkeo_vals.append(float(np.mean(np.abs(psi[1:-1]))))
                lengths.append(float(np.sum(np.abs(np.diff(segment)))))
                velocity = np.diff(segment) * sfreq
                vel_ints.append(float(np.sum(np.abs(velocity))))
            stats_energy = _safe_stats(energies)
            stats_tkeo = _safe_stats(tkeo_vals)
            stats_len = _safe_stats(lengths)
            stats_vel = _safe_stats(vel_ints)
            record: dict[str, float] = {}
            for metric, stats in zip(
                [
                    "blink_signal_energy",
                    "teager_kaiser_energy",
                    "blink_line_length",
                    "blink_velocity_integral",
                ],
                [stats_energy, stats_tkeo, stats_len, stats_vel],
            ):
                for stat_name, value in stats.items():
                    record[f"{metric}_{stat_name}_{ch}"] = value
            records.append(record)
        manual_df = pd.DataFrame.from_records(
            records, index=self.epochs.metadata.index
        )
        manual_df = manual_df.join(blink_counts)
        pd.testing.assert_frame_equal(df, manual_df)
        zero_idx = blink_counts.index[blink_counts["blink_count"] == 0][0]
        self.assertTrue(
            df.drop(columns="blink_count").loc[zero_idx].isna().all()
        )


if __name__ == "__main__":
    unittest.main()
