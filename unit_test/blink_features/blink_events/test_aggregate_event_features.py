"""Tests for aggregating blink event features across all epochs."""
from __future__ import annotations

import unittest
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.blink_events.event_features import (
    aggregate_blink_event_features,
)
from pyblinker.utils import slice_raw_into_mne_epochs
from unit_test.blink_features.utils.helpers import assert_df_has_columns

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestAggregateBlinkFeatures(unittest.TestCase):
    """Validate aggregation of blink features from epochs."""

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
        csv_path = (
            PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_blink_count_epoch.csv"
        )
        self.expected_total = pd.read_csv(csv_path)["blink_count"].sum()

    def test_aggregate_all_features(self) -> None:
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        df = aggregate_blink_event_features(self.epochs, picks=picks)
        expected_cols = ["blink_total", "blink_rate"] + [f"ibi_{p}" for p in picks]
        assert_df_has_columns(self, df, expected_cols)
        self.assertEqual(len(df), 1)

        self.assertEqual(df.loc[0, "blink_total"], self.expected_total)
        epoch_len = self.epochs.tmax - self.epochs.tmin + 1.0 / self.epochs.info["sfreq"]
        total_duration = epoch_len * len(self.epochs)
        expected_rate = self.expected_total / total_duration * 60.0
        self.assertAlmostEqual(df.loc[0, "blink_rate"], expected_rate)

        for col in expected_cols:
            self.assertTrue(np.issubdtype(df[col].dtype, np.number))
        for ch in picks:
            self.assertTrue(np.isfinite(df.loc[0, f"ibi_{ch}"]))

    def test_missing_channel(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_blink_event_features(self.epochs, picks=["BAD-CHAN"])

    def test_select_subset(self) -> None:
        df = aggregate_blink_event_features(
            self.epochs, picks=["EEG-E8"], features=["blink_total"]
        )
        assert_df_has_columns(self, df, ["blink_total"])
        self.assertEqual(list(df.columns), ["blink_total"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

