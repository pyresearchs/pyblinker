"""Unit tests for :mod:`blink_count` feature extraction.

Validates blink counting logic across multiple epochs using metadata.
"""

import logging
from pathlib import Path
import unittest

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.blink_events.event_features.blink_count import (
    blink_count,
)
from pyblinker.utils import slice_raw_into_mne_epochs
from unit_test.blink_features.utils.helpers import assert_df_has_columns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class TestBlinkCount(unittest.TestCase):
    """Unit tests for blink counting from ``mne.Epochs`` metadata."""

    def setUp(self) -> None:
        """Load raw data and slice into epochs for blink counting."""
        logger.info("Setting up epochs for blink count tests...")
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
        # Load ground truth blink counts for cross-verification
        csv_path = (
            PROJECT_ROOT
            / "unit_test"
            / "test_files"
            / "ear_eog_blink_count_epoch.csv"
        )
        expected_full = pd.read_csv(csv_path).set_index("epoch_id")["blink_count"].astype(float)
        # Align ground truth with available epochs
        self.expected_counts = expected_full.loc[self.epochs.metadata.index]
        logger.info("Epoch setup complete.")

    def test_counts(self) -> None:
        """Verify blink counts are derived from metadata correctly."""
        df = blink_count(self.epochs)
        assert_df_has_columns(self, df, ["blink_onset", "blink_duration", "blink_count"])
        self.assertEqual(len(df), len(self.epochs))

        # passthrough metadata check
        pd.testing.assert_series_equal(
            df["blink_onset"], self.epochs.metadata["blink_onset"], check_names=False
        )
        pd.testing.assert_series_equal(
            df["blink_duration"], self.epochs.metadata["blink_duration"], check_names=False
        )
        # Verify blink counts against ground truth CSV for every epoch
        pd.testing.assert_series_equal(
            df["blink_count"], self.expected_counts, check_names=False
        )
        self.assertTrue(np.issubdtype(df["blink_count"].dtype, np.number))
        for idx, expected in self.expected_counts.items():
            self.assertEqual(df.loc[idx, "blink_count"], expected)
            self.assertTrue(np.isfinite(df.loc[idx, "blink_count"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
