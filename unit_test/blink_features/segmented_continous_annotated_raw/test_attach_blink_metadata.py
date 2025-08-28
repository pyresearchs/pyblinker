"""Tests for attaching blink metadata to epochs."""

import logging
from pathlib import Path
import unittest

import mne
import numpy as np
import pandas as pd

from refine_annotation.util import slice_raw_into_mne_epochs_refine_annot
from pyblinker.segment_blink_properties import compute_segment_blink_properties
from pyblinker.utils.blink_metadata import attach_blink_metadata

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestAttachBlinkMetadata(unittest.TestCase):
    """Validate epoch-level blink metadata aggregation."""

    def setUp(self) -> None:
        """Load epochs and compute blink properties."""
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        epochs_full = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        self.params = {
            "base_fraction": 0.5,
            "shut_amp_fraction": 0.9,
            "p_avr_threshold": 3,
            "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),
        }
        blink_epochs = compute_segment_blink_properties(
            epochs_full, self.params, channel="EEG-E8", progress_bar=False
        )
        self.blink_df = blink_epochs.metadata.copy()
        # Drop the first epoch to exercise selection mapping
        self.epochs = epochs_full.copy().drop([0])

    def test_metadata_attachment(self) -> None:
        """Blink metadata merged correctly and long table returned."""
        long_df = attach_blink_metadata(self.epochs, self.blink_df)
        md = self.epochs.metadata

        self.assertEqual(len(md), len(self.epochs))
        self.assertListEqual(list(md.index), list(range(len(self.epochs))))

        for _, row in md.iterrows():
            if row["n_blinks"] == 0:
                self.assertTrue(pd.isna(row["blink_onset"]))
                self.assertTrue(pd.isna(row["blink_duration"]))
            else:
                self.assertEqual(len(row["blink_onset"]), row["n_blinks"])
                self.assertEqual(len(row["blink_duration"]), row["n_blinks"])

        # Numeric summaries allow selection
        _ = self.epochs[self.epochs.metadata["n_blinks"] >= 0]

        # Long table only includes kept epochs
        self.assertTrue(set(long_df["seg_id"]).issubset(set(self.epochs.selection)))

    def test_idempotent(self) -> None:
        """Repeated attachment updates only blink columns."""
        attach_blink_metadata(self.epochs, self.blink_df)
        md_first = self.epochs.metadata.copy()
        attach_blink_metadata(self.epochs, self.blink_df)
        pd.testing.assert_frame_equal(self.epochs.metadata, md_first)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    unittest.main()
