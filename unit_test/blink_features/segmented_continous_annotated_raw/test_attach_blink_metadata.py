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

        expected_cols = [
            "start_blink",
            "max_blink",
            "end_blink",
            "outer_start",
            "outer_end",
            "left_zero",
            "right_zero",
            "max_value",
            "max_blink_alternative",
            "max_pos_vel_frame",
            "max_neg_vel_frame",
            "left_base",
            "right_base",
            "duration_base",
            "duration_zero",
            "pos_amp_vel_ratio_zero",
            "peaks_pos_vel_zero",
            "neg_amp_vel_ratio_zero",
            "pos_amp_vel_ratio_base",
            "peaks_pos_vel_base",
            "neg_amp_vel_ratio_base",
            "closing_time_zero",
            "reopening_time_zero",
            "time_shut_base",
            "peak_max_blink",
            "peak_time_blink",
            "inter_blink_max_amp",
            "inter_blink_max_vel_base",
            "inter_blink_max_vel_zero",
        ]
        self.assertTrue(set(expected_cols).issubset(md.columns))

        for _, row in md.iterrows():
            if row["n_blinks"] == 0:
                self.assertTrue(pd.isna(row["blink_onset"]))
                self.assertTrue(pd.isna(row["blink_duration"]))
                for col in expected_cols:
                    self.assertTrue(pd.isna(row[col]))
            else:
                self.assertEqual(len(row["blink_onset"]), row["n_blinks"])
                self.assertEqual(len(row["blink_duration"]), row["n_blinks"])
                for col in expected_cols:
                    self.assertIsInstance(row[col], list)
                self.assertEqual(len(row["start_blink"]), row["n_blinks"])
                self.assertEqual(len(row["max_blink"]), row["n_blinks"])
                self.assertEqual(len(row["end_blink"]), row["n_blinks"])

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
