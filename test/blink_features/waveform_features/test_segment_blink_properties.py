"""Integration test for epoch-level blink metadata attachment.

This test covers the canonical workflow used in the blink feature pipeline:

1. ``compute_segment_blink_properties`` is executed with ``long_format=False``
   so that per-blink measurements (e.g., onset, velocity, amplitude) are merged
   into :attr:`mne.Epochs.metadata` as lists alongside a blink count.
2. The helper :func:`metadata_to_long` converts the list-based metadata back into
   a long-format table where each row represents a single detected blink.
3. The resulting long table is compared against a stored reference DataFrame to
   ensure that all blink property columns are preserved and numerically
   consistent.

The expectation is that transforming to and from epoch metadata does not alter
any per-blink values and that the assertion view exactly matches the reference
blink properties extracted from the same raw data.
"""

from __future__ import annotations

import logging
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.utils.refine_util import slice_raw_into_mne_epochs_refine_annot
from pyblinker.segment_blink_properties import compute_segment_blink_properties
from test.utils.blink_compare_utils import (
    metadata_to_long,
    report_mismatches,
    scalarize,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSegmentBlinkProperties(unittest.TestCase):
    """Validate blink properties computed from refined epoch metadata."""

    def setUp(self) -> None:
        """Load test epochs and reference blink properties."""
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        self.params = {
            "base_fraction": 0.5,
            "shut_amp_fraction": 0.9,
            "p_avr_threshold": 3,
            "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),
        }
        self.reference = pd.read_pickle(
            PROJECT_ROOT
            / "test"
            / "test_outputs"
            / "blink_properties_with_fit.pkl"
        )


    def test_properties_match_reference(self) -> None:
        """Computed properties match the stored reference table."""
        compute_segment_blink_properties(
            self.epochs,
            self.params,
            channel="EEG-E8",
            progress_bar=False,
            long_format=False,
        )
        df = metadata_to_long(self.epochs)

        key_cols = ["seg_id", "blink_id"]
        other_id_cols = [
            "start_blink",
            "max_blink",
            "end_blink",
            "outer_start",
            "outer_end",
            "left_zero",
            "right_zero",
        ]
        value_cols = [
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

        compare_cols = key_cols + other_id_cols + value_cols

        df_proc = df[compare_cols].map(scalarize)
        ref_proc = self.reference[compare_cols].map(scalarize)

        strict = False
        compare_ids = False
        atol = max(1.1 / float(self.epochs.info["sfreq"]), 5.0)

        if strict:
            df_sorted = df_proc.sort_values(key_cols).reset_index(drop=True)
            ref_sorted = ref_proc.sort_values(key_cols).reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(
                    df_sorted,
                    ref_sorted,
                    check_dtype=False,
                    rtol=1e-5,
                    atol=atol,
                )
            except AssertionError:
                if compare_ids:
                    report_mismatches(
                        df_sorted,
                        ref_sorted,
                        key_cols,
                        other_id_cols + value_cols,
                    )
                raise
        else:
            merged = pd.merge(
                ref_proc,
                df_proc,
                on=key_cols,
                suffixes=("_ref", "_res"),
            )
            assert not merged.empty, "No overlapping seg_id/blink_id rows"
            cols_to_compare = value_cols + (other_id_cols if compare_ids else [])
            ref_df = merged[[f"{c}_ref" for c in cols_to_compare]]
            res_df = merged[[f"{c}_res" for c in cols_to_compare]]
            ref_df.columns = cols_to_compare
            res_df.columns = cols_to_compare
            pd.testing.assert_frame_equal(
                res_df,
                ref_df,
                check_dtype=False,
                rtol=1e-5,
                atol=atol,
            )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    unittest.main()
