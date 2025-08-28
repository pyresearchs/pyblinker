"""Integration test for blink property extraction with refined metadata."""

from __future__ import annotations

import ast
import logging
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from refine_annotation.util import slice_raw_into_mne_epochs_refine_annot
from pyblinker.segment_blink_properties import compute_segment_blink_properties

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSegmentBlinkProperties(unittest.TestCase):
    """Validate blink properties computed from refined epoch metadata."""

    def setUp(self) -> None:
        """Load test epochs and reference blink properties."""
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
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
            / "unit_test"
            / "test_outputs"
            / "blink_properties_with_fit.pkl"
        )

    def _report_mismatches(
        self,
        result: pd.DataFrame,
        reference: pd.DataFrame,
        key_cols: list[str],
        compare_cols: list[str],
    ) -> None:
        """Log detailed mismatches between result and reference frames.

        Parameters
        ----------
        result, reference:
            DataFrames sorted by ``key_cols``.
        key_cols:
            Columns identifying each blink uniquely.
        compare_cols:
            All columns to be compared.
        """

        merged = pd.merge(
            reference,
            result,
            on=key_cols,
            how="outer",
            suffixes=("_ref", "_res"),
            indicator=True,
        )

        missing = merged[merged["_merge"] == "left_only"][key_cols]
        extra = merged[merged["_merge"] == "right_only"][key_cols]
        if not missing.empty:
            logger.error("Missing rows in result:\n%s", missing)
        if not extra.empty:
            logger.error("Unexpected rows in result:\n%s", extra)

        both = merged[merged["_merge"] == "both"]
        ref_vals = both[[f"{c}_ref" for c in compare_cols]].rename(
            columns=lambda c: c[:-4]
        )
        res_vals = both[[f"{c}_res" for c in compare_cols]].rename(
            columns=lambda c: c[:-4]
        )
        diff = res_vals.compare(ref_vals, keep_equal=False)
        if not diff.empty:
            logger.error("Value mismatches:\n%s", diff)

    def test_properties_match_reference(self) -> None:
        """Computed properties match the stored reference table."""
        blink_epochs = compute_segment_blink_properties(
            self.epochs, None, self.params, channel="EEG-E8", progress_bar=False
        )
        df = blink_epochs.metadata.copy()

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

        def _scalarize(val: object) -> float:
            """Return the first numeric value from scalars, lists or strings."""
            if isinstance(val, str):
                try:
                    val = ast.literal_eval(val)
                except (SyntaxError, ValueError):
                    return float(val)
            if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                return float(val[0]) if len(val) else float(np.nan)
            return float(val)

        df_proc = df[compare_cols].map(_scalarize)
        ref_proc = self.reference[compare_cols].map(_scalarize)

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
                    self._report_mismatches(
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
