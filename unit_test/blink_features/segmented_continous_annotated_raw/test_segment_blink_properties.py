"""Integration test for blink property extraction with refined metadata."""

from __future__ import annotations

import ast
import logging
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

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

    def test_properties_match_reference(self) -> None:
        """Computed properties match the stored reference table."""
        blink_epochs = compute_segment_blink_properties(
            self.epochs, None, self.params, channel="EEG-E8", progress_bar=False
        )
        df = blink_epochs.metadata
        cols = [
            "inter_blink_max_vel_base",
            "inter_blink_max_amp",
            "time_shut_base",
            "neg_amp_vel_ratio_base",
            "pos_amp_vel_ratio_zero",
            "duration_base",
        ]
        def _first_scalar(val: object) -> float:
            """Return the first numeric value from scalars, lists or strings."""
            if isinstance(val, str):
                try:
                    val = ast.literal_eval(val)
                except (SyntaxError, ValueError):
                    return float(val)
            if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                return float(val[0]) if len(val) else float("nan")
            return float(val)

        df_first = df[cols].head(3).applymap(_first_scalar).reset_index(drop=True)
        ref_first = (
            self.reference[cols].head(3).applymap(_first_scalar).reset_index(drop=True)
        )
        atol = 1.1 / float(self.epochs.info["sfreq"])
        pd.testing.assert_frame_equal(
            df_first,
            ref_first,
            check_dtype=False,
            rtol=1e-5,
            atol=atol,
        )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    unittest.main()
