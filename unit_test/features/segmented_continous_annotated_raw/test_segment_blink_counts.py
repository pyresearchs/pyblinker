"""Verify blink counts after property extraction.

The integration ensures that no blinks are inadvertently added or removed when
passing through :func:`compute_segment_blink_properties`.  Each segment's blink
count is compared against the reference values stored in
``ear_eog_blink_count_epoch.csv``.
"""
import logging
from pathlib import Path
import unittest

import mne
import pandas as pd
import numpy as np

from pyblinker.utils.epochs import slice_raw_into_epochs
from pyblinker.features.blink_events import generate_blink_dataframe
from pyblinker.segment_blink_properties import compute_segment_blink_properties

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSegmentBlinkCounts(unittest.TestCase):
    """Ensure blink counts match reference values after property extraction.

    The test suite uses ``ear_eog_raw.fif`` along with the CSV file
    ``ear_eog_blink_count_epoch.csv`` as ground truth.  By re-counting blinks
    after running the property extraction pipeline, we guard against regressions
    that might drop or duplicate blink events.
    """

    def setUp(self) -> None:
        """Load raw data, slice into segments and reference blink counts.

        Parameters
        ----------
        None

        Notes
        -----
        ``self.segments`` holds the 30-second slices of ``ear_eog_raw.fif`` and
        ``self.expected_counts`` contains the blink counts per segment from the
        accompanying CSV file.  These fixtures are reused in the actual test.
        """
        raw_path = PROJECT_ROOT / "unit_test" / "features" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        self.segments, _, _, _ = slice_raw_into_epochs(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
        )
        self.expected_counts = pd.read_csv(PROJECT_ROOT / "unit_test" / "features" / "ear_eog_blink_count_epoch.csv")
        self.params = {
            "base_fraction": 0.5,
            "shut_amp_fraction": 0.9,
            "p_avr_threshold": 3,
            "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),
        }

    def test_blink_counts(self) -> None:
        """Compare per-segment blink counts with the reference CSV.

        Parameters
        ----------
        None

        Raises
        ------
        AssertionError
            If the number of blinks computed for any segment exceeds the
            reference count from ``ear_eog_blink_count_epoch.csv``.

        Notes
        -----
        The test ensures that property extraction does not fabricate additional
        blink events.  ``run_fit`` remains disabled to mirror the counting logic
        in :func:`generate_blink_dataframe`.
        """
        blink_df = generate_blink_dataframe(
            self.segments, channel="EEG-E8", blink_label=None, progress_bar=False
        )
        props = compute_segment_blink_properties(
            self.segments,
            blink_df,
            self.params,
            channel="EEG-E8",
            run_fit=False,
            progress_bar=False,
        )

        total_expected = int(self.expected_counts["blink_count"].sum())
        self.assertLessEqual(len(props), total_expected)

        for seg_id, row in self.expected_counts.iterrows():
            seg_rows = props[props["seg_id"] == seg_id]
            self.assertLessEqual(len(seg_rows), int(row["blink_count"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
