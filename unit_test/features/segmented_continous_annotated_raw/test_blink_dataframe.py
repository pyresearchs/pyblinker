"""Validate blink-event DataFrame generation.

This module exercises :func:`pyblinker.features.blink_events.generate_blink_dataframe` on the
example ``ear_eog_raw.fif`` recording using the EEG-E8 channel.  The resulting
DataFrame is compared against ``ear_eog_blink_count_epoch.csv`` to ensure the
blink counts per segment and overall match the reference data.  The docstrings
provide explicit expectations so users know precisely what each assertion
verifies.
"""

import logging
from pathlib import Path
import unittest

import mne
import pandas as pd

from pyblinker.blink_features.blink_events import generate_blink_dataframe
from pyblinker.utils.epochs import slice_raw_into_epochs

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestBlinkDataFrame(unittest.TestCase):
    """Validate blink DataFrame generation from raw segments.

    The ``setUp`` method reads the example FIF file and slices it into
    30-second segments. ``ear_eog_blink_count_epoch.csv`` provides the expected
    blink counts for each segment which are used throughout the test.  The class
    ensures that :func:`generate_blink_dataframe` returns exactly those blink
    events with no duplicates or omissions.
    """

    def setUp(self) -> None:
        """Prepare segments and reference counts.

        Parameters
        ----------
        None

        Notes
        -----
        The raw file is read once and sliced into 30-second epochs.  Blink
        counts for each epoch are loaded from the accompanying CSV file for
        comparison in :meth:`test_row_counts`.
        """
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        self.segments, _, _, _ = slice_raw_into_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        self.expected_counts = pd.read_csv(
            PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_blink_count_epoch.csv"
        )

    def test_row_counts(self) -> None:
        """Verify row counts per segment and in total.

        Parameters
        ----------
        None

        Raises
        ------
        AssertionError
            If the counts derived from :func:`generate_blink_dataframe` do not
            match the reference CSV.

        Notes
        -----
        The check confirms that blink detection neither drops nor duplicates
        events relative to manual annotations.  Both the overall number of rows
        and the distribution across ``seg_id`` are validated.
        """
        df = generate_blink_dataframe(
            self.segments, channel="EEG-E8", blink_label=None, progress_bar=False
        )

        # The total number of blinks in all 60 segments
        total_expected = int(self.expected_counts["blink_count"].sum())
        self.assertEqual(len(df), total_expected)

        # Check that each segment has the expected number of blinks
        for seg_id, row in self.expected_counts.iterrows():
            seg_rows = df[df["seg_id"] == seg_id]
            self.assertEqual(len(seg_rows), int(row["blink_count"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
