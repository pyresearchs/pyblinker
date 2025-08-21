"""
Blink count validation for individual raw epochs using ``prepare_refined_segments``.

Overview
--------
This module validates blink detection using the preprocessing helper that
combines the standard steps:

1. Load a raw EOG signal file (in .fif format).
2. Slice the recording into 30-second segments.
3. Refine blink annotations per segment.
4. Update each segment’s annotations with the refined timings.
5. Count blinks from the updated annotations and validate against ground truth.

Use
---
This test ensures refined annotations produce consistent blink counts
compared to the original ground-truth blink CSV for selected segments.
"""

import logging
from pathlib import Path
import unittest

import mne
import pandas as pd

from pyblinker.utils import prepare_refined_segments
from pyblinker.features.blink_events.event_features.blink_count import blink_count_epoch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]




class TestRawBlinkCount(unittest.TestCase):
    """Compare blink counts from sliced raw epochs to expected values."""

    def setUp(self) -> None:
        """
        Set up test fixture using :func:`prepare_refined_segments` to slice the
        raw file, refine blinks, and update annotations. Expected blink counts
        are loaded from the accompanying CSV for validation.
        """
        logger.info("Entering TestRawBlinkCount.setUp")

        self.raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        self.expected_csv = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_blink_count_epoch.csv"

        # prepare segments using the helper that performs slicing and refinement
        self.segments, self.refined = prepare_refined_segments(
            self.raw_path,
            "EOG-EEG-eog_vert_left",
            progress_bar=False,
        )

        # load expected blink counts and DataFrame for comparison
        self.df = pd.read_csv(self.expected_csv)
        self.expected = self.df.copy()
        logger.info("Exiting TestRawBlinkCount.setUp")

    @staticmethod
    def _count_blinks(raw: mne.io.BaseRaw, label: str | None = "blink") -> int:
        """
        Count blinks in a Raw segment using blink_count_epoch.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw segment to count blinks from.
        label : str | None
            The annotation label to count as blinks (None for default).

        Returns
        -------
        int
            Number of blink annotations.
        """
        from pyblinker.features.blink_events.event_features.blink_count import blink_count_epoch

        return blink_count_epoch(raw, label=label)

    def test_total_blink_count(self) -> None:
        """
        Validate blink counts for selected raw indices against:
        - computed counts
        - DataFrame results
        - expected CSV values
        """
        logger.info("Entering TestRawBlinkCount.test_total_blink_count")
        checks = {0: 2, 13: 4, 49: 13}
        for idx, expected_count in checks.items():
            count = self._count_blinks(self.segments[idx], label=None)
            logger.debug("Segment %d: counted %d blinks", idx, count)

            # compare against DataFrame
            df_count = int(self.df.loc[idx, "blink_count"])
            csv_count = int(self.expected.loc[idx, "blink_count"])

            self.assertEqual(count, expected_count)
            self.assertEqual(count, df_count)
            self.assertEqual(count, csv_count)
        logger.info("Exiting TestRawBlinkCount.test_total_blink_count")

    def test_segments_and_counts(self) -> None:
        """Segments should yield expected blink counts after refinement."""
        segments, refined = prepare_refined_segments(
            self.raw_path,
            "EOG-EEG-eog_vert_left",
            progress_bar=False,
        )
        expected = pd.read_csv(self.expected_csv)
        checks = {0: 2, 13: 4, 49: 13}
        for idx, expected_count in checks.items():
            count = blink_count_epoch(segments[idx], label=None)
            df_count = int(expected.loc[idx, "blink_count"])
            self.assertEqual(count, expected_count)
            self.assertEqual(count, df_count)
        self.assertNotIn("epoch_signal", refined[0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
