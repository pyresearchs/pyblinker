"""Validate blink extraction on a full ``mne.Raw`` recording with fitting.

This module mirrors :mod:`test_long_continous_raw` but enables the
``run_fit`` flag when computing blink properties.  The fitting stage may drop
blinks with insufficient data, so the resulting table can contain fewer rows
than the event table.  The test ensures that tent-based features (e.g.
``duration_tent``) are present for the remaining blinks.
"""
import logging
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.blink_events import generate_blink_dataframe
from pyblinker.segment_blink_properties import (
    compute_segment_blink_properties,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestLongContinuousRawWithFit(unittest.TestCase):
    """Integration tests for a long continuous raw signal with fitting.

    The legacy MATLAB version of the project processes the entire
    ``ear_eog_raw.fif`` file as a single segment.  This test suite reproduces that
    workflow to ensure compatibility with the Python implementation and to
    validate the optional blink-fitting step.  Blink events are detected over
    the continuous recording and the total blink count, as well as extracted
    blink properties, are validated against expectations.
    """

    def setUp(self) -> None:
        """Prepare a single-segment raw object and blink metadata.

        The method reads ``ear_eog_raw.fif`` from the ``unit_test.test_files`` directory and
        stores it as a one-element list in ``self.segments``.  Blink events are
        extracted by :func:`pyblinker.features.blink_events.generate_blink_dataframe` and the
        expected total blink count is loaded from
        ``ear_eog_blink_count_epoch.csv`` which represents the epoch-based
        workflow.  A parameter dictionary for
        :func:`pyblinker.segment_blink_properties.compute_segment_blink_properties`
        is also constructed here.
        """
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        self.raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        self.segments = [self.raw]
        self.blink_df = generate_blink_dataframe(
            self.segments, channel="EEG-E8", blink_label=None, progress_bar=False
        )
        csv_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_blink_count_epoch.csv"
        counts = pd.read_csv(csv_path)
        self.total_expected = int(counts["blink_count"].sum())
        self.params = {
            "base_fraction": 0.5,
            "shut_amp_fraction": 0.9,
            "p_avr_threshold": 3,
            "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),
        }

    def test_dataframe_types(self) -> None:
        """Verify blink DataFrame integrity.

        The blink event table should contain exactly the expected number of
        rows and all sample-index columns must be typed as integers.  This
        mirrors the assumptions made in downstream processing functions.
        """
        self.assertFalse(self.blink_df.empty)
        self.assertEqual(len(self.blink_df), self.total_expected)
        required_cols = [
            "start_blink",
            "end_blink",
            "outer_start",
            "outer_end",
            "left_zero",
        ]
        for col in required_cols:
            with self.subTest(col=col):
                self.assertTrue(pd.api.types.is_integer_dtype(self.blink_df[col]))

    def test_blink_properties_extraction(self) -> None:
        """Compute blink properties with fitting enabled for the raw file.

        Blink properties are extracted for the entire recording using
        :func:`compute_segment_blink_properties` with ``run_fit=True``.  Because
        the fitting stage may drop some blinks, the resulting DataFrame can be
        smaller than the blink event table.  The test verifies that fitting
        produced tent-based columns.
        """
        props = compute_segment_blink_properties(
            self.segments,
            self.params,
            blink_df=self.blink_df,
            channel="EEG-E8",
            run_fit=True,
            progress_bar=False,
        )
        logger.debug("Blink properties head:\n%s", props.head())
        self.assertIsInstance(props, pd.DataFrame)
        self.assertFalse(props.empty)
        self.assertEqual(set(props["seg_id"].unique()), {0})
        self.assertLessEqual(len(props), self.total_expected)
        self.assertIn("duration_tent", props.columns)
        self.assertTrue(props["duration_tent"].notna().any())


if __name__ == "__main__":  # pragma: no cover - manual execution
    logging.basicConfig(level=logging.INFO)
    unittest.main()
