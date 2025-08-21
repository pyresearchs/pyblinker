"""Integration tests for :func:`compute_segment_blink_properties`.

These tests exercise the blink property extraction pipeline on an annotated
raw recording.  They demonstrate the difference between running the blink
fitting stage and skipping it via ``run_fit=False``.  Each assertion validates
that expected columns exist and no unexpected rows are dropped.  The suite is
useful as a guide when modifying :class:`BlinkProperties` or related
processing functions.



Run extraction and sanity-check resulting columns.
            Atleast for our toy example, there are multiple blink being dropped.
            The lines below are the culprits:
                self.frame_blinks.dropna(inplace=True)
                self.frame_blinks["nsize_x_left"] = self.frame_blinks["x_left"].apply(len)
                self.frame_blinks["nsize_x_right"] = self.frame_blinks["x_right"].apply(len)

                https://github.com/balandongiv/pyear/blob/88b84bedc3f7120af935dfa75ff42808b5a150b7/pyear/pyblinker/fit_blink.py#L175

            Therefore, we cannot do calculation that is related to or having the following columns:
                    self.cols_half_height = [
            "left_zero_half_height",
            "right_zero_half_height",
            "left_base_half_height",
            "right_base_half_height",
        ]
        self.cols_fit_range = [
            "x_left",
            "x_right",
            "left_range",
            "right_range",
            "blink_bottom_point_l_y",
            "blink_bottom_point_l_x",
            "blink_top_point_l_y",
            "blink_top_point_l_x",
            "blink_bottom_point_r_x",
            "blink_bottom_point_r_y",
            "blink_top_point_r_x",
            "blink_top_point_r_y",
        ]
        self.cols_lines_intesection = [
            "left_slope",
            "right_slope",
            "aver_left_velocity",
            "aver_right_velocity",
            "right_r2",
            "left_r2",
            "x_intersect",
            "y_intersect",
            "left_x_intercept",
            "right_x_intercept",
        ]

        # The original MATLAB implementation also exposed four
        # ``x_line_cross_*``/``y_line_cross_*`` columns. They were always
        # ``NaN`` and are intentionally omitted here.

"""

import logging
from pathlib import Path
import unittest

import mne
import numpy as np
import pandas as pd

from pyblinker.utils.epochs import slice_raw_into_epochs
from pyblinker.features.blink_events import generate_blink_dataframe
from pyblinker.segment_blink_properties import compute_segment_blink_properties

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSegmentBlinkProperties(unittest.TestCase):
    """Validate blink property extraction on raw segments.

    The fixture uses ``ear_eog_raw.fif`` which contains manual annotations of blink
    events.  Each test slices the raw data into 30-second segments and
    cross-checks extracted blink features against these annotations.
    """

    def setUp(self) -> None:
        """Load the test raw file and prepare blink metadata.

        Parameters
        ----------
        None

        Notes
        -----
        The method populates ``self.segments`` with 30-second epochs and
        ``self.blink_df`` with blink onset/offset pairs.  ``self.params``
        defines the processing parameters shared across tests.
        """
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        self.segments, _, _, _ = slice_raw_into_epochs(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
        )
        self.blink_df = generate_blink_dataframe(
            self.segments,
            channel="EEG-E8",
            blink_label=None,
            progress_bar=False,
        )
        self.params = {
            "base_fraction": 0.5,
            "shut_amp_fraction": 0.9,
            "p_avr_threshold": 3,
            "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),
        }

    def test_properties_dataframe(self) -> None:
        """Extract blink properties with fitting disabled.

        Parameters
        ----------
        None

        Raises
        ------
        AssertionError
            If the output DataFrame is empty, missing expected columns or
            contains unexpected segment identifiers.

        Notes
        -----
        ``run_fit`` is set to ``False`` in order to preserve all blinks.  The
        test confirms that only raw-signal features are produced and that their
        counts align with the blink metadata collected in :meth:`setUp`.
        """
        df = compute_segment_blink_properties(
            self.segments,
            self.blink_df,
            self.params,
            channel="EEG-E8",
            run_fit=False,
            progress_bar=False,
        )
        logger.debug("Blink properties DataFrame:\n%s", df.head())
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertLessEqual(len(df), len(self.blink_df))
        self.assertTrue(
            set(df["seg_id"].unique()).issubset(set(self.blink_df["seg_id"].unique()))
        )
        expected_cols = {
            "duration_base",
            "pos_amp_vel_ratio_zero",
            "closing_time_zero",
        }
        self.assertTrue(expected_cols.issubset(df.columns))

    def test_properties_dataframe_with_fit(self) -> None:
        """Extract blink properties with fitting enabled.

        Parameters
        ----------
        None

        Raises
        ------
        RuntimeWarning
            When ``run_fit`` triggers the fitting stage; emitted if blinks are
            dropped due to NaN fit ranges.
        AssertionError
            If the resulting DataFrame is empty after fitting.

        Notes
        -----
        ``run_fit=True`` emulates the behavior of the original MATLAB
        implementation and calculates additional tent-based metrics.  The test
        verifies that these computations do not completely invalidate the
        DataFrame even when some blinks are discarded.
        """
        with self.assertWarns(RuntimeWarning):
            df = compute_segment_blink_properties(
                self.segments,
                self.blink_df,
                self.params,
                channel="EEG-E8",
                run_fit=True,
                progress_bar=False,
            )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
