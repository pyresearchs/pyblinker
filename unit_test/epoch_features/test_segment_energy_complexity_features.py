"""Tests for time-domain complexity metrics on raw segments."""
import logging
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.utils.epochs import slice_raw_into_epochs
from pyblinker.blink_features.energy.segment_features import compute_time_domain_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSegmentEnergyComplexity(unittest.TestCase):
    """Validate energy and complexity metrics for a real segment."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        self.segments, _, _, _ = slice_raw_into_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        self.sfreq = raw.info["sfreq"]

    def _validate_segments(self, channel: str) -> None:
        """Aggregate metrics from all segments of a channel into a DataFrame.

        Parameters
        ----------
        channel : str
            Name of the channel to pick from each segment.
        """
        records = []
        for segment in self.segments:
            signal = segment.get_data(picks=channel)[0]
            feats = compute_time_domain_features(signal, self.sfreq)
            logger.debug("%s features: %s", channel, feats)
            self.assertSetEqual(
                set(feats.keys()),
                {"energy", "teager", "line_length", "velocity_integral"},
            )
            for value in feats.values():
                self.assertFalse(np.isnan(value))
            records.append(feats)

        df = pd.DataFrame(records)
        self.assertEqual(len(df), len(self.segments))
        self.assertSetEqual(
            set(df.columns),
            {"energy", "teager", "line_length", "velocity_integral"},
        )
        self.assertFalse(df.isna().any().any())

    def test_first_segment_values(self) -> None:
        """Check a subset of feature values for the first segment."""
        signal = self.segments[0].get_data(picks="EAR-avg_ear")[0]
        feats = compute_time_domain_features(signal, self.sfreq)
        logger.debug("Segment time-domain features: %s", feats)
        self.assertAlmostEqual(feats["energy"], 2.608998, places=5)
        self.assertAlmostEqual(feats["line_length"], 14.33153, places=5)
        self.assertTrue(feats["velocity_integral"] > 0)

    def test_all_segments_ear_channel(self) -> None:
        """Validate metrics on all segments using the EAR channel."""
        self._validate_segments("EAR-avg_ear")

    def test_all_segments_eog_channel(self) -> None:
        """Validate metrics on all segments using the EOG channel."""
        self._validate_segments("EOG-EEG-eog_vert_left")

    def test_all_segments_eeg_channel(self) -> None:
        """Validate metrics on all segments using the EEG channel."""
        self._validate_segments("EEG-E8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
