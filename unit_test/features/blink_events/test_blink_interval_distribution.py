"""Unit tests for blink interval distribution features using ``ear_eog_raw.fif``.

Raw data segments (``mne.io.Raw``) are cropped from the test file and used
directly in the feature functions.
"""
import unittest
import math
import logging
from pathlib import Path
import mne

from pyblinker.features.blink_events.event_features.blink_interval_distribution import (
    blink_interval_distribution_segment,
    aggregate_blink_interval_distribution,
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestBlinkIntervalDistribution(unittest.TestCase):
    """Validate blink interval metrics computed on raw segments."""

    def setUp(self) -> None:
        """Load the sample raw file and create two 30s segments."""
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        self.segments = []
        for i in range(2):
            start = 30.0 * i
            stop = start + 30.0
            mini = raw.copy().crop(tmin=start, tmax=stop, include_tmax=False)
            ann = mini.annotations
            shifted = mne.Annotations(
                onset=ann.onset - start,
                duration=ann.duration,
                description=ann.description,
            )
            mini.set_annotations(shifted)
            self.segments.append(mini)

    def test_single_segment_features(self) -> None:
        """First segment has one blink interval of 0.4s."""
        feats = blink_interval_distribution_segment(self.segments[0], blink_label=None)
        logger.debug("Features epoch0: %s", feats)
        self.assertAlmostEqual(feats["blink_interval_min"], 0.4)
        self.assertAlmostEqual(feats["blink_interval_max"], 0.4)
        self.assertTrue(math.isnan(feats["blink_interval_std"]))

    def test_aggregate_across_segments(self) -> None:
        """Aggregation returns expected values for the first two epochs."""
        df = aggregate_blink_interval_distribution(self.segments, blink_label=None)
        logger.debug("Aggregated DF:\n%s", df)
        self.assertAlmostEqual(df.loc[0, "blink_interval_min"], 0.4)
        self.assertTrue(math.isnan(df.loc[1, "blink_interval_min"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
