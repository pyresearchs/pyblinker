"""Unit tests for open-eye period features using synthetic data.

Blink annotations and epoch signals are generated via the
``mock_ear_generation`` fixture creating :class:`mne.Epochs` objects.
"""
import unittest
import math
import logging

from pyblinker.blink_features.open_eye.features import (
    baseline_mean_epoch,
    baseline_drift_epoch,
    baseline_std_epoch,
    perclos_epoch,
    micropause_count_epoch,
)
from unit_test.blink_features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


class TestOpenEyeFeatures(unittest.TestCase):
    """Verify baseline and open-eye metrics."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        per_epoch_signal = {}
        per_epoch_blinks = {}
        for b in blinks:
            idx = b["epoch_index"]
            per_epoch_signal.setdefault(idx, b["epoch_signal"])
            per_epoch_blinks.setdefault(idx, []).append(b)
        self.signal0 = per_epoch_signal[0]
        self.blinks0 = per_epoch_blinks[0]

    def test_baseline_mean(self) -> None:
        """Baseline mean should match expected value."""
        mean_val = baseline_mean_epoch(self.signal0, self.blinks0)
        logger.debug("baseline mean: %s", mean_val)
        self.assertTrue(math.isclose(mean_val, 0.31987358421614165))

    def test_baseline_std(self) -> None:
        """Baseline standard deviation is computed."""
        std_val = baseline_std_epoch(self.signal0, self.blinks0)
        self.assertTrue(math.isclose(std_val, 0.004922817090535525))

    def test_perclos_zero(self) -> None:
        """No closure beyond threshold in mock data."""
        val = perclos_epoch(self.signal0, self.blinks0)
        self.assertEqual(val, 0.0)

    def test_micropause_none(self) -> None:
        """No micropauses expected in mock signal."""
        count = micropause_count_epoch(self.signal0, self.blinks0, self.sfreq)
        self.assertEqual(count, 0)

    def test_baseline_drift_small(self) -> None:
        """Baseline drift should be near zero."""
        slope = baseline_drift_epoch(self.signal0, self.blinks0, self.sfreq)
        self.assertTrue(abs(slope) < 1e-4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
