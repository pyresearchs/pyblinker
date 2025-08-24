"""Unit tests for EAR baseline and extrema features."""

import logging
import unittest

from pyblinker.blink_features.ear_metrics import aggregate_ear_features, ear_before_blink_avg_epoch
from unit_test.features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


class TestEarFeatures(unittest.TestCase):
    """Tests for EAR baseline and extrema calculations."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.blinks = blinks
        self.n_epochs = n_epochs

    def test_first_epoch_preblink(self) -> None:
        """Average EAR before first blink should be below baseline."""
        epoch0_signal = self.blinks[0]["epoch_signal"]
        val = ear_before_blink_avg_epoch(epoch0_signal, [self.blinks[0]], self.sfreq)
        logger.debug("EAR before blink avg: %s", val)
        self.assertTrue(val < epoch0_signal[0])

    def test_aggregate_shape(self) -> None:
        """Aggregated DataFrame has expected columns."""
        df = aggregate_ear_features(self.blinks, self.sfreq, self.n_epochs)
        logger.debug("EAR feature columns: %s", df.columns)
        self.assertIn("EAR_Before_Blink_left_avg", df.columns)
        self.assertIn("EAR_left_min", df.columns)
        self.assertEqual(len(df), self.n_epochs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
