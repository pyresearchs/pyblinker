"""Unit tests for single blink morphology metrics."""

import unittest
import math
import logging

from pyblinker.features.morphology.per_blink import compute_single_blink_features
from unit_test.features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


class TestSingleBlinkFeatures(unittest.TestCase):
    """Verify per-blink morphology calculations."""

    def setUp(self) -> None:
        blinks, sfreq, _epoch_len, _n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.first_blink = [b for b in blinks if b["epoch_index"] == 0][0]

    def test_basic_metrics(self) -> None:
        """Check duration, amplitude and area for the first blink."""
        feats = compute_single_blink_features(self.first_blink, self.sfreq)
        logger.debug("Single blink features: %s", feats)
        self.assertTrue(math.isclose(feats["duration"], 0.1))
        self.assertTrue(math.isclose(feats["amplitude"], 0.19))
        self.assertTrue(math.isclose(feats["area"], 0.0019))

    def test_asymmetry_and_inflection(self) -> None:
        """Check asymmetry ratio and inflection count."""
        feats = compute_single_blink_features(self.first_blink, self.sfreq)
        self.assertTrue(math.isclose(feats["asymmetry"], 1.0))
        self.assertEqual(feats["inflection_count"], 3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
