"""Unit tests for blink morphology feature extraction."""

import unittest
import math
import logging

from pyblinker.blink_features.morphology.morphology_features import compute_morphology_features
from unit_test.blink_features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


class TestMorphologyFeatures(unittest.TestCase):
    """Tests for morphology feature calculations."""

    def setUp(self) -> None:
        logger.info("Preparing mock data for morphology tests...")
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.per_epoch = [[] for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)

    def test_first_epoch_features(self) -> None:
        """Verify basic morphology metrics for the first epoch."""
        feats = compute_morphology_features(self.per_epoch[0], self.sfreq)
        logger.debug(f"Morphology features epoch 0: {feats}")
        self.assertTrue(math.isclose(feats["blink_duration_mean"], 0.1))
        self.assertTrue(math.isclose(feats["blink_amplitude_mean"], 0.19))
        self.assertTrue(math.isclose(feats["blink_asymmetry_mean"], 1.0))
        self.assertTrue(math.isclose(feats["blink_inflection_count_mean"], 3.0))

    def test_nan_with_no_blinks(self) -> None:
        """An epoch without blinks should yield NaN for duration mean."""
        feats = compute_morphology_features(self.per_epoch[3], self.sfreq)
        logger.debug(f"Morphology features epoch 3: {feats}")
        self.assertTrue(math.isnan(feats["blink_duration_mean"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
