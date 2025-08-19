"""
Unit tests for inter-blink interval (IBI) feature extraction.

This module verifies the correctness of IBI feature calculations, particularly
the mean IBI, under realistic conditions (with and without sufficient blink data).
"""

import unittest
import math
import logging

from pyblinker.features.blink_events.event_features.inter_blink_interval import compute_ibi_features
from unit_test.features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)

class TestInterBlinkInterval(unittest.TestCase):
    """
    Tests for `compute_ibi_features`, which calculates metrics on inter-blink intervals
    given blink timestamps and sampling frequency.
    """

    def setUp(self) -> None:
        """
        Load mock blink data and organize it into epochs.
        """
        logger.info("Setting up test fixture for IBI features...")
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.per_epoch = [[] for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)
        logger.debug(f"Blink counts per epoch: {[len(ep) for ep in self.per_epoch]}")

    def test_all_ibi_features_first_epoch(self) -> None:
        """
        Verify all inter-blink interval (IBI) features for the first epoch against expected values.
        Ensures correctness for computed statistical metrics and proper handling of undefined ones.
        """
        logger.info("Testing all IBI features for the first epoch...")
        feats = compute_ibi_features(self.per_epoch[0], self.sfreq)

        # Exact matches (deterministic blink timing)
        self.assertTrue(math.isclose(feats["ibi_mean"], 2.9), "Expected mean IBI ~2.9 for first epoch")
        self.assertTrue(math.isclose(feats["ibi_std"], 0.0), "Expected IBI standard deviation to be 0.0 (no variability)")
        self.assertTrue(math.isclose(feats["ibi_median"], 2.9), "Expected IBI median to equal 2.9 (same as mean)")
        self.assertTrue(math.isclose(feats["ibi_min"], 2.9), "Expected IBI minimum to be 2.9 (only value present)")
        self.assertTrue(math.isclose(feats["ibi_max"], 2.9), "Expected IBI maximum to be 2.9 (only value present)")
        self.assertTrue(math.isclose(feats["ibi_cv"], 0.0), "Expected IBI coefficient of variation to be 0.0 (no variation)")
        self.assertTrue(math.isclose(feats["ibi_rmssd"], 0.0), "Expected IBI RMSSD to be 0.0 (no successive differences)")

        # NaN features (due to insufficient blink count or undefined variability)
        self.assertTrue(math.isnan(feats["poincare_sd1"]), "Expected NaN for Poincaré SD1 due to zero variability")
        self.assertTrue(math.isnan(feats["poincare_sd2"]), "Expected NaN for Poincaré SD2 due to zero variability")
        self.assertTrue(math.isnan(feats["poincare_ratio"]), "Expected NaN for Poincaré ratio (undefined with zero SD2)")
        self.assertTrue(math.isnan(feats["ibi_permutation_entropy"]), "Expected NaN for permutation entropy (not enough unique values)")
        self.assertTrue(math.isnan(feats["ibi_hurst_exponent"]), "Expected NaN for Hurst exponent (requires longer time series)")


    def test_ibi_nan_for_insufficient_blinks(self) -> None:
        """
        If fewer than two blinks are present in an epoch, the mean IBI should be NaN.
        """
        logger.info("Testing IBI mean for an epoch with < 2 blinks (e.g. epoch 3)...")
        feats = compute_ibi_features(self.per_epoch[3], self.sfreq)
        ibi_mean = feats.get("ibi_mean", None)
        logger.debug(f"Computed IBI mean (epoch 3): {ibi_mean}")
        self.assertTrue(math.isnan(ibi_mean), "Expected NaN for mean IBI with insufficient data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
