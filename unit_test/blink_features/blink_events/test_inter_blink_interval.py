"""
Unit tests for inter-blink interval (IBI) feature extraction.

This module verifies the correctness of IBI feature calculations, particularly
the mean IBI, under realistic conditions (with and without sufficient blink data).
"""

import unittest
import math
import logging

from pyblinker.blink_features.blink_events.event_features.inter_blink_interval import compute_ibi_features
from unit_test.blink_features.fixtures.mock_ear_generation import _generate_refined_ear
import logging
from pathlib import Path
import unittest

import mne

from pyblinker.blink_features.blink_events.event_features.blink_count import (
    blink_count_epoch,
)
from pyblinker.utils import onset_entry_to_blinks, slice_raw_into_mne_epochs
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
class TestInterBlinkInterval(unittest.TestCase):
    """
    Tests for `compute_ibi_features`, which calculates metrics on inter-blink intervals
    given blink timestamps and sampling frequency.
    """

    def setUp(self) -> None:
        """Load raw data and slice into epochs for blink counting."""
        logger.info("Setting up epochs for blink count tests...")
        raw_path = (
                PROJECT_ROOT
                / "unit_test"
                / "test_files"
                / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        logger.info("Epoch setup complete.")

        ## for this test, check the ibi for channel "EEG-E8","EOG-EEG-eog_vert_left","EAR-avg_ear"
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




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
