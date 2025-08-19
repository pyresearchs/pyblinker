"""Unit tests for partial and complete blink classification."""

import unittest
import logging

from pyblinker.features.blink_events.classification import aggregate_classification_features
from unit_test.features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


class TestBlinkClassification(unittest.TestCase):
    """Verify partial and complete blink metrics."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.epoch_len = epoch_len
        self.n_epochs = n_epochs
        self.blinks = blinks

    def test_all_complete(self) -> None:
        """Synthetic data contains only complete blinks."""
        df = aggregate_classification_features(
            self.blinks, self.sfreq, self.epoch_len, self.n_epochs, threshold=0.1
        )
        logger.debug("Classification df:\n%s", df)
        self.assertTrue((df["Partial_Blink_Total"] == 0).all())
        self.assertTrue((df["Complete_Blink_Total"] > 0).any())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
