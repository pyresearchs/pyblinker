"""
Unit tests for :mod:`blink_count` feature extraction.

Validates blink counting logic across multiple epochs.
"""

import unittest
import logging

from pyblinker.blink_features.blink_events.event_features.blink_count import blink_count_epoch
from unit_test.blink_features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)

class TestBlinkCount(unittest.TestCase):
    """
    Unit tests for the `blink_count_epoch` function, which returns the number
    of blinks present in a given epoch's blink list.
    """

    def setUp(self) -> None:
        """
        Generate mock blink data and split it by epoch index.
        """
        logger.info("Setting up mock blink data for blink count tests...")
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.per_epoch = [[] for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)
        logger.debug(f"Blink counts per epoch: {[len(ep) for ep in self.per_epoch]}")

    def test_counts(self) -> None:
        """
        Verify blink count for known epochs:
        - Epoch 0: Contains 3 blinks.
        - Epoch 3: Contains 0 blinks.
        """
        logger.info("Testing blink count for epoch 0...")
        count_epoch_0 = blink_count_epoch(self.per_epoch[0])
        logger.debug(f"Epoch 0 blink count: {count_epoch_0}")
        self.assertEqual(count_epoch_0, 3, "Expected 3 blinks in epoch 0")

        logger.info("Testing blink count for epoch 3...")
        count_epoch_3 = blink_count_epoch(self.per_epoch[3])
        logger.debug(f"Epoch 3 blink count: {count_epoch_3}")
        self.assertEqual(count_epoch_3, 0, "Expected 0 blinks in epoch 3")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
