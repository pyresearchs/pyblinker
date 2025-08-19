"""
Unit tests for :mod:`blink_rate` feature extraction.

This module verifies that blink rates are correctly calculated from synthetic blink data.
Specifically, it checks the computation for the first and second epochs out of a total of 5.
"""

import unittest
import logging

from pyblinker.features.blink_events.event_features.blink_rate import blink_rate_epoch
from unit_test.features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)

class TestBlinkRate(unittest.TestCase):
    """
    Unit tests for the `blink_rate_epoch` function, which computes the blink rate
    (in blinks per minute) for a given epoch's blink list and epoch duration.
    """

    def setUp(self) -> None:
        """
        Prepare mock blink data divided across 5 epochs using a test fixture.

        The mock data contains a list of blink dictionaries, each with an 'epoch_index'
        indicating which epoch the blink belongs to.
        """
        logger.info("Setting up mock data for blink rate testing...")
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.epoch_len = epoch_len
        self.per_epoch = [[] for _ in range(n_epochs)]

        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)

        logger.debug(f"Prepared blink lists per epoch: {[len(e) for e in self.per_epoch]}")

    def test_first_and_second_epoch_rates(self) -> None:
        """
        Test blink rate calculations for the first and second epochs.
        Blink rate is calculated as: (number of blinks / epoch_length_seconds) * 60
        """
        logger.info("Testing blink rate for the first epoch...")
        rate_first = blink_rate_epoch(self.per_epoch[0], self.epoch_len)
        logger.debug(f"Blink rate for first epoch: {rate_first}")
        self.assertEqual(rate_first, 18, "Expected blink rate of 18 for the first epoch")

        logger.info("Testing blink rate for the second epoch...")
        rate_second = blink_rate_epoch(self.per_epoch[1], self.epoch_len)
        logger.debug(f"Blink rate for second epoch: {rate_second}")
        self.assertEqual(rate_second, 12, "Expected blink rate of 12 for the second epoch")

    def test_fourth_epoch_zero_blinks(self) -> None:
        """
        Ensure that an epoch with no blinks returns a blink rate of 0.
        """
        logger.info("Testing blink rate for the fourth epoch (no blinks)...")
        rate_fourth = blink_rate_epoch(self.per_epoch[3], self.epoch_len)
        logger.debug(f"Blink rate for fourth epoch: {rate_fourth}")
        self.assertEqual(rate_fourth, 0, "Expected blink rate of 0 for the fourth epoch")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
