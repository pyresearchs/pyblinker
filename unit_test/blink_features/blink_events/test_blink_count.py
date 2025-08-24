"""Unit tests for :mod:`blink_count` feature extraction.

Validates blink counting logic across multiple epochs using metadata.
"""

import logging
from pathlib import Path
import unittest

import mne

from pyblinker.blink_features.blink_events.event_features.blink_count import (
    blink_count_epoch,
)
from pyblinker.utils import onset_entry_to_blinks, slice_raw_into_mne_epochs

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class TestBlinkCount(unittest.TestCase):
    """
    Unit tests for the `blink_count_epoch` function, which returns the number
    of blinks present in a given epoch's blink list.
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

    def test_counts(self) -> None:
        """
        Verify blink count for known epochs:
        - Epoch 0: Contains 2 blinks.
        - Epoch 3: Contains 0 blinks.
        """
        logger.info("Testing blink count for epoch 0...")
        metadata = self.epochs.metadata
        self.assertIsNotNone(metadata, "Epoch metadata is missing")

        blinks_epoch_0 = onset_entry_to_blinks(metadata.loc[0, "blink_onset"])
        count_epoch_0 = blink_count_epoch(blinks_epoch_0)
        logger.debug("Epoch 0 blink count: %d", count_epoch_0)
        self.assertEqual(count_epoch_0, 2, "Expected 2 blinks in epoch 0")

        logger.info("Testing blink count for epoch 3...")
        blinks_epoch_3 = onset_entry_to_blinks(metadata.loc[3, "blink_onset"])
        count_epoch_3 = blink_count_epoch(blinks_epoch_3)
        logger.debug("Epoch 3 blink count: %d", count_epoch_3)
        self.assertEqual(count_epoch_3, 0, "Expected 0 blinks in epoch 3")
        logger.info("Blink count tests completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
