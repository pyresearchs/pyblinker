import unittest
from pyblinker.blink_features.blink_events.event_features import aggregate_blink_event_features
from unit_test.blink_features.fixtures.mock_ear_generation import _generate_refined_ear

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
class TestAggregateBlinkFeatures(unittest.TestCase):
    """Tests for selecting blink features."""

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
    def test_default_features(self) -> None:
        """By default, all feature columns are returned."""
        df = aggregate_blink_event_features(
            self.blinks, self.sfreq, self.epoch_len, self.n_epochs
        )
        self.assertIn("blink_count", df.columns)
        self.assertIn("blink_rate", df.columns)
        self.assertIn("ibi_mean", df.columns)

    def test_select_subset(self) -> None:
        """Selecting only blink_count should omit other columns."""
        df = aggregate_blink_event_features(
            self.blinks,
            self.sfreq,
            self.epoch_len,
            self.n_epochs,
            features=["blink_count"],
        )
        self.assertEqual(list(df.columns), ["blink_count"])


if __name__ == "__main__":
    unittest.main()

