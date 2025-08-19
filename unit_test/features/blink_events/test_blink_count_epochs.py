"""Blink count per epoch using the real ``ear_eog_raw.fif`` dataset.

This test validates ``blink_count_epochs`` which operates on an
:meth:`mne.Epochs` object. Blink annotations are loaded from the sample
file and converted to a :class:`pandas.DataFrame` before processing.
"""
import unittest
import logging
from pathlib import Path
import pandas as pd
import mne

from pyblinker.features.blink_events.event_features.blink_count_epochs import blink_count_epochs
from unit_test.ground_truth.epoch_blink_overlay import summarize_blink_counts

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestBlinkCountEpochs(unittest.TestCase):
    """Verify epoch blink counts match the reference implementation."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unit_test" / "features" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        events = mne.make_fixed_length_events(raw, id=1, duration=30.0)
        self.epochs = mne.Epochs(
            raw,
            events,
            tmin=0.0,
            tmax=30.0 - 1.0 / raw.info["sfreq"],
            baseline=None,
            preload=False,
            verbose=False,
        )
        self.ann_df = pd.DataFrame({
            "onset": raw.annotations.onset,
            "duration": raw.annotations.duration,
            "description": raw.annotations.description,
        })
        self.ref_counts, _ = summarize_blink_counts(raw, epoch_len=30.0, blink_label=None)

    def test_first_twenty_epochs(self) -> None:
        """First twenty epoch counts should match ground truth."""
        df = blink_count_epochs(self.epochs, self.ann_df, blink_label=None)
        result = df.loc[:19, "blink_count"].reset_index(drop=True)
        expected = self.ref_counts.loc[:19, "blink_count"].reset_index(drop=True)
        pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
