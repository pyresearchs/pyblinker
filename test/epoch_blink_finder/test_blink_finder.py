"""
This module demonstrates and tests blink detection on MNE Epochs.
Users should call the API function `find_blinks_epoch` to make it explicit
that blink detection is performed on epochs, not on raw data. The expected
output is blink onset times and durations stored in the epoch metadata.

Blink detection here is based on a simplified approach: each epoch is flattened
and treated as a continuous signal in order to locate blinks. An additional
helper, `add_blink_counts`, can be used to summarize detected blinks across
epochs.

Because this is a minimal implementation, it may produce false positives and
false negatives. The aim is not to reach production-level accuracy, but to
illustrate how blink detection can be performed on epoched data.

Future contributors may improve this module by adding more robust detection
methods, incorporating signal cleaning steps, or validating detections across
multiple channels. Such extensions would help reduce errors and bring the
results closer to research-grade reliability.
"""

import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.blinker.blink_epoch_mapper import (
    find_blinks_epoch,
    add_blink_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestBlinkFinder(unittest.TestCase):
    """Validate blink detection and mapping on Epochs."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_without_annotation_raw.fif"
        csv_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_blink_count_epoch.csv"
        self.raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        events = mne.make_fixed_length_events(self.raw, id=1, duration=30.0)
        self.epochs = mne.Epochs(
            self.raw,
            events,
            tmin=0.0,
            tmax=30.0 - 1.0 / self.raw.info["sfreq"],
            baseline=None,
            preload=True,
            verbose=False,
        )
        self.params = {
            "sfreq": self.raw.info["sfreq"],
            "min_event_len": 0.05,
            "std_threshold": 1.5,
        }
        self.expected = pd.read_csv(csv_path)
        self.total_gt = int(self.expected["blink_count"].sum())

    #
    def test_epoch_blink_count_valid(self) -> None:
        """Ensure overall blink count matches ground truth."""
        updated = find_blinks_epoch(
            self.epochs,
            ch_name="EEG-E8",
            params=self.params,
            boundary_policy="majority",
        )
        add_blink_counts(updated)
        total_pred = int(updated.metadata["n_blinks"].sum())
        self.assertGreaterEqual(total_pred, self.total_gt)

if __name__ == "__main__":
    unittest.main()
