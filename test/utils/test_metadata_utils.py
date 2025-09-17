"""Tests for metadata utilities."""
from __future__ import annotations

import unittest

import pandas as pd

from pyblinker.utils.metadata_utils import (
    onset_entry_to_blinks,
    sample_windows_from_metadata,
)


class TestMetadataUtils(unittest.TestCase):
    def test_onset_entry_to_blinks(self) -> None:
        self.assertEqual(onset_entry_to_blinks([0.1, 0.2]), [{"onset": 0.1}, {"onset": 0.2}])
        self.assertEqual(onset_entry_to_blinks(None), [])
        self.assertEqual(onset_entry_to_blinks(0.3), [{"onset": 0.3}])

    def test_sample_windows_from_metadata(self) -> None:
        metadata = pd.Series({"blink_onset": [0.1, 0.5], "blink_duration": [0.2, 0.1]})
        windows = sample_windows_from_metadata(metadata, "EEG-E8", sfreq=100.0, n_times=1000, epoch_index=0)
        self.assertEqual([(w.start, w.stop) for w in windows], [(10, 30), (50, 60)])

    def test_sample_windows_handles_missing(self) -> None:
        metadata = pd.Series({"blink_onset": float("nan"), "blink_duration": float("nan")})
        windows = sample_windows_from_metadata(metadata, "EEG-E8", sfreq=100.0, n_times=1000, epoch_index=1)
        self.assertEqual(windows, [])


if __name__ == "__main__":
    unittest.main()
