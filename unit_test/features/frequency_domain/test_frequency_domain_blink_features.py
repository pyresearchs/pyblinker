"""Demonstration tests for blink frequency-domain analysis on real data.

Following the style of :mod:`unit_test.features.utils.test_eeg_eog_refinement`,
this module slices the recorded ``ear_eog_raw.fif`` file into 30 s epochs,
detects blinks on both EEG and EOG channels, and validates how spectral blink
metrics are aggregated. The examples illustrate the expected API behaviour
rather than production-grade accuracy. Future contributors are encouraged to
expand the tests with richer signals and additional metrics.
"""
import logging
import math
import unittest
from pathlib import Path

import mne

from pyblinker.blinker.blink_epoch_mapper import find_blinks_epoch
from pyblinker.features.frequency_domain.blink.aggregate import (
    aggregate_frequency_domain_features,
)
from pyblinker.utils import slice_raw_into_mne_epochs, slice_raw_to_segments

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestBlinkFrequencyFeatures(unittest.TestCase):
    """Validate frequency-domain metrics derived from detected blinks."""

    def setUp(self) -> None:
        raw_path = (
            PROJECT_ROOT
            / "unit_test"
            / "test_files"
            / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.sfreq = raw.info["sfreq"]
        self.epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        self.n_epochs = len(self.epochs)

    def _run_channel(self, channel: str) -> None:
        logger.info("Frequency-domain features on %s", channel)
        params = {"sfreq": self.sfreq, "min_event_len": 0.05, "std_threshold": 1.5}
        epochs = find_blinks_epoch(
            self.epochs.copy(), ch_name=channel, params=params, boundary_policy="majority"
        )
        blinks = []
        data = epochs.get_data(picks=[0]).squeeze()
        empty_epoch = None
        for idx, onsets in enumerate(epochs.metadata["blink_onsets"]):
            signal = data[idx]
            if isinstance(onsets, list) and onsets:
                for o in onsets:
                    blinks.append(
                        {
                            "epoch_index": idx,
                            "epoch_signal": signal,
                            "refined_start_frame": int(float(o) * self.sfreq),
                        }
                    )
            else:
                if empty_epoch is None:
                    empty_epoch = idx
        if empty_epoch is None:
            empty_epoch = 0
        freq_df = aggregate_frequency_domain_features(
            blinks, self.sfreq, self.n_epochs
        )
        self.assertEqual(freq_df.shape[0], self.n_epochs)
        self.assertEqual(freq_df.shape[1], 11)
        self.assertTrue(math.isnan(freq_df.loc[empty_epoch, "blink_rate_peak_freq"]))
        if blinks:
            idx = blinks[0]["epoch_index"]
            feats = freq_df.loc[idx]
            logger.debug("Freq features %s epoch %d: %s", channel, idx, feats.to_dict())
            self.assertFalse(math.isnan(feats["blink_rate_peak_freq"]))
            self.assertGreater(feats["wavelet_energy_d1"], 0.0)

    def test_eeg_e8(self) -> None:
        """Run aggregation on EEG channel."""
        self._run_channel("EEG-E8")

    def test_eog_vertical(self) -> None:
        """Run aggregation on EOG channel."""
        self._run_channel("EOG-EEG-eog_vert_left")


class TestSegmentationHelper(unittest.TestCase):
    """Tests for the raw slicing helper."""

    def setUp(self) -> None:
        raw_path = (
            PROJECT_ROOT
            / "unit_test"
            / "test_files"
            / "ear_eog_raw.fif"
        )
        self.raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs(
            self.raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_segment_count(self) -> None:
        """Ensure the helper slices a raw file into multiple segments."""
        segments = slice_raw_to_segments(self.raw, epoch_len=30.0)
        logger.debug(
            "Created %d segments via helper, %d via epoching", len(segments), len(self.epochs)
        )
        self.assertEqual(len(segments), len(self.epochs))
        for seg in segments:
            self.assertIsInstance(seg, mne.io.BaseRaw)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
