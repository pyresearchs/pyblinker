"""Tests for frequency-domain features computed on raw segments."""
import logging
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.utils.epochs import slice_raw_into_epochs
from pyblinker.features.frequency_domain.segment_features import compute_frequency_domain_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSegmentFrequencyFeatures(unittest.TestCase):
    """Validate spectral metrics on 30-second raw segments."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        self.segments, _, _, _ = slice_raw_into_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        self.sfreq = raw.info["sfreq"]

    def _validate_segments(self, channel: str) -> None:
        """Compute and aggregate features for all segments of a channel."""

        expected_keys = {
            "blink_rate_peak_freq",
            "blink_rate_peak_power",
            "broadband_power_0_5_2",
            "broadband_com_0_5_2",
            "high_freq_entropy_2_13",
            "one_over_f_slope",
            "band_power_ratio",
            "wavelet_energy_d1",
            "wavelet_energy_d2",
            "wavelet_energy_d3",
            "wavelet_energy_d4",
        }

        records = []
        for segment in self.segments:
            signal = segment.get_data(picks=channel)[0]
            feats = compute_frequency_domain_features([], signal, self.sfreq)
            logger.debug("%s features: %s", channel, feats)
            self.assertSetEqual(set(feats.keys()), expected_keys)
            for value in feats.values():
                self.assertFalse(np.isnan(value))
            records.append(feats)

        df = pd.DataFrame(records)
        self.assertEqual(len(df), len(self.segments))
        self.assertSetEqual(set(df.columns), expected_keys)
        self.assertFalse(df.isna().any().any())

    def test_first_segment_values(self) -> None:
        """Check a subset of feature values for the first segment."""

        signal = self.segments[0].get_data(picks="EAR-avg_ear")[0]
        feats = compute_frequency_domain_features([], signal, self.sfreq)
        logger.debug("Segment frequency features: %s", feats)
        self.assertAlmostEqual(feats["broadband_power_0_5_2"], 0.13316447, places=5)
        self.assertAlmostEqual(feats["high_freq_entropy_2_13"], 7.4352757, places=5)

    def test_all_segments_ear_channel(self) -> None:
        """Validate metrics on all segments using the EAR channel."""

        self._validate_segments("EAR-avg_ear")

    def test_all_segments_eog_channel(self) -> None:
        """Validate metrics on all segments using the EOG channel."""

        self._validate_segments("EOG-EEG-eog_vert_left")

    def test_all_segments_eeg_channel(self) -> None:
        """Validate metrics on all segments using the EEG channel."""

        self._validate_segments("EEG-E8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
