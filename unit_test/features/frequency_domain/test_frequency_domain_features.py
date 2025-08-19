"""Tests for frequency-domain feature extraction."""
import unittest
import math
import logging
from pathlib import Path
import mne

from pyblinker.features.frequency_domain.features import compute_frequency_domain_features
from pyblinker.features.frequency_domain.aggregate import aggregate_frequency_domain_features
from pyblinker.utils import slice_raw_to_segments
from unit_test.features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestFrequencyFeatures(unittest.TestCase):
    """Validate frequency-domain metrics."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.per_epoch = [[] for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)
        self.blinks = blinks
        self.n_epochs = n_epochs

    def test_single_epoch_features(self) -> None:
        """Check a couple of features for the first epoch."""
        signal = self.per_epoch[0][0]["epoch_signal"]
        feats = compute_frequency_domain_features(self.per_epoch[0], signal, self.sfreq)
        logger.debug("Freq features epoch 0: %s", feats)
        self.assertAlmostEqual(feats["blink_rate_peak_freq"], 0.3, places=1)
        self.assertTrue(feats["wavelet_energy_d1"] > 0)

    def test_aggregate_shape(self) -> None:
        """Aggregating across epochs should return expected DataFrame size."""
        df = aggregate_frequency_domain_features(self.blinks, self.sfreq, self.n_epochs)
        logger.debug("Freq feature DataFrame head: %s", df.head())
        self.assertEqual(df.shape[0], self.n_epochs)
        self.assertEqual(df.shape[1], 11)
        self.assertTrue(math.isnan(df.loc[3, "blink_rate_peak_freq"]))


class TestSegmentationHelper(unittest.TestCase):
    """Tests for the raw slicing helper."""

    def test_segment_count(self) -> None:
        """Ensure the helper slices a raw file into multiple segments."""
        raw_path = PROJECT_ROOT / "unit_test" / "features" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        segments = slice_raw_to_segments(raw, epoch_len=30.0)
        logger.debug("Created %d segments", len(segments))
        self.assertGreater(len(segments), 1)
        for seg in segments:
            self.assertIsInstance(seg, mne.io.BaseRaw)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
