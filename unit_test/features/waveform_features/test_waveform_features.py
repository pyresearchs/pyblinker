"""Waveform feature extraction tests.

Synthetic blinks are generated with ``mock_ear_generation`` for epoch-based
tests.  Real ``mne.io.Raw`` segments from ``ear_eog_raw.fif`` are also used to
validate the aggregation functions on actual data.
"""
import unittest
import math
import logging
from pathlib import Path
import mne

from pyblinker.blink_features.waveform_features import (
    duration_base,
    duration_zero,
    neg_amp_vel_ratio_zero,
    aggregate_waveform_features,
)
from unit_test.features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestWaveformFeatures(unittest.TestCase):
    """Verify waveform-derived feature calculations."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.blinks = blinks
        self.n_epochs = n_epochs
        self.per_epoch = [[] for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)

    def test_individual_functions(self) -> None:
        """Check simple computations for a single blink."""
        blink = self.per_epoch[0][0]
        self.assertTrue(duration_base(blink, self.sfreq) > 0)
        self.assertTrue(duration_zero(blink, self.sfreq) > 0)
        ratio = neg_amp_vel_ratio_zero(blink, self.sfreq)
        self.assertFalse(math.isnan(ratio))

    def test_aggregate_shape(self) -> None:
        """Aggregated DataFrame should contain expected columns."""
        df = aggregate_waveform_features(self.blinks, self.sfreq, self.n_epochs)
        logger.debug("Waveform feature columns: %s", df.columns)
        self.assertIn("duration_base_mean", df.columns)
        self.assertEqual(len(df), self.n_epochs)


class TestWaveformRealRaw(unittest.TestCase):
    """Validate waveform aggregation on a real raw segment."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.sfreq = raw.info["sfreq"]
        start, stop = 0.0, 30.0
        self.signal = raw.get_data(picks="EAR-avg_ear", start=int(start * self.sfreq), stop=int(stop * self.sfreq))[0]
        self.blinks = []
        for onset, dur in zip(raw.annotations.onset, raw.annotations.duration):
            if onset >= start and onset + dur <= stop:
                s = int((onset - start) * self.sfreq)
                e = int((onset + dur - start) * self.sfreq)
                peak = (s + e) // 2
                self.blinks.append(
                    {
                        "refined_start_frame": s,
                        "refined_peak_frame": peak,
                        "refined_end_frame": e,
                        "epoch_signal": self.signal,
                        "epoch_index": 0,
                    }
                )

    def test_first_segment(self) -> None:
        """Waveform features from the first raw segment match expected values."""
        df = aggregate_waveform_features(self.blinks, self.sfreq, 1)
        logger.debug("Real raw waveform features: %s", df.iloc[0].to_dict())
        self.assertAlmostEqual(df.loc[0, "duration_base_mean"], 0.23)
        self.assertAlmostEqual(df.loc[0, "duration_zero_mean"], 0.17)
        self.assertAlmostEqual(df.loc[0, "neg_amp_vel_ratio_zero_mean"], 0.04419523700883515)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
