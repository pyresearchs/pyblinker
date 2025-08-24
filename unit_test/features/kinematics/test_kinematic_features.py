"""Kinematic feature extraction tests.

Synthetic blink waveforms are produced with ``mock_ear_generation`` for
``mne.Epochs``-based tests.  Additional tests rely on raw segments taken from
``ear_eog_raw.fif`` so that real data paths are covered as well.
"""
import unittest
import math
import logging
from pathlib import Path
import mne

from pyblinker.blink_features.kinematics.kinematic_features import compute_kinematic_features
from unit_test.features.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestKinematicFeatures(unittest.TestCase):
    """Tests for kinematic feature calculations."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.per_epoch = [[] for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)

    def test_first_epoch_features(self) -> None:
        """Verify kinematic metrics for the first epoch."""
        feats = compute_kinematic_features(self.per_epoch[0], self.sfreq)
        logger.debug(f"Kinematic features epoch 0: {feats}")
        self.assertTrue(math.isclose(feats["blink_velocity_mean"], 9.5))
        self.assertTrue(math.isclose(feats["blink_acceleration_mean"], 950.0))
        self.assertTrue(math.isclose(feats["blink_jerk_mean"], 71250.0))
        self.assertTrue(math.isclose(feats["blink_avr_mean"], 0.02))

    def test_nan_with_no_blinks(self) -> None:
        """Epoch without blinks should yield NaNs."""
        feats = compute_kinematic_features(self.per_epoch[3], self.sfreq)
        logger.debug(f"Kinematic features epoch 3: {feats}")
        self.assertTrue(math.isnan(feats["blink_velocity_mean"]))
        self.assertTrue(math.isnan(feats["blink_avr_mean"]))


class TestKinematicRealRaw(unittest.TestCase):
    """Validate kinematic metrics on a real raw segment."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.sfreq = raw.info["sfreq"]
        start, stop = 0.0, 30.0
        signal = raw.get_data(picks="EAR-avg_ear", start=int(start * self.sfreq), stop=int(stop * self.sfreq))[0]
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
                        "epoch_signal": signal,
                        "epoch_index": 0,
                    }
                )

    def test_segment_zero_means(self) -> None:
        """Check a few kinematic metrics for the first segment."""
        feats = compute_kinematic_features(self.blinks, self.sfreq)
        logger.debug("Real raw kinematic features: %s", feats)
        self.assertAlmostEqual(feats["blink_velocity_mean"], 3.90395, places=5)
        self.assertAlmostEqual(feats["blink_acceleration_mean"], 162.72143, places=5)
        self.assertAlmostEqual(feats["blink_avr_mean"], 0.0442, places=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
