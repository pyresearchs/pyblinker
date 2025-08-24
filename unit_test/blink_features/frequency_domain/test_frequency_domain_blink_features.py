"""Demonstration tests for blink frequency-domain analysis on real data.
This test assume we already find the blinks location either via BLINKER (via eeg) approach,
manually (via mne annotations), or other algorithms.
This unit test slices the manually annotated ``ear_eog_raw.fif`` file into 30 s
epochs and validates how spectral blink metrics are aggregated for both EEG and
EOG channels. The examples validate the expected API behaviou.
"""
import logging
import math
import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.frequency_domain.blink.aggregate import (
    aggregate_frequency_domain_features,
)
from pyblinker.utils import slice_raw_into_mne_epochs, slice_raw_to_segments

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestBlinkFrequencyFeatures(unittest.TestCase):
    """Validate frequency-domain metrics derived from annotated blinks."""

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
        epochs = self.epochs.copy().pick(channel)
        blinks = []
        data = epochs.get_data(picks=[0]).squeeze()
        empty_epoch = None
        for idx, onset in enumerate(epochs.metadata["blink_onset"]):
            signal = data[idx]
            if onset is None:
                if empty_epoch is None:
                    empty_epoch = idx
                continue
            onset_list = onset if isinstance(onset, list) else [onset]
            for o in onset_list:
                blinks.append(
                    {
                        "epoch_index": idx,
                        "epoch_signal": signal,
                        "refined_start_frame": int(float(o) * self.sfreq),
                    }
                )
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

    def test_ear_avg_ear(self) -> None:
        """Run aggregation on averaged EAR channel."""
        self._run_channel("EAR-avg_ear")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
