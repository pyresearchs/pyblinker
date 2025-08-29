"""Integration test replicating the ``segment_features`` script.

This test illustrates how to compute both time-domain energy and
frequency-domain metrics for 30-second raw segments. It processes the
bundled ``ear_eog_raw.fif`` file and extracts features for all channels whose
names start with ``EAR``, ``EOG`` or ``EEG``. The number of blinks in each
segment is obtained via ``generate_blink_dataframe`` and included in the
result. Metrics from all segments are aggregated into a ``pandas.DataFrame``
per channel just like the original script.
"""
import logging
import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.utils.epochs import slice_raw_into_epochs
from pyblinker.blink_features.energy.segment_features import compute_time_domain_features
from pyblinker.blink_features.frequency_domain.segment_features import compute_frequency_domain_features
from pyblinker.blink_features.blink_events import generate_blink_dataframe

logger = logging.getLogger(__name__)

# Project root is two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSegmentFeaturesScript(unittest.TestCase):
    """Validate combined segment-level feature extraction."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        self.segments, _, _, _ = slice_raw_into_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        self.sfreq = raw.info["sfreq"]
        self.channels = [
            ch for ch in raw.ch_names if ch.startswith(("EAR", "EOG", "EEG"))
        ]
        blink_channel = next((ch for ch in raw.ch_names if ch.startswith("EEG")), raw.ch_names[0])
        blink_df = generate_blink_dataframe(
            self.segments, channel=blink_channel, blink_label=None, progress_bar=False
        )
        self.blink_counts = blink_df.groupby("seg_id").size().to_dict()

    def _compute_features(self, channel: str) -> pd.DataFrame:
        """Return a DataFrame with features for all segments of ``channel``."""
        records = []
        for seg_idx, segment in enumerate(self.segments):
            signal = segment.get_data(picks=channel)[0]
            time_feats = compute_time_domain_features(signal, self.sfreq)
            freq_feats = compute_frequency_domain_features([], signal, self.sfreq)
            blink_count = self.blink_counts.get(seg_idx, 0)
            record = {
                "channel": channel,
                "segment_index": seg_idx,
                "blink_count": blink_count,
            }
            record.update(time_feats)
            record.update(freq_feats)
            records.append(record)
        return pd.DataFrame(records)

    def _validate_dataframe(self, df: pd.DataFrame, n_segments: int) -> None:
        expected_cols = {
            "channel",
            "segment_index",
            "blink_count",
            "energy",
            "teager",
            "line_length",
            "velocity_integral",
            "wavelet_energy_d1",
            "wavelet_energy_d2",
            "wavelet_energy_d3",
            "wavelet_energy_d4",
        }
        self.assertEqual(len(df), n_segments)
        self.assertSetEqual(set(df.columns), expected_cols)
        self.assertFalse(df.isna().any().any())

    def test_features_all_channels(self) -> None:
        """Compute features for all EAR, EOG and EEG channels."""
        result_frames = []
        for ch in self.channels:
            df_ch = self._compute_features(ch)
            self._validate_dataframe(df_ch, len(self.segments))
            result_frames.append(df_ch)
        combined = pd.concat(result_frames, ignore_index=True)
        self.assertEqual(len(combined), len(self.segments) * len(self.channels))

    def test_first_segment_values(self) -> None:
        """Check a subset of values for the first EAR segment."""
        df = self._compute_features("EAR-avg_ear")
        row0 = df.iloc[0]
        logger.debug("First segment features: %s", row0.to_dict())
        self.assertAlmostEqual(row0["energy"], 2.608998, places=5)
        self.assertTrue(row0["wavelet_energy_d1"] >= 0 or pd.isna(row0["wavelet_energy_d1"]))
        self.assertEqual(row0["blink_count"], 2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
