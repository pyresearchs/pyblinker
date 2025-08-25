"""Tests for open-eye baseline features using blink metadata."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.open_eye import (
    baseline_drift_epoch,
    baseline_mad_epoch,
    baseline_mean_epoch,
    baseline_std_epoch,
    eye_opening_rms_epoch,
    micropause_count_epoch,
    perclos_epoch,
    zero_crossing_rate_epoch,
)
from pyblinker.utils import slice_raw_into_mne_epochs

from ..utils.helpers import assert_numeric_or_nan

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _blinks_from_metadata(meta: pd.Series, sfreq: float) -> list[dict[str, int]]:
    """Convert blink onset/duration metadata to frame spans."""
    onset = meta.get("blink_onset")
    duration = meta.get("blink_duration")
    blinks: list[dict[str, int]] = []
    if onset is None or (isinstance(onset, float) and pd.isna(onset)):
        return blinks
    onsets = np.atleast_1d(onset)
    durs = np.atleast_1d(duration if duration is not None else 0.0)
    if durs.size < onsets.size:
        durs = np.pad(durs, (0, onsets.size - durs.size), constant_values=durs[-1])
    for o, d in zip(onsets, durs):
        if pd.isna(o):
            continue
        start = int(float(o) * sfreq)
        end = int((float(o) + float(d or 0.0)) * sfreq)
        blinks.append({"refined_start_frame": start, "refined_end_frame": end})
    return blinks


def _compute_features(signal: np.ndarray, blinks: list[dict[str, int]], sfreq: float) -> pd.Series:
    """Compute baseline features for a single-channel epoch."""
    return pd.Series(
        {
            "baseline_mean": baseline_mean_epoch(signal, blinks),
            "baseline_drift": baseline_drift_epoch(signal, blinks, sfreq),
            "baseline_std": baseline_std_epoch(signal, blinks),
            "baseline_mad": baseline_mad_epoch(signal, blinks),
            "perclos": perclos_epoch(signal, blinks),
            "eye_opening_rms": eye_opening_rms_epoch(signal, blinks),
            "micropause_count": micropause_count_epoch(signal, blinks, sfreq),
            "zero_crossing_rate": zero_crossing_rate_epoch(signal, blinks),
        }
    )


class TestOpenEyeBaselineFeatures(unittest.TestCase):
    """Validate baseline metrics computed over open-eye epochs."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_epoch_values_respect_blinks(self) -> None:
        """Baseline values finite for blink-free epochs and numeric-or-NaN otherwise."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        sfreq = self.epochs.info["sfreq"]

        for idx in range(4):
            meta = self.epochs.metadata.iloc[idx]
            blinks = _blinks_from_metadata(meta, sfreq)
            for ch in picks:
                ch_idx = self.epochs.ch_names.index(ch)
                signal = self.epochs.get_data()[idx, ch_idx, :]
                features = _compute_features(signal, blinks, sfreq)
                if not blinks:
                    self.assertTrue(np.isfinite(features.to_numpy()).all())
                else:
                    assert_numeric_or_nan(self, features)

        for idx in [2, 3, 5, 7]:
            meta = self.epochs.metadata.iloc[idx]
            onset = meta.get("blink_onset")
            self.assertTrue(
                onset is None
                or (isinstance(onset, (list, tuple)) and len(onset) == 0)
                or pd.isna(onset)
            )
            for ch in picks:
                ch_idx = self.epochs.ch_names.index(ch)
                signal = self.epochs.get_data()[idx, ch_idx, :]
                features = _compute_features(signal, [], sfreq)
                self.assertTrue(np.isfinite(features.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
