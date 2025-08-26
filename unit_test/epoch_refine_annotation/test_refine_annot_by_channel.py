"""Tests for slice_raw_into_mne_epochs_refine_annot."""
from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import List

import numpy as np
import mne

from refine_annotation.util import slice_raw_into_mne_epochs_refine_annot

# ---------- config / paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------- helpers ----------
def _listify(x):
    """Return a list of floats for a metadata cell that may be NaN, float, or list."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    return [x]


def _pick_ear_channels(info: mne.Info) -> List[int]:
    """Heuristic to detect EAR channels (keeps test aligned with implementation)."""
    picks = []
    for i, name in enumerate(info["ch_names"]):
        nlow = name.lower()
        if ("ear" in nlow) or ("eye_aspect_ratio" in nlow):
            picks.append(i)
    return picks



class TestFromFile(unittest.TestCase):
    def setUp(self) -> None:
        raw_path = (
                PROJECT_ROOT
                / "unit_test"
                / "test_files"
                / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epoch_len = 30.0
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=self.epoch_len, blink_label=None, progress_bar=False
        )
        self.sfreq = float(self.raw.info["sfreq"])

    def test_epoch_count_matches_duration(self):
        # Expected number of fixed-length epochs
        # mne.make_fixed_length_events starts at 0 and places events every epoch_len
        # while the last epoch must fully fit inside the recording.
        expected = int(np.floor(self.raw.times[-1] / self.epoch_len))
        self.assertEqual(len(self.epochs), expected)

    def test_metadata_columns_presence_by_modalities(self):
        md = self.epochs.metadata
        # Always present:
        for col in ("blink_onset", "blink_duration", "n_blinks"):
            self.assertIn(col, md.columns)

        have_eeg = mne.pick_types(self.epochs.info, eeg=True, eog=False).size > 0
        have_eog = mne.pick_types(self.epochs.info, eeg=False, eog=True).size > 0
        have_ear = len(_pick_ear_channels(self.epochs.info)) > 0

        if have_eeg:
            for col in ("blink_onset_eeg", "blink_duration_eeg", "blink_onset_extremum_eeg"):
                self.assertIn(col, md.columns)
        if have_eog:
            for col in ("blink_onset_eog", "blink_duration_eog", "blink_onset_extremum_eog"):
                self.assertIn(col, md.columns)
        if have_ear:
            for col in ("blink_onset_ear", "blink_duration_ear", "blink_onset_extremum_ear"):
                self.assertIn(col, md.columns)

    def test_value_ranges_and_extremum_logic(self):
        md = self.epochs.metadata
        eps = 1.0 / self.sfreq + 1e-12

        for row in md.itertuples(index=False):
            # Manual fields
            onsets = _listify(row.blink_onset)
            durs = _listify(row.blink_duration)
            self.assertEqual(len(onsets), len(durs))
            for o, d in zip(onsets, durs):
                self.assertGreaterEqual(o, -eps)
                self.assertLessEqual(o, self.epoch_len + eps)
                self.assertGreaterEqual(d, -eps)
                self.assertLessEqual(o + d, self.epoch_len + eps)

            # n_blinks must match manual count
            manual_count = len(onsets)
            self.assertEqual(manual_count, row.n_blinks)

            # Per-modality checks (if columns exist)
            def _check_modality(prefix: str, extremum_name: str, is_trough: bool | None):
                if f"{prefix}_onset" not in md.columns:
                    return
                onset_vals = _listify(getattr(row, f"{prefix}_onset"))
                dur_vals = _listify(getattr(row, f"{prefix}_duration"))
                ext_vals = _listify(getattr(row, extremum_name))
                # Lengths must align with number of blinks in epoch
                if manual_count == 0:
                    self.assertEqual(onset_vals, [])
                    self.assertEqual(dur_vals, [])
                    self.assertEqual(ext_vals, [])
                    return
                self.assertEqual(len(onset_vals), manual_count)
                self.assertEqual(len(dur_vals), manual_count)
                self.assertEqual(len(ext_vals), manual_count)
                for o, d, e in zip(onset_vals, dur_vals, ext_vals):
                    self.assertGreaterEqual(o, -eps)
                    self.assertLessEqual(o, self.epoch_len + eps)
                    self.assertGreaterEqual(d, -eps)
                    self.assertLessEqual(o + d, self.epoch_len + eps)
                    # Extremum should lie inside the refined interval
                    self.assertGreaterEqual(e, o - eps)
                    self.assertLessEqual(e, o + d + eps)

            if "blink_onset_eeg" in md.columns:
                _check_modality("blink_onset_eeg".rsplit("_", 1)[0], "blink_onset_extremum_eeg", is_trough=False)
            if "blink_onset_eog" in md.columns:
                _check_modality("blink_onset_eog".rsplit("_", 1)[0], "blink_onset_extremum_eog", is_trough=False)
            if "blink_onset_ear" in md.columns:
                _check_modality("blink_onset_ear".rsplit("_", 1)[0], "blink_onset_extremum_ear", is_trough=True)




if __name__ == "__main__":
    unittest.main(verbosity=2)
