import unittest
from pathlib import Path
import numpy as np
import mne
import pandas as pd

from pyblinker.blinker.blink_epoch_mapper import (
    find_blinks_epoch,
    add_blink_counts,
    _get_blink_position_epoching,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestBlinkFinderWithDrop(unittest.TestCase):
    """Blink detection after dropping some epochs."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_without_annotation_raw.fif"
        csv_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_blink_count_epoch.csv"

        self.raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        events = mne.make_fixed_length_events(self.raw, id=1, duration=30.0)
        self.epochs = mne.Epochs(
            self.raw,
            events,
            tmin=0.0,
            tmax=30.0 - 1.0 / self.raw.info["sfreq"],
            baseline=None,
            preload=True,
            verbose=False,
        )

        rng = np.random.RandomState(42)
        drop_indices = rng.choice(len(self.epochs), size=max(1, len(self.epochs) // 10), replace=False)
        self.epochs.drop(drop_indices)

        expected = pd.read_csv(csv_path)
        expected = expected[~expected["epoch_id"].isin(drop_indices)].reset_index(drop=True)
        self.expected = expected
        self.total_gt = int(expected["blink_count"].sum())

        self.params = {
            "sfreq": self.raw.info["sfreq"],
            "min_event_len": 0.05,
            "std_threshold": 1.5,
        }

    def test_epoch_blink_counts(self) -> None:
        """Verify predicted counts exceed or equal ground truth after drops."""
        updated = find_blinks_epoch(
            self.epochs, ch_name="EEG-E8", params=self.params, boundary_policy="majority"
        )
        add_blink_counts(updated)
        predicted = updated.metadata["n_blinks"].tolist()
        expected = self.expected["blink_count"].tolist()
        self.assertGreaterEqual(len(predicted), len(expected) - 1)
        self.assertGreaterEqual(sum(predicted), sum(expected) - 5)

    def test_total_blink_count(self) -> None:
        updated = find_blinks_epoch(
            self.epochs, ch_name="EEG-E8", params=self.params, boundary_policy="majority"
        )
        add_blink_counts(updated)
        total_pred = int(updated.metadata["n_blinks"].sum())
        self.assertGreaterEqual(total_pred, self.total_gt - 5)

    def test_blink_position_detection(self) -> None:
        signal = self.epochs.get_data(picks="EEG-E8").flatten()
        df_pos = _get_blink_position_epoching(signal, self.params, ch="EEG-E8", progress_bar=False)
        self.assertGreaterEqual(len(df_pos), self.total_gt)


if __name__ == "__main__":
    unittest.main()
