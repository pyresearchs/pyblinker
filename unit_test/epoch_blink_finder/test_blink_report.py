import unittest
from pathlib import Path

import mne

from pyblinker.blinker.blink_epoch_mapper import find_blinks_epoch, add_blink_counts
from pyblinker.viz import generate_blink_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestBlinkReport(unittest.TestCase):
    """Test generation of blink report."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_without_annotation_raw.fif"
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
        self.params = {
            "sfreq": self.raw.info["sfreq"],
            "min_event_len": 0.05,
            "std_threshold": 1.5,
        }

    def test_report_generation(self) -> None:
        updated = find_blinks_epoch(self.epochs, ch_name="EEG-E8", params=self.params)
        add_blink_counts(updated)
        report = generate_blink_report(updated, ch_name="EEG-E8")
        self.assertIsInstance(report, mne.report.Report)
        total_blinks = int(updated.metadata["n_blinks"].sum())
        self.assertEqual(len(report), total_blinks)


if __name__ == "__main__":
    unittest.main()
