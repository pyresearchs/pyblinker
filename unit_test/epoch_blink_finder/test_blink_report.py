"""
This module demonstrates and tests the generation of blink detection reports
from MNE Epochs.
Users should call the API function `find_blinks_epoch` to make it explicit
that blink detection is performed on epochs, not on raw data. The expected
output is blink onset times and durations stored in the epoch metadata.

One of the key strengths of this repository is the ability to automatically
generate reports for validation. Reports serve multiple purposes:
- Providing a visual summary of detected blinks for quality control
- Allowing users to confirm that detection aligns with expectations
- Facilitating reproducibility by documenting detection parameters and results
- Making it easier to communicate findings with collaborators or in publications

Blink detection here uses a simplified approach where each epoch is flattened
and treated as a continuous signal. The helper function `add_blink_counts` can
be used to summarize detected blinks across epochs, and `generate_blink_report`
integrates results into an MNE `Report` object for convenient inspection.

Because this is a minimal implementation, it may produce false positives and
false negatives. The primary purpose is to illustrate how blink detection,
validation, and reporting can be combined in an end-to-end workflow.

Future contributors may extend the reporting capabilities by adding richer
visualizations, supporting multi-channel overlays, or integrating statistical
summaries. Such improvements would make the reports even more useful for
validation and large-scale EEG/MEG preprocessing pipelines.
"""


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
