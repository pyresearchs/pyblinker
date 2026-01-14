from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "test" / "test_files"

import mne
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.evaluation import mat_data
from test.segment_config import build_segment_config


EXPECTED_COLUMNS = [
		"blink_onset",
		"blink_duration",
		"start__left_base__eeg",
		"end__right_base__eeg",
		"start__left_zero__eeg",
		"end__right_zero__eeg",
		"start__left_x_intercept__eeg",
		"end__right_x_intercept__eeg",
		"start__left_base_half_height__eeg",
		"end__right_base_half_height__eeg",
		"start__left_zero_half_height__eeg",
		"end__right_zero_half_height__eeg",
		"x_intersect__eeg",
		"y_intersect__eeg",
		]


class TestEarRefinementMetadata(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		raw_path = DATA_DIR / "ear_eog_raw.fif"
		csv_path = DATA_DIR / "ear_eog.csv"
		raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
		raw.set_annotations(mat_data.read_annotations_as_mne(csv_path))

		base_config = {
				"eeg": {
						"channel": "EEGE8",
						},
				}

		segmentation_config = build_segment_config(raw, base_config=base_config)
		cls.epochs = slice_raw_into_mne_epochs_refine_annot(
			raw,
			epoch_len=30.0,
			blink_label=None,
			segmentation_type=segmentation_config,
			progress_bar=False,
			)




	def test_metadata_matches_reference(self) -> None:
		got_metadata = self.epochs.metadata
		missing = [col for col in EXPECTED_COLUMNS if col not in got_metadata.columns]
		self.assertFalse(
			missing,
			msg=f"Missing expected EEG refinement columns: {', '.join(missing)}",
			)


if __name__ == "__main__":
	unittest.main(verbosity=2)
