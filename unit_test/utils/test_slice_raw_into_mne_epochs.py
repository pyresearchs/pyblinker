"""Tests for segmenting raw data into MNE epochs."""
import unittest
import logging
from pathlib import Path

import mne

from pyblinker.utils import slice_raw_into_mne_epochs

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestSliceRawIntoMneEpochs(unittest.TestCase):
    """Validate epoch segmentation and annotation integration."""

    def test_annotation_mapping(self) -> None:
        """Annotations should be assigned to their respective epochs."""
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        epoch_len = 30.0
        epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=epoch_len, blink_label=None, progress_bar=False
        )
        self.assertIsInstance(epochs, mne.Epochs)
        expected_n_epochs = int(raw.times[-1] // epoch_len)
        self.assertEqual(len(epochs), expected_n_epochs)
        metadata = epochs.metadata
        self.assertIsNotNone(metadata)
        first_ann = raw.annotations[0]
        idx = int(first_ann["onset"] // epoch_len)
        logger.debug("First annotation %s belongs to epoch %d", first_ann, idx)
        self.assertIn(first_ann["description"], metadata.loc[idx, "annotation"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

