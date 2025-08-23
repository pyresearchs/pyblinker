"""Tests for segmenting raw data into MNE epochs."""
import unittest
import logging
from pathlib import Path

import mne
import pandas as pd

from pyblinker.utils import slice_raw_into_mne_epochs

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestSliceRawIntoMneEpochs(unittest.TestCase):
    """Validate epoch segmentation and annotation integration."""

    def test_annotation_mapping(self) -> None:
        """Annotations should populate onset and duration metadata."""
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

        # Choose an annotation that occurs in an epoch with a single event
        ann = raw.annotations[2]
        idx = int(ann["onset"] // epoch_len)
        logger.debug("Annotation %s belongs to epoch %d", ann, idx)
        self.assertAlmostEqual(
            metadata.loc[idx, "blink_onset"], ann["onset"] - idx * epoch_len
        )
        self.assertAlmostEqual(metadata.loc[idx, "blink_duration"], ann["duration"])

        # Ensure at least one epoch contains no annotations
        empty_idx = metadata["blink_onset"].isna().to_numpy().nonzero()[0][0]
        self.assertTrue(pd.isna(metadata.loc[empty_idx, "blink_duration"]))

    def test_blink_counts_match_ground_truth(self) -> None:
        """Blink counts derived from metadata should match the ground truth."""
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        gt_path = (
            PROJECT_ROOT
            / "unit_test"
            / "test_files"
            / "ear_eog_blink_count_epoch.csv"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        metadata = epochs.metadata
        counts = []
        for onset in metadata["blink_onset"]:
            if isinstance(onset, list):
                counts.append(len(onset))
            elif pd.isna(onset):
                counts.append(0)
            else:
                counts.append(1)
        gt_df = pd.read_csv(gt_path).iloc[: len(counts)]
        assert sum(counts) == int(gt_df["blink_count"].sum())
        for epoch_id, count in enumerate(counts):
            assert count == int(gt_df.loc[epoch_id, "blink_count"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

