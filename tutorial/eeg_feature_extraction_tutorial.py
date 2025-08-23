"""Tutorial: Segmenting Raw Ear/EOG Data and Verifying Blink Counts.

This tutorial mirrors the logic in
``unit_test/utils/test_slice_raw_into_mne_epochs.py``. It demonstrates how to
slice a continuous recording into fixed-length epochs, integrate blink
annotations into each epoch's metadata, and validate the resulting per-epoch
blink counts against a ground truth file.

Flowchart:

1. **Load raw data** from ``ear_eog_raw.fif``.
2. **Segment** the continuous signal into 30 s epochs with
   ``slice_raw_into_mne_epochs``.
3. **Inspect metadata** to confirm ``blink_onset`` and ``blink_duration`` fields
   align with the original annotations.
4. **Count blinks per epoch** and compare totals with
   ``ear_eog_blink_count_epoch.csv``.

The output is an :class:`mne.Epochs` object ready for downstream analyses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import pandas as pd

from pyblinker.utils import slice_raw_into_mne_epochs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the epoch segmentation and blink verification demo."""

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
    logger.info("Created %d epochs", len(epochs))

    metadata = epochs.metadata
    assert metadata is not None

    # Inspect a sample annotation and its epoch mapping
    annotation = raw.annotations[2]
    epoch_idx = int(annotation["onset"] // 30.0)
    logger.info(
        "Annotation at %.2fs mapped to epoch %d", annotation["onset"], epoch_idx
    )
    print(metadata.loc[epoch_idx, ["blink_onset", "blink_duration"]])

    # Count blinks per epoch and validate against ground truth
    counts: list[int] = []
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

    logger.info("Blink counts validated for %d epochs", len(counts))


if __name__ == "__main__":
    main()

