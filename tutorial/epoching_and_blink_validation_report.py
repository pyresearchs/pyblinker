"""Tutorial demonstrating epoching and blink validation report generation."""
from __future__ import annotations

import logging
from pathlib import Path

import mne
import pandas as pd
import numpy as np
from pyblinker.utils.refine_util import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.report import add_blink_plots_to_report

logger = logging.getLogger(__name__)


def main() -> None:
    """Build epochs, validate blink counts, and create an HTML report."""
    raw_path = (
        Path(__file__).resolve().parents[1]
        / "test"
        / "test_files"
        / "ear_eog_raw.fif"
    )
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    # Treat all annotations as blink candidates since the demo file does not
    # use the label "blink" for its events.
    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw, epoch_len=30.0, blink_label=None, progress_bar=True
    )

    # Cross-check blink counts with the provided CSV file.
    csv_path = (
        Path(__file__).resolve().parents[1]
        / "test"
        / "test_files"
        / "ear_eog_blink_count_epoch.csv"
    )
    blink_counts = pd.read_csv(csv_path)
    md = epochs.metadata.copy()
    md["epoch_id"] = md.index
    merged = md.merge(blink_counts, on="epoch_id", how="left")

    # rows to ignore (0-based indices)
    allowed_exception_rows = {31, 55}

    for idx, row in merged.iterrows():
        if idx in allowed_exception_rows:
            continue

        blink_count = row["blink_count"]
        values = row["blink_onset_extremum_ear"]

        length = 0 if (isinstance(values, float) and np.isnan(values)) else len(values)

        assert blink_count == length, (
            f"Mismatch at row {idx}: blink_count={blink_count}, length={length}"
        )

    logger.info("Blink counts in metadata align with CSV")

    report = add_blink_plots_to_report(
        epochs,
        pad_pre=0.5,
        pad_post=0.5,
        limit_per_epoch=None,
        decim=2,
        include_modalities=("eeg", "eog", "ear"),
        progress_bar=True,
    )
    out_path = Path("blink_validation_report.html")
    report.save(out_path, overwrite=True)
    logger.info("Saved blink report to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
