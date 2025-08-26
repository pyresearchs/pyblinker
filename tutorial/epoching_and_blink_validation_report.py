"""Tutorial demonstrating epoching and blink validation report generation."""
from __future__ import annotations

import logging
from pathlib import Path

import mne

from refine_annotation.util import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.report import add_blink_plots_to_report

logger = logging.getLogger(__name__)


def main() -> None:
    """Build epochs and create a blink validation report."""
    raw_path = (
        Path(__file__).resolve().parents[1]
        / "unit_test"
        / "test_files"
        / "ear_eog_raw.fif"
    )
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw, epoch_len=30.0, blink_label="blink", progress_bar=True
    )
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
