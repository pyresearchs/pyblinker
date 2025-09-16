"""Deprecated shim for :mod:`pyblinker.utils.report_utils`."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import mne

from pyblinker.logging import get_logger, set_log_level

from .refinement_utils import slice_raw_into_mne_epochs_refine_annot
from .report_utils import add_blink_plots_to_report, generate_epoch_report

warnings.warn(
    "pyblinker.utils.report is deprecated; import from pyblinker.utils.report_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

logger = get_logger(__name__)


def main() -> None:
    """Build a blink validation report for the demo raw file."""

    raw_path = (
        Path(__file__).resolve().parents[2]
        / "test"
        / "test_files"
        / "ear_eog_raw.fif"
    )
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw, epoch_len=30.0, blink_label=None, progress_bar=True
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


__all__ = ["generate_epoch_report", "add_blink_plots_to_report", "main"]

if __name__ == "__main__":
    set_log_level(logging.INFO)
    main()
