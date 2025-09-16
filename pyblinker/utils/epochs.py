"""Deprecated shim for :mod:`pyblinker.utils.epoch_utils`."""

from __future__ import annotations

import warnings

from .epoch_utils import (
    slice_into_mini_raws,
    slice_raw_into_epochs,
    slice_raw_into_mne_epochs,
)
from .io_utils import save_epoch_raws
from .report_utils import generate_epoch_report

warnings.warn(
    "pyblinker.utils.epochs is deprecated; import from pyblinker.utils.epoch_utils/io_utils/report_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "slice_raw_into_mne_epochs",
    "slice_raw_into_epochs",
    "slice_into_mini_raws",
    "save_epoch_raws",
    "generate_epoch_report",
]
