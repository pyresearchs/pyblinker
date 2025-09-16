"""Deprecated shim for :mod:`pyblinker.utils.epoch_utils`."""

from __future__ import annotations

import warnings

from .epoch_utils import slice_raw_to_segments

warnings.warn(
    "pyblinker.utils.segments is deprecated; import from pyblinker.utils.epoch_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["slice_raw_to_segments"]
