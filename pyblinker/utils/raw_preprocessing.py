"""Deprecated shim for :mod:`pyblinker.utils.io_utils`."""

from __future__ import annotations

import warnings

from .io_utils import prepare_refined_segments

warnings.warn(
    "pyblinker.utils.raw_preprocessing is deprecated; import from pyblinker.utils.io_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["prepare_refined_segments"]
