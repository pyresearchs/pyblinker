"""Deprecated shim for :mod:`pyblinker.utils.metadata_utils`."""

from __future__ import annotations

import warnings

from .metadata_utils import extract_blink_windows

warnings.warn(
    "pyblinker.utils.blink_windows is deprecated; import from pyblinker.utils.metadata_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["extract_blink_windows"]
