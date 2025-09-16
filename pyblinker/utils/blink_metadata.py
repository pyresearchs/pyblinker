"""Deprecated shim for :mod:`pyblinker.utils.metadata_utils`."""

from __future__ import annotations

import warnings

from .metadata_utils import (
    attach_blink_metadata,
    onset_entry_to_blinks,
    sample_windows_from_metadata,
)

warnings.warn(
    "pyblinker.utils.blink_metadata is deprecated; import from pyblinker.utils.metadata_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

_sample_windows_from_metadata = sample_windows_from_metadata

__all__ = [
    "attach_blink_metadata",
    "onset_entry_to_blinks",
    "sample_windows_from_metadata",
    "_sample_windows_from_metadata",
]
