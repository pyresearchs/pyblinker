"""Deprecated shim for :mod:`pyblinker.utils.annotation_utils`."""

from __future__ import annotations

import warnings

from .annotation_utils import create_annotation

warnings.warn(
    "pyblinker.utils.misc is deprecated; import from pyblinker.utils.annotation_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["create_annotation"]
