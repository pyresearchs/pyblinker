"""Deprecated shim for :mod:`pyblinker.utils.velocity_utils`."""

from __future__ import annotations

import warnings

from .velocity_utils import average_velocity

warnings.warn(
    "pyblinker.utils.velocity is deprecated; import from pyblinker.utils.velocity_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["average_velocity"]
