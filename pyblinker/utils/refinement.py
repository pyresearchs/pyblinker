"""Deprecated shim for :mod:`pyblinker.utils.refinement_utils`."""

from __future__ import annotations

import warnings

from .refinement_utils import (
    plot_refined_blinks,
    refine_blinks_from_epochs,
    refine_ear_extrema_and_threshold_stub,
    refine_local_maximum_stub,
)

warnings.warn(
    "pyblinker.utils.refinement is deprecated; import from pyblinker.utils.refinement_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "refine_ear_extrema_and_threshold_stub",
    "refine_local_maximum_stub",
    "plot_refined_blinks",
    "refine_blinks_from_epochs",
]
