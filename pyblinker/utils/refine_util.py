"""Deprecated shim for :mod:`pyblinker.utils.refinement_utils`."""

from __future__ import annotations

import warnings

from .refinement_utils import slice_raw_into_mne_epochs_refine_annot

warnings.warn(
    "pyblinker.utils.refine_util is deprecated; import from pyblinker.utils.refinement_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["slice_raw_into_mne_epochs_refine_annot"]
