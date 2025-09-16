"""Deprecated shim for :mod:`pyblinker.utils.statistics_utils`."""

from __future__ import annotations

import warnings

from .statistics_utils import (
    calculate_good_ratio,
    calculate_within_range,
    get_blink_statistic,
    get_good_blink_mask,
)

warnings.warn(
    "pyblinker.utils.blink_statistics is deprecated; import from pyblinker.utils.statistics_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "calculate_within_range",
    "calculate_good_ratio",
    "get_blink_statistic",
    "get_good_blink_mask",
]
