"""Helper utilities for blink energy features.

The functions in this module are shared across energy feature
calculations. They operate on :class:`pandas.Series` metadata rows and
NumPy arrays representing eyelid aperture signals.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from pyblinker.logging import get_logger
from pyblinker.utils.metadata_utils import extract_blink_windows


logger = get_logger(__name__)


def segment_to_samples(onset_s: float, duration_s: float, sfreq: float, n_times: int) -> slice:
    """Convert blink onset and duration in seconds to a sample slice.

    Parameters
    ----------
    onset_s : float
        Blink onset relative to the start of the epoch in seconds.
    duration_s : float
        Blink duration in seconds.
    sfreq : float
        Sampling frequency of the epochs in Hertz.
    n_times : int
        Number of time points in the epoch.

    Returns
    -------
    slice
        Slice object representing the samples belonging to the blink. The
        slice is clamped to the valid range ``[0, n_times)``.
    """
    logger.info("Entering segment_to_samples")
    start = int(round(onset_s * sfreq))
    stop = start + int(round(duration_s * sfreq))
    start = max(start, 0)
    stop = min(stop, n_times)
    logger.debug("Blink window samples: start=%d stop=%d", start, stop)
    logger.info("Exiting segment_to_samples")
    return slice(start, stop)


def _safe_stats(values: Sequence[float]) -> Dict[str, float]:
    """Compute basic statistics while handling empty input safely.

    Parameters
    ----------
    values : sequence of float
        Values over which to compute statistics.

    Returns
    -------
    dict
        Dictionary with ``mean``, ``std``, and ``cv`` (coefficient of
        variation). ``NaN`` is returned for all values if ``values`` is
        empty or contains only ``NaN``. ``cv`` is ``NaN`` when the mean is
        zero.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {"mean": np.nan, "std": np.nan, "cv": np.nan}

    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=0))
    cv = float(std / mean) if mean != 0 else float("nan")
    return {"mean": mean, "std": std, "cv": cv}


def _tkeo(x: np.ndarray) -> np.ndarray:
    """Compute the Teager\u2013Kaiser Energy Operator of a signal."""
    x = np.asarray(x, dtype=float)
    psi = np.zeros_like(x)
    if x.size >= 3:
        psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return psi

__all__ = ["extract_blink_windows", "segment_to_samples", "_safe_stats", "_tkeo"]
