"""Helper utilities for blink energy features.

The functions in this module are shared across energy feature
calculations. They operate on :class:`pandas.Series` metadata rows and
NumPy arrays representing eyelid aperture signals.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_blink_windows(metadata_row: pd.Series) -> List[Tuple[float, float]]:
    """Extract blink onset and duration pairs from an epoch's metadata.

    Parameters
    ----------
    metadata_row : pandas.Series
        A single row from ``epochs.metadata`` expected to contain
        ``blink_onset`` and ``blink_duration`` entries. Values may be
        scalars or sequences. ``None``/``NaN`` entries are ignored.

    Returns
    -------
    list of tuple of float
        List of ``(onset_seconds, duration_seconds)`` pairs. An empty
        list is returned when no blinks are present.
    """
    onsets = metadata_row.get("blink_onset")
    durations = metadata_row.get("blink_duration")

    if onsets is None or durations is None:
        return []
    if isinstance(onsets, float) and np.isnan(onsets):
        return []
    if isinstance(durations, float) and np.isnan(durations):
        return []

    if not isinstance(onsets, (list, tuple, np.ndarray, pd.Series)):
        onsets = [onsets]
    if not isinstance(durations, (list, tuple, np.ndarray, pd.Series)):
        durations = [durations]

    windows: List[Tuple[float, float]] = []
    for onset, duration in zip(onsets, durations):
        if onset is None or duration is None:
            continue
        if isinstance(onset, float) and np.isnan(onset):
            continue
        if isinstance(duration, float) and np.isnan(duration):
            continue
        windows.append((float(onset), float(duration)))
    logger.debug("Extracted %d blink windows", len(windows))
    return windows


def _segment_to_samples(onset_s: float, duration_s: float, sfreq: float, n_times: int) -> slice:
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
    start = int(round(onset_s * sfreq))
    stop = start + int(round(duration_s * sfreq))
    start = max(start, 0)
    stop = min(stop, n_times)
    logger.debug("Blink window samples: start=%d stop=%d", start, stop)
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
