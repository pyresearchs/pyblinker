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

def _extract_blink_windows(
    metadata_row: pd.Series, channel: str, epoch_index: int
) -> List[Tuple[float, float]]:
    """Extract blink onset and duration pairs from an epoch's metadata.

    The modality (``eeg``, ``eog`` or ``ear``) is inferred from ``channel``
    and used to select modality-specific metadata columns. If those columns
    are missing or empty, the generic ``blink_onset``/``blink_duration``
    entries are used instead. When neither modality-specific nor generic
    keys are present, a :class:`ValueError` is raised.

    Parameters
    ----------
    metadata_row : pandas.Series
        A single row from ``epochs.metadata``.
    channel : str
        Channel name used to infer the modality.
    epoch_index : int
        Index of the epoch within ``epochs``. Included in error messages.

    Returns
    -------
    list of tuple of float
        List of ``(onset_seconds, duration_seconds)`` pairs. An empty list
        is returned when no blinks are present.

    Raises
    ------
    ValueError
        If neither modality-specific nor generic onset/duration metadata
        exist for the provided epoch.
    """

    ch_lower = channel.lower()
    if "ear" in ch_lower:
        mod = "ear"
    elif "eog" in ch_lower:
        mod = "eog"
    else:
        mod = "eeg"

    mod_onset_key = f"blink_onset_{mod}"
    mod_duration_key = f"blink_duration_{mod}"

    def _is_missing(val: object) -> bool:
        return val is None or (isinstance(val, float) and np.isnan(val))

    has_mod_keys = mod_onset_key in metadata_row and mod_duration_key in metadata_row
    if has_mod_keys:
        onsets = metadata_row.get(mod_onset_key)
        durations = metadata_row.get(mod_duration_key)
        if _is_missing(onsets) or _is_missing(durations):
            # fall back to generic keys if available
            onsets = metadata_row.get("blink_onset")
            durations = metadata_row.get("blink_duration")
    else:
        onsets = metadata_row.get("blink_onset")
        durations = metadata_row.get("blink_duration")

    if onsets is None or durations is None:
        if not has_mod_keys and (
            "blink_onset" not in metadata_row or "blink_duration" not in metadata_row
        ):
            raise ValueError(
                "Missing blink onset/duration metadata ('{0}', '{1}') and "
                "'blink_onset', 'blink_duration' for epoch {2}".format(
                    mod_onset_key, mod_duration_key, epoch_index
                )
            )
        return []

    if _is_missing(onsets) or _is_missing(durations):
        return []

    if not isinstance(onsets, (list, tuple, np.ndarray, pd.Series)):
        onsets = [onsets]
    if not isinstance(durations, (list, tuple, np.ndarray, pd.Series)):
        durations = [durations]

    windows: List[Tuple[float, float]] = []
    for onset, duration in zip(onsets, durations):
        if _is_missing(onset) or _is_missing(duration):
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
