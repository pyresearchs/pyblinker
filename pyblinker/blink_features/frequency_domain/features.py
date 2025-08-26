"""Wavelet-based blink energy features.

This module provides the low level helpers for computing discrete wavelet
transform (DWT) energies of blink segments. The public API is exposed via
``aggregate_frequency_domain_features`` in :mod:`aggregate`.
"""

from __future__ import annotations

from typing import List
import logging

import numpy as np
import pywt

logger = logging.getLogger(__name__)


def _compute_wavelet_energies(
    segment: np.ndarray, sfreq: float, max_level: int = 4
) -> List[float]:
    """Compute DWT energies for a single blink segment.

    Parameters
    ----------
    segment : numpy.ndarray
        One-dimensional signal segment containing the blink waveform.
    sfreq : float
        Sampling frequency in Hertz.
    max_level : int, optional
        Highest detail level to compute. Defaults to ``4``.

    Returns
    -------
    list of float
        Energies of DWT detail coefficients ``D1`` .. ``D4``. If a level
        cannot be computed due to segment length or Nyquist constraints the
        corresponding entry is ``NaN``.
    """
    logger.debug("Computing wavelet energies: len=%d sfreq=%.2f", segment.size, sfreq)

    if segment.size == 0:
        return [float("nan")] * max_level

    max_level_available = min(pywt.dwt_max_level(segment.size, "db4"), max_level)
    coeffs = pywt.wavedec(segment, "db4", level=max_level_available)

    energies: List[float] = []
    nyquist = sfreq / 2.0
    for level in range(1, max_level + 1):
        upper_edge = sfreq / (2**level)
        if level > max_level_available or (upper_edge >= nyquist and sfreq < 30):
            logger.debug(
                "Skipping D%d: level unavailable or exceeds Nyquist (%.2f Hz)",
                level,
                nyquist,
            )
            energies.append(float("nan"))
            continue
        coeff = coeffs[level]
        energies.append(float(np.sum(coeff**2)))

    return energies

