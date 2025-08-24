"""Frequency-domain metrics for 30-second segments."""
from typing import Any, Dict, List
import logging

import numpy as np

from .features import compute_frequency_domain_features as _compute_fd_features

logger = logging.getLogger(__name__)


def compute_frequency_domain_features(
    blinks: List[Dict[str, Any]],
    segment_signal: np.ndarray,
    sfreq: float,
) -> Dict[str, float]:
    """Compute spectral and wavelet metrics for one segment.

    This is a thin wrapper around
    :func:`pyblinker.features.frequency_domain.features.compute_frequency_domain_features`
    so that segment-level processing mirrors the epoch-based API.

    Parameters
    ----------
    blinks : list of dict
        Blink annotations belonging to the segment.
    segment_signal : numpy.ndarray
        Eyelid aperture samples for the segment.
    sfreq : float
        Sampling frequency of the recording in Hertz.

    Returns
    -------
    dict
        Dictionary with frequency-domain features.
    """
    logger.info("Computing frequency-domain features for %d blinks", len(blinks))
    return _compute_fd_features(blinks, segment_signal, sfreq)
