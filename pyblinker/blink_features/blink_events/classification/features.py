"""Blink classification metrics.

This module adapts blink summary logic from the
`Jena Facial Palsy Tool <https://github.com/cvjena/JeFaPaTo>`_ to
separate partial from complete blinks based on amplitude.
Such metrics highlight irregular blink behavior often linked to
fatigue or ocular discomfort.
"""

from typing import Any, Dict, List
import logging

from ...morphology.per_blink import compute_single_blink_features

logger = logging.getLogger(__name__)


def classify_blinks_epoch(
    blinks: List[Dict[str, Any]],
    sfreq: float,
    epoch_len: float,
    threshold: float,
) -> Dict[str, float]:
    """Classify blinks within one epoch as partial or complete.

    Parameters
    ----------
    blinks : list of dict
        Blink annotations for a single epoch.
    sfreq : float
        Sampling frequency in Hertz.
    epoch_len : float
        Length of the epoch in seconds.
    threshold : float
        Amplitude threshold below which a blink is considered partial.

    Returns
    -------
    dict
        Counts and frequencies of partial and complete blinks.
    """
    partial = 0
    complete = 0
    for blink in blinks:
        feats = compute_single_blink_features(blink, sfreq)
        amp = feats["amplitude"]
        if amp < threshold:
            partial += 1
        else:
            complete += 1
    freq_partial = partial / epoch_len * 60.0
    freq_complete = complete / epoch_len * 60.0
    logger.debug(
        "Blink classification: partial=%s complete=%s", partial, complete
    )
    return {
        "Partial_Blink_Total": partial,
        "Complete_Blink_Total": complete,
        "Partial_Frequency_bpm": freq_partial,
        "Complete_Frequency_bpm": freq_complete,
    }
