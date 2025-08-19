"""Blink kinematic feature calculations."""
from typing import Any, Dict, List

import logging

from .per_blink import compute_blink_kinematics
from ..morphology.morphology_features import _safe_stats

logger = logging.getLogger(__name__)


def compute_kinematic_features(blinks: List[Dict[str, Any]], sfreq: float) -> Dict[str, float]:
    """Compute aggregated kinematic metrics for a single epoch.

    Blink kinematics describe how quickly and smoothly the eyelids move.
    Research shows that drowsy drivers exhibit slower peak velocity and
    acceleration, reduced jerk, and an elevated amplitude-velocity ratio (AVR).
    These shifts provide sensitive indicators of fatigue-related impairment.

    Parameters
    ----------
    blinks : list of dict
        Blink annotations belonging to one epoch.
    sfreq : float
        Sampling frequency of the recording in Hertz.

    Returns
    -------
    dict
        Dictionary with aggregated kinematic features for the epoch.
    """
    logger.info("Computing kinematic features for %d blinks", len(blinks))
    v_maxs: List[float] = []
    a_maxs: List[float] = []
    j_maxs: List[float] = []
    avrs: List[float] = []

    for blink in blinks:
        single = compute_blink_kinematics(blink, sfreq)
        v_maxs.append(single["v_max"])
        a_maxs.append(single["a_max"])
        j_maxs.append(single["j_max"])
        avrs.append(single["avr"])

    stats_v = _safe_stats(v_maxs)
    stats_a = _safe_stats(a_maxs)
    stats_j = _safe_stats(j_maxs)
    stats_avr = _safe_stats(avrs)

    features = {
        "blink_velocity_mean": stats_v["mean"],
        "blink_velocity_std": stats_v["std"],
        "blink_velocity_cv": stats_v["cv"],
        "blink_acceleration_mean": stats_a["mean"],
        "blink_acceleration_std": stats_a["std"],
        "blink_acceleration_cv": stats_a["cv"],
        "blink_jerk_mean": stats_j["mean"],
        "blink_jerk_std": stats_j["std"],
        "blink_jerk_cv": stats_j["cv"],
        "blink_avr_mean": stats_avr["mean"],
        "blink_avr_std": stats_avr["std"],
        "blink_avr_cv": stats_avr["cv"],
    }

    logger.debug("Kinematic feature values: %s", features)
    return features
