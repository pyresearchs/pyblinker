"""Blink energy and complexity feature calculations."""
from typing import Any, Dict, List
import logging

from .per_blink import compute_blink_energy_complexity
from ..morphology.morphology_features import _safe_stats

logger = logging.getLogger(__name__)


def compute_energy_features(blinks: List[Dict[str, Any]], sfreq: float) -> Dict[str, float]:
    """Compute aggregated energy and complexity metrics for an epoch.

    Signal energy and line length reflect how forcefully and how far the
    eyelid travels, while Teager–Kaiser energy and velocity integral capture
    rapid oscillations and total excursion. Fatigue is often accompanied by
    lower signal energy and smoother, less variable trajectories, making these
    metrics useful markers of drowsiness in driving research.

    Parameters
    ----------
    blinks : list of dict
        Blink annotations belonging to one epoch.
    sfreq : float
        Sampling frequency in Hertz.

    Returns
    -------
    dict
        Dictionary with aggregated energy and complexity features for the epoch.
    """
    logger.info("Computing energy and complexity features for %d blinks", len(blinks))

    energies: List[float] = []
    tkeo_vals: List[float] = []
    lengths: List[float] = []
    vel_ints: List[float] = []

    for blink in blinks:
        single = compute_blink_energy_complexity(blink, sfreq)
        energies.append(single["blink_signal_energy"])
        tkeo_vals.append(single["teager_kaiser_energy"])
        lengths.append(single["blink_line_length"])
        vel_ints.append(single["blink_velocity_integral"])

    stats_energy = _safe_stats(energies)
    stats_tkeo = _safe_stats(tkeo_vals)
    stats_len = _safe_stats(lengths)
    stats_vel = _safe_stats(vel_ints)

    features = {
        "blink_signal_energy_mean": stats_energy["mean"],
        "blink_signal_energy_std": stats_energy["std"],
        "blink_signal_energy_cv": stats_energy["cv"],
        "teager_kaiser_energy_mean": stats_tkeo["mean"],
        "teager_kaiser_energy_std": stats_tkeo["std"],
        "teager_kaiser_energy_cv": stats_tkeo["cv"],
        "blink_line_length_mean": stats_len["mean"],
        "blink_line_length_std": stats_len["std"],
        "blink_line_length_cv": stats_len["cv"],
        "blink_velocity_integral_mean": stats_vel["mean"],
        "blink_velocity_integral_std": stats_vel["std"],
        "blink_velocity_integral_cv": stats_vel["cv"],
    }

    logger.debug("Energy-complexity feature values: %s", features)
    return features
