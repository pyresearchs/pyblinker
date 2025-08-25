"""Per-blink kinematic metrics based on metadata windows."""

from __future__ import annotations

from typing import Dict

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_segment_kinematics(segment: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Compute kinematic quantities for a single blink segment.

    Parameters
    ----------
    segment : numpy.ndarray
        1-D array containing the signal samples for the blink.
    sfreq : float
        Sampling frequency of ``segment`` in Hertz.

    Returns
    -------
    dict
        Mapping of metric name to value. ``NaN`` is returned for metrics that
        cannot be computed due to short segments or non-monotonicity.
    """

    seg = np.asarray(segment, dtype=float)
    if seg.size == 0:
        logger.warning("Empty blink segment provided. Returning NaNs.")
        return {metric: float("nan") for metric in (
            "peak_amp",
            "t2p",
            "vel_mean",
            "vel_peak",
            "acc_mean",
            "acc_peak",
            "rise_time",
            "fall_time",
            "auc",
            "symmetry",
        )}

    abs_seg = np.abs(seg)
    peak_amp = float(np.max(abs_seg))
    peak_idx = int(np.argmax(abs_seg))
    t2p = peak_idx / sfreq

    if seg.size >= 2:
        velocity = np.diff(seg) * sfreq
        abs_vel = np.abs(velocity)
        vel_mean = float(np.mean(abs_vel))
        vel_peak = float(np.max(abs_vel))
    else:
        vel_mean = float("nan")
        vel_peak = float("nan")

    if seg.size >= 3:
        acceleration = np.diff(np.diff(seg)) * (sfreq ** 2)
        abs_acc = np.abs(acceleration)
        acc_mean = float(np.mean(abs_acc))
        acc_peak = float(np.max(abs_acc))
    else:
        acc_mean = float("nan")
        acc_peak = float("nan")

    ten = 0.1 * peak_amp
    ninety = 0.9 * peak_amp

    # Rise time
    idx_10 = next((i for i in range(peak_idx + 1) if abs_seg[i] >= ten), None)
    idx_90 = next((i for i in range(peak_idx + 1) if abs_seg[i] >= ninety), None)
    if (
        idx_10 is not None
        and idx_90 is not None
        and idx_90 > idx_10
        and np.all(np.diff(abs_seg[idx_10 : idx_90 + 1]) >= 0)
    ):
        rise_time = (idx_90 - idx_10) / sfreq
    else:
        rise_time = float("nan")

    # Fall time
    idx_90_fall = None
    for i in range(peak_idx, abs_seg.size):
        if abs_seg[i] <= ninety:
            idx_90_fall = i
            break
    idx_10_fall = None
    if idx_90_fall is not None:
        for i in range(idx_90_fall, abs_seg.size):
            if abs_seg[i] <= ten:
                idx_10_fall = i
                break
    if (
        idx_90_fall is not None
        and idx_10_fall is not None
        and idx_10_fall > idx_90_fall
        and np.all(np.diff(abs_seg[idx_90_fall : idx_10_fall + 1]) <= 0)
    ):
        fall_time = (idx_10_fall - idx_90_fall) / sfreq
    else:
        fall_time = float("nan")

    auc = float(np.sum(abs_seg) / sfreq)
    area_left = float(np.sum(abs_seg[:peak_idx]) / sfreq)
    area_right = float(np.sum(abs_seg[peak_idx:]) / sfreq)
    symmetry = (area_left - area_right) / (area_left + area_right + 1e-8)

    metrics = {
        "peak_amp": peak_amp,
        "t2p": t2p,
        "vel_mean": vel_mean,
        "vel_peak": vel_peak,
        "acc_mean": acc_mean,
        "acc_peak": acc_peak,
        "rise_time": rise_time,
        "fall_time": fall_time,
        "auc": auc,
        "symmetry": symmetry,
    }

    logger.debug("Per-blink kinematics: %s", metrics)
    return metrics

