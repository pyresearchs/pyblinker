"""Kinematic-only blink waveform metrics."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from pyblinker.blink_features._blink_metrics_shared import (
    ALL_METHODS,
    METHODS_BY_MODALITY,
    _method_keys,
    core_nan_dict,
    logger,
)

KINEMATIC_METRIC_STEMS: Sequence[str] = (
    "vel_peak_abs",
    "vel_mean_abs",
    "slope_rise_pos",
    "slope_fall_neg",
    "acc_peak_abs",
    "acc_mean_abs",
)


def compute_blink_kinematic_metrics(
    segment: np.ndarray,
    sfreq: float,
    *,
    start_end_method: str,
    modality: str,
    include_second_derivative: bool = True,
    dx1: np.ndarray | None = None,
    dx2: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute kinematic metrics for a segmented blink waveform."""

    method = start_end_method
    modality_key = modality.lower()
    if modality_key not in METHODS_BY_MODALITY:
        raise ValueError(f"Unsupported modality '{modality}'")

    keys = _method_keys(method, KINEMATIC_METRIC_STEMS)
    if method in ALL_METHODS and method not in METHODS_BY_MODALITY[modality_key]:
        return core_nan_dict(keys)

    if sfreq <= 0:
        logger.warning("Non-positive sampling frequency %s; returning NaNs", sfreq)
        return core_nan_dict(keys)

    seg = np.asarray(segment, dtype=float).reshape(-1)
    if seg.size == 0:
        logger.debug("Empty blink segment provided for method '%s'", method)
        return core_nan_dict(keys)

    velocity = (
        np.diff(seg) * sfreq if dx1 is None else np.asarray(dx1, dtype=float).reshape(-1)
    )
    if velocity.size == 0:
        vel_peak_abs = float("nan")
        vel_mean_abs = float("nan")
        slope_rise_pos = float("nan")
        slope_fall_neg = float("nan")
    else:
        abs_vel = np.abs(velocity)
        vel_peak_abs = float(np.max(abs_vel))
        vel_mean_abs = float(np.mean(abs_vel))
        slope_rise_pos = float(np.max(velocity))
        slope_fall_neg = float(np.min(velocity))

    if include_second_derivative:
        if dx2 is not None:
            acceleration = np.asarray(dx2, dtype=float).reshape(-1)
        elif velocity.size > 1:
            acceleration = np.diff(velocity) * sfreq
        else:
            acceleration = np.asarray([], dtype=float)

        if acceleration.size > 0:
            abs_acc = np.abs(acceleration)
            acc_peak_abs = float(np.max(abs_acc))
            acc_mean_abs = float(np.mean(abs_acc))
        else:
            acc_peak_abs = float("nan")
            acc_mean_abs = float("nan")
    else:
        acc_peak_abs = float("nan")
        acc_mean_abs = float("nan")

    return {
        f"vel_peak_abs_{method}": vel_peak_abs,
        f"vel_mean_abs_{method}": vel_mean_abs,
        f"slope_rise_pos_{method}": slope_rise_pos,
        f"slope_fall_neg_{method}": slope_fall_neg,
        f"acc_peak_abs_{method}": acc_peak_abs,
        f"acc_mean_abs_{method}": acc_mean_abs,
    }


__all__ = ["compute_blink_kinematic_metrics"]
