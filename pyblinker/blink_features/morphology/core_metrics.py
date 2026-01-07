"""Morphology-only blink waveform metrics."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from pyblinker.blink_features._blink_metrics_shared import (
    ALL_METHODS,
    METHODS_BY_MODALITY,
    _first_index_ge,
    _first_index_le,
    _method_keys,
    _symmetry,
    core_nan_dict,
    logger,
)

MORPHOLOGY_METRIC_STEMS: Sequence[str] = (
    "area_abs_total_trapz",
    "area_abs_total_rect",
    "symmetry_trapz",
    "symmetry_rect",
    "rise_time_peak",
    "fall_time_peak",
    "rise_time_10_90",
    "fall_time_10_90",
    "half_width",
    "amp_peak_signed",
    "amp_trough_signed",
    "amp_peak_to_trough",
    "amp_peak_abs",
)


def compute_blink_morphology_metrics(
    segment: np.ndarray,
    sfreq: float,
    *,
    start_end_method: str,
    modality: str,
    use_abs_for_thresholds_and_areas: bool = True,
) -> Dict[str, float]:
    """Compute morphology metrics for a segmented blink waveform."""

    method = start_end_method
    modality_key = modality.lower()
    if modality_key not in METHODS_BY_MODALITY:
        raise ValueError(f"Unsupported modality '{modality}'")

    keys = _method_keys(method, MORPHOLOGY_METRIC_STEMS)
    if method in ALL_METHODS and method not in METHODS_BY_MODALITY[modality_key]:
        return core_nan_dict(keys)

    if sfreq <= 0:
        logger.warning("Non-positive sampling frequency %s; returning NaNs", sfreq)
        return core_nan_dict(keys)

    seg = np.asarray(segment, dtype=float).reshape(-1)
    if seg.size == 0:
        logger.debug("Empty blink segment provided for method '%s'", method)
        return core_nan_dict(keys)

    dt = 1.0 / float(sfreq)

    peak_idx = int(np.argmax(seg))
    trough_idx = int(np.argmin(seg))
    amp_peak_signed = float(seg[peak_idx])
    amp_trough_signed = float(seg[trough_idx])
    amp_peak_to_trough = float(amp_peak_signed - amp_trough_signed)
    amp_peak_abs = float(abs(amp_peak_signed))

    if modality_key == "ear":
        baseline_level = float(np.max(seg))
        magnitude = np.clip(baseline_level - seg, a_min=0.0, a_max=None)
    else:
        magnitude = (
            np.abs(seg) if use_abs_for_thresholds_and_areas else np.asarray(seg)
        )

    mag_peak_idx = int(np.argmax(magnitude))
    mag_peak_value = float(magnitude[mag_peak_idx])

    area_total_trapz = float(np.trapezoid(magnitude, dx=dt))
    area_left_trapz = float(np.trapezoid(magnitude[: mag_peak_idx + 1], dx=dt))
    area_right_trapz = float(np.trapezoid(magnitude[mag_peak_idx:], dx=dt))

    area_left_rect = float(np.sum(magnitude[:mag_peak_idx]) * dt)
    area_right_rect = float(np.sum(magnitude[mag_peak_idx:]) * dt)
    area_total_rect = float(np.sum(magnitude) * dt)

    symmetry_trapz = _symmetry(area_left_trapz, area_right_trapz)
    symmetry_rect = _symmetry(area_left_rect, area_right_rect)

    rise_time_peak = mag_peak_idx / sfreq
    fall_time_peak = (seg.size - 1 - mag_peak_idx) / sfreq

    ten_level = 0.1 * mag_peak_value
    ninety_level = 0.9 * mag_peak_value

    rise_idx_10 = _first_index_ge(magnitude[: mag_peak_idx + 1], ten_level)
    rise_idx_90 = _first_index_ge(magnitude[: mag_peak_idx + 1], ninety_level)
    if rise_idx_10 is None or rise_idx_90 is None or rise_idx_90 <= rise_idx_10:
        rise_time_10_90 = float("nan")
    else:
        rise_time_10_90 = (rise_idx_90 - rise_idx_10) / sfreq

    fall_idx_90_rel = _first_index_le(magnitude[mag_peak_idx:], ninety_level)
    if fall_idx_90_rel is None:
        fall_time_10_90 = float("nan")
    else:
        fall_idx_90 = mag_peak_idx + fall_idx_90_rel
        fall_idx_10_rel = _first_index_le(magnitude[fall_idx_90:], ten_level)
        if fall_idx_10_rel is None:
            fall_time_10_90 = float("nan")
        else:
            fall_idx_10 = fall_idx_90 + fall_idx_10_rel
            if fall_idx_10 <= fall_idx_90:
                fall_time_10_90 = float("nan")
            else:
                fall_time_10_90 = (fall_idx_10 - fall_idx_90) / sfreq

    half_level = 0.5 * mag_peak_value
    left_half_idx = _first_index_ge(magnitude[: mag_peak_idx + 1], half_level)
    right_half_candidates = np.flatnonzero(
        magnitude[mag_peak_idx + 1 :] <= half_level
    )
    if left_half_idx is None or right_half_candidates.size == 0:
        half_width = float("nan")
    else:
        right_half_idx = mag_peak_idx + 1 + int(right_half_candidates[0])
        half_width = (right_half_idx - left_half_idx) / sfreq

    return {
        f"area_abs_total_trapz_{method}": area_total_trapz,
        f"area_abs_total_rect_{method}": area_total_rect,
        f"symmetry_trapz_{method}": symmetry_trapz,
        f"symmetry_rect_{method}": symmetry_rect,
        f"rise_time_peak_{method}": rise_time_peak,
        f"fall_time_peak_{method}": fall_time_peak,
        f"rise_time_10_90_{method}": rise_time_10_90,
        f"fall_time_10_90_{method}": fall_time_10_90,
        f"half_width_{method}": half_width,
        f"amp_peak_signed_{method}": amp_peak_signed,
        f"amp_trough_signed_{method}": amp_trough_signed,
        f"amp_peak_to_trough_{method}": amp_peak_to_trough,
        f"amp_peak_abs_{method}": amp_peak_abs,
    }


__all__ = ["compute_blink_morphology_metrics"]
