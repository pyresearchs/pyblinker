"""Shared blink waveform analytics used by per-blink feature functions."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

from pyblinker.logging import get_logger

logger = get_logger(__name__)

#: Supported segmentation methods per modality. The EEG/EOG modality exposes
#: all historical landmark strategies, while EAR (Eye Aspect Ratio) blinks do
#: not have zero-crossings and therefore omit the ``zero`` variants.
METHODS_BY_MODALITY: Dict[str, Sequence[str]] = {
    "eeg": ("base", "zero", "tent", "half_base", "half_zero"),
    "eog": ("base", "zero", "tent", "half_base", "half_zero"),
    "ear": ("base", "tent", "half_base"),
}

#: Ordered metric stems produced by :func:`compute_blink_core`.  The concrete
#: keys are created by appending ``_{method}`` where ``method`` is one of the
#: segmentation strategies above.
CANONICAL_METRIC_STEMS: Sequence[str] = (
    "area_abs_total_trapz",
    "area_abs_total_rect",
    "symmetry_trapz",
    "symmetry_rect",
    "rise_time_peak",
    "fall_time_peak",
    "rise_time_10_90",
    "fall_time_10_90",
    "half_width",
    "vel_peak_abs",
    "vel_mean_abs",
    "slope_rise_pos",
    "slope_fall_neg",
    "acc_peak_abs",
    "acc_mean_abs",
    "amp_peak_signed",
    "amp_trough_signed",
    "amp_peak_to_trough",
    "amp_peak_abs",
)

_ALL_METHODS = tuple(
    sorted({method for methods in METHODS_BY_MODALITY.values() for method in methods})
)
_EPS = 1e-12


def _method_keys(method: str) -> Sequence[str]:
    return tuple(f"{stem}_{method}" for stem in CANONICAL_METRIC_STEMS)


def core_nan_dict(keys: Iterable[str]) -> Dict[str, float]:
    """Return a dictionary that maps ``keys`` to ``NaN`` values."""

    return {key: float("nan") for key in keys}


def _symmetry(left: float, right: float) -> float:
    denom = left + right
    if np.isnan(left) or np.isnan(right) or abs(denom) <= _EPS:
        return float("nan")
    return (left - right) / denom


def _first_index_ge(values: np.ndarray, threshold: float) -> int | None:
    matches = np.flatnonzero(values >= threshold)
    return int(matches[0]) if matches.size else None


def _first_index_le(values: np.ndarray, threshold: float) -> int | None:
    matches = np.flatnonzero(values <= threshold)
    return int(matches[0]) if matches.size else None


def compute_blink_core(
    segment: np.ndarray,
    sfreq: float,
    *,
    start_end_method: str,
    modality: str,
    include_second_derivative: bool = True,
    use_abs_for_thresholds_and_areas: bool = True,
    dx1: np.ndarray | None = None,
    dx2: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute canonical per-blink metrics for a segmented waveform.

    Parameters
    ----------
    segment
        One-dimensional signal segment spanning a single blink according to the
        requested ``start_end_method``. The array is converted to ``float`` and
        flattened internally.
    sfreq
        Sampling frequency in Hertz.
    start_end_method
        Segmentation strategy name (``"base"``, ``"zero"``, ``"tent"``,
        ``"half_base"``, or ``"half_zero"``).
    modality
        Recording modality. ``"eeg"``/``"eog"`` retain zero-crossing metrics
        whereas ``"ear"`` (Eye Aspect Ratio) omits them and returns ``NaN``.
    include_second_derivative
        If ``True`` (default) velocity and acceleration statistics are
        reported. When ``False`` the acceleration metrics are set to ``NaN``.
    use_abs_for_thresholds_and_areas
        When ``True`` the magnitude used for rise/fall thresholds and area
        calculations is based on ``abs(segment)`` for EEG/EOG data. EAR blinks
        always rely on the dip magnitude relative to their local baseline and
        ignore this flag.

    Returns
    -------
    dict
        Mapping of metric names (with method suffix) to numeric values. If the
        segmentation method is not supported for the modality or the segment is
        invalid, the returned values are ``NaN``.
    """

    method = start_end_method

    modality_key = modality.lower()
    if modality_key not in METHODS_BY_MODALITY:
        raise ValueError(f"Unsupported modality '{modality}'")

    keys = _method_keys(method)
    if method in _ALL_METHODS and method not in METHODS_BY_MODALITY[modality_key]:
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
        # EAR blink segments are dips relative to the open-eye baseline.  The
        # magnitude used for thresholding and area computations therefore
        # reflects the depth of the dip instead of absolute amplitude.
        baseline_level = float(np.max(seg))
        magnitude = np.clip(baseline_level - seg, a_min=0.0, a_max=None)
    else:
        baseline_level = 0.0
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

    velocity = np.diff(seg) * sfreq if dx1 is None else np.asarray(dx1, dtype=float).reshape(-1)
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

    metrics = {
        f"area_abs_total_trapz_{method}": area_total_trapz,
        f"area_abs_total_rect_{method}": area_total_rect,
        f"symmetry_trapz_{method}": symmetry_trapz,
        f"symmetry_rect_{method}": symmetry_rect,
        f"rise_time_peak_{method}": rise_time_peak,
        f"fall_time_peak_{method}": fall_time_peak,
        f"rise_time_10_90_{method}": rise_time_10_90,
        f"fall_time_10_90_{method}": fall_time_10_90,
        f"half_width_{method}": half_width,
        f"vel_peak_abs_{method}": vel_peak_abs,
        f"vel_mean_abs_{method}": vel_mean_abs,
        f"slope_rise_pos_{method}": slope_rise_pos,
        f"slope_fall_neg_{method}": slope_fall_neg,
        f"acc_peak_abs_{method}": acc_peak_abs,
        f"acc_mean_abs_{method}": acc_mean_abs,
        f"amp_peak_signed_{method}": amp_peak_signed,
        f"amp_trough_signed_{method}": amp_trough_signed,
        f"amp_peak_to_trough_{method}": amp_peak_to_trough,
        f"amp_peak_abs_{method}": amp_peak_abs,
    }

    return metrics


ALL_METHODS = _ALL_METHODS


__all__ = [
    "CANONICAL_METRIC_STEMS",
    "METHODS_BY_MODALITY",
    "ALL_METHODS",
    "compute_blink_core",
    "core_nan_dict",
]
