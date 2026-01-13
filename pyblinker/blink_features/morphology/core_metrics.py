"""Morphology-only blink waveform metrics."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from pyblinker.blink_features._blink_metrics_shared import (
    ALL_METHODS,
    METHODS_BY_MODALITY,
    _first_index_ge,
    _first_index_le,
    _method_keys,
    _symmetry,
    core_nan_dict,
    logger,
    normalize_modality,
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


def compute_blink_durations(
    df,
    srate: float,
    *,
    modality: str | None = None,
    fitted: bool = True,
) -> None:
    """Add blink durations in seconds to ``df`` using snake_case columns."""

    constant = 1  # Constant for matching Matlab output
    modality_key = normalize_modality(modality)

    df["duration_base"] = (df["right_base"] - df["left_base"]) / srate
    if {"right_zero", "left_zero"}.issubset(df.columns) and modality_key != "ear":
        df["duration_zero"] = (df["right_zero"] - df["left_zero"]) / srate
    else:
        df["duration_zero"] = np.nan

    if fitted:
        df["duration_tent"] = (df["right_x_intercept"] - df["left_x_intercept"]) / srate
        df["duration_half_base"] = (
            (df["right_base_half_height"] - df["left_base_half_height"]) + constant
        ) / srate
        if {"right_zero_half_height", "left_zero_half_height"}.issubset(df.columns) and modality_key != "ear":
            df["duration_half_zero"] = (
                (df["right_zero_half_height"] - df["left_zero_half_height"]) + constant
            ) / srate
        else:
            df["duration_half_zero"] = np.nan


def _compute_time_shut(
    row,
    data: np.ndarray,
    srate: float,
    shut_amp_fraction: float,
    *,
    key_prefix: str,
    default_no_thresh: float,
) -> float:
    col_left = f"left_{key_prefix.lower()}"
    col_right = f"right_{key_prefix.lower()}"
    left = int(row[col_left])
    right = int(row[col_right])
    threshold = shut_amp_fraction * row["max_value"]
    data_slice = data[left : right + 1]

    cond = data_slice >= threshold
    if not cond.any():
        return default_no_thresh
    start_idx = cond.argmax()

    cond = data_slice[start_idx + 1 :] < threshold
    end_idx = cond.argmax() + 1 if cond.any() else np.nan
    return end_idx / srate


def compute_time_zero_shut(
    df,
    candidate_signal: np.ndarray,
    srate: float,
    *,
    modality: str | None = None,
    shut_amp_fraction: float,
) -> None:
    """Compute zero-crossing closing/reopening and shut times."""

    modality_key = normalize_modality(modality)
    if "left_zero" not in df.columns or modality_key == "ear":
        df["closing_time_zero"] = np.nan
        df["reopening_time_zero"] = np.nan
        df["time_shut_zero"] = np.nan
        return

    df["closing_time_zero"] = (df["max_blink"] - df["left_zero"]) / srate
    df["reopening_time_zero"] = (df["right_zero"] - df["max_blink"]) / srate
    df["time_shut_zero"] = df.apply(
        lambda row: _compute_time_shut(
            row,
            candidate_signal,
            srate,
            shut_amp_fraction,
            key_prefix="Zero",
            default_no_thresh=np.nan,
        ),
        axis=1,
    )


def _compute_time_shut_tent(
    row,
    candidate_signal: np.ndarray,
    srate: float,
    shut_amp_fraction: float,
) -> float:
    left = int(row["left_x_intercept"])
    right = int(row["right_x_intercept"]) + 1
    max_val = row["max_value"]
    amp_threshold = shut_amp_fraction * max_val
    data_slice = candidate_signal[left:right]

    cond_start = data_slice >= amp_threshold
    if not cond_start.any():
        return 0

    start_idx = np.argmax(cond_start)
    cond_end = data_slice[start_idx:-1] < amp_threshold
    end_shut = np.argmax(cond_end) if cond_end.any() else np.nan
    return end_shut / srate


def compute_time_base_shut(
    df,
    candidate_signal: np.ndarray,
    srate: float,
    *,
    shut_amp_fraction: float,
    fitted: bool = True,
) -> None:
    """Compute base closing/reopening and shut times."""

    df["time_shut_base"] = df.apply(
        lambda row: _compute_time_shut(
            row,
            candidate_signal,
            srate,
            shut_amp_fraction,
            key_prefix="Base",
            default_no_thresh=0,
        ),
        axis=1,
    )

    if fitted:
        df["closing_time_tent"] = (df["x_intersect"] - df["left_x_intercept"]) / srate
        df["reopening_time_tent"] = (df["right_x_intercept"] - df["x_intersect"]) / srate
        df["time_shut_tent"] = df.apply(
            lambda row: _compute_time_shut_tent(
                row,
                candidate_signal,
                srate,
                shut_amp_fraction,
            ),
            axis=1,
        )


def compute_blink_peak_times(
    df,
    candidate_signal: np.ndarray,
    srate: float,
    *,
    fitted: bool = True,
) -> None:
    """Compute peak and inter-blink timing features."""

    df["peak_max_blink"] = df["max_value"]
    if fitted:
        df["peak_max_tent"] = df["y_intersect"]
        df["peak_time_tent"] = df["x_intersect"] / srate
    df["peak_time_blink"] = df["max_blink"] / srate

    peaks_with_len = np.append(df["max_blink"].to_numpy(), len(candidate_signal))
    df["inter_blink_max_amp"] = np.diff(peaks_with_len) / srate


def compute_blink_morphology_properties(
    df,
    candidate_signal: np.ndarray,
    srate: float,
    *,
    modality: str | None = None,
    shut_amp_fraction: float,
    fitted: bool = True,
) -> pd.DataFrame:
    """Compute per-blink morphology properties used by BlinkProperties."""

    compute_blink_durations(df, srate, modality=modality, fitted=fitted)
    compute_time_zero_shut(
        df,
        candidate_signal,
        srate,
        modality=modality,
        shut_amp_fraction=shut_amp_fraction,
    )
    compute_time_base_shut(
        df,
        candidate_signal,
        srate,
        shut_amp_fraction=shut_amp_fraction,
        fitted=fitted,
    )
    compute_blink_peak_times(df, candidate_signal, srate, fitted=fitted)
    return df


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


__all__ = [
    "compute_blink_durations",
    "compute_blink_morphology_metrics",
    "compute_blink_morphology_properties",
    "compute_blink_peak_times",
    "compute_time_base_shut",
    "compute_time_zero_shut",
]
