"""Kinematic-only blink waveform metrics."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from pyblinker.blink_features._blink_metrics_shared import (
    ALL_METHODS,
    METHODS_BY_MODALITY,
    _method_keys,
    core_nan_dict,
    logger,
    normalize_modality,
)

KINEMATIC_METRIC_STEMS: Sequence[str] = (
    "vel_peak_abs",
    "vel_mean_abs",
    "slope_rise_pos",
    "slope_fall_neg",
    "acc_peak_abs",
    "acc_mean_abs",
)


def compute_blink_velocity(candidate_signal: np.ndarray) -> np.ndarray:
    """Compute blink velocity as the first derivative of the raw signal."""

    return np.diff(candidate_signal)


def _compute_amplitude_velocity_ratio(
    df,
    candidate_signal: np.ndarray,
    blink_velocity: np.ndarray,
    srate: float,
    *,
    start_key: str,
    end_key: str,
    ratio_key: str,
    aggregator: str = "max",
    idx_col: str | None = None,
) -> None:
    start_vals = df[start_key].to_numpy().astype(int)
    end_vals = df[end_key].to_numpy().astype(int)

    lengths = (end_vals - start_vals + 1).astype(int)
    max_len = lengths.max()
    offsets = np.arange(max_len)[None, :]
    mask = offsets < lengths[:, None]

    all_indices = start_vals[:, None] + offsets
    all_indices = all_indices[mask].astype(int)

    row_idx_all = np.repeat(np.arange(len(lengths)), lengths)
    velocities = blink_velocity[all_indices]

    temp_df = pd.DataFrame(
        {"row_idx": row_idx_all, "velocity": velocities, "index": all_indices}
    )
    if aggregator == "max":
        idx_extreme = temp_df.groupby("row_idx")["velocity"].idxmax()
    else:
        idx_extreme = temp_df.groupby("row_idx")["velocity"].idxmin()

    df_extreme = temp_df.loc[idx_extreme].sort_values("row_idx")

    ratio_vals = (
        100
        * np.abs(
            candidate_signal[df["max_blink"].to_numpy().astype(int)]
            / df_extreme["velocity"].to_numpy()
        )
        / srate
    )

    df[ratio_key] = ratio_vals
    if idx_col:
        df[idx_col] = df_extreme["index"].to_numpy()


def compute_amp_vel_ratio_zero_to_max(
    df,
    candidate_signal: np.ndarray,
    blink_velocity: np.ndarray,
    srate: float,
    *,
    modality: str,
) -> None:
    """Compute zero-to-maximum amplitude-velocity ratios."""

    if modality == "ear":
        df["pos_amp_vel_ratio_zero"] = np.nan
        df["neg_amp_vel_ratio_zero"] = np.nan
        df["peaks_pos_vel_zero"] = np.nan
        return

    _compute_amplitude_velocity_ratio(
        df,
        candidate_signal,
        blink_velocity,
        srate,
        start_key="left_zero",
        end_key="max_blink",
        ratio_key="pos_amp_vel_ratio_zero",
        aggregator="max",
        idx_col="peaks_pos_vel_zero",
    )
    _compute_amplitude_velocity_ratio(
        df,
        candidate_signal,
        blink_velocity,
        srate,
        start_key="max_blink",
        end_key="right_zero",
        ratio_key="neg_amp_vel_ratio_zero",
        aggregator="min",
    )


def compute_amp_vel_ratio_base(
    df,
    candidate_signal: np.ndarray,
    blink_velocity: np.ndarray,
    srate: float,
) -> None:
    """Compute base-to-maximum amplitude-velocity ratios."""

    _compute_amplitude_velocity_ratio(
        df,
        candidate_signal,
        blink_velocity,
        srate,
        start_key="left_base",
        end_key="max_blink",
        ratio_key="pos_amp_vel_ratio_base",
        aggregator="max",
        idx_col="peaks_pos_vel_base",
    )
    _compute_amplitude_velocity_ratio(
        df,
        candidate_signal,
        blink_velocity,
        srate,
        start_key="max_blink",
        end_key="right_base",
        ratio_key="neg_amp_vel_ratio_base",
        aggregator="min",
    )


def compute_amp_vel_ratio_tent(
    df,
    candidate_signal: np.ndarray,
    srate: float,
) -> None:
    """Compute tent-slope amplitude-velocity ratios."""

    df["neg_amp_vel_ratio_tent"] = (
        100
        * np.abs(
            candidate_signal[df["max_blink"].to_numpy().astype(int)] / df["aver_right_velocity"]
        )
        / srate
    )

    df["pos_amp_vel_ratio_tent"] = (
        100
        * np.abs(
            candidate_signal[df["max_blink"].to_numpy().astype(int)] / df["aver_left_velocity"]
        )
        / srate
    )


def compute_inter_blink_max_vel(df, srate: float, *, modality: str) -> None:
    """Compute inter-blink maximum velocity timing features."""

    df["inter_blink_max_vel_base"] = (df["peaks_pos_vel_base"] * -1) / srate
    if modality == "ear":
        df["inter_blink_max_vel_zero"] = np.nan
    else:
        df["inter_blink_max_vel_zero"] = (df["peaks_pos_vel_zero"] * -1) / srate


def compute_blink_kinematic_properties(
    df,
    candidate_signal: np.ndarray,
    srate: float,
    *,
    modality: str | None = None,
    fitted: bool = True,
) -> pd.DataFrame:
    """Compute per-blink kinematic properties used by BlinkProperties."""

    modality_key = normalize_modality(modality)
    blink_velocity = compute_blink_velocity(candidate_signal)

    compute_amp_vel_ratio_zero_to_max(
        df,
        candidate_signal,
        blink_velocity,
        srate,
        modality=modality_key,
    )
    compute_amp_vel_ratio_base(df, candidate_signal, blink_velocity, srate)
    if fitted:
        compute_amp_vel_ratio_tent(df, candidate_signal, srate)
    compute_inter_blink_max_vel(df, srate, modality=modality_key)
    return df


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


__all__ = [
    "compute_amp_vel_ratio_base",
    "compute_amp_vel_ratio_tent",
    "compute_amp_vel_ratio_zero_to_max",
    "compute_blink_kinematic_metrics",
    "compute_blink_kinematic_properties",
    "compute_blink_velocity",
    "compute_inter_blink_max_vel",
]
