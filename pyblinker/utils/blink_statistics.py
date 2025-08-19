"""Blink statistics utilities."""

import numpy as np
import pandas as pd

from pyblinker.fitutils import mad
from pyblinker.blinker.default_setting import SCALING_FACTOR


def calculate_within_range(all_values: np.ndarray, best_median: float, best_robust_std: float) -> int:
    """Return the count of values within two robust standard deviations of ``best_median``."""
    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std
    within_mask = (all_values >= lower_bound) & (all_values <= upper_bound)
    return int(np.sum(within_mask))


def calculate_good_ratio(
    all_values: np.ndarray, best_median: float, best_robust_std: float, all_x: int
) -> float:
    """Return the fraction of ``all_values`` within two robust standard deviations of ``best_median``."""
    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std
    within_mask = (all_values >= lower_bound) & (all_values <= upper_bound)
    return float(np.sum(within_mask) / all_x)


def get_blink_statistic(df: pd.DataFrame, z_thresholds: np.ndarray, signal: np.ndarray | None = None) -> dict:
    """Compute blink statistics for a DataFrame of blink fits."""
    dfx = df.copy()
    dfx[["left_zero", "right_zero"]] = dfx[["left_zero", "right_zero"]] - 1

    indices = np.arange(len(signal))
    blink_mask = np.any(
        [(indices >= lz) & (indices <= rz) for lz, rz in zip(dfx["left_zero"], dfx["right_zero"])],
        axis=0,
    ).astype(bool)

    inside_blink = (signal > 0) & blink_mask
    outside_blink = (signal > 0) & ~blink_mask
    blink_amp_ratio = np.mean(signal[inside_blink]) / np.mean(signal[outside_blink])

    correlation_threshold_bottom, correlation_threshold_top = z_thresholds[0]
    df_data = df[["leftR2", "rightR2", "max_value"]]

    good_mask_top = (df_data["leftR2"] >= correlation_threshold_top) & (
        df_data["rightR2"] >= correlation_threshold_top
    )
    good_mask_bottom = (df_data["leftR2"] >= correlation_threshold_bottom) & (
        df_data["rightR2"] >= correlation_threshold_bottom
    )

    best_values = df_data.loc[good_mask_top, "max_value"].to_numpy()
    worst_values = df_data.loc[~good_mask_bottom, "max_value"].to_numpy()
    good_values = df_data.loc[good_mask_bottom, "max_value"].to_numpy()

    best_median = np.nanmedian(best_values)
    best_robust_std = SCALING_FACTOR * mad(best_values)
    worst_median = np.nanmedian(worst_values)
    worst_robust_std = SCALING_FACTOR * mad(worst_values)

    cutoff = (best_median * worst_robust_std + worst_median * best_robust_std) / (
        best_robust_std + worst_robust_std
    )

    all_x = calculate_within_range(df_data["max_value"].to_numpy(), best_median, best_robust_std)
    good_ratio = (
        np.nan if all_x <= 0 else calculate_good_ratio(good_values, best_median, best_robust_std, all_x)
    )

    number_good_blinks = int(np.sum(good_mask_bottom))

    return {
        "number_blinks": len(df_data),
        "number_good_blinks": number_good_blinks,
        "blink_amp_ratio": blink_amp_ratio,
        "cutoff": cutoff,
        "best_median": best_median,
        "best_robust_std": best_robust_std,
        "good_ratio": good_ratio,
    }


def get_good_blink_mask(
    blink_fits: pd.DataFrame, specified_median: float, specified_std: float, z_thresholds: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return mask of good blinks and subset DataFrame based on correlation and amplitude thresholds."""
    blink_fits = blink_fits.dropna(subset=["leftR2", "rightR2", "max_value"])

    left_r2 = blink_fits["leftR2"].to_numpy()
    right_r2 = blink_fits["rightR2"].to_numpy()
    max_value = blink_fits["max_value"].to_numpy()

    correlation_thresholds = z_thresholds[0]
    z_score_thresholds = z_thresholds[1]

    lower_bounds = np.maximum(0, specified_median - z_score_thresholds * specified_std)
    upper_bounds = specified_median + z_score_thresholds * specified_std

    left_r2 = left_r2[:, None]
    right_r2 = right_r2[:, None]
    max_value = max_value[:, None]
    correlation_thresholds = correlation_thresholds[None, :]
    lower_bounds = lower_bounds[None, :]
    upper_bounds = upper_bounds[None, :]

    masks = (
        (left_r2 >= correlation_thresholds)
        & (right_r2 >= correlation_thresholds)
        & (max_value >= lower_bounds)
        & (max_value <= upper_bounds)
    )
    good_blink_mask = np.any(masks, axis=1)
    selected_rows = blink_fits[good_blink_mask]
    return good_blink_mask, selected_rows
