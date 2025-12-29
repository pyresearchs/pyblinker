"""EAR blink feature extraction built around threshold-refined boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from pyblinker.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EARFeatureConfig:
    """Configuration for EAR-based blink feature extraction."""

    baseline_window: float = 0.2
    classification_threshold: Optional[float] = None
    context_window: Optional[float] = None
    percentiles: Sequence[int] = field(default_factory=lambda: (5, 10, 90, 95))


def _compute_baseline(signal: np.ndarray, start: int, baseline_samples: int) -> float:
    if baseline_samples <= 0:
        return float(signal[start])
    baseline_start = max(0, start - baseline_samples)
    if baseline_start == start:
        return float(signal[start])
    return float(np.mean(signal[baseline_start:start]))


def _safe_gradient(values: np.ndarray, dt: float) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(values, dtype=float)
    return np.gradient(values, dt)


def compute_blink_features(
    signal: np.ndarray,
    sfreq: float,
    threshold: float,
    start_sample: int,
    end_sample: int,
    blink_type: Optional[str],
    feature_config: EARFeatureConfig,
) -> Dict[str, float | str | bool]:
    """Compute EAR-derived features for a single blink window."""

    start_sample = int(max(0, start_sample))
    end_sample = int(min(signal.shape[0] - 1, max(start_sample, end_sample)))
    dt = 1.0 / sfreq

    context_start = start_sample
    context_end = end_sample
    if feature_config.context_window:
        pad = int(round(feature_config.context_window * sfreq))
        context_start = max(0, start_sample - pad)
        context_end = min(signal.shape[0] - 1, end_sample + pad)

    window = signal[start_sample : end_sample + 1]
    context_window = signal[context_start : context_end + 1]

    baseline_samples = int(round(feature_config.baseline_window * sfreq))
    baseline = _compute_baseline(signal, start_sample, baseline_samples)

    mean_val = float(np.mean(context_window))
    median_val = float(np.median(context_window))
    std_val = float(np.std(context_window))
    var_val = float(np.var(context_window))
    mad_val = float(np.median(np.abs(context_window - median_val)))
    iqr_val = float(np.percentile(context_window, 75) - np.percentile(context_window, 25))
    skewness = float(
        np.mean(((context_window - mean_val) / (std_val + 1e-12)) ** 3)
    )
    kurtosis = float(
        np.mean(((context_window - mean_val) / (std_val + 1e-12)) ** 4)
    )

    percentile_values = np.percentile(context_window, feature_config.percentiles)
    percentile_dict = {
        f"ear_p{p}": float(val) for p, val in zip(feature_config.percentiles, percentile_values)
    }

    min_idx_local = int(np.argmin(window))
    min_value = float(window[min_idx_local])
    min_sample = start_sample + min_idx_local
    time_of_min = min_sample / sfreq

    velocity = _safe_gradient(window, dt)
    acceleration = _safe_gradient(velocity, dt)

    max_closing_speed = float(np.min(velocity))
    max_opening_speed = float(np.max(velocity))

    max_negative_acceleration = float(np.min(acceleration))
    max_positive_acceleration = float(np.max(acceleration))

    max_negative_slope = max_closing_speed
    max_positive_slope = max_opening_speed

    closing_velocity = velocity[: min_idx_local + 1]
    reopening_velocity = velocity[min_idx_local:]
    mean_closing_slope = float(np.mean(closing_velocity)) if closing_velocity.size else float("nan")
    mean_reopening_slope = (
        float(np.mean(reopening_velocity)) if reopening_velocity.size else float("nan")
    )

    closure_time = (min_sample - start_sample) * dt
    reopening_time = (end_sample - min_sample) * dt

    under_threshold_mask = window < threshold
    closed_duration = float(under_threshold_mask.sum() * dt)
    closed_fraction = float(np.mean(under_threshold_mask)) if window.size else float("nan")
    auc_below = float(np.sum((threshold - window[under_threshold_mask]) * dt))

    classification_threshold = (
        feature_config.classification_threshold
        if feature_config.classification_threshold is not None
        else threshold
    )
    blink_classification = "full" if min_value < classification_threshold else "partial"

    features: Dict[str, float | str | bool] = {
        "blink_type_original": blink_type,
        "ear_mean": mean_val,
        "ear_median": median_val,
        "ear_std": std_val,
        "ear_var": var_val,
        "ear_mad": mad_val,
        "ear_iqr": iqr_val,
        "ear_skewness": skewness,
        "ear_kurtosis": kurtosis,
        "ear_min": min_value,
        "ear_max": float(np.max(context_window)),
        "ear_time_of_min": time_of_min,
        "ear_baseline": baseline,
        "ear_blink_depth": float(baseline - min_value),
        "max_closing_speed": max_closing_speed,
        "max_opening_speed": max_opening_speed,
        "max_negative_slope": max_negative_slope,
        "max_positive_slope": max_positive_slope,
        "mean_closing_slope": mean_closing_slope,
        "mean_reopening_slope": mean_reopening_slope,
        "max_negative_acceleration": max_negative_acceleration,
        "max_positive_acceleration": max_positive_acceleration,
        "time_to_close": float(closure_time),
        "time_to_reopen": float(reopening_time),
        "closed_duration_seconds": closed_duration,
        "closed_fraction": closed_fraction,
        "time_under_threshold_seconds": closed_duration,
        "auc_below_threshold": auc_below,
        "classification_threshold": float(classification_threshold),
        "blink_classification": blink_classification,
    }
    features.update(percentile_dict)

    return features


class EARBlinkFeatureExtractor:
    """Compute per-blink EAR metrics given refined boundaries."""

    def __init__(
        self,
        signal: np.ndarray,
        sfreq: float,
        threshold: float,
        feature_config: Optional[EARFeatureConfig] = None,
    ):
        self.signal = np.asarray(signal, dtype=float)
        self.sfreq = float(sfreq)
        self.threshold = float(threshold)
        self.feature_config = feature_config or EARFeatureConfig()

    def build_feature_table(self, refined: pd.DataFrame) -> pd.DataFrame:
        """Attach EAR-based blink features to refined annotation rows."""

        required_cols = {"refined_start_sample", "refined_end_sample"}
        missing_cols = required_cols - set(refined.columns)
        if missing_cols:
            raise ValueError(
                f"Refined annotations are missing required columns: {sorted(missing_cols)}"
            )

        records: List[Dict[str, float | str | bool]] = []
        for row in refined.to_dict(orient="records"):
            features = compute_blink_features(
                signal=self.signal,
                sfreq=self.sfreq,
                threshold=self.threshold,
                start_sample=int(row["refined_start_sample"]),
                end_sample=int(row["refined_end_sample"]),
                blink_type=row.get("blink_type"),
                feature_config=self.feature_config,
            )
            combined = {**row, **features}
            combined["time_under_threshold_fraction"] = combined["closed_fraction"]
            combined["refined_duration"] = float(
                (combined["refined_end_sample"] - combined["refined_start_sample"]) / self.sfreq
            )
            combined["refined_duration"] = max(combined["refined_duration"], 0.0)
            records.append(combined)

        df = pd.DataFrame.from_records(records)
        logger.info("Computed EAR features for %s blinks", len(df))
        return df
