"""Blink morphology feature calculations.

This module provides summary statistics of blink width (duration) and
height (amplitude), which are key descriptors of blink intensity.
The implementation adapts blink summary logic from the `Jena Facial
Palsy Tool <https://github.com/cvjena/JeFaPaTo>`_. These metrics are
widely used to study neuromuscular control and fatigue-related changes
in blinking and are also available in the Jena Facial Palsy Toolbox.
"""
from typing import Dict, List, Any

import logging
import numpy as np

from .per_blink import compute_single_blink_features

logger = logging.getLogger(__name__)


def _safe_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "cv": float("nan"),
            "iqr": float("nan"),
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    median = float(np.median(arr))
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    cv = float(std / mean) if mean != 0 and not np.isnan(std) else float("nan")
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": amin,
        "max": amax,
        "cv": cv,
        "iqr": iqr,
    }


def compute_morphology_features(blinks: List[Dict[str, Any]], sfreq: float) -> Dict[str, float]:
    """Compute blink morphology metrics for a single epoch.

    Parameters
    ----------
    blinks : list of dict
        Blink annotations containing ``refined_start_frame``, ``refined_peak_frame``,
        ``refined_end_frame`` and ``epoch_signal``.
    sfreq : float
        Sampling frequency of the recording in Hertz.

    Returns
    -------
    dict
        Dictionary with aggregated morphology features for the epoch.
    """
    logger.info("Computing morphology features for %d blinks", len(blinks))
    durations: List[float] = []
    ttp: List[float] = []
    tfe: List[float] = []
    rise_25_75: List[float] = []
    fall_75_25: List[float] = []
    fwhm: List[float] = []
    amplitudes: List[float] = []
    areas: List[float] = []
    half_area_times: List[float] = []
    asymmetry: List[float] = []
    wave_skews: List[float] = []
    wave_kurts: List[float] = []
    inflections: List[int] = []

    for blink in blinks:
        single = compute_single_blink_features(blink, sfreq)
        durations.append(single["duration"])
        ttp.append(single["time_to_peak"])
        tfe.append(single["time_from_peak_to_end"])
        rise_25_75.append(single["rise_time_25_75"])
        fall_75_25.append(single["fall_time_75_25"])
        fwhm.append(single["fwhm"])
        amplitudes.append(single["amplitude"])
        areas.append(single["area"])
        half_area_times.append(single["half_area_time"])
        asymmetry.append(single["asymmetry"])
        wave_skews.append(single["waveform_skewness"])
        wave_kurts.append(single["waveform_kurtosis"])
        inflections.append(single["inflection_count"])

    stats_duration = _safe_stats(durations)
    stats_ttp = _safe_stats(ttp)
    stats_tfe = _safe_stats(tfe)
    stats_rise = _safe_stats(rise_25_75)
    stats_fall = _safe_stats(fall_75_25)
    stats_fwhm = _safe_stats(fwhm)
    stats_amplitude = _safe_stats(amplitudes)
    stats_area = _safe_stats(areas)
    stats_half_area = _safe_stats(half_area_times)
    arr_asym = np.asarray(asymmetry, dtype=float)
    arr_skew = np.asarray(wave_skews, dtype=float)
    arr_kurt = np.asarray(wave_kurts, dtype=float)
    arr_inflect = np.asarray(inflections, dtype=float)

    features = {
        "blink_duration_mean": stats_duration["mean"],
        "blink_duration_std": stats_duration["std"],
        "blink_duration_median": stats_duration["median"],
        "blink_duration_min": stats_duration["min"],
        "blink_duration_max": stats_duration["max"],
        "blink_duration_cv": stats_duration["cv"],
        "blink_duration_iqr": stats_duration["iqr"],
        "blink_duration_ratio": stats_duration["max"] / stats_duration["min"] if stats_duration["min"] != 0 and not np.isnan(stats_duration["min"]) else float("nan"),
        "time_to_peak_mean": stats_ttp["mean"],
        "time_to_peak_std": stats_ttp["std"],
        "time_to_peak_cv": stats_ttp["cv"],
        "time_from_peak_to_end_mean": stats_tfe["mean"],
        "time_from_peak_to_end_std": stats_tfe["std"],
        "time_from_peak_to_end_cv": stats_tfe["cv"],
        "blink_rise_time_mean": stats_rise["mean"],
        "blink_rise_time_std": stats_rise["std"],
        "blink_rise_time_cv": stats_rise["cv"],
        "blink_fall_time_mean": stats_fall["mean"],
        "blink_fall_time_std": stats_fall["std"],
        "blink_fall_time_cv": stats_fall["cv"],
        "blink_fwhm_mean": stats_fwhm["mean"],
        "blink_fwhm_std": stats_fwhm["std"],
        "blink_fwhm_cv": stats_fwhm["cv"],
        "blink_amplitude_mean": stats_amplitude["mean"],
        "blink_amplitude_std": stats_amplitude["std"],
        "blink_amplitude_median": stats_amplitude["median"],
        "blink_amplitude_min": stats_amplitude["min"],
        "blink_amplitude_max": stats_amplitude["max"],
        "blink_amplitude_cv": stats_amplitude["cv"],
        "blink_area_mean": stats_area["mean"],
        "blink_area_std": stats_area["std"],
        "blink_area_cv": stats_area["cv"],
        "blink_half_area_time_mean": stats_half_area["mean"],
        "blink_half_area_time_std": stats_half_area["std"],
        "blink_half_area_time_cv": stats_half_area["cv"],
        "blink_asymmetry_mean": float(np.nanmean(arr_asym)) if arr_asym.size > 0 else float("nan"),
        "blink_asymmetry_std": float(np.nanstd(arr_asym, ddof=1)) if arr_asym.size > 1 else float("nan"),
        "blink_waveform_skewness_mean": float(np.nanmean(arr_skew)) if arr_skew.size > 0 else float("nan"),
        "blink_waveform_skewness_std": float(np.nanstd(arr_skew, ddof=1)) if arr_skew.size > 1 else float("nan"),
        "blink_waveform_kurtosis_mean": float(np.nanmean(arr_kurt)) if arr_kurt.size > 0 else float("nan"),
        "blink_waveform_kurtosis_std": float(np.nanstd(arr_kurt, ddof=1)) if arr_kurt.size > 1 else float("nan"),
        "blink_inflection_count_mean": float(np.nanmean(arr_inflect)) if arr_inflect.size > 0 else float("nan"),
        "blink_inflection_count_std": float(np.nanstd(arr_inflect, ddof=1)) if arr_inflect.size > 1 else float("nan"),
    }

    logger.info("Computed morphology features")
    logger.debug("Morphology feature values: %s", features)
    return features
