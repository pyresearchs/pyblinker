"""Per-blink morphology feature calculations."""
from typing import Any, Dict

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_single_blink_features(blink: Dict[str, Any], sfreq: float) -> Dict[str, float]:
    """Compute morphology metrics for a single blink.

    Parameters
    ----------
    blink : dict
        Blink annotation containing ``refined_start_frame``, ``refined_peak_frame``,
        ``refined_end_frame`` and ``epoch_signal``.
    sfreq : float
        Sampling frequency in Hertz.

    Returns
    -------
    dict
        Dictionary with morphology metrics for the blink.
    """
    start = int(blink["refined_start_frame"])
    peak = int(blink["refined_peak_frame"])
    end = int(blink["refined_end_frame"])
    signal = np.asarray(blink["epoch_signal"], dtype=float)
    baseline = signal[start]
    segment = signal[start : end + 1]

    duration = (end - start) / sfreq
    t_peak = (peak - start) / sfreq
    t_end = (end - peak) / sfreq
    amplitude = baseline - np.min(segment)

    thresh25 = baseline - 0.25 * amplitude
    thresh75 = baseline - 0.75 * amplitude
    down_seg = signal[start : peak + 1]
    up_seg = signal[peak : end + 1]
    idx25 = next((i for i, v in enumerate(down_seg) if v <= thresh25), peak - start)
    idx75 = next((i for i, v in enumerate(down_seg) if v <= thresh75), peak - start)
    rise_25_75 = (idx75 - idx25) / sfreq
    idx75_up = next((i for i, v in enumerate(up_seg) if v <= thresh75), 0)
    idx25_up = next((i for i, v in enumerate(up_seg) if v <= thresh25), len(up_seg) - 1)
    fall_75_25 = (idx25_up - idx75_up) / sfreq

    thresh50 = baseline - 0.5 * amplitude
    idx_down = next((i for i, v in enumerate(down_seg) if v <= thresh50), peak - start)
    idx_up = next((i for i, v in enumerate(up_seg) if v <= thresh50), len(up_seg) - 1)
    fwhm = (idx_up + peak - start - idx_down) / sfreq

    area = float(np.trapz(baseline - segment, dx=1 / sfreq))
    cumulative = np.cumsum(baseline - segment) / sfreq
    half_area = cumulative[-1] / 2
    idx_half = next((i for i, v in enumerate(cumulative) if v >= half_area), len(cumulative) - 1)
    half_area_time = idx_half / sfreq

    seg_centered = segment - np.mean(segment)
    if seg_centered.size > 2 and np.std(seg_centered, ddof=1) != 0:
        norm = seg_centered / np.std(seg_centered, ddof=1)
        skew = float(np.mean(norm ** 3))
        kurt = float(np.mean(norm ** 4) - 3)
    else:
        skew = float("nan")
        kurt = float("nan")
    second_diff = np.diff(np.sign(np.diff(segment)))
    inflection_count = int(np.sum(second_diff != 0))

    asymmetry = t_peak / t_end if t_end != 0 else float("nan")

    logger.debug(
        "Single blink features computed: duration=%s, amplitude=%s", duration, amplitude
    )

    return {
        "duration": duration,
        "time_to_peak": t_peak,
        "time_from_peak_to_end": t_end,
        "rise_time_25_75": rise_25_75,
        "fall_time_75_25": fall_75_25,
        "fwhm": fwhm,
        "amplitude": amplitude,
        "area": area,
        "half_area_time": half_area_time,
        "asymmetry": asymmetry,
        "waveform_skewness": skew,
        "waveform_kurtosis": kurt,
        "inflection_count": inflection_count,
    }
