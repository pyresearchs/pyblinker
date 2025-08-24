"""Frequency-domain feature calculations."""
from typing import Any, Dict, List
import logging

import numpy as np
import pywt

logger = logging.getLogger(__name__)


def compute_frequency_domain_features(
    blinks: List[Dict[str, Any]],
    epoch_signal: np.ndarray,
    sfreq: float,
) -> Dict[str, float]:
    """Compute spectral and wavelet metrics for one epoch.

    Parameters
    ----------
    blinks : list of dict
        Blink annotations belonging to the epoch.
    epoch_signal : numpy.ndarray
        Eyelid aperture samples for the epoch.
    sfreq : float
        Sampling frequency of the recording in Hertz.

    Returns
    -------
    dict
        Dictionary with frequency-domain features.
    """
    logger.info("Computing frequency-domain features for %d blinks", len(blinks))

    n = len(epoch_signal)
    if n == 0:
        return {
            "blink_rate_peak_freq": float("nan"),
            "blink_rate_peak_power": float("nan"),
            "broadband_power_0_5_2": float("nan"),
            "broadband_com_0_5_2": float("nan"),
            "high_freq_entropy_2_13": float("nan"),
            "one_over_f_slope": float("nan"),
            "band_power_ratio": float("nan"),
            "wavelet_energy_d1": float("nan"),
            "wavelet_energy_d2": float("nan"),
            "wavelet_energy_d3": float("nan"),
            "wavelet_energy_d4": float("nan"),
        }

    dt = 1.0 / sfreq
    freqs = np.fft.rfftfreq(n, dt)
    psd = np.abs(np.fft.rfft(epoch_signal)) ** 2 / n

    def band_power(fmin: float, fmax: float) -> float:
        mask = (freqs >= fmin) & (freqs <= fmax)
        return float(np.sum(psd[mask]))

    band_power_05_2 = band_power(0.5, 2.0)
    mask_band = (freqs >= 0.5) & (freqs <= 2.0)
    if np.any(mask_band) and np.sum(psd[mask_band]) > 0:
        band_com = float(
            np.sum(freqs[mask_band] * psd[mask_band]) / np.sum(psd[mask_band])
        )
    else:
        band_com = float("nan")

    mask_high = (freqs >= 2.0) & (freqs <= 13.0)
    high_power = float(np.sum(psd[mask_high]))
    if np.any(mask_high) and high_power > 0:
        probs = psd[mask_high] / high_power
        high_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    else:
        high_entropy = float("nan")

    mask_all = (freqs >= 0.5) & (freqs <= 13.0)
    if np.any(mask_all):
        slope, _ = np.polyfit(np.log(freqs[mask_all]), np.log(psd[mask_all] + 1e-12), 1)
        one_over_f = float(-slope)
    else:
        one_over_f = float("nan")

    band_ratio = band_power_05_2 / high_power if high_power > 0 else float("nan")

    coeffs = pywt.wavedec(epoch_signal, "db4", level=4)
    energies = [float(np.sum(c ** 2)) for c in coeffs[1:5]]

    indicator = np.zeros(n)
    for blink in blinks:
        idx = int(blink.get("refined_start_frame", 0))
        if 0 <= idx < n:
            indicator[idx] = 1.0
    freqs_blink = np.fft.rfftfreq(n, dt)
    blink_psd = np.abs(np.fft.rfft(indicator)) ** 2 / n
    mask_blink = (freqs_blink >= 0.1) & (freqs_blink <= 0.5)
    if np.any(mask_blink):
        idx_max = np.argmax(blink_psd[mask_blink])
        sub_freqs = freqs_blink[mask_blink]
        sub_power = blink_psd[mask_blink]
        blink_peak_freq = float(sub_freqs[idx_max])
        blink_peak_power = float(sub_power[idx_max])
    else:
        blink_peak_freq = float("nan")
        blink_peak_power = float("nan")

    features = {
        "blink_rate_peak_freq": blink_peak_freq,
        "blink_rate_peak_power": blink_peak_power,
        "broadband_power_0_5_2": band_power_05_2,
        "broadband_com_0_5_2": band_com,
        "high_freq_entropy_2_13": high_entropy,
        "one_over_f_slope": one_over_f,
        "band_power_ratio": band_ratio,
        "wavelet_energy_d1": energies[0],
        "wavelet_energy_d2": energies[1],
        "wavelet_energy_d3": energies[2],
        "wavelet_energy_d4": energies[3],
    }
    logger.debug("Frequency-domain features: %s", features)
    return features
