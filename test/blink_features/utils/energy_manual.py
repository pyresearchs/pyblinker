"""Manual energy computations for tests."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from pyblinker.blink_features.energy.common import compute_energy_metrics
from pyblinker.blink_features.energy.helpers import (
    extract_blink_windows,
    segment_to_samples,
    _safe_stats,
)


def manual_epoch_energy_features(
    epoch_data: np.ndarray,
    metadata_row: pd.Series,
    sfreq: float,
    ch: str,
    epoch_index: int,
) -> Dict[str, float]:
    """Compute manual energy metrics for a single epoch.

    Parameters
    ----------
    epoch_data : np.ndarray
        Samples for one channel within a single epoch.
    metadata_row : pandas.Series
        Epoch metadata containing ``blink_onset`` and ``blink_duration``.
    sfreq : float
        Sampling frequency of the signal.
    ch : str
        Channel name appended to column suffixes.
    epoch_index : int
        Index of the epoch within the original ``Epochs`` object.

    Returns
    -------
    Dict[str, float]
        Mapping of ``f"{metric}_{stat}_{ch}"`` to computed values.
    """
    n_times = epoch_data.size
    windows = extract_blink_windows(metadata_row, ch, epoch_index)
    energies: list[float] = []
    tkeo_vals: list[float] = []
    lengths: list[float] = []
    vel_ints: list[float] = []
    for onset_s, duration_s in windows:
        sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
        segment = epoch_data[sl]
        if segment.size == 0:
            continue
        metrics = compute_energy_metrics(segment, sfreq)
        energies.append(float(metrics["signal_energy"]))
        tkeo_vals.append(float(metrics["teager_kaiser_energy"]))
        lengths.append(float(metrics["line_length"]))
        vel_ints.append(float(metrics["velocity_integral"]))
    stats_energy = _safe_stats(energies)
    stats_tkeo = _safe_stats(tkeo_vals)
    stats_len = _safe_stats(lengths)
    stats_vel = _safe_stats(vel_ints)
    record: Dict[str, float] = {}
    for metric, stats in zip(
        [
            "blink_signal_energy",
            "teager_kaiser_energy",
            "blink_line_length",
            "blink_velocity_integral",
        ],
        [stats_energy, stats_tkeo, stats_len, stats_vel],
    ):
        for stat_name, value in stats.items():
            record[f"{metric}_{stat_name}_{ch}"] = value
    return record
