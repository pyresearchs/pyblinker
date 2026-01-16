"""Blink energy feature calculations.

Features are computed **per channel**, and column names are suffixed with
``_<channel>`` to clearly indicate the source channel.


"""
from __future__ import annotations
from pyblinker.logging import get_logger

from typing import Dict, List, Sequence, Tuple

import mne
import pandas as pd

from .common import compute_energy_metrics
from .helpers import extract_blink_windows, segment_to_samples, _safe_stats

logger = get_logger(__name__)

_METRICS = (
    "blink_signal_energy",
    "teager_kaiser_energy",
    "blink_line_length",
    "blink_velocity_integral",
)
_STATS = ("mean", "std", "cv")


def _make_columns(ch_names: Sequence[str]) -> List[str]:
    """Generate ordered column names for all metrics and statistics."""
    columns: List[str] = []
    for ch in ch_names:
        for metric in _METRICS:
            for stat in _STATS:
                columns.append(f"{metric}_{stat}_{ch}")
    return columns


def _compute_epoch_channel_energy_stats(
    *,
    metadata_row: pd.Series,
    ch: str,
    epoch_index: int,
    signal_1d,
    sfreq: float,
    n_times: int,
) -> Dict[str, Dict[str, float]]:
    """Compute per-metric summary stats for all blinks in one epoch/channel.

    Returns
    -------
    dict
        Mapping of metric name -> stats dict (mean/std/cv). Stats are NaN if
        there are no valid blink segments.
    """
    windows: List[Tuple[float, float]] = extract_blink_windows(metadata_row, ch, epoch_index)

    energies: List[float] = []
    tkeo_vals: List[float] = []
    lengths: List[float] = []
    vel_ints: List[float] = []

    for onset_s, duration_s in windows:
        sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
        # Guard against zero/negative-length slices.
        if getattr(sl, "stop", 0) <= getattr(sl, "start", 0):
            continue
        segment = signal_1d[sl]
        if getattr(segment, "size", 0) == 0:
            continue
        metrics = compute_energy_metrics(segment, sfreq)
        energies.append(float(metrics["signal_energy"]))
        tkeo_vals.append(float(metrics["teager_kaiser_energy"]))
        lengths.append(float(metrics["line_length"]))
        vel_ints.append(float(metrics["velocity_integral"]))

    # Average the metrics over all blinks in the epoch.
    stats_energy = _safe_stats(energies)
    stats_tkeo = _safe_stats(tkeo_vals)
    stats_len = _safe_stats(lengths)
    stats_vel = _safe_stats(vel_ints)

    return {
        _METRICS[0]: stats_energy,
        _METRICS[1]: stats_tkeo,
        _METRICS[2]: stats_len,
        _METRICS[3]: stats_vel,
    }


def compute_energy_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute energy features for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with metadata containing ``blink_onset`` and
        ``blink_duration`` columns.
    picks : str | list of str | None, optional
        Channel name or list of channel names to use. If ``None``, all
        channels are processed.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` with one row per epoch and
        statistics for each metric per channel.

    Raises
    ------
    ValueError
        If any requested channels are missing from ``epochs``.

    Notes
    -----
    For epochs containing no blinks the returned statistics are ``NaN``.
    Features are computed per channel and the resulting columns are
    suffixed with ``_<channel>`` for clarity.
    """
    if picks is None:
        ch_names = epochs.ch_names
    elif isinstance(picks, str):
        ch_names = [picks]
    else:
        ch_names = list(picks)

    missing = [ch for ch in ch_names if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Channels not found: {missing}")

    sfreq = float(epochs.info["sfreq"])
    n_epochs = len(epochs)
    n_times = epochs.get_data(picks=[ch_names[0]]).shape[-1] if n_epochs else 0
    columns = _make_columns(ch_names)
    index = (
        epochs.metadata.index
        if isinstance(epochs.metadata, pd.DataFrame)
        else pd.RangeIndex(n_epochs)
    )
    if n_epochs == 0:
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    data = epochs.get_data(picks=ch_names)
    logger.info("Computing energy features for %d epochs", n_epochs)
    records: List[Dict[str, float]] = []

    for ei in range(n_epochs):
        metadata_row = (
            epochs.metadata.iloc[ei]
            if isinstance(epochs.metadata, pd.DataFrame)
            else pd.Series(dtype=float)
        )
        record: Dict[str, float] = {}
        for ci, ch in enumerate(ch_names):
            stats_by_metric = _compute_epoch_channel_energy_stats(
                metadata_row=metadata_row,
                ch=ch,
                epoch_index=ei,
                signal_1d=data[ei, ci, :],
                sfreq=sfreq,
                n_times=n_times,
            )
            for metric, stats in stats_by_metric.items():
                for stat_name, value in stats.items():
                    record[f"{metric}_{stat_name}_{ch}"] = value
        records.append(record)

    df = pd.DataFrame.from_records(records, index=index, columns=columns)
    logger.debug("Energy feature DataFrame shape: %s", df.shape)
    return df