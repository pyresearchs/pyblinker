"""Aggregate blink waveform-based features.

This module follows the feature definitions from the open-source
`BLINKER`_ project but provides a minimal implementation focused on the
duration and amplitude‑velocity metrics included here.

.. _BLINKER: https://github.com/VisLab/EEG-Blinks
"""
from typing import Iterable, Dict, Any, List, Sequence
import logging
import pandas as pd
import numpy as np
import mne

from ..energy.helpers import _extract_blink_windows, _segment_to_samples

from .features.duration_features import duration_base, duration_zero
from .features.amp_vel_ratio_features import neg_amp_vel_ratio_zero

logger = logging.getLogger(__name__)


def aggregate_waveform_features(
    blinks: Iterable[Dict[str, Any]],
    sfreq: float,
    n_epochs: int,
) -> pd.DataFrame:
    """Aggregate waveform metrics across epochs.

    The aggregation closely mirrors the approach used by the
    `BLINKER`_ repository but is trimmed down to only a handful of
    features in this package.

    Parameters
    ----------
    blinks : Iterable[dict]
        Blink annotations with an ``epoch_index`` field.
    sfreq : float
        Sampling frequency in Hertz.
    n_epochs : int
        Number of epochs to aggregate.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by epoch with mean waveform features.
    """
    logger.info("Aggregating waveform features over %d epochs", n_epochs)
    per_epoch: List[List[Dict[str, Any]]] = [list() for _ in range(n_epochs)]
    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch[idx].append(blink)

    records = []
    for epoch_idx, epoch_blinks in enumerate(per_epoch):
        if epoch_blinks:
            dur_base = [duration_base(b, sfreq) for b in epoch_blinks]
            dur_zero = [duration_zero(b, sfreq) for b in epoch_blinks]
            ratio_neg = [neg_amp_vel_ratio_zero(b, sfreq) for b in epoch_blinks]
            features = {
                "duration_base_mean": float(np.mean(dur_base)),
                "duration_zero_mean": float(np.mean(dur_zero)),
                "neg_amp_vel_ratio_zero_mean": float(np.nanmean(ratio_neg)),
            }
        else:
            features = {
                "duration_base_mean": float("nan"),
                "duration_zero_mean": float("nan"),
                "neg_amp_vel_ratio_zero_mean": float("nan"),
            }
        record = {"epoch": epoch_idx}
        record.update(features)
        records.append(record)

    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated waveform DataFrame shape: %s", df.shape)
    return df


def compute_epoch_waveform_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute waveform features for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with metadata containing ``blink_onset`` and
        ``blink_duration`` entries. These are treated as ground truth and
        no additional blink detection is performed.
    picks : str | sequence of str | None, optional
        Channel name or list of channel names to process. If ``None``, all
        channels are used.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` with one row per epoch and the
        mean waveform features for each channel. Base columns without
        channel suffixes mirror the first channel in ``picks`` for
        convenience.
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

    base_cols = [
        "duration_base_mean",
        "duration_zero_mean",
        "neg_amp_vel_ratio_zero_mean",
    ]
    columns = base_cols + [f"{c}_{ch}" for ch in ch_names for c in base_cols]

    index = (
        epochs.metadata.index
        if isinstance(epochs.metadata, pd.DataFrame)
        else pd.RangeIndex(n_epochs)
    )
    if n_epochs == 0:
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    data = epochs.get_data(picks=ch_names)
    records: List[Dict[str, float]] = []

    for ei in range(n_epochs):
        metadata_row = (
            epochs.metadata.iloc[ei]
            if isinstance(epochs.metadata, pd.DataFrame)
            else pd.Series(dtype=float)
        )
        windows = _extract_blink_windows(metadata_row)
        record: Dict[str, float] = {}
        per_channel: Dict[str, Dict[str, List[float]]] = {
            ch: {c: [] for c in base_cols} for ch in ch_names
        }
        for onset_s, duration_s in windows:
            sl = _segment_to_samples(onset_s, duration_s, sfreq, n_times)
            if sl.stop - sl.start <= 1:
                continue
            for ci, ch in enumerate(ch_names):
                blink = {
                    "refined_start_frame": sl.start,
                    "refined_end_frame": sl.stop - 1,
                    "epoch_signal": data[ei, ci],
                }
                per_channel[ch]["duration_base_mean"].append(
                    duration_base(blink, sfreq)
                )
                per_channel[ch]["duration_zero_mean"].append(
                    duration_zero(blink, sfreq)
                )
                per_channel[ch]["neg_amp_vel_ratio_zero_mean"].append(
                    neg_amp_vel_ratio_zero(blink, sfreq)
                )
        for ch in ch_names:
            for col in base_cols:
                arr = np.asarray(per_channel[ch][col], dtype=float)
                record[f"{col}_{ch}"] = float(np.nanmean(arr)) if arr.size else float("nan")
        for col in base_cols:
            record[col] = record[f"{col}_{ch_names[0]}"]
        records.append(record)

    df = pd.DataFrame.from_records(records, index=index, columns=columns)
    logger.debug("Epoch waveform feature DataFrame shape: %s", df.shape)
    return df
