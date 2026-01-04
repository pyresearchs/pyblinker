"""Blink kinematic feature calculations based on epoch metadata.
All feature calculations rely only on blink onset and blink duration stored in the metadata.
This design intentionally decouples feature extraction from how blink boundaries are defined.

As a result, users should have full flexibility to define blink onset and duration according to their needs.
See pyblinker/segmentation/refinement.py
"""

from __future__ import annotations
from pyblinker.logging import get_logger

from typing import Dict, List, Sequence


import mne
import pandas as pd

from .._core_blink import CANONICAL_METRIC_STEMS
from .per_blink import compute_segment_kinematics
from ..energy.helpers import extract_blink_windows, segment_to_samples, _safe_stats
from ...utils.epoch_utils import build_metric_stat_columns, resolve_channels
from ...utils.channel_utils import normalize_picks

logger = get_logger(__name__)

# Base-method metric and statistic names (kinematics defaults to base per modality)
_METRICS = tuple(f"{stem}_base" for stem in CANONICAL_METRIC_STEMS)
_STATS = ("mean", "std", "cv")


def _infer_modality(channel_name: str, info: mne.Info) -> str:
    """Infer modality label (ear/eeg/eog) from channel metadata."""

    ch_type = info.get_channel_types(picks=[channel_name])[0]
    ch_lower = channel_name.lower()
    if "ear" in ch_lower:
        return "ear"
    if ch_type == "eog" or "eog" in ch_lower:
        return "eog"
    if ch_type == "eeg" or "eeg" in ch_lower:
        return "eeg"
    return ch_type.lower()


def compute_kinematic_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute kinematic blink features for each epoch and channel.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with metadata containing ``blink_onset`` and ``blink_duration``
        columns. Blink windows are derived directly from this metadata.
    picks : str | sequence of str | None, optional
        Channel name or list of channel names to process. ``None`` uses all
        available channels.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` containing aggregated statistics of
        kinematic metrics for each channel.

    Notes
    -----
    If an epoch contains no blinks, all kinematic statistics for that epoch
    are ``NaN``.
    """

    # Resolve channel names and infer modality per channel to avoid defaulting to EEG.
    raw_picks = resolve_channels(epochs, picks, default=lambda ep: normalize_picks(ep.ch_names))
    ch_names: List[str] = []
    modalities: List[str] = []
    for ch in raw_picks:
        ch_names.append(ch)
        modalities.append(_infer_modality(ch, epochs.info))

    sfreq = float(epochs.info["sfreq"])
    n_epochs = len(epochs)
    n_times = epochs.get_data(picks=[ch_names[0]]).shape[-1] if n_epochs else 0

    columns = build_metric_stat_columns(ch_names, _METRICS, _STATS)
    index = (
        epochs.metadata.index
        if isinstance(epochs.metadata, pd.DataFrame)
        else pd.RangeIndex(n_epochs)
    )
    if n_epochs == 0:
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    data = epochs.get_data(picks=ch_names)
    records: List[Dict[str, float]] = []
    logger.info("Computing kinematic features for %d epochs", n_epochs)

    for ei in range(n_epochs):
        metadata_row = (
            epochs.metadata.iloc[ei]
            if isinstance(epochs.metadata, pd.DataFrame)
            else pd.Series(dtype=float)
        )
        record: Dict[str, float] = {}
        for ci, ch in enumerate(ch_names):
            windows = extract_blink_windows(metadata_row, ch, ei)
            per_metric: Dict[str, List[float]] = {m: [] for m in _METRICS}
            modality = modalities[ci]
            for onset_s, duration_s in windows:
                sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
                segment = data[ei, ci, sl]
                if segment.size == 0:
                    continue
                metrics = compute_segment_kinematics(segment, sfreq, modality=modality)
                for m in _METRICS:
                    per_metric[m].append(metrics[m])
            for metric, values in per_metric.items():
                stats = _safe_stats(values)
                for stat_name, value in stats.items():
                    record[f"{metric}_{stat_name}_{ch}"] = value
        records.append(record)

    df = pd.DataFrame.from_records(records, index=index, columns=columns)
    logger.debug("Kinematic feature DataFrame shape: %s", df.shape)
    return df
