"""Aggregate blink morphology features from :class:`mne.Epochs`."""
from __future__ import annotations
from pyblinker.logging import get_logger

from typing import Dict, List, Sequence

import mne
import numpy as np
import pandas as pd

from .per_blink import compute_blink_waveform_metrics
from ..energy.helpers import extract_blink_windows, segment_to_samples, _safe_stats
from ...utils.epoch_utils import build_metric_stat_columns, resolve_channels
from ...utils.modality import infer_modality

logger = get_logger(__name__)

_BASE_METRICS = tuple(compute_blink_waveform_metrics(np.zeros(3), 1.0).keys())

# Derive metric and statistic names instead of hardcoding
_METRICS = _BASE_METRICS + ("duration",)
_STATS = tuple(_safe_stats([]).keys())


def _default_morphology_channels(epochs: mne.Epochs) -> List[str]:
    """Select default EAR/EOG channels when ``picks`` are unspecified."""

    ch_names = [
        ch for ch in epochs.ch_names if "EOG" in ch.upper() or "EAR" in ch.upper()
    ]
    if not ch_names:
        raise ValueError("No default EAR/EOG channels found")
    return ch_names


def compute_epoch_morphology_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute blink morphology statistics for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch object whose ``metadata`` must contain ``blink_onset`` and
        ``blink_duration`` columns.
    picks : str | list of str | None, optional
        Channel name(s) to include. ``None`` selects channels containing
        ``"EOG"`` or ``"EAR"``. If any requested channel is missing a
        :class:`ValueError` is raised.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` containing ``mean``, ``std``, and
        ``cv`` aggregates for each morphology metric per channel.

    Raises
    ------
    ValueError
        If required metadata columns are absent or ``picks`` contain unknown
        channels.
    """
    logger.info("Computing morphology features for epochs")

    if epochs.metadata is None:
        raise ValueError("epochs.metadata must be provided")

    ch_names = resolve_channels(epochs, picks, default=_default_morphology_channels)

    data = epochs.get_data(picks=ch_names)
    sfreq = float(epochs.info["sfreq"])
    n_epochs, n_ch, n_times = data.shape
    index = epochs.metadata.index

    columns = build_metric_stat_columns(ch_names, _METRICS, _STATS)
    if n_epochs == 0:
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    records: List[Dict[str, float]] = []
    for ei in range(n_epochs):
        meta_row = epochs.metadata.iloc[ei]
        record: Dict[str, float] = {}
        for ch_idx, ch_name in enumerate(ch_names):
            windows = extract_blink_windows(meta_row, ch_name, ei)
            per_metric: Dict[str, List[float]] = {m: [] for m in _METRICS}
            channel_modality = infer_modality(ch_name)
            modality_key = "eeg" if channel_modality == "eog" else channel_modality
            for onset_s, duration_s in windows:
                sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
                segment = data[ei, ch_idx, sl]
                metrics = compute_blink_waveform_metrics(
                    segment,
                    sfreq,
                    methods=("base",),
                    modality=modality_key,
                )
                for metric_name in _BASE_METRICS:
                    per_metric[metric_name].append(metrics.get(metric_name, float("nan")))
                per_metric["duration"].append(duration_s)
            for metric in _METRICS:
                stats = _safe_stats(per_metric[metric])
                for stat in _STATS:
                    record[f"{metric}_{stat}_{ch_name}"] = stats[stat]
        records.append(record)

    df = pd.DataFrame.from_records(records, index=index, columns=columns)
    logger.debug("Morphology feature DataFrame shape: %s", df.shape)
    return df