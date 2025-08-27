"""Aggregate blink morphology features from :class:`mne.Epochs`."""
from __future__ import annotations

from typing import Dict, List, Sequence
import logging

import mne
import pandas as pd

from .per_blink import WAVEFORM_METRICS, compute_blink_waveform_metrics
from ..energy.helpers import _extract_blink_windows, _segment_to_samples, _safe_stats

logger = logging.getLogger(__name__)

# Derive metric and statistic names instead of hardcoding
_METRICS = WAVEFORM_METRICS + ("duration",)
_STATS = tuple(_safe_stats([]).keys())


def _make_columns(ch_names: Sequence[str]) -> List[str]:
    """Generate ordered column names for the output DataFrame."""
    columns: List[str] = []
    for ch in ch_names:
        for metric in _METRICS:
            for stat in _STATS:
                columns.append(f"{metric}_{stat}_{ch}")
    return columns


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

    if picks is None:
        ch_names = [
            ch
            for ch in epochs.ch_names
            if "EOG" in ch.upper() or "EAR" in ch.upper()
        ]
        if not ch_names:
            raise ValueError("No default EAR/EOG channels found")
    elif isinstance(picks, str):
        ch_names = [picks]
    else:
        ch_names = list(picks)

    missing = [ch for ch in ch_names if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Channels not found: {missing}")

    data = epochs.get_data(picks=ch_names)
    sfreq = float(epochs.info["sfreq"])
    n_epochs, n_ch, n_times = data.shape
    index = epochs.metadata.index

    columns = _make_columns(ch_names)
    if n_epochs == 0:
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    records: List[Dict[str, float]] = []
    for ei in range(n_epochs):
        meta_row = epochs.metadata.iloc[ei]
        record: Dict[str, float] = {}
        for ch_idx, ch_name in enumerate(ch_names):
            windows = _extract_blink_windows(meta_row, ch_name, ei)
            per_metric: Dict[str, List[float]] = {m: [] for m in _METRICS}
            for onset_s, duration_s in windows:
                sl = _segment_to_samples(onset_s, duration_s, sfreq, n_times)
                segment = data[ei, ch_idx, sl]
                metrics = compute_blink_waveform_metrics(segment, sfreq)
                if metrics is None:
                    continue
                for m, val in metrics.items():
                    per_metric[m].append(val)
                per_metric["duration"].append(duration_s)
            for metric in _METRICS:
                stats = _safe_stats(per_metric[metric])
                for stat in _STATS:
                    record[f"{metric}_{stat}_{ch_name}"] = stats[stat]
        records.append(record)

    df = pd.DataFrame.from_records(records, index=index, columns=columns)
    logger.debug("Morphology feature DataFrame shape: %s", df.shape)
    return df
