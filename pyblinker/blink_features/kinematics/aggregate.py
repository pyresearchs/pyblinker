"""Aggregate blink kinematic features across epochs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import logging
import numpy as np
import pandas as pd

from .per_blink import compute_segment_kinematics
from ..energy.helpers import _safe_stats

logger = logging.getLogger(__name__)

_METRICS = (
    "peak_amp",
    "t2p",
    "vel_mean",
    "vel_peak",
    "acc_mean",
    "acc_peak",
    "rise_time",
    "fall_time",
    "auc",
    "symmetry",
)
_STATS = ("mean", "std", "cv")


def aggregate_kinematic_features(
    blinks: Iterable[Dict[str, Any]], sfreq: float, n_epochs: int
) -> pd.DataFrame:
    """Aggregate kinematic metrics for each epoch.

    Parameters
    ----------
    blinks : iterable of dict
        Blink annotations containing ``epoch_index``, ``refined_start_frame``,
        ``refined_end_frame`` and ``epoch_signal`` fields.
    sfreq : float
        Sampling frequency in Hertz.
    n_epochs : int
        Number of epochs to aggregate.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by epoch with kinematic summary statistics.
    """

    logger.info("Aggregating kinematic features over %d epochs", n_epochs)
    per_epoch: List[List[Dict[str, Any]]] = [list() for _ in range(n_epochs)]
    for blink in blinks:
        idx = blink.get("epoch_index", -1)
        if 0 <= idx < n_epochs:
            per_epoch[idx].append(blink)

    records: List[Dict[str, float]] = []
    for epoch_blinks in per_epoch:
        per_metric: Dict[str, List[float]] = {m: [] for m in _METRICS}
        for blink in epoch_blinks:
            start = int(blink["refined_start_frame"])
            end = int(blink["refined_end_frame"])
            signal = np.asarray(blink["epoch_signal"], dtype=float)
            segment = signal[start:end]
            if segment.size == 0:
                continue
            metrics = compute_segment_kinematics(segment, sfreq)
            for m in _METRICS:
                per_metric[m].append(metrics[m])
        record: Dict[str, float] = {}
        for metric, values in per_metric.items():
            stats = _safe_stats(values)
            for stat_name, value in stats.items():
                record[f"{metric}_{stat_name}"] = value
        records.append(record)

    df = pd.DataFrame.from_records(records, index=pd.RangeIndex(n_epochs))
    logger.debug("Aggregated kinematic DataFrame shape: %s", df.shape)
    return df

