"""Aggregate blink event features using :class:`mne.Epochs` metadata."""
from __future__ import annotations

from typing import Iterable, Sequence

import logging

import mne
import numpy as np
import pandas as pd

from .blink_count import blink_count
from .inter_blink_interval import inter_blink_interval_epochs
from .utils import normalize_picks, require_channels

logger = logging.getLogger(__name__)


def aggregate_blink_event_features(
    epochs: mne.Epochs,
    picks: str | Iterable[str],
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate blink-event metrics across all epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch object with metadata containing ``blink_onset`` and
        ``blink_duration`` columns.
    picks : str or iterable of str
        Channel name(s) used when computing inter-blink interval (IBI)
        statistics. The same IBI values are used for all channels because
        blink timing is not channel-specific in the metadata yet.
    features : sequence of str or None, optional
        Subset of feature groups to compute. Valid keys are
        ``"blink_total"``, ``"blink_rate"`` and ``"ibi"``. Passing ``None``
        (default) computes all features.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame containing the aggregated metrics.

    Raises
    ------
    ValueError
        If an unknown feature key is requested or if required channels are
        missing from ``epochs`` when ``"ibi"`` is selected.
    """

    logger.info("Aggregating blink features from %d epochs", len(epochs))

    valid = {"blink_total", "blink_rate", "ibi"}
    selected = set(features) if features is not None else valid
    invalid = selected - valid
    if invalid:
        raise ValueError(f"Unknown feature keys: {sorted(invalid)}")

    record: dict[str, float] = {}

    counts_df = blink_count(epochs)
    if "blink_total" in selected or "blink_rate" in selected:
        blink_total = float(counts_df["blink_count"].sum())
    if "blink_total" in selected:
        record["blink_total"] = blink_total

    if "blink_rate" in selected:
        epoch_len = epochs.tmax - epochs.tmin + 1.0 / epochs.info["sfreq"]
        total_duration = epoch_len * len(epochs)
        record["blink_rate"] = (
            blink_total / total_duration * 60.0 if total_duration else float("nan")
        )

    if "ibi" in selected:
        picks_list = normalize_picks(picks)
        require_channels(epochs, picks_list)
        ibis_df = inter_blink_interval_epochs(epochs, picks_list)
        for ch in picks_list:
            record[f"ibi_{ch}"] = float(np.nanmean(ibis_df[f"ibi_{ch}"].to_numpy()))

    df = pd.DataFrame([record])
    logger.debug("Aggregated feature row: %s", record)
    return df


__all__ = ["aggregate_blink_event_features"]

