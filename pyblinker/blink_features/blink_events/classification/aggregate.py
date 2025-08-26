"""Aggregate partial and complete blink metrics."""
from typing import Iterable, Dict, Any, List
import logging
import pandas as pd

from .features import classify_blinks_epoch

logger = logging.getLogger(__name__)


def aggregate_classification_features(
    blinks: Iterable[Dict[str, Any]],
    sfreq: float,
    epoch_len: float,
    n_epochs: int,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """Aggregate blink classification metrics across epochs.

    Parameters
    ----------
    blinks : Iterable[dict]
        Blink annotations with ``epoch_index`` field.
    sfreq : float
        Sampling frequency in Hertz.
    epoch_len : float
        Length of each epoch in seconds.
    n_epochs : int
        Number of epochs to aggregate.
    threshold : float, optional
        Amplitude threshold for partial blinks, by default ``0.15``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by epoch with partial and complete blink metrics.
    """
    logger.info("Aggregating blink classification features over %d epochs", n_epochs)

    per_epoch: List[List[Dict[str, Any]]] = [list() for _ in range(n_epochs)]
    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch[idx].append(blink)

    records = []
    for idx, epoch_blinks in enumerate(per_epoch):
        feats = classify_blinks_epoch(epoch_blinks, sfreq, epoch_len, threshold)
        record = {
            "epoch": idx,
            "Partial_Blink_threshold": threshold,
        }
        record.update(feats)
        records.append(record)

    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated blink classification DataFrame shape: %s", df.shape)
    return df
