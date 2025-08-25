"""Aggregate blink energy and complexity features."""
from typing import Any, Dict, Iterable, List
import logging
import pandas as pd

from .energy_complexity_features import compute_energy_features

logger = logging.getLogger(__name__)


def aggregate_energy_complexity_features(
    blinks: Iterable[Dict[str, Any]],
    sfreq: float,
    n_epochs: int,
) -> pd.DataFrame:
    """Aggregate energy and complexity metrics across epochs.

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
        DataFrame indexed by epoch with energy and complexity features.
    """
    logger.info("Aggregating energy and complexity features over %d epochs", n_epochs)
    per_epoch: List[List[Dict[str, Any]]] = [list() for _ in range(n_epochs)]
    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch[idx].append(blink)

    records = []
    for epoch_idx, epoch_blinks in enumerate(per_epoch):
        feats = compute_energy_features(epoch_blinks, sfreq)
        record = {"epoch": epoch_idx}
        record.update(feats)
        records.append(record)

    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated energy-complexity DataFrame shape: %s", df.shape)
    return df
