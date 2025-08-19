"""Aggregate blink kinematic features."""
from typing import Any, Dict, Iterable, List

import logging
import pandas as pd

from .kinematic_features import compute_kinematic_features

logger = logging.getLogger(__name__)


def aggregate_kinematic_features(
    blinks: Iterable[Dict[str, Any]],
    sfreq: float,
    n_epochs: int,
) -> pd.DataFrame:
    """Aggregate kinematic metrics across epochs.

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
        DataFrame indexed by epoch with kinematic features.
    """
    logger.info("Aggregating kinematic features over %d epochs", n_epochs)
    per_epoch: List[List[Dict[str, Any]]] = [list() for _ in range(n_epochs)]
    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch[idx].append(blink)

    records = []
    for epoch_idx, epoch_blinks in enumerate(per_epoch):
        feats = compute_kinematic_features(epoch_blinks, sfreq)
        record = {"epoch": epoch_idx}
        record.update(feats)
        records.append(record)

    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated kinematic DataFrame shape: %s", df.shape)
    return df
