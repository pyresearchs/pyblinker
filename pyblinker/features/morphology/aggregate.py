"""Aggregate blink morphology features."""
from typing import Dict, Iterable, List
import logging
import pandas as pd

from .morphology_features import compute_morphology_features

logger = logging.getLogger(__name__)


def aggregate_morphology_features(
    blinks: Iterable[Dict[str, int]],
    sfreq: float,
    n_epochs: int,
) -> pd.DataFrame:
    """Aggregate morphology metrics across epochs.

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
        DataFrame indexed by epoch with morphology features.
    """
    logger.info("Aggregating morphology features over %d epochs", n_epochs)
    per_epoch: List[List[Dict[str, int]]] = [list() for _ in range(n_epochs)]
    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch[idx].append(blink)

    records = []
    for epoch_idx, epoch_blinks in enumerate(per_epoch):
        feats = compute_morphology_features(epoch_blinks, sfreq)
        record = {"epoch": epoch_idx}
        record.update(feats)
        records.append(record)

    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated morphology DataFrame shape: %s", df.shape)
    return df
