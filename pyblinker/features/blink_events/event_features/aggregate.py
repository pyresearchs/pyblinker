"""Aggregate blink event features."""
from typing import Dict, Iterable, List, Sequence, Set

import logging
import pandas as pd

from .blink_count import blink_count_epoch
from .blink_rate import blink_rate_epoch
from .inter_blink_interval import compute_ibi_features

logger = logging.getLogger(__name__)


def aggregate_blink_event_features(
    blinks: Iterable[Dict[str, int]],
    sfreq: float,
    epoch_len: float,
    n_epochs: int,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate blink-event based metrics for multiple epochs.

    Parameters
    ----------
    blinks : Iterable[dict]
        Iterable of blink annotations containing an ``epoch_index`` field.
    sfreq : float
        Sampling frequency of the recording in Hertz.
    epoch_len : float
        Length of each epoch in seconds.
    n_epochs : int
        Total number of epochs to aggregate.

    features : Sequence[str] | None, optional
        Iterable of feature groups to compute. Valid options are
        ``"blink_count"``, ``"blink_rate"`` and ``"ibi"``.  Passing ``None``
        (default) computes all features.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by epoch containing the selected features for
        each epoch.
    """
    logger.info("Aggregating blink features over %s epochs", n_epochs)

    valid: Set[str] = {"blink_count", "blink_rate", "ibi"}
    selected: Set[str]
    if features is None:
        selected = valid
    else:
        selected = set(features)
        invalid = selected - valid
        if invalid:
            raise ValueError(f"Unknown feature keys: {sorted(invalid)}")

    per_epoch: List[List[Dict[str, int]]] = [list() for _ in range(n_epochs)]
    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch[idx].append(blink)
    records = []
    for epoch_idx, epoch_blinks in enumerate(per_epoch):
        record = {"epoch": epoch_idx}

        if "blink_count" in selected:
            record["blink_count"] = blink_count_epoch(epoch_blinks)

        if "blink_rate" in selected:
            record["blink_rate"] = blink_rate_epoch(epoch_blinks, epoch_len)

        if "ibi" in selected:
            record.update(compute_ibi_features(epoch_blinks, sfreq))

        records.append(record)
    df = pd.DataFrame.from_records(records).set_index("epoch")
    return df
