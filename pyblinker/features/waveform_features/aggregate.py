"""Aggregate blink waveform-based features.

This module follows the feature definitions from the open-source
`BLINKER`_ project but provides a minimal implementation focused on the
duration and amplitude‑velocity metrics included here.

.. _BLINKER: https://github.com/VisLab/EEG-Blinks
"""
from typing import Iterable, Dict, Any, List
import logging
import pandas as pd
import numpy as np

from .features.duration_features import duration_base, duration_zero
from .features.amp_vel_ratio_features import neg_amp_vel_ratio_zero

logger = logging.getLogger(__name__)


def aggregate_waveform_features(
    blinks: Iterable[Dict[str, Any]],
    sfreq: float,
    n_epochs: int,
) -> pd.DataFrame:
    """Aggregate waveform metrics across epochs.

    The aggregation closely mirrors the approach used by the
    `BLINKER`_ repository but is trimmed down to only a handful of
    features in this package.

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
        DataFrame indexed by epoch with mean waveform features.
    """
    logger.info("Aggregating waveform features over %d epochs", n_epochs)
    per_epoch: List[List[Dict[str, Any]]] = [list() for _ in range(n_epochs)]
    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch[idx].append(blink)

    records = []
    for epoch_idx, epoch_blinks in enumerate(per_epoch):
        if epoch_blinks:
            dur_base = [duration_base(b, sfreq) for b in epoch_blinks]
            dur_zero = [duration_zero(b, sfreq) for b in epoch_blinks]
            ratio_neg = [neg_amp_vel_ratio_zero(b, sfreq) for b in epoch_blinks]
            features = {
                "duration_base_mean": float(np.mean(dur_base)),
                "duration_zero_mean": float(np.mean(dur_zero)),
                "neg_amp_vel_ratio_zero_mean": float(np.nanmean(ratio_neg)),
            }
        else:
            features = {
                "duration_base_mean": float("nan"),
                "duration_zero_mean": float("nan"),
                "neg_amp_vel_ratio_zero_mean": float("nan"),
            }
        record = {"epoch": epoch_idx}
        record.update(features)
        records.append(record)

    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated waveform DataFrame shape: %s", df.shape)
    return df
