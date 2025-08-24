"""Utilities for summarizing blink spectrum metrics across epochs.

This module exposes :func:`aggregate_frequency_domain_features`, a high-level
helper that gathers per-blink frequency-domain measurements for each epoch and
returns them in a tidy :class:`pandas.DataFrame`. For a given sequence of blink
annotations, the function groups blinks by their ``epoch_index``, retrieves the
corresponding epoch signal, and delegates the per-epoch spectral calculation to
``compute_frequency_domain_features``. The resulting table contains one row per
epoch with columns such as peak blink-rate frequency, broadband power, spectral
entropy, and wavelet energy. Epochs that lack a signal or blinks are filled with
``NaN`` values so downstream analyses can maintain alignment with the original
epoch numbering.
"""
from typing import Iterable, Dict, Any, List
import logging
import pandas as pd
import numpy as np

from ..features import compute_frequency_domain_features

logger = logging.getLogger(__name__)


def aggregate_frequency_domain_features(
    blinks: Iterable[Dict[str, Any]],
    sfreq: float,
    n_epochs: int,
) -> pd.DataFrame:
    """Aggregate frequency-domain metrics for multiple epochs.

    Parameters
    ----------
    blinks : Iterable[dict]
        Blink annotations with ``epoch_index`` and ``epoch_signal``.
    sfreq : float
        Sampling frequency in Hertz.
    n_epochs : int
        Number of epochs to aggregate.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by epoch with frequency-domain features.
    """
    logger.info("Aggregating frequency-domain features over %d epochs", n_epochs)
    per_epoch_signal: List[np.ndarray | None] = [None for _ in range(n_epochs)]
    per_epoch_blinks: List[List[Dict[str, Any]]] = [list() for _ in range(n_epochs)]

    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs:
            per_epoch_blinks[idx].append(blink)
            if per_epoch_signal[idx] is None:
                per_epoch_signal[idx] = np.asarray(blink["epoch_signal"], dtype=float)

    records = []
    for idx in range(n_epochs):
        signal = per_epoch_signal[idx]
        blinks_epoch = per_epoch_blinks[idx]
        if signal is None:
            feats = {
                "blink_rate_peak_freq": float("nan"),
                "blink_rate_peak_power": float("nan"),
                "broadband_power_0_5_2": float("nan"),
                "broadband_com_0_5_2": float("nan"),
                "high_freq_entropy_2_13": float("nan"),
                "one_over_f_slope": float("nan"),
                "band_power_ratio": float("nan"),
                "wavelet_energy_d1": float("nan"),
                "wavelet_energy_d2": float("nan"),
                "wavelet_energy_d3": float("nan"),
                "wavelet_energy_d4": float("nan"),
            }
        else:
            feats = compute_frequency_domain_features(blinks_epoch, signal, sfreq)
        record = {"epoch": idx}
        record.update(feats)
        records.append(record)

    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated frequency-domain DataFrame shape: %s", df.shape)
    return df
