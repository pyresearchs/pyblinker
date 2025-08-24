"""Blink interval distribution features using raw segments."""
import logging
from typing import Dict, Optional, TYPE_CHECKING

import mne
import numpy as np

if TYPE_CHECKING:
    import pandas as pd  # for type hints only
logger = logging.getLogger(__name__)


def blink_interval_distribution_segment(
    raw: mne.io.BaseRaw,
    *,
    blink_label: Optional[str] = "blink",
) -> Dict[str, float]:
    """Compute blink interval distribution metrics for one raw segment.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw segment containing blink annotations.
    blink_label : str | None, optional
        Annotation description identifying blinks. If ``None``, all annotations
        are used.

    Returns
    -------
    dict
        Dictionary with ``blink_interval_min``, ``blink_interval_max`` and
        ``blink_interval_std`` values computed from successive blink onsets.
    """
    logger.info("Computing blink interval distribution for a segment")
    ann = raw.annotations
    mask = np.ones(len(ann), dtype=bool)
    if blink_label is not None:
        mask &= ann.description == blink_label
    starts = ann.onset[mask]

    if len(starts) < 2:
        logger.debug("Insufficient blinks for interval calculation: %d", len(starts))
        return {
            "blink_interval_min": float("nan"),
            "blink_interval_max": float("nan"),
            "blink_interval_std": float("nan"),
        }

    ibis = np.diff(starts)
    features = {
        "blink_interval_min": float(np.min(ibis)),
        "blink_interval_max": float(np.max(ibis)),
        "blink_interval_std": float(np.std(ibis, ddof=1)) if len(ibis) > 1 else float("nan"),
    }
    logger.debug("Blink intervals: %s", ibis)
    logger.debug("Interval features: %s", features)
    return features


def aggregate_blink_interval_distribution(
    raws: "mne.io.BaseRaw | list[mne.io.BaseRaw] | tuple[mne.io.BaseRaw, ...]",
    *,
    blink_label: Optional[str] = "blink",
) -> "pd.DataFrame":
    """Aggregate blink interval metrics for multiple raw segments."""
    import pandas as pd  # local import to avoid heavy dependency at module load

    logger.info("Aggregating blink interval features over %d segments", len(raws))
    records = []
    for idx, segment in enumerate(raws):
        feats = blink_interval_distribution_segment(segment, blink_label=blink_label)
        record = {"epoch": idx}
        record.update(feats)
        records.append(record)
    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated blink interval DataFrame shape: %s", df.shape)
    return df
