"""Helpers for working with blink metadata."""

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def onset_entry_to_blinks(onset: Any) -> List[Dict[str, float]]:
    """Convert a ``blink_onset`` metadata entry into blink dictionaries.

    Parameters
    ----------
    onset : Any
        Value from an ``Epochs`` metadata ``blink_onset`` column. May be a
        float, list of floats, ``None`` or ``NaN``.

    Returns
    -------
    list of dict
        List of dictionaries with an ``onset`` field for each blink.
    """
    logger.info("Entering onset_entry_to_blinks")
    if isinstance(onset, list):
        blinks = [{"onset": float(o)} for o in onset]
    elif onset is None or pd.isna(onset):
        blinks = []
    else:
        blinks = [{"onset": float(onset)}]
    logger.debug("Converted %s to %d blink entries", onset, len(blinks))
    logger.info("Exiting onset_entry_to_blinks")
    return blinks
