"""Blink count feature utilities."""

from typing import List, Dict, Union, Optional
import logging
import numpy as np
import pandas as pd
import mne


logger = logging.getLogger(__name__)


def blink_count_epoch(
        blinks: Union[List[Dict[str, int]], 'mne.io.BaseRaw', 'mne.Epochs'],
        label: Optional[str] = None
) -> int:
    """Return the number of blinks for a single epoch or from an MNE object.

    Parameters
    ----------
    blinks : list of dict or mne.io.Raw or mne.Epochs
        Blink annotations for one epoch (as a list of dicts) or an MNE Raw object.
    label : str, optional
        If using an MNE Raw object, count only annotations matching this label.

    Returns
    -------
    int
        Total count of blinks.

    Raises
    ------
    NotImplementedError
        If the input is an MNE Epochs object.
    TypeError
        If the input type is not supported.
    """
    logger.info("Calculating blink count for input of type %s", type(blinks))

    if mne and isinstance(blinks, mne.io.BaseRaw):
        logger.debug("Using MNE Raw logic for annotation counting")
        mask = np.ones(len(blinks.annotations), dtype=bool)
        if label is not None:
            mask &= blinks.annotations.description == label
        count = int(mask.sum())
        logger.debug("Found %d blink annotations matching label '%s'", count, label)
        return count

    elif mne and isinstance(blinks, mne.Epochs):
        logger.warning("Blink count from MNE Epochs not implemented")
        raise NotImplementedError("blink_count_epoch does not support MNE Epochs input.")

    elif isinstance(blinks, list):
        logger.debug("Counting %s blinks from list of dicts", len(blinks))
        return len(blinks)

    else:
        logger.error("Unsupported type passed to blink_count_epoch: %s", type(blinks))
        raise TypeError(f"Unsupported input type: {type(blinks)}")


def blink_count(epochs: mne.Epochs) -> pd.DataFrame:
    """Count blinks for each epoch using metadata.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch object whose metadata includes ``blink_onset`` and
        ``blink_duration`` columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` with passthrough metadata columns
        (``blink_onset`` and ``blink_duration``) and a new ``blink_count``
        column. ``blink_count`` is ``0`` when an epoch contains no blinks.

    Raises
    ------
    ValueError
        If required metadata columns are missing.
    """
    logger.info("Counting blinks across %d epochs", len(epochs))
    metadata = epochs.metadata
    if metadata is None or not {"blink_onset", "blink_duration"}.issubset(metadata.columns):
        raise ValueError("Epochs.metadata must contain 'blink_onset' and 'blink_duration' columns")

    df = metadata[["blink_onset", "blink_duration"]].copy()

    def _count(entry: object) -> int:
        if isinstance(entry, list):
            return len(entry)
        if entry is None or pd.isna(entry):
            return 0
        return 1

    df["blink_count"] = metadata["blink_onset"].apply(_count).astype(float)
    logger.debug("Blink counts per epoch: %s", df["blink_count"].tolist())
    return df
