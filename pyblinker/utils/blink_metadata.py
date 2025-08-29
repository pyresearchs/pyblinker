"""Helpers for working with blink metadata."""

import logging
from typing import Any, Dict, List

import mne
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


def attach_blink_metadata(epochs: mne.Epochs, blink_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-blink properties and merge them into epoch metadata.

    All columns from ``blink_df`` (except ``seg_id`` and ``blink_id``) are
    converted into list-valued epoch-level metadata. Lists contain one entry per
    detected blink within the epoch. Additional convenience columns such as
    ``blink_onset`` and ``blink_duration`` in seconds are also added.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch container whose ``metadata`` will be updated in-place. The
        ``selection`` attribute is used to map original segment indices to kept
        epoch indices.
    blink_df : pandas.DataFrame
        Long-format blink table returned by
        :func:`pyblinker.segment_blink_properties.compute_segment_blink_properties`.
        Must contain at least ``seg_id``, ``blink_id``, ``start_blink`` and
        ``end_blink`` columns measured in samples.

    Returns
    -------
    pandas.DataFrame
        Filtered long-format blink table containing only rows from epochs that
        remain after any ``Epochs`` dropping operations. This table is detached
        from ``epochs.metadata`` so that downstream code can operate on
        per-blink rows directly.
    """
    logger.info("Entering attach_blink_metadata")

    sfreq = float(epochs.info["sfreq"])
    selection_map = {orig: new for new, orig in enumerate(epochs.selection)}

    df = blink_df.copy()
    df["epoch_index"] = df["seg_id"].map(selection_map)
    df = df.dropna(subset=["epoch_index"]).reset_index(drop=True)
    df["epoch_index"] = df["epoch_index"].astype(int)

    df["blink_onset"] = df["start_blink"] / sfreq
    df["blink_duration"] = (df["end_blink"] - df["start_blink"]) / sfreq

    group = df.groupby("epoch_index")
    n_epochs = len(epochs)
    epoch_meta = pd.DataFrame(index=range(n_epochs))
    epoch_meta["n_blinks"] = group.size().reindex(epoch_meta.index, fill_value=0)

    def _list_or_nan(series: pd.Series) -> object:
        values = series.dropna().tolist()
        return values if values else float("nan")

    cols_to_attach = [
        c for c in df.columns if c not in {"seg_id", "blink_id", "epoch_index"}
    ]
    for col in cols_to_attach:
        epoch_meta[col] = group[col].apply(_list_or_nan).reindex(epoch_meta.index)

    epoch_meta.index.name = None

    existing = (
        epochs.metadata.copy()
        if isinstance(epochs.metadata, pd.DataFrame)
        else pd.DataFrame(index=range(n_epochs))
    )
    existing = existing.reset_index(drop=True)
    keep_cols = [
        c
        for c in existing.columns
        if (c not in epoch_meta.columns) and not (c.startswith("blink_") or c == "n_blinks")
    ]
    merged = existing[keep_cols].join(epoch_meta)
    epochs.metadata = merged
    epochs.metadata.reset_index(drop=True, inplace=True)

    logger.info("Exiting attach_blink_metadata")
    return df.drop(columns=["epoch_index"])
