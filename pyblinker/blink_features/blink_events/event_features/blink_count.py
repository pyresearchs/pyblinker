"""Blink count feature utilities."""

from typing import Dict, Iterable, List, Optional, Union
import numpy as np
import pandas as pd
import mne
from pyblinker.logging import get_logger
from pyblinker.utils.metadata_utils import extract_blink_windows

from pyblinker.utils import normalize_picks
from pyblinker.utils.modality import infer_modality


logger = get_logger(__name__)


_MODALITY_START_COLUMN = {
    "ear": "start__th_point__ear",
    "eeg": "start__refine__eeg",
    "eog": "start__refine__eog",
}


def _count_from_metadata_start_column(row: pd.Series, column: str) -> float:
    """Count blinks from a modality-specific start landmark metadata column."""

    if column not in row.index:
        return 0.0

    value = row.get(column)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, list):
        return float(sum(0 if pd.isna(v) else 1 for v in value))
    return float(0 if pd.isna(value) else 1)


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


def blink_count(
    epochs: mne.Epochs, picks: str | Iterable[str] | None = None
) -> pd.DataFrame:
    """Count blinks for each epoch using metadata.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch object whose metadata contains blink onset and duration
        information.
    picks : str or iterable of str, optional
        Channel name(s) whose modality determines which blink onset and
        duration columns are used. When channel names are provided the
        resulting column is named ``blink_count_<modality>`` for each unique
        modality encountered. If omitted, the generic ``blink_onset`` and
        ``blink_duration`` columns are used and the output column is simply
        ``blink_count``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` with a leading ``ep`` column. If no
        channel selections are supplied, a single ``blink_count`` column holds
        the per-epoch counts derived from the generic blink metadata. When one
        or more channels are supplied, columns named ``blink_count_<modality>``
        are returned, one for each modality present among the selections.

    Raises
    ------
    ValueError
        If required metadata columns are missing.
    """
    logger.info("Counting blinks across %d epochs", len(epochs))
    metadata = epochs.metadata
    if metadata is None:
        raise ValueError("Epochs.metadata must contain blink information")

    if isinstance(metadata, pd.DataFrame):
        metadata_df = metadata
        index = metadata_df.index
    else:
        metadata_df = pd.DataFrame(metadata)
        index = pd.RangeIndex(len(epochs))

    df = pd.DataFrame(index=index)
    df.insert(0, "ep", index.to_numpy())

    picks_list = normalize_picks(picks) if picks is not None else []
    rows = [row for _, row in metadata_df.iterrows()]

    if not picks_list:
        if "n_blinks" in metadata_df.columns:
            counts = metadata_df["n_blinks"].fillna(0).astype(float).tolist()
        else:
            counts = [
                float(len(extract_blink_windows(row, None, epoch_idx)))
                for epoch_idx, row in enumerate(rows)
            ]
        df["blink_count"] = counts
        logger.debug("Blink counts per epoch: %s", counts)
    else:
        mod_to_channel: Dict[str, str] = {}
        for ch in picks_list:
            mod = infer_modality(ch)
            if mod not in mod_to_channel:
                mod_to_channel[mod] = ch

        for modality, channel in mod_to_channel.items():
            start_column = _MODALITY_START_COLUMN.get(modality)
            mod_onset_key = f"blink_onset_{modality}"
            mod_duration_key = f"blink_duration_{modality}"
            has_window_columns = (
                (
                    mod_onset_key in metadata_df.columns
                    and mod_duration_key in metadata_df.columns
                )
                or (
                    "blink_onset" in metadata_df.columns
                    and "blink_duration" in metadata_df.columns
                )
            )

            if has_window_columns:
                counts = [
                    float(len(extract_blink_windows(row, channel, epoch_idx)))
                    for epoch_idx, row in enumerate(rows)
                ]
            elif start_column is not None and start_column in metadata_df.columns:
                counts = [
                    _count_from_metadata_start_column(row, start_column)
                    for row in rows
                ]
            else:
                counts = [
                    float(len(extract_blink_windows(row, channel, epoch_idx)))
                    for epoch_idx, row in enumerate(rows)
                ]
            col_name = f"blink_count_{modality}"
            df[col_name] = counts
            logger.debug(
                "Blink counts for modality '%s': %s",
                modality,
                counts,
            )

    logger.info("Finished counting blinks")
    return df
