"""Count blinks for each epoch given an :class:`mne.Epochs` object.

This helper operates on pre-segmented epochs and a pandas DataFrame of
annotation onsets. It is useful when blink boundaries have already been
identified in the continuous recording and converted to a tabular format.
"""
from typing import Optional
import logging

import mne
import pandas as pd

logger = logging.getLogger(__name__)


def blink_count_epochs(
    epochs: mne.Epochs,
    ann_df: pd.DataFrame,
    *,
    blink_label: Optional[str] = "blink",
) -> pd.DataFrame:
    """Count blinks per epoch using annotation onsets.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch object defining start and stop times within the original raw
        recording.
    ann_df : pandas.DataFrame
        DataFrame with ``onset`` (seconds) and ``description`` columns
        describing blink annotations relative to the raw.
    blink_label : str | None, optional
        Annotation label used to filter blinks. ``None`` uses all rows.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by epoch with a single ``blink_count`` column.
    """
    n_epochs = len(epochs.events)
    logger.info("Counting blinks across %d epochs", n_epochs)

    onsets = ann_df["onset"].to_numpy()
    descriptions = ann_df["description"].to_numpy()

    sfreq = epochs.info["sfreq"]
    start_times = epochs.events[:, 0] / sfreq + epochs.tmin
    epoch_len = epochs.tmax - epochs.tmin + 1.0 / sfreq

    counts = []
    for start in start_times:
        end = start + epoch_len
        mask = (onsets >= start) & (onsets < end)
        if blink_label is not None:
            mask &= descriptions == blink_label
        counts.append(int(mask.sum()))

    df = pd.DataFrame({"epoch": range(n_epochs), "blink_count": counts})
    logger.debug("Blink counts: %s", counts)
    return df.set_index("epoch")
