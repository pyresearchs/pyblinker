"""Annotation-related helper utilities."""

from __future__ import annotations

from typing import Iterable

import mne
import pandas as pd

from pyblinker.logging import get_logger

logger = get_logger(__name__)


def create_annotation(
    sblink: pd.DataFrame,
    sfreq: float,
    label: str,
) -> mne.Annotations:
    """Convert blink spans into an :class:`mne.Annotations` object.

    Parameters
    ----------
    sblink
        DataFrame containing ``start_blink`` and ``end_blink`` columns with
        blink sample indices.
    sfreq
        Sampling frequency of the signal in Hertz. Must be positive.
    label
        Annotation label applied to each blink.

    Returns
    -------
    mne.Annotations
        Annotation object describing the blinks.

    Raises
    ------
    TypeError
        If ``sblink`` is not a DataFrame or ``label`` is not a string.
    ValueError
        If required columns are missing or ``sfreq`` is non-positive.
    """

    logger.info("Entering create_annotation")
    if not isinstance(sblink, pd.DataFrame):
        raise TypeError("sblink must be a pandas.DataFrame")

    required_cols = {"start_blink", "end_blink"}
    missing = required_cols - set(sblink.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(
            f"sblink is missing required columns: {missing_cols}",
        )

    if not isinstance(sfreq, (int, float)) or sfreq <= 0:
        raise ValueError("sfreq must be a positive number")

    if not isinstance(label, str) or not label:
        raise TypeError("label must be a non-empty string")

    onset = (sblink["start_blink"] / sfreq).to_list()
    duration = (
        (sblink["end_blink"] - sblink["start_blink"]) / sfreq
    ).to_list()
    descriptions: Iterable[str] = [label] * len(onset)

    annot = mne.Annotations(
        onset=onset,
        duration=duration,
        description=list(descriptions),
    )
    logger.info("Exiting create_annotation")
    return annot


__all__ = ["create_annotation"]
