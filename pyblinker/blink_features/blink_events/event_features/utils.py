"""Utility helpers for blink event features."""
from __future__ import annotations

import logging
from typing import Iterable, Sequence, List

import mne

logger = logging.getLogger(__name__)


def normalize_picks(picks: str | Iterable[str]) -> List[str]:
    """Normalize channel picks to a list.

    Parameters
    ----------
    picks : str or iterable of str
        Channel name or collection of channel names.

    Returns
    -------
    list of str
        Normalized list of channel names.
    """
    if isinstance(picks, str):
        return [picks]
    return list(picks)


def require_channels(epochs: mne.Epochs, picks: Sequence[str]) -> None:
    """Validate that all requested channels exist in the epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs whose channel names are checked.
    picks : sequence of str
        Channel names to validate.

    Raises
    ------
    ValueError
        If any channel in ``picks`` is missing from ``epochs``.
    """
    logger.info("Validating channel picks: %s", picks)
    missing = [p for p in picks if p not in epochs.info["ch_names"]]
    if missing:
        raise ValueError(f"Channels not found in epochs: {', '.join(missing)}")
    logger.debug("All channels present")
