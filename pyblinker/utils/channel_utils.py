"""Channel pick normalization and validation utilities."""
from __future__ import annotations

from typing import Iterable, Sequence

import mne
from mne.io import BaseRaw

from pyblinker.logging import get_logger

logger = get_logger(__name__)


def normalize_picks(picks: str | Iterable[str]) -> list[str]:
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


def require_channels(
    data: mne.Epochs | BaseRaw | Sequence[str], picks: Sequence[str]
) -> None:
    """Validate that all requested channels exist in the provided data.

    Parameters
    ----------
    data : mne.Epochs or mne.io.BaseRaw or sequence of str
        Data structure whose channel names are checked.
    picks : sequence of str
        Channel names to validate.

    Raises
    ------
    ValueError
        If any channel in ``picks`` is missing from ``data``.
    """
    logger.info("Validating channel picks: %s", picks)
    if isinstance(data, (mne.Epochs, BaseRaw)):
        ch_names = data.info["ch_names"]
    else:
        ch_names = list(data)
    missing = [p for p in picks if p not in ch_names]
    if missing:
        raise ValueError(f"Channels not found: {', '.join(missing)}")
    logger.debug("All channels present")

