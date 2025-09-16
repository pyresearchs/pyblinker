"""Utilities for working with blink onset and duration metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import List, Tuple

import ast
import numpy as np
import pandas as pd

from pyblinker.logging import get_logger


logger = get_logger(__name__)


def _contains(metadata_row: pd.Series | Mapping[str, object], key: str) -> bool:
    """Return ``True`` when ``key`` is available in ``metadata_row``."""
    if isinstance(metadata_row, pd.Series):
        return key in metadata_row.index
    return key in metadata_row


def extract_blink_windows(
    metadata_row: pd.Series | Mapping[str, object],
    channel: str | None,
    epoch_index: int,
) -> List[Tuple[float, float]]:
    """Extract blink onset/duration pairs for a single epoch.

    Parameters
    ----------
    metadata_row
        Row from ``epochs.metadata`` providing blink annotations. Values may be
        scalars, lists, or string representations of lists.
    channel
        Channel name used to infer the modality-specific metadata columns. If
        set to ``None`` or a generic sentinel (``"generic"``, ``"all"`` or
        ``"any"``), the function bypasses modality-specific columns and
        directly consults ``blink_onset``/``blink_duration``.
    epoch_index
        Integer position of the epoch in ``epochs``. Included in error logs to
        aid debugging.

    Returns
    -------
    list of tuple of float
        Sequence of ``(onset_seconds, duration_seconds)`` pairs. An empty list
        is returned when no blinks are recorded.

    Raises
    ------
    ValueError
        If neither modality-specific nor generic blink metadata columns are
        available for the requested channel.
    TypeError
        If ``metadata_row`` is not a :class:`pandas.Series` or mapping.
    """

    channel_label = channel if channel is not None else "generic"
    logger.info(
        "Entering extract_blink_windows for channel %s (epoch %d)",
        channel_label,
        epoch_index,
    )

    if not isinstance(metadata_row, (pd.Series, Mapping)):
        logger.error("Unsupported metadata row type: %s", type(metadata_row))
        raise TypeError("metadata_row must be a pandas.Series or mapping")

    ch_lower = channel_label.lower()
    prefer_generic = ch_lower in {"generic", "all", "any", ""}
    if "ear" in ch_lower:
        mod = "ear"
    elif "eog" in ch_lower:
        mod = "eog"
    else:
        mod = "eeg"

    mod_onset_key = f"blink_onset_{mod}"
    mod_duration_key = f"blink_duration_{mod}"

    def _is_missing(val: object) -> bool:
        return val is None or (isinstance(val, float) and np.isnan(val))

    if (not prefer_generic) and _contains(metadata_row, mod_onset_key) and _contains(
        metadata_row, mod_duration_key
    ):
        onsets = metadata_row.get(mod_onset_key)
        durations = metadata_row.get(mod_duration_key)
        if _is_missing(onsets) or _is_missing(durations):
            logger.debug(
                "Modality-specific metadata for channel %s contains missing values",
                channel_label,
            )
            windows: List[Tuple[float, float]] = []
            logger.info("Exiting extract_blink_windows")
            return windows
    else:
        generic_keys = ("blink_onset", "blink_duration")
        missing = [key for key in generic_keys if not _contains(metadata_row, key)]
        if missing:
            logger.error("Missing blink metadata columns: %s", ", ".join(sorted(missing)))
            raise ValueError(
                "Epochs.metadata missing required blink columns: "
                + ", ".join(sorted(missing))
            )
        onsets = metadata_row.get("blink_onset")
        durations = metadata_row.get("blink_duration")
        if _is_missing(onsets) or _is_missing(durations):
            logger.debug(
                "Generic metadata contains missing values for channel %s",
                channel_label,
            )
            windows = []
            logger.info("Exiting extract_blink_windows")
            return windows

    def _ensure_list(val: object) -> List[object]:
        """Coerce scalars or string encodings of lists to ``list`` objects."""

        if isinstance(val, str):
            try:
                val = ast.literal_eval(val)
            except (SyntaxError, ValueError):
                pass
        if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
            return list(val)
        return [val]

    onsets_list = _ensure_list(onsets)
    durations_list = _ensure_list(durations)

    windows = []
    for onset, duration in zip(onsets_list, durations_list):
        if _is_missing(onset) or _is_missing(duration):
            continue
        windows.append((float(onset), float(duration)))

    logger.debug("Extracted %d blink windows", len(windows))
    logger.info("Exiting extract_blink_windows")
    return windows


__all__ = ["extract_blink_windows"]

