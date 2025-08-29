"""Segment-level blink property extraction utilities.

This module exposes :func:`compute_segment_blink_properties`, which operates on
refined :class:`mne.Epochs` produced by
``slice_raw_into_mne_epochs_refine_annot``. The epoch metadata provides the
blink windows that drive the computation.

Refactor goals achieved in this version:
- Decomposed into small, focused functions (single responsibility).
- No functions defined inside other functions.
- No nested ``for`` loops; iteration is flattened where needed.
"""

from __future__ import annotations

from typing import Sequence, Dict, Any, List, Iterable, Tuple
import logging
import itertools

import numpy as np
import pandas as pd
import mne
from tqdm import tqdm

from .utils.blink_metadata import attach_blink_metadata

from .blinker.fit_blink import FitBlinks
from .blinker.extract_blink_properties import BlinkProperties
from .blink_features.blink_events.blink_dataframe import left_right_zero_crossing
from .blink_features.waveform_features.aggregate import _sample_windows_from_metadata

logger = logging.getLogger(__name__)

 # ------------------------------ Public API ------------------------------------


def compute_segment_blink_properties(
    epochs: mne.Epochs,
    params: Dict[str, Any],
    *,
    channel: str | Sequence[str] = "EEG-E8",
    progress_bar: bool = True,
    long_format: bool = False,
) -> mne.Epochs | pd.DataFrame:
    """Calculate blink properties from refined epochs.

    Parameters
    ----------
    epochs
        ``mne.Epochs`` instance produced by
        :func:`slice_raw_into_mne_epochs_refine_annot`.
    params
        Parameter dictionary forwarded to :class:`BlinkProperties`.
    channel
        Channel name(s) used for property extraction. Defaults to ``"EEG-E8"``.
    progress_bar
        Whether to display a progress bar during processing.
    long_format
        If ``True``, return a long-format :class:`pandas.DataFrame` of per-blink
        properties. Otherwise, merge blink lists into ``epochs.metadata`` and
        return ``epochs``.

    Returns
    -------
    mne.Epochs or pandas.DataFrame
        Updated ``epochs`` with blink metadata or a long-format blink table
        depending on ``long_format``.
    """
    logger.info("Running refined-epoch blink property computation")
    blink_epochs = compute_from_refined_epochs(
        epochs=epochs,
        params=params,
        channel=channel,
        progress_bar=progress_bar,
    )
    blink_df = blink_epochs.metadata.copy()
    filtered_df = attach_blink_metadata(epochs, blink_df)
    return filtered_df if long_format else epochs


# ------------------------------ Refined epochs path ---------------------------


def compute_from_refined_epochs(
    epochs: mne.Epochs,
    params: Dict[str, Any],
    channel: str | Sequence[str],
    progress_bar: bool,
) -> mne.Epochs:
    """Compute blink properties when given refined :class:`mne.Epochs`."""
    ch_names = resolve_channels(epochs.ch_names, channel)
    validate_channels_exist(epochs.ch_names, ch_names)

    sfreq = float(epochs.info["sfreq"])
    n_epochs = len(epochs)
    n_times = epochs.get_data(picks=[ch_names[0]]).shape[-1] if n_epochs else 0
    data = epochs.get_data(picks=ch_names)

    tasks = build_epoch_channel_tasks(n_epochs, ch_names)

    records: List[pd.DataFrame] = []
    logger.info(
        "Computing blink properties for %d epochs across %d channels",
        n_epochs,
        len(ch_names),
    )

    iterator: Iterable[Tuple[int, int, str]] = tqdm(
        tasks, desc="Epoch×Channel", disable=not progress_bar
    )
    for ei, ci, ch in iterator:
        metadata_row = safe_metadata_row(epochs.metadata, ei)
        signal = data[ei, ci]
        mod = infer_modality_from_channel(ch)

        sample_windows = _sample_windows_from_metadata(
            metadata_row, ch, sfreq, n_times, ei
        )
        if not sample_windows:
            logger.debug("No sample windows for epoch %d channel %s", ei, ch)
            continue

        rows = build_candidate_rows_for_epoch_channel(
            signal=signal,
            sample_windows=sample_windows,
            metadata_row=metadata_row,
            modality=mod,
            channel_name=ch,
        )
        if rows.empty:
            continue

        rows = attach_zero_crossings(signal, rows)
        props = fit_and_extract_properties(signal, rows, sfreq, params, run_fit=False)
        if props is None or props.empty:
            continue

        props["seg_id"] = ei
        props["blink_id"] = range(len(props))
        records.append(props)

    if not records:
        info = mne.create_info(ch_names, sfreq)
        empty_md = pd.DataFrame()
        return mne.EpochsArray(np.zeros((0, len(ch_names), 1)), info, metadata=empty_md)

    result = pd.concat(records, ignore_index=True)
    info = mne.create_info(ch_names, sfreq)
    dummy = np.zeros((len(result), len(ch_names), 1), dtype=float)
    return mne.EpochsArray(dummy, info, metadata=result)


def resolve_channels(available: Sequence[str], channel: str | Sequence[str]) -> List[str]:
    """Normalize the channel input into a list of channel names."""
    return [channel] if isinstance(channel, str) else list(channel)


def validate_channels_exist(available: Sequence[str], requested: Sequence[str]) -> None:
    """Raise if any requested channels are not present."""
    missing = [ch for ch in requested if ch not in available]
    if missing:
        raise ValueError(f"Channels not found: {missing}")


def build_epoch_channel_tasks(
    n_epochs: int, ch_names: Sequence[str]
) -> List[Tuple[int, int, str]]:
    """Create a flattened ``(epoch_index, channel_index, channel_name)`` list."""
    return [
        (ei, ci, ch_names[ci])
        for ei, ci in itertools.product(range(n_epochs), range(len(ch_names)))
    ]


def safe_metadata_row(metadata: pd.DataFrame | None, ei: int) -> pd.Series:
    """Safely access a metadata row; return empty Series if metadata is ``None``."""
    return metadata.iloc[ei] if isinstance(metadata, pd.DataFrame) else pd.Series(dtype=float)


def infer_modality_from_channel(ch_name: str) -> str:
    """Infer modality label (``'eeg'``, ``'eog'``, or ``'ear'``) from a channel name."""
    lower = ch_name.lower()
    if "eeg" in lower:
        return "eeg"
    if "eog" in lower:
        return "eog"
    return "ear"


def build_candidate_rows_for_epoch_channel(
    signal: np.ndarray,
    sample_windows: Sequence[slice],
    metadata_row: pd.Series,
    modality: str,
    channel_name: str,
) -> pd.DataFrame:
    """Construct a candidate blink DataFrame for one ``(epoch, channel)``."""
    starts, ends = window_starts_ends(sample_windows)
    outer_starts, outer_ends = outer_bounds_from_metadata(
        metadata_row, modality, len(starts), starts, ends
    )
    is_ear = modality == "ear"
    max_blinks, max_values = peaks_for_windows(signal, starts, ends, is_ear)

    return pd.DataFrame(
        {
            "start_blink": starts,
            "end_blink": ends,
            "outer_start": outer_starts,
            "outer_end": outer_ends,
            "max_blink": max_blinks,
            "max_value": max_values,
            "channel": channel_name,
            "modality": modality,
        }
    )


def window_starts_ends(sample_windows: Sequence[slice]) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of window start and end (inclusive) indices."""
    starts = np.fromiter((sl.start for sl in sample_windows), dtype=int, count=len(sample_windows))
    ends = np.fromiter((sl.stop - 1 for sl in sample_windows), dtype=int, count=len(sample_windows))
    return starts, ends


def outer_bounds_from_metadata(
    metadata_row: pd.Series,
    modality: str,
    n_windows: int,
    default_starts: np.ndarray,
    default_ends: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect per-window outer bounds from metadata or fall back to defaults."""
    start_key = f"blink_outer_start_{modality}"
    end_key = f"blink_outer_end_{modality}"
    raw_starts = metadata_row.get(start_key, [])
    raw_ends = metadata_row.get(end_key, [])

    starts_arr = normalize_seq(raw_starts, n_windows, None)
    ends_arr = normalize_seq(raw_ends, n_windows, None)

    starts = np.where(isnan_or_none(starts_arr), default_starts, starts_arr).astype(int)
    ends = np.where(isnan_or_none(ends_arr), default_ends, ends_arr).astype(int)
    return starts, ends


def normalize_seq(val: Any, n: int, default_value: Any) -> np.ndarray:
    """Normalize a scalar/list/array/Series to a numpy array of length ``n``."""
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(val)
    else:
        arr = np.asarray([val])
    if arr.size == 0:
        arr = np.asarray([default_value])
    if arr.size < n:
        pad = np.full(n - arr.size, default_value, dtype=object)
        arr = np.concatenate([arr, pad])
    elif arr.size > n:
        arr = arr[:n]
    return arr


def isnan_or_none(arr: np.ndarray) -> np.ndarray:
    """Return boolean mask of elements that are ``None`` or ``NaN``."""
    if arr.dtype == object:
        return np.vectorize(lambda x: x is None or (isinstance(x, float) and np.isnan(x)))(arr)
    return np.isnan(arr)


def peaks_for_windows(
    signal: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    is_ear: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute peak indices and values for each window."""
    peak_indices_abs, peak_values = [], []
    use_arg = np.argmin if is_ear else np.argmax
    for s, e in zip(starts, ends):
        seg = signal[s : e + 1]
        offset = int(use_arg(seg))
        idx = s + offset
        peak_indices_abs.append(idx)
        peak_values.append(float(signal[idx]))
    return np.asarray(peak_indices_abs, dtype=int), np.asarray(peak_values, dtype=float)


def zero_crossing_for_row(signal: np.ndarray, row: pd.Series) -> Tuple[float, float]:
    """Return left/right zero-crossing indices for a blink row."""
    try:
        left, right = left_right_zero_crossing(
            signal,
            int(row["max_blink"]),
            int(row["outer_start"]),
            int(row["outer_end"]),
        )
        right_val = np.nan if right is None else float(right)
        return float(left), right_val
    except Exception:
        return np.nan, np.nan


def attach_zero_crossings(signal: np.ndarray, rows: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach left/right zero crossing columns."""
    zeros = rows.apply(
        lambda r: zero_crossing_for_row(signal, r), axis=1, result_type="expand"
    )
    rows = rows.copy()
    rows["left_zero"] = zeros[0]
    rows["right_zero"] = zeros[1]
    return rows


def fit_and_extract_properties(
    signal: np.ndarray,
    rows: pd.DataFrame,
    sfreq: float,
    params: Dict[str, Any],
    run_fit: bool,
) -> pd.DataFrame | None:
    """Run :class:`FitBlinks` and :class:`BlinkProperties` to obtain metrics."""
    fitter = FitBlinks(candidate_signal=signal, df=rows.copy(), params=params)
    try:
        fitter.dprocess_segment_raw(run_fit=run_fit)
    except Exception:
        return None

    frame_blinks = getattr(fitter, "frame_blinks", None)
    if frame_blinks is None or frame_blinks.empty:
        return None

    return BlinkProperties(signal, frame_blinks, sfreq, params, fitted=run_fit).df


