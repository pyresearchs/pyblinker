import logging
from typing import List, Optional, Sequence, Tuple, Dict

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def left_right_zero_crossing(
    candidate_signal: np.ndarray,
    max_blink: float,
    outer_start: float,
    outer_end: float,
    *,
    signal_type: str = "eeg",
) -> tuple[int, Optional[int]]:
    """Find the nearest zero-crossing indices around ``max_blink``.

    The search is performed in the range ``[outer_start, max_blink)`` on the
    left and ``(max_blink, outer_end]`` on the right. If no negative-valued
    sample is found in the initial window, the search is expanded toward the
    closest signal boundary.

    Parameters
    ----------
    candidate_signal : np.ndarray
        One-dimensional array representing the signal.
    max_blink : float
        Index of the peak around which zero-crossings are located.
    outer_start : float
        Lower bound of the left-side search region.
    outer_end : float
        Upper bound of the right-side search region.
    signal_type : str, optional
        Signal type. Defaults to ``"eeg"``. Other types are accepted, but this
        implementation is tuned for EEG signals and may yield inaccurate results
        otherwise.

    Returns
    -------
    tuple[int, Optional[int]]
        ``(left_zero, right_zero)`` indices. ``right_zero`` may be ``None`` if
        no negative sample exists to the right even after the fallback search.

    Raises
    ------
    ValueError
        If ``left_zero > max_blink`` or ``max_blink > right_zero`` when both
        zeros are found.
    """
    if signal_type.lower() != "eeg":
        logger.warning(
            "left_right_zero_crossing tuned for EEG signals; results may be inaccurate for %s",
            signal_type,
        )

    start_idx = int(outer_start)
    m_frame = int(max_blink)
    end_idx = int(outer_end)

    left_range = np.arange(start_idx, m_frame)
    left_values = candidate_signal[left_range]
    s_ind_left_zero = np.flatnonzero(left_values < 0)

    if s_ind_left_zero.size > 0:
        left_zero = int(left_range[s_ind_left_zero[-1]])
    else:
        full_left_range = np.arange(0, m_frame).astype(int)
        left_neg_idx = np.flatnonzero(candidate_signal[full_left_range] < 0)
        left_zero = int(full_left_range[left_neg_idx[-1]])

    right_range = np.arange(m_frame, end_idx)
    right_values = candidate_signal[right_range]
    s_ind_right_zero = np.flatnonzero(right_values < 0)

    if s_ind_right_zero.size > 0:
        right_zero = int(right_range[s_ind_right_zero[0]])
    else:
        try:
            extreme_outer = np.arange(m_frame, candidate_signal.shape[0]).astype(int)
        except TypeError:
            return left_zero, None

        s_ind_right_zero_ex = np.flatnonzero(candidate_signal[extreme_outer] < 0)
        if s_ind_right_zero_ex.size > 0:
            right_zero = int(extreme_outer[s_ind_right_zero_ex[0]])
        else:
            return left_zero, None

    if left_zero > m_frame:
        raise ValueError(
            "Validation error: left_zero = {left_zero}, max_blink = {max_blink}."
            " Ensure left_zero <= max_blink."
        )

    if m_frame > right_zero:
        raise ValueError(
            "Validation error: max_blink = {max_blink}, right_zero = {right_zero}."
            " Ensure max_blink <= right_zero."
        )

    return left_zero, right_zero


def compute_outer_bounds(peaks: Sequence[int], n_samples: int) -> List[Tuple[int, int]]:
    """Compute search windows around consecutive blink peaks.

    Each blink peak is assigned an ``outer_start`` and ``outer_end`` index such
    that windows do not overlap and cover the entire signal. The first blink's
    window begins at sample ``0`` and extends up to the next peak. The last
    blink's window ends at ``n_samples - 1``. All intermediate blinks span from
    the previous peak to the next peak.

    Parameters
    ----------
    peaks : sequence of int
        Peak indices sorted in ascending order.
    n_samples : int
        Number of samples in the underlying signal.

    Returns
    -------
    list of tuple[int, int]
        ``(outer_start, outer_end)`` pairs for each peak.
    """

    bounds: List[Tuple[int, int]] = []
    for i, max_blink in enumerate(peaks):
        outer_start = 0 if i == 0 else peaks[i - 1]
        outer_end = (n_samples - 1) if i == len(peaks) - 1 else peaks[i + 1]
        bounds.append((outer_start, outer_end))
    return bounds


def _filter_blink_annotations(
    raw: mne.io.BaseRaw, blink_label: str | None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract blink start and end sample indices from ``raw``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw segment containing blink annotations.
    blink_label : str | None
        Annotation label that denotes blinks. ``None`` keeps all annotations.

    Returns
    -------
    tuple of ndarray
        Arrays of start and end sample indices for each blink.
    """

    sfreq = raw.info["sfreq"]
    ann = raw.annotations
    mask = np.ones(len(ann), dtype=bool)
    if blink_label is not None:
        mask &= ann.description == blink_label
    mask &= ann.onset > raw.first_time

    onsets = ann.onset[mask]
    durations = ann.duration[mask]
    starts = ((onsets - raw.first_time) * sfreq).astype(int)
    ends = ((onsets + durations - raw.first_time) * sfreq).astype(int)
    return starts, ends


def _get_channel_type(
    raw: mne.io.BaseRaw, channel: str, provided: str | None
) -> str:
    """Determine ``channel`` type, optionally using the caller-provided value.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw segment from which to infer the channel type.
    channel : str
        Name of the channel to query.
    provided : str | None
        Channel type supplied by the caller. When not ``None`` this value is
        returned directly.

    Returns
    -------
    str
        Detected channel type. If detection fails a warning is logged and
        ``"eeg"`` is returned.
    """

    if provided is not None:
        return provided

    try:
        ch_type = raw.get_channel_types(picks=channel)[0]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Unable to determine channel type for %s: %s. Assuming 'eeg'",
            channel,
            exc,
        )
        ch_type = "eeg"
    return ch_type


def _detect_peaks(
    signal: np.ndarray, starts: np.ndarray, ends: np.ndarray, ch_type: str
) -> List[int]:
    """Detect the peak sample within each blink interval.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional blink signal.
    starts, ends : np.ndarray
        Start and end sample indices for each blink annotation.
    ch_type : str
        Type of the underlying channel (e.g., ``"eeg"``).

    Returns
    -------
    list of int
        Peak index for every blink interval.
    """

    peaks: List[int] = []
    for start, end in zip(starts, ends):
        segment = signal[start : end + 1]
        if ch_type == "eeg":
            peak = int(np.argmax(segment) + start)
        else:
            logger.warning(
                "Peak detection tuned for EEG; using absolute max for %s", ch_type
            )
            peak = int(np.argmax(np.abs(segment)) + start)
        peaks.append(peak)
    return peaks


def _process_segment_blinks(
    seg_id: int,
    raw: mne.io.BaseRaw,
    channel: str,
    blink_label: str | None,
    channel_type: str | None,
    *,
    progress_bar: bool = True,
) -> List[Dict[str, int | None]]:
    """Extract blink information from one raw segment.

    Parameters
    ----------
    seg_id : int
        Index of the segment within ``segments``.
    raw : mne.io.BaseRaw
        Segment containing the blink annotations and data.
    channel : str
        Channel name used for blink detection.
    blink_label : str | None
        Annotation label that marks blinks. ``None`` keeps all annotations.
    channel_type : str | None
        Optional override for the channel type. If ``None`` the type is
        determined from ``raw``.

    Returns
    -------
    list of dict
        Rows describing each detected blink with sample indices.
    """

    signal = raw.get_data(picks=channel)[0]
    ch_type = _get_channel_type(raw, channel, channel_type)

    starts, ends = _filter_blink_annotations(raw, blink_label)
    peaks = _detect_peaks(signal, starts, ends, ch_type)
    bounds = compute_outer_bounds(peaks, len(signal))

    rows: List[Dict[str, int | None]] = []
    for blink_id, (start, end, peak), (outer_start, outer_end) in tqdm(
        zip(range(len(peaks)), zip(starts, ends, peaks), bounds),
        desc=f"Seg {seg_id} blinks",
        leave=False,
        disable=not progress_bar,
    ):
        left_zero, right_zero = left_right_zero_crossing(
            signal,
            peak,
            outer_start,
            outer_end,
            signal_type=ch_type,
        )
        rows.append(
            {
                "seg_id": seg_id,
                "blink_id": blink_id,
                "start_blink": int(start),
                "max_blink": int(peak),
                "end_blink": int(end),
                "outer_start": int(outer_start),
                "outer_end": int(outer_end),
                "left_zero": int(left_zero),
                "right_zero": None if right_zero is None else int(right_zero),
            }
        )
    return rows


def extract_blink_events_dataframe(
    segments: Sequence[mne.io.BaseRaw],
    *,
    channel: str = "EEG-E8",
    blink_label: str | None = "blink",
    channel_type: str | None = None,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """Create a blink event summary for the provided raw segments.

    Parameters
    ----------
    segments : sequence of mne.io.BaseRaw
        Iterable of equally sized raw segments with blink annotations.
    channel : str, optional
        Channel used for blink detection. Defaults to ``"EEG-E8"``.
    blink_label : str | None, optional
        Annotation label that denotes blinks. ``None`` uses all annotations.
    channel_type : str | None, optional
        Explicit channel type. When ``None`` the type is obtained from each
        segment and a warning is emitted if it cannot be determined.

    Returns
    -------
    pandas.DataFrame
        One row per detected blink with sample index information.
    """

    logger.info("Extracting blink events from %d segments", len(segments))
    rows: List[Dict[str, int | None]] = []

    for seg_id, raw in enumerate(
        tqdm(segments, desc="Processing segments", disable=not progress_bar)
    ):
        rows.extend(
            _process_segment_blinks(
                seg_id,
                raw,
                channel,
                blink_label,
                channel_type,
                progress_bar=progress_bar,
            )
        )

    df = pd.DataFrame(rows)
    logger.info("Extracted %d blink events", len(df))
    logger.debug("Blink events preview:\n%s", df.head())
    return df


# Backwards compatibility
generate_blink_dataframe = extract_blink_events_dataframe
