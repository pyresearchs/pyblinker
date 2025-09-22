from pyblinker.logging import get_logger

import numpy as np


logger = get_logger(__name__)


def _find_left_zero_crossing(
    candidate_signal: np.ndarray,
    start_idx: int,
    m_frame: int,
) -> int | float:
    """Search left-to-right for the last negative sample before ``m_frame``.

    The search inspects ``[start_idx, m_frame)`` and falls back to the leading
    portion ``[0, m_frame)`` when no negative values are present.
    """
    left_range = np.arange(start_idx, m_frame, dtype=int)
    left_values = candidate_signal[left_range]
    s_ind_left_zero = np.flatnonzero(left_values < 0)

    if s_ind_left_zero.size > 0:
        return int(left_range[s_ind_left_zero[-1]])

    full_left_range = np.arange(0, m_frame, dtype=int)
    left_neg_idx = np.flatnonzero(candidate_signal[full_left_range] < 0)
    if left_neg_idx.size > 0:
        return int(full_left_range[left_neg_idx[-1]])

    return np.nan


def _find_right_zero_crossing(
    candidate_signal: np.ndarray,
    m_frame: int,
    end_idx: int,
) -> int | float:
    """Search rightward from ``m_frame`` for the first negative sample.

    The search inspects ``[m_frame, end_idx)`` and extends to the signal tail
    when no negative values are encountered within the window.
    """
    right_range = np.arange(m_frame, end_idx, dtype=int)
    right_values = candidate_signal[right_range]
    s_ind_right_zero = np.flatnonzero(right_values < 0)

    if s_ind_right_zero.size > 0:
        return int(right_range[s_ind_right_zero[0]])

    try:
        extreme_outer = np.arange(m_frame, candidate_signal.shape[0], dtype=int)
    except TypeError:
        logger.exception(
            "Failed to extend search range to signal boundary; returning NaN",
            extra={"max_blink": m_frame},
        )
        return np.nan

    s_ind_right_zero_ex = np.flatnonzero(candidate_signal[extreme_outer] < 0)
    if s_ind_right_zero_ex.size > 0:
        return int(extreme_outer[s_ind_right_zero_ex[0]])

    return np.nan


def left_right_zero_crossing(
    candidate_signal: np.ndarray,
    max_blink: float,
    outer_start: float,
    outer_end: float,
    *,
    signal_type: str | None = None,
) -> tuple[int | float, int | float]:
    """Locate zero-crossing indices immediately surrounding a blink peak.

    The search inspects the left side in ``[outer_start, max_blink)`` and the
    right side in ``(max_blink, outer_end]`` for the last/first negative sample
    respectively. If no candidate is found, the window is expanded to the signal
    boundaries. When a crossing still cannot be identified the function returns
    ``numpy.nan`` for the corresponding side so that downstream code can
    gracefully drop the blink.

    Parameters
    ----------
    candidate_signal
        One-dimensional array representing the blink signal.
    max_blink
        Frame index of the blink peak around which to search for crossings.
    outer_start, outer_end
        Search boundaries that delimit the blink window.
    signal_type
        Optional hint describing the modality of ``candidate_signal``. The
        implementation is tuned for EEG traces; other signal types trigger a
        warning but are still processed.

    Returns
    -------
    tuple[int | float, int | float]
        ``(left_zero, right_zero)`` indices. Either element may be
        ``numpy.nan`` when no valid crossing is located.

    Raises
    ------
    ValueError
        If the detected crossings violate the expected ordering relative to
        ``max_blink``.
    """
    if signal_type is not None and signal_type.lower() != "eeg":
        logger.warning(
            "left_right_zero_crossing tuned for EEG signals; results may be inaccurate for %s",
            signal_type,
        )

    start_idx = int(outer_start)
    m_frame = int(max_blink)
    end_idx = int(outer_end)

    left_zero = _find_left_zero_crossing(candidate_signal, start_idx, m_frame)
    right_zero = _find_right_zero_crossing(candidate_signal, m_frame, end_idx)

    if not np.isnan(left_zero) and left_zero > m_frame:
        raise ValueError(
            f"Validation error: left_zero = {left_zero}, max_blink = {m_frame}. Ensure left_zero <= max_blink."
        )

    if not np.isnan(right_zero) and m_frame > right_zero:
        raise ValueError(
            f"Validation error: max_blink = {m_frame}, right_zero = {right_zero}. Ensure max_blink <= right_zero."
        )

    return left_zero, right_zero
