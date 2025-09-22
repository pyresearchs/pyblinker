from functools import wraps
import warnings
from pyblinker.logging import get_logger

import numpy as np

from .fit_range import (
    compute_fit_range as _compute_fit_range,
    get_left_range as _get_left_range,
    get_right_range as _get_right_range,
)


logger = get_logger(__name__)


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

    # Left side search
    left_range = np.arange(start_idx, m_frame, dtype=int)
    left_values = candidate_signal[left_range]
    s_ind_left_zero = np.flatnonzero(left_values < 0)

    if s_ind_left_zero.size > 0:
        left_zero: int | float = int(left_range[s_ind_left_zero[-1]])
    else:
        # Fall back if no negative crossing found in left_range
        full_left_range = np.arange(0, m_frame, dtype=int)
        left_neg_idx = np.flatnonzero(candidate_signal[full_left_range] < 0)
        if left_neg_idx.size > 0:
            left_zero = int(full_left_range[left_neg_idx[-1]])
        else:
            left_zero = np.nan

    # Right side search
    right_range = np.arange(m_frame, end_idx, dtype=int)
    right_values = candidate_signal[right_range]
    s_ind_right_zero = np.flatnonzero(right_values < 0)

    if s_ind_right_zero.size > 0:
        right_zero: int | float = int(right_range[s_ind_right_zero[0]])
    else:
        # Extreme remedy by extending beyond outer_end to the max signal length
        try:
            extreme_outer = np.arange(m_frame, candidate_signal.shape[0], dtype=int)
        except TypeError:
            print("Error")
            # If this except triggers, raise or handle accordingly
            return left_zero, np.nan

        s_ind_right_zero_ex = np.flatnonzero(candidate_signal[extreme_outer] < 0)
        if s_ind_right_zero_ex.size > 0:
            right_zero = int(extreme_outer[s_ind_right_zero_ex[0]])
        else:
            return left_zero, np.nan

    if not np.isnan(left_zero) and left_zero > m_frame:
        raise ValueError(
            "Validation error: left_zero = {left_zero}, max_blink = {max_blink}. Ensure left_zero <= max_blink."
        )

    if not np.isnan(right_zero) and m_frame > right_zero:
        raise ValueError(
            "Validation error: max_blink = {max_blink}, right_zero = {right_zero}. Ensure max_blink <= right_zero."
        )

    return left_zero, right_zero


def get_up_down_stroke(max_blink, left_zero, right_zero):
    """
    Compute the place of maximum positive and negative velocities.
    up_stroke is the interval between left_zero and max_blink,
    down_stroke is the interval between max_blink and right_zero.
    """
    m_frame = int(max_blink)
    l_zero = int(left_zero)
    r_zero = int(right_zero)

    up_stroke = np.arange(l_zero, m_frame + 1)
    down_stroke = np.arange(m_frame, r_zero + 1)
    return up_stroke, down_stroke


def max_pos_vel_frame(blink_velocity, max_blink, left_zero, right_zero):
    """
    In the context of *blink_velocity* time series,
    the `max_pos_vel_frame` and `max_neg_vel_frame` represent the indices where
    the *blink_velocity* reaches its maximum positive value and maximum negative value, respectively.
    """
    m_frame = int(max_blink)
    l_zero = int(left_zero)
    r_zero = int(right_zero)

    up_stroke, down_stroke = get_up_down_stroke(m_frame, l_zero, r_zero)

    # Maximum positive velocity in the up_stroke region
    max_pos_vel_idx = np.argmax(blink_velocity[up_stroke])
    max_pos_vel_frame = up_stroke[max_pos_vel_idx]

    # Maximum negative velocity in the down_stroke region, if it exists
    if down_stroke.size > 0:
        max_neg_vel_idx = np.argmin(blink_velocity[down_stroke])
        max_neg_vel_frame = down_stroke[max_neg_vel_idx]
    else:
        warnings.warn(
            "Force nan but require further investigation why happen like this"
        )
        max_neg_vel_frame = np.nan

    return max_pos_vel_frame, max_neg_vel_frame


@wraps(_get_left_range)
def get_left_range(left_zero, max_blink, candidate_signal, blink_top, blink_bottom):
    warnings.warn(
        "get_left_range has moved to pyblinker.blinker.fit_range.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _get_left_range(left_zero, max_blink, candidate_signal, blink_top, blink_bottom)


@wraps(_get_right_range)
def get_right_range(max_blink, right_zero, candidate_signal, blink_top, blink_bottom):
    warnings.warn(
        "get_right_range has moved to pyblinker.blinker.fit_range.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _get_right_range(max_blink, right_zero, candidate_signal, blink_top, blink_bottom)


@wraps(_compute_fit_range)
def compute_fit_range(
    candidate_signal, max_blink, left_zero, right_zero, base_fraction, top_bottom=None
):
    warnings.warn(
        "compute_fit_range has moved to pyblinker.blinker.fit_range.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _compute_fit_range(
        candidate_signal,
        max_blink,
        left_zero,
        right_zero,
        base_fraction,
        top_bottom,
    )
