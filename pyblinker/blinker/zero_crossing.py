import warnings

from typing import Tuple, Optional
import numpy as np


def get_line_intersection_slope(
    x_intersect, y_intersect, left_x_intersect, right_x_intersect
):
    """
    Original logic retained. Computes slopes at the intersection point.
    """
    # Local variable usage here is minimal since there's only two lines.
    left_slope = y_intersect / (x_intersect - left_x_intersect)
    right_slope = y_intersect / (x_intersect - right_x_intersect)
    return left_slope, right_slope


def get_average_velocity(p_left, p_right, x_left, x_right):
    """
    Original logic retained. Computes average velocities.
    """
    # Using local references is possible, but it's already short.
    aver_left_velocity = p_left.coef[1] / np.std(x_left)
    aver_right_velocity = p_right.coef[1] / np.std(x_right)
    return aver_left_velocity, aver_right_velocity


def left_right_zero_crossing(
    candidate_signal: np.ndarray,
    max_blink: float,
    outer_start: float,
    outer_end: float,
) -> Tuple[int, Optional[int]]:
    """
    Find the nearest zero-crossing indices to the left and right of a given peak in the signal.

    This function searches for the nearest zero-crossings in a 1D signal array around a specified
    `max_blink`. It looks to the left in the range [outer_start, max_blink) and to the right in the
    range (max_blink, outer_end].

    Parameters:
        candidate_signal (np.ndarray): 1D array representing the signal data.
        max_blink (float): The frame index of the peak (or maximum) to evaluate crossings around.
        outer_start (float): The lower bound index of the left-side search region.
        outer_end (float): The upper bound index of the right-side search region.

    Returns:
        Tuple[int, Optional[int]]: A tuple containing two values:
            - Left zero-crossing index (int): nearest zero crossing to the left of max_blink.
            - Right zero-crossing index (Optional[int]): nearest zero crossing to the right of max_blink,
              or None if not found even after fallback search.

    Raises:
        ValueError: If input index boundaries are invalid or logic assumptions break.
    """
    start_idx = int(outer_start)
    m_frame = int(max_blink)
    end_idx = int(outer_end)

    # Left side search
    left_range = np.arange(start_idx, m_frame)
    left_values = candidate_signal[left_range]
    s_ind_left_zero = np.flatnonzero(left_values < 0)

    if s_ind_left_zero.size > 0:
        left_zero = left_range[s_ind_left_zero[-1]]
    else:
        # Fall back if no negative crossing found in left_range
        full_left_range = np.arange(0, m_frame).astype(int)
        left_neg_idx = np.flatnonzero(candidate_signal[full_left_range] < 0)
        left_zero = (
            full_left_range[left_neg_idx[-1]] if left_neg_idx.size > 0 else np.nan
        )

    # Right side search
    right_range = np.arange(m_frame, end_idx)
    right_values = candidate_signal[right_range]
    s_ind_right_zero = np.flatnonzero(right_values < 0)

    if s_ind_right_zero.size > 0:
        right_zero = right_range[s_ind_right_zero[0]]
    else:
        # Extreme remedy by extending beyond outer_end to the max signal length
        try:
            extreme_outer = np.arange(m_frame, candidate_signal.shape[0]).astype(int)
        except TypeError:
            print("Error")
            # If this except triggers, raise or handle accordingly
            return left_zero, np.nan

        s_ind_right_zero_ex = np.flatnonzero(candidate_signal[extreme_outer] < 0)
        if s_ind_right_zero_ex.size > 0:
            right_zero = extreme_outer[s_ind_right_zero_ex[0]]
        else:
            return left_zero, np.nan

    if left_zero > m_frame:
        raise ValueError(
            "Validation error: left_zero = {left_zero}, max_blink = {max_blink}. Ensure left_zero <= max_blink."
        )

    if m_frame > right_zero:
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


def get_left_base(blink_velocity, left_outer, max_pos_vel_frame):
    """
    Determine the left base index from left_outer to max_pos_vel_frame
    by searching for where blink_velocity crosses <= 0.
    """
    l_outer = int(left_outer)
    m_pos_vel = int(max_pos_vel_frame)

    left_range = np.arange(l_outer, m_pos_vel + 1)
    reversed_velocity = np.flip(blink_velocity[left_range])

    left_base_index = np.argmax(reversed_velocity <= 0)
    left_base = m_pos_vel - left_base_index - 1
    return left_base


def get_right_base(candidate_signal, blink_velocity, right_outer, max_neg_vel_frame):
    """Compute the right base frame index.

    Parameters
    ----------
    candidate_signal : numpy.ndarray
        The filtered blink signal from which baseline metrics are derived.
    blink_velocity : numpy.ndarray
        First derivative of ``candidate_signal`` used to locate zero crossings.
    right_outer : int
        Right boundary frame of the blink segment.
    max_neg_vel_frame : float | int | numpy.nan
        Frame index corresponding to the most negative blink velocity.

    Returns
    -------
    int | float
        The frame index of the right base, or ``numpy.nan`` when
        ``max_neg_vel_frame`` is ``NaN``.
    """
    r_outer = int(right_outer)

    """Return ``NaN`` when no negative velocity peak exists."""
    # If the maximum negative velocity frame is undefined (NaN),
    # the right base cannot be determined, so return NaN to
    # indicate that the segment should be ignored by downstream logic.
    if np.isnan(max_neg_vel_frame):
        return np.nan
    m_neg_vel = int(max_neg_vel_frame)

    # Ensure boundaries are valid
    if m_neg_vel > r_outer:
        return None

    max_size = candidate_signal.size
    end_idx = min(r_outer, max_size)
    right_range = np.arange(m_neg_vel, end_idx)

    if right_range.size == 0:
        return None

    # Avoid out-of-bounds indexing for blink_velocity
    if right_range[-1] >= blink_velocity.size:
        right_range = right_range[:-1]
        if right_range.size == 0 or right_range[-1] >= blink_velocity.size:
            # TODO: Handle this case more gracefully
            raise ValueError("Please strategies how to address this")

    right_base_velocity = blink_velocity[right_range]
    right_base_index = np.argmax(right_base_velocity >= 0)
    right_base = m_neg_vel + right_base_index + 1
    return right_base


def get_half_height(
    candidate_signal, max_blink, left_zero, right_zero, left_base, right_outer
):
    """
    left_base_half_height:
        The coordinate of the signal halfway (in height) between
        the blink maximum and the left base value.
    right_base_half_height:
        The coordinate of the signal halfway (in height) between
        the blink maximum and the right base value.
    """
    m_frame = int(max_blink)
    l_zero = int(left_zero)
    r_zero = int(right_zero)
    l_base = int(left_base)
    r_outer = int(right_outer)

    # Halfway point (vertical) from candidate_signal[max_blink] to candidate_signal[left_base]
    max_val = candidate_signal[m_frame]
    left_base_val = candidate_signal[l_base]
    half_height_val = max_val - 0.5 * (max_val - left_base_val)

    # Left side half-height from base
    left_range = np.arange(l_base, m_frame + 1)
    left_vals = candidate_signal[left_range]
    left_index = np.argmax(left_vals >= half_height_val)
    left_base_half_height = l_base + left_index + 1

    # Right side half-height from base
    right_range = np.arange(m_frame, r_outer + 1)
    try:
        right_base_half_height = min(
            r_outer,
            np.argmax(candidate_signal[right_range] <= half_height_val) + m_frame,
        )
    except IndexError:
        # If out-of-bounds, reduce range by 1
        right_range = np.arange(m_frame, r_outer)
        right_base_half_height = min(
            r_outer,
            np.argmax(candidate_signal[right_range] <= half_height_val) + m_frame,
        )

    # Now compute the left and right half-height frames from zero
    # Halfway from candidate_signal[max_blink] down to 0 (the "zero" crossing region).
    # left_zero_half_height
    zero_half_val = 0.5 * max_val
    left_zero_range = np.arange(l_zero, m_frame + 1)
    left_zero_index = np.argmax(candidate_signal[left_zero_range] >= zero_half_val)
    left_zero_half_height = l_zero + left_zero_index + 1

    # right_zero_half_height
    right_zero_range = np.arange(m_frame, r_zero + 1)
    right_zero_index = np.argmax(candidate_signal[right_zero_range] <= zero_half_val)
    right_zero_half_height = min(r_outer, m_frame + right_zero_index)

    return (
        left_zero_half_height,
        right_zero_half_height,
        left_base_half_height,
        right_base_half_height,
    )


def get_left_range(left_zero, max_blink, candidate_signal, blink_top, blink_bottom):
    """
    Identify the left blink range based on blink_top/blink_bottom thresholds
    within candidate_signal.
    """
    l_zero = int(left_zero)
    m_frame = int(max_blink)

    blink_range = np.arange(l_zero, m_frame + 1, dtype=int)
    cand_slice = candidate_signal[blink_range]

    # Indices where candidate_signal < blink_top
    top_idx = np.where(cand_slice < blink_top)[0]
    blink_top_point_idx = top_idx[-1]  # the last occurrence

    # Indices where candidate_signal > blink_bottom
    bottom_idx = np.flatnonzero(cand_slice > blink_bottom)
    blink_bottom_point_idx = bottom_idx[0]  # the first occurrence

    blink_top_point_l_x = blink_range[blink_top_point_idx]
    blink_top_point_l_y = candidate_signal[blink_top_point_l_x]

    blink_bottom_point_l_x = blink_range[blink_bottom_point_idx]
    blink_bottom_point_l_y = candidate_signal[blink_bottom_point_l_x]

    left_range = [blink_bottom_point_l_x, blink_top_point_l_x]

    return (
        left_range,
        blink_top_point_l_x,
        blink_top_point_l_y,
        blink_bottom_point_l_x,
        blink_bottom_point_l_y,
    )


def get_right_range(max_blink, right_zero, candidate_signal, blink_top, blink_bottom):
    """
    Identify the right blink range based on blink_top/blink_bottom thresholds
    within candidate_signal.
    """
    m_frame = int(max_blink)
    r_zero = int(right_zero)

    blink_range = np.arange(m_frame, r_zero + 1, dtype=int)
    cand_slice = candidate_signal[blink_range]

    # Indices where candidate_signal < blink_top
    top_mask = cand_slice < blink_top
    blink_top_point_r = np.argmax(top_mask)  # first True

    # Indices where candidate_signal > blink_bottom
    bottom_mask = cand_slice > blink_bottom
    bottom_idx = np.where(bottom_mask)[0]
    blink_bottom_point_r = bottom_idx[-1]  # last True

    blink_top_point_r_x = blink_range[blink_top_point_r]
    blink_top_point_r_y = candidate_signal[blink_top_point_r_x]

    blink_bottom_point_r_x = blink_range[blink_bottom_point_r]
    blink_bottom_point_r_y = candidate_signal[blink_bottom_point_r_x]

    right_range = [blink_range[blink_top_point_r], blink_range[blink_bottom_point_r]]

    return (
        right_range,
        blink_top_point_r_x,
        blink_top_point_r_y,
        blink_bottom_point_r_x,
        blink_bottom_point_r_y,
    )


def compute_fit_range(
    candidate_signal, max_blink, left_zero, right_zero, base_fraction, top_bottom=None
):
    """
    Computes x_left, x_right, left_range, right_range,
    plus optional top/bottom blink points,
    for the candidate_signal around a blink event.
    """
    m_frame = int(max_blink)
    l_zero = int(left_zero)
    r_zero = int(right_zero)

    # Compute the blink_top/blink_bottom for thresholding
    blink_height = candidate_signal[m_frame] - candidate_signal[l_zero]
    blink_top = candidate_signal[m_frame] - base_fraction * blink_height
    blink_bottom = candidate_signal[l_zero] + base_fraction * blink_height

    (
        left_range,
        blink_top_point_l_x,
        blink_top_point_l_y,
        blink_bottom_point_l_x,
        blink_bottom_point_l_y,
    ) = get_left_range(l_zero, m_frame, candidate_signal, blink_top, blink_bottom)

    (
        right_range,
        blink_top_point_r_x,
        blink_top_point_r_y,
        blink_bottom_point_r_x,
        blink_bottom_point_r_y,
    ) = get_right_range(m_frame, r_zero, candidate_signal, blink_top, blink_bottom)

    # Create arrays for fitting
    x_left = np.arange(
        left_range[0], left_range[1] + 1, dtype=int
    )  # +1 to include the last index
    x_right = np.arange(right_range[0], right_range[1] + 1, dtype=int)

    # Replace empty arrays with np.nan for consistency
    if x_left.size == 0:
        x_left = np.nan
    if x_right.size == 0:
        x_right = np.nan

    if top_bottom is None:
        # Return minimal information
        warnings.warn("To modify this so that all function return the top_bottom point")
        return x_left, x_right, left_range, right_range
    else:
        # Return extended info including top/bottom points
        return (
            x_left,
            x_right,
            left_range,
            right_range,
            blink_bottom_point_l_y,
            blink_bottom_point_l_x,
            blink_top_point_l_y,
            blink_top_point_l_x,
            blink_bottom_point_r_x,
            blink_bottom_point_r_y,
            blink_top_point_r_x,
            blink_top_point_r_y,
        )
