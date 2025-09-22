"""Blink fit range helpers.

This module groups the helper functions that compute the sample ranges used
when fitting blink segments.  The logic was previously embedded inside
``zero_crossing`` but has been extracted so that range computations live in a
dedicated, easier to maintain home.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["compute_fit_range", "get_left_range", "get_right_range"]


def get_left_range(left_zero, max_blink, candidate_signal, blink_top, blink_bottom):
    """Identify the left blink range using the provided thresholds."""

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
    """Identify the right blink range using the provided thresholds."""

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
    """Compute the blink fitting ranges around a blink event."""

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

