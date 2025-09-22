"""Utility helpers for blink stroke analysis."""

from __future__ import annotations

import numpy as np

from pyblinker.logging import get_logger


logger = get_logger(__name__)


def get_up_down_stroke(max_blink, left_zero, right_zero):
    """Compute index ranges for the upward and downward blink strokes."""
    m_frame = int(max_blink)
    l_zero = int(left_zero)
    r_zero = int(right_zero)

    up_stroke = np.arange(l_zero, m_frame + 1)
    down_stroke = np.arange(m_frame, r_zero + 1)
    return up_stroke, down_stroke


def max_pos_vel_frame(blink_velocity, max_blink, left_zero, right_zero):
    """Locate frames with maximum positive and negative blink velocities."""
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
        logger.warning(
            "Down-stroke segment empty; forcing NaN for max negative velocity",
            extra={"down_stroke_size": int(down_stroke.size)},
        )
        max_neg_vel_frame = np.nan

    return max_pos_vel_frame, max_neg_vel_frame
