"""Line intersection utilities replicating MATLAB behavior in Python.

This module provides the :func:`lines_intersection` function which wraps
MATLAB-style helpers to compute line intersections and related metrics.
"""

import numpy as np
from pyblinker.fitutils.forking import (
    corr,
    polyval,
    polyfit,
    get_intersection,
)
from pyblinker.blinker.zero_crossing import get_line_intersection_slope


def lines_intersection(
    *,
    signal: np.ndarray | None = None,
    x_right: np.ndarray | None = None,
    x_left: np.ndarray | None = None,
) -> tuple[float, ...]:
    """Return intersection metrics for the given signal segments.

    Parameters
    ----------
    signal : np.ndarray, optional
        Full signal array containing blink data.
    x_right : np.ndarray, optional
        Sample indices of the right-side segment.
    x_left : np.ndarray, optional
        Sample indices of the left-side segment.

    Returns
    -------
    tuple
        ``(
        left_slope, right_slope, aver_left_velocity, aver_right_velocity,
        right_r2, left_r2, x_intersect, y_intersect,
        left_x_intercept, right_x_intercept
        )``.

    Notes
    -----
    The original MATLAB code also returned ``x_line_cross_l``,
    ``y_line_cross_l``, ``x_line_cross_r``, and ``y_line_cross_r`` as part of
    the results.  These values were always ``NaN`` and are therefore omitted
    from the Python implementation.
    """

    y_right = signal[x_right]
    y_left = signal[x_left]

    degree = 1
    p_left, s_left, mu_left = polyfit(x_left, y_left, degree)
    y_pred_left, _ = polyval(p_left, x_left, S=s_left, mu=mu_left)
    left_r2, _ = corr(y_left, y_pred_left)

    p_right, s_right, mu_right = polyfit(x_right, y_right, 1)
    y_pred_right, _ = polyval(p_right, x_right, S=s_right, mu=mu_right)
    right_r2, _ = corr(y_right, y_pred_right)

    (
        x_intersect,
        y_intersect,
        left_x_intercept,
        right_x_intercept,
    ) = get_intersection(p_left, p_right, mu_left, mu_right)

    left_slope, right_slope = get_line_intersection_slope(
        x_intersect, y_intersect, left_x_intercept, right_x_intercept
    )

    aver_left_velocity = p_left[0] / mu_left[1]
    aver_right_velocity = p_right[0] / mu_right[1]

    return (
        left_slope,
        right_slope,
        aver_left_velocity,
        aver_right_velocity,
        right_r2[0][0],
        left_r2[0][0],
        x_intersect,
        y_intersect,
        left_x_intercept,
        right_x_intercept,
    )
