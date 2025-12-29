from __future__ import annotations

import numpy as np
import pytest

from pyblinker.fitutils.ear_crossing import (
    ThresholdCrossingError,
    compute_threshold_slopes,
    find_threshold_crossing_triplet,
    linear_interpolated_crossing,
)


def test_basic_crossing_and_slopes() -> None:
    t = np.linspace(0.0, 0.5, 6)
    ear = np.array([0.32, 0.26, 0.14, 0.18, 0.31, 0.33])
    theta = 0.2

    result = find_threshold_crossing_triplet(ear, theta, t=t, window=(0, len(ear) - 1))
    closing, opening = compute_threshold_slopes(result, theta)

    assert result.left.time < result.minimum_time < result.right.time
    assert closing < 0
    assert opening > 0
    assert result.status == "ok"


def test_crossings_require_expansion() -> None:
    t = np.arange(0.0, 0.6, 0.1)
    ear = np.array([0.35, 0.3, 0.18, 0.17, 0.28, 0.32])
    theta = 0.2

    # Window misses the downward crossing at index 1; expansion must reach it.
    result = find_threshold_crossing_triplet(
        ear,
        theta,
        t=t,
        window=(2, len(ear) - 1),
        max_expansion=2,
        expansion_step=1,
    )
    assert result.found_by == "expanded"
    assert result.left.index <= 1
    assert result.right.index >= result.minimum_index


def test_no_crossings_even_with_expansion() -> None:
    ear = np.full(10, 0.35)
    theta = 0.2
    with pytest.raises(ThresholdCrossingError):
        find_threshold_crossing_triplet(ear, theta, max_expansion=3)


def test_multiple_noisy_crossings_picks_first_valid_triplet() -> None:
    t = np.arange(0.0, 0.6, 0.1)
    ear = np.array([0.31, 0.22, 0.25, 0.18, 0.17, 0.22])
    theta = 0.2

    result = find_threshold_crossing_triplet(ear, theta, t=t, window=(0, len(ear) - 1))

    assert result.left.index == 2  # first downward crossing
    assert result.right.index == 4  # first upward after the minimum
    assert result.minimum_index == 4
    assert result.minimum_value == ear[4]


def test_plateau_policy_midpoint() -> None:
    t_left = 0.0
    t_right = 1.0
    y0 = y1 = 0.2
    theta = 0.2
    crossing = linear_interpolated_crossing(t_left, y0, t_right, y1, theta)
    assert crossing == pytest.approx(0.5)


def test_zero_denominator_slopes_return_nan() -> None:
    t = np.array([0.0, 1.0, 2.0])
    ear = np.array([0.3, 0.1, 0.3])
    theta = 0.2

    result = find_threshold_crossing_triplet(ear, theta, t=t, window=(0, 2))
    # Force minimum time to coincide with left crossing time
    result.minimum_time = result.left.time
    closing, opening = compute_threshold_slopes(result, theta)
    assert np.isnan(closing)
    assert opening > 0
