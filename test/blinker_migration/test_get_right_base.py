import numpy as np
from pyblinker.blinker.zero_crossing import get_right_base


def test_get_right_base_returns_none_when_range_exceeds_velocity():
    """Return ``None`` when right_range overshoots blink_velocity."""
    candidate_signal = np.zeros(5)
    blink_velocity = np.diff(candidate_signal)
    result = get_right_base(
        candidate_signal=candidate_signal,
        blink_velocity=blink_velocity,
        right_outer=5,
        max_neg_vel_frame=4,
    )
    assert result is None

