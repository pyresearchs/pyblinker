"""Half-height landmark helpers for blink waveforms."""

from __future__ import annotations

import numpy as np


def get_half_height(
    candidate_signal: np.ndarray,
    max_blink,
    left_zero,
    right_zero,
    left_base,
    right_outer,
):
    """Locate half-height landmarks around a blink peak.

    Parameters
    ----------
    candidate_signal
        One-dimensional waveform containing the blink.
    max_blink
        Index of the blink maximum.
    left_zero, right_zero
        Indices of the zero crossings around ``max_blink``.
    left_base
        Baseline index on the left side of the blink.
    right_outer
        Right-most boundary used when scanning for half-height crossings.

    Returns
    -------
    tuple
        ``(left_zero_half_height, right_zero_half_height,
        left_base_half_height, right_base_half_height)`` indices describing the
        locations where the waveform crosses half the blink height relative to
        the zero- and base-line references.
    """

    m_frame = int(max_blink)
    l_zero = int(left_zero)
    r_zero = int(right_zero)
    l_base = int(left_base)
    r_outer = int(right_outer)

    max_val = candidate_signal[m_frame]
    left_base_val = candidate_signal[l_base]
    half_height_val = max_val - 0.5 * (max_val - left_base_val)

    left_range = np.arange(l_base, m_frame + 1)
    left_vals = candidate_signal[left_range]
    left_index = np.argmax(left_vals >= half_height_val)
    left_base_half_height = l_base + left_index + 1

    right_range = np.arange(m_frame, r_outer + 1)
    try:
        right_base_half_height = min(
            r_outer,
            np.argmax(candidate_signal[right_range] <= half_height_val) + m_frame,
        )
    except IndexError:
        right_range = np.arange(m_frame, r_outer)
        right_base_half_height = min(
            r_outer,
            np.argmax(candidate_signal[right_range] <= half_height_val) + m_frame,
        )

    zero_half_val = 0.5 * max_val
    left_zero_range = np.arange(l_zero, m_frame + 1)
    left_zero_index = np.argmax(candidate_signal[left_zero_range] >= zero_half_val)
    left_zero_half_height = l_zero + left_zero_index + 1

    right_zero_range = np.arange(m_frame, r_zero + 1)
    right_zero_index = np.argmax(candidate_signal[right_zero_range] <= zero_half_val)
    right_zero_half_height = min(r_outer, m_frame + right_zero_index)

    return (
        left_zero_half_height,
        right_zero_half_height,
        left_base_half_height,
        right_base_half_height,
    )

