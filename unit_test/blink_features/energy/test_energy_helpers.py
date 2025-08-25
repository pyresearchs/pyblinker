"""Unit tests for helper utilities in the energy feature module."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pyblinker.blink_features.energy.helpers import (
    _extract_blink_windows,
    _segment_to_samples,
    _safe_stats,
    _tkeo,
)


class TestEnergyHelpers(unittest.TestCase):
    """Verify behaviour of low-level helper functions."""

    def test_extract_blink_windows_handles_scalars_and_lists(self) -> None:
        """Both scalar and list metadata are converted to window tuples."""
        row = pd.Series(
            {
                "blink_onset": [0.1, 0.5],
                "blink_duration": [0.2, 0.1],
            }
        )
        windows = _extract_blink_windows(row)
        self.assertEqual(windows, [(0.1, 0.2), (0.5, 0.1)])
        row2 = pd.Series({"blink_onset": 0.3, "blink_duration": 0.2})
        self.assertEqual(_extract_blink_windows(row2), [(0.3, 0.2)])

    def test_segment_to_samples_clamps(self) -> None:
        """Sample slices are clamped to the epoch boundaries."""
        sl = _segment_to_samples(-0.1, 0.5, 100.0, 1000)
        self.assertEqual(sl.start, 0)
        self.assertEqual(sl.stop, 40)  # only 0.4 s fall within the epoch
        sl2 = _segment_to_samples(9.9, 5.0, 100.0, 1000)
        self.assertEqual(sl2.stop, 1000)

    def test_safe_stats_empty(self) -> None:
        """Empty lists result in NaN statistics."""
        stats = _safe_stats([])
        self.assertTrue(np.isnan(stats["mean"]))
        self.assertTrue(np.isnan(stats["std"]))
        self.assertTrue(np.isnan(stats["cv"]))

    def test_tkeo_computation(self) -> None:
        """TKEO returns zero at boundaries and expected inner values."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        psi = _tkeo(x)
        self.assertEqual(len(psi), 4)
        self.assertEqual(psi[0], 0.0)
        self.assertEqual(psi[-1], 0.0)
        self.assertAlmostEqual(psi[1], x[1] ** 2 - x[0] * x[2])


if __name__ == "__main__":
    unittest.main()
