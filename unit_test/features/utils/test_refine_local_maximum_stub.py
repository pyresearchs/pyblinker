import unittest
import numpy as np
from pyblinker.utils.refinement import refine_local_maximum_stub


class TestRefineLocalMaximumStub(unittest.TestCase):
    """Unit tests for refine_local_maximum_stub function."""

    def test_peak_within_cvat(self):
        """If CVAT peak within [start, end], it should be used directly."""
        signal = np.array([0.1, 0.4, 0.9, 0.3, 0.2])
        rs, peak, re = refine_local_maximum_stub(signal, 1, 3, peak_rel_cvat=2)
        self.assertEqual((rs, peak, re), (1, 2, 3))

    def test_peak_outside_range(self):
        """CVAT peak outside window; should pick the max in [1:3]."""
        signal = np.array([0.1, 0.4, 0.9, 0.3, 0.2])
        rs, peak, re = refine_local_maximum_stub(signal, 1, 3, peak_rel_cvat=4)
        # max in slice [1:4] is at index 2
        self.assertEqual((rs, peak, re), (1, 2, 3))

    def test_no_cvat_peak(self):
        """No CVAT peak provided; should find true maximum in full window."""
        signal = np.array([0.1, 0.7, 0.5, 0.9, 0.2])
        rs, peak, re = refine_local_maximum_stub(signal, 0, 4, peak_rel_cvat=None)
        # global max at index 3
        self.assertEqual((rs, peak, re), (0, 3, 4))

    def test_empty_signal(self):
        """Empty segment yields zeros for start, peak, and end."""
        signal = np.array([])
        rs, peak, re = refine_local_maximum_stub(signal, 0, 5)
        self.assertEqual((rs, peak, re), (0, 0, 0))

    def test_start_greater_than_end(self):
        """If start_rel > end_rel, both should clamp to the lower value."""
        signal = np.array([0, 1, 2, 3, 4])
        rs, peak, re = refine_local_maximum_stub(signal, 4, 2, peak_rel_cvat=None)
        # clamp window to [2,2], so max at 2
        self.assertEqual((rs, peak, re), (2, 2, 2))

    def test_negative_start_and_end(self):
        """Negative start and end beyond length should clamp to [0, n-1]."""
        signal = np.array([0, 5, 1, 4, 2])
        rs, peak, re = refine_local_maximum_stub(signal, -3, 10, peak_rel_cvat=None)
        # window becomes [0,4], max at index 1
        self.assertEqual((rs, peak, re), (0, 1, 4))

    def test_equal_maxima(self):
        """When multiple equal maxima, should pick the first occurrence."""
        signal = np.array([0, 2, 2, 1, 2])
        rs, peak, re = refine_local_maximum_stub(signal, 0, 4, peak_rel_cvat=None)
        # equal maxima at indices 1,2,4; pick first at 1
        self.assertEqual(peak, 1)

    def test_negative_values(self):
        """Signal with all negative values should find the least negative (max)."""
        signal = np.array([-3.0, -0.5, -2.0, -1.0])
        rs, peak, re = refine_local_maximum_stub(signal, 0, 3, peak_rel_cvat=None)
        # max is -0.5 at index 1
        self.assertEqual((rs, peak, re), (0, 1, 3))

if __name__ == '__main__':
    unittest.main()
