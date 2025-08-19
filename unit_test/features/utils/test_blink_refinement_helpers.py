import unittest
from pyblinker.utils.blink_refinement_helpers import group_refined_by_epoch

class TestGroupRefinedByEpoch(unittest.TestCase):
    def test_grouping(self):
        # Input: list of blink events
        test_data = [
            {'epoch_index': 0, 'refined_start_frame': 100, 'refined_peak_frame': 110, 'refined_end_frame': 120},
            {'epoch_index': 0, 'refined_start_frame': 130, 'refined_peak_frame': 140, 'refined_end_frame': 150},
            {'epoch_index': 1, 'refined_start_frame': 900, 'refined_peak_frame': 910, 'refined_end_frame': 920},
        ]

        # Expected output
        expected_output = {
            0: [
                {'epoch_index': 0, 'refined_start_frame': 100, 'refined_peak_frame': 110, 'refined_end_frame': 120},
                {'epoch_index': 0, 'refined_start_frame': 130, 'refined_peak_frame': 140, 'refined_end_frame': 150}
            ],
            1: [
                {'epoch_index': 1, 'refined_start_frame': 900, 'refined_peak_frame': 910, 'refined_end_frame': 920}
            ]
        }

        result = group_refined_by_epoch(test_data)
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()
