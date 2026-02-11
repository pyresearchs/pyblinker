'''

We will the following input
test/migration_files/step1bi_data_input_getBlinkPositions.mat
that will be feed into the get_blink_position function
and we will compare the output against the expected output stored in
test/migration_files/step1bi_data_output_getBlinkPositions.mat
'''
import unittest

import numpy as np
import pandas as pd
from pathlib import Path

from pyblinker.blinker.get_blink_positions import get_blink_position
from test.blinker_migration.obs.debugging_tools import load_matlab_data


class TestGetBlinkPosition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load MATLAB input and expected output once for all tests
        base_path = Path(__file__).resolve().parents[1] / "migration_files"
        cls.mat_file_path_input = str(base_path / 'step1bi_data_input_getBlinkPositions.mat')
        cls.mat_file_path_output = str(base_path / 'step1bi_data_output_getBlinkPositions.mat')

        cls.input_data, cls.output_data = load_matlab_data(
            cls.mat_file_path_input, cls.mat_file_path_output
        )

        # Build params for get_blink_position
        # min_event_len is not stored in the .mat; we assume 0.05s as in existing tests
        cls.params = {
            'sfreq': float(cls.input_data['srate']),
            'std_threshold': float(cls.input_data['stdThreshold']),
            'min_event_len': 0.05,
        }

        # Expected blink positions from MATLAB ground truth (1-based indexing)
        cls.blink_positions_mat = cls.output_data['blinkPositions']

    def test_blink_detection(self):
        params = self.params
        blink_component = self.input_data['blinkComp']

        # Run the function (Python implementation uses 0-based indexing)
        result = get_blink_position(
            params=params,
            blink_component=blink_component,
            ch='No_channel',
            progress_bar=False,
        )

        # Check that the result is a DataFrame with expected columns
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), ['start_blink', 'end_blink'])

        # Convert MATLAB ground truth to DataFrame and adjust Python result to 1-based indexing
        expected_df = pd.DataFrame({
            'start_blink': self.blink_positions_mat[0],
            'end_blink': self.blink_positions_mat[1],
        })

        result_1_based = result.copy()
        result_1_based[['start_blink', 'end_blink']] = result_1_based[['start_blink', 'end_blink']] + 1

        # Ensure integer comparison
        result_vals_start = result_1_based['start_blink'].astype(int).values
        result_vals_end = result_1_based['end_blink'].astype(int).values
        expected_vals_start = np.asarray(expected_df['start_blink']).astype(int)
        expected_vals_end = np.asarray(expected_df['end_blink']).astype(int)

        # Compare arrays
        np.testing.assert_array_equal(result_vals_start, expected_vals_start)
        np.testing.assert_array_equal(result_vals_end, expected_vals_end)


if __name__ == '__main__':
    unittest.main()
