import unittest

import pandas as pd
from pathlib import Path

from pyblinker.blinker.get_blink_positions import get_blink_position
from test.blinker_migration.debugging_tools import load_matlab_data


class TestBlinkPosition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the class-level variables for testing.
        Load MATLAB candidate_signal and define parameters for the blink position calculation.
        """
        base_path = Path(__file__).resolve().parents[1] / 'migration_files'
        cls.mat_file_path_input = str(base_path / 'step1bi_data_input_getBlinkPositions.mat')
        cls.mat_file_path_output = str(base_path / 'step1bi_data_output_getBlinkPositions.mat')

        # Parameters for `get_blink_position`
        cls.params = dict(min_event_len=0.05, std_threshold=1.5, sfreq=100)

        # Load MATLAB input and ground truth candidate_signal
        cls.input_data, cls.output_data = load_matlab_data(
            cls.mat_file_path_input, cls.mat_file_path_output
        )
        cls.blinkposition_groundtruth = cls.output_data['blinkPositions']

    def calculate_blink_positions(self, input_data, params):
        """
        Calculate blink positions using the Python implementation.
        """
        blink_comp = input_data['blinkComp']
        # min_blink_frames=5.0
        # threshold=12.241726391783821
        blink_positions = get_blink_position(
            params, blink_component=blink_comp, ch='No_channel', progress_bar=False
        )
        return blink_positions

    def adjust_indices_for_matlab(self, blink_positions, shift_index=1):
        """
        Adjust indices for MATLAB's 1-based indexing.
        """
        blink_positions[['start_blink', 'end_blink']] = (
                blink_positions[['start_blink', 'end_blink']] + shift_index
        )
        return blink_positions

    def compare_dataframes(self, df1, df2):
        """
        Compare two DataFrames for equality.
        """
        column_diff = self.check_column_names(df1, df2)
        data_diff = self.check_data_equality(df1, df2)

        if column_diff or data_diff is not None:
            return {
                "column_difference": column_diff,
                "data_difference": data_diff
            }
        else:
            return None

    def check_column_names(self, df1, df2):
        """
        Check if column names in both DataFrames are the same.
        """
        if df1.columns.tolist() != df2.columns.tolist():
            return {
                "df1_columns": df1.columns.tolist(),
                "df2_columns": df2.columns.tolist()
            }
        return None

    def check_data_equality(self, df1, df2):
        """
        Check if the candidate_signal in both DataFrames is equal.
        """
        if not df1.equals(df2):
            return df1.compare(df2)
        return None

    def test_blink_position_comparison(self):
        """
        Test the blink position outputs of Python implementation against MATLAB ground truth.
        """
        # Calculate blink positions using Python implementation
        blink_positions = self.calculate_blink_positions(self.input_data, self.params)

        # Adjust indices for MATLAB compatibility
        blink_positions_py = self.adjust_indices_for_matlab(blink_positions)

        # Prepare MATLAB ground truth candidate_signal as a DataFrame
        blinkposition_mat = pd.DataFrame({
            'start_blink': self.blinkposition_groundtruth[0],
            'end_blink': self.blinkposition_groundtruth[1]
        })

        # Compare DataFrames
        comparison_result = self.compare_dataframes(
            blink_positions_py.astype(int), blinkposition_mat.astype(int)
        )

        # Assert that there are no differences
        self.assertIsNone(
            comparison_result,
            f"Differences found in blink positions: {comparison_result}"
        )


if __name__ == '__main__':
    unittest.main()
