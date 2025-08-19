import unittest
import numpy as np
import pandas as pd
from unit_test.blinker_migration.debugging_tools import load_matlab_data
from pyblinker.utils.blink_statistics import get_good_blink_mask
from unit_test.blinker_migration.pyblinker.utils.update_pkl_variables import RENAME_MAP
from pathlib import Path


class TestGetGoodBlinkMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by loading input and ground truth candidate_signal.
        """
        base_path = Path(__file__).resolve().parents[1] / 'migration_files'
        cls.mat_file_path_input = base_path / 'step2b_data_input_getGoodBlinkMask.mat'
        cls.mat_file_path_output = base_path / 'step2b_data_output_getGoodBlinkMask.mat'

        # Load candidate_signal
        input_data, output_datax = load_matlab_data(cls.mat_file_path_input, cls.mat_file_path_output)
        cls.input_data = input_data
        # Ground truth good blink mask from MATLAB
        cls.good_blink_mask_output = output_datax['goodBlinkMask'].astype(bool)

        # Blink fits as DataFrame
        cls.blink_fits = pd.DataFrame.from_records(cls.input_data['blinkFits'])
        cls.blink_fits.rename(columns=RENAME_MAP, inplace=True)

        # Use fixed z_thresholds instead of MATLAB values
        cls.z_thresholds = np.array([[0.9, 0.98], [2.0, 5.0]])

    def test_good_blink_mask(self):
        """
        Test the get_good_blink_mask function output against the MATLAB ground truth.
        """
        # Compute good blink mask and selected DataFrame
        good_blink_mask, selected_df = get_good_blink_mask(
            self.blink_fits,
            self.input_data['specifiedMedian'],
            self.input_data['specifiedStd'],
            self.z_thresholds
        )

        # Convert results to DataFrame for comparison
        comparison_df = pd.DataFrame({
            'good_blink_mask': good_blink_mask,
            'good_blink_mask_output': self.good_blink_mask_output
        })

        # Check for inconsistencies
        inconsistent = comparison_df.apply(
            lambda row: row['good_blink_mask'] != row['good_blink_mask_output'], axis=1
        ).any()

        # Log comparison details if inconsistencies exist
        if inconsistent:
            print("\nInconsistent Rows in Good Blink Mask:")
            print(comparison_df[comparison_df['good_blink_mask'] != comparison_df['good_blink_mask_output']])

        # Assert arrays are the same
        self.assertTrue(
            np.array_equal(good_blink_mask, self.good_blink_mask_output),
            "The calculated good_blink_mask does not match the MATLAB output."
        )


if __name__ == '__main__':
    unittest.main()
