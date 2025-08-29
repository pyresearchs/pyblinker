import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from pyblinker.utils.blink_statistics import get_blink_statistic
from test.blinker_migration.debugging_tools import load_matlab_data
from test.blinker_migration.pyblinker.utils.update_pkl_variables import RENAME_MAP, rename_keys


class TestBlinkStatistic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by loading input and ground truth candidate_signal.
        """
        base_path = Path(__file__).resolve().parents[1] / 'migration_files'
        cls.mat_file_path_input = base_path / 'step1bii_v_input_blinkStatProperties.mat'
        cls.mat_file_path_output = base_path / 'step1bii_v_output_blinkStatProperties.mat'

        # Load candidate_signal
        input_data, output_datax = load_matlab_data(cls.mat_file_path_input, cls.mat_file_path_output)
        cls.input_data = input_data
        cls.output_data = output_datax

        # Candidate signals
        cls.signal = cls.input_data['candidateSignals']

        # Blink fits as DataFrame
        cls.df = pd.DataFrame.from_records(cls.input_data['blinkFits'])
        cls.df.rename(columns=RENAME_MAP, inplace=True)

        # Ground truth signal candidate_signal
        cls.signal_data_gt = rename_keys(cls.output_data['blinks']['signalData'], RENAME_MAP)

        # Remove unwanted keys from the ground truth for comparison
        for key in ["signal", "blinkPositions", "signalType", "signalNumber", "signalLabel"]:
            cls.signal_data_gt.pop(key, None)

        # Use fixed z_thresholds
        cls.z_thresholds = np.array([[0.9, 0.98], [2.0, 5.0]])

    def test_blink_statistic(self):
        """
        Test the get_blink_statistic function output against the MATLAB ground truth.
        """
        # Compute blink statistics
        signal_data = get_blink_statistic(self.df, self.z_thresholds, signal=self.signal)

        # Check for differences between the dictionaries
        differences = {}
        for key in self.signal_data_gt.keys():
            if key not in signal_data:
                differences[key] = f"Key '{key}' is missing in the computed signal_data."
            elif not np.allclose(self.signal_data_gt[key], signal_data[key], atol=1e-6, equal_nan=True):
                differences[key] = {
                    'ground_truth': self.signal_data_gt[key],
                    'computed': signal_data[key]
                }

        for key in signal_data.keys():
            if key not in self.signal_data_gt:
                differences[key] = f"Key '{key}' is missing in the ground truth signal_data."

        # Log differences if any
        if differences:
            print("\nDifferences found in signal_data:")
            for key, diff in differences.items():
                print(f"Key: {key}, Difference: {diff}")

        # Assert no differences
        self.assertFalse(
            differences,
            f"The computed signal_data does not match the ground truth. Differences: {differences}"
        )


if __name__ == '__main__':
    unittest.main()
