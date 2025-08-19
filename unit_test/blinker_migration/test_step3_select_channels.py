import unittest
import logging
import pandas as pd
from pathlib import Path
from pyblinker.blinker.get_representative_channel import (
    filter_blink_amplitude_ratios,
    filter_good_blinks,
    filter_good_ratio,
    select_max_good_blinks
)
from unit_test.blinker_migration.debugging_tools import load_matlab_data
from pyblinker.blinker import default_setting
from unit_test.blinker_migration.pyblinker.utils.update_pkl_variables import RENAME_MAP

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSelectChannelCompact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by loading input and ground truth candidate_signal.
        """
        base_path = Path(__file__).resolve().parents[1] / 'migration_files'
        cls.mat_file_path_input = base_path / 'step3a_input_selectChannel_compact.mat'
        cls.mat_file_path_output = base_path / 'step3a_output_selectChannel_compact.mat'

        # Load candidate_signal
        input_data, output_data = load_matlab_data(
            input_path=cls.mat_file_path_input,
            output_path=cls.mat_file_path_output
        )
        cls.input_data = input_data
        cls.output_data = output_data

        # Ground truth signal candidate_signal
        cls.signal_data_gt = pd.DataFrame.from_records(cls.output_data['blinks']['signalData'])
        cls.signal_data_gt = cls.signal_data_gt.drop(columns=['signal', 'blinkPositions', 'signalType', 'signalNumber'])
        cls.signal_data_gt = cls.signal_data_gt.rename(columns={'signalLabel': 'ch'})
        cls.signal_data_gt.rename(columns=RENAME_MAP, inplace=True)

        # Signal candidate_signal for processing
        cls.signal_data = pd.DataFrame.from_records(cls.input_data['signalData'])
        cls.signal_data = cls.signal_data.drop(columns=['signal', 'blinkPositions', 'signalType', 'signalNumber'])
        cls.signal_data = cls.signal_data.rename(columns={'signalLabel': 'ch'})
        cls.signal_data.rename(columns=RENAME_MAP, inplace=True)

        # Parameters
        cls.params = default_setting.DEFAULT_PARAMS.copy()

    def test_select_channel_compact(self):
        """
        Test the blink signal selection process against MATLAB ground truth.
        df is the blink statistics for all the channeles. The statistic candidate_signal include
        - channel
        - number blinks
        - number good blinks
        - blink amp ratio
        - cut off
        - best robust std
        - good ratio



        """
        # Apply the blink signal selection process
        channel_blink_stats = self.signal_data.copy()
        channel_blink_stats = filter_blink_amplitude_ratios(channel_blink_stats, self.params)
        channel_blink_stats = filter_good_blinks(channel_blink_stats, self.params)
        channel_blink_stats = filter_good_ratio(channel_blink_stats, self.params)
        signal_data_output = select_max_good_blinks(channel_blink_stats)

        # Columns to ignore
        columns_to_ignore = ['status', 'select']

        # Log the removal of columns
        logger.info(
            "Removing the following columns from comparison: %s", columns_to_ignore
        )

        # Remove `status` and `select` columns from the comparison
        signal_data_output = signal_data_output.drop(columns=columns_to_ignore, errors='ignore')
        self.signal_data_gt = self.signal_data_gt.drop(columns=columns_to_ignore, errors='ignore')

        # Sort both DataFrames by 'ch' column
        signal_data_output = signal_data_output.sort_values(by='ch').reset_index(drop=True)
        self.signal_data_gt = self.signal_data_gt.sort_values(by='ch').reset_index(drop=True)

        # Check for differences between the DataFrames
        comparison_report = self.signal_data_gt.compare(signal_data_output, align_axis=1)

        # Log differences if any
        if not comparison_report.empty:
            logger.info("\nDifferences found in signal candidate_signal output:")
            logger.info(comparison_report)

        # Assert no differences
        self.assertTrue(
            comparison_report.empty,
            f"The processed signal candidate_signal does not match the ground truth. Differences:\n{comparison_report}"
        )


if __name__ == '__main__':
    unittest.main()
