import logging
import unittest
import os

import matplotlib
import mne
from pyblinker.blinker.pyblinker import BlinkDetector

# Configure logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Use a non-GUI backend for matplotlib
matplotlib.use('Agg')


class TestFullTestMneData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up reusable test resources."""
        cls.sample_data_folder = mne.datasets.sample.data_path()
        cls.sample_data_raw_file = os.path.join(
            cls.sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif'
        )
        if not os.path.exists(cls.sample_data_raw_file):
            raise FileNotFoundError("Sample raw candidate_signal file not found.")

    def setUp(self):
        """Initialize resources for each test."""
        self.raw = mne.io.read_raw_fif(self.sample_data_raw_file, preload=True)
        self.raw.pick_types(eeg=True)
        self.raw.filter(0.5, 20.5, fir_design='firwin')
        self.raw.resample(100)

    def test_full_test_mne_data(self):
        """Test blink detection on EEG candidate_signal."""
        # Select a subset of EEG channels
        drange = [f'EEG 00{X}' for X in range(10)]
        to_drop_ch = list(set(self.raw.ch_names) - set(drange))
        self.raw = self.raw.drop_channels(to_drop_ch)

        # Perform blink detection
        blink_detector = BlinkDetector(self.raw, visualize=False, annot_label=None, filter_low=0.5, filter_high=20.5, resample_rate=100, n_jobs=2)
        annot, ch, number_good_blinks, df, fig_data,ch_selected = blink_detector.get_blink()

        # Assertions to validate expected outcomes
        self.assertIsNotNone(annot, "Annotations should not be None.")
        self.assertGreater(number_good_blinks, 0, "Number of detected blinks should be greater than zero.")
        self.assertIn('EEG 003', ch, "EEG 003 should be part of the analyzed channels.")
        # Assert 'EEG 003' is the first row in the 'channel' column
        self.assertEqual(ch_selected.iloc[0]['ch'], 'EEG 003', "'EEG 003' should be the first row in the channel column.")
        self.assertTrue(not df.empty, "Blink statistics dataframe should not be empty.")

        logger.info("Blink detection test passed with %d blinks detected.", number_good_blinks)


if __name__ == '__main__':
    unittest.main()
