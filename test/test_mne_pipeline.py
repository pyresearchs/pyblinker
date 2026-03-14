import unittest
import os
import mne

from pyblinker.blinker.pyblinker import BlinkDetector

class TestMNEPipeline(unittest.TestCase):
    def setUp(self):
        sample_data_folder = mne.datasets.sample.data_path()
        self.raw_file = os.path.join(
            sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
        )
        self.raw = mne.io.read_raw_fif(self.raw_file, preload=True, verbose=False)
        self.raw.pick_types(eeg=True)
        # Downsample to speed up testing
        self.raw.resample(100)
        # Keep only first 2 EEG channels to speed up
        channel_range = [f"EEG 00{idx}" for idx in range(1, 3)]
        to_drop = list(set(self.raw.ch_names) - set(channel_range))
        if to_drop:
            self.raw = self.raw.drop_channels(to_drop)

    def test_mne_pipeline_execution(self):
        detector = BlinkDetector(self.raw, pipeline="mne")
        annotations, channel, n_good_blinks, df_out, fig_data, ch_selected = detector.get_blink()
        
        # Verify MNE picked an EEG channel
        self.assertTrue(channel.startswith("EEG"))
        
        # Verify we detected some blinks
        self.assertGreater(n_good_blinks, 0)
        
        # Verify df_out is a populated DataFrame with BlinkProperties columns
        self.assertFalse(df_out.empty)
        self.assertIn("max_blink", df_out.columns)
        self.assertIn("left_zero", df_out.columns)
        self.assertIn("right_zero", df_out.columns)
        
        # Annotations
        self.assertTrue(isinstance(annotations, mne.Annotations))
        self.assertEqual(len(annotations), n_good_blinks)

if __name__ == '__main__':
    unittest.main()