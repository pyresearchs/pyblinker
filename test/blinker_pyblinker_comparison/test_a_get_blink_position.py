import unittest
import pandas as pd
import mne
from pyblinker.blinker.get_blink_positions import get_blink_position
import numpy as np
from test.blinker_pyblinker_comparison.utils import (
    load_matlab_blink_positions,
    get_test_file_path,
)

class TestCompareGetBlinkPosition(unittest.TestCase):
    def test_compare_blink_positions_with_matlab(self):
        # Paths
        fif_path = get_test_file_path("ear_eog_resamp-100_raw.fif")
        mat_expected = get_test_file_path("step_a_extract_blinks_resamp-100.mat")

        # Load MATLAB positions (2 x N), convert to DataFrame with 0-based indices
        arr = load_matlab_blink_positions(mat_expected)
        df_mat = pd.DataFrame({
                "start_blink": arr[0, :].astype(np.int64),
                "end_blink": arr[1, :].astype(np.int64),
                })

        self.assertTrue(fif_path.exists(), f"Missing input FIF: {fif_path}")
        self.assertTrue(mat_expected.exists(), f"Missing MATLAB expected file: {mat_expected}")

        # Load raw, pick EEG-E8, resample to 20 Hz (matching the FIF)
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")
        ch_name = "EEG-E8"
        if ch_name not in raw.ch_names:
            # case-insensitive fallback
            lower_map = {c.lower(): c for c in raw.ch_names}
            ch_name = lower_map.get("eeg-e8", ch_name)
        
        raw = raw.copy().pick([ch_name])
        sfreq = 100
        if int(round(raw.info.get("sfreq", sfreq))) != sfreq:
            raw.resample(sfreq)
            
        data = raw.get_data()
        self.assertEqual(data.shape[0], 1, f"Expected single channel, got shape {data.shape}")
        blink_comp = data[0].astype(np.float64)

        # Compute positions via pyblinker
        params = dict(min_event_len=0.05, std_threshold=1.5, sfreq=sfreq)
        df_py = get_blink_position(params, blink_component=blink_comp, ch="No_channel", progress_bar=False)



        # Align dtypes
        df_py = df_py.astype({"start_blink": np.int64, "end_blink": np.int64}, errors="ignore")

        # Compare lengths
        self.assertEqual(len(df_py), len(df_mat), f"Different number of detected blinks: py={len(df_py)} mat={len(df_mat)}")
        
        # Compare exact equality
        pd.testing.assert_frame_equal(df_py.reset_index(drop=True), df_mat.reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
