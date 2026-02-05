import unittest
import pandas as pd
import mne
from pyblinker.blinker.get_blink_positions import get_blink_position
import numpy as np
from pathlib import Path
from scipy.io import loadmat

def _load_matlab_blink_positions(path: str | Path) -> np.ndarray:
    """Load blinkPositions from a complex MATLAB structure.
    
    Structure expected: blinks -> signalData(1) -> blinkPositions
    """
    p = str(path)
    mat = loadmat(p, squeeze_me=True)

    if "blinks" not in mat:
        raise KeyError(f"'blinks' structure not found in {p}. Keys: {list(mat.keys())}")
    
    blinks = mat["blinks"]
    
    # Check for signalData
    if "signalData" not in blinks.dtype.names:
         raise KeyError(f"'signalData' not found in blinks structure. Fields: {blinks.dtype.names}")
    
    signal_data = blinks["signalData"].item() # Get the array of structs
    
    # Handle the case where signal_data might be a single struct or an array of structs
    if hasattr(signal_data, 'dtype') and signal_data.dtype.names is not None:
        # It's a single struct (likely because of squeeze_me=True on a 1x1 array)
        first_signal = signal_data
    elif isinstance(signal_data, (np.ndarray, list)) and len(signal_data) > 0:
        first_signal = signal_data[0]
    else:
        first_signal = signal_data

    if not hasattr(first_signal, 'dtype') or first_signal.dtype.names is None:
         raise ValueError(f"Could not extract a valid struct from signalData. Type: {type(first_signal)}")

    if "blinkPositions" not in first_signal.dtype.names:
        raise KeyError(f"'blinkPositions' not found in signalData. Fields: {first_signal.dtype.names}")
    
    arr = first_signal["blinkPositions"]
    
    # Debug info
    # print(f"DEBUG: arr type={type(arr)}, shape={getattr(arr, 'shape', 'no shape')}")

    # Handle potentially empty or weird shapes
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        return np.empty((2, 0), dtype=np.int64)

    # If it's a scalar array (0-d), it might be containing another array
    if arr.ndim == 0:
        arr = arr.item()

    # normalize shape to (2, N)
    if arr.ndim == 1:
        arr = arr.reshape(2, 1)
    if arr.shape[0] != 2:
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = arr.T
        else:
            raise ValueError(f"Unexpected blinkPositions shape {arr.shape}; expected (2, N)")

    # MATLAB is 1-based -> Python 0-based
    return arr.astype(np.int64) - 1


class TestCompareGetBlinkPosition(unittest.TestCase):
    def test_compare_blink_positions_with_matlab(self):
        # Paths
        fif_path = Path("test/test_files/ear_eog_resamp-100_raw.fif")
        mat_expected = Path("test/test_files/step_a_extract_blinks_resamp-100.mat")

        # Load MATLAB positions (2 x N), convert to DataFrame with 0-based indices
        arr = _load_matlab_blink_positions(mat_expected)
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
        
        raw = raw.copy().pick_channels([ch_name])
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