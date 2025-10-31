'''
Implement the blink detect as per the legacy MATLAB-based BLINK approach, but using Python and MNE-Python.
This code is a basic usage example of the pyblinker library for detecting eye blinks in EEG data.
It reads a raw EEG file, processes the data to filter and resample it,
and then uses the BlinkDetector class to identify and visualize eye blinks.

'''

import logging

import matplotlib
import mne
from test.data_setup import ensure_mne_sample_edf
logging.basicConfig(level=logging.INFO)

matplotlib.use('TkAgg')

def plot_blinks(raw_file: str) -> None:
    """Plot eye close events based on EEG signals.

    Args:
    raw_file (str): Path to the raw EEG candidate_signal file in .fif or .edf format.

    Returns:
    None
    """
    # Let MNE infer based on extension
    if raw_file.lower().endswith('.fif'):
        raw = mne.io.read_raw_fif(raw_file, preload=True)
    else:
        raw = mne.io.read_raw(raw_file, preload=True)


    raw.plot(block=True, title=f'Eye close based on channel')

if __name__ == '__main__':
    # Ensure EDF is available under test/test_files, then plot it
    edf_path = ensure_mne_sample_edf()
    print(f"EDF ready at: {edf_path}")
    plot_blinks(str(edf_path))
