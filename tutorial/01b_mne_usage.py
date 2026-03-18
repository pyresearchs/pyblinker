"""Basic usage example showing how to run the new MNE-based pipeline on sample data."""

import mne
from pyblinker.blinker.pyblinker import BlinkDetector
import os

sample_data_folder = mne.datasets.sample.data_path()
raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
)

print("Loading EEG data...")
raw = mne.io.read_raw_fif(raw_file, preload=True)

# Keep only EEG channels
raw.pick_types(eeg=True)

# Filter the EEG signal
raw.filter(0.5, 20.5, fir_design="firwin")

# Downsample
raw.resample(100)

# Keep only the first 10 EEG channels
channel_range = [f"EEG 00{idx}" for idx in range(10)]
to_drop = list(set(raw.ch_names) - set(channel_range))
if to_drop:
    raw = raw.drop_channels(to_drop)

print("Detecting blinks using legacy pipeline...")
detector_legacy = BlinkDetector(
    raw.copy(),
    pipeline="legacy",
)
_, _, good_legacy, _, _, _ = detector_legacy.get_blink()


print("Detecting blinks using new MNE pipeline...")
detector_mne = BlinkDetector(
    raw.copy(),
    pipeline="mne",
)
annotations_mne, channel_mne, good_mne, df_mne, _, _ = detector_mne.get_blink()

print(f"Legacy Pipeline found {good_legacy} blinks.")
print(f"MNE Pipeline found {good_mne} blinks.")

raw.set_annotations(annotations_mne)
print("Displaying EEG with MNE blink detections...")
raw.plot(block=True, title=f"Eye blinks using MNE Pipeline on channel {channel_mne}")

print("Blink detection complete. Close the plot window to finish.")
