"""Basic usage example showing how to run ``BlinkDetector`` on sample data."""

# 1) Import required libraries
import mne
from pyblinker.blinker.pyblinker import BlinkDetector
# 2) Specify the EEG file path
# Replace this path with the location of your .fif EEG recording file

# sample_data_folder = mne.datasets.sample.data_path()
# raw_file = os.path.join(
# 	sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
# 	)

raw_file=r"ear_eog.fif"
# 3) Load the EEG recording
print("Loading EEG data...")

raw = mne.io.read_raw_fif(raw_file, preload=True)

eeg_chs = [f"EEG-E{i}" for i in range(1, 21)]
raw.pick(eeg_chs)
# 4) Keep only EEG channels
# raw.pick_types(eeg=True)

# 5) Filter the EEG signal between 0.5 Hz and 20.5 Hz
raw.filter(0.5, 20.5, fir_design="firwin")

# 6) Downsample the data to 100 Hz for faster processing
# raw.resample(100)

# 7) Keep only the first 10 EEG channels (EEG 000 - EEG 009)
# channel_range = [f"EEG 00{idx}" for idx in range(10)]
# to_drop = list(set(raw.ch_names) - set(channel_range))
# if to_drop:
# 	raw = raw.drop_channels(to_drop)

# 8) Create and configure the blink detector
print("Detecting blinks...")
detector = BlinkDetector(
	raw,
	visualize=False,
	annot_label=None,
	filter_low=0.5,
	filter_high=30.0,
	resample_rate=100,
	n_jobs=2,
	use_multiprocessing=True,
	)

# 9) Run blink detection
annotations, channel, _good, _df, _fig_data, _selected = detector.get_blink()

# 10) Annotate detected blinks in the raw EEG data
raw.set_annotations(annotations)

# 11) Plot the EEG with blink annotations
print(f"Displaying EEG with blink detections (based on channel {channel})...")
raw.plot(block=True, title=f"Eye close based on channel {channel}")

# 12) Done
print("Blink detection complete. Close the plot window to finish.")
