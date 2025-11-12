# 5) Configuration parameters (tweak as needed)
SAMPLING_RATE_HZ = 200.0                 # sampling rate for the loaded MAT data
CHANNELS_TO_KEEP = (
		"CH1",
		# "CH2",
		# "CH3"
		) # subset of channels for detection
TOLERANCE_SAMPLES = 1                   # blink start/end alignment tolerance
N_PREVIEW_ROWS = 10                      # how many preview rows to print in diff table
N_DIFF_ROWS = 20                      # how many differing rows to print in diff table
# RAW_PLOT_SCALINGS = {"eeg": 0.5}       # optional MNE scaling (example)


# 7) Import helper utilities from this repository (kept top-level for clarity)
from pyblinker.utils.evaluation import (
	blink_comparison
	)
import pandas as pd
import mne
ground_truth_events = pd.read_pickle("../development_data/blinker_results.pkl")['frames']['blinkFits']
detection = pd.read_pickle("../development_data/pyblinker_results.pkl")['events']
raw = mne.io.read_raw_fif("../development_data/9636511.fif", preload=True)
# crop raw to 13 seconds for faster testing
raw.crop(0, 12.0)
#
# # Extract relevant columns from ground truth
# --- Process Ground Truth Events ---
ground_truth_events = (
		ground_truth_events[['leftZero', 'rightZero', 'maxValue']]
		.rename(columns={
				'leftZero': 'start_blink',
				'rightZero': 'end_blink'
				})
)

# Add a new ground truth blink event
ground_truth_events.loc[len(ground_truth_events)] = {
		'start_blink': 1940,
		'end_blink': 2005,
		'maxValue': None
		}

# Sort and crop to first 10 events
ground_truth_events = (
		ground_truth_events
		.sort_values(by='start_blink')
		.reset_index(drop=True)
		.head(10)
)


# --- Process Detection Events ---
detection = (
		detection[['left_zero', 'right_zero', 'max_value']]
		.rename(columns={
				'left_zero': 'start_blink',
				'right_zero': 'end_blink',
				'max_value': 'maxValue'
				})
)

# Add a new detection blink event
detection.loc[len(detection)] = {
		'start_blink': 1918,
		'end_blink': 2042,
		'maxValue': None
		}

# Sort and crop to first 10 events
detection = (
		detection
		.sort_values(by='start_blink')
		.reset_index(drop=True)
		.head(10)
)
signal=raw.get_data(picks=CHANNELS_TO_KEEP[0])[0]
# plot signal
# import matplotlib.pyplot as plt
# import numpy as np
# # Extract the signal (first channel)
# signal = raw.get_data(picks=CHANNELS_TO_KEEP[0])[0]
#
# # Get the sampling frequency to build a time axis
# sfreq = raw.info['sfreq']
# time = np.arange(signal.size) / sfreq
#
# # Plot
# plt.figure(figsize=(12, 4))
# plt.plot(time, signal, linewidth=1)
# plt.title(f"Signal from channel: {CHANNELS_TO_KEEP[0]}")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude (µV)")
# plt.grid(True)
# plt.show()

sampling_rate_hz=200
# # 13) Compare detected vs ground-truth blink intervals (prints previews/diffs)
_diagnostic_raw = blink_comparison.compare_detected_vs_ground_truth(
	detection,
	ground_truth_events,
	sampling_rate_hz,
	tolerance_samples=TOLERANCE_SAMPLES,
	n_preview_rows=N_PREVIEW_ROWS,
	n_diff_rows=N_DIFF_ROWS,
	detected_signal=signal
	)
#
# # # 14) Compute alignment table and summary metrics (matches, differences, etc.)
# alignments, metrics = blink_comparison.compute_alignments_and_metrics(
# 	detected_df=detection,
# 	ground_truth_df=ground_truth_events,
# 	tolerance_samples=TOLERANCE_SAMPLES,
# 	)
# #
# # # 15) Build MNE Annotations to visualize comparisons in the Raw browser
# annotations = blink_comparison.build_comparison_annotations(
# 	ground_truth_starts=ground_truth_events["start_blink"].to_numpy(),
# 	ground_truth_ends=ground_truth_events["end_blink"].to_numpy(),
# 	detected_starts=detection["start_blink"].to_numpy(),
# 	detected_ends=detection["end_blink"].to_numpy(),
# 	sampling_rate_hz=SAMPLING_RATE_HZ,
# 	tolerance_samples=TOLERANCE_SAMPLES,
# 	alignments=alignments,
# 	)
# #
# # # 16) Apply annotations (or clear if none)
# if annotations is not None:
# 	print(f"[mne] Applying {len(annotations)} comparison annotations to the EEG raw")
# 	raw.set_annotations(annotations)
# else:
# 	print("[mne] No blink annotations generated; clearing annotations on the EEG raw")
# 	raw.set_annotations(None)
#
# raw.plot(block=True)
