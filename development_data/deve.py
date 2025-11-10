# 5) Configuration parameters (tweak as needed)
SAMPLING_RATE_HZ = 200.0                 # sampling rate for the loaded MAT data
CHANNELS_TO_KEEP = (
		"CH1",
		# "CH2",
		# "CH3"
		) # subset of channels for detection
TOLERANCE_SAMPLES = 200                   # blink start/end alignment tolerance
N_PREVIEW_ROWS = 10                      # how many preview rows to print in diff table
N_DIFF_ROWS = 30                         # how many differing rows to print in diff table
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
#
# # Extract relevant columns from ground truth
ground_truth_events = ground_truth_events[['leftZero', 'rightZero', 'maxValue']]
ground_truth_events = ground_truth_events.rename(columns={
		'leftZero': 'start_blink',
		'rightZero': 'end_blink'
		})
#
# # Extract and rename columns from detection
detection = detection[['left_zero', 'right_zero', 'max_value']]
detection = detection.rename(columns={
    'left_zero': 'start_blink',
    'right_zero': 'end_blink',
    'max_value': 'maxValue'
})


# # 14) Compute alignment table and summary metrics (matches, differences, etc.)
alignments, metrics = blink_comparison.compute_alignments_and_metrics(
	detected_df=detection,
	ground_truth_df=ground_truth_events,
	tolerance_samples=TOLERANCE_SAMPLES,
	)
#
# # 15) Build MNE Annotations to visualize comparisons in the Raw browser
annotations = blink_comparison.build_comparison_annotations(
	ground_truth_starts=ground_truth_events["start_blink"].to_numpy(),
	ground_truth_ends=ground_truth_events["end_blink"].to_numpy(),
	detected_starts=detection["start_blink"].to_numpy(),
	detected_ends=detection["end_blink"].to_numpy(),
	sampling_rate_hz=SAMPLING_RATE_HZ,
	tolerance_samples=TOLERANCE_SAMPLES,
	alignments=alignments,
	)
#
# # 16) Apply annotations (or clear if none)
if annotations is not None:
	print(f"[mne] Applying {len(annotations)} comparison annotations to the EEG raw")
	raw.set_annotations(annotations)
else:
	print("[mne] No blink annotations generated; clearing annotations on the EEG raw")
	raw.set_annotations(None)

raw.plot(block=True)
