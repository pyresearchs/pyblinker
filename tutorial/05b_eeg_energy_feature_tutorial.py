"""EEG-only energy feature walkthrough.

This tutorial shows how to compute energy metrics from EEG channels without
requiring EAR data. The segmentation config enables EEG refinement by omitting
``seg_type`` (or using any non-empty value), whereas setting ``seg_type=[]``
would disable the modality entirely.
"""
from pathlib import Path

import mne

from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.evaluation import mat_data

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# Load raw data and annotations
# -----------------------------------------------------------------------------
raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
csv_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog.csv"

raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
raw.set_annotations(mat_data.read_annotations_as_mne(csv_path))

# -----------------------------------------------------------------------------
# Select EEG (required) and optional EOG channels
# -----------------------------------------------------------------------------
eeg_channel = "EEG-E8"
eog_channel = "EOG-EEG-eog_vert_left"  # set to None or remove from picks to ignore EOG

picks = [ch for ch in (eeg_channel, eog_channel) if ch and ch in raw.ch_names]
if not picks:
    raise ValueError(
        f"EEG channel '{eeg_channel}' not found in raw data. "
        "Update the channel names to match your recording before running this tutorial."
    )

raw.pick(picks)

SEGMENT_CONFIG = {
    "eeg": {
        "channel": eeg_channel,
        # Leaving ``seg_type`` empty (e.g., []) disables EEG refinement. Omit it
        # or provide any non-empty value to keep EEG metadata columns available.
    }
}

if eog_channel and eog_channel in picks:
    SEGMENT_CONFIG["eog"] = {
        "channel": eog_channel,
        "seg_type": "base",
    }

# Slice raw data into epochs
epochs = slice_raw_into_mne_epochs_refine_annot(
    raw,
    epoch_len=30.0,
    blink_label=None,
    segmentation_type=SEGMENT_CONFIG,
)

# Compute energy features for EEG (and EOG if present)
df = compute_energy_features(
    epochs,
    picks=picks,
)

output_path = PROJECT_ROOT / "tutorial_outputs" / "eeg_energy" / "eeg_energy_features.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved EEG energy features to {output_path}")
