"""EEG-only kinematic feature walkthrough.

The pipeline below demonstrates how to compute blink kinematics from EEG
metadata without defining an EAR channel. SEGMENT_CONFIG includes only the
modalities you want to refine; no placeholder EAR entries are required.
"""
from pathlib import Path
import sys

# ruff: noqa: E402

# Ensure repository root is importable when running directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mne

from pyblinker.blink_features.kinematics import compute_kinematic_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.evaluation import mat_data

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
eog_channel = None  # set to "EOG-EEG-eog_vert_left" to include EOG refinement

picks = [ch for ch in (eeg_channel, eog_channel) if ch and ch in raw.ch_names]
if eeg_channel not in picks:
    raise ValueError(
        f"EEG channel '{eeg_channel}' not found in raw data. "
        "Update the channel name to match your recording before running this tutorial."
    )

raw.pick(picks)

SEGMENT_CONFIG = {
    "eeg": {
        "channel": eeg_channel,
        "seg_type": "base",
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

# Compute kinematic features for EEG (and optionally EOG if picked)
df = compute_kinematic_features(
    epochs,
    picks=picks,
)

output_path = PROJECT_ROOT / "tutorial_outputs" / "eeg_kinematics" / "eeg_kinematic_features.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved EEG kinematic features to {output_path}")
