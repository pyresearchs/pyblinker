"""EAR-only kinematic feature walkthrough.

This tutorial mirrors the EAR energy example but runs the kinematic pipeline
without requiring any EEG channels or dummy configuration entries. Only the
EAR channel is needed; other modalities are optional and safely ignored when
set to ``None``.
"""

from pathlib import Path

import mne

from pyblinker.blink_features.kinematics import compute_kinematic_features
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
# Select EAR (required) and optional EOG channels
# -----------------------------------------------------------------------------
ear_channel = "EAR-avg_ear"
eog_channel = None  # set to "EOG-EEG-eog_vert_left" to include EOG metadata

picks = [ch for ch in (ear_channel, eog_channel) if ch and ch in raw.ch_names]
if ear_channel not in picks:
    raise ValueError(
        f"Required EAR channel '{ear_channel}' not found in raw data. "
        "Update `ear_channel` to match your recording before running this tutorial."
    )

raw.pick(picks)

SEGMENT_CONFIG = {
    "ear": {
        "channel": ear_channel,
        "seg_type": "threshold_interpolation",
        "threshold": 0.260,
        "annotation_time_unit": "seconds",
        "max_extension": 0.35,
        "extension_step": 0.05,
        "padding": 0.05,
        "extend_before": True,
        "extend_after": True,
    }
}

if eog_channel and eog_channel in picks:
    SEGMENT_CONFIG["eog"] = {
        "channel": eog_channel,
        "seg_type": [],
        "threshold": None,
    }

# Slice raw data into epochs and refine blink metadata
epochs = slice_raw_into_mne_epochs_refine_annot(
    raw,
    epoch_len=30.0,
    blink_label=None,
    segmentation_type=SEGMENT_CONFIG,
)

# Compute kinematic features for the EAR channel only
df = compute_kinematic_features(
    epochs,
    picks=ear_channel,
)

output_path = PROJECT_ROOT / "tutorial_outputs" / "ear_kinematics" / "ear_kinematic_features.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved EAR kinematic features to {output_path}")
