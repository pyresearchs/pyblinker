from pathlib import Path

import mne
from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.outside_annotation import build_ear_energy_report
from pyblinker.utils.evaluation import mat_data
from pyblinker.segmentation.refinement import (
    slice_raw_into_mne_epochs_refine_annot,
)

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# Load raw data and annotations
# -----------------------------------------------------------------------------
raw_path = (
    PROJECT_ROOT
    / "test"
    / "test_files"
    / "ear_eog_raw.fif"
)
csv_path = (
    PROJECT_ROOT
    / "test"
    / "test_files"
    / "ear_eog.csv"
)

# Load raw FIF file
raw = mne.io.read_raw_fif(
    raw_path,
    preload=True,
    verbose=False,
)

# Attach manual CSV annotations
# CSV columns: onset (sec), duration (sec), description (label)
raw.set_annotations(
    mat_data.read_annotations_as_mne(csv_path)
)

# -----------------------------------------------------------------------------
# Select channels (EAR required, EEG optional)
# -----------------------------------------------------------------------------
ear_channel = "EAR-avg_ear"
eeg_channel = "EEG-E8"
eog_channel = "EOG-EEG-eog_vert_left"

if ear_channel not in raw.ch_names:
    raise ValueError(
        f"Required EAR channel '{ear_channel}' not found in raw data. "
        "Update `ear_channel` to match your recording before running this tutorial."
    )

picks = [ch for ch in (ear_channel, eeg_channel, eog_channel) if ch in raw.ch_names]
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
    },
}

if eeg_channel in picks:
    SEGMENT_CONFIG["eeg"] = {
        "channel": eeg_channel,
        # ``seg_type=[]`` disables EEG refinement but keeps the channel available
        # for plotting in the report.
        "seg_type": [],
        "threshold": None,
    }

if eog_channel in picks:
    SEGMENT_CONFIG["eog"] = {
        "channel": eog_channel,
        "seg_type": [],
        "threshold": None,
    }
# Slice raw data into epochs
epochs = slice_raw_into_mne_epochs_refine_annot(
    raw,
    epoch_len=30.0,
    blink_label=None,
    segmentation_type=SEGMENT_CONFIG,
)

# # Persist the computed epochs for inspection/reuse
# epochs_out_path = (
#     PROJECT_ROOT
#     / "test"
#     / "test_files"
#     / "ear_metadata_threshold_interpolation.fif"
# )
# epochs_out_path.parent.mkdir(parents=True, exist_ok=True)
# # MNE accepts pathlib.Path, but using str keeps compatibility with older versions.
# epochs.save(str(epochs_out_path), overwrite=True)

report_dir = PROJECT_ROOT / "tutorial_outputs" / "ear_energy"
report_path = report_dir / "ear_energy_report.html"

# Compute energy features
df = compute_energy_features(
    epochs,
    picks=ear_channel,
)

build_ear_energy_report(
    epochs=epochs,
    ear_channel=ear_channel,
    eeg_channel=eeg_channel if eeg_channel in picks else None,
    threshold=SEGMENT_CONFIG["ear"]["threshold"],
    output_path=report_path,
)
