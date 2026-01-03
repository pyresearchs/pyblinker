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
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# -----------------------------------------------------------------------------
# Load raw data and annotations
# -----------------------------------------------------------------------------
raw_path = (
    PROJECT_ROOT
    / "manual_annotation_feature_calculation_data"
    / "ear_eog.fif"
)
csv_path = (
    PROJECT_ROOT
    / "manual_annotation_feature_calculation_data"
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
# Select EAR channel
# -----------------------------------------------------------------------------
ear_channel = "EAR-avg_ear"
eeg_channel = "EEG-E8"
for required in (ear_channel, eeg_channel):
    if required not in raw.ch_names:
        raise ValueError(
            f"Required channel '{required}' not found in raw data."
        )

raw.pick([ear_channel, eeg_channel])
SEGMENT_CONFIG = {
    "ear": {
        "seg_type": "threshold_interpolation",
        "threshold": 0.260,
        "annotation_time_unit": "seconds",
        "max_extension": 0.35,
        "extension_step": 0.05,
        "padding": 0.05,
        "extend_before": True,
        "extend_after": True,
    },
    "eeg": {
        "seg_type": [],
        "threshold": None,
    },
    "eog": {
        "seg_type": [],
        "threshold": None,
    },
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
#     / "manual_annotation_feature_calculation_data"
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
    eeg_channel=eeg_channel,
    threshold=SEGMENT_CONFIG["ear"]["threshold"],
    output_path=report_path,
)
