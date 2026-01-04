"""Scenario B: EEG-only kinematic pipeline coverage."""

from __future__ import annotations

from pathlib import Path

import mne

from pyblinker.blink_features.kinematics import compute_kinematic_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"


def test_eeg_only_runs_without_ear_channel() -> None:
    """EEG-only config (with optional EOG) runs and yields EEG outputs."""

    raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

    segment_config = {
        "eeg": {
            "channel": EEG_CHANNEL,
            "seg_type": "base",
        },
        "eog": {
            "channel": EOG_CHANNEL,
            "seg_type": "base",
        },
    }

    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw,
        epoch_len=30.0,
        blink_label=None,
        progress_bar=False,
        segmentation_type=segment_config,
    )

    df = compute_kinematic_features(epochs, picks=EEG_CHANNEL)

    assert "blink_onset_ear" not in epochs.metadata.columns
    assert "blink_onset_eeg" in epochs.metadata.columns
    assert all(col.endswith(f"_{EEG_CHANNEL}") for col in df.columns)
    assert df.notna().sum().sum() > 0
