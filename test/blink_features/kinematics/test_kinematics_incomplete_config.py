"""Scenario D: Kinematics with omitted modality keys."""

from __future__ import annotations

from pathlib import Path

import mne

from pyblinker.blink_features.kinematics import compute_kinematic_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EAR_CHANNEL = "EAR-avg_ear"
EEG_CHANNEL = "EEG-E8"


def test_missing_eeg_key_does_not_block_ear_processing() -> None:
    """SEGMENT_CONFIG lacking EEG keys still processes EAR and generic metadata."""

    raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

    segment_config = {
        "ear": {
            "channel": EAR_CHANNEL,
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

    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw,
        epoch_len=30.0,
        blink_label=None,
        progress_bar=False,
        segmentation_type=segment_config,
    )

    # Even though EEG is not configured for segmentation, the channel remains available
    # for feature extraction and should fall back to generic blink metadata.
    df = compute_kinematic_features(epochs, picks=[EAR_CHANNEL, EEG_CHANNEL])

    assert "blink_onset_eeg" not in epochs.metadata.columns
    assert "blink_onset_ear" in epochs.metadata.columns
    assert any(col.endswith(f"_{EAR_CHANNEL}") for col in df.columns)
    assert any(col.endswith(f"_{EEG_CHANNEL}") for col in df.columns)
    assert all(
        col.endswith(f"_{EAR_CHANNEL}") or col.endswith(f"_{EEG_CHANNEL}") for col in df.columns
    )
    assert df.notna().sum().sum() > 0
