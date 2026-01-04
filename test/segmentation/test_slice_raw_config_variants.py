"""Variant coverage for ``slice_raw_into_mne_epochs_refine_annot``."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import mne
import pytest

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.evaluation import mat_data
from test.segment_config import DEFAULT_EAR_CHANNEL, DEFAULT_EEG_CHANNEL, DEFAULT_EOG_CHANNEL


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def annotated_raw() -> mne.io.BaseRaw:
    raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
    csv_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog.csv"
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    raw.set_annotations(mat_data.read_annotations_as_mne(csv_path))
    return raw


def _ear_segmentation_block() -> Dict[str, object]:
    return {"channel": DEFAULT_EAR_CHANNEL, "seg_type": "threshold_interpolation", "threshold": 0.18}


def _full_config_with_noops() -> Dict[str, Dict[str, object]]:
    return {
        "ear": _ear_segmentation_block(),
        "eeg": {"channel": DEFAULT_EEG_CHANNEL, "seg_type": [], "threshold": None},
        "eog": {"seg_type": [], "threshold": None},
    }


def _ear_only_config() -> Dict[str, Dict[str, object]]:
    return {"ear": _ear_segmentation_block()}


def _eeg_eog_noop_config() -> Dict[str, Dict[str, object]]:
    return {
        "eeg": {"channel": DEFAULT_EEG_CHANNEL, "seg_type": []},
        "eog": {"channel": DEFAULT_EOG_CHANNEL, "seg_type": []},
    }


def _eeg_only_config() -> Dict[str, Dict[str, object]]:
    return {"eeg": {"channel": DEFAULT_EEG_CHANNEL}}


@pytest.mark.parametrize(
    ("config_builder", "expected_modalities"),
    [
        pytest.param(_full_config_with_noops, {"ear"}, id="full-config-ear-active"),
        pytest.param(_ear_only_config, {"ear"}, id="ear-only"),
        pytest.param(_eeg_eog_noop_config, set(), id="eeg-eog-noop"),
        pytest.param(_eeg_only_config, {"eeg"}, id="eeg-only"),
    ],
)
def test_slice_raw_handles_partial_configs(
    annotated_raw: mne.io.BaseRaw,
    config_builder,
    expected_modalities: Set[str],
) -> None:
    """Ensure partial or noop segmentation configs do not raise and return valid epochs."""

    epochs = slice_raw_into_mne_epochs_refine_annot(
        annotated_raw,
        epoch_len=30.0,
        blink_label=None,
        progress_bar=False,
        segmentation_type=config_builder(),
    )

    assert isinstance(epochs, mne.Epochs)
    md = epochs.metadata
    assert md is not None
    for base_col in ("blink_onset", "blink_duration", "n_blinks"):
        assert base_col in md.columns

    for modality in ("ear", "eeg", "eog"):
        modality_cols_present = f"blink_onset_{modality}" in md.columns
        if modality in expected_modalities:
            assert modality_cols_present
        else:
            assert not modality_cols_present
