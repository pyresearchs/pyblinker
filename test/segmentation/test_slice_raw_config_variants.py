"""Variant coverage for ``slice_raw_into_mne_epochs_refine_annot``."""
from __future__ import annotations

import unittest
from pathlib import Path
from typing import Dict, Set

import mne

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.evaluation import mat_data
from test.segment_config import DEFAULT_EAR_CHANNEL, DEFAULT_EEG_CHANNEL, DEFAULT_EOG_CHANNEL


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


class TestSliceRawConfigVariants(unittest.TestCase):
    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        csv_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog.csv"
        self.raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.raw.set_annotations(mat_data.read_annotations_as_mne(csv_path))

    def _assert_modalities_for_config(self, segmentation_type: Dict[str, Dict[str, object]], expected: Set[str]) -> None:
        epochs = slice_raw_into_mne_epochs_refine_annot(
            self.raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_type,
        )

        self.assertIsInstance(epochs, mne.Epochs)
        md = epochs.metadata
        self.assertIsNotNone(md)
        for base_col in ("blink_onset", "blink_duration", "n_blinks"):
            self.assertIn(base_col, md.columns)

        for modality in ("ear", "eeg", "eog"):
            with self.subTest(modality=modality):
                modality_cols_present = f"blink_onset_{modality}" in md.columns
                if modality in expected:
                    self.assertTrue(modality_cols_present)
                else:
                    self.assertFalse(modality_cols_present)

    def test_full_config_noops(self) -> None:
        self._assert_modalities_for_config(_full_config_with_noops(), {"ear"})

    def test_ear_only(self) -> None:
        self._assert_modalities_for_config(_ear_only_config(), {"ear"})

    def test_eeg_eog_noop(self) -> None:
        self._assert_modalities_for_config(_eeg_eog_noop_config(), set())

    def test_eeg_only(self) -> None:
        self._assert_modalities_for_config(_eeg_only_config(), {"eeg"})


if __name__ == "__main__":
    unittest.main()
