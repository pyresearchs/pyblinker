"""Kinematic pipeline coverage for optional channels and flexible configs."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Dict, Optional

import mne

from pyblinker.blink_features.kinematics import compute_kinematic_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EAR_CHANNEL = "EAR-avg_ear"
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"


def _load_raw() -> mne.io.BaseRaw:
    raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
    return mne.io.read_raw_fif(raw_path, preload=True, verbose=False)


def _segment_config(
    *,
    ear: Optional[str] = None,
    eeg: Optional[str] = None,
    eog: Optional[str] = None,
    include_all_keys: bool = False,
) -> Dict[str, dict]:
    """Build segmentation configs that intentionally allow missing keys."""

    config: Dict[str, dict] = {}
    if include_all_keys or ear is not None:
        config["ear"] = {
            "channel": ear,
            "seg_type": "threshold_interpolation",
            "threshold": 0.260,
            "annotation_time_unit": "seconds",
            "max_extension": 0.35,
            "extension_step": 0.05,
            "padding": 0.05,
            "extend_before": True,
            "extend_after": True,
        }
    if include_all_keys or eeg is not None:
        config["eeg"] = {"channel": eeg, "seg_type": "base"}
    if include_all_keys or eog is not None:
        config["eog"] = {"channel": eog, "seg_type": "base"}
    return config


class TestOptionalChannelsAndConfigs(unittest.TestCase):
    """Ensure kinematics handle missing modalities and partial configs."""

    def _refine(self, segment_config: Dict[str, dict]) -> mne.Epochs:
        raw = _load_raw()
        return slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
        )

    def _assert_columns_use_channels(self, df, expected_channels) -> None:
        expected_set = set(expected_channels)
        observed = {ch for ch in expected_set if any(col.endswith(ch) for col in df.columns)}
        self.assertEqual(observed, expected_set)
        self.assertTrue(all(any(col.endswith(ch) for ch in expected_set) for col in df.columns))

    def test_ear_only_pipeline(self) -> None:
        """Runs with only EAR configured and omits EEG entirely."""

        cfg = _segment_config(ear=EAR_CHANNEL)
        epochs = self._refine(cfg)
        df = compute_kinematic_features(epochs, picks=EAR_CHANNEL)

        self.assertNotIn("blink_onset_eeg", epochs.metadata.columns)
        self.assertTrue(all(col.endswith(f"_{EAR_CHANNEL}") for col in df.columns))
        self.assertGreater(df.notna().sum().sum(), 0)

    def test_eeg_only_pipeline(self) -> None:
        """Runs with only EEG configured and ignores absent EAR metadata."""

        cfg = _segment_config(eeg=EEG_CHANNEL)
        epochs = self._refine(cfg)
        df = compute_kinematic_features(epochs, picks=EEG_CHANNEL)

        self.assertIn("blink_onset_eeg", epochs.metadata.columns)
        self.assertNotIn("blink_onset_ear", epochs.metadata.columns)
        self.assertTrue(all(col.endswith(f"_{EEG_CHANNEL}") for col in df.columns))
        self.assertGreater(df.notna().sum().sum(), 0)

    def test_dual_modality_pipeline(self) -> None:
        """Supports EAR+EEG with full config keys and optional None channels."""

        cfg = _segment_config(
            ear=EAR_CHANNEL,
            eeg=EEG_CHANNEL,
            eog=None,
            include_all_keys=True,
        )
        epochs = self._refine(cfg)
        df = compute_kinematic_features(epochs, picks=[EAR_CHANNEL, EEG_CHANNEL])

        self.assertIn("blink_onset_ear", epochs.metadata.columns)
        self.assertIn("blink_onset_eeg", epochs.metadata.columns)
        self.assertNotIn("blink_onset_eog", epochs.metadata.columns)

        self._assert_columns_use_channels(df, [EAR_CHANNEL, EEG_CHANNEL])
        self.assertGreater(df.notna().sum().sum(), 0)

    def test_incomplete_config_without_ear(self) -> None:
        """Handles configs that skip EAR but keep EEG+EOG refinement."""

        cfg = _segment_config(eeg=EEG_CHANNEL, eog=EOG_CHANNEL)
        epochs = self._refine(cfg)
        df = compute_kinematic_features(epochs, picks=[EEG_CHANNEL, EOG_CHANNEL])

        self.assertNotIn("blink_onset_ear", epochs.metadata.columns)
        self.assertIn("blink_onset_eog", epochs.metadata.columns)
        self._assert_columns_use_channels(df, [EEG_CHANNEL, EOG_CHANNEL])
        self.assertGreater(df.notna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
