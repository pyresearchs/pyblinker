"""Full-modality kinematics coverage (EAR + EEG + EOG)."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.kinematics import compute_kinematic_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EAR_CHANNEL = "EAR-avg_ear"
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"


class TestFullModalityKinematicPipeline(unittest.TestCase):
    """EAR+EEG+EOG kinematic pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"

    def test_full_modality_config_produces_all_channels(self) -> None:
        """EAR+EEG+EOG config yields kinematics for each configured channel."""

        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)

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
            },
            "eeg": {"channel": EEG_CHANNEL, "seg_type": "base"},
            "eog": {"channel": EOG_CHANNEL, "seg_type": "base"},
        }

        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
        )

        df = compute_kinematic_features(
            epochs,
            picks=[EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL],
        )

        self.assertIn("blink_onset_ear", epochs.metadata.columns)
        self.assertIn("blink_onset_eeg", epochs.metadata.columns)
        self.assertIn("blink_onset_eog", epochs.metadata.columns)
        for ch in (EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL):
            self.assertTrue(any(col.endswith(f"__{ch}") for col in df.columns))
        self.assertGreater(df.notna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
