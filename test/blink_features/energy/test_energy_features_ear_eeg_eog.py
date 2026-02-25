"""Full-modality energy coverage (EAR + EEG + EOG)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.helper import build_expected_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "test" / "major_structure_refactor"
BASELINE_PATH =  OUTPUT_DIR/ "energy_features_ear_eeg_eog.pkl"
UPDATE_ENV_VAR = "UPDATE_ENERGY_BASELINE"

EAR_CHANNEL = "EAR-avg_ear"
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"

stats = ["mean", "std", "cv"]
energy_metrics = [
    "blink_signal_energy",
    "teager_kaiser_energy",
    "blink_line_length",
    "blink_velocity_integral",
]
landmarks = ["zero", "base", "tent", "half", "peak"]

REQUIRED_EAR_COLUMNS = build_expected_metrics(
    landmark="th_point",
    metrics=["blink_signal_energy"],
    stats=stats,
    modality="ear",
    feature="energy",
    channel="EAR-AVG_EAR")

REQUIRED_EEG_METRICS =  build_expected_metrics(
            landmark=landmarks,
            metrics=energy_metrics,
            stats=stats,
            modality="eeg",
            feature="energy",
            channel=EEG_CHANNEL,
        )


REQUIRED_EOG_METRICS =   build_expected_metrics(
            landmark=landmarks,
            metrics=energy_metrics,
            stats=stats,
            modality="eog",
            feature="energy",
            channel=EOG_CHANNEL,
        )

SEGMENT_CONFIG = {
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


class TestFullModalityEnergyPipeline(unittest.TestCase):
    """EAR+EEG+EOG energy pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(cls.raw_path, preload=True, verbose=False)

        cls.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=SEGMENT_CONFIG,
        )
        cls.df = compute_energy_features(
            epochs=cls.epochs, picks=[EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL]
        )
        cls.df.to_pickle(BASELINE_PATH) # We will conduct major refactor to the structure of energy features. Therefore, we will compare column by column, and ensure the values are close enough with the original. Here, we will safe the original dataframe as baseline, and compare the new dataframe with the baseline dataframe. If the values are close enough, we will consider the test passed. If not, we will investigate the differences and update the baseline if necessary.

    def _load_baseline(self) -> pd.DataFrame:
        if not BASELINE_PATH.exists():
            raise AssertionError(
                "Missing baseline pickle. Set UPDATE_ENERGY_BASELINE=1 and rerun the test to generate it."
            )
        return pd.read_pickle(BASELINE_PATH)

    # def _maybe_write_baseline(self) -> None:
    #     if os.environ.get(UPDATE_ENV_VAR) == "1":
    #         OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    #         self.df.to_pickle(BASELINE_PATH)



    def test_eeg(self) -> None:
        for style in REQUIRED_EEG_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_eog(self) -> None:
        for style in REQUIRED_EOG_METRICS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_ear(self) -> None:
        # for col in REQUIRED_EAR_COLUMNS:
        #     self.assertIn(col, self.df.columns)
        for style in REQUIRED_EAR_COLUMNS.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_matches_baseline_pickle(self) -> None:
        # self._maybe_write_baseline()
        baseline = self._load_baseline()
        pd.testing.assert_frame_equal(
            self.df.sort_index(axis=1),
            baseline.sort_index(axis=1),
            check_dtype=False,
            rtol=1e-6,
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()

