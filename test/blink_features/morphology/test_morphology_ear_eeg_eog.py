"""Full-modality morphology coverage (EAR + EEG + EOG)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.blink_features.morphology.epoch_features import _available_styles
from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from test.segment_config import build_segment_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "test" / "major_structure_refactor"
BASELINE_PATH = OUTPUT_DIR / "morphology_features_ear_eeg_eog.pkl"
UPDATE_ENV_VAR = "UPDATE_MORPHOLOGY_BASELINE"

EAR_CHANNEL = "EAR-avg_ear"
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"

REQUIRED_EAR_COLUMNS = ['ear__th_point__morphology__amp_peak_abs_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_abs_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_abs_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_signed_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_signed_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_signed_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_to_trough_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_to_trough_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__amp_peak_to_trough_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__amp_trough_signed_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__amp_trough_signed_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__amp_trough_signed_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_rect_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_rect_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_rect_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_trapz_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_trapz_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__area_abs_total_trapz_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__duration_cv__EAR-AVG_EAR', 'ear__th_point__morphology__duration_mean__EAR-AVG_EAR', 'ear__th_point__morphology__duration_std__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_10_90_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_10_90_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_10_90_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_peak_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_peak_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__fall_time_peak_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__half_width_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__half_width_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__half_width_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_10_90_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_10_90_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_10_90_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_peak_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_peak_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__rise_time_peak_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_rect_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_rect_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_rect_base_std__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_trapz_base_cv__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_trapz_base_mean__EAR-AVG_EAR', 'ear__th_point__morphology__symmetry_trapz_base_std__EAR-AVG_EAR']
#

_REQUIRED_LEGACY_MORPHOLOGY_METRICS_EEG = {
        "zero": {
                "duration_zero": [
                        "eeg__zero__morphology__duration_zero_mean__EEG-E8",
                        "eeg__zero__morphology__duration_zero_std__EEG-E8",
                        "eeg__zero__morphology__duration_zero_cv__EEG-E8",
                        ],
                "closing_time_zero": [
                        "eeg__zero__morphology__closing_time_zero_mean__EEG-E8",
                        "eeg__zero__morphology__closing_time_zero_std__EEG-E8",
                        "eeg__zero__morphology__closing_time_zero_cv__EEG-E8",
                        ],
                "reopening_time_zero": [
                        "eeg__zero__morphology__reopening_time_zero_mean__EEG-E8",
                        "eeg__zero__morphology__reopening_time_zero_std__EEG-E8",
                        "eeg__zero__morphology__reopening_time_zero_cv__EEG-E8",
                        ],
                "time_shut_zero": [
                        "eeg__zero__morphology__time_shut_zero_mean__EEG-E8",
                        "eeg__zero__morphology__time_shut_zero_std__EEG-E8",
                        "eeg__zero__morphology__time_shut_zero_cv__EEG-E8",
                        ],
                },
        "base": {
                "duration_base": [
                        "eeg__base__morphology__duration_base_mean__EEG-E8",
                        "eeg__base__morphology__duration_base_std__EEG-E8",
                        "eeg__base__morphology__duration_base_cv__EEG-E8",
                        ],
                "time_shut_base": [
                        "eeg__base__morphology__time_shut_base_mean__EEG-E8",
                        "eeg__base__morphology__time_shut_base_std__EEG-E8",
                        "eeg__base__morphology__time_shut_base_cv__EEG-E8",
                        ],
                },
        "tent": {
                "duration_tent": [
                        "eeg__tent__morphology__duration_tent_mean__EEG-E8",
                        "eeg__tent__morphology__duration_tent_std__EEG-E8",
                        "eeg__tent__morphology__duration_tent_cv__EEG-E8",
                        ],
                "closing_time_tent": [
                        "eeg__tent__morphology__closing_time_tent_mean__EEG-E8",
                        "eeg__tent__morphology__closing_time_tent_std__EEG-E8",
                        "eeg__tent__morphology__closing_time_tent_cv__EEG-E8",
                        ],
                "reopening_time_tent": [
                        "eeg__tent__morphology__reopening_time_tent_mean__EEG-E8",
                        "eeg__tent__morphology__reopening_time_tent_std__EEG-E8",
                        "eeg__tent__morphology__reopening_time_tent_cv__EEG-E8",
                        ],
                "time_shut_tent": [
                        "eeg__tent__morphology__time_shut_tent_mean__EEG-E8",
                        "eeg__tent__morphology__time_shut_tent_std__EEG-E8",
                        "eeg__tent__morphology__time_shut_tent_cv__EEG-E8",
                        ],
                },
        "half": {
                "duration_half_base": [
                        "eeg__half__morphology__duration_half_base_mean__EEG-E8",
                        "eeg__half__morphology__duration_half_base_std__EEG-E8",
                        "eeg__half__morphology__duration_half_base_cv__EEG-E8",
                        ],
                "duration_half_zero": [
                        "eeg__half__morphology__duration_half_zero_mean__EEG-E8",
                        "eeg__half__morphology__duration_half_zero_std__EEG-E8",
                        "eeg__half__morphology__duration_half_zero_cv__EEG-E8",
                        ],
                },
        "peak": {
                "peak_time_blink": [
                        "eeg__peak__morphology__peak_time_blink_mean__EEG-E8",
                        "eeg__peak__morphology__peak_time_blink_std__EEG-E8",
                        "eeg__peak__morphology__peak_time_blink_cv__EEG-E8",
                        ],
                "peak_time_tent": [
                        "eeg__peak__morphology__peak_time_tent_mean__EEG-E8",
                        "eeg__peak__morphology__peak_time_tent_std__EEG-E8",
                        "eeg__peak__morphology__peak_time_tent_cv__EEG-E8",
                        ],
                "peak_max_blink": [
                        "eeg__peak__morphology__peak_max_blink_mean__EEG-E8",
                        "eeg__peak__morphology__peak_max_blink_std__EEG-E8",
                        "eeg__peak__morphology__peak_max_blink_cv__EEG-E8",
                        ],
                "peak_max_tent": [
                        "eeg__peak__morphology__peak_max_tent_mean__EEG-E8",
                        "eeg__peak__morphology__peak_max_tent_std__EEG-E8",
                        "eeg__peak__morphology__peak_max_tent_cv__EEG-E8",
                        ],
                },
        "inter_blink": {
                "inter_blink_max_amp": [
                        "eeg__inter_blink__morphology__inter_blink_max_amp_mean__EEG-E8",
                        "eeg__inter_blink__morphology__inter_blink_max_amp_std__EEG-E8",
                        "eeg__inter_blink__morphology__inter_blink_max_amp_cv__EEG-E8",
                        ],
                },
        }
#
_REQUIRED_LEGACY_MORPHOLOGY_METRICS_EOG =  {
        "zero": {
                "duration_zero": [
                        "eog__zero__morphology__duration_zero_mean__EOG-EEG-eog_vert_left",
                        "eog__zero__morphology__duration_zero_std__EOG-EEG-eog_vert_left",
                        "eog__zero__morphology__duration_zero_cv__EOG-EEG-eog_vert_left",
                        ],
                # "closing_time_zero": [
                # 		"eog__zero__morphology__closing_time_zero_mean__EOG-EEG-eog_vert_left",
                # 		"eog__zero__morphology__closing_time_zero_std__EOG-EEG-eog_vert_left",
                # 		"eog__zero__morphology__closing_time_zero_cv__EOG-EEG-eog_vert_left",
                # 		],
                # "reopening_time_zero": [
                # 		"eog__zero__morphology__reopening_time_zero_mean__EOG-EEG-eog_vert_left",
                # 		"eog__zero__morphology__reopening_time_zero_std__EOG-EEG-eog_vert_left",
                # 		"eog__zero__morphology__reopening_time_zero_cv__EOG-EEG-eog_vert_left",
                # 		],
                # "time_shut_zero": [
                # 		"eog__zero__morphology__time_shut_zero_mean__EOG-EEG-eog_vert_left",
                # 		"eog__zero__morphology__time_shut_zero_std__EOG-EEG-eog_vert_left",
                # 		"eog__zero__morphology__time_shut_zero_cv__EOG-EEG-eog_vert_left",
                # 		],
                },
        # "base": {
        # 		"duration_base": [
        # 				"eog__base__morphology__duration_base_mean__EOG-EEG-eog_vert_left",
        # 				"eog__base__morphology__duration_base_std__EOG-EEG-eog_vert_left",
        # 				"eog__base__morphology__duration_base_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		"time_shut_base": [
        # 				"eog__base__morphology__time_shut_base_mean__EOG-EEG-eog_vert_left",
        # 				"eog__base__morphology__time_shut_base_std__EOG-EEG-eog_vert_left",
        # 				"eog__base__morphology__time_shut_base_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		},
        # "tent": {
        # 		"duration_tent": [
        # 				"eog__tent__morphology__duration_tent_mean__EOG-EEG-eog_vert_left",
        # 				"eog__tent__morphology__duration_tent_std__EOG-EEG-eog_vert_left",
        # 				"eog__tent__morphology__duration_tent_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		"closing_time_tent": [
        # 				"eog__tent__morphology__closing_time_tent_mean__EOG-EEG-eog_vert_left",
        # 				"eog__tent__morphology__closing_time_tent_std__EOG-EEG-eog_vert_left",
        # 				"eog__tent__morphology__closing_time_tent_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		"reopening_time_tent": [
        # 				"eog__tent__morphology__reopening_time_tent_mean__EOG-EEG-eog_vert_left",
        # 				"eog__tent__morphology__reopening_time_tent_std__EOG-EEG-eog_vert_left",
        # 				"eog__tent__morphology__reopening_time_tent_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		"time_shut_tent": [
        # 				"eog__tent__morphology__time_shut_tent_mean__EOG-EEG-eog_vert_left",
        # 				"eog__tent__morphology__time_shut_tent_std__EOG-EEG-eog_vert_left",
        # 				"eog__tent__morphology__time_shut_tent_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		},
        # "half": {
        # 		"duration_half_base": [
        # 				"eog__half__morphology__duration_half_base_mean__EOG-EEG-eog_vert_left",
        # 				"eog__half__morphology__duration_half_base_std__EOG-EEG-eog_vert_left",
        # 				"eog__half__morphology__duration_half_base_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		"duration_half_zero": [
        # 				"eog__half__morphology__duration_half_zero_mean__EOG-EEG-eog_vert_left",
        # 				"eog__half__morphology__duration_half_zero_std__EOG-EEG-eog_vert_left",
        # 				"eog__half__morphology__duration_half_zero_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		},
        # "peak": {
        # 		"peak_time_blink": [
        # 				"eog__peak__morphology__peak_time_blink_mean__EOG-EEG-eog_vert_left",
        # 				"eog__peak__morphology__peak_time_blink_std__EOG-EEG-eog_vert_left",
        # 				"eog__peak__morphology__peak_time_blink_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		"peak_time_tent": [
        # 				"eog__peak__morphology__peak_time_tent_mean__EOG-EEG-eog_vert_left",
        # 				"eog__peak__morphology__peak_time_tent_std__EOG-EEG-eog_vert_left",
        # 				"eog__peak__morphology__peak_time_tent_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		"peak_max_blink": [
        # 				"eog__peak__morphology__peak_max_blink_mean__EOG-EEG-eog_vert_left",
        # 				"eog__peak__morphology__peak_max_blink_std__EOG-EEG-eog_vert_left",
        # 				"eog__peak__morphology__peak_max_blink_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		"peak_max_tent": [
        # 				"eog__peak__morphology__peak_max_tent_mean__EOG-EEG-eog_vert_left",
        # 				"eog__peak__morphology__peak_max_tent_std__EOG-EEG-eog_vert_left",
        # 				"eog__peak__morphology__peak_max_tent_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		},
        # "inter_blink": {
        # 		"inter_blink_max_amp": [
        # 				"eog__inter_blink__morphology__inter_blink_max_amp_mean__EOG-EEG-eog_vert_left",
        # 				"eog__inter_blink__morphology__inter_blink_max_amp_std__EOG-EEG-eog_vert_left",
        # 				"eog__inter_blink__morphology__inter_blink_max_amp_cv__EOG-EEG-eog_vert_left",
        # 				],
        # 		},
        }


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
    "eeg": {"channel": EEG_CHANNEL},
    "eog": {"channel": EOG_CHANNEL},
}


class TestFullModalityKinematicPipeline(unittest.TestCase):
    """EAR+EEG+EOG morphology pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(cls.raw_path, preload=True, verbose=False)
        segment_config = build_segment_config(raw)
        segment_config["ear"] = SEGMENT_CONFIG["ear"]

        cls.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
        )

        cls.df = compute_epoch_morphology_features(
            epochs=cls.epochs, picks=[EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL]
        )


    def _load_baseline(self) -> pd.DataFrame:
        if not BASELINE_PATH.exists():
            raise AssertionError(
                "Missing baseline pickle. Set UPDATE_MORPHOLOGY_BASELINE=1 and rerun the test to generate it."
            )
        return pd.read_pickle(BASELINE_PATH)

    def _maybe_write_baseline(self) -> None:
        if os.environ.get(UPDATE_ENV_VAR) == "1":
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            self.df.to_pickle(BASELINE_PATH)

    # def test_style_windows_uses_landmark_frames_when_available(self) -> None:
    #     metadata_row = {
    #         "start__left_zero__eeg": [10, 30],
    #         "end__right_zero__eeg": [16, 40],
    #         "onset__zero__eeg": [1.0, 2.0],
    #         "duration__zero__eeg": [0.0, 0.0],
    #     }
    #
    #     windows = _style_windows(metadata_row, "eeg", "zero", sfreq=256.0, n_times=2560)
    #     self.assertEqual(windows, [(10, 16), (30, 40)])

    def test_eeg_columns(self) -> None:
        styles = _available_styles(tuple(self.epochs.metadata.columns), "eeg")
        self.assertTrue(styles)
        for style in styles:
            expected = f"eeg__{style}__morphology__duration_mean__{EEG_CHANNEL}"
            self.assertIn(expected, self.df.columns)

        for style in _REQUIRED_LEGACY_MORPHOLOGY_METRICS_EEG.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_eog_columns(self) -> None:
        styles = _available_styles(tuple(self.epochs.metadata.columns), "eog")
        self.assertTrue(styles)
        for style in styles:
            expected = f"eog__{style}__morphology__duration_mean__{EOG_CHANNEL}"
            self.assertIn(expected, self.df.columns)
        #
        for style in _REQUIRED_LEGACY_MORPHOLOGY_METRICS_EOG.values():
            for metric in style.values():
                for stat_name in metric:
                    self.assertIn(stat_name, self.df.columns)

    def test_ear_columns(self) -> None:
        for col in REQUIRED_EAR_COLUMNS:
            self.assertIn(col, self.df.columns)


    def test_matches_baseline_pickle(self) -> None:
        # self._maybe_write_baseline()
        # self.df.to_pickle(BASELINE_PATH)
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
