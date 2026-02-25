"""Scenario A: EAR-only kinematic pipeline coverage."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.blink_features.morphology import compute_epoch_morphology_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EAR_CHANNEL = "EAR-avg_ear"

REQUIRED_LEGACY_MORPHOLOGY_METRICS = {
		"th_point": {
				"amp_peak_abs_base": [
						"ear__th_point__morphology__amp_peak_abs_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__amp_peak_abs_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__amp_peak_abs_base_cv__EAR-AVG_EAR",
						],
				"amp_peak_signed_base": [
						"ear__th_point__morphology__amp_peak_signed_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__amp_peak_signed_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__amp_peak_signed_base_cv__EAR-AVG_EAR",
						],
				"amp_peak_to_trough_base": [
						"ear__th_point__morphology__amp_peak_to_trough_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__amp_peak_to_trough_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__amp_peak_to_trough_base_cv__EAR-AVG_EAR",
						],
				"amp_trough_signed_base": [
						"ear__th_point__morphology__amp_trough_signed_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__amp_trough_signed_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__amp_trough_signed_base_cv__EAR-AVG_EAR",
						],
				"area_abs_total_rect_base": [
						"ear__th_point__morphology__area_abs_total_rect_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__area_abs_total_rect_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__area_abs_total_rect_base_cv__EAR-AVG_EAR",
						],
				"area_abs_total_trapz_base": [
						"ear__th_point__morphology__area_abs_total_trapz_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__area_abs_total_trapz_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__area_abs_total_trapz_base_cv__EAR-AVG_EAR",
						],
				"duration": [
						"ear__th_point__morphology__duration_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__duration_std__EAR-AVG_EAR",
						"ear__th_point__morphology__duration_cv__EAR-AVG_EAR",
						],
				"fall_time_10_90_base": [
						"ear__th_point__morphology__fall_time_10_90_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__fall_time_10_90_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__fall_time_10_90_base_cv__EAR-AVG_EAR",
						],
				"fall_time_peak_base": [
						"ear__th_point__morphology__fall_time_peak_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__fall_time_peak_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__fall_time_peak_base_cv__EAR-AVG_EAR",
						],
				"half_width_base": [
						"ear__th_point__morphology__half_width_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__half_width_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__half_width_base_cv__EAR-AVG_EAR",
						],
				"rise_time_10_90_base": [
						"ear__th_point__morphology__rise_time_10_90_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__rise_time_10_90_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__rise_time_10_90_base_cv__EAR-AVG_EAR",
						],
				"rise_time_peak_base": [
						"ear__th_point__morphology__rise_time_peak_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__rise_time_peak_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__rise_time_peak_base_cv__EAR-AVG_EAR",
						],
				"symmetry_rect_base": [
						"ear__th_point__morphology__symmetry_rect_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__symmetry_rect_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__symmetry_rect_base_cv__EAR-AVG_EAR",
						],
				"symmetry_trapz_base": [
						"ear__th_point__morphology__symmetry_trapz_base_mean__EAR-AVG_EAR",
						"ear__th_point__morphology__symmetry_trapz_base_std__EAR-AVG_EAR",
						"ear__th_point__morphology__symmetry_trapz_base_cv__EAR-AVG_EAR",
						],
				},
		}

class TestEarOnlyKinematicPipeline(unittest.TestCase):
    """Tests for EAR-only kinematic pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"

    def test_ear_only_runs_with_single_modality_config(self) -> None:
        """EAR-only config produces EAR-only outputs without validating EEG."""

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
            }
        }

        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
        )

        df = compute_epoch_morphology_features(epochs, picks=[EAR_CHANNEL])
        for col in required_columns:
            self.assertIn(col, df.columns)




if __name__ == "__main__":
    unittest.main()
