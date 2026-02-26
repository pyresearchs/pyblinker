"""Scenario A: EAR-only kinematic pipeline coverage."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.helper import build_expected_metrics
from pyblinker.blink_features.energy.energy_features import compute_energy_features
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EAR_CHANNEL = "EAR-avg_ear"

required_columns=['ear__th_point__energy__blink_signal_energy_mean__EAR-AVG_EAR', 'ear__th_point__energy__blink_signal_energy_std__EAR-AVG_EAR', 'ear__th_point__energy__blink_signal_energy_cv__EAR-AVG_EAR'
		]

stats = ["mean", "std", "cv"]
metrics = ["blink_signal_energy"]

EAR_CHANNEL = "EAR-avg_ear"
modality = "ear"
landmark = ["th_point"]
feature = "energy"


REQUIRED_MORPHOLOGY_METRICS = build_expected_metrics(
	landmark=landmark,
	metrics=metrics,
	stats=stats,
	modality=modality,
	feature=feature,
	channel=EAR_CHANNEL.upper(),
	)

class TestEarOnlyKinematicPipeline(unittest.TestCase):
	"""Tests for EAR-only kinematic pipeline coverage."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
		raw = mne.io.read_raw_fif(cls.raw_path, preload=True, verbose=False)

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

		cls.epochs = slice_raw_into_mne_epochs_refine_annot(
			raw,
			epoch_len=30.0,
			blink_label=None,
			progress_bar=False,
			segmentation_type=segment_config,
			)

	def test_ear_only_runs_with_single_modality_config(self) -> None:
		"""EAR-only config produces EAR-only outputs without validating EEG."""
		df = compute_energy_features(self.epochs, picks=[EAR_CHANNEL])

		for style in REQUIRED_MORPHOLOGY_METRICS.values():
			for metric in style.values():
				for stat_name in metric:
					self.assertIn(stat_name, df.columns)




if __name__ == "__main__":
	unittest.main()
